"""Fine-tune the CPU GPT for two-sentence continuation and compare held-out results."""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import torch
from tokenizers import Tokenizer

from checkpoints import load_model, sha256
from scripts.story_task import EOS, PAD, encode_example, narrative_sentences
from train_pretrain import PROJECT, lock_run, save_checkpoint, write_atomic, write_json

IGNORE = -100
MAX_NEW_TOKENS = 64


def load_examples(path, tokenizer, context_length):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            prompt, answer = encode_example(tokenizer, record)
            if len(answer) < 2 or len(prompt) + len(answer) > context_length:
                raise ValueError(f"Invalid sequence length for {record['story_id']}")
            rows.append((record, prompt, answer))
    if not rows:
        raise ValueError(f"No examples in {path}")
    return rows


def make_batch(rows, pad_id):
    width = max(len(prompt) + len(answer) - 1 for _, prompt, answer in rows)
    tokens = torch.full((len(rows), width), pad_id, dtype=torch.long)
    targets = torch.full_like(tokens, IGNORE)
    for row, (_, prompt, answer) in enumerate(rows):
        sequence = prompt + answer
        tokens[row, :len(sequence) - 1] = torch.tensor(sequence[:-1])
        first_answer = len(prompt) - 1
        targets[row, first_answer:first_answer + len(answer)] = torch.tensor(answer)
    return tokens, targets, sum(len(answer) for _, _, answer in rows)


@torch.inference_mode()
def response_loss(model, rows, pad_id, batch_size):
    model.eval()
    loss_sum = tokens_scored = 0
    for start in range(0, len(rows), batch_size):
        tokens, targets, count = make_batch(rows[start:start + batch_size], pad_id)
        _, loss = model(tokens, targets)
        loss_sum += loss.item() * count
        tokens_scored += count
    return loss_sum / tokens_scored, tokens_scored


def train(model, tokenizer, source_step, tokenizer_hash, source_hash, args):
    best_path = args.run_dir / "best.pt"
    if best_path.exists():
        raise FileExistsError(f"{best_path} already exists. Choose a new --run-dir.")
    pad_id = tokenizer.token_to_id(PAD)
    train_rows = load_examples(args.data_dir / "sft_train.jsonl", tokenizer, model.config.context_length)
    val_rows = load_examples(args.data_dir / "sft_val.jsonl", tokenizer, model.config.context_length)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    rng = torch.Generator().manual_seed(args.seed)
    baseline, val_tokens = response_loss(model, val_rows, pad_id, args.batch_size)
    history = [{"epoch": 0, "step": 0, "val_loss": baseline, "val_response_tokens": val_tokens}]
    best_loss, best_epoch, step = math.inf, None, 0
    print(f"Pretrained validation CE: {baseline:.4f} over {val_tokens} answer tokens", flush=True)
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(train_rows), generator=rng).tolist()
        loss_sum = tokens_seen = 0
        model.train()
        for start in range(0, len(order), args.batch_size):
            batch = [train_rows[index] for index in order[start:start + args.batch_size]]
            tokens, targets, count = make_batch(batch, pad_id)
            step += 1
            optimizer.param_groups[0]["lr"] = args.lr * min(1, step / 25)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(tokens, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item() * count
            tokens_seen += count
        val_loss, _ = response_loss(model, val_rows, pad_id, args.batch_size)
        if not math.isfinite(val_loss):
            raise RuntimeError("Nonfinite SFT validation loss.")
        row = {
            "epoch": epoch, "step": step, "train_loss": loss_sum / tokens_seen,
            "train_response_tokens": tokens_seen, "val_loss": val_loss,
            "val_response_tokens": val_tokens, "elapsed_seconds": perf_counter() - started,
        }
        history.append(row)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            save_checkpoint(best_path, {
                "model_config": vars(model.config),
                "model_state_dict": model.state_dict(),
                "tokenizer_sha256": tokenizer_hash,
                "pretrain_sha256": source_hash,
                "pretrain_step": source_step,
                "step": step, "epoch": epoch, "val_loss": val_loss,
            })
        write_json(args.run_dir / "training_metrics.json", {
            "seed": args.seed, "learning_rate": args.lr, "threads": args.threads,
            "batch_size": args.batch_size, "pretrain_step": source_step,
            "pretrain_sha256": source_hash, "tokenizer_sha256": tokenizer_hash,
            "splits_sha256": {split: sha256(args.data_dir / f"sft_{split}.jsonl") for split in ("train", "val")},
            "best_epoch": best_epoch, "history": history,
        })
        print(f"epoch {epoch} | step {step} | train CE {row['train_loss']:.4f} | val CE {val_loss:.4f}", flush=True)
    print(f"Selected epoch {best_epoch} using validation CE {best_loss:.4f}.", flush=True)


@torch.inference_mode()
def greedy_batch(model, prompts, eos_id, pad_id):
    """Decode right-padded prompts at their actual last token positions."""
    context = model.config.context_length
    tokens = torch.full((len(prompts), context), pad_id, dtype=torch.long)
    starts = torch.tensor([len(prompt) for prompt in prompts], dtype=torch.long)
    lengths = starts.clone()
    limits = (starts + MAX_NEW_TOKENS).clamp(max=context)
    active = torch.arange(len(prompts))[starts < limits]
    for row, prompt in enumerate(prompts):
        tokens[row, :len(prompt)] = torch.tensor(prompt)
    while active.numel():
        width = int(lengths[active].max())
        logits = model.next_token_logits(tokens[active, :width], lengths[active] - 1)
        next_ids = logits.argmax(dim=-1)
        tokens[active, lengths[active]] = next_ids
        lengths[active] += 1
        active = active[(next_ids != eos_id) & (lengths[active] < limits[active])]
    return [tokens[row, starts[row]:lengths[row]].tolist() for row in range(len(prompts))]


def is_two_sentences(text):
    text = text.strip()
    sentences = list(narrative_sentences(text))
    return len(sentences) == 2 and sentences[-1][1] == len(text)


def score_model(model, rows, tokenizer, args, label):
    eos_id, pad_id = (tokenizer.token_to_id(token) for token in (EOS, PAD))
    loss, count = response_loss(model, rows, pad_id, args.batch_size)
    outputs = []
    started = perf_counter()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        generated = greedy_batch(model, [prompt for _, prompt, _ in batch], eos_id, pad_id)
        for ids in generated:
            text = tokenizer.decode(ids)
            outputs.append({
                "text": text, "token_ids": ids, "two_complete_sentences": is_two_sentences(text),
                "stopped_at_eos": bool(ids and ids[-1] == eos_id),
            })
        if len(outputs) % (args.batch_size * 8) == 0 or len(outputs) == len(rows):
            print(f"{label}: generated {len(outputs)}/{len(rows)}", flush=True)
    two = sum(row["two_complete_sentences"] for row in outputs)
    eos = sum(row["stopped_at_eos"] for row in outputs)
    both = sum(row["two_complete_sentences"] and row["stopped_at_eos"] for row in outputs)
    return outputs, {
        "response_cross_entropy": loss, "response_perplexity": math.exp(loss),
        "response_tokens_scored": count, "two_sentences": two, "two_sentence_rate": two / len(rows),
        "eos_stops": eos, "eos_rate": eos / len(rows),
        "two_sentences_and_eos": both, "two_sentences_and_eos_rate": both / len(rows),
        "generation_seconds": perf_counter() - started,
    }


def compare(base, tokenizer, tokenizer_hash, source_hash, args):
    sft_path = args.run_dir / "best.pt"
    metadata = torch.load(sft_path, weights_only=True, map_location="cpu")
    if metadata["pretrain_sha256"] != source_hash:
        raise ValueError("The SFT weights came from a different pretrained checkpoint.")
    sft, _ = load_model(sft_path, tokenizer_hash)
    if sft.config != base.config:
        raise ValueError("Pretrained and SFT model configurations disagree.")
    test_path = args.data_dir / "sft_test.jsonl"
    rows = load_examples(test_path, tokenizer, base.config.context_length)
    base_outputs, base_metrics = score_model(base, rows, tokenizer, args, "Pretrained")
    sft_outputs, sft_metrics = score_model(sft, rows, tokenizer, args, "SFT")
    records = [
        {**record, "pretrained": before, "sft": after}
        for (record, _, _), before, after in zip(rows, base_outputs, sft_outputs, strict=True)
    ]
    encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records).encode("utf-8")
    write_atomic(args.run_dir / "test_outputs.jsonl", lambda handle: handle.write(encoded))
    metrics = {
        "examples": len(rows), "decoding": "greedy", "max_new_tokens": MAX_NEW_TOKENS,
        "context_length": base.config.context_length, "sft_epoch": metadata["epoch"],
        "test_sha256": sha256(test_path), "pretrain_sha256": source_hash,
        "sft_sha256": sha256(sft_path), "tokenizer_sha256": tokenizer_hash,
        "pretrained": base_metrics, "sft": sft_metrics,
    }
    write_json(args.run_dir / "test_metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data" / "prepared")
    parser.add_argument("--pretrained", type=Path, default=PROJECT / "runs" / "pretrain" / "best.pt")
    parser.add_argument("--run-dir", type=Path, default=PROJECT / "runs" / "sft")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate", action="store_true", help="Compare saved weights on the frozen test set.")
    args = parser.parse_args()
    if min(args.threads, args.batch_size) <= 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Threads, batch size, and learning rate must be positive.")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    tokenizer_path = args.data_dir / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer_hash, source_hash = sha256(tokenizer_path), sha256(args.pretrained)
    model, step = load_model(args.pretrained, tokenizer_hash)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / ".lock").open("a+b") as handle:
        lock_run(handle)
        if args.evaluate:
            compare(model, tokenizer, tokenizer_hash, source_hash, args)
        else:
            train(model, tokenizer, step, tokenizer_hash, source_hash, args)


if __name__ == "__main__":
    main()
