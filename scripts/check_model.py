"""Verify the GPT on CPU by overfitting two windows from the prepared corpus."""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from tokenizers import Tokenizer
import torch

from model import GPT, GPTConfig

PROJECT = Path(__file__).resolve().parents[1]


def load_fixed_batch(data_dir, manifest):
    length = 64
    starts = (0, 512)
    stream = np.memmap(
        data_dir / "pretrain_train.bin", mode="r", dtype=manifest["pretrain_bin_dtype"]
    )
    if len(stream) < starts[-1] + length + 1:
        raise ValueError("The prepared training cache is too short for the model check.")
    windows = np.stack([stream[start : start + length + 1] for start in starts])
    tokens = torch.from_numpy(windows.astype(np.int64))
    return tokens[:, :-1], tokens[:, 1:]


def check_causality(model, tokens):
    cutoff = tokens.shape[1] // 2
    changed = tokens.clone()
    changed[:, cutoff:] = (changed[:, cutoff:] + 1) % model.config.vocab_size
    with torch.no_grad():
        original, _ = model(tokens)
        modified, _ = model(changed)
    expected_shape = (*tokens.shape, model.config.vocab_size)
    if original.shape != expected_shape:
        raise RuntimeError(f"Expected logits of shape {expected_shape}, got {original.shape}.")
    torch.testing.assert_close(original[:, :cutoff], modified[:, :cutoff], atol=1e-6, rtol=1e-5)
    if torch.allclose(original[:, cutoff:], modified[:, cutoff:]):
        raise RuntimeError("Perturbing input tokens should affect the logits at those positions.")
    print(f"Logits shape: {list(original.shape)}. Causality check passed.", flush=True)


def overfit_batch(model, tokens, targets):
    with torch.no_grad():
        _, initial_loss = model(tokens, targets)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(tokens, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, final_loss = model(tokens, targets)
        accuracy = (logits.argmax(-1) == targets).float().mean().item()
    print(f"Fixed batch cross-entropy: {initial_loss.item():.5f} -> {final_loss.item():.5f}")
    print(f"Fixed batch next-token accuracy: {accuracy:.1%}")
    if not math.isfinite(final_loss.item()) or final_loss.item() >= initial_loss.item() * 0.1:
        raise RuntimeError("Expected the fixed batch loss to fall below 10% of its initial value.")
    return logits


def check_reload_and_generate(model, tokens, expected_logits, tokenizer, eos_id, checkpoint):
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=checkpoint.parent, suffix=".tmp", delete_on_close=False) as temp:
        torch.save(
            {"model_config": asdict(model.config), "model_state_dict": model.state_dict()}, temp
        )
        temp.close()
        Path(temp.name).replace(checkpoint)

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored = GPT(GPTConfig(**saved["model_config"])).eval()
    restored.load_state_dict(saved["model_state_dict"])
    if restored.lm_head.weight is not restored.token_embedding.weight:
        raise RuntimeError("The reloaded model did not retain tied embedding and output weights.")
    with torch.no_grad():
        restored_logits, _ = restored(tokens)
    torch.testing.assert_close(restored_logits, expected_logits, atol=0, rtol=0)

    prompt_length = 12
    prompt = tokens[0, :prompt_length]
    result = restored.generate(prompt, max_new_tokens=20, eos_id=eos_id)
    new_tokens = result[prompt_length:]
    if not 1 <= new_tokens.numel() <= 20 or not torch.equal(result[:prompt_length], prompt):
        raise RuntimeError("Generation did not preserve the prompt and append the expected tokens.")
    with torch.no_grad():
        generation_logits, _ = restored(result[:-1].unsqueeze(0))
    expected_tokens = generation_logits[0, prompt_length - 1 :].argmax(dim=-1)
    torch.testing.assert_close(new_tokens, expected_tokens)
    training_match = (new_tokens == tokens[0, prompt_length : result.numel()]).sum().item()
    print(f"Prompt: {tokenizer.decode(prompt.tolist())!r}")
    print(f"Overfit continuation: {tokenizer.decode(new_tokens.tolist())!r}")
    print(f"Checkpoint reloaded with identical logits. Generated {new_tokens.numel()} tokens.")
    print(f"Tokens matching the training continuation: {training_match}/{new_tokens.numel()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data" / "prepared")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "runs" / "model-check" / "overfit.pt")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive.")

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    config = GPTConfig(vocab_size=manifest["vocab_size"], context_length=manifest["context_length"])
    tokenizer = Tokenizer.from_file(str(args.data_dir / "tokenizer.json"))
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("The tokenizer vocabulary and dataset manifest disagree.")
    tokens, targets = load_fixed_batch(args.data_dir, manifest)
    model = GPT(config).eval()
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != 2_590_848 or model.lm_head.weight is not model.token_embedding.weight:
        raise RuntimeError(f"Expected tied weights and 2,590,848 parameters, got {count:,}.")
    print(f"CPU GPT: {count:,} parameters, {args.threads} threads", flush=True)
    check_causality(model, tokens)
    logits = overfit_batch(model, tokens, targets)
    check_reload_and_generate(
        model, tokens, logits, tokenizer, manifest["special_token_ids"]["eos"], args.checkpoint
    )
    print(f"Checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
