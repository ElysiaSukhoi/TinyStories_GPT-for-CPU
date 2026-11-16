"""Benchmark or pretrain the GPT on the local TinyStories token cache."""

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
from tokenizers import Tokenizer
import torch

from model import GPT, GPTConfig

PROJECT = Path(__file__).resolve().parent


class TokenStream:
    def __init__(self, path, dtype, length):
        self.tokens = np.memmap(path, mode="r", dtype=dtype)
        self.offsets = np.arange(length + 1)
        self.start_count = len(self.tokens) - length
        if self.start_count < 1:
            raise ValueError(f"{path} cannot supply a {length}-token training window.")

    def batch(self, starts):
        offsets = np.asarray(starts)[:, None] + self.offsets
        windows = torch.from_numpy(self.tokens[offsets].astype(np.int64))
        return windows[:, :-1], windows[:, 1:]

    def validation_starts(self, batches, batch_size):
        available = (len(self.tokens) - 1) // (len(self.offsets) - 1)
        count = min(batches * batch_size, available)
        return np.linspace(0, available - 1, count, dtype=np.int64) * (len(self.offsets) - 1)


def update(model, optimizer, stream, rng, args, step):
    optimizer.param_groups[0]["lr"] = args.lr * min(1, step / 100)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for _ in range(args.grad_accum):
        starts = torch.randint(stream.start_count, (args.batch_size,), generator=rng).numpy()
        tokens, targets = stream.batch(starts)
        _, loss = model(tokens, targets)
        (loss / args.grad_accum).backward()
        losses.append(loss.detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return torch.stack(losses).mean().item()


@torch.inference_mode()
def evaluate(model, stream, starts, batch_size):
    model.eval()
    start_time = perf_counter()
    loss_sum = 0.0
    for start in range(0, len(starts), batch_size):
        tokens, targets = stream.batch(starts[start : start + batch_size])
        _, loss = model(tokens, targets)
        loss_sum += targets.numel() * loss.item()
    tokens_scored = len(starts) * model.config.context_length
    return loss_sum / tokens_scored, perf_counter() - start_time, tokens_scored


def peak_memory_mib():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class MemoryCounters(ctypes.Structure):
            _fields_ = [
                ("size", wintypes.DWORD),
                ("page_faults", wintypes.DWORD),
                ("peak_working_set", ctypes.c_size_t),
                ("other_counters", ctypes.c_size_t * 7),
            ]

        current_process = ctypes.windll.kernel32.GetCurrentProcess
        current_process.restype = wintypes.HANDLE
        counters = MemoryCounters()
        counters.size = ctypes.sizeof(counters)
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = (wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD)
        if not get_memory(current_process(), ctypes.byref(counters), counters.size):
            raise OSError("GetProcessMemoryInfo failed.")
        return counters.peak_working_set / 1024**2

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024**2 if sys.platform == "darwin" else 1024)


def benchmark(model, optimizer, train_stream, val_stream, val_starts, rng, args):
    warmup = 12
    model.train()
    for step in range(1, warmup + 1):
        update(model, optimizer, train_stream, rng, args, step)
    started = perf_counter()
    for step in range(warmup + 1, warmup + args.steps + 1):
        update(model, optimizer, train_stream, rng, args, step)
    seconds = perf_counter() - started
    loss, val_seconds, val_tokens = evaluate(model, val_stream, val_starts, args.batch_size)
    return {
        "threads": args.threads,
        "batch_size": args.batch_size,
        "context_length": model.config.context_length,
        "grad_accum": args.grad_accum,
        "warmup_steps": warmup,
        "measured_steps": args.steps,
        "train_tokens_per_s": args.steps * tokens_per_step(model, args) / seconds,
        "train_seconds": seconds,
        "val_seconds": val_seconds,
        "val_tokens": val_tokens,
        "val_loss": loss,
        "peak_memory_mib": peak_memory_mib(),
    }


def tokens_per_step(model, args):
    return args.batch_size * args.grad_accum * model.config.context_length


def write_atomic(path, writer):
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete_on_close=False) as temp:
        writer(temp)
        temp.close()
        Path(temp.name).replace(path)


def write_json(path, data):
    write_atomic(path, lambda handle: handle.write((json.dumps(data, indent=2) + "\n").encode("utf-8")))


def save_checkpoint(path, data):
    write_atomic(path, lambda handle: torch.save(data, handle))


def input_digests(data_dir):
    files = ("manifest.json", "tokenizer.json", "pretrain_train.bin", "pretrain_val.bin", "pretrain_val.jsonl")
    digests = {}
    for name in files:
        with (data_dir / name).open("rb") as handle:
            digests[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    return digests


def sample_continuations(model, tokenizer, data_dir, manifest):
    model.eval()
    bos = manifest["special_token_ids"]["bos"]
    eos = manifest["special_token_ids"]["eos"]
    examples = []
    with (data_dir / "pretrain_val.jsonl").open(encoding="utf-8") as handle:
        for index in range(3):
            record = json.loads(next(handle))
            prompt_ids = tokenizer.encode(record["text"], add_special_tokens=False).ids[:12]
            prompt = torch.tensor([bos, *prompt_ids], dtype=torch.long)
            sample_seed = 2026 + index
            output = model.generate(
                prompt, 48, eos_id=eos, temperature=0.8, top_k=20,
                generator=torch.Generator().manual_seed(sample_seed),
            )
            new_ids = output[prompt.numel():].tolist()
            examples.append({
                "story_id": record["story_id"],
                "seed": sample_seed,
                "prompt": tokenizer.decode(prompt_ids),
                "prompt_ids": prompt_ids,
                "continuation": tokenizer.decode(new_ids),
                "completion_ids": new_ids,
                "stopped_at_eos": bool(new_ids and new_ids[-1] == eos),
            })
    return examples


def collect_metrics(model, stream, starts, args, step, train_loss, train_seconds):
    loss, val_seconds, val_tokens = evaluate(model, stream, starts, args.batch_size)
    return {
        "step": step,
        "train_tokens": step * tokens_per_step(model, args),
        "train_loss": train_loss,
        "val_loss": loss,
        "val_ppl": math.exp(loss),
        "val_tokens": val_tokens,
        "val_seconds": val_seconds,
        "train_seconds": train_seconds,
        "peak_memory_mib": peak_memory_mib(),
    }


def print_metrics(row):
    loss = "n/a" if row["train_loss"] is None else f"{row['train_loss']:.4f}"
    print(
        f"step {row['step']:4d} | train {loss} | val {row['val_loss']:.4f} | "
        f"PPL {row['val_ppl']:.2f} | val {row['val_seconds']:.2f} s | "
        f"peak {row['peak_memory_mib']:.0f} MiB",
        flush=True,
    )


def train(model, optimizer, train_stream, val_stream, val_starts, rng, args, manifest):
    last_path = args.run_dir / "last.pt"
    settings = {
        "seed": args.seed,
        "threads": args.threads,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "val_batches": args.val_batches,
        "model_config": asdict(model.config),
        "data_sha256": input_digests(args.data_dir),
    }
    tokenizer = Tokenizer.from_file(str(args.data_dir / "tokenizer.json"))
    if tokenizer.get_vocab_size() != model.config.vocab_size:
        raise ValueError("The tokenizer vocabulary and data manifest disagree.")

    if args.resume:
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        if saved["settings"] != settings:
            raise ValueError("Resume needs the original model, data and training settings.")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        rng.set_state(saved["batch_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        history = saved["history"]
        best_weights = saved.pop("best_model_state_dict")
        best_row = saved["best_row"]
        initial_samples = saved["initial_samples"]
        del saved
        print(f"Resuming from step {history[-1]['step']}.", flush=True)
    else:
        if last_path.exists():
            raise FileExistsError(f"{last_path} exists. Use --resume or a new --run-dir.")
        baseline = collect_metrics(model, val_stream, val_starts, args, 0, None, 0.0)
        print_metrics(baseline)
        initial_samples = sample_continuations(model, tokenizer, args.data_dir, manifest)
        history, best_row, best_weights = [baseline], None, None

    start_step = history[-1]["step"]
    if start_step > args.steps:
        raise ValueError("The requested step precedes the checkpoint.")
    elapsed = history[-1]["train_seconds"]
    losses = []
    model.train()
    started = perf_counter()
    for step in range(start_step + 1, args.steps + 1):
        losses.append(update(model, optimizer, train_stream, rng, args, step))
        if step % args.eval_every != 0 and step != args.steps:
            continue
        elapsed += perf_counter() - started
        row = collect_metrics(model, val_stream, val_starts, args, step, sum(losses) / len(losses), elapsed)
        if not math.isfinite(row["val_loss"]):
            raise RuntimeError("Nonfinite validation loss. The previous checkpoint is intact.")
        history.append(row)
        if best_row is None or row["val_loss"] < best_row["val_loss"]:
            best_row = row
            best_weights = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        save_checkpoint(last_path, {
            "settings": settings,
            "model_config": asdict(model.config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "batch_rng_state": rng.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "history": history,
            "best_row": best_row,
            "best_model_state_dict": best_weights,
            "initial_samples": initial_samples,
        })
        write_json(args.run_dir / "metrics.json", history)
        print_metrics(row)
        model.train()
        losses.clear()
        started = perf_counter()

    save_checkpoint(args.run_dir / "best.pt", {
        "model_config": asdict(model.config),
        "model_state_dict": best_weights,
        "step": best_row["step"],
        "val_loss": best_row["val_loss"],
        "tokenizer_sha256": settings["data_sha256"]["tokenizer.json"],
    })
    model.load_state_dict(best_weights)
    best_samples = sample_continuations(model, tokenizer, args.data_dir, manifest)
    write_json(args.run_dir / "samples.json", {
        "temperature": 0.8, "top_k": 20, "initial_step": 0, "best_step": best_row["step"],
        "initial_samples": initial_samples, "best_samples": best_samples,
    })
    write_json(args.run_dir / "metrics.json", history)
    print(f"Best step {best_row['step']} with validation loss {best_row['val_loss']:.4f}.", flush=True)


def lock_run(handle):
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data" / "prepared")
    parser.add_argument("--run-dir", type=Path, default=PROJECT / "runs" / "pretrain")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4, help="Number of Transformer blocks.")
    parser.add_argument("--d-model", type=int, default=192, help="Hidden width with four attention heads.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, help="Total optimizer steps, including after resume.")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--val-batches", type=int, default=32)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.steps is None:
        args.steps = 30 if args.benchmark else 4000
    if min(args.steps, args.threads, args.batch_size, args.grad_accum, args.eval_every, args.val_batches) <= 0:
        parser.error("Steps, threads, batch size, accumulation, and evaluation counts must be positive.")
    if not math.isfinite(args.lr) or args.lr <= 0 or (args.benchmark and args.resume):
        parser.error("Use a positive learning rate and choose either benchmark or resume.")
    if args.n_layers < 1 or args.d_model < 1 or args.d_model % 4:
        parser.error("--n-layers must be positive and --d-model must be a positive multiple of 4.")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed + 1)
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    config = GPTConfig(
        vocab_size=manifest["vocab_size"], context_length=manifest["context_length"],
        n_layers=args.n_layers, d_model=args.d_model,
    )
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    streams = [
        TokenStream(args.data_dir / f"pretrain_{split}.bin", manifest["pretrain_bin_dtype"], config.context_length)
        for split in ("train", "val")
    ]
    val_starts = streams[1].validation_starts(args.val_batches, args.batch_size)
    if args.benchmark:
        print(json.dumps(benchmark(model, optimizer, *streams, val_starts, rng, args), indent=2))
    else:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        with (args.run_dir / ".lock").open("a+b") as handle:
            lock_run(handle)
            train(model, optimizer, *streams, val_starts, rng, args, manifest)


if __name__ == "__main__":
    main()
