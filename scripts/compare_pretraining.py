"""Verify the matched optimization runs and score both best models on full validation."""

import argparse
import json
import math
from pathlib import Path

import torch

from checkpoints import load_model, sha256
from train_pretrain import PROJECT, TokenStream, evaluate, input_digests, write_json


def inspect_run(directory, steps, data_hashes):
    last = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
    history = last["history"]
    if history[-1]["step"] != steps or history != json.loads(
        (directory / "metrics.json").read_text(encoding="utf-8")
    ):
        raise ValueError(f"{directory} needs a complete {steps}-step run with matching metrics.")
    settings = last["settings"]
    if settings["data_sha256"] != data_hashes:
        raise ValueError(f"{directory} was trained with different prepared data.")
    best = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    best_row = min(history[1:], key=lambda row: row["val_loss"])
    if best["step"] != best_row["step"] or best["val_loss"] != best_row["val_loss"]:
        raise ValueError(f"{directory} does not contain the recorded best weights.")
    if not all(torch.equal(tensor, last["best_model_state_dict"][name])
               for name, tensor in best["model_state_dict"].items()):
        raise ValueError(f"{directory} has inconsistent best weights across checkpoints.")
    return settings, history, best_row


def compare(data, run_root, steps):
    data_hashes = input_digests(data)
    config_a, history_a, best_a = inspect_run(run_root / "a", steps, data_hashes)
    config_b, history_b, best_b = inspect_run(run_root / "b", steps, data_hashes)
    common_a = {name: value for name, value in config_a.items() if name != "model_config"}
    common_b = {name: value for name, value in config_b.items() if name != "model_config"}
    model_a = config_a["model_config"]
    model_b = config_b["model_config"]
    if common_a != common_b or model_a != {**model_b, "d_model": model_a["d_model"]}:
        raise ValueError("The compared models must differ only in hidden width.")
    if (model_a["d_model"], model_b["d_model"]) != (192, 256):
        raise ValueError("Expected widths of 192 and 256 for runs A and B.")
    if [row["step"] for row in history_a] != [row["step"] for row in history_b]:
        raise ValueError("A and B need the same validation schedule.")

    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    batch_size = common_a["batch_size"]
    context = model_a["context_length"]
    total_tokens = steps * batch_size * common_a["grad_accum"] * context
    stream = TokenStream(data / "pretrain_val.bin", manifest["pretrain_bin_dtype"], context)
    full_batch_count = math.ceil((len(stream.tokens) - 1) / (context * batch_size))
    starts = stream.validation_starts(full_batch_count, batch_size)
    torch.set_num_threads(common_a["threads"])

    runs = {}
    for name, history, best_row in (("a", history_a, best_a), ("b", history_b, best_b)):
        directory = run_root / name
        if history[-1]["train_tokens"] != total_tokens:
            raise ValueError(f"Unexpected training token budget for {directory}.")
        path = directory / "best.pt"
        model, step = load_model(path, data_hashes["tokenizer.json"])
        loss, seconds, scored = evaluate(model, stream, starts, batch_size)
        train_seconds = history[-1]["train_seconds"]
        wall = directory / "wall_seconds.json"
        runs[name] = {
            "model_config": (model_a if name == "a" else model_b),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "train_seconds": train_seconds,
            "train_tokens_per_second": total_tokens / train_seconds,
            "peak_training_memory_mib": max(row["peak_memory_mib"] for row in history),
            "wall_seconds": json.loads(wall.read_text(encoding="utf-8")) if wall.is_file() else None,
            "best_step": step,
            "best_monitoring_loss": best_row["val_loss"],
            "best_monitoring_ppl": best_row["val_ppl"],
            "full_validation": {"loss": loss, "ppl": math.exp(loss), "tokens": scored, "seconds": seconds},
            "best_checkpoint_sha256": sha256(path),
            "history": history,
        }
        print(f"{name.upper()}: {scored:,} validation tokens, best full-validation PPL {math.exp(loss):.4f}.", flush=True)
    return {"steps": steps, "train_tokens_per_run": total_tokens, "training_settings": common_a, "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data" / "prepared-v2")
    parser.add_argument("--run-root", type=Path, default=PROJECT / "runs" / "optimization")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--output", type=Path, default=PROJECT / "results" / "optimization_pretrain_ab.json")
    args = parser.parse_args()
    result = compare(args.data_dir, args.run_root, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
