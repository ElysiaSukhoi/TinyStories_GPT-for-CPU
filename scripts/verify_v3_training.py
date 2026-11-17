"""Verify the finished v3 checkpoints and record the held-out validation results."""

import argparse
import math

import torch
from tokenizers import Tokenizer

from checkpoints import load_model, sha256
from scripts.compare_pretraining import inspect_run
from scripts.freeze_baseline import read_json
from scripts.freeze_v3_protocol import DATA, PROTOCOL, RESULTS
from scripts.story_task import PAD
from train_pretrain import PROJECT, TokenStream, evaluate, input_digests, write_json
from train_sft import load_examples, response_loss

RUN = PROJECT / "runs/optimization/v3"
FULL_VALIDATION = RUN / "full_validation.json"


def wall_time(directory):
    path = directory / "wall_seconds.json"
    return read_json(path) if path.is_file() else None


def verify_pretraining(protocol, data_hashes):
    schedule = protocol["pretraining"]
    settings, history, best = inspect_run(RUN, schedule["steps"], data_hashes)
    expected = {
        key: schedule[key] for key in ("seed", "threads", "batch_size", "grad_accum", "lr", "val_batches")
    }
    expected.update(model_config=protocol["v3"]["model_config"], data_sha256=data_hashes)
    if settings != expected:
        raise ValueError("The finished run does not match the frozen v3 settings.")
    steps = [0, schedule["pilot_steps"], *range(schedule["eval_every"], schedule["steps"] + 1,
                                                  schedule["eval_every"])]
    if [row["step"] for row in history] != steps:
        raise ValueError("The pretrained run has a different validation schedule.")
    tokens_per_step = settings["batch_size"] * settings["grad_accum"] * settings["model_config"]["context_length"]
    if (history[-1]["train_tokens"] != schedule["tokens"]
            or any(row["train_tokens"] != row["step"] * tokens_per_step for row in history)):
        raise ValueError("The pretrained run has an incorrect token budget.")
    return history, best


def full_validation(protocol, best, pretrained_hash, model):
    manifest = read_json(DATA / "manifest.json")
    context = model.config.context_length
    batch = protocol["pretraining"]["batch_size"]
    validation = TokenStream(DATA / "pretrain_val.bin", manifest["pretrain_bin_dtype"], context)
    whole_batches = math.ceil((len(validation.tokens) - 1) / (context * batch))
    starts = validation.validation_starts(whole_batches, batch)
    loss, seconds, scored = evaluate(model, validation, starts, batch)
    full = {
        "best_step": best["step"], "checkpoint_sha256": pretrained_hash,
        "validation_sha256": protocol["data_sha256"]["pretrain_val.bin"],
        "loss": loss, "ppl": math.exp(loss), "tokens": scored, "seconds": seconds,
    }
    write_json(FULL_VALIDATION, full)
    print(f"V3 full validation: {scored:,} tokens, CE {loss:.4f}, PPL {full['ppl']:.4f}.")


def verify_sft(protocol, pretrained_hash, tokenizer):
    directory = RUN / "sft"
    metrics = read_json(directory / "training_metrics.json")
    schedule = protocol["sft"]
    if any(metrics[key] != schedule[key] for key in ("seed", "threads", "batch_size", "learning_rate")):
        raise ValueError("The finished SFT run does not match the frozen settings.")
    hashes = protocol["data_sha256"]
    if metrics["splits_sha256"] != {"train": hashes["sft_train.jsonl"], "val": hashes["sft_val.jsonl"]}:
        raise ValueError("The SFT run does not match the frozen data.")
    if metrics["pretrain_sha256"] != pretrained_hash or metrics["tokenizer_sha256"] != hashes["tokenizer.json"]:
        raise ValueError("The SFT run does not match the selected pretraining weight or tokenizer.")

    history = metrics["history"]
    steps_per_epoch = math.ceil(read_json(DATA / "manifest.json")["splits"]["sft_train"]["stories"]
                                / schedule["batch_size"])
    if [(row["epoch"], row["step"]) for row in history] != [
        (epoch, epoch * steps_per_epoch) for epoch in range(schedule["epochs"] + 1)
    ]:
        raise ValueError("The SFT metrics do not describe the complete run.")
    best = min(history[1:], key=lambda row: row["val_loss"])
    path = directory / "best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint["pretrain_sha256"] != pretrained_hash
            or checkpoint["model_config"] != protocol["v3"]["model_config"]
            or not metrics["best_epoch"] == best["epoch"] == checkpoint["epoch"]
            or (best["step"], best["val_loss"]) != (checkpoint["step"], checkpoint["val_loss"])):
        raise ValueError("The SFT checkpoint is not the recorded validation winner.")
    del checkpoint
    model, step = load_model(path, hashes["tokenizer.json"])
    rows = load_examples(DATA / "sft_val.jsonl", tokenizer, model.config.context_length)
    loss, tokens = response_loss(model, rows, tokenizer.token_to_id(PAD), schedule["batch_size"])
    if len(rows) != 400 or tokens != best["val_response_tokens"] or not math.isclose(loss, best["val_loss"], abs_tol=1e-6):
        raise ValueError("The reloaded SFT weight did not reproduce the validation results.")
    print(f"V3 SFT validation: epoch {best['epoch']}, step {step}, CE {loss:.4f}.")
    return {
        "best_step": step, "best_epoch": best["epoch"], "best_checkpoint_sha256": sha256(path),
        "validation_ce": loss, "validation_examples": len(rows), "validation_response_tokens": tokens,
        "peak_memory_mib": max(row["peak_memory_mib"] for row in history[1:]),
        "wall_seconds": wall_time(directory), "training_metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-only", action="store_true", help="Verify pretraining before starting SFT.")
    args = parser.parse_args()
    protocol = read_json(PROTOCOL)
    torch.set_num_threads(protocol["pretraining"]["threads"])
    torch.set_num_interop_threads(1)
    for filename, digest in protocol["data_sha256"].items():
        if sha256(DATA / filename) != digest:
            raise ValueError(f"The frozen data file has changed: {filename}")
    history, best = verify_pretraining(protocol, input_digests(DATA))
    path = RUN / "best.pt"
    pretrained_hash = sha256(path)
    if args.pretrain_only:
        model, _ = load_model(path, protocol["data_sha256"]["tokenizer.json"])
        full_validation(protocol, best, pretrained_hash, model)
        return

    full = read_json(FULL_VALIDATION)
    if (full["best_step"] != best["step"] or full["checkpoint_sha256"] != pretrained_hash
            or full["validation_sha256"] != protocol["data_sha256"]["pretrain_val.bin"]):
        raise ValueError("Re-run the full validation for the current pretrained checkpoint.")
    tokenizer = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    result = {
        "protocol_sha256": sha256(PROTOCOL),
        "model_config": protocol["v3"]["model_config"],
        "parameters": protocol["v3"]["parameters"],
        "pretraining": {
            "history": history, "best_step": best["step"], "best_checkpoint_sha256": pretrained_hash,
            "full_validation": full, "train_seconds": history[-1]["train_seconds"],
            "train_tokens_per_second": history[-1]["train_tokens"] / history[-1]["train_seconds"],
            "peak_memory_mib": max(row["peak_memory_mib"] for row in history),
            "wall_seconds": wall_time(RUN),
        },
        "sft": verify_sft(protocol, pretrained_hash, tokenizer),
    }
    write_json(RESULTS / "v3_training.json", result)


if __name__ == "__main__":
    main()
