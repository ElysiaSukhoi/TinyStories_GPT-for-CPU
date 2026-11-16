"""Record or verify the v1 baseline and the fixed 60-item blind review sample."""

import json
from pathlib import Path
from random import Random

import torch

from checkpoints import sha256
from scripts.story_task import INSTRUCTION, format_prompt
from train_sft import MAX_NEW_TOKENS

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "results" / "optimization_baseline_v1.json"
LOCKED_FILES = (
    "data/prepared/manifest.json",
    "data/prepared/tokenizer.json",
    "data/prepared/pretrain_train.bin",
    "data/prepared/pretrain_val.bin",
    "data/prepared/pretrain_val.jsonl",
    "data/prepared/sft_train.jsonl",
    "data/prepared/sft_val.jsonl",
    "data/prepared/sft_test.jsonl",
    "runs/pretrain/best.pt",
    "runs/pretrain/last.pt",
    "runs/sft/best.pt",
    "runs/sft/test_outputs.jsonl",
    "results/pretrain_metrics.json",
    "results/sft_training_metrics.json",
    "results/sft_test_metrics.json",
)


def read_json(path):
    return json.loads((PROJECT / path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with (PROJECT / path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def make_snapshot():
    files = {path: sha256(PROJECT / path) for path in LOCKED_FILES}
    manifest = read_json("data/prepared/manifest.json")
    test_metrics = read_json("results/sft_test_metrics.json")
    sft_metrics = read_json("results/sft_training_metrics.json")
    checkpoint = torch.load(PROJECT / "runs/pretrain/last.pt", map_location="cpu", weights_only=True)
    if any(files[f"data/prepared/{path}"] != digest
           for path, digest in checkpoint["settings"]["data_sha256"].items()):
        raise ValueError("The current prepared files differ from the v1 pretraining checkpoint.")
    expected = {
        "data/prepared/sft_test.jsonl": test_metrics["test_sha256"],
        "data/prepared/tokenizer.json": test_metrics["tokenizer_sha256"],
        "data/prepared/sft_train.jsonl": sft_metrics["splits_sha256"]["train"],
        "data/prepared/sft_val.jsonl": sft_metrics["splits_sha256"]["val"],
        "runs/pretrain/best.pt": test_metrics["pretrain_sha256"],
        "runs/sft/best.pt": test_metrics["sft_sha256"],
    }
    if any(files[path] != digest for path, digest in expected.items()):
        raise ValueError("The current files do not match the published v1 metric fingerprints.")

    test = read_jsonl("data/prepared/sft_test.jsonl")
    outputs = read_jsonl("runs/sft/test_outputs.jsonl")
    if len(test) != len(outputs) or len(test) != test_metrics["examples"] or len(test) != 400:
        raise ValueError("Expected 400 matched input and output records.")
    if len({row["story_id"] for row in test}) != len(test):
        raise ValueError("The held-out story IDs must be unique.")
    if any(any(actual[key] != value for key, value in row.items())
           for row, actual in zip(test, outputs, strict=True)):
        raise ValueError("The v1 output rows do not align with the held-out test set.")
    if test_metrics["decoding"] != "greedy" or test_metrics["max_new_tokens"] != MAX_NEW_TOKENS:
        raise ValueError("The v1 decoding settings disagree with the evaluation code.")
    if test_metrics["context_length"] != manifest["context_length"]:
        raise ValueError("The v1 evaluation context length disagrees with the data manifest.")

    selection_seed = 42
    rows = Random(selection_seed).sample(range(1, len(test) + 1), 60)
    return {
        "files_sha256": files,
        "dataset_revision": manifest["dataset_revision"],
        "model_config": checkpoint["model_config"],
        "pretrain_step": checkpoint["history"][-1]["step"],
        "sft_epoch": test_metrics["sft_epoch"],
        "sft_test_metrics": test_metrics["sft"],
        "evaluation": {
            "test_examples": len(test),
            "prompt_template": format_prompt(INSTRUCTION, "{opening}"),
            "decoding": test_metrics["decoding"],
            "context_length": test_metrics["context_length"],
            "max_new_tokens": MAX_NEW_TOKENS,
            "eos_id": manifest["special_token_ids"]["eos"],
        },
        "blind_review": {
            "selection_seed": selection_seed,
            "selection": "Random(seed).sample(range(1, 401), 60), kept in review order",
            "model_order_seed": 43,
            "cases": [
                {"review_order": order, "test_row": row, "story_id": test[row - 1]["story_id"]}
                for order, row in enumerate(rows, start=1)
            ],
        },
    }


def main():
    payload = make_snapshot()
    if OUTPUT.exists():
        if read_json(OUTPUT) != payload:
            raise ValueError("The current baseline differs from the frozen snapshot.")
        print(f"Verified the frozen baseline: {OUTPUT}")
        return

    with OUTPUT.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved the baseline and 60 review cases: {OUTPUT}")


if __name__ == "__main__":
    main()
