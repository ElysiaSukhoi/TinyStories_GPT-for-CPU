"""Evaluate the three SFT models and prepare anonymous review questions."""

import json
from argparse import Namespace
from random import Random

import torch
from tokenizers import Tokenizer

from checkpoints import load_model, sha256
from scripts.story_task import INSTRUCTION, format_prompt, narrative_sentences
from train_pretrain import PROJECT, lock_run, write_atomic, write_json
from train_sft import MAX_NEW_TOKENS, load_examples, score_model

RUN = PROJECT / "runs" / "optimization"
DATA = PROJECT / "data" / "prepared-v2"
RESULTS = PROJECT / "results"
COMPARISON = RUN / "comparison"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path, rows):
    contents = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    write_atomic(path, lambda handle: handle.write(contents))


def prepare_review(generated, frozen):
    rng = Random(frozen["model_order_seed"])
    prompts, key = [], []
    for item in frozen["cases"]:
        row = generated[item["test_row"] - 1]
        if item["story_id"] != row["story_id"]:
            raise ValueError("A review case no longer matches its frozen story ID.")
        model_order = sorted(row["models"])
        rng.shuffle(model_order)
        mapping = dict(zip(("X", "Y", "Z"), model_order, strict=True))
        prompts.append({
            **item, "opening": row["input"],
            "candidates": {label: row["models"][model]["text"] for label, model in mapping.items()},
        })
        key.append({"review_order": item["review_order"], "mapping": mapping})
    write_jsonl(COMPARISON / "blind_questions.jsonl", prompts)
    write_json(COMPARISON / "blind_key.json", key)


def evaluate():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    frozen = read_json(RESULTS / "optimization_baseline_v1.json")
    ab = read_json(RESULTS / "optimization_pretrain_ab.json")["runs"]
    data = {
        filename: sha256(DATA / filename)
        for filename in ("tokenizer.json", "sft_train.jsonl", "sft_val.jsonl", "sft_test.jsonl")
    }
    for filename in ("tokenizer.json", "sft_val.jsonl", "sft_test.jsonl"):
        if data[filename] != frozen["files_sha256"][f"data/prepared/{filename}"]:
            raise ValueError(f"{filename} diverged from the frozen baseline.")
    spec = frozen["evaluation"]
    if (spec["decoding"], spec["max_new_tokens"], spec["prompt_template"]) != (
        "greedy", MAX_NEW_TOKENS, format_prompt(INSTRUCTION, "{opening}")
    ):
        raise ValueError("The current decoding constraints differ from the frozen protocol.")

    tokenizer = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    rows = load_examples(DATA / "sft_test.jsonl", tokenizer, spec["context_length"])
    if len(rows) != spec["test_examples"]:
        raise ValueError("Unexpected number of frozen SFT test rows.")
    generated = [{**record, "models": {}} for record, _, _ in rows]
    configs = {}
    sft_settings = None
    for name in ("v1", "a", "b"):
        is_baseline = name == "v1"
        directory = PROJECT / "runs/sft" if is_baseline else RUN / name / "sft"
        checkpoint = directory / "best.pt"
        checkpoint_hash = sha256(checkpoint)
        training = read_json(RESULTS / "sft_training_metrics.json" if is_baseline else directory / "training_metrics.json")
        if is_baseline:
            if checkpoint_hash != frozen["files_sha256"]["runs/sft/best.pt"]:
                raise ValueError("The v1 weights have changed.")
            pretrain_history = read_json(RESULTS / "pretrain_metrics.json")
            pretrain_seconds = pretrain_history[-1]["train_seconds"]
            pretrain_peak = max(item["peak_memory_mib"] for item in pretrain_history)
        else:
            if training["pretrain_sha256"] != ab[name]["best_checkpoint_sha256"]:
                raise ValueError(f"SFT {name} was initialized from a different pretrained checkpoint.")
            if training["splits_sha256"] != {"train": data["sft_train.jsonl"], "val": data["sft_val.jsonl"]}:
                raise ValueError(f"SFT {name} used different training data.")
            common = tuple(training[key] for key in ("threads", "batch_size", "seed", "learning_rate"))
            if sft_settings is not None and common != sft_settings:
                raise ValueError("A and B used different SFT settings.")
            sft_settings = common
            if training["history"][-1]["epoch"] != 3 or training["history"][-1]["step"] != 1500:
                raise ValueError(f"SFT {name} needs a complete three-epoch run.")
            pretrain_seconds = ab[name]["train_seconds"]
            pretrain_peak = ab[name]["peak_training_memory_mib"]
        model, step = load_model(checkpoint, data["tokenizer.json"])
        if model.config.context_length != spec["context_length"]:
            raise ValueError(f"{name} does not use the frozen context length.")
        predictions, metrics = score_model(model, rows, tokenizer, Namespace(batch_size=16), name.upper())
        output_tokens = sum(len(item["token_ids"]) for item in predictions)
        repeat_count = 0
        for item in predictions:
            if item["two_complete_sentences"]:
                sentences = [part.casefold().rstrip(".?!") for part, _ in narrative_sentences(item["text"].strip())]
                repeat_count += sentences[0] == sentences[1]
        metrics["exact_repeated_sentence_pairs"] = repeat_count
        for row, output in zip(generated, predictions, strict=True):
            row["models"][name] = output
        configs[name] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "sft_checkpoint_sha256": checkpoint_hash, "sft_best_step": step,
            "pretrain_train_seconds": pretrain_seconds, "pretrain_peak_mib": pretrain_peak,
            "sft_peak_mib": training["history"][-1].get("peak_memory_mib"),
            "sft": training, "metrics": metrics,
            "generation_tokens": output_tokens,
            "generation_tokens_per_second": output_tokens / metrics["generation_seconds"],
        }

    write_jsonl(COMPARISON / "test_outputs.jsonl", generated)
    prepare_review(generated, frozen["blind_review"])
    metrics_path = RESULTS / "optimization_sft_metrics.json"
    write_json(metrics_path, {
        "evaluation": spec, "threads": 4, "batch_size": 16,
        "data_sha256": data, "models": configs,
    })
    print(f"Saved metrics and anonymous review cases: {metrics_path}")


if __name__ == "__main__":
    COMPARISON.mkdir(parents=True, exist_ok=True)
    with (COMPARISON / ".lock").open("a+b") as handle:
        lock_run(handle)
        evaluate()
