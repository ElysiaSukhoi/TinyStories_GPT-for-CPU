"""Lock the final model settings, v2 comparison inputs, and explicit-goal test cases."""

import json
import unicodedata
from collections import Counter
from dataclasses import asdict

import torch
from tokenizers import Tokenizer

from checkpoints import sha256
from model import GPT, GPTConfig
from scripts.compare_pretraining import inspect_run
from scripts.freeze_baseline import read_json, read_jsonl
from scripts.story_task import EOS, INSTRUCTION, encode_prompt, format_prompt, narrative_sentences
from train_pretrain import PROJECT, input_digests
from train_sft import MAX_NEW_TOKENS

DATA = PROJECT / "data/prepared-v2"
RESULTS = PROJECT / "results"
GOALS = RESULTS / "v3_goal_cases.jsonl"
PROTOCOL = RESULTS / "v3_protocol.json"


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def check_goals(tokenizer, context):
    cases = read_jsonl(GOALS)
    categories = Counter(case["category"] for case in cases)
    if len(cases) != 20 or categories != {"lost_property": 7, "task_completion": 7, "explicit_constraint": 6}:
        raise ValueError("The goal set must have 20 cases in the prescribed categories.")
    openings, lengths = set(), []
    for number, case in enumerate(cases, start=1):
        text = case["opening"]
        sentences = list(narrative_sentences(text))
        if len(sentences) != 2 or sentences[-1][1] != len(text):
            raise ValueError(f"Expected a complete two-sentence opening in case {number}.")
        if case["id"] != f"goal-{number:02d}" or not all(case[field] for field in ("goal", "constraint", "criterion")):
            raise ValueError(f"Invalid or incomplete goal case {number}.")
        openings.add(normalize(text))
        lengths.append(len(encode_prompt(tokenizer, INSTRUCTION, text)))
    if len(openings) != len(cases) or max(lengths) + MAX_NEW_TOKENS > context:
        raise ValueError("Goal cases contain duplicate openings or leave too little context for the answer.")

    for split in ("sft_train", "sft_val", "sft_test"):
        if any(normalize(row["input"]) in openings for row in read_jsonl(DATA / f"{split}.jsonl")):
            raise ValueError(f"A goal opening already appears in {split}.")
    for split in ("pretrain_train", "pretrain_val"):
        for row in read_jsonl(DATA / f"{split}.jsonl"):
            if normalize(row["text"]).startswith(tuple(openings)):
                raise ValueError(f"A goal opening already begins a story in {split}.")

    return {
        "file": GOALS.relative_to(PROJECT).as_posix(), "sha256": sha256(GOALS),
        "examples": len(cases), "categories": dict(categories),
        "prompt_token_range": [min(lengths), max(lengths)], "model_order_seed": 44,
        "scoring": "Pass when the continuation makes concrete progress toward the stated goal and "
                   "respects the opening constraint. Record both judgments and one reason separately.",
    }


def make_protocol():
    frozen = read_json(RESULTS / "optimization_baseline_v1.json")
    ab = read_json(RESULTS / "optimization_pretrain_ab.json")
    compared = read_json(RESULTS / "optimization_sft_metrics.json")
    b = PROJECT / "runs/optimization/b"
    data_hash = input_digests(DATA)
    for filename in ("sft_train.jsonl", "sft_val.jsonl", "sft_test.jsonl"):
        data_hash[filename] = sha256(DATA / filename)
    if any(data_hash[name] != digest for name, digest in (ab["training_settings"]["data_sha256"]
                                                         | compared["data_sha256"]).items()):
        raise ValueError("The prepared data differ from the v2 experiment.")
    for name in ("pretrain_val.bin", "pretrain_val.jsonl", "sft_val.jsonl", "sft_test.jsonl", "tokenizer.json"):
        if data_hash[name] != frozen["files_sha256"][f"data/prepared/{name}"]:
            raise ValueError(f"{name} does not match the frozen v1 file.")
    torch.set_num_threads(4)
    steps = ab["steps"]
    settings, history, best = inspect_run(b, steps, ab["training_settings"]["data_sha256"])
    config_b = ab["runs"]["b"]["model_config"]
    if settings["model_config"] != config_b or history[-1]["train_tokens"] != ab["train_tokens_per_run"]:
        raise ValueError("Unexpected v2 B checkpoint configuration or training budget.")
    if best["step"] != ab["runs"]["b"]["best_step"]:
        raise ValueError("Unexpected v2 B selected pretraining step.")
    weights = {
        "pretrain": sha256(b / "best.pt"),
        "sft": sha256(b / "sft/best.pt"),
    }
    sft_b = compared["models"]["b"]
    if weights != {"pretrain": ab["runs"]["b"]["best_checkpoint_sha256"], "sft": sft_b["sft_checkpoint_sha256"]}:
        raise ValueError("A v2 B weight file differs from its recorded fingerprint.")

    spec = frozen["evaluation"]
    tokenizer = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    if spec != compared["evaluation"] or (spec["max_new_tokens"], spec["eos_id"], spec["prompt_template"]) != (
        MAX_NEW_TOKENS, tokenizer.token_to_id(EOS), format_prompt(INSTRUCTION, "{opening}")
    ):
        raise ValueError("The evaluation setup differs from the frozen v1/v2 protocol.")
    goal_set = check_goals(tokenizer, spec["context_length"])
    config_v3 = GPTConfig(**{**config_b, "n_layers": 6})
    parameters = sum(parameter.numel() for parameter in GPT(config_v3).parameters())
    if parameters != 5_820_416:
        raise ValueError(f"Unexpected v3 parameter count: {parameters:,}")
    return {
        "v2_b": {"model_config": config_b, "weight_sha256": weights},
        "v3": {"model_config": asdict(config_v3), "parameters": parameters},
        "data_sha256": data_hash,
        "pretraining": {
            **{key: value for key, value in ab["training_settings"].items() if key != "data_sha256"},
            "pilot_steps": 250, "steps": steps, "tokens": ab["train_tokens_per_run"], "eval_every": 500,
        },
        "sft": {**{key: sft_b["sft"][key] for key in ("seed", "threads", "batch_size", "learning_rate")}, "epochs": 3},
        "evaluation": spec,
        "review": {"baseline_file": "results/optimization_baseline_v1.json", "baseline_sha256": sha256(RESULTS / "optimization_baseline_v1.json"),
                   "blind_cases": len(frozen["blind_review"]["cases"]), "explicit_goal_set": goal_set},
    }


def main():
    protocol = make_protocol()
    if PROTOCOL.exists():
        if read_json(PROTOCOL) != protocol:
            raise ValueError("The current inputs differ from the frozen v3 protocol.")
        print(f"Verified {PROTOCOL}")
        return
    with PROTOCOL.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(protocol, indent=2) + "\n")
    print(f"Saved {PROTOCOL}")


if __name__ == "__main__":
    main()
