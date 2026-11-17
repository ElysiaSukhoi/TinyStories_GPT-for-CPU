"""Compare the three release models and create the final anonymous review sets."""

from argparse import Namespace
from random import Random

import torch
from tokenizers import Tokenizer

from checkpoints import load_model, sha256
from scripts.evaluate_optimization import read_json, write_jsonl
from scripts.freeze_baseline import read_jsonl
from scripts.freeze_v3_protocol import DATA, GOALS, PROTOCOL, RESULTS
from scripts.story_task import EOS, INSTRUCTION, PAD, encode_prompt, format_prompt, narrative_sentences
from train_pretrain import PROJECT, lock_run, write_json
from train_sft import MAX_NEW_TOKENS, greedy_batch, is_two_sentences, load_examples, score_model

OUTPUT = PROJECT / "runs/optimization/v3/comparison"
WEIGHTS = {
    "v1": PROJECT / "runs/sft/best.pt",
    "v2": PROJECT / "runs/optimization/b/sft/best.pt",
    "v3": PROJECT / "runs/optimization/v3/sft/best.pt",
}


def exact_repeats(predictions):
    count = 0
    for row in predictions:
        if row["two_complete_sentences"]:
            first, second = (sentence.casefold().rstrip(".?!")
                             for sentence, _ in narrative_sentences(row["text"].strip()))
            count += first == second
    return count


def add_goal_outputs(model, tokenizer, examples, name):
    prompts = [encode_prompt(tokenizer, INSTRUCTION, row["opening"]) for row in examples]
    eos, pad = (tokenizer.token_to_id(token) for token in (EOS, PAD))
    for start in range(0, len(prompts), 16):
        outputs = greedy_batch(model, prompts[start:start + 16], eos, pad)
        for row, ids in zip(examples[start:start + 16], outputs, strict=True):
            text = tokenizer.decode(ids)
            row["models"][name] = {
                "text": text, "token_ids": ids, "two_complete_sentences": is_two_sentences(text),
                "stopped_at_eos": bool(ids and ids[-1] == eos),
            }


def write_anonymous(name, cases, seed):
    rng = Random(seed)
    anonymous, key = [], []
    for case in cases:
        order = sorted(case["models"])
        rng.shuffle(order)
        mapping = dict(zip(("X", "Y", "Z"), order, strict=True))
        fields = {key: value for key, value in case.items() if key != "models"}
        anonymous.append({
            **fields, "answers": {label: case["models"][model]["text"] for label, model in mapping.items()},
        })
        key.append({"id": case["id"], "mapping": mapping})
    write_jsonl(OUTPUT / f"{name}_questions.jsonl", anonymous)
    write_json(OUTPUT / f"{name}_key.json", key)


def evaluate():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    protocol = read_json(PROTOCOL)
    frozen_path = PROJECT / protocol["review"]["baseline_file"]
    frozen = read_json(frozen_path)
    v3 = read_json(RESULTS / "v3_training.json")
    goal_spec = protocol["review"]["explicit_goal_set"]
    if (sha256(frozen_path) != protocol["review"]["baseline_sha256"]
            or sha256(GOALS) != goal_spec["sha256"] or v3["protocol_sha256"] != sha256(PROTOCOL)):
        raise ValueError("The final training or review protocol has changed.")
    for name in ("sft_test.jsonl", "tokenizer.json"):
        if sha256(DATA / name) != protocol["data_sha256"][name]:
            raise ValueError(f"The frozen input has changed: {name}")
    spec = protocol["evaluation"]
    if (spec["decoding"], spec["max_new_tokens"], spec["prompt_template"]) != (
        "greedy", MAX_NEW_TOKENS, format_prompt(INSTRUCTION, "{opening}")
    ):
        raise ValueError("The inference configuration differs from the frozen protocol.")
    tokenizer = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    test = load_examples(DATA / "sft_test.jsonl", tokenizer, spec["context_length"])
    goals = read_jsonl(GOALS)
    if len(test) != spec["test_examples"] or len(goals) != goal_spec["examples"]:
        raise ValueError("An evaluation set has the wrong number of examples.")
    test_results = [{**row, "models": {}} for row, _, _ in test]
    goal_results = [{**row, "models": {}} for row in goals]
    weight_hashes = {
        "v1": frozen["files_sha256"]["runs/sft/best.pt"],
        "v2": protocol["v2_b"]["weight_sha256"]["sft"],
        "v3": v3["sft"]["best_checkpoint_sha256"],
    }
    metrics = {}
    for name, path in WEIGHTS.items():
        if sha256(path) != weight_hashes[name]:
            raise ValueError(f"The frozen {name} checkpoint has changed.")
        model, step = load_model(path, protocol["data_sha256"]["tokenizer.json"])
        if model.config.context_length != spec["context_length"]:
            raise ValueError(f"Unexpected context length for {name}.")
        generated, scores = score_model(model, test, tokenizer, Namespace(batch_size=16), name.upper())
        for row, output in zip(test_results, generated, strict=True):
            row["models"][name] = output
        tokens = sum(len(output["token_ids"]) for output in generated)
        metrics[name] = {
            **scores, "exact_repeated_sentence_pairs": exact_repeats(generated),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint_step": step, "checkpoint_sha256": weight_hashes[name],
            "generation_tokens": tokens, "generation_tokens_per_second": tokens / scores["generation_seconds"],
        }
        add_goal_outputs(model, tokenizer, goal_results, name)

    original_cases = []
    blind = frozen["blind_review"]
    earlier_scores = read_json(RESULTS / "optimization_blind_scores.json")["cases"]
    for case, prior in zip(blind["cases"], earlier_scores, strict=True):
        row = test_results[case["test_row"] - 1]
        if row["story_id"] != case["story_id"] or prior["review_order"] != case["review_order"]:
            raise ValueError("A frozen review case no longer matches its story or order.")
        original_cases.append({
            "id": case["review_order"], "test_row": case["test_row"], "story_id": case["story_id"],
            "opening": row["input"], "has_explicit_goal": prior["ratings"]["X"]["goal_progress"] is not None,
            "models": row["models"],
        })
    write_jsonl(OUTPUT / "test_outputs.jsonl", test_results)
    write_jsonl(OUTPUT / "goal_outputs.jsonl", goal_results)
    write_anonymous("blind", original_cases, blind["model_order_seed"])
    write_anonymous("goal", goal_results, goal_spec["model_order_seed"])
    write_json(RESULTS / "v3_comparison.json", {
        "evaluation": spec, "threads": 4, "batch_size": 16,
        "test_sha256": protocol["data_sha256"]["sft_test.jsonl"], "goals_sha256": goal_spec["sha256"],
        "models": metrics,
    })
    print("Saved three-model test metrics and both anonymous review sets.", flush=True)


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".lock").open("a+b") as handle:
        lock_run(handle)
        evaluate()
