"""Verify the frozen anonymous review and aggregate scores by model."""

from collections import Counter
import json

from checkpoints import sha256
from scripts.evaluate_optimization import write_json, write_jsonl
from train_pretrain import PROJECT


COMPARISON = PROJECT / "runs/optimization/v3/comparison"
RESULTS = PROJECT / "results"
MODELS = {"v1", "v2", "v3"}
LABELS = {"X", "Y", "Z"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect(review, section, basename, size):
    questions_path = COMPARISON / f"{basename}_questions.jsonl"
    if sha256(questions_path) != review["question_sha256"][section]:
        raise ValueError(f"{section}: anonymous questions differ from reviewed data")
    questions = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines()]
    keys = read_json(COMPARISON / f"{basename}_key.json")
    ratings = review[section]
    if any(len(rows) != size for rows in (questions, keys, ratings)):
        raise ValueError(f"{section}: expected {size} rows")
    ids = [row["id"] for row in questions]
    if len(set(ids)) != size or any([row["id"] for row in rows] != ids for rows in (keys, ratings)):
        raise ValueError(f"{section}: review IDs do not match")
    return zip(questions, keys, ratings, strict=True)


def main():
    review = read_json(RESULTS / "v3_review_scores.json")
    goals = {
        row["id"]: row
        for row in map(json.loads, (RESULTS / "v3_goal_cases.jsonl").read_text(encoding="utf-8").splitlines())
    }
    summary = {
        "reviewer": review["reviewer"],
        "question_sha256": review["question_sha256"],
        "blind_questions": 60,
        "goal_questions": 20,
        "models": {model: {"blind": Counter(), "goals": Counter(), "goals_by_category": {}}
                   for model in sorted(MODELS)},
    }
    details = []
    for section, basename, size in (("blind", "blind", 60), ("goals", "goal", 20)):
        for question, key, rating in collect(review, section, basename, size):
            answers = question.pop("answers")
            mapping = key["mapping"]
            scores = rating["scores"]
            if any(set(field) != LABELS for field in (answers, mapping, scores)) or set(mapping.values()) != MODELS:
                raise ValueError(f"{section} {question['id']}: invalid labels or model mapping")
            output = {"section": section, **question, "models": {}}
            if section == "goals":
                output["category"] = goals[question["id"]]["category"]
            for label, model in mapping.items():
                score = scores[label]
                counts = summary["models"][model][section]
                metrics = ["continuity", "no_repetition"] if section == "blind" else [
                    "goal_progress", "respects_constraint",
                ]
                if any(type(score[field]) is not bool for field in metrics):
                    raise ValueError(f"{section} {question['id']}: expected boolean scores")
                counts["total"] += 1
                for field in metrics:
                    counts[field] += score[field]
                if section == "blind":
                    applicable = question["has_explicit_goal"]
                    if applicable != (type(score["goal_progress"]) is bool):
                        raise ValueError(f"{question['id']}: goal score applicability differs")
                    counts["goals_applicable"] += applicable
                    counts["goal_progress"] += score["goal_progress"] is True
                else:
                    passed = score["goal_progress"] and score["respects_constraint"]
                    counts["both_pass"] += passed
                    categories = summary["models"][model]["goals_by_category"]
                    category = categories.setdefault(output["category"], Counter())
                    category["total"] += 1
                    category["both_pass"] += passed
                output["models"][model] = {"label": label, "text": answers[label], **score}
            details.append(output)
    write_json(RESULTS / "v3_review.json", summary)
    write_jsonl(RESULTS / "v3_review_outputs.jsonl", details)
    print(json.dumps(summary["models"], indent=2))


if __name__ == "__main__":
    main()
