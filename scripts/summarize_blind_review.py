"""Unblind completed review scores and summarize the results by model."""

from checkpoints import sha256
from scripts.evaluate_optimization import COMPARISON, RESULTS, read_json
from train_pretrain import write_json

DIMENSIONS = ("continuity", "goal_progress", "no_repetition")
LABELS = {"X", "Y", "Z"}
MODELS = {"v1", "a", "b"}


def main():
    score_path = RESULTS / "optimization_blind_scores.json"
    scores = read_json(score_path)
    if scores["questions_sha256"] != sha256(COMPARISON / "blind_questions.jsonl"):
        raise ValueError("The anonymous questions differ from the scored version.")
    key = read_json(COMPARISON / "blind_key.json")
    if len(scores["cases"]) != 60 or len(key) != 60:
        raise ValueError("Expected ratings for all 60 frozen questions.")
    totals = {
        model: {dimension: {"passed": 0, "scored": 0} for dimension in DIMENSIONS}
        for model in sorted(MODELS)
    }
    for order, (rated, labels) in enumerate(zip(scores["cases"], key, strict=True), start=1):
        if rated["review_order"] != order or labels["review_order"] != order:
            raise ValueError("Review order differs from the frozen question order.")
        mapping = labels["mapping"]
        ratings = rated["ratings"]
        if set(ratings) != LABELS or set(mapping) != LABELS or set(mapping.values()) != MODELS:
            raise ValueError(f"Invalid labels for case {order}.")
        for label, rating in ratings.items():
            for dimension in DIMENSIONS:
                value = rating[dimension]
                if dimension == "goal_progress" and value is None:
                    continue
                if type(value) is not bool:
                    raise ValueError(f"Invalid {dimension} rating in case {order}.")
                if not value and not rating["note"].strip():
                    raise ValueError(f"Missing failure note in case {order}, label {label}.")
                tally = totals[mapping[label]][dimension]
                tally["scored"] += 1
                tally["passed"] += int(value)
        if len({rated["ratings"][label]["goal_progress"] is None for label in LABELS}) != 1:
            raise ValueError(f"Goal applicability must match across all candidates in case {order}.")

    for dimensions in totals.values():
        for tally in dimensions.values():
            tally["rate"] = tally["passed"] / tally["scored"] if tally["scored"] else None
    summary = {
        "reviewer": scores["reviewer"], "questions": 60,
        "anonymous_scores_sha256": sha256(score_path),
        "questions_sha256": scores["questions_sha256"],
        "models": totals,
        "model_order_by_case": key,
    }
    path = RESULTS / "optimization_blind_review.json"
    write_json(path, summary)
    print(summary["models"])
    print(path)


if __name__ == "__main__":
    main()
