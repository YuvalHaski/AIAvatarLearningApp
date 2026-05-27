"""Evaluate the fine-tuned Stage 2 feedback model on the held-out val set.

Two kinds of checks:
  1. Faithfulness (deterministic, our code) - because we control the structured
     input, we can verify the generated feedback actually talks about the
     prioritized errors and does NOT invent errors that aren't there.
  2. Quality - optional BERTScore against the reference feedback in val.jsonl.

Point this at a running vLLM server (the same way the app calls the model):
    python training/evaluate.py \
        --endpoint http://localhost:8000/v1 --model-name feedback-model
"""
import argparse
import json
import re

import httpx

# Words/phrases that imply a correction; a perfect attempt must not use them.
_NEGATIVE_MARKERS = [
    "missed", "instead", "wrong", "mistake", "incorrect", "work on",
    "didn't", "did not", "forgot", "try not", "skipped",
]


def _quoted_words(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"'([^']+)'", text)}


def faithfulness(payload: dict, feedback: str) -> dict:
    """Score one prediction against the structured report it was given."""
    fb = feedback.lower()

    focus_words = [p["word"].lower() for p in payload["focus_points"] if p.get("word")]
    covered = sum(1 for w in focus_words if w in fb)
    coverage = covered / len(focus_words) if focus_words else 1.0

    has_errors = bool(
        payload["missing_words"]
        or payload["extra_words"]
        or payload["substitutions"]
        or payload["mispronounced"]
    )
    clean_when_perfect = True
    if not has_errors:
        clean_when_perfect = not any(marker in fb for marker in _NEGATIVE_MARKERS)

    # Any word the feedback puts in quotes must be a real target/heard word.
    allowed = set(re.findall(r"[\w']+", payload["target_sentence"].lower()))
    allowed |= {s["heard"].lower() for s in payload["substitutions"]}
    allowed |= {w.lower() for w in payload["extra_words"]}
    invented = _quoted_words(feedback) - allowed

    passed = coverage >= 0.5 and clean_when_perfect and not invented
    return {
        "coverage": coverage,
        "clean_when_perfect": clean_when_perfect,
        "invented_words": sorted(invented),
        "passed": passed,
    }


def generate(endpoint: str, model_name: str, messages: list[dict]) -> str:
    response = httpx.post(
        endpoint.rstrip("/") + "/chat/completions",
        json={
            "model": model_name,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 160,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return (response.json()["choices"][0]["message"]["content"] or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--model-name", default="feedback-model")
    parser.add_argument("--val", default="training/data/val.jsonl")
    parser.add_argument("--bertscore", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.val, encoding="utf-8")]

    predictions, references, faith_results = [], [], []
    for row in rows:
        messages = row["messages"]
        prompt = messages[:2]  # system + user
        reference = messages[2]["content"]
        payload = json.loads(messages[1]["content"])

        prediction = generate(args.endpoint, args.model_name, prompt)
        predictions.append(prediction)
        references.append(reference)
        faith_results.append(faithfulness(payload, prediction))

    n = len(rows)
    pass_rate = sum(r["passed"] for r in faith_results) / n
    avg_coverage = sum(r["coverage"] for r in faith_results) / n
    n_invented = sum(1 for r in faith_results if r["invented_words"])

    print(f"samples evaluated      : {n}")
    print(f"faithfulness pass rate : {pass_rate:.1%}")
    print(f"avg focus-point cover  : {avg_coverage:.1%}")
    print(f"samples inventing words: {n_invented}")

    if args.bertscore:
        from bert_score import score as bert_score

        _, _, f1 = bert_score(predictions, references, lang="en", verbose=False)
        print(f"BERTScore F1 (vs ref)  : {f1.mean().item():.3f}")

    failures = [
        (i, r) for i, r in enumerate(faith_results) if not r["passed"]
    ]
    if failures:
        print(f"\n{len(failures)} faithfulness failures (first 5):")
        for i, r in failures[:5]:
            print(f"  [{i}] {r}")
            print(f"       feedback: {predictions[i]}")


if __name__ == "__main__":
    main()
