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
import sys
from pathlib import Path

import httpx

# Share the runtime's whole-word matcher so "covered" here means exactly what
# the production verifier means by it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.feedback.generator import _word_in_text

# Words/phrases that imply a correction; a perfect attempt must not use them.
_NEGATIVE_MARKERS = [
    "missed", "instead", "wrong", "mistake", "incorrect", "work on",
    "didn't", "did not", "forgot", "try not", "skipped",
]


def _quoted_words(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"'([^']+)'", text)}


def _required_words(payload: dict) -> list[str]:
    """Every concrete item the feedback must name (see generator verifier)."""
    required = list(payload["missing_words"])
    required += list(payload["extra_words"])
    required += [s["expected"] for s in payload["substitutions"]]
    required += [m["word"] for m in payload["mispronounced"]]
    return [w for w in required if w]


def faithfulness(payload: dict, feedback: str) -> dict:
    """Score one prediction against the structured report it was given."""
    fb = feedback.lower()

    required = _required_words(payload)
    omitted = [w for w in required if not _word_in_text(feedback, w)]
    coverage = (len(required) - len(omitted)) / len(required) if required else 1.0

    has_errors = bool(
        payload["missing_words"]
        or payload["extra_words"]
        or payload["substitutions"]
        or payload["mispronounced"]
    )
    clean_when_perfect = True
    if not has_errors:
        clean_when_perfect = not any(marker in fb for marker in _NEGATIVE_MARKERS)

    # Any word the feedback puts in quotes must be a real target/heard word or
    # an anchor from a mispronunciation hint (e.g. 'th', 'thin').
    allowed = set(re.findall(r"[\w']+", payload["target_sentence"].lower()))
    allowed |= {s["heard"].lower() for s in payload["substitutions"]}
    allowed |= {w.lower() for w in payload["extra_words"]}
    for m in payload["mispronounced"]:
        allowed |= {p.lower() for p in m.get("weak_sounds", [])}
        if m.get("correct_hint"):
            for q in re.findall(r"'([A-Za-z]+)'", m["correct_hint"]):
                allowed.add(q.lower())
    invented = _quoted_words(feedback) - allowed

    passed = not omitted and clean_when_perfect and not invented
    return {
        "coverage": coverage,
        "omitted_words": sorted(omitted),
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
    n_omitting = sum(1 for r in faith_results if r["omitted_words"])

    print(f"samples evaluated      : {n}")
    print(f"faithfulness pass rate : {pass_rate:.1%}")
    print(f"avg required-item cover: {avg_coverage:.1%}")
    print(f"samples omitting items : {n_omitting} ({n_omitting / n:.1%})")
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
