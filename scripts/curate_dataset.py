"""Drop bad samples from train.jsonl / val.jsonl.

Removes a row when its assistant reply:
  - Quotes a word that isn't in the target / heard / extra words (invention).
  - Mentions a class of error (missing / extra / substitution / mispronunciation)
    that isn't actually in the user payload.
  - Uses negative markers ("missed", "wrong", "instead", ...) when the user
    payload has no errors at all ("clean-when-perfect" violation).
  - Is too long (> 380 chars or > 6 sentences) — system prompt asks for 2-5
    short sentences; the slack allows rich multi-error attempts (praise +
    missing + substitution + a couple of distinct-sound hints).
  - Contains markdown (bold, italic, headers, bullet/numbered lists).
  - Repeats the same 4+-word phrase in adjacent positions.

The originals are backed up to *.jsonl.original before the curated versions
overwrite the source files. Run from the repo root:

    python scripts/curate_dataset.py
"""
import json
import re
import shutil
import sys
from pathlib import Path

# Allow `from app...` so we share the runtime's whole-word matcher.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.feedback.generator import _word_in_text

DATA_DIR = Path(__file__).resolve().parents[1] / "training" / "data"

_NEGATIVE_MARKERS = [
    "missed", "instead", "wrong", "mistake", "incorrect", "work on",
    "didn't", "did not", "forgot", "try not", "skipped",
]

_MISSING_HINTS = ["missing", "missed", "forgot", "skipped", "left out", "include", "didn't say"]
_EXTRA_HINTS = ["extra word", "added", "shouldn't have said", "doesn't belong"]
_SUB_HINTS = ["instead of", "said ", "rather than", "substituted"]
_MISPRON_HINTS = ["sound", "pronounce", "pronunciation", "mispron"]

_MARKDOWN_RE = re.compile(r"(\*\*|__|\*[^*]+\*|^\s*[-*]\s|^\s*\d+\.\s|^#{1,6}\s)", re.MULTILINE)
# A quote opens only when NOT preceded by a letter, and closes only when NOT
# followed by a letter. That excludes apostrophes inside contractions
# (I'm, don't, you're) so we don't treat half the reply as one giant quote.
_QUOTED_WORD_RE = re.compile(
    r"(?<![A-Za-z])['\"]([A-Za-z][A-Za-z'-]*)['\"](?![A-Za-z])"
)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+\s+")


def _quoted_words(text: str) -> set[str]:
    return {m.lower() for m in _QUOTED_WORD_RE.findall(text)}


def _required_words(payload: dict) -> list[str]:
    """Every concrete item the feedback is contractually required to name.

    Mirrors the runtime verifier in generator._ensure_coverage so the model is
    curated against the exact contract it will be judged by at request time.
    Extended to include the spelling-driven categories added in the taxonomy
    expansion (silent letters, hard/soft-c, clusters) — the system prompt
    requires the model to name each of those too.
    """
    required = list(payload.get("missing_words", []))
    required += list(payload.get("extra_words", []))
    required += [s["expected"] for s in payload.get("substitutions", [])]
    required += [m["word"] for m in payload.get("mispronounced", [])]
    required += [e["word"] for e in payload.get("silent_letter_errors", [])]
    required += [e["word"] for e in payload.get("hard_soft_c_errors", [])]
    required += [e["word"] for e in payload.get("cluster_errors", [])]
    return [w for w in required if w]


def _has_repetition(text: str) -> bool:
    """Flag if any 4-word phrase appears twice in close proximity.

    Uses 4-grams (not 3) so that two *distinct* anchor hints, which share the
    "<sound> sound like in <word>" template, don't collide: "the 'th' sound
    like in 'thin'" and "the 'r' sound like in 'red'" share the 3-gram "sound
    like in" but no 4-gram. Same-sound words are grouped into one sentence by
    the prompt, so genuine duplication (the same hint twice) is still caught.
    """
    tokens = re.findall(r"\w+", text.lower())
    seen: dict[tuple[str, ...], int] = {}
    for i in range(len(tokens) - 3):
        phrase = tuple(tokens[i : i + 4])
        if phrase in seen and (i - seen[phrase]) < 15:
            return True
        seen[phrase] = i
    return False


def _is_bad(payload: dict, feedback: str) -> str | None:
    """Return a reason string if the sample should be dropped, else None."""
    if not feedback or not feedback.strip():
        return "empty feedback"

    if _MARKDOWN_RE.search(feedback):
        return "markdown formatting"

    if len(feedback) > 380:
        return "too long (chars)"

    sentence_count = len([s for s in _SENTENCE_SPLIT_RE.split(feedback) if s.strip()])
    if sentence_count > 6:
        return "too long (sentences)"

    fb = feedback.lower()

    has_errors = bool(
        payload.get("missing_words")
        or payload.get("extra_words")
        or payload.get("substitutions")
        or payload.get("mispronounced")
    )
    if not has_errors:
        if any(marker in fb for marker in _NEGATIVE_MARKERS):
            return "negative marker on a perfect attempt"

    allowed = set(re.findall(r"[\w']+", payload["target_sentence"].lower()))
    allowed |= {s["heard"].lower() for s in payload.get("substitutions", [])}
    allowed |= {w.lower() for w in payload.get("extra_words", [])}
    # Phoneme labels and anchor words referenced in the analysis are legitimate
    # things to quote (e.g. "the 'th' sound like in 'thin'").
    for m in payload.get("mispronounced", []):
        allowed |= {p.lower() for p in m.get("weak_sounds", [])}
        if m.get("word"):
            allowed.add(m["word"].lower())
        if m.get("correct_hint"):
            for q in re.findall(r"'([A-Za-z]+)'", m["correct_hint"]):
                allowed.add(q.lower())
    # Same rule for substitution sound_hints (minimal pairs) — the quoted
    # anchor/phoneme labels there are contractually part of the reply.
    for s in payload.get("substitutions", []):
        if s.get("sound_hint"):
            for q in re.findall(r"'([A-Za-z]+)'", s["sound_hint"]):
                allowed.add(q.lower())
    # Same for the new spelling-driven categories. Each error's `hint` is a
    # full clause that the model must speak verbatim, so any words it quotes
    # (silent letters, anchor spellings, cluster names) are legitimate.
    for category in ("silent_letter_errors", "hard_soft_c_errors", "cluster_errors"):
        for e in payload.get(category, []):
            if e.get("word"):
                allowed.add(e["word"].lower())
            if e.get("silent_letter"):
                allowed.add(e["silent_letter"].lower())
            if e.get("cluster"):
                allowed.add(e["cluster"].lower())
            if e.get("hint"):
                for q in re.findall(r"'([A-Za-z]+)'", e["hint"]):
                    allowed.add(q.lower())
    invented = _quoted_words(feedback) - allowed
    if invented:
        return f"invented quoted words: {sorted(invented)}"

    # Drop any sample that fails to name a required item — the same omission
    # the runtime verifier would have to patch. We want the model trained only
    # on complete examples.
    omitted = [w for w in _required_words(payload) if not _word_in_text(feedback, w)]
    if omitted:
        return f"omits required words: {sorted(omitted)}"

    if _has_repetition(feedback):
        return "near-duplicate phrase"

    return None


def curate_file(path: Path) -> tuple[int, int, dict[str, int]]:
    backup = path.with_suffix(path.suffix + ".original")
    if not backup.exists():
        shutil.copy2(path, backup)

    kept: list[str] = []
    reasons: dict[str, int] = {}
    total = 0

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        row = json.loads(line)
        messages = row["messages"]
        payload = json.loads(messages[1]["content"])
        feedback = messages[2]["content"]

        reason = _is_bad(payload, feedback)
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        kept.append(line)

    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return total, len(kept), reasons


def main() -> None:
    if not DATA_DIR.exists():
        sys.exit(f"missing {DATA_DIR} - run generate_dataset.py first")

    for name in ("train.jsonl", "val.jsonl"):
        path = DATA_DIR / name
        if not path.exists():
            print(f"skip {name}: not found")
            continue
        total, kept, reasons = curate_file(path)
        dropped = total - kept
        pct = (dropped / total * 100) if total else 0
        print(f"\n=== {name} ===")
        print(f"  total       : {total}")
        print(f"  kept        : {kept}")
        print(f"  dropped     : {dropped} ({pct:.1f}%)")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    - {reason}: {count}")
    print("\noriginals backed up alongside as *.jsonl.original")


if __name__ == "__main__":
    main()
