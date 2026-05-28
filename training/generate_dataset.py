"""Build the (ErrorReport -> feedback) training dataset for the Stage 2 model.

Two steps:
  1. Synthesize a wide spread of ErrorReports by constructing fake AsrResults and
     running them through the *real* Stage 1 analyzer. This guarantees the inputs
     are exactly the shape the model will see in production.
  2. For each report, bootstrap an "ideal" feedback string with a strong model
     (GPT-4) - distillation. Hand-curate the output afterwards: fix tone, enforce
     brevity, and DELETE any sample whose feedback invents an error.

Target sentences come from the `sentences` table in the app database, so the
model trains on exactly the prompts users will practice. Override with
--sentences-file if you want to train without a live DB connection (e.g. in
Colab) by exporting them to a text file first.

Output: JSONL files where each line is {"messages": [system, user, assistant]},
ready for trl's SFTTrainer.

Usage:
    python training/generate_dataset.py --out training/data --limit 0
    (set OPENAI_API_KEY and DATABASE_URL; --limit 0 means no cap)
"""
import argparse
import json
import random
import sys
from pathlib import Path

# Allow `from app...` when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.database import SessionLocal
from app.models.domain import Sentence
from app.schemas.asr import AsrResult, PhonemeScore, PronunciationScores, WordResult
from app.services.feedback.alignment import tokenize
from app.services.feedback.analysis import analyze
from app.services.feedback.generator import build_model_messages


_PRONUNCIATION_FILE = Path(__file__).parent / "sentences_pronunciation.txt"


def _read_sentences_file(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_target_sentences(sentences_file: str | None) -> list[str]:
    """Pull the target sentences from the DB (or a fallback file), then merge
    in `training/sentences_pronunciation.txt` if it exists.

    The DB holds the user-facing practice content; the pronunciation file adds
    phonetically-rich sentences curated specifically to make Azure's phoneme
    scorer flag real difficulties (th/r/l/v/w/etc.). Together they give the
    feedback model a broader spread of ErrorReports to learn from.

    `--sentences-file` overrides the DB for environments without DB access
    (e.g. Colab) — export with `--export-sentences` first.

    Deduped by normalized text (case + trailing-punct insensitive).
    """
    if sentences_file:
        raw = _read_sentences_file(Path(sentences_file))
    else:
        with SessionLocal() as db:
            raw = [row[0] for row in db.query(Sentence.text).all() if row[0]]

    extras: list[str] = []
    if _PRONUNCIATION_FILE.exists():
        extras = _read_sentences_file(_PRONUNCIATION_FILE)
        print(
            f"merging {len(extras)} pronunciation-focused sentences "
            f"from {_PRONUNCIATION_FILE.name}",
            flush=True,
        )

    seen: set[str] = set()
    unique: list[str] = []
    for text in (*raw, *extras):
        key = text.strip().rstrip(".!?").casefold()
        if key and key not in seen:
            seen.add(key)
            unique.append(text)
    return unique

# Commonly-difficult phonemes we may flag as "weak", paired with the orthography
# that produces them. We only assign a phoneme to a word if that word's spelling
# actually contains the trigger — otherwise the training data ends up saying
# things like "weak 'th' sound in 'soup'", which teaches the model nonsense.
_PHONEME_TRIGGERS: list[tuple[str, str]] = [
    ("th", "th"),
    ("sh", "sh"),
    ("ch", "ch"),
    ("ng", "ng"),
    ("r", "r"),
    ("l", "l"),
    ("v", "v"),
    ("w", "w"),
    ("z", "z"),
]


def _plausible_weak_phonemes(word: str) -> list[str]:
    """Phonemes whose orthographic trigger appears in `word`."""
    w = word.lower()
    found = [ph for ph, trigger in _PHONEME_TRIGGERS if trigger in w]
    # "s" only when not part of "sh" (which we already covered).
    if "s" in w and "sh" not in w:
        found.append("s")
    return found


def _pick_mispronunciation(tokens: list[str]) -> tuple[int | None, list[str] | None]:
    """Pick a word index + a weak phoneme set that actually fits that word."""
    eligible = [(i, _plausible_weak_phonemes(t)) for i, t in enumerate(tokens)]
    eligible = [(i, cands) for i, cands in eligible if cands]
    if not eligible:
        return None, None
    i, candidates = random.choice(eligible)
    return i, [random.choice(candidates)]


def _pick_pattern(tokens: list[str]) -> tuple[list[int], str | None]:
    """Find a phoneme present in 2+ words — the basis for a recurring pattern."""
    by_phoneme: dict[str, list[int]] = {}
    for i, tok in enumerate(tokens):
        for ph in _plausible_weak_phonemes(tok):
            by_phoneme.setdefault(ph, []).append(i)
    candidates = [(ph, idxs) for ph, idxs in by_phoneme.items() if len(idxs) >= 2]
    if not candidates:
        return [], None
    ph, idxs = random.choice(candidates)
    return idxs[:2], ph


def _scores(accuracy, fluency, completeness, prosody=None):
    return PronunciationScores(
        accuracy=accuracy,
        fluency=fluency,
        completeness=completeness,
        prosody=accuracy if prosody is None else prosody,
        pronunciation=accuracy,
    )


def _words(tokens, *, accuracy=96, mispron_index=None, weak_phonemes=None):
    words = []
    for i, tok in enumerate(tokens):
        if mispron_index is not None and i == mispron_index:
            phonemes = [
                PhonemeScore(phoneme=p, accuracy_score=24)
                for p in (weak_phonemes or ["s"])
            ]
            words.append(
                WordResult(
                    word=tok, accuracy_score=38,
                    error_type="Mispronunciation", phonemes=phonemes,
                )
            )
        else:
            words.append(
                WordResult(
                    word=tok, accuracy_score=accuracy, error_type="None",
                    phonemes=[PhonemeScore(phoneme="g", accuracy_score=accuracy)],
                )
            )
    return words


def synthesize(target: str) -> list[AsrResult]:
    """Produce a spread of plausible AsrResults for one target sentence."""
    tokens = tokenize(target)
    if len(tokens) < 3:
        return []
    results: list[AsrResult] = []

    # 1. Perfect attempt.
    results.append(
        AsrResult(
            recognized_text=target, scores=_scores(98, 96, 100),
            words=_words(tokens, accuracy=98),
        )
    )

    # 2. One missing word (drop a middle word).
    drop = len(tokens) // 2
    kept = tokens[:drop] + tokens[drop + 1:]
    results.append(
        AsrResult(
            recognized_text=" ".join(kept), scores=_scores(90, 85, 80),
            words=_words(kept, accuracy=92),
        )
    )

    # 3. One extra word.
    extended = tokens[:drop] + ["really"] + tokens[drop:]
    results.append(
        AsrResult(
            recognized_text=" ".join(extended), scores=_scores(91, 84, 100),
            words=_words(extended, accuracy=90),
        )
    )

    # 4. One substitution.
    swapped = list(tokens)
    swapped[drop] = "something"
    results.append(
        AsrResult(
            recognized_text=" ".join(swapped), scores=_scores(85, 82, 95),
            words=_words(swapped, accuracy=86),
        )
    )

    # 5. One mispronounced word. Skip if no word in the sentence has a
    #    plausible weak phoneme (otherwise we'd teach the model nonsense).
    mispron_idx, weak = _pick_mispronunciation(tokens)
    if mispron_idx is not None:
        results.append(
            AsrResult(
                recognized_text=target, scores=_scores(62, 80, 100),
                words=_words(
                    tokens, accuracy=90, mispron_index=mispron_idx,
                    weak_phonemes=weak,
                ),
            )
        )

    # 6. Recurring pattern: same weak phoneme across two different words.
    pattern_idxs, pattern_ph = _pick_pattern(tokens)
    if pattern_ph is not None:
        multi = _words(
            tokens, accuracy=88,
            mispron_index=pattern_idxs[0], weak_phonemes=[pattern_ph],
        )
        multi[pattern_idxs[1]] = WordResult(
            word=tokens[pattern_idxs[1]], accuracy_score=40,
            error_type="Mispronunciation",
            phonemes=[PhonemeScore(phoneme=pattern_ph, accuracy_score=22)],
        )
        results.append(
            AsrResult(
                recognized_text=target, scores=_scores(55, 72, 100), words=multi,
            )
        )

    # 7. Severe attempt: low everything, several words missing.
    half = tokens[: max(1, len(tokens) // 2)]
    results.append(
        AsrResult(
            recognized_text=" ".join(half), scores=_scores(45, 40, 50),
            words=_words(half, accuracy=50),
        )
    )
    return results


def bootstrap_feedback(messages: list[dict], client) -> str:
    """Ask GPT-4 for the ideal assistant reply given the system+user messages."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.4,
        max_tokens=160,
    )
    return (response.choices[0].message.content or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="training/data")
    parser.add_argument("--limit", type=int, default=0, help="cap samples (0 = all)")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="skip GPT calls; emit reports with empty feedback")
    parser.add_argument("--sentences-file", default=None,
                        help="read sentences from a text file (one per line) "
                             "instead of the DB; useful for Colab runs")
    parser.add_argument("--export-sentences", default=None,
                        help="write the DB's sentences to this path and exit "
                             "(use the output later with --sentences-file)")
    args = parser.parse_args()

    if args.export_sentences:
        sentences = load_target_sentences(None)
        Path(args.export_sentences).write_text(
            "\n".join(sentences) + "\n", encoding="utf-8"
        )
        print(f"exported {len(sentences)} sentences -> {args.export_sentences}")
        return

    random.seed(42)

    target_sentences = load_target_sentences(args.sentences_file)
    if not target_sentences:
        raise SystemExit(
            "No target sentences found. Seed the `sentences` table or pass "
            "--sentences-file pointing at a non-empty text file."
        )

    client = None
    if not args.no_bootstrap:
        from openai import OpenAI
        client = OpenAI()

    samples: list[dict] = []
    total = len(target_sentences)
    for i, target in enumerate(target_sentences, start=1):
        print(f"[{i}/{total}] {target}", flush=True)
        for asr in synthesize(target):
            report = analyze(target, asr)
            messages = build_model_messages(report)
            feedback = "" if args.no_bootstrap else bootstrap_feedback(messages, client)
            samples.append({"messages": messages + [
                {"role": "assistant", "content": feedback}
            ]})
            if args.limit and len(samples) >= args.limit:
                break
        if args.limit and len(samples) >= args.limit:
            break

    random.shuffle(samples)
    n_val = int(len(samples) * args.val_fraction)
    val, train = samples[:n_val], samples[n_val:]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} samples -> {path}")

    print(
        "\nNext: hand-curate training/data/train.jsonl - fix tone, enforce "
        "brevity, and DELETE any sample whose feedback mentions an error not "
        "present in the user message. Aim for ~800-1500 curated pairs."
    )


if __name__ == "__main__":
    main()
