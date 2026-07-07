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
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Allow `from app...` when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.asr import AsrResult, PhonemeScore, PronunciationScores, WordResult
from app.services.feedback.alignment import tokenize
from app.services.feedback.analysis import analyze
from app.services.feedback.generator import build_model_messages
from app.services.feedback.phoneme_hints import (
    PHONEME_TRIGGERS as _PHONEME_TRIGGERS,
    plausible_weak_phonemes as _plausible_weak_phonemes,
)
from app.services.feedback.word_hints import (
    cluster_hint_for as _cluster_hint_for,
    hard_soft_c_for as _hard_soft_c_for,
    pick_confusable as _pick_confusable,
    silent_letter_for as _silent_letter_for,
    word_vowel_for as _word_vowel_for,
    MINIMAL_PAIR_CONFUSABLES as _MINIMAL_PAIRS,
)


_PRONUNCIATION_FILE = Path(__file__).parent / "sentences_pronunciation.txt"

# Number of GPT calls to run in parallel. 2 concurrent × ~1s per gpt-4o-mini
# response ≈ 2 req/sec × 780 tokens ≈ 94k TPM, well below tier-1 mini's 200k
# TPM cap with headroom for bursty batches. Concurrency 3+ was chronically
# saturating the window on tier 1 and triggering SDK-level backoff that
# dragged sustained throughput to under 1 row/sec. On tier 2+ this can go to
# 8-10.
_CONCURRENCY = 2
# Seconds to wait between batches when we hit a rate limit. Only applies to
# the outer safety retry below — the SDK already retries with exponential
# backoff (max_retries=6) on each individual call, this handles the case
# where a whole batch runs the account TPM window dry.
_RATE_LIMIT_SLEEP = 20.0


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
        # Imported lazily so the module stays usable in environments without a
        # DB / sqlalchemy (e.g. Colab), where sentences come from --sentences-file.
        from app.core.database import SessionLocal
        from app.models.domain import Sentence

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

def _pick_mispronunciation(tokens: list[str]) -> tuple[int | None, list[str] | None]:
    """Pick a word index + a weak phoneme set that actually fits that word."""
    eligible = [(i, _plausible_weak_phonemes(t)) for i, t in enumerate(tokens)]
    eligible = [(i, cands) for i, cands in eligible if cands]
    if not eligible:
        return None, None
    i, candidates = random.choice(eligible)
    return i, [random.choice(candidates)]


def _pick_mispronunciations(
    tokens: list[str], n: int
) -> list[tuple[int, list[str]]]:
    """Pick up to `n` distinct word indices that each have a fitting weak
    phoneme. Used to synthesize attempts with several mispronunciations at once
    — the multi-error case the model previously dropped words on."""
    eligible = [(i, _plausible_weak_phonemes(t)) for i, t in enumerate(tokens)]
    eligible = [(i, cands) for i, cands in eligible if cands]
    random.shuffle(eligible)
    return [(i, [random.choice(cands)]) for i, cands in eligible[:n]]


def _find_confusable_word(
    tokens: list[str], candidates: list[int] | None = None,
) -> tuple[int | None, str | None]:
    """Find (index, heard_word) for a token that has a curated confusable.

    `candidates` restricts the search to specific indices; when None, all
    tokens are eligible. Returns (None, None) when no token has a confusable
    — callers should skip the substitution scenario rather than emit
    "something" (the old bug this replaces)."""
    indices = candidates if candidates is not None else list(range(len(tokens)))
    random.shuffle(indices)
    for i in indices:
        result = _pick_confusable(tokens[i])
        if result is not None:
            heard, _ = result
            return i, heard
    return None, None


def _find_word_matching(
    tokens: list[str], predicate,
) -> int | None:
    """Return the index of the first token satisfying `predicate`, or None."""
    hits = [i for i, t in enumerate(tokens) if predicate(t)]
    if not hits:
        return None
    return random.choice(hits)


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
                    # Filler phoneme is always strong so it never leaks into a
                    # word's weak_phonemes. A low-scoring word (e.g. the severe
                    # attempt) is still flagged by its word score, and analysis
                    # then derives a real hint from the word's spelling instead
                    # of a meaningless "g"/None.
                    phonemes=[PhonemeScore(phoneme="g", accuracy_score=96)],
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

    # 4. One substitution — with a phonetically PLAUSIBLE confusion word.
    #    Old code used the literal token "something", which Azure would never
    #    return. Now we search for any token in the sentence that has a
    #    curated confusable (word_hints.pick_confusable) and swap it. When
    #    the confusable is a minimal pair, Stage 1 will attach the sound_hint
    #    automatically via sound_hint_for_substitution.
    sub_idx, sub_heard = _find_confusable_word(tokens)
    if sub_idx is not None and sub_heard is not None:
        swapped = list(tokens)
        swapped[sub_idx] = sub_heard
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

    # 8. Hard combo: a substitution PLUS two or three mispronounced words, with
    #    every other word still present. This is the case the model handled
    #    worst (it dropped the mispronunciations when a "louder" error was also
    #    there), so we deliberately over-represent it in the training mix.
    #    The substitution now uses a curated confusable — again no more
    #    "something" placeholder.
    combo = _pick_mispronunciations(tokens, 3)
    if len(combo) >= 2:
        mis_indices = {i for i, _ in combo}
        available = [i for i in range(len(tokens)) if i not in mis_indices]
        combo_sub_idx, combo_sub_heard = _find_confusable_word(tokens, available)
        if combo_sub_idx is not None and combo_sub_heard is not None:
            recognized = list(tokens)
            recognized[combo_sub_idx] = combo_sub_heard
            combo_words = _words(tokens, accuracy=85)
            for i, weak in combo:
                combo_words[i] = WordResult(
                    word=tokens[i], accuracy_score=45,
                    error_type="Mispronunciation",
                    phonemes=[PhonemeScore(phoneme=weak[0], accuracy_score=22)],
                )
            # The recognized word replaces the target at the sub index. Score
            # it as a normally-recognized word (Azure returns the heard word
            # cleanly — alignment turns it into a substitution via missing/
            # extra pairing).
            combo_words[combo_sub_idx] = WordResult(
                word=combo_sub_heard, accuracy_score=88, error_type="None",
                phonemes=[PhonemeScore(phoneme="g", accuracy_score=88)],
            )
            results.append(
                AsrResult(
                    recognized_text=" ".join(recognized),
                    scores=_scores(55, 70, 95),
                    words=combo_words,
                )
            )

    # 9. Borderline accent: one word scores just under the mispronounced cutoff
    #    with ErrorType "None" — Azure's lenient verdict on a clear-but-accented
    #    word (the real "Israel"=79 / "software"=82 case). The raised threshold
    #    still flags it, and with no single weak phoneme the model must coach it
    #    from the word's orthographic anchor (e.g. the 'r' in 'red').
    border_idx, _ = _pick_mispronunciation(tokens)
    if border_idx is not None:
        border_words = _words(tokens, accuracy=95)
        border_words[border_idx] = WordResult(
            word=tokens[border_idx], accuracy_score=82, error_type="None",
            phonemes=[PhonemeScore(phoneme="g", accuracy_score=82)],
        )
        results.append(
            AsrResult(
                recognized_text=target,
                scores=_scores(88, 90, 100, prosody=80),
                words=border_words,
            )
        )

    # 10. Subtle weak sound: Azure flags the word (Mispronunciation) but only one
    #     phoneme is mildly off (~78), just under the phoneme cutoff. Teaches the
    #     model to still name the specific sound instead of a generic line.
    subtle_idx, subtle_weak = _pick_mispronunciation(tokens)
    if subtle_idx is not None:
        subtle_words = _words(tokens, accuracy=92)
        subtle_words[subtle_idx] = WordResult(
            word=tokens[subtle_idx], accuracy_score=88,
            error_type="Mispronunciation",
            phonemes=[PhonemeScore(phoneme=subtle_weak[0], accuracy_score=78)],
        )
        results.append(
            AsrResult(
                recognized_text=target,
                scores=_scores(84, 88, 100, prosody=82),
                words=subtle_words,
            )
        )

    # 11. Clean but not perfect: every word is fine, yet a sub-score (here
    #     prosody) sits below 95 so the score isn't 100. This is the real
    #     "Excellent — try to sound more fluent" case. It yields a passed report
    #     whose only feedback is a single polish tip, teaching the model to
    #     praise + give ONE gentle tip WITHOUT inventing word errors.
    results.append(
        AsrResult(
            recognized_text=target,
            scores=_scores(94, 90, 100, prosody=76),
            words=_words(tokens, accuracy=94),
        )
    )

    # 12. Two missing words (not a severe attempt): drop two spread-out words
    #     but keep the rest intact. Teaches naming MULTIPLE missing words.
    if len(tokens) >= 5:
        drop_a, drop_b = 1, len(tokens) - 2
        kept2 = [t for i, t in enumerate(tokens) if i not in (drop_a, drop_b)]
        results.append(
            AsrResult(
                recognized_text=" ".join(kept2),
                scores=_scores(80, 78, 65),
                words=_words(kept2, accuracy=92),
            )
        )

    # 13. Missing word PLUS a mispronounced word — a very common real combo,
    #     distinct from scenario 8 (which pairs a substitution with mispron).
    mm_idx, mm_weak = _pick_mispronunciation(tokens)
    if mm_idx is not None and len(tokens) >= 4:
        drop_i = 0 if mm_idx != 0 else len(tokens) - 1
        kept3 = [t for i, t in enumerate(tokens) if i != drop_i]
        mis_in_kept = mm_idx if mm_idx < drop_i else mm_idx - 1
        results.append(
            AsrResult(
                recognized_text=" ".join(kept3),
                scores=_scores(66, 76, 80),
                words=_words(
                    kept3, accuracy=90,
                    mispron_index=mis_in_kept, weak_phonemes=mm_weak,
                ),
            )
        )

    # 14. Last word mispronounced — explicitly exercises the final word, the
    #     position most affected by Azure segmentation and the one the
    #     trailing-silence fix targets.
    last_eligible = next(
        (i for i in range(len(tokens) - 1, -1, -1)
         if _plausible_weak_phonemes(tokens[i])),
        None,
    )
    if last_eligible is not None:
        lw = _plausible_weak_phonemes(tokens[last_eligible])[:1]
        results.append(
            AsrResult(
                recognized_text=target,
                scores=_scores(72, 85, 100),
                words=_words(
                    tokens, accuracy=92,
                    mispron_index=last_eligible, weak_phonemes=lw,
                ),
            )
        )

    # 15. Disfluency + mispronunciation: the learner inserts a filler word AND
    #     mispronounces a real one. Distinct from scenario 8 (substitution).
    ex_idx, ex_weak = _pick_mispronunciation(tokens)
    if ex_idx is not None:
        extended2 = tokens[:1] + ["um"] + tokens[1:]
        mis_in_ext = ex_idx if ex_idx == 0 else ex_idx + 1
        results.append(
            AsrResult(
                recognized_text=" ".join(extended2),
                scores=_scores(68, 72, 100),
                words=_words(
                    extended2, accuracy=90,
                    mispron_index=mis_in_ext, weak_phonemes=ex_weak,
                ),
            )
        )

    # 16. Minimal-pair substitution — the target word is swapped for its
    #     minimal-pair neighbor (sheep→ship, right→light, three→tree). Stage 1
    #     will attach the sound_hint automatically, so the model learns to
    #     coach the single distinguishing sound in addition to naming the
    #     swap. This is the ceiling case in our existing memory (mild→make)
    #     that the old pipeline had no realistic training for.
    minimal_idx = _find_word_matching(
        tokens, lambda t: t.lower() in _MINIMAL_PAIRS,
    )
    if minimal_idx is not None:
        mp_result = _pick_confusable(tokens[minimal_idx])
        if mp_result is not None:
            mp_heard, _ = mp_result
            mp_recognized = list(tokens)
            mp_recognized[minimal_idx] = mp_heard
            mp_words = _words(mp_recognized, accuracy=86)
            results.append(
                AsrResult(
                    recognized_text=" ".join(mp_recognized),
                    scores=_scores(76, 82, 90),
                    words=mp_words,
                )
            )

    # 17. Silent-letter mispronunciation — a sentence containing e.g. 'knife',
    #     'walk', 'sign', 'hour'. Azure flags the word as a Mispronunciation
    #     (learner pronounced the silent letter); Stage 1's
    #     _split_spelling_errors then routes the word to silent_letter_errors
    #     instead of the generic mispronounced list, and the model gets a
    #     rule-based coaching hint.
    sl_idx = _find_word_matching(
        tokens, lambda t: _silent_letter_for(t) is not None,
    )
    if sl_idx is not None:
        results.append(
            AsrResult(
                recognized_text=target,
                scores=_scores(70, 80, 100),
                words=_words(
                    tokens, accuracy=88,
                    mispron_index=sl_idx, weak_phonemes=["g"],
                ),
            )
        )

    # 18. Hard/soft-c mispronunciation — sentence contains 'city', 'cycle',
    #     'receive', 'accept', etc. Same synthesis shape as silent-letter;
    #     the routing table decides the category.
    hsc_idx = _find_word_matching(
        tokens, lambda t: _hard_soft_c_for(t) is not None,
    )
    if hsc_idx is not None:
        results.append(
            AsrResult(
                recognized_text=target,
                scores=_scores(70, 82, 100),
                words=_words(
                    tokens, accuracy=88,
                    mispron_index=hsc_idx, weak_phonemes=["g"],
                ),
            )
        )

    # 19. Consonant-cluster mispronunciation — sentence contains 'street',
    #     'spring', 'three', 'shrimp'. Learner likely vowel-inserted or
    #     simplified the cluster.
    cl_idx = _find_word_matching(
        tokens, lambda t: _cluster_hint_for(t) is not None,
    )
    if cl_idx is not None:
        results.append(
            AsrResult(
                recognized_text=target,
                scores=_scores(72, 78, 100),
                words=_words(
                    tokens, accuracy=88,
                    mispron_index=cl_idx, weak_phonemes=["g"],
                ),
            )
        )

    # 20. Vowel-only mispronunciation — a curated vowel-anchored word gets
    #     its vowel phoneme flagged (weak_sounds = ["iy"] / ["ae"] / etc.).
    #     Fills the biggest gap in the previous training set, which never
    #     coached any vowel. Uses word_hints.WORD_VOWEL_ANCHORS to pick the
    #     right phoneme code so contrast_hint_for produces a proper anchor
    #     ("the 'ee' sound like in 'tree'").
    vw_idx = _find_word_matching(
        tokens, lambda t: _word_vowel_for(t) is not None,
    )
    if vw_idx is not None:
        vw_phoneme = _word_vowel_for(tokens[vw_idx])
        # Use a lower phoneme score so contrast_hint_for treats this as a
        # real weak sound and produces the anchor hint.
        vw_words = _words(tokens, accuracy=92)
        vw_words[vw_idx] = WordResult(
            word=tokens[vw_idx], accuracy_score=68,
            error_type="Mispronunciation",
            phonemes=[PhonemeScore(phoneme=vw_phoneme, accuracy_score=42)],
        )
        results.append(
            AsrResult(
                recognized_text=target,
                scores=_scores(72, 88, 100),
                words=vw_words,
            )
        )

    return results


def bootstrap_feedback(messages: list[dict], client) -> str:
    """Ask the teacher model for the ideal assistant reply given the
    system+user messages.

    Uses gpt-4o-mini: this task is rule-following (convert a structured JSON
    verdict into a spoken sentence per SYSTEM_PROMPT rules with slot fills),
    not creative — mini's rule adherence is more than sufficient, and its
    10x higher rate limits + 30x lower cost make the full 5k-row run
    feasible on tier 1.

    Wraps the SDK's built-in max_retries=6 with one more layer of catch-and-
    sleep for RateLimitError: on tier 1 a bursty batch can drain the TPM
    window before the SDK's exponential backoff catches up, and losing a
    whole in-flight batch to a stray 429 was the failure mode that stopped
    the last two runs. Two extra manual retries with a 20s sleep gives the
    window time to reset."""
    import openai as _openai  # local import so the module stays importable
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.4,
                max_tokens=160,
            )
            return (response.choices[0].message.content or "").strip()
        except _openai.RateLimitError:
            if attempt == 2:
                raise
            time.sleep(_RATE_LIMIT_SLEEP)
    return ""


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

    # Build the full, deterministic list of work items first (no API calls).
    # Each item is the chat messages WITHOUT the assistant turn. Doing this up
    # front is what makes the run resumable: a crash (e.g. an OpenAI rate
    # limit) can be continued by simply re-running — the fixed seed reproduces
    # the exact same item order, so we skip the ones already written.
    work: list[list[dict]] = []
    for target in target_sentences:
        for asr in synthesize(target):
            work.append(build_model_messages(analyze(target, asr)))
            if args.limit and len(work) >= args.limit:
                break
        if args.limit and len(work) >= args.limit:
            break

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "_progress.jsonl"

    client = None
    if not args.no_bootstrap:
        from openai import OpenAI
        # max_retries lets the SDK back off on the occasional transient 429;
        # the throttle below keeps us under the steady-state TPM ceiling.
        client = OpenAI(max_retries=6, timeout=60)

    done = 0
    if progress_path.exists():
        done = sum(1 for _ in progress_path.open(encoding="utf-8"))
        if done:
            print(f"resuming: {done}/{len(work)} samples already generated",
                  flush=True)

    # Parallelize GPT-4o calls in batches to hit ~5x throughput vs the old
    # sequential+throttle loop. Results within a batch are written in order,
    # so the resume journal (_progress.jsonl line count) still maps cleanly
    # to the deterministic `work` list on restart.
    def _one(messages: list[dict]) -> str:
        return "" if args.no_bootstrap else bootstrap_feedback(messages, client)

    with progress_path.open("a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=_CONCURRENCY) as executor:
            for batch_start in range(done, len(work), _CONCURRENCY):
                batch_end = min(batch_start + _CONCURRENCY, len(work))
                batch = [work[i] for i in range(batch_start, batch_end)]
                # executor.map preserves input order, so we can write results
                # sequentially without any locking.
                feedbacks = list(executor.map(_one, batch))
                for messages, feedback in zip(batch, feedbacks):
                    sample = {"messages": messages + [
                        {"role": "assistant", "content": feedback}
                    ]}
                    out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{batch_end}/{len(work)}]", flush=True)

    # Every item generated — load the full journal back, shuffle, split, write.
    samples = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    random.shuffle(samples)
    n_val = int(len(samples) * args.val_fraction)
    val, train = samples[:n_val], samples[n_val:]

    for name, rows in (("train", train), ("val", val)):
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} samples -> {path}")

    progress_path.unlink()  # clean up the resume journal on success

    print(
        "\nNext: run scripts/curate_dataset.py to drop any sample whose "
        "feedback omits a required word, then review train.jsonl."
    )


if __name__ == "__main__":
    main()
