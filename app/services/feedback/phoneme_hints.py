"""Phoneme → spoken-friendly anchor hints.

Shared between Stage 1 (`analysis.py`, runtime) and the training-data
generator (`training/generate_dataset.py`). Keeping both sides on the same
table means the model is trained on the exact same hint strings it will be
asked to read aloud in production.

The phoneme codes here are the SAPI-style labels Azure's scripted
Pronunciation Assessment returns (e.g. "th", "sh", "r"), not IPA.
"""
import re

# IPA symbols (and a few variants) mapped back to the SAPI-style codes this
# module is keyed on. We ask Azure for SAPI, but locale/model quirks can still
# leak IPA through; normalizing here means a "th" mistake is coached the same
# way regardless of which alphabet Azure actually returned.
_IPA_TO_SAPI: dict[str, str] = {
    "θ": "th",   # voiceless th, as in 'thin'
    "ð": "dh",   # voiced th, as in 'this'
    "ʃ": "sh",
    "ʒ": "zh",
    "tʃ": "ch",
    "dʒ": "jh",
    "ŋ": "ng",
    "ɹ": "r",
    "ɫ": "l",
}


def canonical_phoneme(phoneme: str) -> str:
    """Normalize a raw Azure phoneme code to our SAPI-style key.

    Lowercases, strips any trailing stress digit Azure appends (e.g. "aa1"),
    and maps IPA symbols to their SAPI equivalents so the rest of the module
    only ever has to reason about one alphabet.
    """
    if not phoneme:
        return ""
    p = re.sub(r"\d+$", "", phoneme.strip().lower())
    return _IPA_TO_SAPI.get(p, p)


# Commonly-difficult phonemes we may flag as "weak", paired with the
# orthography that produces them. A phoneme only applies to a word when the
# word's spelling actually contains the trigger — otherwise we'd be telling
# the learner to "work on the 'th' sound" in a word that has no 'th'.
PHONEME_TRIGGERS: list[tuple[str, str]] = [
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

# Anchor word for each phoneme — short, common, unambiguous, and uses the
# phoneme in a salient position. Spoken as: "the 'th' in 'thin'".
_PHONEME_ANCHOR_WORDS: dict[str, str] = {
    "th": "thin",   # voiceless th
    "dh": "this",   # voiced th
    "sh": "ship",
    "ch": "chair",
    "ng": "sing",
    "r": "red",
    "l": "leg",
    "v": "very",
    "w": "wet",
    "z": "zoo",
    "s": "see",
}

# Some SAPI codes aren't graphemes a learner would recognize. Show a familiar
# spelling instead: voiced "th" (SAPI "dh") is still just "th" to the user, so
# they never see a raw code like "dh".
_PHONEME_DISPLAY: dict[str, str] = {
    "dh": "th",
}

# How many anchors to chain in a single hint before it stops sounding natural.
# Phonemes arrive worst-first, so this keeps the most-mispronounced sounds.
_MAX_ANCHORS_PER_HINT = 3


def plausible_weak_phonemes(word: str) -> list[str]:
    """Phonemes whose orthographic trigger appears in `word`.

    Used by the dataset generator to assign realistic weak phonemes when
    synthesizing fake AsrResults.
    """
    w = word.lower()
    found = [ph for ph, trigger in PHONEME_TRIGGERS if trigger in w]
    # "s" is only safe when not part of "sh" (already covered above).
    if "s" in w and "sh" not in w:
        found.append("s")
    return found


def correct_hint_for(weak_phonemes: list[str]) -> str | None:
    """Return a spoken-friendly anchor for the correct pronunciation.

    The result is a noun phrase that reads naturally inside a tutor-style
    frame like *"For 'thought', practice <hint>."* — so callers don't have
    to know which phoneme it came from.

    Example: `["th", "s"]` ->
    `"the 'th' sound like in 'thin' and the 's' sound like in 'see'"`.

    Returns None when none of the weak phonemes have a known anchor word —
    in that case Stage 2 should fall back to a generic "work on this word"
    line rather than fabricate a hint.
    """
    seen: set[str] = set()
    anchors: list[str] = []
    for raw in weak_phonemes:
        key = canonical_phoneme(raw)
        if not key or key in seen:
            continue
        seen.add(key)
        anchor_word = _PHONEME_ANCHOR_WORDS.get(key)
        if anchor_word is None:
            continue
        grapheme = _PHONEME_DISPLAY.get(key, key)
        anchors.append(f"the '{grapheme}' sound like in '{anchor_word}'")
        if len(anchors) >= _MAX_ANCHORS_PER_HINT:
            break

    if not anchors:
        return None
    if len(anchors) == 1:
        return anchors[0]
    return " and ".join(anchors)
