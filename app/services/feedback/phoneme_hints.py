"""Phoneme → spoken-friendly anchor hints.

Shared between Stage 1 (`analysis.py`, runtime) and the training-data
generator (`training/generate_dataset.py`). Keeping both sides on the same
table means the model is trained on the exact same hint strings it will be
asked to read aloud in production.

Two hint styles per phoneme:
  * ANCHOR — "the 'th' sound like in 'thin'" (used for phoneme *contrast*
    hints, and as one option for plain correct hints).
  * ARTICULATION — "tip of your tongue lightly between your teeth — let
    air flow" (used only for plain correct hints, chosen at random so the
    avatar doesn't sound formulaic).

`correct_hint_for` picks between anchor and articulation randomly per
call; `contrast_hint_for` always uses anchor because the ", not the 'x'
sound" tail only reads coherently on anchor-style phrases.

The phoneme codes here are the SAPI-style labels Azure's scripted
Pronunciation Assessment returns (e.g. "th", "sh", "r"), not IPA.
"""
import random
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
#
# Vowels are included so a vowel mistake (e.g. saying 'soup' with the 'a' of
# 'sap') gets coached on the actual sound instead of falling through to a
# consonant in the spelling. Schwa (SAPI "ax") is deliberately ABSENT: it is
# the reduced, unstressable vowel of function words ("the", "a"), where there
# is nothing meaningful to coach — leaving it out makes correct_hint_for return
# None for those, which analysis.py then suppresses as noise.
_PHONEME_ANCHOR_WORDS: dict[str, str] = {
    # Consonants.
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
    # Vowels (SAPI/ARPABET codes Azure returns; stress digits already stripped).
    "iy": "tree",   # 'ee'
    "ih": "sit",    # short 'i'
    "eh": "bed",    # short 'e'
    "ae": "cat",    # short 'a'
    "aa": "hot",    # 'o'
    "ah": "cup",    # stressed 'uh' (/ʌ/) — NOT schwa
    "ao": "saw",    # 'aw'
    "aw": "now",    # 'ow' diphthong
    "ay": "eye",    # long 'i' (as in 'mild')
    "ey": "day",    # long 'a'
    "ow": "boat",   # long 'o'
    "oy": "boy",    # 'oy'
    "uh": "book",   # 'oo' (/ʊ/, as in 'could')
    "uw": "food",   # 'oo' (/u/, as in 'soup')
    "er": "her",    # 'er'
}

# Some SAPI codes aren't graphemes a learner would recognize. Show a familiar
# spelling instead: voiced "th" (SAPI "dh") is still just "th" to the user, so
# they never see a raw code like "dh". Vowel codes (uw, eh, ...) are never
# learner-friendly, so every anchored vowel needs a display grapheme here.
_PHONEME_DISPLAY: dict[str, str] = {
    "dh": "th",
    # Produced-only display labels for simple consonants. These are deliberately
    # not anchor words, so Stage 1 still cannot coach them as target sounds, but
    # contrast hints can say "not the 'f' sound" when Azure strongly reports it.
    "b": "b",
    "d": "d",
    "f": "f",
    "p": "p",
    "t": "t",
    "iy": "ee",
    "ih": "i",
    "eh": "e",
    "ae": "a",
    "aa": "o",
    "ah": "u",
    "ao": "aw",
    "aw": "ow",
    "ay": "i",
    "ey": "ay",
    "ow": "oa",
    "oy": "oy",
    "uh": "oo",
    "uw": "oo",
    "er": "er",
}

# Articulation-style coaching lines: mouth shape, tongue placement, breath.
# These are what a real pronunciation coach (ELSA / speech therapist / Rachel's
# English) says out loud. Every anchored phoneme should have at least one
# articulation option so `varied_correct_phrase` can rotate through variety.
#
# Kept spoken-friendly and short — each line reads naturally inside the same
# "For '<word>', practice <line>." frame as the anchor hint.
_PHONEME_ARTICULATIONS: dict[str, list[str]] = {
    # Consonants
    "th": [
        "tipping your tongue lightly between your teeth — just let air flow",
        "peeking your tongue out just past your top teeth for a soft hiss",
    ],
    "dh": [
        "putting your tongue between your teeth and voicing it — a soft buzz",
        "the same 'th' shape, but this time add your voice",
    ],
    "sh": [
        "rounding your lips and pushing air through — like hushing someone",
        "a long, quiet 'sh' — no voice, just air",
    ],
    "ch": [
        "starting with a quick 't' and releasing into 'sh' — sharp and short",
        "a burst of air off the roof of your mouth — 't' then 'sh'",
    ],
    "ng": [
        "letting the back of your tongue touch the roof — hum through your nose",
        "keeping your mouth open and humming with your tongue at the back",
    ],
    "r": [
        "curling your tongue up and back — but don't let it touch the roof",
        "pulling your tongue back like a soft growl, lips slightly rounded",
    ],
    "l": [
        "touching the tip of your tongue to the ridge behind your top teeth",
        "letting the air flow around the sides of your tongue",
    ],
    "v": [
        "your top teeth touching your bottom lip, then buzzing — like a bee",
        "biting your bottom lip lightly and adding voice",
    ],
    "w": [
        "rounding your lips into a small circle — like starting to whistle",
        "pushing your lips forward, then releasing into the next sound",
    ],
    "z": [
        "humming a long 's' — same shape, but voice it like a bee",
        "keeping your teeth close and buzzing the air through them",
    ],
    "s": [
        "resting the tip of your tongue near your top teeth and hissing softly",
        "a quiet, steady hiss — no voice, just air",
    ],

    # Vowels — timing, mouth shape, tension
    "iy": [   # long ee
        "smiling wide and stretching the sound — hold it long",
        "keeping your tongue high and lips spread, like saying 'cheese'",
    ],
    "ih": [   # short i
        "relaxing your mouth completely — short and quick, don't stretch it",
        "a lazy, brief 'i' — half the length of the long 'ee'",
    ],
    "ae": [   # cat
        "opening your mouth wide and dropping your jaw",
        "a bright, open 'a' — mouth wide, tongue forward",
    ],
    "eh": [   # bed
        "opening your mouth halfway — tongue in the middle, jaw relaxed",
        "a short, clean 'e' — not as wide as 'a', not as narrow as 'ee'",
    ],
    "ah": [   # cup
        "relaxing completely — like the 'uh' when you're thinking",
        "an open, neutral sound — jaw dropped, tongue at rest",
    ],
    "aa": [   # hot
        "opening your mouth wide, like the dentist chair",
        "a deep, open 'o' — jaw dropped, back of tongue low",
    ],
    "ao": [   # saw
        "rounding your lips a little and dropping your jaw",
        "a long, open 'aw' — lips slightly rounded, mouth wide",
    ],
    "aw": [   # now
        "starting with 'ah' and rounding your lips as you finish",
        "gliding from open 'a' into a rounded 'oo'",
    ],
    "ay": [   # eye / mild
        "starting with 'ah' and sliding up to 'ee' — one smooth motion",
        "an open jaw that closes into a smile as you finish",
    ],
    "ey": [   # day
        "starting with 'eh' and gliding up to 'ee'",
        "a short 'e' that stretches into 'ee' at the end",
    ],
    "ow": [   # boat
        "starting with 'oh' and rounding your lips tighter as you finish",
        "gliding from open 'o' into a small round 'oo'",
    ],
    "oy": [   # boy
        "starting with 'aw' and sliding up into 'ee'",
        "a rounded 'o' that opens up into a smile",
    ],
    "uh": [   # book
        "lightly rounding your lips — quick and relaxed",
        "a short, soft 'oo' — half the length of the long 'oo' in 'food'",
    ],
    "uw": [   # food
        "pushing your lips forward and rounding them tight — long and pushed",
        "a long, tight 'oo' — like blowing out a candle very slowly",
    ],
    "er": [   # her
        "curling your tongue up and holding the 'r' throughout",
        "a long, colored vowel — like the 'r' in 'red' but longer",
    ],
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


# SAPI/ARPABET vowel codes (stress digits already stripped by canonical_phoneme).
# Used to tell a real vowel<->vowel swap from segmentation noise where Azure
# reports a consonant produced in a vowel slot (or vice versa).
_VOWELS: frozenset[str] = frozenset({
    "iy", "ih", "eh", "ae", "aa", "ah", "ao", "aw", "ay", "ey",
    "ow", "oy", "uh", "uw", "er", "ax", "ix", "axr", "ux",
})


def is_vowel(phoneme: str) -> bool:
    """True when the (canonicalized) code is a vowel. Unknown codes are treated
    as consonants, which is the safe default for the cross-class noise filter."""
    return canonical_phoneme(phoneme) in _VOWELS


def has_anchor(phoneme: str) -> bool:
    """True when we have a learner-friendly anchor word for this sound, i.e. we
    can coach it. False for schwa ('ax'), glides ('y'), and consonants with no
    anchor ('k', 'g', ...) — sounds we deliberately don't surface."""
    return canonical_phoneme(phoneme) in _PHONEME_ANCHOR_WORDS


def _display_grapheme(key: str) -> str | None:
    """A learner-friendly spelling for a SAPI code, or None when we have no
    trustworthy one. Vowels carry an explicit display grapheme; consonants are
    readable as their own code. Raw codes with neither (schwa "ax", "hh", …)
    return None so callers never surface a cryptic symbol to the user."""
    if key in _PHONEME_DISPLAY:
        return _PHONEME_DISPLAY[key]
    if key in _PHONEME_ANCHOR_WORDS:
        return key
    return None


def _anchor_phrase(key: str) -> str | None:
    """ "the '<grapheme>' sound like in '<anchor>'" for one SAPI code, or None
    when the code has no anchor word. Deterministic — used by
    `contrast_hint_for` where the ", not the 'x' sound" tail depends on the
    anchor shape."""
    anchor_word = _PHONEME_ANCHOR_WORDS.get(key)
    if anchor_word is None:
        return None
    grapheme = _PHONEME_DISPLAY.get(key, key)
    return f"the '{grapheme}' sound like in '{anchor_word}'"


def _varied_correct_phrase(key: str) -> str | None:
    """Randomly return anchor OR one articulation line for `key`.

    Used only by `correct_hint_for` so the avatar rotates between the
    "sound like in X" anchor and articulation-style coaching (mouth shape,
    tongue placement). Callers of `contrast_hint_for` still get the anchor
    unchanged, since the contrast tail only reads on anchor phrases.

    Returns None when the code has no anchor at all (schwa, unanchored
    consonants) — matches `_anchor_phrase`'s contract so caller logic
    doesn't have to branch."""
    anchor = _anchor_phrase(key)
    if anchor is None:
        return None
    options: list[str] = [anchor, *_PHONEME_ARTICULATIONS.get(key, [])]
    return random.choice(options)


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
        phrase = _varied_correct_phrase(key)
        if phrase is None:
            continue
        anchors.append(phrase)
        if len(anchors) >= _MAX_ANCHORS_PER_HINT:
            break

    if not anchors:
        return None
    if len(anchors) == 1:
        return anchors[0]
    return " and ".join(anchors)


def contrast_hint_for(pairs: list[tuple[str, str | None]]) -> str | None:
    """Like `correct_hint_for`, but each sound can name what the learner
    actually produced so the hint contrasts wrong vs right.

    `pairs` is `(expected_phoneme, produced_phoneme | None)`, worst-first. The
    expected phoneme drives the anchor (the sound to practice); the produced
    phoneme, when known and displayable, adds a ", not the '<x>' sound" tail —
    e.g. `[("eh", "ae")]` -> `"the 'e' sound like in 'bed', not the 'a' sound"`.

    Reads verbatim inside the same "For '<word>', practice <hint>." frame the
    model is trained on, so the contrast rides through Stage 2 untouched.
    Falls back to the plain anchor when produced is unknown/undisplayable, and
    returns None when no expected phoneme has a usable anchor (same contract as
    `correct_hint_for`, so analysis.py can suppress hint-less words).
    """
    seen: set[str] = set()
    phrases: list[str] = []
    for raw_exp, raw_prod in pairs:
        exp_key = canonical_phoneme(raw_exp)
        if not exp_key or exp_key in seen:
            continue
        seen.add(exp_key)
        phrase = _anchor_phrase(exp_key)
        if phrase is None:
            continue
        prod_key = canonical_phoneme(raw_prod) if raw_prod else ""
        prod_grapheme = _display_grapheme(prod_key) if prod_key else None
        exp_grapheme = _PHONEME_DISPLAY.get(exp_key, exp_key)
        # Only contrast when we can name the produced sound and it actually
        # reads as a different grapheme (e.g. "ay" vs "ey" both display "i" —
        # naming "the 'i' sound, not the 'i' sound" would be nonsense).
        if prod_grapheme and prod_grapheme != exp_grapheme:
            phrase += f", not the '{prod_grapheme}' sound"
        phrases.append(phrase)
        if len(phrases) >= _MAX_ANCHORS_PER_HINT:
            break

    if not phrases:
        return None
    if len(phrases) == 1:
        return phrases[0]
    return " and ".join(phrases)
