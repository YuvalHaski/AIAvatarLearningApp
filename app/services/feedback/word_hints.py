"""Word-level coaching tables that complement phoneme_hints.py.

`phoneme_hints.py` coaches at the phoneme layer (Azure said the 'r' was weak
→ practice the 'r' sound like in 'red'). Three classes of L2 English error
aren't phoneme-level and need spelling / orthography coaching instead:

  1. Silent letters (`k` in 'knife', `b` in 'comb', `l` in 'walk')
  2. Hard vs soft 'c' (/k/ in 'cat' vs /s/ in 'city') and 'cc' (/k/ in 'occur')
  3. Difficult consonant clusters (`str`, `spr`, `thr`) that get vowel-inserted
     or simplified

Azure Pronunciation Assessment scores phonemes, not spelling rules. These
tables describe possible coaching hints, but runtime analysis should only emit
them when it has explicit evidence for that spelling-rule mistake. Table
membership alone is not enough.

A fourth class this module handles is *word-level substitutions*. When a
learner mispronounces a word badly enough, Azure returns the closest real
English word it thinks was said (not the placeholder "something" the old
generator used). Two subclasses:

  * MINIMAL_PAIR_CONFUSABLES — the target and the confusable differ by just
    one sound. The learner cannot HEAR the difference by definition, so we
    attach a `sound_hint` teaching the specific sound they need to fix
    ("practice the long 'ee' sound like in 'tree'").
  * GENERAL_CONFUSABLES — the confusion is at the word level, not one sound
    (e.g. was/wants). No sound_hint, just the "you said X instead of Y"
    identification.

Sources: Grammarly's silent-letters guide, Wikipedia hard-and-soft-C,
L2-ARCTIC's most common substitution categories (Z↔S, R deletions, W↔V),
and standard minimal-pair lists for /iː/-/ɪ/, /æ/-/ɛ/, /uː/-/ʊ/.
"""
from __future__ import annotations

import random
import re


# ==========================================================================
# Silent letters
# ==========================================================================
#
# Each entry maps a word to (silent_letter, hint_phrase). The hint is a
# spoken-friendly noun phrase that reads verbatim inside a tutor frame like
# *"Remember, <hint>."*
#
# Only words whose silent letter is confidently silent for every learner are
# included. Regional variants that are only sometimes silent (e.g. the 't' in
# 'often') are omitted so the coach never contradicts a valid pronunciation.

SILENT_LETTER_WORDS: dict[str, tuple[str, str]] = {
    # silent K in kn-
    "knife":     ("k", "the 'k' in 'knife' is silent — say it like 'nife'"),
    "know":      ("k", "the 'k' in 'know' is silent — say it like 'no'"),
    "knew":      ("k", "the 'k' in 'knew' is silent — say it like 'new'"),
    "knee":      ("k", "the 'k' in 'knee' is silent — say it like 'nee'"),
    "knight":    ("k", "the 'k' in 'knight' is silent — say it like 'night'"),
    "knock":     ("k", "the 'k' in 'knock' is silent — say it like 'nock'"),
    "knot":      ("k", "the 'k' in 'knot' is silent — say it like 'not'"),
    "knuckle":   ("k", "the 'k' in 'knuckle' is silent — say it like 'nuckle'"),
    "knowledge": ("k", "the 'k' in 'knowledge' is silent — say it like 'nowledge'"),

    # silent B in -mb / -bt
    "comb":      ("b", "the 'b' in 'comb' is silent — say it like 'come'"),
    "tomb":      ("b", "the 'b' in 'tomb' is silent — say it like 'toom'"),
    "lamb":      ("b", "the 'b' in 'lamb' is silent — say it like 'lam'"),
    "thumb":     ("b", "the 'b' in 'thumb' is silent — say it like 'thum'"),
    "dumb":      ("b", "the 'b' in 'dumb' is silent — say it like 'dum'"),
    "numb":      ("b", "the 'b' in 'numb' is silent — say it like 'num'"),
    "climb":     ("b", "the 'b' in 'climb' is silent — say it like 'clime'"),
    "bomb":      ("b", "the 'b' in 'bomb' is silent — say it like 'bom'"),
    "debt":      ("b", "the 'b' in 'debt' is silent — say it like 'det'"),
    "doubt":     ("b", "the 'b' in 'doubt' is silent — say it like 'dout'"),
    "subtle":    ("b", "the 'b' in 'subtle' is silent — say it like 'suttle'"),

    # silent P in ps- / pn-
    "psalm":     ("p", "the 'p' in 'psalm' is silent — say it like 'salm'"),
    "psychic":   ("p", "the 'p' in 'psychic' is silent — say it like 'sychic'"),
    "psychology": ("p", "the 'p' in 'psychology' is silent — say it like 'sychology'"),
    "pneumonia": ("p", "the 'p' in 'pneumonia' is silent — say it like 'newmonia'"),

    # silent L in -alk / -alf / -ould / -olk
    "walk":      ("l", "the 'l' in 'walk' is silent — say it like 'wok'"),
    "talk":      ("l", "the 'l' in 'talk' is silent — say it like 'tok'"),
    "chalk":     ("l", "the 'l' in 'chalk' is silent — say it like 'chok'"),
    "calf":      ("l", "the 'l' in 'calf' is silent — say it like 'caf'"),
    "half":      ("l", "the 'l' in 'half' is silent — say it like 'haf'"),
    "could":     ("l", "the 'l' in 'could' is silent — say it like 'cood'"),
    "should":    ("l", "the 'l' in 'should' is silent — say it like 'shood'"),
    "would":     ("l", "the 'l' in 'would' is silent — say it like 'wood'"),
    "folk":      ("l", "the 'l' in 'folk' is silent — say it like 'foke'"),
    "yolk":      ("l", "the 'l' in 'yolk' is silent — say it like 'yoke'"),
    "salmon":    ("l", "the 'l' in 'salmon' is silent — say it like 'sammon'"),

    # silent G in gn- / -gn
    "gnat":      ("g", "the 'g' in 'gnat' is silent — say it like 'nat'"),
    "gnaw":      ("g", "the 'g' in 'gnaw' is silent — say it like 'naw'"),
    "gnome":     ("g", "the 'g' in 'gnome' is silent — say it like 'nome'"),
    "sign":      ("g", "the 'g' in 'sign' is silent — say it like 'sine'"),
    "resign":    ("g", "the 'g' in 'resign' is silent — say it like 'resine'"),
    "design":    ("g", "the 'g' in 'design' is silent — say it like 'desine'"),
    "foreign":   ("g", "the 'g' in 'foreign' is silent — say it like 'foren'"),
    "campaign":  ("g", "the 'g' in 'campaign' is silent — say it like 'campain'"),

    # silent H
    "hour":      ("h", "the 'h' in 'hour' is silent — say it like 'our'"),
    "honest":    ("h", "the 'h' in 'honest' is silent — say it like 'onest'"),
    "honor":     ("h", "the 'h' in 'honor' is silent — say it like 'onor'"),
    "ghost":     ("h", "the 'h' in 'ghost' is silent — say it like 'gost'"),
    "ghetto":    ("h", "the 'h' in 'ghetto' is silent — say it like 'getto'"),

    # silent W in wr-
    "write":     ("w", "the 'w' in 'write' is silent — say it like 'rite'"),
    "wrote":     ("w", "the 'w' in 'wrote' is silent — say it like 'rote'"),
    "wrist":     ("w", "the 'w' in 'wrist' is silent — say it like 'rist'"),
    "wrong":     ("w", "the 'w' in 'wrong' is silent — say it like 'rong'"),
    "wrap":      ("w", "the 'w' in 'wrap' is silent — say it like 'rap'"),
    "wreck":     ("w", "the 'w' in 'wreck' is silent — say it like 'reck'"),

    # silent T in -stle / -sten
    "castle":    ("t", "the 't' in 'castle' is silent — say it like 'cassel'"),
    "listen":    ("t", "the 't' in 'listen' is silent — say it like 'lissen'"),
    "whistle":   ("t", "the 't' in 'whistle' is silent — say it like 'whissel'"),
    "fasten":    ("t", "the 't' in 'fasten' is silent — say it like 'fassen'"),
    "hustle":    ("t", "the 't' in 'hustle' is silent — say it like 'hussel'"),

    # silent C in sc-
    "scissors":  ("c", "the 'c' in 'scissors' is silent — say it like 'sizzors'"),
    "scene":     ("c", "the 'c' in 'scene' is silent — say it like 'seen'"),
    "scent":     ("c", "the 'c' in 'scent' is silent — say it like 'sent'"),
    "science":   ("c", "the 'c' in 'science' is silent — say it like 'sience'"),
    "muscle":    ("c", "the 'c' in 'muscle' is silent — say it like 'mussel'"),
}


# ==========================================================================
# Hard vs soft 'c' (and 'cc')
# ==========================================================================
#
# The rule the learner needs to internalize:
#   c → /s/ before e, i, y  (city, cycle, receive)
#   c → /k/ everywhere else (cat, come, cup)
#   cc → /k/ (occur, according)
#   cc → /ks/ before e/i    (accept, accident, success)
#
# We only include words where the wrong choice is a *common* L2 mistake — the
# learner defaults to hard /k/ when soft /s/ is required, or vice versa.

HARD_SOFT_C_WORDS: dict[str, tuple[str, str]] = {
    # soft c (/s/) — likely to be over-hardened
    "city":     ("soft", "in 'city' the 'c' sounds like 's' — say it like 'sity'"),
    "cycle":    ("soft", "in 'cycle' the 'c' sounds like 's' — say it like 'sycle'"),
    "cent":     ("soft", "in 'cent' the 'c' sounds like 's' — say it like 'sent'"),
    "cell":     ("soft", "in 'cell' the 'c' sounds like 's' — say it like 'sell'"),
    "ice":      ("soft", "in 'ice' the 'c' sounds like 's' — say it like 'ise'"),
    "rice":     ("soft", "in 'rice' the 'c' sounds like 's' — say it like 'rise'"),
    "nice":     ("soft", "in 'nice' the 'c' sounds like 's' — say it like 'nise'"),
    "receive":  ("soft", "in 'receive' the 'c' sounds like 's' — say it like 'receeve'"),
    "notice":   ("soft", "in 'notice' the 'c' sounds like 's' — say it like 'notise'"),
    "service":  ("soft", "in 'service' the 'c' sounds like 's' — say it like 'servise'"),
    "office":   ("soft", "in 'office' the 'c' sounds like 's' — say it like 'offise'"),
    "police":   ("soft", "in 'police' the 'c' sounds like 's' — say it like 'polise'"),
    "cinema":   ("soft", "in 'cinema' the 'c' sounds like 's' — say it like 'sinema'"),
    "circle":   ("soft", "in 'circle' the 'c' sounds like 's' — say it like 'sircle'"),
    "pencil":   ("soft", "in 'pencil' the 'c' sounds like 's' — say it like 'pensil'"),
    "dance":    ("soft", "in 'dance' the 'c' sounds like 's' — say it like 'danse'"),
    "chance":   ("soft", "in 'chance' the 'c' sounds like 's' — say it like 'chanse'"),

    # hard c (/k/) — usually fine, but included so the model doesn't
    # over-correct 'cat' to 'sat' when it also sees soft-c coaching.
    "cake":     ("hard", "in 'cake' the 'c' sounds like 'k' — say it like 'kake'"),
    "cook":     ("hard", "in 'cook' the 'c' sounds like 'k' — say it like 'kook'"),

    # double c (cc)
    "accept":    ("cc-ks", "in 'accept' the 'cc' sounds like 'ks' — say it like 'aksept'"),
    "accident":  ("cc-ks", "in 'accident' the 'cc' sounds like 'ks' — say it like 'aksident'"),
    "success":   ("cc-ks", "in 'success' the 'cc' sounds like 'ks' — say it like 'sukses'"),
    "access":    ("cc-ks", "in 'access' the 'cc' sounds like 'ks' — say it like 'akses'"),
    "vaccine":   ("cc-ks", "in 'vaccine' the 'cc' sounds like 'ks' — say it like 'vaksine'"),
    "occur":     ("cc-k",  "in 'occur' the 'cc' sounds like just 'k' — say it like 'okur'"),
    "occasion":  ("cc-k",  "in 'occasion' the 'cc' sounds like just 'k' — say it like 'okasion'"),
    "according": ("cc-k",  "in 'according' the 'cc' sounds like just 'k' — say it like 'akording'"),
    "accurate":  ("cc-k",  "in 'accurate' the 'cc' sounds like just 'k' — say it like 'akurate'"),
    "soccer":    ("cc-k",  "in 'soccer' the 'cc' sounds like just 'k' — say it like 'soker'"),
}


# ==========================================================================
# Consonant clusters
# ==========================================================================
#
# Cluster → (anchor word using the cluster, spoken hint). Non-native learners
# insert a vowel ("estreet" for "street") or drop a consonant ("sreet"). The
# hint frames the whole cluster as one smooth sound.

CLUSTER_PATTERNS: dict[str, tuple[str, str]] = {
    "str": ("street",  "the 'str' cluster — blend it smoothly like in 'street'"),
    "spr": ("spring",  "the 'spr' cluster — blend it smoothly like in 'spring'"),
    "spl": ("split",   "the 'spl' cluster — blend it smoothly like in 'split'"),
    "scr": ("screen",  "the 'scr' cluster — blend it smoothly like in 'screen'"),
    "thr": ("three",   "the 'thr' cluster — blend it smoothly like in 'three'"),
    "shr": ("shrimp",  "the 'shr' cluster — blend it smoothly like in 'shrimp'"),
    "sts": ("fists",   "the '-sts' ending — pronounce all three sounds like in 'fists'"),
    "sks": ("asks",    "the '-sks' ending — pronounce all three sounds like in 'asks'"),
}


def find_cluster(word: str) -> str | None:
    """Return the first difficult cluster found in `word`, or None."""
    w = word.lower()
    for pattern in ("str", "spr", "spl", "scr", "thr", "shr"):
        if w.startswith(pattern):
            return pattern
    if w.endswith("sts"):
        return "sts"
    if w.endswith("sks"):
        return "sks"
    return None


# ==========================================================================
# Substitutions — realistic word-level confusions
# ==========================================================================
#
# When Azure hears a *badly-said* word it returns a DIFFERENT real English
# word, not a nonsense token — replacing the old "something" placeholder.
#
# MINIMAL PAIRS: the target and the heard word differ by exactly one sound
# the learner cannot hear yet (this is why they said the wrong one). We
# attach a `sound_hint` teaching the specific sound so the coaching goes
# beyond just naming the swap.
#
# GENERAL CONFUSABLES: the confusion is not reducible to one sound (was ↔
# wants, verify ↔ very, coffee ↔ copy). No sound_hint; naming the swap is
# already the correction.

# ------ minimal pairs -----------------------------------------------------

# The sound_hint always describes the sound the LEARNER needs to produce
# in the EXPECTED word (i.e. the anchor for the target's contested phoneme).
# Vowel anchors reuse phoneme_hints.py's mapping so all hints read alike.

MINIMAL_PAIR_CONFUSABLES: dict[str, list[tuple[str, str]]] = {
    # /iː/ vs /ɪ/  — long "ee" vs short "i"
    "sheep":  [("ship",   "the long 'ee' sound like in 'tree'")],
    "ship":   [("sheep",  "the short 'i' sound like in 'sit'")],
    "leave":  [("live",   "the long 'ee' sound like in 'tree'")],
    "live":   [("leave",  "the short 'i' sound like in 'sit'")],
    "seat":   [("sit",    "the long 'ee' sound like in 'tree'")],
    "sit":    [("seat",   "the short 'i' sound like in 'sit'")],
    "feel":   [("fill",   "the long 'ee' sound like in 'tree'")],
    "fill":   [("feel",   "the short 'i' sound like in 'sit'")],
    "beat":   [("bit",    "the long 'ee' sound like in 'tree'")],
    "bit":    [("beat",   "the short 'i' sound like in 'sit'")],
    "reach":  [("rich",   "the long 'ee' sound like in 'tree'")],
    "rich":   [("reach",  "the short 'i' sound like in 'sit'")],
    "cheap":  [("chip",   "the long 'ee' sound like in 'tree'")],
    "chip":   [("cheap",  "the short 'i' sound like in 'sit'")],
    "eat":    [("it",     "the long 'ee' sound like in 'tree'")],

    # /æ/ vs /ɛ/  — "a" as in cat vs "e" as in bed
    "bad":    [("bed",    "the 'a' sound like in 'cat'")],
    "bed":    [("bad",    "the 'e' sound like in 'bed'")],
    "man":    [("men",    "the 'a' sound like in 'cat'")],
    "men":    [("man",    "the 'e' sound like in 'bed'")],
    "sat":    [("set",    "the 'a' sound like in 'cat'")],
    "set":    [("sat",    "the 'e' sound like in 'bed'")],
    "pan":    [("pen",    "the 'a' sound like in 'cat'")],
    "pen":    [("pan",    "the 'e' sound like in 'bed'")],
    "sand":   [("send",   "the 'a' sound like in 'cat'")],
    "send":   [("sand",   "the 'e' sound like in 'bed'")],
    "than":   [("then",   "the 'a' sound like in 'cat'")],
    "then":   [("than",   "the 'e' sound like in 'bed'")],

    # /uː/ vs /ʊ/  — "oo" as in food vs "oo" as in book
    "pool":   [("pull",   "the long 'oo' sound like in 'food'")],
    "pull":   [("pool",   "the short 'oo' sound like in 'book'")],
    "food":   [("foot",   "the long 'oo' sound like in 'food'")],
    "foot":   [("food",   "the short 'oo' sound like in 'book'")],
    "fool":   [("full",   "the long 'oo' sound like in 'food'")],
    "full":   [("fool",   "the short 'oo' sound like in 'book'")],

    # /ʌ/ vs /ɒ/  — "u" as in cup vs "o" as in hot
    "cup":    [("cop",    "the 'u' sound like in 'cup'")],
    "cop":    [("cup",    "the 'o' sound like in 'hot'")],
    "nut":    [("not",    "the 'u' sound like in 'cup'")],
    "not":    [("nut",    "the 'o' sound like in 'hot'")],

    # R / L — Japanese, Korean, Mandarin, Cantonese
    "right":  [("light",  "the 'r' sound like in 'red'")],
    "light":  [("right",  "the 'l' sound like in 'leg'")],
    "rice":   [("lice",   "the 'r' sound like in 'red'")],
    "read":   [("lead",   "the 'r' sound like in 'red'")],
    "rock":   [("lock",   "the 'r' sound like in 'red'")],
    "lock":   [("rock",   "the 'l' sound like in 'leg'")],
    "rate":   [("late",   "the 'r' sound like in 'red'")],
    "late":   [("rate",   "the 'l' sound like in 'leg'")],
    "grass":  [("glass",  "the 'r' sound like in 'red'")],
    "glass":  [("grass",  "the 'l' sound like in 'leg'")],
    "pray":   [("play",   "the 'r' sound like in 'red'")],
    "play":   [("pray",   "the 'l' sound like in 'leg'")],

    # V / W — Hindi, German, Japanese
    "vine":   [("wine",   "the 'v' sound like in 'very'")],
    "wine":   [("vine",   "the 'w' sound like in 'wet'")],
    "vest":   [("west",   "the 'v' sound like in 'very'")],
    "west":   [("vest",   "the 'w' sound like in 'wet'")],
    "very":   [("wary",   "the 'v' sound like in 'very'")],
    "worse":  [("verse",  "the 'w' sound like in 'wet'")],
    "verse":  [("worse",  "the 'v' sound like in 'very'")],

    # Th → S / T / D / F — nearly every L1
    "three":  [("tree",   "the 'th' sound like in 'thin'")],
    "tree":   [("three",  "the 'tr' cluster like in 'tree'")],
    "thin":   [("sin",    "the 'th' sound like in 'thin'")],
    "think":  [("sink",   "the 'th' sound like in 'thin'")],
    "sink":   [("think",  "the 's' sound like in 'see'")],
    "thought": [("taught", "the 'th' sound like in 'thin'")],
    "thanks": [("tanks",  "the 'th' sound like in 'thin'")],
    "thigh":  [("sigh",   "the 'th' sound like in 'thin'")],
    "path":   [("pass",   "the 'th' sound like in 'thin'")],
    "bath":   [("bass",   "the 'th' sound like in 'thin'")],
    "mouth":  [("mouse",  "the 'th' sound like in 'thin'")],
    "this":   [("dis",    "the 'th' sound like in 'this'")],
    "that":   [("dat",    "the 'th' sound like in 'this'")],

    # Z ↔ S — the #1 substitution in L2-ARCTIC
    "zoo":    [("sue",    "the 'z' sound like in 'zoo'")],
    "zip":    [("sip",    "the 'z' sound like in 'zoo'")],
    "zeal":   [("seal",   "the 'z' sound like in 'zoo'")],
    "prize":  [("price",  "the 'z' sound like in 'zoo'")],
    "rise":   [("rice",   "the 'z' sound like in 'zoo'")],
    "peas":   [("peace",  "the 'z' sound like in 'zoo'")],
    "please": [("place",  "the 'z' sound like in 'zoo'")],
    "buzz":   [("bus",    "the 'z' sound like in 'zoo'")],
}


# ------ general (non-minimal-pair) confusables ----------------------------

# Word-level confusions where the target and heard word differ by more than
# one sound. The learner isn't confusing two sounds — they're producing a
# different word entirely (grammar slip, disfluency, phrase break). No
# sound_hint; the substitution itself is the correction.

GENERAL_CONFUSABLES: dict[str, list[str]] = {
    "was":     ["wants"],
    "were":    ["where"],
    "of":      ["off"],
    "for":     ["four"],
    "or":      ["are"],
    "and":     ["end"],
    "your":    ["you're"],
    "you're":  ["your"],
    "verify":  ["very"],
    "received": ["recede"],
    "invoice": ["in voice"],
    "wireless": ["wire less"],
    "reliable": ["rely able"],
    "coffee":  ["copy"],
    "walk":    ["work"],
    "work":    ["walk"],
    "prefer":  ["before"],
}


def pick_confusable(word: str) -> tuple[str, str | None] | None:
    """Pick a realistic confusable for `word`.

    Returns `(heard, sound_hint)` — where `sound_hint` is populated only for
    minimal-pair confusions (the target and heard word differ by one sound)
    and None for general word-level confusions. Returns None when no curated
    confusable exists; callers should fall back to another synthesis strategy
    (missing / extra / mispronunciation) rather than emit a placeholder.
    """
    key = word.lower()
    minimal = MINIMAL_PAIR_CONFUSABLES.get(key)
    if minimal:
        heard, hint = minimal[0]
        return heard, hint
    general = GENERAL_CONFUSABLES.get(key)
    if general:
        return general[0], None
    return None


# ==========================================================================
# Lookup helpers used by analysis.py and generate_dataset.py
# ==========================================================================

_WORD_STRIP = re.compile(r"[^\w']+")


def _normalize(word: str) -> str:
    """Lowercase and strip surrounding punctuation the tokenizer leaves in
    (apostrophes are kept — they matter for e.g. 'you're')."""
    return _WORD_STRIP.sub("", word.lower())


# ==========================================================================
# Pattern-level articulation hints
# ==========================================================================
#
# Each per-word coaching hint is paired at lookup time with pattern-level
# articulation coaching from these tables, so the avatar rotates between the
# specific anchor ("in 'knife' the 'k' is silent") and the general rule the
# learner should internalize ("'kn' at the start is always silent"). Real
# pronunciation apps (ELSA / BoldVoice / Speechling) do exactly this — they
# teach the rule, not just the instance.
#
# Silent-letter patterns match on prefix / suffix / substring of the word.

_SILENT_LETTER_PATTERN_HINTS: dict[str, list[str]] = {
    "kn-": [
        "when a word starts with 'kn', the 'k' is always silent",
    ],
    "-mb": [
        "when a word ends in '-mb', the 'b' is silent — just say the 'm'",
    ],
    "-bt": [
        "when 'b' comes before 't', the 'b' is silent",
    ],
    "ps-": [
        "'p' before 's' at the start of a word is silent",
    ],
    "pn-": [
        "'p' before 'n' at the start of a word is silent",
    ],
    "-alk": [
        "in '-alk' words, the 'l' is silent",
    ],
    "-alf": [
        "in '-alf' words, the 'l' is silent",
    ],
    "-ould": [
        "in 'could', 'should', and 'would', the 'l' is silent",
    ],
    "-olk": [
        "in '-olk' words, the 'l' is silent",
    ],
    "gn": [
        "'g' next to 'n' is silent — either at the start of a word or the end",
    ],
    "gh-": [
        "in 'gh-' words, the 'h' is silent — just say the 'g'",
    ],
    "wr-": [
        "when a word starts with 'wr', the 'w' is silent",
    ],
    "-stle": [
        "in '-stle' words, the 't' is silent",
    ],
    "-sten": [
        "in '-sten' words, the 't' is silent",
    ],
    "sc-": [
        "when a word starts with 'sc' followed by 'e' or 'i', the 'c' is silent",
    ],
    # 'h' at the start of certain content words
    "h-honest": [
        "'h' is silent at the start of some words — like 'hour', 'honest', 'honor'",
    ],
}


def _silent_letter_pattern_hints(word: str) -> list[str]:
    """Pattern-level articulation hints that apply to `word` in addition to
    its per-word entry. E.g. 'knife' picks up the 'kn-' pattern hint."""
    w = word.lower()
    hints: list[str] = []
    for pattern, phrases in _SILENT_LETTER_PATTERN_HINTS.items():
        if pattern == "h-honest":
            if w in ("hour", "honest", "honor"):
                hints.extend(phrases)
        elif pattern.startswith("-") and w.endswith(pattern[1:]):
            hints.extend(phrases)
        elif pattern.endswith("-") and w.startswith(pattern[:-1]):
            hints.extend(phrases)
        elif "-" not in pattern and pattern in w:
            hints.extend(phrases)
    return hints


# Hard/soft-c rule-level hints, keyed by the rule tag on each word entry.

_HARD_SOFT_C_RULE_HINTS: dict[str, list[str]] = {
    "soft": [
        "'c' before 'e', 'i', or 'y' usually sounds like 's'",
        "when 'c' meets 'e', 'i', or 'y', it softens to 's'",
    ],
    "hard": [
        "'c' before 'a', 'o', or 'u' usually sounds like 'k'",
    ],
    "cc-ks": [
        "'cc' followed by 'e' or 'i' sounds like 'ks' — first hard, then soft",
    ],
    "cc-k": [
        "'cc' followed by other letters sounds like just 'k'",
    ],
}


# Cluster-level articulation — how to blend, not just what to blend.

_CLUSTER_GENERAL_HINTS: list[str] = [
    "blending all the sounds together smoothly — don't slip a vowel in between",
    "starting on the first sound and rolling straight into the next, no pause",
]


def silent_letter_for(word: str) -> tuple[str, str] | None:
    """Return (silent_letter, hint) if `word` has a curated silent-letter
    entry, else None. Case- and punctuation-insensitive.

    The `hint` is randomly picked from the per-word anchor plus any
    pattern-level articulation hints that apply, so callers get variety
    without having to know about the pool structure."""
    entry = SILENT_LETTER_WORDS.get(_normalize(word))
    if entry is None:
        return None
    letter, anchor_hint = entry
    options = [anchor_hint, *_silent_letter_pattern_hints(word)]
    return letter, random.choice(options)


def hard_soft_c_for(word: str) -> tuple[str, str] | None:
    """Return (rule, hint) if `word` has a curated hard/soft-c entry, else
    None. Randomly picks between the per-word anchor and the rule-level
    articulation matched by the entry's rule tag."""
    entry = HARD_SOFT_C_WORDS.get(_normalize(word))
    if entry is None:
        return None
    rule, anchor_hint = entry
    options = [anchor_hint, *_HARD_SOFT_C_RULE_HINTS.get(rule, [])]
    return rule, random.choice(options)


def cluster_hint_for(word: str) -> tuple[str, str] | None:
    """Return (cluster, hint) if `word` starts or ends with a difficult
    consonant cluster, else None. Randomly picks between the anchor and
    the general blend-smoothly articulation."""
    cluster = find_cluster(_normalize(word))
    if cluster is None:
        return None
    entry = CLUSTER_PATTERNS.get(cluster)
    if entry is None:
        return None
    _, anchor_hint = entry
    options = [anchor_hint, *_CLUSTER_GENERAL_HINTS]
    return cluster, random.choice(options)


def sound_hint_for_substitution(expected: str, heard: str) -> str | None:
    """If (expected, heard) is a known minimal pair, return the sound_hint;
    otherwise None. Called by analysis.py when a real Azure substitution
    happens to match a curated minimal pair, so the runtime path gets the
    same coaching the training set was built with."""
    for candidate, hint in MINIMAL_PAIR_CONFUSABLES.get(expected.lower(), []):
        if candidate.lower() == heard.lower():
            return hint
    return None


# ==========================================================================
# Word → dominant vowel phoneme
# ==========================================================================
#
# phoneme_hints.PHONEME_TRIGGERS only maps *consonant* letter patterns, so
# `plausible_weak_phonemes` never returns a vowel code. That's why the old
# training set had zero vowel coaching. This curated table maps words with
# a salient vowel (usually the one that carries a minimal-pair distinction)
# to their SAPI vowel code, so the generator can synthesize vowel-focused
# mispronunciation rows and Stage 1's correct_hint_for produces the right
# vowel anchor (e.g. "the 'ee' sound like in 'tree'").
#
# Codes match `phoneme_hints._PHONEME_ANCHOR_WORDS` (iy=ee tree, ih=i sit,
# ae=a cat, eh=e bed, aa=o hot, ah=u cup, ao=aw saw, ay=i eye, ey=ay day,
# ow=oa boat, oy=oy boy, uh=oo book, uw=oo food, er=er her).

WORD_VOWEL_ANCHORS: dict[str, str] = {
    # /iː/  ee — tree
    "sheep": "iy", "leave": "iy", "seat": "iy", "feel": "iy",
    "beat": "iy", "reach": "iy", "cheap": "iy", "eat": "iy",
    "green": "iy", "three": "iy", "beach": "iy",

    # /ɪ/  i — sit
    "ship": "ih", "live": "ih", "sit": "ih", "fill": "ih",
    "bit": "ih", "rich": "ih", "chip": "ih", "it": "ih",
    "in": "ih", "sin": "ih",

    # /æ/  a — cat
    "bad": "ae", "man": "ae", "sat": "ae", "pan": "ae",
    "sand": "ae", "cat": "ae", "hat": "ae", "map": "ae",
    "than": "ae", "back": "ae", "black": "ae",

    # /ɛ/  e — bed
    "bed": "eh", "men": "eh", "set": "eh", "pen": "eh",
    "send": "eh", "then": "eh", "get": "eh", "head": "eh",
    "when": "eh", "yes": "eh",

    # /ʌ/  u — cup
    "cup": "ah", "nut": "ah", "cut": "ah", "but": "ah",
    "much": "ah", "come": "ah", "up": "ah",

    # /ɒ/  o — hot
    "cop": "aa", "not": "aa", "hot": "aa", "lot": "aa",
    "stop": "aa", "top": "aa", "shop": "aa",

    # /uː/  oo — food
    "pool": "uw", "food": "uw", "fool": "uw", "soup": "uw",
    "two": "uw", "shoe": "uw", "true": "uw",

    # /ʊ/  oo — book
    "pull": "uh", "foot": "uh", "full": "uh", "book": "uh",
    "look": "uh", "good": "uh",

    # /ɔː/  aw — saw
    "saw": "ao", "law": "ao", "call": "ao", "small": "ao",
    "walk": "ao", "talk": "ao",

    # /aɪ/  i — eye (long i / diphthong)
    "eye": "ay", "mild": "ay", "kind": "ay", "find": "ay",
    "night": "ay", "light": "ay", "time": "ay", "life": "ay",
    "mile": "ay", "high": "ay",

    # /eɪ/  ay — day
    "day": "ey", "way": "ey", "say": "ey", "play": "ey",
    "made": "ey", "name": "ey", "late": "ey", "great": "ey",

    # /oʊ/  oa — boat
    "boat": "ow", "go": "ow", "so": "ow", "no": "ow",
    "know": "ow", "note": "ow", "home": "ow",

    # /ɔɪ/  oy — boy
    "boy": "oy", "toy": "oy", "voice": "oy", "choice": "oy",
    "join": "oy", "point": "oy", "noise": "oy",

    # /aʊ/  ow — now
    "now": "aw", "how": "aw", "out": "aw", "about": "aw",
    "found": "aw", "sound": "aw", "brown": "aw",

    # /ɜːr/  er — her
    "her": "er", "were": "er", "learn": "er", "first": "er",
    "world": "er", "work": "er", "third": "er", "person": "er",
    "word": "er",
}


def word_vowel_for(word: str) -> str | None:
    """Return the dominant vowel phoneme code for `word` if curated, else
    None. Used by generate_dataset.py to synthesize vowel-mispronunciation
    rows the old training set had zero coverage of."""
    return WORD_VOWEL_ANCHORS.get(_normalize(word))
