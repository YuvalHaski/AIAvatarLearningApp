"""Word-level alignment of the target sentence against what was recognized.

This is a Needleman-Wunsch / Levenshtein edit-distance alignment over word
tokens. It is fully deterministic and is what tells us, precisely, which words
were missing, extra, or substituted - independent of the ASR's own miscue flags.
"""
import re
from dataclasses import dataclass

_NON_WORD = re.compile(r"[^\w']+")


@dataclass
class AlignmentOp:
    """One step of the alignment.

    kind: match | substitution | missing | extra
    - match/substitution carry both target and recognized words
    - missing carries only the target word (the learner skipped it)
    - extra carries only the recognized word (the learner added it)
    """
    kind: str
    target: str | None
    recognized: str | None


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into comparable word tokens."""
    return [tok for tok in _NON_WORD.split(text.lower().strip()) if tok]


def align(target_tokens: list[str], recognized_tokens: list[str]) -> list[AlignmentOp]:
    """Align two token lists; returns the ordered list of edit operations.

    Costs: match = 0, substitution = 1, missing = 1, extra = 1.
    """
    n, m = len(target_tokens), len(recognized_tokens)

    # dp[i][j] = min edit cost to align target[:i] with recognized[:j].
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if target_tokens[i - 1] == recognized_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + sub_cost,  # match / substitution
                dp[i - 1][j] + 1,             # missing  (target word skipped)
                dp[i][j - 1] + 1,             # extra    (word added)
            )

    # Backtrack from dp[n][m] to dp[0][0].
    ops: list[AlignmentOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            same = target_tokens[i - 1] == recognized_tokens[j - 1]
            sub_cost = 0 if same else 1
            if dp[i][j] == dp[i - 1][j - 1] + sub_cost:
                ops.append(
                    AlignmentOp(
                        kind="match" if same else "substitution",
                        target=target_tokens[i - 1],
                        recognized=recognized_tokens[j - 1],
                    )
                )
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(AlignmentOp(kind="missing", target=target_tokens[i - 1], recognized=None))
            i -= 1
            continue
        ops.append(AlignmentOp(kind="extra", target=None, recognized=recognized_tokens[j - 1]))
        j -= 1

    ops.reverse()
    return ops
