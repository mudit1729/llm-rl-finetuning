"""Per-turn shaping for Wordle: pure functions, no verifiers/textarena imports.

Each model turn gets a small shaped reward; the episode keeps a dominant
outcome reward. The turn-level credit algorithm (prime-rl `turn_grpo`) turns
these into per-turn advantages.

    r_t = w_info · IG_t                    information gain of a valid guess, in [0, 1]
          − p_invalid · [move rejected]    no [word], wrong length, repeat, not a word
          − p_cutoff · [hit token limit]   thought ran past the per-turn budget
          − p_length · ramp(tokens)        0 below soft_start, 1 at the budget
          − p_inconsistent · [valid guess contradicts earlier feedback]

    IG_t = log2(|C_before| / |C_after|) / log2(|pool|)

C is the set of pool words still consistent with all feedback so far. The pool is
the game's own secret-word list (156 nouns), known to the environment, never shown
to the model.

    outcome = win · (1 + speed_bonus · guesses_left)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

WORD_LEN = 5
MAX_GUESSES = 6


def feedback(guess: str, secret: str) -> str:
    """Wordle feedback with TextArena's duplicate-letter rules: G, Y or X per letter."""
    out = [""] * WORD_LEN
    remaining: list[str | None] = list(secret)
    for i, ch in enumerate(guess):
        if ch == secret[i]:
            out[i] = "G"
            remaining[i] = None
    for i, ch in enumerate(guess):
        if out[i]:
            continue
        if ch in remaining:
            out[i] = "Y"
            remaining[remaining.index(ch)] = None
        else:
            out[i] = "X"
    return "".join(out)


def candidates(pool: Iterable[str], history: Sequence[tuple[str, str]]) -> list[str]:
    """Pool words that would have produced exactly the feedback seen so far."""
    return [w for w in pool if all(feedback(g, w) == fb for g, fb in history)]


def consistent(guess: str, history: Sequence[tuple[str, str]]) -> bool:
    """Whether `guess` could be the answer given the feedback so far (pool-independent)."""
    return all(feedback(g, guess) == fb for g, fb in history)


@dataclass(frozen=True)
class ShapingWeights:
    info: float = 0.5
    invalid: float = 0.5
    cutoff: float = 0.5
    length: float = 0.2
    inconsistent: float = 0.2
    soft_start: int = 768
    budget: int = 1024
    speed_bonus: float = 0.1


@dataclass
class TurnRecord:
    """One model turn, as stored in trace.info["turns"] (JSON-serializable)."""

    index: int
    tokens: int
    cutoff: bool
    move: str | None  # the bracketed word the game read, if any
    accepted: bool  # the game accepted the guess (it entered the guess history)
    rejected_reason: str | None
    consistent: bool | None  # accepted guesses only
    cand_before: int | None
    cand_after: int | None
    info_gain: float
    reward: float

    def to_dict(self) -> dict:
        return asdict(self)


def info_gain(cand_before: int, cand_after: int, pool_size: int) -> float:
    if cand_before <= 0 or cand_after <= 0 or pool_size <= 1:
        return 0.0
    return max(0.0, math.log2(cand_before / cand_after)) / math.log2(pool_size)


def length_ramp(tokens: int, w: ShapingWeights) -> float:
    if tokens <= w.soft_start:
        return 0.0
    return min(1.0, (tokens - w.soft_start) / max(1, w.budget - w.soft_start))


def turn_reward(*, accepted: bool, cutoff: bool, tokens: int, is_consistent: bool | None, ig: float,
                w: ShapingWeights) -> float:
    r = w.info * ig if accepted else -w.invalid
    if cutoff:
        r -= w.cutoff
    r -= w.length * length_ramp(tokens, w)
    if accepted and is_consistent is False:
        r -= w.inconsistent
    return r


def outcome_reward(won: bool, valid_guesses: int, w: ShapingWeights) -> float:
    if not won:
        return 0.0
    return 1.0 + w.speed_bonus * max(0, MAX_GUESSES - valid_guesses)


def answer_text(reply: str) -> str:
    """The part of a reply the game should read: text after the last </think>.

    A reply that opens <think> but never closes it was cut off mid-thought, so it has
    no answer (''). Replies without think tags are returned unchanged. This makes the
    env independent of whether the inference server runs a reasoning parser."""
    if "</think>" in reply:
        return reply.rsplit("</think>", 1)[1].strip()
    if "<think>" in reply:
        return ""
    return reply.strip()
