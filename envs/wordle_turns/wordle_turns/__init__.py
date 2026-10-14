"""wordle-turns: the `wordle` taskset's tasks, played by an env that records per-turn shaped rewards."""

from wordle.taskset import WordleTaskset

from wordle_turns.env import WordleTurnsEnv, WordleTurnsEnvConfig


class WordleTurnsTaskset(WordleTaskset):
    """Identical tasks to `wordle` (seed i = task i), so seed selection carries over."""


__all__ = ["WordleTurnsEnv", "WordleTurnsEnvConfig", "WordleTurnsTaskset"]
