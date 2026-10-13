"""Held-out Wordle split: eval words never appear in training.

The verifiers `wordle` taskset is procedural (task i = TextArena Wordle-v0 seeded with i)
and draws secret words from only 156 nouns, so seeds alone don't hold words out.
This replays the engine's seeding (exactly as TextArenaEnv.run does) to get each seed's
secret word, writes configs/heldout_split.json, and injects the train selection into the
RL configs (scripts/apply_split.py):

  eval   seeds 0..19 (the prime-rl example's eval set: select.limit = 20)
  train  seeds 20..20+N, minus every seed whose word is an eval word

prime-rl materializes a closed include range and cycles it in order, so training never
reaches an excluded seed however long it runs.

    uv run python scripts/make_split.py [--train-seeds 6000]
"""

import argparse
import contextlib
import copy
import io
import json
import random
from pathlib import Path

import textarena as ta

ROOT = Path(__file__).resolve().parent.parent
N_EVAL = 20


def secret_word(seed: int) -> str:
    """Exactly what TextArenaEnv.run does: seed, make, reset (slow: make rebuilds the word list)."""
    random.seed(seed)
    env = ta.make(env_id="Wordle-v0")
    env.reset(num_players=1)
    return _inner(env).state.game_state["secret_word"]


def _inner(env):
    while hasattr(env, "env"):
        env = env.env
    return env


_TEMPLATE = None


def secret_word_fast(seed: int) -> str:
    """Seed, then reset a copy of one prebuilt env (what the taskset's load() does).
    Equivalent to secret_word() because make() draws nothing from the global RNG;
    main() checks that on a sample of seeds."""
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = ta.make(env_id="Wordle-v0")
    random.seed(seed)
    env = copy.deepcopy(_TEMPLATE)
    env.reset(num_players=1)
    return _inner(env).state.game_state["secret_word"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seeds", type=int, default=6000, help="size of the train seed range before exclusion")
    args = ap.parse_args()

    with contextlib.redirect_stderr(io.StringIO()):
        start, stop = N_EVAL, N_EVAL + args.train_seeds
        sample = list(range(N_EVAL)) + random.Random(0).sample(range(start, stop), 30)
        mismatch = [s for s in sample if secret_word(s) != secret_word_fast(s)]
        assert not mismatch, f"fast path disagrees with the env's seeding on seeds {mismatch}"
        eval_words = {s: secret_word_fast(s) for s in range(N_EVAL)}
        held_out = set(eval_words.values())
        train_words = {s: secret_word_fast(s) for s in range(start, stop)}
    excluded = [s for s, w in train_words.items() if w in held_out]
    kept_words = {w for s, w in train_words.items() if w not in held_out}

    (ROOT / "configs" / "heldout_split.json").write_text(json.dumps(
        {"eval_seeds": eval_words, "held_out_words": sorted(held_out), "train_range": [start, stop],
         "excluded_train_seeds": excluded, "train_words": sorted(kept_words)}, indent=1))
    assert not kept_words & held_out
    print(f"eval: {N_EVAL} seeds, {len(held_out)} words | train: {args.train_seeds - len(excluded)} seeds, "
          f"{len(kept_words)} words | excluded {len(excluded)} seeds | overlap: 0")
    import apply_split  # noqa: E402  (same dir)
    apply_split.main()


if __name__ == "__main__":
    main()
