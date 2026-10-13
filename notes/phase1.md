# Phase 1: local smoke test (2026-10-12)

Setup: verifiers @ 2cb0a6d (prime-rl's pinned submodule) + wordle taskset on the Mac (uv, Python 3.12);
Qwen3-1.7B served by mlx-lm 4-bit on 127.0.0.1:8080; temperature 1.0, 1024 max tokens per turn
(the example eval's settings). Config: `configs/local_eval.toml`. 5 games (seeds 0-4), 1 rollout each.

| games | wins | avg reward | avg turns | format-error games | games ended by invalid move | turns truncated at 1024 tok | crashes |
|---|---|---|---|---|---|---|---|
| 5 | 0 | 0.10 | 3.4 | 5/5 | 5/5 | 9/17 (53%) | 0 |

What happens: Qwen3 thinks for 800-1000+ tokens per turn. About half of turns hit the 1024-token cap
before any `[word]` appears (counted as a format error), and the model often repeats an earlier guess.
Two strikes end the game. This matches the prime-rl example's description of the base model
(about 0% wins, mostly format failures). These numbers come from a 4-bit model, so they are a plumbing
check only.

Notes
- The vf-eval log's `rollout done ... reward=0.000` is printed before TextArenaEnv records the game
  reward. The trace's `rewards.game_reward.score` is authoritative, and scripts/summarize_eval.py reads that.
- Reproduce: `make local-serve` (terminal 1), `make local-test`, `make local-summary`.
