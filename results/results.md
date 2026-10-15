# Results: Qwen3-1.7B on Wordle (prime-rl)

Eval: `configs/wordle/eval.toml` for every model: 20 held-out games (seeds 0-19, 18 secret words
never seen in SFT or RL) × 10 rollouts = 200 games, temperature 1.0, 1024 tokens per turn.

| Model | Win rate (95% CI) | Avg reward | Avg turns | Guesses to solve | Format-error rate | GPU-hours |
|---|---|---|---|---|---|---|
| Qwen3-1.7B (base) | 0.0% (0–0%) | 0.17 | 3.33 | no wins | 61.9% | 0.00 |
| + SFT warmup | 2.5% (0–6%) | 0.30 | 5.11 | 4.20 | 13.9% | 0.16 |
| + SFT + RL (full) | 15.0% (10–20%) | 0.43 | 5.83 | 4.47 | 9.4% | 7.44 |
| + SFT + RL (LoRA r32) | 21.0% (12–31%) | 0.47 | 6.02 | 4.38 | 11.0% | 9.56 |
| + SFT + RL (per-turn credit) | 17.0% (10–24%) | 0.45 | 6.96 | 4.32 | 2.4% | 4.94 |
| + structured-thinking SFT | 11.5% (5–20%) | 0.42 | 6.00 | 4.57 | 4.1% | 0.31 |
| + structured SFT + RL (reweighted) | 45.5% (30–61%) | 0.66 | 5.62 | 4.36 | 12.3% | 10.31 |
| + structured SFT + RL, rerun | 41.0% (28–56%) | 0.66 | 5.61 | 4.29 | 6.6% | 4.80 |

### Same eval with reasoning stripped (diagnostic)

`configs/wordle/eval.toml` unchanged; the vLLM server adds `--vllm.reasoning-parser qwen3`, so the game
sees only the answer after `</think>`, as in training (prime-rl's renderer splits it there). The example
eval passes the raw text, and the game plays the first bracketed word, often a guess quoted in the thinking.

| Model | Win rate (95% CI) | Avg reward | Avg turns | Guesses to solve | Format-error rate |
|---|---|---|---|---|---|
| Qwen3-1.7B (base) | 0.0% (0–0%) | 0.21 | 3.72 | no wins | 47.7% |
| + SFT warmup | 12.5% (8–18%) | 0.40 | 5.31 | 4.08 | 31.2% |
| + SFT + RL (full) | 40.5% (28–53%) | 0.66 | 5.78 | 4.27 | 22.5% |
| + SFT + RL (LoRA r32) | 42.5% (30–55%) | 0.65 | 5.53 | 4.16 | 22.8% |
| + SFT + RL (per-turn credit) | 41.5% (28–56%) | 0.64 | 5.88 | 4.53 | 9.1% |
| + structured-thinking SFT | 0.5%† (0–2%) | 0.24 | 3.97 | 3.00 | 56.0% |
| + structured SFT + RL (reweighted) | 2.5%† (0–5%) | 0.29 | 4.09 | 3.20 | 55.4% |
| + structured SFT + RL, rerun | 5.5%† (2–10%) | 0.33 | 4.25 | 2.64 | 53.6% |

† Not comparable: with the parser on, the eval client sends earlier turns back without their
thinking; the structured-thinking models were trained only on games that keep it, and without it
they stop giving an answer. Training and the example eval keep it.

### Held-out vs training words (diagnostic)

Same eval settings on 20 training games (`configs/wordle/eval_seen.toml`), 10 tries each.

| Model | Win rate, held-out | Win rate, training words | Reward, held-out | Reward, training words |
|---|---|---|---|---|
| + SFT + RL (full) | 15.0% | 15.5% | 0.43 | 0.46 |
| + SFT + RL (LoRA r32) | 21.0% | 11.0% | 0.47 | 0.41 |

- **Win rate**: games solved within 6 guesses. **Avg reward**: TextArena game reward (1.0 win; otherwise
  (greens + 0.5 × yellows) / 5 of the last valid guess). **Avg turns**: model turns per game, invalid
  attempts included. **Guesses to solve**: valid guesses in won games (1-6), the usual Wordle score.
- **Format-error rate**: share of model turns that did not produce a well-formed move: cut off at the
  1024-token limit, or rejected for format (no `[word]`, or not 5 letters). A cut-off `<think>` has no
  `</think>`, so vLLM returns the whole thought as the answer and the game reads the first bracketed word
  in it (usually a quoted earlier guess) as a repeat; those turns count as format errors here.
- **GPU-hours**: training compute behind each model, cumulative (SFT; SFT + RL), from run
  timestamps × 2 GPUs. Setup and eval time are not included.

Plots: `plots/train_reward.png`, `plots/win_rate.png`.
