# Qwen3-1.7B learns Wordle with prime-rl: report

**Bottom line.** Training worked. On 200 games over 18 secret words the model never saw in training,
SFT + RL raised the win rate from **0% to 15% (full fine-tune) and 21% (LoRA)** on prime-rl's example
eval. When the eval strips the model's reasoning before the game reads the move, as training does, the
win rates are **40.5% and 42.5%**. Average reward rose from 0.17 to 0.43–0.47 (example eval) or from
0.21 to 0.65–0.66 (reasoning stripped). Turns-to-solve did not go down.

## Results (200 held-out games: 20 words × 10 tries, T = 1.0, 1,024 tokens per turn)

| Model | Win rate, example eval | Win rate, reasoning stripped | Avg reward (example / stripped) | Guesses to solve (stripped) | Format errors / turn (example) | Training GPU-h |
|---|---|---|---|---|---|---|
| Qwen3-1.7B base | 0.0% | 0.0% | 0.17 / 0.21 | no wins | 61.9% | 0 |
| + SFT warmup (20 steps) | 2.5% (0–6) | 12.5% (8–18) | 0.30 / 0.40 | 4.08 | 13.9% | 0.16 |
| + RL, full fine-tune (200 steps) | 15.0% (10–20) | **40.5%** (28–53) | 0.43 / 0.66 | 4.27 | 9.4% | 7.44 |
| + RL, LoRA r32 (200 steps) | **21.0%** (12–31) | **42.5%** (30–55) | 0.47 / 0.65 | 4.16 | 11.0% | 9.56 |
| + RL, per-turn credit (200 steps) | 17.0% (10–24) | 41.5% (28–56) | 0.45 / 0.64 | 4.53 | 2.4% | 4.94 |

95% intervals resample the 20 eval games. Full tables: `results/results.md`. Plots: `results/plots/`.

## Run 2: per-turn credit (the main bet from notes/IDEAS.md)

**Setup.** A new `wordle-turns` env scores every turn:
- information gain over the 156-word list;
- penalties for rejected moves, turns cut off at the token limit, thinking past 768 tokens, and guesses that contradict earlier clues;
- a win reward scaled by speed.

A small prime-rl patch (`turn_grpo`) gives each turn's tokens its own advantage: the GRPO outcome advantage plus discounted per-turn credit. The SFT recipe and the 200 full-fine-tune RL steps are otherwise the same as the baseline.

**Results** (thinking removed, 200 held-out games):

| | Baseline full RL | Per-turn credit |
|---|---|---|
| Win rate | 40.5% | 41.5% |
| Turns cut off at 1,024 tokens | 15% | **1.4%** |
| Format errors per turn | 22.5% | **9.1%** |
| Games lost to invalid moves | 88 | **39** |
| Games lost after all 6 guesses | 31 | 78 |
| Guesses consistent with earlier clues | 54% | 47% |
| Guesses that could still be the answer | 14% | 12% |

**Takeaway.** The cut-off and validity fixes worked. Run 2 also trained about 30% faster, because thinking got shorter. The win rate did not move: the games it rescued now run out of guesses, because the model uses its clues *worse*. The information-gain bonus pays for exploratory guesses that cannot be the answer, and the clue-violation penalty is too weak to offset it.

**Next run (run 3):**
- Lower the information weight to about 0.1.
- Raise the clue-violation penalty to about 0.5, and lower the turn weight to about 0.3, so that winning with consistent guesses dominates.
- Add the clue-state summary to observations (idea 3 in IDEAS.md).
- Train longer; the reward was still rising at step 200.

## Success criteria

- **Win rate up:** met. 0% → 15–21% (example eval), or 0% → 40–42% (reasoning stripped).
- **Average reward up:** met. 0.17 → 0.43–0.47, or 0.21 → 0.65–0.66.
- **Turns-to-solve down:** not met. The base model never wins, so it has no baseline. From SFT to RL, valid
  guesses per win stayed flat (4.08 → 4.27 / 4.16, within noise; SFT has only 25 wins). Games got longer on
  average because fewer of them end early on an invalid move.

## What we found

1. **The example's eval words were not held out.** Eval seeds 0–19 are also the first training seeds;
   all 156 secret words recur constantly; 98 of the 1,000 SFT games end on eval words. We held out the
   18 eval words everywhere (RL: 5,325 seeds over 138 words; SFT: 902 games), and verified it.
2. **The learning generalizes.** On 20 *training* words the RL models score about the same as on
   held-out words (full: 15.5% vs 15.0%; LoRA: 11.0% vs 21.0%). The gains are not memorized words.
3. **The example eval and training feed the game different text.** In training, prime-rl's renderer
   splits off `<think>` and the game sees only the answer. In the eval, the vLLM server has no reasoning
   parser, so the game reads the first bracketed word anywhere in the reply. That is often a guess the
   model quoted while thinking ("the feedback for [crane]…"), so it gets rejected as a repeat. Adding
   `--vllm.reasoning-parser qwen3` to the eval server changes nothing else and moves full RL from 15% to
   40.5%. The base model's reward with it (0.21) matches prime-rl's published baseline (~0.2), which
   suggests the published numbers were measured this way.
4. **LoRA matched full fine-tuning.** The training reward tracked step for step, and the evals agree within
   their intervals. It used 9 GiB of trainer memory against 30 GiB, but each prime-rl step was slower (~80 s vs ~62 s).
5. **Where the model still fails:** with reasoning stripped, 18% of the RL model's turns run out of the
   1,024-token budget mid-thought and produce no move. Shorter thinking (a length penalty) or a bigger
   per-turn budget is the clearest next gain.
6. **Compared with prime-rl's ~60%:** ours is 40–42% on truly held-out words, with half the RL data
   (25.6k vs 51.2k games) and one inference GPU instead of six.

## How it ran

- **Code:** prime-rl `1e53181` (verifiers `2cb0a6d`), torch 2.13 cu130, vLLM 0.31.
- **Setup:** Mac for code; vast.ai 2× RTX 3090 for the smoke test (reverse-text: 0.047 → 0.813, matching
  the README) and 2× A100-SXM4-80GB for Wordle. One inference GPU and one trainer GPU, batch 128 games
  (8 words × 16 tries), seq 8,192.
- **Fixes along the way:**
  - CUDA 13 image (not 12).
  - `UV_NO_SYNC=1` (GitHub 504s).
  - `python3.12-dev` for Triton.
  - A 16-worker env pool. TextArena takes 0.87 s of synchronous CPU to build a game, which stalled the default single worker; a step went from 3.5 min to ~70 s.
  - Deleting optimizer state after export to fit the 100 GB disk.

| Phase | Hardware | Hours |
|---|---|---|
| 3 smoke test | 2× RTX 3090 | 1.30 |
| 4 Wordle (incl. 9 diagnostic evals) | 2× A100 SXM4 | 10.26 |

## Next steps

1. Make the reasoning-parser eval the standard one, and report the example eval only for comparison.
2. Add prime-rl's length penalty, or raise per-turn tokens to 2,048, to cut the 18% of empty turns.
3. Train longer or with a larger batch (the reward was still rising at step 200), on 4–8 GPUs if approved.

Artifacts on the Mac: eval traces, metrics and logs in `artifacts/wordle-rl-wordle-55623081/`, and
the LoRA adapter in `…/outputs/wordle-rl-lora/lora_adapter_step_200/`. Full-model weights were not kept.
