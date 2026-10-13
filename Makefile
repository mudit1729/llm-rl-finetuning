# wordle-rl: one target per step. Mac-side targets run locally; vast-* and the
# training/eval targets drive a vast.ai instance over SSH (all GPU work is remote).
#
#   make help
#
# Targets whose script is not written yet fail with a message naming the phase.

SHELL := /bin/bash
.DEFAULT_GOAL := help

OFFER   ?=
HOURS   ?=
PROFILE ?= smoke
PHASE   ?=
STEP    ?=

need = @test -x $(1) || { echo "$(1) is not written yet ($(2))."; exit 1; }

.PHONY: help
help: ## List targets
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n",$$1,$$2}'

# ---------------------------------------------------------------- Phase 1 (Mac)
.PHONY: setup local-serve local-test local-summary
setup: ## Install Mac-side tooling (verifiers + wordle at the pinned commit)
	uv sync

local-serve: ## Serve Qwen3-1.7B (4-bit MLX) on 127.0.0.1:8080 (foreground)
	./scripts/local_serve.sh

local-test: ## Play ~5 Wordle games against the local server (needs local-serve)
	uv run vf-eval @ configs/local_eval.toml --no-rich

local-summary: ## Summarize the latest local smoke-test run
	uv run python scripts/summarize_eval.py results/local

# ---------------------------------------------------------- Phase 2 (vast.ai)
.PHONY: vast-offers vast-up vast-setup vast-push vast-pull vast-ssh vast-dashboard vast-status vast-watchdog vast-down
vast-offers: ## Top 5 offers, creates nothing: make vast-offers PROFILE=smoke|wordle [HOURS=2]
	$(call need,scripts/vast_up.sh,Phase 2)
	./scripts/vast_up.sh search $(PROFILE) $(if $(HOURS),--hours $(HOURS))

vast-up: ## Rent the offer you picked: make vast-up OFFER=<id> HOURS=<expected>
	$(call need,scripts/vast_up.sh,Phase 2)
	./scripts/vast_up.sh create $(OFFER) --hours $(HOURS)

vast-setup: ## On the instance (tmux job): uv, prime-rl @ pinned SHA, submodules, uv sync, checks
	$(call need,scripts/vast_setup.sh,Phase 2)
	./scripts/remote_job.sh start setup && ./scripts/remote_job.sh wait setup

vast-push: ## Push configs/ and scripts/ to the instance
	$(call need,scripts/vast_sync.sh,Phase 2)
	./scripts/vast_sync.sh push

vast-pull: ## Pull results, logs and final checkpoints back to the Mac
	$(call need,scripts/vast_sync.sh,Phase 2)
	./scripts/vast_sync.sh pull

vast-ssh: ## Open a shell on the instance
	$(call need,scripts/vast_ssh.sh,Phase 2)
	./scripts/vast_ssh.sh

vast-dashboard: ## Forward the prime-rl dashboard to http://localhost:7788
	$(call need,scripts/vast_ssh.sh,Phase 2)
	./scripts/vast_ssh.sh --dashboard

vast-status: ## Running jobs, exit codes and GPU use on the instance
	./scripts/remote_job.sh status

vast-watchdog: ## Destroy the instance after 15 idle min (pulls first); touch .vast/keep to pause
	./scripts/vast_watchdog.sh $(if $(PHASE),--phase $(PHASE))

vast-down: ## Pull, destroy, confirm gone with `vastai show instances`: [PHASE=3]
	$(call need,scripts/vast_down.sh,Phase 2)
	./scripts/vast_down.sh $(if $(PHASE),--phase $(PHASE))

# Provider-neutral aliases for the GPU machine lifecycle (same as the vast-* targets).
.PHONY: gpu-offers gpu-up gpu-setup gpu-down
gpu-offers: vast-offers ## Alias: list GPU offers (creates nothing)
gpu-up: vast-up         ## Alias: rent the offer you picked
gpu-setup: vast-setup   ## Alias: install prime-rl on the GPU machine
gpu-down: vast-down     ## Alias: pull results, release the machine

# ------------------------------------------- Phase 3-4 (remote GPU jobs, in tmux)
# Each runs scripts/remote_job.sh <step> on the instance inside tmux, logging to
# ~/wordle-rl/logs/<step>.log. `make attach STEP=<step>` follows a job.
.PHONY: wait smoke split data eval-base sft eval-sft rl eval-rl rl-lora eval-lora attach
smoke:     ## Phase 3: prime-rl reverse-text example end to end
	$(call need,scripts/remote_job.sh,Phase 3)
	./scripts/remote_job.sh start smoke
split:     ## Phase 4 (Mac): held-out split -> configs/heldout_split.json + RL configs (~7 min)
	uv run python scripts/make_split.py
data:      ## Phase 4: on the instance, drop SFT games whose answer is an eval word
	$(call need,scripts/remote_job.sh,Phase 4)
	./scripts/remote_job.sh start data
eval-base: ## Phase 4a: baseline eval of Qwen/Qwen3-1.7B (~200 games)
	$(call need,scripts/remote_job.sh,Phase 4)
	./scripts/remote_job.sh start eval-base
sft:       ## Phase 4b: SFT warmup (sft.toml)
	$(call need,scripts/remote_job.sh,Phase 4)
	./scripts/remote_job.sh start sft
eval-sft:  ## Phase 4b: eval the SFT checkpoint
	$(call need,scripts/remote_job.sh,Phase 4)
	./scripts/remote_job.sh start eval-sft
rl:        ## Phase 4c: full-weight RL (rl.toml, 2-GPU overrides)
	$(call need,scripts/remote_job.sh,Phase 4)
	./scripts/remote_job.sh start rl
eval-rl:   ## Phase 4c: eval the RL checkpoint
	$(call need,scripts/remote_job.sh,Phase 4)
	./scripts/remote_job.sh start eval-rl
rl-lora:   ## Phase 4d: LoRA RL with the same settings
	$(call need,scripts/remote_job.sh,Phase 4)
	./scripts/remote_job.sh start rl-lora
eval-lora: ## Phase 4d: eval the LoRA RL checkpoint
	$(call need,scripts/remote_job.sh,Phase 4)
	./scripts/remote_job.sh start eval-lora
attach:    ## Tail a remote job's log: make attach STEP=rl
	$(call need,scripts/remote_job.sh,Phase 3)
	./scripts/remote_job.sh tail $(STEP)
wait:      ## Block until a remote job ends: make wait STEP=rl
	./scripts/remote_job.sh wait $(STEP)

# ------------------------------------------------------------- Phase 5 (report)
.PHONY: report page
page: ## Refresh notes/experiment.html with the latest pulled metrics and evals (then republish)
	uv run python scripts/build_page.py

report: ## Build results/results.md and the plots from pulled artifacts
	$(call need,scripts/make_report.py,Phase 5)
	uv run python scripts/make_report.py
