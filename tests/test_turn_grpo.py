"""Tests for the prime-rl `turn_grpo` patch, runnable on the Mac.

prime-rl itself only installs on Linux+CUDA, so this loads the patched sources
(algo/base.py, routing.py, turn_grpo.py) with stub modules for the GPU-side imports,
and scores REAL training traces from the Phase 4 RL run (token ids + masks from the
renderer) to check that per-turn advantages land on the right tokens.

Needs: a patched prime-rl checkout at $PRIME_RL_SRC (default: the session's reference clone)
and artifacts/*/results/train_samples/wordle-rl.jsonl (pulled from the instance).
"""

import asyncio
import glob
import importlib.util
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import verifiers.v1 as vf

ROOT = Path(__file__).resolve().parent.parent
PRIME_RL_SRC = Path(os.environ.get(
    "PRIME_RL_SRC",
    "/private/tmp/claude-501/-Users-muditj-Desktop-Projects-RL-finetuning/b78f09e3-92c3-4d26-a8f3-e9840dfa72d5/scratchpad/ref/prime-rl",
))
ALGO = PRIME_RL_SRC / "src/prime_rl/orchestrator/algo"


@dataclass
class TurnGRPOAlgoConfig:
    turn_weight: float = 0.5
    discount: float = 0.9
    turn_info_key: str = "turns"
    action_loss_type: str = "rl"


def _load_patched():
    if not (ALGO / "turn_grpo.py").exists():
        pytest.skip(f"no patched prime-rl at {PRIME_RL_SRC}")
    stubs = {
        "prime_rl": types.ModuleType("prime_rl"),
        "prime_rl.configs": types.ModuleType("prime_rl.configs"),
        "prime_rl.configs.algorithm": types.SimpleNamespace(
            AlgoConfig=object, FrozenModelConfig=object, ActionLossType=str, TurnGRPOAlgoConfig=TurnGRPOAlgoConfig),
        "prime_rl.orchestrator": types.ModuleType("prime_rl.orchestrator"),
        "prime_rl.orchestrator.clients": types.SimpleNamespace(InferenceClient=object, connect_frozen_client=None),
        "prime_rl.orchestrator.algo": types.ModuleType("prime_rl.orchestrator.algo"),
        "prime_rl.transports": types.ModuleType("prime_rl.transports"),
        "prime_rl.transports.batch": types.SimpleNamespace(TrainingSample=object),
    }
    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)
    loaded = {}
    for mod in ("base", "routing", "turn_grpo"):
        name = f"prime_rl.orchestrator.algo.{mod}"
        spec = importlib.util.spec_from_file_location(name, ALGO / f"{mod}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        loaded[mod] = module
    return loaded


def test_turn_advantages_math():
    ta = _load_patched()["turn_grpo"].turn_advantages
    # Two rollouts, equal outcomes: only the turn term remains.
    adv = ta([1.0, 1.0], [[0.5, 0.0], [0.5, -1.0]], turn_weight=1.0, discount=0.5)
    # baselines: t0 = 0.5, t1 = -0.5 -> a = [[0, 0.5], [0, -0.5]]; G = [[0.25, 0.5], [-0.25, -0.5]]
    assert adv == [[0.25, 0.5], [-0.25, -0.5]]
    # Outcome term only (no turn rewards differ): equals GRPO.
    adv = ta([1.3, 0.0], [[0.2], [0.2]], turn_weight=0.5, discount=0.9)
    assert adv == [[0.65], [-0.65]]
    # A turn only one rollout reached has zero turn credit.
    adv = ta([0.0, 0.0], [[0.1], [0.1, -1.0]], turn_weight=1.0, discount=1.0)
    assert adv == [[0.0], [0.0, 0.0]]


def _training_episodes():
    files = sorted(glob.glob(str(ROOT / "artifacts/*/results/train_samples/wordle-rl.jsonl")))
    if not files:
        pytest.skip("no pulled training traces")
    return [vf.Episode.model_validate(json.loads(line)) for line in open(files[-1]) if line.strip()]


def test_per_turn_advantages_land_on_each_turns_tokens():
    mods = _load_patched()
    trainable_nodes = mods["routing"].trainable_nodes
    episodes = _training_episodes()[:16]
    # One "group": attach synthetic per-turn rewards (turn k of rollout i gets reward i/10 - k/10).
    traces = [t for _, t in mods["base"].iter_trainable_traces(episodes)]
    assert traces, "no trainable traces in the sample"
    for i, trace in enumerate(traces):
        n_turns = sum(1 for node in trace.nodes if node.sampled)
        trace.info["turns"] = [{"reward": i / 10 - k / 10} for k in range(n_turns)]
    algo = mods["turn_grpo"].TurnGRPOAlgorithm(TurnGRPOAlgoConfig(), clients=None)
    asyncio.run(algo.score_group(episodes))

    matched = 0
    for trace in traces:
        nodes = trainable_nodes(trace)
        assert len(nodes) == len(trace.info["turns"]), "one trainable node per model turn"
        values = [node.advantages for node in nodes]
        assert all(v is not None and len(v) == sum(node.mask) for v, node in zip(values, nodes))
        assert all(len(set(v)) == 1 for v in values), "a turn's tokens share one advantage"
        if len({v[0] for v in values}) > 1:
            matched += 1
    assert matched > 0, "per-turn advantages differ across turns"


def test_fallback_without_turn_records():
    mods = _load_patched()
    episodes = _training_episodes()[16:32]
    algo = mods["turn_grpo"].TurnGRPOAlgorithm(TurnGRPOAlgoConfig(), clients=None)
    asyncio.run(algo.score_group(episodes))
    traces = [t for _, t in mods["base"].iter_trainable_traces(episodes)]
    rewards = [t.reward for t in traces]
    mean = sum(rewards) / len(rewards)
    for trace in traces:
        vals = {v for node in mods["routing"].trainable_nodes(trace) for v in node.advantages}
        assert len(vals) == 1, "no turn records -> one scalar advantage per trace (plain GRPO)"
        assert abs(vals.pop() - (trace.reward - mean)) < 1e-6
