"""Shared differentiable rollouts, progress reporting, and device synchronization."""

import json

import torch

from hedging_gym.environment.rollout import run_episode

from .controllers import policy_controller


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _report(stage, **values):
    print(json.dumps({"stage": stage, **values}, allow_nan=False), flush=True)


def rollout(policy, bank, *, record_positions=False):
    """Adapt a pathwise policy to the public, gradient-preserving episode runner."""
    policy.check_config(bank.config)
    return run_episode(policy_controller(policy, evaluation=False), bank,
                       record_positions=record_positions)
