"""Complete batched episodes for any controller, with gradients left intact."""

import torch

from hedging_gym.interfaces import Controller

from .finance import MarketBank
from .gym_env import TensorHedgingEnv


def run_episode(controller: Controller, bank: MarketBank, *, record_positions=False):
    """Return the environment's terminal results using absolute target holdings.

    The controller sees only the current observation, ledger, date and config.
    This runner neither changes network mode nor disables gradients. Training
    owns those choices; evaluation calls it under ``torch.no_grad()``. Optional
    position records are detached copies of the executed ledger-dtype targets.
    """
    env = TensorHedgingEnv(bank)
    observed = env.reset()
    positions = []
    for time_index in range(bank.config.n_decisions):
        target = controller(observed, env.state, time_index, bank.config)
        if not isinstance(target, torch.Tensor) or target.shape != env.state.positions.shape:
            raise ValueError("controller must return a tensor shaped [batch,n_assets]")
        if target.device != observed.device:
            raise ValueError("controller targets and tensor environment must share a device")
        target = target.to(dtype=env.state.positions.dtype)
        if record_positions:
            positions.append(target.detach().clone())
        observed, _, terminated, truncated, result = env.step(target)
    if not terminated or truncated:
        raise RuntimeError("complete financial episodes are required for terminal loss metrics")
    if record_positions:
        result["positions"] = torch.stack(positions, dim=1)
    return result
