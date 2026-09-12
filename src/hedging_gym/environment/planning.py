"""Conditional market samples and batched financial branches for planners.

No action selection, value network, risk aggregation or search algorithm lives
here. Scenarios start from current observations, never an evaluation path's
future. Candidate actions share the same sampled market continuations.
"""

from dataclasses import dataclass, fields

import torch

from hedging_gym.interfaces import Controller

from .finance import (
    LedgerState,
    decode_market_observation,
    liquidate,
    mark_state,
    observation_from_state,
    trade_step,
    transition,
)


@dataclass
class ConditionalPaths:
    """Market continuations ordered by root then scenario, including expiry."""

    states: list[tuple[torch.Tensor, torch.Tensor]]
    marks: list[torch.Tensor]
    terminal_liability: torch.Tensor
    scenarios: int


@torch.no_grad()
def sample_continuation(observed, time_index, config, scenarios, generator):
    """Sample fresh conditional paths using a caller-owned random generator.

    Contract marks use the environment's tensor pricing backend. Instruments
    must be markable from the supplied spot/variance state; missing historical
    state for a path-dependent instrument is not inferred from future data.
    """
    spot, variance = decode_market_observation(observed, config)
    spot = spot[:, None].expand(-1, scenarios).reshape(-1).double()
    variance = variance[:, None].expand(-1, scenarios).reshape(-1).double()
    states, marks = [], []
    for date in range(time_index, config.n_steps + 1):
        if date > time_index:
            spot, variance = transition(spot, variance, config=config.market,
                                       dt=config.dt, generator=generator)
        mid, liability = mark_state(spot, variance, date, config)
        states.append((spot.to(observed.dtype), variance.to(observed.dtype)))
        marks.append(mid.to(observed.dtype))
    return ConditionalPaths(states, marks, liability.to(observed.dtype), scenarios)


def rollout_branches(targets, ledger, time_index, config, paths: ConditionalPaths,
                     continuation: Controller):
    """Return terminal losses [root, candidate, scenario] with action gradients.

    Execute each root candidate, then ask the continuation controller for later
    trades. Ledgers are branched without changing the caller's holdings. The
    caller owns model mode, gradients, risk aggregation and action selection.
    """
    batch, candidates, assets = targets.shape
    scenarios = paths.scenarios

    def repeat_ledger(value):
        return value[:, None, None].expand(-1, candidates, scenarios, *value.shape[1:]).reshape(
            -1, *value.shape[1:])

    state = LedgerState(*(repeat_ledger(getattr(ledger, field.name)) for field in fields(LedgerState)))

    def branch(value):
        shaped = value.reshape(batch, scenarios, *value.shape[1:])
        return shaped[:, None].expand(-1, candidates, -1, *value.shape[1:]).reshape(
            batch * candidates * scenarios, *value.shape[1:])

    target = targets[:, :, None].expand(-1, -1, scenarios, -1).reshape(-1, assets)
    for offset, date in enumerate(range(time_index, config.n_decisions)):
        mid = branch(paths.marks[offset])
        if offset:
            spot, variance = (branch(value) for value in paths.states[offset])
            observed = observation_from_state(spot, variance, date, state, mid, config)
            target = continuation(observed, state, date, config)
        if (not isinstance(target, torch.Tensor) or target.shape != state.positions.shape
                or target.device != state.positions.device):
            raise ValueError("actions must be [batch,n_assets] tensors on the environment device")
        # Match TensorHedgingEnv: own each action buffer before the controller
        # reuses it, and execute in ledger precision without breaking autograd.
        target = target.to(dtype=state.positions.dtype).clone()
        state = trade_step(state, target, mid, config)
    losses = liquidate(state, branch(paths.marks[-1]),
                       branch(paths.terminal_liability), config)["terminal_loss"]
    return losses.reshape(batch, candidates, scenarios)
