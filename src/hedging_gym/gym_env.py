"""Thin Gymnasium access to the shared batched, differentiable hedge ledger.

The tensor class is the training fast path; Gymnasium converts only at its
NumPy boundary. Precomputed exogenous markets are never included
in observations. Planning uses finance.transition/mark_state from the current
state with independent shocks, not the bank's realized future.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from .finance import (
    MarketBank, common_config, generate_market_bank, initial_state, liquidate,
    observation, observation_fields, trade_step, feasible_targets,
)


class TensorHedgingEnv:
    """Synchronous batched episodes on CPU/CUDA; actions are target holdings.

    No autoreset, clipping, penalties or hidden objective changes. Actions must
    already satisfy common bounds and execution rules; infeasible trades raise.
    Actions/candidates use the ledger dtype; this differentiable boundary cast
    is shared by masks and execution, never lot rounding or action projection.
    Tensor actions preserve gradients through all cash/holding transitions.
    Reward is zero until settlement, then cost-inclusive terminal P&L. To train
    ES95, apply the global Rockafellar--Uryasev loss to the episode batch,
    not a new conditional CVaR objective at every date.
    """

    def __init__(self, bank: MarketBank):
        self._bank = bank
        self.config = bank.config
        self.reset()

    def reset(self):
        self.time_index = 0
        self.state = initial_state(self._bank)
        return observation(self._bank, 0, self.state)

    def action_mask(self, candidate_targets):
        """Current-holdings feasibility: [K,A] or [B,K,A] targets -> [B,K].

        The mask is for ordinary decisions, not forced terminal liquidation.
        No candidates are rounded, projected or selected by the environment.
        """
        candidates = torch.as_tensor(candidate_targets, device=self.state.positions.device,
                                     dtype=self.state.positions.dtype)
        if (candidates.ndim not in (2, 3) or candidates.shape[-1] != self.config.n_assets
                or (candidates.ndim == 3 and candidates.shape[0] != len(self.state.positions))):
            raise ValueError("candidate targets must be [K,n_assets] or [batch,K,n_assets]")
        if candidates.ndim == 2:
            candidates = candidates[None].expand(len(self.state.positions), -1, -1)
        return feasible_targets(self.state.positions[:, None], candidates, self.config).all(-1)

    def step(self, target_holdings):
        if self.time_index >= self.config.n_steps:
            raise RuntimeError("episode has ended; reset before stepping")
        if (not isinstance(target_holdings, torch.Tensor)
                or target_holdings.shape != self.state.positions.shape
                or target_holdings.device != self.state.positions.device):
            raise ValueError("actions must be [batch,n_assets] tensors on the environment device")
        target_holdings = target_holdings.to(dtype=self.state.positions.dtype)
        # New-suite execution is checked by trade_step; legacy configs still
        # need boundary validation here. Do not synchronize the GPU twice.
        if not self.config.execution_features and not bool(feasible_targets(
                self.state.positions, target_holdings, self.config).all()):
            raise ValueError("actions must be finite bounded targets satisfying minimum-trade and lot rules")
        # Own the executed holdings: generic RL callers often reuse an action
        # buffer in place. Cloning preserves gradients but prevents that buffer
        # from rewriting yesterday's holdings without a cash movement.
        self.state = trade_step(self.state, target_holdings.clone(),
                                self._bank.marks[:, self.time_index], self.config)
        self.time_index += 1
        terminated = self.time_index == self.config.n_steps
        info = {}
        reward = torch.zeros_like(self.state.cash)
        if terminated:
            info = liquidate(self.state, self._bank.marks[:, -1],
                             self._bank.liability[:, -1], self.config)
            reward = info["terminal_pnl"]
            self.state = trade_step(self.state, torch.zeros_like(self.state.positions),
                                    self._bank.marks[:, -1], self.config, liquidating=True)
            # Terminal observation is after liquidation but before paying the
            # liability, matching liquidate's explicit cash/payoff convention.
        return observation(self._bank, self.time_index, self.state), reward, terminated, False, info


class HestonHedgingVectorEnv(VectorEnv):
    """One batched simulation, not a Python loop over single-path environments.

    ``reset``/``step`` implement Gymnasium's NumPy VectorEnv API. Torch learners
    use ``reset_tensor``/``step_tensor`` to keep observations, actions, rewards
    and accounting gradients on CPU/CUDA without a host round-trip. The two
    interfaces drive the same state, not separate simulations.

    All episodes have the same horizon and reset together. Autoreset is disabled:
    the terminal observation/info are returned before an explicit reset starts
    fresh paths. A seed controls the whole batch's RNG stream (as in native
    batched environments), not a list of independently seeded scalar wrappers.
    Seeding is reproducible for a fixed batch size/device/resolution; comparisons should
    use a shared MarketBank through TensorHedgingEnv/the common evaluator.

    Reset includes selected-model path generation and option pricing. Step only
    exposes the current date and executes the shared ledger. Default terminal
    reward is P&L; optional risk_threshold gives negative global RU loss at .95,
    not per-step CVaR. The threshold is fitted by the learner, never the env.
    simulation_substeps refines internal integration, not the trading calendar.
    """

    metadata = {"render_modes": [], "autoreset_mode": AutoresetMode.DISABLED}

    def __init__(self, num_envs, config=None, *, device="cpu", risk_threshold=None,
                 price_chunk_size=1024, simulation_substeps=1):
        if not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs = num_envs
        self.config = config or common_config()
        self.device = torch.device(device)
        self.risk_threshold = risk_threshold
        self.price_chunk_size = price_chunk_size
        if not isinstance(simulation_substeps, int) or simulation_substeps < 1:
            raise ValueError("simulation_substeps must be a positive integer")
        self.simulation_substeps = simulation_substeps
        self.single_action_space = gym.spaces.Box(
            np.asarray(self.config.holding_lower, dtype=np.float32),
            np.asarray(self.config.holding_upper, dtype=np.float32))
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, (len(observation_fields(self.config)),), np.float32)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.render_mode = None
        self._tensor_env = None

    def reset_tensor(self, *, seed=None, options=None):
        if options:
            # Fixed-horizon paths share a date; a partial reset would silently
            # put some paths on the wrong market date.
            if set(options) != {"reset_mask"} or not np.array_equal(
                    options["reset_mask"], np.ones(self.num_envs, dtype=bool)):
                raise ValueError("fixed-horizon batches support only a full reset")
        super().reset(seed=seed)
        market_seed = int(self.np_random.integers(0, 2**63 - 1))
        bank = generate_market_bank(self.config, self.num_envs, market_seed,
                                    device=self.device, price_chunk_size=self.price_chunk_size,
                                    substeps=self.simulation_substeps)
        self._tensor_env = TensorHedgingEnv(bank)
        return self._tensor_env.reset(), {}

    def step_tensor(self, actions):
        if self._tensor_env is None:
            raise RuntimeError("reset before stepping")
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if actions.shape != self.action_space.shape:
            raise ValueError("actions must be [num_envs,n_assets] target holdings")
        obs, reward, terminated, truncated, info = self._tensor_env.step(actions)
        if terminated and self.risk_threshold is not None:
            reward = -self.risk_threshold - (info["terminal_loss"] - self.risk_threshold).relu() / .05
        return (obs, reward,
                torch.full((self.num_envs,), terminated, dtype=torch.bool, device=self.device),
                torch.full((self.num_envs,), truncated, dtype=torch.bool, device=self.device), info)

    def action_mask_tensor(self, candidate_targets):
        """Torch [B,K] mask for shared [K,A] or per-path [B,K,A] targets."""
        if self._tensor_env is None:
            raise RuntimeError("reset before requesting an action mask")
        return self._tensor_env.action_mask(candidate_targets)

    def action_mask(self, candidate_targets):
        """NumPy counterpart of action_mask_tensor, with the same feasibility."""
        return self.action_mask_tensor(candidate_targets).detach().cpu().numpy()

    def reset(self, *, seed=None, options=None):
        obs, info = self.reset_tensor(seed=seed, options=options)
        return obs.detach().cpu().numpy(), info

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.step_tensor(actions)
        array_info = {}
        for key, value in info.items():
            array_info[key] = value.detach().cpu().numpy()
            array_info[f"_{key}"] = np.ones(self.num_envs, dtype=bool)
        return (*(value.detach().cpu().numpy() for value in
                  (obs, reward, terminated, truncated)), array_info)

    def close_extras(self, **kwargs):
        self._tensor_env = None


class HestonHedgingEnv(gym.Env):
    """Single-path standard Gymnasium API with reproducible fresh market paths.

    Defaults to the basic common benchmark. The NumPy API is convenient for
    generic RL packages, not the accelerated path for direct-gradient methods.
    A fixed risk_threshold optionally returns terminal negative RU loss at
    alpha=.95. Fitting that one global threshold remains the learner's job.
    simulation_substeps refines integration while preserving all trading dates.
    """

    metadata = {"render_modes": []}

    def __init__(self, config=None, *, risk_threshold=None, simulation_substeps=1):
        self.config = config or common_config()
        self.risk_threshold = risk_threshold
        if not isinstance(simulation_substeps, int) or simulation_substeps < 1:
            raise ValueError("simulation_substeps must be a positive integer")
        self.simulation_substeps = simulation_substeps
        self.action_space = gym.spaces.Box(np.asarray(self.config.holding_lower, dtype=np.float32),
                                          np.asarray(self.config.holding_upper, dtype=np.float32))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                               (len(observation_fields(self.config)),), np.float32)
        self._tensor_env = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        market_seed = int(self.np_random.integers(0, 2**63 - 1))
        bank = generate_market_bank(self.config, 1, market_seed, substeps=self.simulation_substeps)
        self._tensor_env = TensorHedgingEnv(bank)
        return self._tensor_env.reset()[0].numpy(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError("action must be [n_assets] target holdings")
        if self._tensor_env is None:
            raise RuntimeError("reset before stepping")
        obs, reward, terminated, truncated, info = self._tensor_env.step(torch.from_numpy(action)[None])
        info = {key: value[0].detach().numpy() for key, value in info.items()}
        reward = float(reward[0])
        if terminated and self.risk_threshold is not None:
            loss = float(info["terminal_loss"])
            reward = -self.risk_threshold - max(loss - self.risk_threshold, 0.) / .05
        return obs[0].detach().numpy(), reward, terminated, truncated, info

    def action_mask(self, candidate_targets):
        """NumPy [K] current-state mask for [K,n_assets] candidate targets."""
        if self._tensor_env is None:
            raise RuntimeError("reset before requesting an action mask")
        candidates = np.asarray(candidate_targets, dtype=np.float32)
        if candidates.ndim != 2:
            raise ValueError("candidate targets must be [K,n_assets]")
        return self._tensor_env.action_mask(candidates)[0].detach().numpy()


# Generic names for new market backends; historical Heston names are retained.
HedgingEnv = HestonHedgingEnv
HedgingVectorEnv = HestonHedgingVectorEnv
