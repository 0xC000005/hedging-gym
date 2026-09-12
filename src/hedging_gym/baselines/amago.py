"""Official AMAGO learners on common-ledger hedging sequences.

Papers: AMAGO, https://arxiv.org/abs/2310.09971;
AMAGO-2, https://arxiv.org/abs/2411.11188.
Upstream: https://github.com/UT-Austin-RPL/amago/tree/54c25ab6da9371614c47352f569a56c0fe938d3b
Implementation notes: docs/amago-context.md.

The upstream learner owns training, replay and attention. With gamma=1 and a
fixed global Rockafellar--Uryasev threshold, terminal negative RU rewards
optimize expected RU loss at that threshold, not joint ES minimization.

Concatenated independent books form a training sequence, not a new market
model. Declared market parameters remain observed; no future path is exposed.
"""

from dataclasses import replace

import gymnasium as gym
import numpy as np
import torch
from amago.envs.exploration import EpsilonGreedy

from hedging_gym.environment.finance import bank_subset, bank_to, observation_fields
from hedging_gym.environment.gym_env import TensorHedgingEnv


def _check_sequence_contract(config):
    # The retained runner and context schedule count pre-maturity decisions.
    if config.time_grid.trade_at_maturity:
        raise ValueError("AMAGO sequences do not support trading at maturity")


class ExplicitEpsilonGreedy(EpsilonGreedy):
    """Expose two properties formerly forwarded implicitly by Gymnasium 0.29.

    No change to official action sampling, exploration noise or replay logic.
    This lets the official synchronous pipeline use common Gymnasium 1.2.
    """

    @property
    def env_name(self):
        return self.env.env_name

    @property
    def step_count(self):
        return self.env.step_count


class MemoryHedgingTask(gym.Env):
    """Several independent books in one observed market per memory sequence.

    Each underlying episode uses TensorHedgingEnv unchanged, including cash
    reset between books, transaction costs and terminal liquidation. Market
    banks are training-only and exogenous; observations expose only current data.
    """

    def __init__(self, banks, *, threshold, episodes=3, seed=7, num_envs=1):
        self.banks = tuple(banks)
        if not self.banks or episodes < 1 or num_envs < 1:
            raise ValueError("provide training banks and at least one episode")
        self.num_envs = num_envs
        self.config = self.banks[0].config
        _check_sequence_contract(self.config)
        for bank in self.banks:
            if replace(bank.config, market=self.config.market) != self.config:
                raise ValueError("training tasks may differ only in observed market parameters")
        self.threshold, self.episodes = float(threshold), episodes
        self.completed_books = 0
        self.terminal_losses = []
        self._rng = np.random.default_rng(seed)
        self.action_space = gym.spaces.Box(
            np.asarray(self.config.execution.vector("holding_lower", self.config.n_assets), np.float32),
            np.asarray(self.config.execution.vector("holding_upper", self.config.n_assets), np.float32))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
            (len(observation_fields(self.config)),), np.float32)

    def _start_book(self):
        bank = self.banks[self.task]
        paths = torch.from_numpy(self._rng.integers(len(bank.spot), size=self.num_envs))
        self.env = TensorHedgingEnv(bank_subset(bank, paths))
        return self._observation(self.env.reset())

    def _observation(self, values):
        array = values.numpy()
        return array[0] if self.num_envs == 1 else array

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.task = int(self._rng.integers(len(self.banks)))
        self.completed = 0
        return self._start_book(), {}

    def step(self, action):
        target = torch.as_tensor(action, dtype=self.env.state.positions.dtype).reshape(
            self.num_envs, self.config.n_assets)
        observed, _, terminated, truncated, raw = self.env.step(target)
        reward, info = np.zeros(self.num_envs, np.float32), {}
        if terminated:
            self.completed_books += self.num_envs
            self.terminal_losses.append(raw["terminal_loss"].numpy().copy())
            reward = -self.config.risk.loss(raw["terminal_loss"], self.threshold).numpy()
            info = {"AMAGO_LOG_METRIC terminal_loss": raw["terminal_loss"].numpy(),
                    "AMAGO_LOG_METRIC transaction_cost": raw["transaction_cost"].numpy()}
            self.completed += 1
            if self.completed < self.episodes:
                observed = self._start_book()
                terminated = False
            elif self.num_envs > 1:
                # Native AlreadyVectorizedEnv requires same-step autoreset.
                # All books have equal horizons, so the whole tensor batch resets
                # together only after all original terminal settlements finish.
                observed, _ = self.reset()
            else:
                observed = self._observation(observed)
        else:
            observed = self._observation(observed)
        if self.num_envs == 1:
            return observed, float(reward[0]), terminated, truncated, info
        return (observed, reward, np.full(self.num_envs, terminated),
                np.full(self.num_envs, truncated), info)


class AmagoController:
    """Official native policy inference on a common-evaluator book batch.

    Query batches start with independent memory, optionally primed using complete
    *training-context* books. Query outcomes never become context for other
    queries. Each context bank has one independently drawn path per query.
    """

    action_selection = "Official AMAGO deterministic actor; memory reset per book batch"

    def __init__(self, agent, *, context_banks=(), risk_threshold=None, initial_time=0):
        self.agent = agent
        self.hidden = None
        self.previous_action = None
        self.context_banks = tuple(context_banks)
        for bank in self.context_banks:
            _check_sequence_contract(bank.config)
        self.threshold = risk_threshold
        self.initial_time = initial_time
        if self.context_banks and risk_threshold is None:
            raise ValueError("context rewards require the unchanged training RU threshold")
        self.offset, self.context_decisions = 0, 0
        self.action_selection = ("Official AMAGO actor with prior completed training books"
            if self.context_banks else type(self).action_selection)

    def _action(self, observed, config):
        rl2 = torch.cat((self.previous_reward[:, None], self.previous_action), -1)
        actions, self.hidden = self.agent.get_actions(
            obs={"observation": observed[:, None]}, rl2s=rl2[:, None],
            time_idxs=torch.full((len(observed), 1, 1), self.history_step,
                device=observed.device, dtype=torch.long), hidden_state=self.hidden, sample=False)
        self.previous_action = actions[:, 0]
        self.previous_reward = observed.new_zeros(len(observed))
        self.history_step += 1
        low = observed.new_tensor(config.execution.vector("holding_lower", config.n_assets))
        high = observed.new_tensor(config.execution.vector("holding_upper", config.n_assets))
        return low + (self.previous_action + 1.) * .5 * (high-low)

    def __call__(self, observed, ledger, time_index, config):
        _check_sequence_contract(config)
        if time_index == 0:
            count = len(observed)
            self.hidden = self.agent.traj_encoder.init_hidden_state(count, observed.device)
            self.previous_action = torch.zeros_like(ledger.positions)
            self.previous_reward = observed.new_zeros(count)
            self.history_step = self.initial_time
            for bank in self.context_banks:
                if self.offset + count > len(bank.spot):
                    raise ValueError("supply a distinct completed context path per held-out query")
                sample = bank_to(bank_subset(bank, slice(self.offset, self.offset+count)), observed.device)
                context = TensorHedgingEnv(sample)
                current = context.reset()
                for _ in range(sample.config.n_steps):
                    target = self._action(current, sample.config)
                    current, _, _, _, raw = context.step(target)
                self.previous_reward = -sample.config.risk.loss(raw["terminal_loss"], self.threshold)
                self.context_decisions += count * sample.config.n_steps
            self.offset += count
        return self._action(observed, config)
