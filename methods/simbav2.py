"""Finance bridge to the unmodified official SimBaV2 JAX learner.

Source: https://github.com/DAVIAN-Robotics/SimbaV2
Pinned revision: 86899c277cdc697b2b02d827243de1ea93f20a1d (Apache-2.0).
The donor supplies every network, optimizer, update and normalization equation.
This module supplies holdings coordinates, raw-loss replay and complete local
checkpoints. It is a terminal-ES finance transfer, not an ICML reproduction.
"""
from contextlib import contextmanager
from dataclasses import asdict
import math
import os
from pathlib import Path
import subprocess
import sys

import gymnasium as gym
import numpy as np
import torch

from hedging_gym.finance import observation_fields
from hedging_gym.config import RiskConfig, config_from_dict

SOURCE_COMMIT = "86899c277cdc697b2b02d827243de1ea93f20a1d"
NETWORKS = ("_actor", "_critic", "_target_critic", "_temperature")


def load_donor(path):
    """Import a pinned external checkout without installing its simulator stack."""
    path = Path(path).resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "scale_rl", "configs/agent/simbaV2.yaml"],
                                    cwd=path, text=True)
    if head != SOURCE_COMMIT or dirty:
        raise ValueError(f"SimBaV2 requires clean donor source {SOURCE_COMMIT}; found {head}")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(path))
    from scale_rl.agents.simbaV2 import simbaV2_agent
    if not Path(simbaV2_agent.__file__).resolve().is_relative_to(path):
        raise RuntimeError("a different scale_rl checkout was already imported")
    return path


def donor_config(path, *, seed, total_updates, actor_width=128, critic_width=512):
    """Resolve original YAML defaults, with explicit finite-episode changes."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(Path(path) / "configs/agent/simbaV2.yaml"), resolve=False)
    cfg.pop("agent_type")
    cfg.update(seed=seed, gamma=1., n_step=1, critic_use_cdq=True,
               load_only_param=False, learning_rate_decay_step=max(1, total_updates),
               actor_hidden_dim=actor_width, critic_hidden_dim=critic_width,
               critic_min_v=-cfg["normalized_g_max"], critic_max_v=cfg["normalized_g_max"])
    for name, width in (("actor", actor_width), ("critic", critic_width)):
        cfg[f"{name}_scaler_init"] = math.sqrt(2 / width)
        cfg[f"{name}_scaler_scale"] = math.sqrt(2 / width)
        cfg[f"{name}_alpha_init"] = 1 / (cfg[f"{name}_num_blocks"] + 1)
        cfg[f"{name}_alpha_scale"] = 1 / math.sqrt(width)
    return cfg


class RawLossReplay:
    """Uniform one-step replay; only settlement records carry raw terminal loss.

    Never store normalized or RU-transformed rewards. Recompute them for each
    sample using the current episode-global threshold, before donor scaling.
    """
    def __init__(self, capacity, seed):
        if capacity < 1:
            raise ValueError("positive replay capacity required")
        self.capacity, self.size, self.cursor = capacity, 0, 0
        self.arrays = {}
        self.rng = np.random.default_rng(seed)

    def add(self, observation, action, next_observation, terminated, terminal_loss):
        values = dict(observation=np.asarray(observation, np.float32), action=np.asarray(action, np.float32),
                      next_observation=np.asarray(next_observation, np.float32),
                      terminated=np.asarray(terminated, bool), terminal_loss=np.asarray(terminal_loss, np.float64))
        count = min(len(action), self.capacity)
        if not self.arrays:
            self.arrays = {key: np.empty((self.capacity, *value.shape[1:]), value.dtype)
                           for key, value in values.items()}
        slots = (np.arange(count) + self.cursor) % self.capacity
        for key, value in values.items():
            self.arrays[key][slots] = value[-count:]
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    @staticmethod
    def rewards(losses, terminated, zeta, alpha):
        raw = torch.as_tensor(np.asarray(losses, np.float64))
        terminal_rewards = -RiskConfig(alpha).loss(raw, zeta).numpy()
        return np.where(terminated, terminal_rewards, 0.).astype(np.float32)

    def sample(self, count, *, zeta, alpha):
        if not self.size:
            raise ValueError("cannot sample empty replay")
        indices = self.rng.integers(self.size, size=count)
        batch = {key: value[indices].copy() for key, value in self.arrays.items()}
        batch["reward"] = self.rewards(batch.pop("terminal_loss"), batch["terminated"], zeta, alpha)
        batch["truncated"] = np.zeros(count, bool)
        return batch

    def terminal_rewards(self, zeta, alpha):
        if not self.size:
            return np.empty(0, np.float32)
        terminal = self.arrays["terminated"][:self.size]
        losses = self.arrays["terminal_loss"][:self.size][terminal]
        return self.rewards(losses, np.ones(len(losses), bool), zeta, alpha)

    def state_dict(self):
        return dict(capacity=self.capacity, size=self.size, cursor=self.cursor,
                    arrays={k: v[:self.size].copy() for k, v in self.arrays.items()},
                    rng=self.rng.bit_generator.state)

    def load_state_dict(self, state):
        self.capacity, self.size, self.cursor = state["capacity"], state["size"], state["cursor"]
        self.arrays = {}
        for key, value in state["arrays"].items():
            self.arrays[key] = np.empty((self.capacity, *value.shape[1:]), value.dtype)
            self.arrays[key][:self.size] = value
        self.rng.bit_generator.state = state["rng"]


class SimBaV2Hedger:
    def __init__(self, config, donor_path, *, seed=7, total_updates=1000,
                 replay_capacity=1_000_000, actor_width=128, critic_width=512):
        if config.risk.objective != "es":
            raise ValueError("SimBaV2 replay adapter supports terminal ES only")
        if not all(math.isfinite(value) for name in ("holding_lower", "holding_upper")
                   for value in config.execution.vector(name, config.n_assets)):
            raise ValueError("SimBaV2 requires finite holding bounds")
        self.donor_path = load_donor(donor_path)
        from scale_rl.agents.simbaV2.simbaV2_agent import SimbaV2Agent
        from scale_rl.agents.wrappers.normalization import ObservationNormalizer, RewardNormalizer
        self.config, self.seed = config, seed
        self.cfg = donor_config(self.donor_path, seed=seed, total_updates=total_updates,
                                actor_width=actor_width, critic_width=critic_width)
        observations = gym.spaces.Box(-np.inf, np.inf, (len(observation_fields(config)),), np.float32)
        actions = gym.spaces.Box(-1., 1., (config.n_assets,), np.float32)
        self.core = SimbaV2Agent(observations, actions, dict(self.cfg))
        self.observation_normalizer = ObservationNormalizer(self.core)
        self.agent = RewardNormalizer(self.observation_normalizer, gamma=1., g_max=self.cfg["normalized_g_max"])
        self.replay = RawLossReplay(replay_capacity, seed+1)
        self.generator = np.random.default_rng(seed+2)
        self.zeta, self.transitions, self.updates, self.collections = 0., 0, 0, 0
        self.lower = np.asarray(config.execution.vector("holding_lower", config.n_assets), np.float32)
        self.upper = np.asarray(config.execution.vector("holding_upper", config.n_assets), np.float32)
        if any(config.execution.vector(name, config.n_assets)[i] > 0
               for name in ("minimum_trade", "trade_lot") for i in range(config.n_assets)):
            raise ValueError("continuous SimBaV2 does not parameterize minimum trades or lots")

    def normalized_actions(self, observed, *, training=False, deterministic=True, previous=None):
        observed = np.asarray(observed, np.float32)
        if training:
            previous = dict(previous or {})
            previous["next_observation"] = observed.copy()
            return self.agent.sample_actions(self.transitions, previous, training=True)
        # Frozen normalization and independent evaluation RNG: donor's training
        # flag controls action sampling as well as statistics in its wrappers.
        normalized = self.observation_normalizer._normalize(observed)
        return self.core.sample_actions(self.transitions, {"next_observation": normalized},
                                        training=not deterministic)

    def holdings(self, actions):
        return self.lower + (actions + 1.) * (self.upper - self.lower) / 2.

    def set_threshold(self, zeta):
        self.zeta = float(zeta)
        # The donor uses max-return / g_max as a scaling floor. Cover all current
        # relabeled replay terminals too, so a phase change cannot newly clip
        # settlement rewards at the categorical support boundary.
        rewards = self.replay.terminal_rewards(self.zeta, self.config.risk.alpha)
        if len(rewards):
            self.agent.G_r_max = max(self.agent.G_r_max, float(np.abs(rewards).max()))

    def update(self, batch_size):
        batch = self.replay.sample(batch_size, zeta=self.zeta, alpha=self.config.risk.alpha)
        # New terminal records can arrive just before the next action updates
        # donor return statistics. Bound their scaling immediately as well.
        self.agent.G_r_max = max(self.agent.G_r_max, float(np.abs(batch["reward"]).max()))
        info = self.agent.update(self.updates, batch)
        self.updates += 1
        return {key: value for key, value in info.items() if not isinstance(value, dict)}

    @contextmanager
    def evaluation_rng(self, seed):
        import jax
        original = self.core._rng
        self.core._rng = jax.random.PRNGKey(seed)
        try:
            yield
        finally:
            self.core._rng = original

    def controller(self, *, deterministic=True):
        def controller(observed, ledger, time_index, config):
            action = self.normalized_actions(observed.detach().cpu().numpy(), deterministic=deterministic)
            return torch.as_tensor(self.holdings(action), device=observed.device, dtype=observed.dtype)
        controller.action_selection = "official SimBaV2 deterministic" if deterministic else "official SimBaV2 sampled"
        return controller

    def state_dict(self):
        from flax import serialization
        network_state = {name: getattr(self.core, name) for name in NETWORKS}
        network_state["rng"] = self.core._rng
        return dict(source_commit=SOURCE_COMMIT, config=asdict(self.config), cfg=self.cfg,
                    seed=self.seed, networks=serialization.to_bytes(network_state),
                    observation_stats=vars(self.observation_normalizer.obs_rms).copy(),
                    reward_stats=vars(self.agent.G_rms).copy(), G=self.agent.G,
                    G_r_max=self.agent.G_r_max, replay=self.replay.state_dict(),
                    generator=self.generator.bit_generator.state, zeta=self.zeta,
                    transitions=self.transitions, updates=self.updates, collections=self.collections)

    def load_state_dict(self, state):
        from flax import serialization
        if (state["source_commit"] != SOURCE_COMMIT
                or config_from_dict(state["config"]) != self.config or state["cfg"] != self.cfg):
            raise ValueError("resume requires identical source, financial contract and learner configuration")
        template = {name: getattr(self.core, name) for name in NETWORKS}
        template["rng"] = self.core._rng
        restored = serialization.from_bytes(template, state["networks"])
        for name in NETWORKS:
            setattr(self.core, name, restored[name])
        self.core._rng = restored["rng"]
        vars(self.observation_normalizer.obs_rms).update(state["observation_stats"])
        vars(self.agent.G_rms).update(state["reward_stats"])
        self.agent.G, self.agent.G_r_max = state["G"], state["G_r_max"]
        self.replay.load_state_dict(state["replay"])
        self.generator.bit_generator.state = state["generator"]
        for name in ("zeta", "transitions", "updates", "collections"):
            setattr(self, name, state[name])

    def save(self, path, *, runner=None):
        """Trusted-local checkpoint, written atomically at an episode boundary."""
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(dict(learner=self.state_dict(), runner=runner), temporary)
        temporary.replace(path)
