"""SB3-Contrib 2.9.0 CrossQ/TQC on the shared terminal-risk hedging task.

The algorithms, networks, targets, entropy tuning and optimizers are upstream:
https://sb3-contrib.readthedocs.io/en/v2.9.0/modules/crossq.html
https://sb3-contrib.readthedocs.io/en/v2.9.0/modules/tqc.html
Only replay reward relabeling and the financial/checkpoint boundary live here.
These are finance adaptations, not reproductions of the papers' benchmarks.
"""
from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import torch
from sb3_contrib import CrossQ, TQC
from stable_baselines3.common.buffers import ReplayBuffer

from hedging_gym.config import RiskConfig
from hedging_gym.gym_env import TensorHedgingEnv
from methods.sb3 import sb3_controller


ALGORITHMS = {"crossq": CrossQ, "tqc": TQC}


class TerminalRiskReplayBuffer(ReplayBuffer):
    """Store raw terminal loss in SB3's reward slot; relabel when sampled.

    The global RU threshold changes between complete episodes. Keeping the
    collection-time RU reward would mix incompatible objectives in replay.
    Nonterminal entries contain zero, terminal entries contain raw loss, and
    the sampled reward is -RU(loss, CURRENT zeta) only at actual settlement.
    One-step replay without reward normalization preserves this convention.
    """

    def __init__(self, *args, risk_alpha=.95, risk_threshold=0., **kwargs):
        super().__init__(*args, **kwargs)
        if self.optimize_memory_usage:
            raise ValueError("terminal-risk replay requires explicit next observations")
        self.risk = RiskConfig(alpha=risk_alpha)
        self.risk_threshold = float(risk_threshold)

    @property
    def terminal_losses(self):
        return self.rewards

    def add(self, obs, next_obs, action, reward, done, infos):
        losses = np.zeros(self.n_envs, np.float32)
        for i, terminal in enumerate(done):
            if infos[i].get("TimeLimit.truncated", False):
                raise ValueError("financial settlement must not be a time-limit truncation")
            if terminal:
                losses[i] = infos[i]["terminal_loss"]
        super().add(obs, next_obs, action, losses, done, infos)

    def _get_samples(self, batch_inds, env=None):
        if env is not None:
            raise ValueError("terminal-risk replay does not use reward normalization")
        samples = super()._get_samples(batch_inds, env=None)
        reward = torch.where(samples.dones.bool(),
            -self.risk.loss(samples.rewards, self.risk_threshold),
            torch.zeros_like(samples.rewards))
        return samples._replace(rewards=reward)


def build_off_policy(algorithm, env, *, seed=7, device="cpu", buffer_size=1_000_000,
                     learning_starts=100, batch_size=256, policy_kwargs=None):
    """Keep source defaults except gamma=1 and vector-aware update accounting.

    SB3 counts train_freq in vector steps but gradient_steps in optimizer steps.
    -1 performs num_envs updates per vector step: source ratio one update per
    transition. Passing the upstream literal default 1 with 512 envs would
    reduce that ratio by 512. CrossQ retains its source policy_delay=3.
    """
    return ALGORITHMS[algorithm]("MlpPolicy", env, seed=seed, device=device,
        gamma=1., train_freq=1, gradient_steps=-1, buffer_size=buffer_size,
        learning_starts=learning_starts, batch_size=batch_size,
        replay_buffer_class=TerminalRiskReplayBuffer,
        replay_buffer_kwargs=dict(risk_alpha=env.config.risk.alpha,
                                  risk_threshold=env.risk_threshold),
        policy_kwargs=policy_kwargs, verbose=0)


def off_policy_controller(model, *, deterministic=True):
    """Use upstream predict and the existing exact affine holding transform."""
    controller = sb3_controller(model, deterministic=deterministic)
    controller.action_selection = f"SB3-Contrib {type(model).__name__} " + (
        "deterministic" if deterministic else "sampled")
    return controller


def set_risk_threshold(model, env, zeta):
    if env.tensor_env is not None and env.tensor_env.time_index != 0:
        raise ValueError("change the global threshold only at an episode boundary")
    env.risk_threshold = model.replay_buffer.risk_threshold = float(zeta)


def save_off_policy(model, env, directory, *, runner_state=None):
    """Save after learn returns at a complete episode batch boundary.

    SB3 saves policy/critic/target parameters, optimizer and entropy state,
    update counters and last observations. Its separate replay API saves raw
    losses and replay position. The sidecar saves RNG streams and the already
    autoreset next batch, so loading does not silently draw a different bank.
    """
    if (env.tensor_env is None or env.tensor_env.time_index != 0
            or model.num_timesteps % (env.num_envs * env.config.n_steps)):
        raise ValueError("checkpoint requires a completed episode batch")
    if model.replay_buffer.risk_threshold != env.risk_threshold:
        raise ValueError("environment and replay thresholds differ")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    runtime = dict(algorithm=type(model).__name__.lower(), config=env.config,
        num_envs=env.num_envs, risk_threshold=env.risk_threshold,
        batch=env.tensor_env._bank, bank_rng=env.generator.get_state(),
        pending_seeds=env._seeds.copy(),
        action_rng=deepcopy(model.action_space.np_random.bit_generator.state),
        python_rng=random.getstate(), numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        runner_state=runner_state or {})
    model.save(directory / "model")
    model.save_replay_buffer(directory / "replay.pkl")
    # Written last: an interrupted save has no completed runtime sidecar.
    torch.save(runtime, directory / "runtime.pt")


def load_off_policy(directory, env, *, device="cpu"):
    directory = Path(directory)
    runtime = torch.load(directory / "runtime.pt", map_location="cpu", weights_only=False)
    if runtime["config"] != env.config or runtime["num_envs"] != env.num_envs:
        raise ValueError("resume requires the saved configuration and vector batch size")
    model = ALGORITHMS[runtime["algorithm"]].load(directory / "model", env=env,
                                                device=device, force_reset=False)
    model.load_replay_buffer(directory / "replay.pkl")
    env.tensor_env = TensorHedgingEnv(runtime["batch"])
    env.generator.set_state(runtime["bank_rng"])
    env._seeds = runtime["pending_seeds"]
    model.action_space.np_random.bit_generator.state = runtime["action_rng"]
    set_risk_threshold(model, env, runtime["risk_threshold"])
    random.setstate(runtime["python_rng"])
    np.random.set_state(runtime["numpy_rng"])
    torch.set_rng_state(runtime["torch_rng"])
    if runtime["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(runtime["cuda_rng"])
    return model, runtime["runner_state"]
