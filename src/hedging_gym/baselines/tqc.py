"""TQC using the unchanged SB3-Contrib 2.9.0 learner.

Paper: Kuznetsov et al., Controlling Overestimation Bias with Truncated Mixture
of Continuous Distributional Quantile Critics (ICML 2020).
https://proceedings.mlr.press/v119/kuznetsov20a.html
Implementation:
https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/tree/v2.9.0/sb3_contrib/tqc

The common terminal-risk objective is supplied through a shared financial
replay adapter. Algorithms, networks, targets, entropy tuning and optimizers
remain upstream. This is a finance adaptation, not the paper's native benchmark.
See docs/baseline-methods.md for the integration and qualification boundaries.
"""
from sb3_contrib import TQC

from hedging_gym.baselines._shared.sb3_off_policy import (
    TerminalRiskReplayBuffer,
    load_off_policy,
)
from hedging_gym.baselines._shared.sb3_off_policy import (
    off_policy_controller as make_controller,
)
from hedging_gym.baselines._shared.sb3_off_policy import save_off_policy as save

__all__ = ["build_tqc", "load", "save", "make_controller"]


def build_tqc(env, *, seed=7, device="cpu", buffer_size=1_000_000,
                     learning_starts=100, batch_size=256, policy_kwargs=None):
    """Keep source defaults except gamma=1 and vector-aware update accounting.

    SB3 counts train_freq in vector steps but gradient_steps in optimizer steps.
    -1 performs num_envs updates per vector step: source ratio one update per
    transition. Passing the upstream literal default 1 with 512 envs would
    reduce that ratio by 512.
    """
    if env.config.risk.objective != "es":
        raise ValueError("CrossQ/TQC replay adapter supports terminal ES only")
    return TQC("MlpPolicy", env, seed=seed, device=device,
        gamma=1., train_freq=1, gradient_steps=-1, buffer_size=buffer_size,
        learning_starts=learning_starts, batch_size=batch_size,
        replay_buffer_class=TerminalRiskReplayBuffer,
        replay_buffer_kwargs=dict(risk_alpha=env.config.risk.alpha,
                                  risk_threshold=env.risk_threshold),
        policy_kwargs=policy_kwargs, verbose=0)


def load(directory, env, *, device="cpu"):
    """Load this method with its financial replay and episode continuation state."""
    return load_off_policy(TQC, directory, env, device=device)
