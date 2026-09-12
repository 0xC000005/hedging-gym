"""Proximal Policy Optimization using the unchanged Stable-Baselines3 learner.

Paper: Schulman et al., Proximal Policy Optimization Algorithms (2017).
https://arxiv.org/abs/1707.06347
Implementation: Stable-Baselines3 2.9.0.
https://github.com/DLR-RM/stable-baselines3/tree/v2.9.0/stable_baselines3/ppo

Complete-episode rollouts use the shared terminal-risk objective and a global
threshold fitted between training phases. This is a finance adaptation, not a
reproduction of the original control benchmarks. See docs/baseline-methods.md.
"""
import torch
from stable_baselines3 import PPO


def build_ppo(env, *, seed=7, device="cpu", epochs=10):
    """Construct the established complete-horizon hedging training recipe.

    The upstream model owns learn(), save() and load(); training phases and
    training-only threshold calibration remain in the benchmark recipe.
    """
    steps_per_rollout = env.num_envs * env.config.n_decisions
    return PPO("MlpPolicy", env, seed=seed, device=device, verbose=0,
        n_steps=env.config.n_decisions, batch_size=min(1024, steps_per_rollout),
        n_epochs=epochs, gamma=1., gae_lambda=1., learning_rate=3e-4,
        policy_kwargs=dict(net_arch=dict(pi=[64,64], vf=[64,64]), log_std_init=-1.))


def ppo_controller(model, *, deterministic=True):
    """Tensor-native PPO inference, with the same actions as SB3 ``predict``.

    This wrapper's Box is [-1, 1]. SB3's policy distribution is unchanged; its
    sampled actions are clipped before mapping to holdings, just as ``predict``
    does. Avoid the tensor -> CPU NumPy -> GPU -> CPU -> tensor round trip in
    the shared evaluator. SB3 training still uses its original NumPy VecEnv.
    """
    @torch.no_grad()
    def controller(observed, ledger, time_index, config):
        model.policy.set_training_mode(False)
        features = observed.to(device=model.device, dtype=torch.float32)
        action = model.policy.get_distribution(features).get_actions(deterministic=deterministic)
        action = action.clamp(-1., 1.).to(device=observed.device)
        lower = observed.new_tensor(config.execution.vector("holding_lower", config.n_assets))
        upper = observed.new_tensor(config.execution.vector("holding_upper", config.n_assets))
        return lower + (action + 1.) * (upper-lower) / 2.
    controller.action_selection = "SB3 PPO deterministic" if deterministic else "SB3 PPO sampled"
    return controller
