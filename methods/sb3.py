"""Thin SB3 interface to the existing tensor ledger, not another market model.

SB3's VecEnv differs from Gymnasium's VectorEnv: it autoresets and returns a
list of per-episode infos. A reset samples a batch from the supplied training
bank; observations never expose those paths' future. All paths terminate
together. Use complete-horizon PPO rollouts for the global terminal ES reward.
"""
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv

from hedging_gym.finance import bank_subset, observation_fields
from hedging_gym.gym_env import TensorHedgingEnv


class SB3HedgingVecEnv(VecEnv):
    """Batched bank sampling and exact affine action coordinates in [-1, 1].

    The bank must be training data on CPU. Sampling has its own seeded RNG;
    seed denotes the whole batch stream, not independently seeded sub-envs.
    No automatic projection for lots/minimum trades is provided.
    Change the global risk threshold only between complete-episode rollouts.
    """
    render_mode = None

    def __init__(self, bank, num_envs=512, *, risk_threshold=0.):
        if bank.spot.device.type != "cpu" or not 0 < num_envs <= len(bank.spot):
            raise ValueError("provide a CPU training bank with at least num_envs paths")
        self.bank, self.config = bank, bank.config
        self.risk_threshold = float(risk_threshold)
        self.generator = torch.Generator()
        self.tensor_env = None
        self.lower = np.asarray(self.config.execution.vector("holding_lower", self.config.n_assets), np.float32)
        self.upper = np.asarray(self.config.execution.vector("holding_upper", self.config.n_assets), np.float32)
        super().__init__(num_envs,
            gym.spaces.Box(-np.inf, np.inf, (len(observation_fields(self.config)),), np.float32),
            gym.spaces.Box(-1., 1., (self.config.n_assets,), np.float32))

    def reset(self):
        if self._seeds[0] is not None:
            self.generator.manual_seed(self._seeds[0])
        self._reset_seeds()
        self._reset_options()
        selected = torch.randperm(len(self.bank.spot), generator=self.generator)[:self.num_envs]
        self.tensor_env = TensorHedgingEnv(bank_subset(self.bank, selected))
        return self.tensor_env.reset().numpy().astype(np.float32)

    def step_async(self, actions):
        # Own the caller's buffer until step_wait consumes it.
        self.actions = np.array(actions, dtype=np.float32, copy=True)

    def step_wait(self):
        target = self.lower + (self.actions + 1.) * (self.upper - self.lower) / 2.
        with torch.no_grad():
            obs, reward, done, _, result = self.tensor_env.step(torch.from_numpy(target))
        observed = obs.numpy().astype(np.float32)
        infos = [{} for _ in range(self.num_envs)]
        if done:
            reward = -self.config.risk.loss(result["terminal_loss"], self.risk_threshold)
            for i, info in enumerate(infos):
                info.update(terminal_observation=observed[i].copy(),
                            terminal_loss=float(result["terminal_loss"][i]),
                            **{"TimeLimit.truncated": False})
            observed = self.reset()
        return observed, reward.numpy(), np.full(self.num_envs, done), infos

    def close(self):
        self.tensor_env = None

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        if list(self._get_indices(indices)) != list(range(self.num_envs)):
            raise ValueError("a synchronous batch shares attributes across all episodes")
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        if list(self._get_indices(indices)) != list(range(self.num_envs)):
            raise ValueError("a synchronous batch supports only full-batch method calls")
        result = getattr(self, method_name)(*args, **kwargs)
        return [result for _ in range(self.num_envs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


def sb3_controller(model, *, deterministic=True):
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
