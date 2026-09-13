"""Cao/Chen/Hull/Poulos two-moment DDPG hedging.

Paper: Deep Hedging of Derivatives Using Reinforcement Learning (2021),
https://ssrn.com/abstract=3514586.
Upstream: https://github.com/rotmanfinhub/deep-hedging-research/tree/b4d031a185fe2547dd81ad7a67081f6dbe52c5bc
Implementation notes: docs/baseline-methods.md.

This PyTorch reimplementation follows the paper and pinned author reference
implementation. It controls common-task holdings using the shared observations.
Dense net-portfolio P&L telescopes to terminal P&L. The original mean-minus-
1.5-standard-deviation objective is distinct from terminal ES and QR-D4PG.

Source corrections use pointwise nonnegative variance rather than K.max(var,0),
copy target weights after initialization, and sample every occupied replay slot.
Actor batch normalization keeps source inference semantics; critic normalization
and its shared input layer keep the Keras update convention. Each 128-row replay
draw gives four shuffled 32-row fits per moment critic and one actor update.
"""
import math
import time
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from hedging_gym.baselines._shared.checkpoints import (
    load_checkpoint,
    restore_rng,
    rng_state,
    save_checkpoint,
)
from hedging_gym.baselines._shared.controllers import (
    policy_controller as make_controller,
)
from hedging_gym.baselines._shared.policy import (
    BUY,
    HOLD,
    SELL,
    PolicyAction,
    _ConfiguredPolicy,
)
from hedging_gym.environment.finance import bank_subset, bank_to
from hedging_gym.environment.gym_env import TensorHedgingEnv

SOURCE = 'https://github.com/rotmanfinhub/deep-hedging-research/tree/b4d031a185fe2547dd81ad7a67081f6dbe52c5bc'


__all__ = ["HullDDPGPolicy", "HullLearner", "train_hull_ddpg", "make_controller"]


class KerasBatchNorm(nn.Module):
    """Keras2.3.1 BN: eps=.001, old-stat weight=.99 and its exact correction."""
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))
        self.register_buffer('running_mean', torch.zeros(width))
        self.register_buffer('running_var', torch.ones(width))

    def forward(self, values):
        if self.training:
            variance, mean = torch.var_mean(values, dim=0, correction=0)
            count = values.shape[0]
            with torch.no_grad():
                self.running_mean.mul_(.99).add_(mean, alpha=.01)
                self.running_var.mul_(.99).add_(variance * (count/(count-1.001)), alpha=.01)
        else:
            mean, variance = self.running_mean, self.running_var
        return (values-mean) * torch.rsqrt(variance+.001) * self.weight + self.bias


def _head(inputs, outputs):
    result = nn.Sequential(nn.Linear(inputs, 32), nn.ReLU(), KerasBatchNorm(32),
        nn.Linear(32, 64), nn.ReLU(), KerasBatchNorm(64), nn.Linear(64, outputs))
    for layer in result:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return result


class MomentActor(nn.Module):
    def __init__(self, inputs, assets):
        super().__init__()
        self.input_bn = KerasBatchNorm(inputs)
        self.head = _head(inputs, assets)

    def forward(self, observed):
        return self.head(self.input_bn(observed)).sigmoid()


class HullDDPGPolicy(_ConfiguredPolicy):
    def __init__(self, config):
        super().__init__(config)
        self.actor = MomentActor(self.feature_dim, self.n_assets)
        self.register_buffer('lower', torch.tensor(config.execution.vector('holding_lower', self.n_assets)))
        self.register_buffer('upper', torch.tensor(config.execution.vector('holding_upper', self.n_assets)))

    def targets(self, features, lower=None, upper=None):
        lo = self.lower if lower is None else torch.as_tensor(lower, device=features.device, dtype=features.dtype)
        hi = self.upper if upper is None else torch.as_tensor(upper, device=features.device, dtype=features.dtype)
        return lo+(hi-lo)*self.actor(features)

    def forward(self, features, holdings, lower, upper, *, deterministic=True, generator=None):
        del deterministic, generator
        targets = self.targets(features, lower, upper)
        modes = torch.where(targets > holdings, BUY, torch.where(targets < holdings, SELL, HOLD))
        return PolicyAction(targets, modes, features.new_zeros(len(features)))


class MomentCritic(nn.Module):
    """Conditional first and raw second moments of cumulative P&L rewards."""
    def __init__(self, features, assets):
        super().__init__()
        self.input_bn = KerasBatchNorm(features+assets)
        # head1 estimates E[R]; head2 estimates E[R**2], not variance.
        self.head1, self.head2 = _head(features+assets, 1), _head(features+assets, 1)

    def forward(self, observations, actions):
        normalized = self.input_bn(torch.cat((observations, actions), -1))
        return self.head1(normalized).squeeze(-1), self.head2(normalized).squeeze(-1)

    def moment(self, observations, actions, second=False):
        values = self.input_bn(torch.cat((observations, actions), -1))
        return (self.head2 if second else self.head1)(values).squeeze(-1)

    def moment_parameters(self, second=False):
        return tuple(self.input_bn.parameters())+tuple((self.head2 if second else self.head1).parameters())


def moment_targets(rewards, terminal, next_mean, next_second):
    alive = (~terminal).to(rewards.dtype)
    return (rewards+alive*next_mean,
            rewards.square()+alive*(2*rewards*next_mean+next_second))


def risk_score(mean, second, c=1.5, variance_floor=1e-8):
    """Pointwise paper score; tiny floor only resolves sqrt's derivative at zero."""
    return mean-c*(second-mean.square()).clamp_min(variance_floor).sqrt()


class SourceAdam:
    """Original TF/Keras Adam epsilon placement, with eagerly allocated state.

    Tensor step counters also advance under CUDA graph replay. No optimizer
    reset occurs between training blocks or checkpoint resumes.
    """
    def __init__(self, parameters, learning_rate, epsilon):
        self.parameters = tuple(parameters)
        self.lr, self.epsilon = learning_rate, epsilon
        self.count = self.parameters[0].new_zeros(())
        self.first = [torch.zeros_like(value) for value in self.parameters]
        self.second = [torch.zeros_like(value) for value in self.parameters]

    @torch.no_grad()
    def step(self, gradients):
        self.count.add_(1)
        rate = self.lr*(1-.999**self.count).sqrt()/(1-.9**self.count)
        for parameter, gradient, first, second in zip(self.parameters, gradients, self.first, self.second):
            first.mul_(.9).add_(gradient, alpha=.1)
            second.mul_(.999).addcmul_(gradient, gradient, value=.001)
            parameter.add_(-rate*first/(second.sqrt()+self.epsilon))

    def state_dict(self):
        return dict(count=self.count, first=self.first, second=self.second)

    @torch.no_grad()
    def load_state_dict(self, state):
        self.count.copy_(state['count'])
        for current, previous in zip(self.first+self.second, state['first']+state['second']):
            current.copy_(previous)


class PrioritizedReplay:
    """Tensor storage plus vectorized CPU segment trees; source stratified PER."""
    def __init__(self, capacity=600000, alpha=.6):
        self.capacity, self.alpha, self.size, self.cursor = capacity, alpha, 0, 0
        self.leaves = 1 << (capacity-1).bit_length()
        self.sums = np.zeros(2*self.leaves, dtype=np.float64)
        self.minimum = np.full(2*self.leaves, np.inf)
        self.maximum_priority = 1.
        self.arrays = None

    def priorities(self, indices, values):
        indices, values = np.asarray(indices), np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise FloatingPointError('Hull moment critic produced a non-finite/nonpositive priority')
        # Native sequential updates: the last occurrence wins for duplicate slots.
        _, reverse = np.unique(indices[::-1], return_index=True)
        keep = len(indices)-1-reverse
        slots = indices[keep]+self.leaves
        weights = values[keep]**self.alpha
        self.sums[slots], self.minimum[slots] = weights, weights
        self.maximum_priority = max(self.maximum_priority, float(values.max()))
        while len(slots) and slots[0] > 1:
            slots = np.unique(slots//2)
            self.sums[slots] = self.sums[2*slots]+self.sums[2*slots+1]
            self.minimum[slots] = np.minimum(self.minimum[2*slots], self.minimum[2*slots+1])

    def add(self, values):
        if self.arrays is None:
            self.arrays = [value.new_zeros((self.capacity, *value.shape[1:])) for value in values]
        count = min(len(values[0]), self.capacity)
        indices = (np.arange(count)+self.cursor) % self.capacity
        tensor_indices = torch.as_tensor(indices, device=values[0].device)
        for stored, value in zip(self.arrays, values):
            stored[tensor_indices] = value[-count:]
        self.priorities(indices, np.full(count, self.maximum_priority))
        self.cursor = (self.cursor+count) % self.capacity
        self.size = min(self.size+count, self.capacity)

    def sample(self, count, beta, generator):
        masses = (np.arange(count)+generator.random(count))*(self.sums[1]/count)
        nodes = np.ones(count, dtype=np.int64)
        while nodes[0] < self.leaves:
            left = 2*nodes
            right = masses >= self.sums[left]
            masses = masses-np.where(right, self.sums[left], 0.)
            nodes = left+right
        indices = nodes-self.leaves
        weights = (self.minimum[1]/self.sums[nodes])**beta
        tensor_indices = torch.as_tensor(indices, device=self.arrays[0].device)
        values = tuple(stored[tensor_indices] for stored in self.arrays)
        return (*values, values[0].new_tensor(weights)), indices


class HullLearner:
    def __init__(self, config, *, device='cpu', dtype=torch.float32, learning_rate=1e-4,
                 target_rate=1e-5, variance_floor=1e-8):
        self.actor = HullDDPGPolicy(config).to(device=device, dtype=dtype).eval()
        self.critic = MomentCritic(self.actor.feature_dim, config.n_assets).to(device=device, dtype=dtype)
        self.target_actor, self.target_critic = deepcopy(self.actor), deepcopy(self.critic).eval()
        self.actor_optimizer = SourceAdam(self.actor.parameters(), learning_rate, 1e-8)
        self.first_optimizer = SourceAdam(self.critic.moment_parameters(), learning_rate, 1e-7)
        self.second_optimizer = SourceAdam(self.critic.moment_parameters(True), learning_rate, 1e-7)
        self.target_rate, self.variance_floor = target_rate, variance_floor
        self.graph = None

    def update(self, states, actions, rewards, following, terminal, weights, order1, order2):
        with torch.no_grad():
            next_actions = self.target_actor.targets(following)
            targets = moment_targets(rewards, terminal, *self.target_critic(following, next_actions))
            self.critic.eval()
            priority = (self.critic.moment(states, actions, True)-targets[1]).abs()+1e-6
        losses = []
        self.critic.train()
        for second, order, optimizer, target in zip((False, True), (order1, order2),
                (self.first_optimizer, self.second_optimizer), targets):
            for indices in order.reshape(-1, 32):
                prediction = self.critic.moment(states[indices], actions[indices], second)
                loss = ((prediction-target[indices]).square()*weights[indices]).mean()
                gradients = torch.autograd.grad(loss, optimizer.parameters)
                optimizer.step(gradients)
                losses.append(loss.detach())
        self.critic.eval()
        # Native actor optimization and critic action derivatives use inference BN.
        score = risk_score(*self.critic(states, self.actor.targets(states)),
                           variance_floor=self.variance_floor)
        actor_loss = -score.sum()  # tf.gradients sums, not averages, the batch.
        self.actor_optimizer.step(torch.autograd.grad(actor_loss, self.actor_optimizer.parameters))
        with torch.no_grad():
            for online, target in ((self.actor, self.target_actor), (self.critic, self.target_critic)):
                for value, destination in zip(online.parameters(), target.parameters()):
                    destination.lerp_(value, self.target_rate)
                for value, destination in zip(online.buffers(), target.buffers()):
                    destination.lerp_(value, self.target_rate)
        return priority, torch.stack(losses).mean(), actor_loss.detach()/len(states)

    def state_dict(self):
        return {name: getattr(self, name).state_dict() for name in
            ('actor', 'critic', 'target_actor', 'target_critic', 'actor_optimizer',
             'first_optimizer', 'second_optimizer')}

    def load_state_dict(self, state):
        for name, values in state.items():
            getattr(self, name).load_state_dict(values)

    def enable_cuda_graph(self, sample):
        """Capture one unchanged fixed-size learner update; restore probe updates."""
        if self.actor.lower.device.type != 'cuda':
            raise ValueError('CUDA graph requires CUDA')
        snapshot = deepcopy(self.state_dict())
        self.graph_inputs = tuple(value.clone() for value in sample)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.update(*self.graph_inputs)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.graph_outputs = self.update(*self.graph_inputs)
        self.load_state_dict(snapshot)

    def step(self, sample):
        if self.graph is None:
            return self.update(*sample)
        for destination, value in zip(self.graph_inputs, sample):
            destination.copy_(value)
        self.graph.replay()
        return self.graph_outputs


def train_hull_ddpg(bank, *, episodes=50001, seed=7, device='cpu', collection_batch=16,
                    replay_capacity=600000, learning_rate=1e-4, target_rate=1e-5,
                    variance_floor=1e-8, reward_scale=None, cuda_graph=False,
                    checkpoint_path=None, checkpoint_every=1024, resume_from=None,
                    progress=True):
    """One replay/actor update per inserted transition after 128-row warmup.

    Paths are advanced in parallel, then one update per lane is performed.
    Collection has at most collection_batch transitions of policy staleness;
    it does not collect whole episode banks under one frozen actor.
    """
    # Author units: 100 shares of a $100 initial stock. Only learner money
    # changes; execution, common observations, and evaluated cash do not.
    if reward_scale is None:
        reward_scale = bank.config.market.spot0/10000
    if episodes < 1 or collection_batch < 1 or replay_capacity < 129 or reward_scale <= 0:
        raise ValueError('positive episodes/batch/scale and replay capacity >128 required')
    options = dict(episodes=episodes, seed=seed, collection_batch=collection_batch,
        replay_capacity=replay_capacity, learning_rate=learning_rate, target_rate=target_rate,
        variance_floor=variance_floor, reward_scale=reward_scale, cuda_graph=cuda_graph)
    config = bank.config
    saved = None if resume_from is None else load_checkpoint(resume_from, method='hull_ddpg2021', config=config)
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    bank = bank_to(bank, device)
    learner = HullLearner(config, device=device, dtype=bank.spot.dtype, learning_rate=learning_rate,
                          target_rate=target_rate, variance_floor=variance_floor)
    replay = PrioritizedReplay(replay_capacity)
    completed, updates, transitions, elapsed_before = 0, 0, 0, 0.
    order = generator.permutation(len(bank.spot))
    if saved is not None:
        if {k:v for k,v in saved['options'].items() if k != 'episodes'} != {k:v for k,v in options.items() if k != 'episodes'}:
            raise ValueError('resume requires identical Hull recipe except total episodes')
        learner.load_state_dict(saved['learner'])
        replay = saved['replay']
        replay.arrays = [value.to(device) for value in replay.arrays]
        generator.bit_generator.state = saved['numpy_rng']
        restore_rng(saved['torch_rng'])
        completed, updates, transitions = saved['step'], saved['updates'], saved['transitions']
        elapsed_before, order = saved['elapsed_seconds'], saved['path_order']
        if completed > episodes:
            raise ValueError('requested episodes precede checkpoint')
    started = time.perf_counter()
    last_print = started
    epsilon_decays = math.ceil(math.log(.1)/math.log(.99994))
    if progress:
        print(dict(method='Hull2021', config=asdict(config), options=options, device=str(device),
                   transitions=episodes*config.n_decisions, updates=episodes*config.n_decisions-128), flush=True)
    last_loss, last_actor = float('nan'), float('nan')
    while completed < episodes:
        count = min(collection_batch, episodes-completed)
        rows = torch.as_tensor(order[np.arange(completed, completed+count) % len(order)], device=device)
        subset = bank_subset(bank, rows)
        env = TensorHedgingEnv(subset)
        observations = env.reset()
        wealth = observations.new_zeros(count)
        epsilons = observations.new_tensor(.99994**np.minimum(np.arange(completed, completed+count), epsilon_decays))
        for date in range(config.n_decisions):
            with torch.no_grad():
                targets = learner.actor.targets(observations)
                explore = torch.rand(count, device=device) <= epsilons
                random_actions = learner.actor.lower+(learner.actor.upper-learner.actor.lower)*torch.rand_like(targets)
                actions = torch.where(explore[:, None], random_actions, targets)
                following, _, terminal, _, info = env.step(actions)
                next_wealth = (info['terminal_pnl'] if terminal else
                    env.state.cash+(env.state.positions*subset.marks[:, date+1]).sum(-1)-subset.liability[:, date+1])
                rewards = (next_wealth-wealth)/reward_scale
                replay.add((observations, actions, rewards, following,
                            torch.full((count,), terminal, device=device, dtype=torch.bool)))
                observations, wealth = following, next_wealth
            previous = transitions
            transitions += count
            number_updates = max(0, transitions-max(previous, 128))
            for lane in range(number_updates):
                beta = .4+.6*min((completed+lane)/50001, 1.)
                sample, indices = replay.sample(128, beta, generator)
                sample = (*sample, torch.randperm(128, device=device), torch.randperm(128, device=device))
                if cuda_graph and learner.graph is None:
                    learner.enable_cuda_graph(sample)
                priority, loss, actor_loss = learner.step(sample)
                replay.priorities(indices, priority.detach().cpu().numpy())
                updates += 1
                last_loss, last_actor = loss, actor_loss
        completed += count
        now = time.perf_counter()
        save = checkpoint_path is not None and (completed == episodes or completed % checkpoint_every < count)
        if progress and (now-last_print >= 30 or save or completed == episodes):
            if str(device).startswith('cuda'):
                torch.cuda.synchronize()
            elapsed = elapsed_before+time.perf_counter()-started
            print(dict(episodes=completed, total=episodes, updates=updates,
                elapsed_seconds=elapsed, eta_seconds=elapsed*(episodes-completed)/completed,
                epsilon=float(epsilons[-1]), critic_loss=float(last_loss), actor_objective=float(last_actor)), flush=True)
            last_print = now
        if save:
            save_checkpoint(checkpoint_path, dict(method='hull_ddpg2021', config=asdict(config), options=options,
                learner=learner.state_dict(), replay=replay, numpy_rng=generator.bit_generator.state,
                torch_rng=rng_state(), step=completed, updates=updates, transitions=transitions,
                elapsed_seconds=elapsed_before+time.perf_counter()-started, path_order=order))
    elapsed = elapsed_before+time.perf_counter()-started
    metadata = dict(source=SOURCE, options=options, episodes=completed, learner_updates=updates,
        critic_optimizer_steps=8*updates, actor_optimizer_steps=updates,
        transitions=transitions, elapsed_seconds=elapsed, objective='mean loss + 1.5 standard deviation',
        original_batch_max_corrected=True, copied_initial_targets=True,
        full_occupied_PER_sampling=True, blockwise_collection_staleness=collection_batch,
        training_bank_paths=len(bank.spot), learner_money_multiplier=1/reward_scale,
        critic_loss=None if not updates else float(last_loss),
        actor_objective=None if not updates else float(last_actor))
    return learner.actor.eval(), metadata
