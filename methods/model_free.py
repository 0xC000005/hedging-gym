"""Model-free QR-D4PG and EX-D4PG adapters for the common terminal-ES task.

Donors are Cao et al. (Rotman/Hull), gamma-vega-rl-hedging@77dc483, and
Malekzadeh et al., EX-DRL@f1abe99. See agent/learning.py and
agent/distributional.py in those public repositories. This is a batched PyTorch
adaptation, not a reproduction of their TensorFlow option-arrival experiments.

Both retain replay, n-step distributional targets, a target actor/critic and
deterministic policy gradients. EX-D4PG additionally fits a generalized Pareto
distribution to the critic's upper loss quantiles and uses that tail in both
Bellman targets and actor improvement. Fitted tail draws are NOT market data.

Important objective adaptation: the papers use conditional VaR/CVaR. Here one
episode-global threshold optimizes the common precommitment terminal ES via
Rockafellar--Uryasev (RU). The actor minimizes the critic's expected RU loss;
the threshold is updated using initial-state predictions only. Critic error can
bias that training estimate; final risk always comes from the shared evaluator.
No derivative passes through a market transition or the financial ledger.
"""

from copy import deepcopy
from dataclasses import asdict
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from hedging_gym.finance import bank_subset, bank_to
from hedging_gym.gym_env import TensorHedgingEnv

from .policies import DirectDHPolicy, _network
from .training import _report, _sync


SOURCES = {
    "hull_rl": "https://github.com/rotmanfinhub/gamma-vega-rl-hedging/tree/77dc48326da000d983b1fb750edb2177e38c75fd",
    "exdrl": "https://github.com/pmalekzadeh/EX-DRL/tree/f1abe99df7fa9efaa65af6b9dd416c3425c64098",
}


def quantile_huber_loss(predictions, targets, probabilities):
    """QR-D4PG's pairwise quantile regression, on upper-tail terminal losses."""
    error = targets[:, None, :] - predictions[:, :, None]
    huber = F.huber_loss(error, torch.zeros_like(error), reduction="none")
    weights = (probabilities[None, :, None] - (error.detach() < 0).to(error.dtype)).abs()
    return (weights * huber).mean()


def gpd_expected_excess(cutoff, scale, shape):
    """E[(Y-cutoff)+] for Y~GPD(scale, shape), 0<shape<1.

    Integrating the survival function avoids the author's rejection-sampling
    loop for the tail expectation. This is the same Pareto distribution, not a
    Gaussian or clipped-quantile substitute.
    """
    positive = cutoff.clamp_min(0)
    survival = torch.exp(-torch.log1p(shape * positive / scale) / shape)
    return survival * (scale + shape * positive) / (1-shape) + (-cutoff).clamp_min(0)


def gpd_nll(excess, scale, shape):
    """Negative log-likelihood of nonnegative Pareto excess observations."""
    return torch.log(scale) + (1 + 1/shape) * torch.log1p(shape * excess / scale)


class DistributionalCritic(nn.Module):
    """Quantile body and optional learned GPD tail; returns are scaled losses."""

    def __init__(self, feature_dim, n_assets, hidden=(64, 64), *, quantiles=128,
                 tail_threshold=None):
        super().__init__()
        self.quantiles = _network(feature_dim+n_assets, quantiles, hidden)
        self.tail = (_network(feature_dim+n_assets, 2, hidden)
                     if tail_threshold is not None else None)
        self.tail_threshold = tail_threshold
        self.register_buffer("probabilities", (torch.arange(quantiles)+.5)/quantiles)
        # Quantile integration weights below the GPD splice sum to its mass.
        edges = torch.arange(quantiles+1)/quantiles
        threshold = 1. if tail_threshold is None else tail_threshold
        self.register_buffer("body_weights", (edges[1:].clamp_max(threshold)
                                             - edges[:-1].clamp_max(threshold)))

    def forward(self, observed, actions):
        inputs = torch.cat((observed, actions), -1)
        values = self.quantiles(inputs)
        if self.tail is None:
            return values, None, None
        raw = self.tail(inputs)
        epsilon = torch.finfo(raw.dtype).eps
        scale = F.softplus(raw[:, 0]) + epsilon
        # Author heavy_tail=True also restricts shape to (0,1), finite-mean GPD.
        shape = raw[:, 1].sigmoid().clamp(epsilon, 1-epsilon)
        return values, scale, shape

    def threshold_value(self, values):
        values = values.sort(-1).values
        index = self.tail_threshold * values.shape[-1] - .5
        left = min(max(math.floor(index), 0), values.shape[-1]-1)
        right = min(left+1, values.shape[-1]-1)
        return values[:, left] + (index-left) * (values[:, right]-values[:, left])

    def distribution(self, observed, actions):
        """Equal-probability quadrature for the target loss distribution."""
        values, scale, shape = self(observed, actions)
        values = values.sort(-1).values
        if self.tail is None:
            return values
        location = self.threshold_value(values)
        tail_probabilities = ((self.probabilities-self.tail_threshold)
                              / (1-self.tail_threshold)).clamp_min(0)
        excess = (scale[:, None] / shape[:, None]
                  * torch.expm1(-shape[:, None] * torch.log1p(-tail_probabilities)))
        return torch.where(self.probabilities[None] > self.tail_threshold,
                           location[:, None]+excess, values)

    def expected_ru(self, observed, actions, threshold, alpha):
        values, scale, shape = self(observed, actions)
        values = values.sort(-1).values
        positive_part = ((values-threshold).clamp_min(0)*self.body_weights).sum(-1)
        if self.tail is not None:
            location = self.threshold_value(values)
            positive_part = positive_part + (1-self.tail_threshold) * gpd_expected_excess(
                threshold-location, scale, shape)
        return threshold + positive_part/(1-alpha)

    def tail_loss(self, observed, actions):
        """Fit excess upper quantiles by MLE, as in the EX-DRL author learner."""
        values, scale, shape = self(observed, actions)
        values = values.detach().sort(-1).values
        excess = (values[:, self.probabilities > self.tail_threshold]
                  - self.threshold_value(values)[:, None]).clamp_min(0)
        return gpd_nll(excess, scale[:, None], shape[:, None]).mean()


def _targets(actor, observed, config):
    indices = [actor.observation_fields.index(f"{name}_position")
               for name in actor.instrument_names]
    return actor(observed, observed[:, indices], config.execution.holding_lower,
                 config.execution.holding_upper).target_holdings


@torch.no_grad()
def collect_episodes(actor, bank, *, noise=0., generator=None, n_step=5):
    """Causal observations and executed actions; future paths only enter labels.

    The environment is completely detached. N-step costs are zero until the
    authoritative terminal loss; bootstrap is disabled at/after termination.
    """
    config = bank.config
    env = TensorHedgingEnv(bank)
    observed = env.reset()
    observations, actions = [observed], []
    lower = observed.new_tensor(config.execution.vector("holding_lower", config.n_assets))
    upper = observed.new_tensor(config.execution.vector("holding_upper", config.n_assets))
    for _ in range(config.n_steps):
        target = _targets(actor, observed, config)
        if noise:
            target = (target + noise*(upper-lower)*torch.randn(
                target.shape, generator=generator, device=target.device, dtype=target.dtype)).clamp(lower, upper)
        actions.append(target)
        observed, _, _, _, result = env.step(target)
        observations.append(observed)
    observations = torch.stack(observations, 1)
    actions = torch.stack(actions, 1)
    times = torch.arange(config.n_steps, device=observed.device)
    next_times = (times+n_step).clamp_max(config.n_steps)
    terminal = (next_times == config.n_steps)[None].expand(len(observed), -1)
    costs = torch.where(terminal, result["terminal_loss"][:, None], 0.)
    transitions = (observations[:, :-1].flatten(0, 1), actions.flatten(0, 1),
                   costs.flatten(), observations[:, next_times].flatten(0, 1),
                   terminal.flatten())
    return transitions, result["terminal_loss"], observations[:, 0]


class _Replay:
    """Fixed-size tensor ring, avoiding a Python transition/worker framework."""

    def __init__(self, capacity):
        self.capacity, self.size, self.cursor = capacity, 0, 0
        self.arrays = None

    def add(self, values):
        if self.arrays is None:
            self.arrays = [value.new_empty((self.capacity, *value.shape[1:])) for value in values]
        count = min(len(values[0]), self.capacity)
        slots = (torch.arange(count, device=values[0].device)+self.cursor) % self.capacity
        for stored, value in zip(self.arrays, values):
            stored[slots] = value[-count:]
        self.cursor = (self.cursor+count) % self.capacity
        self.size = min(self.size+count, self.capacity)

    def sample(self, count, generator):
        indices = torch.randint(self.size, (count,), device=self.arrays[0].device, generator=generator)
        return tuple(value[indices] for value in self.arrays)


def train_model_free(method, train_bank, *, seed=7, updates=100, batch_size=32,
                     hidden=(64, 64), device="cpu", progress=True,
                     learning_rate=1e-3, critic_learning_rate=1e-3,
                     zeta_learning_rate=3e-4, quantiles=128,
                     tail_threshold=.96, n_step=5, gradient_steps=4,
                     replay_capacity=65536, exploration_noise=.1, target_rate=.01):
    """Return DirectDHPolicy-compatible actor and source/cost/training metadata.

    hull_rl is the Rotman quantile-D4PG family; exdrl adds the genuine GPD tail.
    Updates each collect batch_size complete episodes, then gradient_steps
    replay minibatches. Loss scaling is fixed from initial training episodes.
    The ES threshold and its learning rate stay in portfolio-money units, as
    in Deep Hedging; only critic inputs/outputs use standardized loss units.
    Uniform replay and Polyak targets replace Acme/Reverb infrastructure.
    Continuous actors support bounded trades and fees, but not discrete lots or
    minimum-order constraints; no silent projection changes their problem.
    """
    if method not in SOURCES or min(updates, batch_size, gradient_steps, replay_capacity, n_step) < 1:
        raise ValueError("choose hull_rl/exdrl and positive training work")
    if quantiles < 4 or not 0 < tail_threshold < 1 or (method == "exdrl" and quantiles*(1-tail_threshold) < 2):
        raise ValueError("EX-D4PG needs at least two quantiles above its GPD threshold")
    if min(learning_rate, critic_learning_rate, zeta_learning_rate) <= 0 or not 0 < target_rate <= 1:
        raise ValueError("learning rates must be positive and target_rate in (0,1]")
    if exploration_noise < 0 or len(train_bank.spot) < 1:
        raise ValueError("exploration noise must be nonnegative and the bank nonempty")
    config = train_bank.config
    if any(config.execution.vector("minimum_trade", config.n_assets)
           + config.execution.vector("trade_lot", config.n_assets)):
        raise ValueError("continuous D4PG does not implement minimum-trade or lot constraints")
    started = time.perf_counter()
    device = torch.device(device)
    torch.manual_seed(seed)
    bank = bank_to(train_bank, device)
    actor = DirectDHPolicy(config, hidden=hidden).to(device=device, dtype=bank.spot.dtype)
    critic = DistributionalCritic(actor.feature_dim, config.n_assets, hidden,
        quantiles=quantiles, tail_threshold=tail_threshold if method == "exdrl" else None).to(
            device=device, dtype=bank.spot.dtype)
    generator = torch.Generator(device=device).manual_seed(seed+100003)
    options = dict(updates=updates, batch_size=batch_size, hidden=list(hidden),
        learning_rate=learning_rate, critic_learning_rate=critic_learning_rate,
        zeta_learning_rate=zeta_learning_rate, quantiles=quantiles,
        tail_threshold=tail_threshold if method == "exdrl" else None,
        n_step=n_step, gradient_steps=gradient_steps, replay_capacity=replay_capacity,
        exploration_noise=exploration_noise, target_rate=target_rate)
    if progress:
        _report("train_start", method=method, seed=seed, device=str(device),
                options=options, expected_episode_rollouts=updates*batch_size,
                initialization_paths=min(256, len(bank.spot)))
    initial_bank = bank_subset(bank, slice(0, min(256, len(bank.spot))))
    initial_transitions, initial_losses, initial_observed = collect_episodes(actor, initial_bank,
                                                                           n_step=n_step)
    return_scale = initial_losses.std(unbiased=False).clamp_min(config.market.spot0*1e-6)
    scaled_losses = initial_losses/return_scale
    zeta = nn.Parameter(torch.quantile(initial_losses, config.risk.alpha))
    with torch.no_grad():
        critic.quantiles[-1].weight.mul_(.01)
        critic.quantiles[-1].bias.copy_(torch.quantile(scaled_losses, critic.probabilities))
        if critic.tail is not None:
            excess = (scaled_losses-torch.quantile(scaled_losses, tail_threshold)).clamp_min(0)
            scale = (excess.sum()/(excess > 0).sum().clamp_min(1)*.8).clamp_min(.01)
            critic.tail[-1].weight.mul_(.01)
            critic.tail[-1].bias.copy_(torch.stack((torch.log(torch.expm1(scale)), scale.new_tensor(math.log(.2/.8)))))
    target_actor, target_critic = deepcopy(actor).requires_grad_(False), deepcopy(critic).requires_grad_(False)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = torch.optim.Adam(critic.quantiles.parameters(), lr=critic_learning_rate)
    tail_optimizer = (torch.optim.Adam(critic.tail.parameters(), lr=critic_learning_rate)
                      if critic.tail is not None else None)
    threshold_optimizer = torch.optim.Adam([zeta], lr=zeta_learning_rate)
    replay = _Replay(replay_capacity)
    initial_states, initial_actions, initial_costs, next_states, initial_done = initial_transitions
    replay.add((initial_states, initial_actions, initial_costs/return_scale, next_states, initial_done))
    _sync(device)
    initialization_seconds = time.perf_counter()-started
    training_started, history = time.perf_counter(), []
    for update in range(1, updates+1):
        indices = torch.randint(len(bank.spot), (batch_size,), generator=generator, device=device)
        transitions, episode_losses, _ = collect_episodes(actor, bank_subset(bank, indices),
            noise=exploration_noise, generator=generator, n_step=n_step)
        observed, actions, costs, next_observed, done = transitions
        replay.add((observed, actions, costs/return_scale, next_observed, done))
        for _ in range(gradient_steps):
            observed, actions, costs, next_observed, done = replay.sample(batch_size, generator)
            with torch.no_grad():
                future = target_critic.distribution(next_observed, _targets(target_actor, next_observed, config))
                target_losses = costs[:, None] + (~done)[:, None] * future
            predicted, _, _ = critic(observed, actions)
            critic_loss = quantile_huber_loss(predicted, target_losses, critic.probabilities)
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.quantiles.parameters(), 5., error_if_nonfinite=True)
            critic_optimizer.step()
            if tail_optimizer is not None:
                tail_loss = critic.tail_loss(observed, actions)
                tail_optimizer.zero_grad(set_to_none=True)
                tail_loss.backward()
                nn.utils.clip_grad_norm_(critic.tail.parameters(), 5., error_if_nonfinite=True)
                tail_optimizer.step()
            # Freeze critic weights, not its action derivative. Never backprop
            # through sampled transitions, the simulator, or cash accounting.
            critic.requires_grad_(False)
            actor_loss = critic.expected_ru(observed, _targets(actor, observed, config),
                                            zeta.detach()/return_scale, config.risk.alpha).mean()
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 5., error_if_nonfinite=True)
            actor_optimizer.step()
            # Only initial-state predictions define the global ES threshold.
            with torch.no_grad():
                initial_actions = _targets(actor, initial_observed, config)
            threshold_loss = return_scale * critic.expected_ru(
                initial_observed, initial_actions, zeta/return_scale, config.risk.alpha).mean()
            threshold_optimizer.zero_grad(set_to_none=True)
            threshold_loss.backward()
            threshold_optimizer.step()
            critic.requires_grad_(True)
            with torch.no_grad():
                for target, current in zip(target_actor.parameters(), actor.parameters()):
                    target.lerp_(current, target_rate)
                for target, current in zip(target_critic.parameters(), critic.parameters()):
                    target.lerp_(current, target_rate)
        if update == 1 or update % 20 == 0 or update == updates:
            _sync(device)
            elapsed = time.perf_counter()-training_started
            record = dict(completed=update, total=updates, elapsed_seconds=elapsed,
                eta_seconds=elapsed*(updates-update)/update, replay_transitions=replay.size,
                critic_loss=float(critic_loss.detach()), actor_ru_estimate=float(actor_loss.detach()*return_scale),
                zeta=float(zeta.detach()), mean_exploration_loss=float(episode_losses.mean()))
            if tail_optimizer is not None:
                record["gpd_nll"] = float(tail_loss.detach())
            history.append(record)
            if progress:
                _report("train_progress", method=method, seed=seed, **record)
    _sync(device)
    metadata = dict(method=method, algorithm="EX-D4PG" if method == "exdrl" else "QR-D4PG",
        source=SOURCES[method], scope="common-environment adaptation, not author benchmark reproduction",
        objective="episode-global terminal ES via critic expected RU; initial-state threshold update",
        deviations=["PyTorch batched tensor environment instead of Acme/Reverb",
                    "Heston common book, all configured hedge instruments, authoritative terminal cash loss",
                    "global terminal ES replaces native conditional VaR/CVaR",
                    "uniform replay; Polyak targets; GPD inverse-CDF quadrature and analytic tail expectation"],
        seed=seed, device=str(device), options=options, config=asdict(config), history=history,
        zeta=float(zeta.detach()), threshold_units="portfolio_money", return_scale=float(return_scale),
        expected_episode_rollouts=updates*batch_size, initialization_paths=len(initial_bank.spot),
        transition_count=updates*batch_size*config.n_steps, gradient_updates=updates*gradient_steps,
        initialization_seconds=initialization_seconds, training_seconds=time.perf_counter()-training_started,
        total_seconds=time.perf_counter()-started,
        parameter_count=sum(parameter.numel() for parameter in actor.parameters()),
        critic_parameter_count=sum(parameter.numel() for parameter in critic.parameters()))
    return actor, metadata
