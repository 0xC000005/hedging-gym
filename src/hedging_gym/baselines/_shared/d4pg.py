"""Shared batched collection, replay and update machinery for QR/EX-D4PG.

Method modules own their critics and source attribution. This loop receives
that critic explicitly; it does not select public method implementations.
The actor minimizes expected Rockafellar--Uryasev loss with an episode-global
threshold fitted from initial-state predictions. No derivative passes through
the simulator or ledger. See docs/baseline-methods.md for source deviations.
"""
import math
import time
from copy import deepcopy
from dataclasses import asdict

import torch
from torch import nn
from torch.nn import functional as F

from hedging_gym.baselines.deep_hedging import DirectDHPolicy
from hedging_gym.environment.finance import bank_subset, bank_to
from hedging_gym.environment.gym_env import TensorHedgingEnv

from .checkpoints import (
    check_resume_options,
    due_checkpoint,
    load_checkpoint,
    restore_rng,
    rng_state,
    save_checkpoint,
)
from .training import _report, _sync


def quantile_huber_loss(predictions, targets, probabilities, kappa=1.):
    """Quantile loss with explicit smoothing in standardized return units.

    kappa=0 is pinball loss. A fixed nonzero Huber threshold changes the
    population quantile optimum; return normalization must not hide its units.
    """
    error = targets[:, None, :] - predictions[:, :, None]
    if kappa < 0:
        raise ValueError("quantile Huber kappa must be nonnegative")
    huber = (error.abs() if kappa == 0 else
             F.huber_loss(error, torch.zeros_like(error), delta=kappa, reduction="none") / kappa)
    weights = (probabilities[None, :, None] - (error.detach() < 0).to(error.dtype)).abs()
    return (weights * huber).mean()


def action_gradient_loss(values, actions, clip=None):
    """Deterministic policy-gradient surrogate with per-state dQ/da clipping.

    Clip before the minibatch mean, as in the donor DPG learner. This bounds
    the critic's action derivative, separately from actor parameter gradients.
    """
    derivative, = torch.autograd.grad(values.sum(), actions, retain_graph=True)
    derivative = derivative.detach()
    if clip is not None:
        derivative = derivative * (clip / derivative.norm(dim=-1, keepdim=True).clamp_min(clip))
    return (actions * derivative).sum(-1).mean()


def _targets(actor, observed, config):
    indices = [actor.observation_fields.index(f"{name}_position")
               for name in actor.instrument_names]
    return actor(observed, observed[:, indices], config.execution.holding_lower,
                 config.execution.holding_upper).target_holdings


def _accumulated_cost(observed, actor, config, initial_cash):
    """Initial premium minus current marked hedge wealth, in money units.

    This known state potential permits native-style dense P&L labels without
    changing terminal ES. It excludes the liability until final settlement.
    """
    fields = actor.observation_fields
    positions = observed[:, [fields.index(f"{name}_position") for name in actor.instrument_names]]
    mids = observed[:, [fields.index(f"{name}_mid") for name in actor.instrument_names]]
    wealth = config.market.spot0*(observed[:, fields.index("cash")] + (positions*mids).sum(-1))
    return initial_cash-wealth


@torch.no_grad()
def collect_episodes(actor, bank, *, noise=0., generator=None, n_step=5, dense_rewards=False,
                     first_action=None):
    """Causal observations and executed actions; future paths only enter labels.

    The environment is completely detached. N-step costs are zero until the
    authoritative terminal loss; bootstrap is disabled at/after termination.
    With dense_rewards, differences of marked hedge wealth telescope to the
    same loss. The critic then predicts remaining rather than total loss.
    """
    config = bank.config
    env = TensorHedgingEnv(bank)
    observed = env.reset()
    observations, actions = [observed], []
    lower = observed.new_tensor(config.execution.vector("holding_lower", config.n_assets))
    upper = observed.new_tensor(config.execution.vector("holding_upper", config.n_assets))
    for time_index in range(config.n_decisions):
        target = (first_action.expand(len(observed), -1) if time_index == 0 and first_action is not None
                  else _targets(actor, observed, config))
        if noise:
            target = (target + noise*(upper-lower)*torch.randn(
                target.shape, generator=generator, device=target.device, dtype=target.dtype)).clamp(lower, upper)
        actions.append(target)
        observed, _, _, _, result = env.step(target)
        observations.append(observed)
    observations = torch.stack(observations, 1)
    actions = torch.stack(actions, 1)
    times = torch.arange(config.n_decisions, device=observed.device)
    next_times = (times+n_step).clamp_max(config.n_decisions)
    terminal = (next_times == config.n_decisions)[None].expand(len(observed), -1)
    costs = torch.where(terminal, result["terminal_loss"][:, None], 0.)
    if dense_rewards:
        potentials = _accumulated_cost(observations.flatten(0, 1), actor, config,
            bank.liability[0, 0]).reshape(len(observed), config.n_decisions+1)
        potentials[:, -1] = result["terminal_loss"]
        costs = potentials[:, next_times]-potentials[:, :-1]
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


def train_d4pg(method, train_bank, *, critic_class, source, seed=7, updates=100, batch_size=32,
                     hidden=(64, 64), device="cpu", progress=True,
                     learning_rate=1e-3, critic_learning_rate=1e-3,
                     zeta_learning_rate=3e-4, quantiles=128,
                     tail_threshold=.96, n_step=5, gradient_steps=4,
                     replay_capacity=65536, exploration_noise=.1, target_rate=.01,
                     collection_batch_size=None, dense_rewards=False,
                     quantile_kappa=1., tail_learning_rate=None, tail_policy_actions=False,
                     actor_warmup_updates=0, actor_update_period=1, action_gradient_clip=None,
                     extend_frozen_warmup=False,
                     checkpoint_path=None, checkpoint_every=200, resume_from=None):
    """Return DirectDHPolicy-compatible actor and source/cost/training metadata.

    hull_rl is the Rotman quantile-D4PG family; exdrl adds the genuine GPD tail.
    Updates each collect collection_batch_size complete episodes (batch_size
    by default), then gradient_steps replay minibatches of batch_size samples.
    This separates simulator work from the donor's replay-samples-per-insert
    training intensity. Loss scaling is fixed from initial training episodes.
    With dense_rewards=True, the critic learns remaining marked-PnL loss and
    the actor's RU threshold subtracts accumulated loss. The objective is still
    the same episode-global terminal ES, not a nested sequence of conditional ES.
    The ES threshold and its learning rate stay in portfolio-money units, as
    in Deep Hedging; only critic inputs/outputs use standardized loss units.
    Uniform replay and Polyak targets replace Acme/Reverb infrastructure.
    Continuous actors support bounded trades and fees, but not discrete lots or
    minimum-order constraints; no silent projection changes their problem.
    Checkpoints contain the replay, all trainable/target state and both RNGs;
    resume may increase total updates but does not change the training recipe.
    Explicitly extending a still-frozen actor warmup is also supported: neither
    the actor nor the global threshold may have taken an optimizer step yet.
    Smaller quantile smoothing and delayed actor updates are disclosed
    stabilization adaptations, not claimed to reproduce the donor recipe.
    """
    collection_batch_size = batch_size if collection_batch_size is None else collection_batch_size
    tail_learning_rate = critic_learning_rate if tail_learning_rate is None else tail_learning_rate
    if method not in ("hull_rl", "exdrl") or min(updates, batch_size, collection_batch_size, gradient_steps, replay_capacity, n_step, checkpoint_every) < 1:
        raise ValueError("choose hull_rl/exdrl and positive training work")
    if quantiles < 4 or not 0 < tail_threshold < 1 or (method == "exdrl" and quantiles*(1-tail_threshold) < 2):
        raise ValueError("EX-D4PG needs at least two quantiles above its GPD threshold")
    if min(learning_rate, critic_learning_rate, zeta_learning_rate) <= 0 or not 0 < target_rate <= 1:
        raise ValueError("learning rates must be positive and target_rate in (0,1]")
    if (quantile_kappa < 0 or tail_learning_rate <= 0 or actor_warmup_updates < 0
            or actor_update_period < 1 or (action_gradient_clip is not None and action_gradient_clip <= 0)):
        raise ValueError("invalid quantile smoothing, tail rate, actor schedule or action derivative clip")
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
    critic_options = dict(quantiles=quantiles)
    if method == "exdrl":
        critic_options["tail_threshold"] = tail_threshold
    critic = critic_class(actor.feature_dim, config.n_assets, hidden, **critic_options).to(
        device=device, dtype=bank.spot.dtype)
    generator = torch.Generator(device=device).manual_seed(seed+100003)
    options = dict(updates=updates, batch_size=batch_size, hidden=list(hidden),
        learning_rate=learning_rate, critic_learning_rate=critic_learning_rate,
        zeta_learning_rate=zeta_learning_rate, quantiles=quantiles,
        tail_threshold=tail_threshold if method == "exdrl" else None,
        n_step=n_step, gradient_steps=gradient_steps, replay_capacity=replay_capacity,
        exploration_noise=exploration_noise, target_rate=target_rate,
        collection_batch_size=collection_batch_size, dense_rewards=dense_rewards,
        quantile_kappa=quantile_kappa, tail_learning_rate=tail_learning_rate,
        tail_policy_actions=tail_policy_actions, actor_warmup_updates=actor_warmup_updates,
        actor_update_period=actor_update_period, action_gradient_clip=action_gradient_clip)
    if progress:
        _report("train_start", method=method, seed=seed, device=str(device),
                options=options, expected_episode_rollouts=updates*collection_batch_size,
                initialization_paths=min(256, len(bank.spot)))
    initial_bank = bank_subset(bank, slice(0, min(256, len(bank.spot))))
    # Native D4PG explores during replay warmup too. Thousands of deterministic
    # initial transitions otherwise dominate the first actor updates while the
    # critic has almost no evidence of how alternative actions change returns.
    initial_transitions, initial_losses, initial_observed = collect_episodes(
        actor, initial_bank, n_step=n_step, noise=exploration_noise, generator=generator,
        dense_rewards=dense_rewards)
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
    tail_optimizer = (torch.optim.Adam(critic.tail.parameters(), lr=tail_learning_rate)
                      if critic.tail is not None else None)
    threshold_optimizer = torch.optim.Adam([zeta], lr=zeta_learning_rate)
    replay = _Replay(replay_capacity)
    initial_states, initial_actions, initial_costs, next_states, initial_done = initial_transitions
    replay.add((initial_states, initial_actions, initial_costs/return_scale, next_states, initial_done))
    completed, history, previous_seconds = 0, [], 0.
    if resume_from is not None:
        saved = load_checkpoint(resume_from, method=method, config=config)
        saved_for_check = saved
        previous_warmup = saved["options"]["actor_warmup_updates"]
        if extend_frozen_warmup and actor_warmup_updates != previous_warmup:
            if (actor_warmup_updates < previous_warmup or saved["step"] > previous_warmup
                    or saved["actor_optimizer"]["state"] or saved["threshold_optimizer"]["state"]):
                raise ValueError("warmup extension requires a checkpoint whose actor has never updated")
            saved_for_check = dict(saved, options=dict(saved["options"],
                                                       actor_warmup_updates=actor_warmup_updates))
            if progress:
                _report("extend_frozen_warmup", completed=saved["step"],
                        previous=previous_warmup, new=actor_warmup_updates)
        check_resume_options(saved_for_check, options)
        if saved["seed"] != seed:
            raise ValueError("resume requires the original training seed")
        actor.load_state_dict(saved["actor"])
        critic.load_state_dict(saved["critic"])
        target_actor.load_state_dict(saved["target_actor"])
        target_critic.load_state_dict(saved["target_critic"])
        with torch.no_grad():
            zeta.copy_(saved["zeta"].to(device))
        return_scale = saved["return_scale"].to(device)
        initial_observed = saved["initial_observed"].to(device)
        actor_optimizer.load_state_dict(saved["actor_optimizer"])
        critic_optimizer.load_state_dict(saved["critic_optimizer"])
        threshold_optimizer.load_state_dict(saved["threshold_optimizer"])
        if tail_optimizer is not None:
            tail_optimizer.load_state_dict(saved["tail_optimizer"])
        replay.arrays = [value.to(device) for value in saved["replay"]["arrays"]]
        replay.size, replay.cursor = saved["replay"]["size"], saved["replay"]["cursor"]
        generator.set_state(saved["generator_state"])
        restore_rng(saved["rng_state"])
        completed, history = saved["step"], saved["history"]
        previous_seconds = saved["training_seconds"]
        if completed > updates:
            raise ValueError("total updates cannot precede the saved step")
        if progress:
            _report("train_resume", method=method, seed=seed, completed=completed, total=updates)
    _sync(device)
    initialization_seconds = time.perf_counter()-started
    training_started = time.perf_counter()
    for update in range(completed+1, updates+1):
        indices = torch.randint(len(bank.spot), (collection_batch_size,), generator=generator, device=device)
        transitions, episode_losses, _ = collect_episodes(actor, bank_subset(bank, indices),
            noise=exploration_noise, generator=generator, n_step=n_step, dense_rewards=dense_rewards)
        observed, actions, costs, next_observed, done = transitions
        replay.add((observed, actions, costs/return_scale, next_observed, done))
        actor_updated = False
        for gradient_step in range(gradient_steps):
            observed, actions, costs, next_observed, done = replay.sample(batch_size, generator)
            with torch.no_grad():
                future = target_critic.distribution(next_observed, _targets(target_actor, next_observed, config))
                target_losses = costs[:, None] + (~done)[:, None] * future
            predicted, _, _ = critic(observed, actions)
            critic_loss = quantile_huber_loss(predicted, target_losses, critic.probabilities, quantile_kappa)
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.quantiles.parameters(), 5., error_if_nonfinite=True)
            critic_optimizer.step()
            if tail_optimizer is not None:
                tail_observed, tail_actions = observed, actions
                if tail_policy_actions:
                    # The author fits the tail at next-state target-policy
                    # actions. Terminal states have no continuation value;
                    # use their preceding state with the same target policy.
                    tail_observed = torch.where(done[:, None], observed, next_observed)
                    with torch.no_grad():
                        tail_actions = _targets(target_actor, tail_observed, config)
                tail_loss = critic.tail_loss(tail_observed, tail_actions)
                tail_optimizer.zero_grad(set_to_none=True)
                tail_loss.backward()
                nn.utils.clip_grad_norm_(critic.tail.parameters(), 5., error_if_nonfinite=True)
                tail_optimizer.step()
            # Freeze critic weights, not its action derivative. Never backprop
            # through sampled transitions, the simulator, or cash accounting.
            critic.requires_grad_(False)
            accumulated = (_accumulated_cost(observed, actor, config, bank.liability[0, 0])
                           if dense_rewards else observed.new_zeros(len(observed)))
            actor_actions = _targets(actor, observed, config)
            actor_values = (critic.expected_ru(observed, actor_actions,
                (zeta.detach()-accumulated)/return_scale, config.risk.alpha)
                + accumulated/return_scale)
            actor_loss = actor_values.mean()
            global_gradient_step = (update-1)*gradient_steps+gradient_step+1
            improve_actor = (update > actor_warmup_updates
                             and global_gradient_step % actor_update_period == 0)
            if improve_actor:
                actor_optimizer.zero_grad(set_to_none=True)
                surrogate = (actor_loss if action_gradient_clip is None else
                             action_gradient_loss(actor_values, actor_actions, action_gradient_clip))
                surrogate.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 5., error_if_nonfinite=True)
                actor_optimizer.step()
                actor_updated = True
            # Only initial-state predictions define the global ES threshold.
            with torch.no_grad():
                initial_actions = _targets(actor, initial_observed, config)
            threshold_loss = return_scale * critic.expected_ru(
                initial_observed, initial_actions, zeta/return_scale, config.risk.alpha).mean()
            if improve_actor:
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
            elapsed = previous_seconds+time.perf_counter()-training_started
            record = dict(completed=update, total=updates, elapsed_seconds=elapsed,
                eta_seconds=elapsed*(updates-update)/update, replay_transitions=replay.size,
                critic_loss=float(critic_loss.detach()), actor_ru_estimate=float(actor_loss.detach()*return_scale),
                zeta=float(zeta.detach()), mean_exploration_loss=float(episode_losses.mean()),
                actor_updated=actor_updated)
            if tail_optimizer is not None:
                record["gpd_nll"] = float(tail_loss.detach())
            history.append(record)
            if progress:
                _report("train_progress", method=method, seed=seed, **record)
        if checkpoint_path is not None and due_checkpoint(update, updates, checkpoint_every):
            save_checkpoint(checkpoint_path, dict(method=method, config=asdict(config),
                options=options, seed=seed, step=update, actor=actor.state_dict(),
                critic=critic.state_dict(), target_actor=target_actor.state_dict(),
                target_critic=target_critic.state_dict(), actor_optimizer=actor_optimizer.state_dict(),
                critic_optimizer=critic_optimizer.state_dict(),
                tail_optimizer=None if tail_optimizer is None else tail_optimizer.state_dict(),
                threshold_optimizer=threshold_optimizer.state_dict(), zeta=zeta.detach(),
                return_scale=return_scale, initial_observed=initial_observed,
                replay=dict(arrays=replay.arrays, size=replay.size, cursor=replay.cursor),
                generator_state=generator.get_state(), rng_state=rng_state(), history=history,
                training_seconds=previous_seconds+time.perf_counter()-training_started))
    _sync(device)
    metadata = dict(method=method, algorithm="EX-D4PG" if method == "exdrl" else "QR-D4PG",
        source=source, scope="common-environment adaptation, not author benchmark reproduction",
        objective="episode-global terminal ES via critic expected RU; initial-state threshold update",
        reward_labels="telescoping dense marked-PnL" if dense_rewards else "sparse terminal loss",
        deviations=["PyTorch batched tensor environment instead of Acme/Reverb",
                    "Heston common book, all configured hedge instruments, authoritative terminal cash loss",
                    "global terminal ES replaces native conditional VaR/CVaR",
                    "uniform replay; Polyak targets; GPD inverse-CDF quadrature and analytic tail expectation"],
        seed=seed, device=str(device), options=options, config=asdict(config), history=history,
        zeta=float(zeta.detach()), threshold_units="portfolio_money", return_scale=float(return_scale),
        quantile_kappa_money=quantile_kappa*float(return_scale),
        initial_threshold_source="empirical quantile of exploratory initialization episodes",
        expected_episode_rollouts=updates*collection_batch_size, initialization_paths=len(initial_bank.spot),
        transition_count=updates*collection_batch_size*config.n_decisions,
        replay_samples_per_insert=gradient_steps*batch_size/(collection_batch_size*config.n_decisions),
        gradient_updates=updates*gradient_steps,
        actor_gradient_updates=max(0, updates*gradient_steps//actor_update_period
            - min(updates, actor_warmup_updates)*gradient_steps//actor_update_period),
        initialization_seconds=initialization_seconds,
        training_seconds=previous_seconds+time.perf_counter()-training_started,
        total_seconds=time.perf_counter()-started,
        parameter_count=sum(parameter.numel() for parameter in actor.parameters()),
        critic_parameter_count=sum(parameter.numel() for parameter in critic.parameters()))
    return actor, metadata
