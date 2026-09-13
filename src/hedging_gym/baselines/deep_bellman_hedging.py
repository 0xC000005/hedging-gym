"""Deep Bellman Hedging: actor-critic value iteration under a monetary utility.

Paper: Buehler, Murray and Wood, https://arxiv.org/abs/2207.00932v3.
Implementation: equation-based PyTorch implementation; the paper has no public
code. The utility list follows the paper and the author's vanilla Deep Hedging
objectives (SOURCE below, GPL-3.0, equations only). Notes: docs/baseline-methods.md.

The value function is the risk-adjusted excess of a book over its mark (paper
section 2), so it starts and settles at exactly zero. Rewards are one-step
changes of marked wealth from the shared ledger primitives; with the enforced
zero rates they telescope to terminal P&L. Training samples random dates and
random holdings from the simulated bank instead of tabulated history, so one
trained model covers arbitrary initial books (paper section 3.1).

Source differences: the reward enters the utility as in Definition 1 rather than
outside it as displayed in the actor/critic objectives (11)-(13); the critic
minimizes the unconditional squared loss the paper shows has the same gradient;
cash is not a network input because monetary utilities are cash-invariant; and a
configured terminal closeout fee is charged inside the final reward. Learner
quantities are expressed in units of the one-day stock move spot0*sqrt(v0*dt),
which equals scaling the risk aversion, so that optimizer steps match the value
scale; risk aversion and reported values stay in money units. Beyond the
published one-step scheme, `steps` selects the paper's n-step operator T_n (whose
fixed point differs from the one-step one unless the utility is time-consistent,
as the paper's remark on multiple time steps notes), `scenarios` averages targets
over fresh conditional continuations, and `aggregate="entropic"` uses the
closed-form entropic certainty equivalent.
"""
import math
import time
from copy import deepcopy
from dataclasses import asdict

import torch
from torch import nn

from hedging_gym.baselines._shared.checkpoints import (
    check_resume_options,
    due_checkpoint,
    load_checkpoint,
    restore_rng,
    rng_state,
    save_checkpoint,
)
from hedging_gym.baselines._shared.controllers import (
    policy_controller as make_controller,
)
from hedging_gym.baselines._shared.policy import _ConfiguredPolicy, _network
from hedging_gym.baselines._shared.training import _report, _sync
from hedging_gym.baselines.deep_hedging import DirectDHPolicy
from hedging_gym.environment.finance import (
    LedgerState,
    bank_subset,
    bank_to,
    initial_state,
    liquidate,
    mark_state,
    observation,
    observation_from_state,
    trade_step,
)
from hedging_gym.environment.finance import transition as market_transition

SOURCE = "https://github.com/hansbuehler/deephedging/tree/fe53fd9a7e402d4ba18bd5b9ec704e0361554f7c"

__all__ = ["UTILITIES", "DeepBellmanPolicy", "BuehlerZero", "DeepBellmanLearner",
           "train_deep_bellman", "make_controller"]


def _entropy(x, lam):
    return -torch.expm1(-lam*x)/lam


# Paper section 2.1: u(0)=0 and u'(0)=1 where differentiable; lam is the risk aversion.
UTILITIES = {
    "identity": lambda x, lam: x,
    "entropy": _entropy,
    "cvar": lambda x, lam: (1+lam)*x.clamp_max(0),
    "truncated_entropy": lambda x, lam: torch.where(x > 0, _entropy(x.clamp_min(0), lam), x-.5*lam*x.square()),
    "vicky": lambda x, lam: (1+lam*x-(1+(lam*x).square()).sqrt())/lam,
    "quadratic": lambda x, lam: torch.where(x < 1/lam, .5/lam-.5*lam*(x-1/lam).square(), .5/lam),
}


def oce(utility, outcome, shift, lam):
    """Optimized certainty equivalent integrand u(X + y) - y (paper definition 2)."""
    return utility(outcome+shift, lam)-shift


def entropic_certainty(outcome, lam):
    """Closed-form entropic certainty equivalent over the scenario axis (paper section 2.1)."""
    return -(torch.logsumexp(-lam*outcome, 1)-math.log(outcome.shape[1]))/lam


def position_bounds(config, position_range):
    """Per-asset sampling interval for training books: holding bounds cut to position_range."""
    n = config.n_assets
    lower = list(config.execution.vector("holding_lower", n))
    upper = list(config.execution.vector("holding_upper", n))
    if position_range is not None:
        ranges = ([float(position_range)]*n if isinstance(position_range, (int, float))
                  else [float(value) for value in position_range])
        if len(ranges) != n or min(ranges) <= 0:
            raise ValueError("position_range needs one positive value per asset")
        lower = [max(lo, -r) for lo, r in zip(lower, ranges)]
        upper = [min(hi, r) for hi, r in zip(upper, ranges)]
    if not all(math.isfinite(lo) and math.isfinite(hi) for lo, hi in zip(lower, upper)):
        raise ValueError("position_range is required for unbounded holdings")
    return lower, upper


class _Inputs(nn.Module):
    """Observation columns without cash, holdings in units of their sampling range."""
    def __init__(self, fields, instruments, scale):
        super().__init__()
        keep = [index for index, name in enumerate(fields) if name != "cash"]
        divisor = torch.ones(len(keep))
        for name, value in zip(instruments, scale):
            divisor[keep.index(fields.index(f"{name}_position"))] = value
        self.register_buffer("columns", torch.tensor(keep))
        self.register_buffer("divisor", divisor)
        self.width = len(keep)

    def forward(self, observed):
        return observed[..., self.columns]/self.divisor


def _head(fields, instruments, scale, outputs, hidden):
    inputs = _Inputs(fields, instruments, scale)
    return nn.Sequential(inputs, _network(inputs.width, outputs, hidden))


def _scale(lower, upper):
    return [max(abs(lo), abs(hi)) or 1. for lo, hi in zip(lower, upper)]


class DeepBellmanPolicy(DirectDHPolicy):
    """Deep Hedging actor and target bounding on cash-free, holding-scaled inputs."""
    def __init__(self, config, hidden=(32, 32), *, position_range=None):
        _ConfiguredPolicy.__init__(self, config)
        scale = _scale(*position_bounds(config, position_range))
        self.continuous = _head(self.observation_fields, self.instrument_names, scale, self.n_assets, hidden)


class BuehlerZero(nn.Module):
    """N(theta; x) - eta N(theta0; x) with eta in [0, 1], starting at one (paper footnote 12)."""
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.initial = deepcopy(network).requires_grad_(False)
        self.eta = nn.Parameter(torch.ones(()))

    def forward(self, features):
        return self.network(features)-self.eta.clamp(0, 1)*self.initial(features)


def sample_states(bank, count, generator, lower, upper):
    """Random (path, date, holdings) rows; cash is the premium and is never a network input."""
    config = bank.config
    rows = torch.randint(len(bank.spot), (count,), generator=generator)
    dates = torch.randint(config.n_decisions, (count,), generator=generator)
    uniform = torch.rand((count, config.n_assets), generator=generator, dtype=lower.dtype)
    positions = lower+(upper-lower)*uniform.to(lower.device)
    rows, dates = rows.to(bank.spot.device), dates.to(bank.spot.device)
    cash = bank.liability[rows, 0].clone()
    zero = torch.zeros_like(cash)
    return rows, dates, LedgerState(cash, positions, zero, torch.zeros_like(positions), zero.clone())


def observe(bank, rows, dates, state):
    index = dates.clamp_max(bank.config.n_steps)
    return observation_from_state(bank.spot[rows, index], bank.variance[rows, index], index, state,
                                  bank.marks[rows, index], bank.config)


def _transition(config, state, targets, marks, liability, following, next_spot, next_variance,
                next_marks, next_liability, terminal):
    traded = trade_step(state, targets, marks, config)
    wealth = state.cash+(state.positions*marks).sum(-1)-liability
    next_wealth = traded.cash+(traded.positions*next_marks).sum(-1)-next_liability
    settled = liquidate(traded, next_marks, next_liability, config)["terminal_pnl"]
    next_wealth = torch.where(terminal, settled, next_wealth)
    next_observed = observation_from_state(next_spot, next_variance, following, traded, next_marks, config)
    return next_wealth-wealth, next_observed, ~terminal, traded


def transition(bank, rows, dates, state, targets):
    """One-step change of marked wealth and the next observation; gradients flow through targets.

    Wealth is cash + holdings * marks - liability. The final decision settles with
    the configured liquidation, so the rewards of a path sum to terminal P&L.
    """
    config = bank.config
    now, following = dates.clamp_max(config.n_steps), (dates+1).clamp_max(config.n_steps)
    return _transition(config, state, targets, bank.marks[rows, now], bank.liability[rows, now], following,
                       bank.spot[rows, following], bank.variance[rows, following], bank.marks[rows, following],
                       bank.liability[rows, following], dates == config.n_decisions-1)[:3]


def multistep_transition(bank, rows, dates, state, targets, steps, policy_targets):
    """Rewards over `steps` decisions along the bank's path, then the bootstrap observation.

    Paper remark "Multiple time steps" (operator T_n): the first targets are given,
    later ones come from `policy_targets(observed, positions)` with gradients kept,
    as in pathwise Deep Hedging. Rows that settle earlier stop accumulating and do
    not bootstrap; `steps` at or beyond the horizon gives pure terminal targets.
    """
    config = bank.config
    reward = torch.zeros_like(state.cash)
    alive = torch.ones_like(state.cash, dtype=torch.bool)
    for step in range(steps):
        current = dates+step
        now, following = current.clamp_max(config.n_steps), (current+1).clamp_max(config.n_steps)
        if step:
            observed = observation_from_state(bank.spot[rows, now], bank.variance[rows, now], now, state,
                                              bank.marks[rows, now], config)
            targets = policy_targets(observed, state.positions)
        step_reward, _, continuing, traded = _transition(
            config, state, targets, bank.marks[rows, now], bank.liability[rows, now], following,
            bank.spot[rows, following], bank.variance[rows, following], bank.marks[rows, following],
            bank.liability[rows, following], current == config.n_decisions-1)
        reward = reward+torch.where(alive, step_reward, torch.zeros_like(step_reward))
        state = LedgerState(torch.where(alive, traded.cash, state.cash),
                            torch.where(alive[:, None], traded.positions, state.positions),
                            state.total_cost, state.turnover, state.tickets)
        alive = alive & continuing
    following = (dates+steps).clamp_max(config.n_steps)
    next_observed = observation_from_state(bank.spot[rows, following], bank.variance[rows, following], following,
                                           state, bank.marks[rows, following], config)
    return reward, next_observed, alive


def _repeat(state, scenarios):
    return LedgerState(*(getattr(state, name).repeat_interleave(scenarios, 0) for name in
                         ("cash", "positions", "total_cost", "turnover", "tickets")))


def conditional_transition(bank, rows, dates, state, targets, scenarios, generator):
    """`transition` with `scenarios` fresh one-day continuations per state instead of the bank's path.

    Rows are repeated `scenarios` times in order. The paper has one historical next
    day per state; a simulator can average the Bellman target over conditional
    futures, which lowers the variance of the one-day P&L in actor and critic targets.
    """
    config = bank.config
    spot = bank.spot[rows, dates].double().repeat_interleave(scenarios, 0)
    variance = bank.variance[rows, dates].double().repeat_interleave(scenarios, 0)
    spot, variance = market_transition(spot, variance, config=config.market, dt=config.dt, generator=generator)
    following = (dates+1).repeat_interleave(scenarios, 0)
    next_marks, next_liability = mark_state(spot, variance, following, config)
    dtype = bank.spot.dtype
    return _transition(config, _repeat(state, scenarios), targets.repeat_interleave(scenarios, 0),
                       bank.marks[rows, dates].repeat_interleave(scenarios, 0),
                       bank.liability[rows, dates].repeat_interleave(scenarios, 0), following,
                       spot.to(dtype), variance.to(dtype), next_marks.to(dtype), next_liability.to(dtype),
                       (dates == config.n_decisions-1).repeat_interleave(scenarios, 0))[:3]


class DeepBellmanLearner:
    def __init__(self, config, *, hidden, position_range, utility, risk_aversion, learning_rate,
                 critic_learning_rate, device, dtype, scenarios=1, scenario_generator=None, steps=1,
                 aggregate="oce"):
        self.config = config
        self.scenarios, self.scenario_generator, self.steps = scenarios, scenario_generator, steps
        self.aggregate = aggregate
        lower, upper = position_bounds(config, position_range)
        self.lower = torch.tensor(lower, device=device, dtype=dtype)
        self.upper = torch.tensor(upper, device=device, dtype=dtype)
        scale = _scale(lower, upper)
        self.policy = DeepBellmanPolicy(config, hidden, position_range=position_range).to(device=device, dtype=dtype)
        fields, names = self.policy.observation_fields, self.policy.instrument_names
        self.shift = _head(fields, names, scale, 1, hidden).to(device=device, dtype=dtype)
        self.critic = BuehlerZero(_head(fields, names, scale, 1, hidden)).to(device=device, dtype=dtype)
        self.reward_scale = config.market.spot0*math.sqrt(config.market.v0*config.dt)
        self.utility = UTILITIES[utility]
        # CVaR is coherent, so its aversion is unit-free; the others rescale with money.
        self.lam = risk_aversion if utility == "cvar" else risk_aversion*self.reward_scale
        self.actor_parameters = [*self.policy.parameters(), *self.shift.parameters()]
        self.critic_parameters = [value for value in self.critic.parameters() if value.requires_grad]
        self.actor_optimizer = torch.optim.Adam(self.actor_parameters, lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=critic_learning_rate)

    def targets(self, observed, positions):
        execution = self.config.execution
        return self.policy(observed, positions, execution.holding_lower, execution.holding_upper).target_holdings

    def value(self, observed):
        """Critic value in money units."""
        return self.critic(observed).squeeze(-1)*self.reward_scale

    def outcome(self, bank, rows, dates, state, observed):
        """Continuation value plus reward in learner units, and the OCE shift y(s), per scenario."""
        targets = self.targets(observed, state.positions)
        if self.steps > 1:
            reward, following, alive = multistep_transition(bank, rows, dates, state, targets, self.steps,
                                                            self.targets)
        elif self.scenarios == 1:
            reward, following, alive = transition(bank, rows, dates, state, targets)
        else:
            reward, following, alive = conditional_transition(bank, rows, dates, state, targets,
                                                              self.scenarios, self.scenario_generator)
        shift = self.shift(observed).squeeze(-1).repeat_interleave(self.scenarios, 0)
        return alive*self.critic(following).squeeze(-1)+reward/self.reward_scale, shift

    def certainty(self, outcome, shift):
        """Per-state risk-adjusted target: the OCE integrand averaged over scenarios, or the
        closed-form entropic certainty equivalent of the scenarios."""
        if self.aggregate == "entropic":
            return entropic_certainty(outcome.reshape(-1, self.scenarios), self.lam)
        return oce(self.utility, outcome, shift, self.lam).reshape(-1, self.scenarios).mean(1)

    def update(self, bank, rows, dates, state, *, actor_steps, critic_steps):
        """Paper section 3: actor step under the previous critic, then critic regression."""
        observed = observe(bank, rows, dates, state)
        for _ in range(actor_steps):
            objective = self.certainty(*self.outcome(bank, rows, dates, state, observed)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            (-objective).backward(inputs=self.actor_parameters)
            nn.utils.clip_grad_norm_(self.actor_parameters, 5., error_if_nonfinite=True)
            self.actor_optimizer.step()
        with torch.no_grad():
            target = self.certainty(*self.outcome(bank, rows, dates, state, observed))
        for _ in range(critic_steps):
            residual = self.critic(observed).squeeze(-1)-target
            loss = residual.square().mean()
            self.critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic_parameters, 5., error_if_nonfinite=True)
            self.critic_optimizer.step()
        return objective.detach()*self.reward_scale, loss.detach().sqrt()*self.reward_scale

    def state_dict(self):
        return {name: getattr(self, name).state_dict() for name in
                ("policy", "shift", "critic", "actor_optimizer", "critic_optimizer")}

    def load_state_dict(self, state):
        for name, values in state.items():
            getattr(self, name).load_state_dict(values)


def train_deep_bellman(bank, *, seed=7, updates=8, batch_size=32, hidden=(32, 32), learning_rate=1e-3,
                       critic_learning_rate=None, utility=None, risk_aversion=None, actor_steps=1,
                       critic_steps=1, scenarios=1, steps=1, aggregate="oce", position_range=None,
                       device="cpu", progress=True, checkpoint_path=None, checkpoint_every=200,
                       resume_from=None):
    """Return a trained policy and compact metadata; no model selection occurs.

    The utility defaults to the configured objective: CVaR with 1 + lam = 1/(1 - alpha)
    for ES, or the entropic utility with the configured risk aversion. Other
    utilities default to the configured risk aversion, in the ledger's money
    units. `scenarios=1` uses the bank's own next day as the paper does; more
    scenarios simulate fresh conditional continuations per sampled state.
    `steps` selects the paper's n-step operator T_n: rewards of `steps` decisions
    by the current policy before bootstrapping. `aggregate="entropic"` replaces
    the learned OCE shift by the closed-form entropic certainty equivalent over
    the scenarios (paper section 2.1), which needs the entropic utility and more
    than one scenario. History records the actor objective, the Bellman residual
    (root mean square) and the initial-state value in money units.
    """
    config = bank.config
    if utility is None:
        utility = {"es": "cvar", "entropy": "entropy"}.get(config.risk.objective)
        if utility is None:
            raise ValueError("MSE is not a monetary utility; choose a utility explicitly")
    if utility not in UTILITIES:
        raise ValueError(f"choose a utility among {sorted(UTILITIES)}")
    if risk_aversion is None:
        risk_aversion = (config.risk.alpha/(1-config.risk.alpha) if utility == "cvar"
                         else config.risk.risk_aversion)
    if critic_learning_rate is None:
        critic_learning_rate = learning_rate
    if min(updates, batch_size, actor_steps, critic_steps, scenarios, steps, checkpoint_every, len(bank.spot)) < 1:
        raise ValueError("provide positive training work, steps, scenarios and a nonempty bank")
    if scenarios > 1 and steps > 1:
        raise ValueError("conditional scenarios apply to one-step targets only")
    if aggregate not in ("oce", "entropic"):
        raise ValueError("aggregate targets with oce or entropic")
    if aggregate == "entropic" and (utility != "entropy" or scenarios < 2):
        raise ValueError("the closed-form entropic aggregate needs the entropic utility and several scenarios")
    if min(learning_rate, critic_learning_rate, risk_aversion) <= 0:
        raise ValueError("learning rates and risk aversion must be positive")
    if any(config.execution.vector("minimum_trade", config.n_assets)
           + config.execution.vector("trade_lot", config.n_assets)):
        raise ValueError("continuous Deep Bellman training does not support minimum-trade or lot constraints")
    if scenarios > 1 and (config.time_grid.trade_at_maturity or any(
            instrument.needs_integrated_variance for instrument in (config.portfolio.liability, *config.portfolio.hedges))):
        raise ValueError("conditional scenarios need a decision before maturity and markable instruments")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; select --device cpu")
    started = time.perf_counter()
    options = dict(updates=updates, batch_size=batch_size, hidden=list(hidden), learning_rate=learning_rate,
                   critic_learning_rate=critic_learning_rate, utility=utility, risk_aversion=risk_aversion,
                   actor_steps=actor_steps, critic_steps=critic_steps, scenarios=scenarios, steps=steps,
                   aggregate=aggregate,
                   position_range=None if position_range is None else
                   [float(value) for value in ([position_range]*config.n_assets
                                               if isinstance(position_range, (int, float)) else position_range)])
    if progress:
        _report("train_start", method="deep_bellman", label="Deep Bellman Hedging", seed=seed,
                sampling_seed=seed+100003, device=str(device), workers=torch.get_num_threads(),
                config=asdict(config), options=options,
                expected_transitions=updates*batch_size*scenarios*steps*(actor_steps+1))
    torch.manual_seed(seed)
    train_device = bank_to(bank, device)
    scenario_generator = torch.Generator(device=device).manual_seed(seed+200003) if scenarios > 1 else None
    learner = DeepBellmanLearner(config, hidden=hidden, position_range=options["position_range"],
                                 utility=utility, risk_aversion=risk_aversion, learning_rate=learning_rate,
                                 critic_learning_rate=critic_learning_rate, device=device, dtype=bank.spot.dtype,
                                 scenarios=scenarios, scenario_generator=scenario_generator, steps=steps,
                                 aggregate=aggregate)
    generator = torch.Generator().manual_seed(seed+100003)
    first_step, previous_seconds, history = 0, 0., []
    if resume_from:
        saved = load_checkpoint(resume_from, method="deep_bellman", config=config)
        check_resume_options(saved, options)
        if saved["seed"] != seed or saved["training_paths"] != len(bank.spot):
            raise ValueError("resume requires the original seed and frozen training bank")
        learner.load_state_dict(saved["learner"])
        generator.set_state(saved["sampling_rng"])
        if scenario_generator is not None:
            scenario_generator.set_state(saved["scenario_rng"].to(device))
        restore_rng(saved["rng"])
        first_step, history, previous_seconds = saved["step"], saved["history"], saved["elapsed_seconds"]
    probe = bank_subset(train_device, slice(0, min(1024, len(train_device.spot))))
    initial_observed = observation(probe, 0, initial_state(probe))
    _sync(device)
    initialization_seconds = time.perf_counter()-started
    training_start = time.perf_counter()
    learner.policy.train()
    for update in range(first_step+1, updates+1):
        rows, dates, state = sample_states(train_device, batch_size, generator, learner.lower, learner.upper)
        objective, residual = learner.update(train_device, rows, dates, state,
                                             actor_steps=actor_steps, critic_steps=critic_steps)
        if update == 1 or update % 20 == 0 or update == updates:
            _sync(device)
            elapsed = time.perf_counter()-training_start
            with torch.no_grad():
                value_initial = float(learner.value(initial_observed).mean())
            record = dict(completed=update, total=updates, elapsed_seconds=elapsed,
                          updates_per_second=(update-first_step)/max(elapsed, 1e-12),
                          eta_seconds=elapsed*(updates-update)/(update-first_step),
                          actor_objective=float(objective), bellman_residual=float(residual),
                          eta=float(learner.critic.eta.detach().clamp(0, 1)), value_initial=value_initial)
            history.append(record)
            if progress:
                _report("train_progress", method="deep_bellman", seed=seed, **record)
        if checkpoint_path and due_checkpoint(update, updates, checkpoint_every):
            save_checkpoint(checkpoint_path, dict(method="deep_bellman", config=asdict(config), seed=seed,
                options=options, step=update, learner=learner.state_dict(), rng=rng_state(),
                sampling_rng=generator.get_state(), history=history, training_paths=len(bank.spot),
                scenario_rng=None if scenario_generator is None else scenario_generator.get_state().cpu(),
                elapsed_seconds=previous_seconds+time.perf_counter()-started))
    _sync(device)
    metadata = dict(label="Baseline training", method="deep_bellman", method_label="Deep Bellman Hedging",
                    source=SOURCE, seed=seed, sampling_seed=seed+100003, device=str(device), options=options,
                    config=asdict(config), observation_fields=list(learner.policy.observation_fields),
                    instrument_names=list(learner.policy.instrument_names),
                    objective=f"Bellman optimized certainty equivalent, {utility} utility, risk aversion {risk_aversion}",
                    reward_scale=learner.reward_scale, zeta=None, history=history,
                    eta=history[-1]["eta"] if history else 1.,
                    value_initial=history[-1]["value_initial"] if history else 0.,
                    actor_optimizer_steps=(updates-first_step)*actor_steps,
                    critic_optimizer_steps=(updates-first_step)*critic_steps,
                    transitions=(updates-first_step)*batch_size*scenarios*steps*(actor_steps+1),
                    initialization_seconds=initialization_seconds,
                    training_seconds=time.perf_counter()-training_start,
                    total_seconds=previous_seconds+time.perf_counter()-started,
                    resumed_from=str(resume_from) if resume_from else None, resumed_step=first_step,
                    parameter_count=sum(value.numel() for value in learner.actor_parameters))
    return learner.policy.eval(), metadata
