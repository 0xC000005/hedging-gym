"""Stochastic AlphaZero adaptation: learned PUCT search, then visit/return fitting.

The policy/value and search-improvement loop follows Szehr's public hedger:
https://github.com/plan64/minimalHedger_AlphaZero/tree/3111c378fcd17e45f94d2fc668a3aa117126ecba
(MCTS.py and Trainer.py). This is a common-environment adaptation, not a paper
reproduction. It uses an independent holding lattice, legal-action masking,
continuous-market chance nodes, and terminal expected-shortfall cost instead of
the author's reward. Chance outcomes are sampled/reused uniformly, never chosen
as profitable actions. The old research implementation's progressive-widening
kernel is retained without its HPO proposals, Gumbel variants or tree caches.
Separate actor/critic fitting, marked-wealth features, and completed greedy
continuation reanalysis repair this ES adaptation; they are not presented as
features of the source paper or as changes to the common financial environment.

Independent roots share batched pricing/inference. Tree traversal remains on
CPU, so small trees need not be faster on GPU. Larger books should supply an
explicit action table: the default Cartesian lattice grows exponentially.
"""

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from itertools import product
import math
from statistics import NormalDist
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hedging_gym import finance
from hedging_gym.gym_env import TensorHedgingEnv
from hedging_gym.evaluation import empirical_es

from .policies import _ConfiguredPolicy, _network
from .training import _report, _sync
from .checkpoints import (check_resume_options, due_checkpoint, load_checkpoint,
                          restore_rng, rng_state, save_checkpoint)


def holding_grid(config, points=3):
    """Configured absolute holdings; zero and legal lot multiples are included."""
    if points < 2:
        raise ValueError("the holding lattice requires at least two axis points")
    if config.execution.holding_lower is None or config.execution.holding_upper is None:
        raise ValueError("discrete search requires an explicitly bounded action grid")
    axes = []
    for lower, upper, lot in zip(config.execution.vector("holding_lower", config.n_assets),
                                 config.execution.vector("holding_upper", config.n_assets),
                                 config.execution.vector("trade_lot", config.n_assets)):
        axis = np.linspace(lower, upper, points)
        if lot:
            # Keep floating representations such as 3*.1 inside a .3 bound;
            # the shared lot check below removes any genuinely off-lot endpoint.
            axis = np.clip(np.round(axis / lot) * lot, lower, upper)
        axis = np.unique(np.r_[axis, 0.])
        axes.append(axis[(axis >= lower) & (axis <= upper)])
    grid = torch.tensor(list(product(*axes)), dtype=torch.float64)
    legal_inventory = finance.feasible_targets(torch.zeros_like(grid), grid, config, liquidating=True).all(-1)
    return grid[legal_inventory]


class AlphaZeroPolicy(_ConfiguredPolicy):
    """Separate actor and critic; nonnegative excess over the global RU threshold.

    The global ES threshold is observable and fixed during an entire self-play
    block/search. It is calibrated from completed greedy training losses, never by
    re-optimizing a different conditional tail objective at each tree node.
    """

    def __init__(self, config, hidden=(64, 64), *, targets=None, grid_points=3):
        super().__init__(config)
        if config.risk.objective != "es":
            raise ValueError("this RU-value AlphaZero adapter is ES-only; use source_alphazero for MSE")
        if config.time_grid.trade_at_maturity or any(instrument.needs_integrated_variance
                for instrument in (config.portfolio.liability, *config.portfolio.hedges)):
            raise ValueError("this planner is not qualified for terminal trades or path-dependent instruments")
        targets = (holding_grid(config, grid_points) if targets is None
                   else torch.as_tensor(targets, dtype=torch.float64))
        lo = targets.new_tensor(config.execution.vector("holding_lower", config.n_assets))
        hi = targets.new_tensor(config.execution.vector("holding_upper", config.n_assets))
        if (targets.ndim != 2 or targets.shape[1] != config.n_assets or not len(targets)
                or not torch.isfinite(targets).all() or ((targets < lo) | (targets > hi)).any()):
            raise ValueError("targets must be a finite [actions,assets] table within holding limits")
        # Preserve decimal lot sizes until the caller chooses its ledger dtype.
        self.register_buffer("targets", torch.unique(targets, dim=0))
        self.register_buffer("zeta", torch.zeros(()))
        self.register_buffer("value_scale", torch.tensor(max(
            config.market.spot0 * math.sqrt(config.market.v0 * config.time_grid.horizon), 1e-6)))
        # The deployed actor is fixed during threshold calibration/value fitting.
        # In particular, changing zeta must not change its actions underneath the
        # completed-rollout targets. Separate fitting also avoids sparse RU tails
        # dominating the policy gradients through a shared representation.
        self.register_buffer("search_scale", self.value_scale.clone())
        self.network = _network(self.feature_dim, len(self.targets) + 1, hidden)
        self.value_network = _network(self.feature_dim + 2, 1, hidden)

    def forward(self, features):
        return self.network(features[:, :-1]), F.softplus(self.value_network(self.value_features(features))[:, 0])

    def value_features(self, features):
        """Expose cash plus marked holdings exactly, in the critic's risk units.

        A self-financing trade exchanges cash for holdings. Learning that large
        cancellation from raw coordinates obscures small hedge-loss differences.
        This adds an accounting identity, not a new price or reward approximation.
        """
        fields = self.observation_fields
        positions = features[:, [fields.index(f"{name}_position") for name in self.instrument_names]]
        marks = features[:, [fields.index(f"{name}_mid") for name in self.instrument_names]]
        wealth = features[:, fields.index("cash")] + (positions*marks).sum(-1)
        # Observation cash/marks/zeta are divided by spot0; search_scale is money.
        relative_scale = self.search_scale / features[:, fields.index("spot0")]
        normalized = torch.cat((features[:, :-1], (features[:, -1]/relative_scale)[:, None],
                                (wealth/relative_scale)[:, None]), dim=1)
        return normalized

    def features(self, observed, *, spot0):
        return torch.cat((observed, self.zeta.expand(len(observed), 1) / spot0), dim=1)

    def candidates(self, positions, config):
        grid = self.targets.to(positions)[None].expand(len(positions), -1, -1)
        candidates = torch.cat((grid, positions[:, None]), dim=1)
        legal = finance.feasible_targets(positions[:, None], candidates, config).all(-1)
        # Absolute grid actions keep their identity when inventory is unchanged,
        # as in the donor. Mask the redundant HOLD, not the matching grid action;
        # otherwise a persistent grid preference is forced into needless trades.
        legal[:, -1] &= ~(grid == positions[:, None]).all(-1).any(-1)
        return candidates, legal


@dataclass
class _State:
    spot: torch.Tensor
    variance: torch.Tensor
    time_index: int
    ledger: finance.LedgerState
    marks: torch.Tensor
    terminal_cost: float | None = None
    terminal_loss: float | None = None


@dataclass
class _Node:
    state: object
    prior: np.ndarray | None = None
    value: float = 0.
    visits: np.ndarray | None = None
    sums: np.ndarray | None = None
    children: dict = field(default_factory=dict)


def _search_steps(state, model, rng, *, simulations, c_puct, root_noise):
    """Coroutine: yield batched leaf/transition work, preserve sequential backup.

    sqrt(N) chance widening admits increasingly many independent market draws;
    revisits sample uniformly from those draws. This trades finite-budget bias
    for deeper look-ahead, without assuming the market is a second opponent.
    """
    work = dict(simulations=simulations, transition_samples=0, network_rows=0,
                terminal_evaluations=0, maximum_depth=0)

    def expand(node):
        cost = model.terminal_cost(node.state)
        if cost is not None:
            node.value = cost
            work["terminal_evaluations"] += 1
            return cost
        prior, value = yield ("leaf", node.state)
        node.prior, node.value = prior, float(value)
        node.visits = np.zeros(len(prior), dtype=np.int64)
        node.sums = np.zeros(len(prior), dtype=np.float64)
        work["network_rows"] += 1
        return node.value

    root = _Node(state)
    yield from expand(root)
    if root.prior is None:
        raise ValueError("AlphaZero search needs a nonterminal state")
    if root_noise:
        legal = root.prior > 0
        noise = rng.dirichlet(np.full(int(legal.sum()), .3))
        root.prior[legal] = (1-root_noise)*root.prior[legal] + root_noise*noise

    def visit(node, depth):
        work["maximum_depth"] = max(work["maximum_depth"], depth)
        if node.prior is None:
            return node.value
        q = np.divide(node.sums, node.visits, out=np.full(len(node.prior), node.value),
                      where=node.visits > 0)
        score = -q + c_puct * node.prior * np.sqrt(1+node.visits.sum()) / (1+node.visits)
        score[node.prior == 0] = -np.inf
        action = int(rng.choice(np.flatnonzero(score == score.max())))
        children = node.children.setdefault(action, [])
        limit = max(1, math.ceil(math.sqrt(1+node.visits[action])))
        if len(children) < limit:
            child = _Node((yield ("transition", node.state, action, rng)))
            children.append(child)
            work["transition_samples"] += 1
            value = yield from expand(child)
            work["maximum_depth"] = max(work["maximum_depth"], depth+1)
        else:
            # Crucial stochastic boundary: do not maximize over market samples.
            child = children[int(rng.integers(len(children)))]
            value = yield from visit(child, depth+1)
        node.visits[action] += 1
        node.sums[action] += value
        return value

    for _ in range(simulations):
        yield from visit(root, 0)
    return dict(policy=root.visits/root.visits.sum(), visits=root.visits.copy(),
                value=float(root.sums.sum()/root.visits.sum()), work=work)


def stochastic_search_batch(states, model, rngs, *, simulations=32, c_puct=1., root_noise=0.):
    """Batch independent search roots; no concurrent stale backup within a tree.

    The small model boundary also permits an exact finite-problem test of the
    search's chance semantics without substituting that problem for finance.
    """
    if simulations < 1 or c_puct < 0 or not 0 <= root_noise <= 1:
        raise ValueError("positive search work and valid exploration controls required")
    if not states or len(states) != len(rngs):
        raise ValueError("one random stream per nonempty search state required")
    generators = [_search_steps(state, model, rng, simulations=simulations,
                    c_puct=c_puct, root_noise=root_noise) for state, rng in zip(states, rngs)]
    pending = {i: next(generator) for i, generator in enumerate(generators)}
    results = [None] * len(states)
    while pending:
        groups = defaultdict(list)
        for index, request in pending.items():
            groups[request[0]].append(index)
        responses = {}
        for kind, indices in groups.items():
            requests = [pending[index] for index in indices]
            states = [request[1] for request in requests]
            if kind == "leaf":
                outputs = model.evaluate(states)
            else:
                outputs = model.advance(states, [r[2] for r in requests], [r[3] for r in requests])
            responses.update(zip(indices, outputs))
        for index in list(pending):
            try:
                pending[index] = generators[index].send(responses[index])
            except StopIteration as finished:
                results[index] = finished.value
                del pending[index]
    return results


_LEDGER_FIELDS = ("cash", "positions", "total_cost", "turnover", "tickets")


def _ledger_slice(ledger, index):
    return finance.LedgerState(**{key: getattr(ledger, key)[index:index+1] for key in _LEDGER_FIELDS})


class _FinanceSearch:
    def __init__(self, policy, config):
        policy.check_config(config)
        self.policy, self.config = policy, config

    def roots(self, observed, ledger, time_index):
        spot, variance = finance.decode_market_observation(observed, self.config)
        fields = finance.observation_fields(self.config)
        indices = [fields.index(f"{name}_mid") for name in self.policy.instrument_names]
        marks = observed[:, indices] * self.config.market.spot0
        dates = ([time_index] * len(spot) if isinstance(time_index, int) else time_index.tolist())
        return [_State(spot[i:i+1], variance[i:i+1], dates[i],
                       _ledger_slice(ledger, i), marks[i:i+1]) for i in range(len(spot))]

    def observed_roots(self, observed):
        """Restore the cash/holdings sufficient state for terminal-loss rollouts.

        Cumulative diagnostic counters do not enter future cash accounting;
        already-paid costs are included in the observed cash balance.
        """
        fields = finance.observation_fields(self.config)
        cash = observed[:, fields.index("cash")] * self.config.market.spot0
        positions = observed[:, [fields.index(f"{name}_position")
                                  for name in self.policy.instrument_names]]
        zero = torch.zeros_like(cash)
        ledger = finance.LedgerState(cash, positions, zero, torch.zeros_like(positions), zero)
        dates = (observed[:, fields.index("time_fraction")] * self.config.n_steps).round().long()
        return self.roots(observed, ledger, dates)

    def _join(self, states):
        ledger = finance.LedgerState(**{key: torch.cat([getattr(s.ledger, key) for s in states])
                                       for key in _LEDGER_FIELDS})
        spot, variance, marks = [torch.cat([getattr(s, key) for s in states])
                                 for key in ("spot", "variance", "marks")]
        times = torch.tensor([s.time_index for s in states], device=spot.device)
        return spot, variance, times, ledger, marks

    def terminal_cost(self, state):
        return state.terminal_cost

    @torch.no_grad()
    def evaluate(self, states):
        spot, variance, times, ledger, marks = self._join(states)
        observed = finance.observation_from_state(spot, variance, times, ledger, marks, self.config)
        _, legal = self.policy.candidates(ledger.positions, self.config)
        features = self.policy.features(observed, spot0=self.config.market.spot0)
        logits, value = self.policy(features)
        probabilities = logits.masked_fill(~legal, -torch.inf).softmax(-1).cpu().numpy()
        costs = (self.policy.zeta + self.policy.value_scale*value).cpu().numpy()
        return list(zip(probabilities, costs))

    @torch.no_grad()
    def advance(self, states, actions, rngs):
        spot, variance, times, ledger, marks = self._join(states)
        candidates, _ = self.policy.candidates(ledger.positions, self.config)
        targets = candidates[torch.arange(len(states), device=spot.device),
                             torch.tensor(actions, device=spot.device)]
        ledger = finance.trade_step(ledger, targets, marks, self.config)
        ledger_dtype = spot.dtype
        shocks = torch.as_tensor(np.stack([finance.market_shocks(self.config.market, rng,
                                  dt=self.config.dt) for rng in rngs]), dtype=torch.float64, device=spot.device)
        spot, variance = finance.transition(spot.double(), variance.double(), shocks,
                                            self.config.market, dt=self.config.dt)
        times = times + 1
        marks, liability = finance.mark_state(spot, variance, times, self.config)
        spot, variance, marks, liability = (value.to(ledger_dtype)
                                           for value in (spot, variance, marks, liability))
        terminal = times == self.config.n_steps
        costs = torch.zeros_like(spot)
        raw_losses = torch.zeros_like(spot)
        if terminal.any():
            final = finance.LedgerState(**{key: getattr(ledger, key)[terminal] for key in _LEDGER_FIELDS})
            losses = finance.liquidate(final, marks[terminal], liability[terminal], self.config)["terminal_loss"]
            raw_losses[terminal] = losses
            costs[terminal] = self.config.risk.loss(losses, self.policy.zeta)
        dates, terminal, costs = times.tolist(), terminal.tolist(), costs.tolist()
        raw_losses = raw_losses.tolist()
        return [_State(spot[i:i+1], variance[i:i+1], dates[i], _ledger_slice(ledger, i),
                       marks[i:i+1], costs[i] if terminal[i] else None,
                       raw_losses[i] if terminal[i] else None) for i in range(len(states))]


@torch.no_grad()
def _completed_rollouts(model, states, seeds, *, first_actions=None):
    """Fresh conditional paths through settlement under the frozen greedy actor.

    Repeated seeds provide common market draws across counterfactual actions;
    none of the supplied states contains a realized future market path.
    """
    rngs = [np.random.default_rng(int(seed)) for seed in seeds]
    losses = np.empty(len(states), dtype=np.float64)
    active = list(range(len(states)))
    work = dict(transition_samples=0, network_rows=0, terminal_evaluations=0)
    first_values = None
    while active:
        if first_actions is not None:
            actions, first_actions = first_actions, None
        else:
            evaluated = model.evaluate(states)
            actions = [int(prior.argmax()) for prior, _ in evaluated]
            work["network_rows"] += len(states)
        states = model.advance(states, actions, [rngs[i] for i in active])
        work["transition_samples"] += len(states)
        if first_values is None:
            first_values = np.empty(len(states))
            nonterminal = [i for i, state in enumerate(states) if state.terminal_loss is None]
            estimates = model.evaluate([states[i] for i in nonterminal]) if nonterminal else []
            work["network_rows"] += len(nonterminal)
            for i, (_, value) in zip(nonterminal, estimates):
                first_values[i] = value
            for i, state in enumerate(states):
                if state.terminal_loss is not None:
                    first_values[i] = state.terminal_cost
        remaining, next_states = [], []
        for index, state in zip(active, states):
            if state.terminal_loss is None:
                remaining.append(index)
                next_states.append(state)
            else:
                losses[index] = state.terminal_loss
                work["terminal_evaluations"] += 1
        active, states = remaining, next_states
    return losses, first_values, work


@torch.no_grad()
def _calibration_rollout(policy, bank):
    """Training-only greedy episodes; the actor does not depend on zeta."""
    env = TensorHedgingEnv(bank)
    observed, rows = env.reset(), []
    controller = alphazero_controller(policy, simulations=0)
    for date in range(bank.config.n_steps):
        rows.append(observed)
        observed, _, _, _, info = env.step(controller(observed, env.state, date, bank.config))
    return torch.cat(rows), info["terminal_loss"]


@torch.no_grad()
def completed_rollout_action_diagnostic(policy, config, *, samples=128, seed=70001):
    """Bounded initial-action check of the critic against completed continuations.

    This conditional greedy-policy diagnostic is not a final ES comparison or
    a proof of calibration at every state. It runs before search qualification.
    """
    if samples < 2:
        raise ValueError("action diagnostics need at least two paired market draws")
    started = time.perf_counter()
    policy.eval()
    reference = policy.targets.new_full((1,), config.market.spot0)
    variance = torch.full_like(reference, config.market.v0)
    marks, _ = finance.mark_state(reference, variance, 0, config)
    ledger = finance.initial_ledger(reference, config)
    root = _State(reference, variance, 0, ledger, marks)
    model = _FinanceSearch(policy, config)
    candidates, legal = policy.candidates(ledger.positions, config)
    actions = torch.nonzero(legal[0]).flatten().tolist()
    seeds = np.random.default_rng(seed).integers(2**63-1, size=samples)
    raw, predicted, work = _completed_rollouts(model, [root] * (len(actions)*samples),
        np.tile(seeds, len(actions)), first_actions=np.repeat(actions, samples).tolist())
    raw = torch.as_tensor(raw.reshape(len(actions), samples))
    ru = config.risk.loss(raw, float(policy.zeta)).numpy()
    predicted = predicted.reshape(len(actions), samples).mean(-1)
    actual = ru.mean(-1)
    selected, best = int(predicted.argmin()), int(actual.argmin())
    paired = ru[selected] - ru[best]
    se = float(paired.std(ddof=1)/math.sqrt(samples))
    # Both actions are selected using these draws. Cover all action pairs,
    # rather than treating that selected comparison as fixed in advance.
    # This normal-approximation screen is not a tail-risk certificate.
    pairs = max(1, len(actions)*(len(actions)-1)//2)
    critical = NormalDist().inv_cdf(1-.05/(2*pairs))
    result = dict(samples=samples, seed=seed, continuation="frozen_greedy_policy",
        zeta=float(policy.zeta), action_indices=actions,
        action_targets=candidates[0, actions].cpu().tolist(),
        predicted_ru=predicted.tolist(), completed_ru=actual.tolist(),
        completed_es=[empirical_es(row, config.risk.alpha) for row in raw],
        critic_selected_action=actions[selected], rollout_best_action=actions[best],
        paired_regret=float(paired.mean()), paired_regret_se=se,
        paired_regret_critical_z=critical,
        action_ordering_screen_failed=bool(paired.mean() > critical*se + 1e-8),
        mean_absolute_value_error=float(np.abs(predicted-actual).mean()),
        action_value_range=float(np.ptp(actual)),
        seconds=time.perf_counter()-started, work=work,
        scope="training_diagnostic_not_final_test_or_global_value_certificate")
    return result, dict(terminal_loss=raw, predicted_ru=torch.as_tensor(predicted))


def alphazero_controller(policy, *, simulations=32, seed=30001, c_puct=1., progress=False):
    """Visit-greedy search controller; simulations=0 is the learned-policy ablation.

    Seed/batch order are part of the search budget contract. A new controller
    restarts its random streams; reusing it deliberately continues those streams.
    """
    stream = np.random.default_rng(seed)
    work = dict(decision_calls=0, transition_samples=0, network_rows=0,
                terminal_evaluations=0, maximum_depth=0)
    started = time.perf_counter()

    @torch.no_grad()
    def control(observed, ledger, time_index, config):
        policy.check_config(config)
        policy.eval()
        candidates, legal = policy.candidates(ledger.positions, config)
        if simulations == 0:
            # Policy-only deployment needs no critic or wealth-feature work.
            # Actor inputs are the observed state, without the value threshold.
            logits = policy.network(observed)
            actions = logits.masked_fill(~legal, -torch.inf).argmax(-1)
            work["network_rows"] += len(observed)
        else:
            model = _FinanceSearch(policy, config)
            rngs = [np.random.default_rng(int(value)) for value in stream.integers(2**63-1, size=len(observed))]
            results = stochastic_search_batch(model.roots(observed, ledger, time_index), model, rngs,
                        simulations=simulations, c_puct=c_puct*float(policy.search_scale))
            actions = torch.tensor([int(result["visits"].argmax()) for result in results], device=observed.device)
            for result in results:
                for key in ("transition_samples", "network_rows", "terminal_evaluations"):
                    work[key] += result["work"][key]
                work["maximum_depth"] = max(work["maximum_depth"], result["work"]["maximum_depth"])
        work["decision_calls"] += 1
        if progress and (time_index == 0 or (time_index+1) % 5 == 0 or time_index+1 == config.n_steps):
            _report("az_search_progress", date=time_index+1, dates=config.n_steps,
                    roots=len(observed), simulations=simulations,
                    elapsed_seconds=time.perf_counter()-started, **work)
        return candidates[torch.arange(len(observed), device=observed.device), actions].clone()

    control.action_selection = "visit_greedy_stochastic_search" if simulations else "greedy_policy"
    control.work = work
    return control


def train_alphazero(train_bank, *, seed=7, updates=8, batch_size=16, hidden=(32, 32),
                   learning_rate=1e-3, device="cpu", simulations=32, grid_points=3,
                   targets=None, gradient_steps=4, replay_batches=4, c_puct=1., progress=True,
                   calibration_paths=1024, reanalysis_states=128, reanalysis_samples=32,
                   value_gradient_steps=128,
                   checkpoint_path=None, checkpoint_every=4, resume_from=None):
    """Full single-agent search/improvement loop on the supplied training bank.

    Each iteration collects complete episodes under visit-sampled search,
    then fits policy targets from visit frequencies. Raw terminal losses remain
    in replay: censoring them at a previous threshold would prevent relabeling.
    After actor fitting, training-only greedy rollouts calibrate the global ES
    threshold. Fresh conditional greedy continuations reanalyse visited states,
    and a separate critic is fitted at that threshold before checkpointing.
    Search leaves therefore estimate the frozen greedy actor's terminal RU;
    search is a separately evaluated policy improvement, not the critic target.
    Exogenous training paths are used only for live episode steps; planning
    generates independent conditional paths via the same financial model.
    ``updates`` counts self-play batches, not neural optimizer steps.
    Checkpoints contain the optimizer, rolling replay, ES threshold and every
    random stream; resuming extends self-play without discarding learned state.
    """
    if min(updates, batch_size, simulations, gradient_steps, replay_batches, checkpoint_every,
           calibration_paths, reanalysis_states, reanalysis_samples, value_gradient_steps) < 1:
        raise ValueError("training/search/replay work must be positive")
    total_started = time.perf_counter()
    config, device = train_bank.config, torch.device(device)
    torch.manual_seed(seed)
    policy = AlphaZeroPolicy(config, hidden, targets=targets, grid_points=grid_points).to(
        device=device, dtype=train_bank.spot.dtype)
    bank = finance.bank_to(train_bank, device)
    with torch.no_grad():
        # Training-only no-trade losses initialize the common global threshold.
        policy.zeta.copy_(torch.quantile(bank.liability[:, -1]-bank.liability[:, 0], config.risk.alpha))
    optimizer = torch.optim.Adam(policy.network.parameters(), lr=learning_rate)
    value_optimizer = torch.optim.Adam(policy.value_network.parameters(), lr=learning_rate)
    index_rng = torch.Generator().manual_seed(seed+100003)
    action_rng = torch.Generator(device=device).manual_seed(seed+200003)
    search_rng = np.random.default_rng(seed+300003)
    rollout_rng = np.random.default_rng(seed+500003)
    replay, history = deque(maxlen=replay_batches), []
    value_replay = None
    options = dict(updates=updates, batch_size=batch_size, simulations=simulations,
                   gradient_steps=gradient_steps, replay_batches=replay_batches,
                   hidden=list(hidden), grid_points=grid_points, learning_rate=learning_rate,
                   c_puct=c_puct, action_targets=policy.targets.cpu().tolist(),
                   action_encoding="absolute_grid_with_off_grid_hold",
                   value_target_version="raw_loss_greedy_reanalysis_v3",
                   calibration_paths=calibration_paths, reanalysis_states=reanalysis_states,
                   reanalysis_samples=reanalysis_samples, value_gradient_steps=value_gradient_steps,
                   value_continuation="frozen_greedy_policy")
    start_step, previous_seconds = 0, 0.
    work = dict(transition_samples=0, network_rows=0, terminal_evaluations=0, maximum_depth=0)
    refit_work = dict(transition_samples=0, network_rows=0, terminal_evaluations=0,
                     calibration_episodes=0, calibration_network_rows=0)
    if resume_from is not None:
        saved = load_checkpoint(resume_from, method="alphazero", config=config)
        check_resume_options(saved, options)
        if saved["seed"] != seed or saved["step"] > updates:
            raise ValueError("resume requires the saved seed and at least its completed updates")
        policy.load_state_dict(saved["policy"])
        optimizer.load_state_dict(saved["optimizer"])
        value_optimizer.load_state_dict(saved["value_optimizer"])
        replay.extend(tuple(value.to(device) for value in block) for block in saved["replay"])
        history, work = saved["history"], saved["work"]
        index_rng.set_state(saved["index_rng"])
        action_rng.set_state(saved["action_rng"])
        search_rng.bit_generator.state = saved["search_rng"]
        rollout_rng.bit_generator.state = saved["rollout_rng"]
        value_replay, refit_work = saved["value_replay"], saved["refit_work"]
        restore_rng(saved["rng"])
        start_step, previous_seconds = saved["step"], saved["training_seconds"]
    started = time.perf_counter()
    if progress:
        _report("train_start", method="alphazero", seed=seed, device=str(device),
                updates=updates, batch_size=batch_size, simulations=simulations,
                dates=config.n_steps, actions=len(policy.targets)+1,
                expected_root_searches=updates*batch_size*config.n_steps, resumed_step=start_step,
                calibration_paths=min(calibration_paths, len(bank.spot)),
                conditional_rollouts_per_update=reanalysis_states*reanalysis_samples,
                value_gradient_steps=value_gradient_steps)
    for update in range(start_step+1, updates+1):
        indices = torch.randint(len(bank.spot), (batch_size,), generator=index_rng).to(device)
        env = TensorHedgingEnv(finance.bank_subset(bank, indices))
        observed = env.reset()
        features, target_policies, masks = [], [], []
        with torch.no_grad():
            fixed_zeta = policy.zeta.clone()
            model = _FinanceSearch(policy, config)
            policy.eval()
            for date in range(config.n_steps):
                rngs = [np.random.default_rng(int(value)) for value in search_rng.integers(2**63-1, size=batch_size)]
                result = stochastic_search_batch(model.roots(observed, env.state, date), model, rngs,
                        simulations=simulations, c_puct=c_puct*float(policy.search_scale), root_noise=.25)
                targets_pi = observed.new_tensor(np.stack([item["policy"] for item in result]))
                candidates, legal = policy.candidates(env.state.positions, config)
                actions = torch.multinomial(targets_pi, 1, generator=action_rng).squeeze(-1)
                features.append(policy.features(observed, spot0=config.market.spot0).clone())
                target_policies.append(targets_pi)
                masks.append(legal)
                for item in result:
                    for key in ("transition_samples", "network_rows", "terminal_evaluations"):
                        work[key] += item["work"][key]
                    work["maximum_depth"] = max(work["maximum_depth"], item["work"]["maximum_depth"])
                observed, _, _, _, info = env.step(candidates[torch.arange(batch_size, device=device), actions])
                if progress and (date == 0 or (date+1) % 5 == 0 or date+1 == config.n_steps):
                    completed = (update-start_step-1)*config.n_steps + date+1
                    elapsed = time.perf_counter()-started
                    _report("az_self_play", update=update, updates=updates, date=date+1,
                            dates=config.n_steps, elapsed_seconds=elapsed,
                            eta_seconds=elapsed*((updates-start_step)*config.n_steps-completed)/completed)
            terminal_loss = info["terminal_loss"]
            replay.append((torch.cat(features), torch.cat(target_policies), torch.cat(masks),
                           terminal_loss.repeat(config.n_steps).clone()))
        x, pi, legal, _ = [torch.cat([block[column] for block in replay]) for column in range(4)]
        policy.train()
        for _ in range(gradient_steps):
            logits = policy.network(x[:, :-1])
            policy_loss = F.cross_entropy(logits.masked_fill(~legal, torch.finfo(logits.dtype).min), pi)
            optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.network.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        policy.eval()
        refit_started = time.perf_counter()
        with torch.no_grad():
            calibration_bank = finance.bank_subset(bank, slice(0, calibration_paths))
            calibration_rows, calibration_losses = _calibration_rollout(policy, calibration_bank)
            policy.zeta.copy_(torch.quantile(calibration_losses, config.risk.alpha))
            refit_work["calibration_episodes"] += len(calibration_losses)
            refit_work["calibration_network_rows"] += len(calibration_rows)
            # Mix deployed-actor states, search exploration, and explicit
            # counterfactual successors. Reanalyse under the SAME frozen actor;
            # old exploratory terminal costs are not current-policy value labels.
            model = _FinanceSearch(policy, config)
            initial_root = model.observed_roots(calibration_rows[:1])[0]
            _, initial_legal = policy.candidates(initial_root.ledger.positions, config)
            legal_actions = torch.nonzero(initial_legal[0]).flatten()
            n_actions = min(len(legal_actions), reanalysis_states//2)
            selected_actions = torch.randperm(len(legal_actions), generator=index_rng)[:n_actions]
            action_ids = legal_actions[selected_actions.to(device)].tolist()
            # Counterfactual successor states are where PUCT queries the critic.
            # Include them explicitly instead of hoping live episodes cover them.
            common_seed = int(rollout_rng.integers(2**63-1))
            counterfactual = model.advance([initial_root]*n_actions, action_ids,
                [np.random.default_rng(common_seed) for _ in range(n_actions)]) if n_actions else []
            counterfactual = [state for state in counterfactual if state.terminal_loss is None]
            n_live = (reanalysis_states-len(counterfactual)+1)//2
            live = torch.randint(len(calibration_rows), (n_live,), generator=index_rng).to(device)
            old = torch.randint(len(x), (reanalysis_states-len(counterfactual)-n_live,), generator=index_rng).to(device)
            observed_value = torch.cat((calibration_rows[live], x[old, :-1]))
            roots = model.observed_roots(observed_value)
            if counterfactual:
                spot, variance, dates, ledger, marks = model._join(counterfactual)
                observed_value = torch.cat((observed_value,
                    finance.observation_from_state(spot, variance, dates, ledger, marks, config)))
                roots.extend(counterfactual)
            states = [state for state in roots for _ in range(reanalysis_samples)]
            rollout_seeds = rollout_rng.integers(2**63-1, size=len(states))
            if counterfactual:
                # Pair future market draws across those initial-action alternatives.
                rollout_seeds[-len(counterfactual)*reanalysis_samples:] = np.tile(
                    rollout_rng.integers(2**63-1, size=reanalysis_samples), len(counterfactual))
            raw, _, counts = _completed_rollouts(model, states, rollout_seeds)
            counts["transition_samples"] += n_actions
            counts["terminal_evaluations"] += n_actions-len(counterfactual)
            raw = observed_value.new_tensor(raw).reshape(reanalysis_states, reanalysis_samples)
            excess = (config.risk.loss(raw, policy.zeta)-policy.zeta).mean(-1)
            # Positive scaling changes neither RU nor action rankings. Exploration
            # retains its independent market-unit scale, so refitting cannot
            # silently change the declared PUCT exploration budget.
            policy.value_scale.copy_(excess.square().mean().sqrt().clamp_min(policy.search_scale))
            value_x = policy.features(observed_value, spot0=config.market.spot0)
            value_target = excess/policy.value_scale
            value_replay = (observed_value.detach().cpu(), raw.detach().cpu())
            for key, count in counts.items():
                refit_work[key] += count
        for _ in range(value_gradient_steps):
            values = F.softplus(policy.value_network(policy.value_features(value_x))[:, 0])
            value_loss = F.mse_loss(values, value_target)
            value_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.value_network.parameters(), 5., error_if_nonfinite=True)
            value_optimizer.step()
        policy.eval()
        if progress:
            _report("az_value_refit", update=update, zeta=float(policy.zeta),
                    calibration_es=empirical_es(calibration_losses, config.risk.alpha),
                    raw_exceedance=float((raw > policy.zeta).float().mean()),
                    conditional_rollouts=len(states), seconds=time.perf_counter()-refit_started)
        _sync(device)
        segment_elapsed = time.perf_counter()-started
        elapsed = previous_seconds+segment_elapsed
        record = dict(completed=update, total=updates, elapsed_seconds=elapsed,
                      eta_seconds=segment_elapsed*(updates-update)/(update-start_step), policy_loss=float(policy_loss.detach()),
                      value_loss=float(value_loss.detach()), zeta=float(policy.zeta),
                      self_play_zeta=float(fixed_zeta),
                      calibration_es=empirical_es(calibration_losses, config.risk.alpha),
                      calibration_exceedance=float((calibration_losses > policy.zeta).float().mean()),
                      value_scale=float(policy.value_scale), refit_work=refit_work.copy(), **work)
        history.append(record)
        if progress:
            _report("train_progress", method="alphazero", **record)
        if checkpoint_path is not None and due_checkpoint(update, updates, checkpoint_every):
            save_checkpoint(checkpoint_path, dict(method="alphazero", config=asdict(config),
                options=options, seed=seed, step=update, policy=policy.state_dict(),
                optimizer=optimizer.state_dict(), replay=list(replay), history=history,
                value_optimizer=value_optimizer.state_dict(), value_replay=value_replay,
                refit_work=refit_work,
                work=work, index_rng=index_rng.get_state(), action_rng=action_rng.get_state(),
                search_rng=search_rng.bit_generator.state, rollout_rng=rollout_rng.bit_generator.state,
                rng=rng_state(), training_seconds=elapsed))
    policy.eval()
    return policy, dict(method="alphazero", method_label="Stochastic AlphaZero adaptation",
            source_repo="plan64/minimalHedger_AlphaZero", source_commit="3111c378fcd17e45f94d2fc668a3aa117126ecba",
            classification="common_environment_adaptation_not_paper_reproduction", seed=seed,
            config=asdict(config), device=str(device), zeta=float(policy.zeta), history=history,
            initialization_seconds=started-total_started,
            training_seconds=previous_seconds+time.perf_counter()-started,
            total_seconds=previous_seconds+time.perf_counter()-total_started,
            options=options, resumed_step=start_step,
            action_targets=policy.targets.cpu().tolist(), hold_action=len(policy.targets),
            expected_episode_rollouts=updates*batch_size, optimizer_steps=updates*gradient_steps,
            optimizer_steps_scope="actor_only", value_optimizer_steps=updates*value_gradient_steps,
            total_optimizer_steps=updates*(gradient_steps+value_gradient_steps),
            parameter_count=sum(p.numel() for p in policy.parameters()), work=work, refit_work=refit_work)
