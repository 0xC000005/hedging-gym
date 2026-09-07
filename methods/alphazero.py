"""Stochastic AlphaZero adaptation: learned PUCT search, then visit/return fitting.

The policy/value and search-improvement loop follows Szehr's public hedger:
https://github.com/plan64/minimalHedger_AlphaZero/tree/3111c378fcd17e45f94d2fc668a3aa117126ecba
(MCTS.py and Trainer.py). This is a common-environment adaptation, not a paper
reproduction. It uses an independent holding lattice, legal-action masking,
continuous-market chance nodes, and terminal expected-shortfall cost instead of
the author's reward. Chance outcomes are sampled/reused uniformly, never chosen
as profitable actions. The old research implementation's progressive-widening
kernel is retained without its HPO proposals, Gumbel variants or tree caches.

Independent roots share batched pricing/inference. Tree traversal remains on
CPU, so small trees need not be faster on GPU. Larger books should supply an
explicit action table: the default Cartesian lattice grows exponentially.
"""

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from itertools import product
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hedging_gym import finance
from hedging_gym.gym_env import TensorHedgingEnv

from .policies import _ConfiguredPolicy, _network
from .training import _report, _sync
from .checkpoints import (check_resume_options, due_checkpoint, load_checkpoint,
                          restore_rng, rng_state, save_checkpoint)


def holding_grid(config, points=3):
    """Configured absolute holdings; zero and legal lot multiples are included."""
    if points < 2:
        raise ValueError("the holding lattice requires at least two axis points")
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
    """Shared representation; nonnegative excess over the global RU threshold.

    The global ES threshold is observable and fixed during an entire self-play
    block/search. It is updated only from completed training losses, never by
    re-optimizing a different conditional tail objective at each tree node.
    """

    def __init__(self, config, hidden=(64, 64), *, targets=None, grid_points=3):
        super().__init__(config)
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
        self.network = _network(self.feature_dim + 1, len(self.targets) + 2, hidden)

    def forward(self, features):
        output = self.network(features)
        return output[:, :-1], F.softplus(output[:, -1])

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
        return [_State(spot[i:i+1], variance[i:i+1], time_index,
                       _ledger_slice(ledger, i), marks[i:i+1]) for i in range(len(spot))]

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
        if terminal.any():
            final = finance.LedgerState(**{key: getattr(ledger, key)[terminal] for key in _LEDGER_FIELDS})
            losses = finance.liquidate(final, marks[terminal], liability[terminal], self.config)["terminal_loss"]
            costs[terminal] = self.config.risk.loss(losses, self.policy.zeta)
        dates, terminal, costs = times.tolist(), terminal.tolist(), costs.tolist()
        return [_State(spot[i:i+1], variance[i:i+1], dates[i], _ledger_slice(ledger, i),
                       marks[i:i+1], costs[i] if terminal[i] else None) for i in range(len(states))]


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
            logits, _ = policy(policy.features(observed, spot0=config.market.spot0))
            actions = logits.masked_fill(~legal, -torch.inf).argmax(-1)
        else:
            model = _FinanceSearch(policy, config)
            rngs = [np.random.default_rng(int(value)) for value in stream.integers(2**63-1, size=len(observed))]
            results = stochastic_search_batch(model.roots(observed, ledger, time_index), model, rngs,
                        simulations=simulations, c_puct=c_puct*float(policy.value_scale))
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
                   checkpoint_path=None, checkpoint_every=4, resume_from=None):
    """Full single-agent search/improvement loop on the supplied training bank.

    Each iteration collects complete episodes under visit-sampled search,
    then fits policy targets from visit frequencies and value targets from the
    *realized* terminal RU cost. A small rolling replay retains earlier blocks.
    Exogenous training paths are used only for live episode steps; planning
    generates independent conditional paths via the same financial model.
    ``updates`` counts self-play batches, not neural optimizer steps.
    Checkpoints contain the optimizer, rolling replay, ES threshold and every
    random stream; resuming extends self-play without discarding learned state.
    """
    if min(updates, batch_size, simulations, gradient_steps, replay_batches, checkpoint_every) < 1:
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
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    index_rng = torch.Generator().manual_seed(seed+100003)
    action_rng = torch.Generator(device=device).manual_seed(seed+200003)
    search_rng = np.random.default_rng(seed+300003)
    replay, history = deque(maxlen=replay_batches), []
    options = dict(updates=updates, batch_size=batch_size, simulations=simulations,
                   gradient_steps=gradient_steps, replay_batches=replay_batches,
                   hidden=list(hidden), grid_points=grid_points, learning_rate=learning_rate,
                   c_puct=c_puct, action_targets=policy.targets.cpu().tolist(),
                   action_encoding="absolute_grid_with_off_grid_hold")
    start_step, previous_seconds = 0, 0.
    work = dict(transition_samples=0, network_rows=0, terminal_evaluations=0, maximum_depth=0)
    if resume_from is not None:
        saved = load_checkpoint(resume_from, method="alphazero", config=config)
        check_resume_options(saved, options)
        if saved["seed"] != seed or saved["step"] > updates:
            raise ValueError("resume requires the saved seed and at least its completed updates")
        policy.load_state_dict(saved["policy"])
        optimizer.load_state_dict(saved["optimizer"])
        replay.extend(tuple(value.to(device) for value in block) for block in saved["replay"])
        history, work = saved["history"], saved["work"]
        index_rng.set_state(saved["index_rng"])
        action_rng.set_state(saved["action_rng"])
        search_rng.bit_generator.state = saved["search_rng"]
        restore_rng(saved["rng"])
        start_step, previous_seconds = saved["step"], saved["training_seconds"]
    started = time.perf_counter()
    if progress:
        _report("train_start", method="alphazero", seed=seed, device=str(device),
                updates=updates, batch_size=batch_size, simulations=simulations,
                dates=config.n_steps, actions=len(policy.targets)+1,
                expected_root_searches=updates*batch_size*config.n_steps, resumed_step=start_step)
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
                        simulations=simulations, c_puct=c_puct*float(policy.value_scale), root_noise=.25)
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
            costs = config.risk.loss(terminal_loss, fixed_zeta)
            replay.append((torch.cat(features), torch.cat(target_policies), torch.cat(masks),
                           ((costs-fixed_zeta)/policy.value_scale).repeat(config.n_steps)))
        x, pi, legal, cost = [torch.cat([block[column] for block in replay]) for column in range(4)]
        policy.train()
        for _ in range(gradient_steps):
            logits, values = policy(x)
            policy_loss = F.cross_entropy(logits.masked_fill(~legal, torch.finfo(logits.dtype).min), pi)
            value_loss = F.mse_loss(values, cost)
            objective = policy_loss + value_loss
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        with torch.no_grad():
            # Coordinate minimization in zeta, after (never during) self-play.
            policy.zeta.copy_(torch.quantile(terminal_loss, config.risk.alpha))
        _sync(device)
        segment_elapsed = time.perf_counter()-started
        elapsed = previous_seconds+segment_elapsed
        record = dict(completed=update, total=updates, elapsed_seconds=elapsed,
                      eta_seconds=segment_elapsed*(updates-update)/(update-start_step), policy_loss=float(policy_loss.detach()),
                      value_loss=float(value_loss.detach()), zeta=float(policy.zeta), **work)
        history.append(record)
        if progress:
            _report("train_progress", method="alphazero", **record)
        if checkpoint_path is not None and due_checkpoint(update, updates, checkpoint_every):
            save_checkpoint(checkpoint_path, dict(method="alphazero", config=asdict(config),
                options=options, seed=seed, step=update, policy=policy.state_dict(),
                optimizer=optimizer.state_dict(), replay=list(replay), history=history,
                work=work, index_rng=index_rng.get_state(), action_rng=action_rng.get_state(),
                search_rng=search_rng.bit_generator.state, rng=rng_state(), training_seconds=elapsed))
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
            parameter_count=sum(p.numel() for p in policy.parameters()), work=work)
