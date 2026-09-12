"""Published AlphaZero search/training with a common financial game adapter.

Paper: Szehr, Hedging of Financial Derivative Contracts via Monte Carlo Tree
Search, https://arxiv.org/abs/2102.06274.
Upstream: https://github.com/plan64/minimalHedger_AlphaZero/tree/3111c378fcd17e45f94d2fc668a3aa117126ecba
Implementation notes: docs/source-alphazero.md.

The donor supplies MCTS, Trainer and supervised fitting. This module supplies
the game/network boundary and shared financial accounting. Its evaluation
controller deploys the learned policy's argmax, matching source validation;
that deployment does not run a new tree search. Objective and state-aggregation
changes are documented separately from the maintained AlphaZero adaptation.
"""
import importlib
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hedging_gym.environment import finance
from hedging_gym.environment.pricing import quantlib_mark_state

from .alphazero import holding_grid


@dataclass
class GameState:
    spot: torch.Tensor
    variance: torch.Tensor
    date: int
    ledger: finance.LedgerState
    marks: torch.Tensor
    loss: float | None = None
    integrated_variance: torch.Tensor | None = None


class SourceHedgingGame:
    """Scalar branching API over the same model, book and ledger as the Gym.

    No realized path is stored in a state. Search draws a fresh conditional
    market move on each call. Only dictionary keys are rounded, never finances.
    The source has no action masks, so this adapter currently qualifies Basic
    (and fees), not minimum-order sizes or lots.
    """
    def __init__(self, config, *, seed, zeta, scale, objective=None, grid_points=5, output=None):
        if objective is not None:
            config = replace(config, risk=replace(config.risk, objective=objective))
        if any(config.execution.vector("minimum_trade", config.n_assets)):
            raise ValueError("source MCTS has no state-dependent action mask; use Basic")
        if scale <= 0:
            raise ValueError("positive monetary reward scale required")
        self.config, self.zeta, self.scale = config, float(zeta), float(scale)
        self.objective = config.risk.objective
        self.targets = holding_grid(config, grid_points)
        self.rng = np.random.default_rng(seed)
        self.output = Path(output) if output is not None else None
        spot = torch.tensor([config.market.spot0], dtype=torch.float64)
        variance = torch.tensor([config.market.v0], dtype=torch.float64)
        self.needs_integral = any(instrument.needs_integrated_variance for instrument in
                                 (config.portfolio.liability, *config.portfolio.hedges))
        integral = torch.zeros_like(spot) if self.needs_integral else None
        marks, _ = self.mark(spot, variance, 0, integral)
        self.initial = GameState(spot, variance, 0, finance.initial_ledger(spot, config), marks,
                                  integrated_variance=integral)
        if not finance.feasible_targets(self.initial.ledger.positions[:, None], self.targets[None], config).all():
            raise ValueError("source MCTS needs every grid target legal from initial inventory")
        self.feature_dim = len(finance.observation_fields(config))
        self.transitions = 0
        self.search_stats = dict(calls=0, maximum_depth=0, terminal_returns=0)

    def getInitState(self):
        # Finance functions return new ledger tensors; the initial state is read-only.
        return self.initial

    def getActionSize(self):
        return len(self.targets)

    def getNormalizations(self):
        # The shared observation function already applies its documented units.
        return np.ones(self.feature_dim)

    def observe(self, state):
        return finance.observation_from_state(state.spot, state.variance, state.date,
            state.ledger, state.marks, self.config)[0].numpy()

    def mark(self, spot, variance, date, integral=None):
        """Use the environment's scalar pricing backend for donor game queries."""
        return quantlib_mark_state(spot, variance, date, self.config,
                                   integrated_variance=integral)

    @torch.no_grad()
    def getNextState(self, state, player, action):
        if state.date >= self.config.n_decisions:
            raise ValueError("cannot advance a settled state")
        ledger = finance.trade_step(state.ledger, self.targets[int(action)][None], state.marks, self.config)
        spot, variance, integral = state.spot, state.variance, state.integrated_variance
        if state.date < self.config.n_steps:
            shocks = finance.market_shocks(self.config.market, self.rng, count=1, dt=self.config.dt)
            spot, variance = finance.transition(state.spot, state.variance, shocks,
                                               self.config.market, dt=self.config.dt)
            if integral is not None:
                integral = integral + .5*(state.variance + variance)*self.config.dt
        date = state.date + 1
        marks, liability = self.mark(spot, variance, date, integral)
        loss = (float(finance.liquidate(ledger, marks, liability, self.config)["terminal_loss"][0])
                if date == self.config.n_decisions else None)
        self.transitions += 1
        return GameState(spot, variance, date, ledger, marks, loss, integral), player

    def getGameEnded(self, state, player):
        if state.loss is None:
            return 0.
        # A constant shift keeps perfect hedges distinct from the source's
        # zero/nonterminal flag. Scaling changes units, not the objective's optimum.
        loss = self.config.risk.loss(torch.tensor(state.loss, dtype=torch.float64), self.zeta)
        if self.objective == "mse":
            return -1. - float(loss)/self.scale**2
        # Fixed-zeta RU minimization for the separately selected ES experiment.
        return -1. - float(loss-self.zeta)/self.scale

    def stringRepresentation(self, state):
        if state.loss is not None:
            # Never cache one rounded state's realized reward for another path.
            return ("terminal", state.loss)
        # Source Heston bins: variance 3 decimals, holdings/cash/spot 2.
        # This is approximate transposition reuse, not rounding executed trades.
        key = (state.date, round(float(state.variance[0]), 3),
                *(round(float(x), 2) for x in state.ledger.positions[0]),
                round(float(state.ledger.cash[0]), 2), round(float(state.spot[0]), 2))
        return key if state.integrated_variance is None else (*key, float(state.integrated_variance[0]))


def load_source(donor_path):
    """Load a pinned checkout in a fresh process; no vendor copy in this repo.

    Remove the donor's unused torchvision import in the external checkout.
    All compatibility changes belong in the run's source diff.
    """
    sys.path.insert(0, str(Path(donor_path).resolve()))
    trainer_module = importlib.import_module("Trainer")
    source_mcts = importlib.import_module("MCTS").MCTS
    source_wrapper = importlib.import_module("hedger_TV.neuralNet.trainerNeuralNet_simpleFF").NNetWrapper
    source_network = importlib.import_module("hedger_TV.neuralNet.hedgerNeuralNet_simpleFF").HedgerNNet

    class MeasuredMCTS(source_mcts):
        def __init__(self, *args):
            super().__init__(*args)
            self.depth = 0

        def search(self, state):
            self.game.search_stats["maximum_depth"] = max(self.game.search_stats["maximum_depth"], self.depth)
            self.game.search_stats["calls"] += 1
            self.game.search_stats["terminal_returns"] += int(state.loss is not None)
            self.depth += 1
            try:
                return super().search(state)
            finally:
                self.depth -= 1

    trainer_module.MCTS = MeasuredMCTS  # Instrumentation only; source recursion/backup unchanged.

    class CommonNetwork(source_network):
        def __init__(self, game, args):
            super().__init__(game, args)
            self.lin1 = nn.Linear(game.feature_dim, args["num_channels"])

        def forward(self, s):
            # Same six hidden layers, batch normalization, dropout and heads.
            # Only input width and the terminal value activation differ.
            s = F.relu(self.bn1(self.lin1(s)))
            s = F.relu(self.bn2(self.lin2(s)))
            s = F.relu(self.bn3(self.lin3(s)))
            s = F.relu(self.bn4(self.lin4(s)))
            s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args["dropout"], training=self.training)
            s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args["dropout"], training=self.training)
            return F.log_softmax(self.fc3(s), dim=1), self.fc4(s)

    class CommonWrapper(source_wrapper):
        def __init__(self, game, args):
            self.game, self.args = game, args
            self.nnet = CommonNetwork(game, args)
            self.action_size = game.getActionSize()
            self.normalizations = game.getNormalizations()
            self.cuda = False  # Native scalar inference was faster on CPU; batches stay on CPU too.
            self.fit_count = 0
            self.fit_records = []

        def train(self, examples):
            if len(examples) < max(2, self.args["batch_size"]):
                raise ValueError("not enough self-play examples for a source training minibatch")
            before = {k: v.detach().clone() for k, v in self.nnet.named_parameters()}
            converted = [(self.game.observe(s), pi, value, action) for s, pi, value, action in examples]
            super().train(converted)
            changed = any(not torch.equal(before[k], v) for k, v in self.nnet.named_parameters())
            self.fit_count += 1
            self.fit_records.append(dict(examples=len(examples), weights_changed=changed,
                optimizer_steps=(len(examples)//self.args["batch_size"])*self.args["epochs"]))
            if self.game.output is not None:
                self.save_checkpoint(str(self.game.output), f"candidate-{self.fit_count}.pt")

        def predict(self, state):
            return super().predict(self.game.observe(state))

        @torch.no_grad()
        def predict_batch(self, observed):
            self.nnet.eval()
            logits, values = self.nnet(observed.detach().cpu().float())
            return logits.exp(), values[:, 0]

        def load_checkpoint(self, folder, filename=None):
            path = Path(folder)/filename if filename is not None else Path(folder)
            self.nnet.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])

    return trainer_module.Trainer, CommonWrapper


def source_controller(network, game):
    """Batched greedy policy deployment, as in the source Trainer's validation."""
    def controller(observed, ledger, date, config):
        if config != game.config:
            raise ValueError("evaluate with the checkpoint's financial configuration")
        probabilities, _ = network.predict_batch(observed)
        return game.targets[probabilities.argmax(-1)].to(observed)
    controller.action_selection = "source_policy_argmax"
    return controller
