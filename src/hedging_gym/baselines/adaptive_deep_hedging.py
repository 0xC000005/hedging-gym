"""Online Deep Hedging and task-embedding adaptation on the common ledger.

The embedding mechanism follows Schmid and Oeltz (2025), §2.2:
https://arxiv.org/html/2504.16436v1. Source markets jointly fit shared weights
and task vectors; a new market fits a vector initialized at their mean while
the shared weights stay fixed. This is an implementation of that mechanism,
not a native reproduction: we use the configured cost-inclusive objective, causal
observation schema and multi-instrument book instead of their frictionless
stock-only squared-error experiment. This is a paper-based implementation.
See docs/fast-adaptation.md for source mapping and qualification.

Pretraining banks must be declared separately from future evaluation regimes.
Controllers never infer a task vector from evaluation paths. An updater receives
only the current stage's training bank, after its pre-update evaluation.
"""

import math
import time
from dataclasses import asdict, replace

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
from hedging_gym.baselines._shared.policy import (
    BUY,
    HOLD,
    SELL,
    PolicyAction,
    _bounds,
    _ConfiguredPolicy,
    _network,
)
from hedging_gym.baselines._shared.training import _report, _sync, rollout
from hedging_gym.environment.finance import bank_subset, bank_to

__all__ = ["TaskEmbeddedPolicy", "train_multitask", "make_controller", "make_updater"]


def make_updater(policy, **options):
    """Fit a task embedding with shared policy weights frozen."""
    from hedging_gym.baselines._shared.adaptation import AdaptationUpdater

    return AdaptationUpdater(policy, mode="embedding", **options)


def _continuous_contract(config):
    if any(config.execution.vector("minimum_trade", config.n_assets)
           + config.execution.vector("trade_lot", config.n_assets)):
        raise ValueError("continuous adaptation does not support minimum-trade or lot constraints")


class TaskEmbeddedPolicy(_ConfiguredPolicy):
    """One shared hedger conditioned on a learned, low-dimensional task vector.

    ``active_task`` is used only by the pretraining loop. Evaluation always uses
    ``embedding``, the most recently fitted vector; it does not select a stored
    vector by looking up the evaluation regime.
    """

    def __init__(self, config, *, n_tasks, embedding_dim=4, hidden=(32, 32)):
        super().__init__(config)
        if not all(math.isfinite(value) for name in ("holding_lower", "holding_upper")
                   for value in config.execution.vector(name, config.n_assets)):
            raise ValueError("task-embedding adaptation requires finite holding bounds")
        if n_tasks < 1 or embedding_dim < 1:
            raise ValueError("positive task count and embedding dimension required")
        self.shared = _network(self.feature_dim + embedding_dim, self.n_assets, hidden)
        self.source_embeddings = nn.Parameter(torch.randn(n_tasks, embedding_dim) * .1)
        self.embedding = nn.Parameter(torch.zeros(embedding_dim), requires_grad=False)
        self.active_task = None

    def prepare_adaptation(self):
        """Freeze learned source structure and initialize a new task at its mean."""
        self.shared.requires_grad_(False)
        self.source_embeddings.requires_grad_(False)
        self.embedding.requires_grad_(True)
        self.active_task = None
        self.reset_embedding()

    def reset_embedding(self):
        with torch.no_grad():
            self.embedding.copy_(self.source_embeddings.mean(dim=0))

    def forward(self, features, holdings, lower, upper, *, deterministic=True,
                generator=None):
        del deterministic, generator
        lo, hi = _bounds(features, holdings, lower, upper, self.feature_dim, self.n_assets)
        vector = (self.embedding if self.active_task is None
                  else self.source_embeddings[self.active_task])
        inputs = torch.cat((features, vector.expand(len(features), -1)), dim=-1)
        target = lo + (hi-lo) * self.shared(inputs).sigmoid()
        modes = torch.where(target > holdings, BUY, torch.where(target < holdings, SELL, HOLD))
        return PolicyAction(target, modes, features.new_zeros(len(features)))




def train_multitask(train_banks, *, seed=7, updates=8, batch_size=32,
                    hidden=(32, 32), embedding_dim=4, learning_rate=1e-3,
                    zeta_learning_rate=3e-4, device="cpu", progress=True,
                    checkpoint_path=None, checkpoint_every=100, resume_from=None,
                    policy_class=TaskEmbeddedPolicy):
    """Jointly fit source tasks, then return an embedding-adaptable policy.

    ``updates`` is the total number of optimizer steps, not steps per market.
    Tasks are visited round-robin, with a separately fitted ES threshold per
    task. Returned evaluation behavior uses the mean learned source embedding.
    Source configurations and work counts are retained in metadata. A context
    policy factory can replace the input-embedding architecture while keeping
    the source banks, optimizer and risk objective identical.
    """
    banks = tuple(train_banks)
    if len(banks) < 2 or updates < len(banks) or batch_size < 1:
        raise ValueError("multitask pretraining needs at least two banks and one update per task")
    if learning_rate <= 0 or zeta_learning_rate <= 0 or checkpoint_every < 1:
        raise ValueError("learning rates must be positive")
    config = banks[0].config
    _continuous_contract(config)
    for bank in banks:
        if len(bank.spot) < 1 or replace(bank.config, market=config.market) != config:
            raise ValueError("nonempty source banks must share book, clock, execution and risk")
    device = torch.device(device)
    started = time.perf_counter()
    torch.manual_seed(seed)
    policy = policy_class(config, n_tasks=len(banks), embedding_dim=embedding_dim,
                                hidden=hidden).to(device=device, dtype=banks[0].spot.dtype)
    method_name = getattr(policy, "method_name", "adaptive_dh")
    for bank in banks:
        policy.check_config(bank.config)
    banks = tuple(bank_to(bank, device) for bank in banks)
    zeta = nn.Parameter(banks[0].spot.new_zeros(len(banks)))
    optimizer = torch.optim.Adam([
        {"params": [*policy.shared.parameters(), policy.source_embeddings], "lr": learning_rate},
        {"params": [zeta], "lr": zeta_learning_rate},
    ])
    options = dict(updates=updates, batch_size=batch_size, hidden=list(hidden),
                   embedding_dim=embedding_dim, learning_rate=learning_rate,
                   zeta_learning_rate=zeta_learning_rate)
    if type(policy) is not TaskEmbeddedPolicy:
        options["policy_class"] = type(policy).__name__
    if progress:
        _report("train_start", method=method_name, seed=seed, device=str(device),
                workers=torch.get_num_threads(), source_markets=[asdict(b.config.market) for b in banks],
                options=options, expected_episode_rollouts=updates*batch_size,
                initialization_paths=sum(min(1024, len(b.spot)) for b in banks))
    index_generator = torch.Generator().manual_seed(seed+100003)
    history, completed, prior_seconds = [], 0, 0.
    source_configs = [asdict(bank.config) for bank in banks]
    if resume_from is not None:
        saved = load_checkpoint(resume_from, method=method_name, config=config)
        check_resume_options(saved, options)
        if (saved["seed"] != seed or saved["source_configs"] != source_configs
                or saved["source_paths"] != [len(bank.spot) for bank in banks]):
            raise ValueError("resume requires the saved seed, source markets and bank sizes")
        policy.load_state_dict(saved["policy"])
        zeta.data.copy_(saved["zeta"].to(device))
        optimizer.load_state_dict(saved["optimizer"])
        index_generator.set_state(saved["index_rng"])
        restore_rng(saved["rng"])
        completed, history = saved["step"], saved["history"]
        prior_seconds = saved["training_seconds"]
        if completed > updates:
            raise ValueError("total updates cannot precede the checkpoint")
    else:
        with torch.no_grad():
            for task, bank in enumerate(banks):
                policy.active_task = task
                losses = rollout(policy, bank_subset(bank, slice(0, min(1024, len(bank.spot)))))
                zeta[task] = torch.quantile(losses["terminal_loss"], config.risk.alpha)
    _sync(device)
    initialization_seconds = time.perf_counter()-started
    training_start = time.perf_counter()
    for step in range(completed+1, updates+1):
        task = (step-1) % len(banks)
        policy.active_task = task
        bank = banks[task]
        indices = torch.randint(len(bank.spot), (batch_size,), generator=index_generator)
        losses = rollout(policy, bank_subset(bank, indices.to(device)))["terminal_loss"]
        objective = config.risk.loss(losses, zeta[task]).mean()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_([*policy.shared.parameters(), policy.source_embeddings, zeta],
                                       5., error_if_nonfinite=True)
        optimizer.step()
        if step <= len(banks) or (step-1) % (20*len(banks)) < len(banks) or step == updates:
            _sync(device)
            elapsed = prior_seconds+time.perf_counter()-training_start
            record = dict(completed=step, total=updates, task=task,
                          batch_risk_loss=float(objective.detach()), elapsed_seconds=elapsed,
                          eta_seconds=elapsed*(updates-step)/step, zeta=float(zeta[task].detach()))
            history.append(record)
            if progress:
                _report("train_progress", method=method_name, **record)
        if checkpoint_path is not None and due_checkpoint(step, updates, checkpoint_every):
            save_checkpoint(checkpoint_path, dict(method=method_name, phase="pretraining",
                step=step, seed=seed, config=asdict(config), source_configs=source_configs,
                source_paths=[len(bank.spot) for bank in banks], options=options,
                policy=policy.state_dict(), zeta=zeta.detach(), optimizer=optimizer.state_dict(),
                index_rng=index_generator.get_state(), rng=rng_state(), history=history,
                training_seconds=prior_seconds+time.perf_counter()-training_start))
    policy.prepare_adaptation()
    _sync(device)
    # This mean initializes the optimizer's auxiliary threshold; it is not a
    # claim that the mixture's VaR is the average source-task VaR.
    metadata = dict(method=method_name, method_label=getattr(policy, "method_label", "Task-embedding adaptive Deep Hedging"),
        policy_class=type(policy).__name__,
        classification="Common-benchmark adaptation, not paper reproduction",
        source="https://arxiv.org/html/2504.16436v1#S2.SS2",
        source_changes=["configured cost-inclusive objective replaces the source squared-error task", "common causal observations",
                        "multi-instrument bounded holdings", "common tanh architecture and optimizer",
                        "PyTorch implementation"],
        seed=seed, minibatch_seed=seed+100003, device=str(device), options=options,
        config=asdict(config), source_configs=[asdict(bank.config) for bank in banks],
        source_paths=[len(bank.spot) for bank in banks],
        updates_per_source=[updates//len(banks)+(i < updates % len(banks)) for i in range(len(banks))],
        source_zetas=zeta.detach().cpu().tolist(), zeta=float(zeta.detach().mean()),
        history=history, initialization_seconds=initialization_seconds,
        training_seconds=prior_seconds+time.perf_counter()-training_start,
        total_seconds=prior_seconds+time.perf_counter()-started,
        expected_episode_rollouts=updates*batch_size,
        observation_fields=list(policy.observation_fields), instrument_names=list(policy.instrument_names),
        parameter_count=sum(p.numel() for p in policy.parameters()),
        adaptation_policy_parameter_count=policy.embedding.numel(),
        adaptation_threshold_parameter_count=1)
    return policy, metadata
