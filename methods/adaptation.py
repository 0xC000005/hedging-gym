"""Online Deep Hedging and task-embedding adaptation on the common ledger.

The embedding mechanism follows Schmid and Oeltz (2025), §2.2:
https://arxiv.org/html/2504.16436v1. Source markets jointly fit shared weights
and task vectors; a new market fits a vector initialized at their mean while
the shared weights stay fixed. This is an implementation of that mechanism,
not a reproduction: we use the common cost-inclusive ES objective, causal
observation schema and multi-instrument book instead of their frictionless
stock-only squared-error experiment. No official experiment code was available.

Pretraining banks must be declared separately from future evaluation regimes.
Controllers never infer a task vector from evaluation paths. An updater receives
only the current stage's training bank, after its pre-update evaluation.
"""

from dataclasses import asdict, replace
import math
import time

import torch
from torch import nn

from hedging_gym.finance import bank_subset, bank_to
from .checkpoints import (check_resume_options, due_checkpoint, load_checkpoint,
                          restore_rng, rng_state, save_checkpoint, saved_config, saved_market)
from .policies import BUY, HOLD, SELL, PolicyAction, _ConfiguredPolicy, _bounds, _network
from .training import _report, _sync, rollout, train_policy


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


def train_online_finetune(train_bank, **kwargs):
    """Initial ordinary DH training; subsequent updates use AdaptationUpdater.

    Before any new-market updates, this *is* the ordinary DH policy. A distinct
    adaptation advantage cannot be claimed from this initial training alone.
    """
    policy, metadata = train_policy("dh", train_bank, **kwargs)
    metadata.update(method="finetune_dh", method_label="Online fine-tuned Deep Hedging",
                    adaptation="Full-network updates on current-stage training paths")
    return policy, metadata


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
        if (saved["seed"] != seed
                or [saved_config(value) for value in saved["source_configs"]] != [bank.config for bank in banks]
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
        source_changes=["cost-inclusive ES replaces squared error", "common causal observations",
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


class AdaptationUpdater:
    """Persistent current-stage trainer, callable as ``update(training_bank)``.

    Fine-tuning keeps the policy, ES threshold and Adam moments across calls and
    market changes. Its adaptation Adam starts fresh after initial pretraining.
    Embedding adaptation keeps shared weights frozen. On a newly encountered
    stage market it starts from the source-vector mean with a fresh embedding
    Adam, matching source new-task calibration; repeated calls in that market
    keep the optimizer. No cache silently restores a known evaluation regime.

    A call fits ``updates`` minibatches. Checkpoints include the in-flight call,
    so resuming it neither repeats an update nor silently resets a task vector.
    """

    def __init__(self, policy, *, metadata=None, mode=None, seed=7, updates=1,
                 batch_size=32, learning_rate=1e-3, zeta_learning_rate=3e-4,
                 progress=True, checkpoint_path=None, checkpoint_every=100):
        inferred = "embedding" if isinstance(policy, TaskEmbeddedPolicy) else "finetune"
        self.mode = inferred if mode is None else mode
        if self.mode not in ("finetune", "embedding") or (self.mode == "embedding" and inferred != self.mode):
            raise ValueError("embedding updates require a TaskEmbeddedPolicy")
        if (updates < 1 or batch_size < 1 or learning_rate <= 0 or zeta_learning_rate <= 0
                or checkpoint_every < 1):
            raise ValueError("positive update count, batch size and learning rates required")
        self.policy, self.updates, self.batch_size = policy, updates, batch_size
        self.learning_rate, self.zeta_learning_rate = learning_rate, zeta_learning_rate
        self.progress, self.history = progress, []
        self.seed = seed
        self.checkpoint_path, self.checkpoint_every = checkpoint_path, checkpoint_every
        self.completed_steps, self.pending_call = 0, None
        self.index_generator = torch.Generator().manual_seed(seed+100003)
        parameter = next(policy.parameters())
        self.zeta = nn.Parameter(parameter.new_tensor(0. if metadata is None else metadata["zeta"]))
        self.last_market = None
        if self.mode == "embedding":
            policy.prepare_adaptation()
        else:
            policy.requires_grad_(True)
        self._new_optimizer()

    def state_dict(self):
        """Complete adaptation state, including frozen-parameter semantics."""
        return dict(mode=self.mode, policy=self.policy.state_dict(),
            requires_grad={name: parameter.requires_grad for name, parameter in self.policy.named_parameters()},
            active_task=getattr(self.policy, "active_task", None), zeta=self.zeta.detach(),
            optimizer=self.optimizer.state_dict(), index_rng=self.index_generator.get_state(),
            rng=rng_state(), last_market=self.last_market, history=self.history,
            pending_call=self.pending_call, completed_steps=self.completed_steps,
            options=dict(updates=self.updates, batch_size=self.batch_size,
                         learning_rate=self.learning_rate, zeta_learning_rate=self.zeta_learning_rate,
                         seed=self.seed))

    def load_state_dict(self, saved):
        if saved["mode"] != self.mode or saved["options"] != self.state_dict()["options"]:
            raise ValueError("adaptation resume requires the saved mode and update recipe")
        self.policy.load_state_dict(saved["policy"])
        for name, parameter in self.policy.named_parameters():
            parameter.requires_grad_(saved["requires_grad"][name])
        if isinstance(self.policy, TaskEmbeddedPolicy):
            self.policy.active_task = saved["active_task"]
        self.zeta.data.copy_(saved["zeta"].to(self.zeta.device))
        self._new_optimizer()
        self.optimizer.load_state_dict(saved["optimizer"])
        self.index_generator.set_state(saved["index_rng"])
        self.last_market, self.history = saved_market(saved["last_market"]), saved["history"]
        self.pending_call, self.completed_steps = saved["pending_call"], saved["completed_steps"]
        restore_rng(saved["rng"])

    def _new_optimizer(self):
        self.trainable = ([self.policy.embedding] if self.mode == "embedding"
                          else list(self.policy.parameters()))
        self.optimizer = torch.optim.Adam([
            {"params": self.trainable, "lr": self.learning_rate},
            {"params": [self.zeta], "lr": self.zeta_learning_rate},
        ])

    def __call__(self, training_bank, *, initial_embedding=None):
        config = training_bank.config
        self.policy.check_config(config)
        _continuous_contract(config)
        device = self.zeta.device
        started = time.perf_counter()
        bank = bank_to(training_bank, device)
        continuing = self.pending_call is not None
        if continuing and self.last_market != config.market:
            raise ValueError("finish the saved current-market update before changing market")
        reset = not continuing and self.mode == "embedding" and self.last_market != config.market
        initialization_paths = 0
        if reset:
            self.policy.reset_embedding()
            # Retrieval or a context encoder may supply an initialization using
            # current training data. It must not be erased by the mean restart.
            if initial_embedding is not None:
                with torch.no_grad():
                    self.policy.embedding.copy_(initial_embedding)
            self._new_optimizer()
            initialization_paths = min(1024, len(bank.spot))
            with torch.no_grad():
                losses = rollout(self.policy, bank_subset(bank, slice(0, initialization_paths)))
                self.zeta.copy_(torch.quantile(losses["terminal_loss"], config.risk.alpha))
        self.last_market = config.market
        if not continuing:
            self.pending_call = dict(completed=0, reset_embedding=reset,
                initialization_paths=initialization_paths, elapsed_seconds=0.)
        pending = self.pending_call
        prior_seconds = pending["elapsed_seconds"]
        if self.progress:
            _report("adapt_start", method=self.mode, call=len(self.history)+1,
                    market=asdict(config.market), updates=self.updates, batch_size=self.batch_size,
                    device=str(device), reset_embedding=reset, initialization_paths=initialization_paths)
        for step in range(pending["completed"]+1, self.updates+1):
            indices = torch.randint(len(bank.spot), (self.batch_size,), generator=self.index_generator)
            sample = bank_subset(bank, indices.to(device))
            self.policy.train()
            losses = rollout(self.policy, sample)["terminal_loss"]
            objective = config.risk.loss(losses, self.zeta).mean()
            self.optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_([*self.trainable, self.zeta], 5., error_if_nonfinite=True)
            self.optimizer.step()
            self.completed_steps += 1
            pending["completed"] = step
            pending["elapsed_seconds"] = prior_seconds+time.perf_counter()-started
            if self.progress and (step == 1 or step % 20 == 0 or step == self.updates):
                _sync(device)
                elapsed = time.perf_counter()-started
                _report("adapt_progress", method=self.mode, completed=step, total=self.updates,
                        elapsed_seconds=elapsed, eta_seconds=elapsed*(self.updates-step)/step,
                        batch_risk_loss=float(objective.detach()))
            if step == self.updates:
                record = dict(call=len(self.history)+1, mode=self.mode, market=asdict(config.market),
                    updates=self.updates, expected_episode_rollouts=self.updates*self.batch_size,
                    initialization_paths=pending["initialization_paths"],
                    reset_embedding=pending["reset_embedding"], zeta=float(self.zeta.detach()),
                    elapsed_seconds=pending["elapsed_seconds"])
                self.history.append(record)
                self.pending_call = None
            if self.checkpoint_path is not None and due_checkpoint(step, self.updates, self.checkpoint_every):
                save_checkpoint(self.checkpoint_path, dict(method=self.mode+"_adaptation",
                    config=asdict(config), step=step,
                    total_completed_steps=self.completed_steps, state=self.state_dict()))
        _sync(device)
        return self.history[-1]
