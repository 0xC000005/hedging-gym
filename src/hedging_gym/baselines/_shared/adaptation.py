"""Shared stateful optimizer for full-network and task-embedding adaptation."""
import time
from dataclasses import asdict

import torch
from torch import nn

from hedging_gym.baselines.adaptive_deep_hedging import (
    TaskEmbeddedPolicy,
    _continuous_contract,
)
from hedging_gym.environment.finance import bank_subset, bank_to

from .checkpoints import due_checkpoint, restore_rng, rng_state, save_checkpoint
from .training import _report, _sync, rollout


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
        self.last_market, self.history = saved["last_market"], saved["history"]
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
