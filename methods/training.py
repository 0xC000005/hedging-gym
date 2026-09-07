"""Deep Hedging/band training through the shared differentiable environment.

Adam jointly fits the policy and a global expected-shortfall threshold, with
separate learning rates. Training samples only its own market bank; evaluation
and model selection stay outside this adapter.
"""

from dataclasses import asdict
import json
import time

import torch

from hedging_gym.finance import bank_subset, bank_to
from hedging_gym.gym_env import TensorHedgingEnv

from .policies import DirectDHPolicy, NoTransactionBandPolicy
from .checkpoints import (save_checkpoint, load_checkpoint, rng_state, restore_rng,
                          check_resume_options, due_checkpoint)


POLICIES = {"dh": DirectDHPolicy, "ntb": NoTransactionBandPolicy}
METHOD_LABELS = {
    "dh": "Deep Hedging",
    "ntb": "Learned no-transaction bands",
}


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _report(stage, **values):
    print(json.dumps({"stage": stage, **values}, allow_nan=False), flush=True)


def rollout(policy, bank, *, record_positions=False):
    """Differentiable complete episode using the identical evaluation ledger.

    The source causal policy loop now calls TensorHedgingEnv directly. No holding
    rounding, penalty objective, straight-through gradient or cash equation lives
    in this adapter. Terminal loss includes the mandatory liquidation and fees.
    """
    policy.check_config(bank.config)
    env = TensorHedgingEnv(bank)
    observed = env.reset()
    positions = []
    for _ in range(bank.config.n_steps):
        target = policy(observed, env.state.positions, bank.config.execution.holding_lower,
                        bank.config.execution.holding_upper, deterministic=True).target_holdings
        if record_positions:
            positions.append(target.detach().clone())
        observed, _, terminated, truncated, result = env.step(target)
    if not terminated or truncated:
        raise RuntimeError("training requires complete terminal financial episodes")
    if record_positions:
        result["positions"] = torch.stack(positions, dim=1)
    return result


def train_policy(method, train_bank, *, seed=7, updates=8, batch_size=32,
                 hidden=(32, 32), learning_rate=1e-3, zeta_learning_rate=3e-4,
                 device="cpu", progress=True, checkpoint_path=None,
                 checkpoint_every=200, resume_from=None):
    """Return a trained policy and compact metadata; no model selection occurs.

    The configured tail risk uses one learned episode-global threshold,
    initialized from training losses at risk.alpha. Subsequent minibatches use
    the shared RiskConfig.loss implementation.
    Ordinary pathwise derivatives miss moving fixed-fee/hold-gate boundary terms.
    Continuous outputs are not adapted to lot/minimum-order constraints, so those
    contracts fail explicitly before training instead of being projected.
    """
    if method not in POLICIES or updates < 1 or batch_size < 1 or len(train_bank.spot) < 1:
        raise ValueError("choose dh/ntb with positive training work and a nonempty bank")
    config = train_bank.config
    if any(config.execution.vector("minimum_trade", config.n_assets)
           + config.execution.vector("trade_lot", config.n_assets)):
        raise ValueError("continuous DH/NTB training does not support minimum-trade or lot constraints")
    if learning_rate <= 0 or zeta_learning_rate <= 0 or checkpoint_every < 1:
        raise ValueError("learning rates must be positive")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; select --device cpu")
    started = time.perf_counter()
    options = dict(updates=updates, batch_size=batch_size, hidden=list(hidden),
                   learning_rate=learning_rate, zeta_learning_rate=zeta_learning_rate)
    if progress:
        _report("train_start", method=method, label=METHOD_LABELS[method], seed=seed,
                minibatch_seed=seed+100003, device=str(device), workers=torch.get_num_threads(),
                config=asdict(config), options=options,
                expected_episode_rollouts=updates*batch_size,
                initialization_paths=min(1024, len(train_bank.spot)))
    # A separate CPU generator owns minibatch sampling, as in the source run.
    torch.manual_seed(seed)
    policy = POLICIES[method](config, hidden=hidden)
    policy = policy.to(device=device, dtype=train_bank.spot.dtype)
    train_device = bank_to(train_bank, device)
    zeta = torch.nn.Parameter(torch.zeros((), device=device, dtype=train_bank.spot.dtype))
    optimizer = torch.optim.Adam([
        {"params": policy.parameters(), "lr": learning_rate},
        {"params": [zeta], "lr": zeta_learning_rate},
    ])
    index_generator = torch.Generator().manual_seed(seed+100003)
    first_step, previous_seconds, history = 0, 0., []
    if resume_from:
        saved = load_checkpoint(resume_from, method=method, config=config)
        check_resume_options(saved, options)
        if saved["seed"] != seed or saved["training_paths"] != len(train_bank.spot):
            raise ValueError("resume requires the original seed and frozen training bank")
        policy.load_state_dict(saved["policy"])
        with torch.no_grad():
            zeta.copy_(saved["zeta"])
        optimizer.load_state_dict(saved["optimizer"])
        index_generator.set_state(saved["index_rng"])
        restore_rng(saved["rng"])
        first_step, history = saved["step"], saved["history"]
        previous_seconds = saved["elapsed_seconds"]
    else:
        with torch.no_grad():
            sample = bank_subset(train_device, slice(0, min(1024, len(train_device.spot))))
            zeta.copy_(torch.quantile(rollout(policy, sample)["terminal_loss"], config.risk.alpha))
    _sync(device)
    initialization_seconds = time.perf_counter()-started
    training_start = time.perf_counter()
    for update in range(first_step+1, updates+1):
        indices = torch.randint(len(train_bank.spot), (batch_size,), generator=index_generator)
        sample = bank_subset(train_device, indices.to(device))
        policy.train()
        losses = rollout(policy, sample)["terminal_loss"]
        objective = config.risk.loss(losses, zeta).mean()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_([*policy.parameters(), zeta], 5.0, error_if_nonfinite=True)
        optimizer.step()
        if update == 1 or update % 20 == 0 or update == updates:
            _sync(device)
            elapsed = time.perf_counter()-training_start
            record = dict(completed=update, total=updates, elapsed_seconds=elapsed,
                          updates_per_second=(update-first_step)/max(elapsed, 1e-12),
                          eta_seconds=elapsed*(updates-update)/(update-first_step),
                          batch_risk_loss=float(objective.detach()), risk_alpha=config.risk.alpha,
                          zeta=float(zeta.detach()))
            history.append(record)
            if progress:
                _report("train_progress", method=method, seed=seed, **record)
        if checkpoint_path and due_checkpoint(update, updates, checkpoint_every):
            save_checkpoint(checkpoint_path, dict(method=method, config=asdict(config),
                seed=seed, options=options, step=update, policy=policy.state_dict(),
                optimizer=optimizer.state_dict(), zeta=zeta.detach(), rng=rng_state(),
                index_rng=index_generator.get_state(), history=history,
                training_paths=len(train_bank.spot),
                elapsed_seconds=previous_seconds+time.perf_counter()-started))
    _sync(device)
    metadata = dict(label="Baseline training", method=method,
                    method_label=METHOD_LABELS[method], seed=seed, minibatch_seed=seed+100003,
                    device=str(device), options=options, config=asdict(config),
                    observation_fields=list(policy.observation_fields),
                    instrument_names=list(policy.instrument_names),
                    zeta=float(zeta.detach()), history=history,
                    initialization_seconds=initialization_seconds,
                    training_seconds=time.perf_counter()-training_start,
                    total_seconds=previous_seconds+time.perf_counter()-started,
                    resumed_from=str(resume_from) if resume_from else None, resumed_step=first_step,
                    expected_episode_rollouts=updates*batch_size,
                    parameter_count=sum(p.numel() for p in policy.parameters()))
    return policy, metadata
