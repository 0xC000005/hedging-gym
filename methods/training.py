"""ADAPTATION: source DH/NTB pathwise training through the common tensor env.

Retains the historical global RU threshold, Adam parameter groups, independent
minibatch RNG and gradient clipping from heston_v1/run.py. Checkpoint selection,
hybrid score estimators, native donor stacks and research orchestration are not
part of this small development adapter. No test bank enters training.
"""

from dataclasses import asdict
import json
import time

import torch

from hedging_gym.finance import bank_subset, bank_to, observation_fields
from hedging_gym.gym_env import TensorHedgingEnv

from .policies import DirectDHPolicy, NoTransactionBandPolicy


POLICIES = {"dh": DirectDHPolicy, "ntb": NoTransactionBandPolicy}
METHOD_LABELS = {
    "dh": "ADAPTATION / bounded direct Deep Hedging",
    "ntb": "ADAPTATION / learned no-transaction band, conditional pathwise training",
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
    env = TensorHedgingEnv(bank)
    observed = env.reset()
    positions = []
    for _ in range(bank.config.n_steps):
        target = policy(observed, env.state.positions, bank.config.holding_lower,
                        bank.config.holding_upper, deterministic=True).target_holdings
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
                 device="cpu", progress=True):
    """Return a trained policy and compact metadata; no model selection occurs.

    ES95 uses one learned episode-global threshold, initialized once from training
    losses. Its subsequent minibatch objective is zeta + relu(loss-zeta)/0.05.
    Ordinary pathwise derivatives miss moving fixed-fee/hold-gate boundary terms.
    Continuous outputs are not adapted to lot/minimum-order constraints, so those
    contracts fail explicitly before training instead of being projected.
    """
    if method not in POLICIES or updates < 1 or batch_size < 1 or len(train_bank.spot) < 1:
        raise ValueError("choose dh/ntb with positive training work and a nonempty bank")
    if any(train_bank.config.minimum_trade + train_bank.config.trade_lot):
        raise ValueError("continuous DH/NTB training does not support minimum-trade or lot constraints")
    if learning_rate <= 0 or zeta_learning_rate <= 0:
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
                config=asdict(train_bank.config), options=options,
                expected_episode_rollouts=updates*batch_size,
                initialization_paths=min(1024, len(train_bank.spot)))
    # A separate CPU generator owns minibatch sampling, as in the source run.
    torch.manual_seed(seed)
    policy = POLICIES[method](len(observation_fields(train_bank.config)),
                              n_assets=train_bank.config.n_assets, hidden=hidden)
    policy = policy.to(device=device, dtype=train_bank.spot.dtype)
    train_device = bank_to(train_bank, device)
    zeta = torch.nn.Parameter(torch.zeros((), device=device, dtype=train_bank.spot.dtype))
    optimizer = torch.optim.Adam([
        {"params": policy.parameters(), "lr": learning_rate},
        {"params": [zeta], "lr": zeta_learning_rate},
    ])
    index_generator = torch.Generator().manual_seed(seed+100003)
    with torch.no_grad():
        sample = bank_subset(train_device, slice(0, min(1024, len(train_device.spot))))
        zeta.copy_(torch.quantile(rollout(policy, sample)["terminal_loss"], .95))
    _sync(device)
    initialization_seconds = time.perf_counter()-started
    training_start = time.perf_counter()
    history = []
    for update in range(1, updates+1):
        indices = torch.randint(len(train_bank.spot), (batch_size,), generator=index_generator)
        sample = bank_subset(train_device, indices.to(device))
        policy.train()
        losses = rollout(policy, sample)["terminal_loss"]
        objective = (zeta+(losses-zeta).relu()/.05).mean()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_([*policy.parameters(), zeta], 5.0, error_if_nonfinite=True)
        optimizer.step()
        if update == 1 or update % 20 == 0 or update == updates:
            _sync(device)
            elapsed = time.perf_counter()-training_start
            record = dict(completed=update, total=updates, elapsed_seconds=elapsed,
                          updates_per_second=update/max(elapsed, 1e-12),
                          eta_seconds=elapsed*(updates-update)/update,
                          batch_ru_es95=float(objective.detach()), zeta=float(zeta.detach()))
            history.append(record)
            if progress:
                _report("train_progress", method=method, seed=seed, **record)
    _sync(device)
    metadata = dict(label="DEVELOPMENT / ADAPTATION; no publication comparison", method=method,
                    method_label=METHOD_LABELS[method], seed=seed, minibatch_seed=seed+100003,
                    device=str(device), options=options, config=asdict(train_bank.config),
                    zeta=float(zeta.detach()), history=history,
                    initialization_seconds=initialization_seconds,
                    training_seconds=time.perf_counter()-training_start,
                    total_seconds=time.perf_counter()-started,
                    expected_episode_rollouts=updates*batch_size,
                    parameter_count=sum(p.numel() for p in policy.parameters()))
    return policy, metadata
