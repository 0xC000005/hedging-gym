"""One algorithm-independent evaluator for the batched hedging environment."""
import math
import time
from typing import TYPE_CHECKING

import torch

from .environment.finance import bank_subset, bank_to
from .environment.rollout import run_episode

if TYPE_CHECKING:
    from .interfaces import Controller


def empirical_es(values: torch.Tensor, alpha: float) -> float:
    """Integral empirical ES with fractional weight at the tail boundary."""
    if not 0 < alpha < 1 or values.ndim != 1 or values.numel() == 0:
        raise ValueError("ES needs nonempty scalar path losses and alpha in (0,1)")
    ordered = values.detach().double().sort(descending=True).values
    mass = (1 - alpha) * len(ordered)
    whole = math.floor(mass)
    fraction = mass - whole
    tail = ordered[:whole].sum()
    if fraction > 0:
        tail = tail + fraction * ordered[whole]
    return float(tail / mass)


@torch.no_grad()
def evaluate_controller(controller: "Controller", bank, *, device=None, batch_size=1024, mode_seed=30001,
                        label="Controller evaluation", zeta=None, progress=False):
    """Run complete shared tensor episodes; MSE, ES and entropic risk use all pooled path losses.

    Trade tapes retain the executed quantities. Batch size and order are part of the sampled-policy RNG contract, as in native evaluation.
    Timing includes transfers, controller calls, ledger execution and tapes;
    bank generation/training are external costs and are not silently zeroed.
    Controller inputs are read-only. Clone ledger.positions when constructing
    targets by in-place adjustment; only the environment updates its ledger.
    """
    if batch_size < 1 or len(bank.spot) == 0:
        raise ValueError("positive batch size and a nonempty market bank required")
    device = torch.device(device or bank.spot.device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started, batches = time.perf_counter(), []
    if progress:
        print(f"{label}: {len(bank.spot)} paths, {bank.config.n_decisions} decisions, batch {batch_size}, "
              f"mode seed {mode_seed}, device {device}", flush=True)
    keys = ("terminal_loss", "transaction_cost", "turnover", "tickets", "constraint_violations", "positions")
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(mode_seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(mode_seed)
        for offset in range(0, len(bank.spot), batch_size):
            sample = bank_to(bank_subset(bank, slice(offset, offset+batch_size)), device)
            violations = torch.zeros(len(sample.spot), dtype=torch.long, device=device)
            result = run_episode(controller, sample, record_positions=True)
            # Successful environment steps reject all infeasible trades; the
            # independent NumPy tape audit checks feasibility again offline.
            result["constraint_violations"] = violations
            batches.append({key: result[key].cpu() for key in keys})
            if progress:
                completed = min(offset+batch_size, len(bank.spot))
                elapsed = time.perf_counter()-started
                print(f"Evaluation {completed}/{len(bank.spot)} paths; {elapsed:.2f}s; "
                      f"{completed/elapsed:.2f} paths/s; ETA {elapsed*(len(bank.spot)-completed)/completed:.2f}s", flush=True)
    raw = {key: torch.cat([batch[key] for batch in batches]) for key in keys}
    loss = raw["terminal_loss"]
    if not torch.isfinite(loss).all():
        raise FloatingPointError("nonfinite controller terminal losses")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    alpha = bank.config.risk.alpha
    metrics = dict(label=label, paths=len(loss), mean_loss=float(loss.double().mean()),
        mse=float(loss.double().square().mean()),
        rmse=float(loss.double().square().mean().sqrt()),
        risk_alpha=alpha, expected_shortfall=empirical_es(loss, alpha),
        es95=empirical_es(loss, .95), es99=empirical_es(loss, .99),
        transaction_cost_mean=float(raw["transaction_cost"].double().mean()),
        turnover_mean_by_asset=raw["turnover"].double().mean(0).tolist(),
        turnover_mean_total=float(raw["turnover"].double().sum(-1).mean()), tickets_mean=float(raw["tickets"].double().mean()),
        constraint_violations=int(raw["constraint_violations"].sum()), mode_seed=mode_seed, evaluation_batch_size=batch_size,
        action_selection=getattr(controller, "action_selection", "caller_defined"),
        effective_tail_paths=(1-alpha)*len(loss),
        effective_tail_paths95=.05*len(loss), effective_tail_paths99=.01*len(loss),
        evaluation_seconds=time.perf_counter()-started, device=str(device),
        timing_scope="Whole frozen evaluation: transfers, decisions, tensor ledger and CPU tapes; excludes bank generation/training")
    risk = bank.config.risk
    metrics["objective"] = risk.objective
    metrics["objective_value"] = (metrics["mse"] if risk.objective == "mse"
                                  else float(risk.entropic_risk(loss.double())) if risk.objective == "entropy"
                                  else metrics["expected_shortfall"])
    if zeta is not None and risk.objective != "mse":
        threshold = float(zeta.detach().cpu()) if isinstance(zeta, torch.Tensor) else float(zeta)
        metrics.update(zeta=threshold, ru_at_training_zeta=float(risk.loss(loss.double(), threshold).mean()))
    return metrics, raw
