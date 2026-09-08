"""Reassess saved CEM choices; no training or changes to deployed search.

Replay captures the original selection scores without changing RNG consumption.
New futures evaluate fixed choices against the original policy at the SAME
historical ledger. The quantity is conditional global-threshold RU, not a
new per-node CVaR or a full repeatedly-searching policy's ES.
"""

import argparse
import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from experiments.qualify_policies import load_bank
from hedging_gym import finance
from hedging_gym.evaluation import evaluate_controller
from methods.hybrid import HybridPolicy
from methods.planning import RolloutPlanner, _conditional_paths, _repeat_ledger
from methods.training import _report

DATES = (0, 9, 19, 29)
INDICES = tuple(np.linspace(0, 511, 16, dtype=int).tolist())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class CapturePlanner(RolloutPlanner):
    """Observe existing score calls; never insert candidates or consume RNG."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def _score(self, targets, ledger, time_index, config, paths):
        score = super()._score(targets, ledger, time_index, config, paths)
        if self.capture_indices:
            self.trials.append(
                (
                    targets[self.capture_indices].detach().cpu(),
                    score[self.capture_indices].detach().cpu(),
                )
            )
            self.capture_paths = paths
        return score

    @torch.no_grad()
    def __call__(self, observed, ledger, time_index, config):
        offset = (self.calls // config.n_steps) * 128
        self.capture_indices = (
            [i - offset for i in INDICES if offset <= i < offset + len(observed)]
            if time_index in DATES
            else []
        )
        self.trials = []
        chosen = super().__call__(observed, ledger, time_index, config)
        for position, local in enumerate(self.capture_indices):
            actions = torch.cat([t[position] for t, s in self.trials])
            scores = torch.cat([s[position] for t, s in self.trials])
            selected = chosen[local].cpu()
            selected_score = scores[(actions == selected).all(-1)].min()
            proposal = self.trials[0][0][position, 1]
            policy_score = self.trials[0][1][position, 1]

            def take(value, index=local):
                return (
                    value.reshape(len(observed), self.scenarios, *value.shape[1:])[
                        index
                    ]
                    .detach()
                    .cpu()
                )

            states, marks, payoff = self.capture_paths
            self.records.append(
                {
                    "path": offset + local,
                    "date": time_index,
                    "observed": observed[local].detach().cpu(),
                    "ledger": {
                        f.name: getattr(ledger, f.name)[local].detach().cpu()
                        for f in fields(ledger)
                    },
                    "chosen": selected,
                    "proposal": proposal,
                    "selection_scores": torch.stack((policy_score, selected_score)),
                    "selection_paths": (
                        [(take(s), take(v)) for s, v in states],
                        [take(m) for m in marks],
                        take(payoff),
                    ),
                    "selection_tie": bool(selected_score == policy_score),
                    "chose_hold": bool(
                        torch.equal(selected, ledger.positions[local].cpu())
                    ),
                }
            )
        return chosen


def load_case(args):
    bank = finance.bank_subset(
        load_bank(args.source / "development_bank.pt", args.preset), slice(0, 512)
    )
    folder = args.source / "policies"
    if args.preset != "basic":
        folder = folder / args.preset
    checkpoint = (
        folder / "hpo-seed7" / f"step-{3000 if args.preset == 'basic' else 1000}.pt"
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy = HybridPolicy(bank.config, hidden=saved["options"]["hidden"]).to(
        args.device
    )
    policy.load_state_dict(saved["policy"])
    policy.eval()
    old = (
        args.source
        / "search"
        / ("basic64-batched" if args.preset == "basic" else "fixed64-batched")
    )
    return bank, policy, float(saved["zeta"]), old


def replay(args):
    output = args.output / args.preset
    output.mkdir(parents=True, exist_ok=True)
    if (output / "capture.pt").exists():
        return
    bank, policy, zeta, old = load_case(args)
    planner = CapturePlanner(
        policy,
        zeta=zeta,
        candidates=16,
        scenarios=64,
        iterations=2,
        seed=200010,
        guided=True,
        progress=True,
    )
    metrics, tape = evaluate_controller(
        planner,
        bank,
        device=args.device,
        batch_size=128,
        label="original guided CEM replay",
        progress=True,
    )
    reference = torch.load(old / "guided-tape.pt", weights_only=False)
    errors = {
        key: float((value - reference[key]).abs().max()) for key, value in tape.items()
    }
    if any(errors.values()):
        raise ValueError(f"original replay differs: {errors}")
    torch.save({"records": planner.records, "zeta": zeta}, output / "capture.pt")
    write_json(
        output / "replay.json",
        {
            "metrics": metrics,
            "errors": errors,
            "records": len(planner.records),
            "planning": planner.metadata(),
        },
    )
    _report(
        "replay_verified",
        preset=args.preset,
        records=len(planner.records),
        errors=errors,
    )


@torch.no_grad()
def conditional_outcomes(policy, ledger, targets, date, config, paths):
    """Expose per-future outcomes using the shared ledger, plus audit positions."""
    states, marks, payoff = paths
    batch, candidates, assets = targets.shape
    scenarios = len(payoff) // batch
    state = _repeat_ledger(ledger, candidates, scenarios)

    def branch(value):
        value = value.reshape(batch, scenarios, *value.shape[1:])
        return (
            value[:, None]
            .expand(-1, candidates, -1, *value.shape[2:])
            .reshape(batch * candidates * scenarios, *value.shape[2:])
        )

    target = targets[:, :, None].expand(-1, -1, scenarios, -1).reshape(-1, assets)
    audit_indices = torch.arange(batch * candidates, device=targets.device) * scenarios
    positions = []
    for offset, now in enumerate(range(date, config.n_steps)):
        mid = branch(marks[offset])
        if offset:
            spot, variance = (branch(v) for v in states[offset])
            observed = finance.observation_from_state(
                spot, variance, now, state, mid, config
            )
            target = policy(
                observed,
                state.positions,
                config.execution.holding_lower,
                config.execution.holding_upper,
                deterministic=True,
            ).target_holdings
        positions.append(target[audit_indices].clone())
        state = finance.trade_step(state, target, mid, config)
    result = finance.liquidate(state, branch(marks[-1]), branch(payoff), config)
    return (
        result["terminal_loss"].reshape(batch, candidates, scenarios),
        result["transaction_cost"].reshape(batch, candidates, scenarios),
        torch.stack(positions, 1).cpu(),
    )


def root_tensors(records, device):
    observed = torch.stack([r["observed"] for r in records]).to(device)
    ledger = finance.LedgerState(
        **{
            k: torch.stack([r["ledger"][k] for r in records]).to(device)
            for k in records[0]["ledger"]
        }
    )
    targets = torch.stack(
        [torch.stack((r["proposal"], r["chosen"])) for r in records]
    ).to(device)
    return observed, ledger, targets


@torch.no_grad()
def probe(args):
    output = args.output / args.preset
    capture = torch.load(output / "capture.pt", weights_only=False)
    bank, policy, zeta, old = load_case(args)
    original_positions = torch.load(old / "guided-tape.pt", weights_only=False)[
        "positions"
    ]
    records = capture["records"]
    started = time.perf_counter()
    completed = 0
    for date in DATES:
        date_records = [r for r in records if r["date"] == date]
        for offset in range(0, len(date_records), 4):
            destination = output / f"probe-{date}-{offset}.pt"
            if destination.exists():
                completed += 4
                continue
            block = date_records[offset : offset + 4]
            observed, ledger, targets = root_tensors(block, args.device)
            old_paths = (
                [
                    tuple(
                        torch.cat([r["selection_paths"][0][t][k] for r in block]).to(
                            args.device
                        )
                        for k in (0, 1)
                    )
                    for t in range(bank.config.n_steps - date + 1)
                ],
                [
                    torch.cat([r["selection_paths"][1][t] for r in block]).to(
                        args.device
                    )
                    for t in range(bank.config.n_steps - date + 1)
                ],
                torch.cat([r["selection_paths"][2] for r in block]).to(args.device),
            )
            old_losses, _, _ = conditional_outcomes(
                policy, ledger, targets, date, bank.config, old_paths
            )
            reference_scores = torch.stack(
                [r["selection_scores"] for r in block]
            ).double()
            rescored = bank.config.risk.loss(old_losses, zeta).mean(-1).double().cpu()
            score_error = float((rescored - reference_scores).abs().max())
            if not np.isfinite(score_error) or score_error > 2e-6:
                raise ValueError(
                    f"original 64-scenario scoring parity failed: {score_error}"
                )
            # A separate seed per fixed block permits resume without RNG drift.
            generator = torch.Generator(device=args.device).manual_seed(
                910020 + date * 100 + offset
            )
            chunks = []
            cash_error = 0.0
            for start in range(0, args.futures, 1024):
                count = min(1024, args.futures - start)
                paths = _conditional_paths(
                    observed, date, bank.config, count, generator
                )
                losses, costs, positions = conditional_outcomes(
                    policy, ledger, targets, date, bank.config, paths
                )
                ru = bank.config.risk.loss(losses, zeta)
                chunks.append(
                    {"loss": losses.cpu(), "cost": costs.cpu(), "ru": ru.cpu()}
                )
                if start == 0:
                    _, marks, payoff = paths
                    # Complete the old prefix with the first new future per root.
                    # The independent NumPy ledger starts at the original premium.
                    marks = (
                        torch.stack(marks, 1)
                        .cpu()
                        .reshape(len(block), count, -1, bank.config.n_assets)
                    )
                    for i, r in enumerate(block):
                        for action in range(2):
                            all_marks = torch.cat(
                                (bank.marks[r["path"], :date], marks[i, 0])
                            )
                            all_positions = torch.cat(
                                (
                                    original_positions[r["path"], :date],
                                    positions[i * 2 + action],
                                )
                            )
                            independent = finance.numpy_ledger(
                                all_marks[None].numpy(),
                                all_positions[None].numpy(),
                                bank.liability[r["path"], 0].numpy(),
                                payoff.reshape(len(block), count)[i, 0].cpu().numpy(),
                                bank.config,
                            )
                            error = abs(
                                float(independent["terminal_loss"][0])
                                - float(losses[i, action, 0])
                            )
                            cash_error = max(cash_error, error)
            if not np.isfinite(cash_error) or cash_error > 2e-6:
                raise ValueError(f"conditional ledger audit failed: {cash_error}")
            outcomes = {k: torch.cat([c[k] for c in chunks], dim=-1) for k in chunks[0]}
            torch.save(
                {
                    "records": block,
                    "outcomes": outcomes,
                    "cash_error": cash_error,
                    "selection_score_error": score_error,
                    "futures": args.futures,
                    "seed": 910020 + date * 100 + offset,
                    "zeta": zeta,
                },
                destination,
            )
            completed += len(block)
            elapsed = time.perf_counter() - started
            _report(
                "rerank_progress",
                preset=args.preset,
                roots=completed,
                total=len(records),
                futures=args.futures,
                elapsed_seconds=elapsed,
                eta_seconds=elapsed * (len(records) - completed) / completed,
            )


def summarize(args):
    rows = []
    max_cash_error = 0.0
    max_score_error = 0.0
    for preset in ("basic", "operational_fixed"):
        for date in DATES:
            for offset in range(0, 16, 4):
                saved = torch.load(
                    args.output / preset / f"probe-{date}-{offset}.pt",
                    weights_only=False,
                )
                out = saved["outcomes"]
                expected = {(path, date) for path in INDICES[offset : offset + 4]}
                actual = {(r["path"], r["date"]) for r in saved["records"]}
                if (
                    actual != expected
                    or saved["futures"] != args.futures
                    or any(
                        value.shape != (4, 2, args.futures) for value in out.values()
                    )
                ):
                    raise ValueError(
                        f"incomplete probe: {preset}, date {date}, offset {offset}"
                    )
                for i, r in enumerate(saved["records"]):
                    delta = (
                        out["ru"][i, 1].double() - out["ru"][i, 0].double()
                    ).numpy()
                    row = {
                        "preset": preset,
                        "path": r["path"],
                        "date": date,
                        "selection_delta": float(
                            r["selection_scores"][1] - r["selection_scores"][0]
                        ),
                        "tie": r["selection_tie"],
                        "hold": r["chose_hold"],
                        "changed": not torch.equal(r["chosen"], r["proposal"]),
                        "verification_delta": float(delta.mean()),
                        "paired_se": float(delta.std(ddof=1) / np.sqrt(len(delta))),
                        "tail_counts": (out["loss"][i] > saved["zeta"])
                        .sum(-1)
                        .tolist(),
                        "mean_loss_delta": float(
                            (
                                out["loss"][i, 1].double() - out["loss"][i, 0].double()
                            ).mean()
                        ),
                        "mean_cost_delta": float(
                            (
                                out["cost"][i, 1].double() - out["cost"][i, 0].double()
                            ).mean()
                        ),
                    }
                    rows.append(row)
                max_cash_error = max(max_cash_error, saved["cash_error"])
                max_score_error = max(max_score_error, saved["selection_score_error"])
    summary = {}
    for preset in ("basic", "operational_fixed"):
        subset = [r for r in rows if r["preset"] == preset]
        summary[preset] = {
            "roots": len(subset),
            "changed": sum(r["changed"] for r in subset),
            "ties": sum(r["tie"] for r in subset),
            "tied_hold": sum(r["tie"] and r["hold"] for r in subset),
            "mean_selection_delta": float(
                np.mean([r["selection_delta"] for r in subset])
            ),
            "mean_verification_delta": float(
                np.mean([r["verification_delta"] for r in subset])
            ),
            "two_se_worse": sum(
                r["verification_delta"] > 2 * r["paired_se"] for r in subset
            ),
            "two_se_better": sum(
                r["verification_delta"] < -2 * r["paired_se"] for r in subset
            ),
        }
    result = {
        "summary": summary,
        "roots": rows,
        "max_cash_error": max_cash_error,
        "max_selection_score_error": max_score_error,
        "futures_per_root": args.futures,
        "scope": "Fixed selected actions vs policy at same searched ledger; conditional RU with frozen feedback, not deployed-policy ES",
        "uncertainty": "Root paired Monte Carlo SE; two-SE counts descriptive and unadjusted, not simultaneous significance",
    }
    write_json(args.output / "summary.json", result)
    _report("rerank_summary", summary=summary, max_cash_error=max_cash_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("replay", "probe", "summarize"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preset", choices=("basic", "operational_fixed"), default="basic"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--futures", type=int, default=8192)
    args = parser.parse_args()
    torch.set_num_threads(2)
    _report(
        "rerank_start",
        options={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    )
    {"replay": replay, "probe": probe, "summarize": summarize}[args.stage](args)


if __name__ == "__main__":
    main()
