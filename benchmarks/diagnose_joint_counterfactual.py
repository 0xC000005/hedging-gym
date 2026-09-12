"""Training-only mode/sizing and gradient-noise diagnostics after joint updates."""
import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import torch

from benchmarks.qualify_counterfactual import _load_policy
from benchmarks.qualify_policies import load_bank
from hedging_gym.baselines._shared.training import _report, _sync
from hedging_gym.environment.finance import (
    BANK_FIELDS,
    MarketBank,
    bank_subset,
    bank_to,
)
from hedging_gym.evaluation import empirical_es
from hedging_gym.extensions.counterfactual import (
    _counterfactual_rollout,
    _modes,
    counterfactual_rollout,
)
from hedging_gym.extensions.joint_counterfactual import joint_objective


@torch.no_grad()
def precision_audit(policy, bank, seed):
    """Same marks and uniform draws; count TRADE modes yielding exact zero trades."""
    results = {}
    uniforms = torch.rand((len(bank.spot), bank.config.n_steps), dtype=torch.float64,
        device=bank.spot.device, generator=torch.Generator(device=bank.spot.device).manual_seed(seed+950003))
    for dtype in (torch.float32, torch.float64):
        local = MarketBank(*(getattr(bank, key).to(dtype) for key in BANK_FIELDS), bank.config)
        converted = deepcopy(policy).to(dtype=dtype)
        branches = _counterfactual_rollout(converted, local, time_index=0, algorithm="sampled",
            uniforms=uniforms, retain_tape=True, retain_scores=True)
        positions = branches["positions"][:, 0]
        previous = torch.cat((positions.new_zeros((len(positions), 1, bank.config.n_assets)),
                              positions[:, :-1]), 1)
        actual_trade = positions != previous
        modes = torch.stack([_modes(converted.discrete(branches["histories"][date, :, 0]).softmax(-1),
                                      uniforms[:, date]) for date in range(bank.config.n_steps)], 1)
        proposed_trade = converted.trade_mask[modes]
        results[str(dtype)] = dict(es95=empirical_es(branches["terminal_losses"][:, 0], .95),
            proposed_trade_modes_per_path=float(proposed_trade.sum((1, 2)).double().mean()),
            actual_decision_tickets_per_path=float(actual_trade.sum((1, 2)).double().mean()),
            trade_modes_with_exact_zero_delta_per_path=float((proposed_trade & ~actual_trade).sum((1, 2)).double().mean()),
            hold_modes_with_nonzero_delta=int((~proposed_trade & actual_trade).sum()),
            scope="Decision-date tickets only, excludes liquidation; same market marks and common float64 uniforms")
    return results


def gradient_noise(policy, bank, zeta, seed, repeats=32, batch_size=64, score_scope="trajectory"):
    """Variance at fixed parameters; independent minibatches and matched work.

    The sample mean is not a known true gradient. Report trace variance and
    estimated mean-gradient norm separately; do not call this a signal oracle.
    """
    policy.requires_grad_(True)
    policy.value.requires_grad_(False)
    parameters = [*policy.discrete.parameters(), *policy.continuous.parameters()]
    split = sum(value.numel() for value in policy.discrete.parameters())
    output = {}
    for algorithm in ("all_mode", "sampled"):
        generator = torch.Generator(device=bank.spot.device).manual_seed(seed+710003)
        dates = torch.Generator().manual_seed(seed+810003)
        gradients, work = [], 0
        started = time.perf_counter()
        for _ in range(repeats):
            date = int(torch.randint(bank.config.n_steps, (), generator=dates))
            roots = batch_size if algorithm == "all_mode" else math.ceil(batch_size*(
                date+policy.n_modes*(bank.config.n_steps-date))/bank.config.n_steps)
            indices = torch.randint(len(bank.spot), (roots,), device=bank.spot.device, generator=generator)
            objective, branches, _ = joint_objective(policy, bank_subset(bank, indices),
                time_index=date, zeta=zeta, algorithm=algorithm, generator=generator,
                score_scope=score_scope)
            values = torch.autograd.grad(objective, parameters)
            gradients.append(torch.cat([value.flatten().detach() for value in values]))
            work += branches["ledger_steps"]
        _sync(bank.spot.device)
        samples = torch.stack(gradients).double()
        elapsed = time.perf_counter()-started
        output[algorithm] = dict(repeats=repeats, decision_ledger_steps=work,
            seconds=elapsed, trace_variance=float(samples.var(0, unbiased=True).sum()),
            categorical_trace_variance=float(samples[:, :split].var(0, unbiased=True).sum()),
            continuous_trace_variance=float(samples[:, split:].var(0, unbiased=True).sum()),
            estimated_mean_gradient_norm=float(samples.mean(0).norm()))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--gradient-repeats", type=int, default=32)
    parser.add_argument("--precision-only", action="store_true")
    parser.add_argument("--controls", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    report = json.loads((args.run/"comparison.json").read_text())
    bank = bank_to(load_bank(Path(report["arguments"]["source_dir"])/"train_bank.pt",
                            report["arguments"]["preset"]), args.device)
    output = dict(scope="Training-only diagnosis; no final paths, no oracle gradient", seeds={})
    if args.precision_only:
        output["scope"] = "Frozen-policy numerical mechanism diagnosis; first4096training paths, no new finance rules"
        sample = bank_subset(bank, slice(0,4096))
        for seed_text,item in report["seeds"].items():
            seed = int(seed_text)
            paths = dict(source=Path(item["source_checkpoint"]))
            paths.update({name:args.run/f"seed-{seed}/{name}/latest.pt" for name in ("all_mode","sampled")})
            if args.controls:
                paths["full_hpo"] = args.controls/f"seed-{seed}/full_hpo/latest.pt"
            rows = {}
            for name,path in paths.items():
                policy = _load_policy(torch.load(path,map_location="cpu",weights_only=False),
                    bank.config,args.device,bank.spot.dtype)
                rows[name] = precision_audit(policy,sample,seed)
            output["seeds"][seed_text] = rows
            _report("counterfactual_precision_diagnosis",seed=seed,results=rows)
        (args.run/"precision-diagnostics.json").write_text(json.dumps(output,indent=2)+"\n")
        return
    for seed_text, item in report["seeds"].items():
        seed = int(seed_text)
        saved = torch.load(item["source_checkpoint"], map_location="cpu", weights_only=False)
        source = _load_policy(saved, bank.config, args.device, bank.spot.dtype)
        threshold = bank.spot.new_tensor(item["fixed_zeta"])
        date = bank.config.n_steps//2
        uniforms = torch.rand((1024, bank.config.n_steps), device=args.device,
            generator=torch.Generator(device=args.device).manual_seed(seed+910003))
        branches = counterfactual_rollout(source, bank_subset(bank, slice(0,1024)),
            time_index=date, uniforms=uniforms, retain_tape=True)
        observed = branches["observed"]
        holdings = branches["positions"][:, 0, date-1]
        with torch.no_grad():
            source_probabilities = source.discrete(observed).softmax(-1)
            source_targets = source.candidates(observed, holdings,
                bank.config.execution.holding_lower, bank.config.execution.holding_upper)
        seed_result = dict(source_gradient_noise={scope: gradient_noise(source, bank, threshold, seed,
            repeats=args.gradient_repeats, score_scope=scope) for scope in ("sampled_date", "trajectory")},
            common_source_prefix={})
        for algorithm in ("source", "all_mode", "sampled", "full_hpo"):
            if algorithm not in item["results"]:
                continue
            policy = source if algorithm == "source" else _load_policy(torch.load(
                args.run/f"seed-{seed}"/algorithm/"latest.pt", map_location="cpu", weights_only=False),
                bank.config, args.device, bank.spot.dtype)
            with torch.no_grad():
                probabilities = policy.discrete(observed).softmax(-1)
                targets = policy.candidates(observed, holdings,
                    bank.config.execution.holding_lower, bank.config.execution.holding_upper)
                mode = source_probabilities.argmax(-1)
                rows = torch.arange(len(mode), device=args.device)
                stats = dict(root_date=date, paths=len(mode),
                    mean_entropy=float(-(probabilities*probabilities.clamp_min(1e-30).log()).sum(-1).mean()),
                    mean_largest_mode_probability=float(probabilities.max(-1).values.mean()),
                    changed_greedy_mode_fraction=float((probabilities.argmax(-1)!=mode).float().mean()),
                    mean_mode_probabilities=probabilities.mean(0).tolist(),
                    incumbent_mode_target_rms_change=float((targets[rows,mode]-source_targets[rows,mode]).square().mean().sqrt()),
                    scope="All policies evaluated at identical incumbent-prefix states; not their own visitation distributions")
            seed_result["common_source_prefix"][algorithm] = stats
        output["seeds"][seed_text] = seed_result
        (args.run/"training-diagnostics.json").write_text(json.dumps(output,indent=2)+"\n")
        _report("joint_diagnostics", seed=seed, **seed_result)


if __name__ == "__main__":
    main()
