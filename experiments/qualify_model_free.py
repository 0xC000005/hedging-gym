"""Source-reasoned model-free training on saved common training/development banks.

Uses native replay intensity and separate conservative actor/tail learning rates.
Frozen-policy calibration and paired action checks are reported at warmup.
Run to the warmup boundary and inspect those diagnostics before a longer run.
This is diagnostic policy training, not calibrated-baseline qualification or a
published-benchmark reproduction. No diagnostic heuristic blocks continuation.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from hedging_gym.config import config_from_dict
from hedging_gym.evaluation import evaluate_controller
from hedging_gym.finance import BANK_FIELDS, MarketBank, bank_subset, bank_to, numpy_ledger
from methods.controllers import policy_controller
from methods.model_free import DistributionalCritic, collect_episodes, train_model_free
from methods.policies import DirectDHPolicy


def verify_result(directory, bank, *, device="cpu"):
    """Reload the saved actor and independently reconstruct its saved cash tape."""
    started = time.perf_counter()
    saved = torch.load(directory/"policy.pt", map_location="cpu", weights_only=False)
    step = saved["metadata"]["options"]["updates"]
    tape = torch.load(directory/f"development-{step}.pt", map_location="cpu", weights_only=False)
    policy = DirectDHPolicy(bank.config, hidden=saved["metadata"]["options"]["hidden"]).to(
        device=device, dtype=bank.spot.dtype)
    policy.load_state_dict(saved["policy"])
    metrics, reloaded = evaluate_controller(policy_controller(policy), bank,
        device=device, batch_size=1024, zeta=saved["metadata"]["zeta"])
    reconstructed = numpy_ledger(bank.marks.cpu().numpy(), tape["positions"].cpu().numpy(),
        bank.liability[:, 0].cpu().numpy(), bank.liability[:, -1].cpu().numpy(), bank.config)
    difference = reconstructed["terminal_loss"]-tape["terminal_loss"].cpu().numpy()
    result = dict(step=step, paths=len(bank.spot), reloaded_es95=metrics["es95"],
        maximum_reload_loss_difference=float((reloaded["terminal_loss"]-tape["terminal_loss"]).abs().max()),
        maximum_numpy_ledger_loss_difference=float(np.abs(difference).max()),
        all_losses_finite=bool(np.isfinite(reconstructed["terminal_loss"]).all()),
        constraint_violations=metrics["constraint_violations"],
        verification_seconds=time.perf_counter()-started)
    (directory/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


@torch.no_grad()
def diagnose(checkpoint, bank, *, device="cpu", probe_paths=2048):
    """Read-only fixed-policy calibration and paired first-action intervention.

    The intervention changes one legal first action, then follows the saved
    policy through the detached common ledger. All variants use the same paths.
    It tests critic action directions without simulator gradients or training.
    """
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    config, options = bank.config, saved["options"]
    bank = bank_to(bank, device)
    actor = DirectDHPolicy(config, hidden=options["hidden"]).to(device=device, dtype=bank.spot.dtype)
    actor.load_state_dict(saved["actor"])
    critic = DistributionalCritic(actor.feature_dim, config.n_assets, options["hidden"],
        quantiles=options["quantiles"], tail_threshold=options["tail_threshold"]).to(device=device, dtype=bank.spot.dtype)
    critic.load_state_dict(saved["critic"])
    transitions, losses, initial = collect_episodes(actor, bank, n_step=config.n_steps)
    actions = transitions[1].reshape(len(losses), config.n_steps, -1)[:, 0]
    scale, zeta = saved["return_scale"], saved["zeta"]
    distribution = critic.distribution(initial, actions)*scale
    raw, tail_scale, tail_shape = critic(initial[:1], actions[:1])
    raw = raw.sort(-1).values*scale
    q_index = config.risk.alpha*options["quantiles"]-.5
    left = int(q_index)
    q95 = distribution[:, left] + (q_index-left)*(distribution[:, left+1]-distribution[:, left])
    body_q95 = raw[:, left]+(q_index-left)*(raw[:, left+1]-raw[:, left])
    predicted_ru = scale*critic.expected_ru(initial, actions, zeta/scale, config.risk.alpha).mean()
    realized_ru = config.risk.loss(losses, zeta).mean()
    observed_var = torch.quantile(losses, config.risk.alpha)
    # Scalar optimum isolates Huber smoothing from neural/Bellman errors.
    normalized = losses/scale
    low, high = normalized.min(), normalized.max()
    kappa = options.get("quantile_kappa", 1.)
    for _ in range(50):
        middle = (low+high)/2
        error = normalized-middle
        derivative = (error.sign() if kappa == 0 else error.clamp(-kappa, kappa))
        equation = (torch.where(error > 0, config.risk.alpha, 1-config.risk.alpha)*derivative).mean()
        low, high = (middle, high) if equation > 0 else (low, middle)
    huber_quantile = (low+high)/2*scale
    probe = bank_subset(bank, slice(0, min(probe_paths, len(losses))))
    first = actions[0].clone()
    lower = first.new_tensor(config.execution.vector("holding_lower", config.n_assets))
    upper = first.new_tensor(config.execution.vector("holding_upper", config.n_assets))
    exact, predicted = [], []
    for asset in range(config.n_assets):
        minus, plus = first.clone(), first.clone()
        distance = .02*(upper[asset]-lower[asset])
        minus[asset] = (minus[asset]-distance).clamp(lower[asset], upper[asset])
        plus[asset] = (plus[asset]+distance).clamp(lower[asset], upper[asset])
        step = plus[asset]-minus[asset]
        actual_values, critic_values = [], []
        for action in (minus, plus):
            _, changed_losses, _ = collect_episodes(actor, probe, first_action=action)
            actual_values.append(config.risk.loss(changed_losses, zeta).mean())
            critic_values.append(scale*critic.expected_ru(initial[:1], action[None],
                                                          zeta/scale, config.risk.alpha).mean())
        exact.append((actual_values[1]-actual_values[0])/step)
        predicted.append((critic_values[1]-critic_values[0])/step)
    exact, predicted = torch.stack(exact), torch.stack(predicted)
    cosine = (exact*predicted).sum()/(exact.norm()*predicted.norm()).clamp_min(1e-12)
    exceedance = float((losses > q95).float().mean())
    ru_relative_error = float((predicted_ru-realized_ru).abs()/realized_ru.abs().clamp_min(1e-6))
    return dict(step=saved["step"], initial_action=first.tolist(),
        zeta=float(zeta), zeta_exceedance=float((losses > zeta).float().mean()),
        body_q95=float(body_q95), spliced_q95=float(q95.mean()),
        gpd_scale_money=None if tail_scale is None else float(tail_scale*scale),
        gpd_shape=None if tail_shape is None else float(tail_shape),
        predicted_ru=float(predicted_ru), realized_ru=float(realized_ru),
        ru_relative_error=ru_relative_error, q95_exceedance=exceedance,
        empirical_var=float(observed_var), huber_optimum=float(huber_quantile),
        huber_optimum_exceedance=float((losses > huber_quantile).float().mean()),
        paired_action_gradient=exact.tolist(), critic_action_gradient=predicted.tolist(),
        action_gradient_cosine=float(cosine), probe_paths=len(probe.spot),
        initial_sanity_screen_passed=(.02 <= exceedance <= .08 and ru_relative_error <= .35 and float(cosine) > 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="directory containing train_bank.pt and development_bank.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("hull_rl", "exdrl"), required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--stage-updates", type=int, default=50)
    parser.add_argument("--actor-warmup-updates", type=int, default=50)
    parser.add_argument("--actor-update-period", type=int, default=5)
    parser.add_argument("--quantile-kappa", type=float, default=.01)
    parser.add_argument("--quantiles", type=int, default=399)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-5)
    parser.add_argument("--critic-learning-rate", type=float, default=1e-4)
    parser.add_argument("--tail-learning-rate", type=float, default=1e-6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--reward-labels", choices=("dense", "terminal"), default="dense")
    parser.add_argument("--resume", type=Path,
                        help="trusted local complete training checkpoint")
    parser.add_argument("--extend-frozen-warmup", action="store_true",
                        help="explicitly extend warmup only while the resumed actor has never updated")
    args = parser.parse_args()
    if min(args.stage_updates, args.updates, args.threads) < 1 or args.actor_warmup_updates < 1:
        parser.error("qualification requires positive work and a frozen-actor warmup")
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def bank(name):
        saved = torch.load(args.run_dir/name, map_location="cpu", weights_only=False)
        return MarketBank(*(saved[key] for key in BANK_FIELDS), config_from_dict(saved["config"]))

    training, development = bank("train_bank.pt"), bank("development_bank.pt")
    # Eight collected episodes * dates *32 samples per insertion / batch128.
    gradient_steps = 2*training.config.n_steps
    options = dict(
        seed=args.seed, updates=args.updates, batch_size=128, collection_batch_size=8,
        hidden=(64,64), device=args.device, gradient_steps=gradient_steps,
        learning_rate=args.actor_learning_rate, critic_learning_rate=args.critic_learning_rate,
        tail_learning_rate=args.tail_learning_rate, tail_policy_actions=True,
        quantile_kappa=args.quantile_kappa, quantiles=args.quantiles, tail_threshold=.95,
        actor_warmup_updates=args.actor_warmup_updates, actor_update_period=args.actor_update_period,
        action_gradient_clip=1., dense_rewards=args.reward_labels == "dense", checkpoint_every=50,
        checkpoint_path=args.output_dir/"latest.pt")
    start = 0 if args.resume is None else torch.load(args.resume, weights_only=False, map_location="cpu")["step"]
    records = []
    previous_report = {}
    report_path = args.output_dir/"report.json"
    if args.resume is not None and report_path.exists():
        previous_report = json.loads(report_path.read_text())
        records = previous_report["records"]
    calibration = bank_subset(training, slice(0, min(4096, len(training.spot))))
    initial_resume_diagnostic = previous_report.get("initial_resume_diagnostic")
    if args.resume is not None and start == args.actor_warmup_updates:
        previous_check = diagnose(args.resume, calibration, device=args.device)
        initial_resume_diagnostic = previous_check
        print(json.dumps(dict(training_only_initial_check=previous_check,
            interpretation="diagnostic heuristic only; this is not calibration certification")), flush=True)
    status = "completed"
    resume = args.resume
    while start < args.updates:
        stop = min(start+args.stage_updates, args.updates)
        # Always inspect the fixed policy before the first actor update.
        if start < args.actor_warmup_updates < stop:
            stop = args.actor_warmup_updates
        options["updates"] = stop
        policy, metadata = train_model_free(args.method, training, **options,
            resume_from=resume, extend_frozen_warmup=args.extend_frozen_warmup)
        metrics, tape = evaluate_controller(policy_controller(policy), development,
            device=args.device, batch_size=1024, zeta=metadata["zeta"], progress=True,
            label=f"{args.method}: development qualification step {stop}")
        diagnostic = diagnose(options["checkpoint_path"], development, device=args.device)
        training_diagnostic = (diagnose(options["checkpoint_path"], calibration, device=args.device)
                               if stop == args.actor_warmup_updates else None)
        records.append(dict(step=stop, metrics=metrics, diagnostic=diagnostic,
                            training_calibration=training_diagnostic))
        torch.save(dict(policy=policy.state_dict(), config=asdict(training.config), metadata=metadata),
                   args.output_dir/"policy.pt")
        torch.save(tape, args.output_dir/f"development-{stop}.pt")
        warnings = [f"step {record['step']}: frozen critic missed the initial sanity screen; not calibration certification"
                    for record in records if record.get("training_calibration") is not None
                    and not record["training_calibration"].get("initial_sanity_screen_passed",
                              record["training_calibration"].get("permits_actor_training", False))]
        if initial_resume_diagnostic is not None and not initial_resume_diagnostic["initial_sanity_screen_passed"]:
            warnings.append("resumed frozen critic missed the initial sanity screen; continuation is a diagnostic policy-training attempt")
        report = dict(scope="diagnostic policy training, not calibrated-baseline qualification", metadata=metadata,
            metrics=metrics, records=records, status=status,
            warnings=warnings,
            continuation_rule="report-only training diagnostics; explicit fixed training budget; development is report-only",
            resume_from=None if args.resume is None else str(args.resume),
            initial_resume_diagnostic=initial_resume_diagnostic,
            extended_frozen_warmup=args.extend_frozen_warmup,
            training_bank=str(args.run_dir/"train_bank.pt"), development_bank=str(args.run_dir/"development_bank.pt"))
        report_path.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(dict(status=status, **records[-1])), flush=True)
        if status != "completed" or stop == args.updates:
            print(json.dumps(dict(verification=verify_result(args.output_dir, development, device=args.device))), flush=True)
            break
        start = stop
        resume = options["checkpoint_path"]


if __name__ == "__main__":
    main()
