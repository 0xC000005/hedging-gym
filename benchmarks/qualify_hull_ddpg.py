"""Original-objective Hull/Cao2021 training and common-ledger qualification.

The training objective is mean loss +1.5 standard deviation, not global ES.
Development evaluation is report-only; there is no checkpoint-selection gate.
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from hedging_gym.baselines._shared.controllers import policy_controller
from hedging_gym.baselines.hull_ddpg import (
    SOURCE,
    HullDDPGPolicy,
    HullLearner,
    train_hull_ddpg,
)
from hedging_gym.environment.config import config_from_dict
from hedging_gym.environment.finance import BANK_FIELDS, MarketBank, numpy_ledger
from hedging_gym.evaluation import evaluate_controller


def load_bank(path):
    saved = torch.load(path, weights_only=False, map_location='cpu')
    return MarketBank(*(saved[key] for key in BANK_FIELDS), config_from_dict(saved['config']))


def evaluate(policy, bank, device):
    metrics, tape = evaluate_controller(policy_controller(policy), bank,
        device=device, batch_size=1024, progress=True, label='Hull2021 development')
    loss = tape['terminal_loss'].double()
    metrics['mean_plus_1_5_std'] = float(loss.mean()+1.5*loss.std(correction=0))
    return metrics, tape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=50001)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--collection-batch', type=int, default=16)
    parser.add_argument('--checkpoint-every', type=int, default=1024)
    parser.add_argument('--cuda-graph', action='store_true')
    parser.add_argument('--reward-scale', type=float, default=None,
                        help='common money per learner dollar; default S0/10000 follows native notional')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.cuda_graph and args.device != 'cuda':
        parser.error('--cuda-graph requires --device cuda')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training = load_bank(args.run_dir/'train_bank.pt')
    development = load_bank(args.run_dir/'development_bank.pt')
    if training.config != development.config:
        raise ValueError('training/development financial contracts differ')
    options = dict(episodes=args.episodes, seed=args.seed, device=args.device,
        collection_batch=args.collection_batch, replay_capacity=600000,
        learning_rate=1e-4, target_rate=1e-5, variance_floor=1e-8,
        reward_scale=args.reward_scale, cuda_graph=args.cuda_graph, progress=True,
        checkpoint_path=args.output_dir/'latest.pt', checkpoint_every=args.checkpoint_every)
    initial_metrics = None
    if args.resume is None:
        if (args.output_dir/'initial-policy.pt').exists():
            raise FileExistsError('existing initial policy: use a fresh output or explicit resume')
        torch.manual_seed(args.seed)
        initial = HullLearner(training.config, device=args.device, dtype=training.spot.dtype).actor
        torch.save(dict(policy=initial.state_dict(), config=asdict(training.config), source=SOURCE),
                   args.output_dir/'initial-policy.pt')
        initial_metrics, initial_tape = evaluate(initial, development, args.device)
        torch.save(initial_tape, args.output_dir/'initial-development.pt')
        (args.output_dir/'initial-metrics.json').write_text(json.dumps(initial_metrics, indent=2)+'\n')
        del initial, initial_tape
    else:
        initial_directory = (args.output_dir if (args.output_dir/'initial-metrics.json').exists()
                             else args.resume.parent)
        if (initial_directory/'initial-metrics.json').exists():
            initial_metrics = json.loads((initial_directory/'initial-metrics.json').read_text())
    started = time.perf_counter()
    policy, metadata = train_hull_ddpg(training, **options, resume_from=args.resume)
    torch.save(dict(policy=policy.state_dict(), config=asdict(training.config), metadata=metadata),
               args.output_dir/'policy.pt')
    metrics, tape = evaluate(policy, development, args.device)
    torch.save(tape, args.output_dir/'development.pt')
    # Reload from disk, and compare with a separate cash recursion.
    saved = torch.load(args.output_dir/'policy.pt', weights_only=False, map_location='cpu')
    restored = HullDDPGPolicy(development.config).to(device=args.device, dtype=development.spot.dtype).eval()
    restored.load_state_dict(saved['policy'])
    reloaded_metrics, reloaded_tape = evaluate(restored, development, args.device)
    ledger = numpy_ledger(development.marks.numpy(), tape['positions'].cpu().numpy(),
        development.liability[:, 0].numpy(), development.liability[:, -1].numpy(), development.config)
    verification = dict(
        maximum_reload_loss_difference=float((reloaded_tape['terminal_loss']-tape['terminal_loss']).abs().max()),
        maximum_reload_action_difference=float((reloaded_tape['positions']-tape['positions']).abs().max()),
        maximum_independent_ledger_difference=float(np.abs(ledger['terminal_loss']-tape['terminal_loss'].cpu().numpy()).max()),
        all_losses_finite=bool(np.isfinite(ledger['terminal_loss']).all()),
        constraint_violations=metrics['constraint_violations'])
    report = dict(classification='2021 two-moment DDPG common-task adaptation; one declared training round',
        source=SOURCE, training_objective='mean loss + 1.5 standard deviation',
        evaluation='common terminal loss and ES; development report-only, no checkpoint selection',
        training_bank=str(args.run_dir/'train_bank.pt'), development_bank=str(args.run_dir/'development_bank.pt'),
        financial_config=asdict(training.config), metadata=metadata, initial_metrics=initial_metrics,
        metrics=metrics, verification=verification, stage_wall_seconds=time.perf_counter()-started,
        resume_from=None if args.resume is None else str(args.resume))
    (args.output_dir/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
