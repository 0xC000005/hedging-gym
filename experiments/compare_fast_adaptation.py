"""Compare three context-adaptation mechanisms on unchanged basic Heston.

Generate shared banks first, then run independent method/seed jobs. Outputs are
external checkpoints, complete evaluation tapes and a compact JSON readout.
The default comparison is fixed before evaluation; --smoke is plumbing only.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from functools import partial
import json
from pathlib import Path
import time

import torch

from hedging_gym.config import config_from_dict
from hedging_gym.evaluation import evaluate_controller
from hedging_gym.finance import BANK_FIELDS, generate_market_bank
from experiments.qualify_adaptation import load_bank
from methods.adaptation import AdaptationUpdater, train_multitask
from methods.belief_adaptation import (BeliefDynamicsEncoder, BeliefEmbeddedPolicy,
    encode_bank_context, train_belief_encoder)
from methods.controllers import policy_controller
from methods.geps import GEPSPolicy
from methods.skill_retrieval import nearest_context, rank_contexts, train_retriever
from methods.training import _report, _sync


SOURCE_MARKETS = ((.04,.04,3,.3,-.5), (.032,.032,3,.3,-.5),
    (.02,.02,2,.2,-.3), (.06,.06,4,.4,-.7), (.08,.08,3,.3,-.5),
    (.11,.11,2,.4,-.7), (.025,.06,4,.2,-.3), (.07,.03,2,.4,-.7))
TARGET_MARKETS = dict(A=SOURCE_MARKETS[0], B=(.09,.09,3,.3,-.5),
    C=(.05,.05,3,.35,-.6), D=(.04,.065,2.5,.25,-.4))


def market_config(values):
    # These saved experiments predate the paper defaults and QE-M simulator.
    base = config_from_dict(json.loads(
        (Path(__file__).parent / "configs" / "legacy-basic-heston.json").read_text()))
    changes = dict(zip(("v0", "theta", "kappa", "sigma", "rho"), values))
    return replace(base, market=replace(base.market, **changes))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n")


def prepare_banks(args):
    directory = args.output / "banks"
    directory.mkdir(parents=True, exist_ok=True)
    requests = []
    for index, values in enumerate(SOURCE_MARKETS):
        requests.extend(((f"source-{index}", values, 8192, 910000+index),
                         (f"source-cal-{index}", values, 1024, 920000+index)))
    for index, (name, values) in enumerate(TARGET_MARKETS.items()):
        requests.extend(((f"{name}-train", values, 4096, 930000+index),
                         (f"{name}-cal", values, 1024, 940000+index),
                         (f"{name}-eval", values, 8192, 950000+index)))
    for number, (name, values, count, seed) in enumerate(requests, 1):
        count = 32 if args.smoke else count
        path = directory / f"{name}.pt"
        config = market_config(values)
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if (saved["seed"] != seed or len(saved["spot"]) != count
                    or config_from_dict(saved["config"]) != config):
                raise ValueError("bank does not match this comparison; use a separate output directory")
            continue
        _report("bank_start", completed=number-1, total=len(requests), name=name,
                paths=count, seed=seed, device=args.device)
        started = time.perf_counter()
        bank = generate_market_bank(config, count, seed, device=args.device)
        _sync(torch.device(args.device))
        torch.save(dict(config=asdict(config), seed=seed,
            generation_seconds=time.perf_counter()-started,
            **{key:getattr(bank,key).cpu() for key in BANK_FIELDS}), path)
        _report("bank_complete", name=name, completed=number, total=len(requests),
                seconds=time.perf_counter()-started)


def source_policy(args, output, banks, calibration):
    factory, donor_work = {}, {}
    if args.method == "geps":
        factory["policy_class"] = GEPSPolicy
    elif args.method == "belief":
        encoder_path = output / "encoder.pt"
        if encoder_path.exists():
            saved = torch.load(encoder_path, map_location="cpu", weights_only=False)
            encoder = BeliefDynamicsEncoder(**saved["architecture"]).to(args.device)
            encoder.load_state_dict(saved["encoder"])
            encoder.requires_grad_(False).eval()
            encoder_work = saved["metadata"]
        else:
            encoder, encoder_work = train_belief_encoder(calibration, seed=args.seed+6000,
                updates=8 if args.smoke else 1000, device=args.device)
            torch.save(dict(encoder=encoder.state_dict(), architecture=encoder.options,
                            metadata=encoder_work), encoder_path)
        vectors, context_work = zip(*(encode_bank_context(encoder, bank) for bank in calibration))
        factory["policy_class"] = partial(BeliefEmbeddedPolicy, encoder=encoder,
                                          source_contexts=torch.stack(vectors))
        donor_work = dict(encoder=encoder_work, source_contexts=list(context_work))
    checkpoint = output / "pretrain-latest.pt"
    policy, metadata = train_multitask(banks, seed=args.seed,
        updates=16 if args.smoke else 6000, batch_size=32 if args.smoke else 256,
        hidden=(64,64), embedding_dim=4, device=args.device,
        checkpoint_path=checkpoint, checkpoint_every=200,
        resume_from=checkpoint if checkpoint.exists() else None, **factory)
    if args.method == "geps":
        metadata.update(source="https://github.com/itsakk/geps",
            source_commit="e9a865218ecffacb7007ac7d719f3741afcf8c02",
            source_changes=["financial ES policy instead of PDE predictor", "common tanh activations",
                            "same context-only target optimizer as Adaptive DH"])
    elif args.method == "belief":
        metadata.update(source="https://github.com/maxsbob/BeliefConditionedFB",
            source_commit="30e7487ca033c3619ec744ed55f916ece005c425",
            source_changes=["context encoder only, not full FB or Rotation-FB RL",
                "direct-gradient financial ES hedger", "observed parameters retained"],
            donor_work=donor_work)
    metadata["timing_caveat"] = "Independent jobs may overlap; these are not uncontended speed measurements"
    previous_metadata = output / "pretraining.json"
    if previous_metadata.exists():
        previous = json.loads(previous_metadata.read_text())
        if previous["options"] == metadata["options"] and previous["seed"] == args.seed:
            # A completed checkpoint reload is not new pretraining. Preserve
            # its original initialization and all-in cost instead of replacing
            # them with the time spent loading it in this process.
            metadata = previous
    torch.save(dict(policy=policy.state_dict(), metadata=metadata), output / "pretrained.pt")
    write_json(output / "pretraining.json", metadata)
    return policy, metadata


@torch.no_grad()
def rescore_contexts(policy, bank, ranking, device):
    """Real target calibration, not evaluation-path model selection."""
    scorer = deepcopy(policy)
    scores = []
    started = time.perf_counter()
    for index in ranking:
        scorer.embedding.copy_(scorer.source_embeddings[index])
        metrics, _ = evaluate_controller(policy_controller(scorer), bank, device=device)
        scores.append((index, metrics["es95"]))
    return scores, dict(episode_rollouts=len(bank.spot)*len(ranking),
                        seconds=time.perf_counter()-started)


def evaluate_curve(args, output, policy, metadata, target, variant,
                   train, calibration, evaluation, initial=None, extra_work=None):
    directory = output / target / variant
    directory.mkdir(parents=True, exist_ok=True)
    complete = directory / "curve.json"
    if complete.exists():
        return json.loads(complete.read_text())
    policy = deepcopy(policy)
    if isinstance(policy, BeliefEmbeddedPolicy):
        extra_work = dict(extra_work or {}, context=policy.infer_context(calibration))
    updater = AdaptationUpdater(policy, metadata=metadata, seed=args.seed+1000,
        updates=1 if args.smoke else 10, batch_size=32 if args.smoke else 256,
        progress=False, checkpoint_path=directory / "adapt-latest.pt")
    if initial is not None:
        with torch.no_grad():
            policy.embedding.copy_(initial)
    initial_vector = policy.embedding.detach().clone()
    frozen = {name: value.detach().clone() for name,value in policy.named_parameters()
              if name != "embedding"}
    result = dict(method=variant, target=target, policy_seed=args.seed,
        config=asdict(train.config), extra_work=extra_work or {}, milestones=[])
    milestones = (0,1,2) if args.smoke else (0,10,50,200)
    checkpoint = directory / "adapt-latest.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        updater.load_state_dict(saved["state"])
    completed = updater.completed_steps
    for budget in milestones:
        milestone_path = directory / f"{budget}.json"
        if milestone_path.exists():
            result["milestones"].append(json.loads(milestone_path.read_text()))
            continue
        if completed > budget:
            raise RuntimeError("checkpoint passed an unsaved evaluation milestone")
        started = time.perf_counter()
        while completed < budget:
            updater(train, initial_embedding=initial_vector)
            completed = updater.completed_steps
        metrics, tape = evaluate_controller(policy_controller(policy), evaluation,
            device=args.device, batch_size=1024, label=f"{variant}/{target}/{budget}")
        changed = [name for name,value in policy.named_parameters()
                   if name in frozen and not torch.equal(value.detach(),frozen[name])]
        if changed:
            raise RuntimeError(f"context-only adaptation altered shared parameters: {changed}")
        record = dict(updates=budget, metrics=metrics,
            adaptation_episode_rollouts=budget*updater.batch_size,
            threshold_initialization_paths=sum(h["initialization_paths"] for h in updater.history),
            adaptation_seconds=sum(h["elapsed_seconds"] for h in updater.history),
            context=policy.embedding.detach().cpu().tolist())
        torch.save(tape, directory / f"{budget}-tape.pt")
        torch.save(dict(policy=policy.state_dict(), updater=updater.state_dict()),
                   directory / f"{budget}-policy.pt")
        write_json(milestone_path,record)
        result["milestones"].append(record)
        _report("curve_point", method=variant, seed=args.seed, target=target,
                updates=budget, es95=metrics["es95"], seconds=time.perf_counter()-started)
    write_json(complete,result)
    return result


def compare(args):
    output = args.output / args.method / f"seed-{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    bank_dir = args.output / "banks"
    banks = [load_bank(bank_dir / f"source-{i}.pt") for i in range(8)]
    calibration = [load_bank(bank_dir / f"source-cal-{i}.pt") for i in range(8)]
    policy, metadata = source_policy(args,output,banks,calibration)
    predictor = None
    if args.method == "adh":
        predictor, retrieval_work = train_retriever(policy,calibration,seed=args.seed+7000,
            updates=8 if args.smoke else 300, device=args.device,
            checkpoint_path=output / "retrieval.pt")
        write_json(output / "retrieval.json",retrieval_work)
    for target in TARGET_MARKETS:
        train, prior, evaluation = (load_bank(bank_dir / f"{target}-{role}.pt")
                                    for role in ("train","cal","eval"))
        evaluate_curve(args,output,policy,metadata,target,args.method,
                       train,prior,evaluation)
        if predictor is not None:
            ranking = rank_contexts(predictor,policy,train.config)
            scores, work = rescore_contexts(policy,prior,ranking,args.device)
            score_map = dict(scores)
            choices = dict(srsa_top1=ranking[0], nearest=nearest_context(predictor,train.config),
                srsa_top5=min(ranking[:5],key=score_map.get),
                exhaustive=min(ranking,key=score_map.get))
            for variant,index in choices.items():
                query_count = 5 if variant == "srsa_top5" else 8 if variant == "exhaustive" else 0
                # Each method is charged its own conceptual query workload;
                # the screen computes all eight once to share deterministic work.
                extra = dict(selected_context=index, predicted_ranking=ranking,
                    rescored_contexts=query_count, calibration_episode_rollouts=query_count*len(prior.spot),
                    all_scores=scores if query_count else [],
                    shared_all_context_scoring_seconds=work["seconds"])
                evaluate_curve(args,output,policy,metadata,target,variant,
                    train,prior,evaluation,initial=policy.source_embeddings[index].detach(),extra_work=extra)
    _report("method_complete", method=args.method, seed=args.seed, output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("banks","compare"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--method",choices=("adh","geps","belief"),default="adh")
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    _report("comparison_start", options={key:str(value) if isinstance(value,Path) else value
                                         for key,value in vars(args).items()})
    (prepare_banks if args.stage == "banks" else compare)(args)


if __name__ == "__main__":
    main()
