"""First-order adaptation-aware pretraining on the shared financial ledger.

Algorithmic reference: cbfinn/maml, commit
a7f45f1bcd7457fe97b227a21e89b8a82cc5fa49, ``maml.py`` lines 91--103
(``stop_grad``) and 144--150 (final post-adaptation query objective). That
author code detaches inner gradients and differentiates query loss through the
remaining identity connection to initial weights. Here we explicitly transfer
the fast policy's query gradients to the initialization. This is a finance
adaptation of that first-order principle, not author finance code or an exact
MAML reproduction: inner Adam, task vectors, and terminal ES replace the source
supervised tasks and inner gradient descent. No fast-weight interpolation or
second derivatives are used.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import math
import time

import torch

from hedging_gym.finance import bank_subset, bank_to
from .adaptation import AdaptationUpdater, TaskEmbeddedPolicy, _continuous_contract
from .checkpoints import load_checkpoint, restore_rng, rng_state, save_checkpoint
from .training import _report, _sync, rollout


DONOR_URL = "https://github.com/cbfinn/maml"
DONOR_COMMIT = "a7f45f1bcd7457fe97b227a21e89b8a82cc5fa49"
METHOD = "adapt_aware_dh"


def differentiable_empirical_es(values, alpha):
    """Same fractional empirical tail integral as evaluation, retaining gradients.

    Sorting selects the tail; away from ties each selected loss receives its
    exact tail-mass weight. At a tie autograd selects a valid subgradient. The
    double conversion matches the independent evaluator's summation precision.
    """
    if not 0 < alpha < 1 or values.ndim != 1 or values.numel() == 0:
        raise ValueError("ES needs nonempty scalar path losses and alpha in (0,1)")
    ordered = values.double().sort(descending=True).values
    mass = (1-alpha)*len(ordered)
    whole = math.floor(mass)
    fraction = mass-whole
    tail = ordered[:whole].sum()
    if fraction > 0:
        tail = tail + fraction*ordered[whole]
    return tail/mass


def split_source_bank(bank):
    """Fixed disjoint support/query halves; 8192 paths become 4096/4096."""
    if len(bank.spot) < 2 or len(bank.spot) % 2:
        raise ValueError("source banks need an even positive support/query split")
    midpoint = len(bank.spot)//2
    return (bank_subset(bank, slice(0, midpoint)),
            bank_subset(bank, slice(midpoint, len(bank.spot))))


def _assign_first_order_gradients(policy, fast_policy, task, query_objective):
    """Set initial gradients to post-update query gradients, without inner Jacobians."""
    initial = dict(policy.shared.named_parameters())
    fast = dict(fast_policy.shared.named_parameters())
    if initial.keys() != fast.keys():
        raise ValueError("initial and adapted policies must have matching shared weights")
    gradients = torch.autograd.grad(query_objective,
        [*fast.values(), fast_policy.embedding], create_graph=False)
    for parameter, gradient in zip(initial.values(), gradients[:-1]):
        parameter.grad = gradient.detach().clone()
    policy.source_embeddings.grad = torch.zeros_like(policy.source_embeddings)
    policy.source_embeddings.grad[task].copy_(gradients[-1].detach())


def train_adapt_aware(policy, metadata, source_banks, *, seed=7, episodes=600,
                      inner_updates=5, batch_size=256, query_size=1024,
                      inner_lr=3e-4, outer_lr=1e-4, checkpoint_path=None,
                      progress=True, resume_from=None):
    """Refine an existing TaskEmbeddedPolicy for post-adaptation terminal ES.

    Each round-robin episode clones the current initialization and source task
    vector, initializes its ES threshold from up to 1024 support paths, and
    performs fresh full-network Adam adaptation through AdaptationUpdater.
    Query paths are sampled without replacement from the other half-bank, with
    an independent RNG. Only the final query ES drives the outer Adam update.
    With ``inner_updates=0``, ordinary continued multitask training instead
    samples from the entire source bank and uses no support or initialization
    rollouts. This control retains the architecture, query ES and outer Adam.

    Checkpoints after each complete metaepisode include outer Adam, both index
    RNGs, global RNG, task position, history and accumulated work/time. Resume
    may extend ``episodes`` but must retain the recipe and declared source banks.
    An interrupted in-flight episode is replayed from the last complete one.
    Architecture metadata remains compatible with the ordinary policy loader.
    """
    banks = tuple(source_banks)
    if type(policy) is not TaskEmbeddedPolicy or len(banks) != len(policy.source_embeddings):
        raise ValueError("one source bank per ordinary TaskEmbeddedPolicy source vector required")
    if (len(banks) < 2 or min(episodes, batch_size, query_size) < 1 or inner_updates < 0
            or min(inner_lr, outer_lr) <= 0):
        raise ValueError("at least two banks and positive work counts/learning rates required")
    config = banks[0].config
    _continuous_contract(config)
    for bank in banks:
        if replace(bank.config, market=config.market) != config:
            raise ValueError("source banks must share book, clock, execution and risk")
        policy.check_config(bank.config)
    parameter = next(policy.parameters())
    device = parameter.device
    if any(bank.spot.dtype != parameter.dtype for bank in banks):
        raise ValueError("policy and source-bank dtypes must match")
    started = time.perf_counter()
    banks = tuple(bank_to(bank, device) for bank in banks)
    splits = (tuple(split_source_bank(bank) for bank in banks) if inner_updates
              else tuple((None, bank) for bank in banks))
    if any(query_size > len(query.spot) for _, query in splits):
        raise ValueError("query_size cannot exceed a source's independent query half")
    method = METHOD if inner_updates else "query_es_dh"
    recipe = "first_order_post_adaptation_es" if inner_updates else "ordinary_query_es_continuation"
    options = dict(episodes=episodes, inner_updates=inner_updates, batch_size=batch_size,
        query_size=query_size, inner_lr=inner_lr, inner_zeta_lr=3e-4,
        outer_lr=outer_lr, gradient_clip=5., initialization_paths_per_episode=1024 if inner_updates else 0,
        recipe=recipe)
    source_configs = [asdict(bank.config) for bank in banks]
    source_paths = [len(bank.spot) for bank in banks]
    for key, actual in (("source_configs", source_configs), ("source_paths", source_paths)):
        if key in metadata and metadata[key] != actual:
            raise ValueError("adaptation-aware pretraining requires the declared original source banks")
    initial_metadata = deepcopy(metadata)
    policy.active_task = None
    policy.shared.requires_grad_(True)
    policy.source_embeddings.requires_grad_(True)
    policy.embedding.requires_grad_(False)
    outer_parameters = [*policy.shared.parameters(), policy.source_embeddings]
    optimizer = torch.optim.Adam(outer_parameters, lr=outer_lr)
    support_generator = torch.Generator().manual_seed(seed+100003)
    query_generator = torch.Generator().manual_seed(seed+200003)
    history, completed, previous_seconds = [], 0, 0.
    source_zetas = list(metadata.get("source_zetas", [metadata.get("zeta", 0.)]*len(banks)))
    if resume_from is not None:
        saved = load_checkpoint(resume_from, method=method, config=config)
        prior_options = {key: value for key, value in saved["options"].items() if key != "episodes"}
        current_options = {key: value for key, value in options.items() if key != "episodes"}
        if (prior_options != current_options or saved["seed"] != seed
                or saved["source_configs"] != source_configs or saved["source_paths"] != source_paths):
            raise ValueError("resume requires the saved recipe, seed and source banks")
        completed = saved["step"]
        if completed > episodes:
            raise ValueError("requested episodes precede the saved completed step")
        policy.load_state_dict(saved["policy"])
        optimizer.load_state_dict(saved["optimizer"])
        support_generator.set_state(saved["support_rng"])
        query_generator.set_state(saved["query_rng"])
        restore_rng(saved["rng"])
        history, previous_seconds = saved["history"], saved["training_seconds"]
        source_zetas, initial_metadata = saved["source_zetas"], saved["initial_metadata"]
    else:
        torch.manual_seed(seed)

    def work_counts(count):
        visits = [count//len(banks)+(task < count % len(banks)) for task in range(len(banks))]
        initialization = sum(visits[task]*min(1024, len(support.spot))
                             for task, (support, _) in enumerate(splits)) if inner_updates else 0
        support_work = count*inner_updates*batch_size
        query_work = count*query_size
        return dict(metaepisodes=count, outer_optimizer_steps=count,
            inner_optimizer_steps=count*inner_updates, episodes_per_source=visits,
            source_unique_paths=sum(source_paths),
            source_support_paths=[len(support.spot) if support is not None else 0 for support, _ in splits],
            source_query_paths=[len(query.spot) for _, query in splits],
            initialization_forward_rollouts=initialization,
            support_gradient_rollouts=support_work, query_gradient_rollouts=query_work,
            gradient_episode_rollouts=support_work+query_work,
            total_episode_rollouts=initialization+support_work+query_work)

    if progress:
        _report("meta_train_start", method=method, seed=seed, device=str(device),
            workers=torch.get_num_threads(), config=asdict(config), source_configs=source_configs,
            options=options, completed=completed, expected_work=work_counts(episodes),
            support_minibatch_seed=seed+100003, query_minibatch_seed=seed+200003)
    for step in range(completed+1, episodes+1):
        task = (step-1) % len(banks)
        support, query = splits[task]
        fast = deepcopy(policy)
        fast.active_task = None
        fast.embedding.requires_grad_(True)
        fast.source_embeddings.requires_grad_(False)
        with torch.no_grad():
            fast.embedding.copy_(policy.source_embeddings[task])
        if inner_updates:
            with torch.no_grad():
                initial_losses = rollout(fast, bank_subset(support,
                    slice(0, min(1024, len(support.spot)))))["terminal_loss"]
                threshold = torch.quantile(initial_losses, config.risk.alpha)
            updater = AdaptationUpdater(fast, metadata={"zeta": float(threshold)}, mode="finetune",
                seed=seed, updates=inner_updates, batch_size=batch_size,
                learning_rate=inner_lr, zeta_learning_rate=3e-4, progress=False)
            fast.source_embeddings.requires_grad_(False)
            updater.last_market = support.config.market
            updater.index_generator = support_generator
            updater(support)
        # Query draws never consume the support minibatch stream, and can never
        # include episode support paths in the adaptation-aware arm. K0 has no
        # support and can use the whole original source bank.
        query_indices = torch.randperm(len(query.spot), generator=query_generator)[:query_size]
        query_losses = rollout(fast, bank_subset(query, query_indices.to(device)))["terminal_loss"]
        objective = differentiable_empirical_es(query_losses, config.risk.alpha)
        optimizer.zero_grad(set_to_none=True)
        _assign_first_order_gradients(policy, fast, task, objective)
        gradient_norm = torch.nn.utils.clip_grad_norm_(outer_parameters, 5., error_if_nonfinite=True)
        optimizer.step()
        if inner_updates:
            source_zetas[task] = float(updater.zeta.detach())
        if step == 1 or step % 20 == 0 or step == episodes:
            _sync(device)
            elapsed = previous_seconds+time.perf_counter()-started
            current_elapsed = time.perf_counter()-started
            record = dict(completed=step, total=episodes, task=task,
                query_es=float(objective.detach()), outer_gradient_norm=float(gradient_norm),
                elapsed_seconds=elapsed, eta_seconds=current_elapsed*(episodes-step)/(step-completed),
                zeta=source_zetas[task])
            history.append(record)
            if progress:
                _report("meta_train_progress", method=method, **record)
        if checkpoint_path is not None:
            _sync(device)
            save_checkpoint(checkpoint_path, dict(method=method, phase="meta_pretraining",
                step=step, seed=seed, config=asdict(config), source_configs=source_configs,
                source_paths=source_paths, options=options, policy=policy.state_dict(),
                optimizer=optimizer.state_dict(), support_rng=support_generator.get_state(),
                query_rng=query_generator.get_state(), rng=rng_state(), history=history,
                source_zetas=source_zetas, initial_metadata=initial_metadata,
                training_seconds=previous_seconds+time.perf_counter()-started,
                work=work_counts(step), donor_url=DONOR_URL, donor_commit=DONOR_COMMIT))
    policy.prepare_adaptation()
    policy.eval()
    _sync(device)
    additional_seconds = previous_seconds+time.perf_counter()-started
    work = work_counts(episodes)
    result = deepcopy(initial_metadata)
    result.update(method=method,
        method_label=("First-order adaptation-aware Deep Hedging" if inner_updates
                      else "Ordinary query-ES continued multitask Deep Hedging"),
        classification=("Finance adaptation of first-order MAML, not author-code reproduction" if inner_updates
                        else "Ordinary continued multitask pretraining control; no within-episode adaptation"),
        initial_pretraining=initial_metadata,
        meta_pretraining=dict(source=DONOR_URL if inner_updates else None,
            source_commit=DONOR_COMMIT if inner_updates else None, recipe=recipe,
            source_reference=("maml.py stop_grad inner updates and final post-update query objective"
                              if inner_updates else None),
            source_changes=(["fresh inner Adam instead of author gradient descent",
                "cost-inclusive terminal ES on the unchanged common finance ledger",
                "source task embeddings and disjoint financial support/query paths",
                "one round-robin source task per outer step", "explicit detached gradient transfer"] if inner_updates
                else ["continue original multitask policy with exact query ES and the matched outer Adam"]),
            seed=seed, options=options, work=work, history=history,
            training_seconds=additional_seconds, device=str(device),
            objective=("exact fractional empirical ES after all inner updates" if inner_updates
                       else "ordinary exact fractional empirical ES at the current source initialization"),
            gradient=("first-order identity-Jacobian approximation; no differentiation through Adam" if inner_updates
                      else "ordinary query-loss gradient without inner adaptation"),
            query_sampling=("without replacement from fixed disjoint second half of each source bank" if inner_updates
                            else "without replacement from the entire original source bank"),
            query_holdout_scope="disjoint from episode support only; original ADH pretraining used the source banks",
            source_embedding_optimizer="dense Adam; only selected row gets query gradient, all rows retain moments",
            timing_scope="additional setup, initialization rollouts, inner/query training and checkpoints; excludes original training and bank generation"),
        source_zetas=source_zetas, zeta=sum(source_zetas)/len(source_zetas),
        training_seconds=initial_metadata.get("training_seconds", 0.)+additional_seconds,
        total_seconds=initial_metadata.get("total_seconds", initial_metadata.get("training_seconds", 0.))+additional_seconds,
        expected_episode_rollouts=initial_metadata.get("expected_episode_rollouts", 0)+work["gradient_episode_rollouts"],
        adaptation_policy_parameter_count=sum(p.numel() for p in policy.shared.parameters())+policy.embedding.numel(),
        adaptation="Full-network updates from the returned source initialization")
    return policy, result
