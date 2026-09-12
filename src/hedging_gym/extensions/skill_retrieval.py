"""SRSA-style retrieval from a frozen Adaptive DH context library.

Paper: Guo et al., ICLR 2025, https://arxiv.org/abs/2503.04538, equation (1).
Upstream: https://github.com/NVlabs/SRSA/tree/2bed3f7ecee73be29eaeccd1b3a6fe03d4482702
Implementation notes: docs/skill-retrieval.md.

The paired-feature ReLU/MSE predictor follows SuccPreNetwork.fc. Here skills
are source task vectors, and lower cost-inclusive ES replaces higher robotic
success. This transfers retrieval, not the complete robotic system.
"""

import time
from copy import deepcopy
from dataclasses import asdict, replace

import torch
from torch import nn

from hedging_gym.baselines._shared.checkpoints import rng_state, save_checkpoint
from hedging_gym.baselines._shared.training import _report, _sync, rollout
from hedging_gym.environment.finance import bank_subset, bank_to
from hedging_gym.evaluation import empirical_es

SOURCE_COMMIT = "2bed3f7ecee73be29eaeccd1b3a6fe03d4482702"


class TransferRiskPredictor(nn.Module):
    """Predict ES(source context, target) minus ES(mean context, target).

    Raw market parameters are standardized using source tasks only. The source
    and target features are concatenated in that order: transfer is asymmetric.
    Normalization and the immutable source library descriptors are saved buffers.
    """

    def __init__(self, source_configs, *, hidden=32):
        super().__init__()
        configs = tuple(source_configs)
        self.market_model = configs[0].market.model
        self.market_scheme = getattr(configs[0].market, "scheme", None)
        # Model and discretization scheme are categorical, not numeric inputs.
        self.fields = tuple(key for key in asdict(configs[0].market)
                            if key not in ("model", "scheme"))
        if len(configs) < 2 or any(c.market.model != self.market_model for c in configs):
            raise ValueError("retrieval needs at least two source tasks from one market family")
        if any(getattr(c.market, "scheme", None) != self.market_scheme for c in configs):
            raise ValueError("retrieval source tasks must use the same market scheme")
        source = torch.tensor([[getattr(c.market, key) for key in self.fields]
                               for c in configs])
        scale = source.std(dim=0, correction=0)
        self.register_buffer("source_features", source)
        self.register_buffer("feature_mean", source.mean(dim=0))
        self.register_buffer("feature_scale", torch.where(scale > 0, scale, 1.))
        self.register_buffer("risk_scale", torch.ones(()))
        # Same head structure and initialization as the donor; width is smaller
        # because the first finance library has 8 tasks, not 90 robotic skills.
        self.head = nn.Sequential(nn.Linear(2*len(self.fields), hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, .01)

    def features(self, config):
        if config.market.model != self.market_model:
            raise ValueError("retrieval does not extrapolate between market model families")
        if getattr(config.market, "scheme", None) != self.market_scheme:
            raise ValueError("retrieval target must use the source market scheme")
        return self.source_features.new_tensor([getattr(config.market, key) for key in self.fields])

    def forward(self, source_features, target_features):
        source = (source_features-self.feature_mean)/self.feature_scale
        target = (target_features-self.feature_mean)/self.feature_scale
        return self.head(torch.cat((source, target), dim=-1)).squeeze(-1)*self.risk_scale

    @torch.no_grad()
    def predict_scores(self, target_config):
        """One score per source context; lower predicted relative ES is better."""
        target = self.features(target_config).expand_as(self.source_features)
        return self(self.source_features, target)


@torch.no_grad()
def score_source_contexts(policy, source_banks, *, device="cpu", chunk_size=4096,
                          progress=True):
    """Score every source context on fresh, dedicated source-calibration paths.

    ``source_banks[j]`` belongs to source task j, not its original training bank.
    The caller supplies and records distinct calibration seeds. ES is pooled
    across all paths, never averaged across chunks. Policy state is untouched.
    Returns the source-by-target ES matrix, mean-context ES, and measured work.
    """
    banks = tuple(source_banks)
    if len(banks) != len(policy.source_embeddings) or chunk_size < 1:
        raise ValueError("one calibration bank per ordered source context is required")
    config = banks[0].config
    if any(replace(b.config, market=config.market) != config for b in banks):
        raise ValueError("source tasks may differ only in market parameters")
    started = time.perf_counter()
    scorer = deepcopy(policy).to(device)
    scorer.eval()
    scorer.active_task = None
    scores = torch.empty((len(banks), len(banks)), dtype=torch.float64)
    mean_scores = torch.empty(len(banks), dtype=torch.float64)
    work = 0
    for target, bank in enumerate(banks):
        bank = bank_to(bank, device)
        scorer.check_config(bank.config)
        for source in range(len(banks)+1):
            context = (scorer.source_embeddings.mean(0) if source == len(banks)
                       else scorer.source_embeddings[source])
            scorer.embedding.copy_(context)
            losses = []
            for first in range(0, len(bank.spot), chunk_size):
                sample = bank_subset(bank, slice(first, first+chunk_size))
                losses.append(rollout(scorer, sample)["terminal_loss"].cpu())
            score = empirical_es(torch.cat(losses), bank.config.risk.alpha)
            if source == len(banks):
                mean_scores[target] = score
            else:
                scores[source, target] = score
            work += len(bank.spot)
        if progress:
            elapsed = time.perf_counter()-started
            _report("retrieval_labels", completed=target+1, total=len(banks),
                    elapsed_seconds=elapsed,
                    eta_seconds=elapsed*(len(banks)-target-1)/(target+1))
    _sync(device)
    return scores, mean_scores, dict(episode_rollouts=work,
        ledger_decisions=work*config.n_steps, seconds=time.perf_counter()-started,
        source_paths=[len(b.spot) for b in banks])


def train_retriever(policy, source_banks, *, updates=300, seed=7, device="cpu",
                    progress=True, hidden=32, learning_rate=1e-3, chunk_size=4096,
                    checkpoint_path=None):
    """Fit the donor's paired MSE predictor without touching the source hedger.

    All n² source-task pairs are used per inexpensive predictor update. The
    expensive calibration rollouts happen once. No target-market path, target
    evaluation metric, or result-dependent checkpoint selection is consumed.
    """
    banks = tuple(source_banks)
    if updates < 1 or learning_rate <= 0:
        raise ValueError("positive predictor updates and learning rate required")
    started = time.perf_counter()
    if progress:
        _report("retrieval_start", seed=seed, device=str(device), sources=len(banks),
                updates=updates, expected_label_rollouts=(len(banks)+1)*sum(len(b.spot) for b in banks))
    transfer_es, mean_es, label_work = score_source_contexts(
        policy, banks, device=device, chunk_size=chunk_size, progress=progress)
    torch.manual_seed(seed)
    predictor = TransferRiskPredictor([b.config for b in banks], hidden=hidden).to(
        device=device, dtype=next(policy.parameters()).dtype)
    relative_es = (transfer_es-mean_es.unsqueeze(0)).to(predictor.source_features)
    # Scaling changes optimizer conditioning, not pair rankings or the ES target.
    scale = relative_es.square().mean().sqrt()
    predictor.risk_scale.copy_(torch.where(scale > 0, scale, 1.))
    source, target = torch.meshgrid(torch.arange(len(banks), device=device),
                                   torch.arange(len(banks), device=device), indexing="ij")
    source = predictor.source_features[source.flatten()]
    target = predictor.source_features[target.flatten()]
    labels = relative_es.flatten()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
    history = []
    training_start = time.perf_counter()
    for step in range(1, updates+1):
        loss = ((predictor(source, target)-labels)/predictor.risk_scale).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == updates:
            _sync(device)
            elapsed = time.perf_counter()-training_start
            record = dict(completed=step, total=updates, normalized_mse=float(loss.detach()),
                          elapsed_seconds=elapsed, eta_seconds=elapsed*(updates-step)/step)
            history.append(record)
            if progress:
                _report("retrieval_fit", **record)
        if checkpoint_path is not None and (step == 1 or step % 100 == 0 or step == updates):
            save_checkpoint(checkpoint_path, dict(method="srsa_retrieval", step=step, seed=seed,
                source_configs=[asdict(b.config) for b in banks], hidden=hidden,
                predictor=predictor.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state(),
                transfer_es=transfer_es, mean_context_es=mean_es, label_work=label_work,
                options=dict(updates=updates, learning_rate=learning_rate), history=history))
    predictor.eval()
    with torch.no_grad():
        fitted = predictor(source, target).reshape(len(banks), len(banks))
    metadata = dict(method="srsa_retrieval", classification="Finance retrieval adaptation, not full SRSA",
        source="https://arxiv.org/abs/2503.04538", source_commit=SOURCE_COMMIT,
        source_configs=[asdict(b.config) for b in banks], feature_fields=list(predictor.fields),
        seed=seed, device=str(device), updates=updates, hidden=hidden, learning_rate=learning_rate,
        pair_count=len(banks)**2, predictor_pair_evaluations=updates*len(banks)**2,
        transfer_es=transfer_es.tolist(), mean_context_es=mean_es.tolist(),
        relative_es=relative_es.cpu().tolist(), fitted_relative_es=fitted.cpu().tolist(),
        label_work=label_work, training_seconds=time.perf_counter()-training_start,
        total_seconds=time.perf_counter()-started, history=history,
        prediction_only_selection=True)
    return predictor, metadata


@torch.no_grad()
def rank_contexts(retriever, policy, target_config):
    """Return source indices from lowest to highest predicted target ES.

    No target rollout occurs here. The caller can use top-1 directly or rescore a
    predeclared top-k on separate target-calibration paths, charging that work.
    """
    policy.check_config(target_config)
    if len(policy.source_embeddings) != len(retriever.source_features):
        raise ValueError("retrieval requires the unchanged ordered source context library")
    return torch.argsort(retriever.predict_scores(target_config), stable=True).cpu().tolist()


def select_context(retriever, policy, target_config):
    return rank_contexts(retriever, policy, target_config)[0]


@torch.no_grad()
def nearest_context(retriever, target_config):
    """Nonlearned control using the identical source-only feature scaling."""
    differences = (retriever.source_features-retriever.features(target_config))/retriever.feature_scale
    return int(differences.square().sum(-1).argmin())
