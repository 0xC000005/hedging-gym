"""Belief-FB dynamics-encoder transfer into task-conditioned Deep Hedging.

Modified PyTorch port of DynamicsTransformer/Encoder1DBlock/NextStatePrediction
and context_encoder_loss from maxsbob/BeliefConditionedFB, commit
30e7487ca033c3619ec744ed55f916ece005c425 (Apache-2.0; license in
docs/belief-fb-LICENSE.txt). Source links and deliberate changes are in
docs/belief-adaptation.md. This is neither full Belief-FB nor Rotation-FB.

The encoder learns only from declared prior training histories. Observed Heston
parameters remain in the unchanged common policy observation. Financial training
continues through TaskEmbeddedPolicy and the common terminal-ES cash ledger.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import time

import torch
from torch import nn

from .adaptation import TaskEmbeddedPolicy
from .training import _report, _sync


DONOR_COMMIT = "30e7487ca033c3619ec744ed55f916ece005c425"
DONOR_URL = "https://github.com/maxsbob/BeliefConditionedFB"


class _EncoderBlock(nn.Module):
    """Port of native Encoder1DBlock: pre-norm attention and GELU residual MLP."""

    def __init__(self, width, heads, mlp_dim):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width, eps=1e-6)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(width, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(width, mlp_dim), nn.GELU(),
                                 nn.Linear(mlp_dim, width))

    def forward(self, inputs):
        normalized = self.attention_norm(inputs)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        residual = inputs + attended
        return residual + self.mlp(self.mlp_norm(residual))


class BeliefDynamicsEncoder(nn.Module):
    """Native transition-set architecture and Gaussian context distribution.

    No positional encodings or attention masks: permuting complete transitions
    does not alter the inferred context. The action channel is zero because the
    common market process is exogenous to the hedger's actions. No hidden market
    parameters, option values, future evaluation states, or payoff labels enter.
    """

    def __init__(self, *, embedding_dim=4, width=48, heads=4, layers=2, mlp_dim=96):
        super().__init__()
        if min(embedding_dim, width, heads, layers, mlp_dim) < 1 or width % 3 or width % heads:
            raise ValueError("positive dimensions; width must divide into three channels and heads")
        self.options = dict(embedding_dim=embedding_dim, width=width, heads=heads,
                            layers=layers, mlp_dim=mlp_dim)
        self.embedding_dim = embedding_dim
        self.state_projection = nn.Linear(2, width // 3)
        self.action_projection = nn.Linear(1, width // 3)
        self.next_state_projection = nn.Linear(2, width // 3)
        self.blocks = nn.ModuleList(_EncoderBlock(width, heads, mlp_dim) for _ in range(layers))
        self.context_mean = nn.Linear(width, embedding_dim)
        self.context_log_std = nn.Linear(width, embedding_dim)
        self.predictor = nn.Sequential(nn.Linear(3 + embedding_dim, mlp_dim), nn.GELU(),
            nn.Linear(mlp_dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, 2))

    def forward(self, states, actions, next_states):
        if (states.ndim != 3 or states.shape[-1] != 2 or next_states.shape != states.shape
                or actions.shape != (*states.shape[:2], 1) or states.shape[1] < 1):
            raise ValueError("expected transition sets [batch,context,2/1/2]")
        tokens = torch.cat((self.state_projection(states), self.action_projection(actions),
                            self.next_state_projection(next_states)), dim=-1)
        for block in self.blocks:
            tokens = block(tokens)
        pooled = tokens.mean(dim=1)
        return self.context_mean(pooled), self.context_log_std(pooled)

    def prediction_loss(self, states, actions, next_states, *, generator=None):
        """Source loss: Gaussian context sample and half squared next-state error.

        There is no added KL, variance, parameter-label, or financial reward
        objective. This matches the released context_encoder_loss, including
        reconstructing transitions from the same supplied context window.
        """
        mean, log_std = self(states, actions, next_states)
        noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        context = mean + noise * log_std.exp()
        repeated = context[:, None].expand(-1, states.shape[1], -1)
        prediction = self.predictor(torch.cat((states, actions, repeated), dim=-1))
        return 0.5 * (prediction - next_states).square().mean()


def _transitions(bank, *, device=None, dtype=None):
    """Fixed market coordinates; parameters are not inferred from normalizers.

    Spot is expressed relative to its declared unit S0, then divided by 0.2;
    variance is divided by a fixed 0.04, not the current market's v0 or theta.
    This changes representation/units, not the native next-state objective.
    """
    if (bank.spot.ndim != 2 or bank.variance.shape != bank.spot.shape
            or bank.spot.shape[1] != bank.config.n_steps + 1 or len(bank.spot) < 1):
        raise ValueError("nonempty complete prior-history bank required")
    if not bool(torch.isfinite(bank.spot).all() & torch.isfinite(bank.variance).all()):
        raise ValueError("prior histories must be finite")
    if bool((bank.spot <= 0).any() | (bank.variance < 0).any()):
        raise ValueError("positive prior spot and nonnegative variance required")
    states = torch.stack(((bank.spot / bank.config.market.spot0).log() / .2,
                          bank.variance / .04), dim=-1).to(device=device, dtype=dtype)
    return states[:, :-1], states.new_zeros((*states[:, :-1].shape[:2], 1)), states[:, 1:]


def train_belief_encoder(source_history_banks, *, seed=7, updates=1000, batch_size=32,
                         context_length=20, embedding_dim=4, width=48, heads=4,
                         layers=2, mlp_dim=96, learning_rate=1e-3, device="cpu",
                         progress=True):
    """Pretrain the source-native encoder before financial policy pretraining.

    Every minibatch uses one source market (round-robin) and contiguous windows
    from independently sampled historical paths; contexts never mix regimes.
    Caller owns the declared history-bank/evaluation-bank separation. A bank is
    not accepted here implicitly from evaluation or from a live policy rollout.
    """
    banks = tuple(source_history_banks)
    if len(banks) < 2 or updates < 1 or batch_size < 1 or context_length < 1 or learning_rate <= 0:
        raise ValueError("two source-history banks and positive training work required")
    config = banks[0].config
    for bank in banks:
        if replace(bank.config, market=config.market) != config or context_length > bank.config.n_steps:
            raise ValueError("history banks must share full book/clock/risk and support context_length")
    device = torch.device(device)
    started = time.perf_counter()
    torch.manual_seed(seed)
    encoder = BeliefDynamicsEncoder(embedding_dim=embedding_dim, width=width,
        heads=heads, layers=layers, mlp_dim=mlp_dim).to(device=device, dtype=banks[0].spot.dtype)
    transitions = tuple(_transitions(bank, device=device, dtype=banks[0].spot.dtype) for bank in banks)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    index_rng = torch.Generator().manual_seed(seed + 200003)
    noise_rng = torch.Generator(device=device).manual_seed(seed + 300007)
    options = dict(updates=updates, batch_size=batch_size, context_length=context_length,
                   learning_rate=learning_rate, **encoder.options)
    if progress:
        _report("belief_encoder_start", source_commit=DONOR_COMMIT, seed=seed,
            device=str(device), workers=torch.get_num_threads(), options=options,
            source_markets=[asdict(bank.config.market) for bank in banks],
            expected_transition_presentations=updates * batch_size * context_length)
    history = []
    for step in range(1, updates + 1):
        task = (step - 1) % len(banks)
        path_ids = torch.randint(len(banks[task].spot), (batch_size,), generator=index_rng).to(device)
        starts = torch.randint(config.n_steps - context_length + 1,
                               (batch_size,), generator=index_rng).to(device)
        dates = starts[:, None] + torch.arange(context_length, device=device)[None]
        sample = tuple(value[path_ids[:, None], dates] for value in transitions[task])
        objective = encoder.prediction_loss(*sample, generator=noise_rng)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == updates:
            _sync(device)
            elapsed = time.perf_counter() - started
            record = dict(completed=step, total=updates, task=task, loss=float(objective.detach()),
                          elapsed_seconds=elapsed, eta_seconds=elapsed * (updates-step) / step)
            history.append(record)
            if progress:
                _report("belief_encoder_progress", **record)
    encoder.requires_grad_(False).eval()
    _sync(device)
    metadata = dict(method="belief_fb_encoder_transfer", classification="Source-encoder transfer, not full FB/RFB",
        donor_url=DONOR_URL, donor_commit=DONOR_COMMIT, seed=seed, device=str(device),
        options=options, history=history, training_seconds=time.perf_counter()-started,
        source_history_paths=[len(bank.spot) for bank in banks],
        source_history_transitions=[len(bank.spot) * bank.config.n_steps for bank in banks],
        source_configs=[asdict(bank.config) for bank in banks],
        transition_presentations=updates * batch_size * context_length,
        encoder_parameter_count=sum(parameter.numel() for parameter in encoder.parameters()),
        history_bank_generation_seconds="Caller must add separately",
        objective="Native sampled-Gaussian-context next-state half squared error; no KL or variance objective")
    return encoder, metadata


@torch.no_grad()
def encode_bank_context(encoder, prior_training_bank, *, history_paths=32, context_length=20):
    """Infer a fixed context from completed prior paths, never the evaluated book.

    Uses the first context_length transitions of history_paths prior paths and
    averages their posterior means. Insufficient history fails rather than
    silently changing the declared information budget.
    """
    if (history_paths < 1 or context_length < 1 or history_paths > len(prior_training_bank.spot)
            or context_length > prior_training_bank.config.n_steps):
        raise ValueError("insufficient declared prior paths or transitions")
    parameter = next(encoder.parameters())
    _sync(parameter.device)
    started = time.perf_counter()
    # Slice before transformation so unused histories are neither processed nor
    # counted as encoder observations. Marks and liabilities are never accessed.
    from hedging_gym.finance import bank_subset
    selected = bank_subset(prior_training_bank, slice(0, history_paths))
    transitions = _transitions(selected, device=parameter.device, dtype=parameter.dtype)
    mean, _ = encoder(*(value[:, :context_length] for value in transitions))
    context = mean.mean(dim=0)
    _sync(parameter.device)
    return context, dict(history_paths=history_paths, context_length=context_length,
        history_transitions=history_paths * context_length,
        prior_bank_paths=len(prior_training_bank.spot),
        prior_bank_generated_transitions=len(prior_training_bank.spot) * prior_training_bank.config.n_steps,
        inference_seconds=time.perf_counter()-started,
        context_mean_dispersion=float(mean.std(dim=0, unbiased=False).mean()),
        context_norm=float(context.norm()),
        history_required_before_evaluation=True, history_bank_generation_seconds="Caller must add separately")


class BeliefEmbeddedPolicy(TaskEmbeddedPolicy):
    """TaskEmbeddedPolicy with fixed encoded source vectors and prior inference.

    Compatible with the shared policy factory via functools.partial supplying
    encoder and source_contexts. The common optimizer may list source_embeddings,
    but they stay frozen; only shared DH weights learn during source training.
    After infer_context(), AdaptationUpdater may optimize the same context vector
    through terminal ES. Resets use the freshly inferred baseline, not a lookup
    of evaluation-regime identities or a previously calibrated regime cache.
    """

    def __init__(self, config, *, n_tasks, embedding_dim=4, hidden=(32, 32),
                 encoder, source_contexts):
        super().__init__(config, n_tasks=n_tasks, embedding_dim=embedding_dim, hidden=hidden)
        if (source_contexts.shape != self.source_embeddings.shape
                or embedding_dim != encoder.embedding_dim):
            raise ValueError("source contexts and encoder must match the task-vector geometry")
        self.encoder = deepcopy(encoder).requires_grad_(False).eval()
        with torch.no_grad():
            self.source_embeddings.copy_(source_contexts)
        self.source_embeddings.requires_grad_(False)
        self.register_buffer("prior_context", self.source_embeddings.detach().mean(dim=0).clone())
        self.register_buffer("has_target_history", torch.tensor(False))
        self.reset_embedding()

    def reset_embedding(self):
        with torch.no_grad():
            self.embedding.copy_(self.prior_context)

    def infer_context(self, prior_training_bank, *, history_paths=32, context_length=20):
        self.check_config(prior_training_bank.config)
        vector, metadata = encode_bank_context(self.encoder, prior_training_bank,
            history_paths=history_paths, context_length=context_length)
        with torch.no_grad():
            self.prior_context.copy_(vector)
            self.has_target_history.fill_(True)
        self.active_task = None
        self.reset_embedding()
        return metadata
