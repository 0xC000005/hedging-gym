"""GEPS context-controlled low-rank layers for the common hedging policy.

Paper: Kassaï Koupaï et al., NeurIPS 2024,
https://arxiv.org/abs/2410.23889, equations (4)--(5).
Upstream: https://github.com/itsakk/geps/tree/e9a865218ecffacb7007ac7d719f3741afcf8c02
Implementation notes: docs/geps-adaptation.md.

This layer transfer jointly fits shared matrices and source contexts; there is
no inner meta-gradient loop. Adaptation fits a new task context only. Source
checks concern the layer equations, not reproduction of a PDE experiment.
"""

import math

import torch
from torch import nn

from .adaptive_deep_hedging import TaskEmbeddedPolicy

GEPS_SOURCE_COMMIT = "e9a865218ecffacb7007ac7d719f3741afcf8c02"
GEPS_SOURCE_URL = f"https://github.com/itsakk/geps/tree/{GEPS_SOURCE_COMMIT}"


class GEPSLinear(nn.Module):
    """Row-vector form: x(W + A diag(c) B) + b + c b_context.

    The factor is fixed to one, as in the paper equations. Reassociation avoids
    materializing one full weight matrix per path, without changing the layer
    or its gradients. Parameters use the author's [input, output] orientation.
    """

    def __init__(self, in_features, out_features, context_dim):
        super().__init__()
        if min(in_features, out_features, context_dim) < 1:
            raise ValueError("GEPS layer dimensions must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.context_dim = context_dim
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.A = nn.Parameter(torch.empty(in_features, context_dim))
        self.B = nn.Parameter(torch.empty(context_dim, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.bias_context = nn.Parameter(torch.empty(context_dim, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Same distributions as the author's GEPSLinear.reset_parameters.
        # Its [input, output] weight orientation implies fan_in=out_features.
        for parameter in (self.weight, self.A, self.B):
            nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.out_features)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.bias_context, -bound, bound)

    def forward(self, inputs, context):
        return (inputs @ self.weight + ((inputs @ self.A) * context) @ self.B
                + self.bias + context @ self.bias_context)


class _GEPSNetwork(nn.Module):
    """Accept the unchanged TaskEmbeddedPolicy concatenation interface."""

    def __init__(self, feature_dim, output_dim, context_dim, hidden):
        super().__init__()
        dimensions = (feature_dim, *hidden, output_dim)
        if any(width < 1 for width in dimensions) or context_dim < 1:
            raise ValueError("network dimensions must be positive")
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.layers = nn.ModuleList(
            GEPSLinear(left, right, context_dim)
            for left, right in zip(dimensions[:-1], dimensions[1:])
        )

    def forward(self, inputs):
        features, context = inputs.split((self.feature_dim, self.context_dim), dim=-1)
        for layer in self.layers[:-1]:
            features = torch.tanh(layer(features, context))
        return self.layers[-1](features, context)


class GEPSPolicy(TaskEmbeddedPolicy):
    """GEPS conditioning with the unchanged financial and adaptation interface.

    The same context modulates every layer. Source embeddings, new-context mean
    initialization, bounded holdings, observations and context-only freezing are
    inherited. The common trainer jointly optimizes ``shared`` and
    ``source_embeddings`` exactly as for TaskEmbeddedPolicy.
    """

    method_name = "geps_dh"
    method_label = "GEPS internal-context Deep Hedging"

    def __init__(self, config, *, n_tasks, embedding_dim=4, hidden=(64, 64)):
        super().__init__(config, n_tasks=n_tasks, embedding_dim=embedding_dim, hidden=hidden)
        self.shared = _GEPSNetwork(self.feature_dim, self.n_assets, embedding_dim, hidden)
        # Native geps/model/forecasters.py: Derivative initializes all codes at zero.
        with torch.no_grad():
            self.source_embeddings.zero_()
