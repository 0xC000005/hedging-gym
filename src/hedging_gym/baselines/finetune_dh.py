"""Ordinary full-network fine-tuning of a pretrained Deep Hedging policy.

Foundation: Bühler et al., https://arxiv.org/abs/1802.03042v1.
This is a local comparison control, not a separate paper reproduction.
See docs/fast-adaptation.md for matched embedding/full-network experiments.
"""
from hedging_gym.baselines._shared.controllers import (
    policy_controller as make_controller,
)

from .deep_hedging import train as train_deep_hedging

__all__ = ["train_online_finetune", "make_controller", "make_updater"]


def make_updater(policy, **options):
    """Fine-tune all policy weights on the current market's training data."""
    from hedging_gym.baselines._shared.adaptation import AdaptationUpdater

    return AdaptationUpdater(policy, mode="finetune", **options)


def train_online_finetune(train_bank, **kwargs):
    """Initial ordinary DH training; subsequent updates use AdaptationUpdater.

    Before any new-market updates, this *is* the ordinary DH policy. A distinct
    adaptation advantage cannot be claimed from this initial training alone.
    """
    policy, metadata = train_deep_hedging(train_bank, **kwargs)
    metadata.update(method="finetune_dh", method_label="Online fine-tuned Deep Hedging",
                    adaptation="Full-network updates on current-stage training paths")
    return policy, metadata
