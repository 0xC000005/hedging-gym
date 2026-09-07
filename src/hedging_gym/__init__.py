"""Batched hedging research environment; v0.1 alpha, not a production risk engine.

The public API shares one financial implementation across all methods.
"""
from .finance import GBMConfig, HestonConfig, BatesConfig
from .gym_env import HedgingEnv, HedgingVectorEnv, TensorHedgingEnv
from .benchmark import benchmark_config, operational_config, adaptation_configs, evaluate_adaptation
from .evaluation import evaluate_controller, empirical_es

__all__ = [
    "GBMConfig", "HestonConfig", "BatesConfig", "HedgingEnv", "HedgingVectorEnv",
    "TensorHedgingEnv", "benchmark_config", "operational_config", "adaptation_configs",
    "evaluate_adaptation", "evaluate_controller", "empirical_es",
]
