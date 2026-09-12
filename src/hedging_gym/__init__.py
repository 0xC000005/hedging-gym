"""Batched hedging research environment; v0.1 alpha, not a production risk engine.

The public API shares one financial implementation across all methods.
"""
from .environment.config import (
    GBMConfig, HestonConfig, BatesConfig, TimeGrid, EuropeanOption, PortfolioConfig,
    ExecutionConfig, RiskConfig, SettlementConfig, HedgingConfig, config_from_dict,
)
from .environment.gym_env import HedgingEnv, HedgingVectorEnv, TensorHedgingEnv
from .environment.instruments import Instrument, VarianceSwap, register_instrument
from .environment.benchmark import benchmark_config, operational_config, adaptation_configs, evaluate_adaptation
from .evaluation import evaluate_controller, empirical_es

__all__ = [
    "GBMConfig", "HestonConfig", "BatesConfig", "HedgingEnv", "HedgingVectorEnv",
    "TensorHedgingEnv", "benchmark_config", "operational_config", "adaptation_configs",
    "evaluate_adaptation", "evaluate_controller", "empirical_es", "TimeGrid",
    "EuropeanOption", "PortfolioConfig", "ExecutionConfig", "RiskConfig", "HedgingConfig",
    "config_from_dict", "Instrument", "VarianceSwap", "register_instrument", "SettlementConfig",
]
