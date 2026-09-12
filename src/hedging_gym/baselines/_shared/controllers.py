"""Shared adapter for policies returning PolicyAction records.

Only the environment executes trades. Inputs contain current observations and
read-only ledger state; the adapter does not receive evaluation futures.
"""
from contextlib import contextmanager, nullcontext

from hedging_gym.interfaces import Controller


@contextmanager
def _evaluation_mode(module):
    training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(training)


def policy_controller(policy, *, evaluation=True) -> Controller:
    """Adapt a PolicyAction network to target holdings without detaching gradients.

    Evaluation temporarily selects eval mode and restores the original mode.
    Training and differentiable planning use ``evaluation=False`` to leave the
    caller's mode untouched. This adapter never updates network weights.
    """
    def control(observed, ledger, time_index, config):
        policy.check_config(config)
        with _evaluation_mode(policy) if evaluation else nullcontext():
            return policy(observed, ledger.positions, config.execution.holding_lower,
                          config.execution.holding_upper, deterministic=True).target_holdings
    control.action_selection = "deterministic"
    return control
