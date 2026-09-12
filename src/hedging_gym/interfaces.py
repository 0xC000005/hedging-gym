"""The decision contract shared by neural, classical and search controllers."""
from typing import Protocol

from torch import Tensor

from hedging_gym.environment.config import HedgingConfig
from hedging_gym.environment.finance import LedgerState


class Controller(Protocol):
    """Return absolute target holdings [batch, n_assets] on the input device.

    Inputs are read-only; only the financial environment updates the ledger.
    Controllers may use different training libraries or perform explicit search.
    They are not required to be neural networks or to implement training.
    """

    def __call__(self, observed: Tensor, ledger: LedgerState, time_index: int,
                 config: HedgingConfig) -> Tensor: ...
