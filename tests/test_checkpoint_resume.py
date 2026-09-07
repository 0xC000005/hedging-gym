"""An interrupted learner must continue the same training trajectory."""
import pytest
import torch

from hedging_gym import benchmark_config
from hedging_gym.config import TimeGrid
from hedging_gym.finance import generate_market_bank
from methods.training import train_policy
from methods.hybrid import train_hybrid


@pytest.mark.parametrize("method", ["dh", "ntb", "hpo"])
def test_split_training_matches_uninterrupted(method, tmp_path):
    bank = generate_market_bank(benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3)),
                                24, 440, dtype=torch.float64)
    def train(**kwargs):
        return (train_hybrid(bank, **kwargs) if method == "hpo"
                else train_policy(method, bank, **kwargs))
    options = dict(seed=7, batch_size=8, hidden=(8,), progress=False)
    full, full_meta = train(updates=4, **options)
    path = tmp_path / "latest.pt"
    train(updates=2, checkpoint_path=path, checkpoint_every=1, **options)
    resumed, meta = train(updates=4, resume_from=path, checkpoint_path=path, **options)
    for name, value in full.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
        else:
            assert value == resumed.state_dict()[name]
    assert meta["zeta"] == full_meta["zeta"]
    assert (tmp_path / "latest-early.pt").exists()
    saved = torch.load(path, weights_only=False)
    assert saved["step"] == 4 and saved["optimizer"]["state"]
    with pytest.raises(ValueError, match="precede"):
        train(updates=1, resume_from=path, **options)
