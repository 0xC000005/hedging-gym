"""Shared Greek calculations and two-sensitivity sizing algebra."""
import torch

from hedging_gym.environment.finance import MarketBank, decode_market_observation
from hedging_gym.interfaces import Controller


def _pair_greeks(spot, variance, time_index, config, *, hedge_index, chunk_size, gamma=False,
                integrated_variance=None):
    if not 0 <= hedge_index < len(config.portfolio.hedges) or chunk_size < 1:
        raise ValueError("select an available hedge option and a positive chunk size")
    liability = config.portfolio.liability
    hedge = config.portfolio.hedges[hedge_index]
    s, v, t = torch.broadcast_tensors(
        spot.detach().to(torch.float64),
        torch.as_tensor(variance, dtype=torch.float64, device=spot.device).detach(),
        torch.as_tensor(time_index, dtype=torch.float64, device=spot.device),
    )
    shape = s.shape
    integral = (None if integrated_variance is None else
                torch.broadcast_to(integrated_variance.to(s), shape).reshape(-1))
    s, v, time = s.reshape(-1), v.reshape(-1), t.reshape(-1) * config.dt
    maturities = torch.stack((liability.maturity - time, hedge.maturity - time), dim=-1)
    if bool((maturities <= 0).any()):
        raise ValueError("hedge Greeks require decisions strictly before settlement")
    deltas, dvariances = [], []
    for offset in range(0, s.numel(), chunk_size):
        sl = slice(offset, offset + chunk_size)
        with torch.enable_grad():
            ss = s[sl, None].expand(-1, 2).clone().requires_grad_(True)
            vv = v[sl, None].expand(-1, 2).clone().requires_grad_(True)
            # The same instrument interface handles options and the variance
            # claim. Past realized variance is held fixed in current-state Greeks.
            realized = None if integral is None else integral[sl]
            prices = torch.stack((
                config.portfolio.liability_quantity * liability.mark(
                    ss[:, 0], vv[:, 0], maturities[sl, 0], config.market, integrated_variance=realized),
                hedge.mark(ss[:, 1], vv[:, 1], maturities[sl, 1], config.market,
                            integrated_variance=realized),
            ), dim=-1)
            if gamma:
                ds, = torch.autograd.grad(prices.sum(), ss, create_graph=True)
                dv, = torch.autograd.grad(ds.sum(), ss)
            else:
                ds, dv = torch.autograd.grad(prices.sum(), (ss, vv))
        deltas.append(ds.detach())
        dvariances.append(dv.detach())
    return (torch.cat(deltas).reshape(shape + (2,)),
            torch.cat(dvariances).reshape(shape + (2,)))


def _two_sensitivity_positions(bank: MarketBank, hedge_index, greeks):
    """Size stock and one hedge from delta and a second supplied sensitivity.

    Callers choose gamma or variance exposure. The holdings algebra is shared:
    match the second exposure first, then offset the remaining stock delta.
    """
    if not 0 <= hedge_index < len(bank.config.portfolio.hedges):
        raise ValueError("select an available hedge option")
    c = bank.config
    lower = c.execution.vector("holding_lower", c.n_assets)
    upper = c.execution.vector("holding_upper", c.n_assets)
    delta, sensitivity = greeks
    if delta.shape != bank.spot[:, :-1].shape + (2,) or sensitivity.shape != delta.shape:
        raise ValueError("cached Greeks must have shape [paths,n_steps,2]")
    if not bool(torch.isfinite(delta).all() and torch.isfinite(sensitivity).all()
                and (sensitivity[..., 1] > 0).all()):
        raise ValueError("hedge sensitivity needs pricing-domain requalification")
    instrument = hedge_index + 1
    hedge = (sensitivity[..., 0] / sensitivity[..., 1]).clamp(lower[instrument], upper[instrument])
    stock = (delta[..., 0] - hedge * delta[..., 1]).clamp(lower[0], upper[0])
    positions = stock.new_zeros((*stock.shape, c.n_assets))
    positions[..., 0] = stock
    positions[..., instrument] = hedge
    return positions.to(bank.spot.dtype)


def two_sensitivity_controller(greek, *, hedge_index=0, chunk_size=1024) -> Controller:
    """Match one secondary exposure, then hedge the remaining spot delta."""
    def control(observed, ledger, time_index, config):
        spot, variance = decode_market_observation(observed, config)
        lower = config.execution.vector("holding_lower", config.n_assets)
        upper = config.execution.vector("holding_upper", config.n_assets)
        ds, exposure = greek(spot, variance, time_index, config, hedge_index=hedge_index, chunk_size=chunk_size)
        if not bool(torch.isfinite(ds).all() and torch.isfinite(exposure).all() and (exposure[:, 1] > 0).all()):
            raise ValueError("hedge sensitivity needs pricing-domain requalification")
        instrument = hedge_index + 1
        hedge = (exposure[:, 0]/exposure[:, 1]).clamp(lower[instrument], upper[instrument])
        stock = (ds[:, 0]-hedge*ds[:, 1]).clamp(lower[0], upper[0])
        targets = torch.zeros_like(ledger.positions)
        targets[:, 0], targets[:, instrument] = stock, hedge
        return targets
    control.action_selection = "deterministic"
    return control
