"""GBM backend: exact constant-variance stock steps and Black--Scholes calls.

Physical stock drift is config.mu; pricing uses the shared r=q=0 convention.
This distinction is intentional and does not change the portfolio ledger.
"""
from __future__ import annotations

import torch


def gbm_transition(spot, variance, shocks=None, config=None, *, generator=None):
    """Exact GBM step; channel 2 is the stock normal in the common shock tuple.

    Variance stays constant along each path. Channels 0/1 are unused, keeping
    conditional search compatible with the common three-channel interface.
    """
    if config is None:
        from .finance import GBMConfig
        config = GBMConfig()
    spot, variance = torch.broadcast_tensors(spot, variance)
    if shocks is None:
        normal = torch.randn(spot.shape, dtype=spot.dtype, device=spot.device, generator=generator)
    else:
        shocks = torch.as_tensor(shocks, dtype=spot.dtype, device=spot.device)
        if shocks.shape[-1] != 3:
            raise ValueError("GBM requires three shock channels; stock normal is channel 2")
        spot, variance, normal = torch.broadcast_tensors(spot, variance, shocks[..., 2])
    next_spot = spot * torch.exp((config.mu - .5 * variance) * config.dt
                                + (variance * config.dt).sqrt() * normal)
    return next_spot, variance


def gbm_call_price(spot, variance, maturity, strike, config=None):
    """Exact European call at r=q=0, including zero variance and maturity.

    Internals use float64 and retain autograd, then restore the spot dtype.
    The physical drift mu never enters a risk-neutral option price.
    """
    if config is not None and (config.r != 0 or config.q != 0):
        raise ValueError("the shared Black--Scholes pricer requires r=q=0")
    if not isinstance(spot, torch.Tensor):
        spot = torch.as_tensor(spot, dtype=torch.float64)
    result_dtype, reference = spot.dtype, spot.to(torch.float64)
    s, v, tau, k = torch.broadcast_tensors(reference, *(torch.as_tensor(
        value, dtype=reference.dtype, device=reference.device) for value in (variance, maturity, strike)))
    if not bool((torch.isfinite(s) & torch.isfinite(v) & torch.isfinite(tau) & torch.isfinite(k)
                 & (s > 0) & (v >= 0) & (tau >= 0) & (k > 0)).all()):
        raise ValueError("finite positive spot/strike and nonnegative variance/time required")
    active = (v > 0) & (tau > 0)
    # Make the inactive branch differentiable too: sqrt(0) followed by where
    # would still produce 0*infinite gradients at expiry or zero variance.
    total_variance = torch.where(active, v * tau, torch.ones_like(s))
    std = total_variance.sqrt()
    d1 = (torch.log(s / k) + .5 * total_variance) / std
    d2 = d1 - std
    value = s * torch.special.ndtr(d1) - k * torch.special.ndtr(d2)
    intrinsic = (s - k).clamp_min(0)
    return torch.where(active, value.clamp_min(0), intrinsic).to(result_dtype)
