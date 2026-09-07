"""Bates backend: Heston QE diffusion plus independent lognormal Poisson jumps.

Jump aggregation is exact within each step. The continuous component remains
the disclosed, approximate Heston QE/log-spot scheme. P/Q share jump parameters
and zero drift here; the compensator preserves the jump factor's mean of one.
"""
from __future__ import annotations

import math

import torch


def jump_compensator(config):
    """E[exp(Y)-1] for Y ~ Normal(jump_mean, jump_std²)."""
    return math.expm1(config.jump_mean + .5 * config.jump_std**2)


def jump_characteristic(u, maturity, config):
    """Compensated independent log-jump characteristic-function factor."""
    mark_exponent = 1j * u * config.jump_mean - .5 * config.jump_std**2 * u.square()
    return torch.exp(config.jump_intensity * maturity
                     * (torch.expm1(mark_exponent) - 1j * u * jump_compensator(config)))


def bates_transition(spot, variance, shocks=None, config=None, *, dt, generator=None):
    """QE diffusion times exp(-lambda*k*dt + N*mean + sqrt(N)*std*Z).

    Explicit shocks have five channels: normalV, uniformV, normalS,
    PoissonCount, normalJump. The count is already sampled at lambda*dt by the
    caller. Three-channel input is rejected rather than silently losing jumps.
    Without explicit shocks, Torch draws the count and independent normal from
    the supplied generator; no inverse-CDF or count truncation is introduced.
    """
    from .config import BatesConfig
    from .finance import heston_transition
    config = config or BatesConfig()
    if shocks is not None:
        shocks = torch.as_tensor(shocks, dtype=spot.dtype, device=spot.device)
        if shocks.shape[-1] != 5:
            raise ValueError("Bates requires five shock channels, including Poisson count and jump normal")
        count, normal = shocks[..., 3], shocks[..., 4]
        if not bool((torch.isfinite(count) & (count >= 0) & (count == count.floor())
                     & torch.isfinite(normal)).all()):
            raise ValueError("explicit jump counts must be nonnegative integers and normals finite")
        if config.jump_intensity == 0 and bool((count != 0).any()):
            raise ValueError("zero jump intensity requires zero explicit counts")
    continuous, next_variance = heston_transition(
        spot, variance, None if shocks is None else shocks[..., :3], config, dt=dt, generator=generator)
    # Preserve the complete Heston generator stream when jumps are disabled.
    if config.jump_intensity == 0:
        return continuous, next_variance
    if shocks is None:
        rate = torch.full_like(continuous, config.jump_intensity * dt)
        count = torch.poisson(rate, generator=generator)
        normal = torch.randn(continuous.shape, dtype=continuous.dtype, device=continuous.device,
                             generator=generator)
    log_jump = (-config.jump_intensity * jump_compensator(config) * dt
                + count * config.jump_mean + count.sqrt() * config.jump_std * normal)
    return continuous * log_jump.exp(), next_variance


def bates_call_price(spot, variance, maturity, strike, config=None):
    """Use the shared Fourier pricer with the Bates characteristic factor."""
    from .config import BatesConfig
    from .finance import heston_call_price
    return heston_call_price(spot, variance, maturity, strike, config or BatesConfig())
