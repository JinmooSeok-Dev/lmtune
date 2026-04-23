"""NSGA-II — Non-dominated Sorting Genetic Algorithm (multi-objective).

Population-based evolutionary sampler that maintains a front of non-dominated
trials over N objectives. Crossover + mutation produce children; the
next generation is chosen by (a) non-dominated rank, (b) crowding distance.

Reference: Deb et al., 2002. Optuna's NSGAIISampler is the canonical
implementation for HPO. For continuous-only problems NSGA-III can be marginally
better, but NSGA-II handles mixed spaces (categorical + continuous) cleanly
and is the standard multi-obj default in 2026.
"""

from __future__ import annotations

import optuna


def make_nsga2(
    seed: int | None = None,
    population_size: int = 20,
) -> optuna.samplers.NSGAIISampler:
    return optuna.samplers.NSGAIISampler(
        seed=seed,
        population_size=int(population_size),
    )


def make_nsga3(
    seed: int | None = None,
    population_size: int = 20,
) -> optuna.samplers.NSGAIIISampler:
    """NSGA-III — reference-point guided variant, often preferred for ≥4 objectives."""
    return optuna.samplers.NSGAIIISampler(
        seed=seed,
        population_size=int(population_size),
    )
