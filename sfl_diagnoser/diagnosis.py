"""
Extended Spectrum-Based Fault Localization (SFL).

This module adapts classical SFL similarity coefficients (Ochiai,
Tarantula) to work with:

  - Binary spectra  (full or partial traces), and
  - Probabilistic spectra from Rec-Weighted.

Reference:
  Sotto-Mayor et al., "Spectrum-based fault diagnosis with partial traces",
  Section 5.1.2 (Diagnosis via extended SFL), Algorithm 4.
"""

from __future__ import annotations

import math
from typing import Tuple, Callable, Dict, List

from .data_models import SpectrumData


# ---------------------------------------------------------------------------
# Extended contingency table (Algorithm 4)
# ---------------------------------------------------------------------------

def compute_counts(spectrum: SpectrumData, j: int) -> Tuple[float, float, float, float]:
    """
    Compute the extended contingency table (n11, n10, n01, n00) for component j.

    Meaning of the counts (per component j):
      - n11: executed in failing tests
      - n10: executed in passing tests
      - n01: not executed in failing tests
      - n00: not executed in passing tests

    If A_recon is present, we use it and interpret A[i, j] as
    Pr(component j executed in test i). Otherwise, we use A_partial
    (binary 0/1) and get the standard SFL counts.
    """
    # Active spectrum:
    #   - A_recon if available (probabilistic),
    #   - otherwise A_partial (binary).
    A = spectrum.A_recon if spectrum.A_recon is not None else spectrum.A_partial

    e = spectrum.e
    num_tests = spectrum.num_tests

    n11 = n10 = n01 = n00 = 0.0

    for i in range(num_tests):
        # Probability that component j executed in test i
        p_exec = float(A[i, j])

        if e[i] == 1:  # failing test
            n11 += p_exec
            n01 += (1.0 - p_exec)
        else:  # passing test
            n10 += p_exec
            n00 += (1.0 - p_exec)

    return n11, n10, n01, n00


# ---------------------------------------------------------------------------
# Similarity coefficients (Ochiai, Tarantula)
# ---------------------------------------------------------------------------

def ochiai(n11: float, n10: float, n01: float, n00: float) -> float:
    """
    Ochiai suspiciousness.

    Formula (using extended counts):
        score = n11 / sqrt( (n11 + n01) * (n11 + n10) )

    If the denominator is 0, return 0.0.
    """
    denom = math.sqrt((n11 + n01) * (n11 + n10))
    if denom == 0.0:
        return 0.0
    return n11 / denom


def tarantula(n11: float, n10: float, n01: float, n00: float) -> float:
    """
    Tarantula suspiciousness.

    Let:
        fail_rate = n11 / (n11 + n01)   # fraction of failing tests that execute the component
        pass_rate = n10 / (n10 + n00)   # fraction of passing tests that execute the component
        score     = fail_rate / (fail_rate + pass_rate)

    If both rates are 0, return 0.0.
    """
    fail_total = n11 + n01
    pass_total = n10 + n00

    fail_rate = n11 / fail_total if fail_total > 0.0 else 0.0
    pass_rate = n10 / pass_total if pass_total > 0.0 else 0.0

    denom = fail_rate + pass_rate
    if denom == 0.0:
        return 0.0

    return fail_rate / denom


# Registry of available metrics
_METRICS: Dict[str, Callable[[float, float, float, float], float]] = {
    "ochiai": ochiai,
    "tarantula": tarantula,
}


# ---------------------------------------------------------------------------
# High-level diagnosis API
# ---------------------------------------------------------------------------

def run_sfl(spectrum: SpectrumData, metric: str = "ochiai") -> List[Tuple[str, float]]:
    """
    Run (extended) SFL on the given spectrum.

    If spectrum.A_recon is present:
        - Rank all components in C using A_recon (probabilistic spectrum).

    Otherwise:
        - Rank only the visible components C \\ H using A_partial (binary).

    Args:
        spectrum: SpectrumData with A_partial and optionally A_recon.
        metric:   "ochiai" or "tarantula".

    Returns:
        A list of (component_name, suspiciousness_score) sorted
        in descending suspiciousness.
    """
    if metric not in _METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Available: {list(_METRICS.keys())}")

    metric_fn = _METRICS[metric]

    # Choose which components and matrix columns we iterate over:
    #   - With reconstruction: all components in C (aligned with A_recon).
    #   - Without reconstruction: only visible components (aligned with A_partial).
    if spectrum.A_recon is not None:
        components = spectrum.C
    else:
        components = spectrum.visible_components

    scores: List[Tuple[str, float]] = []

    for j, comp in enumerate(components):
        n11, n10, n01, n00 = compute_counts(spectrum, j)
        score = metric_fn(n11, n10, n01, n00)
        scores.append((comp, score))

    # Sort by descending suspiciousness
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
