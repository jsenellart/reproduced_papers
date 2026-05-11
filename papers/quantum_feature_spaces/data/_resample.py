"""Iterative filter-and-resample helper used by all dataset generators.

Algorithm
---------
1. Draw ``target_size`` raw samples (one batch).
2. **Hard filter** by ``min_margin`` — these samples are gone forever.
3. **Estimate** what the post-balance yield would be (without actually
   discarding any majority-class samples — they can rebalance naturally as
   more draws come in):

       balance_yield = 2 · min(n_class_0, n_class_1) / target_size   if balanced
       balance_yield = len(kept) / target_size                       otherwise

4. If ``balance_yield < low_survival_threshold`` after the first batch,
   **raise** :class:`LowYieldError` — silently returning a short dataset
   would produce misleading downstream metrics.  Pass
   ``low_survival_threshold=0`` (or ``--bail-threshold 0`` from the CLI)
   to disable the check and force iterative resampling.
5. Otherwise iterate: draw more raw, hard-filter by margin, accumulate.
   Estimate the deficit (for the minority class when balanced) and the
   per-raw-sample yield (= margin_rate × minority-class fraction) to size
   the next draw.  Stop when ``balance_yield >= 1``.
6. Apply the actual balancing **only at the end**, so any majority-class
   surplus accumulated during iterations is reused if the minority class
   catches up.

This makes ``--dataset-size N`` mean *the post-filter, post-balance size*
across all generators, with one consistent UX.
"""

from __future__ import annotations

from typing import Callable

import torch


class LowYieldError(RuntimeError):
    """Raised when the first-batch post-balance yield is below the threshold.

    Attributes
    ----------
    margin_survival : float
        Fraction of the first batch that survived the hard margin filter.
    balance_yield : float
        Estimated post-balance yield ``= 2·min(n0,n1) / target_size``.
    threshold : float
        The configured ``low_survival_threshold`` that triggered the error.
    samples_kept : int
        How many samples survived margin filtering on the first batch.
    target_size : int
        The originally requested dataset size.
    """

    def __init__(
        self,
        margin_survival: float,
        balance_yield: float,
        threshold: float,
        samples_kept: int,
        target_size: int,
    ) -> None:
        self.margin_survival = margin_survival
        self.balance_yield = balance_yield
        self.threshold = threshold
        self.samples_kept = samples_kept
        self.target_size = target_size
        super().__init__(
            f"First-batch yield too low: "
            f"margin_survival={margin_survival:.1%}, "
            f"balance_yield={balance_yield:.1%} < threshold {threshold:.0%}. "
            f"Only {samples_kept}/{target_size} samples survived margin filtering, "
            f"and the post-balance estimate is below the cutoff. "
            f"Either reduce --min-margin, drop --balanced, or pass "
            f"--bail-threshold 0 to force iterative resampling (potentially expensive)."
        )


def filter_resample(
    target_size: int,
    balanced: bool,
    min_margin: float,
    draw_fn: Callable[[int], tuple[torch.Tensor, ...]],
    *,
    low_survival_threshold: float = 0.30,
    max_iter: int = 10,
    perm_seed: int = 0,
) -> tuple[tuple[torch.Tensor, ...], dict]:
    """Iteratively call ``draw_fn`` to obtain ``target_size`` post-filter samples.

    Parameters
    ----------
    target_size:
        Final number of samples to return.
    balanced:
        If True, return ``target_size // 2`` rows per class.
    min_margin:
        Hard-filter threshold: drop samples with ``confidence < min_margin``.
        ``confidence`` is the last tensor returned by ``draw_fn``.
    draw_fn:
        ``draw_fn(n) -> (X, y, *extras, confidence)``. All returned tensors
        must have first-dim length ``n``.

        - ``X``: feature tensor of shape ``(n, ...)``
        - ``y``: integer labels of shape ``(n,)`` in ``{0, 1}``
        - ``extras``: any tensors that travel with samples (parity, scores,
          probs ...).  Arbitrary shape ``(n, ...)``.
        - ``confidence``: 1D tensor of length ``n`` in ``[0, 1]``, used for
          margin filtering.
    low_survival_threshold:
        If the first-batch *post-balance* yield is below this fraction,
        bail out without resampling.  Default 0.30.
    max_iter:
        Maximum number of resample iterations after the initial batch.
    perm_seed:
        RNG seed for the final shuffle when ``balanced=True``.

    Returns
    -------
    samples:
        Tuple ``(X, y, *extras)`` — same arity as ``draw_fn`` minus the
        confidence column.  First-dim length is ``target_size``.
    info:
        Diagnostic dict:

        - ``total_raw_drawn`` (int): total raw samples drawn across all
          iterations (useful for cost accounting).
        - ``margin_survival`` (float): fraction of the first batch that
          survived the hard margin filter.
        - ``balance_yield`` (float): the *first-batch* estimate of
          ``post-balance / target_size``.
        - ``n_iters`` (int): number of resample iterations performed.

    Raises
    ------
    LowYieldError
        If the first-batch post-balance yield is below
        ``low_survival_threshold``.  Pass ``low_survival_threshold=0`` to
        disable this check.
    """
    def _draw_and_margin(n: int) -> tuple[torch.Tensor, ...]:
        """Draw n raw samples and hard-filter by min_margin."""
        result = draw_fn(n)
        if len(result) < 3:
            raise ValueError(
                "draw_fn must return at least (X, y, confidence); got "
                f"{len(result)} tensors."
            )
        conf = result[-1]
        tensors = result[:-1]
        if min_margin > 0.0:
            keep = conf >= min_margin
            tensors = tuple(t[keep] for t in tensors)
        return tensors

    def _yields(tensors: tuple[torch.Tensor, ...]) -> tuple[float, float]:
        """Return (margin_yield, balance_yield), both as fractions of target_size.

        - margin_yield  = len(kept) / target_size               (always ≥ 0)
        - balance_yield = 2·min(n0, n1) / target_size            if balanced
                       = margin_yield                            otherwise
        """
        X, y = tensors[0], tensors[1]
        if target_size <= 0:
            return 1.0, 1.0
        m_yield = len(X) / target_size
        if balanced:
            n0 = int((y == 0).sum().item())
            n1 = int((y == 1).sum().item())
            b_yield = (2 * min(n0, n1)) / target_size
        else:
            b_yield = m_yield
        return m_yield, b_yield

    def _final_balance(
        tensors: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Apply balancing (if requested) and trim to exactly target_size."""
        X, y = tensors[0], tensors[1]
        if balanced and len(X) > 0:
            n_per_class = target_size // 2
            idx0 = (y == 0).nonzero(as_tuple=True)[0]
            idx1 = (y == 1).nonzero(as_tuple=True)[0]
            n_each = min(n_per_class, len(idx0), len(idx1))
            idx = torch.cat([idx0[:n_each], idx1[:n_each]])
            perm = torch.randperm(
                len(idx), generator=torch.Generator().manual_seed(perm_seed)
            )
            return tuple(t[idx[perm]] for t in tensors)
        if len(X) > target_size:
            return tuple(t[:target_size] for t in tensors)
        return tensors

    # ---- First pass: draw target_size raw samples, hard-filter by margin ----
    tensors = _draw_and_margin(target_size)
    total_raw = target_size
    margin_survival, first_balance_yield = _yields(tensors)

    # Bail-out: post-balance yield too low ⇒ resampling won't help much.
    # Raise instead of returning a short dataset — silently truncating produces
    # misleading downstream metrics.  Pass low_survival_threshold=0 to disable.
    if first_balance_yield < low_survival_threshold:
        raise LowYieldError(
            margin_survival=margin_survival,
            balance_yield=first_balance_yield,
            threshold=low_survival_threshold,
            samples_kept=len(tensors[0]),
            target_size=target_size,
        )

    # ---- Iterative top-up ----
    # Balance is NOT applied during iterations: any majority-class surplus is
    # kept and may be matched by future minority-class draws.
    n_iters = 0
    for _ in range(max_iter):
        _, b_yield = _yields(tensors)
        if b_yield >= 1.0:
            break

        X, y = tensors[0], tensors[1]
        if balanced:
            n_per_class = target_size // 2
            n0 = int((y == 0).sum().item())
            n1 = int((y == 1).sum().item())
            deficit_per_class = max(n_per_class - n0, n_per_class - n1)
            kept = max(len(y), 1)
            # Per-raw-sample yield for the minority class:
            #   margin_rate × minority_class_fraction
            class_rate = max(min(n0, n1) / kept, 0.05)
            margin_rate = max(kept / total_raw, 0.05)
            eff_rate = max(margin_rate * class_rate, 0.05)
            raw_to_draw = max(int(deficit_per_class / eff_rate * 1.5), 100)
        else:
            deficit = target_size - len(X)
            margin_rate = max(len(X) / total_raw, 0.05)
            eff_rate = max(margin_rate, 0.05)
            raw_to_draw = max(int(deficit / eff_rate * 1.5), 100)

        new_tensors = _draw_and_margin(raw_to_draw)
        total_raw += raw_to_draw
        tensors = tuple(torch.cat([t, e]) for t, e in zip(tensors, new_tensors))
        n_iters += 1

    # ---- Final balance + trim ----
    return _final_balance(tensors), {
        "total_raw_drawn": total_raw,
        "margin_survival": margin_survival,
        "balance_yield": first_balance_yield,
        "n_iters": n_iters,
    }
