"""
API cost guard — tracks running API costs and enforces a hard cap.
Only relevant when --use-api flag is set. Local inference = $0 cost.
"""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


class CostGuard:
    """
    Tracks cumulative API cost and enforces a hard limit.

    Usage:
        guard = CostGuard(max_usd=20.0)
        guard.check_estimate(estimated_cost)  # prompts user, raises if cap exceeded
        guard.add(actual_cost)
    """

    def __init__(self, max_usd: float = 20.0) -> None:
        self._max = max_usd
        self._total = 0.0

    @property
    def total(self) -> float:
        return self._total

    @property
    def remaining(self) -> float:
        return max(0.0, self._max - self._total)

    def add(self, cost: float) -> None:
        self._total += cost
        if self._total > self._max:
            raise RuntimeError(
                f"API cost cap exceeded: ${self._total:.2f} > ${self._max:.2f} hard limit. "
                f"Increase max_api_cost_usd in config to continue."
            )

    def check_estimate(self, estimated_cost: float, auto_confirm: bool = False) -> None:
        """
        Show estimate and prompt user to confirm before proceeding.
        Raises SystemExit if user declines or cap would be exceeded.

        Args:
            estimated_cost: projected cost for this run in USD
            auto_confirm: skip prompt (for non-interactive use)
        """
        projected_total = self._total + estimated_cost

        if projected_total > self._max:
            raise RuntimeError(
                f"Estimated cost ${estimated_cost:.2f} would exceed cap "
                f"(current: ${self._total:.2f}, max: ${self._max:.2f}). "
                f"Increase max_api_cost_usd in config or use local weights."
            )

        if auto_confirm:
            logger.info("Estimated API cost: $%.2f (total: $%.2f)", estimated_cost, projected_total)
            return

        print(f"\nEstimated API cost: ${estimated_cost:.2f}")
        print(f"Running total will be: ${projected_total:.2f} / ${self._max:.2f} cap")
        response = input("Proceed? [y/N] ").strip().lower()
        if response not in ("y", "yes"):
            print("Cancelled.")
            sys.exit(0)
