"""UX Friction Scoring & Evaluation Engine.

This module quantifies the usability cost of a completed test episode by
computing a normalised **UX Friction Score** (0–100, lower is better) from
four orthogonal dimensions:

1. **Action Bloat** — Did the agent take far more steps than a reasonable
   user would need?  Penalises excessive navigation and indecisive loops.
2. **Unexpected Popups** — How often was the critical path blocked by
   modals, cookie banners, or interstitials the agent had to dismiss?
3. **Backtracking** — How frequently did the agent issue ``go_back``
   actions, indicating confused or dead-end navigation?
4. **Missing Accessibility Labels** — What percentage of interaction
   targets lacked WCAG-compliant accessible names?

Each sub-score is independently normalised to ``[0, 25]``, then summed
for a composite ``[0, 100]`` range.

Public API:
    ``calculate_friction_score(steps, metrics_log)``
        → ``FrictionReport``
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_MAX_SUB_SCORE = 25.0
_OPTIMAL_STEP_ESTIMATE = 5  # Expected steps for a well-designed UX flow


@dataclass(frozen=True)
class FrictionReport:
    """Immutable report summarising friction analysis for one test episode.

    Attributes:
        overall_score: Composite UX Friction Score in ``[0, 100]``.
            ``0`` = frictionless, ``100`` = maximally problematic.
        action_bloat_score: Sub-score for excess step count (0–25).
        popup_score: Sub-score for unexpected popup frequency (0–25).
        backtrack_score: Sub-score for backtracking frequency (0–25).
        a11y_score: Sub-score for missing accessibility labels (0–25).
        total_steps: Total number of steps in the episode.
        popup_count: Number of steps with unexpected popups.
        backtrack_count: Number of ``go_back`` actions.
        missing_label_count: Number of steps with missing a11y labels.
        summary: Human-readable one-line summary.
    """

    overall_score: float
    action_bloat_score: float
    popup_score: float
    backtrack_score: float
    a11y_score: float
    total_steps: int
    popup_count: int
    backtrack_count: int
    missing_label_count: int
    summary: str = field(default="")


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def calculate_friction_score(
    steps: List[Dict[str, Any]],
    metrics_log: List[Dict[str, Any]],
) -> FrictionReport:
    """Compute the normalised UX Friction Score for a completed episode.

    Args:
        steps: Ordered list of ``AgentAction`` dicts (from
            ``AgentState["history"]``).  Each dict must have at
            minimum an ``"action_type"`` key.
        metrics_log: Ordered list of ``ScreenFriction`` dicts (from
            ``AgentState["metrics_log"]``).  Each dict should have
            ``"has_unexpected_popup"`` and ``"missing_a11y_label"``
            boolean keys.

    Returns:
        A ``FrictionReport`` instance with the composite score and
        per-dimension breakdown.

    Example:
        >>> report = calculate_friction_score(steps, metrics_log)
        >>> print(f"Friction: {report.overall_score:.1f}/100")
    """
    total_steps = max(len(steps), 1)

    # -----------------------------------------------------------------
    # 1. Action Bloat sub-score
    # -----------------------------------------------------------------
    bloat_ratio = max(0, total_steps - _OPTIMAL_STEP_ESTIMATE) / _OPTIMAL_STEP_ESTIMATE
    action_bloat_score = min(_MAX_SUB_SCORE, bloat_ratio * _MAX_SUB_SCORE)

    # -----------------------------------------------------------------
    # 2. Unexpected Popup sub-score
    # -----------------------------------------------------------------
    popup_count = sum(
        1 for m in metrics_log
        if m.get("has_unexpected_popup", False)
    )
    popup_ratio = popup_count / total_steps
    popup_score = min(_MAX_SUB_SCORE, popup_ratio * _MAX_SUB_SCORE * 4)
    # Multiplier of 4: even 25% popup rate → full 25-point penalty

    # -----------------------------------------------------------------
    # 3. Backtracking sub-score
    # -----------------------------------------------------------------
    backtrack_count = sum(
        1 for s in steps
        if s.get("action_type") == "go_back"
    )
    backtrack_ratio = backtrack_count / total_steps
    backtrack_score = min(
        _MAX_SUB_SCORE, backtrack_ratio * _MAX_SUB_SCORE * 5
    )
    # Multiplier of 5: even 20% backtrack rate → full penalty

    # -----------------------------------------------------------------
    # 4. Missing Accessibility Labels sub-score
    # -----------------------------------------------------------------
    missing_label_count = sum(
        1 for m in metrics_log
        if m.get("missing_a11y_label", False)
    )
    a11y_ratio = missing_label_count / total_steps
    a11y_score = min(_MAX_SUB_SCORE, a11y_ratio * _MAX_SUB_SCORE * 2)
    # Multiplier of 2: 50%+ missing labels → full penalty

    # -----------------------------------------------------------------
    # Composite
    # -----------------------------------------------------------------
    overall = action_bloat_score + popup_score + backtrack_score + a11y_score
    overall = round(min(100.0, max(0.0, overall)), 2)

    # Human-readable verdict
    if overall <= 15:
        verdict = "Excellent"
    elif overall <= 35:
        verdict = "Good"
    elif overall <= 55:
        verdict = "Fair"
    elif overall <= 75:
        verdict = "Poor"
    else:
        verdict = "Critical"

    summary = (
        f"UX Friction Score: {overall}/100 ({verdict}) — "
        f"{total_steps} steps, {popup_count} popups, "
        f"{backtrack_count} backtracks, "
        f"{missing_label_count} missing a11y labels."
    )

    report = FrictionReport(
        overall_score=overall,
        action_bloat_score=round(action_bloat_score, 2),
        popup_score=round(popup_score, 2),
        backtrack_score=round(backtrack_score, 2),
        a11y_score=round(a11y_score, 2),
        total_steps=total_steps,
        popup_count=popup_count,
        backtrack_count=backtrack_count,
        missing_label_count=missing_label_count,
        summary=summary,
    )

    logger.info(summary)
    return report
