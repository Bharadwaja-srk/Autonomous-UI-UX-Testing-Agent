"""Standalone HTML Audit Dashboard Generator.

This module renders a **single, self-contained HTML file** that serves as
the post-run audit report for a completed test episode.  The report is
designed to be shared with stakeholders without any server infrastructure.

Highlights:
    - Styled with **Tailwind CSS via CDN** for a responsive, modern layout.
    - Step screenshots are embedded as **base64 data-URIs** so the report
      works entirely offline (only the Tailwind CDN requires connectivity
      for initial CSS load, with a graceful degradation fallback).
    - Step-by-step timeline displays the agent's chain-of-thought, action
      taken, and friction flags with colour-coded severity badges.
    - A visual friction-score gauge provides at-a-glance UX quality.
    - Accessibility compliance flaws are highlighted with distinct
      warning indicators.

Public API:
    ``generate_html_report(run_id, task_goal, steps_list, friction_score)``
        → HTML string
"""

from __future__ import annotations

import base64
import html
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluator import FrictionReport

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def generate_html_report(
    run_id: str,
    task_goal: str,
    steps_list: List[Dict[str, Any]],
    friction_report: FrictionReport,
    screenshot_dir: Optional[Path] = None,
) -> str:
    """Generate a standalone HTML audit dashboard for a test episode.

    Args:
        run_id: Unique identifier for this test run (e.g., a timestamp
            or UUID slug).
        task_goal: Natural-language description of the test objective.
        steps_list: Ordered list of step dicts, each containing at
            minimum: ``step_number``, ``thought``, ``action_type``,
            ``target_mark_id``, ``friction_metrics`` (dict with
            ``has_unexpected_popup``, ``missing_a11y_label``,
            ``friction_notes``).
        friction_report: The ``FrictionReport`` instance produced by
            the evaluator.
        screenshot_dir: Optional path to the directory containing
            ``step_NNN.png`` screenshots.  If provided, screenshots are
            embedded as base64 in the report.

    Returns:
        A complete HTML string ready to be written to a ``.html`` file.

    Example:
        >>> html_content = generate_html_report(
        ...     run_id="20260918_152000",
        ...     task_goal="Navigate to checkout",
        ...     steps_list=steps,
        ...     friction_report=report,
        ...     screenshot_dir=Path("./runs/20260918_152000"),
        ... )
        >>> Path("report.html").write_text(html_content, encoding="utf-8")
    """
    logger.info("Generating HTML report for run '%s'.", run_id)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    steps_html = _render_steps(steps_list, screenshot_dir)
    score_color = _score_color(friction_report.overall_score)
    verdict = _score_verdict(friction_report.overall_score)

    report_html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UX Audit Report — {html.escape(run_id)}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* Graceful fallback if Tailwind CDN is unavailable */
        body {{ font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif; }}
        .gauge-ring {{ transition: stroke-dashoffset 1s ease-in-out; }}
    </style>
</head>
<body class="bg-gray-950 text-gray-100 min-h-screen">

    <!-- ═══════════════ Header ═══════════════ -->
    <header class="bg-gradient-to-r from-indigo-900 via-purple-900 to-indigo-900
                    border-b border-indigo-700/40 px-6 py-8">
        <div class="max-w-6xl mx-auto">
            <h1 class="text-3xl font-bold text-white tracking-tight">
                🔍 Autonomous UX Audit Report
            </h1>
            <p class="mt-2 text-indigo-300 text-sm">
                Run ID: <code class="bg-indigo-800/50 px-2 py-0.5 rounded text-xs">
                    {html.escape(run_id)}</code>
                &nbsp;·&nbsp; Generated: {timestamp}
            </p>
        </div>
    </header>

    <main class="max-w-6xl mx-auto px-6 py-8 space-y-10">

        <!-- ═══════════════ Goal ═══════════════ -->
        <section class="bg-gray-900 rounded-xl border border-gray-800 p-6">
            <h2 class="text-lg font-semibold text-gray-200 mb-2">🎯 Test Goal</h2>
            <p class="text-gray-400 leading-relaxed">{html.escape(task_goal)}</p>
        </section>

        <!-- ═══════════════ Friction Score ═══════════════ -->
        <section class="bg-gray-900 rounded-xl border border-gray-800 p-6">
            <h2 class="text-lg font-semibold text-gray-200 mb-6">📊 UX Friction Score</h2>
            <div class="flex flex-col md:flex-row items-center gap-8">

                <!-- SVG Gauge -->
                <div class="relative w-40 h-40 flex-shrink-0">
                    <svg viewBox="0 0 120 120" class="w-full h-full -rotate-90">
                        <circle cx="60" cy="60" r="52" fill="none"
                                stroke="#1e293b" stroke-width="10"/>
                        <circle cx="60" cy="60" r="52" fill="none"
                                stroke="{score_color}" stroke-width="10"
                                stroke-linecap="round"
                                class="gauge-ring"
                                stroke-dasharray="{2 * 3.14159 * 52:.1f}"
                                stroke-dashoffset="{2 * 3.14159 * 52 * (1 - friction_report.overall_score / 100):.1f}"/>
                    </svg>
                    <div class="absolute inset-0 flex flex-col items-center justify-center">
                        <span class="text-3xl font-bold" style="color: {score_color}">
                            {friction_report.overall_score:.0f}
                        </span>
                        <span class="text-xs text-gray-500">/ 100</span>
                    </div>
                </div>

                <!-- Sub-scores -->
                <div class="grid grid-cols-2 gap-4 flex-1 w-full">
                    {_render_sub_score("Action Bloat", friction_report.action_bloat_score, "🚶")}
                    {_render_sub_score("Popup Friction", friction_report.popup_score, "🪟")}
                    {_render_sub_score("Backtracking", friction_report.backtrack_score, "🔄")}
                    {_render_sub_score("A11y Gaps", friction_report.a11y_score, "♿")}
                </div>
            </div>
            <p class="mt-4 text-sm text-gray-500">
                Verdict: <span class="font-semibold" style="color: {score_color}">{verdict}</span>
                &nbsp;·&nbsp; {friction_report.total_steps} steps
                &nbsp;·&nbsp; {friction_report.popup_count} popups
                &nbsp;·&nbsp; {friction_report.backtrack_count} backtracks
                &nbsp;·&nbsp; {friction_report.missing_label_count} missing labels
            </p>
        </section>

        <!-- ═══════════════ Step Timeline ═══════════════ -->
        <section class="bg-gray-900 rounded-xl border border-gray-800 p-6">
            <h2 class="text-lg font-semibold text-gray-200 mb-6">
                📋 Step-by-Step Timeline
            </h2>
            <div class="space-y-6">
                {steps_html}
            </div>
        </section>

    </main>

    <!-- ═══════════════ Footer ═══════════════ -->
    <footer class="border-t border-gray-800 mt-12 px-6 py-6 text-center text-gray-600 text-xs">
        Autonomous Agentic UI/UX Testing Framework — P8 &nbsp;·&nbsp;
        Report generated automatically. Lower friction score = better UX.
    </footer>

</body>
</html>"""

    logger.info(
        "HTML report generated: %d bytes, %d steps.",
        len(report_html),
        len(steps_list),
    )
    return report_html


# ═══════════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════════

def _render_steps(
    steps_list: List[Dict[str, Any]],
    screenshot_dir: Optional[Path],
) -> str:
    """Render the HTML for each step in the timeline.

    Args:
        steps_list: Ordered step dicts.
        screenshot_dir: Optional directory with step PNG screenshots.

    Returns:
        Concatenated HTML string for all steps.
    """
    fragments: List[str] = []

    for step in steps_list:
        step_num = step.get("step_number", "?")
        thought = html.escape(step.get("thought", ""))
        action_type = step.get("action_type", "unknown")
        mark_id = step.get("target_mark_id")
        text_input = step.get("text_input")

        friction = step.get("friction_metrics", {})
        has_popup = friction.get("has_unexpected_popup", False)
        missing_label = friction.get("missing_a11y_label", False)
        friction_notes = html.escape(friction.get("friction_notes", ""))

        # Build action summary
        action_summary = f"<code class=\"text-cyan-400\">{html.escape(action_type)}</code>"
        if mark_id is not None:
            action_summary += f" → mark <strong>[{mark_id}]</strong>"
        if text_input:
            action_summary += (
                f" → <span class=\"text-yellow-400\">"
                f"\"{html.escape(text_input[:60])}\"</span>"
            )

        # Friction badges
        badges = ""
        if has_popup:
            badges += (
                '<span class="inline-block bg-orange-900/60 text-orange-300 '
                'text-xs px-2 py-0.5 rounded-full mr-1">⚠ Popup</span>'
            )
        if missing_label:
            badges += (
                '<span class="inline-block bg-red-900/60 text-red-300 '
                'text-xs px-2 py-0.5 rounded-full mr-1">♿ No Label</span>'
            )

        # Embedded screenshot (base64)
        screenshot_html = ""
        if screenshot_dir is not None:
            png_path = screenshot_dir / f"step_{step_num:03d}.png"
            if png_path.exists():
                try:
                    b64 = base64.b64encode(
                        png_path.read_bytes()
                    ).decode("ascii")
                    screenshot_html = (
                        f'<img src="data:image/png;base64,{b64}" '
                        f'alt="Step {step_num} screenshot" '
                        f'class="mt-3 rounded-lg border border-gray-700 '
                        f'max-w-full shadow-lg" loading="lazy"/>'
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not embed screenshot for step %s: %s",
                        step_num, exc,
                    )

        fragment = f"""\
<div class="bg-gray-800/60 rounded-lg border border-gray-700/50 p-4
            hover:border-indigo-600/40 transition-colors">
    <div class="flex items-start gap-3">
        <div class="flex-shrink-0 w-10 h-10 rounded-full bg-indigo-900/80
                    flex items-center justify-center text-sm font-bold
                    text-indigo-300 border border-indigo-700/50">
            {step_num}
        </div>
        <div class="flex-1 min-w-0">
            <div class="flex flex-wrap items-center gap-2 mb-1">
                <span class="text-sm font-medium text-gray-300">
                    Action: {action_summary}
                </span>
                {badges}
            </div>
            <p class="text-xs text-gray-500 leading-relaxed mb-1">
                💭 {thought}
            </p>
            {"<p class='text-xs text-orange-400/80 mt-1'>📝 " + friction_notes + "</p>" if friction_notes else ""}
            {screenshot_html}
        </div>
    </div>
</div>"""
        fragments.append(fragment)

    return "\n".join(fragments)


def _render_sub_score(label: str, score: float, icon: str) -> str:
    """Render a single sub-score card for the friction breakdown grid.

    Args:
        label: Human-readable dimension name.
        score: Score value in [0, 25].
        icon: Emoji icon for visual identification.

    Returns:
        HTML string for one sub-score card.
    """
    pct = min(100.0, (score / 25.0) * 100)
    bar_color = _score_color(score * 4)  # Scale 0–25 → 0–100 for coloring
    return f"""\
<div class="bg-gray-800/50 rounded-lg p-3 border border-gray-700/30">
    <div class="flex justify-between items-center mb-1">
        <span class="text-xs text-gray-400">{icon} {html.escape(label)}</span>
        <span class="text-xs font-mono text-gray-300">{score:.1f}/25</span>
    </div>
    <div class="w-full bg-gray-700 rounded-full h-1.5">
        <div class="h-1.5 rounded-full transition-all duration-700"
             style="width: {pct:.0f}%; background: {bar_color}"></div>
    </div>
</div>"""


def _score_color(score: float) -> str:
    """Map a friction score (0–100) to a CSS colour.

    Low scores → green (good UX), high scores → red (poor UX).

    Args:
        score: Friction score in ``[0, 100]``.

    Returns:
        CSS hex colour string.
    """
    if score <= 20:
        return "#22c55e"  # green-500
    elif score <= 40:
        return "#84cc16"  # lime-500
    elif score <= 60:
        return "#eab308"  # yellow-500
    elif score <= 80:
        return "#f97316"  # orange-500
    else:
        return "#ef4444"  # red-500


def _score_verdict(score: float) -> str:
    """Map a friction score to a human-readable verdict label.

    Args:
        score: Friction score in ``[0, 100]``.

    Returns:
        Verdict string (e.g., ``"Excellent"``, ``"Critical"``).
    """
    if score <= 15:
        return "Excellent"
    elif score <= 35:
        return "Good"
    elif score <= 55:
        return "Fair"
    elif score <= 75:
        return "Poor"
    else:
        return "Critical"
