"""LangGraph State Machine & CLI Orchestrator for Autonomous UI/UX Testing.

This is the **brain** of the P8 framework.  It wires the four functional
layers—perception, reasoning, execution, and evaluation—into a cyclic
LangGraph state machine and exposes an ``argparse`` CLI for end-users.

Architecture:
    ┌──────────────┐
    │  perceive    │ ← capture_and_annotate → annotated image + a11y text
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │   reason     │ ← decide_next_step → AgentAction
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │   execute    │ ← BlackBoxDriver.execute_action → browser interaction
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │  evaluate    │ ← routing decision: FINISH → END, else → perceive
    └──────┬───────┘
           │ loop ↩

Key Features:
    - **Self-Healing Loop Detection**: MD5 hash of each screenshot is
      computed.  If N consecutive screenshots hash identically, a warning
      is injected into history and a ``go_back`` strategy is forced.
    - **Ghost Mode**: ``--demo-mode`` replays cached artifacts without
      calling the VLM, guaranteeing zero-flakiness stage demos.
    - **Safety Circuit Breaker**: ``--max-steps`` hard cap prevents
      runaway episodes.

CLI Flags:
    --url          Target URL to test (required)
    --goal         Natural-language test objective (required)
    --demo-mode    Replay cached run; bypass LLM inference
    --output       Output directory for runs/reports (default: ./runs)
    --headless     Run browser headlessly
    --max-steps    Maximum allowed steps (default: 50)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from langgraph.graph import END, StateGraph

from agent import decide_next_step
from driver import BlackBoxDriver
from evaluator import FrictionReport, calculate_friction_score
from perception import capture_and_annotate
from report import generate_html_report
from schema import AgentAction, AgentState, ScreenFriction

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_CONSECUTIVE_SAME_STATE = 3  # Trigger loop recovery after this many
_DEFAULT_MAX_STEPS = 50


# ═══════════════════════════════════════════════════════════════════════════
# LangGraph Node Functions
# ═══════════════════════════════════════════════════════════════════════════
# Each node accepts the full AgentState dict and returns a partial state
# update dict.  LangGraph merges the update into the running state.
# ═══════════════════════════════════════════════════════════════════════════

async def perceive_node(
    state: AgentState,
    *,
    driver: BlackBoxDriver,
) -> Dict[str, Any]:
    """Capture the current viewport and annotate interactive elements.

    This node invokes the perception layer to produce a Set-of-Mark
    annotated screenshot, an accessibility-tree text summary, and the
    mark-ID → coordinate lookup table.

    Args:
        state: Current LangGraph agent state.
        driver: Active ``BlackBoxDriver`` with a loaded page.

    Returns:
        Partial state update with ``current_image``, ``a11y_tree``,
        ``id_to_coords``, and ``state_hashes``.
    """
    logger.info("═══ PERCEIVE NODE (step %d) ═══", state.get("step_number", 1))
    try:
        annotated_bytes, a11y_text, id_to_coords = await capture_and_annotate(
            driver.page
        )

        # Compute MD5 hash of the annotated image for loop detection
        img_hash = hashlib.md5(annotated_bytes).hexdigest()
        state_hashes: List[str] = list(state.get("state_hashes", []))
        state_hashes.append(img_hash)

        return {
            "current_image": annotated_bytes,
            "a11y_tree": a11y_text,
            "id_to_coords": id_to_coords,
            "state_hashes": state_hashes,
        }
    except Exception as exc:
        logger.error("Perception failed: %s", exc)
        raise


async def reason_node(
    state: AgentState,
    *,
    api_key: str,
    demo_mode: bool = False,
    driver: Optional[BlackBoxDriver] = None,
) -> Dict[str, Any]:
    """Determine the next browser action via the VLM or Ghost Mode replay.

    In live mode, calls ``decide_next_step`` with the current perception
    data.  In demo mode, loads the cached artifact for the current step
    number instead.

    Self-healing loop detection is applied here: if the last N screenshots
    have identical MD5 hashes, a warning is injected and a ``go_back``
    action is forced.

    Args:
        state: Current LangGraph agent state.
        api_key: Google API key for Gemini (unused in demo mode).
        demo_mode: If ``True``, read from cached artifacts.
        driver: The ``BlackBoxDriver`` (needed for Ghost Mode artifact
            loading path).

    Returns:
        Partial state update with ``last_action`` and appended
        ``history`` entry.
    """
    step_number: int = state.get("step_number", 1)
    logger.info("═══ REASON NODE (step %d) ═══", step_number)

    history: List[Dict[str, Any]] = list(state.get("history", []))

    # -----------------------------------------------------------------
    # Self-Healing Loop Detection
    # -----------------------------------------------------------------
    state_hashes = state.get("state_hashes", [])
    if _detect_loop(state_hashes, history=history):
        logger.warning(
            "Loop detected! Last %d screenshots have identical hashes. "
            "Forcing go_back.",
            _MAX_CONSECUTIVE_SAME_STATE,
        )
        forced_action = AgentAction(
            step_number=step_number,
            thought=(
                "LOOP DETECTED: The page state has not changed across "
                f"{_MAX_CONSECUTIVE_SAME_STATE} consecutive steps. "
                "Forcing go_back to escape the loop."
            ),
            action_type="go_back",
            target_mark_id=None,
            text_input=None,
            friction_metrics=ScreenFriction(
                has_unexpected_popup=False,
                missing_a11y_label=False,
                friction_notes="Automated loop-recovery triggered.",
            ),
        )
        action_dict = forced_action.model_dump(mode="json")
        history.append(action_dict)
        return {
            "last_action": action_dict,
            "history": history,
        }

    # -----------------------------------------------------------------
    # Ghost Mode: replay from cache
    # -----------------------------------------------------------------
    if demo_mode and driver is not None:
        cached = driver.load_step_artifact(step_number)
        if cached is not None:
            logger.info("Ghost Mode: replaying cached step %d.", step_number)
            history.append(cached)
            return {
                "last_action": cached,
                "history": history,
            }
        else:
            logger.warning(
                "Ghost Mode: no artifact for step %d; falling through "
                "to LLM inference.",
                step_number,
            )

    # -----------------------------------------------------------------
    # Live Mode: VLM inference
    # -----------------------------------------------------------------
    # Build step-history summaries for context
    step_summaries: List[str] = []
    for h in history:
        summary = (
            f"Step {h.get('step_number', '?')}: "
            f"{h.get('action_type', '?')}"
        )
        if h.get("target_mark_id") is not None:
            summary += f" → mark [{h['target_mark_id']}]"
        if h.get("text_input"):
            summary += f" (typed: \"{h['text_input'][:40]}\")"
        step_summaries.append(summary)

    action: AgentAction = decide_next_step(
        goal=state.get("goal", ""),
        annotated_image_bytes=state.get("current_image", b""),
        a11y_text=state.get("a11y_tree", ""),
        step_history=step_summaries,
        api_key=api_key,
    )

    action_dict = action.model_dump(mode="json")
    history.append(action_dict)

    return {
        "last_action": action_dict,
        "history": history,
    }


async def execute_node(
    state: AgentState,
    *,
    driver: BlackBoxDriver,
) -> Dict[str, Any]:
    """Execute the decided action in the browser and record artifacts.

    This node dispatches the ``last_action`` through the
    ``BlackBoxDriver``, captures a post-action screenshot, saves a
    Ghost Mode artifact, and increments the step counter.

    Args:
        state: Current LangGraph agent state.
        driver: Active ``BlackBoxDriver``.

    Returns:
        Partial state update with incremented ``step_number``,
        updated ``metrics_log``, and fresh ``current_image``.
    """
    step_number: int = state.get("step_number", 1)
    logger.info("═══ EXECUTE NODE (step %d) ═══", step_number)

    action_dict = state.get("last_action", {})
    id_to_coords = state.get("id_to_coords", {})

    try:
        # Reconstruct AgentAction for the driver
        action = AgentAction.model_validate(action_dict)

        # Execute in browser
        await driver.execute_action(action, id_to_coords)

        # Capture post-action screenshot
        screenshot = await driver.take_screenshot()

        # Record Ghost Mode artifact
        driver.save_step_artifact(step_number, action, screenshot)

        # Update metrics log
        metrics_log: List[Dict[str, Any]] = list(
            state.get("metrics_log", [])
        )
        friction_dict = action_dict.get("friction_metrics", {})
        metrics_log.append(friction_dict)

        return {
            "step_number": step_number + 1,
            "current_image": screenshot,
            "metrics_log": metrics_log,
        }

    except Exception as exc:
        logger.error("Execution failed at step %d: %s", step_number, exc)
        # On failure, still increment to avoid infinite retry loops
        return {
            "step_number": step_number + 1,
        }


def evaluate_node(state: AgentState) -> Dict[str, Any]:
    """Route the state machine: continue looping or terminate.

    This is a pure routing node—it inspects the ``last_action`` to
    decide whether to loop back to perception or end the episode.

    Args:
        state: Current LangGraph agent state.

    Returns:
        Empty dict (routing is handled by the conditional edge).
    """
    step_number = state.get("step_number", 1)
    last_action = state.get("last_action", {})
    action_type = last_action.get("action_type", "")

    logger.info(
        "═══ EVALUATE NODE (step %d) — last_action=%s ═══",
        step_number,
        action_type,
    )
    return {}


# ═══════════════════════════════════════════════════════════════════════════
# Routing function for conditional edges
# ═══════════════════════════════════════════════════════════════════════════

def _should_continue(
    state: AgentState,
    *,
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> Literal["perceive", "__end__"]:
    """Determine whether to continue the perception loop or end.

    Args:
        state: Current agent state.
        max_steps: Hard cap on episode length.

    Returns:
        ``"perceive"`` to continue the loop, or ``"__end__"`` to
        terminate the episode.
    """
    last_action = state.get("last_action", {})
    action_type = last_action.get("action_type", "")
    step_number = state.get("step_number", 1)

    if action_type == "done":
        logger.info("Agent signalled DONE at step %d.", step_number - 1)
        return "__end__"

    if step_number > max_steps:
        logger.warning(
            "Maximum step limit (%d) reached. Terminating.", max_steps
        )
        return "__end__"

    return "perceive"


# ═══════════════════════════════════════════════════════════════════════════
# Loop Detection Helper
# ═══════════════════════════════════════════════════════════════════════════

def _detect_loop(
    state_hashes: List[str],
    history: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Check for stuck-state patterns: identical hashes, oscillation, or
    non-productive action repetition.

    Detects three patterns:
        1. **Identical hashes**: Last N screenshots have the same MD5.
        2. **Oscillation**: Last 2N hashes alternate between exactly two
           values (A-B-A-B), indicating the agent is bouncing between
           two states without progress.
        3. **Non-productive actions**: Last N actions are all scrolls
           (``scroll_up`` / ``scroll_down``) with no clicks, typing, or
           navigation — indicating aimless scrolling.

    Args:
        state_hashes: List of MD5 hex digests of screenshots.
        history: Optional list of action dicts from the agent state.
            When provided, enables action-repetition detection.

    Returns:
        ``True`` if any loop pattern is detected.
    """
    n = _MAX_CONSECUTIVE_SAME_STATE

    # Pattern 1: Identical consecutive screenshots
    if len(state_hashes) >= n:
        recent = state_hashes[-n:]
        if len(set(recent)) == 1:
            logger.warning("Loop pattern 1: %d identical screenshots.", n)
            return True

    # Pattern 2: Oscillation (A-B-A-B) over 2N screenshots
    oscillation_window = n * 2
    if len(state_hashes) >= oscillation_window:
        recent = state_hashes[-oscillation_window:]
        unique = set(recent)
        if len(unique) == 2:
            # Check if it's truly alternating (even indices same, odd indices same)
            evens = set(recent[::2])
            odds = set(recent[1::2])
            if len(evens) == 1 and len(odds) == 1 and evens != odds:
                logger.warning(
                    "Loop pattern 2: oscillation between 2 states over %d steps.",
                    oscillation_window,
                )
                return True

    # Pattern 3: Non-productive action repetition (scroll-only)
    _SCROLL_ACTIONS = {"scroll_up", "scroll_down"}
    if history and len(history) >= n + 1:  # n+1 because scrolls need to be sustained
        recent_actions = [
            h.get("action_type", "") for h in history[-(n + 1):]
        ]
        if all(a in _SCROLL_ACTIONS for a in recent_actions):
            logger.warning(
                "Loop pattern 3: %d consecutive scroll-only actions.",
                n + 1,
            )
            return True

    return False


# ═══════════════════════════════════════════════════════════════════════════
# Graph Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_graph(
    driver: BlackBoxDriver,
    api_key: str,
    demo_mode: bool = False,
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> StateGraph:
    """Construct the LangGraph state machine for the agent loop.

    Wires four nodes—perceive, reason, execute, evaluate—into a cyclic
    graph with a conditional edge from evaluate back to perceive (or END).

    Args:
        driver: Active ``BlackBoxDriver`` instance.
        api_key: Google API key for Gemini.
        demo_mode: Enable Ghost Mode replay.
        max_steps: Safety circuit breaker.

    Returns:
        A compiled LangGraph ``StateGraph`` ready to be invoked.
    """
    graph = StateGraph(AgentState)

    # --- Register nodes with bound dependencies --------------------------
    async def _perceive(state: AgentState) -> Dict[str, Any]:
        return await perceive_node(state, driver=driver)

    async def _reason(state: AgentState) -> Dict[str, Any]:
        return await reason_node(
            state, api_key=api_key, demo_mode=demo_mode, driver=driver
        )

    async def _execute(state: AgentState) -> Dict[str, Any]:
        return await execute_node(state, driver=driver)

    def _evaluate(state: AgentState) -> Dict[str, Any]:
        return evaluate_node(state)

    graph.add_node("perceive", _perceive)
    graph.add_node("reason", _reason)
    graph.add_node("execute", _execute)
    graph.add_node("evaluate", _evaluate)

    # --- Wire edges -------------------------------------------------------
    graph.set_entry_point("perceive")
    graph.add_edge("perceive", "reason")
    graph.add_edge("reason", "execute")
    graph.add_edge("execute", "evaluate")

    # Conditional routing from evaluate
    def _routing(state: AgentState) -> Literal["perceive", "__end__"]:
        return _should_continue(state, max_steps=max_steps)

    graph.add_conditional_edges(
        "evaluate",
        _routing,
        {
            "perceive": "perceive",
            "__end__": END,
        },
    )

    return graph.compile()


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the P8 framework.

    Args:
        argv: Optional list of argument strings (defaults to
            ``sys.argv[1:]``).

    Returns:
        Parsed ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(
        prog="p8-agent",
        description=(
            "Autonomous Agentic Black-Box UI/UX & Accessibility "
            "Testing Framework (Project P8)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --url https://example.com "
            '--goal "Find the contact page"\n'
            "  python main.py --url https://example.com "
            '--goal "Add item to cart" --headless\n'
            "  python main.py --demo-mode "
            "--output ./runs/20260918_150000\n"
        ),
    )

    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Target URL to test (required unless --demo-mode).",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Natural-language test objective (required unless --demo-mode).",
    )
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        default=False,
        help=(
            "Ghost Mode: replay a previous run from cached artifacts. "
            "Bypasses LLM inference for zero-flakiness demos."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./runs",
        help="Base directory for run artifacts and reports (default: ./runs).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run the browser in headless mode.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=_DEFAULT_MAX_STEPS,
        help=f"Maximum steps before forced termination (default: {_DEFAULT_MAX_STEPS}).",
    )

    args = parser.parse_args(argv)

    # Validation
    if not args.demo_mode:
        if not args.url:
            parser.error("--url is required unless --demo-mode is set.")
        if not args.goal:
            parser.error("--goal is required unless --demo-mode is set.")

    return args


async def run_agent(args: argparse.Namespace) -> None:
    """Execute the full autonomous testing pipeline.

    Orchestrates browser startup, LangGraph execution, friction scoring,
    and HTML report generation.

    Args:
        args: Parsed CLI arguments.
    """
    # -----------------------------------------------------------------
    # Resolve API key
    # -----------------------------------------------------------------
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key and not args.demo_mode:
        logger.error(
            "GOOGLE_API_KEY environment variable is not set. "
            "Set it or use --demo-mode."
        )
        sys.exit(1)

    # -----------------------------------------------------------------
    # Build run directory
    # -----------------------------------------------------------------
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir)

    # -----------------------------------------------------------------
    # Launch driver & graph
    # -----------------------------------------------------------------
    async with BlackBoxDriver(
        headless=args.headless,
        demo_mode=args.demo_mode,
        run_dir=str(run_dir),
    ) as driver:

        # Navigate to target URL
        if args.url:
            await driver.navigate(args.url)

        # Build and compile the LangGraph state machine
        compiled_graph = build_graph(
            driver=driver,
            api_key=api_key,
            demo_mode=args.demo_mode,
            max_steps=args.max_steps,
        )

        # Initial state
        initial_state: AgentState = {
            "url": args.url or "",
            "goal": args.goal or "",
            "step_number": 1,
            "history": [],
            "current_image": b"",
            "a11y_tree": "",
            "id_to_coords": {},
            "last_action": None,
            "metrics_log": [],
            "state_hashes": [],
        }

        logger.info(
            "Starting agent loop: url=%s, goal='%s', max_steps=%d",
            args.url, args.goal, args.max_steps,
        )

        # -----------------------------------------------------------------
        # Run the state machine
        # -----------------------------------------------------------------
        final_state: Dict[str, Any] = {}
        try:
            async for state_update in compiled_graph.astream(
                initial_state,
                config={"recursion_limit": args.max_steps * 4 + 10},
            ):
                # state_update is {node_name: partial_state_dict}
                for node_name, partial in state_update.items():
                    if isinstance(partial, dict):
                        final_state.update(partial)
                    logger.debug(
                        "Node '%s' completed. Keys updated: %s",
                        node_name,
                        list(partial.keys()) if isinstance(partial, dict) else "N/A",
                    )
        except Exception as exc:
            logger.error("Agent loop terminated with error: %s", exc)

    # -----------------------------------------------------------------
    # Post-run: Friction scoring & HTML report
    # -----------------------------------------------------------------
    history = final_state.get("history", [])
    metrics_log = final_state.get("metrics_log", [])

    logger.info("Agent completed: %d steps recorded.", len(history))

    # Calculate friction score
    friction_report: FrictionReport = calculate_friction_score(
        steps=history,
        metrics_log=metrics_log,
    )
    logger.info("Friction Score: %s", friction_report.summary)

    # Generate HTML report
    html_content = generate_html_report(
        run_id=run_id,
        task_goal=args.goal or "(demo replay)",
        steps_list=history,
        friction_report=friction_report,
        screenshot_dir=run_dir,
    )

    report_path = run_dir / "report.html"
    report_path.write_text(html_content, encoding="utf-8")
    logger.info("HTML report saved to: %s", report_path)

    # Save friction report as JSON
    friction_json_path = run_dir / "friction_report.json"
    friction_json_path.write_text(
        json.dumps(
            {
                "overall_score": friction_report.overall_score,
                "action_bloat_score": friction_report.action_bloat_score,
                "popup_score": friction_report.popup_score,
                "backtrack_score": friction_report.backtrack_score,
                "a11y_score": friction_report.a11y_score,
                "total_steps": friction_report.total_steps,
                "popup_count": friction_report.popup_count,
                "backtrack_count": friction_report.backtrack_count,
                "missing_label_count": friction_report.missing_label_count,
                "summary": friction_report.summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Friction report JSON saved to: %s", friction_json_path)

    # Final console output (ASCII-safe for Windows cp1252 terminals)
    print("\n" + "=" * 60)
    print("  AUTONOMOUS UX AUDIT COMPLETE")
    print("=" * 60)
    print(f"  Run ID:          {run_id}")
    print(f"  Steps:           {friction_report.total_steps}")
    print(f"  Friction Score:  {friction_report.overall_score}/100")
    try:
        verdict = friction_report.summary.split("(")[1].split(")")[0]
    except (IndexError, AttributeError):
        verdict = "N/A"
    print(f"  Verdict:         {verdict}")
    print(f"  Report:          {report_path.resolve()}")
    print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """CLI entry point.  Configures logging and launches the async agent."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    asyncio.run(run_agent(args))


if __name__ == "__main__":
    main()
