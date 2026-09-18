"""Schema definitions for the Autonomous Agentic UI/UX Testing Framework.

This module defines the canonical data contracts that flow through the
agent's perception–reasoning–action loop.  Every model is designed for
deterministic serialisation to / from Gemini's structured-output JSON,
clean episodic-memory snapshots, and zero-ambiguity type checking at
both static-analysis and runtime boundaries.

Modules:
    ScreenFriction: Usability-telemetry signal collected per action step.
    AgentAction:    Strict contract for the JSON payload returned by Gemini.
    AgentState:     LangGraph TypedDict carrying the full execution state.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, conint, constr
from typing_extensions import TypedDict


class ScreenFriction(BaseModel):
    """Captures usability-friction signals observed on a single screen state.

    Each autonomous step records one ``ScreenFriction`` instance to build a
    quantitative friction heatmap over the user journey under test.

    Attributes:
        has_unexpected_popup: ``True`` when a modal, overlay, cookie banner,
            or any DOM element not predicted by the goal-plan appears and
            obstructs the critical path.
        missing_a11y_label: ``True`` when the target interactive element
            (button, link, input) lacks an accessible name per WCAG 2.2
            Success Criterion 4.1.2 (Name, Role, Value).
        friction_notes: Free-text annotation the agent attaches to explain
            *why* friction was flagged.  An empty string signals that no
            friction-worthy observation was made.

    Example:
        >>> friction = ScreenFriction(
        ...     has_unexpected_popup=True,
        ...     missing_a11y_label=False,
        ...     friction_notes="Cookie-consent banner blocked CTA button.",
        ... )
    """

    model_config = {"frozen": True}

    has_unexpected_popup: bool = Field(
        ...,
        description=(
            "Whether an unexpected popup, modal, overlay, or interstitial "
            "was detected on screen that was not anticipated by the test plan."
        ),
    )
    missing_a11y_label: bool = Field(
        ...,
        description=(
            "Whether the primary interaction target is missing an accessible "
            "label (aria-label, aria-labelledby, or visible label association) "
            "per WCAG 2.2 SC 4.1.2."
        ),
    )
    friction_notes: str = Field(
        default="",
        description=(
            "Human-readable annotation describing the observed friction. "
            "Empty string when no friction is observed."
        ),
    )


ACTION_TYPES = Literal[
    "click",
    "type",
    "scroll_up",
    "scroll_down",
    "wait",
    "go_back",
    "goto_url",
    "done",
]


class AgentAction(BaseModel):
    """Enforces the exact JSON schema that Gemini must return per step.

    This model acts as the single source of truth for the
    perception → reasoning → action contract.  Any deviation from the
    expected payload structure is caught at Pydantic-validation time,
    preventing malformed actions from reaching the browser driver.

    Attributes:
        step_number: Monotonically increasing 1-based index of the current
            step within the episode.  Used for ordered replay and debugging.
        thought: The agent's chain-of-thought reasoning that led to this
            action.  Stored verbatim for post-hoc explainability audits.
        action_type: One of the canonical action verbs the browser driver
            can execute.
        target_mark_id: The numeric Set-of-Mark identifier of the DOM
            element to interact with.  Required for ``click`` and ``type``;
            ``None`` for actions that do not target a specific element
            (e.g., ``scroll_up``, ``wait``, ``done``).
        text_input: The string to type into the targeted element.  Required
            when ``action_type`` is ``type``; ``None`` otherwise.
        friction_metrics: Per-step usability telemetry captured alongside
            the action.

    Example:
        >>> action = AgentAction(
        ...     step_number=3,
        ...     thought="The search box is visible at mark 12. Typing query.",
        ...     action_type="type",
        ...     target_mark_id=12,
        ...     text_input="autonomous testing",
        ...     friction_metrics=ScreenFriction(
        ...         has_unexpected_popup=False,
        ...         missing_a11y_label=True,
        ...         friction_notes="Search input has no aria-label.",
        ...     ),
        ... )
    """

    model_config = {"frozen": True}

    step_number: int = Field(
        ...,
        ge=1,
        description=(
            "1-based, monotonically increasing step index within the "
            "current agent episode."
        ),
    )
    thought: str = Field(
        ...,
        min_length=1,
        description=(
            "Chain-of-thought reasoning that justifies the chosen action. "
            "Must be non-empty for auditability."
        ),
    )
    action_type: ACTION_TYPES = Field(
        ...,
        description=(
            "The canonical browser-driver verb to execute. One of: "
            "click, type, scroll_up, scroll_down, wait, go_back, "
            "goto_url, done."
        ),
    )
    target_mark_id: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Set-of-Mark numeric identifier of the target DOM element. "
            "Required for 'click' and 'type' actions; None otherwise."
        ),
    )
    text_input: Optional[str] = Field(
        default=None,
        description=(
            "Text payload to type into the target element. "
            "Required when action_type is 'type'; None otherwise."
        ),
    )
    friction_metrics: ScreenFriction = Field(
        ...,
        description=(
            "Usability-friction telemetry captured for this step."
        ),
    )


# ---------------------------------------------------------------------------
# 3. AgentState – LangGraph execution-state TypedDict
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """LangGraph-compatible execution state for the autonomous testing agent.

    This TypedDict defines the canonical shape of the state dictionary
    that flows between LangGraph nodes.  Using ``total=False`` allows
    nodes to perform partial updates without supplying every key on
    every transition.

    Attributes:
        url: The URL currently loaded in the controlled browser instance.
        goal: Natural-language description of the test objective the agent
            is pursuing (e.g., "Add an item to the cart and proceed to
            checkout").
        step_number: Current 1-based step index within the episode.
            Incremented by the action-execution node after each successful
            browser interaction.
        history: Ordered list of ``AgentAction`` dicts representing every
            action the agent has taken so far in this episode.  Appended
            to (never mutated in place) to preserve episodic memory.
        current_image: Raw screenshot bytes (PNG) of the browser viewport
            after the latest navigation or action.  Passed to the vision
            model for perception.
        a11y_tree: Serialised accessibility-tree snapshot of the current
            page, used for element identification and WCAG analysis.
        id_to_coords: Mapping from Set-of-Mark integer IDs to their
            ``(x, y)`` viewport coordinates, enabling the browser driver
            to translate mark IDs into click targets.
        last_action: The most recent ``AgentAction`` dict, surfaced as a
            convenience key so downstream nodes can inspect it without
            indexing into ``history``.
        metrics_log: Accumulated list of ``ScreenFriction`` dicts—one per
            step—used for post-episode friction reporting.
        state_hashes: Set of perceptual hashes (e.g., pHash of screenshots)
            used to detect navigation loops.  If a hash repeats, the
            orchestrator can trigger a recovery strategy.
    """

    url: str
    goal: str
    step_number: int
    history: List[Dict[str, Any]]
    current_image: bytes
    a11y_tree: str
    id_to_coords: Dict[int, Tuple[int, int]]
    last_action: Optional[Dict[str, Any]]
    metrics_log: List[Dict[str, Any]]
    state_hashes: List[str]
