"""Core multimodal decision loop for the Autonomous UI/UX Testing Agent.

This module encapsulates the single-responsibility ``decide_next_step``
function that bridges the perception layer (annotated screenshots +
accessibility tree) with the action layer (browser driver commands)
via the Google GenAI SDK (``google-genai``) and Gemini 2.0 Flash.

Design Principles:
    1. **Defensive Programming** – Every external call (client init,
       content generation, response parsing) is wrapped in its own
       ``try/except`` with structured ``logging`` output.
    2. **Determinism** – ``temperature=0.0`` plus Gemini's native
       ``response_schema`` mapped to the ``AgentAction`` Pydantic model
       eliminates output drift across identical inputs.
    3. **Modular Design** – This module has zero knowledge of Playwright,
       Selenium, or any browser driver.  It accepts raw bytes and text,
       and returns a validated ``AgentAction``.
"""

from __future__ import annotations

import json
import logging
import time
from typing import List

from google import genai
from google.genai import types
from pydantic import ValidationError

from schema import AgentAction

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL_ID = "gemini-3.6-flash"

_SYSTEM_INSTRUCTION = """\
You are an autonomous QA and accessibility testing agent operating inside a
real web browser.  You receive:

1. A **Set-of-Mark annotated screenshot** (PNG) where every interactive
   element is overlaid with a unique numeric mark ID.
2. A **serialised accessibility tree** describing each element's role,
   name, state, and mark ID.
3. The **test goal** you must accomplish.
4. A **step history** of actions you have already taken.

────────────────────────────────────────────
DECISION RULES
────────────────────────────────────────────
• Identify the element that best advances the goal by correlating the
  screenshot marks with the accessibility tree entries.
• If an **unexpected modal, cookie banner, or overlay** blocks the target
  element, dismiss it first (click its close/dismiss control or press
  Escape) and note the friction in ``friction_metrics``.
• If an interactive element is **missing an accessible label** (no
  ``aria-label``, ``aria-labelledby``, or associated ``<label>``),
  set ``friction_metrics.missing_a11y_label = true`` and explain in
  ``friction_notes``.
• **Avoid infinite loops**: if the last 3+ actions targeted the same
  mark ID or the page state appears unchanged, choose ``go_back``,
  ``scroll_up``, ``scroll_down``, or ``done`` with an explanation.
• **Scroll oscillation ban**: if the step history shows 2 or more
  consecutive scroll actions (``scroll_up`` / ``scroll_down``) without
  any click or type in between, you MUST NOT issue another scroll.
  Instead, try one of these recovery strategies:
  1. Click a visible interactive element that may advance the goal.
  2. Use ``go_back`` to return to a previous page and try a different
     navigation path.
  3. Emit ``action_type = "done"`` with a ``thought`` explaining that
     the target element could not be located after scrolling.
• When the goal is fully achieved, emit ``action_type = "done"``.

────────────────────────────────────────────
OUTPUT CONTRACT
────────────────────────────────────────────
Return a single JSON object conforming to the ``AgentAction`` schema.
Every field is mandatory except ``target_mark_id`` and ``text_input``
(which are conditional on the action type).
"""


def decide_next_step(
    goal: str,
    annotated_image_bytes: bytes,
    a11y_text: str,
    step_history: list[str],
    api_key: str,
) -> AgentAction:
    """Determine the next browser action for the autonomous testing loop.

    This function is **side-effect-free** with respect to the browser – it
    only reads perception data and returns a validated action contract.

    Args:
        goal: Natural-language description of the test objective
            (e.g. ``"Add an item to the cart and proceed to checkout"``).
        annotated_image_bytes: Raw PNG bytes of the Set-of-Mark annotated
            browser viewport screenshot.
        a11y_text: Serialised accessibility-tree snapshot of the current
            page (element roles, names, states, and mark IDs).
        step_history: Chronological list of human-readable summaries of
            previous steps.  Forwarded as context so the LLM can avoid
            repetition and detect loops.
        api_key: Google API key authorised for the Gemini API.

    Returns:
        A Pydantic-validated :class:`AgentAction` instance ready for the
        browser-driver layer to execute.

    Raises:
        RuntimeError: If the GenAI client cannot be initialised or the
            model call fails after exhausting retries.
        pydantic.ValidationError: If the model response does not conform
            to the ``AgentAction`` schema.
    """

    # -----------------------------------------------------------------
    # 1. Initialise the GenAI client
    # -----------------------------------------------------------------
    client = _init_client(api_key)

    # -----------------------------------------------------------------
    # 2. Assemble the multimodal content parts
    # -----------------------------------------------------------------
    contents = _build_contents(goal, annotated_image_bytes, a11y_text, step_history)

    # -----------------------------------------------------------------
    # 3. Call Gemini with deterministic settings + structured output
    # -----------------------------------------------------------------
    raw_json = _call_model(client, contents)

    # -----------------------------------------------------------------
    # 4. Parse and validate the response into AgentAction
    # -----------------------------------------------------------------
    action = _parse_response(raw_json)

    logger.info(
        "Step %d decided: action=%s, target=%s | thought=%s",
        action.step_number,
        action.action_type,
        action.target_mark_id,
        action.thought[:120],
    )
    return action


# ═══════════════════════════════════════════════════════════════════════════
# Private helpers (each wraps exactly one concern)
# ═══════════════════════════════════════════════════════════════════════════

def _init_client(api_key: str) -> genai.Client:
    """Create and return an authenticated ``genai.Client``.

    Raises:
        RuntimeError: On any failure during client construction so the
            caller receives a uniform exception type.
    """
    try:
        client = genai.Client(api_key=api_key)
        logger.info("genai.Client initialised successfully.")
        return client
    except Exception as exc:
        logger.error("Failed to initialise genai.Client: %s", exc)
        raise RuntimeError(
            f"GenAI client initialisation failed: {exc}"
        ) from exc


def _build_contents(
    goal: str,
    image_bytes: bytes,
    a11y_text: str,
    step_history: list[str],
) -> list:
    """Assemble the multimodal ``contents`` list for ``generate_content``.

    The SDK expects a flat list where strings and ``types.Part`` objects
    can be intermixed.  Image data is inlined via
    ``types.Part.from_bytes``.
    """
    try:
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png",
        )

        history_block = "\n".join(
            f"  {i}. {entry}" for i, entry in enumerate(step_history, 1)
        ) if step_history else "  (no prior steps)"

        text_context = (
            f"## Test Goal\n{goal}\n\n"
            f"## Accessibility Tree\n{a11y_text}\n\n"
            f"## Step History\n{history_block}"
        )

        contents = [text_context, image_part]
        logger.info(
            "Content payload assembled: %d text chars, image %d bytes.",
            len(text_context),
            len(image_bytes),
        )
        return contents

    except Exception as exc:
        logger.error("Failed to assemble content payload: %s", exc)
        raise RuntimeError(
            f"Content assembly failed: {exc}"
        ) from exc


def _call_model(
    client: genai.Client,
    contents: list,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> str:
    """Invoke Gemini and return the raw JSON string from the response.

    Includes exponential-backoff retry logic to handle transient errors
    (HTTP 503 Service Unavailable, 429 Rate Limit, network timeouts).

    Determinism knobs:
        - ``temperature = 0.0`` (greedy decoding)
        - ``response_schema = AgentAction`` (structured output)
        - ``response_mime_type = "application/json"``

    Args:
        client: Authenticated ``genai.Client``.
        contents: Multimodal content parts list.
        max_retries: Maximum number of retry attempts (default: 3).
        base_delay: Base delay in seconds for exponential backoff
            (default: 2.0 → delays of 2s, 4s, 8s).

    Raises:
        RuntimeError: On any transport, quota, or generation error
            after exhausting all retries.
    """
    config = types.GenerateContentConfig(
        system_instruction=_SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=AgentAction,
        temperature=0.0,
    )

    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=_MODEL_ID,
                contents=contents,
                config=config,
            )

            raw_json: str = response.text
            if not raw_json or not raw_json.strip():
                raise ValueError("Gemini returned an empty response body.")

            logger.info("Gemini response received (%d chars).", len(raw_json))
            logger.debug("Raw JSON payload:\n%s", raw_json)
            return raw_json.strip()

        except Exception as exc:
            last_exc = exc
            exc_str = str(exc)
            is_retryable = any(
                code in exc_str for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "timeout")
            )

            if is_retryable and attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Gemini call failed (attempt %d/%d): %s. "
                    "Retrying in %.1fs...",
                    attempt, max_retries, exc, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "Gemini call failed (attempt %d/%d, final): %s",
                    attempt, max_retries, exc,
                )
                raise RuntimeError(
                    f"Gemini model call failed after {attempt} attempt(s): {exc}"
                ) from exc

    # Should never reach here, but satisfy type checker
    raise RuntimeError(
        f"Gemini model call failed after {max_retries} retries: {last_exc}"
    )


def _parse_response(raw_json: str) -> AgentAction:
    """Deserialise and validate the JSON string into an ``AgentAction``.

    Uses Pydantic v2's ``model_validate_json`` for single-pass parsing
    and validation.

    Raises:
        pydantic.ValidationError: When the payload violates the schema.
        RuntimeError: On unexpected deserialisation errors.
    """
    try:
        action = AgentAction.model_validate_json(raw_json)
        logger.info("AgentAction validated successfully.")
        return action
    except ValidationError as ve:
        logger.error(
            "AgentAction schema validation failed:\n%s",
            ve.json(indent=2),
        )
        raise
    except json.JSONDecodeError as je:
        logger.error("Response is not valid JSON: %s", je)
        raise RuntimeError(
            f"Gemini returned malformed JSON: {je}"
        ) from je
    except Exception as exc:
        logger.error("Unexpected error during response parsing: %s", exc)
        raise RuntimeError(
            f"Response parsing failed: {exc}"
        ) from exc
