"""Playwright Browser Driver & Ghost Mode Caching Engine.

This module provides ``BlackBoxDriver``, the **hands** of the autonomous
testing agent.  It translates high-level ``AgentAction`` commands into
precise Playwright browser interactions using the Set-of-Mark coordinate
lookup table produced by the perception layer.

Key Capabilities:
    - Manages the full Playwright lifecycle (browser → context → page).
    - Dispatches clicks, typing, scrolling, navigation via coordinate-based
      mouse/keyboard primitives—completely decoupled from CSS selectors.
    - **Ghost Mode Engine**: during live runs, every step's action payload
      and screenshot are serialised to ``./runs/{run_id}/`` as JSON
      artifacts.  In ``--demo-mode``, these artifacts are replayed
      sequentially, **bypassing LLM inference entirely**, to guarantee
      deterministic, zero-flakiness stage demonstrations.

Design Principles:
    - Every Playwright call is wrapped in ``try/except`` with structured
      ``logging`` output and clean fallback semantics.
    - The driver has *zero* knowledge of the VLM, perception, or scoring
      layers—it accepts an ``AgentAction`` and a coordinate map, period.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

from schema import AgentAction

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_VIEWPORT = {"width": 1280, "height": 720}
_NAV_TIMEOUT_MS = 30_000
_ACTION_SETTLE_MS = 800
_SCROLL_DELTA = 500


class BlackBoxDriver:
    """Manages Playwright browser lifecycle and action execution.

    This driver translates ``AgentAction`` decisions into real browser
    interactions.  It operates exclusively through coordinate-based mouse
    clicks and keyboard input—never CSS selectors—making it a true
    black-box automation layer.

    Attributes:
        headless: Whether to run the browser in headless mode.
        demo_mode: If ``True``, replay cached artifacts instead of
            performing live browser interactions.
        run_dir: Filesystem path where step artifacts are stored.
        page: The active Playwright ``Page`` instance.

    Example:
        >>> async with BlackBoxDriver(headless=True, run_dir="./runs/run_001") as driver:
        ...     await driver.navigate("https://example.com")
        ...     await driver.execute_action(action, id_to_coords)
    """

    def __init__(
        self,
        headless: bool = True,
        demo_mode: bool = False,
        run_dir: str = "./runs/default",
    ) -> None:
        """Initialise the driver configuration.

        Args:
            headless: Run Chromium in headless mode if ``True``.
            demo_mode: Enable Ghost Mode artifact replay if ``True``.
            run_dir: Directory path for reading/writing step artifacts.
        """
        self.headless: bool = headless
        self.demo_mode: bool = demo_mode
        self.run_dir: Path = Path(run_dir)

        # Playwright objects (populated in ``start``)
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

        # Ghost Mode replay index
        self._replay_index: int = 0

    # ------------------------------------------------------------------
    # Async context-manager protocol
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "BlackBoxDriver":
        """Start the browser when entering the async context."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Tear down the browser when exiting the async context."""
        await self.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch Chromium and create a browser context and page.

        Creates the ``run_dir`` on disk if it does not exist (for artifact
        storage during live runs).

        Raises:
            RuntimeError: If Playwright or browser launch fails.
        """
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
            )
            self._context = await self._browser.new_context(
                viewport=_DEFAULT_VIEWPORT,
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            self.page = await self._context.new_page()
            logger.info(
                "BlackBoxDriver started: headless=%s, demo_mode=%s, "
                "run_dir=%s",
                self.headless,
                self.demo_mode,
                self.run_dir,
            )
        except Exception as exc:
            logger.error("Failed to start BlackBoxDriver: %s", exc)
            raise RuntimeError(
                f"Playwright browser launch failed: {exc}"
            ) from exc

    async def close(self) -> None:
        """Gracefully tear down the browser, context, and Playwright.

        Safe to call multiple times; silently ignores already-closed
        resources.
        """
        for resource_name, resource in [
            ("page", self.page),
            ("context", self._context),
            ("browser", self._browser),
        ]:
            if resource is not None:
                try:
                    await resource.close()
                    logger.debug("Closed %s.", resource_name)
                except Exception as exc:
                    logger.warning(
                        "Error closing %s: %s", resource_name, exc
                    )

        if self._playwright is not None:
            try:
                await self._playwright.stop()
                logger.debug("Playwright stopped.")
            except Exception as exc:
                logger.warning("Error stopping Playwright: %s", exc)

        self.page = None
        self._context = None
        self._browser = None
        self._playwright = None
        logger.info("BlackBoxDriver closed.")

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    async def navigate(self, url: str) -> None:
        """Navigate to the given URL and wait for DOM stability.

        Waits for ``networkidle`` (no outstanding network requests for
        500 ms) with a configurable timeout, then pauses briefly to let
        late JavaScript settle.

        Args:
            url: Fully qualified URL to navigate to.

        Raises:
            RuntimeError: If navigation or load waiting fails.
        """
        if self.page is None:
            raise RuntimeError("Driver not started. Call start() first.")
        try:
            logger.info("Navigating to %s", url)
            await self.page.goto(url, wait_until="networkidle", timeout=_NAV_TIMEOUT_MS)
            await self.page.wait_for_timeout(_ACTION_SETTLE_MS)
            logger.info("Navigation complete, DOM stable.")
        except Exception as exc:
            logger.error("Navigation to %s failed: %s", url, exc)
            raise RuntimeError(
                f"Navigation failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    async def execute_action(
        self,
        action: AgentAction,
        id_to_coords: Dict[int, Dict[str, int]],
    ) -> None:
        """Dispatch a browser action based on the ``AgentAction`` payload.

        Translates the high-level action type and mark-ID into precise
        Playwright mouse/keyboard primitives.

        Args:
            action: Validated ``AgentAction`` from the reasoning engine.
            id_to_coords: Mark-ID → ``{"x": int, "y": int}`` lookup
                table from the perception layer.

        Raises:
            RuntimeError: If the action cannot be executed (e.g., mark
                ID not found, page context destroyed).
        """
        if self.page is None:
            raise RuntimeError("Driver not started. Call start() first.")

        action_type = action.action_type
        mark_id = action.target_mark_id
        text_input = action.text_input

        logger.info(
            "Executing action: type=%s, mark=%s, text=%s",
            action_type,
            mark_id,
            repr(text_input)[:60] if text_input else None,
        )

        try:
            if action_type == "click":
                coords = self._resolve_coords(mark_id, id_to_coords)
                await self.page.mouse.click(coords["x"], coords["y"])
                logger.info(
                    "Clicked at (%d, %d) [mark %d].",
                    coords["x"], coords["y"], mark_id,
                )

            elif action_type == "type":
                coords = self._resolve_coords(mark_id, id_to_coords)
                await self.page.mouse.click(coords["x"], coords["y"])
                # Triple-click to select all existing text, then overwrite
                await self.page.mouse.click(
                    coords["x"], coords["y"], click_count=3
                )
                await self.page.keyboard.type(
                    text_input or "", delay=30
                )
                logger.info(
                    "Typed %d chars into mark %d.",
                    len(text_input or ""), mark_id,
                )

            elif action_type == "scroll_down":
                await self.page.mouse.wheel(0, _SCROLL_DELTA)
                logger.info("Scrolled down by %d px.", _SCROLL_DELTA)

            elif action_type == "scroll_up":
                await self.page.mouse.wheel(0, -_SCROLL_DELTA)
                logger.info("Scrolled up by %d px.", _SCROLL_DELTA)

            elif action_type == "go_back":
                await self.page.go_back(
                    wait_until="networkidle", timeout=_NAV_TIMEOUT_MS
                )
                logger.info("Navigated back.")

            elif action_type == "goto_url":
                if text_input:
                    await self.navigate(text_input)
                else:
                    logger.warning(
                        "goto_url action with no text_input; skipping."
                    )

            elif action_type == "wait":
                await self.page.wait_for_timeout(2000)
                logger.info("Waited 2 s.")

            elif action_type == "done":
                logger.info("Action 'done' — no browser interaction.")

            else:
                logger.warning(
                    "Unknown action type '%s'; skipping.", action_type
                )

            # Post-action settle time
            if action_type not in ("done", "wait"):
                await self.page.wait_for_timeout(_ACTION_SETTLE_MS)

        except Exception as exc:
            logger.error(
                "Action execution failed (type=%s, mark=%s): %s",
                action_type, mark_id, exc,
            )
            raise RuntimeError(
                f"Action execution failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------

    async def take_screenshot(self) -> bytes:
        """Capture a PNG screenshot of the current viewport.

        Returns:
            Raw PNG bytes.

        Raises:
            RuntimeError: If screenshot capture fails.
        """
        if self.page is None:
            raise RuntimeError("Driver not started. Call start() first.")
        try:
            png_bytes = await self.page.screenshot(type="png")
            logger.debug("Screenshot: %d bytes.", len(png_bytes))
            return png_bytes
        except Exception as exc:
            logger.error("Screenshot failed: %s", exc)
            raise RuntimeError(
                f"Screenshot capture failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Ghost Mode: artifact recording & playback
    # ------------------------------------------------------------------

    def save_step_artifact(
        self,
        step_number: int,
        action: AgentAction,
        screenshot_bytes: bytes,
    ) -> Path:
        """Persist a step's action payload and screenshot to disk.

        Writes two files under ``run_dir``:
            - ``step_{N:03d}.json``: JSON-serialised ``AgentAction``.
            - ``step_{N:03d}.png``: Post-action viewport screenshot.

        Args:
            step_number: 1-based step index.
            action: The ``AgentAction`` executed this step.
            screenshot_bytes: Post-action screenshot PNG bytes.

        Returns:
            Path to the JSON artifact file.
        """
        step_prefix = f"step_{step_number:03d}"
        json_path = self.run_dir / f"{step_prefix}.json"
        png_path = self.run_dir / f"{step_prefix}.png"

        try:
            # Serialise action to JSON
            action_dict = action.model_dump(mode="json")
            json_path.write_text(
                json.dumps(action_dict, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Write screenshot
            png_path.write_bytes(screenshot_bytes)

            logger.info(
                "Ghost Mode artifact saved: %s (+screenshot).", json_path
            )
            return json_path

        except Exception as exc:
            logger.error(
                "Failed to save step %d artifact: %s", step_number, exc
            )
            raise RuntimeError(
                f"Artifact save failed: {exc}"
            ) from exc

    def load_step_artifact(
        self,
        step_number: int,
    ) -> Optional[Dict[str, Any]]:
        """Load a cached step artifact from disk for Ghost Mode replay.

        Args:
            step_number: 1-based step index to load.

        Returns:
            Parsed JSON dict of the ``AgentAction`` payload, or ``None``
            if the artifact file does not exist.
        """
        json_path = self.run_dir / f"step_{step_number:03d}.json"
        if not json_path.exists():
            logger.info(
                "No artifact for step %d at %s.", step_number, json_path
            )
            return None

        try:
            raw = json_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            logger.info(
                "Ghost Mode loaded artifact for step %d.", step_number
            )
            return data
        except Exception as exc:
            logger.error(
                "Failed to load step %d artifact: %s", step_number, exc
            )
            return None

    def list_cached_steps(self) -> List[int]:
        """Return sorted list of step numbers with cached artifacts.

        Returns:
            List of 1-based step numbers for which JSON artifacts exist.
        """
        steps: List[int] = []
        if not self.run_dir.exists():
            return steps
        for path in sorted(self.run_dir.glob("step_*.json")):
            try:
                # Extract step number from filename like "step_003.json"
                num = int(path.stem.split("_")[1])
                steps.append(num)
            except (IndexError, ValueError):
                continue
        return steps

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_coords(
        mark_id: Optional[int],
        id_to_coords: Dict[int, Dict[str, int]],
    ) -> Dict[str, int]:
        """Look up viewport coordinates for a mark ID.

        Args:
            mark_id: The Set-of-Mark numeric identifier.
            id_to_coords: The coordinate lookup table.

        Returns:
            ``{"x": int, "y": int}`` viewport centre of the element.

        Raises:
            RuntimeError: If ``mark_id`` is ``None`` or not found in
                the lookup table.
        """
        if mark_id is None:
            raise RuntimeError(
                "Action requires a target_mark_id but received None."
            )
        if mark_id not in id_to_coords:
            raise RuntimeError(
                f"Mark ID {mark_id} not found in coordinate map. "
                f"Available IDs: {sorted(id_to_coords.keys())[:20]}"
            )
        return id_to_coords[mark_id]
