"""Set-of-Mark Perception & Accessibility Annotation Engine.

This module is the **eyes** of the autonomous testing agent.  It bridges
the raw browser viewport with the vision-language model by:

1. Injecting a JavaScript routine via Playwright that enumerates *every*
   interactive DOM element (links, buttons, inputs, selects, ARIA widgets)
   and returns their bounding-box geometry, accessible names, and roles.
2. Filtering out invisible, off-screen, or zero-dimension elements so the
   model never wastes tokens on phantom targets.
3. Drawing high-contrast numbered badges (``[1]``, ``[2]``, …) on top of
   a clean screenshot using **Pillow**, producing the *Set-of-Mark*
   annotated image the VLM consumes.
4. Building a serialised accessibility-tree text block that pairs each
   mark ID with its role, name, and state for non-visual reasoning.

Public API:
    ``capture_and_annotate(page)``
        → ``(annotated_png_bytes, a11y_text, id_to_coords)``
"""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import Page

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BADGE_BG_COLOR = (230, 50, 50)       # Vibrant red background for badges
_BADGE_TEXT_COLOR = (255, 255, 255)    # White text
_BADGE_FONT_SIZE = 12
_BADGE_PADDING = 2

# JavaScript to enumerate all interactive elements and their metadata.
# Returns a JSON-serialisable array of element descriptors.
_JS_ENUMERATE_ELEMENTS = """
() => {
    const INTERACTIVE_SELECTORS = [
        'a[href]',
        'button',
        'input',
        'select',
        'textarea',
        '[role="button"]',
        '[role="link"]',
        '[role="checkbox"]',
        '[role="radio"]',
        '[role="tab"]',
        '[role="menuitem"]',
        '[role="option"]',
        '[role="switch"]',
        '[role="combobox"]',
        '[role="textbox"]',
        '[role="searchbox"]',
        '[contenteditable="true"]',
        '[tabindex]:not([tabindex="-1"])',
    ];

    const selector = INTERACTIVE_SELECTORS.join(', ');
    const candidates = document.querySelectorAll(selector);
    const results = [];

    for (const el of candidates) {
        // --- Visibility gate -------------------------------------------
        const style = window.getComputedStyle(el);
        if (
            style.display === 'none' ||
            style.visibility === 'hidden' ||
            parseFloat(style.opacity) === 0
        ) {
            continue;
        }

        const rect = el.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) continue;

        // Filter elements entirely outside the viewport
        if (
            rect.bottom < 0 ||
            rect.right < 0 ||
            rect.top > window.innerHeight ||
            rect.left > window.innerWidth
        ) {
            continue;
        }

        // --- Accessible name resolution --------------------------------
        const ariaLabel = el.getAttribute('aria-label') || '';
        const ariaLabelledBy = el.getAttribute('aria-labelledby') || '';
        let accessibleName = ariaLabel;
        if (!accessibleName && ariaLabelledBy) {
            const labelEl = document.getElementById(ariaLabelledBy);
            accessibleName = labelEl ? labelEl.textContent.trim() : '';
        }
        if (!accessibleName) {
            // Fallback: visible text, title, placeholder, value
            accessibleName = (
                el.textContent?.trim()?.substring(0, 80) ||
                el.getAttribute('title') ||
                el.getAttribute('placeholder') ||
                el.getAttribute('value') ||
                ''
            );
        }

        const role =
            el.getAttribute('role') ||
            el.tagName.toLowerCase();

        const hasA11yLabel = !!(
            el.getAttribute('aria-label') ||
            el.getAttribute('aria-labelledby') ||
            el.closest('label') ||
            (el.id && document.querySelector(`label[for="${el.id}"]`))
        );

        results.push({
            tag: el.tagName.toLowerCase(),
            role: role,
            name: accessibleName.substring(0, 120),
            hasA11yLabel: hasA11yLabel,
            x: Math.round(rect.x),
            y: Math.round(rect.y),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
            cx: Math.round(rect.x + rect.width / 2),
            cy: Math.round(rect.y + rect.height / 2),
        });
    }

    return results;
}
"""


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

async def capture_and_annotate(
    page: Page,
) -> Tuple[bytes, str, Dict[int, Dict[str, int]]]:
    """Capture the current viewport, annotate interactive elements, and
    build an accessibility-tree text summary.

    This is the primary entry point for the perception layer.  It performs
    the full pipeline: screenshot → element enumeration → badge drawing →
    accessibility text assembly.

    Args:
        page: An active Playwright ``Page`` instance with the target URL
            already loaded and stable.

    Returns:
        A 3-tuple of:
            - **annotated_image_bytes** (``bytes``): PNG bytes of the
              viewport screenshot with Set-of-Mark badges overlaid.
            - **a11y_text** (``str``): Serialised accessibility tree where
              each line describes one interactive element with its mark
              ID, role, name, and label status.
            - **id_to_coords** (``Dict[int, Dict[str, int]]``): Mapping
              from mark ID → ``{"x": int, "y": int}`` giving the exact
              viewport centre of each annotated element.

    Raises:
        RuntimeError: If the JavaScript evaluation or screenshot capture
            fails (e.g., page navigated away, context destroyed).
    """
    # 1. Enumerate interactive elements via injected JS
    elements = await _enumerate_elements(page)
    logger.info("Enumerated %d interactive elements.", len(elements))

    # 2. Capture a clean screenshot (before badge overlay)
    screenshot_bytes = await _take_screenshot(page)

    # 3. Draw Set-of-Mark badges on the screenshot
    annotated_bytes, id_to_coords = _draw_badges(screenshot_bytes, elements)

    # 4. Build the accessibility-tree text
    a11y_text = _build_a11y_text(elements)

    logger.info(
        "Perception complete: %d marks, %d bytes annotated image.",
        len(id_to_coords),
        len(annotated_bytes),
    )
    return annotated_bytes, a11y_text, id_to_coords


# ═══════════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════════

async def _enumerate_elements(page: Page) -> List[Dict[str, Any]]:
    """Inject JavaScript and return a list of interactive-element metadata.

    Each element dict contains keys: ``tag``, ``role``, ``name``,
    ``hasA11yLabel``, ``x``, ``y``, ``width``, ``height``, ``cx``, ``cy``.

    Raises:
        RuntimeError: If JS evaluation fails.
    """
    try:
        elements = await page.evaluate(_JS_ENUMERATE_ELEMENTS)
        if not isinstance(elements, list):
            logger.warning(
                "JS enumerate returned non-list type %s; defaulting to [].",
                type(elements).__name__,
            )
            return []
        return elements
    except Exception as exc:
        logger.error("Element enumeration JS failed: %s", exc)
        raise RuntimeError(
            f"Failed to enumerate interactive elements: {exc}"
        ) from exc


async def _take_screenshot(page: Page) -> bytes:
    """Capture a full-viewport PNG screenshot.

    Raises:
        RuntimeError: If the screenshot capture fails.
    """
    try:
        png_bytes: bytes = await page.screenshot(type="png")
        logger.debug("Screenshot captured: %d bytes.", len(png_bytes))
        return png_bytes
    except Exception as exc:
        logger.error("Screenshot capture failed: %s", exc)
        raise RuntimeError(
            f"Failed to capture screenshot: {exc}"
        ) from exc


def _draw_badges(
    screenshot_bytes: bytes,
    elements: List[Dict[str, Any]],
) -> Tuple[bytes, Dict[int, Dict[str, int]]]:
    """Overlay numbered Set-of-Mark badges on the screenshot image.

    Each visible interactive element receives a red rectangle with a white
    number label positioned near its top-left corner.

    Args:
        screenshot_bytes: Raw PNG bytes of the clean viewport screenshot.
        elements: List of element metadata dicts from ``_enumerate_elements``.

    Returns:
        Tuple of (annotated PNG bytes, mark-ID → coordinate mapping).
    """
    image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Attempt to load a monospaced font; fall back to Pillow default.
    try:
        font = ImageFont.truetype("arial.ttf", _BADGE_FONT_SIZE)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                _BADGE_FONT_SIZE,
            )
        except (OSError, IOError):
            font = ImageFont.load_default()
            logger.debug("Using Pillow default font for badge labels.")

    id_to_coords: Dict[int, Dict[str, int]] = {}

    for mark_id, elem in enumerate(elements, start=1):
        cx: int = elem["cx"]
        cy: int = elem["cy"]
        ex: int = elem["x"]
        ey: int = elem["y"]

        label = f"[{mark_id}]"

        # Measure text bounding box
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Position badge at the top-left corner of the element
        badge_x = max(0, ex - _BADGE_PADDING)
        badge_y = max(0, ey - text_h - 2 * _BADGE_PADDING)

        # Draw filled rectangle background
        draw.rectangle(
            [
                badge_x,
                badge_y,
                badge_x + text_w + 2 * _BADGE_PADDING,
                badge_y + text_h + 2 * _BADGE_PADDING,
            ],
            fill=(*_BADGE_BG_COLOR, 220),
        )

        # Draw text label
        draw.text(
            (badge_x + _BADGE_PADDING, badge_y + _BADGE_PADDING),
            label,
            fill=(*_BADGE_TEXT_COLOR, 255),
            font=font,
        )

        # Store coordinate mapping
        id_to_coords[mark_id] = {"x": cx, "y": cy}

    # Composite overlay onto original image
    composited = Image.alpha_composite(image, overlay).convert("RGB")

    # Encode to PNG bytes
    buffer = io.BytesIO()
    composited.save(buffer, format="PNG")
    annotated_bytes = buffer.getvalue()

    logger.debug("Drew %d badges on screenshot.", len(id_to_coords))
    return annotated_bytes, id_to_coords


def _build_a11y_text(elements: List[Dict[str, Any]]) -> str:
    """Serialise element metadata into a structured accessibility tree.

    Each line has the format::

        [mark_id] role="<role>" name="<name>" a11y_label=<yes|NO>

    The ``NO`` flag (uppercase) draws the VLM's attention to missing
    labels so it can flag them in ``friction_metrics``.

    Args:
        elements: List of element metadata dicts from ``_enumerate_elements``.

    Returns:
        Multi-line string representing the accessibility tree.
    """
    lines: List[str] = []
    for mark_id, elem in enumerate(elements, start=1):
        a11y_flag = "yes" if elem.get("hasA11yLabel", False) else "NO"
        name_display = elem.get("name", "").strip() or "(unnamed)"
        line = (
            f"[{mark_id}] role=\"{elem.get('role', 'unknown')}\" "
            f"name=\"{name_display}\" "
            f"a11y_label={a11y_flag}"
        )
        lines.append(line)

    a11y_text = "\n".join(lines)
    logger.debug("Built a11y text with %d entries.", len(lines))
    return a11y_text
