from __future__ import annotations

from typing import Optional, Tuple

from PIL import Image


def crop_with_margin(
        image: Image.Image,
        bbox_xywh: Tuple[float, float, float, float],
        margin_ratio: float = 0.1,
        margin_px: Optional[int] = None,
        min_side: int = 8,
) -> Optional[Image.Image]:
    """
    Crop an image patch from a bounding box with additional margins.

    Args:
        image: PIL image of the full frame.
        bbox_xywh: Bounding box in pixel coordinates (x, y, w, h), where (x, y) is the top-left.
        margin_ratio: Relative margin w.r.t bbox size (applied when margin_px is None).
        margin_px: Absolute pixel margin on both axes. If provided, overrides margin_ratio.
        min_side: Enforce a minimal side length for the crop (in pixels).

    Returns:
        A cropped PIL image, or None if the bbox is invalid.
    """
    x, y, w, h = map(float, bbox_xywh)
    W, H = image.size

    if w <= 0 or h <= 0:
        return None

    # Compute margins
    if margin_px is not None:
        pad_x = float(margin_px)
        pad_y = float(margin_px)
    else:
        pad_x = float(margin_ratio) * w
        pad_y = float(margin_ratio) * h

    # Expand bbox and clamp to image bounds
    x1 = max(0.0, x - pad_x)
    y1 = max(0.0, y - pad_y)
    x2 = min(float(W), x + w + pad_x)
    y2 = min(float(H), y + h + pad_y)

    # Enforce minimal size
    if x2 - x1 < min_side:
        cx = (x1 + x2) / 2.0
        x1 = max(0.0, cx - min_side / 2.0)
        x2 = min(float(W), cx + min_side / 2.0)
    if y2 - y1 < min_side:
        cy = (y1 + y2) / 2.0
        y1 = max(0.0, cy - min_side / 2.0)
        y2 = min(float(H), cy + min_side / 2.0)

    # Integer box and final guard
    box = (int(x1), int(y1), int(x2), int(y2))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return image.crop(box)


def xywh_to_xyxy_with_margin(
        bbox_xywh: Tuple[float, float, float, float],
        image_size: Tuple[int, int],
        margin_ratio: float = 0.1,
        margin_px: Optional[int] = None,
        min_side: int = 8,
) -> Tuple[int, int, int, int]:
    """
    Compute an expanded x1,y1,x2,y2 box from xywh with margins, clipped to image bounds.
    Useful if you want the coordinates without doing the actual PIL crop.
    """
    x, y, w, h = map(float, bbox_xywh)
    W, H = map(float, image_size)

    if margin_px is not None:
        pad_x = float(margin_px)
        pad_y = float(margin_px)
    else:
        pad_x = float(margin_ratio) * w
        pad_y = float(margin_ratio) * h

    x1 = max(0.0, x - pad_x)
    y1 = max(0.0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)

    if x2 - x1 < min_side:
        cx = (x1 + x2) / 2.0
        x1 = max(0.0, cx - min_side / 2.0)
        x2 = min(W, cx + min_side / 2.0)
    if y2 - y1 < min_side:
        cy = (y1 + y2) / 2.0
        y1 = max(0.0, cy - min_side / 2.0)
        y2 = min(H, cy + min_side / 2.0)

    return int(x1), int(y1), int(x2), int(y2)


