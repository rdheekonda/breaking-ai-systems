"""Image transforms for multimodal attack demonstrations."""

import base64
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont
from PIL.ImageFont import ImageFont as PILImageFont


def add_text_overlay(
    image_path: str,
    text: str,
    output_path: str,
    font_size: int = 36,
    color: tuple = (255, 0, 0),
) -> str:
    """Render bold text overlay on an image and save the result.

    Args:
        image_path:  Path to the source image.
        text:        Text to render on the image.
        output_path: Where to save the modified image.
        font_size:   Font size in pixels.
        color:       RGB tuple for text color.

    Returns:
        The output_path for convenience.
    """
    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Use default font at requested size
    font: FreeTypeFont | PILImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except OSError:
            font = ImageFont.load_default(size=font_size)

    # Center the text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (img.width - tw) // 2
    y = (img.height - th) // 2

    # Semi-transparent black background behind text
    padding = 12
    draw.rectangle(
        (x - padding, y - padding, x + tw + padding, y + th + padding),
        fill=(0, 0, 0, 180),
    )
    draw.text((x, y), text, fill=color + (255,), font=font)

    result = Image.alpha_composite(img, overlay).convert("RGB")
    result.save(output_path)
    return output_path


def image_to_base64(image_path: str) -> str:
    """Read an image file and return a data-URI string for litellm.

    Args:
        image_path: Path to a PNG/JPEG image.

    Returns:
        String in the format ``data:image/png;base64,...``
    """
    data = Path(image_path).read_bytes()
    b64 = base64.b64encode(data).decode()
    ext = Path(image_path).suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{b64}"
