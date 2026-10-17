"""Draw the macOS window chrome the demo is composited into.

Writes demo/out/frame.png: a 1920x1080 canvas with a gradient backdrop, a
soft drop shadow, and a dark terminal window (title bar, traffic lights,
title, hairline border, rounded corners) whose content area is transparent.
render.sh overlays the terminal recording under it at CONTENT_X/CONTENT_Y.
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080
WIN_X, WIN_Y, WIN_W, WIN_H = 150, 76, 1620, 928
BAR = 40
RADIUS = 12
CONTENT_X, CONTENT_Y = WIN_X, WIN_Y + BAR  # must match render.sh
TITLE = "releves - claude - 162×44"


def backdrop():
    top, bottom = (48, 43, 99), (186, 92, 128)
    grad = Image.new("RGB", (1, H))
    for y in range(H):
        t = y / (H - 1)
        grad.putpixel((0, y), tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)))
    return grad.resize((W, H)).convert("RGBA")


def main(out):
    img = backdrop()

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (WIN_X, WIN_Y + 18, WIN_X + WIN_W, WIN_Y + WIN_H + 18), RADIUS, fill=(0, 0, 0, 150)
    )
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(28)))

    # Window body, then punch the content area out so the video shows through.
    win = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(win)
    box = (WIN_X, WIN_Y, WIN_X + WIN_W - 1, WIN_Y + WIN_H - 1)
    d.rounded_rectangle(box, RADIUS, fill=(40, 40, 54, 255))
    img = Image.alpha_composite(img, win)

    hole = Image.new("L", (W, H), 0)
    hd = ImageDraw.Draw(hole)
    hd.rounded_rectangle(box, RADIUS, fill=255)
    hd.rectangle((WIN_X, WIN_Y, WIN_X + WIN_W, CONTENT_Y - 1), fill=0)
    alpha = img.getchannel("A").point(lambda a: a)
    alpha.paste(0, mask=hole)
    img.putalpha(alpha)

    d = ImageDraw.Draw(img)
    d.line((WIN_X, CONTENT_Y - 1, WIN_X + WIN_W, CONTENT_Y - 1), fill=(20, 20, 28, 255))
    for i, color in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
        cx, cy = WIN_X + 22 + i * 21, WIN_Y + BAR // 2
        d.ellipse((cx - 6.5, cy - 6.5, cx + 6.5, cy + 6.5), fill=color + (255,))
    font = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 15)
    tw = d.textlength(TITLE, font=font)
    d.text((WIN_X + (WIN_W - tw) / 2, WIN_Y + BAR / 2), TITLE, font=font,
           fill=(200, 200, 212, 255), anchor="lm")

    # Hairline border on top of everything, like a real window edge.
    edge = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(edge).rounded_rectangle(box, RADIUS, outline=(255, 255, 255, 40), width=1)
    img = Image.alpha_composite(img, edge)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "out" / "frame.png"))
