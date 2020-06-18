from typing import Tuple, Sized, List

import numpy as np


def hex_to_rgb(hex: str) -> Tuple[int, ...]:
    hex = hex.lstrip("#")
    return tuple(int(hex[i: i + 2], 16) for i in (0, 2, 4))


def n_colors(n: int) -> List[Tuple[int, int, int]]:
    colors = []
    r = int(np.random.random() * 256)
    g = int(np.random.random() * 256)
    b = int(np.random.random() * 256)
    step = 256 / n
    for _ in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        colors.append((r, g, b))
    return colors

def parse_and_validate_color(color) -> Tuple[int, int, int]:
    if not isinstance(color, str) and not (
        isinstance(color, Sized) and len(color) == 3
    ):
        raise ValueError("Colors needs to be defined either by 3 ints or a hex string")
    if isinstance(color, str):
        color = hex_to_rgb(color)
    color = np.array(color).astype(np.float32)
    if (color <= 1.0).all():
        color = np.round(color * 255)
    color = color.astype(np.int32)
    if np.max(color) > 255 or np.min(color) < 0:
        raise ValueError("A color should have values between 0 and 255")
    return color[0], color[1], color[2]