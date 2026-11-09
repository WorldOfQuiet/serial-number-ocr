"""Small plotting helpers for the static README figure; not used by inference."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

BG = "#f5f6f3"
INK = "#16392e"
MUTED = "#697971"
LINE = "#d9e1d9"
GREEN = "#285d49"
MONO = "DejaVu Sans Mono"


def canvas(width=1800, height=1120):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path"})
    figure = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=BG)
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set(xlim=(0, width), ylim=(height, 0))
    axis.set_autoscale_on(False)
    axis.axis("off")
    return figure, axis


def text(axis, x, y, value, size=16, color=INK, weight="normal", mono=False, **kwargs):
    return axis.text(x, y, value, fontsize=size, color=color, weight=weight,
                     fontfamily=MONO if mono else "DejaVu Sans", va="top", **kwargs)


def box(axis, x, y, width, height, fill="white", edge=LINE, radius=18):
    patch = FancyBboxPatch((x, y), width, height, boxstyle=f"round,pad=0,rounding_size={radius}",
                           facecolor=fill, edgecolor=edge, linewidth=1, zorder=1)
    axis.add_patch(patch)
    return patch


def photo(axis, pixels, x, y, width, height):
    pixels = np.asarray(pixels)
    scale = min(width / pixels.shape[1], height / pixels.shape[0])
    w, h = pixels.shape[1] * scale, pixels.shape[0] * scale
    left, top = x + (width - w) / 2, y + (height - h) / 2
    axis.imshow(pixels, extent=(left, left + w, top + h, top),
                interpolation="bilinear", aspect="auto", zorder=2)
    return left, top, scale


def step(axis, x, y, number, label):
    text(axis, x, y, number, 13, GREEN, mono=True, weight="bold")
    text(axis, x + 40, y - 2, label, 16, INK, weight="bold")


def arrow(axis, start, end, color=GREEN):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=19,
                                  linewidth=1.5, color=color, zorder=5))


def save(figure, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width, height = figure.canvas.get_width_height()
    for axis in figure.axes:
        for label in axis.texts:
            bounds = label.get_window_extent(renderer)
            if bounds.x0 < 0 or bounds.y0 < 0 or bounds.x1 > width or bounds.y1 > height:
                raise ValueError(f"Text would be clipped: {label.get_text()}")
    figure.savefig(destination.with_suffix(".png"), dpi=100, facecolor=BG)
    svg = destination.with_suffix(".svg")
    figure.savefig(svg, facecolor=BG)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(figure)
