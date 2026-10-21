import csv

"Current paper implementation; provenance is recorded in provenance/module_map.json."
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json, re, string, inspect
import numpy as np
from traser.io import ROOT, write, sha
from traser.visualization.render import read, calendar_path

SCI = ROOT / "evidence/source_data"
OUT = ROOT / "outputs/figures"
REV = OUT
PREV = OUT
FIG = OUT
PAPER = OUT
DS = ("blm_x", "hk_weibo")
HOURS = {"blm_x": (6, 12, 24), "hk_weibo": (3, 6, 12)}
DATA = SCI
METHODS = ("TRASER", "Pure BCM", "WAR", "WR", "LQD")
NAMES = {"blm_x": "BLM / X", "hk_weibo": "Weibo"}


def render():
    import pandas as pd
    import traser.visualization.panels as u

    u.OUT = OUT
    u.setup()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    data = pd.read_csv(DATA / "reference_comparison_origins.csv")
    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 94 / 25.4), facecolor="white")
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.17, top=0.88, wspace=0.46)
    for col, dataset in enumerate(DS):
        ax = axes[col]
        high = 0.0
        for index, method in enumerate(METHODS):
            values = (
                data[(data.dataset == dataset) & (data.method == method)]
                .sort_values("origin")
                .scaled_sliced_w2.to_numpy()
            )
            high = max(high, float(values.max()))
            colour = u.GREEN if index == 0 else u.GREY
            ax.scatter(
                values,
                index + np.linspace(-0.13, 0.13, len(values)),
                s=18,
                facecolors="white",
                edgecolors=colour,
                linewidths=1,
                alpha=0.8,
                zorder=3,
            )
            ax.plot(
                [values.min(), values.max()], [index, index], color=colour, linewidth=1.5, zorder=2
            )
            ax.plot(
                values.mean(),
                index,
                marker="D",
                markersize=6,
                color=colour,
                markeredgewidth=1,
                zorder=4,
            )
        ax.set_yticks(range(5), METHODS)
        ax.set_ylim(4.5, -0.5)
        ax.set_xlim(0, high * 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.set_xlabel("Forecast error")
        ax.tick_params(axis="y", length=0, pad=7)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color="#e5e8e8", linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        u.panel(ax, "ab"[col], NAMES[dataset])
    u.save(fig, "hssc_statistical_references")
