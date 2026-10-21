"""Current paper implementation; provenance is recorded in provenance/module_map.json."""

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
PALETTE = {
    "INK": "#253B4B",
    "GREY": "#84949E",
    "ORANGE": "#C17D44",
    "GREEN": "#187F78",
    "BLUE": "#416E99",
    "PURPLE": "#806B9D",
    "ROSE": "#AC657D",
}


def finish(fig, name, exclude=None, **alignment):
    import traser.visualization.panels as u
    from matplotlib.text import Text
    from matplotlib.lines import Line2D

    for ax in fig.axes:
        ax.set_facecolor("white")
        for s in ax.spines.values():
            s.set_color(PALETTE["INK"])
        if ax.get_label() != "colorbar":
            ax.tick_params(direction="out", length=3.5, pad=3)
        for t in ax.texts:
            if t.get_text().startswith("n ="):
                t.set_text(t.get_text().replace("n = ", "n = "))
    if name == "hssc_semantic":
        for ax in fig.axes:
            for t in ax.texts:
                if t.get_text().startswith("n ="):
                    t.set_visible(False)
        fig.axes[0].set_position([0.18, 0.59, 0.355, 0.24])
        fig.axes[1].set_position([0.625, 0.59, 0.355, 0.24])
        fig.axes[2].set_position([0.18, 0.105, 0.355, 0.25])
        fig.axes[3].set_position([0.625, 0.105, 0.355, 0.25])
        for t in fig.texts:
            if t.get_text() == "Same-window reconstruction":
                t.set_position((0.18, 0.975))
            if t.get_text() == "Next-window prediction":
                t.set_position((0.18, 0.425))
        for ax in fig.axes[2:]:
            for collection in ax.collections:
                if len(collection.get_offsets()) > 0:
                    collection.set_facecolor(PALETTE["BLUE"])
                    collection.set_edgecolor(PALETTE["BLUE"])
            for line in ax.lines:
                if line.get_linestyle() != "--":
                    line.set_color(PALETTE["BLUE"])
    if name == "hssc_updating":
        import pandas as pd

        data = pd.read_csv(SCI / "particle_forecasts.csv")
        base = pd.read_csv(SCI / "forecast_results.csv")
        data = pd.concat([data, base], ignore_index=True)
        data = data[
            (data.dataset == "hk_weibo") & (data.window_hours == 6) & data.group.isin(["H0", "HA"])
        ]
        grouped = (
            data.groupby(["particles", "origin", "group"]).scaled_sliced_w2.mean().unstack("group")
        )
        delta = grouped.HA - grouped.H0
        mean = delta.groupby("particles").mean()
        assert set(mean.index) == {500, 1000, 2000}
        ax = fig.axes[2]
        ax.clear()
        ax.axhline(0, color=PALETTE["GREY"], ls="--", lw=1)
        ax.plot(range(3), mean.values * 1000, color=PALETTE["INK"], lw=1.2)
        ax.scatter(
            range(3),
            mean.values * 1000,
            c=[PALETTE["GREEN"] if v < 0 else PALETTE["ORANGE"] for v in mean],
            s=38,
            zorder=5,
        )
        ax.set_xticks(range(3), [f"{int(n):,}" for n in mean.index])
        ax.set_xlabel("Particles")
        ax.set_ylabel("Difference × 1,000")
        ax.set_title("c  Weibo · 6 h particle sensitivity", loc="left", fontweight="bold", pad=5)
        ax.set_xlim(-0.15, 2.15)
        write(
            OUT / "particle_panel.json",
            {
                "dataset": "hk_weibo",
                "hours": 6,
                "n_origins_per_budget": 123,
                "aggregation": "Mean three seeds within each origin, then equal-origin mean; no retuning.",
                "mean_differences": {str(int(k)): float(v) for k, v in mean.items()},
            },
        )
    if name == "hssc_interventions":
        short = {
            "blm_x": ["Accountability", "Action", "Anti-racism", "Counter-frames", "Institutions"],
            "hk_weibo": ["Casualties", "Mourning", "Investigation", "Relief", "Survivors"],
        }
        for row in range(3):
            for col, d in enumerate(("blm_x", "hk_weibo")):
                ax = fig.axes[row * 2 + col]
                p = ax.get_position()
                ax.set_position([0.2 if col == 0 else 0.69, p.y0, 0.285, p.height])
                if row < 2:
                    ax.set_yticks(range(5), short[d])
        for t in fig.texts:
            t.set_x(0.2 if "BLM" in t.get_text() else 0.69)
        fig.axes[4].set_ylabel("Injected topic")
    if name == "hssc_update_parameters":
        for ax in fig.axes:
            ax.set_ylabel("Penalty / baseline")
            for line in ax.lines:
                if list(line.get_xdata()) == [0, 1] and list(line.get_ydata()) == [1, 1]:
                    line.set_zorder(0)
                    line.set_linestyle(":")
    if name.startswith("hssc_latent_"):
        for ax in fig.axes:
            if ax.get_label() == "colorbar":
                ax._colorbar.set_ticks([1e-06, 0.001, 1], labels=["0.000001", "0.001", "1"])
                ax.minorticks_off()
    if name == "hssc_latent_combined":
        import matplotlib.dates as dates
        from datetime import datetime

        data = read(OUT / "exact_distributions.json")
        for col, d in enumerate(u.DS):
            q = data[d]
            k = next((i for i, w in enumerate(q["calendar"]) if w["window"] == q["target"]))
            w = q["calendar"][k]
            x = (
                dates.date2num(datetime.fromisoformat(w["start_utc"]))
                + dates.date2num(datetime.fromisoformat(w["end_exclusive_utc"]))
            ) / 2
            ax = fig.axes[col]
            ax.axvline(x, color=PALETTE["BLUE"], ls=":", lw=1, zorder=1)
            ax.scatter([x], [q["history"][k]["rms64"]], s=24, color=PALETTE["BLUE"], zorder=4)
    for t in fig.findobj(Text):
        s = t.get_text()
        if s == "Change × 1,000":
            s = "Change × 1,000"
        t.set_text(s)
    original = u.OUT
    u.OUT = OUT
    try:
        BASE_SAVE(fig, name, exclude, **alignment)
    finally:
        u.OUT = original
