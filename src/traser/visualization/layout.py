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
import traser.visualization.panels as u


def compact_save(fig, name, exclude=None, **alignment):
    """Place comparable plots in a tighter grid without scaling any type."""
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt

    if name in ("hssc_observed", "hssc_latent_combined"):
        pass
    elif name == "hssc_semantic_shapes":
        fig.set_size_inches(180 / 25.4, 171 / 25.4)
        fig.subplots_adjust(left=0.1, right=0.98, top=0.81, bottom=0.09, wspace=0.23, hspace=0.5)
        for ax in fig.axes:
            ax.title.set_y(1)
            ax.set_title(ax.get_title(loc="left"), loc="left", fontweight="bold", pad=5)
            for t in ax.texts:
                if t.get_text().startswith("P(0)"):
                    t.set_visible(False)
    elif name == "hssc_semantic":
        fig.set_size_inches(180 / 25.4, 150 / 25.4)
        for row in range(2):
            for col in range(2):
                ax = fig.axes[row * 2 + col]
                ax.set_position(
                    [0.18 if col == 0 else 0.625, 0.58 if row == 0 else 0.11, 0.355, 0.23]
                )
                if col == 1:
                    ax.tick_params(labelleft=False)
        for ax in fig.axes:
            for t in ax.texts:
                if t.get_text().startswith("n ="):
                    t.set_position((0, 1.18))
            ax.set_title(ax.get_title(loc="left"), loc="left", fontweight="bold", pad=5)
        for t in fig.texts:
            if t.get_text() == "Next-window prediction":
                t.set_y(0.445)
            if t.get_text().startswith("Difference ="):
                t.set_visible(False)
    elif name == "hssc_updating":
        fig.set_size_inches(180 / 25.4, 150 / 25.4)
        fig.axes[0].set_position([0.12, 0.49, 0.37, 0.31])
        fig.axes[1].set_position([0.61, 0.49, 0.37, 0.31])
        fig.axes[2].set_position([0.12, 0.105, 0.86, 0.23])
        fig.axes[2].set_ylabel("Mean difference")
        for ax in fig.axes:
            ax.set_title(ax.get_title(loc="left"), loc="left", fontweight="bold", pad=5)
        for t in fig.texts:
            t.set_visible(False)
        fig.legends[0].set_bbox_to_anchor((0.55, 0.99))
    elif name == "hssc_interventions":
        fig.set_size_inches(180 / 25.4, 176 / 25.4)
        fig.subplots_adjust(left=0.13, right=0.98, bottom=0.26, top=0.87, wspace=0.28, hspace=0.68)
        for ax in fig.axes[:6]:
            ax.set_title(ax.get_title(loc="left"), loc="left", fontweight="bold", pad=5)
        for t in fig.texts:
            t.set_x(0.13 if t.get_text() == u.NAMES[u.DS[0]] else 0.61)
        fig.axes[-1].set_position([0.3, 0.067, 0.52, 0.018])
    elif name.startswith("hssc_semantic_trajectory"):
        fig.set_size_inches(160 / 25.4, 168 / 25.4)
        fig.subplots_adjust(left=0.13, right=0.98, bottom=0.1, top=0.85, wspace=0.3, hspace=0.49)
        for ax in fig.axes:
            ax.set_title(ax.get_title(loc="left"), loc="left", fontweight="bold", pad=5)
    elif name.startswith("hssc_semantic_shapes"):
        nrows = len(fig.axes) // 2
        fig.set_size_inches(160 / 25.4, (32 + nrows * 26) / 25.4)
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.095, top=0.83, wspace=0.27, hspace=0.5)
        for ax in fig.axes:
            ax.set_title(ax.get_title(loc="left"), loc="left", fontweight="bold", pad=5)
        for leg in fig.legends:
            leg.set_ncols(4)
            leg.set_bbox_to_anchor((0.55, 1.01))
    elif name.startswith("hssc_latent_"):
        fig.set_size_inches(160 / 25.4, 119 / 25.4)
        fig.subplots_adjust(left=0.11, right=0.98, bottom=0.24, top=0.88, wspace=0.28, hspace=0.59)
        for ax in fig.axes[:6]:
            ax.set_title(ax.get_title(loc="left"), loc="left", fontweight="bold", pad=5)
            ax.xaxis.set_major_locator(MaxNLocator(2))
        for i, t in enumerate(fig.texts):
            t.set_y(0.95 if i == 0 else 0.55)
        fig.axes[-1].set_position([0.25, 0.105, 0.53, 0.021])
    elif name == "hssc_update_parameters":
        fig.set_size_inches(160 / 25.4, 109 / 25.4)
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.13, top=0.84, hspace=0.49)
        for ax in fig.axes:
            ax.set_title(ax.get_title(loc="left"), loc="left", fontweight="bold", pad=5)
    elif name == "hssc_origin_diagnostics":
        fig.set_size_inches(160 / 25.4, 137 / 25.4)
        fig.subplots_adjust(left=0.15, right=0.98, bottom=0.11, top=0.93, wspace=0.31, hspace=0.6)
        for ax in fig.axes:
            ax.set_title(ax.get_title(loc="left"), loc="left", fontweight="bold", pad=5)
    elif name.startswith("hssc_injection_all_"):
        fig.set_size_inches(160 / 25.4, 132 / 25.4)
        fig.subplots_adjust(left=0.28, right=0.96, bottom=0.3, top=0.91)
        fig.axes[-1].set_position([0.36, 0.095, 0.48, 0.022])
    else:
        raise ValueError(name)
    original_out = u.OUT
    u.OUT = OUT
    try:
        ORIGINAL_SAVE(fig, name, exclude, **alignment)
    finally:
        u.OUT = original_out


def latent():
    import matplotlib.pyplot as plt
    import matplotlib.dates as dates
    from datetime import datetime
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import MaxNLocator

    data = read(PREV / "exact_distributions.json")
    old = read(REV / "shape_distributions.json")["datasets"]
    fig, axes = plt.subplots(
        4, 2, figsize=(180 / 25.4, 154 / 25.4), gridspec_kw={"height_ratios": [0.75, 1, 1, 1]}
    )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.18, top=0.88, wspace=0.24, hspace=0.62)
    for col, d in enumerate(u.DS):
        q = data[d]
        xx = [dates.date2num(datetime.fromisoformat(w["start_utc"])) for w in q["calendar"]]
        end = dates.date2num(datetime.fromisoformat(q["calendar"][-1]["end_exclusive_utc"]))
        centres = (np.array(xx) + np.r_[xx[1:], end]) / 2
        ax = axes[0, col]
        ax.plot(centres, [h["rms64"] for h in q["history"]], color=u.INK, lw=1.5)
        ax.set_xlim(xx[0], end)
        ax.set_xticks([xx[0], end])
        ax.xaxis.set_major_formatter(dates.DateFormatter("%d %b"))
        ax.get_xticklabels()[0].set_ha("left")
        ax.get_xticklabels()[-1].set_ha("right")
        ax.yaxis.set_major_locator(MaxNLocator(2))
        ax.set_title(f"{'ab'[col]}  Latent dispersion", loc="left", fontweight="bold", pad=5)
        if col == 0:
            ax.set_ylabel("RMS")
        src = old[d]
        for row, kind in enumerate(("Observed", "Forecast", "Analysis"), 1):
            rr = [r for r in src["records"] if r["window"] == q["target"] and r["kind"] == kind]
            m = np.mean([r["latent_mass"] for r in rr], axis=0)
            assert np.isclose(m.sum(), 1)
            ax = axes[row, col]
            im = ax.pcolormesh(
                *src["latent_edges"],
                np.ma.masked_equal(m.T, 0),
                cmap="Blues",
                norm=LogNorm(1e-06, 1),
                rasterized=True,
            )
            ax.set_title(
                f"{'cdefgh'[(row - 1) * 2 + col]}  {kind}",
                loc="left",
                fontweight="bold",
                pad=5,
                color=u.COLORS[kind],
            )
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(2))
            if col == 0:
                ax.set_ylabel("PC2")
            if row < 3:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("PC1", labelpad=2)
        fig.text(0.11 if col == 0 else 0.595, 0.96, u.NAMES[d], fontweight="bold")
    cb = fig.add_axes([0.28, 0.085, 0.51, 0.017])
    cb.set_label("colorbar")
    fig.colorbar(im, cax=cb, orientation="horizontal", ticks=[1e-06, 0.001, 1], format="%.6g")
    cb.minorticks_off()
    cb.set_xlabel("Probability mass per bin", labelpad=3)
    compact_save(fig, "hssc_latent_combined", exclude=[cb])


def observed():
    import matplotlib.pyplot as plt
    import matplotlib.dates as dates
    from datetime import datetime
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator, NullLocator, FuncFormatter

    rec = read(SCI / "reconstruction.json")
    fig, axs = plt.subplots(3, 2, figsize=(180 / 25.4, 140 / 25.4))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.11, top=0.86, wspace=0.29, hspace=0.5)
    phasecolors = ["#EDF0F3", "#CBD3DC", "#9EADBD"]
    fig.legend(
        handles=[
            Patch(fc=c, ec=u.INK, label=p)
            for c, p in zip(phasecolors, ["Train", "Development", "Test"])
        ],
        loc="upper center",
        bbox_to_anchor=(0.55, 1),
        ncol=3,
        frameon=False,
    )
    for col, d in enumerate(u.DS):
        cal = read(calendar_path(d, 12))["windows"]
        rr = {
            r["window"]: r
            for r in rec
            if (r["dataset"], r["window_hours"], r["group"]) == (d, 12, "HA")
        }
        starts = np.array([dates.date2num(datetime.fromisoformat(w["start_utc"])) for w in cal])
        ends = np.array(
            [dates.date2num(datetime.fromisoformat(w["end_exclusive_utc"])) for w in cal]
        )
        x = (starts + ends) / 2
        ax = axs[0, col]
        ax.plot(x, [w["eligible_occurrences"] for w in cal], color=u.BLUE)
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticks([1000, 100000] if col == 0 else [10, 1000])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_title(f"{'ab'[col]}  {u.NAMES[d]} · Posts", loc="left", fontweight="bold", pad=5)
        for phase, color in zip(["train", "development", "test"], phasecolors):
            group = [i for i, w in enumerate(cal) if w["phase"] == phase]
            ax.axvspan(
                starts[group[0]], ends[group[-1]], ymin=0.97, ymax=1, color=color, ec=u.INK, lw=1
            )
        for row, indices, palette in [
            (1, [0, 1, 4] if col == 0 else [0, 1, 3], [u.BLUE, u.GREEN, u.PURPLE]),
            (2, [5, 6], [u.ORANGE, u.ROSE]),
        ]:
            ax = axs[row, col]
            for j, color in zip(indices, palette):
                ax.plot(x, [rr[w["window"]]["observed_scores"][j] for w in cal], color=color)
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(3))
            labels = [
                (u.TOPICS[d] + u.EMOTIONS)[j].replace("Collective action", "Action")
                for j in indices
            ]
            ax.legend(
                handles=[Line2D([], [], color=c, label=label) for c, label in zip(palette, labels)],
                loc="lower left",
                bbox_to_anchor=(0, 1.02),
                ncol=2,
                frameon=False,
                handlelength=1.1,
                columnspacing=0.8,
                handletextpad=0.4,
                borderaxespad=0,
            )
            ax.text(
                -0.13,
                1.08,
                "cdef"[(row - 1) * 2 + col],
                transform=ax.transAxes,
                fontweight="bold",
                va="bottom",
            )
        for row in range(3):
            ax = axs[row, col]
            for phase in ["development", "test"]:
                ax.axvline(
                    starts[next((i for i, w in enumerate(cal) if w["phase"] == phase))],
                    color=u.GREY,
                    ls=":",
                    lw=1,
                )
            ax.set_xlim(starts[0], ends[-1])
            ax.set_xticks([starts[0], (starts[0] + ends[-1]) / 2, ends[-1]])
            ax.xaxis.set_major_formatter(dates.DateFormatter("%d %b"))
            if row < 2:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Date (UTC)")
                ax.get_xticklabels()[0].set_ha("left")
                ax.get_xticklabels()[-1].set_ha("right")
            if col == 0:
                ax.set_ylabel("Posts" if row == 0 else "Mean score")
    compact_save(fig, "hssc_observed")
