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
DS = ("blm_x", "hk_weibo")
NAMES = {"blm_x": "BLM / X", "hk_weibo": "Weibo"}
DS = ("blm_x", "hk_weibo")
NAMES = {"blm_x": "BLM / X", "hk_weibo": "Weibo"}
TOPICS = {
    "blm_x": [
        "Accountability",
        "Collective action",
        "Anti-racism",
        "Counter-narratives",
        "Institutions",
    ],
    "hk_weibo": ["Casualties", "Mourning", "Investigation", "Relief", "Survivor accounts"],
}
EMOTIONS = ["Anger", "Sadness", "Fear", "Disgust", "Joy", "Surprise"]
INK = "#243746"
BLUE = "#2F668B"
GREEN = "#29816E"
ORANGE = "#BD7043"
PURPLE = "#78669B"
GREY = "#657580"
INK = "#243746"
BLUE = "#2F668B"
GREEN = "#29816E"
ORANGE = "#BD7043"
PURPLE = "#78669B"
GREY = "#657580"
INK = "#243746"
BLUE = "#2F668B"
GREEN = "#29816E"
ORANGE = "#BD7043"
PURPLE = "#78669B"
GREY = "#657580"
INK = "#243746"
BLUE = "#2F668B"
GREEN = "#29816E"
ORANGE = "#BD7043"
PURPLE = "#78669B"
GREY = "#657580"
INK = "#243746"
BLUE = "#2F668B"
GREEN = "#29816E"
ORANGE = "#BD7043"
PURPLE = "#78669B"
GREY = "#657580"
INK = "#243746"
BLUE = "#2F668B"
GREEN = "#29816E"
ORANGE = "#BD7043"
PURPLE = "#78669B"
GREY = "#657580"
PALETTE = {"Observed": INK, "H0": GREY, "Forecast": ORANGE, "Analysis": GREEN}
STYLES = {"Observed": "-", "H0": ":", "Forecast": "--", "Analysis": "-."}


def labelled(ax, label, title):
    ax.set_title(f"{label}  {title}", loc="left", fontweight="bold", pad=10)


def legends(fig, labels, columns=2, y=0.99):
    from matplotlib.lines import Line2D

    display = {
        "H0": "Pure BCM",
        "Forecast": "Forecast",
        "Analysis": "Analysis",
        "Observed": "Observed",
    }
    handles = [Line2D([], [], color=PALETTE[k], ls=STYLES[k], label=display[k]) for k in labels]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=columns,
        frameon=False,
        handlelength=2,
        columnspacing=1.3,
        handletextpad=0.6,
    )


def mean_records(data, t, kind, field):
    rows = [r for r in data["records"] if r["window"] == t and r["kind"] == kind]
    return np.array([r[field] for r in rows])


def shapes(manifest):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    data = read(OUT / "shape_distributions.json")["datasets"]

    def latent(name, which):
        fig, axs = plt.subplots(2, 3, figsize=(160 / 25.4, 142 / 25.4))
        fig.subplots_adjust(left=0.11, right=0.98, bottom=0.2, top=0.9, wspace=0.3, hspace=0.65)
        vals = []
        for d in DS:
            q = data[d]
            middle = q["selected_test_targets"][1]
            for j in range(3):
                t = [3, 14, 26][j] if which == "observed" else middle
                kind = (
                    "Observed" if which == "observed" else ["Observed", "Forecast", "Analysis"][j]
                )
                vals.append(mean_records(q, t, kind, "latent_mass").mean(0))
        norm = LogNorm(vmin=1e-06, vmax=1)
        for i, d in enumerate(DS):
            q = data[d]
            middle = q["selected_test_targets"][1]
            for j in range(3):
                ax = axs[i, j]
                t = [3, 14, 26][j] if which == "observed" else middle
                cal = next((x for x in q["calendar"] if x["window"] == t))
                title = (
                    datetime.fromisoformat(cal["start_utc"]).strftime("%d %b")
                    if which == "observed"
                    else ["Observed", "Forecast", "Analysis"][j]
                )
                labelled(ax, string.ascii_lowercase[i * 3 + j], title)
                mesh = ax.pcolormesh(
                    *q["latent_edges"],
                    np.ma.masked_equal(vals[i * 3 + j].T, 0),
                    cmap="Blues",
                    norm=norm,
                    rasterized=True,
                )
                ax.xaxis.set_major_locator(MaxNLocator(3))
                ax.yaxis.set_major_locator(MaxNLocator(3))
                if i == 1:
                    ax.set_xlabel("PC1")
                if j == 0:
                    ax.set_ylabel("PC2")
            fig.text(0.11, 0.955 if i == 0 else 0.55, NAMES[d], fontweight="bold", fontsize=11)
        cb = fig.add_axes([0.22, 0.075, 0.56, 0.028])
        cb.set_label("colorbar")
        fig.colorbar(
            mesh, cax=cb, orientation="horizontal", ticks=[1e-05, 0.001, 0.1], format="%.5g"
        )
        cb.minorticks_off()
        cb.set_xlabel("Probability mass per bin", labelpad=3)
        save(fig, name, manifest)

    latent("hssc_latent_observed", "observed")
    latent("hssc_latent_comparison", "comparison")

    def semantic(name, position, axes=None):
        axissets = {
            d: ([0, 1, 4, 5, 6] if d == "blm_x" else [0, 1, 3, 5, 6]) if axes is None else axes
            for d in DS
        }
        nr = len(axissets[DS[0]])
        fig, axs = plt.subplots(nr, 2, figsize=(160 / 25.4, (40 + nr * 30) / 25.4), squeeze=False)
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.08, top=0.84, wspace=0.4, hspace=0.65)
        legends(fig, ["Observed", "H0", "Forecast", "Analysis"], 2)
        for c, d in enumerate(DS):
            q = data[d]
            t = q["selected_test_targets"][position]
            for r, j in enumerate(axissets[d]):
                ax = axs[r, c]
                for kind in ["H0", "Forecast", "Analysis", "Observed"]:
                    h = mean_records(q, t, kind, "semantic_mass")[:, j, :]
                    mean = h.mean(0)
                    if len(h) > 1:
                        ax.fill_between(
                            q["semantic_edges"],
                            np.r_[h.min(0), h.min(0)[-1]],
                            np.r_[h.max(0), h.max(0)[-1]],
                            step="post",
                            alpha=0.1,
                            color=PALETTE[kind],
                            lw=0,
                        )
                    ax.stairs(
                        mean, q["semantic_edges"], color=PALETTE[kind], ls=STYLES[kind], lw=1.5
                    )
                title = (TOPICS[d] + EMOTIONS)[j]
                labelled(ax, string.ascii_lowercase[r * 2 + c], title)
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 0.5, 1])
                ax.set_ylim(bottom=0)
                ax.yaxis.set_major_locator(MaxNLocator(3))
                if r == nr - 1:
                    ax.set_xlabel("Mapped score")
                else:
                    ax.tick_params(labelbottom=False)
                if c == 0:
                    ax.set_ylabel("Mass")
            fig.text(0.12 if c == 0 else 0.61, 0.895, NAMES[d], fontweight="bold", fontsize=11)
        save(fig, name, manifest)

    semantic("hssc_semantic_shapes", 1)
    semantic("hssc_semantic_shapes_first", 0)
    semantic("hssc_semantic_shapes_last", 2)
    semantic("hssc_semantic_shapes_all_topics", 1, list(range(5)))
    semantic("hssc_semantic_shapes_all_emotions", 1, list(range(5, 11)))
    for pos, suffix in [(0, "first"), (2, "last")]:
        fig, axs = plt.subplots(2, 3, figsize=(160 / 25.4, 142 / 25.4))
        fig.subplots_adjust(left=0.11, right=0.98, bottom=0.2, top=0.9, wspace=0.3, hspace=0.65)
        vmax = max(
            (
                mean_records(data[d], data[d]["selected_test_targets"][pos], k, "latent_mass")
                .mean(0)
                .max()
                for d in DS
                for k in ["Observed", "Forecast", "Analysis"]
            )
        )
        for i, d in enumerate(DS):
            q = data[d]
            t = q["selected_test_targets"][pos]
            for j, k in enumerate(["Observed", "Forecast", "Analysis"]):
                ax = axs[i, j]
                labelled(ax, string.ascii_lowercase[i * 3 + j], k)
                mesh = ax.pcolormesh(
                    *q["latent_edges"],
                    np.ma.masked_equal(mean_records(q, t, k, "latent_mass").mean(0).T, 0),
                    cmap="Blues",
                    norm=LogNorm(vmin=1e-06, vmax=1),
                    rasterized=True,
                )
                ax.xaxis.set_major_locator(MaxNLocator(3))
                ax.yaxis.set_major_locator(MaxNLocator(3))
                if i == 1:
                    ax.set_xlabel("PC1")
                if j == 0:
                    ax.set_ylabel("PC2")
            fig.text(0.11, 0.955 if i == 0 else 0.55, NAMES[d], fontweight="bold", fontsize=11)
        cb = fig.add_axes([0.22, 0.09, 0.56, 0.028])
        cb.set_label("colorbar")
        fig.colorbar(
            mesh, cax=cb, orientation="horizontal", ticks=[1e-05, 0.001, 0.1], format="%.5g"
        )
        cb.minorticks_off()
        cb.set_xlabel("Probability mass per bin", labelpad=1)
        save(fig, "hssc_latent_comparison_" + suffix, manifest)


def trajectories(manifest):
    import matplotlib.pyplot as plt
    import matplotlib.dates as dates
    from matplotlib.ticker import MaxNLocator, NullLocator

    rec = read(SCI / "reconstruction.json")

    def make(hours, full):
        nr = 5 if full else 6
        fig, axs = plt.subplots(nr, 2, figsize=(160 / 25.4, (52 + nr * 30 if full else 205) / 25.4))
        fig.subplots_adjust(left=0.15, right=0.98, bottom=0.09, top=0.86, wspace=0.48, hspace=0.65)
        for c, d in enumerate(DS):
            actual_hours = HOURS[d][(6, 12, 24).index(hours)] if full else 12
            cal = read(calendar_path(d, actual_hours))["windows"]
            by = {q["window"]: q for q in cal}
            rr = [
                q
                for q in rec
                if (q["dataset"], q["window_hours"], q["group"]) == (d, actual_hours, "HA")
            ]
            obs = {q["window"]: q for q in rr}
            ts = sorted(obs)
            x = [
                datetime.fromisoformat(by[t]["start_utc"])
                + (
                    datetime.fromisoformat(by[t]["end_exclusive_utc"])
                    - datetime.fromisoformat(by[t]["start_utc"])
                )
                / 2
                for t in ts
            ]
            if full:
                for row, j in enumerate([0, 1, 4, 5, 6] if c == 0 else [0, 1, 3, 5, 6]):
                    ax = axs[row, c]
                    ax.plot(x, [obs[t]["observed_scores"][j] for t in ts], c=INK, label="Observed")
                    for kind, color, ls in [("prior", ORANGE, "--"), ("analysis", GREEN, "-.")]:
                        for phase in ["train", "development", "test"]:
                            pp = [q for q in rr if q["phase"] == phase and q["kind"] == kind]
                            ws = sorted({q["window"] for q in pp})
                            xp = [x[ts.index(t)] for t in ws]
                            vv = np.array(
                                [[q["scores"][j] for q in pp if q["window"] == t] for t in ws],
                                dtype=float,
                            )
                            ax.plot(xp, vv.mean(axis=1), c=color, ls=ls)
                            ax.fill_between(
                                xp,
                                vv.min(axis=1),
                                vv.max(axis=1),
                                color=color,
                                alpha=0.15,
                                linewidth=0,
                            )
                    labelled(ax, string.ascii_lowercase[row * 2 + c], (TOPICS[d] + EMOTIONS)[j])
                    ax.set_ylim(bottom=0)
            else:
                axs[0, c].plot(x, [obs[t]["target_count"] for t in ts], c=BLUE)
                axs[0, c].set_yscale("log")
                axs[0, c].yaxis.set_minor_locator(NullLocator())
                axs[0, c].set_yticks([1000, 100000] if c == 0 else [10, 1000])
                axs[0, c].set_yticklabels(["1,000", "100,000"] if c == 0 else ["10", "1,000"])
                labelled(axs[0, c], string.ascii_lowercase[c], "Posting volume")
                selected = [0, 1, 4, 5, 6] if c == 0 else [0, 1, 3, 5, 6]
                for row, (j, color) in enumerate(
                    zip(selected, [BLUE, GREEN, PURPLE, ORANGE, PURPLE]), 1
                ):
                    axs[row, c].plot(x, [obs[t]["observed_scores"][j] for t in ts], c=color)
                    labelled(
                        axs[row, c], string.ascii_lowercase[row * 2 + c], (TOPICS[d] + EMOTIONS)[j]
                    )
                    axs[row, c].set_ylim(bottom=0)
            boundaries = [
                datetime.fromisoformat(next((w["start_utc"] for w in cal if w["phase"] == p)))
                for p in ["development", "test"]
            ]
            for row in range(nr):
                ax = axs[row, c]
                ax.set_xlim(x[0], x[-1])
                ax.set_xticks([x[0], x[len(x) // 2], x[-1]])
                ax.xaxis.set_major_formatter(dates.DateFormatter("%d %b"))
                if full:
                    for b in boundaries:
                        ax.axvline(b, c=GREY, ls=":", lw=1)
                if row < nr - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel("Date (UTC)")
                    tt = ax.get_xticklabels()
                    tt[0].set_ha("left")
                    tt[-1].set_ha("right")
                if full or row:
                    ax.yaxis.set_major_locator(MaxNLocator(3))
                if c == 0:
                    ax.set_ylabel("Score" if full or row else "Posts")
            fig.text(
                0.15 if c == 0 else 0.63,
                0.91 if full else 0.95,
                NAMES[d] + f" · {actual_hours} h",
                fontweight="bold",
                fontsize=11,
            )
        if full:
            legends(fig, ["Observed", "Forecast", "Analysis"], 3)
        save(fig, f"hssc_semantic_trajectory_{hours}h" if full else "hssc_observed", manifest)

    make(12, False)
    for h in [6, 12, 24]:
        make(h, True)
