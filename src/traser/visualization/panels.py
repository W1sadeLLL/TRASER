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
INK, GREY, ORANGE, GREEN = ("#263445", "#8B96A5", "#C47A3A", "#168577")
BLUE, PURPLE, ROSE = ("#406D99", "#8265A7", "#AD5D81")
COLORS = {"Observed": INK, "H0": GREY, "Forecast": ORANGE, "Analysis": GREEN}
STYLES = {"Observed": "-", "H0": ":", "Forecast": "--", "Analysis": "-."}
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


def panel(ax, letter, title, color=INK):
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", pad=10, color=color)


def semantic():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    from traser.visualization.aggregation import grouped_values

    reconstruction = read(SCI / "reconstruction.json")
    with (SCI / "semantic_forecasts.csv").open(encoding="utf-8", newline="") as stream:
        forecasts = list(csv.DictReader(stream))
    for r in forecasts:
        r["window_hours"], r["origin"] = (int(r["window_hours"]), int(r["origin"]))
        for k in ("mean_absolute_error", "marginal_W1"):
            r[k] = json.loads(r[k])
    fig, axs = plt.subplots(2, 2, figsize=(180 / 25.4, 200 / 25.4))
    fig.subplots_adjust(left=0.18, right=0.98, top=0.83, bottom=0.09, hspace=0.82, wspace=0.73)
    exported = []
    for col, d in enumerate(DS):
        ax = axs[0, col]
        panel(ax, "ab"[col], NAMES[d])
        for k, indices in enumerate((slice(0, 5), slice(5, 11))):
            prior = grouped_values(reconstruction, d, "HA", "mean_absolute_error", indices, "prior")
            analysis = grouped_values(
                reconstruction, d, "HA", "mean_absolute_error", indices, "analysis"
            )
            assert len(prior) == len(analysis) == (25 if col == 0 else 71)
            offsets = np.linspace(-0.18, 0.18, len(prior))
            for j, (p, a) in enumerate(zip(prior, analysis)):
                ax.plot([p, a], [k + offsets[j]] * 2, color=GREY, alpha=0.22, lw=1)
            ax.scatter(prior, k + offsets, color=ORANGE, s=7, alpha=0.35)
            ax.scatter(analysis, k + offsets, color=GREEN, s=7, alpha=0.35)
            ax.plot([prior.mean(), analysis.mean()], [k, k], color=INK, lw=2)
            ax.scatter(prior.mean(), k, color=ORANGE, marker="o", s=44, zorder=5)
            ax.scatter(analysis.mean(), k, color=GREEN, marker="D", s=40, zorder=5)
            exported.append(
                {
                    "dataset": d,
                    "task": "reconstruction",
                    "family": k,
                    "prior": prior.tolist(),
                    "analysis": analysis.tolist(),
                }
            )
        ax.set_yticks([0, 1], ["Topics", "Emotions"])
        ax.set_ylim(1.45, -0.45)
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.set_xlabel("Mean-score error")
        ax.text(0, 1.23, f"n = {len(prior)} windows", transform=ax.transAxes)
        ax = axs[1, col]
        panel(ax, "cd"[col], NAMES[d])
        ax.axvline(0, color=INK, ls="--", lw=1)
        groups = [
            ("mean_absolute_error", slice(0, 5)),
            ("mean_absolute_error", slice(5, 11)),
            ("marginal_W1", slice(0, 5)),
            ("marginal_W1", slice(5, 11)),
        ]
        for k, (field, indices) in enumerate(groups):
            pure = grouped_values(forecasts, d, "H0", field, indices)
            pred = grouped_values(forecasts, d, "HA", field, indices)
            delta = pred - pure
            assert len(delta) == (5 if col == 0 else 63)
            ax.scatter(
                delta,
                k + np.linspace(-0.15, 0.15, len(delta)),
                color=ORANGE,
                alpha=0.38,
                s=19,
                zorder=3,
            )
            avg = delta.mean()
            ax.plot([delta.min(), delta.max()], [k, k], color=ORANGE, lw=1.3)
            ax.scatter(avg, k, color=ORANGE, marker="D", s=40, zorder=5)
            exported.append(
                {
                    "dataset": d,
                    "task": "forecast",
                    "metric": field,
                    "family": indices.start,
                    "pure": pure.tolist(),
                    "forecast": pred.tolist(),
                    "difference": delta.tolist(),
                    "mean_difference": float(avg),
                }
            )
        ax.set_yticks(range(4), ["Topic mean", "Emotion mean", "Topic W1", "Emotion W1"])
        ax.set_ylim(3.5, -0.5)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.set_xlabel("Forecast-error difference")
        ax.text(0, 1.23, f"n = {len(delta)} origins", transform=ax.transAxes)
    fig.text(0.18, 0.975, "Same-window reconstruction", fontweight="bold")
    fig.legend(
        handles=[
            Line2D([], [], color=ORANGE, marker="o", label="Prior"),
            Line2D([], [], color=GREEN, marker="D", label="Analysis"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.58, 0.955),
        ncol=2,
        frameon=False,
    )
    fig.text(0.18, 0.49, "Next-window prediction", fontweight="bold")
    fig.text(0.18, 0.455, "Difference = with assimilation − pure BCM")
    write(OUT / "semantic_pairs.json", exported)
    save(fig, "hssc_semantic")


def ecdf():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    data = read(OUT / "exact_distributions.json")
    fig, axs = plt.subplots(5, 2, figsize=(180 / 25.4, 215 / 25.4))
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.07, top=0.82, wspace=0.28, hspace=0.84)
    fig.legend(
        handles=[
            Line2D([], [], color=COLORS[k], ls=STYLES[k], label={"H0": "Pure BCM"}.get(k, k))
            for k in ("Observed", "H0", "Forecast", "Analysis")
        ],
        loc="upper center",
        bbox_to_anchor=(0.55, 1),
        ncol=2,
        frameon=False,
    )
    for col, d in enumerate(DS):
        q = data[d]
        fig.text(0.1 if col == 0 else 0.59, 0.9, NAMES[d], fontweight="bold")
        fig.text(
            0.1 if col == 0 else 0.59, 0.865, f"n = {q['observed_n']:,}; N = {q['particles']:,}"
        )
        for row, j in enumerate([0, 1, 4, 5, 6] if col == 0 else [0, 1, 3, 5, 6]):
            ax = axs[row, col]
            panel(ax, string.ascii_lowercase[row * 2 + col], (TOPICS[d] + EMOTIONS)[j])
            for k in ("H0", "Forecast", "Analysis", "Observed"):
                v = q["ecdf"][k][j]
                x = np.asarray(v["x"])
                avg = np.asarray(v["mean"])
                if k != "Observed":
                    ax.fill_between(
                        x,
                        v["min"],
                        v["max"],
                        step="post",
                        color=COLORS[k],
                        alpha=0.12,
                        lw=0,
                        rasterized=True,
                    )
                ax.step(
                    np.r_[0, x],
                    np.r_[0, avg],
                    where="post",
                    color=COLORS[k],
                    ls=STYLES[k],
                    lw=1.5,
                    rasterized=True,
                )
            v = q["ecdf"]["Observed"][j]
            p1 = "0" if v["p1"] == 0 else f"{v['p1']:.1g}"
            ax.text(
                0.97,
                0.08,
                f"P(0) {v['p0']:.3f}; P(1) {p1}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.035)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            if row < 4:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Mapped score")
            if col == 0:
                ax.set_ylabel("ECDF")
    save(fig, "hssc_semantic_shapes")


def scenarios():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    rows = read(SCI / "interventions/w12/summary.json")
    fig, axs = plt.subplots(3, 2, figsize=(180 / 25.4, 214 / 25.4))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.25, top=0.88, wspace=0.6, hspace=0.68)
    conditions = ["expression_injection", "reduce_reach", "silence_sources", "redirect_influence"]
    colors = [ORANGE, BLUE, PURPLE, GREEN]
    markers = ["o", "o", "s", "D"]
    matrices = {}
    exported = []
    for col, d in enumerate(DS):
        for k, condition in enumerate(conditions):
            ax = axs[0 if k == 0 else 1, col]
            for j in range(5):
                q = next(
                    (
                        v
                        for v in rows
                        if (v["dataset"], v["condition"], v["topic"], v["fraction"])
                        == (d, condition, f"C0{j + 1}", 0.1)
                    )
                )
                factor = 1 if k == 0 else 1000
                y = j + (0 if k == 0 else (k - 2) * 0.21)
                ax.plot(
                    [q["delta_min"][j] * factor, q["delta_max"][j] * factor],
                    [y, y],
                    color=colors[k],
                    lw=1.5,
                )
                ax.scatter(
                    q["delta_mean"][j] * factor,
                    y,
                    color=colors[k],
                    marker=markers[k],
                    s=25,
                    zorder=4,
                )
                exported.append(q)
        for row, title in [(0, "Injection"), (1, "Influence changes")]:
            ax = axs[row, col]
            panel(ax, "abcd"[row * 2 + col], title)
            ax.set_yticks(range(5), [f"T{j + 1}" for j in range(5)])
            ax.set_ylim(4.6, -0.6)
            ax.axvline(0, color=GREY, ls="--", lw=1)
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.set_xlabel("Target-score change" if row == 0 else "Change × 1,000")
        matrix = np.array(
            [
                next(
                    (
                        v
                        for v in rows
                        if (v["dataset"], v["condition"], v["topic"], v["fraction"])
                        == (d, "expression_injection", f"C0{j + 1}", 0.1)
                    )
                )["delta_mean"]
                for j in range(5)
            ]
        )
        for j in range(5):
            matrix[j, j] = np.nan
        matrices[d] = matrix
        fig.text(0.18 if col == 0 else 0.67, 0.94, NAMES[d], fontweight="bold")
    limit = max((float(np.nanmax(abs(m))) for m in matrices.values()))
    cmap = LinearSegmentedColormap.from_list("score_changes", [ORANGE, "#FAFAF8", GREEN])
    for col, d in enumerate(DS):
        ax = axs[2, col]
        panel(ax, "ef"[col], "Other-axis changes")
        im = ax.imshow(
            np.ma.masked_invalid(matrices[d]),
            cmap=cmap,
            norm=TwoSlopeNorm(0, -limit, limit),
            aspect="auto",
        )
        for j in range(5):
            ax.text(j, j, "×", ha="center", va="center", color=GREY)
        ax.set_yticks(range(5), [f"T{j + 1}" for j in range(5)])
        ax.set_xticks(
            range(11),
            [f"T{j + 1}" for j in range(5)] + EMOTIONS,
            rotation=90,
            rotation_mode="anchor",
            ha="right",
            va="center",
        )
        ax.tick_params(axis="x", pad=7)
        if col == 0:
            ax.set_ylabel("Injected topic")
    fig.legend(
        handles=[
            Line2D([], [], color=colors[k], marker=markers[k], ls="", label=label)
            for k, label in zip((1, 2, 3), ("Reduce", "Silence", "Redirect"))
        ],
        loc="upper center",
        bbox_to_anchor=(0.58, 1.0),
        ncol=3,
        frameon=False,
        columnspacing=1,
    )
    cb = fig.add_axes([0.3, 0.07, 0.52, 0.017])
    cb.set_label("colorbar")
    fig.colorbar(im, cax=cb, orientation="horizontal", ticks=[-limit, 0, limit], format="%.3f")
    cb.set_xlabel("Scenario − matched control", labelpad=4)
    write(
        OUT / "scenario_source.json",
        {
            "records": exported,
            "off_target_color_limit": limit,
            "masked_diagonal": "Target values are shown in panels a/b; no data dropped from source.",
            "topic_names": TOPICS,
        },
    )
    save(fig, "hssc_interventions", exclude=[cb])


def supplement(names=None):
    """Use the same state colours for the retained supplementary displays."""
    import traser.visualization.distributions as original
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    active = set(
        [
            "hssc_injection_all_blm_x",
            "hssc_injection_all_hk_weibo",
            "hssc_latent_comparison_first",
            "hssc_latent_comparison_last",
            "hssc_latent_observed",
            "hssc_origin_diagnostics",
            "hssc_semantic_shapes_all_emotions",
            "hssc_semantic_shapes_all_topics",
            "hssc_semantic_shapes_first",
            "hssc_semantic_shapes_last",
            "hssc_semantic_trajectory_12h",
            "hssc_semantic_trajectory_24h",
            "hssc_semantic_trajectory_6h",
            "hssc_update_parameters",
        ]
    )
    if names:
        active &= set(names)

    def staged_save(fig, name, manifest):
        if name in active:
            if name == "hssc_latent_observed":
                fig.set_size_inches(160 / 25.4, 160 / 25.4)
                fig.subplots_adjust(bottom=0.28)
                fig.axes[-1].set_position([0.22, 0.1, 0.56, 0.028])
                for text in fig.texts:
                    if text.get_text() == "Weibo":
                        text.set_y(0.59)
            save(fig, name, exclude=[a for a in fig.axes if a.get_label() == "colorbar"])
        else:
            plt.close(fig)

    original.save = staged_save
    original.PALETTE = COLORS
    original.INK, original.GREY, original.ORANGE, original.GREEN = (INK, GREY, ORANGE, GREEN)
    original.shapes([])
    original.trajectories([])
    rows = read(SCI / "interventions/w12/summary.json")
    matrices = {
        d: np.array(
            [
                next(
                    (
                        v
                        for v in rows
                        if (v["dataset"], v["condition"], v["topic"], v["fraction"])
                        == (d, "expression_injection", f"C0{j + 1}", 0.1)
                    )
                )["delta_mean"]
                for j in range(5)
            ]
        ).T
        for d in DS
    }
    limit = max((float(abs(m).max()) for m in matrices.values()))
    cmap = LinearSegmentedColormap.from_list("signed_effect", [ORANGE, "#FFFFFF", GREEN])
    for d in DS:
        if "hssc_injection_all_" + d not in active:
            continue
        fig, ax = plt.subplots(figsize=(160 / 25.4, 147 / 25.4))
        fig.subplots_adjust(left=0.28, right=0.94, bottom=0.27, top=0.9)
        mesh = ax.imshow(matrices[d], aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
        ax.set_yticks(range(11), original.TOPICS[d] + original.EMOTIONS)
        ax.set_xticks(
            range(5),
            [
                x.replace("Counter-narratives", "Counter\nnarratives").replace(" ", "\n")
                for x in original.TOPICS[d]
            ],
        )
        ax.set_xlabel("Injected topic")
        ax.set_title(NAMES[d] + " · all semantic changes", fontweight="bold", pad=12)
        cb = fig.add_axes([0.36, 0.075, 0.48, 0.025])
        fig.colorbar(
            mesh, cax=cb, orientation="horizontal", ticks=[-limit, 0, limit], format="%.2f"
        )
        cb.set_xlabel("Mean score change", labelpad=3)
        save(fig, "hssc_injection_all_" + d, exclude=[cb])
