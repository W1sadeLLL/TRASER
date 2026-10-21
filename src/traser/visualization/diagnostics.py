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


def adapt(function, replacements, namespace, name):
    source = inspect.getsource(function)
    for old, new in replacements:
        assert old in source, (name, old)
        source = source.replace(old, new)
    directory = OUT / "plot_sources"
    directory.mkdir(exist_ok=True)
    path = directory / (name + ".py")
    path.write_text(source, encoding="utf-8")
    scope = dict(function.__globals__)
    scope.update(namespace)
    exec(compile(source, str(path), "exec"), scope)
    return scope[function.__name__]


def updating(u, save):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    stats = read(SCI / "paired_statistics.json")
    fig = plt.figure(figsize=(180 / 25.4, 150 / 25.4))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.12,
        right=0.98,
        bottom=0.105,
        top=0.8,
        hspace=0.8,
        wspace=0.33,
        height_ratios=[1.35, 1],
    )
    for c, dataset in enumerate(DS):
        ax = fig.add_subplot(gs[0, c])
        u.panel(ax, "ab"[c], u.NAMES[dataset])
        for y, hours in enumerate(HOURS[dataset]):
            q = next(
                (
                    r
                    for r in stats
                    if (r["dataset"], r["window_hours"], r["reference"], r["metric"])
                    == (dataset, hours, "H0", "scaled_sliced_w2")
                )
            )
            values = np.asarray(list(q["origins"].values()))
            colors = [
                u.GREEN if v < -1e-12 else u.GREY if abs(v) <= 1e-12 else u.ORANGE for v in values
            ]
            ax.scatter(
                values,
                y + np.linspace(-0.14, 0.08, len(values)),
                c=colors,
                s=12,
                alpha=0.65,
                zorder=4,
            )
            if q["interval_95"]:
                ax.plot(q["interval_95"], [y + 0.24] * 2, color=u.INK)
            ax.scatter(q["mean_difference"], y + 0.24, color=u.INK, marker="D", s=28, zorder=6)
            ax.text(
                0.01,
                y + 0.46,
                f"n={len(values)}, B={q['blocks']}",
                transform=ax.get_yaxis_transform(),
                va="top",
            )
        ax.axvline(0, color=u.GREY, ls="--")
        ax.set_yticks(range(3), [f"{h} h" for h in HOURS[dataset]])
        ax.set_ylim(2.9, -0.45)
        ax.set_xlabel("Forecast-error difference")
        ax.xaxis.set_major_locator(MaxNLocator(3))
    ax = fig.add_subplot(gs[1, :])
    q = next(
        (
            r
            for r in stats
            if (r["dataset"], r["window_hours"], r["reference"], r["metric"])
            == ("hk_weibo", 12, "H0", "scaled_sliced_w2")
        )
    )
    origins = sorted(map(int, q["leave_one_out"]))
    values = [q["leave_one_out"][str(t)] for t in origins]
    ax.axhline(0, color=u.GREY, ls="--")
    ax.axhline(q["mean_difference"], color=u.INK, ls=":")
    ax.scatter(origins, values, c=[u.ORANGE if v > 0 else u.GREEN for v in values], s=20)
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.set_xlabel("Omitted origin")
    ax.set_ylabel("Mean difference")
    u.panel(ax, "c", "Weibo · 12 h sensitivity")
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=u.GREEN, label="Lower"),
            Line2D([], [], marker="o", ls="", color=u.GREY, label="Equal"),
            Line2D([], [], marker="o", ls="", color=u.ORANGE, label="Higher"),
            Line2D([], [], marker="D", color=u.INK, label="Mean / interval"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    save(fig, "hssc_updating")


def diagnostics(u, save):
    import matplotlib.pyplot as plt
    import matplotlib.dates as dates
    from matplotlib.ticker import MaxNLocator, NullLocator

    updates = read(SCI / "all_updates.json")
    stats = read(SCI / "paired_statistics.json")
    fig, axs = plt.subplots(2, 1, figsize=(160 / 25.4, 109 / 25.4))
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.13, top=0.9, hspace=0.7)
    for c, dataset in enumerate(DS):
        ax = axs[c]
        xs, ratios = ([], [])
        for k, hours in enumerate(HOURS[dataset]):
            cal = {w["window"]: w for w in read(calendar_path(dataset, hours))["windows"]}
            current = [
                r
                for r in updates
                if (r["dataset"], r["window_hours"], r["phase"], r["group"])
                == (dataset, hours, "test", "HA")
            ]
            windows = sorted({r["window"] for r in current})
            x = [datetime.fromisoformat(cal[t]["start_utc"]) for t in windows]
            values = [
                [
                    r["strength"] / r["base_strength"]
                    for r in current
                    if r["window"] == t and r.get("strength") is not None
                ]
                for t in windows
            ]
            mean = [np.mean(v) if v else np.nan for v in values]
            ratios.extend((z for v in values for z in v if np.isfinite(z) and z > 0))
            low, high = [[func(v) if v else np.nan for v in values] for func in (min, max)]
            ax.plot(
                x,
                mean,
                color=[u.BLUE, u.GREEN, u.PURPLE][k],
                ls=["-", "--", "-."][k],
                label=f"{hours} h",
            )
            ax.fill_between(x, low, high, color=[u.BLUE, u.GREEN, u.PURPLE][k], alpha=0.12, lw=0)
            xs.extend(x)
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(NullLocator())
        low, high = (min(ratios), max(ratios))
        ax.set_ylim(low * 0.8, high * 1.25)
        ticks = [v for v in (0.03, 0.1, 0.3, 1, 3, 10, 30) if low * 0.8 <= v <= high * 1.25]
        ax.set_yticks(ticks, [f"{v:g}" for v in ticks])
        ax.axhline(1, color=u.GREY)
        ax.set_ylabel("Regularisation ratio")
        ax.set_xticks([min(xs), min(xs) + (max(xs) - min(xs)) / 2, max(xs)])
        ax.xaxis.set_major_formatter(dates.DateFormatter("%d %b"))
        u.panel(ax, "ab"[c], u.NAMES[dataset])
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1, 1.02),
            borderaxespad=0,
            frameon=False,
            ncol=3,
            handlelength=1.2,
            columnspacing=0.8,
        )
    axs[1].set_xlabel("Date (UTC)")
    save(fig, "hssc_update_parameters")
    fig, axs = plt.subplots(3, 2, figsize=(160 / 25.4, 137 / 25.4))
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.11, top=0.93, wspace=0.31, hspace=0.6)
    for c, dataset in enumerate(DS):
        for k, hours in enumerate(HOURS[dataset]):
            q = next(
                (
                    r
                    for r in stats
                    if (r["dataset"], r["window_hours"], r["reference"], r["metric"])
                    == (dataset, hours, "H0", "scaled_sliced_w2")
                )
            )
            x, y = (list(map(int, q["origins"])), list(q["origins"].values()))
            ax = axs[k, c]
            ax.bar(x, y, color=[u.ORANGE if v > 0 else u.GREEN for v in y])
            ax.axhline(0, color=u.GREY)
            ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            u.panel(ax, "abcdef"[k * 2 + c], f"{u.NAMES[dataset]} · {hours} h")
            if c == 0:
                ax.set_ylabel("Error difference")
            if k == 2:
                ax.set_xlabel("Issue window")
    save(fig, "hssc_origin_diagnostics")
