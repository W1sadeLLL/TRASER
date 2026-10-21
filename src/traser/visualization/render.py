"""Re-render current quantitative panels from complete archived arrays only."""

from pathlib import Path
import gzip, json, shutil, os, sys
import numpy as np
from traser.io import ROOT, write, sha

SOURCE = ROOT / "evidence/source_data"
OUT = ROOT / "outputs/figures"
MANIFEST = []


def read(path):
    path = Path(path)
    if not path.exists():
        path = SOURCE / path.name
    if path.suffix == ".json" and (not path.exists()) and path.with_suffix(".json.gz").exists():
        with gzip.open(path.with_suffix(".json.gz"), "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def calendar_path(dataset, hours):
    return ROOT / f"assets/{dataset}/w{hours}/calendar.json"


def setup():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1,
            "lines.linewidth": 1.5,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "xtick.minor.width": 1,
            "ytick.minor.width": 1,
        }
    )


def save(fig, name, exclude=None, **alignment):
    from matplotlib.text import Text
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt

    for t in fig.findobj(Text):
        t.set_fontfamily("DejaVu Sans")
        t.set_fontsize(11)
    for line in fig.findobj(Line2D):
        line.set_linewidth(max(1, line.get_linewidth()))
        line.set_markeredgewidth(max(1, line.get_markeredgewidth()))
    for patch in fig.findobj(Patch):
        patch.set_linewidth(max(1, patch.get_linewidth()))
    if name == "hssc_observed":
        for ax in fig.axes:
            if ax.get_ylabel() == "Posts":
                ax.yaxis.labelpad = 0
    if name in [
        "hssc_latent_observed",
        "hssc_latent_comparison_first",
        "hssc_latent_comparison_last",
    ]:
        for i, ax in enumerate(fig.axes[:6]):
            pos = ax.get_position()
            ax.set_position([0.11 + 0.325 * (i % 3), pos.y0, 0.23, pos.height])
    fig.canvas.draw()
    if os.environ.get("TRASER_FIGURE_AUDIT_PATH"):
        sys.path.insert(0, os.environ["TRASER_FIGURE_AUDIT_PATH"])
        from audit_panel_alignment import require_matplotlib_panel_alignment

        require_matplotlib_panel_alignment(
            fig,
            json_out=OUT / (name + ".alignment.json"),
            exclude_axes=exclude or [],
            strict=True,
            **alignment,
        )
    rectangles = [
        {"label": ax.get_label(), "bounds": list(ax.get_position().bounds)} for ax in fig.axes
    ]
    write(
        OUT / (name + ".layout.json"),
        {
            "width_inches": fig.get_figwidth(),
            "height_inches": fig.get_figheight(),
            "axes": rectangles,
            "alignment_exemptions": alignment,
            "font_pt": 11,
        },
    )
    for ext in ["pdf", "svg"]:
        fig.savefig(OUT / (name + "." + ext), dpi=450, facecolor="white")
    MANIFEST.append(
        {
            "name": name,
            "pdf_sha256": sha(OUT / (name + ".pdf")),
            "font_pt": 11,
            "source": "frozen complete array archive; no new model computation",
            "font_substitution": "DejaVu Sans for portable release; archived reference uses TeX Gyre Heros",
        }
    )
    print("Rendered " + name, flush=True)
    plt.close(fig)


def render():
    import traser.visualization.panels as u
    import traser.visualization.layout as c
    import traser.visualization.distributions as o
    import traser.visualization.diagnostics as r
    import traser.visualization.style as f
    import traser.visualization.references as ref

    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.clear()
    for m in [u, c, o, r, f, ref]:
        m.OUT = OUT
        m.SCI = SOURCE
        m.read = read
    u.setup = setup
    u.save = save
    f.BASE_SAVE = save
    c.ORIGINAL_SAVE = f.finish
    for m in [u, o]:
        for k, v in f.PALETTE.items():
            setattr(m, k, v)
    u.COLORS = {"Observed": u.INK, "H0": u.GREY, "Forecast": u.ORANGE, "Analysis": u.GREEN}
    o.PALETTE = dict(u.COLORS)
    c.REV = c.PREV = u.REV = OUT
    setup()
    u.save = c.compact_save
    c.observed()
    c.latent()
    u.ecdf()
    u.semantic()
    r.updating(u, c.compact_save)
    u.scenarios()
    u.supplement()
    r.diagnostics(u, c.compact_save)
    u.save = f.finish
    ref.render()
    shutil.copy2(ROOT / "evidence/figures/Figure_1.pdf", OUT / "Figure_1.pdf")
    write(
        OUT / "manifest.json",
        {
            "quantitative_figures": MANIFEST,
            "framework": "exact existing Figure_1.pdf copied, not regenerated",
            "scientific_models_executed": False,
        },
    )
    assert len(MANIFEST) == 21, len(MANIFEST)
    return MANIFEST
