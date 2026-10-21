"""Complete development selection orchestration, isolated from CLI parsing."""

from traser.io import ROOT
from traser.experiments.replay import Experiment, requirements
from traser.experiments.runtime import exclusive_experiment


@exclusive_experiment
def run(dataset, asset_root=None):
    root = (asset_root or ROOT / "assets").resolve()
    cfg, paths = requirements(dataset, root)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen assets: " + ", ".join(missing))
    for hours in cfg["window_hours"]:
        exp = Experiment(dataset, hours, root)
        exp.out = ROOT / f"outputs/selection/{dataset}/w{hours}"
        exp.out.mkdir(parents=True, exist_ok=True)
        if dataset == "blm_x":
            from traser.experiments.select_blm import selection

            selection(exp, cfg, float("inf"))
        else:
            from traser.experiments.select_weibo import select

            select(exp, cfg)
        exp.clear_phase()
