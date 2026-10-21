import importlib, json, sys
from traser.io import ROOT, write


def doctor():
    import importlib.metadata as md

    deps = {}
    for module, distribution in [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("pyarrow", "pyarrow"),
        ("torch", "torch"),
        ("matplotlib", "matplotlib"),
        ("threadpoolctl", "threadpoolctl"),
    ]:
        try:
            importlib.import_module(module)
            deps[distribution] = md.version(distribution)
        except ImportError as e:
            deps[distribution] = {"missing": str(e)}
    import torch

    modules = [
        "traser.dynamics.operators",
        "traser.assimilation.experiment",
        "traser.assimilation.transport",
        "traser.experiments.chains",
        "traser.experiments.interventions",
        "traser.baselines.statistical",
        "traser.representations.text",
        "traser.representations.moments",
    ]
    for name in modules:
        importlib.import_module(name)
    result = {
        "python": sys.version,
        "dependencies": deps,
        "numerical_imports": "PASS",
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "full_scientific_run": False,
    }
    write(ROOT / "outputs/environment.json", result)
    print(json.dumps(result, indent=2))
    return result
