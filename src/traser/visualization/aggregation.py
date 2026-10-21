"""Current paper implementation; provenance is recorded in provenance/module_map.json."""

from collections import defaultdict
import numpy as np


def grouped_values(rows, dataset, group, field, indices, kind=None):
    pools = defaultdict(list)
    for r in rows:
        if r["dataset"] != dataset or r["window_hours"] != 12 or r["group"] != group:
            continue
        if kind is not None and (r["kind"] != kind or r["target_count"] <= 0):
            continue
        key = (r["phase"], r["window"] if kind else r["origin"])
        pools[key].append(float(np.mean(np.asarray(r[field])[indices])))
    assert pools and all((len(v) == 3 for v in pools.values()))
    return np.array([np.mean(pools[k]) for k in sorted(pools)])
