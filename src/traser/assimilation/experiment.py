"""Current paper implementation; provenance is recorded in provenance/module_map.json."""

from pathlib import Path
import time, itertools, hashlib, json
import numpy as np
from traser.io import ROOT, read, write, sha, stamp
from traser.semantics.diagnostics import Experiment as FrozenExperiment


class Experiment(FrozenExperiment):
    def clouds(self, phase):
        if phase not in self.cloud_cache:
            ts = np.asarray(self.observation_index["window"])
            ix = np.asarray(self.observation_index["coordinate_row"])
            result = {
                t: np.asarray(self.all_coordinates[ix[ts == t]], dtype=np.float64)
                for t in self.phases[phase]
            }
            for t, x in result.items():
                if len(x) != self.windows[t]["eligible_occurrences"] or not np.isfinite(x).all():
                    raise RuntimeError("Incomplete time-resolution target")
            self.cloud_cache[phase] = result
        return self.cloud_cache[phase]

    def semantic(self, x, phase, t):
        if self.windows[t]["eligible_occurrences"]:
            return super().semantic(x, phase, t)
        v, c = self.mapped(x)
        xx = x.double()
        return {
            "scores": v.mean(0).tolist(),
            "observed_scores": [None] * 11,
            "mean_absolute_error": [None] * 11,
            "marginal_W1": [None] * 11,
            "clipped_fraction": c.tolist(),
            "observed_clipped_fraction": [None] * 11,
            "rms": float(((xx - xx.mean(0)) ** 2).sum(1).mean().sqrt()),
            "target_count": 0,
            "observation_available": False,
            "full_target_mass": True,
        }

    def adaptive_update(self, prior, phase, t, spec):
        from traser.assimilation.transport import effective_strength, transport_update

        n = self.windows[t]["eligible_occurrences"]
        if n == 0:
            return (prior, {"skipped": "empty_observation", "window": t, "strength": None})
        discrepancy = self.score(prior, phase, t)["scaled_sliced_w2"]
        strength = effective_strength(spec, n, discrepancy, self.references)
        if spec["family"] == "MMD":
            old = self.cfg
            self.cfg = {
                **old,
                "assimilation": {
                    **old["assimilation"],
                    "steps": spec["steps"],
                    "learning_rate": spec["learning_rate"],
                },
            }
            try:
                x, diag = super().update(prior, phase, t, strength)
            finally:
                self.cfg = old
        else:
            target = self.analysis_distance.prepare(self.clouds(phase)[t], (phase, t))
            x, diag = transport_update(
                prior, target, self.analysis_distance, strength, spec["steps"]
            )
        diag.update(
            window=t,
            observed_count=n,
            prior_observation_discrepancy=discrepancy,
            base_strength=spec["strength"],
            count_power=spec.get("count_power", 0),
            innovation_response=spec.get("innovation_response", 0),
            future_target_used=False,
        )
        return (x, diag)

    def clear_phase(self):
        self.cloud_cache.clear()
        self.target_cache.clear()
        self.gpu_observation_cache.clear()
        self.semantic_cache.clear()
        self.additional_target_cache.clear()
        for distance in (
            self.distance,
            self.sensitivity_distance,
            self.marginal_distance,
            self.analysis_distance,
        ):
            distance.cache.clear()
        self.torch.cuda.empty_cache()


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]
