"""Current paper implementation; provenance is recorded in provenance/module_map.json."""

from pathlib import Path
import os, time, json, hashlib, itertools, math
import numpy as np
from traser.io import ROOT, read, write, sha, stamp
import traser.dynamics.base as parent


class Experiment(parent.Experiment):
    def mapped(self, x):
        torch = self.torch
        m = self.mapper
        x = torch.as_tensor(x, device="cuda", dtype=torch.float64)
        raw = (x - m["mean"]) / m["sd"] @ m["coef"] + m["intercept"]
        if not bool(torch.isfinite(raw).all()):
            raise RuntimeError("Nonfinite readout")
        return (raw.clamp(0, 1), ((raw < 0) | (raw > 1)).double().mean(0))

    def semantic_target(self, phase, t):
        key = (phase, t)
        if key not in self.semantic_cache:
            y = self.clouds(phase)[t]
            chunks = []
            clip = []
            for a in range(0, len(y), 65536):
                v, c = self.mapped(y[a : a + 65536])
                chunks.append(v)
                clip.append(c * len(v))
            v = self.torch.cat(chunks)
            mean = v.mean(0)
            target = {
                "sorted": v.T.contiguous().sort(1).values,
                "mean": mean,
                "clipped_fraction": self.torch.stack(clip).sum(0) / len(v),
                "count": len(v),
            }
            self.semantic_cache[key] = target
        return self.semantic_cache[key]

    def semantic(self, x, phase, t):
        torch = self.torch
        target = self.semantic_target(phase, t)
        v, clip = self.mapped(x)
        xs = v.T.contiguous().sort(1).values
        ys = target["sorted"]
        w1 = []
        for j in range(len(self.axes)):
            a, b = (xs[j], ys[j])
            merged = torch.cat((a, b)).sort().values
            gap = (
                torch.searchsorted(a, merged, right=True).double() / len(a)
                - torch.searchsorted(b, merged, right=True).double() / len(b)
            ).abs()
            w1.append(float(((merged[1:] - merged[:-1]) * gap[:-1]).sum()))
        mean = v.mean(0)
        xt = torch.as_tensor(x, device="cuda", dtype=torch.float64)
        return {
            "scores": mean.tolist(),
            "observed_scores": target["mean"].tolist(),
            "mean_absolute_error": (mean - target["mean"]).abs().tolist(),
            "marginal_W1": w1,
            "clipped_fraction": clip.tolist(),
            "observed_clipped_fraction": target["clipped_fraction"].tolist(),
            "rms": float((xt - xt.mean(0)).square().sum(1).mean().sqrt()),
            "target_count": target["count"],
            "full_target_mass": True,
        }

    def metrics(self, prediction, target_window, scored):
        result = super().metrics(prediction, target_window, scored)
        torch = self.torch
        if not hasattr(self, "additional_target_cache"):
            self.additional_target_cache = {}
        if target_window not in self.additional_target_cache:
            y = torch.as_tensor(
                self.clouds("test")[target_window], device="cuda", dtype=torch.float64
            )
            yc = y - y.mean(0)
            self.additional_target_cache[target_window] = (
                y.T.contiguous().sort(1).values,
                yc.T @ yc / len(y),
            )
        ys, ycov = self.additional_target_cache[target_window]
        x = torch.as_tensor(prediction, device="cuda", dtype=torch.float64)
        xc = x - x.mean(0)
        xs = x.T.contiguous().sort(1).values
        w1 = []
        ks = []
        for j in range(64):
            a, b = (xs[j], ys[j])
            merged = torch.cat((a, b)).sort().values
            gap = (
                torch.searchsorted(a, merged, right=True).double() / len(a)
                - torch.searchsorted(b, merged, right=True).double() / len(b)
            ).abs()
            w1.append(float(((merged[1:] - merged[:-1]) * gap[:-1]).sum()))
            ks.append(float(gap.max()))
        result["metrics"].update(
            marginal_W1_mean=float(np.mean(w1)),
            marginal_KS_mean=float(np.mean(ks)),
            covariance_Frobenius=float(torch.linalg.norm(xc.T @ xc / len(x) - ycov)),
        )
        return result
