"""Current paper implementation; provenance is recorded in provenance/module_map.json."""

from pathlib import Path
import os, time, json, hashlib, itertools, math
import numpy as np
from traser.io import ROOT, read, write, sha, stamp


def atomic_torch(value, path):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def atomic_array(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("wb") as file:
        np.save(file, value)
    os.replace(temp, path)


def seal(path, identity):
    if path.exists():
        if read(path) != identity:
            raise RuntimeError(f"Frozen identity differs; preserve existing run: {path}")
    else:
        write(path, identity)


class Experiment:
    def initial(self, phase, seed, origin=None):
        from traser.dynamics.seeding import _initial

        t = self.phases[phase][0] if origin is None else origin
        return self.torch.tensor(
            _initial(self.clouds(phase)[t], self.budget, self.dataset, t, seed),
            device="cuda",
            dtype=self.torch.float32,
        )

    def innovation(self, phase, seed, origin, step, epoch=0):
        code = [
            20261013,
            1 if self.dataset == "hk_weibo" else 2,
            {"train": 1, "development": 2, "test": 3}[phase],
            seed,
            origin,
            step,
            epoch,
        ]
        value = np.random.default_rng(np.random.SeedSequence(code)).normal(size=(self.budget, 64))
        return self.torch.tensor(value, dtype=self.torch.float32, device="cuda")

    def transition(self, model, x, previous, bcm, noise, residual=True):
        from traser.dynamics.operators import bcm_step

        self.guard()
        y = bcm_step(x, bcm["alpha"], bcm["radius"], self.cfg["resources"]["bcm_row_block"])
        if residual:
            r, energy, diagnostic = model(x, y, previous)
        else:
            r, energy, diagnostic = (0.0, x.new_zeros(()), {})
        result = y + r + bcm["sigma"] * noise
        if not bool(self.torch.isfinite(result).all()):
            raise RuntimeError("Nonfinite prediction")
        return (result, energy, diagnostic)

    def update(self, prior, phase, target_window, strength):
        from traser.dynamics.operators import mmd_update

        self.guard()
        key = (phase, target_window)
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        cold = key not in self.gpu_observation_cache
        if cold:
            observed = self.clouds(phase)[target_window]
            unique, counts = np.unique(np.asarray(observed, np.float32), axis=0, return_counts=True)
            target = {
                "points": self.torch.as_tensor(unique, device="cuda"),
                "mass": self.torch.as_tensor(
                    counts / len(observed), device="cuda", dtype=self.torch.float32
                ),
                "count": len(observed),
            }
            self.torch.cuda.synchronize()
            target["prepare_seconds"] = time.perf_counter() - started
            self.gpu_observation_cache[key] = target
        target = self.gpu_observation_cache[key]
        with self.torch.enable_grad():
            state, diagnostic = mmd_update(
                prior, target, strength, self.features["bandwidth"], self.cfg["assimilation"]
            )
        self.torch.cuda.synchronize()
        diagnostic["actual_seconds"] = time.perf_counter() - started
        diagnostic["target_prepare_seconds"] = target["prepare_seconds"]
        diagnostic["cold_equivalent_seconds"] = diagnostic["actual_seconds"] + (
            0 if cold else target["prepare_seconds"]
        )
        return (state, diagnostic)

    def score(self, prediction, phase, target_window):
        with self.torch.no_grad():
            return self.distance.score(
                prediction, self.clouds(phase)[target_window], (phase, target_window)
            )

    def reference_candidates(self):
        yield ("WAR", {})
        for rank in self.cfg["references"]["WR"]:
            yield ("WR", rank)
        grid = self.cfg["references"]["LQD"]
        for bandwidth, rank, ridge in itertools.product(
            grid["bandwidths"], grid["ranks"], grid["ridges"]
        ):
            yield ("LQD", {"bandwidth": bandwidth, "rank": rank, "ridge": ridge})

    def reference_forecasts(self, directory, model, name, phase, horizons, seed):
        import traser.baselines.statistical as baselines

        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "complete.json").exists():
            return read(directory / "complete.json")
        rows_file = directory / "pairs.json"
        rows = read(rows_file) if rows_file.exists() else []
        seen = {(r["origin"], r["horizon"]) for r in rows}
        predict = getattr(baselines, "predict_" + name.lower())
        windows = self.phases[phase]
        for t in windows[:-1]:
            initial = self.initial(phase, seed, t).cpu().numpy()
            for h in horizons:
                if t + h not in windows or (t, h) in seen:
                    continue
                predicted = predict(model, initial, h)
                filename = f"origin_{t}_h_{h}.npy"
                atomic_array(predicted, directory / filename)
                tensor = self.torch.as_tensor(predicted, device="cuda", dtype=self.torch.float64)
                record = self.score(tensor, phase, t + h)
                rows.append(
                    {
                        "origin": t,
                        "target": t + h,
                        "horizon": h,
                        "seed": seed,
                        "prediction": filename,
                        **record,
                    }
                )
                write(rows_file, rows)
                self.progress(
                    "REFERENCE_" + phase.upper(), method=name, origin=t, horizon=h, seed=seed
                )
        result = {
            "state": "COMPLETE",
            "pairs": rows,
            "means": {
                str(h): float(np.mean([r["scaled_sliced_w2"] for r in rows if r["horizon"] == h]))
                for h in horizons
            },
        }
        write(directory / "complete.json", result)
        return result

    def select_references(self):
        import traser.baselines.statistical as baselines

        selection_file = self.out / "reference_selection.json"
        if selection_file.exists():
            return read(selection_file)
        candidates = []
        for index, (name, parameters) in enumerate(self.reference_candidates()):
            directory = self.out / "references" / f"candidate_{index}"
            directory.mkdir(parents=True, exist_ok=True)
            model_file = directory / "model.pt"
            if model_file.exists():
                model = self.torch.load(model_file, map_location="cpu", weights_only=False)
            else:
                model = getattr(baselines, "fit_" + name.lower())(
                    self.clouds("train"), **parameters
                )
                atomic_torch(model, model_file)
            scores = [
                self.reference_forecasts(
                    directory / "development" / f"seed_{seed}",
                    model,
                    name,
                    "development",
                    [1],
                    seed,
                )["means"]["1"]
                for seed in self.cfg["seeds"]
            ]
            candidates.append(
                {
                    "index": index,
                    "name": name,
                    "parameters": parameters,
                    "score": float(np.mean(scores)),
                    "model_sha256": sha(model_file),
                }
            )
            write(self.out / "reference_candidates.json", candidates)
        selected = {
            name: min(
                (r for r in candidates if r["name"] == name), key=lambda r: (r["score"], r["index"])
            )
            for name in ("WAR", "WR", "LQD")
        }
        write(selection_file, selected)
        return selected

    def metrics(self, prediction, target_window, scored):
        from traser.metrics.density import density

        observed = self.clouds("test")[target_window]
        tensor = self.torch.as_tensor(prediction, device="cuda", dtype=self.torch.float64)
        sensitivity = self.sensitivity_distance.score(tensor, observed, ("test", target_window))
        target = self.marginal_distance.prepare(observed, ("test", target_window))
        marginal2 = self.marginal_distance.squared(tensor, target, precise=True)
        values, failures = ([], [])
        for d in range(64):
            try:
                values.append({"coordinate": d, **density(prediction[:, d], observed[:, d])})
            except ValueError as exc:
                failures.append({"coordinate": d, "reason": str(exc)})
        names = [
            "symmetric_KL",
            "sqrt_JS_arithmetic",
            "sqrt_JS_geometric_unnormalized",
            "density_L1",
        ]
        metrics = {
            key: float(np.mean([v[key] for v in values])) if len(values) == 64 else None
            for key in names
        }
        metrics.update(
            scaled_sliced_w2=scored["scaled_sliced_w2"],
            projection_sensitivity_scaled_sliced_w2=sensitivity["scaled_sliced_w2"],
            marginal_W2_mean=float(marginal2.clamp_min(0).sqrt().mean()),
            population_mean_MSE=scored["population_mean_MSE"],
            centered_scaled_sliced_w2=scored["centered_scaled_sliced_w2"],
            mean_displacement_l2=scored["mean_displacement_l2"],
            centered_rms_forecast=float(
                np.sqrt(np.mean(np.sum((prediction - prediction.mean(0)) ** 2, axis=1)))
            ),
            centered_rms_observed=target["rms"],
        )
        return {
            "metrics": metrics,
            "density_per_coordinate": values,
            "density_failures": failures,
            "distance_definition": "sqrt(64 * mean projected W2 squared); finite 256 orthogonal directions; not exact joint W2",
            "full_target_mass": True,
            "sensitivity_used_for_selection": False,
        }
