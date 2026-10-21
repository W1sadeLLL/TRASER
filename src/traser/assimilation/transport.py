"""Current paper implementation; provenance is recorded in provenance/module_map.json."""

import numpy as np


def effective_strength(spec, n, innovation, references):
    ratio = np.clip(references["count"] / max(n, 1), 1 / 16, 16)
    multiplier = ratio ** spec.get("count_power", 0) / (
        1 + spec.get("innovation_response", 0) * (innovation / references["innovation"]) ** 2
    )
    return float(spec["strength"] * np.clip(multiplier, 1 / 32, 32))


def transport_update(prior, target, distance, strength, steps):
    """Majorisation step for orthogonal sliced W2 plus displacement tether.

    Sorted quantile-bin means retain all target occurrence mass. Freezing each
    projection assignment gives a quadratic majoriser. Four orthonormal bases
    make its normal matrix scalar; accepting only descending finite objectives
    protects round-off and returns the best available finite-step analysis.
    """
    import torch

    with torch.no_grad():
        anchor = prior.detach()
        x = anchor.clone()
        theta = distance.theta
        q = target["q"]
        scale = distance.scale
        directions = theta.shape[1]

        def objective(v):
            discrepancy = scale * distance.squared(v, target).mean()
            return float(discrepancy + strength * (v - anchor).square().sum(1).mean())

        value = objective(x)
        curve = [value]
        used = 0
        for iteration in range(steps):
            values, order = (x @ theta).sort(0)
            difference = torch.empty_like(values)
            difference.scatter_(0, order, values - q)
            proposal = x - (
                scale / directions * (difference @ theta.T) + strength * (x - anchor)
            ) / (1 + strength)
            new = objective(proposal)
            if not np.isfinite(new):
                raise RuntimeError("Nonfinite transport analysis objective")
            if new > value + 1e-07 * max(1, abs(value)):
                accepted = False
                for factor in (0.5, 0.25, 0.125, 0.0625):
                    candidate = x + factor * (proposal - x)
                    test = objective(candidate)
                    if test <= value:
                        proposal, new, accepted = (candidate, test, True)
                        break
                if not accepted:
                    break
            if new >= value:
                break
            change = value - new
            x = proposal
            value = new
            used = iteration + 1
            curve.append(value)
            if change <= 1e-08 * max(1, abs(value)):
                break
        return (
            x,
            {
                "method": "full_mass_regularised_sliced_transport",
                "strength": strength,
                "requested_steps": steps,
                "selected_step": used,
                "objective_curve": curve,
                "target_occurrences": target["count"],
                "full_target_mass": True,
                "directions": directions,
                "observation_fit_is_not_forecast_validation": True,
            },
        )
