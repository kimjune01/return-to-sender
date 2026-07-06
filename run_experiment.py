"""Main study + failure-mode ablations. Writes results.json and prints tables."""

import json

import numpy as np

from return_to_sender.sim import World, run_trial

Q = 0.10
TAU = 1.0
P_THROTTLE = 0.5
REPS = 200
WINDOW_GRID = [10, 25, 50, 100, 200]


def replicate(reps, n_windows, p_throttle, seed_base, **kw):
    rows = []
    for r in range(reps):
        rng = np.random.default_rng(seed_base + r)
        world = World().build(rng)
        rows.append(
            run_trial(
                world,
                n_windows,
                p_throttle,
                kw.get("tau", TAU),
                Q,
                rng,
                confounded=kw.get("confounded", False),
                estimator=kw.get("estimator", "aipw"),
            )
        )
    return {
        "fdr": float(np.mean([x["fdp"] for x in rows])),
        "power": float(np.mean([x["power"] for x in rows])),
        "mean_flagged": float(np.mean([x["n_flagged"] for x in rows])),
        "subcontractor_flags_total": int(
            np.sum([x["subcontractors_flagged"] for x in rows])
        ),
        "reps": reps,
    }


def show(t, row):
    print(
        f"{t:>8} {row['fdr']:>7.3f} {row['power']:>7.3f}"
        f" {row['mean_flagged']:>8.1f} {row['subcontractor_flags_total']:>11}"
    )


def main():
    results = {"q": Q, "tau": TAU, "p_throttle": P_THROTTLE, "reps": REPS}
    header = f"{'windows':>8} {'FDR':>7} {'power':>7} {'flagged':>8} {'subc.flags':>11}"

    for est in ["ht", "aipw"]:
        print(f"\nMain study ({est.upper()}): q={Q}, tau={TAU}, p={P_THROTTLE}")
        print(header)
        study = {}
        for t in WINDOW_GRID:
            row = replicate(REPS, t, P_THROTTLE, seed_base=1000 * t, estimator=est)
            study[t] = row
            show(t, row)
        results[f"main_{est}"] = study

    print("\nAblation A: confounded (reactive) throttling, analyst assumes design")
    for tau in [TAU, 0.25]:
        print(f"  tau={tau}")
        print(header)
        arm = {}
        for t in [50, 100]:
            row = replicate(
                REPS, t, P_THROTTLE, seed_base=7_000_000 + t, confounded=True, tau=tau
            )
            arm[t] = row
            show(t, row)
        results[f"ablation_confounded_tau{tau}"] = arm
    # Control: same small tau under the honest design must keep FDR <= q.
    print(f"  tau=0.25, designed randomization (control)")
    print(header)
    arm = {}
    for t in [50, 100]:
        row = replicate(REPS, t, P_THROTTLE, seed_base=8_000_000 + t, tau=0.25)
        arm[t] = row
        show(t, row)
    results["control_tau0.25"] = arm

    print("\nAblation B: rare friction (p=0.05), positivity weakens")
    print(header)
    ablation_b = {}
    for t in WINDOW_GRID:
        row = replicate(REPS, t, 0.05, seed_base=9_000_000 + t)
        ablation_b[t] = row
        show(t, row)
    results["ablation_rare_friction"] = ablation_b

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote results.json")


if __name__ == "__main__":
    main()
