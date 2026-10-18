"""DAgger start states: where the learned policy actually drifts off the approach.

    python3 scripts/dagger_starts.py runs/rollouts/vla-instruments-window-near2-15000-50? \\
      --first-seed 601 --output config/dagger/c172p-r1.json

Each rollout (the policy flown on a training seed with fly_c172p.ps1 -Rollout) is
sampled at fixed distances before the threshold. States off the approach path
(cross-track or height error beyond the limits) become numbered start states; a
training_starts profile with "states" pointing at the file lets the expert
autoland start there (fly_c172p.ps1 -Autoland -StartProfile <profile> -Seed <n>).
Evaluation flights (runs/eval, seeds 1000+) are refused so they stay held out.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from landing.core import (  # noqa: E402
    EVALUATION_SEED_START,
    START_KEYS,
    approach_guidance,
    read_json,
    runway_position,
)
from landing.storage import atomic_json  # noqa: E402


def states(flight, distances, min_agl_ft):
    """Airborne states nearest each distance, before any ground contact."""
    meta = read_json(flight / "metadata.json")
    seed = (meta.get("initial") or {}).get("seed")
    if meta.get("purpose") != "rollout" or not 0 < seed < EVALUATION_SEED_START:
        raise ValueError(f"{flight}: not a training-seed rollout")
    scenario, rows = meta["scenario"], []
    for line in (flight / "samples.jsonl").open(encoding="utf-8"):
        s = json.loads(line)["state"]
        if s["wow"] > 0 or s.get("crashed"):
            break
        rows.append(s)
    r = scenario["runway_geometry"]
    for distance in distances:
        near = [s for s in rows if abs(runway_position(s, r)[0] + distance) < 60]
        if near:
            s = min(near, key=lambda s: abs(runway_position(s, r)[0] + distance))
            if s["agl_ft"] >= min_agl_ft:
                yield distance, s, approach_guidance(s, scenario)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("rollouts", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--first-seed", type=int, required=True)
    p.add_argument(
        "--distances", type=float, nargs="+", default=[2500, 1800, 1200, 800, 500]
    )
    p.add_argument("--min-cross-m", type=float, default=20)
    p.add_argument("--min-height-error-ft", type=float, default=30)
    p.add_argument("--min-agl-ft", type=float, default=40)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit(f"{a.output} exists; start states must not change once used")
    starts, seed = {}, a.first_seed
    for flight in a.rollouts:
        for distance, s, g in states(flight, a.distances, a.min_agl_ft):
            off = abs(g["cross_m"]) >= a.min_cross_m or (
                abs(g["height_error_ft"]) >= a.min_height_error_ft
            )
            print(
                f"{flight.name} {distance:5.0f} m: cross {g['cross_m']:+6.1f} m, "
                f"height {g['height_error_ft']:+6.1f} ft"
                + (f" -> seed {seed}" if off else "")
            )
            if not off:
                continue
            if seed >= EVALUATION_SEED_START:
                raise SystemExit("Seeds would reach the evaluation range")
            starts[str(seed)] = {
                **{k: s[k] for k in START_KEYS},
                "source": f"{flight.as_posix()} t={s['time']:.2f}",
                "distance_m": distance,
                "cross_m": g["cross_m"],
                "height_error_ft": g["height_error_ft"],
            }
            seed += 1
    if not starts:
        raise SystemExit("No off-path states found")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(
        a.output,
        {
            "description": "DAgger start states from policy rollouts on training seeds",
            "starts": starts,
        },
    )
    print(f"saved {len(starts)} starts (seeds {a.first_seed}-{seed - 1}) to {a.output}")


if __name__ == "__main__":
    main()
