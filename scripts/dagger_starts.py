"""DAgger start states: where the learned policy actually drifts off the approach.

    python3 scripts/dagger_starts.py runs/rollouts/vla-instruments-window-near2-15000-50? \\
      --first-seed 601 --output config/dagger/c172p-r1.json

Each rollout (the policy flown on a training seed with fly_c172p.ps1 -Rollout) is
sampled at fixed distances before the threshold. States off the approach path
(cross-track or height error beyond the limits) become numbered start states; a
training_starts profile with "states" pointing at the file lets the expert
autoland start there (fly_c172p.ps1 -Autoland -StartProfile <profile> -Seed <n>).
Evaluation flights (runs/eval, seeds 1000+) are refused so they stay held out.

    python3 scripts/dagger_starts.py --mirror config/dagger/c172p-r1.json \\
      --first-seed 646 --output config/dagger/c172p-r1-mirror.json

A biased policy gives one-sided starts (round 1: right of the centerline and low),
and the expert's recoveries from them carry that bias into the data. --mirror
reflects each start about the centerline and the glide path (cross-track, height
error and heading error change sign; distance and airspeed stay), so the expert
also flies the opposite recoveries. The mirrored states are flown, not
approximated by flipping images.

    python3 scripts/dagger_starts.py runs/rollouts/vla-instruments-window-dagger2-15000-51? \\
      --only-high --max-height-error-ft 450 --distances 3300 3000 2700 2400 2100 \\
      --first-seed 701 --output config/dagger/c172p-r2-high.json

--only-high keeps states above the glide path. Rollout starts also keep the
attitude (pitch_deg): a nose-high policy loses the runway from the windshield
view, and the expert shows how to lower the nose and find it again. States too
far above the path for a landing (--max-height-error-ft) are left out.
"""

import argparse
import json
import math
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

EARTH_M = 6371000


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


def mirrored(start, scenario):
    """Start reflected about the centerline and the glide path."""
    r = scenario["runway_geometry"]
    along, cross = runway_position(start, r)
    g = approach_guidance({**start, "down_fps": 0, "groundspeed_kt": 0}, scenario)
    h = math.radians(r["heading_deg"])
    north = along * math.cos(h) + cross * math.sin(h)
    east = along * math.sin(h) - cross * math.cos(h)
    return {
        "latitude": r["latitude"] + math.degrees(north / EARTH_M),
        "longitude": r["longitude"]
        + math.degrees(east / (EARTH_M * math.cos(math.radians(r["latitude"])))),
        "altitude_ft": start["altitude_ft"] - 2 * g["height_error_ft"],
        "heading_deg": (2 * r["heading_deg"] - start["heading_deg"]) % 360,
        "airspeed_kt": start["airspeed_kt"],
    }


def mirror_starts(path, scenario, first_seed):
    starts, seed = {}, first_seed
    for number, s in read_json(path)["starts"].items():
        m = mirrored(s, scenario)
        g = approach_guidance({**m, "down_fps": 0, "groundspeed_kt": 0}, scenario)
        print(
            f"{number}: cross {s['cross_m']:+6.1f} -> {g['cross_m']:+6.1f} m, "
            f"height {s['height_error_ft']:+6.1f} -> {g['height_error_ft']:+6.1f} ft"
            f" -> seed {seed}"
        )
        starts[str(seed)] = {
            **m,
            "source": f"mirror of {path.as_posix()} seed {number} ({s.get('source')})",
            "distance_m": s.get("distance_m"),
            "cross_m": g["cross_m"],
            "height_error_ft": g["height_error_ft"],
        }
        seed += 1
    return starts, seed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("rollouts", nargs="*", type=Path)
    p.add_argument(
        "--mirror", type=Path, help="Reflect the starts of this file instead"
    )
    p.add_argument(
        "--scenario",
        type=Path,
        default=ROOT / "config/scenarios/c172p-rjtt-34r-approach.json",
        help="Scenario whose runway and glide path --mirror reflects about",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--first-seed", type=int, required=True)
    p.add_argument(
        "--distances", type=float, nargs="+", default=[2500, 1800, 1200, 800, 500]
    )
    p.add_argument("--min-cross-m", type=float, default=20)
    p.add_argument("--min-height-error-ft", type=float, default=30)
    p.add_argument("--min-agl-ft", type=float, default=40)
    p.add_argument(
        "--only-high", action="store_true", help="Only states above the path"
    )
    p.add_argument("--max-height-error-ft", type=float)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit(f"{a.output} exists; start states must not change once used")
    if bool(a.mirror) == bool(a.rollouts):
        raise SystemExit("Give rollouts or --mirror, not both")
    starts, seed = {}, a.first_seed
    if a.mirror:
        starts, seed = mirror_starts(a.mirror, read_json(a.scenario), a.first_seed)
    for flight in a.rollouts:
        for distance, s, g in states(flight, a.distances, a.min_agl_ft):
            off = abs(g["cross_m"]) >= a.min_cross_m or (
                abs(g["height_error_ft"]) >= a.min_height_error_ft
            )
            if a.only_high:
                off = g["height_error_ft"] >= a.min_height_error_ft
            if a.max_height_error_ft is not None:
                off = off and abs(g["height_error_ft"]) <= a.max_height_error_ft
            print(
                f"{flight.name} {distance:5.0f} m: cross {g['cross_m']:+6.1f} m, "
                f"height {g['height_error_ft']:+6.1f} ft, pitch {s['pitch_deg']:+5.1f}"
                + (f" -> seed {seed}" if off else "")
            )
            if not off:
                continue
            if seed >= EVALUATION_SEED_START:
                raise SystemExit("Seeds would reach the evaluation range")
            starts[str(seed)] = {
                **{k: s[k] for k in START_KEYS},
                "pitch_deg": s["pitch_deg"],
                "source": f"{flight.as_posix()} t={s['time']:.2f}",
                "distance_m": distance,
                "cross_m": g["cross_m"],
                "height_error_ft": g["height_error_ft"],
            }
            seed += 1
    if not starts:
        raise SystemExit("No off-path states found")
    if seed > EVALUATION_SEED_START:
        raise SystemExit("Seeds would reach the evaluation range")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(
        a.output,
        {
            "description": (
                f"Mirror of {a.mirror.as_posix()} about the centerline and glide path"
                if a.mirror
                else "DAgger start states from policy rollouts on training seeds"
            ),
            "starts": starts,
        },
    )
    print(f"saved {len(starts)} starts (seeds {a.first_seed}-{seed - 1}) to {a.output}")


if __name__ == "__main__":
    main()
