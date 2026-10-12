"""Shared, dependency-free schema and inference for Windows and WSL."""

import hashlib
import json
import math
import random
from pathlib import Path

from .storage import read_json as read_json

SCHEMA = 1
# One FlightGear generic packet samples these properties together.
FIELDS = {
    "time": "/sim/time/elapsed-sec",
    "latitude": "/position/latitude-deg",
    "longitude": "/position/longitude-deg",
    "altitude_ft": "/position/altitude-ft",
    "agl_ft": "/position/altitude-agl-ft",
    "airspeed_kt": "/velocities/airspeed-kt",
    "groundspeed_kt": "/velocities/groundspeed-kt",
    "north_fps": "/velocities/speed-north-fps",
    "east_fps": "/velocities/speed-east-fps",
    "down_fps": "/velocities/speed-down-fps",
    "roll_deg": "/orientation/roll-deg",
    "pitch_deg": "/orientation/pitch-deg",
    "heading_deg": "/orientation/heading-deg",
    "roll_rate": "/orientation/roll-rate-degps",
    "pitch_rate": "/orientation/pitch-rate-degps",
    "yaw_rate": "/orientation/yaw-rate-degps",
    "aileron": "/controls/flight/aileron",
    "elevator": "/controls/flight/elevator",
    "rudder": "/controls/flight/rudder",
    "throttle": "/controls/engines/engine[0]/throttle",
    "throttle1": "/controls/engines/engine[1]/throttle",
    "throttle2": "/controls/engines/engine[2]/throttle",
    "throttle3": "/controls/engines/engine[3]/throttle",
    "brake_left": "/controls/gear/brake-left",
    "brake_right": "/controls/gear/brake-right",
    "trim": "/controls/flight/elevator-trim",
    "flaps": "/controls/flight/flaps",
    "gear": "/controls/gear/gear-down-command",
    "wow": "/gear/gear[1]/wow",
    "paused": "/sim/freeze/master",
    "clock_paused": "/sim/freeze/clock",
    "crashed": "/sim/crashed",
}
# Control stick, rudder and common throttle. Brakes, trim, flaps and gear stay fixed.
ACTIONS = ["aileron", "elevator", "rudder", "throttle"]
LIMITS = [(-1, 1), (-1, 1), (-1, 1), (0, 1)]
FEATURES = [
    "along_m",
    "cross_m",
    "height_m",
    "agl_m",
    "airspeed_kt",
    "groundspeed_kt",
    "along_speed_mps",
    "cross_speed_mps",
    "down_mps",
    "roll_deg",
    "pitch_deg",
    "heading_sin",
    "heading_cos",
    "roll_rate",
    "pitch_rate",
    "yaw_rate",
    "wow",
]


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# Keys that describe how a flight is judged or displayed, not the task itself.
# Training start profiles widen data collection only; evaluation uses the nominal start.
NON_TASK_KEYS = (
    "description",
    "evaluation",
    "guidance",
    "damage_properties",
    "training_starts",
)
NOMINAL_START = "nominal"


def scenario_hash(scenario):
    """Task identity: tuning success criteria must not invalidate recorded flights."""
    return digest({k: v for k, v in scenario.items() if k not in NON_TASK_KEYS})


def current_criteria(scenario):
    """Judge a recorded scenario with the repository's current criteria."""
    key = scenario_hash(scenario)
    root = Path(__file__).resolve().parents[1] / "config/scenarios"
    for path in sorted(root.glob("*.json")):
        config = read_json(path)
        if scenario_hash(config) == key:
            return {**scenario, **{k: config[k] for k in NON_TASK_KEYS if k in config}}
    return scenario


def packet(data):
    values = data.decode("ascii").strip().split(",")
    if len(values) != len(FIELDS):
        raise ValueError(f"Telemetry schema mismatch: {len(values)} != {len(FIELDS)}")
    values = [float(v) for v in values]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Non-finite telemetry")
    return dict(zip(FIELDS, values))


def runway_position(state, runway):
    north = math.radians(state["latitude"] - runway["latitude"]) * 6371000
    east = (
        math.radians(state["longitude"] - runway["longitude"])
        * 6371000
        * math.cos(math.radians(runway["latitude"]))
    )
    h = math.radians(runway["heading_deg"])
    return north * math.cos(h) + east * math.sin(h), east * math.cos(
        h
    ) - north * math.sin(h)


# Seeds from this value up are held out for evaluation and never used as demonstrations.
EVALUATION_SEED_START = 1000


def initial_state(scenario, seed, profile=NOMINAL_START):
    """Start state from the runway-relative preset plus a seeded uniform offset.

    Seed 0 is the nominal start without offsets. A training start profile moves
    the start closer on the same path and widens the offsets, scaled with distance.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Seed must be a non-negative integer")
    r, start = scenario["runway_geometry"], dict(scenario["initial"])
    variation = scenario.get("initial_variation", {})
    if profile != NOMINAL_START:
        if seed >= EVALUATION_SEED_START:
            raise ValueError("Evaluation seeds use the nominal start")
        if profile not in scenario.get("training_starts", {}):
            raise ValueError(f"Unknown start profile: {profile}")
        p = scenario["training_starts"][profile]
        scale = p["distance_scale"]
        # Stay on the line from the nominal start to the aiming point.
        aim = scenario.get("guidance", {}).get("aim_point_m", 0)
        distance = start["distance_m"] * scale
        start["height_ft"] *= (distance + aim) / (start["distance_m"] + aim)
        start["distance_m"] = distance
        variation = {
            k: v * scale if k in ("cross_m", "height_ft") else v
            for k, v in p["variation"].items()
        }
    rng = random.Random(seed)
    offsets = {
        key: rng.uniform(-limit, limit) if seed else 0.0
        for key, limit in sorted(variation.items())
    }
    along = -start["distance_m"]
    cross = start["cross_m"] + offsets.get("cross_m", 0)
    h = math.radians(r["heading_deg"])
    north = along * math.cos(h) - cross * math.sin(h)
    east = along * math.sin(h) + cross * math.cos(h)
    return {
        "seed": seed,
        "profile": profile,
        "offsets": offsets,
        "latitude": r["latitude"] + math.degrees(north / 6371000),
        "longitude": r["longitude"]
        + math.degrees(east / (6371000 * math.cos(math.radians(r["latitude"])))),
        "altitude_ft": r["elevation_ft"]
        + start["height_ft"]
        + offsets.get("height_ft", 0),
        "heading_deg": (start["heading_deg"] + offsets.get("heading_deg", 0)) % 360,
        "airspeed_kt": start["airspeed_kt"] + offsets.get("airspeed_kt", 0),
    }


def features(s, scenario):
    r = scenario["runway_geometry"]
    along, cross = runway_position(s, r)
    h = math.radians(r["heading_deg"])
    error = math.radians(s["heading_deg"] - r["heading_deg"])
    return [
        along,
        cross,
        (s["altitude_ft"] - r["elevation_ft"]) * 0.3048,
        s["agl_ft"] * 0.3048,
        s["airspeed_kt"],
        s["groundspeed_kt"],
        (s["north_fps"] * math.cos(h) + s["east_fps"] * math.sin(h)) * 0.3048,
        (s["east_fps"] * math.cos(h) - s["north_fps"] * math.sin(h)) * 0.3048,
        s["down_fps"] * 0.3048,
        s["roll_deg"],
        s["pitch_deg"],
        math.sin(error),
        math.cos(error),
        s["roll_rate"],
        s["pitch_rate"],
        s["yaw_rate"],
        s["wow"],
    ]


def clamp_actions(action):
    if len(action) != len(ACTIONS) or not all(math.isfinite(v) for v in action):
        raise ValueError("Invalid action vector")
    return [max(lo, min(hi, v)) for v, (lo, hi) in zip(action, LIMITS)]


def action_packet(action, engine_count):
    if engine_count not in (1, 2, 3, 4):
        raise ValueError("Invalid engine count")
    a = dict(zip(ACTIONS, clamp_actions(action)))
    # Protocol has four engine channels. Only selected aircraft engines receive nonzero thrust.
    values = [a["aileron"], a["elevator"], a["rudder"]]
    values += [a["throttle"] if i < engine_count else 0 for i in range(4)]
    return (",".join(f"{v:.8f}" for v in values) + "\n").encode("ascii")


class Policy:
    """CPU-only deployment of an exported small MLP; no Windows ML dependencies."""

    def __init__(self, model):
        if (
            model["schema"] != SCHEMA
            or model["features"] != FEATURES
            or model["actions"] != ACTIONS
        ):
            raise ValueError("Policy schema mismatch")
        if len(model["mean"]) != len(FEATURES) or len(model["std"]) != len(FEATURES):
            raise ValueError("Normalization shape mismatch")
        if not all(math.isfinite(v) for v in model["mean"]):
            raise ValueError("Invalid normalization mean")
        if any(not math.isfinite(v) or v <= 0 for v in model["std"]):
            raise ValueError("Invalid normalization scale")
        width = len(FEATURES)
        for layer in model["layers"]:
            if not all(
                math.isfinite(v) for row in layer["weight"] for v in row
            ) or not all(math.isfinite(v) for v in layer["bias"]):
                raise ValueError("Non-finite network weights")
            if len(layer["weight"]) != len(layer["bias"]) or any(
                len(row) != width for row in layer["weight"]
            ):
                raise ValueError("Invalid network shape")
            width = len(layer["bias"])
        if width != len(ACTIONS):
            raise ValueError("Invalid network output")
        self.model = model

    def predict(self, observation):
        if len(observation) != len(FEATURES) or not all(
            math.isfinite(v) for v in observation
        ):
            raise ValueError("Invalid observation")
        x = [
            (v - m) / s
            for v, m, s in zip(observation, self.model["mean"], self.model["std"])
        ]
        for i, layer in enumerate(self.model["layers"]):
            x = [
                sum(w * v for w, v in zip(row, x)) + bias
                for row, bias in zip(layer["weight"], layer["bias"])
            ]
            if i + 1 != len(self.model["layers"]):
                x = [max(0, v) for v in x]
        return clamp_actions(x)


def evaluate(rows, scenario, damage=None):
    """Observed landing metrics, not certification or automatic dataset approval.

    damage maps aircraft damage properties read at the end of the flight to flags.
    """
    if not rows:
        return {"candidate_success": False, "reason": "no_samples"}
    states = [r["state"] for r in rows]
    touch = touchdown_index(states)
    result = {
        "candidate_success": False,
        "touchdown_detected": touch is not None,
        "crashed": any(s["crashed"] > 0.5 for s in states),
    }
    if touch is None:
        result["reason"] = "no_air_to_ground_transition"
        return result
    s = states[touch]
    r, e = scenario["runway_geometry"], scenario["evaluation"]
    along, cross = runway_position(s, r)
    # Last airborne sample approximates contact sink rate; it is not impact-load measurement.
    sink = states[touch - 1]["down_fps"]
    heading_error = (s["heading_deg"] - r["heading_deg"] + 180) % 360 - 180
    rollout = states[touch:]
    # Height regained after first contact; a skip of a few inches is not a bounce.
    rebound = max(
        (v["agl_ft"] - s["agl_ft"] for v in rollout if v["wow"] < 0.5), default=0.0
    )
    rollout_seconds = states[-1]["time"] - s["time"]
    result.update(
        touchdown_time=s["time"],
        touchdown_along_m=along,
        touchdown_cross_m=cross,
        pre_touchdown_sink_fps=sink,
        touchdown_bank_deg=s["roll_deg"],
        touchdown_pitch_deg=s["pitch_deg"],
        touchdown_heading_error_deg=heading_error,
        touchdown_airspeed_kt=s["airspeed_kt"],
        rollout_seconds=rollout_seconds,
        bounce_height_ft=rebound,
    )
    result["checks"] = {
        "no_crash": not result["crashed"],
        "touchdown_zone": 0 <= along <= e["touchdown_max_m"],
        "centerline": abs(cross) <= e["max_cross_m"],
        # A floating flare can read a few mm/s of climb at contact; real climbs are bounces.
        "sink_rate": e.get("min_sink_fps", 0) <= sink <= e["max_sink_fps"],
        "bank": abs(s["roll_deg"]) <= e["max_bank_deg"],
        "main_gear_first": s["pitch_deg"] >= e["min_pitch_deg"],
        "heading": abs(heading_error) <= e["max_heading_error_deg"],
        "no_bounce": rebound <= e["max_bounce_ft"]
        if "max_bounce_ft" in e
        else all(v["wow"] > 0.5 for v in rollout),
        "rollout_inside": all(
            0 <= runway_position(v, r)[0] <= r["length_m"]
            and abs(runway_position(v, r)[1]) <= r["width_m"] / 2
            for v in rollout
        ),
        "rollout_complete": rollout_seconds >= e["min_rollout_seconds"],
    }
    if scenario.get("damage_properties") or damage is not None:
        # Unread damage counts as a failure; a broken gear is never a success.
        result["damage"] = damage
        result["checks"]["no_damage"] = damage is not None and not any(damage.values())
    result["candidate_success"] = all(result["checks"].values())
    return result


def approach_guidance(state, scenario):
    """Pilot aid: deviation from the scenario glide path; not an AI input."""
    r, g = scenario["runway_geometry"], scenario["guidance"]
    along, cross = runway_position(state, r)
    slope = math.tan(math.radians(g["glide_slope_deg"]))
    if "aim_point_m" in g:
        # Glide path to the runway aiming point (ICAO: 400 m on long runways).
        target_ft = max(0.0, g["aim_point_m"] - along) * slope / 0.3048
    else:
        # Older recorded scenarios: fixed threshold crossing height.
        target_ft = (
            g["threshold_crossing_height_ft"] + max(0.0, -along) * slope / 0.3048
        )
    return {
        "distance_m": -along,
        "height_error_ft": state["altitude_ft"] - r["elevation_ft"] - target_ft,
        "cross_m": cross,
        "airspeed_kt": state["airspeed_kt"],
        "sink_fpm": state["down_fps"] * 60,
        # Groundspeed along a 3 degree path needs roughly 5.3 ft/min per knot.
        "target_sink_fpm": state["groundspeed_kt"] * 6076.12 / 60 * slope,
    }


def touchdown_index(states):
    return next(
        (
            i
            for i in range(1, len(states))
            if states[i - 1]["wow"] < 0.5 <= states[i]["wow"]
        ),
        None,
    )


if __name__ == "__main__":
    # Launcher helper: canonical scenario hash and seeded start, as exact strings.
    import sys

    scenario = read_json(sys.argv[1])
    start = initial_state(
        scenario, int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else NOMINAL_START
    )
    keys = ("latitude", "longitude", "altitude_ft", "heading_deg", "airspeed_kt")
    print(
        json.dumps(
            {"hash": scenario_hash(scenario), **{k: repr(start[k]) for k in keys}}
        )
    )
