"""Shared, dependency-free schema and inference for Windows and WSL."""

import hashlib
import json
import math

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
ACTIONS = [
    "aileron",
    "elevator",
    "rudder",
    "throttle",
    "brake_left",
    "brake_right",
    "trim",
]
LIMITS = [(-1, 1), (-1, 1), (-1, 1), (0, 1), (0, 1), (0, 1), (-1, 1)]
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
    values += [a["brake_left"], a["brake_right"], a["trim"]]
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


def evaluate(rows, scenario):
    """Observed landing metrics, not certification or automatic dataset approval."""
    if not rows:
        return {"candidate_success": False, "reason": "no_samples"}
    states = [r["state"] for r in rows]
    touch = next(
        (
            i
            for i in range(1, len(states))
            if states[i - 1]["wow"] < 0.5 <= states[i]["wow"]
        ),
        None,
    )
    result = {
        "candidate_success": False,
        "touchdown_detected": touch is not None,
        "crashed": any(s["crashed"] > 0.5 for s in states),
    }
    if touch is None:
        result["reason"] = "no_air_to_ground_transition"
        return result
    s = states[touch]
    along, cross = runway_position(s, scenario["runway_geometry"])
    r, e = scenario["runway_geometry"], scenario["evaluation"]
    # Last airborne sample approximates contact sink rate; it is not impact-load measurement.
    sink = states[touch - 1]["down_fps"]
    rollout = states[touch:]
    inside = all(
        0 <= runway_position(v, r)[0] <= r["length_m"]
        and abs(runway_position(v, r)[1]) <= r["width_m"] / 2
        for v in rollout
    )
    settled = states[-1]["time"] - s["time"] >= e["min_rollout_seconds"]
    result.update(
        touchdown_time=s["time"],
        touchdown_along_m=along,
        touchdown_cross_m=cross,
        pre_touchdown_sink_fps=sink,
        touchdown_bank_deg=s["roll_deg"],
        rollout_inside=inside,
        final_groundspeed_kt=states[-1]["groundspeed_kt"],
    )
    result["candidate_success"] = bool(
        not result["crashed"]
        and inside
        and settled
        and 0 <= along <= e["touchdown_max_m"]
        and 0 <= sink <= e["max_sink_fps"]
        and abs(s["roll_deg"]) <= e["max_bank_deg"]
        and states[-1]["wow"] > 0.5
        and states[-1]["groundspeed_kt"] <= e["stop_speed_kt"]
    )
    return result
