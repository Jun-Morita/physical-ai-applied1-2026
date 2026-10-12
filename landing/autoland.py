"""Scripted expert landing: a programmatic demonstrator, not the learned AI.

It reads the same telemetry as the learned policies and outputs the same four
controls. Approach: track the centreline and the scenario glide path, hold the
approach speed with the throttle. Flare: idle, wings level, align with rudder,
reduce the sink rate with height. Rollout: idle and steer with rudder.
"""

import hashlib
import math
from pathlib import Path

from .core import approach_guidance, scenario_hash

APPROACH_KT = 65.0
# Height of the telemetry agl reference above the wheels on the ground (FlightGear C172P).
GEAR_AGL_FT = 3.5
FLARE_FT = 20.0


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap(deg):
    return (deg + 180) % 360 - 180


class Autoland:
    def __init__(self, scenario, dt=0.05):
        self.scenario = scenario
        self.dt = dt
        self.last_time = None
        # Identifies the demonstrator in flight metadata, like a learned policy's manifest.
        self.model = {
            "type": "autoland",
            "scenario_hash": scenario_hash(scenario),
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }
        self.runway = scenario["runway_geometry"]["heading_deg"]
        self.pitch_hold = 2.0
        self.elevator_hold = 0.0
        self.throttle_hold = scenario["controls"]["throttle"]
        self.phase = "approach"
        self.flare_throttle = None

    def predict(self, s):
        # Integrators use the telemetry interval, which varies with dropped packets.
        if self.last_time is not None:
            self.dt = clamp(s["time"] - self.last_time, 0.0, 0.5)
        self.last_time = s["time"]
        g = approach_guidance(s, self.scenario)
        h = math.radians(self.runway)
        lateral = (s["east_fps"] * math.cos(h) - s["north_fps"] * math.sin(h)) * 0.3048
        wheels_ft = s["agl_ft"] - GEAR_AGL_FT
        if s["wow"] > 0.5:
            self.phase = "rollout"
        elif self.phase == "approach" and wheels_ft < FLARE_FT and g["distance_m"] < 0:
            self.phase = "flare"
        heading_error = wrap(self.runway - s["heading_deg"])

        if self.phase == "rollout":
            # Hold a low nose attitude so leftover lift cannot balloon the aircraft.
            rudder = clamp(0.08 * heading_error - 0.01 * g["cross_m"], -1, 1)
            elevator = clamp(
                0.05 * (s["pitch_deg"] - 1.0) + 0.02 * s["pitch_rate"], -0.3, 0.3
            )
            return [0.0, elevator, rudder, 0.0]

        # Lateral: steer the ground track toward the centreline, bank to turn.
        track = math.degrees(math.atan2(s["east_fps"], s["north_fps"]))
        limit = 20.0 if self.phase == "approach" else 3.0
        track_cmd = self.runway + clamp(-0.25 * g["cross_m"], -limit, limit)
        roll_limit = 15.0 if self.phase == "approach" else 4.0
        roll_cmd = clamp(
            2.0 * wrap(track_cmd - track) - 1.0 * lateral, -roll_limit, roll_limit
        )
        aileron = clamp(
            0.05 * (roll_cmd - s["roll_deg"]) - 0.02 * s["roll_rate"], -0.6, 0.6
        )

        # Vertical: a sink-rate command from the glide path error, flown by pitch.
        sink_fpm = s["down_fps"] * 60
        if self.phase == "approach":
            sink_cmd = clamp(g["target_sink_fpm"] + 3.0 * g["height_error_ft"], 0, 900)
            pitch_limit = (-8.0, 8.0)
        else:
            # Exponential flare: the commanded sink shrinks with wheel height.
            sink_cmd = clamp(60 * (0.12 * wheels_ft + 0.4), 20, 400)
            pitch_limit = (-2.0, 10.0)
        sink_error = sink_fpm - sink_cmd
        self.pitch_hold = clamp(
            self.pitch_hold + 0.0015 * sink_error * self.dt, *pitch_limit
        )
        gain = 0.006 if self.phase == "approach" else 0.015
        pitch_cmd = clamp(self.pitch_hold + gain * sink_error, *pitch_limit)
        pitch_error = pitch_cmd - s["pitch_deg"]
        # Negative elevator raises the nose; the integral absorbs the fixed trim.
        self.elevator_hold = clamp(
            self.elevator_hold - 0.02 * pitch_error * self.dt, -0.5, 0.5
        )
        kp = 0.06 if self.phase == "approach" else 0.1
        elevator = clamp(
            self.elevator_hold - kp * pitch_error + 0.02 * s["pitch_rate"], -0.8, 0.6
        )

        # Speed with throttle; idle progressively once the flare starts.
        if self.phase == "approach":
            speed_error = APPROACH_KT - s["airspeed_kt"]
            self.throttle_hold = clamp(
                self.throttle_hold + 0.01 * speed_error * self.dt, 0.05, 1.0
            )
            throttle = clamp(self.throttle_hold + 0.04 * speed_error, 0.0, 1.0)
        else:
            if self.flare_throttle is None:
                self.flare_throttle = self.throttle_hold
            self.flare_throttle = max(0.0, self.flare_throttle - 0.5 * self.dt)
            throttle = self.flare_throttle
        rudder = (
            clamp(0.03 * heading_error, -0.4, 0.4) if self.phase == "flare" else 0.0
        )
        return [aileron, elevator, rudder, throttle]
