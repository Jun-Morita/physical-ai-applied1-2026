"""Dependency-free image capture and shared-directory VLA transport (Windows)."""

import base64
import hashlib
import math
import time
import urllib.request
import uuid
from pathlib import Path

from .core import (
    ACTIONS,
    FEATURES,
    NAV_FEATURES,
    clamp_actions,
    features,
    read_json,
    scenario_hash,
)
from .storage import atomic_json as atomic_json

STATE_SCALE = [
    10000,
    1000,
    1000,
    1000,
    200,
    200,
    100,
    100,
    10,
    45,
    20,
    1,
    1,
    30,
    30,
    30,
    1,
]
INSTRUCTION = "Land on runway 34R at Haneda and keep the aircraft on the runway."


def capture(path, get_time, timeout=5):
    before = float(get_time())
    started = time.monotonic()
    with urllib.request.urlopen(
        "http://127.0.0.1:8080/screenshot?type=jpg", timeout=timeout
    ) as response:
        data = response.read(20 * 1024 * 1024 + 1)
    after = float(get_time())
    if (
        len(data) > 20 * 1024 * 1024
        or not data.startswith(b"\xff\xd8")
        or not data.endswith(b"\xff\xd9")
    ):
        raise ValueError("FlightGear did not return a complete JPEG screenshot")
    if not all(math.isfinite(t) for t in (before, after)):
        raise ValueError("Non-finite image timestamp")
    if after < before:
        raise ValueError("Simulation reset during image capture")
    Path(path).write_bytes(data)
    return {
        "file": "frames/" + Path(path).name,
        "sim_before": before,
        "sim_after": after,
        "capture_seconds": time.monotonic() - started,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


class BridgePolicy:
    def __init__(self, directory, scenario, instruction, timeout=60):
        self.directory = Path(directory)
        info = read_json(self.directory / "ready.json")
        self.model = info["manifest"]
        if (
            self.model["scenario_hash"] != scenario_hash(scenario)
            or self.model["instruction"] != instruction
        ):
            raise ValueError("VLA worker scenario/instruction mismatch")
        # The request carries all FEATURES; the worker keeps those its model uses.
        if (
            not set(self.model["features"]) <= set(FEATURES) | set(NAV_FEATURES)
            or self.model["actions"] != ACTIONS
        ):
            raise ValueError("VLA worker feature/action schema mismatch")
        self.worker_id = info["worker_id"]
        self.session_id = uuid.uuid4().hex
        self.timeout = timeout

    def predict_frame(self, state, scenario, image):
        request_id = uuid.uuid4().hex
        request = {
            "worker_id": self.worker_id,
            "session_id": self.session_id,
            "scenario_hash": scenario_hash(scenario),
            "features": features(state, scenario),
            "image": base64.b64encode(Path(image).read_bytes()).decode("ascii"),
        }
        response_path = self.directory / (request_id + ".response.json")
        request_path = self.directory / (request_id + ".request.json")
        atomic_json(request_path, request)
        deadline = time.monotonic() + self.timeout
        try:
            while time.monotonic() < deadline:
                if response_path.exists():
                    result = read_json(response_path)
                    if result.get("worker_id") != self.worker_id:
                        raise ValueError("VLA worker changed during flight")
                    if "error" in result:
                        raise RuntimeError(result["error"])
                    return clamp_actions(result["action"])
                time.sleep(0.02)
            raise TimeoutError("VLA inference timed out; simulator remains paused")
        finally:
            request_path.unlink(missing_ok=True)
            response_path.unlink(missing_ok=True)
