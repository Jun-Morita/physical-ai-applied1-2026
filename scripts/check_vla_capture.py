"""Read-only check of a paused FlightGear instance, run with Windows Python."""

import argparse
import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from landing.core import digest, packet, read_json
from landing.session import prop
from landing.vision import capture

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--scenario", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
scenario = read_json(a.scenario)
deadline = time.monotonic() + 60
while time.monotonic() < deadline:
    try:
        if prop("/sim/sceneryloaded") and prop("/sim/signals/fdm-initialized"):
            break
    except OSError:
        pass
    time.sleep(0.5)
else:
    raise TimeoutError("FlightGear not ready")
if not (prop("/sim/freeze/master") and prop("/sim/freeze/clock")):
    raise RuntimeError("Pause FlightGear first")
if prop("/sim/physical-ai/scenario-hash") != digest(scenario):
    raise ValueError("Scenario mismatch")
a.output.mkdir(parents=True, exist_ok=False)
(a.output / "frames").mkdir()
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    sock.bind(("127.0.0.1", scenario["telemetry_port"]))
    sock.settimeout(5)
    first = packet(sock.recv(65535))
    frame = capture(
        a.output / "frames/screen.jpg", lambda: prop("/sim/time/elapsed-sec")
    )
    sock.setblocking(False)
    last = first
    try:
        while True:
            last = packet(sock.recv(65535))
    except BlockingIOError:
        pass
report = {
    "aircraft": prop("/sim/aircraft"),
    "first_state": first,
    "last_state": last,
    "frame": frame,
    "view_number": prop("/sim/current-view/view-number"),
    "note": "Paused diagnostic, not a flight demonstration.",
}
(a.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
