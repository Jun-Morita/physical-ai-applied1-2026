"""Windows stdlib runner: record, fly, inspect and review FlightGear episodes."""

import argparse
import hashlib
import json
import math
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from .core import (
    SCHEMA,
    Policy,
    action_packet,
    digest,
    evaluate,
    features,
    packet,
    read_json,
)
from .recording import finish_video
from .storage import atomic_json
from .storage import rows_at as rows_at
from .vision import INSTRUCTION, BridgePolicy, capture


def prop(path):
    with urllib.request.urlopen(
        "http://127.0.0.1:8080/json" + path, timeout=3
    ) as response:
        return json.load(response).get("value")


def set_prop(path, value, kind="bool"):
    parent, name = path.rsplit("/", 1)
    req = urllib.request.Request(
        "http://127.0.0.1:8080/json" + parent,
        data=json.dumps({"name": name, "type": kind, "value": value}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=3):
        pass


def pause(value):
    for name in ("master", "clock"):
        set_prop("/sim/freeze/" + name, value)


def configure_controls(scenario):
    """Apply fixed approach controls after aircraft initialization, while paused."""
    for name in ("flaps", "trim"):
        node = "elevator-trim" if name == "trim" else name
        set_prop("/controls/flight/" + node, scenario["controls"][name], "double")
    set_prop("/controls/gear/gear-down-command", True)
    for i in range(scenario["engine_count"]):
        set_prop(
            f"/controls/engines/engine[{i}]/throttle",
            scenario["controls"]["throttle"],
            "double",
        )


def run(args):
    path = Path(args.output)
    if path.exists():
        raise ValueError("Output already exists; choose a new episode directory")
    scenario = read_json(args.scenario)
    bridge = getattr(args, "vla_bridge", None)
    instruction = getattr(args, "instruction", INSTRUCTION)
    policy = (
        BridgePolicy(bridge, scenario, instruction)
        if bridge
        else (Policy(read_json(args.policy)) if args.policy else None)
    )
    images = bool(bridge or getattr(args, "images", False))
    if policy and policy.model["scenario_hash"] != digest(scenario):
        raise ValueError("Policy and scenario differ; use the training scenario")
    if prop("/sim/aircraft") != scenario["aircraft"]:
        raise ValueError("Running aircraft differs from scenario")
    if not prop("/sim/sceneryloaded") or not prop("/sim/signals/fdm-initialized"):
        raise ValueError("Wait until FlightGear scenery and FDM are ready")
    if not prop("/sim/freeze/master"):
        raise ValueError("Start from a paused scenario")
    # FlightGear initialization can reset clock freeze even when passed on its CLI.
    pause(True)
    if not prop("/sim/freeze/clock"):
        raise ValueError("FlightGear did not acknowledge clock freeze")
    if prop("/sim/physical-ai/scenario-hash") != digest(scenario):
        raise ValueError("Restart with start_flightgear.ps1 and this scenario")
    if bool(prop("/sim/input/no-joystick-input")) != bool(policy):
        raise ValueError("Launcher Mode must match human/ai session")
    configure_controls(scenario)
    for name, expected in (("flaps", scenario["controls"]["flaps"]),):
        if abs(float(prop("/controls/flight/" + name)) - expected) > 0.02:
            raise ValueError("Configure flaps to the scenario value before starting")
    if not prop("/controls/gear/gear-down-command"):
        raise ValueError("Lower landing gear before starting")
    path.mkdir(parents=True, exist_ok=False)
    if images:
        (path / "frames").mkdir()
    metadata = {
        "schema": SCHEMA,
        "source": "ai" if policy else "human",
        "status": "running",
        "outcome": "unreviewed",
        "requested_seconds": args.seconds,
        "purpose": getattr(args, "purpose", "evaluation"),
        "scenario": scenario,
        "scenario_hash": digest(scenario),
        "policy_hash": digest(policy.model) if policy else None,
        "policy_manifest": policy.model if bridge else None,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "implementation_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        "aircraft_lock": read_json(
            Path(__file__).resolve().parents[1] / "config/aircraft-lock.json"
        ),
        "instruction": instruction,
        "images": images,
        "execution_mode": "paused_inference" if bridge else "continuous",
        "image_period": getattr(args, "image_period", 0.2),
        "video": bool(args.video),
        "playback_speed": 1,
        "rows": 0,
    }

    def save():
        atomic_json(path / "metadata.json", metadata)

    save()
    recorder = None
    video_log = None
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rows = []
    try:
        sock.bind(("127.0.0.1", scenario["telemetry_port"]))
        sock.settimeout(args.timeout)
        if args.video:
            video_log = (path / "ffmpeg.log").open("w")
            recorder = subprocess.Popen(
                [
                    args.ffmpeg,
                    "-hide_banner",
                    "-nostats",
                    "-f",
                    "gdigrab",
                    "-framerate",
                    "20",
                    "-i",
                    "title=FlightGear",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-pix_fmt",
                    "yuv420p",
                    str(path / "flight.mp4"),
                ],
                stdin=subprocess.PIPE,
                stdout=video_log,
                stderr=video_log,
            )
            time.sleep(1)
            if recorder.poll() is not None:
                raise RuntimeError(
                    "FFmpeg failed; inspect ffmpeg.log and FlightGear window title"
                )
        # Establish a valid telemetry channel before releasing the paused simulator.
        packet(sock.recv(65535))
        pause(False)
        last_time = None
        next_image_time = float("-inf")
        started = time.monotonic()
        with (path / "samples.jsonl").open("w", encoding="utf-8") as stream:
            while time.monotonic() - started < args.seconds:
                state = packet(sock.recv(65535))
                # Consume the newest queued packet so AI never steers using an old backlog.
                sock.setblocking(False)
                dropped = 0
                try:
                    while True:
                        state = packet(sock.recv(65535))
                        dropped += 1
                except BlockingIOError:
                    pass
                finally:
                    sock.settimeout(args.timeout)
                if last_time is not None and state["time"] < last_time:
                    raise RuntimeError(
                        "Simulation time reset; start a separate episode"
                    )
                if state["paused"] or state["clock_paused"]:
                    continue
                if last_time == state["time"]:
                    continue
                frame = None
                inference_seconds = None
                command = None
                if images and state["time"] >= next_image_time:
                    if bridge:
                        pause(True)
                        # Wait for a telemetry sample acknowledging both freeze flags.
                        freeze_deadline = time.monotonic() + args.timeout
                        while not (state["paused"] and state["clock_paused"]):
                            if time.monotonic() >= freeze_deadline:
                                raise TimeoutError(
                                    "FlightGear did not acknowledge pause"
                                )
                            state = packet(sock.recv(65535))
                    image_path = path / "frames" / f"{len(rows):08d}.jpg"
                    frame = capture(image_path, lambda: prop("/sim/time/elapsed-sec"))
                    frame["view_number"] = prop("/sim/current-view/view-number")
                    next_image_time = state["time"] + getattr(args, "image_period", 0.2)
                    if bridge:
                        infer_started = time.monotonic()
                        command = policy.predict_frame(state, scenario, image_path)
                        inference_seconds = time.monotonic() - infer_started
                if policy and not bridge and command is None:
                    command = policy.predict(features(state, scenario))
                if command is not None:
                    sender.sendto(
                        action_packet(command, scenario["engine_count"]),
                        ("127.0.0.1", scenario["control_port"]),
                    )
                if bridge and frame is not None:
                    pause(False)
                row = {
                    "state": state,
                    "image": frame,
                    "inference_seconds": inference_seconds,
                    "command": command,
                    "wall_elapsed": time.monotonic() - started,
                    "dropped_packets": dropped,
                }
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                rows.append(row)
                last_time = state["time"]
                if recorder and recorder.poll() is not None:
                    raise RuntimeError("Video capture stopped unexpectedly")
                if state["crashed"]:
                    raise RuntimeError("Simulator reports a crash")
        metadata["status"] = "complete" if rows else "incomplete"
        metadata["end_reason"] = "duration_limit"
    except KeyboardInterrupt:
        metadata["end_reason"] = "user_interrupted"
        metadata["status"] = "complete" if rows else "incomplete"
    except Exception as exc:
        metadata.update(status="incomplete", end_reason="error", error=str(exc))
        raise
    finally:
        try:
            pause(True)
        except Exception as exc:
            metadata.update(status="incomplete", pause_error=str(exc))
            print("Could not pause FlightGear:", exc)
        sock.close()
        sender.close()
        finish_video(recorder, video_log, metadata)
        metadata["rows"] = len(rows)
        try:
            metadata["evaluation"] = evaluate(rows, scenario)
        except (KeyError, TypeError, ValueError) as exc:
            metadata.update(status="incomplete", evaluation_error=str(exc))
        save()
        print(json.dumps(metadata, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("record", "fly"):
        q = sub.add_parser(name)
        q.add_argument("--scenario", required=True, type=Path)
        q.add_argument("--output", required=True, type=Path)
        q.add_argument("--seconds", type=float, default=240)
        q.add_argument("--timeout", type=float, default=3)
        q.add_argument(
            "--purpose", choices=["evaluation", "diagnostic"], default="evaluation"
        )
        q.add_argument("--video", action="store_true")
        q.add_argument("--ffmpeg", default="ffmpeg")
        q.add_argument("--policy", type=Path, default=None)
        q.add_argument("--vla-bridge", type=Path)
        q.add_argument("--images", action="store_true")
        q.add_argument("--instruction", default=INSTRUCTION)
        q.add_argument("--image-period", type=float, default=0.2)
    for name in ("inspect", "label"):
        q = sub.add_parser(name)
        q.add_argument("episode", type=Path)
        if name == "label":
            q.add_argument(
                "--outcome", choices=["success", "failure", "exclude"], required=True
            )
            q.add_argument("--note", required=True)
    a = p.parse_args()
    if a.command in ("record", "fly"):
        if not all(math.isfinite(v) and v > 0 for v in (a.seconds, a.timeout)):
            p.error("Durations must be positive")
        if not 0.125 <= a.image_period <= 5:
            p.error("--image-period must be between .125 and 5 seconds")
        if a.command == "record" and (a.policy or a.vla_bridge):
            p.error("Use fly for AI control")
        if a.command == "fly" and bool(a.policy) == bool(a.vla_bridge):
            p.error("Choose exactly one of --policy and --vla-bridge")
        run(a)
    else:
        meta = read_json(a.episode / "metadata.json")
        meta["evaluation"] = evaluate(rows_at(a.episode), meta["scenario"])
        if a.command == "label":
            if meta.get("purpose", "evaluation") != "evaluation":
                raise ValueError(
                    "Diagnostic episodes cannot be reviewed as demonstrations"
                )
            if meta["source"] != "human" or meta["status"] != "complete":
                raise ValueError(
                    "Only complete human episodes can be reviewed for training"
                )
            meta.update(outcome=a.outcome, review_note=a.note)
            atomic_json(a.episode / "metadata.json", meta)
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
