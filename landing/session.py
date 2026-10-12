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

from .autoland import Autoland
from .feedback import feedback
from .core import (
    SCHEMA,
    approach_guidance,
    current_criteria,
    scenario_hash,
    initial_state,
    runway_position,
    touchdown_index,
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


def engine_status():
    """Diagnostic engine readout: a flight that lost power is a failure to explain."""
    try:
        return {name: prop(f"/engines/engine[0]/{name}") for name in ("running", "rpm")}
    except Exception as exc:
        return {"error": str(exc)}


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


def guidance_line(state, scenario):
    g = approach_guidance(state, scenario)
    side = "右" if g["cross_m"] >= 0 else "左"
    sink = f"降下 {g['sink_fpm']:4.0f}fpm"
    if g["distance_m"] > 0:
        error = g["height_error_ft"]
        path = f"経路より {abs(error):3.0f}ft {'高い' if error >= 0 else '低い'}"
        where = f"進入端まで {g['distance_m'] / 1000:3.1f}km"
        sink += f"（目安 {g['target_sink_fpm']:3.0f}）"
    else:
        path = f"対地 {state['agl_ft']:4.0f}ft"
        where = "滑走路上"
    return (
        f"{state['time']:5.0f}s {where} | {path} | 中心線の{side} {abs(g['cross_m']):3.0f}m"
        f" | 速度 {g['airspeed_kt']:3.0f}kt | {sink} | スロットル {state['throttle']:.2f}"
    )


def summary(metadata):
    evaluation = metadata.get("evaluation", {})
    lines = [
        f"status={metadata['status']} end_reason={metadata.get('end_reason')}"
        f" rows={metadata['rows']} control_resets={len(metadata['control_resets'])}",
        f"candidate_success={evaluation.get('candidate_success')}"
        f" reason={evaluation.get('reason', '-')}",
    ]
    for name, value in evaluation.items():
        if isinstance(value, float):
            lines.append(f"  {name} = {value:.2f}")
    for name, passed in evaluation.get("checks", {}).items():
        lines.append(f"  {'OK ' if passed else 'NG '} {name}")
    if metadata.get("error"):
        lines.append(f"error={metadata['error']}")
    return "\n".join(lines)


def approved(directory, scenario):
    """Paths of reviewed successes recorded for this scenario's task, for training."""
    key = scenario_hash(scenario)
    return [
        str(meta_path.parent)
        for meta_path in sorted(Path(directory).glob("*/metadata.json"))
        if (meta := read_json(meta_path)).get("outcome") == "success"
        and scenario_hash(meta["scenario"]) == key
    ]


def review(directory):
    """One line per recorded flight, to choose demonstrations and track progress."""
    lines = []
    for meta_path in sorted(Path(directory).glob("*/metadata.json")):
        meta = read_json(meta_path)
        rows = rows_at(meta_path.parent) if meta.get("rows") else []
        try:
            result = evaluate(
                rows, current_criteria(meta["scenario"]), meta.get("damage")
            )
        except (KeyError, TypeError, ValueError) as exc:
            result = {"candidate_success": False, "reason": str(exc)}
        failed = [k for k, v in result.get("checks", {}).items() if not v]
        lines.append(
            f"{meta_path.parent.name:24} seed={(meta.get('initial') or {}).get('seed', '-')!s:>4}"
            f" {meta.get('purpose', 'evaluation'):10} {meta['status']:10}"
            f" outcome={meta.get('outcome'):10}"
            f" {'PASS' if result['candidate_success'] else 'FAIL'}"
            f" {', '.join(failed) or result.get('reason', '')}"
        )
    passed = sum(" PASS " in line for line in lines)
    lines.append(f"{len(lines)} flights, {passed} passing every check")
    return "\n".join(lines)


def configure_controls(scenario):
    """Apply fixed approach controls after aircraft initialization, while paused."""
    for name in ("flaps", "trim"):
        node = "elevator-trim" if name == "trim" else name
        set_prop("/controls/flight/" + node, scenario["controls"][name], "double")
    set_prop("/controls/gear/gear-down-command", True)
    for side in ("left", "right"):
        set_prop(f"/controls/gear/brake-{side}", 0.0, "double")
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
    expert = getattr(args, "autoland", False)
    if expert:
        policy = Autoland(scenario)
    elif bridge:
        policy = BridgePolicy(bridge, scenario, instruction)
    else:
        policy = Policy(read_json(args.policy)) if args.policy else None
    images = bool(bridge or getattr(args, "images", False))
    if policy and policy.model["scenario_hash"] != scenario_hash(scenario):
        raise ValueError("Policy and scenario differ; use the training scenario")
    if prop("/sim/aircraft") != scenario["aircraft"]:
        raise ValueError("Running aircraft differs from scenario")
    if not prop("/sim/sceneryloaded") or not prop("/sim/signals/fdm-initialized"):
        raise ValueError("Wait until FlightGear scenery and FDM are ready")
    # FlightGear initialization can reset either freeze flag even when passed on its CLI;
    # one remaining flag still stops the simulation, and the seeded start check below
    # rejects an aircraft that has already flown.
    if not (prop("/sim/freeze/master") or prop("/sim/freeze/clock")):
        raise ValueError("Start from a paused scenario")
    pause(True)
    if not (prop("/sim/freeze/master") and prop("/sim/freeze/clock")):
        raise ValueError("FlightGear did not acknowledge pause")
    if prop("/sim/physical-ai/scenario-hash") != scenario_hash(scenario):
        raise ValueError("Restart with start_flightgear.ps1 and this scenario")
    if bool(prop("/sim/input/no-joystick-input")) != bool(policy):
        raise ValueError("Launcher Mode must match human/ai session")
    start = initial_state(
        scenario,
        int(prop("/sim/physical-ai/seed")),
        prop("/sim/physical-ai/start-profile"),
    )
    # The launcher and runner derive the start independently; a mismatch means a stale launch.
    position = {k: float(prop(f"/position/{k}-deg")) for k in ("latitude", "longitude")}
    r = scenario["runway_geometry"]
    if math.dist(runway_position(position, r), runway_position(start, r)) > 30:
        raise ValueError("Aircraft is not at the seeded start; restart the launcher")
    configure_controls(scenario)
    for name, expected in (("flaps", scenario["controls"]["flaps"]),):
        if abs(float(prop("/controls/flight/" + name)) - expected) > 0.02:
            raise ValueError("Configure flaps to the scenario value before starting")
    if not prop("/controls/gear/gear-down-command"):
        raise ValueError("Lower landing gear before starting")
    engine_start = engine_status()
    path.mkdir(parents=True, exist_ok=False)
    if images:
        (path / "frames").mkdir()
    metadata = {
        "schema": SCHEMA,
        "source": "expert" if expert else ("ai" if policy else "human"),
        "status": "running",
        "outcome": "unreviewed",
        "requested_seconds": args.seconds,
        "purpose": getattr(args, "purpose", "evaluation"),
        "scenario": scenario,
        "scenario_hash": scenario_hash(scenario),
        "initial": start,
        "policy_hash": digest(policy.model) if policy else None,
        "policy_manifest": policy.model if bridge or expert else None,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": {"start": engine_start},
        "implementation_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        "aircraft_lock": read_json(
            Path(__file__).resolve().parents[1] / "config/aircraft-lock.json"
        ),
        "instruction": instruction,
        "images": images,
        "execution_mode": "paused_inference"
        if bridge
        else ("paused_capture" if expert and images else "continuous"),
        "image_period": getattr(args, "image_period", 0.2),
        "video": bool(args.video),
        "playback_speed": 1,
        "rows": 0,
        "control_resets": [],
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

        def overridden(state):
            flaps = scenario["controls"]["flaps"]
            if abs(state["flaps"] - flaps) <= 0.02:
                return False
            # Nobody operates the fixed flaps, so a change means an aircraft script
            # (the C172P state manager sets flaps and throttle 5 s after an in-air
            # start) overrode the controls. Restore both and drop the sample.
            throttle = (
                rows[-1]["state"]["throttle"]
                if rows
                else scenario["controls"]["throttle"]
            )
            # Throttle first: a sample taken in between still shows the moved
            # flaps, so it is detected and dropped as well.
            for i in range(scenario["engine_count"]):
                set_prop(f"/controls/engines/engine[{i}]/throttle", throttle, "double")
            set_prop("/controls/flight/flaps", flaps, "double")
            metadata["control_resets"].append(
                {
                    "time": state["time"],
                    "observed_flaps": state["flaps"],
                    "observed_throttle": state["throttle"],
                    "restored_throttle": throttle,
                }
            )
            return True

        # Establish a valid telemetry channel before releasing the paused simulator.
        packet(sock.recv(65535))
        pause(False)
        last_time = None
        touchdown = None
        next_guidance = 0.0
        rollout_seconds = scenario["evaluation"]["min_rollout_seconds"]
        end_reason = "duration_limit"
        stopped = None
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
                if overridden(state):
                    continue
                frame = None
                inference_seconds = None
                command = None
                if images and state["time"] >= next_image_time:
                    # Captures take 0.3-0.9 s; the bridge and the expert pause the
                    # simulation so control keeps its 20 Hz in simulation time.
                    if bridge or expert:
                        pause(True)
                        # Wait for a telemetry sample acknowledging both freeze flags.
                        freeze_deadline = time.monotonic() + args.timeout
                        while not (state["paused"] and state["clock_paused"]):
                            if time.monotonic() >= freeze_deadline:
                                raise TimeoutError(
                                    "FlightGear did not acknowledge pause"
                                )
                            state = packet(sock.recv(65535))
                        # The paused sample replaces the checked one; check it too.
                        if overridden(state):
                            pause(False)
                            continue
                    image_path = path / "frames" / f"{len(rows):08d}.jpg"
                    frame = capture(image_path, lambda: prop("/sim/time/elapsed-sec"))
                    frame["view_number"] = prop("/sim/current-view/view-number")
                    next_image_time = state["time"] + getattr(args, "image_period", 0.2)
                    if bridge:
                        infer_started = time.monotonic()
                        command = policy.predict_frame(state, scenario, image_path)
                        inference_seconds = time.monotonic() - infer_started
                if expert:
                    command = policy.predict(state)
                elif policy and not bridge and command is None:
                    command = policy.predict(features(state, scenario))
                if command is not None:
                    sender.sendto(
                        action_packet(command, scenario["engine_count"]),
                        ("127.0.0.1", scenario["control_port"]),
                    )
                if (bridge or expert) and frame is not None:
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
                if "guidance" in scenario and state["time"] >= next_guidance:
                    print(guidance_line(state, scenario), flush=True)
                    next_guidance = state["time"] + 1
                if recorder and recorder.poll() is not None:
                    raise RuntimeError("Video capture stopped unexpectedly")
                # A crash or a stop off the runway (such as in water) ends the flight
                # as a failure; nothing is gained by recording it any longer.
                if state["crashed"]:
                    end_reason = "crashed"
                    break
                if state["wow"] < 0.5 and state["groundspeed_kt"] < 5:
                    stopped = stopped if stopped is not None else state["time"]
                    if state["time"] - stopped >= 3:
                        end_reason = "stopped_off_runway"
                        break
                else:
                    stopped = None
                # The task ends after a short rollout; deceleration is out of scope.
                if touchdown is None:
                    touchdown = touchdown_index([r["state"] for r in rows[-2:]])
                    touchdown = state["time"] if touchdown is not None else None
                elif state["time"] - touchdown >= rollout_seconds:
                    end_reason = "rollout_complete"
                    break
        metadata["status"] = "complete" if rows else "incomplete"
        metadata["end_reason"] = end_reason
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
        metadata["engine"]["end"] = engine_status()
        if scenario.get("damage_properties"):
            try:
                metadata["damage"] = {
                    p: bool(prop(p)) for p in scenario["damage_properties"]
                }
            except Exception as exc:
                metadata["damage_error"] = str(exc)
        try:
            metadata["evaluation"] = evaluate(rows, scenario, metadata.get("damage"))
        except (KeyError, TypeError, ValueError) as exc:
            metadata.update(status="incomplete", evaluation_error=str(exc))
        save()
        print(summary(metadata))
        if "evaluation" in metadata:
            print("\n".join(feedback(rows, scenario, metadata["evaluation"])))
        print(f"Saved {path / 'metadata.json'}")


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
        q.add_argument(
            "--autoland",
            action="store_true",
            help="Fly the scripted expert landing (training demonstrations)",
        )
        q.add_argument("--images", action="store_true")
        q.add_argument("--instruction", default=INSTRUCTION)
        q.add_argument("--image-period", type=float, default=0.2)
    q = sub.add_parser("review", help="List flights with their landing checks")
    q.add_argument("directory", type=Path)
    q = sub.add_parser(
        "approved", help="Print reviewed successes for a scenario (training input)"
    )
    q.add_argument("directory", type=Path)
    q.add_argument("--scenario", type=Path, required=True)
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
        if a.command == "record" and (a.policy or a.vla_bridge or a.autoland):
            p.error("Use fly for AI control")
        if (
            a.command == "fly"
            and sum(map(bool, (a.policy, a.vla_bridge, a.autoland))) != 1
        ):
            p.error("Choose exactly one of --policy, --vla-bridge and --autoland")
        run(a)
    elif a.command == "review":
        print(review(a.directory))
    elif a.command == "approved":
        print("\n".join(approved(a.directory, read_json(a.scenario))))
    else:
        meta = read_json(a.episode / "metadata.json")
        meta["evaluation"] = evaluate(
            rows_at(a.episode), current_criteria(meta["scenario"]), meta.get("damage")
        )
        if a.command == "label":
            if meta.get("purpose", "evaluation") != "evaluation":
                raise ValueError(
                    "Diagnostic episodes cannot be reviewed as demonstrations"
                )
            if (
                meta["source"] not in ("human", "expert")
                or meta["status"] != "complete"
            ):
                raise ValueError(
                    "Only complete human or expert episodes can be reviewed for training"
                )
            if a.outcome == "success" and not meta["evaluation"]["candidate_success"]:
                failed = [
                    k for k, v in meta["evaluation"].get("checks", {}).items() if not v
                ]
                raise ValueError(
                    f"Only flights passing every landing check can be successes: {failed}"
                )
            meta.update(outcome=a.outcome, review_note=a.note)
            atomic_json(a.episode / "metadata.json", meta)
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
