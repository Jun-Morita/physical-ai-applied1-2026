import argparse
import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from test_pipeline import ROOT, SCENARIO, START_PROPS, state, wire

from landing import session
from landing.compare import compare
from landing.core import ACTIONS, FEATURES, read_json, scenario_hash
from landing.vision import INSTRUCTION, BridgePolicy, atomic_json, capture
from landing.vla import decode_action, state_normalization, training_episodes


class VLATests(unittest.TestCase):
    def test_fixed_controls_after_initialization(self):
        with patch.object(session, "set_prop") as setter:
            session.configure_controls(SCENARIO)
            setter.assert_any_call(
                "/controls/flight/flaps", SCENARIO["controls"]["flaps"], "double"
            )
            setter.assert_any_call(
                "/controls/flight/elevator-trim", SCENARIO["controls"]["trim"], "double"
            )
            setter.assert_any_call(
                "/controls/engines/engine[0]/throttle",
                SCENARIO["controls"]["throttle"],
                "double",
            )
            setter.assert_any_call("/controls/gear/brake-left", 0.0, "double")
            self.assertEqual(setter.call_count, 6)

    def test_state_normalization_uses_training_statistics(self):
        from landing.vision import STATE_SCALE

        # The untrained baseline keeps the fixed scale; fine-tuned models carry stats.
        self.assertEqual(state_normalization({})[1], STATE_SCALE)
        stats = {
            "state_mean": [1.0] * len(FEATURES),
            "state_std": [2.0] * len(FEATURES),
        }
        self.assertEqual(
            state_normalization(stats), (stats["state_mean"], stats["state_std"])
        )

    def test_action_mapping(self):
        self.assertEqual(decode_action([0] * 4), [0, 0, 0, 0.5])
        with self.assertRaises(ValueError):
            decode_action([float("nan")] * 4)

    def test_image_capture_bounds_and_validation(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("landing.vision.urllib.request.urlopen") as opened,
        ):
            opened.return_value.__enter__.return_value.read.return_value = (
                b"\xff\xd8example\xff\xd9"
            )
            result = capture(Path(tmp) / "frame.jpg", iter([1.0, 1.1]).__next__)
            self.assertEqual((result["sim_before"], result["sim_after"]), (1.0, 1.1))
            self.assertEqual(
                hashlib.sha256((Path(tmp) / "frame.jpg").read_bytes()).hexdigest(),
                result["sha256"],
            )
            opened.return_value.__enter__.return_value.read.return_value = (
                b"<html>Error</html>"
            )
            with self.assertRaises(ValueError):
                capture(Path(tmp) / "bad.jpg", lambda: 1.0)

    def test_bridge_request_response_and_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            atomic_json(
                path / "ready.json",
                {
                    "worker_id": "test",
                    "manifest": {
                        "scenario_hash": scenario_hash(SCENARIO),
                        "instruction": INSTRUCTION,
                        "features": FEATURES,
                        "actions": ACTIONS,
                    },
                },
            )
            image = path / "frame.jpg"
            image.write_bytes(b"example")
            policy = BridgePolicy(path, SCENARIO, INSTRUCTION, timeout=2)

            def worker():
                for _ in range(200):
                    requests = list(path.glob("*.request.json"))
                    if requests:
                        req = requests[0]
                        atomic_json(
                            req.with_name(req.name.replace(".request", ".response")),
                            {"worker_id": "test", "action": [0] * 4},
                        )
                        return
                    time.sleep(0.005)

            thread = threading.Thread(target=worker)
            thread.start()
            self.assertEqual(policy.predict_frame(state(), SCENARIO, image), [0] * 4)
            thread.join()
            self.assertFalse(list(path.glob("*.request.json")))
            policy.timeout = 0.01
            with self.assertRaises(TimeoutError):
                policy.predict_frame(state(), SCENARIO, image)

    def test_vla_session_pauses_before_inference(self):
        props = {
            "/sim/aircraft": "c172p",
            "/sim/sceneryloaded": True,
            "/sim/signals/fdm-initialized": True,
            "/sim/freeze/master": True,
            "/sim/freeze/clock": True,
            "/sim/physical-ai/scenario-hash": scenario_hash(SCENARIO),
            **START_PROPS,
            "/sim/input/no-joystick-input": True,
            "/controls/flight/flaps": SCENARIO["controls"]["flaps"],
            "/controls/gear/gear-down-command": True,
            "/sim/current-view/view-number": 0,
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(session, "prop", side_effect=props.__getitem__),
            patch.object(session, "pause") as pause,
            patch.object(session, "configure_controls"),
            patch.object(session.socket, "socket") as sock,
            patch.object(session, "BridgePolicy") as bridge,
            patch.object(session, "capture", return_value={"file": "frames/test.jpg"}),
            patch("builtins.print"),
        ):
            bridge.return_value.model = {"scenario_hash": scenario_hash(SCENARIO)}
            bridge.return_value.predict_frame.return_value = [0, 0, 0, 0.4]
            paused = state(0.1)
            paused.update(paused=1, clock_paused=1)
            sock.return_value.recv.side_effect = [
                wire(state()),
                wire(state(0.05)),
                BlockingIOError(),
                wire(paused),
                KeyboardInterrupt(),
            ]
            args = argparse.Namespace(
                scenario=ROOT / "config/scenarios/c172p-rjtt-34r-approach.json",
                policy=None,
                vla_bridge=Path(tmp) / "bridge",
                output=Path(tmp) / "flight",
                timeout=1,
                seconds=10,
                video=False,
                instruction=INSTRUCTION,
            )
            session.run(args)
            self.assertEqual(
                [c.args[0] for c in pause.call_args_list],
                [True, False, True, False, True],
            )
            bridge.return_value.predict_frame.assert_called_once()
            sock.return_value.sendto.assert_called_once()
            meta = read_json(args.output / "metadata.json")
            self.assertEqual(meta["execution_mode"], "paused_inference")
            self.assertEqual(meta["source"], "ai")

    def test_expert_pauses_simulation_while_capturing(self):
        props = {
            "/sim/aircraft": "c172p",
            "/sim/sceneryloaded": True,
            "/sim/signals/fdm-initialized": True,
            "/sim/freeze/master": True,
            "/sim/freeze/clock": True,
            "/sim/physical-ai/scenario-hash": scenario_hash(SCENARIO),
            **START_PROPS,
            "/sim/input/no-joystick-input": True,
            "/controls/flight/flaps": SCENARIO["controls"]["flaps"],
            "/controls/gear/gear-down-command": True,
            "/sim/current-view/view-number": 0,
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(session, "prop", side_effect=props.__getitem__),
            patch.object(session, "pause") as pause,
            patch.object(session, "configure_controls"),
            patch.object(session.socket, "socket") as sock,
            patch.object(session, "capture", return_value={"file": "frames/test.jpg"}),
            patch("builtins.print"),
        ):
            paused = state(0.05)
            paused.update(paused=1, clock_paused=1)
            sock.return_value.recv.side_effect = [
                wire(state()),
                wire(state(0.05)),
                BlockingIOError(),
                wire(paused),
                KeyboardInterrupt(),
            ]
            args = argparse.Namespace(
                scenario=ROOT / "config/scenarios/c172p-rjtt-34r-approach.json",
                policy=None,
                autoland=True,
                images=True,
                output=Path(tmp) / "expert",
                timeout=1,
                seconds=10,
                video=False,
            )
            session.run(args)
            # Start, pause for the capture, resume after the command, final pause.
            self.assertEqual(
                [c.args[0] for c in pause.call_args_list],
                [True, False, True, False, True],
            )
            sock.return_value.sendto.assert_called_once()
            meta = read_json(args.output / "metadata.json")
            self.assertEqual(meta["execution_mode"], "paused_capture")
            self.assertEqual(meta["source"], "expert")

    def test_override_arriving_while_pausing_for_capture_is_dropped(self):
        props = {
            "/sim/aircraft": "c172p",
            "/sim/sceneryloaded": True,
            "/sim/signals/fdm-initialized": True,
            "/sim/freeze/master": True,
            "/sim/freeze/clock": True,
            "/sim/physical-ai/scenario-hash": scenario_hash(SCENARIO),
            **START_PROPS,
            "/sim/input/no-joystick-input": True,
            "/controls/flight/flaps": SCENARIO["controls"]["flaps"],
            "/controls/gear/gear-down-command": True,
            "/sim/current-view/view-number": 0,
        }
        frozen = {"paused": 1, "clock_paused": 1}
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(session, "prop", side_effect=props.__getitem__),
            patch.object(session, "pause"),
            patch.object(session, "configure_controls"),
            patch.object(session, "set_prop"),
            patch.object(session.socket, "socket") as sock,
            patch.object(
                session, "capture", return_value={"file": "frames/test.jpg"}
            ) as capture,
            patch("builtins.print"),
        ):
            # The C172P override lands in the sample acknowledging the pause.
            sock.return_value.recv.side_effect = [
                wire(state()),
                wire(state(0.05)),
                BlockingIOError(),
                wire({**state(0.05), **frozen, "flaps": 1.0, "throttle": 0.75}),
                wire(state(0.1)),
                BlockingIOError(),
                wire({**state(0.1), **frozen}),
                KeyboardInterrupt(),
            ]
            args = argparse.Namespace(
                scenario=ROOT / "config/scenarios/c172p-rjtt-34r-approach.json",
                policy=None,
                autoland=True,
                images=True,
                output=Path(tmp) / "expert",
                timeout=1,
                seconds=10,
                video=False,
            )
            session.run(args)
            capture.assert_called_once()
            meta = read_json(args.output / "metadata.json")
            self.assertEqual(len(meta["control_resets"]), 1)
            rows = session.rows_at(args.output)
            self.assertEqual([r["state"]["time"] for r in rows], [0.1])

    def test_comparison_keeps_failed_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name, stage in [("before", "base_adapted"), ("after", "finetuned")]:
                path = Path(tmp) / name
                path.mkdir()
                paths.append(path)
                atomic_json(
                    path / "metadata.json",
                    {
                        "source": "ai",
                        "status": "incomplete",
                        "policy_manifest": {
                            "stage": stage,
                            "seed": 42,
                            "weights_sha256": name,
                            "parent_weights_sha256": "before",
                        },
                        "scenario": SCENARIO,
                        "instruction": INSTRUCTION,
                        "execution_mode": "paused_inference",
                        "image_period": 0.2,
                        "policy_hash": name,
                        "initial": {"seed": 1000},
                    },
                )
            result = compare(paths[:1], paths[1:])
            self.assertEqual(result["before"]["count"], 1)
            self.assertEqual(result["after"]["candidate_success_rate"], 0)
            self.assertEqual(result["after"]["touchdown_rate"], 0)
            meta = read_json(paths[1] / "metadata.json")
            for seed, error in [(1001, "same start seeds"), (1, "held-out")]:
                atomic_json(
                    paths[1] / "metadata.json", {**meta, "initial": {"seed": seed}}
                )
                with self.assertRaisesRegex(ValueError, error):
                    compare(paths[:1], paths[1:])
            meta["image_period"] = 0.5
            atomic_json(paths[1] / "metadata.json", meta)
            with self.assertRaisesRegex(ValueError, "conditions"):
                compare(paths[:1], paths[1:])

    def test_expert_paused_captures_pair_with_their_commands(self):
        from landing.train import load_episodes

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for episode in range(3):
                path = Path(tmp) / f"expert-{episode}"
                (path / "frames").mkdir(parents=True)
                (path / "frames/f.jpg").write_bytes(b"synthetic-test-image")
                image_hash = hashlib.sha256(b"synthetic-test-image").hexdigest()
                atomic_json(
                    path / "metadata.json",
                    {
                        "source": "expert",
                        "status": "complete",
                        "outcome": "success",
                        "scenario": SCENARIO,
                        "initial": {"seed": 1},
                        "evaluation": {"candidate_success": True},
                        "instruction": INSTRUCTION,
                        "execution_mode": "paused_capture",
                    },
                )
                rows = []
                for i in range(25):
                    s = {**state(i * 0.2), "paused": 1, "clock_paused": 1}
                    s["altitude_ft"] += episode
                    image = {
                        "file": "frames/f.jpg",
                        "sim_before": i * 0.2,
                        "sim_after": i * 0.2,
                        "sha256": image_hash,
                        "view_number": 0,
                    }
                    rows.append(
                        {"state": s, "image": image, "command": [0.1, -0.1, 0, 0.5]}
                    )
                (path / "samples.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows)
                )
                paths.append(path)
            manifest = {
                "scenario_hash": scenario_hash(SCENARIO),
                "instruction": INSTRUCTION,
            }
            # Paused captures are exact pairs, not rejected as crossing a pause.
            episodes = training_episodes(paths, manifest)
            self.assertEqual(len(episodes[0]["samples"]), 25)
            self.assertEqual(episodes[0]["samples"][0]["action"], [0.1, -0.1, 0, 0.5])
            self.assertEqual(load_episodes(paths)[0]["y"][0], [0.1, -0.1, 0, 0.5])
            rows[0]["command"] = [2, 0, 0, 0.5]
            (paths[0] / "samples.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            with self.assertRaisesRegex(ValueError, "Invalid expert command"):
                load_episodes([paths[0]])
            # A logged aircraft-script override row (recorded before the
            # paused-capture fix) is dropped; an unlogged one rejects the flight.
            rows[0]["command"] = [0.1, -0.1, 0, 0.5]
            rows[1]["state"].update(flaps=1.0, throttle=0.75)
            (paths[0] / "samples.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                load_episodes([paths[0]])
            meta = read_json(paths[0] / "metadata.json")
            atomic_json(
                paths[0] / "metadata.json",
                {**meta, "control_resets": [{"time": 0.2, "observed_flaps": 1.0}]},
            )
            self.assertEqual(len(load_episodes([paths[0]])[0]["y"]), 24)

    def test_visual_dataset_filters_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for episode in range(3):
                path = Path(tmp) / str(episode)
                path.mkdir()
                (path / "frames").mkdir()
                (path / "frames/f.jpg").write_bytes(b"synthetic-test-image")
                image_hash = hashlib.sha256(b"synthetic-test-image").hexdigest()
                atomic_json(
                    path / "metadata.json",
                    {
                        "source": "human",
                        "status": "complete",
                        "outcome": "success",
                        "scenario": SCENARIO,
                        "initial": {"seed": 1},
                        "evaluation": {"candidate_success": True},
                        "instruction": INSTRUCTION,
                    },
                )
                rows = []
                for i in range(12):
                    s = state(i * 0.2)
                    s["altitude_ft"] += episode
                    rows += [
                        {
                            "state": s,
                            "image": {
                                "file": "frames/f.jpg",
                                "sim_before": i * 0.2,
                                "sim_after": i * 0.2 + 0.01,
                                "sha256": image_hash,
                                "view_number": 0,
                            },
                        },
                        {"state": state(i * 0.2 + 0.05)},
                    ]
                (path / "samples.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows)
                )
                paths.append(path)
            manifest = {
                "scenario_hash": scenario_hash(SCENARIO),
                "instruction": INSTRUCTION,
            }
            self.assertEqual(len(training_episodes(paths, manifest)), 3)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                training_episodes([paths[0]] * 3, manifest)
            (paths[0] / "frames/f.jpg").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                training_episodes(paths, manifest)


if __name__ == "__main__":
    unittest.main()
