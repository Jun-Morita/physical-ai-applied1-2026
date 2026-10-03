import argparse
import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from test_pipeline import ROOT, SCENARIO, state, wire

from landing import session
from landing.compare import compare
from landing.core import ACTIONS, FEATURES, digest, read_json
from landing.vision import INSTRUCTION, BridgePolicy, atomic_json, capture
from landing.vla import decode_action, training_episodes


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
            self.assertEqual(setter.call_count, 4)

    def test_action_mapping(self):
        self.assertEqual(decode_action([0] * 7), [0, 0, 0, 0.5, 0.5, 0.5, 0])
        with self.assertRaises(ValueError):
            decode_action([float("nan")] * 7)

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
                        "scenario_hash": digest(SCENARIO),
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
                            {"worker_id": "test", "action": [0] * 7},
                        )
                        return
                    time.sleep(0.005)

            thread = threading.Thread(target=worker)
            thread.start()
            self.assertEqual(policy.predict_frame(state(), SCENARIO, image), [0] * 7)
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
            "/sim/physical-ai/scenario-hash": digest(SCENARIO),
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
            bridge.return_value.model = {"scenario_hash": digest(SCENARIO)}
            bridge.return_value.predict_frame.return_value = [0, 0, 0, 0.4, 0, 0, 0]
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
                    },
                )
            result = compare(paths[:1], paths[1:])
            self.assertEqual(result["before"]["count"], 1)
            self.assertEqual(result["after"]["candidate_success_rate"], 0)
            meta = read_json(paths[1] / "metadata.json")
            meta["image_period"] = 0.5
            atomic_json(paths[1] / "metadata.json", meta)
            with self.assertRaisesRegex(ValueError, "conditions"):
                compare(paths[:1], paths[1:])

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
            manifest = {"scenario_hash": digest(SCENARIO), "instruction": INSTRUCTION}
            self.assertEqual(len(training_episodes(paths, manifest)), 3)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                training_episodes([paths[0]] * 3, manifest)
            (paths[0] / "frames/f.jpg").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                training_episodes(paths, manifest)


if __name__ == "__main__":
    unittest.main()
