"""Failures discovered during the pre-collection review."""

import argparse
import copy
import hashlib
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from test_pipeline import NO_DAMAGE, ROOT, SCENARIO, START_PROPS, state, wire

from landing import session
from landing.compare import compare
from landing.core import scenario_hash
from landing.feedback import feedback
from landing.data import demonstration_action, episode_digest, training_episodes
from landing.recording import finish_video
from landing.storage import atomic_json, file_digest, read_json
from landing.train import load_episodes
from landing.vision import capture


def write_episode(path, rows):
    path.mkdir(exist_ok=True)
    atomic_json(
        path / "metadata.json",
        {
            "source": "human",
            "status": "complete",
            "outcome": "success",
            "scenario": SCENARIO,
            "initial": {"seed": 1},
            "evaluation": {"candidate_success": True},
            "instruction": "land",
        },
    )
    (path / "samples.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


class RegressionTests(unittest.TestCase):
    def test_mlp_rejects_reset_crash_out_of_range_and_nonfinite_state(self):
        for field, value in [
            ("time", -1),
            ("crashed", 1),
            ("aileron", 2),
            ("latitude", math.nan),
        ]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                rows = [{"state": state(i * 0.05)} for i in range(30)]
                rows[15]["state"][field] = value
                write_episode(Path(tmp), rows)
                with self.assertRaises(ValueError):
                    load_episodes([Path(tmp)])

    def test_duplicate_digest_ignores_wall_clock_and_image_filename(self):
        rows = [
            {
                "state": state(i * 0.05),
                "wall_elapsed": i,
                "image": {
                    "file": "frames/a.jpg",
                    "sha256": "a",
                    "sim_before": i * 0.05,
                    "sim_after": i * 0.05,
                    "view_number": 0,
                },
            }
            for i in range(30)
        ]
        changed = copy.deepcopy(rows)
        for row in changed:
            row["wall_elapsed"] += 100
            row["image"]["file"] = "frames/renamed.jpg"
        self.assertEqual(
            episode_digest(rows, visual=True), episode_digest(changed, visual=True)
        )
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            write_episode(a, rows)
            write_episode(b, changed)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_episodes([a, b])

    def test_visual_pair_cannot_cross_a_pause(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "frames").mkdir()
            (path / "frames/test.jpg").write_bytes(b"test")
            rows = []
            for i in range(10):
                t = i * 0.2
                rows.extend(
                    [
                        {
                            "state": state(t),
                            "image": {
                                "file": "frames/test.jpg",
                                "sha256": hashlib.sha256(b"test").hexdigest(),
                                "sim_before": t,
                                "sim_after": t + 0.08,
                                "view_number": 0,
                            },
                        },
                        {"state": state(t + 0.1)},
                    ]
                )
            paused = state(0.05)
            paused["paused"] = 1
            rows.insert(1, {"state": paused})
            write_episode(path, rows)
            with self.assertRaisesRegex(ValueError, "fewer than 10"):
                training_episodes(
                    [path],
                    {"scenario_hash": scenario_hash(SCENARIO), "instruction": "land"},
                )

    def test_video_failure_cannot_prevent_metadata_persistence(self):
        for failure in (
            OSError("broken stdin"),
            subprocess.TimeoutExpired("ffmpeg", 15),
        ):
            with (
                self.subTest(failure=type(failure).__name__),
                tempfile.TemporaryDirectory() as tmp,
            ):
                recorder = Mock()
                recorder.communicate.side_effect = [failure, (b"", b"")]
                recorder.returncode = -9
                log = Mock()
                meta = {"status": "complete"}
                finish_video(recorder, log, meta)
                atomic_json(Path(tmp) / "metadata.json", meta)
                self.assertEqual(
                    read_json(Path(tmp) / "metadata.json")["status"], "incomplete"
                )
                recorder.kill.assert_called_once()
                log.close.assert_called_once()

    def test_empty_session_is_incomplete(self):
        props = {
            "/sim/aircraft": "c172p",
            "/sim/sceneryloaded": True,
            "/sim/signals/fdm-initialized": True,
            "/sim/freeze/master": True,
            "/sim/freeze/clock": True,
            "/sim/physical-ai/scenario-hash": scenario_hash(SCENARIO),
            **START_PROPS,
            "/sim/input/no-joystick-input": False,
            "/controls/flight/flaps": SCENARIO["controls"]["flaps"],
            "/controls/gear/gear-down-command": True,
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(session, "prop", side_effect=props.__getitem__),
            patch.object(session, "pause"),
            patch.object(session, "configure_controls"),
            patch.object(session.socket, "socket") as sock,
            patch("builtins.print"),
        ):
            sock.return_value.recv.return_value = wire(state())
            args = argparse.Namespace(
                output=Path(tmp) / "empty",
                scenario=ROOT / "config/scenarios/c172p-rjtt-34r-approach.json",
                policy=None,
                timeout=1,
                seconds=1e-12,
                video=False,
            )
            session.run(args)
            meta = read_json(args.output / "metadata.json")
            self.assertEqual((meta["status"], meta["rows"]), ("incomplete", 0))

    def test_session_reverts_aircraft_script_override(self):
        props = {
            "/sim/aircraft": "c172p",
            "/sim/sceneryloaded": True,
            "/sim/signals/fdm-initialized": True,
            "/sim/freeze/master": True,
            "/sim/freeze/clock": True,
            "/sim/physical-ai/scenario-hash": scenario_hash(SCENARIO),
            **START_PROPS,
            "/sim/input/no-joystick-input": False,
            "/controls/flight/flaps": SCENARIO["controls"]["flaps"],
            "/controls/gear/gear-down-command": True,
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(session, "prop", side_effect=props.__getitem__),
            patch.object(session, "pause"),
            patch.object(session, "configure_controls"),
            patch.object(session, "set_prop") as setter,
            patch.object(session.socket, "socket") as sock,
            patch("builtins.print"),
        ):
            sock.return_value.recv.side_effect = [
                wire(state()),
                wire({**state(0.05), "throttle": 0.42}),
                BlockingIOError(),
                wire({**state(0.1), "flaps": 1.0, "throttle": 0.75}),
                BlockingIOError(),
                wire({**state(0.15), "throttle": 0.42}),
                BlockingIOError(),
                KeyboardInterrupt(),
            ]
            args = argparse.Namespace(
                output=Path(tmp) / "episode",
                scenario=ROOT / "config/scenarios/c172p-rjtt-34r-approach.json",
                policy=None,
                timeout=1,
                seconds=10,
                video=False,
            )
            session.run(args)
            self.assertEqual(
                [c.args for c in setter.call_args_list],
                [
                    ("/controls/engines/engine[0]/throttle", 0.42, "double"),
                    ("/controls/flight/flaps", SCENARIO["controls"]["flaps"], "double"),
                ],
            )
            meta = read_json(args.output / "metadata.json")
            self.assertEqual(
                meta["control_resets"],
                [
                    {
                        "time": 0.1,
                        "observed_flaps": 1.0,
                        "observed_throttle": 0.75,
                        "restored_throttle": 0.42,
                    }
                ],
            )
            self.assertEqual(
                [r["state"]["time"] for r in session.rows_at(args.output)], [0.05, 0.15]
            )

    def test_session_ends_after_rollout_and_rejects_fixed_control_changes(self):
        props = {
            "/sim/aircraft": "c172p",
            "/sim/sceneryloaded": True,
            "/sim/signals/fdm-initialized": True,
            "/sim/freeze/master": True,
            "/sim/freeze/clock": True,
            "/sim/physical-ai/scenario-hash": scenario_hash(SCENARIO),
            **START_PROPS,
            "/sim/input/no-joystick-input": False,
            "/controls/flight/flaps": SCENARIO["controls"]["flaps"],
            "/controls/gear/gear-down-command": True,
        }
        rollout = SCENARIO["evaluation"]["min_rollout_seconds"]
        packets = [wire(state())]
        for t, wow in [(0.05, 0), (1, 1), (1 + rollout, 1), (2 + rollout, 1)]:
            packets += [wire({**state(t), "wow": wow}), BlockingIOError()]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(session, "prop", side_effect=props.__getitem__),
            patch.object(session, "pause"),
            patch.object(session, "configure_controls"),
            patch.object(session.socket, "socket") as sock,
            patch("builtins.print"),
        ):
            sock.return_value.recv.side_effect = packets
            args = argparse.Namespace(
                output=Path(tmp) / "episode",
                scenario=ROOT / "config/scenarios/c172p-rjtt-34r-approach.json",
                policy=None,
                timeout=1,
                seconds=600,
                video=False,
            )
            session.run(args)
            meta = read_json(args.output / "metadata.json")
            self.assertEqual(
                (meta["end_reason"], meta["rows"]), ("rollout_complete", 3)
            )
            self.assertTrue(meta["evaluation"]["checks"]["rollout_complete"])
            props["/position/latitude-deg"] += 0.001
            with self.assertRaisesRegex(ValueError, "seeded start"):
                session.run(
                    argparse.Namespace(**{**vars(args), "output": Path(tmp) / "b"})
                )
        with tempfile.TemporaryDirectory() as tmp:
            write_episode(Path(tmp), [{"state": state(i * 0.05)} for i in range(3)])
            meta = read_json(Path(tmp) / "metadata.json")
            atomic_json(
                Path(tmp) / "metadata.json", {**meta, "initial": {"seed": 1000}}
            )
            with self.assertRaisesRegex(ValueError, "training seed"):
                load_episodes([Path(tmp)])
        for change in ({"brake_left": 0.5}, {"trim": 0.1}):
            with self.assertRaisesRegex(ValueError, "brake"):
                demonstration_action({**state(), **change}, SCENARIO)

    def test_session_ends_when_the_aircraft_stops_off_the_runway(self):
        props = {
            "/sim/aircraft": "c172p",
            "/sim/sceneryloaded": True,
            "/sim/signals/fdm-initialized": True,
            "/sim/freeze/master": True,
            "/sim/freeze/clock": True,
            "/sim/physical-ai/scenario-hash": scenario_hash(SCENARIO),
            **START_PROPS,
            "/sim/input/no-joystick-input": False,
            "/controls/flight/flaps": SCENARIO["controls"]["flaps"],
            "/controls/gear/gear-down-command": True,
        }
        # Flying, then stuck in water: no gear contact and no ground speed.
        packets = [wire(state())]
        for t, speed in [(0.05, 60), (1, 0), (2, 0), (4.5, 0), (9, 0)]:
            packets += [wire({**state(t), "groundspeed_kt": speed}), BlockingIOError()]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(session, "prop", side_effect=props.__getitem__),
            patch.object(session, "pause"),
            patch.object(session, "configure_controls"),
            patch.object(session.socket, "socket") as sock,
            patch("builtins.print"),
        ):
            sock.return_value.recv.side_effect = packets
            args = argparse.Namespace(
                output=Path(tmp) / "episode",
                scenario=ROOT / "config/scenarios/c172p-rjtt-34r-approach.json",
                policy=None,
                timeout=1,
                seconds=600,
                video=False,
            )
            session.run(args)
            meta = read_json(args.output / "metadata.json")
            self.assertEqual(
                (meta["end_reason"], meta["status"], meta["rows"]),
                ("stopped_off_runway", "complete", 4),
            )
            self.assertFalse(meta["evaluation"]["candidate_success"])

    def test_approved_lists_successes_of_the_current_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            other = copy.deepcopy(SCENARIO)
            other["runway_geometry"]["latitude"] += 0.01
            for name, outcome, scenario in [
                ("a", "success", SCENARIO),
                ("b", "unreviewed", SCENARIO),
                ("c", "success", other),
            ]:
                path = Path(tmp) / name
                path.mkdir()
                atomic_json(
                    path / "metadata.json", {"outcome": outcome, "scenario": scenario}
                )
            self.assertEqual(session.approved(tmp, SCENARIO), [str(Path(tmp) / "a")])

    def test_feedback_label_gate_and_review(self):
        from landing.core import evaluate, initial_state

        # 1200 m before the threshold, 200 ft above the nominal start-height path.
        near = {**SCENARIO, "initial": {**SCENARIO["initial"], "distance_m": 1200}}
        start = initial_state(near, 0)
        approach = {**state(), **start, "altitude_ft": 3 + 50 + 206 + 200}
        rows = [{"state": approach}] + [{"state": state(i + 1)} for i in range(9)]
        for row in rows[3:]:
            row["state"]["wow"] = 1
        rows[3]["state"]["pitch_deg"] = -2
        result = evaluate(rows, SCENARIO, NO_DAMAGE)
        text = "\n".join(feedback(rows, SCENARIO, result))
        self.assertIn("ft高すぎます", text)
        self.assertIn("フレア", text)
        rows[3]["state"]["pitch_deg"] = 0
        passed = evaluate(rows, SCENARIO, NO_DAMAGE)
        self.assertTrue(passed["candidate_success"])
        self.assertIn("全項目に合格", "\n".join(feedback(rows, SCENARIO, passed)))
        with tempfile.TemporaryDirectory() as tmp:
            for name, damage in [("good", NO_DAMAGE), ("broken", None)]:
                path = Path(tmp) / name
                write_episode(path, rows)
                meta = read_json(path / "metadata.json")
                atomic_json(
                    path / "metadata.json",
                    {
                        **meta,
                        "damage": damage,
                        "rows": len(rows),
                        "outcome": "unreviewed",
                    },
                )
            listing = session.review(Path(tmp))
            self.assertIn("2 flights, 1 passing every check", listing)
            self.assertIn("no_damage", listing)
            for name, error in [("broken", "every landing check"), ("good", None)]:
                argv = ["session", "label", str(Path(tmp) / name)]
                argv += ["--outcome", "success", "--note", "checked video"]
                with patch("sys.argv", argv), patch("builtins.print"):
                    if error:
                        with self.assertRaisesRegex(ValueError, error):
                            session.main()
                    else:
                        session.main()
            self.assertEqual(
                read_json(Path(tmp) / "good/metadata.json")["outcome"], "success"
            )

    def test_autoland_flies_sessions_as_reviewable_expert(self):
        from landing.autoland import Autoland
        from landing.core import ACTIONS, LIMITS, initial_state

        pilot = Autoland(SCENARIO)
        start = {**state(), **initial_state(SCENARIO, 0)}
        start["altitude_ft"] = initial_state(SCENARIO, 0)["altitude_ft"]
        command = pilot.predict(start)
        self.assertEqual(len(command), len(ACTIONS))
        self.assertTrue(all(lo <= v <= hi for v, (lo, hi) in zip(command, LIMITS)))
        self.assertEqual(pilot.phase, "approach")
        pilot.predict({**state(0.05), "wow": 1})
        self.assertEqual(pilot.phase, "rollout")
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
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(session, "prop", side_effect=props.__getitem__),
            patch.object(session, "pause"),
            patch.object(session, "configure_controls"),
            patch.object(session.socket, "socket") as sock,
            patch("builtins.print"),
        ):
            sock.return_value.recv.side_effect = [
                wire(state()),
                wire(state(0.05)),
                BlockingIOError(),
                KeyboardInterrupt(),
            ]
            args = argparse.Namespace(
                output=Path(tmp) / "expert",
                scenario=ROOT / "config/scenarios/c172p-rjtt-34r-approach.json",
                policy=None,
                autoland=True,
                timeout=1,
                seconds=10,
                video=False,
            )
            session.run(args)
            sock.return_value.sendto.assert_called_once()
            meta = read_json(args.output / "metadata.json")
            self.assertEqual(meta["source"], "expert")
            self.assertEqual(meta["policy_manifest"]["type"], "autoland")
            # Expert flights pass the same review gate as human ones.
            atomic_json(
                args.output / "metadata.json",
                {
                    **meta,
                    "outcome": "success",
                    "evaluation": {"candidate_success": True},
                },
            )
            with self.assertRaisesRegex(ValueError, "fewer than|training seed"):
                load_episodes([args.output])

    def test_comparison_rejects_diagnostic_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            atomic_json(path / "metadata.json", {"purpose": "integration_smoke_test"})
            with self.assertRaisesRegex(ValueError, "Diagnostic"):
                compare([path], [path / "other"])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                compare([path], [path])
            with self.assertRaises(ValueError):
                compare([], [])

    def test_nonfinite_capture_timestamp_and_atomic_write(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("landing.vision.urllib.request.urlopen") as response,
        ):
            response.return_value.__enter__.return_value.read.return_value = (
                b"\xff\xd8test\xff\xd9"
            )
            with self.assertRaisesRegex(ValueError, "timestamp"):
                capture(Path(tmp) / "bad.jpg", lambda: math.nan)
            self.assertFalse((Path(tmp) / "bad.jpg").exists())
            target = Path(tmp) / "metadata.json"
            atomic_json(target, {"note": "着陸"})
            with self.assertRaises(ValueError):
                atomic_json(target, {"bad": math.nan})
            self.assertEqual(read_json(target), {"note": "着陸"})
            self.assertEqual(
                file_digest(target), hashlib.sha256(target.read_bytes()).hexdigest()
            )


if __name__ == "__main__":
    unittest.main()
