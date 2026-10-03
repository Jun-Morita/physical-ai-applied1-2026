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

from test_pipeline import ROOT, SCENARIO, state, wire

from landing import session
from landing.compare import compare
from landing.core import digest
from landing.data import episode_digest, training_episodes
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
                    [path], {"scenario_hash": digest(SCENARIO), "instruction": "land"}
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
            "/sim/physical-ai/scenario-hash": digest(SCENARIO),
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
