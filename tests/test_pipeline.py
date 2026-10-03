import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from landing import core, session
from landing.train import load_episodes, train

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = core.read_json(ROOT / "config/scenarios/c172p-rjtt-34r-approach.json")


def state(t=0):
    s = dict.fromkeys(core.FIELDS, 0.0)
    s.update(
        time=t,
        latitude=SCENARIO["runway_geometry"]["latitude"],
        longitude=SCENARIO["runway_geometry"]["longitude"],
        heading_deg=330.02,
        altitude_ft=100,
        agl_ft=97,
        airspeed_kt=75,
        gear=1,
        flaps=SCENARIO["controls"]["flaps"],
        down_fps=3,
    )
    return s


def wire(s):
    return (",".join(str(s[k]) for k in core.FIELDS) + "\n").encode()


class PipelineTests(unittest.TestCase):
    def test_wire_and_geometry(self):
        s = state()
        self.assertEqual(core.packet(wire(s)), s)
        self.assertEqual(core.runway_position(s, SCENARIO["runway_geometry"]), (0, 0))
        for bad in (b"1,2", wire({**s, "time": math.nan})):
            with self.assertRaises(ValueError):
                core.packet(bad)
        values = list(
            map(float, core.action_packet([2, -2, 0, 0.4, 0, 0, 0], 2).split(b","))
        )
        self.assertEqual(values[:3], [1, -1, 0])
        self.assertEqual(values[3:7], [0.4, 0.4, 0, 0])

    def test_landing_requires_contact_and_rollout(self):
        rows = [{"state": state(i)} for i in range(10)]
        self.assertFalse(core.evaluate(rows, SCENARIO)["candidate_success"])
        for row in rows[2:]:
            row["state"]["wow"] = 1
        self.assertTrue(core.evaluate(rows, SCENARIO)["candidate_success"])
        rows[-1]["state"]["longitude"] += 0.01
        self.assertFalse(core.evaluate(rows, SCENARIO)["candidate_success"])

    def test_training_split_and_export(self):
        # Synthetic inputs exercise implementation only; never saved as human demonstrations.
        rng = np.random.default_rng(7)
        episodes = [
            {
                "path": f"synthetic-{i}",
                "hash": str(i),
                "scenario": SCENARIO,
                "x": rng.normal(i, 1, (32, len(core.FEATURES))).tolist(),
                "y": rng.uniform(-0.2, 0.2, (32, len(core.ACTIONS))).tolist(),
            }
            for i in range(3)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            model, report = train(episodes, Path(tmp) / "model", epochs=2, device="cpu")
            self.assertFalse(
                set(e["path"] for e in report["train_episodes"])
                & set(e["path"] for e in report["validation_episodes"])
            )
            train_indices = [
                int(e["path"].split("-")[1]) for e in report["train_episodes"]
            ]
            expected = np.asarray(
                [row for i in train_indices for row in episodes[i]["x"]],
                dtype=np.float32,
            ).mean(0)
            np.testing.assert_allclose(model["mean"], expected)
            observation = episodes[0]["x"][0]
            x = torch.tensor(
                [
                    (v - m) / s
                    for v, m, s in zip(observation, model["mean"], model["std"])
                ]
            )
            for i, layer in enumerate(model["layers"]):
                x = torch.nn.functional.linear(
                    x, torch.tensor(layer["weight"]), torch.tensor(layer["bias"])
                )
                if i < len(model["layers"]) - 1:
                    x = torch.relu(x)
            np.testing.assert_allclose(
                core.Policy(model).predict(observation),
                core.clamp_actions(x.tolist()),
                atol=1e-6,
            )

    def test_dataset_review_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            meta = {
                "source": "ai",
                "outcome": "success",
                "status": "complete",
                "scenario": SCENARIO,
            }
            (path / "metadata.json").write_text(json.dumps(meta))
            rows = [{"state": state(i * 0.05)} for i in range(30)]
            (path / "samples.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            with self.assertRaisesRegex(ValueError, "human"):
                load_episodes([path])
            meta["source"] = "human"
            (path / "metadata.json").write_text(json.dumps(meta))
            self.assertEqual(len(load_episodes([path])[0]["x"]), 29)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_episodes([path, path])
            rows[10]["state"]["flaps"] = 0
            (path / "samples.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            with self.assertRaisesRegex(ValueError, "flaps"):
                load_episodes([path])

    def test_ai_session_sends_controls_and_finalizes_video(self):
        properties = {
            "/sim/aircraft": "c172p",
            "/sim/sceneryloaded": True,
            "/sim/signals/fdm-initialized": True,
            "/sim/freeze/master": True,
            "/sim/freeze/clock": True,
            "/sim/physical-ai/scenario-hash": core.digest(SCENARIO),
            "/sim/input/no-joystick-input": True,
            "/controls/flight/flaps": SCENARIO["controls"]["flaps"],
            "/controls/gear/gear-down-command": True,
        }
        model = {
            "schema": core.SCHEMA,
            "features": core.FEATURES,
            "actions": core.ACTIONS,
            "mean": [0] * len(core.FEATURES),
            "std": [1] * len(core.FEATURES),
            "scenario_hash": core.digest(SCENARIO),
            "layers": [
                {
                    "weight": [[0] * len(core.FEATURES) for _ in core.ACTIONS],
                    "bias": [0, 0, 0, 0.4, 0, 0, 0],
                }
            ],
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(session, "prop", side_effect=properties.__getitem__),
            patch.object(session, "pause") as pause,
            patch.object(session, "configure_controls"),
            patch.object(session.socket, "socket") as sock,
            patch.object(session.subprocess, "Popen") as process,
            patch.object(session.time, "sleep"),
            patch("builtins.print"),
        ):
            root = Path(tmp)
            (root / "policy.json").write_text(json.dumps(model))
            sock.return_value.recv.side_effect = [
                wire(state()),
                wire(state(0.05)),
                BlockingIOError(),
                KeyboardInterrupt(),
            ]
            process.return_value.poll.return_value = None
            process.return_value.returncode = 0
            args = argparse.Namespace(
                scenario=ROOT / "config/scenarios/c172p-rjtt-34r-approach.json",
                policy=root / "policy.json",
                output=root / "episode",
                timeout=1,
                seconds=10,
                video=True,
                ffmpeg="ffmpeg",
            )
            session.run(args)
            sock.return_value.sendto.assert_called_once_with(
                core.action_packet([0, 0, 0, 0.4, 0, 0, 0], 1), ("127.0.0.1", 5502)
            )
            process.return_value.communicate.assert_called_once_with(b"q\n", timeout=15)
            pause.assert_called_with(True)
            meta = core.read_json(args.output / "metadata.json")
            self.assertEqual(
                (meta["source"], meta["status"], meta["rows"]), ("ai", "complete", 1)
            )
            self.assertEqual(
                session.rows_at(args.output)[0]["command"], [0, 0, 0, 0.4, 0, 0, 0]
            )

    def test_timeout_pauses_and_marks_incomplete(self):
        properties = {
            "/sim/aircraft": "c172p",
            "/sim/sceneryloaded": True,
            "/sim/signals/fdm-initialized": True,
            "/sim/freeze/master": True,
            "/sim/freeze/clock": True,
            "/sim/physical-ai/scenario-hash": core.digest(SCENARIO),
            "/sim/input/no-joystick-input": False,
            "/controls/flight/flaps": SCENARIO["controls"]["flaps"],
            "/controls/gear/gear-down-command": True,
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(session, "prop", side_effect=properties.__getitem__),
            patch.object(session, "pause") as pause,
            patch.object(session, "configure_controls"),
            patch.object(session.socket, "socket") as sock,
            patch("builtins.print"),
        ):
            sock.return_value.recv.side_effect = TimeoutError("no telemetry")
            args = argparse.Namespace(
                scenario=ROOT / "config/scenarios/c172p-rjtt-34r-approach.json",
                policy=None,
                output=Path(tmp) / "episode",
                timeout=0.01,
                seconds=1,
                video=False,
            )
            with self.assertRaises(TimeoutError):
                session.run(args)
            self.assertEqual(
                core.read_json(args.output / "metadata.json")["status"], "incomplete"
            )
            pause.assert_called_with(True)
            self.assertFalse(
                any(call.args == (False,) for call in pause.call_args_list)
            )


if __name__ == "__main__":
    unittest.main()
