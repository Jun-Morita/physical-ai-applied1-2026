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
START = core.initial_state(SCENARIO, 0)
NO_DAMAGE = dict.fromkeys(SCENARIO["damage_properties"], False)
START_PROPS = {
    "/sim/physical-ai/seed": 0,
    "/sim/physical-ai/start-profile": "nominal",
    "/position/latitude-deg": START["latitude"],
    "/position/longitude-deg": START["longitude"],
}


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
        values = list(map(float, core.action_packet([2, -2, 0, 0.4], 2).split(b",")))
        self.assertEqual(values[:3], [1, -1, 0])
        self.assertEqual(values[3:7], [0.4, 0.4, 0, 0])

    def test_landing_requires_contact_and_rollout(self):
        rows = [{"state": state(i)} for i in range(10)]
        self.assertFalse(core.evaluate(rows, SCENARIO)["candidate_success"])
        for row in rows[2:]:
            row["state"]["wow"] = 1
        self.assertTrue(core.evaluate(rows, SCENARIO, NO_DAMAGE)["candidate_success"])
        rows[-1]["state"]["longitude"] += 0.01
        self.assertFalse(core.evaluate(rows, SCENARIO, NO_DAMAGE)["candidate_success"])

    def test_seeded_start_is_reproducible_and_bounded(self):
        r, start = SCENARIO["runway_geometry"], SCENARIO["initial"]
        variation = SCENARIO["initial_variation"]
        nominal = core.initial_state(SCENARIO, 0)
        self.assertTrue(all(v == 0 for v in nominal["offsets"].values()))
        along, cross = core.runway_position(nominal, r)
        self.assertAlmostEqual(along, -start["distance_m"], places=3)
        self.assertAlmostEqual(cross, 0, places=3)
        starts = [core.initial_state(SCENARIO, seed) for seed in range(1, 50)]
        self.assertEqual(starts[0], core.initial_state(SCENARIO, 1))
        self.assertGreater(len({round(s["altitude_ft"], 6) for s in starts}), 40)
        for s in starts:
            self.assertTrue(
                all(abs(s["offsets"][k]) <= variation[k] for k in variation)
            )
            self.assertAlmostEqual(
                core.runway_position(s, r)[1], s["offsets"]["cross_m"], places=3
            )
        for bad in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                core.initial_state(SCENARIO, bad)

    def test_recorded_dagger_starts(self):
        import subprocess
        import sys

        r = SCENARIO["runway_geometry"]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # A rollout of a policy on a training seed that drifts right and low.
            rollout = tmp / "rollout-501"
            rollout.mkdir()
            rows = []
            for i in range(60):
                along = -3000 + i * 50
                s = state()
                pos = core.initial_state(SCENARIO, 0)
                h = math.radians(r["heading_deg"])
                cross = 80.0
                s.update(
                    latitude=r["latitude"]
                    + math.degrees(
                        (along * math.cos(h) - cross * math.sin(h)) / 6371000
                    ),
                    longitude=r["longitude"]
                    + math.degrees(
                        (along * math.sin(h) + cross * math.cos(h))
                        / (6371000 * math.cos(math.radians(r["latitude"])))
                    ),
                    altitude_ft=pos["altitude_ft"] * (-along + 400) / 4104 - 60,
                    agl_ft=200,
                    wow=0,
                    time=i,
                )
                rows.append({"state": s})
            (rollout / "samples.jsonl").write_text(
                "\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8"
            )
            meta = {
                "purpose": "rollout",
                "initial": {"seed": 501},
                "scenario": SCENARIO,
            }
            (rollout / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
            out = tmp / "starts.json"
            cmd = [sys.executable, str(ROOT / "scripts/dagger_starts.py"), str(rollout)]
            cmd += [
                "--first-seed",
                "601",
                "--output",
                str(out),
                "--distances",
                "2000",
                "1000",
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            starts = core.read_json(out)["starts"]
            self.assertEqual(sorted(starts), ["601", "602"])
            scenario = {
                **SCENARIO,
                "training_starts": {"dagger": {"states": str(out)}},
            }
            start = core.initial_state(scenario, 601, "dagger")
            along, cross = core.runway_position(start, r)
            self.assertAlmostEqual(along, -2000, delta=60)
            self.assertAlmostEqual(cross, 80, delta=1)
            with self.assertRaisesRegex(ValueError, "No recorded start"):
                core.initial_state(scenario, 603, "dagger")
            with self.assertRaisesRegex(ValueError, "nominal start"):
                core.initial_state(scenario, 1000, "dagger")
            # The start file is fixed once written; evaluation flights are refused.
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            meta["purpose"] = "evaluation"
            (rollout / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
            cmd[cmd.index(str(out))] = str(tmp / "other.json")
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)

    def test_training_start_profiles(self):
        r = SCENARIO["runway_geometry"]
        nominal = core.initial_state(SCENARIO, 0)
        g = core.approach_guidance({**state(), **nominal}, SCENARIO)
        for name, p in SCENARIO["training_starts"].items():
            if "states" in p:  # recorded DAgger starts: see test_recorded_dagger_starts
                continue
            s = core.initial_state(SCENARIO, 0, name)
            along, _ = core.runway_position(s, r)
            scale = p["distance_scale"]
            self.assertAlmostEqual(along, -SCENARIO["initial"]["distance_m"] * scale, 3)
            # Closer starts stay on the nominal start's path to the aiming point.
            near = core.approach_guidance({**state(), **s}, SCENARIO)
            self.assertAlmostEqual(near["height_error_ft"], g["height_error_ft"], 0)
            for seed in range(1, 30):
                offsets = core.initial_state(SCENARIO, seed, name)["offsets"]
                # Near-runway profiles keep full-size offsets to teach recovery.
                shrink = scale if p.get("scale_offsets", True) else 1.0
                for key in ("height_ft", "cross_m"):
                    limit = p["variation"][key] * shrink
                    self.assertLessEqual(abs(offsets[key]), limit)
        # Profiles widen data collection only: the task identity is unchanged.
        bare = {k: v for k, v in SCENARIO.items() if k != "training_starts"}
        self.assertEqual(core.scenario_hash(bare), core.scenario_hash(SCENARIO))
        with self.assertRaisesRegex(ValueError, "nominal start"):
            core.initial_state(SCENARIO, 1000, "wide")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            core.initial_state(SCENARIO, 1, "narrow")

    def test_guidance_damage_and_summary(self):
        start = {**state(), **core.initial_state(SCENARIO, 0)}
        start["altitude_ft"] = START["altitude_ft"]
        g = core.approach_guidance(start, SCENARIO)
        self.assertAlmostEqual(g["distance_m"], SCENARIO["initial"]["distance_m"], 0)
        # The preset starts on the 3 degree path to the 400 m aiming point (rounded to 1 ft).
        self.assertAlmostEqual(g["height_error_ft"], 0.4, 0)
        self.assertIn("経路より", session.guidance_line(start, SCENARIO))
        self.assertIn("滑走路上", session.guidance_line(state(), SCENARIO))
        rows = [{"state": state(i)} for i in range(10)]
        for row in rows[2:]:
            row["state"]["wow"] = 1
        broken = {"/fdm/jsbsim/gear/unit[0]/broken": True}
        result = core.evaluate(rows, SCENARIO, broken)
        self.assertFalse(result["checks"]["no_damage"])
        self.assertFalse(result["candidate_success"])
        # Unread damage is a failure, not a pass.
        self.assertFalse(core.evaluate(rows, SCENARIO)["checks"]["no_damage"])
        text = session.summary(
            {
                "status": "complete",
                "end_reason": "rollout_complete",
                "rows": 10,
                "control_resets": [],
                "evaluation": result,
            }
        )
        self.assertIn("NG  no_damage", text)

    def test_touchdown_quality_checks(self):
        def landing(index=2, **values):
            rows = [{"state": state(i)} for i in range(10)]
            for row in rows[2:]:
                row["state"]["wow"] = 1
            rows[index]["state"].update(values)
            return core.evaluate(rows, SCENARIO, NO_DAMAGE)

        self.assertTrue(all(landing(heading_deg=333)["checks"].values()))
        # Touchdown agl is 97 ft in this fixture: a 0.5 ft skip is not a bounce.
        skip = landing(4, wow=0, agl_ft=97.5)
        self.assertTrue(skip["checks"]["no_bounce"])
        self.assertAlmostEqual(skip["bounce_height_ft"], 0.5)
        # A floating flare reads a few mm/s of climb at contact; a real climb fails.
        self.assertTrue(landing(1, down_fps=-0.05)["checks"]["sink_rate"])
        for result, check in [
            (landing(pitch_deg=-2), "main_gear_first"),
            (landing(heading_deg=340), "heading"),
            (landing(1, down_fps=-1), "sink_rate"),
            (landing(4, wow=0, agl_ft=100), "no_bounce"),
        ]:
            self.assertFalse(result["checks"][check], check)
            self.assertFalse(result["candidate_success"])

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
                "initial": {"seed": 1},
                "evaluation": {"candidate_success": True},
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
            "/sim/physical-ai/scenario-hash": core.scenario_hash(SCENARIO),
            **START_PROPS,
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
            "scenario_hash": core.scenario_hash(SCENARIO),
            "layers": [
                {
                    "weight": [[0] * len(core.FEATURES) for _ in core.ACTIONS],
                    "bias": [0, 0, 0, 0.4],
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
                core.action_packet([0, 0, 0, 0.4], 1), ("127.0.0.1", 5502)
            )
            process.return_value.communicate.assert_called_once_with(b"q\n", timeout=15)
            pause.assert_called_with(True)
            meta = core.read_json(args.output / "metadata.json")
            self.assertEqual(
                (meta["source"], meta["status"], meta["rows"]), ("ai", "complete", 1)
            )
            self.assertEqual(session.rows_at(args.output)[0]["command"], [0, 0, 0, 0.4])

    def test_timeout_pauses_and_marks_incomplete(self):
        properties = {
            "/sim/aircraft": "c172p",
            "/sim/sceneryloaded": True,
            "/sim/signals/fdm-initialized": True,
            "/sim/freeze/master": True,
            "/sim/freeze/clock": True,
            "/sim/physical-ai/scenario-hash": core.scenario_hash(SCENARIO),
            **START_PROPS,
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
