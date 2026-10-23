"""Fly the scripted expert (or a trained MLP) offline in JSBSim with FlightGear's C172P.

This is a fast tuning check without the FlightGear window, not a substitute for
recording in FlightGear: Nasal scripts, terrain and some FlightGear-only
properties are approximated (flat runway, preset fuel and pilot weight).

    uv run --locked python scripts/offline_autoland.py \\
      --fgdata /mnt/c/Users/<user>/FlightGear/Downloads/fgdata_2024_1 --seeds 0 1 2

`--policy runs/mlp/<name>/policy.json` flies a trained MLP instead of the expert.
"""

import argparse
import math
import os
import re
import sys
from pathlib import Path

import jsbsim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from landing.autoland import Autoland  # noqa: E402
from landing.core import (  # noqa: E402
    Policy,
    evaluate,
    features,
    NOMINAL_START,
    initial_state,
    read_json,
)

# Values FlightGear sets through c172p-main.xml and Nasal (state-manager.nas).
FG_DEFAULTS = {
    "/engines/active-engine/damage_allowed": 0,
    "aero/coefficient/spiral-propwash-coeff": 0.25,
    "/controls/flight/adv-yaw-fact1": 0.2,
    "/controls/flight/adv-yaw-fact2": 0.2,
    "/pax/pilot/present": 1,
    "inertia/pointmass-weight-lbs[0]": 170,
    "inertia/pointmass-weight-lbs[1]": 0,
    "/engines/active-engine/running": 1,
    "/controls/engines/active-engine": 0,
    "/environment/temperature-degc": 15,
    "/environment/pressure-inhg": 29.92,
    "/controls/engines/current-engine/mixture": 1.0,
    "/systems/electrical/outputs/flaps": 28.0,
    "/engines/active-engine/oil-temperature-degf": 180,
    # FlightGear's default fuel load is not recorded; about 20 gal per wing tank.
    "propulsion/tank[0]/contents-lbs": 120,
    "propulsion/tank[1]/contents-lbs": 120,
    "/consumables/fuel/tank[0]/level-gal_us": 20,
    "/consumables/fuel/tank[1]/level-gal_us": 20,
    "/consumables/fuel/tank[0]/unusable-vol-gal": 1.5,
    "/consumables/fuel/tank[1]/unusable-vol-gal": 1.5,
}
MOORING_OUTPUTS = (
    "altitude-diff-ft force-down-lbs force-east-lbs force-north-lbs latitude-diff-ft"
    " longitude-diff-ft rope-force-lbs total-distance-ft total-force-lbs"
).split()
# JSBSim objects can throw from their destructors on FlightGear-only properties.
KEEP = []


def prepare(fgdata, work):
    """Link the C172P into a work tree, stubbing tie-downs that need FlightGear."""
    source = Path(fgdata) / "Aircraft"
    aircraft = work / "aircraft" / "c172p"
    systems = aircraft / "Systems"
    systems.mkdir(parents=True, exist_ok=True)
    for path in (source / "c172p").iterdir():
        if path.name != "Systems" and not (aircraft / path.name).exists():
            (aircraft / path.name).symlink_to(path)
    for directory in (source / "c172p/Systems", source / "Generic/JSBSim/Systems"):
        for path in directory.glob("*.xml"):
            target = systems / path.name
            if path.name != "mooring-jsbsim.xml" and not target.exists():
                target.symlink_to(path)
    functions = "".join(
        f'<fcs_function name="mooring/{n}"><function><value>0</value></function></fcs_function>'
        for n in MOORING_OUTPUTS
    )
    (systems / "mooring-jsbsim.xml").write_text(
        f'<?xml version="1.0"?><system name="mooring stub"><channel name="m">{functions}</channel></system>'
    )
    return work / "aircraft"


def make(aircraft, work, ic):
    """Load the model, creating FlightGear-only properties it asks for."""
    cache = work / "missing-properties.txt"
    names = cache.read_text().split() if cache.exists() else []
    while True:
        fdm = jsbsim.FGFDMExec(None)
        KEEP.append(fdm)
        fdm.set_debug_level(0)
        for name in names:
            fdm.set_property_value(name, 0.0)
        fdm.set_aircraft_path(str(aircraft))
        fdm.set_engine_path(str(aircraft / "c172p/Engines"))
        fdm.set_systems_path(str(aircraft / "c172p/Systems"))
        try:
            if not fdm.load_model("c172p"):
                raise RuntimeError("Could not load the C172P model")
            for key, value in {**FG_DEFAULTS, **ic}.items():
                fdm[key] = value
            fdm.run_ic()
            fdm.run()
            return fdm
        except Exception as exc:
            match = re.search(r"property (\S+) does not exist", str(exc))
            if not match:
                raise
            names.append(match.group(1))
            cache.write_text("\n".join(names))


def telemetry(fdm, command):
    deg = math.degrees(1)
    s = {
        "time": fdm.get_sim_time(),
        "latitude": fdm["position/lat-geod-deg"],
        "longitude": fdm["position/long-gc-deg"],
        "altitude_ft": fdm["position/h-sl-ft"],
        "agl_ft": fdm["position/h-agl-ft"],
        "airspeed_kt": fdm["velocities/vc-kts"],
        "groundspeed_kt": fdm["velocities/vg-fps"] / 1.68781,
        "north_fps": fdm["velocities/v-north-fps"],
        "east_fps": fdm["velocities/v-east-fps"],
        "down_fps": fdm["velocities/v-down-fps"],
        "roll_deg": fdm["attitude/phi-deg"],
        "pitch_deg": fdm["attitude/theta-deg"],
        "heading_deg": fdm["attitude/psi-deg"],
        "roll_rate": fdm["velocities/p-rad_sec"] * deg,
        "pitch_rate": fdm["velocities/q-rad_sec"] * deg,
        "yaw_rate": fdm["velocities/r-rad_sec"] * deg,
        "wow": float(fdm["gear/unit[1]/WOW"]),
    }
    s.update(zip(("aileron", "elevator", "rudder", "throttle"), command))
    s.update(throttle1=0, throttle2=0, throttle3=0, brake_left=0, brake_right=0)
    s.update(trim=0, flaps=fdm["fcs/flap-cmd-norm"], gear=1, paused=0)
    s.update(clock_paused=0, crashed=0)
    return s


def fly(scenario, seed, aircraft, work, model=None, limit=240, profile=NOMINAL_START):
    start = initial_state(scenario, seed, profile)
    controls = scenario["controls"]
    fdm = make(
        aircraft,
        work,
        {
            "ic/h-sl-ft": start["altitude_ft"],
            "ic/terrain-elevation-ft": scenario["runway_geometry"]["elevation_ft"],
            "ic/vc-kts": start["airspeed_kt"],
            "ic/psi-true-deg": start["heading_deg"],
            "ic/lat-geod-deg": start["latitude"],
            "ic/long-gc-deg": start["longitude"],
            **({"ic/theta-deg": start["pitch_deg"]} if "pitch_deg" in start else {}),
            "fcs/flap-cmd-norm": controls["flaps"],
            "/controls/flight/flaps": controls["flaps"],
            "fcs/mixture-cmd-norm[0]": 1,
            "settings/damage": 1,
        },
    )
    fdm["propulsion/set-running"] = -1
    if model is None:
        predict = Autoland(scenario).predict
    else:
        policy = Policy(model)

        def predict(state):
            return policy.predict(features(state, scenario))

    every = round(0.05 / fdm.get_delta_t())
    command = [0.0, 0.0, 0.0, controls["throttle"]]
    rows, step, touchdown = [], 0, None
    while fdm.get_sim_time() < limit:
        if step % every == 0:
            state = telemetry(fdm, command)
            command = predict(state)
            state.update(zip(("aileron", "elevator", "rudder", "throttle"), command))
            rows.append({"state": state})
            if touchdown is None and state["wow"] > 0.5:
                touchdown = state["time"]
            rollout = scenario["evaluation"]["min_rollout_seconds"]
            if touchdown is not None and state["time"] - touchdown >= rollout:
                break
        aileron, elevator, rudder, throttle = command
        fdm["fcs/aileron-cmd-norm"] = aileron
        fdm["fcs/elevator-cmd-norm"] = elevator
        # FlightGear passes the rudder to JSBSim with the opposite sign.
        fdm["fcs/rudder-cmd-norm"] = -rudder
        fdm["/controls/flight/rudder"] = rudder
        # The C172P enables nose wheel steering from these FlightGear gear outputs.
        fdm["/gear/gear[0]/compression-norm"] = fdm["gear/unit[0]/compression-ft"]
        fdm["/gear/gear[0]/rollspeed-ms"] = fdm["gear/unit[0]/wheel-speed-fps"] * 0.3048
        fdm["fcs/throttle-cmd-norm[0]"] = throttle
        fdm["/controls/engines/current-engine/throttle"] = throttle
        fdm["propulsion/magneto_cmd"] = 3
        fdm.run()
        step += 1
    damage = {
        f"/fdm/jsbsim/gear/unit[{i}]/broken": bool(fdm[f"gear/unit[{i}]/broken"])
        for i in range(3)
    }
    return evaluate(rows, scenario, damage)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fgdata", required=True, type=Path)
    p.add_argument(
        "--scenario",
        type=Path,
        default=ROOT / "config/scenarios/c172p-rjtt-34r-approach.json",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--work", type=Path, default=ROOT / "runs/offline-c172p")
    p.add_argument("--policy", type=Path, help="Trained MLP policy.json")
    p.add_argument("--start-profile", default=NOMINAL_START)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    aircraft = prepare(a.fgdata, a.work.resolve())
    scenario = read_json(a.scenario)
    model = read_json(a.policy) if a.policy else None
    passed = 0
    for seed in a.seeds:
        result = fly(
            scenario, seed, aircraft, a.work.resolve(), model, profile=a.start_profile
        )
        failed = [k for k, v in result.get("checks", {}).items() if not v]
        passed += result["candidate_success"]
        metrics = " ".join(
            f"{k.replace('touchdown_', '')}={result[k]:.1f}"
            for k in (
                "touchdown_along_m",
                "touchdown_cross_m",
                "pre_touchdown_sink_fps",
                "touchdown_pitch_deg",
                "bounce_height_ft",
                "touchdown_airspeed_kt",
            )
            if k in result
        )
        verdict = "PASS" if result["candidate_success"] else "FAIL"
        print(
            f"seed {seed}: {verdict} {metrics} {failed or result.get('reason', '')}",
            flush=True,
        )
    print(
        f"{passed}/{len(a.seeds)} passed (offline JSBSim, not FlightGear)", flush=True
    )
    # Skip JSBSim destructors, which can throw on FlightGear-only properties.
    os._exit(0)


if __name__ == "__main__":
    main()
