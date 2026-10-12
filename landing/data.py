"""Shared admission checks for human demonstrations and visual sample alignment."""

import math
from bisect import bisect_left
from pathlib import Path

from .core import (
    ACTIONS,
    EVALUATION_SEED_START,
    FIELDS,
    LIMITS,
    SCHEMA,
    digest,
    scenario_hash,
    features,
)
from .storage import file_digest, read_json, rows_at


def validate_rows(rows):
    previous = None
    for row in rows:
        state = row.get("state", {})
        if any(
            key not in state
            or not isinstance(state[key], (float, int))
            or not math.isfinite(state[key])
            for key in FIELDS
        ):
            raise ValueError("Incomplete or non-finite state")
        if previous is not None and state["time"] <= previous:
            raise ValueError("Non-increasing simulation timestamps")
        previous = state["time"]


def reviewed_episode(path):
    path = Path(path)
    meta = read_json(path / "metadata.json")
    if meta.get("schema", SCHEMA) != SCHEMA:
        raise ValueError("Unsupported episode schema")
    # Demonstrations come from the human pilot or the scripted expert landing.
    if meta.get("source") not in ("human", "expert") or (
        meta.get("status"),
        meta.get("outcome"),
    ) != ("complete", "success"):
        raise ValueError(
            f"{path}: only reviewed, completed human or expert successes are allowed"
        )
    if meta.get("purpose", "evaluation") != "evaluation":
        raise ValueError("Diagnostic episodes cannot be training demonstrations")
    # Flights recorded before the task-only hash stored the full scenario digest.
    stored = meta.get("scenario_hash", scenario_hash(meta["scenario"]))
    if stored not in (scenario_hash(meta["scenario"]), digest(meta["scenario"])):
        raise ValueError("Scenario hash mismatch")
    if not (meta.get("evaluation") or {}).get("candidate_success"):
        raise ValueError(f"{path}: demonstrations must pass every landing check")
    seed = (meta.get("initial") or {}).get("seed")
    if not isinstance(seed, int) or not 0 <= seed < EVALUATION_SEED_START:
        raise ValueError(f"{path}: demonstrations need a training seed below 1000")
    rows = rows_at(path)
    validate_rows(rows)
    if meta.get("control_resets"):
        # Before 2026-10-12 a sample of the logged C172P override (flaps 1.0,
        # throttle 0.75) could slip in during a paused capture; it is not a
        # demonstration. Other fixed-control changes still reject the episode.
        flaps = meta["scenario"]["controls"]["flaps"]
        rows = [r for r in rows if abs(r["state"]["flaps"] - flaps) <= 0.02]
    if any(row["state"]["crashed"] for row in rows):
        raise ValueError("Crashed episode cannot be a successful demonstration")
    for row in rows:
        demonstration_action(row["state"], meta["scenario"])
    return meta, rows


def demonstration_action(state, scenario):
    controls = scenario["controls"]
    if (
        state["gear"] < 0.5
        or abs(state["flaps"] - controls["flaps"]) > 0.02
        or abs(state["trim"] - controls["trim"]) > 0.02
        or max(state["brake_left"], state["brake_right"]) > 0.02
    ):
        raise ValueError("Fixed flaps/gear/trim/brake configuration changed")
    if any(
        abs(state[f"throttle{i}"] - state["throttle"]) > 0.02
        for i in range(1, scenario["engine_count"])
    ):
        raise ValueError("Asymmetric thrust is unsupported")
    values = [state[key] for key in ACTIONS]
    if any(
        not math.isfinite(v) or not lo <= v <= hi for v, (lo, hi) in zip(values, LIMITS)
    ):
        raise ValueError("Invalid demonstration action")
    return values


def command_action(row, scenario):
    """The expert's own output for this row's telemetry: an exact, delay-free label."""
    demonstration_action(row["state"], scenario)
    values = row.get("command")
    if not isinstance(values, list) or len(values) != len(ACTIONS):
        raise ValueError("Expert row without a command")
    if any(
        not math.isfinite(v) or not lo <= v <= hi for v, (lo, hi) in zip(values, LIMITS)
    ):
        raise ValueError("Invalid expert command")
    return values


def episode_digest(rows, visual=False):
    # Ignore incidental wall-clock logs and image paths, so copied flights cannot leak across splits.
    values = []
    for row in rows:
        item = {"state": row["state"]}
        image = row.get("image")
        if visual and image:
            item["image"] = {
                key: image.get(key)
                for key in ("sha256", "sim_before", "sim_after", "view_number")
            }
        values.append(item)
    return digest(values)


def training_episodes(paths, manifest):
    episodes, seen = [], set()
    for path in paths:
        path = Path(path)
        meta, rows = reviewed_episode(path)
        if (
            scenario_hash(meta["scenario"]) != manifest["scenario_hash"]
            or meta.get("instruction") != manifest["instruction"]
        ):
            raise ValueError("Scenario/instruction mismatch")
        h = episode_digest(rows, visual=True)
        if h in seen:
            raise ValueError("Duplicate flight")
        seen.add(h)
        times = [row["state"]["time"] for row in rows]
        samples, views = [], set()
        # A pause anywhere between observation and target invalidates their alignment.
        paused = [0]
        for row in rows:
            paused.append(
                paused[-1]
                + bool(row["state"]["paused"] or row["state"]["clock_paused"])
            )
        for i, row in enumerate(rows):
            image = row.get("image")
            if not image:
                continue
            view = image.get("view_number")
            if (
                not isinstance(view, (int, float))
                or not math.isfinite(view)
                or view < 0
            ):
                raise ValueError("Missing or invalid camera view")
            views.add(view)
            image_path = (path / image["file"]).resolve()
            if meta["source"] == "expert":
                # Expert captures pause the simulation: image, state and the
                # command computed from them belong to one instant.
                if not image_path.is_relative_to(path.resolve()):
                    raise ValueError("Image path escapes episode")
                if file_digest(image_path) != image["sha256"]:
                    raise ValueError("Image content hash mismatch")
                samples.append(
                    {
                        "observation": features(row["state"], meta["scenario"]),
                        "action": command_action(row, meta["scenario"]),
                        "image": str(image_path),
                    }
                )
                continue
            before, after = image["sim_before"], image["sim_after"]
            if not all(math.isfinite(v) for v in (before, after)):
                raise ValueError("Non-finite image timestamp")
            s = row["state"]
            if not 0 <= after - before <= 0.25 or not 0 <= before - s["time"] <= 0.15:
                continue
            j = bisect_left(times, after, lo=i + 1)
            if j == len(rows) or times[j] - after > 0.1 or paused[j + 1] != paused[i]:
                continue
            following = rows[j]["state"]
            demonstration_action(s, meta["scenario"])
            action = demonstration_action(following, meta["scenario"])
            if not image_path.is_relative_to(path.resolve()):
                raise ValueError("Image path escapes episode")
            if file_digest(image_path) != image["sha256"]:
                raise ValueError("Image content hash mismatch")
            samples.append(
                {
                    "observation": features(s, meta["scenario"]),
                    "action": action,
                    "image": str(image_path),
                }
            )
        if len(samples) < 10:
            raise ValueError(f"{path}: fewer than 10 aligned image/action samples")
        if len(views) != 1:
            raise ValueError("Camera view changed during flight")
        episodes.append(
            {
                "path": str(path),
                "hash": h,
                "samples": samples,
                "view": next(iter(views)),
            }
        )
    if len(episodes) < 3:
        raise ValueError("At least three separate reviewed image flights required")
    if len({ep["view"] for ep in episodes}) != 1:
        raise ValueError("Camera views differ between flights")
    return episodes
