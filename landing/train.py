"""Train an aircraft/scenario-specific behavioral cloning baseline with uv."""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from .core import ACTIONS, FEATURES, SCHEMA, Policy, features, scenario_hash
from .data import command_action, demonstration_action, episode_digest, reviewed_episode


def load_episodes(paths):
    episodes = []
    identity = None
    for path in paths:
        path = Path(path)
        meta, rows = reviewed_episode(path)
        scenario = meta["scenario"]
        key = scenario_hash(scenario)
        if identity is not None and identity != key:
            raise ValueError("Train one aircraft/scenario per policy")
        identity = key
        x, y = [], []
        if meta["source"] == "expert":
            # The expert's command for each telemetry row is the label.
            for row in rows:
                if row.get("command") is not None:
                    x.append(features(row["state"], scenario))
                    y.append(command_action(row, scenario))
        else:
            # Human controls take effect on the next sample.
            for previous, current in zip(rows, rows[1:]):
                a, b = previous["state"], current["state"]
                dt = b["time"] - a["time"]
                if not 0 < dt <= 1.5 / scenario["rate_hz"]:
                    continue
                if any(s["paused"] or s["clock_paused"] for s in (a, b)):
                    continue
                demonstration_action(a, scenario)
                x.append(features(a, scenario))
                y.append(demonstration_action(b, scenario))
        if len(x) < 20:
            raise ValueError(f"{path}: fewer than 20 valid adjacent pairs")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Non-finite training data")
        episodes.append(
            {
                "path": str(path),
                "hash": episode_digest(rows),
                "x": x,
                "y": y,
                "scenario": scenario,
                "source": meta["source"],
            }
        )
    if len({e["hash"] for e in episodes}) != len(episodes):
        raise ValueError("Duplicate episodes would leak across the split")
    return episodes


def train(episodes, output, epochs=100, seed=42, device="auto"):
    if len(episodes) < 3:
        raise ValueError("At least three separate reviewed flights are required")
    output = Path(output)
    if output.exists():
        raise ValueError("Output already exists; choose a new model directory")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    order = list(range(len(episodes)))
    random.shuffle(order)
    nval = max(1, round(len(order) * 0.2))
    val, fit = order[:nval], order[nval:]

    def array(indices, key):
        return np.asarray(
            [row for i in indices for row in episodes[i][key]], dtype=np.float32
        )

    x, y, vx, vy = array(fit, "x"), array(fit, "y"), array(val, "x"), array(val, "y")
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-5)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    x, y, vx, vy = [
        torch.tensor(v, device=device)
        for v in ((x - mean) / std, y, (vx - mean) / std, vy)
    ]
    net = torch.nn.Sequential(
        torch.nn.Linear(len(FEATURES), 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, len(ACTIONS)),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, saved, history = float("inf"), None, []
    for epoch in range(epochs):
        net.train()
        for batch in torch.randperm(len(x), device=device).split(256):
            loss = torch.nn.functional.mse_loss(net(x[batch]), y[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            score = torch.nn.functional.mse_loss(net(vx), vy).item()
        history.append({"epoch": epoch + 1, "validation_mse": score})
        if score < best:
            best, saved = score, copy.deepcopy(net.state_dict())
    if saved is None:
        raise ValueError("Training did not produce a finite model")
    net.load_state_dict(saved)
    model = {
        "schema": SCHEMA,
        "features": FEATURES,
        "actions": ACTIONS,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "scenario": episodes[0]["scenario"],
        "scenario_hash": scenario_hash(episodes[0]["scenario"]),
        "layers": [
            {
                "weight": layer.weight.detach().cpu().tolist(),
                "bias": layer.bias.detach().cpu().tolist(),
            }
            for layer in net
            if isinstance(layer, torch.nn.Linear)
        ],
    }
    Policy(model)
    output.mkdir(parents=True)
    (output / "policy.json").write_text(json.dumps(model, indent=2))
    report = {
        "seed": seed,
        "device": device,
        "validation_mse": best,
        "history": history,
        "train_episodes": [
            {"path": episodes[i]["path"], "hash": episodes[i]["hash"]} for i in fit
        ],
        "validation_episodes": [
            {"path": episodes[i]["path"], "hash": episodes[i]["hash"]} for i in val
        ],
        "train_pairs": len(x),
        "validation_pairs": len(vx),
        "target": {
            "expert": "expert command for the same telemetry sample",
            "human": "next telemetry sample controls",
        },
        "sources": sorted({episodes[i].get("source", "human") for i in fit + val}),
        "limitation": "Validation control error does not establish closed-loop landing success.",
    }
    (output / "training.json").write_text(json.dumps(report, indent=2))
    return model, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episodes", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a = p.parse_args()
    if a.epochs < 1:
        p.error("--epochs must be positive")
    _, report = train(load_episodes(a.episodes), a.output, a.epochs, a.seed, a.device)
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
