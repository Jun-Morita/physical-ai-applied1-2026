"""Sensitivity of the fine-tuned SmolVLA to each input, along recorded flights.

    uv run --project environments/vla --locked python scripts/analyze_sensitivity.py \\
      --model runs/vla/c172p-expert-wide-norm-15000 \\
      --output runs/analysis/sensitivity.json runs/eval/vla-expert-wide-norm-15000-100?

Applies the pilot-model analysis of Suzuki et al. (JSME 2002,
doi:10.1299/jsmetld.2002.11.373) to the VLA. For each recorded frame, each state
input is moved by +/- delta (0.1 training standard deviations) on its own and the
central difference (eq. 4 of the paper) gives the action change per standard
deviation. The image is tested by swapping it for a uniform gray frame and for
the camera view from the other end of the approach (start <-> touchdown); the
same swap of the state alone gives the scale to compare it with. All
rows of a frame share one flow-matching noise sample, so differences come from
the input alone; a repeated unchanged row measures the numerical floor.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from landing.core import ACTIONS, FEATURES, features  # noqa: E402
from landing.vla import load_policy, predict, state_normalization  # noqa: E402

DELTA = 0.1  # in training standard deviations
SKIP = {"wow"}  # binary: a small step has no meaning


def touchdown_time(rows):
    for r in rows:
        if r["state"]["wow"] > 0:
            return r["state"]["time"]
    return None


def analyze(policy, tokenizer, manifest, flight, every, device):
    rows = [json.loads(line) for line in open(flight / "samples.jsonl")]
    scenario = manifest["scenario"]
    td = touchdown_time(rows)
    _, std = state_normalization(manifest)
    used = manifest["features"]
    inputs = [i for i, name in enumerate(FEATURES) if name in used and name not in SKIP]
    recorded = [r for r in rows if r.get("image")]
    frames = recorded[::every]
    # Views from the start of the approach and the last airborne frame.
    airborne = [r for r in recorded if r["state"]["wow"] == 0]
    far, near = (
        (
            Image.open(flight / r["image"]["file"]).convert("RGB"),
            features(r["state"], scenario),
        )
        for r in (airborne[0], airborne[-1])
    )
    out = []
    for n, r in enumerate(frames):
        x = features(r["state"], scenario)
        image = Image.open(flight / r["image"]["file"]).convert("RGB")
        obs, ims = [x], [image]
        for i in inputs:
            for sign in (1, -1):
                y = list(x)
                y[i] += sign * DELTA * std[used.index(FEATURES[i])]
                obs.append(y)
                ims.append(image)
        other_image, other_state = near if r["state"]["agl_ft"] > 100 else far
        gray = Image.new("RGB", image.size, (128, 128, 128))
        obs += [x, x, x, other_state]
        ims += [image, gray, other_image, image]
        noise = torch.randn(
            (1, policy.config.chunk_size, policy.config.max_action_dim),
            generator=torch.Generator().manual_seed(n),
        ).to(device)
        policy.model.sample_noise = lambda shape, dev: noise.expand(shape).to(dev)
        policy.reset()
        acts = predict(policy, tokenizer, manifest, obs, ims, device)
        base = acts[0]
        sens = {}
        for k, i in enumerate(inputs):
            plus, minus = acts[1 + 2 * k], acts[2 + 2 * k]
            sens[FEATURES[i]] = [(p - m) / (2 * DELTA) for p, m in zip(plus, minus)]
        out.append(
            {
                "time": r["state"]["time"],
                "to_touchdown": None if td is None else r["state"]["time"] - td,
                "agl_ft": r["state"]["agl_ft"],
                "wow": r["state"]["wow"],
                "action": base,
                "sensitivity": sens,
                "repeat_change": [g - b for g, b in zip(acts[-4], base)],
                "gray_image_change": [g - b for g, b in zip(acts[-3], base)],
                "swapped_image_change": [g - b for g, b in zip(acts[-2], base)],
                "swapped_state_change": [g - b for g, b in zip(acts[-1], base)],
            }
        )
        print(f"{flight.name} {n + 1}/{len(frames)}", flush=True)
    return {
        "flight": flight.name,
        "path": str(flight),
        "touchdown_time": td,
        "frames": out,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("flights", type=Path, nargs="+")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--every", type=int, default=2, help="Use every Nth recorded frame")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    policy, tokenizer, manifest = load_policy(a.model, a.device)
    result = {
        "model": str(a.model),
        "delta_std": DELTA,
        "actions": ACTIONS,
        "flights": [
            analyze(policy, tokenizer, manifest, f, a.every, a.device)
            for f in a.flights
        ],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result), encoding="utf-8")
    print("saved", a.output)


if __name__ == "__main__":
    main()
