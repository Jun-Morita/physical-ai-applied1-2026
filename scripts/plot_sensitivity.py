"""Report figure: what the fine-tuned SmolVLA responds to, and its flare.

    uv run --with matplotlib==3.10.3 python scripts/plot_sensitivity.py \\
      runs/analysis/sensitivity.json --model runs/vla/c172p-expert-wide-norm-15000 \\
      --font /mnt/c/Windows/Fonts/BIZ-UDGothicR.ttc \\
      --output docs/report/figures/fig4-sensitivity.png

Left: elevator sensitivity (central difference, per training standard deviation)
to height, sink rate and pitch along the approach, in the style of the 2002
pilot-model paper. Right: elevator before touchdown, demonstrations used for
training vs the evaluation flights. Both are medians in time bins relative to
touchdown. Prints the image-swap vs state-swap comparison for the report table.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

MUTED, INK, AFTER = "#8a8984", "#52514e", "#2a78d6"
INPUTS = [  # validated categorical set, fixed order
    ("高さ", "height_m", "#1baf7a"),
    ("降下率", "down_mps", "#eda100"),
    ("ピッチ角", "pitch_deg", "#6250d6"),
]
ELEVATOR = 1


def binned(points, start, end, width):
    bins = {}
    for t, v in points:
        if t is not None and start <= t < end:
            bins.setdefault(int((t - start) // width), []).append(v)
    keys = sorted(bins)
    return [start + (k + 0.5) * width for k in keys], [
        statistics.median(bins[k]) for k in keys
    ]


def elevator_trace(directory):
    rows = [json.loads(line) for line in open(directory / "samples.jsonl")]
    td = next((r["state"]["time"] for r in rows if r["state"]["wow"] > 0), None)
    if td is None:
        return []
    return [(r["state"]["time"] - td, r["state"]["elevator"]) for r in rows]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("analysis", type=Path)
    p.add_argument("--model", type=Path, required=True, help="For its training list")
    p.add_argument("--font", type=Path, help="Japanese font file")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.font:
        font_manager.fontManager.addfont(str(a.font))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(a.font)
        ).get_name()
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": MUTED})
    data = json.loads(a.analysis.read_text(encoding="utf-8"))
    frames = [f for flight in data["flights"] for f in flight["frames"]]

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(8, 3.1), gridspec_kw={"width_ratios": [3, 2]}
    )
    for label, name, color in INPUTS:
        xs, ys = binned(
            [
                (f["to_touchdown"], abs(f["sensitivity"][name][ELEVATOR]))
                for f in frames
            ],
            -120,
            4,
            4,
        )
        left.plot(xs, ys, color=color, linewidth=2, label=label)
        left.text(xs[-1] + 2, ys[-1], label, color=INK, fontsize=8.5, va="center")
    left.set_xlim(-120, 22)
    left.set_ylim(bottom=0)
    left.set_xlabel("接地までの時間（秒）")
    left.set_ylabel("エレベーターの感度")
    left.set_title("入力ごとの感度（評価5回）", fontsize=10, loc="left")
    left.legend(loc="upper left", frameon=False, fontsize=8.5)

    training = json.loads((a.model / "training.json").read_text(encoding="utf-8"))
    root = Path.cwd()
    expert = [
        pt
        for ep in training["train_episodes"]
        for pt in elevator_trace(root / ep["path"])
    ]
    flights = [
        pt for flight in data["flights"] for pt in elevator_trace(Path(flight["path"]))
    ]
    for label, points, style in (
        ("お手本", expert, dict(color=INK, linestyle="--")),
        ("SmolVLA", flights, dict(color=AFTER)),
    ):
        xs, ys = binned(points, -12, 2, 0.5)
        right.plot(xs, ys, linewidth=2, label=label, **style)
    right.set_xlim(-12, 2)
    right.set_xlabel("接地までの時間（秒）")
    right.set_ylabel("エレベーター")
    right.set_title("接地直前の操作（引く＝正）", fontsize=10, loc="left")
    right.legend(loc="upper left", frameon=False, fontsize=8.5)
    for ax in (left, right):
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.text(0.4, ax.get_ylim()[0], "接地", color=MUTED, fontsize=8.5, va="bottom")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    fig.tight_layout()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=200)
    print("saved", a.output)

    airborne = [f for f in frames if f["wow"] == 0]
    for i, action in enumerate(data["actions"]):
        image, state = (
            statistics.median(abs(f[k][i]) for f in airborne)
            for k in ("swapped_image_change", "swapped_state_change")
        )
        print(f"{action}: image swap {image:.4f}, state swap {state:.4f}")


if __name__ == "__main__":
    main()
