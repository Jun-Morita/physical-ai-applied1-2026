"""Report figure: before/after flight paths on the same evaluation starts.

    uv run --with matplotlib==3.10.3 python scripts/plot_report_figures.py \\
      --before runs/eval/vla-base-aim400-1000 runs/eval/vla-base-aim400-b-100? \\
      --after runs/eval/vla-expert-wide-norm-15000-100? \\
      --font /mnt/c/Windows/Fonts/BIZ-UDGothicR.ttc --output docs/report/figures/fig3-paths.png
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from landing.core import read_json, runway_position  # noqa: E402

BEFORE, AFTER = "#eb6834", "#2a78d6"  # validated categorical slots 2 and 1
MUTED, INK = "#8a8984", "#52514e"


def path(directory):
    meta = read_json(directory / "metadata.json")
    runway = meta["scenario"]["runway_geometry"]
    xs, hs, cs, touchdown = [], [], [], None
    for line in (directory / "samples.jsonl").open(encoding="utf-8"):
        s = json.loads(line)["state"]
        along, cross = runway_position(s, runway)
        xs.append(along / 1000)
        hs.append(s["altitude_ft"] - runway["elevation_ft"])
        cs.append(cross)
        if touchdown is None and s["wow"] > 0.5:
            touchdown = len(xs) - 1
    return meta, xs, hs, cs, touchdown


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", nargs="+", type=Path, required=True)
    p.add_argument("--after", nargs="+", type=Path, required=True)
    p.add_argument("--font", type=Path, help="Japanese font file")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.font:
        font_manager.fontManager.addfont(str(a.font))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(a.font)
        ).get_name()
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED})

    fig, (side, top) = plt.subplots(
        2, 1, figsize=(8, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    scenario = None
    for group, color, label in (
        (a.before, BEFORE, "学習前"),
        (a.after, AFTER, "学習後"),
    ):
        for i, directory in enumerate(group):
            scenario, xs, hs, cs, td = path(directory)
            scenario = scenario["scenario"]
            kw = dict(color=color, lw=1.6, alpha=0.9, label=label if i == 0 else None)
            side.plot(xs, hs, **kw)
            top.plot(xs, cs, color=color, lw=1.6, alpha=0.9)
            end = td if td is not None else len(xs) - 1
            marker = "o" if td is not None else "x"
            side.plot(
                xs[end],
                hs[end],
                marker,
                color=color,
                ms=6,
                mec="white" if td is not None else color,
            )
            top.plot(
                xs[end],
                cs[end],
                marker,
                color=color,
                ms=6,
                mec="white" if td is not None else color,
            )

    # Reference: the 3 degree path to the aiming point, and the runway.
    runway = scenario["runway_geometry"]
    aim = scenario["guidance"]["aim_point_m"] / 1000
    start = -scenario["initial"]["distance_m"] / 1000
    slope = (
        math.tan(math.radians(scenario["guidance"]["glide_slope_deg"])) / 0.3048 * 1000
    )
    side.plot(
        [start, aim],
        [(aim - start) * slope, 0],
        "--",
        color=MUTED,
        lw=1.2,
        label="3度の進入経路",
    )
    length = runway["length_m"] / 1000
    for ax in (side, top):
        ax.axvspan(0, length, color="#e8e6df", lw=0, zorder=0)
        ax.grid(color="#ecebe7", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=INK)
    half = runway["width_m"] / 2
    top.fill_between([0, length], -half, half, color="#cfcdc4", lw=0, zorder=0)
    side.axvline(aim, color=MUTED, lw=0.8, ls=":")
    side.text(aim + 0.03, 640, "接地目標\n（400 m）", color=INK, fontsize=9, va="top")
    side.text(0.05, 830, "滑走路", color=INK, fontsize=9, va="top")
    side.text(
        start + 0.4,
        230,
        "学習前：開始15〜19秒で\n裏返って海に落ちた（×）",
        color=BEFORE,
        fontsize=9,
    )
    side.text(-1.75, 60, "学習後：5回とも接地帯に接地（●）", color=AFTER, fontsize=9)

    side.set_ylabel("滑走路からの高さ（ft）", color=INK)
    side.set_ylim(-20, 850)
    side.legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=9
    )
    side.set_title(
        "横から見た飛行経路（評価用の番号1000〜1004）",
        loc="left",
        color=INK,
        fontsize=11,
    )
    top.set_ylabel("中心線からのずれ（m）", color=INK)
    top.set_ylim(-120, 120)
    top.set_xlabel("進入端からの距離（km、負は進入端の手前）", color=INK)
    top.set_title(
        "上から見た飛行経路（濃い帯が滑走路の幅）", loc="left", color=INK, fontsize=11
    )
    top.set_xlim(start - 0.1, 1.0)
    fig.tight_layout()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=200)
    print("saved", a.output)


if __name__ == "__main__":
    main()
