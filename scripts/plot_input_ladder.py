"""Report figure: flight paths as pilot instruments are added to the inputs.

    uv run --with matplotlib==3.10.3 python scripts/plot_input_ladder.py \\
      --group "計器＋景色" runs/eval/vla-instruments-window-dagger3-15000-r2-100? \\
      --group "＋ILS" runs/eval/vla-instruments-ils-window-15000-100? \\
      --group "＋ILS＋DME" runs/eval/vla-instruments-ils-dme-window-15000-100? \\
      --font /mnt/c/Windows/Fonts/BIZ-UDGothicR.ttc \\
      --output docs/report/figures/fig5-input-ladder.png

One column per model, on the same evaluation starts (1000-1004) and scales:
height above the runway (top) and distance from the centerline (bottom).
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_report_figures import INK, MUTED, path  # noqa: E402

LINE, PASS = "#2a78d6", "#eb6834"  # validated categorical slots 1 and 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--group",
        nargs="+",
        action="append",
        required=True,
        metavar=("LABEL", "FLIGHT"),
        help="Column label followed by its evaluation flights",
    )
    p.add_argument("--font", type=Path, help="Japanese font file")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if any(len(g) < 2 for g in a.group):
        p.error("Each --group needs a label and at least one flight")
    if a.font:
        font_manager.fontManager.addfont(str(a.font))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(a.font)
        ).get_name()
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED})

    columns = len(a.group)
    fig, axes = plt.subplots(
        2,
        columns,
        figsize=(3.0 * columns, 4.6),
        sharex=True,
        sharey="row",
        gridspec_kw={"height_ratios": [3, 2]},
        squeeze=False,
    )
    scenario = None
    for col, (label, *flights) in enumerate(a.group):
        side, top = axes[0][col], axes[1][col]
        passed = 0
        for directory in map(Path, flights):
            meta, xs, hs, cs, td = path(directory)
            scenario = meta["scenario"]
            ok = (meta.get("evaluation") or {}).get("candidate_success")
            passed += bool(ok)
            color, width, z = (PASS, 2.0, 3) if ok else (LINE, 1.2, 2)
            end = td if td is not None else len(xs) - 1
            for ax, ys in ((side, hs), (top, cs)):
                ax.plot(xs, ys, color=color, lw=width, alpha=0.9, zorder=z)
                ax.plot(
                    xs[end],
                    ys[end],
                    "o" if td is not None else "x",
                    color=color,
                    ms=5,
                    mec="white" if td is not None else color,
                    zorder=z,
                )
        side.set_title(
            f"{label}（全項目合格 {passed}/{len(flights)}）",
            loc="left",
            color=INK,
            fontsize=10,
        )

    runway = scenario["runway_geometry"]
    aim = scenario["guidance"]["aim_point_m"] / 1000
    start = -scenario["initial"]["distance_m"] / 1000
    slope = (
        math.tan(math.radians(scenario["guidance"]["glide_slope_deg"])) / 0.3048 * 1000
    )
    length = runway["length_m"] / 1000
    half = runway["width_m"] / 2
    for col in range(columns):
        side, top = axes[0][col], axes[1][col]
        side.plot(
            [start, aim],
            [(aim - start) * slope, 0],
            "--",
            color=MUTED,
            lw=1.0,
            zorder=1,
        )
        top.axhline(0, color=MUTED, lw=1.0, ls="--", zorder=1)
        top.fill_between([0, length], -half, half, color="#cfcdc4", lw=0, zorder=0)
        for ax in (side, top):
            ax.axvspan(0, length, color="#e8e6df", lw=0, zorder=0)
            ax.grid(color="#ecebe7", lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(colors=INK)
        top.set_xlabel("進入端からの距離（km）", color=INK)
    axes[0][0].set_ylabel("滑走路からの高さ（ft）", color=INK)
    axes[1][0].set_ylabel("中心線からのずれ（m）", color=INK)
    axes[0][0].set_ylim(-20, 820)
    axes[1][0].set_ylim(-450, 450)
    axes[0][0].set_xlim(start - 0.1, 1.0)
    axes[0][0].text(start + 0.1, 30, "破線：3度の進入経路", color=INK, fontsize=8)
    fig.text(
        0.01,
        0.005,
        "橙：全項目に合格した飛行　●：接地　×：接地せずに終了　灰色の帯：滑走路",
        color=INK,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=200)
    print("saved", a.output)


if __name__ == "__main__":
    main()
