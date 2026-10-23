"""Build the submission video (1-3 min) from recorded flights.

    uv run --with pillow==12.0.0 python scripts/make_video.py \\
      --before runs/eval/vla-instruments-window-dagger3-15000-r2-1001 \\
      --after runs/eval/vla-instruments-ils-dme-window-15000-1001 \\
      --expert datasets/c172p/expert-201 \\
      --font /mnt/c/Windows/Fonts/BIZ-UDGothicB.ttc --output runs/video/PAI最終課題_143108.mp4

Recordings are wall-clock screen captures that stand still while the simulator
is paused for image capture and inference. Each kept frame is mapped to
simulation time through the telemetry log, so the edited clips play at
simulation speed with the paused time removed; this is stated in the video.
"""

import argparse
import bisect
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from landing.core import approach_guidance, read_json  # noqa: E402

W, H, FPS = 1280, 720, 30
NAVY, BLUE, INK, MUTED = "#183b5b", "#2f6f9f", "#243342", "#667788"
PALE, LINE, WARM = "#eef5f9", "#d9e3ea", "#f7f4ed"
BEFORE, AFTER = "#eb6834", "#2a78d6"


def run(*args):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True
    )


# ---------------------------------------------------------------- flight clips


def sim_time_frames(flight, work):
    """Extract changing frames and the simulation time at which each was shown."""
    frames = work / "frames"
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    log = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(flight / "flight.mp4"),
            "-vf",
            "mpdecimate=hi=64*12:lo=64*5:frac=0.33,showinfo",
            "-vsync",
            "vfr",
            "-q:v",
            "3",
            str(frames / "%06d.jpg"),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stderr
    times = [float(t) for t in re.findall(r"pts_time:([0-9.]+)", log)]
    files = sorted(frames.glob("*.jpg"))
    rows = [json.loads(line) for line in (flight / "samples.jsonl").open()]
    duration = float(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(flight / "flight.mp4"),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    # The recorder starts shortly before the control loop and stops right after it.
    offset = duration - rows[-1]["wall_elapsed"]
    walls = [r["wall_elapsed"] for r in rows]
    sims = [r["state"]["time"] for r in rows]

    def sim_at(wall):
        i = bisect.bisect_left(walls, wall)
        if i <= 0:
            return sims[0]
        if i >= len(walls):
            return sims[-1]
        f = (wall - walls[i - 1]) / max(walls[i] - walls[i - 1], 1e-6)
        return sims[i - 1] + f * (sims[i] - sims[i - 1])

    kept, last = [], -1.0
    for path, t in zip(files, times):
        sim = sim_at(t - offset)
        if sim > last + 0.02:
            kept.append((path, sim))
            last = sim
    return kept, rows


def telemetry_ass(rows, scenario, t0, t1, path, label, color):
    """Subtitle track with height, speed and distance, on the clip's own timeline."""

    def ts(sec):
        sec = max(sec, 0)
        return f"{int(sec // 3600)}:{int(sec % 3600 // 60):02d}:{sec % 60:05.2f}"

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {W}",
        f"PlayResY: {H}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV",
        "Style: T,BIZ UDGothic,40,&H00FFFFFF,&H00000000,&H90000000,1,3,10,0,1,24,24,24",
        "Style: L,BIZ UDGothic,46,&H00FFFFFF,&H00000000,&H00{},1,3,12,0,3,24,24,24".format(
            color[5:7] + color[3:5] + color[1:3]
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Text",
        f"Dialogue: 0,{ts(0)},{ts(t1 - t0 + 30)},L,{label}",
    ]
    step = 0.25
    t = t0
    sims = [r["state"]["time"] for r in rows]
    while t < t1:
        s = rows[min(bisect.bisect_left(sims, t), len(rows) - 1)]["state"]
        g = approach_guidance(s, scenario)
        dist = g["distance_m"] / 1000
        where = (
            f"進入端まで {dist:4.1f} km" if dist > 0 else f"進入端から {-dist:4.1f} km"
        )
        text = f"高度 {max(s['agl_ft'] - 3.5, 0):4.0f} ft　速度 {s['airspeed_kt']:3.0f} kt\\N{where}"
        lines.append(f"Dialogue: 0,{ts(t - t0)},{ts(t - t0 + step)},T,{text}")
        t += step
    path.write_text("\n".join(lines), encoding="utf-8")


def flight_clip(flight, work, out, label, color, t0=None, t1=None, fonts=None):
    kept, rows = sim_time_frames(flight, work)
    scenario = read_json(flight / "metadata.json")["scenario"]
    t0 = kept[0][1] if t0 is None else t0
    t1 = kept[-1][1] if t1 is None else t1
    kept = [(p, s) for p, s in kept if t0 <= s <= t1]
    concat = work / "frames.txt"
    with concat.open("w") as f:
        for (path, sim), (_, nxt) in zip(kept, kept[1:] + [(None, kept[-1][1] + 0.05)]):
            f.write(f"file '{path.resolve()}'\nduration {max(nxt - sim, 0.01):.4f}\n")
    ass = work / "telemetry.ass"
    telemetry_ass(rows, scenario, t0, t1, ass, label, color)
    run(
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat),
        "-vf",
        f"fps={FPS},crop=iw:ih-24:0:24,scale={W}:{H},subtitles={ass}:fontsdir={fonts}",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(out),
    )
    return t1 - t0


# ---------------------------------------------------------------- cards


class Card:
    def __init__(self, font):
        self.font = font
        self.im = Image.new("RGB", (W, H), "white")
        self.d = ImageDraw.Draw(self.im)

    def f(self, size):
        return ImageFont.truetype(self.font, size)

    def text(self, xy, s, size, fill=INK, anchor="la"):
        self.d.text(xy, s, font=self.f(size), fill=fill, anchor=anchor)

    def kicker(self, s="Physical AI 応用編1 ｜ 最終課題"):
        self.text((80, 60), s, 22, BLUE)

    def save(self, path):
        self.im.save(path)


def still(png, out, seconds):
    run(
        "-loop",
        "1",
        "-t",
        str(seconds),
        "-i",
        str(png),
        "-vf",
        f"fps={FPS},format=yuv420p,fade=in:0:{FPS // 2},fade=out:st={seconds - 0.5}:d=0.5",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        str(out),
    )


def cards(font, work, expert_clip):
    out = {}
    c = Card(font)
    c.kicker()
    c.text((80, 190), "パイロットと同じ情報で着陸できるか", 64, NAVY)
    c.text((80, 285), "SmolVLAによる小型機の着陸と、AIに足りなかった距離感", 40, NAVY)
    c.d.rectangle((80, 360, 86, 470), fill=BLUE)
    c.text((110, 365), "FlightGearで、セスナ172Pを羽田空港の滑走路34Rに着陸させる", 28)
    c.text((110, 410), "窓の外の景色と計器 → 操縦桿・ラダー・スロットル", 28)
    c.save(work / "title.png")
    out["title"] = work / "title.png"

    c = Card(font)
    c.kicker()
    c.text((80, 110), "課題の設定", 46, NAVY)
    rows = [
        ("開始", "滑走路の2 NM（約3.7 km）手前。乱数で横・高さ・方位・速度をずらす"),
        ("AIの入力", "窓の外の景色と、パイロットが読める計器だけ（ILS・DMEを含む）"),
        ("AIの出力", "エルロン・エレベーター・ラダー・スロットルの4つ"),
        ("モデル", "SmolVLA（ロボット用に公開されたVLA）を追加学習"),
        ("お手本", "自作の自動着陸の飛行224回（全項目に合格したもの）"),
        ("成功", "接地位置・中心線・降下率・バウンド・脚の損傷など11項目すべてに合格"),
    ]
    y = 200
    for k, v in rows:
        c.d.rounded_rectangle(
            (80, y, 1200, y + 62),
            10,
            fill=PALE if rows.index((k, v)) % 2 == 0 else "white",
            outline=LINE,
        )
        c.text((105, y + 31), k, 26, NAVY, "lm")
        c.text((300, y + 31), v, 26, INK, "lm")
        y += 72
    c.save(work / "task.png")
    out["task"] = work / "task.png"

    c = Card(font)
    c.kicker()
    c.text((80, 110), "結果：計器を加えるごとに、できることが増えた", 46, NAVY)
    c.text(
        (80, 175),
        "開始位置の異なる5ケースで評価（いずれも学習には使っていない位置）",
        24,
        MUTED,
        "lm",
    )
    table = [
        ("AIの入力", "横のずれ", "高さのずれ", "全項目合格"),
        ("計器＋景色", "戻せない", "戻せない", "0/5"),
        ("＋ILS（コースと進入角の針）", "戻せる", "戻せない", "1/5"),
        ("＋ILS＋DME（距離計）", "戻せる", "戻し始める", "1/5"),
    ]
    y = 215
    for i, row in enumerate(table):
        head = i == 0
        c.d.rectangle(
            (80, y, 1200, y + 70), fill=PALE if head else "white", outline=LINE
        )
        c.text((105, y + 35), row[0], 28, NAVY if head else INK, "lm")
        for x, v in zip((690, 900, 1100), row[1:]):
            good = v.startswith("戻せる") or v.startswith("戻し") or v == "1/5"
            color = NAVY if head else (AFTER if good else BEFORE)
            c.text((x, y + 35), v, 28, color, "mm")
        y += 70
    c.text(
        (80, 540),
        "景色だけでは、AIは自分の位置と距離を読み取れなかった。",
        28,
        INK,
    )
    c.text(
        (80, 590),
        "足りなかったのは操縦ではなく、人間が景色から自然に得ている距離感だった。",
        28,
        INK,
    )
    c.save(work / "result.png")
    out["result"] = work / "result.png"

    c = Card(font)
    c.kicker()
    c.text((80, 110), "今後の課題", 46, NAVY)
    items = [
        (
            "景色から距離を読ませる",
            "数枚の画像を続けて入力して動きを見せる。画像を読む部分も飛行の画像で学習させる",
        ),
        (
            "高さの制御",
            "開始直後に深く降りすぎる癖が残った。経路のずれから戻す場面を重く学習させる",
        ),
        (
            "リアルタイム化",
            "今回は画像の取得と推論のたびにシミュレーションを止めている",
        ),
        ("ほかの機体へ", "旅客機（747-400）・戦闘機（F-14B）の進入シナリオを用意済み"),
    ]
    y = 210
    for k, v in items:
        c.d.rounded_rectangle((80, y, 1200, y + 100), 12, fill=WARM)
        c.text((110, y + 16), k, 28, NAVY)
        c.text((110, y + 60), v, 23, INK)
        y += 112
    c.save(work / "next.png")
    out["next"] = work / "next.png"
    return out


def caption_png(font, path, title, sub, color=NAVY):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 104 if sub else 84), fill=(255, 255, 255, 235))
    d.text((40, 22), title, font=ImageFont.truetype(font, 32), fill=color)
    if sub:
        d.text((40, 66), sub, font=ImageFont.truetype(font, 22), fill=MUTED)
    im.save(path)


def compare(font, before, after, work, out, fonts):
    """Before and after side by side from the same start, at twice simulation speed."""
    b = work / "before.mp4"
    a = work / "after.mp4"
    flight_clip(before, work / "b", b, "計器＋景色", BEFORE, fonts=fonts)
    length = flight_clip(after, work / "a", a, "＋ILS＋DME", AFTER, fonts=fonts)
    top = work / "compare-top.png"
    caption_png(
        font,
        top,
        "同じ開始位置から、2つのAIを同時に再生",
        "評価ケース2（学習に使っていない開始位置）｜ 2倍速 ｜ AIの計算中は一時停止し、その時間はカット",
    )
    panel = Image.open(top).convert("RGBA")
    d = ImageDraw.Draw(panel)
    small = ImageFont.truetype(font, 24)
    for x, color, head, body in (
        (
            13,
            BEFORE,
            "計器＋景色",
            ["自分の位置と距離が分からず、", "高さのずれも横のずれも戻せない"],
        ),
        (
            647,
            AFTER,
            "＋ILS＋DME（パイロットが使う計器）",
            [
                "低くなっても経路に戻し、中心線から",
                "0.2 mの位置に接地。全11項目に合格",
            ],
        ),
    ):
        d.rounded_rectangle((x, 520, x + 620, 680), 12, fill=(255, 255, 255, 255))
        d.rectangle((x, 520, x + 8, 680), fill=color)
        d.text((x + 28, 536), head, font=ImageFont.truetype(font, 30), fill=color)
        for i, line in enumerate(body):
            d.text((x + 28, 584 + i * 36), line, font=small, fill=INK)
    panel.save(top)
    crash = work / "crash.png"
    im = Image.new("RGBA", (620, 349), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((60, 130, 560, 220), 14, fill=(255, 255, 255, 230))
    d.text(
        (310, 175),
        "滑走路の上を高く通り過ぎた",
        font=ImageFont.truetype(font, 36),
        fill=BEFORE,
        anchor="mm",
    )
    im.save(crash)
    seconds = length / 2
    run(
        "-i",
        str(b),
        "-i",
        str(a),
        "-i",
        str(top),
        "-i",
        str(crash),
        "-filter_complex",
        f"[0]setpts=PTS/2,scale=620:349,tpad=stop_mode=clone:stop_duration={seconds}[b0];"
        f"[b0][3]overlay=0:0:enable='gte(t,{102 / 2})'[b];"
        f"[1]setpts=PTS/2,scale=620:349[a];"
        f"color=c=#eef5f9:s={W}x{H}:d={seconds}[bg];"
        f"[bg][b]overlay=13:140[x];[x][a]overlay=647:140[y];[y][2]overlay=0:0,"
        f"fps={FPS},trim=duration={seconds},format=yuv420p,fade=in:0:{FPS // 2},fade=out:st={seconds - 0.7}:d=0.7[v]",
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        str(out),
    )


def expert_segment(font, expert, work, out, fonts):
    clip = work / "expert-full.mp4"
    kept, rows = sim_time_frames(expert, work / "e")
    end = kept[-1][1]
    flight_clip(
        expert, work / "e", clip, "お手本", "#183b5b", t0=end - 18, t1=end, fonts=fonts
    )
    top = work / "expert-top.png"
    caption_png(
        font,
        top,
        "お手本：自作の自動着陸（PID制御）",
        "このような飛行224回でSmolVLAを追加学習 ｜ 等倍 ｜ 画像の撮影中は一時停止し、その時間はカット",
    )
    run(
        "-i",
        str(clip),
        "-i",
        str(top),
        "-filter_complex",
        f"[0][1]overlay=0:0,fps={FPS},format=yuv420p,fade=in:0:{FPS // 2},fade=out:st=17.3:d=0.7[v]",
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        str(out),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--expert", type=Path, required=True)
    p.add_argument("--font", type=Path, required=True)
    p.add_argument("--fonts-dir", type=Path, default=Path("/mnt/c/Windows/Fonts"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--music", type=Path, help="Background music (played once; silence after it)"
    )
    p.add_argument("--music-volume", type=float, default=0.35)
    p.add_argument("--opening", type=Path, help="Course opening jingle (optional)")
    p.add_argument("--closing", type=Path, help="Course closing jingle (optional)")
    p.add_argument("--work", type=Path, default=Path("runs/video/work").resolve())
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    font = str(a.font)
    c = cards(font, a.work, None)
    parts = []
    for name, seconds in (("title", 5), ("task", 12)):
        out = a.work / f"{name}.mp4"
        still(c[name], out, seconds)
        parts.append(out)
    expert_seg = a.work / "expert.mp4"
    expert_segment(font, a.expert, a.work, expert_seg, a.fonts_dir)
    parts.append(expert_seg)
    compare_seg = a.work / "compare.mp4"
    compare(font, a.before, a.after, a.work, compare_seg, a.fonts_dir)
    parts.append(compare_seg)
    for name, seconds in (("result", 20), ("next", 15)):
        out = a.work / f"{name}.mp4"
        still(c[name], out, seconds)
        parts.append(out)
    listing = a.work / "parts.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    bookends = a.opening or a.closing
    main_out = a.work / "main.mp4" if bookends else a.output
    silent = a.work / "silent.mp4" if a.music else main_out
    run(
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(listing),
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(FPS),
        "-movflags",
        "+faststart",
        str(silent),
    )
    if a.music:
        seconds = float(
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    str(silent),
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        # Quiet music under the visuals, played once: it fades out at the end of
        # the track (or of the video) and the closing cards may run on in silence.
        track = float(
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    str(a.music),
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        end = min(seconds, track)
        run(
            "-i",
            str(silent),
            "-i",
            str(a.music),
            "-filter_complex",
            f"[1:a]volume={a.music_volume},afade=t=in:d=2,"
            f"afade=t=out:st={end - 4}:d=4,atrim=duration={end},"
            f"apad=whole_dur={seconds}[m]",
            "-map",
            "0:v",
            "-map",
            "[m]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(main_out),
        )
    if bookends:
        bookend(main_out, a.opening, a.closing, a.output, main_audio=bool(a.music))
    print("saved", a.output)


def bookend(main, opening, closing, out, main_audio=True):
    """Course jingles before and after the video, at its size and frame rate.

    The jingles have no sound; silence is added so the audio stays in step.
    """
    inputs, labels = [], []
    for clip in (opening, main, closing):
        if clip is None:
            continue
        n = len(inputs) // 2
        inputs += ["-i", str(clip)]
        v = f"[{n}:v]scale={W}:{H},setsar=1,fps={FPS},format=yuv420p[v{n}];"
        if clip == main and main_audio:
            a = f"[{n}:a]aresample=48000,aformat=channel_layouts=stereo[a{n}];"
        else:
            a = (
                f"anullsrc=r=48000:cl=stereo,atrim=duration="
                f"{probe_duration(clip)}[a{n}];"
            )
        labels.append((v + a, f"[v{n}][a{n}]"))
    graph = "".join(f for f, _ in labels) + "".join(s for _, s in labels)
    graph += f"concat=n={len(labels)}:v=1:a=1[v][a]"
    run(
        *inputs,
        "-filter_complex",
        graph,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(out),
    )


def probe_duration(path):
    return float(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )


if __name__ == "__main__":
    main()
