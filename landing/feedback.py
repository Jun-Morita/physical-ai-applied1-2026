"""Post-flight advice for the human pilot, so later demonstrations succeed more often."""

from .core import approach_guidance, touchdown_index

# Distances before the threshold at which the approach is checked, in metres.
GATES = (1500, 1000, 500, 0)
# Glide path deviation that deserves a comment, in feet.
PATH_TOLERANCE_FT = 50

ADVICE = {
    "no_crash": "クラッシュしました。降下率と速度を落としてから接地してください。",
    "touchdown_zone": "接地位置が接地帯（進入端から900 m以内）の外です。進入端から400 mの接地目標点（太い白線2本）に向かう経路に早めに乗せてください。",
    "centerline": "接地時に中心線から離れています。進入端までに中心線へ合わせ、最後は小さな修正にとどめてください。",
    "sink_rate": "接地が強すぎます。対地20 ft付近でスロットルを絞り、スティックを少しずつ引いて降下を止めてから接地してください。",
    "bank": "傾いたまま接地しています。接地前に翼を水平に戻してください。",
    "main_gear_first": "機首が下がったまま接地しています。接地直前にスティックを引いて機首を少し上げ（フレア）、主脚から接地してください。",
    "heading": "機首が滑走路の向きとずれたまま接地しています。接地前にラダー（右スティック左右）で機首を滑走路に合わせてください。",
    "no_bounce": "接地後に再び浮きました。速度と降下率を落とし、機首を上げすぎずに接地してください。",
    "rollout_inside": "接地後に滑走路から外れました。ラダーでまっすぐ滑走してください。",
    "rollout_complete": "接地後の滑走が規定時間に達していません。",
    "no_damage": "脚が壊れたか、損傷を確認できませんでした。強い接地と機首からの接地を避けてください。",
}


def gate_states(rows, scenario):
    """First sample at or past each gate; None when the flight never reached it."""
    states = [r["state"] for r in rows]
    found = {}
    for gate in GATES:
        found[gate] = next(
            (s for s in states if approach_guidance(s, scenario)["distance_m"] <= gate),
            None,
        )
    return found


def feedback(rows, scenario, evaluation):
    """Japanese lines describing the approach and what to change next time."""
    if not rows or "guidance" not in scenario:
        return ["フィードバックに必要な記録がありません。"]
    lines = ["[進入の経過]"]
    tips = []
    deviated = set()
    for gate, state in gate_states(rows, scenario).items():
        if state is None:
            lines.append(f"  進入端の{gate} m手前に到達していません")
            continue
        g = approach_guidance(state, scenario)
        error = g["height_error_ft"]
        label = "進入端の上空" if gate == 0 else f"進入端の{gate} m手前"
        lines.append(
            f"  {label}: 経路より{abs(error):.0f} ft{'高い' if error >= 0 else '低い'}"
            f"、中心線の{'右' if g['cross_m'] >= 0 else '左'}{abs(g['cross_m']):.0f} m"
            f"、{g['airspeed_kt']:.0f} kt"
        )
        # One tip per direction, at the first gate where the deviation appears.
        if gate and abs(error) > PATH_TOLERANCE_FT and (error > 0) not in deviated:
            deviated.add(error > 0)
            tips.append(
                f"進入端の{gate} m手前で経路より{error:.0f} ft高すぎます。もっと早くスロットルを絞って降下してください。"
                if error > 0
                else f"進入端の{gate} m手前で経路より{-error:.0f} ft低すぎます。スロットルを上げて降下を緩めてください。"
            )
        if gate == 0 and g["airspeed_kt"] > 70:
            tips.append("進入端の通過速度が速すぎます。60〜65 ktまで落としてください。")
        elif gate == 0 and g["airspeed_kt"] < 55:
            tips.append(
                "進入端の通過速度が遅すぎます。失速に注意し、60〜65 ktを保ってください。"
            )
    states = [r["state"] for r in rows]
    touch = touchdown_index(states)
    if touch is not None:
        s = states[touch]
        along = -approach_guidance(s, scenario)["distance_m"]
        aim = scenario["guidance"].get("aim_point_m")
        where = f"進入端から{along:.0f} m"
        if aim is not None:
            where += f"（接地目標点の{abs(along - aim):.0f} m{'先' if along >= aim else '手前'}）"
        lines.append(
            f"  接地: {where}、降下率 {states[touch - 1]['down_fps'] * 60:.0f} fpm、"
            f"ピッチ {s['pitch_deg']:.1f}度、速度 {s['airspeed_kt']:.0f} kt"
        )
    checks = evaluation.get("checks", {})
    tips += [ADVICE[name] for name, passed in checks.items() if not passed]
    if evaluation.get("reason") == "no_air_to_ground_transition":
        tips.append("接地を検知できませんでした。")
    if evaluation.get("candidate_success"):
        lines.append(
            "[判定] 全項目に合格しました。動画を確認して、良ければ success のラベルを付けてください。"
        )
    else:
        lines.append("[判定] 不合格です。次回の改善点:")
        lines += [f"  - {tip}" for tip in tips] or [
            "  - (改善点を特定できませんでした)"
        ]
    return lines
