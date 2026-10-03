# VLAによる着陸：収録・学習前の実行・追加学習・学習後の実行

2026-10-03にVLAを採用する方針に合意した。vLLM（推論エンジン）ではなく、SmolVLA（画像・言語・状態から行動を出すモデル）を使う。まずC172P / RJTT 34Rで検証し、747-400・F-14Bにも別々のモデルを用意する。既存の数値入力MLPは比較用に保持する。

## 実装の範囲

WindowsのFlightGearからJPEG画像・数値状態・操作を収録し、UbuntuのRTX 3060でSmolVLAを動かす。WindowsとWSLの共有ディレクトリを介して推論を要求するため、推論用の外部HTTPサーバーやファイアウォール設定は不要。

最初の実装は**推論中にFlightGearを一時停止する方式**。通常の物理計算を再開して0.2シミュレーション秒以上進むと、再び止めて次の操作を求める。実時間20 HzのVLA操縦ではない。収録した動画には推論待ちの停止が含まれ、これを隠してリアルタイム性能と報告しない。記録には推論時間と `execution_mode=paused_inference` を残す。

SmolVLAの重みはロボット由来で、飛行操縦の知識を保証しない。事前学習時の操作次元を機体の7操作に対応させたものを `base_adapted` と呼び、追加学習後の `finetuned` と比較する。学習前の着陸成功を前提にしない。

数値入力は既存の17状態。固定の物理スケールで正規化し、7操作は各範囲を[-1, 1]に変換する。画像はRGBで縦横比を保って256×256へ余白付きで変換する。SmolVLAには1画像・指示・状態を渡し、1ステップの操作を学習・実行する。元の50ステップチャンク構成とは異なる課題用の設定。逆推力やスポイラー等は現状の操作出力に含めない。

## 1. VLA専用環境とベースモデルの準備（Ubuntu）

リポジトリ直下で実行する。

```bash
uv sync --project environments/vla --locked
uv run --project environments/vla --locked python -m landing.vla prepare \
  --scenario config/scenarios/c172p-rjtt-34r-approach.json \
  --output runs/vla/c172p-base
uv run --project environments/vla --locked python -m landing.vla check \
  --model runs/vla/c172p-base --backward
```

専用の `environments/vla/pyproject.toml` と `uv.lock` を管理する。Pythonは3.12、LeRobot 0.4.3、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128を使用し、ルートのMLP環境には変更を加えない。

`prepare` は `config/vla-model-lock.json` に固定したHugging Faceのリビジョンからダウンロードし、機体用の入出力設定と重みを保存する。キャッシュも `runs/` 内。ダウンロードと専用環境には数GB以上必要。外部への学習データ・モデルのアップロードはしない。

`check --backward` は黒画像とゼロ状態による推論・逆伝播・重み更新の診断で、更新後の重みを保存しない。飛行データでの学習や着陸実績とは区別する。

## 2. 画像付きの手動飛行データを取得（Windows）

既存の[着陸パイプライン](landing-pipeline.md)に従い、同じ初期状態とカメラ視点を使う。F310の割当、エンジン・脚・フラップ・開始状態を先に確認する。手動モードでFlightGearを起動し、別のPowerShellで収録する。収録・AI実行の開始前に、停止中の機体へシナリオのフラップ・脚・トリム・スロットルを再適用する（機体の初期化による上書きへの対応）。エンジン運転や重量・燃料をこの処理で保証するものではない。両方のターミナルはリポジトリ直下を作業場所とする。

```powershell
Set-Location '\\wsl.localhost\Ubuntu\home\jun\physical-ai-applied1-2026'
powershell -NoProfile -ExecutionPolicy Bypass -File ./scripts/start_flightgear.ps1 -Scenario config/scenarios/c172p-rjtt-34r-approach.json -Mode human
```

```powershell
py -3 -m landing.session record --scenario config/scenarios/c172p-rjtt-34r-approach.json --output datasets/c172p-visual/flight-001 --images --image-period 0.2 --video --ffmpeg C:\ffmpeg\bin\ffmpeg.exe
py -3 -m landing.session inspect datasets/c172p-visual/flight-001
py -3 -m landing.session label datasets/c172p-visual/flight-001 --outcome success --note "動画・画像と操縦を確認。初期状態、カメラ、脚・フラップの変更なし。"
```

着陸・減速後に収録側でCtrl+Cを押す。失敗や調整の飛行は `failure` / `exclude` とする。各飛行には新しい保存先を使う。

`frames/` にFlightGear自身のHTTPスクリーンショットを保存し、各フレームの取得前後のシミュレーション時刻、所要時間、画像ハッシュ、視点番号を `samples.jsonl` に記録する。画像とFDM状態は厳密に同時の観測ではない。画像取得中は手動飛行を止めないので受信間隔が広がる可能性がある。通信・描画・画像取得時間を実機で確認する。

既存のMP4だけの収録は、このVLA学習入力には使わない。画像時刻の対応付けが必要なため、`--images` 付きで新たに記録する。

指示の既定値は `landing.vision.INSTRUCTION` の英語文。変更する場合、`prepare` と `record` と `fly` に同一の `--instruction` を指定する。カメラはFlightGearの現在のビューを使い、飛行中に切り替えない。

## 3. 追加学習前のモデルで実行・録画

Ubuntu側で推論ワーカーを起動する。

```bash
uv run --project environments/vla --locked python -m landing.vla serve \
  --model runs/vla/c172p-base --bridge runs/vla/bridge-before-001 --seed 42
```

`VLA worker ready` を待つ。Windows側ではFlightGearを終了してAIモードで同じシナリオを起動する。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ./scripts/start_flightgear.ps1 -Scenario config/scenarios/c172p-rjtt-34r-approach.json -Mode ai
```

別のWindows PowerShellで実行する。

```powershell
py -3 -m landing.session fly --scenario config/scenarios/c172p-rjtt-34r-approach.json --vla-bridge runs/vla/bridge-before-001 --output runs/vla/before-001 --seconds 240 --video --ffmpeg C:\ffmpeg\bin\ffmpeg.exe
```

推論要求には画像と状態、ワーカー・飛行の識別子を含む。返答がない場合はタイムアウトで停止する。シナリオ・指示の一致を確認してから動作する。AIモードの停止は収録コンソールでCtrl+C。ワーカーも不要になったらUbuntu側でCtrl+C。ワーカーを再起動する場合は新しいbridgeディレクトリを指定する。

`--seconds` は実時間の上限で、推論待ちを含む。同じ上限でも推論速度により到達するシミュレーション時刻が違う可能性がある。比較時はタスク終了まで実行できたかも確認する。実時間のままMP4を保存するので、編集・倍速使用時はその旨を明記する。

## 4. 手動データで追加学習（Ubuntu）

```bash
uv run --project environments/vla --locked python -m landing.vla train \
  datasets/c172p-visual/flight-001 datasets/c172p-visual/flight-002 datasets/c172p-visual/flight-003 \
  --base runs/vla/c172p-base --output runs/vla/c172p-finetuned-001 \
  --steps 1000 --batch-size 1 --device cuda --seed 42
```

最低3飛行・各10組以上の有効な画像／操作ペアが必要。これはコードの受付条件であり、十分なデータ量ではない。成功確認済みの手動飛行だけを使い、飛行単位で約20%を検証用に分ける。AIの試行結果や診断記録を教師データへ混ぜない。状態・画像内容による重複検査を行い、受信時刻や画像ファイル名だけを変えたコピーも拒否する。

前処理では画像取得区間が0.25秒以内、直前の状態が画像取得開始から0.15秒以内のものを使う。画像取得完了以後で最初に得られた操作を教師とし、その遅れが0.1秒を超える組を除外する。停止をまたぐペアを除外する。クラッシュを含む飛行は拒否し、画像ハッシュ、時刻順序、脚・フラップ固定、対称推力、視点の一貫性と操作範囲を検査する。視点番号が不明な飛行も拒否する。各閾値は初期設定であり、実測で見直す。

LeRobotのSmolVLA本体を使うが、学習データは本課題のJSONL/JPEGを読む独自のPyTorch学習ループ。LeRobotDataset変換や `lerobot-train` を使用する構成ではない。画像エンコーダ等の凍結設定はSmolVLAの設定を引き継ぐ。まずbatch size 1からGPU使用量を確認する。

最終ステップのモデルと `landing.json`、`training.json`、`status.json` を保存する。学習ログの検証損失は固定した検証サブセットのflow-matching損失であり、全検証データの操縦誤差や着陸成功率ではない。チェックポイント選択、全検証飛行での予測グラフ、未見条件への汎化評価は今後の拡張。

## 5. 学習後のモデルで再実行・比較

Ubuntuで学習後のモデルを読み込む。

```bash
uv run --project environments/vla --locked python -m landing.vla serve \
  --model runs/vla/c172p-finetuned-001 --bridge runs/vla/bridge-after-001 --seed 42
```

WindowsのFlightGearを同じAIモード・シナリオで新しく起動してから実行する。

```powershell
py -3 -m landing.session fly --scenario config/scenarios/c172p-rjtt-34r-approach.json --vla-bridge runs/vla/bridge-after-001 --output runs/vla/after-001 --seconds 240 --video --ffmpeg C:\ffmpeg\bin\ffmpeg.exe
```

Ubuntuで比較を保存する。

```bash
uv run --locked python -m landing.compare \
  --before runs/vla/before-001 --after runs/vla/after-001 \
  --output runs/vla/comparison-001.json
```

複数試行は `--before` / `--after` に各飛行のパスを列挙する。失敗・中断も含める。比較は同一シナリオ・指示・実行方式・画像周期・指定実行時間・推論seedを要求し、参考判定の成功率と各試行の指標を出力する。接地信号・滑走路寸法はまだ実機照合が必要で、自動の成功判定を確定結果として使わない。燃料・重量・画角などの完全な一致は現状この比較コマンドだけでは保証しない。

短い接続確認は `fly --purpose diagnostic` として記録し、比較から除外する。既存の `integration_smoke_test` 記録も拒否する。指定実行時間の記録がない旧形式と新形式は混ぜて比較できない。

固定指示のみの実験は、言語理解の効果を証明しない。MLPと比較してもモデル構造・入力・実行周期が異なるので、改善を画像入力だけの効果と断定しない。

## 参照した一次資料

- [SmolVLA公式手順（LeRobot v0.4.3）](https://huggingface.co/docs/lerobot/v0.4.3/en/smolvla)：自分のデータで追加学習する構成。
- [SmolVLAモデル実装](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/policies/smolvla/modeling_smolvla.py)：状態・画像・行動のパディング、forward・select_action。
- [FlightGear画像取得の実装](https://github.com/FlightGear/flightgear/blob/next/src/Network/http/ScreenshotUriHandler.cxx)：`/screenshot?type=jpg`。参照したnextブランチの仕様に基づき実装し、導入済み2024.1.4でJPEG取得を確認済み。


## 実施済みの検証（2026-10-03）

- 収録前の整理後、自動テスト20件に合格（[点検記録](code-review.md)）。既存MLP経路、画像整合性、VLA通信とタイムアウト、推論中の停止、比較条件、初期操作設定を含む。
- Windows PythonでVLA用CLIを読み込み、PowerShellの構文解析に成功。
- RTX 3060で実際のSmolVLA重みを使い、模擬入力の推論・逆伝播・AdamW更新に成功。PyTorchが報告する最大確保メモリは約2.08 GB（GPU全体の使用量ではない）。初回推論は約12.2秒。
- 模擬データの追加学習1ステップ、検証、保存、再読込、再推論に成功。テストではデータ読み込みを模擬し、人の成功飛行として保存せず、作成したモデルは一時ディレクトリとともに削除した。
- FlightGear 2024.1.4の停止中C172Pから、32項目のUDP状態・JPEG画像・視点番号0を取得。画像取得前後のシミュレーション時刻は同じだった。動作中の同期精度や着陸性能の証明ではない。
- 起動オプションだけでは時計の停止が維持されなかったため、実行開始前にも物理・時計の停止を再適用し、応答を確認する。
- 初期化でフラップが0に上書きされたため、初期化後に設定を適用する処理を追加。HTTP経由で0.3333の反映を確認した。

モデルの追加学習・保存・再読込の診断（Ubuntu、合成入力のみ）：

```bash
uv run --project environments/vla --locked python scripts/check_vla_training.py --base runs/vla/c172p-base
```

停止中のFlightGearの読み取り診断（Windows、保存先は毎回新しい名前）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ./scripts/set_simulation_pause.ps1
py -3 scripts/check_vla_capture.py --scenario config/scenarios/c172p-rjtt-34r-approach.json --output runs/vla/capture-check-new
```

診断もUDPポート5501を使うため、収録コマンドと同時に実行しない。


### 実FlightGearとVLAの結合確認

`runs/vla/live-smoke-001/` に学習前のモデルによる短時間の診断結果を保存した（Git対象外）。8行の状態ログと2回のVLA推論を記録し、送信したエルロン・エレベータ値が次のUDP状態に反映されることを確認した。最初の推論は約9.16秒、次は約0.80秒。この2回だけで定常性能を判断しない。

シミュレーション時刻は0.70〜1.17秒の範囲。推論中は停止し、実時間の動画には停止が含まれる。MP4の保存とFFmpegの正常終了を確認した。ffprobeでH.264・1024×768・13.5秒・312,963バイトと確認し、終了後の物理・時計がともに停止していることも確認した。これは操作・録画経路の診断であり、着陸試行の成績には含めない。成功飛行データの収集、実データでの追加学習、学習後の着陸評価は未実施。
