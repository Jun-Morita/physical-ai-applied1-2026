# 着陸データから模倣学習・AI操縦・録画まで

2026-09-30：データ取得に先立ち、一連のコードを実装。テストの模擬データは動作検証用であり、手動操縦データ・着陸成功実績ではない。2026-10-03にC172Pで実通信・VLA操作反映・短時間録画を確認した（[VLA検証記録](vla-pipeline.md)）。F310の操縦割当と着陸に適した進入初期状態は引き続き確認が必要。

## 構成

- Windows：FlightGear、F310、標準ライブラリだけで動く `landing.session`、FFmpeg。
- Ubuntu / WSL：uvで管理するPyTorchによる `landing.train`。RTX 3060を使用可能。
- UDPはWindowsのループバック内で完結。状態は5501番、AI制御は5502番、20 Hz。HTTP 8080番は状態確認・一時停止に使用する。
- `config/protocols/` のXMLとPythonの列順は `scripts/generate_protocols.py` で同期する。スキーマを変えたら再生成し、FlightGearを再起動する。異なるスキーマの収録を混ぜない。
- `config/scenarios/*-rjtt-34r-approach.json` は3機種の仮の進入プリセット。5 NM手前、3度相当の高さ、昼間・無風。開始速度、トリム、フラップ、エンジン・燃料・重量は機種ごとの確認・調整が必要。自動操縦は切る。
- 滑走路進入端の緯度経度・方向は取得済みRJTTシーナリーの `RJTT.threshold.xml` に由来する。標高・長さ・幅と評価閾値は暫定値。評価指標として使う前に描画・接地面と照合する。

## 1. Windowsで開始する

Windows Python 3.12以降とFFmpegが必要。確認環境にはPython 3.13、`C:\ffmpeg\bin\ffmpeg.exe` がある。Python環境と依存関係の構築はUbuntuのREADME手順で `uv sync --locked` を実行する。Windows側にはPythonの追加パッケージは不要。

PowerShellでリポジトリへ移動する。WSLのディストリビューション名とユーザー名は自分の環境に置き換える。

```powershell
Set-Location '\\wsl.localhost\Ubuntu\home\jun\physical-ai-applied1-2026'
py -3 --version
& C:\ffmpeg\bin\ffmpeg.exe -version
powershell -NoProfile -ExecutionPolicy Bypass -File ./scripts/start_flightgear.ps1 -Aircraft c172p -Scenario ./config/scenarios/c172p-rjtt-34r-approach.json -Mode human
```

別のPowerShellも同じリポジトリへ移動して使う。起動スクリプトは通信XMLをFlightGear基本データのProtocolディレクトリへコピーする。初期状態をやり直すときはFlightGearを終了し、同じコマンドで再起動する。

初回は機体システムの初期化、エンジン運転、脚・フラップ・速度、F310の各軸とブレーキ割当を確認する。確認のため動かした後は新しく起動して収録する。各機体の起動スクリプトが設定を上書きすることがあるため、JSONだけで適切な進入状態が保証されるわけではない。

## 2. 手動操縦を収録する（後日実施）

```powershell
py -3 -m landing.session record --scenario config/scenarios/c172p-rjtt-34r-approach.json --output datasets/c172p/flight-001 --seconds 240 --video --ffmpeg C:\ffmpeg\bin\ffmpeg.exe
```

地形・飛行モデルの初期化と一時停止を確認し、シナリオのフラップ・脚・トリム・スロットルを再適用したうえで、収録側が時刻と物理の停止を解除する。着陸し減速したら収録コンソールでCtrl+C。指定時間に達した場合も終了する。終了時・通信異常時にはFlightGearを停止し、動画を閉じる。強制的にプロセスを終了すると、この後処理は保証されない。

`--seconds` は実時間の上限。動画はFlightGearウィンドウだけを通常速度で記録する。ウィンドウのタイトルが `FlightGear` でない場合はFFmpegが失敗するので `ffmpeg.log` を確認する。録画中はウィンドウを最小化しない。

保存先は毎回新しい名前にする。既存の飛行は上書きしない。短い接続確認には `--purpose diagnostic` を付け、学習・成績比較から分離する。通常の試行の既定値は `evaluation`。記録が空の場合や録画の終了に失敗した場合は `incomplete` として理由を保存する。

| ファイル | 内容 |
| --- | --- |
| `metadata.json` | 機体・シナリオ全文とハッシュ、実装ハッシュ、取得予定版の機体ロック情報、記録種別、完了状態、レビュー結果、着陸指標 |
| `samples.jsonl` | シミュレーション時刻、状態、実際の操縦プロパティ、AIの場合は送信した命令、受信時刻差、間引いたパケット数 |
| `flight.mp4` | `--video` 指定時の画面録画、音声なし |
| `ffmpeg.log` | エンコーダの診断情報 |

機体ロックは取得設定の記録であり、実行中の機体ファイル全体がその版と一致することの証明ではない。重量・燃料や機体固有の自動制御もまだ完全には固定・記録していない。収録開始前の機種別検証が必要。

## 3. 確認して学習対象を選ぶ

```powershell
py -3 -m landing.session inspect datasets/c172p/flight-001
py -3 -m landing.session label datasets/c172p/flight-001 --outcome success --note "動画を確認。滑走路内に接地し減速。操作設定の変更なし。"
```

失敗は `--outcome failure`、設定確認や不要な飛行は `exclude` とする。`candidate_success` は接地検知、接地直前の降下率、姿勢、滑走路内の減速を使った参考判定で、人による確認の代わりにはしない。接地検知は `/gear/gear[1]/wow` を使うため、3機種で実際に機能するか確認する。

## 4. Ubuntuで学習する

少なくとも別々の成功飛行3本が必要。3本は動作開始の最低条件で、十分な学習量を意味しない。

```bash
uv run --locked python -m landing.train \
  datasets/c172p/flight-001 datasets/c172p/flight-002 datasets/c172p/flight-003 \
  --output runs/c172p-bc-001 --epochs 100 --device cuda
```

機体・シナリオが同一の、完了かつ人が成功確認した手動飛行だけを受け付ける。AI飛行、診断記録、重複した飛行は拒否する。ログ時刻や画像ファイル名だけを変更したコピーも重複として扱う。時刻の逆行・重複、非有限値、クラッシュ、範囲外の操作値を含む飛行も拒否する。飛行単位で約20%を検証用に分離し、正規化統計は学習側だけから計算する。

入力は滑走路相対位置・高度・速度・姿勢・角速度・接地状態の17値。出力はエルロン、エレベータ、ラダー、全エンジン共通スロットル、左右ブレーキ、エレベータトリムの7値。フラップ・脚は固定し、左右異なる推力は扱わない。設定を変更したデータは拒否する。逆噴射・スポイラー・F-14固有制御はこのベースラインの学習出力に含まれない。

`状態(t) → 次の受信サンプルの操縦値(t+Δt)` を学習する。Δtが正で75 ms以下の隣接ペアのみ使い、停止中と欠落区間を除く。これは通信サンプルに基づく近似で、FDMの厳密なステップ前後を保証しない。最初の実通信確認で遅延と信号の意味を検証する。

64×64のMLPを学習し、検証MSEが最小の重みを `policy.json` に書き出す。`training.json` に分割・ハッシュ・誤差を保存する。検証誤差が低くても、連続したAI操縦で着陸できることは保証されない。

## 5. AIに操縦させて録画する

FlightGearを閉じ、PowerShellから同じシナリオをAIモードで起動する。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ./scripts/start_flightgear.ps1 -Aircraft c172p -Scenario ./config/scenarios/c172p-rjtt-34r-approach.json -Mode ai
```

別のPowerShellから開始する。

```powershell
py -3 -m landing.session fly --scenario config/scenarios/c172p-rjtt-34r-approach.json --policy runs/c172p-bc-001/policy.json --output runs/c172p-ai-flight-001 --seconds 240 --video --ffmpeg C:\ffmpeg\bin\ffmpeg.exe
```

AIモードではFlightGearのゲームパッド・マウス・キーボード入力を無効にして、入力の競合を避ける。停止はPython側のCtrl+Cで行う。手動飛行に戻す際は `-Mode human` で再起動する。

学習済みモデルとシナリオのハッシュが違う場合は実行を拒否する。機体・初期条件を変えるときは別シナリオ・別モデルで検証する。現在の実装は同一シナリオ内のベースラインで、未知の条件への汎化評価は追加課題。

747-400、F-14Bはパス中の `c172p` とシナリオをそれぞれ変更して独立に収録・学習する。データとモデルを混ぜない。

動画は生の実験記録。提出時は1〜5分、倍速使用時の明記、PDFレポートと合わせたZIP 200 MB以下など、[応用編1の提出要件](final-project-requirements.md) を確認する。

## コードの検証

```bash
uv run --locked python -m unittest discover -s tests -v
```

通信パケット、不正値、滑走路座標・接地判定、収録対象の制限、飛行単位の分割、正規化、PyTorchと標準ライブラリ推論の一致、通信断での停止、模擬AI送信と動画終了を検証する。実際のFlightGear・FFmpeg・F310を使った結合確認はこれらとは別に実施する。

2026-09-30のコード確認結果：自動テスト6件に合格。Windows PythonでCLIを読み込み、PowerShell起動スクリプトの構文解析に成功。RTX 3060上でも、模擬3エピソード（学習64ペア・検証32ペア）を2エポック学習し、有限の検証誤差とJSONモデル出力を確認した。模擬データとモデルは一時ディレクトリで削除済み。実飛行による性能評価ではない。

2026-10-03の収録前整理：共通のデータ検査・保存・録画終了処理を分離し、回帰テストを含む20件に合格。[修正内容と検証範囲](code-review.md)を参照。
