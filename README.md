# physical-ai-applied1-2026

Physical AI 2026 応用編1の最終課題として、**人のゲームパッド操縦から学ぶAIの着陸**に取り組むリポジトリです。
実在機を模擬したシミュレータで手動操縦のデータを収集し、模倣学習による着陸を目指します。

## タスクと使用機体

滑走路が視界に入っている最終進入から着陸するタスクを扱います。
空港は **羽田空港（HND / RJTT）**、機体は次の3機種に決定しました。

| 区分 | 機体 | 採用予定のモデル |
| --- | --- | --- |
| 軽飛行機 | Cessna 172P（C172P） | FlightGear 2024.1.4基本データ同梱版 |
| 旅客機 | Boeing 747-400（GE CF6-80C2B1F） | LincolnWorks-FlightGearの747-400 |
| 戦闘機 | F-14B | Zaretto系FlightGearモデル |

モデルは再現度の根拠を追えることを重視して選定しました。機体の取得と飛行モデルの読み込みを確認済みで、操縦・飛行性能は未検証です。
選定理由と制約は [機体選定](docs/aircraft-selection.md) に記録しています。

初期の実装範囲は、最終進入からフレア、接地、減速までを想定しています。
最初のコード検証用プリセットはRJTT 34Rへの進入です。空港・滑走路・天候・初期状態・機体はJSON設定から変更します。進入状態の飛行確認はこれからです。

## 技術構成と開発環境

- 利用可能な環境：WSL2 / Ubuntu 22.04、NVIDIA RTX 3060（12GB）、WindowsにUSB接続したLogicool F310。
- Pythonの環境構築・依存関係管理：**uv**。
- シミュレータ構成：Windows版FlightGearで描画・ゲームパッド入力・機体システムを扱い、WSL側のPythonで学習します。収録・AI操縦はWindowsのPythonからUDPで行います。実通信・操縦の検証はこれからです。
- データ収集：ゲームパッドの操縦入力と機体状態を時刻を揃えて記録します。
- 学習方法：手動操縦データによる模倣学習を軸にします。AIの入力は数値状態17値とし、画像入力は今後の拡張候補です。

## 環境構築

構成は **Windows：FlightGear・F310入力／WSL2 Ubuntu：Python・学習** です。
WindowsにUSB接続したF310をWSLに転送する必要はありません。

### 1. 前提条件

- Windows上でWSL2とUbuntuを利用できること（確認環境：Ubuntu 22.04.5）。
- Windows側にWSL対応のNVIDIAドライバがあり、Ubuntuで `nvidia-smi` が成功すること（確認GPU：RTX 3060、12GB）。
- UbuntuにGitとuvがあること（今回使用したuv：0.10.12）。uvがない場合は [公式手順](https://docs.astral.sh/uv/getting-started/installation/) で導入する。
- Python依存関係とFlightGearの基本データを取得するためのネット接続と、数GB以上の空き容量があること。

以下のUbuntuコマンドは、このリポジトリを取得したディレクトリで実行します。

### 2. Python・GPU環境（Ubuntu）

```bash
uv python install 3.12.12
uv sync --locked
uv run --locked python scripts/check_environment.py
```

`.python-version` でPython 3.12.12を指定し、`pyproject.toml` と `uv.lock` で依存関係を管理します。
PyTorchは公式CUDA 12.8インデックスから取得します。今回のwheel実行に必要なCUDAランタイムは依存関係に含まれます。
`uv run` を使用するため、仮想環境を手動で有効化する必要はありません。

診断はRTX 3060上で学習の1ステップ（順伝播・逆伝播・更新）を行い、JSBSimの付属C172Pで1秒間の物理計算を実行します。
成功するとGPU名、各バージョン、`cuda_training_step: passed`、`jsbsim_simulated_seconds`（約1秒）を表示します。
この診断はJSBSimパッケージ付属モデルを使用します。課題で使うFlightGear同梱C172Pとは区別し、3機種の操縦・飛行性能は別途検証します。

### 3. FlightGear（Windows PowerShell）

```powershell
winget install --id FlightGear.FlightGear --exact --source winget --version 2024.1.4
```

導入する本体は2024.1.4に固定します。インストール後、WindowsのスタートメニューからFlightGearを起動し、初回画面で対応する基本データをダウンロードします。
本体のインストールだけでは基本データの取得は完了しません。追加の747-400・F-14Bと羽田の風景データは、版を確認して別途導入します。

### 4. Logicool F310（Windows）

1. F310をUSB接続する。
2. `Win + R` → `joy.cpl` で、スティック・ボタンに反応することを確認する。
3. FlightGearのJoystick Configurationでピッチ・ロール・ラダー・スロットル等を割り当て、軸反転とデッドゾーンを調整する。

FlightGear付属の診断ツールでも認識を確認できます。PowerShellで実行し、終了時は `Ctrl + C` を押します。インストール先を変えた場合はパスを読み替えてください。

```powershell
& "C:\Program Files\FlightGear 2024.1\bin\js_demo.exe"
```

この環境では `Controller (Gamepad F310)` として検出し、軸値を読み取れました。
F310のXInput／DirectInputモードを切り替えると割当が変わる可能性があるため、現在認識されているモードで設定します。
実際のスティック操作との対応、3機種の操縦、WindowsとWSL間の通信は未検証です。

詳細な確認状況と参考資料は [環境構築メモ](docs/environment-setup.md) を参照してください。

### 5. 使用機体の準備（Ubuntu）

C172PはFlightGear 2024.1.4基本データに同梱されており、追加取得は不要です。747-400・F-14Bは以下で取得します。

以下のWindowsユーザー名を自分のものに置き換えて実行します。

```bash
uv run --locked python scripts/install_aircraft.py \
  --destination "/mnt/c/Users/<Windowsユーザー名>/FlightGear/Custom Aircraft/physical-ai"
```

取得元とコミットは `config/aircraft-lock.json` に固定しています。F-14には `patches/f14-nasal-2024.patch` を適用し、FlightGear 2024.1で予約語と衝突する旧変数名を修正します。空力データは変更しません。既存のモデルに変更がある場合は上書きせず停止します。
外部機体データはこのリポジトリに同梱しません。747はLincolnWorks版、F-14はZarettoの開発ブランチを固定したものです。

### 6. 羽田周辺の風景取得（Ubuntu）

```bash
uv run --locked python scripts/download_rjtt_scenery.py \
  --destination "/mnt/c/Users/<Windowsユーザー名>/FlightGear/Custom Scenery/physical-ai"
```

羽田を含む東経139度・北緯35度のタイルと空港補助データをTerraSync配布元から取得し、SHA-1とサイズで検証します。
今回の取得は296ファイル・約81MB。取得したファイルのハッシュは保存先の `physical-ai-scenery-manifest.json` に記録します。
このスクリプトは取得時点の配布索引を参照するため、将来同じ内容になる保証はありません。共有建物モデルなどは別途TerraSyncで取得します。

### 7. 羽田での導入確認（Windows PowerShell）

このリポジトリのディレクトリで実行します。WSLのリポジトリはエクスプローラーの `\\wsl.localhost\Ubuntu\home\jun\physical-ai-applied1-2026` から開けます（ディストリビューション名とユーザー名は環境に合わせて変更）。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ./scripts/start_flightgear.ps1 -Aircraft c172p
# 起動中のFlightGearを終了してから別機種を起動する
powershell -NoProfile -ExecutionPolicy Bypass -File ./scripts/start_flightgear.ps1 -Aircraft 747-400
# 747を終了してからF-14Bを起動する
powershell -NoProfile -ExecutionPolicy Bypass -File ./scripts/start_flightgear.ps1 -Aircraft f-14b
```

`config/scenarios/*-rjtt-check.json` は羽田34R・昼間・無風・地上・一時停止で始める導入確認用設定です。
34Rは初期確認用であり、着陸学習に使う滑走路の最終決定ではありません。
別の設定は `-Scenario <JSONファイル>` で指定できます。FlightGearの保存場所は `-FlightGearRoot`、実行ファイルは `-Executable` で変更できます。

取得済みカスタム風景を優先し、TerraSync（`https://terrasync.b-cdn.net`）も有効にします。初回は取得待ちがあり、取得後に再起動が必要な場合があります。
風景データは更新されるため、機体と異なり現段階では完全な版固定をしていません。
実行ログとシナリオはWindows側の `FlightGear/Logs/physical-ai/` に保存します。
診断用HTTPサーバーをWindowsの `127.0.0.1:8080` に限定して有効にします。WSLとの通信経路は別途確認します。

起動後の初期化は、別のWindows PowerShellで確認できます。`-ExpectedAircraft` は起動した機種に合わせます。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ./scripts/check_flightgear.ps1 -ExpectedAircraft c172p
```

地形・飛行モデルの初期化を最大120秒待ち、機体名、高度、停止状態、ゲームパッド設定を表示します。失敗時はエラーで終了します。
この診断は操縦性能や入力割当の検証ではありません。C172Pは初期化直後に描画用照明値の未設定エラーが出ましたが、照明値が揃った後の短時間再実行では再発しませんでした。

起動後の停止・再開は、別のWindows PowerShellで以下を実行できます。物理計算と時計の両方を切り替えます。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ./scripts/set_simulation_pause.ps1 -Resume
# 再び停止する
powershell -NoProfile -ExecutionPolicy Bypass -File ./scripts/set_simulation_pause.ps1
```

747は一時停止で初回起動すると脚高度が未初期化になり、自動飛行スクリプトの警告が出ます。
地形読込後に一度数秒再開すると初期化されます。この状態から再び停止できます。飛行性能の検証はこれからです。

## 収録・模倣学習・AI操縦のコード

データ取得前に一連のコードを作成しました。実行する順番とWindows / Ubuntuのコマンドは **[着陸パイプライン手順](docs/landing-pipeline.md)** にまとめています。

| 工程 | 実行場所・コマンド |
| --- | --- |
| 進入状態で起動 | Windows：`scripts/start_flightgear.ps1 -Scenario config/scenarios/c172p-rjtt-34r-approach.json -Mode human` |
| 手動操縦の収録 | Windows：`py -3 -m landing.session record` |
| データ確認・採否 | Windows：`py -3 -m landing.session inspect` / `label` |
| 模倣学習 | Ubuntu：`uv run --locked python -m landing.train` |
| AI操縦・動画保存 | Windows：AIモードで再起動後 `py -3 -m landing.session fly --video` |

表のコマンドにはシナリオ・出力先などの引数が必要です。上の手順書にそのまま実行できる例があります。収録・モデル・動画は `datasets/` と `runs/` に保存し、Gitには含めません。
WindowsのPythonは標準ライブラリのみ使用し、学習環境は既存の `uv.lock` で再現します。動画記録にはWindows版FFmpegが必要です。

コードの検証はUbuntuで実行します。

```bash
uv run --locked python -m unittest discover -s tests -v
```

現在の進入速度・フラップ・トリムは仮設定です。3機種で初期状態と信号を確認してから収録します。テスト用の模擬データを学習実績や着陸成功として扱いません。

## 現在の状況と開発予定

uvの依存関係を固定し、Windows版FlightGear本体の導入とF310の認識確認を行いました。
Python環境の構築と診断は完了しました。RTX 3060（12GB）上のPyTorch学習ステップと、JSBSimの1秒間の物理計算が成功しています。

| 確認項目 | バージョン・結果 |
| --- | --- |
| Python | 3.12.12 |
| PyTorch / CUDAランタイム | 2.11.0+cu128 / 12.8 |
| JSBSim / NumPy | 1.3.1 / 2.5.3 |
| GPU診断 | 順伝播・逆伝播・更新に成功 |
| FlightGear | Windows版本体2024.1.4を導入済み。基本データも取得済み（ログと保存先を確認） |
| F310 | 付属診断ツールで認識・軸値読み取りに成功 |

747-400・F-14Bを取得し、JSBSim飛行モデルと羽田の地形の読み込みを確認しました。C172Pも羽田34Rで起動し、地形・飛行モデルの初期化と短時間の物理計算を確認済みです。F-14のNasal互換性問題はパッチで修正済みです。
[導入確認と既知の問題](docs/simulator-validation.md) に互換性パッチと警告を記録しています。
収録・確認・模倣学習・AI操縦・録画のコードを追加しました。模擬データで検証しており、実通信・手動操縦データの取得・実データによる学習・AI着陸は未実施です。

1. 3機種と羽田で、ゲームパッド設定・操縦・着陸を確認する。
2. 追加した進入シナリオ、再起動による初期化、収録・制御・録画コードを実機で確認する。
3. 操縦データを収集し、飛行単位で学習用・評価用に分割して模倣学習を行う。
4. 未使用の初期条件で、着陸成功率・接地位置・降下速度・姿勢・滑走路逸脱を評価する。
5. 提出用の動画・レポート・コードを整理する。

合意事項と未確定事項の詳細は [プロジェクト方針](docs/project-plan.md) を参照してください。

## 講座・課題・研修資料

- [講座情報（東京大学 松尾・岩澤研究室）](https://weblab.t.u-tokyo.ac.jp/lecture/course-list/physical-ai/advanced1/)
- [最終課題の要件・提出チェックリスト](docs/final-project-requirements.md)
- [研修資料の一覧・保管場所](materials/README.md)
- [研修資料からの開発メモ](docs/lecture-notes.md)
- [基礎編の参考メモ：模倣学習・データ収集・評価](docs/foundations-notes.md)

最終課題の要件は、受講者提供の9/17版の画像に基づいています。
研修資料の原本は `materials/lectures/` に保管し、資料の開示制限に基づきGit管理・提出コードの対象外としています。
