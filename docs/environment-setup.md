# 開発環境のセットアップ

## 構成

- WSL2 / Ubuntu 22.04：Python、データ記録・学習・評価。
- Windows：FlightGear、描画、Windowsに接続したゲームパッドの入力。
- NVIDIA RTX 3060（12GB）：PyTorchのCUDA計算。FlightGearと同時に動かす場合はGPUメモリを共有するため、描画負荷と学習バッチサイズを調整する。
- Python 3.12.12、uv。依存パッケージの解決結果は `uv.lock` に固定する。

ゲームパッドはWindows版FlightGearで認識・割当を確認する。WSLの `/dev/input` にデバイスが見えることは、この構成では必要ない。
C172P・747-400・F-14Bの3機種を使用する。FlightGearとPythonの収録用通信、リセット・記録は今後実装する。
FlightGear内のJSBSimとPython版JSBSimは別の実行環境であり、モデルや物理計算が自動的に同期するわけではない。

## WSL側

リポジトリ直下で実行する。

```bash
uv sync --locked
uv run --locked python scripts/check_environment.py
```

PyTorchは公式CUDA 12.8インデックスから取得する。通常のwheel実行に必要なCUDAランタイムは依存関係に含まれる。
診断スクリプトはGPU上で順伝播・逆伝播・パラメータ更新を行い、JSBSimの付属C172Pを1秒進める。
診断に使うJSBSim付属C172Pと、課題に使用するFlightGear同梱C172Pは区別する。3機種の操縦・性能検証は別途必要。

## Windows側

FlightGear 2024.1.4のWindows本体をwingetでインストール済み。最新版を意味するものではない。基本データの取得は初回画面で別途行う。
PowerShellから再現する場合：

```powershell
winget install --id FlightGear.FlightGear --exact --source winget --version 2024.1.4
```

FlightGear本体に加えて対応する基本データが必要。初回起動で取得を求められた場合は案内に従う。
機体アドオンと羽田の風景データは別途導入し、取得元・版・ライセンスを記録する。

ゲームパッドをWindowsに接続し、`joy.cpl` で入力を確認した後、FlightGearのJoystick Configurationで軸・ボタンを割り当てる。
使用機種はLogicool F310（WindowsにUSB接続）。FlightGear付属の `js_demo.exe` が `Controller (Gamepad F310)` として検出し、軸の値を読み出せることを確認した。実際にスティックを動かしたときの各軸の対応と、飛行中の操縦は未確認。
F310にはXInput／DirectInput切替がある。現在認識されているモードで進め、切り替えた場合は軸番号とボタン割当を再確認する。
ロール・ピッチ・ラダー・スロットルの対応、軸反転、デッドゾーンは実際に動かして調整する。

WSLへUSBを直接渡す代案にはusbipd-winがあるが、Linux側の対応ドライバが必要で、WSLに接続中はWindowsから利用できない。
Bluetooth接続をUSBデバイスと同じ手順で渡せるとは限らないため、まずWindows側での利用を優先する。

## 確認状況

- Python 3.12.12を使用するuv環境を作成し、JSBSim 1.3.1、NumPy 2.5.3、PyTorch 2.11.0+cu128をロック済み。
- `uv sync --locked` が完了。診断スクリプトでRTX 3060（12GB）上の学習ステップが成功し、JSBSimのシミュレーション時刻が約1秒に進むことを確認済み。
- Windows版FlightGear 2024.1.4本体はインストール済み。基本データも取得済み。2026-09-28にログの展開完了記録と `C:/Users/mjuni/FlightGear/Downloads/fgdata_2024_1` を確認した。
- FlightGear付属の診断ツールでF310の認識・軸値読み取りを確認済み。操縦操作の対応は未確認。
- 両機の取得と飛行モデル読込、羽田周辺の地形取得まで実施。操縦・性能・収録用通信・着陸データ収集は未実施。[導入確認記録](simulator-validation.md) を参照。

## 参考

- [uvとPyTorchの設定](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [PyTorchのインストールとCUDA確認](https://pytorch.org/get-started/locally/)
- [FlightGearのダウンロード](https://www.flightgear.org/download/)
- [FlightGearの入力デバイス設定](https://wiki.flightgear.org/Input_device)
- [FlightGearの外部通信](https://wiki.flightgear.org/Generic_Protocol)
- [WSLへのUSB接続](https://learn.microsoft.com/en-us/windows/wsl/connect-usb)

- [F310公式クイックスタートガイド](https://www.logitech.com/assets/47879/f310-gamepad-quick-start-guide.pdf)

C172Pも羽田34Rでの起動・初期化・短時間の物理計算を確認済み。`scripts/check_flightgear.ps1` で初期化状態を再確認できる。詳細と初期描画エラーの経過は [導入確認記録](simulator-validation.md) を参照。
