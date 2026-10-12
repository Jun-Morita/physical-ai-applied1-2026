param(
    # Seeded start: training flights use 1-999, evaluation flights 1000+ (see README).
    [Parameter(Mandatory = $true)][ValidateRange(1, 99999)][int]$Seed,
    # Fly the scripted expert landing instead of the gamepad.
    [switch]$Autoland,
    # Fly a trained MLP (policy.json) on an evaluation seed.
    [string]$Policy,
    # Fly SmolVLA through a running worker's bridge directory on an evaluation seed.
    [string]$VlaBridge,
    # Training start profile (wide, wide-half, wide-quarter); evaluation uses nominal.
    [string]$StartProfile = "nominal",
    [string]$FFmpeg = "C:\ffmpeg\bin\ffmpeg.exe"
)
$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot/.."
$scenario = "config/scenarios/c172p-rjtt-34r-approach.json"
if ((@($Autoland.IsPresent, [bool]$Policy, [bool]$VlaBridge) -eq $true).Count -gt 1) {
    throw "Choose one of -Autoland, -Policy and -VlaBridge"
}
if ($Policy -or $VlaBridge) {
    if ($Seed -lt 1000) { throw "Evaluate policies on held-out seeds 1000+" }
    if ($StartProfile -ne "nominal") { throw "Evaluate policies from the nominal start" }
    $name = if ($Policy) { Split-Path (Split-Path $Policy -Parent) -Leaf } else { Split-Path $VlaBridge -Leaf }
    $output = "runs/eval/$name-$Seed"
} else {
    if ($Seed -gt 999) { throw "Training flights use seeds 1-999" }
    $kind = if ($Autoland) { "expert" } else { "flight" }
    $output = "datasets/c172p/$kind-{0:D3}" -f $Seed
}
if (Test-Path -LiteralPath $output) { throw "$output already exists; choose an unused -Seed" }

# Each flight needs a fresh simulator at its seeded start.
$deadline = (Get-Date).AddSeconds(30)
while ($running = Get-Process fgfs -ErrorAction SilentlyContinue) {
    if ((Get-Date) -gt $deadline) { throw "FlightGear did not exit; close it manually" }
    $running | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}
$mode = if ($Autoland -or $Policy -or $VlaBridge) { "ai" } else { "human" }
& "$PSScriptRoot/start_flightgear.ps1" -Aircraft c172p -Scenario $scenario -Mode $mode -Seed $Seed -StartProfile $StartProfile
& "$PSScriptRoot/check_flightgear.ps1" -ExpectedAircraft c172p | Out-Null
Write-Host "Ready. Recording $output; the flight starts now."
# Images are required for VLA training; the MP4 is for review and the video.
# Wall-clock limit: the expert pauses for every capture, so it needs longer.
$seconds = if ($Autoland) { "900" } else { "240" }
if ($Policy) {
    # No images: the MLP reads telemetry only and runs without pauses.
    & py -3 -m landing.session fly --policy $Policy --scenario $scenario --output $output --seconds $seconds --video --ffmpeg $FFmpeg
    exit $LASTEXITCODE
}
if ($VlaBridge) {
    # The simulation pauses for every capture and inference, so allow a long wall-clock time.
    & py -3 -m landing.session fly --vla-bridge $VlaBridge --scenario $scenario --output $output --seconds 1800 --image-period 0.2 --video --ffmpeg $FFmpeg
    exit $LASTEXITCODE
}
$common = @("--scenario", $scenario, "--output", $output, "--seconds", $seconds, "--images", "--image-period", "0.2", "--video", "--ffmpeg", $FFmpeg)
if ($Autoland) {
    & py -3 -m landing.session fly --autoland @common
} else {
    & py -3 -m landing.session record @common
}
