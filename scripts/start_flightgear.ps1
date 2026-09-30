param(
    [ValidateSet("c172p", "747-400", "f-14b")][string]$Aircraft = "747-400",
    [string]$Scenario = "",
    [ValidateSet("human", "ai")][string]$Mode = "human",
    [string]$FlightGearRoot = "$env:USERPROFILE/FlightGear",
    [string]$Executable = "C:/Program Files/FlightGear 2024.1/bin/fgfs.exe"
)
$ErrorActionPreference = "Stop"
if (Get-Process fgfs -ErrorAction SilentlyContinue) {
    throw "Close the existing FlightGear instance before starting another scenario."
}
if (!$Scenario) { $Scenario = "$PSScriptRoot/../config/scenarios/$Aircraft-rjtt-check.json" }
$config = Get-Content -LiteralPath $Scenario -Raw | ConvertFrom-Json
if ($config.aircraft -notin @("c172p", "747-400", "f-14b")) { throw "Unsupported aircraft" }
$aircraftRoot = "$FlightGearRoot/Custom Aircraft/physical-ai"
$searchPath = "$aircraftRoot;$aircraftRoot/zaretto/aircraft"
$model = switch ($config.aircraft) {
    "c172p" { "$FlightGearRoot/Downloads/fgdata_2024_1/Aircraft/c172p/c172p-set.xml" }
    "747-400" { "$aircraftRoot/747-400/747-400-set.xml" }
    "f-14b" { "$aircraftRoot/zaretto/aircraft/f-14b/f-14b-set.xml" }
}
foreach ($path in @($Executable, "$FlightGearRoot/Downloads/fgdata_2024_1/version", $model)) {
    if (!(Test-Path -LiteralPath $path)) { throw "Missing installation file: $path" }
}
$logDir = "$FlightGearRoot/Logs/physical-ai"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log = "$logDir/$($config.aircraft)-$stamp"
New-Item -ItemType Directory -Force -Path $log | Out-Null
$config | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 "$logDir/$($config.aircraft)-$stamp.scenario.json"
$options = @(
    "--fg-root=$FlightGearRoot/Downloads/fgdata_2024_1",
    "--fg-aircraft=$searchPath",
    "--fg-scenery=$FlightGearRoot/Custom Scenery/physical-ai;$FlightGearRoot/Downloads/TerraSync;$FlightGearRoot/Downloads/fgdata_2024_1/Scenery",
    "--aircraft=$($config.aircraft)",
    "--airport=$($config.airport)",
    "--runway=$($config.runway)",
    "--timeofday=$($config.timeofday)",
    "--wind=$($config.wind)",
    "--disable-real-weather-fetch",
    "--enable-terrasync",
    "--prop:/sim/terrasync/http-server=https://terrasync.b-cdn.net",
    "--terrasync-dir=$FlightGearRoot/Downloads/TerraSync",
    "--httpd=127.0.0.1:8080",
    "--log-level=info",
    "--log-dir=$log"
)
if ($config.initial) {
    # A Python-computed canonical hash is identical to the runner/trainer implementation.
    $scenarioHash = & py -3 -c 'import sys; from landing.core import digest, read_json; print(digest(read_json(sys.argv[1])))' $Scenario
    if ($LASTEXITCODE -ne 0) { throw "Run from the repository root with Windows Python available" }
    foreach ($name in @("physical-ai-state", "physical-ai-action")) {
        Copy-Item -LiteralPath "$PSScriptRoot/../config/protocols/$name.xml" -Destination "$FlightGearRoot/Downloads/fgdata_2024_1/Protocol/$name.xml" -Force
    }
    $options += @(
        "--in-air", "--lat=$($config.initial.latitude)", "--lon=$($config.initial.longitude)",
        "--altitude=$($config.initial.altitude_ft)", "--heading=$($config.initial.heading_deg)",
        "--vc=$($config.initial.airspeed_kt)",
        "--prop:/sim/physical-ai/scenario-hash=$scenarioHash",
        "--generic=socket,out,$($config.rate_hz),127.0.0.1,$($config.telemetry_port),udp,physical-ai-state",
        "--prop:/controls/flight/flaps=$($config.controls.flaps)",
        "--prop:/controls/flight/elevator-trim=$($config.controls.trim)",
        "--prop:/controls/gear/gear-down-command=true"
    )
    for ($i = 0; $i -lt $config.engine_count; $i++) {
        $options += "--prop:/controls/engines/engine[$i]/throttle=$($config.controls.throttle)"
    }
    if ($Mode -eq "ai") {
        $options += "--generic=socket,in,$($config.rate_hz),127.0.0.1,$($config.control_port),udp,physical-ai-action"
    }
    $disabled = if ($Mode -eq "ai") { "true" } else { "false" }
    foreach ($device in @("joystick", "event", "hid", "mouse", "keyboard")) {
        $options += "--prop:/sim/input/no-$device-input=$disabled"
    }
} elseif ($Mode -eq "ai") { throw "AI mode requires an approach scenario" }
if ($config.on_ground) { $options += "--on-ground" }
if ($config.freeze) { $options += "--enable-freeze" }
Write-Host "Aircraft: $($config.aircraft), airport: $($config.airport), runway: $($config.runway)"
Write-Host "Log: $log"
& $Executable @options
exit $LASTEXITCODE
