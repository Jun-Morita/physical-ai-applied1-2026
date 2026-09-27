param(
    [ValidateSet("747-400", "f-14b")][string]$Aircraft = "747-400",
    [string]$Scenario = "",
    [string]$FlightGearRoot = "$env:USERPROFILE/FlightGear",
    [string]$Executable = "C:/Program Files/FlightGear 2024.1/bin/fgfs.exe"
)
$ErrorActionPreference = "Stop"
if (Get-Process fgfs -ErrorAction SilentlyContinue) {
    throw "Close the existing FlightGear instance before starting another scenario."
}
if (!$Scenario) { $Scenario = "$PSScriptRoot/../config/scenarios/$Aircraft-rjtt-check.json" }
$config = Get-Content -LiteralPath $Scenario -Raw | ConvertFrom-Json
if ($config.aircraft -notin @("747-400", "f-14b")) { throw "Unsupported aircraft" }
$aircraftRoot = "$FlightGearRoot/Custom Aircraft/physical-ai"
$searchPath = "$aircraftRoot;$aircraftRoot/zaretto/aircraft"
$model = if ($config.aircraft -eq "747-400") { "$aircraftRoot/747-400/747-400-set.xml" } else { "$aircraftRoot/zaretto/aircraft/f-14b/f-14b-set.xml" }
foreach ($path in @($Executable, "$FlightGearRoot/Downloads/fgdata_2024_1/version", $model)) {
    if (!(Test-Path -LiteralPath $path)) { throw "Missing installation file: $path" }
}
$logDir = "$FlightGearRoot/Logs/physical-ai"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log = "$logDir/$($config.aircraft)-$stamp"
New-Item -ItemType Directory -Force -Path $log | Out-Null
$config | ConvertTo-Json | Set-Content -Encoding UTF8 "$logDir/$($config.aircraft)-$stamp.scenario.json"
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
if ($config.on_ground) { $options += "--on-ground" }
if ($config.freeze) { $options += "--enable-freeze" }
Write-Host "Aircraft: $($config.aircraft), airport: $($config.airport), runway: $($config.runway)"
Write-Host "Log: $log"
& $Executable @options
exit $LASTEXITCODE
