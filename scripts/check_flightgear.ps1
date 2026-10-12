param(
    [ValidateSet("c172p", "747-400", "f-14b")][string]$ExpectedAircraft,
    [int]$TimeoutSeconds = 120
)
$ErrorActionPreference = "Stop"
$base = "http://127.0.0.1:8080/json"
function Read-Value([string]$Path) {
    (Invoke-RestMethod ($base + $Path) -TimeoutSec 5).value
}
# Joystick properties depend on how bindings were loaded; report absence instead of failing.
function Read-Optional([string]$Path) {
    try { Read-Value $Path } catch { "(missing: $Path)" }
}
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$ready = $false
while ((Get-Date) -lt $deadline) {
    try {
        $scenery = Read-Value "/sim/sceneryloaded"
        $fdm = Read-Value "/sim/signals/fdm-initialized"
        if ($scenery -eq $true -and $fdm -eq $true) { $ready = $true; break }
    } catch { }
    Start-Sleep -Seconds 2
}
if (!$ready) { throw "FlightGear scenery/FDM not ready after $TimeoutSeconds seconds; inspect its log." }
$aircraft = Read-Value "/sim/aircraft"
if ($ExpectedAircraft -and $aircraft -ne $ExpectedAircraft) {
    throw "Expected $ExpectedAircraft but running $aircraft"
}
$altitude = Read-Value "/position/altitude-ft"
$ground = Read-Value "/position/ground-elev-ft"
foreach ($value in @($altitude, $ground)) {
    if ($null -eq $value -or [double]::IsNaN([double]$value) -or [double]::IsInfinity([double]$value) -or $value -eq -9999) {
        throw "Altitude/ground elevation is not initialized"
    }
}
[ordered]@{
    aircraft = $aircraft
    scenery_loaded = $scenery
    fdm_initialized = $fdm
    altitude_ft = $altitude
    ground_elevation_ft = $ground
    paused = Read-Value "/sim/freeze/master"
    clock_paused = Read-Value "/sim/freeze/clock"
    joystick = Read-Optional "/input/joysticks/js/id"
    joystick_config = Read-Optional "/input/joysticks/js/source"
    joystick_axis4 = Read-Optional "/input/joysticks/js/axis[4]/desc"
    scope = "Initialization only; not a flight-performance or input-mapping test"
} | ConvertTo-Json
