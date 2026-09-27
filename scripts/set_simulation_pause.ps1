param([switch]$Resume)
$ErrorActionPreference = "Stop"
$uri = "http://127.0.0.1:8080/json/sim/freeze"
foreach ($name in @("master", "clock")) {
    $body = @{ name = $name; type = "bool"; value = (!$Resume.IsPresent) } | ConvertTo-Json
    Invoke-RestMethod $uri -Method Post -ContentType "application/json" -Body $body | Out-Null
}
foreach ($name in @("master", "clock")) {
    $state = Invoke-RestMethod "$uri/$name"
    Write-Host "$name = $($state.value)"
}
