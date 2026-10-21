#Requires -RunAsAdministrator
param(
    [Parameter(Mandatory=$true)][System.Net.IPAddress]$ShieldIP,
    [Parameter(Mandatory=$true)][string]$Program,
    [ValidateRange(1,65535)][int]$Port = 6677
)
$ErrorActionPreference = 'Stop'
if ($ShieldIP.AddressFamily -ne [System.Net.Sockets.AddressFamily]::InterNetwork) {
    throw 'ShieldIP must be IPv4'
}
$resolvedProgram = (Resolve-Path -LiteralPath $Program).Path
$ruleName = "EMGViewer-UDP-$ShieldIP-$Port"
Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -Name $ruleName -DisplayName "EMG Viewer UDP $ShieldIP port $Port" `
    -Direction Inbound -Action Allow -Protocol UDP -LocalPort $Port `
    -RemoteAddress $ShieldIP.IPAddressToString -Program $resolvedProgram -Profile Any | Out-Null
Write-Host "Allowed UDP $Port from $ShieldIP for $resolvedProgram"
