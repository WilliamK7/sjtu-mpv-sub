$ErrorActionPreference = "Stop"

$BaseKey = "HKCU:\Software\Classes\sjtu-mpv"
if (Test-Path $BaseKey) {
    Remove-Item -Path $BaseKey -Recurse -Force
    Write-Host "Unregistered sjtu-mpv:// URL protocol."
} else {
    Write-Host "sjtu-mpv:// URL protocol is not registered."
}
