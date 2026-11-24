$ErrorActionPreference = "Stop"

$RootDir = $PSScriptRoot
$WakeVbs = Join-Path $RootDir "wake.vbs"
if (-not (Test-Path $WakeVbs)) {
    throw "wake.vbs not found at $WakeVbs"
}

$BaseKey = "HKCU:\Software\Classes\sjtu-mpv"
$CmdKey = "$BaseKey\shell\open\command"

New-Item -Path $CmdKey -Force | Out-Null
Set-ItemProperty -Path $BaseKey -Name "(Default)" -Value "URL:SJTU MPV Bilingual Player"
Set-ItemProperty -Path $BaseKey -Name "URL Protocol" -Value ""
Set-ItemProperty -Path $CmdKey -Name "(Default)" -Value ('wscript.exe "' + $WakeVbs + '" "%1"')

Write-Host "Registered sjtu-mpv:// -> wscript.exe `"$WakeVbs`" `"%1`""
