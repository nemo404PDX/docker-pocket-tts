# run_local.ps1 — start the TTS server locally (no Docker) with the project venv.
# Mirrors `python server.py` but uses the venv interpreter and passes through any extra args.
# server.py loads .env itself, so no env plumbing is needed here.
#
# Usage:
#   pwsh ./scripts/powershell/run_local.ps1                       # defaults from .env / Config
#   pwsh ./scripts/powershell/run_local.ps1 -Port 49112 -BriefingMode
#   pwsh ./scripts/powershell/run_local.ps1 --port 49112          # raw passthrough after the script

[CmdletBinding()]
param(
    [string]$BindHost = '',
    [int]$Port = 0,
    [switch]$BriefingMode,
    [switch]$NoNormalize,
    [ValidateSet('none','paragraph','sentence')][string]$Segmentation = ''
)

$ErrorActionPreference = 'Stop'
$ServiceRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $ServiceRoot

$VenvPy = Join-Path $ServiceRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $VenvPy)) {
    throw "No .venv found at $ServiceRoot\.venv — run install_pocket_tts.ps1 first."
}

$serverArgs = @()
if ($BindHost) { $serverArgs += @('--host', $BindHost) }
if ($Port)   { $serverArgs += @('--port', [string]$Port) }
if ($BriefingMode)  { $serverArgs += '--briefing-mode' }
if ($NoNormalize)   { $serverArgs += '--no-normalize' }
if ($Segmentation)  { $serverArgs += @('--segmentation', $Segmentation) }

# Anything after the script's own params is passed straight through to server.py.
$all = @($serverArgs) + @($args)
Write-Host "Starting local TTS server: & `"$VenvPy`" server.py $($all -join ' ')"
& $VenvPy server.py @all
exit $LASTEXITCODE
