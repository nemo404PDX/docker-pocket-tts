# install_pocket_tts.ps1 — create the project venv and install CPU-only deps.
# Windows counterpart of scripts/bash/install_pocket_tts.sh.
#
# Usage:
#   pwsh ./scripts/powershell/install_pocket_tts.ps1            # python 3.11 (matches CI)
#   pwsh ./scripts/powershell/install_pocket_tts.ps1 -Python 3.12

[CmdletBinding()]
param(
    [string]$Python = '3.11'
)

$ErrorActionPreference = 'Stop'
$ServiceRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $ServiceRoot

# Prefer uv (fast, and what this repo's venv was created with); fall back to python -m venv.
if (Get-Command uv -ErrorAction SilentlyContinue) {
    if (-not (Test-Path .venv)) {
        Write-Host "Creating .venv with Python $Python via uv..."
        uv venv --python $Python .venv
    }
    # Install into the existing venv from both requirement files. cpu_requirements.txt
    # carries the CPU-only torch index, so this never pulls multi-GB CUDA wheels.
    uv pip install --python (Join-Path $ServiceRoot '.venv') `
        -r requirements/cpu_requirements.txt `
        -r requirements/dev_requirements.txt
} else {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        throw "Neither 'uv' nor 'python' found on PATH. Install one and retry."
    }
    if (-not (Test-Path .venv)) {
        Write-Host "Creating .venv with Python $Python..."
        python -m venv .venv
    }
    $VenvPy = Join-Path $ServiceRoot '.venv\Scripts\python.exe'
    & $VenvPy -m pip install --upgrade pip setuptools wheel
    & $VenvPy -m pip install `
        -r requirements/cpu_requirements.txt `
        -r requirements/dev_requirements.txt
}

Write-Host ""
Write-Host "Done. Verify with:"
Write-Host "  pwsh ./scripts/powershell/run_tests.ps1"
