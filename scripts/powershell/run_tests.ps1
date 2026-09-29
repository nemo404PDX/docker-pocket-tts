# run_tests.ps1 — run the pytest suite with the project venv.
# Windows counterpart of: .venv/Scripts/python -m pytest tests/ -q
#
# Usage:
#   pwsh ./scripts/powershell/run_tests.ps1              # full suite, quiet
#   pwsh ./scripts/powershell/run_tests.ps1 -TestPath tests/test_normalize.py

[CmdletBinding()]
param(
    [string]$TestPath = 'tests/'
)

$ErrorActionPreference = 'Stop'
$ServiceRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $ServiceRoot

$VenvPy = Join-Path $ServiceRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $VenvPy)) {
    throw "No .venv found at $ServiceRoot\.venv — run install_pocket_tts.ps1 first."
}

& $VenvPy -m pytest $TestPath -q
exit $LASTEXITCODE
