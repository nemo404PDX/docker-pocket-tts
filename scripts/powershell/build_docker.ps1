# build_docker.ps1 — build and start the Docker container.
# Windows counterpart of scripts/bash/build_docker.sh.
#
# Usage:
#   pwsh ./scripts/powershell/build_docker.ps1            # up --build (foreground)
#   pwsh ./scripts/powershell/build_docker.ps1 -Detach    # run in background

[CmdletBinding()]
param(
    [switch]$Detach
)

$ErrorActionPreference = 'Stop'
$ServiceRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $ServiceRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "docker not found on PATH. Start Docker Desktop and retry."
}

$args = @('compose', '-f', 'deploy/docker-compose.yml', 'up', '--build')
if ($Detach) { $args += '-d' }
& docker @args
exit $LASTEXITCODE
