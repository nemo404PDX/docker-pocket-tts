# docker_curl_test.ps1 — POST a TTS request to the running server and save the audio.
# Windows counterpart of scripts/bash/docker_curl_test.sh (works against Docker or local).
# Uses curl.exe, which ships with Windows 10/11 — same call as the bash script.
#
# Usage:
#   pwsh ./scripts/powershell/docker_curl_test.ps1
#   pwsh ./scripts/powershell/docker_curl_test.ps1 -Port 49112 -Text "hello from docker"

[CmdletBinding()]
param(
    [string]$BindHost = 'localhost',
    [int]$Port = 49112,
    [string]$Text = 'hello from docker',
    [string]$Voice = 'default.wav',
    [ValidateSet('wav','mp3','flac','pcm')][string]$Format = 'wav'
)

$ErrorActionPreference = 'Stop'
if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    throw "curl.exe not found on PATH."
}

$ServiceRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $ServiceRoot
$OutFile = Join-Path $ServiceRoot ("test.$Format")

$body = @{ input = $Text; voice = $Voice; response_format = $Format } | ConvertTo-Json -Compress
$uri  = "http://${BindHost}:${Port}/tts"

Write-Host "POST $uri  (voice=$Voice, format=$Format)"
& curl.exe -sS -X POST $uri -H 'Content-Type: application/json' -d $body --output $OutFile
if ($LASTEXITCODE -ne 0) { throw "curl failed with exit code $LASTEXITCODE." }

Write-Host "Saved: $OutFile"
