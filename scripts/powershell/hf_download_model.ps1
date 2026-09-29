# hf_download_model.ps1 — download exactly the HF files the active PocketTTS config
# references, into the HF cache layout, so the server loads from disk without hitting
# HuggingFace at startup. Windows counterpart of scripts/bash/hf_download_model.sh.
#
# Unlike a bare `hf download <repo>` (which pulls every language/variant in the repo),
# this parses the hf:// URIs out of the model config and fetches only those pinned files —
# the same call (hf_hub_download) the app makes at load time, so it can't drift.
#
# Usage:
#   pwsh ./scripts/powershell/hf_download_model.ps1                  # default 'english' config
#   pwsh ./scripts/powershell/hf_download_model.ps1 -ModelPath english_2026-09
#   pwsh ./scripts/powershell/hf_download_model.ps1 -Token hf_xxx    # explicit token

[CmdletBinding()]
param(
    [string]$Token = '',
    [string]$ModelPath = ''
)

$ErrorActionPreference = 'Stop'
$ServiceRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $ServiceRoot

# --- Load .env into the process environment (KEY=VALUE lines, no overwrite). ---
function Read-DotEnv {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return }
    foreach ($line in Get-Content -LiteralPath $Path) {
        $t = $line.Trim()
        if ($t -eq '' -or $t.StartsWith('#')) { continue }
        if ($t -match '^export\s+') { $t = $t -replace '^export\s+', '' }
        $idx = $t.IndexOf('=')
        if ($idx -lt 1) { continue }
        $key = $t.Substring(0, $idx).Trim()
        $val = $t.Substring($idx + 1).Trim().Trim('"').Trim("'")
        if (-not (Test-Path "env:$key")) { Set-Item "env:$key" -Value $val }
    }
}

$EnvFile = Join-Path $ServiceRoot '.env'
if (-not (Test-Path $EnvFile)) {
    Write-Warning ".env not found at $EnvFile — relying on process environment only."
} else {
    Read-DotEnv -Path $EnvFile
}

# --- Locate the venv python. ---
$VenvPy = Join-Path $ServiceRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $VenvPy)) {
    throw "No .venv found at $ServiceRoot\.venv — run install_pocket_tts.ps1 first."
}

# --- Token resolution mirrors app/services/tts.py::_load_hf_token(): explicit arg,
#     then HF_TOKEN env (set by .env), then the file at HF_TOKEN_FILE. This lets local
#     dev use the same token-file convention as Docker without a literal secret in .env. ---
$UserToken = $Token
if (-not $UserToken) { $UserToken = $env:HF_TOKEN }
if ((-not $UserToken -or $UserToken.Trim() -eq '') -and $env:HF_TOKEN_FILE) {
    $tokenFile = Resolve-Path -Path $env:HF_TOKEN_FILE -ErrorAction SilentlyContinue
    if ($tokenFile -and (Test-Path $tokenFile)) {
        $UserToken = ((Get-Content -LiteralPath $tokenFile -Raw) -replace '\s', '')
    }
}
if (-not $UserToken -or $UserToken.Trim() -eq '') {
    throw "USER_TOKEN not provided (pass -Token, set HF_TOKEN in .env, or point HF_TOKEN_FILE at a token file)."
}

# --- The app resolves weights via hf_hub_download(), which reads the Hugging Face cache
#     layout under HUGGINGFACE_HUB_CACHE (= POCKET_TTS_MODEL_CACHE_DIR). So we populate
#     that cache dir — NOT a flat --local-dir, which the app never looks at. ---
$CacheDir = $env:POCKET_TTS_MODEL_CACHE_DIR
if (-not $CacheDir) { $CacheDir = Join-Path $ServiceRoot 'data\models\pocket-tts' }

# Model config to resolve (variant name or .yaml path). Defaults to POCKET_TTS_MODEL_PATH, else 'english'.
$Model = $ModelPath
if (-not $Model) { $Model = $env:POCKET_TTS_MODEL_PATH }
if (-not $Model) { $Model = 'english' }

$env:HF_TOKEN = $UserToken
$env:HUGGINGFACE_HUB_CACHE = $CacheDir
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

Write-Host "Downloading config '$Model' files into HF cache: $CacheDir"
& $VenvPy (Join-Path $PSScriptRoot '_hf_download.py') $Model
if ($LASTEXITCODE -ne 0) { throw "download failed (exit $LASTEXITCODE)." }

Write-Host ""
Write-Host "Done. The server will now load from this cache without hitting HuggingFace."
