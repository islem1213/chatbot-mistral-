<#
.SYNOPSIS
Automates setup and execution of the Mistral AI RAG project on Windows.

.DESCRIPTION
This script activates the local virtual environment, installs dependencies,
ensures a .env file exists, and optionally runs one of the project actions.

.PARAMETER Action
The action to execute after setup. Valid values: ingest, index, app, all.
Default is none.

.EXAMPLE
./setup.ps1 -Action all
# Activates the venv, installs requirements, copies .env, and runs the full pipeline.
#>

param(
    [ValidateSet('ingest','index','app','all','none')]
    [string]$Action = 'none'
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location -Path $projectRoot

$venvActivate = Join-Path -Path $projectRoot -ChildPath '.venv\Scripts\Activate.ps1'
if (-not (Test-Path -Path $venvActivate)) {
    Write-Error "Virtual environment activation script not found: $venvActivate"
    exit 1
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& $venvActivate

if (-not (Test-Path -Path 'requirements.txt')) {
    Write-Error "requirements.txt not found in project root."
    exit 1
}

Write-Host "Installing dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

if (-not (Test-Path -Path '.env')) {
    if (Test-Path -Path '.env.example') {
        Write-Host "Creating .env from .env.example..." -ForegroundColor Cyan
        Copy-Item -Path '.env.example' -Destination '.env'
        Write-Host "A .env file has been created. Please edit it and add your MISTRAL_API_KEY." -ForegroundColor Yellow
    }
    else {
        Write-Host "No .env or .env.example file found. Please create a .env file manually." -ForegroundColor Yellow
    }
}
else {
    Write-Host ".env file already exists." -ForegroundColor Green
}

if ($Action -ne 'none') {
    Write-Host "Running action: $Action" -ForegroundColor Cyan
    python .\run.py $Action
}
else {
    Write-Host "Setup complete. Run `python .\run.py app` or `python .\run.py all` when ready." -ForegroundColor Green
}
