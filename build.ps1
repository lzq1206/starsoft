$ErrorActionPreference = "Stop"

$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $project ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    py -3.12 -m venv (Join-Path $project ".venv")
    $python = Join-Path $project ".venv\Scripts\python.exe"
}

& $python -m pip install -r (Join-Path $project "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "依赖安装失败。" }

Push-Location $project
try {
    & $python -m PyInstaller --noconfirm --clean --onefile --windowed --name "星点柔焦" --collect-all rawpy --collect-all sep --collect-all tifffile --add-data "ui\index.html;ui" app.py
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller 打包失败。" }
} finally {
    Pop-Location
}

Write-Host "完成：$(Join-Path $project 'dist\星点柔焦.exe')"
