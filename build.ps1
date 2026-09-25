$ErrorActionPreference = "Stop"

$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $project ".venv\Scripts\python.exe"
$versionFile = Join-Path $project "version.py"
$releaseDir = Join-Path $project "releases"
if (-not (Test-Path $python)) {
    py -3.12 -m venv (Join-Path $project ".venv")
    $python = Join-Path $project ".venv\Scripts\python.exe"
}

$versionSource = [System.IO.File]::ReadAllText($versionFile)
$versionMatch = [regex]::Match($versionSource, 'APP_VERSION\s*=\s*"(?<major>\d+)\.(?<minor>\d+)\.(?<patch>\d+)"')
if (-not $versionMatch.Success) { throw "version.py 中未找到有效的主.次.修订版本号。" }
$major = [int]$versionMatch.Groups['major'].Value
$minor = [int]$versionMatch.Groups['minor'].Value
$patch = [int]$versionMatch.Groups['patch'].Value
New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null
do {
    $version = "$major.$minor.$patch"
    $zipPath = Join-Path $releaseDir "星点柔焦-v$version-win-x64.zip"
    if (Test-Path $zipPath) {
        $patch += 1
        $versionSource = "APP_VERSION = `"$major.$minor.$patch`"`n"
        [System.IO.File]::WriteAllText($versionFile, $versionSource, [System.Text.UTF8Encoding]::new($false))
    }
} while (Test-Path $zipPath)

& $python -m pip install -r (Join-Path $project "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "依赖安装失败。" }

$buildRoot = Join-Path $env:TEMP ("starsoft-build-" + [guid]::NewGuid().ToString("N"))
$distDir = Join-Path $buildRoot "dist"
$workDir = Join-Path $buildRoot "work"
$packageDir = Join-Path $buildRoot "package"
$solverDir = Join-Path $buildRoot "solver"
try {
    New-Item -ItemType Directory -Path $distDir, $workDir, $packageDir -Force | Out-Null
    $solverScript = Join-Path $project "scripts\prepare_astap.py"
    & $python $solverScript --output $solverDir --platform windows
    if ($LASTEXITCODE -ne 0) { throw "下载或整理本机板解算组件失败。" }
    $uiFile = Join-Path $project "ui\index.html"
    $dataArgument = "$uiFile;ui"
    $starNamesFile = Join-Path $project "data\hyg_named_stars.csv"
    if (-not (Test-Path -LiteralPath $starNamesFile)) { throw "缺少本地恒星名称索引：$starNamesFile" }
    $starNamesArgument = "$starNamesFile;data"
    $exePath = Join-Path $distDir "星点柔焦.exe"

    Push-Location $project
    try {
        & $python -m PyInstaller --noconfirm --clean --onefile --windowed --name "星点柔焦" --distpath $distDir --workpath $workDir --specpath $buildRoot --collect-all rawpy --collect-all sep --collect-all tifffile --collect-all astropy --collect-all seiza --add-data $dataArgument --add-data $starNamesArgument app.py
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller 打包失败。" }
    } finally {
        Pop-Location
    }

    Copy-Item -LiteralPath $exePath -Destination (Join-Path $packageDir "星点柔焦.exe")
    Copy-Item -LiteralPath $solverDir -Destination (Join-Path $packageDir "solver") -Recurse
    Copy-Item -LiteralPath (Join-Path $project "README.md") -Destination $packageDir
    $licenseFile = Join-Path $project "LICENSE"
    if (Test-Path $licenseFile) { Copy-Item -LiteralPath $licenseFile -Destination $packageDir }
    $thirdPartyLicense = Join-Path $project "licenses\Seiza-Apache-2.0.txt"
    if (Test-Path $thirdPartyLicense) { Copy-Item -LiteralPath $thirdPartyLicense -Destination $packageDir }
    $thirdPartyNotice = Join-Path $project "licenses\THIRD_PARTY_NOTICES.txt"
    if (Test-Path $thirdPartyNotice) { Copy-Item -LiteralPath $thirdPartyNotice -Destination $packageDir }
    $hygLicense = Join-Path $project "licenses\CC-BY-SA-4.0.txt"
    if (Test-Path $hygLicense) { Copy-Item -LiteralPath $hygLicense -Destination $packageDir }
    Set-Content -LiteralPath (Join-Path $packageDir "VERSION.txt") -Value $version -Encoding utf8
    $exeHash = (Get-FileHash -LiteralPath (Join-Path $packageDir "星点柔焦.exe") -Algorithm SHA256).Hash.ToLowerInvariant()
    Set-Content -LiteralPath (Join-Path $packageDir "SHA256SUMS.txt") -Value "$exeHash  星点柔焦.exe" -Encoding utf8
    Compress-Archive -Path (Join-Path $packageDir "*") -DestinationPath $zipPath -CompressionLevel Optimal
    Write-Host "完成：$zipPath"
} finally {
    if (Test-Path $buildRoot) { Remove-Item -LiteralPath $buildRoot -Recurse -Force }
}
