$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$zip = 'deps\OpenBLAS-0.3.28-x64.zip'
$expected = '4CBD0E5DAA3FB083B18F5E5FA6EEFE79E2F2C51A6D539F98A3C6309A21160042'
New-Item -ItemType Directory -Path 'deps' -Force | Out-Null
if (-not (Test-Path -LiteralPath $zip)) {
    Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.28/OpenBLAS-0.3.28-x64.zip' -OutFile $zip
}
if ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash -ne $expected) {
    throw 'OpenBLAS archive hash mismatch'
}
if (-not (Test-Path -LiteralPath 'deps\openblas')) {
    Expand-Archive -LiteralPath $zip -DestinationPath 'deps\openblas'
}
