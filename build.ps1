param([switch]$CpuOnly)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$vswhere = Join-Path ([Environment]::GetEnvironmentVariable('ProgramFiles(x86)')) 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "Visual Studio C++ build tools required" }
Import-Module (Join-Path $vs 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
New-Item -ItemType Directory -Path 'build' -Force | Out-Null
$flags = @('/nologo','/O2','/EHsc','/std:c++17','/arch:AVX2','/openmp','/W3')
$extra = @()
if (-not $CpuOnly) {
    if (-not $env:CUDA_PATH) { throw "CUDA_PATH required (or use -CpuOnly)" }
    $flags += @('/DAVEC_CUDA', "/I$env:CUDA_PATH\include")
    & cl @flags 'gpu.cpp' '/c' '/Fo:build\gpu.obj'
    if ($LASTEXITCODE) { throw "CUDA host compilation failed" }
    $extra = @('build\gpu.obj', "/LIBPATH:$env:CUDA_PATH\lib\x64", 'cudart.lib','cublas.lib','cuda.lib','nvrtc.lib')
    $env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
}
& cl @flags 'essai.cpp' 'bench.cpp' '/Fo:build\' '/Fe:build\bench.exe' '/link' @extra
if ($LASTEXITCODE) { throw "C++ build failed" }
Get-ChildItem -LiteralPath 'deps\openblas\bin' -Filter '*.dll' | Copy-Item -Destination 'build' -Force
& '.\build\bench.exe' --self-test
if ($LASTEXITCODE) { throw "CPU correctness check failed" }
if (-not $CpuOnly) {
    & '.\build\bench.exe' --self-test --backend gpu --precision fp32
    if ($LASTEXITCODE) { throw "CUDA FP32 correctness check failed" }
    & '.\build\bench.exe' --self-test --backend gpu --precision fp16
    if ($LASTEXITCODE) { throw "CUDA FP16 correctness check failed" }
}
