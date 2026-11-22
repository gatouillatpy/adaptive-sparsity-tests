param([switch]$CpuOnly, [int[]]$Seeds = @(42,43,44), [int[]]$Threads = @(1,8),
      [int]$Queries = 24, [int]$Repeats = 3, [string]$Output = "")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if (-not $Output) { $Output = 'results\' + (Get-Date -Format 'yyyyMMdd-HHmmss') }
if (Test-Path -LiteralPath $Output) { throw "Output directory exists; use a new run name" }
New-Item -ItemType Directory -Path $Output | Out-Null
if ($env:CUDA_PATH) { $env:PATH = "$env:CUDA_PATH\bin;$env:PATH" }
$info = [ordered]@{
    date = (Get-Date -Format o)
    cpu = (Get-ItemProperty -LiteralPath 'HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor\0').ProcessorNameString
    logical_processors = [Environment]::ProcessorCount
    os = [Environment]::OSVersion.VersionString
    cuda_path = $env:CUDA_PATH
    seeds = $Seeds; threads = $Threads; queries = $Queries; repeats = $Repeats
    command = $MyInvocation.Line
    input = 'Synthetic Gaussian weights, Gaussian or ReLU input vectors; no model checkpoints.'
    order = 'Dense first; keep budgets .05,.10,.15 next. Four warmup calls per path.'
}
& nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv | Set-Content -LiteralPath (Join-Path $Output 'gpu.txt') -Encoding UTF8
$info | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $Output 'machine.json') -Encoding UTF8
Get-ChildItem -Path '*.cpp','*.hpp','*.cu','*.ps1','CMakeLists.txt' | Get-FileHash -Algorithm SHA256 |
    Select-Object @{Name='File';Expression={Split-Path $_.Path -Leaf}},Hash |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Output 'sources.json') -Encoding UTF8
function Mesure([string]$Name, [string[]]$BenchArgs) {
    Write-Host $Name
    $log = Join-Path $Output "$Name.log"
    $ErrorActionPreference = 'Continue'
    & '.\build\bench.exe' @BenchArgs --csv (Join-Path $Output "$Name.csv") 2> $log
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($code) { Get-Content -LiteralPath $log; throw "Benchmark failed: $Name" }
}
foreach ($seed in $Seeds) {
    foreach ($preset in @('gpt2','llama')) {
        foreach ($dist in @('gaussian','relu')) {
            $base = @('--preset',$preset,'--seed',"$seed",'--dist',$dist,'--queries',"$Queries",'--repeats',"$Repeats")
            foreach ($thread in $Threads) {
                Mesure "$preset-$dist-cpu-t$thread-s$seed" ($base + @('--threads',"$thread"))
            }
            if (-not $CpuOnly) {
                foreach ($precision in @('fp32','fp16')) {
                    Mesure "$preset-$dist-gpu-$precision-s$seed" ($base + @('--backend','gpu','--precision',$precision))
                }
            }
        }
    }
}
Write-Host "Saved to $Output"
