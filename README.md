# adaptive-sparsity-tests

Small C++/CUDA experiment: use SimHash buckets to choose rows of a dense layer before doing the dot products. Compare the complete sparse path with OpenBLAS or cuBLAS, including selection overhead.

## Idee

For input x and row w_i, try to identify large activations without evaluating every w_i dot x.

- Build L tables, each using K random Gaussian hyperplanes.
- Hash each weight row by the signs of its projections.
- Query the matching buckets for x; optional one-bit probes.
- Rank candidates by collision count, truncate to the requested row budget, then sort row IDs for locality.
- Compute selected rows and leave the others at zero.

This is approximate candidate selection, not an exact top-k algorithm. SimHash approximates angular similarity, whereas large dot products also depend on norms. Collision counts are a weak ranking signal for independent high-dimensional random vectors. Linear activation queries both x and -x to include large negative responses; GELU/ReLU queries x only.

Reference: [BOLT, arXiv:2303.17727](https://arxiv.org/abs/2303.17727), especially sections 3 and 4.3. The paper describes sparse training/inference for search and recommendation. This code borrows the LSH idea; it does not reproduce BOLT, its training procedure, its autotuning, or its published results. Hash functions are fixed random projections, not learned MONGOOSE hashes.

## Scope

One batch-1, bias-free projection y = activation(Wx), with skipped entries zeroed.

| Preset | Input dimension | Neurons | Meaning |
| --- | ---: | ---: | --- |
| gpt2 | 768 | 3072 | GPT-2-small-sized up projection |
| llama | 4096 | 11008 | LLaMA-7B-sized up projection |

The names specify dimensions only. Default weights are Gaussian with standard deviation 1/sqrt(input dimension); inputs are Gaussian or their ReLU. Default activation is tanh-approximate GELU for both presets. There are no pretrained checkpoints, attention, down projections, SwiGLU gates, tokenization, text prompts, training, or perplexity evaluations. In particular, the llama preset is not a complete LLaMA FFN.

CPU uses FP32. GPU uses FP32 or FP16 weights/inputs with FP32 accumulation/output. The GPU selector stays on the CPU and uses the FP32 master weights and input; the two GPU execution paths use identical quantized operands. This is not a GPU-resident LSH implementation.

## Build / run on Windows

Prerequisites: x64 Visual Studio C++ tools; CUDA toolkit for GPU runs. Local validation used MSVC 14.51, CUDA 12.8, and OpenBLAS 0.3.28.

From the repository directory:

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1

.\build\bench.exe --preset gpt2 --threads 1
.\build\bench.exe --preset llama --backend gpu --precision fp16

powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1
~~~

setup.ps1 downloads the official [OpenBLAS Windows x64 archive](https://github.com/OpenMathLib/OpenBLAS/releases/tag/v0.3.28) into deps/ and checks its pinned SHA-256. Dependencies and binaries are excluded from Git. It does not install them globally. To build/run only the CPU path, pass -CpuOnly to both build.ps1 and run.ps1.

GPU kernels are compiled once at process startup through NVRTC, then launched through the CUDA driver API. This avoids the installed nvcc/Visual Studio header incompatibility. JIT time, allocation, and weight upload are outside steady-state timings. Set CUDA_PATH to your toolkit and include its bin directory in PATH when invoking the executable directly. The scripts handle PATH for their own process. Run from this directory so gpu.cu can be read.

The sparse CPU loop uses AVX2 and OpenMP. OpenBLAS and that loop use the same requested thread count. The selector itself is serial.

An optional CMake build is also included (validated with CMake 3.26 and NMake). It requires OpenMP, a discoverable OpenBLAS shared library at runtime, and the CUDA toolkit when configured with -DAVEC_CUDA=ON. On Linux, set CUDA_PATH to the toolkit directory for NVRTC includes. Both the Windows script build and the CMake/NMake CUDA build passed their three correctness suites on this machine. Linux has not been tested.

## Measurements

Recorded run: [20261122-030655](results/20261122-030655/summary.md), November 22, 2026.

- AMD Ryzen 9 5950X, 16 cores / 32 logical processors.
- NVIDIA GeForce GTX 1070, 8 GiB, compute capability 6.1; driver 572.61.
- Windows build 26200; CUDA 12.8; OpenBLAS 0.3.28 (Zen kernels).
- Three seeds: 42, 43, 44.
- Two shapes, Gaussian/ReLU inputs, CPU 1/8 threads, GPU FP32/FP16.
- Row budgets 5%, 10%, 15%; K=8 bits, L=16 tables, exact buckets.
- 24 input vectors per run, repeated three times; four warmup calls per execution path.
- 48 processes, 144 configuration rows. Repeats reuse the same inputs; they are not independent data samples.

Selected Gaussian-input results at a 5% row budget:

| Shape | Backend | Dense ms | Sparse total ms | Speedup | Relative MSE | Cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 768 x 3072 | CPU, 1 thread | 0.131 | 0.029 | 4.62x | 0.930 | 0.264 |
| 768 x 3072 | CPU, 8 threads | 0.100 | 0.018 | 5.66x | 0.930 | 0.264 |
| 768 x 3072 | GPU, FP32 | 0.088 | 0.084 | 1.05x | 0.930 | 0.264 |
| 768 x 3072 | GPU, FP16 | 0.075 | 0.081 | 0.93x | 0.930 | 0.264 |
| 4096 x 11008 | CPU, 1 thread | 7.737 | 0.619 | 12.50x | 0.942 | 0.240 |
| 4096 x 11008 | CPU, 8 threads | 6.372 | 0.384 | 16.43x | 0.942 | 0.240 |
| 4096 x 11008 | GPU, FP32 | 0.883 | 0.201 | 4.38x | 0.942 | 0.240 |
| 4096 x 11008 | GPU, FP16 | 0.530 | 0.208 | 2.55x | 0.942 | 0.240 |

Cells summarize medians across the three seeds. Each latency is a median over per-vector timings within that run; speedup is the median of the per-seed dense/sparse ratios, so it need not equal the ratio of the displayed medians. The full summary also gives seed ranges and all configurations.

**The high error is the main result.** Relative MSE around 0.93–0.94 and cosine around 0.24–0.26 do not show a useful approximation. Most dense output energy is missing. A zero output has relative MSE 1 when the reference is nonzero. The sparse path can be fast largely because it discards useful work; no end-to-end LLM acceleration or preserved model quality is established.

On the small shape the GPU is near parity or slower; on the large shape it is faster at this poor quality level. This does not confirm the remembered 2023 outcome that GPU execution rarely beat cuBLAS. Different code, data, and hardware can change that result. The notes mention an RTX 3090; the detected GPU is a GTX 1070.

### What is timed

- Dense CPU: OpenBLAS SGEMV plus activation.
- Sparse CPU: hashing, bucket lookup, deduplication, candidate truncation/sorting, output zeroing, selected dot products, and activation.
- Dense GPU: host FP16 conversion when requested, input H2D transfer, cuBLAS SGEMV (FP32) or GemmEx (FP16), activation kernel, and full output D2H transfer.
- Sparse GPU: CPU selection plus input/ID transfers, output zeroing, selected CUDA dot products/activation, and full output D2H transfer.

Weights stay resident. Allocations, initial index construction, and NVRTC compilation are excluded; index build time is reported separately. There is no amortization claim across model updates. CPU/GPU figures include different execution/transfer costs and are not device-only kernel throughput.

Dense timings run first, then the row budgets. CPU affinity, clocks, power, and other desktop activity are not controlled. The display GPU is shared with Windows. There are only three seeds; the reported seed range is not a confidence interval.

### Reading the CSV

- keep_budget is an upper bound, not guaranteed occupancy. With exact buckets, only about 6% of rows are retrieved here, so 10% and 15% budgets often produce the same mask. No random fallback is inserted to fill it.
- kept_mean is the actual fraction evaluated; kept_min/max count selected rows over inputs.
- select_p50_ms, eval_p50_ms, and total_p50_ms are separate medians; component medians need not sum exactly to the total median.
- MSE = mean((sparse-dense)^2). Relative MSE = sum((sparse-dense)^2)/sum(dense^2). Metrics cover the entire post-activation vector, including skipped entries.
- oracle_mse uses the largest dense activation magnitudes at the same actual row count. It is a post-hoc quality bound, not a fast execution path.
- random_mse masks the dense result at uniformly shuffled rows with the same count. It is a selection-quality control, not a timed algorithm.
- oracle_recall is overlap with that top-magnitude mask; cosine is against the same-precision dense output.
- index_mib counts allocated projections, bucket storage, row IDs, and counters; it excludes allocator bookkeeping and temporary index-build workspace. weights_mib is the host FP32 matrix size, including for GPU FP16 runs. These are not total-process or total-device memory measurements.

Raw CSVs, stderr logs, machine metadata, and a SHA-256 snapshot of the benchmark sources are in the run directory. Windows PowerShell wraps the first stderr line as NativeCommandError in logs even for successful processes; the runner separately checks exit codes. CSVs are the numeric record.

To regenerate the table (standard-library Python only):

~~~powershell
py -3 .\resume_resultats.py .\results\20261122-030655
~~~

If your Python launcher has no Python 3 registered, invoke the Python 3 executable directly.

CMake verification, from a Visual Studio developer shell with OpenBLAS available on PATH:

~~~powershell
$env:PATH = "$PWD\build;$env:CUDA_PATH\bin;$env:PATH"
cmake -S . -B build\cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DAVEC_CUDA=ON
cmake --build build\cmake
ctest --test-dir build\cmake --output-on-failure
~~~

## More experiments

~~~powershell
.\build\bench.exe --preset llama --bits 6 --tables 16 --probes 1 --keep .05,.1,.2 --csv build\probes.csv
.\build\bench.exe --dim 128 --width 512 --activation linear --keep 1 --queries 8
.\build\bench.exe --help
~~~

--weights and --inputs accept raw little-endian FP32, row-major, exact sizes width*dim and queries*dim respectively. Specify dimensions/query count explicitly. A GPT-2 Conv1D checkpoint matrix may need transposition to become neuron rows. No checkpoint loader is included.

## Checks

build.ps1 runs CPU and GPU FP32/FP16 correctness checks:
- Full-selection agreement with dense execution on non-square matrices and dimensions not divisible by the vector width.
- Matching-row retrieval, candidate deduplication, budget enforcement, empty selection, and opposite-sign lookup.
- Hand-calculated MSE/oracle masks.
- GPU sparse subset and empty-output checks.

All three suites passed on the machine above. These checks establish implementation consistency, not task-level model accuracy.

## Next

- Start with real checkpoint weights and captured hidden states; measure full FFN output error.
- Tune table count, bits, and probing at a fixed quality target, rather than comparing speed at arbitrary sparsity.
- Test angular hashing against a norm-aware selection method.
- Try block-level selection to reduce irregular memory access.
- Only then attempt token-level throughput and perplexity.
