# tandem.cpp-amd

**Experimental.** An AMD (HIP/ROCm) port of [tandem.cpp](https://github.com/ewhacc/tandem.cpp), built for tests in the ITLab lab. On our RX 9070 XT it did not beat [Strata](https://github.com/Niko1221/Strata), so we stopped working on it. We publish it as is, for anyone who wants to try tandem.cpp on their own AMD hardware.

Write-up: [English](https://itlab.cl/en/tandem-cpp-on-amd-port-rx-9070-xt/) · [Español](https://itlab.cl/tandem-cpp-en-amd-port-rx-9070-xt/)

No guarantees: it ran on one card, with one model, in short tests. Read [Known weak points](#known-weak-points) and [Not validated](#not-validated) before you rely on it.

## Credit

All the real work is in these two projects. This repository only adds 47 lines and removes 10 so that it builds and runs with HIP.

- **[tandem.cpp](https://github.com/ewhacc/tandem.cpp)** by ewhacc (MIT): the patch set that keeps MoE experts in host RAM, caches the most-used ones in VRAM, and splits the misses between the CPU and uploads. Upstream supports CUDA on Linux only.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** by the ggml authors (MIT).

This port is not affiliated with ewhacc or ggml-org. Please do not report problems with this port to them.

## What is in this repository

| File | Content |
|---|---|
| `tandem-amd.patch` | The AMD port, applied on top of `tandem.patch` |
| `build.sh` | Clones llama.cpp, downloads `tandem.patch` and checks its SHA-256, applies both patches, builds `llama-server` with HIP |
| `LLAMA_CPP_COMMIT` | llama.cpp base: `f5222f25c18da062ab2b8a2dbbfbf4feae398566` |
| `TANDEM_COMMIT` | tandem.cpp commit: `e1acb8877bfcf92b4ae7c9f0e0f1d896dbf6c683` |

`tandem.patch` is not copied here. `build.sh` downloads it from the tandem.cpp repository at that commit and checks SHA-256 `8a2e5898d4b750127f9fc474a28b7781615724ea1dbac4ac1c35633df1c1fb5d`.

## Build

Needs ROCm (we used 7.2.4) and its `amdclang`/`amdclang++`.

```
bash build.sh                              # gfx1201 (RX 9070 / 9070 XT), ROCm in /opt/rocm
AMDGPU_TARGETS=gfx1100 bash build.sh       # another GPU
NO_BUILD=1 bash build.sh                   # only prepare the sources
```

or by hand:

```
git clone https://github.com/ggml-org/llama.cpp tandem-amd && cd tandem-amd
git checkout f5222f25c18da062ab2b8a2dbbfbf4feae398566
curl -fsSLO https://raw.githubusercontent.com/ewhacc/tandem.cpp/e1acb8877bfcf92b4ae7c9f0e0f1d896dbf6c683/tandem.patch
git apply tandem.patch
git apply ../tandem-amd.patch
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++
cmake --build build -j --target llama-server
```

Only gfx1201 was tested.

## What the patch changes

| File | Change |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `ggml_cuda_time_ns` and `ggml_cuda_nanosleep` replace `%%globaltimer` and `__nanosleep` (NVIDIA only) in `k_host_add_wait` and `k_moe_doorbell` |
| `ggml/src/ggml-cuda/vendors/hip.h` | Aliases for `cudaEventCreate`, `cudaEventQuery`, `cudaEventElapsedTime`, `cudaErrorNotSupported` |
| `ggml/src/ggml-cuda/hc-fused.cu` | No `cuda_bf16.h` under HIP; explicit width `32` in `__shfl_xor_sync` |
| `ggml/src/ggml-cuda/mmvq.cu` | 64-bit mask in `__ballot_sync`, as HIP requires |
| `src/llama-model.cpp` | Looks for the `ROCm` backend as well as `CUDA`, accepts the `ROCm_Host` buffer, and does not register file-backed memory on ROCm |

The backend name matters: tandem.cpp looks for `"CUDA"`, but under HIP the backend is `"ROCm"`. Without the fix, pinned experts and the per-GPU embedding table are silently disabled.

## Results on an RX 9070 XT

Card: RX 9070 XT, 16 GiB, gfx1201, ROCm 7.2.4. Model: Qwen3.8-Flash-Next IQ3_S (a Swift-1.5 GSQ-RCO variant, 79 GiB) with the MTP head [`ewhacc/qwen38-mtp-outq4_K-expsq4_K`](https://huggingface.co/ewhacc/qwen38-mtp-outq4_K-expsq4_K).

Short runs: 3,000 tokens, MTP with 3 drafts, context 32,768, `LLAMA_PLE_GPU=0`, fixed prompt, seed 42.

| Group | Runs | Decode, mean | Range |
|---|---:|---:|---|
| Pinned experts | 5 | 45.45 tok/s | 34.96 to 53.43 |
| Not pinned | 4 | 34.85 tok/s | 28.73 to 40.62 |

- Best single run: 53.43 tok/s.
- With pinned experts, 144 expert tensors (46.84 GiB) are pinned in host RAM. Pinning takes 23.5 to 63.7 s at load.
- Expert cache: 5.83 GiB for 47.96 GiB of experts, with 58 to 62 % hits.
- For reference, Strata on the same card and model gave 66.26 tok/s sustained over our test battery. **This is not a head-to-head comparison:** the tasks, lengths and contexts were different.

**Correctness check.** We compared this port on the RX 9070 XT with unmodified tandem.cpp on an RTX 3060. Same text, 200-token prefix, 60 positions, no MTP, temperature 0, top-20 logits per position:

| Measure | Result | tandem.cpp README |
|---|---:|---:|
| Same top token | 55 of 60 (91.7 %) | 95.5 % |
| Mean KL divergence | 0.00890 | 0.0093 |
| Max KL divergence | 0.09783 | — |

No sign of an error, but this is a small test. It does not cover MTP or multi-token batches, and it was done before the last `src/llama-model.cpp` changes. Those changes do not touch compute kernels, but the test was not repeated.

## Example run

The settings of the 53.43 tok/s run. Check that about 50 GiB of RAM is free first.

```
LLAMA_MTP_DRAFT_VOCAB=100000 HIP_VISIBLE_DEVICES=0 LLAMA_PLE_GPU=0 \
./build/bin/llama-server -m model-00001-of-00002.gguf \
  --host 127.0.0.1 --port 8091 --no-webui --jinja \
  -ngl 99 -cmoe --moe-cache-mib auto -np 1 -c 32768 \
  --spec-type draft-mtp -md qwen38-mtp-outq4_K-expsq4_K.gguf \
  --spec-draft-n-max 3 --spec-draft-sampling probabilistic \
  -bs -ot '^token_embd\.weight=ROCm0'
```

See the [tandem.cpp README](https://github.com/ewhacc/tandem.cpp) for what each option does.

## Warning: per-GPU embedding table

**Never enable the per-GPU embedding table (`LLAMA_PLE_GPU`) on ROCm without this patch's change to `src/llama-model.cpp`.** Without it, tandem.cpp registers the 27 GiB table that is mapped from the model file with HIP. On our card that hung the `amdgpu` driver: the server stuck in D state in `svm_range_set_attr` for many minutes, and tools such as `ps` hung too. The process only exited about 50 s after `SIGKILL`. Setting `LLAMA_PLE_GPU_COPY=0` does not prevent it.

With the patch, ROCm never registers file-backed memory. We did not find the cause inside the driver. Registering anonymous, resident memory was fast in our tests (4 GiB in 0.094 s).

The patched path copies the table into RAM instead. In the one run we did, without pinned experts, it gave only 9.78 tok/s. We do not know why.

## Known weak points

- **Fixed 100 MHz clock.** `ggml_cuda_time_ns` assumes `wall_clock64` runs at 100 MHz. That is what our card reports (`hipDeviceAttributeWallClockRate` = 100,000 kHz), but it should be read from the device. It only affects the 2 s wait limit and timing, not compute.
- **Busy wait.** `ggml_cuda_nanosleep` spins, because HIP has no direct equivalent of `__nanosleep`.
- **32-thread groups.** Several kernels assume 32-thread warps. That matches gfx12, but we only checked it indirectly, through the correctness test above. GPUs with 64-thread wavefronts (for example gfx9 / CDNA) may fail.
- **ARGSORT warning.** With `-bs`, the log says `device 'ROCm0' does not have support for op ARGSORT needed for sampler 'top-p'`. We did not check its effect.

## Not validated

- Multi-token batches, which MTP uses.
- Quality with expert skipping (`LLAMA_MOE_TAIL_SKIP=7`).
- A full task with a verdict, and our 49-task battery.
- A head-to-head comparison with Strata on the same task, length and context.
- Sustained speed with pinned experts: the figures above are from 3,000-token runs.
- Prompt processing speed with pinned experts.
- Large contexts: we tested 32,768.
- Why pinned runs vary so much (34.96 to 53.43 tok/s).
- Any AMD card other than the RX 9070 XT.

## Licence

MIT, see [LICENSE](LICENSE). tandem.cpp is Copyright (c) 2026 ewhacc, and llama.cpp is Copyright (c) 2023-2026 The ggml authors, both under MIT.
