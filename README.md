# experimental support of Ryzen 7x40 (Linux)
in my case a 7940HS with 64Go of RAM.with fedora:41/rocm-hip:6.2.1

The backend only add mulmat(bf16) support OP with hip (no use for rocblas.) There is no limit on RAM usage (GTT/VRAM) weight are allocate on RAM.

If you want to test:

```sh
# build:
rm -rf build/igpu
cmake -S . -B build/igpu -DGGML_IGPU=ON -DAMDGPU_TARGETS=gfx1103 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build/igpu --config Release -- -j 8

# run: (please use -ngl 999 --no-mmap -ctk bf16 -ctv bf16 for the best)
build/igpu/bin/llama-cli --color -ngl 999 --no-mmap -ctk bf16 -ctv bf16 -m Meta-Llama-3.1-8B-Instruct.BF16.gguf
```

to be fare there is some aleatory crache with 'MES' error, may need some correction on AMD firmware

01/03/2025: 1er version of kernel (V1)  (support only BF16 quantisation)
14/03/2025: create a new kernel (V2)    (support only BF16 quantisation)
01/04/2025: V4 optimise small N
15/04/2025: V5 kernel support BF16 & FP16 quant
25/05/2025: V7 optimised tensor loading (WIP)

Next:
  - create kernel for FP8 and support optional conversion of weight (FP16/BF16/FP32) to BFP on load.
  - create true block kernel for CPU ("blis" like)?

Some result (when it did not crash):

## Llama-3.2-1B-Instruct/BF16.gguf 
| model           |       size |   params | type_k | type_v |  test |    CPU |    V1   |     V2  |     V4  |      V7 |  Vulkan |
| --------------- | ---------: | -------: | -----: | -----: | ----: | -----: | ------: | ------: | ------: | ------: | ------: |
| llama 1B BF16   |   2.30 GiB |   1.24 B |   bf16 |   bf16 |   pp1 |  23.26 |   18.53 |   27.59 |   30.14 |   29.49 |   30.99 |
| llama 1B BF16   |   2.30 GiB |   1.24 B |   bf16 |   bf16 |   pp2 |  45.39 |   36.20 |   34.22 |   57.68 |   56.36 |   60.76 |
| llama 1B BF16   |   2.30 GiB |   1.24 B |   bf16 |   bf16 |   pp4 |  90.47 |   71.78 |   65.12 |  111.07 |  109.02 |  117.07 |
| llama 1B BF16   |   2.30 GiB |   1.24 B |   bf16 |   bf16 |   pp8 | 176.86 |  139.26 |  119.79 |  200.94 |  197.59 |  229.28 |
| llama 1B BF16   |   2.30 GiB |   1.24 B |   bf16 |   bf16 |  pp16 | 344.33 |  266.42 |  200.51 |  315.39 |  311.96 |  196.28 |
| llama 1B BF16   |   2.30 GiB |   1.24 B |   bf16 |   bf16 |  pp32 | 562.30 |  422.50 |  429.52 |  423.95 |  475.87 |  366.10 |
| llama 1B BF16   |   2.30 GiB |   1.24 B |   bf16 |   bf16 |  pp48 | 665.70 |  653.25 |  601.83 |  597.82 |  681.69 |  594.74 |
| llama 1B BF16   |   2.30 GiB |   1.24 B |   bf16 |   bf16 |  pp64 | 679.13 |  717.96 |  760.94 |  764.79 |  875.60 |  744.75 |
| llama 1B BF16   |   2.30 GiB |   1.24 B |   bf16 |   bf16 | pp128 | 723.15 |  990.37 | 1062.69 | 1061.43 | 1348.84 | 1007.61 |
| llama 1B BF16   |   2.30 GiB |   1.24 B |   bf16 |   bf16 | pp192 | 738.65 | 1131.50 | 1304.20 | 1298.02 | 1544.39 | 1054.13 |
| llama 1B BF16   |   2.30 GiB |   1.24 B |   bf16 |   bf16 | pp256 | 746.87 | 1151.29 | 1326.96 | 1329.72 | 1631.61 | 1153.88 |
| llama 1B BF16   |   2.30 GiB |   1.24 B |   bf16 |   bf16 | pp384 | 714.54 | 1178.65 | 1220.25 | 1197.43 | 1339.24 | 1238.02 |
| llama 1B BF16   |   2.30 GiB |   1.24 B |   bf16 |   bf16 | pp512 | 677.09 |  963.16 |  950.69 |  946.85 |  957.09 | 1207.43 |
| llama 1B BF16   |   2.30 GiB |   1.24 B |   bf16 |   bf16 | pp768 | 665.30 |  901.93 |  884.07 |  874.94 |  899.83 | 1162.78 |
| llama 1B BF16   |   2.30 GiB |   1.24 B |   bf16 |   bf16 |  tg16 |  23.00 |   18.26 |   27.69 |   30.13 |   30.18 |   31.17 |


## Llama-3.2-3B-Instruct/BF16.gguf
| model           |       size |   params | type_k | type_v |  test |    CPU |   V1   |    V2  |    V4  |     V7 | Vulkan |
| --------------- | ---------: | -------: | -----: | -----: | ----: | -----: | -----: | -----: | -----: | -----: | -----: |
| llama 3B BF16   |   5.98 GiB |   3.21 B |   bf16 |   bf16 |   pp1 |   8.94 |   7.85 |  11.03 |  11.84 |  11.81 |  12.07 |
| llama 3B BF16   |   5.98 GiB |   3.21 B |   bf16 |   bf16 |   pp2 |  17.56 |  15.67 |  14.61 |  23.08 |  23.10 |  23.67 |
| llama 3B BF16   |   5.98 GiB |   3.21 B |   bf16 |   bf16 |   pp4 |  35.02 |  31.11 |  27.86 |  44.61 |  44.66 |  44.96 |
| llama 3B BF16   |   5.98 GiB |   3.21 B |   bf16 |   bf16 |   pp8 |  69.18 |  61.01 |  51.21 |  82.57 |  81.93 |  90.41 |
| llama 3B BF16   |   5.98 GiB |   3.21 B |   bf16 |   bf16 |  pp16 | 131.72 | 117.77 |  86.80 | 135.50 | 134.94 |  78.39 |
| llama 3B BF16   |   5.98 GiB |   3.21 B |   bf16 |   bf16 |  pp32 | 209.28 | 185.05 | 178.08 | 176.60 | 196.36 | 142.46 |
| llama 3B BF16   |   5.98 GiB |   3.21 B |   bf16 |   bf16 |  pp48 | 232.70 | 273.60 | 249.61 | 251.45 | 282.22 | 196.73 |
| llama 3B BF16   |   5.98 GiB |   3.21 B |   bf16 |   bf16 |  pp64 | 237.90 | 300.62 | 313.17 | 316.92 | 356.20 | 246.77 |
| llama 3B BF16   |   5.98 GiB |   3.21 B |   bf16 |   bf16 | pp128 | 261.37 | 390.84 | 438.12 | 438.36 | 555.42 | 316.93 |
| llama 3B BF16   |   5.98 GiB |   3.21 B |   bf16 |   bf16 | pp192 | 263.82 | 445.00 | 506.12 | 504.17 | 576.38 | 368.65 |
| llama 3B BF16   |   5.98 GiB |   3.21 B |   bf16 |   bf16 | pp256 | 265.27 | 450.11 | 516.21 | 512.75 | 621.90 | 373.97 |
| llama 3B BF16   |   5.98 GiB |   3.21 B |   bf16 |   bf16 | pp384 | 261.27 | 470.54 | 485.27 | 476.42 | 581.70 | 400.52 |
| llama 3B BF16   |   5.98 GiB |   3.21 B |   bf16 |   bf16 | pp512 | 254.72 | 441.51 | 480.40 | 479.50 | 557.48 | 390.60 |
| llama 3B BF16   |   5.98 GiB |   3.21 B |   bf16 |   bf16 | pp768 | 253.87 | 429.79 | 462.86 | 462.20 | 542.42 | 384.43 |
| llama 3B BF16   |   5.98 GiB |   3.21 B |   bf16 |   bf16 |  tg16 |   8.90 |   7.85 |  11.02 |  11.88 |  11.88 |  12.30 |


## Meta-Llama-3.1-8B-Instruct/BF16.gguf
| model           |       size |   params | type_k | type_v |  test |    CPU |   V1   |    V2  |    V4  |     V7 | Vulkan |
| --------------- | ---------: | -------: | -----: | -----: | ----: | -----: | -----: | -----: | -----: | -----: | -----: |
| llama 8B BF16   |  14.96 GiB |   8.03 B |   bf16 |   bf16 |   pp1 |   3.88 |   3.88 |   4.88 |   5.21 |   5.20 |   5.35 |
| llama 8B BF16   |  14.96 GiB |   8.03 B |   bf16 |   bf16 |   pp2 |   7.59 |   7.74 |   7.40 |  10.12 |  10.13 |  10.60 |
| llama 8B BF16   |  14.96 GiB |   8.03 B |   bf16 |   bf16 |   pp4 |  15.04 |  15.43 |  14.20 |  19.67 |  19.65 |  20.59 |
| llama 8B BF16   |  14.96 GiB |   8.03 B |   bf16 |   bf16 |   pp8 |  29.73 |  30.23 |  26.37 |  36.74 |  36.59 |  40.71 |
| llama 8B BF16   |  14.96 GiB |   8.03 B |   bf16 |   bf16 |  pp16 |  56.55 |  58.55 |  45.95 |  61.51 |  60.66 |  41.17 |
| llama 8B BF16   |  14.96 GiB |   8.03 B |   bf16 |   bf16 |  pp32 |  84.81 |  91.54 |  83.38 |  81.09 |  94.22 |  75.68 |
| llama 8B BF16   |  14.96 GiB |   8.03 B |   bf16 |   bf16 |  pp48 |  90.43 | 114.77 | 116.55 | 114.14 | 131.60 | 106.00 |
| llama 8B BF16   |  14.96 GiB |   8.03 B |   bf16 |   bf16 |  pp64 |  85.45 | 137.17 | 139.46 | 142.46 | 165.54 | 132.83 |
| llama 8B BF16   |  14.96 GiB |   8.03 B |   bf16 |   bf16 | pp128 | 103.68 | 152.59 | 195.33 | 192.79 | 239.67 | 150.98 |
| llama 8B BF16   |  14.96 GiB |   8.03 B |   bf16 |   bf16 | pp192 | 107.07 | 183.30 | 215.62 | 217.06 | 260.26 | 159.43 |
| llama 8B BF16   |  14.96 GiB |   8.03 B |   bf16 |   bf16 | pp256 | 107.43 | 185.74 | 235.19 | 233.90 | 290.99 | 164.52 |
| llama 8B BF16   |  14.96 GiB |   8.03 B |   bf16 |   bf16 | pp384 | 106.74 | 213.56 | 230.65 | 229.00 | 290.74 | 168.15 |
| llama 8B BF16   |  14.96 GiB |   8.03 B |   bf16 |   bf16 | pp512 | 104.39 | 203.01 | 232.16 | 231.73 | 288.41 | 167.31 |
| llama 8B BF16   |  14.96 GiB |   8.03 B |   bf16 |   bf16 | pp768 | 104.19 | 194.98 | 225.46 | 225.09 | 281.33 | 165.74 |
| llama 8B BF16   |  14.96 GiB |   8.03 B |   bf16 |   bf16 |  tg16 |   3.88 |   3.88 |   4.87 |   5.21 |   5.20 |   5.36 |


## Mistral-Nemo-Instruct-2407/BF16.gguf
| model           |       size |   params | type_k | type_v |  test |    CPU |   V1   |    V2  |    V4  | Vulkan |
| --------------- | ---------: | -------: | -----: | -----: | ----: | -----: | -----: | -----: | -----: | -----: |
| llama 12B BF16  |  22.81 GiB |  12.25 B |   bf16 |   bf16 |   pp1 |   2.52 |   2.76 |   3.16 |   3.39 |   3.47 |
| llama 12B BF16  |  22.81 GiB |  12.25 B |   bf16 |   bf16 |   pp2 |   4.94 |   5.49 |   4.90 |   6.59 |   6.89 |
| llama 12B BF16  |  22.81 GiB |  12.25 B |   bf16 |   bf16 |   pp4 |   9.82 |  10.92 |   9.42 |  12.85 |  13.38 |
| llama 12B BF16  |  22.81 GiB |  12.25 B |   bf16 |   bf16 |   pp8 |  19.40 |  21.60 |  17.56 |  23.92 |  25.51 |
| llama 12B BF16  |  22.81 GiB |  12.25 B |   bf16 |   bf16 |  pp16 |  36.85 |  42.03 |  30.77 |  40.88 |  12.83 |
| llama 12B BF16  |  22.81 GiB |  12.25 B |   bf16 |   bf16 |  pp32 |  50.40 |  65.33 |  56.43 |  55.22 |  22.44 |
| llama 12B BF16  |  22.81 GiB |  12.25 B |   bf16 |   bf16 |  pp48 |  52.77 |  77.46 |  76.93 |  75.94 |  37.75 |
| llama 12B BF16  |  22.81 GiB |  12.25 B |   bf16 |   bf16 |  pp64 |  54.65 |  94.48 |  93.57 |  94.02 |  48.15 |
| llama 12B BF16  |  22.81 GiB |  12.25 B |   bf16 |   bf16 | pp128 |  65.72 | 103.87 | 127.90 | 128.54 |  51.19 |
| llama 12B BF16  |  22.81 GiB |  12.25 B |   bf16 |   bf16 | pp192 |  67.66 | 121.43 | 143.60 | 147.41 |  54.16 |
| llama 12B BF16  |  22.81 GiB |  12.25 B |   bf16 |   bf16 | pp256 |  68.45 | 130.03 | 156.00 | 155.52 |  54.07 |
| llama 12B BF16  |  22.81 GiB |  12.25 B |   bf16 |   bf16 | pp384 |  67.64 | 142.89 | 154.52 | 153.33 |  54.42 |
| llama 12B BF16  |  22.81 GiB |  12.25 B |   bf16 |   bf16 | pp512 |  67.02 | 136.18 | 156.22 | 156.51 |  46.71 |
| llama 12B BF16  |  22.81 GiB |  12.25 B |   bf16 |   bf16 | pp768 |  66.74 | 130.78 | 151.59 | 151.78 |  46.73 |
| llama 12B BF16  |  22.81 GiB |  12.25 B |   bf16 |   bf16 |  tg16 |   2.52 |   2.76 |   3.16 |   3.39 |   3.48 |


## Mistral-Small-24B-Instruct-2501/BF16.gguf
| model           |       size |   params | type_k | type_v |  test |    CPU |   V1   |    V2  |     V4  |
| --------------- | ---------: | -------: | -----: | -----: | ----: | -----: | -----: | -----: | ------: |
| llama 24B BF16  |  43.91 GiB |  23.57 B |   bf16 |   bf16 |   pp1 |   1.28 |   1.39 |   1.64 |   1.73  |
| llama 24B BF16  |  43.91 GiB |  23.57 B |   bf16 |   bf16 |   pp2 |   2.52 |   2.76 |   2.71 |   3.40  |
| llama 24B BF16  |  43.91 GiB |  23.57 B |   bf16 |   bf16 |   pp4 |   5.02 |   5.50 |   5.26 |   6.63  |
| llama 24B BF16  |  43.91 GiB |  23.57 B |   bf16 |   bf16 |   pp8 |   9.87 |  10.89 |   9.94 |  12.52  |
| llama 24B BF16  |  43.91 GiB |  23.57 B |   bf16 |   bf16 |  pp16 |  18.32 |  21.32 |  17.86 |  22.36  |
| llama 24B BF16  |  43.91 GiB |  23.57 B |   bf16 |   bf16 |  pp32 |  25.53 |  34.65 |  31.50 |  30.18  |
| llama 24B BF16  |  43.91 GiB |  23.57 B |   bf16 |   bf16 |  pp48 |  24.53 |  36.05 |  43.93 |  43.43  |
| llama 24B BF16  |  43.91 GiB |  23.57 B |   bf16 |   bf16 |  pp64 |  25.88 |  47.87 |  53.96 |  53.73  |
| llama 24B BF16  |  43.91 GiB |  23.57 B |   bf16 |   bf16 | pp128 |  29.69 |  52.03 |  69.64 |  65.84  |
| llama 24B BF16  |  43.91 GiB |  23.57 B |   bf16 |   bf16 | pp192 |  29.99 |  61.00 |  79.73 |  80.14  |
| llama 24B BF16  |  43.91 GiB |  23.57 B |   bf16 |   bf16 | pp256 |  30.94 |  63.11 |  87.30 |  87.01  |
| llama 24B BF16  |  43.91 GiB |  23.57 B |   bf16 |   bf16 | pp384 |  32.51 |  75.00 |  86.26 |    -    |
| llama 24B BF16  |  43.91 GiB |  23.57 B |   bf16 |   bf16 | pp512 |  32.28 |  71.11 |  88.11 |    -    |
| llama 24B BF16  |  43.91 GiB |  23.57 B |   bf16 |   bf16 | pp768 |  32.02 |  67.33 |  85.47 |    -    |
| llama 24B BF16  |  43.91 GiB |  23.57 B |   bf16 |   bf16 |  tg16 |   1.28 |   1.38 |   1.62 |    -    |

-------------------------------

# llama.cpp

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)

[Roadmap](https://github.com/users/ggerganov/projects/7) / [Project status](https://github.com/ggml-org/llama.cpp/discussions/3471) / [Manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml)

Inference of Meta's [LLaMA](https://arxiv.org/abs/2302.13971) model (and others) in pure C/C++

## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Hot topics

- 🔥 Multimodal support arrived in `llama-server`: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [documentation](./docs/multimodal.md)
- **GGML developer experience survey (organized and reviewed by NVIDIA):** [link](https://forms.gle/Gasw3cRgyhNEnrwK9)
- A new binary `llama-mtmd-cli` is introduced to replace `llava-cli`, `minicpmv-cli`, `gemma3-cli` ([#13012](https://github.com/ggml-org/llama.cpp/pull/13012)) and `qwen2vl-cli` ([#13141](https://github.com/ggml-org/llama.cpp/pull/13141)), `libllava` will be deprecated
- VS Code extension for FIM completions: https://github.com/ggml-org/llama.vscode
- Universal [tool call support](./docs/function-calling.md) in `llama-server` https://github.com/ggml-org/llama.cpp/pull/9639
- Vim/Neovim plugin for FIM completions: https://github.com/ggml-org/llama.vim
- Introducing GGUF-my-LoRA https://github.com/ggml-org/llama.cpp/discussions/10123
- Hugging Face Inference Endpoints now support GGUF out of the box! https://github.com/ggml-org/llama.cpp/discussions/9669
- Hugging Face GGUF editor: [discussion](https://github.com/ggml-org/llama.cpp/discussions/9268) | [tool](https://huggingface.co/spaces/CISCai/gguf-editor)

----

## Description

The main goal of `llama.cpp` is to enable LLM inference with minimal setup and state-of-the-art performance on a wide
range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads MTT GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is the main playground for developing new features for the [ggml](https://github.com/ggml-org/ggml) library.

<details>
<summary>Models</summary>

Typically finetunes of the base models below are supported as well.

Instructions for adding support for new models: [HOWTO-add-model.md](docs/development/HOWTO-add-model.md)

#### Text-only

- [X] LLaMA 🦙
- [x] LLaMA 2 🦙🦙
- [x] LLaMA 3 🦙🦙🦙
- [X] [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [x] [DBRX](https://huggingface.co/databricks/dbrx-instruct)
- [X] [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [X] [Chinese LLaMA / Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) and [Chinese LLaMA-2 / Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)
- [X] [Vigogne (French)](https://github.com/bofenghuang/vigogne)
- [X] [BERT](https://github.com/ggml-org/llama.cpp/pull/5423)
- [X] [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- [X] [Baichuan 1 & 2](https://huggingface.co/models?search=baichuan-inc/Baichuan) + [derivations](https://huggingface.co/hiyouga/baichuan-7b-sft)
- [X] [Aquila 1 & 2](https://huggingface.co/models?search=BAAI/Aquila)
- [X] [Starcoder models](https://github.com/ggml-org/llama.cpp/pull/3187)
- [X] [Refact](https://huggingface.co/smallcloudai/Refact-1_6B-fim)
- [X] [MPT](https://github.com/ggml-org/llama.cpp/pull/3417)
- [X] [Bloom](https://github.com/ggml-org/llama.cpp/pull/3553)
- [x] [Yi models](https://huggingface.co/models?search=01-ai/Yi)
- [X] [StableLM models](https://huggingface.co/stabilityai)
- [x] [Deepseek models](https://huggingface.co/models?search=deepseek-ai/deepseek)
- [x] [Qwen models](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [PLaMo-13B](https://github.com/ggml-org/llama.cpp/pull/3557)
- [x] [Phi models](https://huggingface.co/models?search=microsoft/phi)
- [x] [PhiMoE](https://github.com/ggml-org/llama.cpp/pull/11003)
- [x] [GPT-2](https://huggingface.co/gpt2)
- [x] [Orion 14B](https://github.com/ggml-org/llama.cpp/pull/5118)
- [x] [InternLM2](https://huggingface.co/models?search=internlm2)
- [x] [CodeShell](https://github.com/WisdomShell/codeshell)
- [x] [Gemma](https://ai.google.dev/gemma)
- [x] [Mamba](https://github.com/state-spaces/mamba)
- [x] [Grok-1](https://huggingface.co/keyfan/grok-1-hf)
- [x] [Xverse](https://huggingface.co/models?search=xverse)
- [x] [Command-R models](https://huggingface.co/models?search=CohereForAI/c4ai-command-r)
- [x] [SEA-LION](https://huggingface.co/models?search=sea-lion)
- [x] [GritLM-7B](https://huggingface.co/GritLM/GritLM-7B) + [GritLM-8x7B](https://huggingface.co/GritLM/GritLM-8x7B)
- [x] [OLMo](https://allenai.org/olmo)
- [x] [OLMo 2](https://allenai.org/olmo)
- [x] [OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
- [x] [Granite models](https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330)
- [x] [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) + [Pythia](https://github.com/EleutherAI/pythia)
- [x] [Snowflake-Arctic MoE](https://huggingface.co/collections/Snowflake/arctic-66290090abe542894a5ac520)
- [x] [Smaug](https://huggingface.co/models?search=Smaug)
- [x] [Poro 34B](https://huggingface.co/LumiOpen/Poro-34B)
- [x] [Bitnet b1.58 models](https://huggingface.co/1bitLLM)
- [x] [Flan T5](https://huggingface.co/models?search=flan-t5)
- [x] [Open Elm models](https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca)
- [x] [ChatGLM3-6b](https://huggingface.co/THUDM/chatglm3-6b) + [ChatGLM4-9b](https://huggingface.co/THUDM/glm-4-9b) + [GLMEdge-1.5b](https://huggingface.co/THUDM/glm-edge-1.5b-chat) + [GLMEdge-4b](https://huggingface.co/THUDM/glm-edge-4b-chat)
- [x] [GLM-4-0414](https://huggingface.co/collections/THUDM/glm-4-0414-67f3cbcb34dd9d252707cb2e)
- [x] [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)
- [x] [EXAONE-3.0-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)
- [x] [FalconMamba Models](https://huggingface.co/collections/tiiuae/falconmamba-7b-66b9a580324dd1598b0f6d4a)
- [x] [Jais](https://huggingface.co/inceptionai/jais-13b-chat)
- [x] [Bielik-11B-v2.3](https://huggingface.co/collections/speakleash/bielik-11b-v23-66ee813238d9b526a072408a)
- [x] [RWKV-6](https://github.com/BlinkDL/RWKV-LM)
- [x] [QRWKV-6](https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1)
- [x] [GigaChat-20B-A3B](https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct)
- [X] [Trillion-7B-preview](https://huggingface.co/trillionlabs/Trillion-7B-preview)
- [x] [Ling models](https://huggingface.co/collections/inclusionAI/ling-67c51c85b34a7ea0aba94c32)

#### Multimodal

- [x] [LLaVA 1.5 models](https://huggingface.co/collections/liuhaotian/llava-15-653aac15d994e992e2677a7e), [LLaVA 1.6 models](https://huggingface.co/collections/liuhaotian/llava-16-65b9e40155f60fd046a5ccf2)
- [x] [BakLLaVA](https://huggingface.co/models?search=SkunkworksAI/Bakllava)
- [x] [Obsidian](https://huggingface.co/NousResearch/Obsidian-3B-V0.5)
- [x] [ShareGPT4V](https://huggingface.co/models?search=Lin-Chen/ShareGPT4V)
- [x] [MobileVLM 1.7B/3B models](https://huggingface.co/models?search=mobileVLM)
- [x] [Yi-VL](https://huggingface.co/models?search=Yi-VL)
- [x] [Mini CPM](https://huggingface.co/models?search=MiniCPM)
- [x] [Moondream](https://huggingface.co/vikhyatk/moondream2)
- [x] [Bunny](https://github.com/BAAI-DCAI/Bunny)
- [x] [GLM-EDGE](https://huggingface.co/models?search=glm-edge)
- [x] [Qwen2-VL](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)

</details>

<details>
<summary>Bindings</summary>

- Python: [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Go: [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp)
- Node.js: [withcatai/node-llama-cpp](https://github.com/withcatai/node-llama-cpp)
- JS/TS (llama.cpp server client): [lgrammel/modelfusion](https://modelfusion.dev/integration/model-provider/llamacpp)
- JS/TS (Programmable Prompt Engine CLI): [offline-ai/cli](https://github.com/offline-ai/cli)
- JavaScript/Wasm (works in browser): [tangledgroup/llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm)
- Typescript/Wasm (nicer API, available on npm): [ngxson/wllama](https://github.com/ngxson/wllama)
- Ruby: [yoshoku/llama_cpp.rb](https://github.com/yoshoku/llama_cpp.rb)
- Rust (more features): [edgenai/llama_cpp-rs](https://github.com/edgenai/llama_cpp-rs)
- Rust (nicer API): [mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp)
- Rust (more direct bindings): [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)
- Rust (automated build from crates.io): [ShelbyJenkins/llm_client](https://github.com/ShelbyJenkins/llm_client)
- C#/.NET: [SciSharp/LLamaSharp](https://github.com/SciSharp/LLamaSharp)
- C#/VB.NET (more features - community license): [LM-Kit.NET](https://docs.lm-kit.com/lm-kit-net/index.html)
- Scala 3: [donderom/llm4s](https://github.com/donderom/llm4s)
- Clojure: [phronmophobic/llama.clj](https://github.com/phronmophobic/llama.clj)
- React Native: [mybigday/llama.rn](https://github.com/mybigday/llama.rn)
- Java: [kherud/java-llama.cpp](https://github.com/kherud/java-llama.cpp)
- Zig: [deins/llama.cpp.zig](https://github.com/Deins/llama.cpp.zig)
- Flutter/Dart: [netdur/llama_cpp_dart](https://github.com/netdur/llama_cpp_dart)
- Flutter: [xuegao-tzx/Fllama](https://github.com/xuegao-tzx/Fllama)
- PHP (API bindings and features built on top of llama.cpp): [distantmagic/resonance](https://github.com/distantmagic/resonance) [(more info)](https://github.com/ggml-org/llama.cpp/pull/6326)
- Guile Scheme: [guile_llama_cpp](https://savannah.nongnu.org/projects/guile-llama-cpp)
- Swift [srgtuszy/llama-cpp-swift](https://github.com/srgtuszy/llama-cpp-swift)
- Swift [ShenghaiWang/SwiftLlama](https://github.com/ShenghaiWang/SwiftLlama)
- Delphi [Embarcadero/llama-cpp-delphi](https://github.com/Embarcadero/llama-cpp-delphi)

</details>

<details>
<summary>UIs</summary>

*(to have a project listed here, it should clearly state that it depends on `llama.cpp`)*

- [AI Sublime Text plugin](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) (MIT)
- [cztomsik/ava](https://github.com/cztomsik/ava) (MIT)
- [Dot](https://github.com/alexpinel/Dot) (GPL)
- [eva](https://github.com/ylsdamxssjxxdd/eva) (MIT)
- [iohub/collama](https://github.com/iohub/coLLaMA) (Apache-2.0)
- [janhq/jan](https://github.com/janhq/jan) (AGPL)
- [johnbean393/Sidekick](https://github.com/johnbean393/Sidekick) (MIT)
- [KanTV](https://github.com/zhouwg/kantv?tab=readme-ov-file) (Apache-2.0)
- [KodiBot](https://github.com/firatkiral/kodibot) (GPL)
- [llama.vim](https://github.com/ggml-org/llama.vim) (MIT)
- [LARS](https://github.com/abgulati/LARS) (AGPL)
- [Llama Assistant](https://github.com/vietanhdev/llama-assistant) (GPL)
- [LLMFarm](https://github.com/guinmoon/LLMFarm?tab=readme-ov-file) (MIT)
- [LLMUnity](https://github.com/undreamai/LLMUnity) (MIT)
- [LMStudio](https://lmstudio.ai/) (proprietary)
- [LocalAI](https://github.com/mudler/LocalAI) (MIT)
- [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp) (AGPL)
- [MindMac](https://mindmac.app) (proprietary)
- [MindWorkAI/AI-Studio](https://github.com/MindWorkAI/AI-Studio) (FSL-1.1-MIT)
- [Mobile-Artificial-Intelligence/maid](https://github.com/Mobile-Artificial-Intelligence/maid) (MIT)
- [Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile) (Apache-2.0)
- [nat/openplayground](https://github.com/nat/openplayground) (MIT)
- [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all) (MIT)
- [ollama/ollama](https://github.com/ollama/ollama) (MIT)
- [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) (AGPL)
- [PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai) (MIT)
- [psugihara/FreeChat](https://github.com/psugihara/FreeChat) (MIT)
- [ptsochantaris/emeltal](https://github.com/ptsochantaris/emeltal) (MIT)
- [pythops/tenere](https://github.com/pythops/tenere) (AGPL)
- [ramalama](https://github.com/containers/ramalama) (MIT)
- [semperai/amica](https://github.com/semperai/amica) (MIT)
- [withcatai/catai](https://github.com/withcatai/catai) (MIT)
- [Autopen](https://github.com/blackhole89/autopen) (GPL)

</details>

<details>
<summary>Tools</summary>

- [akx/ggify](https://github.com/akx/ggify) – download PyTorch models from HuggingFace Hub and convert them to GGML
- [akx/ollama-dl](https://github.com/akx/ollama-dl) – download models from the Ollama library to be used directly with llama.cpp
- [crashr/gppm](https://github.com/crashr/gppm) – launch llama.cpp instances utilizing NVIDIA Tesla P40 or P100 GPUs with reduced idle power consumption
- [gpustack/gguf-parser](https://github.com/gpustack/gguf-parser-go/tree/main/cmd/gguf-parser) - review/check the GGUF file and estimate the memory usage
- [Styled Lines](https://marketplace.unity.com/packages/tools/generative-ai/styled-lines-llama-cpp-model-292902) (proprietary licensed, async wrapper of inference part for game development in Unity3d with pre-built Mobile and Web platform wrappers and a model example)

</details>

<details>
<summary>Infrastructure</summary>

- [Paddler](https://github.com/distantmagic/paddler) - Stateful load balancer custom-tailored for llama.cpp
- [GPUStack](https://github.com/gpustack/gpustack) - Manage GPU clusters for running LLMs
- [llama_cpp_canister](https://github.com/onicai/llama_cpp_canister) - llama.cpp as a smart contract on the Internet Computer, using WebAssembly
- [llama-swap](https://github.com/mostlygeek/llama-swap) - transparent proxy that adds automatic model switching with llama-server
- [Kalavai](https://github.com/kalavai-net/kalavai-client) - Crowdsource end to end LLM deployment at any scale
- [llmaz](https://github.com/InftyAI/llmaz) - ☸️ Easy, advanced inference platform for large language models on Kubernetes.
</details>

<details>
<summary>Games</summary>

- [Lucy's Labyrinth](https://github.com/MorganRO8/Lucys_Labyrinth) - A simple maze game where agents controlled by an AI model will try to trick you.

</details>

## Supported backends

| Backend | Target devices |
| --- | --- |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [SYCL](docs/backend/SYCL.md) | Intel and Nvidia GPU |
| [MUSA](docs/build.md#musa) | Moore Threads MTT GPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |

## Building the project

The main product of this project is the `llama` library. Its C-style interface can be found in [include/llama.h](include/llama.h).
The project also includes many example programs and tools using the `llama` library. The examples range from simple, minimal code snippets to sophisticated sub-projects such as an OpenAI-compatible HTTP server. Possible methods for obtaining the binaries:

- Clone this repository and build locally, see [how to build](docs/build.md)
- On MacOS or Linux, install `llama.cpp` via [brew, flox or nix](docs/install.md)
- Use a Docker image, see [documentation for Docker](docs/docker.md)
- Download pre-built binaries from [releases](https://github.com/ggml-org/llama.cpp/releases)

## Obtaining and quantizing models

The [Hugging Face](https://huggingface.co) platform hosts a [number of LLMs](https://huggingface.co/models?library=gguf&sort=trending) compatible with `llama.cpp`:

- [Trending](https://huggingface.co/models?library=gguf&sort=trending)
- [LLaMA](https://huggingface.co/models?sort=trending&search=llama+gguf)

You can either manually download the GGUF file or directly use any `llama.cpp`-compatible models from [Hugging Face](https://huggingface.co/) or other model hosting sites, such as [ModelScope](https://modelscope.cn/), by using this CLI argument: `-hf <user>/<model>[:quant]`.

By default, the CLI would download from Hugging Face, you can switch to other options with the environment variable `MODEL_ENDPOINT`. For example, you may opt to downloading model checkpoints from ModelScope or other model sharing communities by setting the environment variable, e.g. `MODEL_ENDPOINT=https://www.modelscope.cn/`.

After downloading a model, use the CLI tools to run it locally - see below.

`llama.cpp` requires the model to be stored in the [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) file format. Models in other data formats can be converted to GGUF using the `convert_*.py` Python scripts in this repo.

The Hugging Face platform provides a variety of online tools for converting, quantizing and hosting models with `llama.cpp`:

- Use the [GGUF-my-repo space](https://huggingface.co/spaces/ggml-org/gguf-my-repo) to convert to GGUF format and quantize model weights to smaller sizes
- Use the [GGUF-my-LoRA space](https://huggingface.co/spaces/ggml-org/gguf-my-lora) to convert LoRA adapters to GGUF format (more info: https://github.com/ggml-org/llama.cpp/discussions/10123)
- Use the [GGUF-editor space](https://huggingface.co/spaces/CISCai/gguf-editor) to edit GGUF meta data in the browser (more info: https://github.com/ggml-org/llama.cpp/discussions/9268)
- Use the [Inference Endpoints](https://ui.endpoints.huggingface.co/) to directly host `llama.cpp` in the cloud (more info: https://github.com/ggml-org/llama.cpp/discussions/9669)

To learn more about model quantization, [read this documentation](tools/quantize/README.md)

## [`llama-cli`](tools/main)

#### A CLI tool for accessing and experimenting with most of `llama.cpp`'s functionality.

- <details open>
    <summary>Run in conversation mode</summary>

    Models with a built-in chat template will automatically activate conversation mode. If this doesn't occur, you can manually enable it by adding `-cnv` and specifying a suitable chat template with `--chat-template NAME`

    ```bash
    llama-cli -m model.gguf

    # > hi, who are you?
    # Hi there! I'm your helpful assistant! I'm an AI-powered chatbot designed to assist and provide information to users like you. I'm here to help answer your questions, provide guidance, and offer support on a wide range of topics. I'm a friendly and knowledgeable AI, and I'm always happy to help with anything you need. What's on your mind, and how can I assist you today?
    #
    # > what is 1+1?
    # Easy peasy! The answer to 1+1 is... 2!
    ```

    </details>

- <details>
    <summary>Run in conversation mode with custom chat template</summary>

    ```bash
    # use the "chatml" template (use -h to see the list of supported templates)
    llama-cli -m model.gguf -cnv --chat-template chatml

    # use a custom template
    llama-cli -m model.gguf -cnv --in-prefix 'User: ' --reverse-prompt 'User:'
    ```

    </details>

- <details>
    <summary>Run simple text completion</summary>

    To disable conversation mode explicitly, use `-no-cnv`

    ```bash
    llama-cli -m model.gguf -p "I believe the meaning of life is" -n 128 -no-cnv

    # I believe the meaning of life is to find your own truth and to live in accordance with it. For me, this means being true to myself and following my passions, even if they don't align with societal expectations. I think that's what I love about yoga – it's not just a physical practice, but a spiritual one too. It's about connecting with yourself, listening to your inner voice, and honoring your own unique journey.
    ```

    </details>

- <details>
    <summary>Constrain the output with a custom grammar</summary>

    ```bash
    llama-cli -m model.gguf -n 256 --grammar-file grammars/json.gbnf -p 'Request: schedule a call at 8pm; Command:'

    # {"appointmentTime": "8pm", "appointmentDetails": "schedule a a call"}
    ```

    The [grammars/](grammars/) folder contains a handful of sample grammars. To write your own, check out the [GBNF Guide](grammars/README.md).

    For authoring more complex JSON grammars, check out https://grammar.intrinsiclabs.ai/

    </details>


## [`llama-server`](tools/server)

#### A lightweight, [OpenAI API](https://github.com/openai/openai-openapi) compatible, HTTP server for serving LLMs.

- <details open>
    <summary>Start a local HTTP server with default configuration on port 8080</summary>

    ```bash
    llama-server -m model.gguf --port 8080

    # Basic web UI can be accessed via browser: http://localhost:8080
    # Chat completion endpoint: http://localhost:8080/v1/chat/completions
    ```

    </details>

- <details>
    <summary>Support multiple-users and parallel decoding</summary>

    ```bash
    # up to 4 concurrent requests, each with 4096 max context
    llama-server -m model.gguf -c 16384 -np 4
    ```

    </details>

- <details>
    <summary>Enable speculative decoding</summary>

    ```bash
    # the draft.gguf model should be a small variant of the target model.gguf
    llama-server -m model.gguf -md draft.gguf
    ```

    </details>

- <details>
    <summary>Serve an embedding model</summary>

    ```bash
    # use the /embedding endpoint
    llama-server -m model.gguf --embedding --pooling cls -ub 8192
    ```

    </details>

- <details>
    <summary>Serve a reranking model</summary>

    ```bash
    # use the /reranking endpoint
    llama-server -m model.gguf --reranking
    ```

    </details>

- <details>
    <summary>Constrain all outputs with a grammar</summary>

    ```bash
    # custom grammar
    llama-server -m model.gguf --grammar-file grammar.gbnf

    # JSON
    llama-server -m model.gguf --grammar-file grammars/json.gbnf
    ```

    </details>


## [`llama-perplexity`](tools/perplexity)

#### A tool for measuring the perplexity [^1][^2] (and other quality metrics) of a model over a given text.

- <details open>
    <summary>Measure the perplexity over a text file</summary>

    ```bash
    llama-perplexity -m model.gguf -f file.txt

    # [1]15.2701,[2]5.4007,[3]5.3073,[4]6.2965,[5]5.8940,[6]5.6096,[7]5.7942,[8]4.9297, ...
    # Final estimate: PPL = 5.4007 +/- 0.67339
    ```

    </details>

- <details>
    <summary>Measure KL divergence</summary>

    ```bash
    # TODO
    ```

    </details>

[^1]: [tools/perplexity/README.md](./tools/perplexity/README.md)
[^2]: [https://huggingface.co/docs/transformers/perplexity](https://huggingface.co/docs/transformers/perplexity)

## [`llama-bench`](tools/llama-bench)

#### Benchmark the performance of the inference for various parameters.

- <details open>
    <summary>Run default benchmark</summary>

    ```bash
    llama-bench -m model.gguf

    # Output:
    # | model               |       size |     params | backend    | threads |          test |                  t/s |
    # | ------------------- | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         pp512 |      5765.41 ± 20.55 |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         tg128 |        197.71 ± 0.81 |
    #
    # build: 3e0ba0e60 (4229)
    ```

    </details>

## [`llama-run`](tools/run)

#### A comprehensive example for running `llama.cpp` models. Useful for inferencing. Used with RamaLama [^3].

- <details>
    <summary>Run a model with a specific prompt (by default it's pulled from Ollama registry)</summary>

    ```bash
    llama-run granite-code
    ```

    </details>

[^3]: [RamaLama](https://github.com/containers/ramalama)

## [`llama-simple`](examples/simple)

#### A minimal example for implementing apps with `llama.cpp`. Useful for developers.

- <details>
    <summary>Basic text completion</summary>

    ```bash
    llama-simple -m model.gguf

    # Hello my name is Kaitlyn and I am a 16 year old girl. I am a junior in high school and I am currently taking a class called "The Art of
    ```

    </details>


## Contributing

- Contributors can open PRs
- Collaborators can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Collaborators will be invited based on contributions
- Any help with managing issues, PRs and projects is very appreciated!
- See [good first issues](https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for tasks suitable for first contributions
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information
- Make sure to read this: [Inference at the edge](https://github.com/ggml-org/llama.cpp/discussions/205)
- A bit of backstory for those who are interested: [Changelog podcast](https://changelog.com/podcast/532)

## Other documentation

- [main (cli)](tools/main/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development documentation

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

#### Seminal papers and background on the models

If your issue is with model generation quality, then please at least scan the following links and papers to understand the limitations of LLaMA models. This is especially important when choosing an appropriate model size and appreciating both the significant and subtle differences between LLaMA models and ChatGPT:
- LLaMA:
    - [Introducing LLaMA: A foundational, 65-billion-parameter large language model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
    - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- GPT-3
    - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- GPT-3.5 / InstructGPT / ChatGPT:
    - [Aligning language models to follow instructions](https://openai.com/research/instruction-following)
    - [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

## XCFramework
The XCFramework is a precompiled version of the library for iOS, visionOS, tvOS,
and macOS. It can be used in Swift projects without the need to compile the
library from source. For example:
```swift
// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MyLlamaPackage",
    targets: [
        .executableTarget(
            name: "MyLlamaPackage",
            dependencies: [
                "LlamaFramework"
            ]),
        .binaryTarget(
            name: "LlamaFramework",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b5046/llama-b5046-xcframework.zip",
            checksum: "c19be78b5f00d8d29a25da41042cb7afa094cbf6280a225abe614b03b20029ab"
        )
    ]
)
```
The above example is using an intermediate build `b5046` of the library. This can be modified
to use a different version by changing the URL and checksum.

## Completions
Command-line completion is available for some environments.

#### Bash Completion
```bash
$ build/bin/llama-cli --completion-bash > ~/.llama-completion.bash
$ source ~/.llama-completion.bash
```
Optionally this can be added to your `.bashrc` or `.bash_profile` to load it
automatically. For example:
```console
$ echo "source ~/.llama-completion.bash" >> ~/.bashrc
```

## Dependencies

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [minja](https://github.com/google/minja) - Minimal Jinja parser in C++, used by various tools/examples - MIT License
- [linenoise.cpp](./tools/run/linenoise.cpp/linenoise.cpp) - C++ library that provides readline-like line editing capabilities, used by `llama-run` - BSD 2-Clause License
- [curl](https://curl.se/) - Client-side URL transfer library, used by various tools/examples - [CURL License](https://curl.se/docs/copyright.html)
