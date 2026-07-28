# advanced-gguf-quantizer

`advanced-gguf-quantizer` is a llama.cpp-derived, CUDA accelerated GGUF quantization toolkit for
creating, inspecting, evaluating, improving, and testing GGUF models. The first
focus initially was Blackwell NVFP4 and MXFP6, but now this quantizer has
expanded to OCP MXFP8 as well. A BF16 GGUF, imatrix, dataset, and KLD file are
used together
to search and tune for the best possible combination of scales and mixed tensor types.

This tool is still in the absolute earliest stages of development and is still a rapidly changing work in progress.  More features and improvements are still on the way.

A new strategy called Refined Scale Fit (RSF) is  used on Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, and NVFP4 for changing the scales and re-fitting using imatrix supported data.

Logit KLD files are used during the process to tune and adjust through a series of candidate policies and different types of quantizations to determine which has the lowest error for that particular tensor and how the imatrix affects the ppl/kld data.  
The tool will also self determine using a weighted score how to promote more error prone or sensitive tensors to higher bit GGUF types such as Q4_K, Q5_K, Q6_K, MXFP6 (not by default), MXFP8, or Q8_0 when the measured error profile justifies both bpw/size and speed tradeoff.

Models created with this tool are posted at:
https://huggingface.co/michaelw9999

This repository keeps a limited set of the original llama.cpp tools:

- `llama-quantize`: the primary command and preferred entry
  point
- `advanced-gguf-quantizer`:  binary for new subcommands.
- `llama-imatrix`: calibration imatrix generation.
- `llama-perplexity`: PPL and saved-logit KLD reporter
- `llama-completion`: quickly tests produced GGUFs.


This tool requires the input GGUF files to be present locally.
It does not download models, import Hugging Face checkpoints, or convert checkpoints.

## Alpha Notice

This tool and MXFP6/E2M3 is experimental and is not supported by NVIDIA or llama.cpp. If
official MXFP6 support appears later, the official format may change and GGUF
models created here may not work with future runtimes.

MXFP6 is not enabled by default, but the tool can make MXFP6 quantizations and also use MXFP6 mixed with other tensor types in the model evaluation strategy.

MXFP8 is the standard OCP MX E4M3 format: 32 E4M3 values share one E8M0
scale. It is available as a direct 8.25 bpw GGUF profile with CPU and CUDA
runtime support. Unlike the repository's MXFP6_E2M3 extension, MXFP8 is not a
private experimental encoding.

MXFP6_E2M3 uses maintainer-owned GGUF tensor and file type ID 50. Older MXFP6
artifacts that used tensor type ID 42 must be migrated or regenerated because
upstream llama.cpp now assigns tensor type ID 42 to Q2_0.

Feedback is requested. The latest full CUDA MXFP6 branch can be installed from
<https://github.com/michaelw9999/llama.cpp/tree/mxfp6-cuda>.
The initial llama.cpp MXFP6 CPU only PR is located at
<https://github.com/ggml-org/llama.cpp/pull/22671>.

## What This Does

- Designed to create NVFP4, MXFP8, MXFP6, and mixed NVFP4/MXFP6 GGUFs with proper tensor and input scales.
- Uses RSF (Refined Scale Fit) for supported K-quants and NVFP4 scale fitting.
- Selects quantization types tensor-by-tensor based on set parameters/recipe or mode.
- Offers `fast`, `normal`, and `deep` work modes for quantizer search depth.
- Allows existing GGUFs to easily be patched to further improve them or swap tensor types.
- Uses CUDA accelerated improvements for tensor encoding, `llama-perplexity`,
  `llama-imatrix`, and in-memory runtime patch edits.
- Allows start and stop and resume on quantizing, creating imatrix, or saving logits.
- Creates activation-aware quantization for NVFP4 using imatrix input scales.
- Offers recipe and project files for reproducible runs.
- Creates run artifacts: locked recipe, run log, tensor assignment log, manifest,
  `checkpoint-key.json`, quantization report, and validation scripts.
- Checkpointed candidate search so interrupted long runs can resume safely.
- In-place repair and edit feature for existing GGUFs.
- p99 and p999 KLD gates in quality mode, visible in reports.
- Fused decision units for grouped tensor choices.
- Best-candidate reporting determined by defined quality and budget choices.
- A text UI and wizard for guided recipe creation, monitoring, recovery, repair,
  and inspection. [Still needs improvement]
- Agent-guided quantization in [SKILLS.md](SKILLS.md) for automated runs.

- This README is still a work in progress.  The tool has more features and improvements than are listed here.

## Build

Use a fresh build directory:

```bash
cmake -S . -B build -DGGML_CUDA=ON
cmake --build build -j 20
```

Expected binaries are in `build/bin/`.

## Quick Start

Inspect the source GGUF:

```bash
./build/bin/llama-quantize inspect models/source-bf16.gguf
```

Create a starting recipe:

```bash
./build/bin/advanced-gguf-quantizer recipe init --profile nvfp4 --output recipes/model.toml
```

Edit the recipe fields for your files:

```toml
[io]
input = "models/source-bf16.gguf"
output = "runs/model/model-nvfp4-mxfp6.gguf"

[calibration]
corpus = "data/calibration.txt"
imatrix = "runs/model/imatrix.dat"

[evaluation]
bf16_reference = "models/source-bf16.gguf"
corpus = "data/eval.txt"
kld_base = "runs/model/bf16.kld"

[target]
precision_mode = "nvfp4"
target_bpw = 4.8

[quantizer]
mode = "normal" # fast | normal | deep
```

Generate an imatrix:

```bash
./build/bin/llama-imatrix \
  -m models/source-bf16.gguf \
  -f data/calibration.txt \
  -o runs/model/imatrix.dat
```

Generate a saved-logit KLD base:

```bash
./build/bin/advanced-gguf-quantizer kld-command recipes/model.toml
```

Inspect the planned command without writing a model:

```bash
./build/bin/advanced-gguf-quantizer recipe validate recipes/model.toml
./build/bin/advanced-gguf-quantizer plan recipes/model.toml
```

Run real quantization:

```bash
./build/bin/advanced-gguf-quantizer run recipes/model.toml --project runs/model --yes
```

Inspect and test the result:

```bash
./build/bin/llama-quantize inspect runs/model/model-nvfp4.gguf
./build/bin/llama-completion -m runs/model/model-nvfp4.gguf -p "The model is"
```

## Example Paths

Compact Qwen-style dense or MTP model:

```bash
./build/bin/advanced-gguf-quantizer recipe init --profile nvfp4_mxfp6 --output recipes/qwen.toml
./build/bin/advanced-gguf-quantizer run recipes/qwen.toml --project runs/qwen --yes
```

Quality-first MoE model:

```bash
./build/bin/advanced-gguf-quantizer recipe init --profile mxfp6-primary --output recipes/moe.toml
./build/bin/advanced-gguf-quantizer run recipes/moe.toml --project runs/moe --yes
```

Direct OCP MXFP8 model:

```bash
./build/bin/advanced-gguf-quantizer recipe init --profile mxfp8 --output recipes/mxfp8.toml
./build/bin/advanced-gguf-quantizer run recipes/mxfp8.toml --project runs/mxfp8 --yes
```

Repair or edit an existing GGUF:

```bash
./build/bin/advanced-gguf-quantizer recipe init --profile repair --output recipes/repair.toml
./build/bin/advanced-gguf-quantizer run recipes/repair.toml --project runs/repair --yes
```

Use `plan` to inspect a recipe. For model production, launch `run`; it writes
real artifacts and does not waste time on fake quantization passes.

## Choosing NVFP4, MXFP8, MXFP6, Or Mixed

`NVFP4` when smaller size and Blackwell speed matter most. It is the smallest
advanced mode but has more quantization error, so quality runs should rely on
imatrix, saved-logit KLD evidence, p99/p999 tails, and tensor-by-tensor
promotions for more error prone or sensitive layers.

`MXFP8` when a standardized microscaling E4M3 format and higher precision
matter more than compact size. It uses 8.25 bpw, has CPU execution, and uses
native scaled FP8 matrix paths on Blackwell CUDA where the operation shape
allows it.

`MXFP6_E2M3` when quality matters more than file size. It is larger than
NVFP4 but useful for sensitive tensors, MoE experts, output/head tensors,
or models where NVFP4-only error is high. MXFP6 is still experimental and should
be selected deliberately.

`NVFP4_MXFP6` will make the fastest, best balanced model. The allocator can either:

- use MXFP6 to improve an NVFP4-first model while keeping the model compact;
- use MXFP6 as the primary target and demote selected tensors to NVFP4 to increase speed
and reduce model size, when measured quality loss is acceptable.

Fallback types such as `MXFP6_E2M3`, `Q4_K`, `Q6_K`, `Q8_0`, `BF16`, and
`F16` remain available for tensor policy, size and speed-aware candidate assignment,
output/head handling, exclusions, or repair.

For KLD selector runs, choose a quantizer mode:

- `fast`: smallest search for relatively quick iteration.
- `normal`: default time/search balance for typical use
- `deep`: longest, deepest search for finding incremental last remaining quality

## Documentation

- [Documentation index](docs/README.md)
- [Notices and attribution](NOTICE.md)
- [Flag and workflow guide](docs/advanced-gguf-quantizer-flags.md)
- [Layer policy guide](docs/advanced-gguf-quantizer-layer-policy.md)
- [CUDA acceleration guide](docs/advanced-gguf-quantizer-cuda-acceleration.md)
- [Imatrix and KLD guide](docs/advanced-gguf-quantizer-imatrix-kld.md)
- [NVFP4 GGUF contract](docs/advanced-gguf-quantizer-nvfp4-contract.md)
- [MXFP8 GGUF contract](docs/advanced-gguf-quantizer-mxfp8-contract.md)
- [AI agent guide](SKILLS.md)
