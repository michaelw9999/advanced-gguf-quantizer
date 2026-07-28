# Layer Policy Guide

Layer policy decides which tensor type each GGUF tensor receives. The default is
conservative: protect sensitive tensors, preserve model structure, and spend
extra bits only where evidence justifies them.

## Core Rules

- Preserve MTP and NextN tensors by default, including the whole appended
  decoder block on Qwen-style MTP models.
- Use `base.mtp_tensor_type` / `--mtp-tensor-type` when a recipe deliberately
  converts MTP matrix tensors. `NVFP4`, `MXFP8`, `MXFP6_E2M3`, `Q8_0`,
  `BF16`, and `F16` are valid release choices.
- Keep token embeddings as `NVFP4` in pure NVFP4 artifacts.
- A separate `output.weight` may use `Q6_K`, `Q8_0`, `BF16`, or another stronger
  type when the recipe asks for it.
- Do not silently turn a tied token embedding into an output fallback type.
- Use `MXFP8` for all matrix tensors in the direct MXFP8 profile unless the
  recipe explicitly overrides embedding, output, or MTP policy.
- Use `MXFP6_E2M3` only when a recipe or candidate assignment selects it.
- Keep output/head choices explicit in the recipe and report.
- Treat fallback types such as `MXFP6_E2M3`, `Q4_K`, `Q6_K`, `Q8_0`, `BF16`,
  and `F16` as budgeted policy choices, not hidden defaults.

MXFP6_E2M3 is experimental and unsupported by NVIDIA and llama.cpp. Future
official support may use a different format, so mixed or MXFP6-primary GGUFs
created here may not remain compatible with future runtimes.

GGUF type ID 42 is reserved for upstream `Q2_0`. This repository writes
maintainer-owned `MXFP6_E2M3` tensor and file types as ID 50. Older MXFP6 files
that used tensor ID 42 or the briefly used local ID 44 must be migrated or
regenerated so they cannot be misread as Q2_0.

## Fused Decision Units

The allocator should make coherent decisions for related tensors before using
per-tensor exceptions:

- `qkv`: attention q, k, and v projections in the same layer, including fused
  QKV tensors.
- `gate_up`: dense MLP gate and up projections.
- `expert_pairs`: MoE expert tensors grouped by layer and expert.
- `mtp_heads`: MTP, NextN, and draft-head tensors.
- `lm_head_and_embeddings`: token embeddings, tied embeddings, separate output,
  and lm head tensors.

The final `assignment.jsonl` still records per-tensor types. The report should
explain the group decision that produced those assignments.

## Sensitive Tensor Classes

These tensor classes often need stronger handling:

- early `ffn_down` tensors;
- attention output projections;
- q/k/v siblings when one projection dominates quality loss;
- MoE expert down projections;
- output/lm head tensors;
- MTP/NextN heads;
- small control or state-space tensors where quantizing saves little space.

For mixed NVFP4/MXFP6 runs, sensitive tensors are good MXFP6 or fallback
candidates. For compact NVFP4-first runs, let the p99/p999 KLD gates decide
whether the extra bytes are worth it.

## Embeddings And Output

Recommended defaults:

- pure NVFP4: token embeddings `NVFP4`, separate output stronger than NVFP4 when
  needed, MTP matrix weights `NVFP4` when MTP is present;
- mixed NVFP4/MXFP6: keep embeddings and output controlled by explicit recipe
  fields; MTP matrix weights default to `NVFP4` unless the recipe requests
  `MXFP6_E2M3` or another fallback type;
- MXFP6-primary: use MXFP6 for sensitive output/head tensors unless a budget
  rule proves a lower type is acceptable; MTP matrix weights default to
  `MXFP6_E2M3`;
- tied embeddings: follow the token embedding rule unless the architecture has a
  separate output tensor.

Inspect finished artifacts:

```bash
./build/bin/llama-quantize inspect runs/model/output.gguf --tensors
```

## Reporting

A layer policy report should include:

- tensor type counts and bytes by type;
- scale and input-scale tensor counts;
- MTP/NextN preservation status;
- token embedding and output policy;
- fused decision unit summaries;
- exceptions and fallback tensors;
- BPW and output file size;
- PPL/KLD/tail metrics that justified sensitive assignments.

The strongest explanation is not that a tensor is generally believed to be
sensitive. The strongest explanation is measured evidence under the same
source, imatrix, KLD base, recipe, and commit.
