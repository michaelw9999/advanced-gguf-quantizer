# advanced-gguf-quantizer Flag And Workflow Guide

This guide explains the public command surface, recipe fields, and quality
controls for producing NVFP4, OCP MXFP8, MXFP6_E2M3, and mixed NVFP4/MXFP6
GGUF models.
Prefer recipes and project files over one-off shell state so runs are
reproducible and resumable.

## Commands

- `advanced-gguf-quantizer recipe init --profile NAME --output recipe.toml`
  creates a starting recipe.
- `advanced-gguf-quantizer recipe validate recipe.toml`
  checks required fields and incompatible settings.
- `advanced-gguf-quantizer plan recipe.toml`
  prints the planned run without writing a GGUF.
- `advanced-gguf-quantizer run recipe.toml --project DIR --yes`
  writes the quantized GGUF and run artifacts.
- `llama-quantize inspect model.gguf`
  reports tensor types, scale tensors, metadata, MTP/NextN status, and output
  policy.
- `advanced-gguf-quantizer candidates recipe.toml`
  expands a recipe into candidate recipes or candidate records.
- `advanced-gguf-quantizer best metrics.jsonl --real-ppl-kld --quality-only`
  reports non-dominated candidates from real PPL/KLD metrics.
- `advanced-gguf-quantizer what-if sensitivity.jsonl --tensor NAME`
  inspects exact sensitivity rows from runs that produced them.
- `advanced-gguf-quantizer kld-command recipe.toml`
  prints the `llama-perplexity` command for a saved-logit KLD base.
- `advanced-gguf-quantizer imatrix-command recipe.toml`
  prints the `llama-imatrix` command for calibration.

`advanced-gguf-quantizer` owns the recipe/project/report helper surface.
`llama-quantize` is the retained model-writing engine used by `run` and remains
available for direct quantization and GGUF inspection.

## No Fake Production Runs

For model-production requests, use `run`. Use `plan`, `recipe validate`, and
`inspect` only for no-write inspection. `fast`, `normal`, and `deep` are all
real quantization runs; they differ by search breadth and tensor-plan evidence
thresholds, not by replacing the model with a fake or partial artifact.

## Recipe Fields

Core inputs:

- `io.input`: source GGUF.
- `io.output`: final GGUF path.
- `base.ftype`: llama.cpp base quantization type when using stock fallback
  behavior.
- `base.output_tensor_type`, `base.token_embedding_type`: optional explicit
  output/head and token embedding tensor types.
- `base.mtp_tensor_type`: optional explicit MTP/NextN matrix tensor type.
  NVFP4 profiles default this to `NVFP4`; MXFP6 profiles default it to
  `MXFP6_E2M3`; the MXFP8 profile defaults it to `MXFP8`. Norm/vector MTP
  tensors are still preserved.
- `target.precision_mode`: `nvfp4`, `mxfp8`, `mxfp6`, `nvfp4_mxfp6`, or
  `mxfp6-primary`.
- `target.target_bpw`: final bits-per-weight goal for mixed allocation.
- `target.vram_gb`: optional runtime budget for allocation choices.

Calibration:

- `calibration.corpus`: text or token data used for imatrix generation.
- `calibration.imatrix`: imatrix file used for activation-aware scaling.
- `calibration.ctx_size`, `calibration.batch_size`, and
  `calibration.ubatch_size`: calibration shape.
- `calibration.n_gpu_layers`: optional `llama-imatrix -ngl` value for imatrix
  generation. Leave blank for the tool default, or use `auto`, `all`, or a
  layer count when you deliberately want to control calibration offload.

Evaluation:

- `evaluation.bf16_reference`: trusted source model for saved-logit evidence.
- `evaluation.corpus`: held-out evaluation corpus.
- `evaluation.kld_base`: saved-logit KLD base.
- `evaluation.perplexity_bin`: optional path to the retained
  `llama-perplexity` executable.

Saved-logit KLD base generation is intentionally minimal. Use the command
printed by `advanced-gguf-quantizer kld-command`; it should contain only the
reference model, evaluation corpus, and saved-logit output path. Let
`llama-perplexity` use its defaults for runtime shape and scheduling.

Runtime and memory:

- `base.threads`: CPU worker count for general execution.
- choose a conservative run profile for shared machines and small GPUs;
- leave CPU and memory headroom for CUDA, file cache, and the OS;
- use checkpoints and resume for long runs;
- reduce measured-candidate breadth when memory is tight.

Stream counts, row chunk sizes, selector workers, and logits workspace internals
are resolved implementation details. They may appear in locked run manifests for
reproducibility, but they are not public tuning knobs.

## Work Modes

Production selector runs use one work mode:

- `fast`: compact KLD-guided search for quick iteration.
- `normal`: default production search.
- `deep`: broader search for final comparisons or difficult models.

Direct `llama-quantize` saved-logit selector runs use
`--mode fast|normal|deep`. Recipe runs store the same value in
`quantizer.mode`.

## Profiles

- `nvfp4`: compact Blackwell NVFP4 model with NVFP4 MTP matrix weights.
- `mxfp8`: direct 8.25 bpw OCP MXFP8 E4M3 model with MXFP8 embeddings,
  output, and MTP matrix weights. This direct profile does not require an
  imatrix or saved-logit KLD base.
- `mxfp6`: MXFP6_E2M3 model with MXFP6_E2M3 MTP matrix weights.
- `nvfp4_mxfp6`: NVFP4-first mixed model that promotes sensitive tensors to
  MXFP6 or stronger fallback types while keeping MTP matrix weights NVFP4.
- `mxfp6-primary`: MXFP6-first model that demotes selected tensors to NVFP4
  when measured quality allows it, with MXFP6_E2M3 MTP matrix weights by
  default.
- `repair`: edit or repair an existing GGUF with targeted tensor policy.
- `q8_0`: conservative baseline or calibration helper.

MXFP6_E2M3 is experimental and unsupported by NVIDIA and llama.cpp. Future
official MXFP6 support may use a different format, so GGUFs created here may
not remain compatible. Feedback is requested through the MXFP6-capable branch:
<https://github.com/michaelw9999/llama.cpp/tree/mxfp6-cuda>.

## Staged Pipeline

Quality mode is reported in five named stages:

1. `probe`: inspect the source GGUF, tensor names, MTP/NextN metadata, output
   policy, and expected scale tensors.
2. `candidate_search`: run cheap proxy scoring and prune obviously bad
   policies.
3. `measured_ppl_kld_eval`: evaluate shortlisted policies with runtime patching
   and real saved-logit PPL/KLD evidence.
4. `quality_repair`: apply targeted repair or edit decisions through the same
   GGUF writer path used by normal quantization.
5. `export`: write the final GGUF, inspect it, emit reports, and prepare a
   validation smoke script.

## Quality Metrics

Quality mode treats PPL as a guardrail and KLD/tail behavior as the main
selection signal. Reports should include:

- PPL and log-PPL ratio;
- mean KLD;
- p95, p99, and p999 KLD;
- KLD tail mean or CVaR;
- max KLD as a risk signal;
- RMS probability delta;
- same-top rate;
- top-flip weight;
- teacher-top probability RMSE;
- entropy RMSE;
- non-finite logit counts.

p99 and p999 controls are visible by default:

- `selector.ranking.p99_penalty`
- `selector.ranking.p999_penalty`
- `selector.ranking.p99_hard_gate`
- `selector.ranking.p999_hard_gate`
- `selector.ranking.p99_threshold`
- `selector.ranking.p999_threshold`

Hard gates reject candidates that cross configured risk thresholds. Soft
penalties let a candidate survive only when it earns the tail risk with better
overall evidence.

## RSF And Candidate Families

Refined scale fit (RSF) is part of the normal quantization path by default. Each
tensor can refine its tensor scale while the local block/subblock fit remains
imatrix-weighted. RSF is a first-pass scale-fitting strategy, not a late
repair-only path. Set the granularity, search depth, or report path directly:

```bash
./build/bin/llama-quantize ... --selector-rsf-mode tensor \
  --selector-rsf-depth deeper \
  --selector-rsf-report runs/model/rsf-report.txt
```

Recipe files can set the RSF granularity explicitly for diagnostics or
reproduction:

```toml
[nvfp4.rsf]
mode = "tensor"       # tensor | slice | expert | group
depth = "exhaustive"  # normal | deep | deeper | exhaustive
```

RSF variants are shortlisted by the existing proxy path and measured directly
instead of carrying parallel non-RSF siblings through the selector. When a KLD
base is available, the full-base exact KLD score validates the composed tensor
plan. Adaptive four-over-six remains the default NVFP4 encoder path for these
variants and the base policies they extend.
When `base.mtp_tensor_type = "NVFP4"`, MTP/NextN matrix weights use the same
NVFP4 encoder policy, auxiliary scale tensors, input-scale policy, and RSF
configuration as the rest of the NVFP4 model.
During measured Stage-B policy materialization, direct runtime patching skips
the extra tensor reconstruction-stat pass by default because the policy rank is
computed from saved-logit runtime metrics: PPL, mean and tail KLD, RMS
probability delta, same-top rate, top-flip weight, teacher-top probability
RMSE, and entropy RMSE. Use `--selector-stageb-patch-eval` when a
diagnostic run also needs patch-time tensor RMSE/abs/max aggregates in the log.
Use `--selector-stageb-direct-verify` or
`--selector-no-stageb-direct-verify` to override direct-patch readback
verification for runtime-patch diagnostics.

The default RSF selector budget is a compact KLD-guided real-artifact search:
proxy-ranked seeds are refined into a small RSF scale-fit frontier,
duplicate-equivalent policies are removed before measured survey/eval, and
Stage B spends its KLD budget on the most distinct candidates. Use explicit
selector fields only when intentionally broadening the search or running a
targeted reproduction.
Proxy-only survey passes may use a bounded RSF depth so that expensive
exhaustive scale fitting is reserved for measured KLD/PPL candidate evaluation
and final materialization.

After the selected policy is materialized, the main selector can also
consider speed-aware tensor type candidates for the worst baseline-error tensors.
This is not a separate repair pass: it is part of the final exact tensor map.
With a saved-logit KLD selector, the default cap is mode-dependent and based on
outlier error rather than an arbitrary fixed fraction. Format-local RSF repairs
are tried first; only tensors where the baseline format still has high error are
allowed to move to stronger tensor types.

```bash
./build/bin/llama-quantize ... --selector-candidate-types Q5_0,Q4_K,Q5_K,Q6_K,Q8_0
```

Mixed NVFP4/MXFP6 runs prepend `MXFP6_E2M3` to that candidate list by default,
so MXFP6 can win the tensor before Q4_K when it is available. Candidate ranking
uses tensor-local baseline error with static size and speed penalties; it does not
run benchmarks during quantization. Use `--selector-candidate-fraction F`
or `--selector-candidate-top N` to make the cap stricter or broader.

For targeted diagnostics, `--selector-include-policy name` and
`--selector-include-policies a,b` limit selector work to exact policy
names while still allowing the internal `seed_keep` checkpoint policy.

Fresh selector runs build a per-tensor policy plan before final
materialization. Tensor-local proxy evidence chooses policy changes across the
full tensor set, then the loaded runtime scores the whole plan against the KLD
base before writing the final GGUF. This is enabled by default for measured
selector runs; use `--selector-no-tensor-policy-map` only for diagnostics
or whole-policy comparison runs.

The tensor-policy proxy gate is tied to work mode: `fast` uses a 0.0005
absolute gain floor, `normal` uses 0.0002, and `deep` uses 0.0001, with a small
relative guard to avoid floating-point noise.

## Selector Ledger

The recipe surface can keep a selector evidence ledger:

```toml
[selector]
ledger = "runs/model/selector-ledger.jsonl"
```

`selector.ledger` passes through to `--selector-ledger` when set and
appends raw selector evidence rows to JSONL. When exact stage-B KLD rows in the
ledger match the current cache key, the selector can reuse them as a read-through
cache before remeasuring a policy. Leave it empty for normal runs.

Ledger estimates are not release-quality evidence. They can help avoid
recomputing matching exact rows, but final claims still need the same exact
artifact evaluation and completion smoke required for any production run.

Inspect a ledger without running model evaluation:

```bash
./build/bin/advanced-gguf-quantizer evidence inspect runs/model/selector-ledger.jsonl --top 40
```

## Best Candidate Reports

Use `best` for non-dominated candidate sets. Internally this is Pareto-style
selection: a candidate is retained when no other feasible candidate is no worse
on every selected objective and strictly better on at least one.

Recommended quality report:

```bash
./build/bin/advanced-gguf-quantizer best runs/model/metrics.jsonl \
  --real-ppl-kld \
  --quality-only \
  --report runs/model/best-report.json
```

Use `--quality-only` when deciding which candidate is best by quality. Treat
BPW, size, speed, and VRAM as explicit constraints unless you are intentionally
making a tradeoff report.

Metrics records should include identity fields, source GGUF, recipe lock,
commit, CUDA device, calibration corpus, imatrix, evaluation corpus, context,
KLD base, final bytes, BPW, tensor mix, and quality metrics.

## Mixed NVFP4/MXFP6 Allocation

The mixed allocator can work in two directions:

- `nvfp4_mxfp6`: start from compact NVFP4 and spend MXFP6 on sensitive tensors.
- `mxfp6-primary`: start from quality-oriented MXFP6 and use NVFP4 where the
  measured loss is acceptable.

Budget and fallback controls:

- `target.target_bpw`: desired final BPW.
- `target.vram_gb`: optional runtime memory budget.
- `mixed.nvfp4_budget_bias`: favors more or fewer NVFP4 assignments.
- `mixed.quality_margin`: acceptable measured loss for a demotion.
- `mixed.fallback_types`: allowed fallback types such as `MXFP6_E2M3`, `Q4_K`,
  `Q6_K`, `Q8_0`, `BF16`, and `F16`.
- `layer_policy.output_type`: separate output/head policy.
- `layer_policy.embedding_type`: token embedding policy.

Do not force every sensitive tensor into NVFP4 just to hit a size target. Use
the candidate report to understand the cost of each assignment.

## Advanced Quality Controls

The recipe keeps a small set of low-level quality controls for reproducible
expert runs:

- `mxfp6.input_scale_denom` and `mxfp6.input_scale_quantile`: activation-aware
  MXFP6 input-scale calibration from imatrix data.
- `mxfp6.tensor_scale_sample_blocks` and `mxfp6.tensor_scale_steps`: effort used
  by MXFP6 tensor-scale search.
- `mixed.sample_blocks` and `mixed.sample_cap`: proxy sample breadth for
  NVFP4/MXFP6 allocation.
- `mixed.imatrix_weight_blend`, `mixed.imatrix_weight_power`,
  `mixed.imatrix_weight_min`, and `mixed.imatrix_weight_max`: imatrix shaping for
  mixed-format proxy scoring.

Leave these empty for normal runs. Set them only when reproducing an earlier
artifact, comparing optimizer behavior, or deliberately spending more work on
the mixed-format selector.

## Fused Decision Units

Layer policy should keep related tensors coherent:

- q/k/v projections;
- gate/up MLP projections;
- MoE expert pairs;
- MTP or NextN heads;
- token embeddings, tied embeddings, and separate `output.weight`;
- lm head and output adapter tensors.

Per-tensor overrides are still allowed, but coherent groups should be the
default unit for candidate generation and reporting.

## Checkpoints

Runs write a checkpoint key report that includes:

- source model;
- imatrix;
- KLD base;
- recipe lock;
- resolved quality policy and allocation settings;
- code commit.

If the key changes, use a new run directory or intentionally clear old
checkpoint/cache files. This prevents stale resumed work from mixing incompatible
recipes.

## Artifacts

Every `run` should leave:

- `recipe.lock.toml`
- `run.jsonl`
- `assignment.jsonl`
- `run-manifest.json`
- `checkpoint-key.json`
- `quantization-report.md`
- `validate-output-smoke.sh`
- final GGUF output

The assignment log records tensor decisions, fused decision units, and tail gate
settings. The report is the human-readable summary.

## Memory-Bounded Runs

For models too large to keep all evidence hot in memory:

- choose a conservative run profile for the target GPU;
- leave CPU cores and memory headroom;
- use checkpoint and runtime-cache reuse;
- prefer smaller measured candidate batches;
- resume from the project directory rather than launching a wider fresh run.

Layer-by-layer streaming is the public mental model: load or patch only the
needed tensor groups, write checkpoints, and avoid retaining every candidate in
host memory.

## Validation

Before expensive runs:

```bash
./build/bin/llama-quantize inspect models/source-bf16.gguf
./build/bin/advanced-gguf-quantizer recipe validate recipes/model.toml
./build/bin/advanced-gguf-quantizer plan recipes/model.toml
```

After a run:

```bash
./build/bin/llama-quantize inspect runs/model/output.gguf
./build/bin/llama-completion -m runs/model/output.gguf -p "The model is"
./build/bin/llama-perplexity -m runs/model/output.gguf -f data/eval.txt
```

For strong claims, compare only artifacts with matching source, calibration,
imatrix, evaluation corpus, KLD base, and build commit.
