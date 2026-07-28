# MXFP8 GGUF Contract

MXFP8 uses the OCP microscaling E4M3 data format with one E8M0 scale for every
32 values. This repository stores eight 32-value sub-blocks in each 256-value
GGML block.

## Binary Layout

The public GGML type is `GGML_TYPE_MXFP8` (type ID 43). A block contains:

- 8 E8M0 scale bytes;
- 8 groups of 32 E4M3 value bytes;
- 256 logical values and 264 stored bytes;
- 8.25 bits per weight.

The mostly-type selectors are `GGML_FTYPE_MOSTLY_MXFP8` and
`LLAMA_FTYPE_MOSTLY_MXFP8`. Writers must emit the canonical 264-byte layout;
runtime-only CUDA activation tiles are not a GGUF storage format.

## Quantization

For each 32-value sub-block, the encoder chooses a power-of-two E8M0 scale that
keeps the largest finite magnitude representable by E4M3, then rounds each
scaled value to E4M3. Zero sub-blocks use the neutral E8M0 scale code.

The direct recipe is:

```bash
./build/bin/advanced-gguf-quantizer recipe init \
  --profile mxfp8 \
  --output recipes/mxfp8.toml
```

It defaults the base, token embedding, output, and MTP matrix types to `MXFP8`
with a target of 8.25 bpw. It is a direct format recipe, so calibration and KLD
evidence are optional rather than launch requirements.

## Runtime Support

CPU support includes reference quantization/dequantization, generic dot
products, AVX2 dot products on x86, and NEON dot products on Arm.

CUDA support includes dequantization, conversion, `GET_ROWS`, generic
matrix-vector execution, and matrix-matrix execution. Blackwell builds use
native block-scaled E4M3 MMA for MXFP8 MMQ and the one-column inference path;
other supported CUDA architectures use the generic quantized path.

## Validation

After writing an artifact:

```bash
./build/bin/llama-quantize inspect model-mxfp8.gguf --tensors
./build/bin/llama-completion -m model-mxfp8.gguf -p "The model is"
```

Inspection should report MXFP8 tensor counts, preserve MTP/NextN metadata and
non-matrix tensors, and show an output policy consistent with the locked
recipe.
