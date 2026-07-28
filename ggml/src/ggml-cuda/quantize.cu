#include "quantize.cuh"
#include <cstdint>

__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)
static __global__ void quantize_q8_1(
        const float * x_ptr, void * vy_ptr,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT x  = x_ptr;
    void        * GGML_CUDA_RESTRICT vy = vy_ptr;
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3*ne2.z;
    const int64_t i1 = blockIdx.y;

    const int64_t & i00 = i0;
    const int64_t & i01 = i1;
    const int64_t & i02 = i2;
    const int64_t & i03 = i3;

    const int64_t i_cont = ((i3*ne2.z + i2) * ne1 + i1) * ne0 + i0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib  = i_cont / QK8_1; // block index
    const int64_t iqs = i_cont % QK8_1; // quant index

    ggml_cuda_pdl_sync();
    const float xi = i0 < ne00 ? x[i03*s03 + i02*s02 + i01*s01 + i00] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds = make_half2(d, sum);
}

__device__ __forceinline__ uint8_t compute_e8m0_scale(float amax) {
    if (!(amax > 0.0f)) {
        return 0;
    }

    // FP4 E2M1: max exponent (unbiased) is 2.
    constexpr int FP4_E2M1_EMAX = 2;

    const float e = log2f(amax);

    // "even" -> round-to-nearest integer, ties-to-even
    const int e_int = __float2int_rn(e);

    const int shared_exp = e_int - FP4_E2M1_EMAX;

    int biased = shared_exp + 127;

    biased = max(biased, 0);
    biased = min(biased, 254);

    return static_cast<uint8_t>(biased);
}

template <bool has_ids, bool has_scale>
static __global__ void quantize_mmq_nvfp4(const float * __restrict__ x,
                                          const int32_t * __restrict__ ids,
                                          const int32_t * __restrict__ ids_expert,
                                          void * __restrict__ vy,
                                          const float * __restrict__ scale_activation,
                                          const int64_t scale_activation_ne,
                                          const int64_t ne00,
                                          const int64_t s01,
                                          const int64_t s02,
                                          const int64_t s03,
                                          const int64_t ne0,
                                          const int64_t ne1,
                                          const int64_t ne2) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t i0_base = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * QK_NVFP4_SUB;
    if (i0_base >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;
    const int64_t i01 = has_ids ? ids[i1] : i1;
    float inv_input_scale = 1.0f;
    if constexpr (has_scale) {
        const int64_t scale_idx = scale_activation_ne <= 1 ? 0 : (has_ids && ids_expert ? ids_expert[i1] : i01);
        float input_scale = scale_activation[scale_idx];
        if (!(input_scale > 0.0f) || !isfinite(input_scale)) {
            input_scale = 1.0f;
        }
        inv_input_scale = 1.0f / input_scale;
    }

    const int64_t k_block = i0_base / QK_K;
    const int64_t blocks_per_col = (ne0 + QK_K - 1) / QK_K;
    if (k_block >= blocks_per_col) {
        return;
    }

    const int64_t batch_offset = (int64_t) blockIdx.z * (blocks_per_col * ne1);
    block_nvfp4_mmq * yb = (block_nvfp4_mmq *) vy + batch_offset + k_block * ne1 + blockIdx.x;
    const int sub = (i0_base % QK_K) / QK_NVFP4_SUB;

    float vals0[8];
    float vals1[8];
    float sub_max = 0.0f;
    const int64_t base_idx = i3 * s03 + i2 * s02 + i01 * s01;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int64_t i00 = i0_base + k;
        const float v = i00 < ne00 ? x[base_idx + i00] * inv_input_scale : 0.0f;
        vals0[k] = v;
        sub_max = fmaxf(sub_max, fabsf(v));
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int64_t i00 = i0_base + 8 + k;
        const float v = i00 < ne00 ? x[base_idx + i00] * inv_input_scale : 0.0f;
        vals1[k] = v;
        sub_max = fmaxf(sub_max, fabsf(v));
    }

    static constexpr int test_offsets[5] = { 0, -1, 1, -2, 2 };
    const int first_fp8_code = (int) ggml_cuda_fp32_to_ue4m3(sub_max * (1.0f / 6.0f));
    const int max_fp8_code = 0x7E;

    float best_err = 1e30f;
    uint8_t fp8_code = 0;
    float subblock_scale = 0.0f;

#pragma unroll
    for (int i = 0; i < 5; ++i) {
        const int test_code = first_fp8_code + test_offsets[i];
        if (test_code < 0 || test_code > max_fp8_code) {
            continue;
        }

        const float test_scale = ggml_cuda_nvfp4_scale_to_fp32_half((uint8_t) test_code);
        if (!(test_scale > 0.0f) || !isfinite(test_scale)) {
            continue;
        }

        float cur_err = 0.0f;
        const float test_inv_scale = 0.5f / test_scale;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint8_t q0 = ggml_cuda_float_to_fp4_e2m1(vals0[k], test_inv_scale);
            const uint8_t q1 = ggml_cuda_float_to_fp4_e2m1(vals1[k], test_inv_scale);
            const float err0 = fabsf(vals0[k]) - fabsf(kvalues_mxfp4[q0 & 0x7]) * test_scale;
            const float err1 = fabsf(vals1[k]) - fabsf(kvalues_mxfp4[q1 & 0x7]) * test_scale;
            cur_err += err0 * err0 + err1 * err1;
        }

        if (cur_err < best_err) {
            best_err = cur_err;
            fp8_code = (uint8_t) test_code;
            subblock_scale = test_scale;
        }
    }

    const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
    uint32_t packed_lo = 0;
    uint32_t packed_hi = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        packed_lo |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals0[k], inv_scale) << (4 * k);
        packed_hi |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals1[k], inv_scale) << (4 * k);
    }
    yb->qs_u32[2 * sub + 0] = packed_lo;
    yb->qs_u32[2 * sub + 1] = packed_hi;
    reinterpret_cast<uint8_t *>(yb->sc4_u32)[sub] = fp8_code;
#else
    NO_DEVICE_CODE; // This is for Blackwell NVFP4 activations only.
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

static __device__ __forceinline__ uint8_t mxfp6_e2m3_scale_code_from_amax(float amax) {
    if (!(amax > 0.0f) || !isfinite(amax)) {
        return 127;
    }

    const float scaled = amax * (1.0f / 7.5f);
    const uint32_t bits = __float_as_uint(scaled);
    const int exp_bits = int((bits >> 23) & 0xff);
    int code = 0;
    if (exp_bits != 0) {
        const int exp = exp_bits - 127;
        const int mantissa = int(bits & 0x007fffff);
        code = exp + (mantissa != 0) + 127;
    }
    code = max(0, min(254, code));
    return (uint8_t) code;
}

static __device__ __forceinline__ uint8_t mxfp6_e2m3_quant(float x, float inv_scale) {
    if (!isfinite(x) || x == 0.0f || !(inv_scale > 0.0f)) {
        return 0;
    }

#if defined(GGML_CUDA_MXFP6_E2M3_NATIVE_FP6)
    return ggml_cuda_mxfp6_e2m3_cvt_f32(x * inv_scale);
#else
    const int sign = x < 0.0f ? 0x20 : 0x00;
    const float q = fabsf(x) * inv_scale * 8.0f;
    int code = 0;
    if (q < 15.5f) {
        code = max(0, min(15, __float2int_rn(q)));
    } else if (q < 31.0f) {
        const int mantissa = __float2int_rn((q - 16.0f) * 0.5f);
        code = 16 + max(0, min(7, mantissa));
    } else {
        const int mantissa = __float2int_rn((q - 32.0f) * 0.25f);
        code = 24 + max(0, min(7, mantissa));
    }
    return (uint8_t) (sign | code);
#endif // defined(GGML_CUDA_MXFP6_E2M3_NATIVE_FP6)
}

template <bool has_ids, bool has_scale>
static __global__ void quantize_mmq_mxfp6_e2m3(const float * __restrict__ x,
                                               const int32_t * __restrict__ ids,
                                               const int32_t * __restrict__ ids_expert,
                                               void * __restrict__ vy,
                                               const float * __restrict__ scale_activation,
                                               const int64_t scale_activation_ne,
                                               const int64_t ne00,
                                               const int64_t s01,
                                               const int64_t s02,
                                               const int64_t s03,
                                               const int64_t ne0,
                                               const int64_t ne1,
                                               const int64_t ne2) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    constexpr int warps_per_block = 8;
    const int64_t k_block = blockIdx.y;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int64_t i1 = int64_t(blockIdx.x) * warps_per_block + warp;
    if (i1 >= ne1) {
        return;
    }
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;
    const int64_t i01 = has_ids ? ids[i1] : i1;

    float inv_input_scale = 1.0f;
    bool use_input_scale = false;
    if constexpr (has_scale) {
        if (lane == 0) {
            const int64_t scale_idx = scale_activation_ne <= 1 ? 0 : (has_ids && ids_expert ? ids_expert[i1] : i01);
            const float input_scale = scale_activation[scale_idx];
            if (input_scale != 1.0f && input_scale > 0.0f && isfinite(input_scale)) {
                inv_input_scale = 1.0f / input_scale;
            }
        }
        inv_input_scale = __shfl_sync(0xFFFFFFFFu, inv_input_scale, 0);
        use_input_scale = inv_input_scale != 1.0f;
    }

    const int64_t blocks_per_col = (ne0 + QK_MXFP6_E2M3 - 1) / QK_MXFP6_E2M3;
    if (k_block >= blocks_per_col) {
        return;
    }

    const int64_t batch_offset = (int64_t) blockIdx.z * (blocks_per_col * ne1);
    tile_mxfp6_e2m3_mmq * y = (tile_mxfp6_e2m3_mmq *) vy + batch_offset + k_block * ne1 + i1;

    const int64_t base_idx = i3 * s03 + i2 * s02 + i01 * s01;
    for (int frag = 0; frag < QK_MXFP6_E2M3_FRAGS; ++frag) {
        const int64_t i00 = k_block * QK_MXFP6_E2M3 + frag * QK_MXFP6_E2M3_SUB + lane;
        const float loaded = i00 < ne00 ? (use_input_scale ? x[base_idx + i00] * inv_input_scale : x[base_idx + i00]) : 0.0f;
        const float val = isfinite(loaded) ? loaded : 0.0f;

        const float amax = warp_reduce_max<QK_MXFP6_E2M3_SUB>(fabsf(val));
        const uint8_t scale_code = mxfp6_e2m3_scale_code_from_amax(amax);
        const float scale = ggml_cuda_e8m0_to_fp32(scale_code);
        const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        const uint32_t code = (uint32_t) mxfp6_e2m3_quant(val, inv_scale);

        if (lane == 0) {
            y->e[frag] = scale_code;
        }

        ((uint8_t *) y->qs_u32[frag])[lane] = (uint8_t) (code & 0x3Fu);
    }
#else
    GGML_UNUSED_VARS(x, ids, ids_expert, vy, scale_activation, scale_activation_ne, ne00, s01, s02, s03, ne0, ne1, ne2);
    NO_DEVICE_CODE;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

static __device__ __forceinline__ uint8_t fp8_scale_code_from_amax(float amax) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    if (!(amax > 0.0f) || !isfinite(amax)) {
        return 127;
    }
    int e = (int) ceilf(log2f(amax / 448.0f)) + 127;
    e = max(0, min(254, e));
    return (uint8_t) e;
#else
    GGML_UNUSED(amax);
    NO_DEVICE_CODE;
    return 127;
#endif
}

static __device__ __forceinline__ uint8_t fp8_quant_e4m3(float x, float inv_scale) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    if (!isfinite(x) || x == 0.0f || !(inv_scale > 0.0f)) {
        return 0;
    }

    const uint8_t sign = x < 0.0f ? 0x80 : 0x00;
    const float ax = fabsf(x) * inv_scale;
    if (!(ax > 0.0f)) {
        return 0;
    }
    if (ax >= 448.0f) {
        return sign | 0x7E;
    }

    const int floor_exp = (int) floorf(log2f(ax));
    int exp = floor_exp + 7;
    int man = 0;
    if (exp <= 0) {
        man = (int) floorf(ax * 512.0f + 0.5f);
        man = max(0, min(7, man));
        return man == 0 ? 0 : (sign | (uint8_t) man);
    }

    const float base = exp2f((float) floor_exp);
    man = (int) floorf((ax / base - 1.0f) * 8.0f + 0.5f);
    if (man >= 8) {
        man = 0;
        ++exp;
    }
    if (exp > 15 || (exp == 15 && man > 6)) {
        return sign | 0x7E;
    }
    return sign | (uint8_t) ((exp << 3) | man);
#else
    GGML_UNUSED_VARS(x, inv_scale);
    NO_DEVICE_CODE;
    return 0;
#endif
}

template <bool has_ids, bool has_scale>
static __global__ void quantize_mmq_fp8_e4m3(const float * __restrict__ x,
                                               const int32_t * __restrict__ ids,
                                               const int32_t * __restrict__ ids_expert,
                                               void * __restrict__ vy,
                                               const float * __restrict__ scale_activation,
                                               const int64_t scale_activation_ne,
                                               const int64_t ne00,
                                               const int64_t s01,
                                               const int64_t s02,
                                               const int64_t s03,
                                               const int64_t ne0,
                                               const int64_t ne1,
                                               const int64_t ne2) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t k_block = blockIdx.y;
    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;
    const int64_t i01 = has_ids ? ids[i1] : i1;

    float inv_input_scale = 1.0f;
    if constexpr (has_scale) {
        const int64_t scale_idx = scale_activation_ne <= 1 ? 0 : (has_ids && ids_expert ? ids_expert[i1] : i01);
        float input_scale = scale_activation[scale_idx];
        if (!(input_scale > 0.0f) || !isfinite(input_scale)) {
            input_scale = 1.0f;
        }
        inv_input_scale = 1.0f / input_scale;
    }

    const int64_t blocks_per_col = (ne0 + QK_FP8 - 1) / QK_FP8;
    if (k_block >= blocks_per_col) {
        return;
    }

    const int64_t batch_offset = (int64_t) blockIdx.z * (blocks_per_col * ne1);
    block_fp8 * y = (block_fp8 *) vy + batch_offset + k_block * ne1 + i1;

    constexpr int warps_per_block = 4;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int64_t base_idx = i3 * s03 + i2 * s02 + i01 * s01;
    for (int frag = warp; frag < QK_FP8_FRAGS; frag += warps_per_block) {
        const int64_t i00 = k_block * QK_FP8 + frag * QK_FP8_SUB + lane;
        const float loaded = i00 < ne00 ? x[base_idx + i00] * inv_input_scale : 0.0f;
        const float val = isfinite(loaded) ? loaded : 0.0f;

        const float amax = warp_reduce_max<QK_FP8_SUB>(fabsf(val));
        const uint8_t scale_code = fp8_scale_code_from_amax(amax);
        const float scale = ggml_cuda_e8m0_to_fp32(scale_code);
        const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        const uint32_t code = (uint32_t) fp8_quant_e4m3(val, inv_scale);

        if (lane == 0) {
            y->e[frag] = scale_code;
        }

        const int pack_lane = lane & ~3;
        const uint32_t packed =
            (__shfl_sync(0xFFFFFFFFu, code, pack_lane + 0) <<  0) |
            (__shfl_sync(0xFFFFFFFFu, code, pack_lane + 1) <<  8) |
            (__shfl_sync(0xFFFFFFFFu, code, pack_lane + 2) << 16) |
            (__shfl_sync(0xFFFFFFFFu, code, pack_lane + 3) << 24);
        if ((lane & 3) == 0) {
            ggml_cuda_fp8_set4_u8containers(*y, frag, lane >> 2, packed);
        }
        if (frag == 0 && lane == 0) {
            y->pad[0] = 0;
        }
    }
#else
    GGML_UNUSED_VARS(x, ids, ids_expert, vy, scale_activation, scale_activation_ne, ne00, s01, s02, s03, ne0, ne1, ne2);
    NO_DEVICE_CODE;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

static __global__ void quantize_mmq_mxfp8(
        const float * __restrict__ x, const int32_t * __restrict__ ids,
        void * __restrict__ vy, const int64_t ne00,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) {
        return;
    }

    ggml_cuda_pdl_sync();
    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t base_idx = i3 * s03 + i2 * s02 + i01 * s01;

    const float4 xi = i0 < ne00 ?
        make_float4(x[base_idx + i0 + 0], x[base_idx + i0 + 1],
                    x[base_idx + i0 + 2], x[base_idx + i0 + 3]) :
        make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float amax = fmaxf(fabsf(xi.x), fabsf(xi.y));
    amax = fmaxf(amax, fmaxf(fabsf(xi.z), fabsf(xi.w)));
    amax = warp_reduce_max<QK_MXFP8_SUB / 4>(amax);

    const uint8_t e = fp8_scale_code_from_amax(amax);
    const float scale = ggml_cuda_e8m0_to_fp32(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    const uint32_t packed =
        ((uint32_t) fp8_quant_e4m3(xi.x, inv_scale) <<  0) |
        ((uint32_t) fp8_quant_e4m3(xi.y, inv_scale) <<  8) |
        ((uint32_t) fp8_quant_e4m3(xi.z, inv_scale) << 16) |
        ((uint32_t) fp8_quant_e4m3(xi.w, inv_scale) << 24);

    block_mxfp8_mmq * y = (block_mxfp8_mmq *) vy;
    const int64_t ib0 = blockIdx.z * ((int64_t) ne1 * ne0 / (4 * QK8_1));
    const int64_t ib = ib0 + (i0 / (4 * QK8_1)) * ne1 + i1;
    const int64_t iqs = i0 % (4 * QK8_1);
    ((uint32_t *) y[ib].qs)[iqs / 4] = packed;
    if (iqs % QK_MXFP8_SUB == 0) {
        y[ib].d4[iqs / QK_MXFP8_SUB] = e;
    }
#else
    GGML_UNUSED_VARS(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
    NO_DEVICE_CODE;
#endif
}

template <bool has_ids, bool has_scale>
static __global__ void quantize_row_fp8_e4m3(const float * __restrict__ x,
                                               const int32_t * __restrict__ ids,
                                               void * __restrict__ vy,
                                               const float * __restrict__ scale_activation,
                                               const int64_t scale_activation_ne,
                                               const int64_t ne00,
                                               const int64_t s01,
                                               const int64_t s02,
                                               const int64_t s03,
                                               const int64_t ne0,
                                               const int64_t ne1,
                                               const int64_t ne2) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t k_block = blockIdx.y;
    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;
    const int64_t i01 = has_ids ? ids[i1] : i1;

    float inv_input_scale = 1.0f;
    if constexpr (has_scale) {
        const int64_t scale_idx = scale_activation_ne <= 1 ? 0 : i01;
        float input_scale = scale_activation[scale_idx];
        if (!(input_scale > 0.0f) || !isfinite(input_scale)) {
            input_scale = 1.0f;
        }
        inv_input_scale = 1.0f / input_scale;
    }

    const int64_t blocks_per_col = (ne0 + QK_FP8 - 1) / QK_FP8;
    if (k_block >= blocks_per_col) {
        return;
    }

    const int64_t batch_offset = (int64_t) blockIdx.z * (ne1 * blocks_per_col);
    block_fp8 * y = (block_fp8 *) vy + batch_offset + i1 * blocks_per_col + k_block;

    constexpr int warps_per_block = 4;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int64_t base_idx = i3 * s03 + i2 * s02 + i01 * s01;
    for (int frag = warp; frag < QK_FP8_FRAGS; frag += warps_per_block) {
        const int64_t i00 = k_block * QK_FP8 + frag * QK_FP8_SUB + lane;
        const float loaded = i00 < ne00 ? x[base_idx + i00] * inv_input_scale : 0.0f;
        const float val = isfinite(loaded) ? loaded : 0.0f;

        const float amax = warp_reduce_max<QK_FP8_SUB>(fabsf(val));
        const uint8_t scale_code = fp8_scale_code_from_amax(amax);
        const float scale = ggml_cuda_e8m0_to_fp32(scale_code);
        const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        const uint32_t code = (uint32_t) fp8_quant_e4m3(val, inv_scale);

        if (lane == 0) {
            y->e[frag] = scale_code;
        }

        const int pack_lane = lane & ~3;
        const uint32_t packed =
            (__shfl_sync(0xFFFFFFFFu, code, pack_lane + 0) <<  0) |
            (__shfl_sync(0xFFFFFFFFu, code, pack_lane + 1) <<  8) |
            (__shfl_sync(0xFFFFFFFFu, code, pack_lane + 2) << 16) |
            (__shfl_sync(0xFFFFFFFFu, code, pack_lane + 3) << 24);
        if ((lane & 3) == 0) {
            ggml_cuda_fp8_set4_u8containers(*y, frag, lane >> 2, packed);
        }
        if (frag == 0 && lane == 0) {
            y->pad[0] = 0;
        }
    }
#else
    GGML_UNUSED_VARS(x, ids, vy, scale_activation, scale_activation_ne, ne00, s01, s02, s03, ne0, ne1, ne2);
    NO_DEVICE_CODE;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

// quantize values in the format mxfp4 is stored which is interleaved nibbles
// i.e. a block a0-a31 is represented as a0a16,a1a17 ...a15a31
static __global__ void quantize_mmq_mxfp4(const float * __restrict__ x,
                                          const int32_t * __restrict__ ids,
                                          void * __restrict__ vy,
                                          const int64_t ne00,
                                          const int64_t s01,
                                          const int64_t s02,
                                          const int64_t s03,
                                          const int64_t ne0,
                                          const int     ne1,
                                          const int     ne2) {
    constexpr int vals_per_scale = 32;
    constexpr int vals_per_warp  = 2 * vals_per_scale;  // Each warp processes 2 blocks of 32 = 64 values

    const int warp_id = threadIdx.y;
    const int lane_id_32 = threadIdx.x;

    const int nwarps = blockDim.y;

    const int64_t warp_start_offset = (blockIdx.y * nwarps + warp_id) * vals_per_warp;

    if (warp_start_offset >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    ggml_cuda_pdl_sync();
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    block_fp4_mmq * y = (block_fp4_mmq *) vy;

    const int64_t block_fp4_mmq_size = 8 * QK_MXFP4;  // 256 values
    const int64_t ib0                = blockIdx.z * ((int64_t) ne1 * (ne0 / block_fp4_mmq_size));
    const int64_t ib = ib0 + (warp_start_offset / block_fp4_mmq_size) * ne1 + blockIdx.x;
    const int64_t quad_idx_in_block  = (warp_start_offset % block_fp4_mmq_size) / vals_per_warp;

    const int group_id = lane_id_32 / 4;
    const int lane_in_group = lane_id_32 % 4;
    const int base = group_id * 2;
    char2 * yqs2 = (char2 *) y[ib].qs;

    const int64_t base_pos = i03 * s03 + i02 * s02 + i01 * s01;

    uint8_t scales[2];

#pragma unroll
    for (int b = 0; b < 2; ++b) {
        const int64_t i0 = warp_start_offset + b * vals_per_scale + lane_id_32;
        const float xi = (i0 < ne00) ? x[base_pos + i0] : 0.0f;

        float amax = fabsf(xi);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, WARP_SIZE));
        }

        const uint8_t e = compute_e8m0_scale(amax);
        scales[b] = e;
        const float inv_s = (amax == 0.0f) ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(e));

#if CUDART_VERSION >= 12080
        const float scaled_val = xi * inv_s;

        const float val0 = __shfl_sync(0xFFFFFFFF, scaled_val, base, WARP_SIZE);
        const float val1 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 16, WARP_SIZE);
        const float val2 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 1, WARP_SIZE);
        const float val3 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 17, WARP_SIZE);

        if (lane_in_group == 0) {
            __nv_fp4x4_e2m1 fp4_packed(make_float4(val0, val1, val2, val3));

            yqs2[quad_idx_in_block * 16 + b * 8 + group_id] = *(char2 *) &fp4_packed;
        }
#else
        // Fallback: manual FP4 conversion using LUT
        const uint8_t q_val = ggml_cuda_float_to_fp4_e2m1(xi, inv_s);

        const uint8_t q_lo_0 = __shfl_sync(0xFFFFFFFF, q_val, base,      WARP_SIZE);
        const uint8_t q_lo_1 = __shfl_sync(0xFFFFFFFF, q_val, base + 1,  WARP_SIZE);
        const uint8_t q_hi_0 = __shfl_sync(0xFFFFFFFF, q_val, base + 16, WARP_SIZE);
        const uint8_t q_hi_1 = __shfl_sync(0xFFFFFFFF, q_val, base + 17, WARP_SIZE);

        if (lane_in_group == 0) {
            char2 q;
            q.x = (q_hi_0 << 4) | q_lo_0;
            q.y = (q_hi_1 << 4) | q_lo_1;
            yqs2[quad_idx_in_block * 16 + b * 8 + group_id] = q;
        }
#endif // CUDART_VERSION >= 12080
    }

    if (lane_id_32 == 0) {
        // Store 2 scales packed into 1 uint32
        y[ib].d4[quad_idx_in_block] = (scales[1] << 8) | scales[0];
    }
}

template <mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i00 = i0;
    ggml_cuda_pdl_sync();
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    const float4 * x4 = (const float4 *) x;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + (i0 / (4*QK8_1))*ne1 + blockIdx.x;                    // block index in channel
    const int64_t iqs = i0 % (4*QK8_1);                                             // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = i0 < ne00 ? x4[(i03*s03 + i02*s02 + i01*s01 + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);

    // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }

        y[ib].d2s6[2 + iqs/16] = sum;

        if (iqs % 64 != 0) {
            return;
        }

        const float d = 1.0f / d_inv;

        y[ib].d2s6[iqs/64] = d;

        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    const float d = 1.0f / d_inv;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs/32]  = d;
    }
}

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(!ids);
    GGML_ASSERT(ne0 % QK8_1 == 0);

    const uint3 ne2_fastdiv = init_fastdiv_values(ne2);

    const int64_t block_num_x = (ne0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ne1, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, block_size, 0, stream);
    ggml_cuda_kernel_launch(quantize_q8_1, launch_params, x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
    GGML_UNUSED(type_src0);
}

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % (4*QK8_1) == 0);

    // ne1 tends to assume the highest values, therefore use it as the "x" dimension of the CUDA grid:
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void quantize_mmq_mxfp8_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        cudaStream_t stream) {
    GGML_ASSERT(type_src0 == GGML_TYPE_MXFP8);
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % (4 * QK8_1) == 0);

    const int64_t block_num_y =
        (ne0 + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    quantize_mmq_mxfp8<<<num_blocks, block_size, 0, stream>>>(
        x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}

void quantize_mmq_nvfp4_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    quantize_mmq_nvfp4_cuda(x, ids, nullptr, vy, type_src0, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, nullptr, 0, stream);
}

void quantize_mmq_nvfp4_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const float * scale_activation, const int64_t scale_activation_ne, cudaStream_t stream) {
    quantize_mmq_nvfp4_cuda(x, ids, nullptr, vy, type_src0, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, scale_activation, scale_activation_ne, stream);
}

void quantize_mmq_nvfp4_cuda(
        const float * x, const int32_t * ids, const int32_t * ids_expert, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const float * scale_activation, const int64_t scale_activation_ne, cudaStream_t stream) {
    GGML_ASSERT(type_src0 == GGML_TYPE_NVFP4);
    GGML_ASSERT(ne00 % 8 == 0);
    GGML_ASSERT(ne0 > 0);

    constexpr int nvfp4_block_size = 128;
    const int64_t block_num_y = (ne0 + QK_NVFP4_SUB * nvfp4_block_size - 1) / (QK_NVFP4_SUB * nvfp4_block_size);
    const dim3 num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3 block_size(nvfp4_block_size, 1, 1);
    if (ids) {
        if (scale_activation) {
            quantize_mmq_nvfp4<true, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids, ids_expert, vy, scale_activation, scale_activation_ne, ne00, s01, s02, s03, ne0, ne1, ne2);
        } else {
            quantize_mmq_nvfp4<true, false><<<num_blocks, block_size, 0, stream>>>(
                x, ids, ids_expert, vy, nullptr, 0, ne00, s01, s02, s03, ne0, ne1, ne2);
        }
    } else if (scale_activation) {
        quantize_mmq_nvfp4<false, true><<<num_blocks, block_size, 0, stream>>>(
            x, ids, nullptr, vy, scale_activation, scale_activation_ne, ne00, s01, s02, s03, ne0, ne1, ne2);
    } else {
        quantize_mmq_nvfp4<false, false><<<num_blocks, block_size, 0, stream>>>(
            x, ids, nullptr, vy, nullptr, 0, ne00, s01, s02, s03, ne0, ne1, ne2);
    }
}

void quantize_mmq_mxfp6_e2m3_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    quantize_mmq_mxfp6_e2m3_cuda(x, ids, nullptr, vy, type_src0, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, nullptr, 0, stream);
}

void quantize_mmq_mxfp6_e2m3_cuda(
        const float * x, const int32_t * ids, const int32_t * ids_expert, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const float * scale_activation, const int64_t scale_activation_ne, cudaStream_t stream) {
    GGML_ASSERT(type_src0 == GGML_TYPE_MXFP6_E2M3);
    GGML_ASSERT(ne0 > 0);

    constexpr int warps_per_block = 8;
    const int64_t block_num_y = (ne0 + QK_MXFP6_E2M3 - 1) / QK_MXFP6_E2M3;
    const dim3 num_blocks((ne1 + warps_per_block - 1) / warps_per_block, block_num_y, ne2 * ne3);
    const dim3 block_size(warps_per_block * QK_MXFP6_E2M3_SUB, 1, 1);
    if (ids) {
        if (scale_activation) {
            quantize_mmq_mxfp6_e2m3<true, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids, ids_expert, vy, scale_activation, scale_activation_ne, ne00, s01, s02, s03, ne0, ne1, ne2);
        } else {
            quantize_mmq_mxfp6_e2m3<true, false><<<num_blocks, block_size, 0, stream>>>(
                x, ids, ids_expert, vy, nullptr, 0, ne00, s01, s02, s03, ne0, ne1, ne2);
        }
    } else if (scale_activation) {
        quantize_mmq_mxfp6_e2m3<false, true><<<num_blocks, block_size, 0, stream>>>(
            x, ids, nullptr, vy, scale_activation, scale_activation_ne, ne00, s01, s02, s03, ne0, ne1, ne2);
    } else {
        quantize_mmq_mxfp6_e2m3<false, false><<<num_blocks, block_size, 0, stream>>>(
            x, ids, nullptr, vy, nullptr, 0, ne00, s01, s02, s03, ne0, ne1, ne2);
    }
}

void quantize_mmq_fp8_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    quantize_mmq_fp8_cuda(x, ids, nullptr, vy, type_src0, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, nullptr, 0, stream);
}

void quantize_row_fp8_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    quantize_row_fp8_cuda(x, ids, vy, type_src0, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, nullptr, 0, stream);
}

void quantize_row_fp8_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const float * scale_activation, const int64_t scale_activation_ne, cudaStream_t stream) {
    GGML_ASSERT(type_src0 == GGML_TYPE_NVFP4 || type_src0 == GGML_TYPE_MXFP6_E2M3);
    GGML_ASSERT(ne0 > 0);

    const int64_t block_num_y = (ne0 + QK_FP8 - 1) / QK_FP8;
    const dim3 num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3 block_size(4 * QK_FP8_SUB, 1, 1);

    if (ids) {
        if (scale_activation) {
            quantize_row_fp8_e4m3<true, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids, vy, scale_activation, scale_activation_ne, ne00, s01, s02, s03, ne0, ne1, ne2);
        } else {
            quantize_row_fp8_e4m3<true, false><<<num_blocks, block_size, 0, stream>>>(
                x, ids, vy, nullptr, 0, ne00, s01, s02, s03, ne0, ne1, ne2);
        }
    } else if (scale_activation) {
        quantize_row_fp8_e4m3<false, true><<<num_blocks, block_size, 0, stream>>>(
            x, ids, vy, scale_activation, scale_activation_ne, ne00, s01, s02, s03, ne0, ne1, ne2);
    } else {
        quantize_row_fp8_e4m3<false, false><<<num_blocks, block_size, 0, stream>>>(
            x, ids, vy, nullptr, 0, ne00, s01, s02, s03, ne0, ne1, ne2);
    }
}

void quantize_mmq_fp8_cuda(
        const float * x, const int32_t * ids, const int32_t * ids_expert, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const float * scale_activation, const int64_t scale_activation_ne, cudaStream_t stream) {
    GGML_ASSERT(type_src0 == GGML_TYPE_NVFP4 || type_src0 == GGML_TYPE_MXFP6_E2M3);
    GGML_ASSERT(ne0 > 0);

    const int64_t block_num_y = (ne0 + QK_FP8 - 1) / QK_FP8;
    const dim3 num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3 block_size(4 * QK_FP8_SUB, 1, 1);
    if (ids) {
        if (scale_activation) {
            quantize_mmq_fp8_e4m3<true, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids, ids_expert, vy, scale_activation, scale_activation_ne, ne00, s01, s02, s03, ne0, ne1, ne2);
        } else {
            quantize_mmq_fp8_e4m3<true, false><<<num_blocks, block_size, 0, stream>>>(
                x, ids, ids_expert, vy, nullptr, 0, ne00, s01, s02, s03, ne0, ne1, ne2);
        }
    } else if (scale_activation) {
        quantize_mmq_fp8_e4m3<false, true><<<num_blocks, block_size, 0, stream>>>(
            x, ids, nullptr, vy, scale_activation, scale_activation_ne, ne00, s01, s02, s03, ne0, ne1, ne2);
    } else {
        quantize_mmq_fp8_e4m3<false, false><<<num_blocks, block_size, 0, stream>>>(
            x, ids, nullptr, vy, nullptr, 0, ne00, s01, s02, s03, ne0, ne1, ne2);
    }
}

void quantize_mmq_fp4_cuda(const float *                    x,
                           const int32_t *                  ids,
                           void *                           vy,
                           [[maybe_unused]] const ggml_type type_src0,
                           const int64_t                    ne00,
                           const int64_t                    s01,
                           const int64_t                    s02,
                           const int64_t                    s03,
                           const int64_t                    ne0,
                           const int64_t                    ne1,
                           const int64_t                    ne2,
                           const int64_t                    ne3,
                           cudaStream_t                     stream) {
    GGML_ASSERT(ne0 % (2 * QK_MXFP4) == 0);

    constexpr int nwarps = 8;
    constexpr int vals_per_warp  = 2 * QK_MXFP4;
    constexpr int vals_per_block = nwarps * vals_per_warp;

    const int64_t block_num_y = (ne0 + vals_per_block - 1) / vals_per_block;
    const dim3    num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3    block_size(WARP_SIZE, nwarps, 1);

    quantize_mmq_mxfp4<<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}
