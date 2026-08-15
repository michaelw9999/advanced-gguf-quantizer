#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <cstdlib>
#include <cstring>

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP8:
            mul_mat_q_case<GGML_TYPE_MXFP8>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP6_E2M3:
            mul_mat_q_case<GGML_TYPE_MXFP6_E2M3>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static bool ggml_cuda_mxfp6_use_fp8_acts(const char * name) {
    const char * mode = getenv("GGML_CUDA_MXFP6_FP8_ACTS");
    if (mode == nullptr) {
        return false;
    }
    if (strcmp(mode, "0") == 0 || strcmp(mode, "none") == 0) {
        return false;
    }
    if (strcmp(mode, "1") == 0 || strcmp(mode, "all") == 0) {
        return true;
    }
    if (strcmp(mode, "attn") == 0) {
        return strstr(name, "attn_") != nullptr;
    }
    if (strcmp(mode, "ffn") == 0) {
        return strstr(name, "ffn_") != nullptr;
    }
    if (strcmp(mode, "ssm") == 0) {
        return strstr(name, "ssm_") != nullptr;
    }
    return strstr(name, mode) != nullptr;
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const bool src0_has_blackwell_input_scale = src0->type == GGML_TYPE_NVFP4;
    const ggml_tensor * scale_x_t = src0_has_blackwell_input_scale ? ggml_cuda_mul_mat_input_scale(dst) : nullptr;
    const float * scale_x_d = scale_x_t ? (const float *) scale_x_t->data : nullptr;
    const int64_t scale_x_ne = scale_x_t ? ggml_nelements(scale_x_t) : 0;
#if defined(BLACKWELL_MMA_AVAILABLE)
    const ggml_tensor * scale_x_src = src0_has_blackwell_input_scale ? src0->src[1] : nullptr;
    const bool scale_x_in_header = src0_has_blackwell_input_scale &&
        scale_x_src != nullptr && ggml_is_scalar(scale_x_src);
    const float * scale_x_q_d = scale_x_d != nullptr ? scale_x_d :
        scale_x_in_header ?
            &((const block_nvfp4_blackwell_tensor *) src0_d)->input_scale : nullptr;
    const int64_t scale_x_q_ne = scale_x_d != nullptr ? scale_x_ne :
        scale_x_in_header ? 1 : 0;
#else
    const float * scale_x_q_d = scale_x_d;
    const int64_t scale_x_q_ne = scale_x_ne;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (src0->type != GGML_TYPE_NVFP4 &&
            src0->type != GGML_TYPE_MXFP6_E2M3 &&
            ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;
    const bool use_stream_k_hw = (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                              || GGML_CUDA_CC_IS_CDNA(cc);
    // TODO: tighter pool buffer size vs q8 path
    const bool use_native_mxfp4 = blackwell_mma_available(cc) && src0->type == GGML_TYPE_MXFP4;
    const bool use_native_mxfp8 = blackwell_mma_available(cc) && src0->type == GGML_TYPE_MXFP8;
    const bool use_native_mxfp6 = blackwell_mma_available(cc) &&
        ggml_is_contiguous(src0) && src0->view_src == nullptr &&
        src0->type == GGML_TYPE_MXFP6_E2M3 &&
        ne00 % QK_MXFP6_E2M3 == 0;
#if defined(BLACKWELL_MMA_AVAILABLE)
    const bool use_native_nvfp4 = src0->type == GGML_TYPE_NVFP4;
#else
    const bool use_native_nvfp4 = false;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
    const bool use_stream_k = use_stream_k_hw;
    int64_t s01_mmq = s01;
    int64_t s02_mmq = s02;
    int64_t s03_mmq = s03;
#if defined(BLACKWELL_MMA_AVAILABLE)
    const bool use_nvfp4_layout = src0->type == GGML_TYPE_NVFP4;
    const bool use_mxfp6_layout = use_native_mxfp6;
    const bool use_mxfp8_layout = use_native_mxfp8;
    if (use_nvfp4_layout) {
        s01_mmq = ggml_cuda_nvfp4_blocks_per_row(ne00);
        s02_mmq = ggml_cuda_bw_div_up(ne01, 16) * s01_mmq;
        s03_mmq = (s03 / s02) * s02_mmq;
    } else if (use_mxfp6_layout) {
        s01_mmq = ggml_cuda_mxfp6_e2m3_blocks_per_row(ne00);
        s02_mmq = ggml_cuda_bw_div_up(ne01, 16) * s01_mmq;
        s03_mmq = (s03 / s02) * s02_mmq;
    } else if (use_mxfp8_layout) {
        s01_mmq = ggml_cuda_mxfp8_blocks_per_row(ne00);
        s02_mmq = ggml_cuda_mxfp8_tile_rows(ne01) * s01_mmq;
        s03_mmq = (s03 / s02) * s02_mmq;
    }
#endif // defined(BLACKWELL_MMA_AVAILABLE)
    if (!ids) {
        const bool use_fp8_acts = use_native_mxfp6 && ggml_cuda_mxfp6_use_fp8_acts(src0->name);
        const bool use_mxfp6_acts = use_native_mxfp6 && !use_fp8_acts;
        const size_t nbytes_src1_q8_1 = use_fp8_acts ?
            ne13*ne12 * ne11*ggml_cuda_fp8_blocks_per_row(ne10_padded) * sizeof(block_fp8) +
                get_mmq_x_max_host(cc)*sizeof(block_fp8) :
            use_mxfp6_acts ?
            ne13*ne12 * ne11*ggml_cuda_mxfp6_e2m3_blocks_per_row(ne10_padded) * sizeof(tile_mxfp6_e2m3_mmq) +
                get_mmq_x_max_host(cc)*sizeof(tile_mxfp6_e2m3_mmq) :
            ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1 +
                get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
	            if (use_native_mxfp4) {
	                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
	                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
	                                      ne11, ne12, ne13, stream);
	            } else if (use_native_mxfp8) {
	                static_assert(sizeof(block_mxfp8_mmq) == 4 * sizeof(block_q8_1));
	                quantize_mmq_mxfp8_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
	                                        ne10_padded, ne11, ne12, ne13, stream);
	            } else if (use_fp8_acts) {
		            quantize_mmq_fp8_cuda(src1_d, nullptr, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
		                                  ne10_padded, ne11, ne12, ne13, nullptr, 0, stream);
	            } else if (use_mxfp6_acts) {
		            quantize_mmq_mxfp6_e2m3_cuda(src1_d, nullptr, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
		                                         ne10_padded, ne11, ne12, ne13, scale_x_q_d, scale_x_q_ne, stream);
#if defined(BLACKWELL_MMA_AVAILABLE)
            } else if (use_native_nvfp4) {
                quantize_mmq_nvfp4_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, scale_x_q_d, scale_x_q_ne, stream);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_mxfp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) /
                                    (8 * QK_MXFP4 * sizeof(int))  // block_fp4_mmq holds 256 values (8 blocks of 32)
	                            : use_fp8_acts ?
	                                ne11 * ggml_cuda_fp8_blocks_per_row(ne10_padded) * sizeof(block_fp8) / sizeof(int)
	                            : use_mxfp6_acts ?
	                                ne11 * ggml_cuda_mxfp6_e2m3_blocks_per_row(ne10_padded) * sizeof(tile_mxfp6_e2m3_mmq) / sizeof(int)
#if defined(BLACKWELL_MMA_AVAILABLE)
	                            : use_native_nvfp4 ?
                                ne11 * ggml_cuda_nvfp4_blocks_per_row(ne10_padded) * sizeof(block_nvfp4_mmq) / sizeof(int)
#endif // defined(BLACKWELL_MMA_AVAILABLE)
                                :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            ne00, ne01, ne1, s01_mmq, ne11, s1,
            ne02, ne12, s02_mmq, s12, s2,
            ne03, ne13, s03_mmq, s13, s3,
            use_stream_k, use_native_mxfp6, use_fp8_acts, ne1};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_expert(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, stream, ids_expert.get());
        CUDA_CHECK(cudaGetLastError());
    }

    const bool use_fp8_acts = use_native_mxfp6 && ggml_cuda_mxfp6_use_fp8_acts(src0->name);
    const bool use_mxfp6_acts = use_native_mxfp6 && !use_fp8_acts;
    const size_t nbytes_src1_q8_1 = use_fp8_acts ?
        ne12*n_expert_used*ggml_cuda_fp8_blocks_per_row(ne10_padded) * sizeof(block_fp8) +
            get_mmq_x_max_host(cc)*sizeof(block_fp8) :
        use_mxfp6_acts ?
        ne12*n_expert_used*ggml_cuda_mxfp6_e2m3_blocks_per_row(ne10_padded) * sizeof(tile_mxfp6_e2m3_mmq) +
            get_mmq_x_max_host(cc)*sizeof(tile_mxfp6_e2m3_mmq) :
        ne12*n_expert_used*ne10_padded * sizeof(block_q8_1)/QK8_1 +
            get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

	        if (use_native_mxfp4) {
	            quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
	                                  ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
	        } else if (use_native_mxfp8) {
	            quantize_mmq_mxfp8_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
	                                    ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
	        } else if (use_fp8_acts) {
	            quantize_mmq_fp8_cuda(src1_d, ids_src1.get(), ids_expert.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
	                                  ne10_padded, ne11_flat, ne12_flat, ne13_flat, nullptr, 0, stream);
	        } else if (use_mxfp6_acts) {
	            quantize_mmq_mxfp6_e2m3_cuda(src1_d, ids_src1.get(), ids_expert.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
	                                         ne10_padded, ne11_flat, ne12_flat, ne13_flat, scale_x_q_d, scale_x_q_ne, stream);
#if defined(BLACKWELL_MMA_AVAILABLE)
        } else if (use_native_nvfp4) {
            quantize_mmq_nvfp4_cuda(src1_d, ids_src1.get(), ids_expert.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                    ne10_padded, ne11_flat, ne12_flat, ne13_flat, scale_x_q_d, scale_x_q_ne, stream);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t s12 = use_native_mxfp4 ?
                            ne11 * ne10_padded * sizeof(block_fp4_mmq) / (8 * QK_MXFP4 * sizeof(int)) :
                        use_fp8_acts ?
                            ne11_flat * ggml_cuda_fp8_blocks_per_row(ne10_padded) * sizeof(block_fp8) / sizeof(int) :
                        use_mxfp6_acts ?
                            ne11_flat * ggml_cuda_mxfp6_e2m3_blocks_per_row(ne10_padded) * sizeof(tile_mxfp6_e2m3_mmq) / sizeof(int) :
#if defined(BLACKWELL_MMA_AVAILABLE)
                        use_native_nvfp4 ?
                            ne11 * ggml_cuda_nvfp4_blocks_per_row(ne10_padded) * sizeof(block_nvfp4_mmq) / sizeof(int) :
#endif // defined(BLACKWELL_MMA_AVAILABLE)
                            ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        ne00, ne01, ne_get_rows, s01_mmq, ne_get_rows, s1,
        ne02, ne02, s02_mmq, s12, s2,
        ne03, ne13, s03_mmq, s13, s3,
        use_stream_k, use_native_mxfp6, use_fp8_acts, ne12};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
}

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
#if defined(BLACKWELL_MMA_AVAILABLE)
    const bool use_nvfp4_layout = src0->type == GGML_TYPE_NVFP4;
    const bool use_mxfp6_layout = ggml_is_contiguous(src0) && src0->view_src == nullptr &&
        src0->type == GGML_TYPE_MXFP6_E2M3 && ne00 % QK_MXFP6_E2M3 == 0 &&
        blackwell_mma_available(cc);
#else
    const bool use_nvfp4_layout = false;
    const bool use_mxfp6_layout = false;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t stride01 = use_nvfp4_layout ? ggml_cuda_nvfp4_blocks_per_row(ne00) :
        use_mxfp6_layout ? ggml_cuda_mxfp6_e2m3_blocks_per_row(ne00) : ne00 / ggml_blck_size(src0->type);
#else
    const int64_t stride01 = ne00 / ggml_blck_size(src0->type);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    // The stream-k decomposition is only faster for recent NVIDIA GPUs.
    // Also its fixup needs to allocate a temporary buffer in the memory pool.
    // There are multiple parallel CUDA streams for src1_ncols != ne11 which would introduce a race condition for this buffer.
    const bool use_stream_k = ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc))
                            && src1_ncols == ne11;
    const mmq_args args = {
        src0_dd_i, src0->type, (const int *) src1_ddq_i, nullptr, nullptr, dst_dd_i,
        ne00, row_diff, src1_ncols, stride01, ne11, nrows_dst,
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        use_stream_k, use_mxfp6_layout, false, src1_ncols};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_padded_row_size);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    if (type == GGML_TYPE_MXFP8) {
        return blackwell_mma_available(cc);
    }

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_MXFP6_E2M3:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
