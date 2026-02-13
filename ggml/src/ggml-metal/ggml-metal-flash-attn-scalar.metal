// Flash Attention SCALAR fallback for GPUs without simdgroup matrix multiply
// Algorithm from Apple MLX sdpa_vector.h, adapted for ggml tensor layout
// Runs on any Metal GPU with simdgroup_reduction (AMD, older Apple, etc.)
//
// Thread decomposition: BN simdgroups × BD threads
//   - Each simdgroup processes one KV position at a time, stepping by BN
//   - Each thread handles DK/BD elements of Q/K and DV/BD elements of V
//   - QK dot product reduced via simd_sum within each simdgroup
//   - Cross-simdgroup reduction via shared memory after all KV processed

#define SCALAR_BN 4    // simdgroups per threadgroup
#define SCALAR_BD 32   // threads per simdgroup (fixed by Metal)

template<int DK, int DV>
inline void flash_attn_ext_scalar_impl(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    uint    simd_gid [[simdgroup_index_in_threadgroup]],
    uint    simd_lid [[thread_index_in_simdgroup]])
{
    constexpr int BN = SCALAR_BN;
    constexpr int BD = SCALAR_BD;
    constexpr int qk_per_thread = DK / BD;
    constexpr int v_per_thread  = DV / BD;

    const int iq1 = tgpig.x;  // query position
    const int iq2 = tgpig.y;  // Q head index
    const int iq3 = tgpig.z;  // batch index

    if (iq1 >= args.ne01) return;

    const int KV = args.ne11;

    // GQA: map Q head → K/V head
    const int ik2 = iq2 / (args.ne02 / args.ne_12_2);

    // Q pointer (f32, contiguous within row)
    device const float * q_ptr = (device const float *)(q + iq1 * args.nb01 + iq2 * args.nb02 + iq3 * args.nb03);

    // K/V base pointers (f16) — offset by KV head and batch
    device const char * k_head = k + ik2 * args.nb12 + iq3 * args.nb13;
    device const char * v_head = v + ik2 * args.nb22 + iq3 * args.nb23;

    // Mask pointer (f16) — offset by query position, head, batch
    const bool has_mask = args.ne31 > 0;
    device const half * mask_row = nullptr;
    if (has_mask) {
        mask_row = (device const half *)(mask
            + iq1 * args.nb31
            + (iq2 % args.ne32) * args.nb32
            + (iq3 % args.ne33) * args.nb33);
    }

    // ALiBi slope
    float slope = 1.0f;
    if (args.max_bias > 0.0f) {
        const short h = iq2;
        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;
        slope = pow(base, float(exph));
    }

    const bool has_softcap = args.logit_softcap != 0.0f;

    // Load Q into registers (pre-scaled; args.scale already divided by softcap if needed)
    float q_reg[qk_per_thread];
    for (int j = 0; j < qk_per_thread; j++) {
        q_reg[j] = q_ptr[simd_lid * qk_per_thread + j] * args.scale;
    }

    // Per-thread accumulators
    float o[v_per_thread];
    for (int j = 0; j < v_per_thread; j++) {
        o[j] = 0;
    }
    float max_score = -__FLT_MAX__ / 2.0f;
    float sum_exp   = 0;

    // Main KV loop: each simdgroup steps through KV by BN
    for (int i = simd_gid; i < KV; i += BN) {
        device const half * k_ptr = (device const half *)(k_head + (uint64_t)i * args.nb11);

        // QK dot product: partial per thread, then simd_sum
        float score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
            score += q_reg[j] * float(k_ptr[simd_lid * qk_per_thread + j]);
        }
        score = simd_sum(score);

        // Logit softcap: tanh(score/softcap) * softcap (score already divided by softcap via prescale)
        if (has_softcap) {
            score = args.logit_softcap * precise::tanh(score);
        }

        // Apply additive mask with ALiBi slope. -INF entries naturally → exp(-INF) = 0
        if (has_mask) {
            score += slope * float(mask_row[i]);
        }

        // Online softmax update (all lanes have same score after simd_sum)
        float new_max   = max(max_score, score);
        float factor    = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp   = sum_exp * factor + exp_score;

        // Fused V accumulation: O = O * factor + exp_score * V
        device const half * v_ptr = (device const half *)(v_head + (uint64_t)i * args.nb21);
        for (int j = 0; j < v_per_thread; j++) {
            o[j] = o[j] * factor + exp_score * float(v_ptr[simd_lid * v_per_thread + j]);
        }
    }

    // --- Cross-simdgroup reduction ---
    // Shared memory layout: [BN max] [BN sum] [BN*BD output scratch]
    threadgroup float * sg_max = shmem;
    threadgroup float * sg_sum = shmem + BN;
    threadgroup float * sg_out = shmem + 2 * BN;

    if (simd_lid == 0) {
        sg_max[simd_gid] = max_score;
        sg_sum[simd_gid] = sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Global max across all simdgroups
    float global_max = sg_max[0];
    for (int g = 1; g < BN; g++) {
        global_max = max(global_max, sg_max[g]);
    }

    // Global sum_exp, rescaled to global max
    float global_sum_exp = 0;
    for (int g = 0; g < BN; g++) {
        global_sum_exp += sg_sum[g] * fast::exp(sg_max[g] - global_max);
    }

    // Rescale this simdgroup's partial output to global max
    float rescale = fast::exp(max_score - global_max);
    for (int j = 0; j < v_per_thread; j++) {
        o[j] *= rescale;
    }

    // Sum output across simdgroups via shared memory
    for (int j = 0; j < v_per_thread; j++) {
        sg_out[simd_gid * BD + simd_lid] = o[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_gid == 0) {
            float total = 0;
            for (int g = 0; g < BN; g++) {
                total += sg_out[g * BD + simd_lid];
            }
            o[j] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Attention sinks: fold pre-computed per-head sink value into softmax state
    const bool has_sinks = args.ns10 > 0;
    if (has_sinks && simd_gid == 0) {
        const float s = simd_lid == 0 ? ((device const float *) sinks)[iq2] : -__FLT_MAX__ / 2.0f;
        const float sink_val = simd_max(max(global_max, s));

        const float ms = fast::exp(global_max - sink_val);
        const float vs = fast::exp(s - sink_val);

        global_sum_exp = global_sum_exp * ms + simd_sum(vs);

        for (int j = 0; j < v_per_thread; j++) {
            o[j] *= ms;
        }
    }

    // Final normalization
    if (simd_gid == 0) {
        for (int j = 0; j < v_per_thread; j++) {
            o[j] = (global_sum_exp > 0) ? (o[j] / global_sum_exp) : 0;
        }
    }

    // Write output (only simdgroup 0)
    // dst layout: [DV, ne1(heads), ne2(N), ne3(batch)] contiguous f32
    if (simd_gid == 0) {
        device float * dst_ptr = (device float *)dst;
        const int dst_base = iq3 * args.ne2 * args.ne1 * DV
                           + iq1 * args.ne1 * DV
                           + iq2 * DV;
        for (int j = 0; j < v_per_thread; j++) {
            dst_ptr[dst_base + simd_lid * v_per_thread + j] = o[j];
        }
    }
}

// Kernel wrappers for common head sizes
[[host_name("kernel_flash_attn_ext_scalar_dk64_dv64")]]
kernel void kernel_flash_attn_ext_scalar_dk64_dv64(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    uint    simd_gid [[simdgroup_index_in_threadgroup]],
    uint    simd_lid [[thread_index_in_simdgroup]])
{
    flash_attn_ext_scalar_impl<64, 64>(args, q, k, v, mask, dst, sinks, shmem, tgpig, simd_gid, simd_lid);
}

[[host_name("kernel_flash_attn_ext_scalar_dk128_dv128")]]
kernel void kernel_flash_attn_ext_scalar_dk128_dv128(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    uint    simd_gid [[simdgroup_index_in_threadgroup]],
    uint    simd_lid [[thread_index_in_simdgroup]])
{
    flash_attn_ext_scalar_impl<128, 128>(args, q, k, v, mask, dst, sinks, shmem, tgpig, simd_gid, simd_lid);
}

[[host_name("kernel_flash_attn_ext_scalar_dk256_dv256")]]
kernel void kernel_flash_attn_ext_scalar_dk256_dv256(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q     [[buffer(1)]],
    device const char * k     [[buffer(2)]],
    device const char * v     [[buffer(3)]],
    device const char * mask  [[buffer(4)]],
    device       char * dst   [[buffer(5)]],
    device const char * sinks [[buffer(6)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig    [[threadgroup_position_in_grid]],
    uint    simd_gid [[simdgroup_index_in_threadgroup]],
    uint    simd_lid [[thread_index_in_simdgroup]])
{
    flash_attn_ext_scalar_impl<256, 256>(args, q, k, v, mask, dst, sinks, shmem, tgpig, simd_gid, simd_lid);
}
