// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstdint>

#include "../utils/kernel_generator.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "primitive_inst.h"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

// The KV cache update stage: the kernel that stores new K/V rows in the paged KV cache, and what it needs to run.
// Shared by the paged_attention and paged_kv_cache_write primitives.
namespace ov::intel_gpu::ocl {

enum class PagedAttentionStage : uint8_t { GENERATE = 0, PREFILL = 1, MIXED = 2, UNKNOWN = 3 };

// Inputs of the primitive that the stage reads
struct KVCacheUpdateInputs {
    size_t key;
    size_t value;
    size_t key_cache;
    size_t value_cache;
    size_t past_lens;
    size_t subsequence_begins;
    size_t block_indices;
    size_t block_indices_begins;
};

struct KVCacheUpdateDesc {
    KVCacheUpdateInputs inputs;
    size_t k_head_size;
    size_t v_head_size;
    size_t kv_heads_num;
    bool is_key_by_channel;
};

// For a paged_attention or a paged_kv_cache_write primitive
KVCacheUpdateDesc get_kv_cache_update_desc(const kernel_impl_params& params);

// Runtime parameters of the primitive must derive from this
struct KVCacheUpdateRuntimeParams : public ImplRuntimeParams {
    PagedAttentionStage stage = PagedAttentionStage::UNKNOWN;
    size_t paged_attention_aligned_seq_len = 0;
};

// rows_input_idx: an input with one row per new token
PagedAttentionStage get_kv_cache_update_stage(const kernel_impl_params& params, size_t rows_input_idx, size_t past_lens_input_idx);

// Total number of new rows, with the rows of each subsequence aligned to target_seq_len_block_size
int64_t get_kv_cache_update_aligned_seq_len(const kernel_impl_params& params,
                                            const KVCacheUpdateInputs& inputs,
                                            const PagedAttentionStage& stage,
                                            int64_t target_seq_len_block_size = 16);

size_t get_num_k_head_size_partitions(bool is_key_by_channel, int64_t k_head_size);

// Layout of a u4 KV cache
JitConstants make_uint4_kv_cache_layout_jit_constants(const kernel_impl_params& params, const KVCacheUpdateDesc& desc);

// Uses internal buffers 0, 1, 2 (see add_kv_cache_update_blocks) in the prefill and mixed stages
class KVCacheUpdateGenerator : public KernelGenerator {
public:
    KVCacheUpdateGenerator() : KernelGenerator("pa_kv_cache_update_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
};

using KVCacheUpdateBufferLock = mem_lock<int32_t, mem_lock_type::write>;

// Splits rows [seq_start, seq_end) of a subsequence, stored from position past_len on, at the KV cache block borders.
// Adds one entry per block to the three block distribution buffers, from entry `index` on: the first row, the row
// after the last one, and the subsequence. Returns the next free entry.
inline size_t add_kv_cache_update_blocks(int32_t subsequence,
                                         int32_t past_len,
                                         int32_t seq_start,
                                         int32_t seq_end,
                                         size_t index,
                                         KVCacheUpdateBufferLock& blocks_indexes_start,
                                         KVCacheUpdateBufferLock& blocks_indexes_end,
                                         KVCacheUpdateBufferLock& blocked_gws_subseq_mapping) {
    const auto pa_block_size = static_cast<int32_t>(paged_attention::block_size);
    const auto seq_length = seq_end - seq_start;

    int32_t j = 0;
    if (past_len != 0) {
        auto block_start_pos = seq_start;
        auto empty_slots = pa_block_size - (past_len % pa_block_size);
        auto block_end_pos = seq_start + std::min(empty_slots, seq_length);

        blocks_indexes_start[index] = block_start_pos;
        blocks_indexes_end[index] = block_end_pos;
        blocked_gws_subseq_mapping[index] = subsequence;

        index++;

        auto added_slots = block_end_pos - block_start_pos;
        j += added_slots;
    }

    for (; j < seq_length; j += pa_block_size) {
        auto block_start_pos = seq_start + j;
        auto block_end_pos = std::min(block_start_pos + pa_block_size, seq_end);

        blocks_indexes_start[index] = block_start_pos;
        blocks_indexes_end[index] = block_end_pos;
        blocked_gws_subseq_mapping[index] = subsequence;

        index++;
    }

    return index;
}

// Fills internal buffers 0, 1, 2 of the instance for the prefill and mixed stages
void fill_kv_cache_update_buffers(primitive_inst& instance, const KVCacheUpdateInputs& inputs);

}  // namespace ov::intel_gpu::ocl
