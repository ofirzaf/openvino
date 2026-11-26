// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_kv_cache_update.hpp"

#include <array>
#include <cassert>

#include "../utils/jitter.hpp"
#include "common_tools.h"
#include "common_utils/jitter.hpp"
#include "intel_gpu/primitives/paged_kv_cache_write.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "openvino/runtime/internal_properties.hpp"

namespace ov::intel_gpu::ocl {
namespace {

constexpr size_t paged_attention_block_size = 16;
constexpr size_t subgroup_size = 16;
constexpr size_t u4_elems_per_byte = 2;

inline size_t get_generate_stage_block_size(size_t head_size) {
    auto preferred_block_size = {4, 2, 1};
    for (const auto& block_size : preferred_block_size) {
        if (head_size % (block_size * subgroup_size) == 0) {
            return block_size;
        }
    }
    return 1;
}

}  // namespace

KVCacheUpdateDesc get_kv_cache_update_desc(const kernel_impl_params& params) {
    if (params.is_type<paged_kv_cache_write>()) {
        using Idx = paged_kv_cache_write::InputIdx;
        const auto desc = params.typed_desc<paged_kv_cache_write>();
        return {{Idx::KEY, Idx::VALUE, Idx::KEY_CACHE, Idx::VALUE_CACHE, Idx::PAST_LENS, Idx::SUBSEQUENCE_BEGINS, Idx::BLOCK_INDICES, Idx::BLOCK_INDICES_BEGINS},
                desc->k_head_size,
                desc->v_head_size,
                desc->kv_heads_num,
                desc->is_key_by_channel};
    }

    using Idx = paged_attention::PagedAttentionInputIdx;
    const auto desc = params.typed_desc<paged_attention>();
    return {{Idx::KEY, Idx::VALUE, Idx::KEY_CACHE, Idx::VALUE_CACHE, Idx::PAST_LENS, Idx::SUBSEQUENCE_BEGINS, Idx::BLOCK_INDICES, Idx::BLOCK_INDICES_BEGINS},
            desc->k_head_size,
            desc->v_head_size,
            desc->kv_heads_num,
            desc->is_key_by_channel};
}

PagedAttentionStage get_kv_cache_update_stage(const kernel_impl_params& impl_param, size_t rows_input_idx, size_t past_lens_input_idx) {
    const auto& rows_shape = impl_param.get_input_layout(rows_input_idx).get_partial_shape();
    const auto& past_lens_shape = impl_param.get_input_layout(past_lens_input_idx).get_partial_shape();

    if (rows_shape.is_static() && past_lens_shape.is_static()) {
        if (rows_shape[0].get_length() == past_lens_shape[0].get_length()) {
            return PagedAttentionStage::GENERATE;
        }

        const auto& memory_deps = impl_param.memory_deps;
        const auto past_lens_mem = memory_deps.at(past_lens_input_idx);
        mem_lock<int32_t, mem_lock_type::read> past_lens_mem_lock(past_lens_mem, *impl_param.strm);

        const auto past_lens_size = past_lens_mem_lock.size();
        for (size_t i = 0; i < past_lens_size; i++) {
            if (past_lens_mem_lock[i] != 0) {
                return PagedAttentionStage::MIXED;
            }
        }
        return PagedAttentionStage::PREFILL;
    }
    return PagedAttentionStage::UNKNOWN;
}

int64_t get_kv_cache_update_aligned_seq_len(const kernel_impl_params& impl_param,
                                            const KVCacheUpdateInputs& inputs,
                                            const PagedAttentionStage& stage,
                                            int64_t target_seq_len_block_size,
                                            bool block_table_fits_rows) {
    // Since at prefill stage Q, K, V inputs may contain multiple sequences with arbitrary
    // target sequence lengths each (shape is [sequences_num * target_seq_len, num_heads * head_size]),
    // to apply blocking to the first dimension (target_seq_len of each sequence), we need to calculate aligned total
    // target sequence length for proper kernel dispatching
    // For instance, if input contains two sequences with 35 and 28 sequence lengths each,
    // the Q, K, V inputs at prefill stage will have shapes [35 + 28, num_heads * head_size]; considering kernel's
    // target_seq_len_block_size equals 16, we need to launch kernel instances for the following ranges:
    // [0, 15], [16, 31], [32, 34], [35, 50], [51, 62], so aligned target_seq_len_block_size should be 5 * 16 = 80,
    // and 5 kernels instances should be launched (for each range, some of them containing leftovers)
    //
    // In general, to obtain length for each sequence, we have to parse subsequence_begins input,
    // which contains begin and end indexes for each sequence (for above example it will contain three values: {0, 35, 63})
    // However, as long as kernel's target_seq_len_block_size matches with vLLM's block_size,
    // we can reuse block_indices_shape[0] size to determine total aligned sequences length size, avoiding
    // memory access at runtime, because vLLM internally uses similar logic to configure blocks for KV cache

    auto calculate_aligned_seq_len = [&]() {
        const auto& input_mem = impl_param.memory_deps;
        const auto subsequence_begins_mem = input_mem.at(inputs.subsequence_begins);
        mem_lock<int32_t, mem_lock_type::read> subsequence_begins_mem_lock(subsequence_begins_mem, *impl_param.strm);

        int64_t aligned_seq_len = 0;
        if (stage == PagedAttentionStage::MIXED) {
            const auto past_lens_mem = input_mem.at(inputs.past_lens);
            mem_lock<int32_t, mem_lock_type::read> past_lens_mem_lock(past_lens_mem, *impl_param.strm);

            for (size_t i = 0; i < subsequence_begins_mem_lock.size() - 1; i++) {
                auto past_len = past_lens_mem_lock[i];
                int64_t seq_length = subsequence_begins_mem_lock[i + 1] - subsequence_begins_mem_lock[i];

                // Since in MIXED execution mode the present KV-cache can be appended to the past KV-cache at any offset inside block,
                // to ensure proper alignment and update_kv_cache kernel scheduling, we need to account for the number of unaligned tokens
                // in the first block
                // For example, if we need to store values in the following slots:
                //
                // block0: |O|O|O|O|O|O|O|O|O|O|O|O|U|U|U|U|
                // block1: |U|U|U|U|U|U|U|U|U|U|U|U|U|U|U|U|
                // block2: |U|U|U|U|U|U|E|E|E|E|E|E|E|E|E|E|
                // Where O - occupied slots, U - currently beeing updated slots, E - empty slots
                //
                // We need to schedule 3 update_kv_cache operations:
                // - For ranges of block0: [12-15]
                // - For ranges of block1: [0-15]
                // - For ranges of block2: [0-5]
                //
                // Therefore, consider an additional increment of aligned_seq_len to properly process all the blocks

                auto occupied_slots_num = past_len % target_seq_len_block_size;
                if (past_len != 0 && seq_length + occupied_slots_num > target_seq_len_block_size) {
                    aligned_seq_len += target_seq_len_block_size;
                    seq_length -= target_seq_len_block_size - occupied_slots_num;
                }

                aligned_seq_len += align_to(seq_length, target_seq_len_block_size);
            }
        } else {
            for (size_t i = 0; i < subsequence_begins_mem_lock.size() - 1; i++) {
                auto prompt_length = subsequence_begins_mem_lock[i + 1] - subsequence_begins_mem_lock[i];
                aligned_seq_len += align_to(prompt_length, target_seq_len_block_size);
            }
        }

        return aligned_seq_len;
    };

    int64_t aligned_seq_len = 0;
    if (stage == PagedAttentionStage::PREFILL) {
        if (block_table_fits_rows && static_cast<int64_t>(paged_attention::block_size) == target_seq_len_block_size) {
            const auto& block_indices_ps = impl_param.get_input_layout(inputs.block_indices).get_partial_shape();

            aligned_seq_len = block_indices_ps[0].get_length() * target_seq_len_block_size;
        } else {
            aligned_seq_len = calculate_aligned_seq_len();
        }
    } else {
        aligned_seq_len = calculate_aligned_seq_len();
    }

    return aligned_seq_len;
}

size_t get_num_k_head_size_partitions(bool is_key_by_channel, int64_t k_head_size) {
    size_t head_size_partition = 1;
    if (is_key_by_channel) {
        if (k_head_size % subgroup_size == 0) {
            head_size_partition = std::max(static_cast<size_t>(1), static_cast<size_t>(k_head_size / subgroup_size));
            head_size_partition = std::min(static_cast<size_t>(16), head_size_partition);
        }
    }
    return head_size_partition;
}

JitConstants make_uint4_kv_cache_layout_jit_constants(const kernel_impl_params& params, const KVCacheUpdateDesc& desc) {
    ov::intel_gpu::JitConstants jit;
    const auto kv_cache_dt = params.get_program().get_config().get_kv_cache_precision();
    const auto& kv_dt = params.input_layouts[desc.inputs.key].data_type;

    if (data_type_traits::is_i4_u4(kv_cache_dt)) {
        const auto scales_zp_size = ov::element::Type(kv_dt).size() * 2;  // fp16 scale + fp16 zp = 4 bytes
        jit.make("IS_INT4_COMPRESSED", true);
        if (desc.is_key_by_channel) {
            // K BY_CHANNEL: head_size is NOT packed (outer dim), block_size IS packed (inner dim)
            jit.make("PACKED_K_HEAD_SIZE", desc.k_head_size);
            jit.make("PACKED_ADJUSTED_K_HEAD_SIZE", desc.k_head_size);
            jit.make("PACKED_K_BLOCK_SIZE", paged_attention_block_size / u4_elems_per_byte);
        } else {
            // K BY_TOKEN: head_size IS packed (inner dim), block_size is NOT packed
            const auto packed_k_head_size = kernel_selector::Align(desc.k_head_size / u4_elems_per_byte, subgroup_size);
            jit.make("PACKED_K_HEAD_SIZE", packed_k_head_size);
            jit.make("PACKED_ADJUSTED_K_HEAD_SIZE", packed_k_head_size + scales_zp_size);
            jit.make("PACKED_K_BLOCK_SIZE", paged_attention_block_size);
        }
        // V per-token: head_size IS packed (inner dim)
        jit.make("PACKED_V_HEAD_SIZE", (kernel_selector::Align(desc.v_head_size / u4_elems_per_byte, subgroup_size)));
        jit.make("PACKED_ADJUSTED_V_HEAD_SIZE", (kernel_selector::Align(desc.v_head_size / u4_elems_per_byte, subgroup_size)) + scales_zp_size);

        // Dual-nibble V-cache optimization: each lane reads 1 packed byte (2 nibbles) via BLOCK_READN
        // Only applicable when PACKED_V_HEAD_SIZE is divisible by SUBGROUP_SIZE (no padding lanes needed)
        if (desc.v_head_size / u4_elems_per_byte % subgroup_size == 0) {
            jit.make("USE_DUAL_NIBBLE_V_OPT", 1);
        }
    } else {
        jit.make("IS_INT4_COMPRESSED", false);
    }

    return jit;
}

JitConstants KVCacheUpdateGenerator::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = make_base_jit_constants(params);

    const auto desc = get_kv_cache_update_desc(params);
    const auto& in_offsets_map = params.in_port_to_shape_info_offset;

    const std::array input_ids = {desc.inputs.key,
                                  desc.inputs.value,
                                  desc.inputs.past_lens,
                                  desc.inputs.block_indices,
                                  desc.inputs.block_indices_begins,
                                  desc.inputs.subsequence_begins};

    for (size_t i = 0; i < input_ids.size(); i++) {
        const size_t tensor_id = input_ids.at(i);
        jit.add(make_layout_jit_constants("INPUT" + to_code_string(i), params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
    }

    const size_t key_cache_id = desc.inputs.key_cache;
    const size_t value_cache_id = desc.inputs.value_cache;

    jit.add(make_layout_jit_constants("OUTPUT", params.input_layouts[key_cache_id], in_offsets_map.at(key_cache_id)));
    jit.add(make_layout_jit_constants("OUTPUT" + to_code_string(1), params.input_layouts[value_cache_id], in_offsets_map.at(value_cache_id)));

    const auto kv_cache_dt = params.get_program().get_config().get_kv_cache_precision();
    const auto is_key_by_channel = desc.is_key_by_channel;
    OPENVINO_ASSERT(is_key_by_channel == (params.get_program().get_config().get_key_cache_quant_mode() == ov::internal::CacheQuantMode::BY_CHANNEL),
                    "[GPU] Paged Attention key cache quantization mode mismatch: prim.key_cache_by_channel : ",
                    is_key_by_channel,
                    " and exec_config : ",
                    params.get_program().get_config().get_key_cache_quant_mode());

    jit.make("U4_ELEMS_PER_BYTE", u4_elems_per_byte);
    jit.make("K_HEAD_SIZE", desc.k_head_size);
    jit.make("V_HEAD_SIZE", desc.v_head_size);
    jit.make("KV_HEADS_NUM", desc.kv_heads_num);
    jit.make("PAGED_ATTENTION_BLOCK_SIZE", paged_attention_block_size);
    jit.make("SUBGROUP_SIZE", subgroup_size);
    jit.make("GENERATE_STAGE_K_BLOCK_SIZE", get_generate_stage_block_size(desc.k_head_size));
    jit.make("GENERATE_STAGE_V_BLOCK_SIZE", get_generate_stage_block_size(desc.v_head_size));

    const auto& key_cache_layout = params.input_layouts[key_cache_id];
    const bool is_kv_compressed = data_type_traits::is_i8_u8(key_cache_layout.data_type) || data_type_traits::is_i4_u4(key_cache_layout.data_type);
    jit.make("IS_KV_COMPRESSED", is_kv_compressed ? 1 : 0);
    if (is_kv_compressed) {
        auto data_type = params.input_layouts[desc.inputs.key].data_type;  // key tensor data size
        auto scales_zp_size = ov::element::Type(data_type).size() * 2;     // scale + zp
        if (data_type_traits::is_i4_u4(kv_cache_dt)) {
            jit.make("SCALE_ZP_SIZE_PER_TOKEN", scales_zp_size);
            jit.add(make_uint4_kv_cache_layout_jit_constants(params, desc));
        }

        if (data_type_traits::is_i4_u4(kv_cache_dt)) {
            if (is_key_by_channel) {
                // INT4 BY_CHANNEL: K dim order {0,1,3,2}, block_size packed in innermost dim
                jit.make("IS_KEY_BY_CHANNEL", 1);
                jit.make("ADJUSTED_K_HEAD_SIZE", desc.k_head_size);
                jit.make("ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE", paged_attention_block_size / u4_elems_per_byte + scales_zp_size);
            } else {
                // INT4 BY_TOKEN: K dim order {0,1,2,3}, head_size packed
                jit.make("ADJUSTED_K_HEAD_SIZE", desc.k_head_size + scales_zp_size);
                jit.make("ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE", paged_attention_block_size);
            }
            jit.make("NUM_K_HEAD_SIZE_PARTITIONS", get_num_k_head_size_partitions(desc.is_key_by_channel, desc.k_head_size));
        } else if (is_key_by_channel) {
            jit.make("IS_KEY_BY_CHANNEL", 1);
            jit.make("ADJUSTED_K_HEAD_SIZE", desc.k_head_size);
            jit.make("ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE", paged_attention_block_size + scales_zp_size);
            jit.make("NUM_K_HEAD_SIZE_PARTITIONS", get_num_k_head_size_partitions(desc.is_key_by_channel, desc.k_head_size));
        } else {
            jit.make("ADJUSTED_K_HEAD_SIZE", desc.k_head_size + scales_zp_size);
            jit.make("ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE", paged_attention_block_size);
        }
        jit.make("ADJUSTED_V_HEAD_SIZE", desc.v_head_size + scales_zp_size);
    } else {
        jit.make("ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE", paged_attention_block_size);
        jit.make("ADJUSTED_K_HEAD_SIZE", desc.k_head_size);
        jit.make("ADJUSTED_V_HEAD_SIZE", desc.v_head_size);
    }

    return jit;
}

Arguments KVCacheUpdateGenerator::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;
    const auto inputs = get_kv_cache_update_desc(params).inputs;

    if (params.is_dynamic()) {
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
    }

    // Inputs
    args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(inputs.key)});
    args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(inputs.value)});
    args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(inputs.past_lens)});
    args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(inputs.block_indices)});
    args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(inputs.block_indices_begins)});
    args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(inputs.subsequence_begins)});

    // Outputs
    args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(inputs.key_cache)});
    args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(inputs.value_cache)});

    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});

    return args;
}

DispatchDataFunc KVCacheUpdateGenerator::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        assert(!params.is_dynamic());
        auto& wgs = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        scalars.resize(1);

        const auto desc = get_kv_cache_update_desc(params);
        auto* rtp = static_cast<KVCacheUpdateRuntimeParams*>(rt_params);

        const auto is_prefill = rtp->stage == PagedAttentionStage::PREFILL || rtp->stage == PagedAttentionStage::MIXED;
        auto heads_number = desc.kv_heads_num;

        if (is_prefill) {
            const auto blocks_number = rtp->paged_attention_aligned_seq_len / paged_attention_block_size;

            wgs.global = {blocks_number, heads_number, subgroup_size};
            wgs.local = {1, 1, subgroup_size};
        } else {
            const auto& key_input = params.input_layouts[desc.inputs.key];
            const auto sequences_number = key_input.get_partial_shape()[0].get_length();
            size_t head_size_partition = get_num_k_head_size_partitions(desc.is_key_by_channel, desc.k_head_size);
            wgs.global = {static_cast<size_t>(sequences_number), heads_number, subgroup_size * head_size_partition};
            wgs.local = {1, 1, subgroup_size};
        }

        scalars[0].t = ScalarDescriptor::Types::UINT32;
        scalars[0].v.u32 = static_cast<uint32_t>(is_prefill);
    }};
}

void fill_kv_cache_update_buffers(primitive_inst& instance, const KVCacheUpdateInputs& inputs) {
    auto& stream = instance.get_network().get_stream();
    const auto& intermediates_memories = instance.get_intermediates_memories();
    OPENVINO_ASSERT(intermediates_memories.size() >= 3, "[GPU] Unexpected number of intermediates buffers for the KV cache update");

    mem_lock<int32_t, mem_lock_type::read> past_lens_mem_lock(instance.input_memory_ptr(inputs.past_lens), stream);
    mem_lock<int32_t, mem_lock_type::read> subsequence_begins_mem_lock(instance.input_memory_ptr(inputs.subsequence_begins), stream);
    KVCacheUpdateBufferLock blocks_indexes_start_lock(intermediates_memories[0], stream);
    KVCacheUpdateBufferLock blocks_indexes_end_lock(intermediates_memories[1], stream);
    KVCacheUpdateBufferLock blocked_gws_subseq_mapping_mem_lock(intermediates_memories[2], stream);

    size_t index = 0;
    for (size_t i = 0; i < subsequence_begins_mem_lock.size() - 1; i++) {
        index = add_kv_cache_update_blocks(static_cast<int32_t>(i),
                                           past_lens_mem_lock[i],
                                           subsequence_begins_mem_lock[i],
                                           subsequence_begins_mem_lock[i + 1],
                                           index,
                                           blocks_indexes_start_lock,
                                           blocks_indexes_end_lock,
                                           blocked_gws_subseq_mapping_mem_lock);
    }
}

}  // namespace ov::intel_gpu::ocl
