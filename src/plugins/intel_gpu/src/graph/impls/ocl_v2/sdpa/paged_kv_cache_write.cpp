// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_kv_cache_write.hpp"

#include <algorithm>
#include <cassert>
#include <memory>
#include <vector>

#include "../primitive_ocl_base.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "pa_kv_cache_update.hpp"
#include "paged_kv_cache_write_inst.h"

namespace ov::intel_gpu::ocl {
namespace {

using InputIdx = paged_kv_cache_write::InputIdx;

// Runs the KV cache update stage of paged_attention, and nothing else
class PagedKVCacheWriteImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::PagedKVCacheWriteImpl)

    // Dispatch (e.g. PREFILL vs MIXED stage) is decided by reading past_lens/subsequence_begins contents
    bool supports_replay() const override {
        return false;
    }

    Stage::Ptr kv_cache_update = make_stage<KVCacheUpdateGenerator>();

    PagedKVCacheWriteImpl() : PrimitiveImplOCL(PagedKVCacheWrite::get_type_info_static()) {}
    explicit PagedKVCacheWriteImpl(const kernel_impl_params& params) : PagedKVCacheWriteImpl() {
        add_stage(kv_cache_update, params);
    }

    void update_rt_params(const primitive_inst& instance) override {
        update_stages_flags(instance);
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<KVCacheUpdateRuntimeParams>();
        }
        auto* rt_params = static_cast<KVCacheUpdateRuntimeParams*>(m_rt_params.get());
        const auto& params = *instance.get_impl_params();

        rt_params->stage = get_kv_cache_update_stage(params, InputIdx::KEY, InputIdx::PAST_LENS);
        if ((rt_params->stage == PagedAttentionStage::PREFILL || rt_params->stage == PagedAttentionStage::MIXED) && !params.is_dynamic()) {
            const auto inputs = get_kv_cache_update_desc(params).inputs;
            rt_params->paged_attention_aligned_seq_len = static_cast<size_t>(get_kv_cache_update_aligned_seq_len(params, inputs, rt_params->stage));
        }
    }

    void update(primitive_inst& inst, const kernel_impl_params& impl_params) override {
        PrimitiveImplOCL::update(inst, impl_params);
        update_rt_params(inst);
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        const auto& params = *instance.get_impl_params();

        update_stages_flags(instance);
        kernel_dump_info.clear_entries();
        auto* rt_params = static_cast<KVCacheUpdateRuntimeParams*>(m_rt_params.get());
        assert(rt_params != nullptr);

        if (params.get_input_layout(InputIdx::KEY).count() == 0) {
            return instance.get_network().get_stream().aggregate_events(events, events.size() > 1, instance.is_output());
        }

        if (rt_params->stage == PagedAttentionStage::PREFILL || rt_params->stage == PagedAttentionStage::MIXED) {
            fill_kv_cache_update_buffers(instance, get_kv_cache_update_desc(params).inputs);
        }
        return execute_stage(events, instance, kv_cache_update);
    }

    // The rows to write may change (MIXED stage) even if the input shapes haven't been changed
    bool requires_update(primitive_inst& inst, const kernel_impl_params& impl_params) const override {
        return get_kv_cache_update_stage(impl_params, InputIdx::KEY, InputIdx::PAST_LENS) == PagedAttentionStage::MIXED;
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        // The block distribution buffers [0, 1, 2] of the KV cache update stage
        const auto indexes_dt = ov::element::u8;
        const auto element_size = 4;  // 4 bytes
        const int64_t target_seq_len_block_size = 16;

        int64_t aligned_seq_len = -1;
        if (!params.is_dynamic()) {
            const auto stage = get_kv_cache_update_stage(params, InputIdx::KEY, InputIdx::PAST_LENS);
            aligned_seq_len = get_kv_cache_update_aligned_seq_len(params, get_kv_cache_update_desc(params).inputs, stage);
        }
        const auto target_seq_len = std::max<int64_t>(aligned_seq_len, 1);
        const auto indexes_buf_size = static_cast<int64_t>(ceil_div(target_seq_len, target_seq_len_block_size)) * element_size;

        const bool lockable = true;
        const bool not_shareable = false;
        std::vector<BufferDescriptor> internal_buffers;
        for (size_t i = 0; i < 3; i++) {
            internal_buffers.emplace_back(indexes_buf_size, indexes_dt, lockable, not_shareable);
        }
        return internal_buffers;
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<PagedKVCacheWriteImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> PagedKVCacheWrite::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<paged_kv_cache_write>());
    return std::make_unique<PagedKVCacheWriteImpl>(params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::paged_kv_cache_write)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::PagedKVCacheWriteImpl)
