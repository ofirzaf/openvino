// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_kv_cache_write.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/primitives/paged_kv_cache_write.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/runtime/internal_properties.hpp"

namespace ov::intel_gpu {

static void CreatePagedKVCacheWriteOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::PagedKVCacheWrite>& op) {
    using CacheWrite = ov::op::internal::PagedKVCacheWrite;
    validate_inputs_count(op, {8});

    // The primitive writes to the caches in place; its outputs are the caches. A PagedAttentionExtension reads the
    // caches from these outputs, so it runs after the write.
    for (size_t port = 0; port < op->get_output_size(); ++port) {
        for (const auto& target : op->get_output_target_inputs(port)) {
            const auto* consumer = target.get_node();
            OPENVINO_ASSERT(ov::is_type<ov::op::PagedAttentionExtension>(consumer) && target.get_index() == port + 3,
                            "[GPU] PagedKVCacheWrite ", op->get_friendly_name(),
                            " output ", port, " must feed the cache input of a PagedAttentionExtension");
        }
    }

    auto prim = cldnn::paged_kv_cache_write(layer_type_name_ID(op), p.GetInputInfo(op));

    const auto& rt_info = op->get_rt_info();
    const auto* const k_head_size_id = "k_head_size";
    const auto* const v_head_size_id = "v_head_size";
    const auto* const num_k_heads_id = "num_k_heads";
    OPENVINO_ASSERT(rt_info.count(k_head_size_id) && rt_info.count(v_head_size_id) && rt_info.count(num_k_heads_id),
                    "[GPU] PagedKVCacheWrite ", op->get_friendly_name(), " rt_info is missing k_head_size/v_head_size/num_k_heads");
    prim.k_head_size = rt_info.at(k_head_size_id).as<int64_t>();
    prim.v_head_size = rt_info.at(v_head_size_id).as<int64_t>();
    prim.kv_heads_num = rt_info.at(num_k_heads_id).as<int64_t>();
    prim.is_key_by_channel = p.get_config().get_key_cache_quant_mode() == ov::internal::CacheQuantMode::BY_CHANNEL;

    // As for PagedAttentionExtension, the block size of the value cache tells whether XAttention is on
    const auto& value_cache_ps = op->get_input_partial_shape(CacheWrite::VALUE_CACHE);
    prim.has_xattention = value_cache_ps.rank().is_static() && value_cache_ps[2].is_static() &&
                          value_cache_ps[2].get_length() == static_cast<int64_t>(cldnn::paged_attention::block_size_xattn);

    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, PagedKVCacheWrite);

}  // namespace ov::intel_gpu
