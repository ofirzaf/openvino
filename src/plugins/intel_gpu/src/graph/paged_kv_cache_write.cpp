// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>
#include <string>

#include "json_object.h"
#include "paged_kv_cache_write_inst.h"
#include "primitive_type_base.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(paged_kv_cache_write)

// The caches are written in place: the outputs are the cache inputs
layout paged_kv_cache_write_inst::calc_output_layout(const paged_kv_cache_write_node& /*node*/, const kernel_impl_params& impl_param) {
    return impl_param.get_input_layout(paged_kv_cache_write::InputIdx::KEY_CACHE);
}

template <typename ShapeType>
std::vector<layout> paged_kv_cache_write_inst::calc_output_layouts(const paged_kv_cache_write_node& /*node*/, const kernel_impl_params& impl_param) {
    return {impl_param.get_input_layout(paged_kv_cache_write::InputIdx::KEY_CACHE),
            impl_param.get_input_layout(paged_kv_cache_write::InputIdx::VALUE_CACHE)};
}

template std::vector<layout> paged_kv_cache_write_inst::calc_output_layouts<ov::PartialShape>(const paged_kv_cache_write_node& node,
                                                                                              const kernel_impl_params& impl_param);

std::string paged_kv_cache_write_inst::to_string(const paged_kv_cache_write_node& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    json_composite info;
    info.add("k_head_size", desc->k_head_size);
    info.add("v_head_size", desc->v_head_size);
    info.add("kv_heads_num", desc->kv_heads_num);
    info.add("is_key_by_channel", desc->is_key_by_channel);
    info.add("key_cache_dt", node.get_input_layout(paged_kv_cache_write::InputIdx::KEY_CACHE).data_type);
    info.add("value_cache_dt", node.get_input_layout(paged_kv_cache_write::InputIdx::VALUE_CACHE).data_type);
    node_info->add("paged_kv_cache_write primitive info", info);

    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

paged_kv_cache_write_inst::typed_primitive_inst(network& network, const paged_kv_cache_write_node& node) : parent(network, node) {
    const auto desc = node.get_primitive();
    OPENVINO_ASSERT(desc->k_head_size > 0 && desc->v_head_size > 0 && desc->kv_heads_num > 0,
                    "[GPU] paged_kv_cache_write expects positive head sizes and KV heads number");
}

void paged_kv_cache_write_inst::update_shape() {
    parent::update_shape();
    use_caches_as_outputs();
}

void paged_kv_cache_write_inst::use_caches_as_outputs() {
    OPENVINO_ASSERT(_outputs.size() == 2, "[GPU] paged_kv_cache_write must have 2 outputs");
    build_deps();
    const size_t cache_inputs[] = {paged_kv_cache_write::InputIdx::KEY_CACHE, paged_kv_cache_write::InputIdx::VALUE_CACHE};
    for (size_t i = 0; i < 2; i++) {
        const auto cache = input_memory_ptr(cache_inputs[i]);
        OPENVINO_ASSERT(cache != nullptr, "[GPU] paged_kv_cache_write cache input ", cache_inputs[i], " is not allocated");
        if (!_outputs[i] || !_network.get_engine().is_the_same_buffer(*_outputs[i], *cache)) {
            _outputs[i] = cache;
            // The generic output allocation keeps an output that is large enough
            _max_output_layout_count[i] = cache->get_layout().get_linear_size();
            set_flag(ExecutionFlags::MEMORY_CHANGED);
        }
    }
}
}  // namespace cldnn
