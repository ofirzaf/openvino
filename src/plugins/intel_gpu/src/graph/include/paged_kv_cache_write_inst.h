// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/paged_kv_cache_write.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<paged_kv_cache_write> : public typed_program_node_base<paged_kv_cache_write> {
    using parent = typed_program_node_base<paged_kv_cache_write>;

public:
    using parent::parent;

    // The stage (prefill, generate or mixed) and the cache blocks to write follow from the contents of these inputs
    std::set<size_t> get_lockable_input_ids() const override {
        return {paged_kv_cache_write::InputIdx::PAST_LENS, paged_kv_cache_write::InputIdx::SUBSEQUENCE_BEGINS};
    }

    std::vector<size_t> get_shape_infer_dependencies() const override {
        return {paged_kv_cache_write::InputIdx::PAST_LENS, paged_kv_cache_write::InputIdx::SUBSEQUENCE_BEGINS};
    }
};

using paged_kv_cache_write_node = typed_program_node<paged_kv_cache_write>;

template <>
class typed_primitive_inst<paged_kv_cache_write> : public typed_primitive_inst_base<paged_kv_cache_write> {
    using parent = typed_primitive_inst_base<paged_kv_cache_write>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(paged_kv_cache_write_node const& node, const kernel_impl_params& impl_param);
    static layout calc_output_layout(paged_kv_cache_write_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(paged_kv_cache_write_node const& node);

    typed_primitive_inst(network& network, paged_kv_cache_write_node const& node);

private:
    void update_shape() override;
    // Makes the cache inputs the outputs, so that no output memory is allocated
    void use_caches_as_outputs();
};

using paged_kv_cache_write_inst = typed_primitive_inst<paged_kv_cache_write>;

}  // namespace cldnn
