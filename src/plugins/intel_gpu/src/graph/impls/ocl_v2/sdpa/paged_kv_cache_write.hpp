// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <memory>
#include <utility>

#include "paged_kv_cache_write_inst.h"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct PagedKVCacheWrite : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::paged_kv_cache_write")
    explicit PagedKVCacheWrite(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_kv_types = {
            ov::element::f32,
            ov::element::f16,
            ov::element::i8,
        };

        // The KV cache of XAttention has another layout
        if (node.as<paged_kv_cache_write>().get_primitive()->has_xattention) {
            return false;
        }

        const auto& k_layout = node.get_input_layout(paged_kv_cache_write::InputIdx::KEY);
        const auto& v_layout = node.get_input_layout(paged_kv_cache_write::InputIdx::VALUE);
        if (!everyone_is(format::bfyx, k_layout.format, v_layout.format)) {
            return false;
        }

        return one_of(k_layout.data_type, supported_kv_types) && one_of(v_layout.data_type, supported_kv_types);
    }
};

}  // namespace ov::intel_gpu::ocl
