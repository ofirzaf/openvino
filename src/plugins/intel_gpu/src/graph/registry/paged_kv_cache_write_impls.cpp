// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/paged_kv_cache_write.hpp"
#include "primitive_inst.h"
#include "registry.hpp"

#if OV_GPU_WITH_OCL
#    include "impls/ocl_v2/sdpa/paged_kv_cache_write.hpp"
#endif

namespace ov {
namespace intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<paged_kv_cache_write>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {OV_GPU_CREATE_INSTANCE_OCL(ocl::PagedKVCacheWrite, shape_types::any)};

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov
