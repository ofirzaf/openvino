// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Writes new K/V rows into the paged KV cache without computing attention.
/// The caches are updated in place. The two outputs are the key and value caches after the write: the same memory as
/// inputs KEY_CACHE and VALUE_CACHE. A primitive that reads the caches from these outputs runs after the write.
struct paged_kv_cache_write : public primitive_base<paged_kv_cache_write> {
    CLDNN_DECLARE_PRIMITIVE(paged_kv_cache_write)

    // Matches ov::op::internal::PagedKVCacheWrite::InputIdx
    enum InputIdx {
        KEY = 0,
        VALUE = 1,
        KEY_CACHE = 2,
        VALUE_CACHE = 3,
        PAST_LENS = 4,
        SUBSEQUENCE_BEGINS = 5,
        BLOCK_INDICES = 6,
        BLOCK_INDICES_BEGINS = 7,
    };

    paged_kv_cache_write() : primitive_base("", {}, 2) {}

    paged_kv_cache_write(primitive_id id, std::vector<input_info> inputs) : primitive_base(std::move(id), std::move(inputs), 2) {
        OPENVINO_ASSERT(input.size() == 8, "[GPU] Unexpected inputs number for paged_kv_cache_write primitive: ", input.size());
    }

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, k_head_size);
        seed = hash_combine(seed, v_head_size);
        seed = hash_combine(seed, kv_heads_num);
        seed = hash_combine(seed, is_key_by_channel);
        seed = hash_combine(seed, has_xattention);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs)) {
            return false;
        }
        auto rhs_casted = downcast<const paged_kv_cache_write>(rhs);
        return k_head_size == rhs_casted.k_head_size && v_head_size == rhs_casted.v_head_size && kv_heads_num == rhs_casted.kv_heads_num &&
               is_key_by_channel == rhs_casted.is_key_by_channel && has_xattention == rhs_casted.has_xattention;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<paged_kv_cache_write>::save(ob);
        ob << k_head_size;
        ob << v_head_size;
        ob << kv_heads_num;
        ob << is_key_by_channel;
        ob << has_xattention;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<paged_kv_cache_write>::load(ib);
        ib >> k_head_size;
        ib >> v_head_size;
        ib >> kv_heads_num;
        ib >> is_key_by_channel;
        ib >> has_xattention;
    }

    size_t k_head_size = 0;
    size_t v_head_size = 0;
    size_t kv_heads_num = 0;
    bool is_key_by_channel = false;
    // Align with paged_attention::has_xattention
    bool has_xattention = false;
};
}  // namespace cldnn
