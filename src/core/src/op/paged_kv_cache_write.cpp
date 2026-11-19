// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_kv_cache_write.hpp"

#include <algorithm>

#include "itt.hpp"

namespace ov::op::internal {

PagedKVCacheWrite::PagedKVCacheWrite(const Output<Node>& key,
                                     const Output<Node>& value,
                                     const Output<Node>& key_cache,
                                     const Output<Node>& value_cache,
                                     const Output<Node>& past_lens,
                                     const Output<Node>& subsequence_begins,
                                     const Output<Node>& block_indices,
                                     const Output<Node>& block_indices_begins)
    : Op({key, value, key_cache, value_cache, past_lens, subsequence_begins, block_indices, block_indices_begins}) {
    constructor_validate_and_infer_types();
}

PagedKVCacheWrite::PagedKVCacheWrite(const ov::OutputVector& args) : Op(args) {
    constructor_validate_and_infer_types();
}

bool PagedKVCacheWrite::visit_attributes(ov::AttributeVisitor& /*visitor*/) {
    OV_OP_SCOPE(PagedKVCacheWrite_visit_attributes);
    return true;
}

void PagedKVCacheWrite::validate_and_infer_types() {
    OV_OP_SCOPE(PagedKVCacheWrite_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() == 8, "PagedKVCacheWrite expects 8 inputs, got ", get_input_size());

    const auto check_rank = [this](size_t idx, std::initializer_list<int64_t> ranks, const char* name) {
        const auto& rank = get_input_partial_shape(idx).rank();
        NODE_VALIDATION_CHECK(this,
                              rank.is_dynamic() || std::find(ranks.begin(), ranks.end(), rank.get_length()) != ranks.end(),
                              "Input `",
                              name,
                              "` has unexpected rank ",
                              rank);
    };
    check_rank(KEY, {2}, "key");
    check_rank(VALUE, {2}, "value");
    check_rank(KEY_CACHE, {2, 3, 4, 5}, "key_cache");
    check_rank(VALUE_CACHE, {2, 3, 4, 5}, "value_cache");
    for (size_t idx = PAST_LENS; idx <= BLOCK_INDICES_BEGINS; ++idx) {
        check_rank(idx, {1}, "indices");
        const auto& type = get_input_element_type(idx);
        NODE_VALIDATION_CHECK(this,
                              type.is_dynamic() || type == ov::element::i32,
                              "Input ",
                              idx,
                              " (indices) must be i32, got ",
                              type);
    }

    set_output_type(0, get_input_element_type(KEY_CACHE), get_input_partial_shape(KEY_CACHE));
    set_output_type(1, get_input_element_type(VALUE_CACHE), get_input_partial_shape(VALUE_CACHE));
}

std::shared_ptr<Node> PagedKVCacheWrite::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(PagedKVCacheWrite_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<PagedKVCacheWrite>(new_args);
}

}  // namespace ov::op::internal
