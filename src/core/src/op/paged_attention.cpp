// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_attention.hpp"

#include "dimension_util.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"
#include "openvino/util/common_util.hpp"
#include "paged_attention_shape_inference.hpp"

namespace {

// Validates input rank and type. Dynamic rank/type always passes.
// Empty {} means any rank/type.
inline void input_check(const ov::Node* node,
                        size_t idx,
                        const std::string_view input_name,
                        std::initializer_list<ov::Rank>&& allowed_ranks,
                        const std::vector<ov::element::Type>& allowed_types) {
    using namespace ov;
    using namespace ov::util;
    using namespace ov::element;

    const auto& rank = node->get_input_partial_shape(idx).rank();
    const auto& tp = node->get_input_element_type(idx);

    auto rank_check = [&](const Rank& rank) {
        return rank.is_dynamic() || empty(allowed_ranks) || is_rank_compatible_any_of(rank.get_length(), allowed_ranks);
    };

    auto type_check = [&](const Type& type) {
        auto it = std::find(allowed_types.begin(), allowed_types.end(), type);
        return type.is_dynamic() || allowed_types.empty() || it != allowed_types.end();
    };

    NODE_VALIDATION_CHECK(node,
                          rank_check(rank),
                          "Rank of `",
                          input_name,
                          "` input should be in [dynamic, ",
                          join(allowed_ranks),
                          "] list, but it is ",
                          rank,
                          ".");

    NODE_VALIDATION_CHECK(node,
                          type_check(tp),
                          "Element type of `",
                          input_name,
                          "` input should be in [dynamic, ",
                          join(allowed_types),
                          "] list, but it is ",
                          tp,
                          ".");
}

std::vector<ov::element::Type> get_real_types() {
    std::vector<ov::element::Type> real_types;
    for (const auto& type : ov::element::Type::get_known_types()) {
        if (type->is_real()) {
            real_types.push_back(*type);
        }
    }
    return real_types;
}

}  // namespace

namespace ov {
namespace op {

PagedAttentionExtension::PagedAttentionExtension(const ov::OutputVector& args, bool write_kv_cache)
    : ov::op::Op(args),
      m_write_kv_cache(write_kv_cache) {
    constructor_validate_and_infer_types();
}

void PagedAttentionExtension::validate_and_infer_types() {
    OV_OP_SCOPE(PagedAttentionExtension_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 28 || get_input_size() == 31,
                          "PagedAttentionExtension expects 28 legacy inputs or 31 inputs with prefix K/V, but it has ",
                          get_input_size());

    // format: Node*, input_idx, name, {rank_list}, {type_list}
    input_check(this, 0, "query", {2}, {});
    input_check(this, 1, "key", {2}, {});
    input_check(this, 2, "value", {2}, {});
    input_check(this, 3, "key_cache", {2, 3, 4, 5}, {});
    input_check(this, 4, "value_cache", {2, 3, 4, 5}, {});
    input_check(this, 5, "past_lens", {1}, {element::i32});
    input_check(this, 6, "subsequence_begins", {1}, {element::i32});
    input_check(this, 7, "block_indices", {1}, {element::i32});
    input_check(this, 8, "block_indices_begins", {1}, {element::i32});
    input_check(this, 9, "scale", {0}, get_real_types());
    input_check(this, 10, "sliding_window", {0}, {element::i32});
    input_check(this, 11, "alibi_slopes", {1}, get_real_types());
    input_check(this, 12, "max_context_len", {0}, {element::i32});
    input_check(this, 13, "score_aggregation_window", {0, 1}, {element::i32});
    input_check(this, 14, "rotated_block_indices", {1}, {element::i32});
    input_check(this, 15, "rotation_deltas", {1, 2}, {element::i32});
    input_check(this, 16, "rotation_trig_lut", {1, 2}, {element::f16, element::f32});
    input_check(this, 17, "xattention_threshold", {1}, {element::f16, element::f32});
    input_check(this, 18, "xattention_block_size", {0}, {element::i32});
    input_check(this, 19, "xattention_stride", {0}, {element::i32});
    input_check(this, 20, "sinks", {1, 4}, {});

    input_check(this, 21, "adaptive_rkv_start_size", {0}, {element::i32});
    input_check(this, 22, "adaptive_rkv_evictable_sizes", {1}, {element::i32});
    input_check(this, 23, "adaptive_rkv_diversity_block_set_indices", {1}, {element::i32});
    input_check(this, 24, "adaptive_rkv_diversity_block_set_indices_begins", {1}, {element::i32});
    input_check(this, 25, "token_type_ids", {1, 2}, {element::i32});
    input_check(this, 26, "qq_bias", {1}, {element::u8});
    input_check(this, 27, "qq_bias_begins", {1}, {element::i32});
    if (get_input_size() == 31) {
        input_check(this, 28, "prefix_key", {2}, {});
        input_check(this, 29, "prefix_value", {2}, {});
        input_check(this, 30, "prefix_subsequence_begins", {1}, {element::i32});

        const auto& prefix_key_shape = get_input_partial_shape(28);
        const auto& prefix_value_shape = get_input_partial_shape(29);
        const auto& prefix_key_type = get_input_element_type(28);
        const auto& prefix_value_type = get_input_element_type(29);
        const auto& key_type = get_input_element_type(1);
        const auto& value_type = get_input_element_type(2);
        NODE_VALIDATION_CHECK(this,
                              prefix_key_type.is_dynamic() || key_type.is_dynamic() || prefix_key_type == key_type,
                              "Prefix key element type must match key, got ",
                              prefix_key_type,
                              " and ",
                              key_type,
                              ".");
        NODE_VALIDATION_CHECK(
            this,
            prefix_value_type.is_dynamic() || value_type.is_dynamic() || prefix_value_type == value_type,
            "Prefix value element type must match value, got ",
            prefix_value_type,
            " and ",
            value_type,
            ".");
        NODE_VALIDATION_CHECK(this,
                              prefix_key_shape.rank().is_dynamic() || prefix_value_shape.rank().is_dynamic() ||
                                  prefix_key_shape[0].compatible(prefix_value_shape[0]),
                              "Prefix key and value token dimensions must be compatible, got ",
                              prefix_key_shape,
                              " and ",
                              prefix_value_shape,
                              ".");

        const auto& key_shape = get_input_partial_shape(1);
        const auto& value_shape = get_input_partial_shape(2);
        NODE_VALIDATION_CHECK(this,
                              prefix_key_shape.rank().is_dynamic() || key_shape.rank().is_dynamic() ||
                                  prefix_key_shape[1].compatible(key_shape[1]),
                              "Prefix key feature dimension must be compatible with key, got ",
                              prefix_key_shape,
                              " and ",
                              key_shape,
                              ".");
        NODE_VALIDATION_CHECK(this,
                              prefix_value_shape.rank().is_dynamic() || value_shape.rank().is_dynamic() ||
                                  prefix_value_shape[1].compatible(value_shape[1]),
                              "Prefix value feature dimension must be compatible with value, got ",
                              prefix_value_shape,
                              " and ",
                              value_shape,
                              ".");
    }

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);
    // Use m_output_type overrides when set, otherwise fall back to query type
    for (int i = 0; i < 3; ++i) {
        const auto et = m_output_type[i].is_dynamic() ? get_input_element_type(0) : m_output_type[i];
        set_output_type(i, et, output_shapes[i]);
    }
}

std::shared_ptr<ov::Node> PagedAttentionExtension::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(PagedAttentionExtension_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<PagedAttentionExtension>(new_args, m_write_kv_cache);
}

bool PagedAttentionExtension::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("write_kv_cache", m_write_kv_cache);
    return true;
}

const ov::element::Type PagedAttentionExtension::get_out_type(int index) const {
    OPENVINO_ASSERT(index < 3, "Output index should be 0, 1 or 2, but got " + std::to_string(index));
    return m_output_type[index];
}

void PagedAttentionExtension::set_out_type(int index, const ov::element::Type& output_type) {
    OPENVINO_ASSERT(index < 3, "Output index should be 0, 1 or 2, but got " + std::to_string(index));
    m_output_type[index] = output_type;
}

}  // namespace op
}  // namespace ov
