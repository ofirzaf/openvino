// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace internal {

/// \brief Writes new K/V rows into the paged KV cache without computing attention.
///
/// Row `i` of sequence `s` is stored at logical position `past_lens[s] + i`, exactly as the cache write step of
/// PagedAttentionExtension. The outputs are the key and value caches after the write; they alias inputs 2 and 3,
/// which are updated in place. Connecting them to a PagedAttentionExtension cache input orders the write before
/// the attention that reads the cache.
class OPENVINO_API PagedKVCacheWrite : public ov::op::Op {
public:
    OPENVINO_OP("PagedKVCacheWrite");

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

    PagedKVCacheWrite() = default;

    PagedKVCacheWrite(const Output<Node>& key,
                      const Output<Node>& value,
                      const Output<Node>& key_cache,
                      const Output<Node>& value_cache,
                      const Output<Node>& past_lens,
                      const Output<Node>& subsequence_begins,
                      const Output<Node>& block_indices,
                      const Output<Node>& block_indices_begins);

    explicit PagedKVCacheWrite(const ov::OutputVector& args);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace internal
}  // namespace op
}  // namespace ov
