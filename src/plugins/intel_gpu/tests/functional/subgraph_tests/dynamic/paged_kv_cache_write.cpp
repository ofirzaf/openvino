// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// PagedKVCacheWrite + PagedAttentionExtension on block rows must match one PagedAttentionExtension over
// [context rows ; block rows] in which only block rows (token_type_ids == 1) are bidirectional:
// the K/V caches must be identical and the block-row outputs must match.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <map>
#include <numeric>
#include <random>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/paged_kv_cache_write.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/core.hpp"

namespace {

using ov::op::v0::Constant;
using ov::op::v0::Parameter;

constexpr int64_t kHeads = 32;
constexpr int64_t kKvHeads = 8;
constexpr int64_t kHeadSize = 128;
constexpr int32_t kBlockSize = 16;

struct Sequence {
    int32_t past;
    int32_t context;
    int32_t block;
};

// One inference per step. Steps of a case share the caches; sequence s is the same request in every step.
struct WriteCase {
    std::string name;
    std::vector<Sequence> sequences;
    int32_t sliding_window = 0;
    std::vector<std::vector<Sequence>> later_steps = {};
};

std::shared_ptr<Parameter> param(ov::element::Type type, const ov::PartialShape& shape, const std::string& name) {
    auto p = std::make_shared<Parameter>(type, shape);
    p->set_friendly_name(name);
    p->output(0).set_names({name});
    return p;
}

std::shared_ptr<ov::op::PagedAttentionExtension> make_pa(const ov::OutputVector& qkv_and_caches,
                                                          const ov::OutputVector& metadata,
                                                          int32_t sliding_window = 0) {
    // metadata: past_lens, subsequence_begins, block_indices, block_indices_begins, max_context_len, token_type_ids
    const auto i32_empty = Constant::create(ov::element::i32, ov::Shape{0}, std::vector<int32_t>{});
    const auto i32_zero = Constant::create(ov::element::i32, ov::Shape{}, {0});
    ov::OutputVector inputs = qkv_and_caches;
    inputs.insert(inputs.end(), {metadata[0], metadata[1], metadata[2], metadata[3]});
    inputs.push_back(Constant::create(ov::element::f32, ov::Shape{}, {1.0f / std::sqrt(static_cast<float>(kHeadSize))}));
    inputs.push_back(Constant::create(ov::element::i32, ov::Shape{}, {sliding_window}));  // sliding_window
    inputs.push_back(Constant::create(ov::element::f32, ov::Shape{0}, std::vector<float>{}));  // alibi
    inputs.push_back(metadata[4]);                                                         // max_context_len
    inputs.push_back(i32_zero);                                                            // score_aggregation_window
    inputs.push_back(i32_empty);                                                           // rotated_block_indices
    inputs.push_back(i32_empty);                                                           // rotation_deltas
    inputs.push_back(Constant::create(ov::element::f32, ov::Shape{0}, std::vector<float>{}));  // rotation_trig_lut
    inputs.push_back(Constant::create(ov::element::f32, ov::Shape{0}, std::vector<float>{}));  // xattention_threshold
    inputs.push_back(i32_zero);                                                            // xattention_block_size
    inputs.push_back(i32_zero);                                                            // xattention_stride
    inputs.push_back(Constant::create(ov::element::f16, ov::Shape{0}, std::vector<float>{}));  // sinks
    inputs.push_back(i32_zero);                                                            // adaptive_rkv_start_size
    inputs.push_back(i32_empty);                                                           // adaptive_rkv_evictable_sizes
    inputs.push_back(i32_empty);                                                           // diversity block set indices
    inputs.push_back(i32_empty);                                                           // diversity block set begins
    inputs.push_back(metadata[5]);                                                         // token_type_ids
    inputs.push_back(Constant::create(ov::element::u8, ov::Shape{0}, std::vector<uint8_t>{}));  // qq_bias
    inputs.push_back(i32_empty);                                                           // qq_bias_begins
    auto pa = std::make_shared<ov::op::PagedAttentionExtension>(inputs);
    pa->get_rt_info()["num_k_heads"] = kKvHeads;
    pa->get_rt_info()["k_head_size"] = kHeadSize;
    pa->get_rt_info()["num_v_heads"] = kKvHeads;
    pa->get_rt_info()["v_head_size"] = kHeadSize;
    return pa;
}

ov::ParameterVector pa_metadata(const std::string& prefix) {
    return {param(ov::element::i32, {-1}, prefix + "past_lens"),
            param(ov::element::i32, {-1}, prefix + "subsequence_begins"),
            param(ov::element::i32, {-1}, prefix + "block_indices"),
            param(ov::element::i32, {-1}, prefix + "block_indices_begins"),
            param(ov::element::i32, {}, "max_context_len"),
            param(ov::element::i32, {-1}, "token_type_ids")};
}

// One PA over all new rows of every sequence.
std::shared_ptr<ov::Model> reference_model(int32_t sliding_window = 0) {
    auto q = param(ov::element::f16, {-1, kHeads * kHeadSize}, "q");
    auto k = param(ov::element::f16, {-1, kKvHeads * kHeadSize}, "k");
    auto v = param(ov::element::f16, {-1, kKvHeads * kHeadSize}, "v");
    auto key_cache = param(ov::element::f16, ov::PartialShape::dynamic(4), "key_cache.0");
    auto value_cache = param(ov::element::f16, ov::PartialShape::dynamic(4), "value_cache.0");
    auto metadata = pa_metadata("");
    auto pa = make_pa({q, k, v, key_cache, value_cache},
                      ov::OutputVector(metadata.begin(), metadata.end()),
                      sliding_window);
    ov::ParameterVector params{q, k, v, key_cache, value_cache};
    params.insert(params.end(), metadata.begin(), metadata.end());
    return std::make_shared<ov::Model>(ov::OutputVector{pa->output(0)}, params);
}

// The writer stores context rows, then PA runs on block rows only.
std::shared_ptr<ov::Model> writer_model(int32_t sliding_window = 0) {
    auto k_ctx = param(ov::element::f16, {-1, kKvHeads * kHeadSize}, "k_ctx");
    auto v_ctx = param(ov::element::f16, {-1, kKvHeads * kHeadSize}, "v_ctx");
    auto q = param(ov::element::f16, {-1, kHeads * kHeadSize}, "q");
    auto k = param(ov::element::f16, {-1, kKvHeads * kHeadSize}, "k");
    auto v = param(ov::element::f16, {-1, kKvHeads * kHeadSize}, "v");
    auto key_cache = param(ov::element::f16, ov::PartialShape::dynamic(4), "key_cache.0");
    auto value_cache = param(ov::element::f16, ov::PartialShape::dynamic(4), "value_cache.0");
    ov::ParameterVector writer_metadata{param(ov::element::i32, {-1}, "kv_write_past_lens"),
                                        param(ov::element::i32, {-1}, "kv_write_subsequence_begins"),
                                        param(ov::element::i32, {-1}, "kv_write_block_indices"),
                                        param(ov::element::i32, {-1}, "kv_write_block_indices_begins")};
    auto writer = std::make_shared<ov::op::internal::PagedKVCacheWrite>(k_ctx,
                                                                        v_ctx,
                                                                        key_cache,
                                                                        value_cache,
                                                                        writer_metadata[0],
                                                                        writer_metadata[1],
                                                                        writer_metadata[2],
                                                                        writer_metadata[3]);
    auto metadata = pa_metadata("");
    auto pa = make_pa({q, k, v, writer->output(0), writer->output(1)},
                      ov::OutputVector(metadata.begin(), metadata.end()),
                      sliding_window);
    writer->get_rt_info() = pa->get_rt_info();
    ov::ParameterVector params{k_ctx, v_ctx, q, k, v, key_cache, value_cache};
    params.insert(params.end(), writer_metadata.begin(), writer_metadata.end());
    params.insert(params.end(), metadata.begin(), metadata.end());
    return std::make_shared<ov::Model>(ov::OutputVector{pa->output(0)}, params);
}

ov::Tensor i32_tensor(const std::vector<int32_t>& values, bool scalar = false) {
    ov::Tensor t(ov::element::i32, scalar ? ov::Shape{} : ov::Shape{values.size()});
    std::copy(values.begin(), values.end(), t.data<int32_t>());
    return t;
}

ov::Tensor rows_tensor(const std::vector<ov::float16>& data, int64_t width) {
    const size_t rows = data.size() / width;
    ov::Tensor t(ov::element::f16, ov::Shape{rows, static_cast<size_t>(width)});
    std::copy(data.begin(), data.end(), t.data<ov::float16>());
    return t;
}

// Metadata of a set of sequences that contribute `rows` new rows each.
struct Metadata {
    std::vector<int32_t> past_lens, subsequence_begins{0}, block_indices, block_indices_begins{0};
};

class PagedKVCacheWriteTest : public testing::TestWithParam<WriteCase> {};

TEST_P(PagedKVCacheWriteTest, MatchesSinglePagedAttention) {
    std::vector<std::vector<Sequence>> steps{GetParam().sequences};
    steps.insert(steps.end(), GetParam().later_steps.begin(), GetParam().later_steps.end());
    const size_t num_sequences = steps.front().size();
    const int64_t q_width = kHeads * kHeadSize;
    const int64_t kv_width = kKvHeads * kHeadSize;

    // Block tables: sequence s owns enough blocks for its longest step, interleaved across sequences.
    std::vector<std::vector<int32_t>> tables(num_sequences);
    int32_t next_block = 1;
    for (size_t s = 0; s < num_sequences; ++s) {
        int32_t total = 0;
        for (const auto& step : steps) {
            ASSERT_EQ(step.size(), num_sequences);
            total = std::max(total, step[s].past + step[s].context + step[s].block);
        }
        for (int32_t b = 0; b < std::max(1, (total + kBlockSize - 1) / kBlockSize); ++b) {
            tables[s].push_back(next_block);
            next_block += 2;
        }
    }
    const size_t num_blocks = static_cast<size_t>(next_block);

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto random_rows = [&](int64_t rows, int64_t width) {
        std::vector<ov::float16> data(rows * width);
        for (auto& x : data) {
            x = ov::float16(dist(rng));
        }
        return data;
    };

    ov::Core core;
    const ov::AnyMap config{ov::hint::inference_precision(ov::element::f16), ov::hint::kv_cache_precision(ov::element::f16)};
    auto reference = core.compile_model(reference_model(GetParam().sliding_window), "GPU", config);
    auto split = core.compile_model(writer_model(GetParam().sliding_window), "GPU", config);
    auto context = core.get_default_context("GPU");

    // Both models get identical, randomly initialized caches (the past rows) as USM host tensors, so the in-place
    // updates are visible on the host after inference.
    auto make_caches = [&](const ov::CompiledModel& compiled) {
        std::vector<ov::Tensor> caches;
        for (const auto* name : {"key_cache.0", "value_cache.0"}) {
            const auto& port = compiled.input(name);
            auto shape = port.get_partial_shape();
            ov::Shape static_shape{num_blocks};
            for (size_t i = 1; i < 4; ++i) {
                static_shape.push_back(shape[i].get_length());
            }
            caches.push_back(context.create_host_tensor(port.get_element_type(), static_shape));
        }
        return caches;
    };
    auto reference_caches = make_caches(reference);
    auto split_caches = make_caches(split);
    for (size_t i = 0; i < 2; ++i) {
        auto* data = reference_caches[i].data<ov::float16>();
        for (size_t j = 0; j < reference_caches[i].get_size(); ++j) {
            data[j] = ov::float16(dist(rng));
        }
        std::copy(data, data + reference_caches[i].get_size(), split_caches[i].data<ov::float16>());
    }

    // The block table of a step covers the step's past + context + block rows, as GenAI schedules it.
    // GPU PA crashes or hangs on prefill when the table is much longer than the rows need, so tables stay tight.
    auto add_sequence = [&](Metadata& m, size_t s, int32_t past, int32_t rows, int32_t step_tokens) {
        m.past_lens.push_back(past);
        m.subsequence_begins.push_back(m.subsequence_begins.back() + rows);
        const auto used = std::max(1, (step_tokens + kBlockSize - 1) / kBlockSize);
        m.block_indices.insert(m.block_indices.end(), tables[s].begin(), tables[s].begin() + used);
        m.block_indices_begins.push_back(static_cast<int32_t>(m.block_indices.size()));
    };
    auto set_metadata = [&](ov::InferRequest& request, const Metadata& m, const std::string& prefix) {
        request.set_tensor(prefix + "past_lens", i32_tensor(m.past_lens));
        request.set_tensor(prefix + "subsequence_begins", i32_tensor(m.subsequence_begins));
        request.set_tensor(prefix + "block_indices", i32_tensor(m.block_indices));
        request.set_tensor(prefix + "block_indices_begins", i32_tensor(m.block_indices_begins));
    };

    auto ref_request = reference.create_infer_request();
    auto split_request = split.create_infer_request();
    ref_request.set_tensor("key_cache.0", reference_caches[0]);
    ref_request.set_tensor("value_cache.0", reference_caches[1]);
    split_request.set_tensor("key_cache.0", split_caches[0]);
    split_request.set_tensor("value_cache.0", split_caches[1]);

    for (size_t step_idx = 0; step_idx < steps.size(); ++step_idx) {
    SCOPED_TRACE("step " + std::to_string(step_idx));
    const auto& sequences = steps[step_idx];
    int32_t max_context = 1;
    for (const auto& seq : sequences) {
        max_context = std::max(max_context, seq.past + seq.context + seq.block);
    }

    // New rows per sequence: [context ; block].
    std::vector<std::vector<ov::float16>> q_rows, k_rows, v_rows;
    for (const auto& seq : sequences) {
        const auto rows = seq.context + seq.block;
        q_rows.push_back(random_rows(rows, q_width));
        k_rows.push_back(random_rows(rows, kv_width));
        v_rows.push_back(random_rows(rows, kv_width));
    }

    // Reference: every sequence with new rows, token types 0 for context rows and 1 for block rows.
    Metadata ref_meta;
    std::vector<ov::float16> ref_q, ref_k, ref_v;
    std::vector<int32_t> ref_types;
    std::vector<std::pair<size_t, size_t>> ref_block_rows;  // (first output row, rows) per sequence with a block
    for (size_t s = 0; s < sequences.size(); ++s) {
        const auto& seq = sequences[s];
        const auto rows = seq.context + seq.block;
        if (rows == 0) {
            continue;
        }
        if (seq.block > 0) {
            ref_block_rows.emplace_back(ref_meta.subsequence_begins.back() + seq.context, seq.block);
        }
        add_sequence(ref_meta, s, seq.past, rows, seq.past + rows);
        ref_q.insert(ref_q.end(), q_rows[s].begin(), q_rows[s].end());
        ref_k.insert(ref_k.end(), k_rows[s].begin(), k_rows[s].end());
        ref_v.insert(ref_v.end(), v_rows[s].begin(), v_rows[s].end());
        ref_types.insert(ref_types.end(), seq.context, 0);
        ref_types.insert(ref_types.end(), seq.block, 1);
    }
    ref_request.set_tensor("q", rows_tensor(ref_q, q_width));
    ref_request.set_tensor("k", rows_tensor(ref_k, kv_width));
    ref_request.set_tensor("v", rows_tensor(ref_v, kv_width));
    set_metadata(ref_request, ref_meta, "");
    ref_request.set_tensor("max_context_len", i32_tensor({max_context}, true));
    ref_request.set_tensor("token_type_ids", i32_tensor(ref_types));
    ref_request.infer();

    // Writer: sequences with context rows. PA: sequences with block rows, past = committed + context rows.
    Metadata write_meta, pa_meta;
    std::vector<ov::float16> w_k, w_v, b_q, b_k, b_v;
    for (size_t s = 0; s < sequences.size(); ++s) {
        const auto& seq = sequences[s];
        if (seq.context > 0) {
            add_sequence(write_meta, s, seq.past, seq.context, seq.past + seq.context + seq.block);
            w_k.insert(w_k.end(), k_rows[s].begin(), k_rows[s].begin() + seq.context * kv_width);
            w_v.insert(w_v.end(), v_rows[s].begin(), v_rows[s].begin() + seq.context * kv_width);
        }
        if (seq.block > 0) {
            add_sequence(pa_meta, s, seq.past + seq.context, seq.block, seq.past + seq.context + seq.block);
            b_q.insert(b_q.end(), q_rows[s].begin() + seq.context * q_width, q_rows[s].end());
            b_k.insert(b_k.end(), k_rows[s].begin() + seq.context * kv_width, k_rows[s].end());
            b_v.insert(b_v.end(), v_rows[s].begin() + seq.context * kv_width, v_rows[s].end());
        }
    }
    split_request.set_tensor("k_ctx", rows_tensor(w_k, kv_width));
    split_request.set_tensor("v_ctx", rows_tensor(w_v, kv_width));
    split_request.set_tensor("q", rows_tensor(b_q, q_width));
    split_request.set_tensor("k", rows_tensor(b_k, kv_width));
    split_request.set_tensor("v", rows_tensor(b_v, kv_width));
    set_metadata(split_request, write_meta, "kv_write_");
    set_metadata(split_request, pa_meta, "");
    split_request.set_tensor("max_context_len", i32_tensor({max_context}, true));
    split_request.set_tensor("token_type_ids", i32_tensor(std::vector<int32_t>(b_q.size() / q_width, 1)));
    split_request.infer();

    // Every new row must be in the split model's cache at its logical position.
    // Key cache: [blocks, kv_heads, head_size, block_size]; value cache: [blocks, kv_heads, block_size, head_size].
    size_t missing_rows = 0;
    const auto* key_cache_data = split_caches[0].data<ov::float16>();
    const auto* value_cache_data = split_caches[1].data<ov::float16>();
    for (size_t s = 0; s < sequences.size(); ++s) {
        for (int32_t r = 0; r < sequences[s].context + sequences[s].block; ++r) {
            const int32_t position = sequences[s].past + r;
            const size_t block = tables[s][position / kBlockSize];
            const size_t slot = position % kBlockSize;
            bool row_ok = true;
            for (int64_t h = 0; h < kKvHeads; ++h) {
                for (int64_t d = 0; d < kHeadSize; ++d) {
                    const auto src = k_rows[s][r * kv_width + h * kHeadSize + d].to_bits();
                    const auto key = key_cache_data[((block * kKvHeads + h) * kHeadSize + d) * kBlockSize + slot].to_bits();
                    const auto vsrc = v_rows[s][r * kv_width + h * kHeadSize + d].to_bits();
                    const auto val = value_cache_data[((block * kKvHeads + h) * kBlockSize + slot) * kHeadSize + d].to_bits();
                    row_ok = row_ok && src == key && vsrc == val;
                }
            }
            missing_rows += row_ok ? 0 : 1;
        }
    }
    EXPECT_EQ(missing_rows, 0u) << "rows not found at their cache positions";

    for (size_t i = 0; i < 2; ++i) {
        const auto* expected = reference_caches[i].data<ov::float16>();
        const auto* actual = split_caches[i].data<ov::float16>();
        size_t mismatches = 0;
        for (size_t j = 0; j < reference_caches[i].get_size(); ++j) {
            mismatches += static_cast<uint16_t>(expected[j].to_bits() != actual[j].to_bits());
        }
        EXPECT_EQ(mismatches, 0u) << (i == 0 ? "key" : "value") << " cache differs";
    }

    const auto ref_out = ref_request.get_output_tensor(0);
    const auto split_out = split_request.get_output_tensor(0);
    size_t block_rows = 0;
    for (const auto& [first, rows] : ref_block_rows) {
        block_rows += rows;
    }
    ASSERT_EQ(split_out.get_shape()[0], block_rows);
    if (block_rows == 0) {
        continue;
    }
    const auto* expected = ref_out.data<ov::float16>();
    const auto* actual = split_out.data<ov::float16>();
    float max_diff = 0.0f;
    size_t out_row = 0;
    for (const auto& [first, rows] : ref_block_rows) {
        for (size_t r = 0; r < rows; ++r, ++out_row) {
            for (int64_t c = 0; c < q_width; ++c) {
                const float e = expected[(first + r) * q_width + c];
                const float a = actual[out_row * q_width + c];
                max_diff = std::max(max_diff, std::abs(e - a));
            }
        }
    }
    std::cout << GetParam().name << " step " << step_idx << ": block rows " << block_rows << ", max |diff| " << max_diff
              << std::endl;
    EXPECT_LT(max_diff, 2e-2f) << "block-row attention output differs";
    }
}

// GPU time of one draft prefill layer: one PA over [N context ; 16 block] rows against writer(N) + PA(16 rows).
// PA on B block rows of one sequence with `past` rows in the cache, timed by wall clock (inputs are small).
// The GPU plugin picks the kernel at run time; select it with OV_GPU_PA_GQA_MULTI_TOKENS_MAX /
// OV_GPU_PA_GQA_MULTI_TOKENS_KERNEL in separate processes. Run with --gtest_also_run_disabled_tests.
TEST(PagedKVCacheWritePerf, DISABLED_BlockRowsAttention) {
    constexpr int iterations = 50;
    const int64_t q_width = kHeads * kHeadSize;
    const int64_t kv_width = kKvHeads * kHeadSize;
    const char* tt_env = std::getenv("PA_PERF_TOKEN_TYPES");
    const bool with_token_types = tt_env == nullptr || std::string(tt_env) != "0";

    ov::Core core;
    const ov::AnyMap config{ov::hint::inference_precision(ov::element::f16), ov::hint::kv_cache_precision(ov::element::f16)};
    auto context = core.get_default_context("GPU");
    for (const int32_t sliding_window : {0, 4096}) {
        auto q = param(ov::element::f16, {-1, q_width}, "q");
        auto k = param(ov::element::f16, {-1, kv_width}, "k");
        auto v = param(ov::element::f16, {-1, kv_width}, "v");
        auto key_cache = param(ov::element::f16, ov::PartialShape::dynamic(4), "key_cache.0");
        auto value_cache = param(ov::element::f16, ov::PartialShape::dynamic(4), "value_cache.0");
        auto metadata = pa_metadata("");
        ov::OutputVector pa_meta(metadata.begin(), metadata.end());
        if (!with_token_types) {
            pa_meta[5] = Constant::create(ov::element::i32, ov::Shape{0}, std::vector<int32_t>{});
            metadata.pop_back();
        }
        auto pa = make_pa({q, k, v, key_cache, value_cache}, pa_meta, sliding_window);
        ov::ParameterVector params{q, k, v, key_cache, value_cache};
        params.insert(params.end(), metadata.begin(), metadata.end());
        auto compiled = core.compile_model(std::make_shared<ov::Model>(ov::OutputVector{pa->output(0)}, params), "GPU", config);

        for (const int32_t block : {8, 16}) {
            for (const int32_t past : {512, 4096, 16384, 65536}) {
                const int32_t blocks = (past + block + kBlockSize - 1) / kBlockSize;
                std::vector<ov::Tensor> caches;
                for (const auto* name : {"key_cache.0", "value_cache.0"}) {
                    const auto& port = compiled.input(name);
                    ov::Shape shape{static_cast<size_t>(blocks)};
                    for (size_t i = 1; i < 4; ++i) {
                        shape.push_back(port.get_partial_shape()[i].get_length());
                    }
                    caches.push_back(context.create_host_tensor(port.get_element_type(), shape));
                    std::fill_n(caches.back().data<ov::float16>(), caches.back().get_size(), ov::float16(0.01f));
                }
                auto request = compiled.create_infer_request();
                auto rows = [&](int64_t width, float value) {
                    ov::Tensor t(ov::element::f16, {static_cast<size_t>(block), static_cast<size_t>(width)});
                    std::fill_n(t.data<ov::float16>(), t.get_size(), ov::float16(value));
                    return t;
                };
                request.set_tensor("q", rows(q_width, 0.01f));
                request.set_tensor("k", rows(kv_width, 0.02f));
                request.set_tensor("v", rows(kv_width, 0.03f));
                request.set_tensor("key_cache.0", caches[0]);
                request.set_tensor("value_cache.0", caches[1]);
                request.set_tensor("past_lens", i32_tensor({past}));
                request.set_tensor("subsequence_begins", i32_tensor({0, block}));
                std::vector<int32_t> table(blocks);
                std::iota(table.begin(), table.end(), 0);
                request.set_tensor("block_indices", i32_tensor(table));
                request.set_tensor("block_indices_begins", i32_tensor({0, blocks}));
                request.set_tensor("max_context_len", i32_tensor({past + block}, true));
                if (with_token_types) {
                    request.set_tensor("token_type_ids", i32_tensor(std::vector<int32_t>(block, 1)));
                }
                std::vector<double> times;
                for (int i = 0; i < iterations + 10; ++i) {
                    const auto start = std::chrono::steady_clock::now();
                    request.infer();
                    const auto end = std::chrono::steady_clock::now();
                    if (i >= 10) {
                        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
                    }
                }
                std::sort(times.begin(), times.end());
                std::cout << "block_pa tt=" << with_token_types << " sw=" << sliding_window << " B=" << block
                          << " past=" << past << " median_ms=" << times[times.size() / 2] << std::endl;
            }
        }
    }
}

// Run with --gtest_also_run_disabled_tests.
TEST(PagedKVCacheWritePerf, DISABLED_PrefillLayer) {
    constexpr int32_t block = 16;
    constexpr int iterations = 7;
    const int64_t q_width = kHeads * kHeadSize;
    const int64_t kv_width = kKvHeads * kHeadSize;

    ov::Core core;
    const ov::AnyMap config{ov::hint::inference_precision(ov::element::f16),
                            ov::hint::kv_cache_precision(ov::element::f16),
                            ov::enable_profiling(true)};
    auto reference = core.compile_model(reference_model(), "GPU", config);
    auto split = core.compile_model(writer_model(), "GPU", config);
    auto context = core.get_default_context("GPU");

    for (const int32_t context_rows : {4096, 16384, 65536}) {
        const int32_t rows = context_rows + block;
        const int32_t blocks = (rows + kBlockSize - 1) / kBlockSize;
        auto filled = [](ov::element::Type type, const ov::Shape& shape, float value) {
            ov::Tensor t(type, shape);
            std::fill_n(t.data<ov::float16>(), t.get_size(), ov::float16(value));
            return t;
        };
        auto make_caches = [&](const ov::CompiledModel& compiled) {
            std::vector<ov::Tensor> caches;
            for (const auto* name : {"key_cache.0", "value_cache.0"}) {
                const auto& port = compiled.input(name);
                ov::Shape shape{static_cast<size_t>(blocks)};
                for (size_t i = 1; i < 4; ++i) {
                    shape.push_back(port.get_partial_shape()[i].get_length());
                }
                caches.push_back(context.create_host_tensor(port.get_element_type(), shape));
            }
            return caches;
        };
        std::vector<int32_t> table(blocks);
        std::iota(table.begin(), table.end(), 0);
        const auto block_indices = i32_tensor(table);
        const auto block_indices_begins = i32_tensor({0, blocks});
        const auto max_context = i32_tensor({rows}, true);

        auto ref_request = reference.create_infer_request();
        auto ref_caches = make_caches(reference);
        ref_request.set_tensor("q", filled(ov::element::f16, {static_cast<size_t>(rows), static_cast<size_t>(q_width)}, 0.01f));
        ref_request.set_tensor("k", filled(ov::element::f16, {static_cast<size_t>(rows), static_cast<size_t>(kv_width)}, 0.02f));
        ref_request.set_tensor("v", filled(ov::element::f16, {static_cast<size_t>(rows), static_cast<size_t>(kv_width)}, 0.03f));
        ref_request.set_tensor("key_cache.0", ref_caches[0]);
        ref_request.set_tensor("value_cache.0", ref_caches[1]);
        ref_request.set_tensor("past_lens", i32_tensor({0}));
        ref_request.set_tensor("subsequence_begins", i32_tensor({0, rows}));
        ref_request.set_tensor("block_indices", block_indices);
        ref_request.set_tensor("block_indices_begins", block_indices_begins);
        ref_request.set_tensor("max_context_len", max_context);
        std::vector<int32_t> types(rows, 0);
        std::fill(types.begin() + context_rows, types.end(), 1);
        ref_request.set_tensor("token_type_ids", i32_tensor(types));

        auto split_request = split.create_infer_request();
        auto split_caches = make_caches(split);
        split_request.set_tensor("k_ctx", filled(ov::element::f16, {static_cast<size_t>(context_rows), static_cast<size_t>(kv_width)}, 0.02f));
        split_request.set_tensor("v_ctx", filled(ov::element::f16, {static_cast<size_t>(context_rows), static_cast<size_t>(kv_width)}, 0.03f));
        split_request.set_tensor("q", filled(ov::element::f16, {block, static_cast<size_t>(q_width)}, 0.01f));
        split_request.set_tensor("k", filled(ov::element::f16, {block, static_cast<size_t>(kv_width)}, 0.02f));
        split_request.set_tensor("v", filled(ov::element::f16, {block, static_cast<size_t>(kv_width)}, 0.03f));
        split_request.set_tensor("key_cache.0", split_caches[0]);
        split_request.set_tensor("value_cache.0", split_caches[1]);
        split_request.set_tensor("kv_write_past_lens", i32_tensor({0}));
        split_request.set_tensor("kv_write_subsequence_begins", i32_tensor({0, context_rows}));
        split_request.set_tensor("kv_write_block_indices", block_indices);
        split_request.set_tensor("kv_write_block_indices_begins", block_indices_begins);
        split_request.set_tensor("past_lens", i32_tensor({context_rows}));
        split_request.set_tensor("subsequence_begins", i32_tensor({0, block}));
        split_request.set_tensor("block_indices", block_indices);
        split_request.set_tensor("block_indices_begins", block_indices_begins);
        split_request.set_tensor("max_context_len", max_context);
        split_request.set_tensor("token_type_ids", i32_tensor(std::vector<int32_t>(block, 1)));

        // Median over iterations of the GPU time of each op type, in ms.
        auto profile = [&](ov::InferRequest& request) {
            std::map<std::string, std::vector<double>> times;
            for (int i = 0; i < iterations + 2; ++i) {
                request.infer();
                if (i < 2) {
                    continue;
                }
                std::map<std::string, double> sum;
                for (const auto& info : request.get_profiling_info()) {
                    if (info.status == ov::ProfilingInfo::Status::EXECUTED) {
                        sum[info.node_type] += info.real_time.count() / 1000.0;
                    }
                }
                for (const auto& [type, ms] : sum) {
                    times[type].push_back(ms);
                }
            }
            std::map<std::string, double> median;
            for (auto& [type, values] : times) {
                std::sort(values.begin(), values.end());
                median[type] = values[values.size() / 2];
            }
            return median;
        };
        // Profiling of a multi-stage primitive covers only its last kernel, so also time whole inferences.
        auto wall = [&](ov::InferRequest& request) {
            std::vector<double> times;
            for (int i = 0; i < iterations + 2; ++i) {
                const auto start = std::chrono::steady_clock::now();
                request.infer();
                const auto end = std::chrono::steady_clock::now();
                if (i >= 2) {
                    times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
                }
            }
            std::sort(times.begin(), times.end());
            return times[times.size() / 2];
        };
        const auto ref_ms = profile(ref_request);
        const auto split_ms = profile(split_request);
        std::cout << "context " << context_rows << " + block " << block << ": wall single PA " << wall(ref_request)
                  << " ms | wall writer + block PA " << wall(split_request) << " ms" << std::endl;
        auto at = [](const std::map<std::string, double>& m, const std::string& key) {
            const auto it = m.find(key);
            return it == m.end() ? 0.0 : it->second;
        };
        std::cout << "context " << context_rows << " + block " << block << ": single PA "
                  << at(ref_ms, "PagedAttentionExtension") << " ms | writer " << at(split_ms, "PagedKVCacheWrite")
                  << " ms + block PA " << at(split_ms, "PagedAttentionExtension") << " ms" << std::endl;
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_PagedKVCacheWrite,
                         PagedKVCacheWriteTest,
                         testing::Values(WriteCase{"prefill_with_block", {{0, 37, 16}}},
                                         WriteCase{"long_prefill_with_block", {{0, 1000, 16}}},
                                         // The writer's block table also covers the block rows: 1025 blocks for
                                         // 1024 blocks of context rows.
                                         WriteCase{"prefill_16k_with_block", {{0, 16384, 16}}},
                                         WriteCase{"decode_step", {{100, 5, 16}}},
                                         WriteCase{"unaligned_past", {{37, 3, 8}}},
                                         WriteCase{"batch", {{0, 40, 16}, {64, 7, 16}, {13, 1, 6}}},
                                         WriteCase{"batch_some_without_block", {{0, 40, 16}, {20, 33, 0}, {13, 4, 9}}},
                                         WriteCase{"prefill_only", {{0, 50, 0}, {16, 21, 0}}},
                                         WriteCase{"single_prefill_only", {{0, 300, 0}}},
                                         WriteCase{"single_prefill_only_chunk", {{256, 77, 0}}},
                                         WriteCase{"no_context_rows", {{20, 0, 16}, {5, 0, 3}}},
                                         // All drafts rejected: one context row (the target's token) per request.
                                         WriteCase{"single_all_rejected", {{50, 1, 16}}},
                                         WriteCase{"batch_all_rejected", {{50, 1, 16}, {700, 1, 16}, {33, 1, 8}}},
                                         WriteCase{"batch_some_rejected", {{0, 40, 16}, {64, 1, 16}, {13, 6, 16}}},
                                         // Sliding window 4096 with context longer than the window.
                                         WriteCase{"window_prefill_with_block", {{0, 5000, 16}}, 4096},
                                         WriteCase{"window_decode_step", {{6000, 5, 16}}, 4096},
                                         WriteCase{"window_all_rejected", {{4100, 1, 16}}, 4096},
                                         WriteCase{"window_batch", {{0, 4500, 16}, {5000, 3, 16}, {100, 1, 8}}, 4096},
                                         // Chunked prefill: prompt chunks without a block, then the last chunk with
                                         // the first block, then generation steps. Each step's past drops the
                                         // previous block rows, as GenAI does after verification.
                                         WriteCase{"chunked_prefill",
                                                   {{0, 700, 0}},
                                                   0,
                                                   {{{700, 700, 0}}, {{1400, 300, 16}}, {{1700, 4, 16}}, {{1704, 1, 16}}}},
                                         WriteCase{"window_chunked_prefill",
                                                   {{0, 3000, 0}},
                                                   4096,
                                                   {{{3000, 3000, 0}}, {{6000, 500, 16}}, {{6500, 3, 16}}, {{6503, 1, 16}}}},
                                         // Request 0 prefills in chunks while request 1 generates.
                                         WriteCase{"window_chunked_prefill_with_generation",
                                                   {{0, 2500, 0}, {4200, 4, 16}},
                                                   4096,
                                                   {{{2500, 2500, 0}, {4204, 1, 16}},
                                                    {{5000, 600, 16}, {4205, 7, 16}},
                                                    {{5600, 2, 16}, {4212, 1, 8}}}}),
                         [](const testing::TestParamInfo<WriteCase>& info) {
                             return info.param.name;
                         });

}  // namespace
