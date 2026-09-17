// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <numeric>

#include "openvino/reference/paged_attention.hpp"

namespace {
using ov::reference::paged_attention_cache::EvictionPolicy;
using ov::reference::paged_attention_cache::PagedCacheManager;

// Zero queries/keys make the independent expected result the mean of the
// visible values. Distinct sequence values expose cross-request indexing bugs.
struct AsymmetricPAFixture {
    static constexpr size_t heads = 4;
    static constexpr size_t kv_heads = 1;
    static constexpr size_t head_size = 4;
    static constexpr uintptr_t node = 1;
    ov::Shape cache_shape{32, kv_heads, 4, head_size};
    std::vector<float> empty_cache = std::vector<float>(ov::shape_size(cache_shape), 0.f);
    PagedCacheManager manager{ov::element::f32, EvictionPolicy::FIFO, 0};

    std::vector<float> run(const std::vector<int32_t>& past,
                           const std::vector<int32_t>& kv_begins,
                           const std::vector<int32_t>& q_begins,
                           const std::vector<int32_t>& types,
                           const std::vector<float>& values,
                           int32_t window) {
        const size_t nq = q_begins.back();
        const size_t nk = kv_begins.back();
        std::vector<float> q(nq * heads * head_size, 0.f);
        std::vector<float> k(nk * kv_heads * head_size, 0.f);
        std::vector<float> v(nk * kv_heads * head_size);
        for (size_t i = 0; i < nk; ++i)
            std::fill_n(v.begin() + i * head_size, head_size, values[i]);
        std::vector<float> out(q.size(), -999.f);
        float scale = 1.f;
        ov::reference::paged_attention<float>(
            node, &manager, out.data(), nullptr, nullptr, q.data(), k.data(), v.data(),
            empty_cache.data(), empty_cache.data(), past.data(), kv_begins.data(),
            nullptr, 0, nullptr, 0, &scale, ov::element::f32, &window,
            nullptr, ov::element::dynamic, {}, nullptr, nullptr, 0,
            nullptr, 0, nullptr, {}, nullptr, ov::element::dynamic, {},
            nullptr, ov::element::dynamic, nullptr, nullptr, nullptr, ov::element::dynamic,
            nullptr, nullptr, nullptr, nullptr, types.data(), types.size(),
            {nq, heads * head_size}, {nk, kv_heads * head_size}, {nk, kv_heads * head_size},
            cache_shape, cache_shape, {past.size()}, {kv_begins.size()}, 7,
            q_begins.data(), q_begins.size());
        return out;
    }

    static void expect_rows(const std::vector<float>& out, const std::vector<float>& expected) {
        ASSERT_EQ(out.size(), expected.size() * heads * head_size);
        for (size_t i = 0; i < expected.size(); ++i)
            for (size_t j = 0; j < heads * head_size; ++j)
                EXPECT_NEAR(out[i * heads * head_size + j], expected[i], 1e-5f) << "query " << i;
    }
};
}  // namespace

TEST(PagedAttentionRefTest, AsymmetricMixedRequestsAndProposalTailRollback) {
    AsymmetricPAFixture f;
    // Two requests: D=3/B=2 and D=1/B=3. Proposal groups may not
    // extend across the packed request boundary.
    auto first = f.run({0, 0}, {0, 5, 9}, {0, 2, 5},
                       {0, 0, 0, 1, 1, 0, 1, 1, 1},
                       {1, 2, 3, 100, 200, 10, 20, 30, 40}, 0);
    AsymmetricPAFixture::expect_rows(first, {61.2f, 61.2f, 25.f, 25.f, 25.f});

    // Retain only target-backed prefixes (3 and 1), replacing ALL proposal
    // rows, including the accepted ones, with new target-backed K/V.
    auto second = f.run({3, 1}, {0, 3, 7}, {0, 2, 4},
                        {0, 1, 1, 0, 0, 1, 1},
                        {4, 5, 6, 11, 12, 13, 14}, 0);
    AsymmetricPAFixture::expect_rows(second, {3.5f, 3.5f, 12.f, 12.f});
}

TEST(PagedAttentionRefTest, AsymmetricSlidingWindowKeepsCompleteProposalBlock) {
    AsymmetricPAFixture f;
    auto result = f.run({0}, {0, 6}, {0, 4}, {0, 0, 1, 1, 1, 1}, {1, 2, 3, 4, 5, 6}, 2);
    // First query includes the preceding context row. Later queries retain
    // every proposal row even when earlier proposals fall outside the window.
    AsymmetricPAFixture::expect_rows(result, {4.f, 4.5f, 4.5f, 4.5f});
    auto next = f.run({2}, {0, 5}, {0, 3}, {0, 0, 1, 1, 1}, {7, 8, 9, 10, 11}, 2);
    AsymmetricPAFixture::expect_rows(next, {9.5f, 10.f, 10.f});
}

TEST(PagedAttentionRefTest, AsymmetricCausalQueriesUseSuffixPositions) {
    AsymmetricPAFixture f;
    auto result = f.run({0}, {0, 5}, {0, 2}, {0, 0, 0, 0, 0}, {1, 2, 3, 4, 5}, 0);
    AsymmetricPAFixture::expect_rows(result, {2.5f, 3.f});
}

TEST(PagedAttentionRefTest, AsymmetricRejectsInvalidPerRequestBoundaries) {
    AsymmetricPAFixture f;
    EXPECT_THROW(f.run({0, 0}, {0, 1, 5}, {0, 2, 3}, {0, 0, 0, 1, 1}, {1, 2, 3, 4, 5}, 0),
                 ov::Exception);
}

TEST(PagedAttentionRefTest, AsymmetricQueriesMixCausalAndBidirectionalGroups) {
    AsymmetricPAFixture f;
    auto result = f.run({0}, {0, 6}, {0, 4},
                        {0, 0, 0, 1, 1, 0}, {1, 2, 3, 4, 5, 6}, 0);
    AsymmetricPAFixture::expect_rows(result, {2.f, 3.f, 3.f, 3.5f});
    auto next = f.run({2}, {0, 5}, {0, 4},
                      {0, 0, 1, 1, 0}, {7, 8, 9, 10, 11}, 2);
    AsymmetricPAFixture::expect_rows(next, {7.5f, 9.f, 9.5f, 10.5f});
}
