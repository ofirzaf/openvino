// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/rms_clip.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/rms.hpp"
#include "plugin/transformations/rms_clip_fusion.hpp"

namespace {

struct ClipOptions {
    ov::Shape gamma_shape = {1, 1, 1, 1};
    float gamma = 2.0f;
    float factor = -1.0f;            // the multiplier of the Maximum, `aten::neg`
    bool factor_behind_convert = true;  // as exported: Convert(Constant(i32 -1))
    bool swap_inputs = false;        // Minimum(limit, x) and Maximum(limit, x)
    bool other_minimum_input = false;  // the Minimum reads another tensor than the RMS
    bool expose_limit = false;       // the RMS result has a third reader
    bool swap_subtract = false;      // Relu(-Maximum) - Relu(Minimum)
    int64_t axis = -1;
};

struct Clip {
    std::shared_ptr<ov::op::v0::Parameter> input;
    std::shared_ptr<ov::Model> model;
};

// The exported form of Inkling2's L2 clip after RMSFusion.
Clip build_clip(const ClipOptions& options = {}, const ov::PartialShape& shape = {-1, -1, 24, 128}) {
    Clip clip;
    clip.input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    ov::ParameterVector parameters{clip.input};
    ov::Output<ov::Node> minimum_input = clip.input;
    if (options.other_minimum_input) {
        auto other = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        parameters.push_back(other);
        minimum_input = other;
    }
    auto gamma = ov::op::v0::Constant::create(ov::element::f32, options.gamma_shape, std::vector<float>(ov::shape_size(options.gamma_shape), options.gamma));
    auto limit = std::make_shared<ov::op::internal::RMS>(clip.input, gamma, 1e-12, ov::element::f32, options.axis);
    auto lower = options.swap_inputs ? std::make_shared<ov::op::v1::Minimum>(limit, minimum_input) : std::make_shared<ov::op::v1::Minimum>(minimum_input, limit);
    auto upper = options.swap_inputs ? std::make_shared<ov::op::v1::Maximum>(limit, clip.input) : std::make_shared<ov::op::v1::Maximum>(clip.input, limit);
    ov::Output<ov::Node> factor;
    if (options.factor_behind_convert) {
        factor = std::make_shared<ov::op::v0::Convert>(ov::op::v0::Constant::create(ov::element::i32, {}, {static_cast<int32_t>(options.factor)}), ov::element::f32);
    } else {
        factor = ov::op::v0::Constant::create(ov::element::f32, {}, {options.factor});
    }
    auto positive_part = std::make_shared<ov::op::v0::Relu>(lower);
    auto negative_part = std::make_shared<ov::op::v0::Relu>(std::make_shared<ov::op::v1::Multiply>(upper, factor));
    auto clipped = options.swap_subtract ? std::make_shared<ov::op::v1::Subtract>(negative_part, positive_part)
                                         : std::make_shared<ov::op::v1::Subtract>(positive_part, negative_part);
    clipped->set_friendly_name("clip");
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(clipped)};
    if (options.expose_limit) {
        results.push_back(std::make_shared<ov::op::v0::Result>(limit));
    }
    clip.model = std::make_shared<ov::Model>(results, parameters);
    return clip;
}

std::shared_ptr<ov::intel_gpu::op::RMSClip> run_fusion(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::RMSClipFusion>();
    manager.run_passes(model);
    std::shared_ptr<ov::intel_gpu::op::RMSClip> found;
    size_t pattern_nodes = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto rms_clip = ov::as_type_ptr<ov::intel_gpu::op::RMSClip>(node)) {
            found = rms_clip;
        }
        pattern_nodes += ov::is_type<ov::op::internal::RMS>(node) || ov::is_type<ov::op::v1::Minimum>(node) || ov::is_type<ov::op::v1::Maximum>(node);
    }
    // nothing of the pattern may stay next to the fused node
    if (found) {
        EXPECT_EQ(pattern_nodes, 0);
    }
    return found;
}

TEST(RMSClipFusionTest, FusesExportedForm) {
    auto clip = build_clip();
    auto rms_clip = run_fusion(clip.model);

    ASSERT_NE(rms_clip, nullptr);
    EXPECT_EQ(rms_clip->get_input_node_shared_ptr(0), clip.input);
    EXPECT_DOUBLE_EQ(rms_clip->get_epsilon(), 1e-12);
    EXPECT_EQ(rms_clip->get_friendly_name(), "clip");
    EXPECT_EQ(rms_clip->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(rms_clip->get_output_partial_shape(0), (ov::PartialShape{-1, -1, 24, 128}));
    const auto gamma = ov::as_type_ptr<ov::op::v0::Constant>(rms_clip->get_input_node_shared_ptr(1));
    ASSERT_NE(gamma, nullptr);
    EXPECT_FLOAT_EQ(gamma->cast_vector<float>().at(0), 2.0f);
    // the fused node is the only operation between the input and the result
    EXPECT_EQ(clip.model->get_ops().size(), 4);
}

TEST(RMSClipFusionTest, FusesPlainConstantFactorAndSwappedInputs) {
    ClipOptions options;
    options.factor_behind_convert = false;
    options.swap_inputs = true;
    options.gamma_shape = {};
    EXPECT_NE(run_fusion(build_clip(options).model), nullptr);
}

TEST(RMSClipFusionTest, FusesExplicitLastAxisAndRank3) {
    ClipOptions options;
    options.axis = 2;
    options.gamma_shape = {1};
    EXPECT_NE(run_fusion(build_clip(options, {-1, -1, 16}).model), nullptr);
}

TEST(RMSClipFusionTest, RejectsPerElementGamma) {
    ClipOptions options;
    options.gamma_shape = {128};
    EXPECT_EQ(run_fusion(build_clip(options).model), nullptr);
}

TEST(RMSClipFusionTest, RejectsNonPositiveGamma) {
    ClipOptions options;
    options.gamma = -2.0f;
    EXPECT_EQ(run_fusion(build_clip(options).model), nullptr);
    options.gamma = 0.0f;
    EXPECT_EQ(run_fusion(build_clip(options).model), nullptr);
}

TEST(RMSClipFusionTest, RejectsOtherFactor) {
    ClipOptions options;
    options.factor = 1.0f;
    EXPECT_EQ(run_fusion(build_clip(options).model), nullptr);
    options.factor = -2.0f;
    EXPECT_EQ(run_fusion(build_clip(options).model), nullptr);
}

TEST(RMSClipFusionTest, RejectsMinimumOfAnotherTensor) {
    ClipOptions options;
    options.other_minimum_input = true;
    EXPECT_EQ(run_fusion(build_clip(options).model), nullptr);
}

TEST(RMSClipFusionTest, RejectsLimitWithAnotherReader) {
    ClipOptions options;
    options.expose_limit = true;
    EXPECT_EQ(run_fusion(build_clip(options).model), nullptr);
}

TEST(RMSClipFusionTest, RejectsSwappedSubtract) {
    ClipOptions options;
    options.swap_subtract = true;
    EXPECT_EQ(run_fusion(build_clip(options).model), nullptr);
}

TEST(RMSClipFusionTest, RejectsOtherAxis) {
    ClipOptions options;
    options.axis = 1;
    EXPECT_EQ(run_fusion(build_clip(options).model), nullptr);
}

}  // namespace
