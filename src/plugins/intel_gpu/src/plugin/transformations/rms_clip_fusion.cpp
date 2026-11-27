// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rms_clip_fusion.hpp"

#include "intel_gpu/op/rms_clip.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rms.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

namespace {
/// The single value of a constant, also behind a Convert (`aten::neg` exports its -1 that way).
bool single_value(const ov::Output<ov::Node>& output, float& value) {
    const auto constant = ov::util::get_constant_from_source(output);
    return constant && ov::shape_size(constant->get_shape()) == 1 && ov::op::util::get_single_value(constant, value);
}
}  // namespace

RMSClipFusion::RMSClipFusion() {
    using namespace ov::pass::pattern;

    auto x = any_input();
    auto gamma = any_input();
    // the limit is read by the Minimum and the Maximum only
    auto limit = wrap_type<ov::op::internal::RMS>({x, gamma}, consumers_count(2));
    auto lower = wrap_type<ov::op::v1::Minimum>({x, limit}, consumers_count(1));
    auto upper = wrap_type<ov::op::v1::Maximum>({x, limit}, consumers_count(1));
    auto positive_part = wrap_type<ov::op::v0::Relu>({lower}, consumers_count(1));
    auto minus_one = any_input();
    auto negated_upper = wrap_type<ov::op::v1::Multiply>({upper, minus_one}, consumers_count(1));
    auto negative_part = wrap_type<ov::op::v0::Relu>({negated_upper}, consumers_count(1));
    auto clipped = wrap_type<ov::op::v1::Subtract>({positive_part, negative_part});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        const auto root = pm.at(clipped).get_node_shared_ptr();
        const auto rms = ov::as_type_ptr<ov::op::internal::RMS>(pm.at(limit).get_node_shared_ptr());
        if (!rms || transformation_callback(root)) {
            return false;
        }

        // The kernel clips with a scalar gamma, along the last axis.
        const auto rank = rms->get_input_partial_shape(0).rank();
        if (rank.is_dynamic() || !rms->get_elementwise_affine() || (rms->get_axis() != -1 && rms->get_axis() != rank.get_length() - 1)) {
            return false;
        }
        // The equality with the element of smaller magnitude needs the limit to keep the sign of x: gamma > 0.
        float gamma_value = 0.0F;
        float factor = 0.0F;
        if (!single_value(pm.at(gamma), gamma_value) || !(gamma_value > 0.0F) || !single_value(pm.at(minus_one), factor) || factor != -1.0F) {
            return false;
        }
        // No type change inside the pattern: RMSClip writes the type it reads.
        if (rms->get_output_element_type(0) != pm.at(x).get_element_type() || root->get_output_element_type(0) != pm.at(x).get_element_type()) {
            return false;
        }

        auto rms_clip = std::make_shared<ov::intel_gpu::op::RMSClip>(pm.at(x), pm.at(gamma), rms->get_epsilon());
        rms_clip->set_friendly_name(root->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), rms_clip);
        ov::replace_node(root, rms_clip);
        return true;
    };

    auto m = std::make_shared<Matcher>(clipped, "RMSClipFusion");
    register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
