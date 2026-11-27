// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/rms_clip.hpp"

#include "openvino/core/attribute_visitor.hpp"

namespace ov::intel_gpu::op {

RMSClip::RMSClip(const ov::Output<Node>& data, const ov::Output<Node>& gamma, double epsilon) : Op({data, gamma}), m_epsilon(epsilon) {
    validate_and_infer_types();
}

void RMSClip::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() == 2, "RMSClip takes data and gamma");
    OPENVINO_ASSERT(get_input_element_type(0).is_dynamic() || get_input_element_type(0).is_real(), "RMSClip requires a floating-point input");
    const auto& gamma_shape = get_input_partial_shape(1);
    OPENVINO_ASSERT(gamma_shape.is_static() && ov::shape_size(gamma_shape.to_shape()) == 1, "RMSClip requires a scalar gamma");
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool RMSClip::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("epsilon", m_epsilon);
    return true;
}

std::shared_ptr<ov::Node> RMSClip::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<RMSClip>(new_args.at(0), new_args.at(1), m_epsilon);
}

}  // namespace ov::intel_gpu::op
