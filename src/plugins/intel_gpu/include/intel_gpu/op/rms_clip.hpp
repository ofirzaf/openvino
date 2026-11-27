// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::intel_gpu::op {

/// \brief Clip of the root mean square along the last axis:
///        `data * min(1, gamma * rsqrt(mean(data^2) + epsilon))`.
///        A row whose root mean square is above `gamma` is scaled down to it, a shorter row is passed through.
///        It is the RMS operation with its scale limited to 1, and runs on the RMS primitive.
class RMSClip : public ov::op::Op {
public:
    OPENVINO_OP("RMSClip", "gpu_opset");

    RMSClip() = default;

    /// \param data Input tensor
    /// \param gamma Positive scalar constant: the root mean square that a row is clipped to
    /// \param epsilon Added to the mean of squares, keeps a zero row finite
    RMSClip(const ov::Output<Node>& data, const ov::Output<Node>& gamma, double epsilon);

    void validate_and_infer_types() override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    double get_epsilon() const {
        return m_epsilon;
    }

private:
    double m_epsilon = 0.0;
};

}  // namespace ov::intel_gpu::op
