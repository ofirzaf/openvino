// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// Fuses a clip of the root mean square, written with an RMS operation and a per-element choice, into one RMSClip:
///
///     limit = RMS(x, gamma)                                  x scaled to the root mean square gamma (a positive scalar)
///     y     = Relu(Minimum(x, limit)) - Relu(-Maximum(x, limit))
///
/// x and limit have the same sign in every element, so y is the one with the smaller magnitude: x when its root mean
/// square is at most gamma, limit otherwise. That is `x * min(1, gamma * rsqrt(mean(x^2) + eps))`.
/// Runs after RMSFusion. The three primitives (RMS, Minimum, Maximum) become one.
class RMSClipFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RMSClipFusion");
    RMSClipFusion();
};

}  // namespace ov::intel_gpu
