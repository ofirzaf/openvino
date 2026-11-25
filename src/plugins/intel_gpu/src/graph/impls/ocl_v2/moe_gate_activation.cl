// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Gate activation of the MoE expert MLPs, MOE_GATE_ACT(x) on a float: SwiGLU (Swish, default), GeGLU-Tanh,
// GeGLU-ERF or softcapped SwiGLU. GATE_ACT_GELU_ERF takes precedence over GATE_ACT_GELU_TANH when both are set.
// The expert output before the down projection is MOE_UP_ACT(up) * MOE_GATE_ACT(gate); MOE_UP_ACT is the
// identity except for the softcapped SwiGLU, which bounds both factors with softcap(x, c) = c * tanh(x / c):
//   softcap(gate, GATE_SOFTCAP) * sigmoid(gate) * softcap(up, UP_SOFTCAP)
#ifdef GATE_ACT_SWIGLU_SOFTCAP
#    define MOE_GATE_ACT(x) ((GATE_SOFTCAP) * tanh((x) / (GATE_SOFTCAP)) / (1.0f + exp(-(x))))
#    define MOE_UP_ACT(u)   ((UP_SOFTCAP) * tanh((float)(u) / (UP_SOFTCAP)))
#elif defined(GATE_ACT_GELU_ERF)
// ERF Gelu: 0.5 * x * (1 + erf(x / sqrt(2))); A&S 7.1.26 fast erf approximation
inline float FUNC(moe_mlp_fast_erf)(float x) {
    const float p  = 0.3275911f;
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;
    float z = fabs(x);
    float t = 1.0f / (1.0f + p * z);
    float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-(z * z));
    return (x >= 0.0f) ? y : -y;
}
#    define MOE_GATE_ACT(x) (0.5f * (x) * (1.0f + FUNC_CALL(moe_mlp_fast_erf)((x) * 0.7071067811865475f)))
#elif defined(GATE_ACT_GELU_TANH)
#    define MOE_GATE_ACT(x) (0.5f * (x) * (1.0f + (tanh(0.79788458347320556640625f * (x) * (1.0f + 0.044715f * (x) * (x))))))
#else
#    define MOE_GATE_ACT(x) ((x) / (1.0f + exp(-(x))))
#endif

#ifndef MOE_UP_ACT
#    define MOE_UP_ACT(u) (u)
#endif
