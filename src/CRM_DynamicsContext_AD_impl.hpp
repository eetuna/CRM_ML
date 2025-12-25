#pragma once

// This file contains template implementations that require the full definition
// of DYNNLEqnParams. It should be included AFTER CRMDYN.hpp.

#include "CRM_DynamicsContext_AD.hpp"
#include "CRMDYN.hpp"

namespace CRMCatheterModel {
namespace dynnl_ad_eigen {

template <typename Scalar>
LearnableParamsAD<Scalar>::LearnableParamsAD(const DYNNLEqnParams& params, int flex_seg_idx)
    : actMass(Scalar(0.0))
{
    // Validate flex_seg_idx bounds
    if (flex_seg_idx < 0 || flex_seg_idx >= params.no_flex_seg) {
        // Zero-initialize K_diag and ustar if out of range
        K_diag.setZero();
        ustar.setZero();
    } else {
        // Extract flexible segment parameters from flex_seg[flex_seg_idx]
        K_diag(0) = Scalar(params.K[flex_seg_idx](0, 0));
        K_diag(1) = Scalar(params.K[flex_seg_idx](1, 1));
        K_diag(2) = Scalar(params.K[flex_seg_idx](2, 2));

        for (int i = 0; i < 3; ++i) {
            ustar(i) = Scalar(params.ustar[flex_seg_idx](i));
        }
    }

    // Extract actuator parameters from actuator[0]
    for (int i = 0; i < 6; ++i) {
        damping(i) = Scalar(params.dynamics.actuators[0].damping(i));
    }
    actMass = Scalar(params.ActMass[0]);
    for (int i = 0; i < 3; ++i) {
        MagMoment(i) = Scalar(params.MagMoment[0](i));
    }

    // Populate CATAM for all actuators
    catam.resize(params.no_act_set);
    for (int j = 0; j < params.no_act_set; ++j) {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                catam[j](r, c) = Scalar(params.CoilAlignmentTurnAreaMatrix[j](r, c));
            }
        }
    }
}

template <typename Scalar>
DynamicsContextAD<Scalar>::DynamicsContextAD(const DYNNLEqnParams& params, int flex_seg_idx)
    : DELTA_T(params.dynamics.DELTA_T),
      integrator_type(params.dynamics.integrator_type),
      learnable(params, flex_seg_idx),
      insertion_length(Scalar(params.InsertedLength)),
      geometry(&params)
{
    // Extract physical constants
    for (int i = 0; i < 3; ++i) {
        B0(i) = params.B0[i];
        g(i) = params.g[i];
    }
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            actInertia(r, c) = params.dynamics.actuators[0].inertia(r, c);
        }
    }

    // Copy dynamics state from params.dynamics
    actuators.reserve(params.dynamics.size());
    for (int i = 0; i < params.dynamics.size(); ++i) {
        actuators.emplace_back(params.dynamics.actuators[i]);
    }
}

} // namespace dynnl_ad_eigen
} // namespace CRMCatheterModel
