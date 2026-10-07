#pragma once

#include <cmath>
#include <cstdint>

namespace Pale {
    enum class LocalLayerDepthMode : uint32_t {
        NormalDistance = 0u,
        SymmetricRayDepth = 1u
    };

    // Shared scalar predicates for renderer slab assembly and uncapped overlap
    // diagnostics. Keep capacity/opacity/visibility decisions at the call site.
    inline float localLayerRayHalfWidth(float referenceViewCosine, float depthEpsilon,
                                        LocalLayerDepthMode mode) {
        return mode == LocalLayerDepthMode::SymmetricRayDepth
            ? depthEpsilon : depthEpsilon / std::fmax(std::fabs(referenceViewCosine), 0.05f);
    }

    inline float localLayerFacingNormalAgreement(float normalDot, float referenceDotRay,
                                                 float candidateDotRay) {
        // Face both normals toward the ray origin; this is not abs(normalDot).
        return (referenceDotRay > 0.0f) != (candidateDotRay > 0.0f) ? -normalDot : normalDot;
    }

    inline bool localLayerDepthRangeContains(float anchorT, float candidateT, float rayHalfWidth,
                                             LocalLayerDepthMode mode, float rayEpsilon) {
        const float minimum = mode == LocalLayerDepthMode::SymmetricRayDepth
            ? anchorT - rayHalfWidth : anchorT;
        return !(candidateT + rayEpsilon < minimum || candidateT > anchorT + rayHalfWidth);
    }

    inline bool localLayerSurfaceMatches(float normalAgreement, float normalDistance,
                                         float normalCosineThreshold, float depthEpsilon,
                                         LocalLayerDepthMode mode) {
        if (normalAgreement < normalCosineThreshold) return false;
        return !(mode == LocalLayerDepthMode::NormalDistance && normalDistance > depthEpsilon);
    }
}
