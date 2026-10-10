#pragma once

#include <algorithm>

namespace Pale {
    struct AdjointSlabSampling {
        float qNull;
        float qReflect;
        bool reflectionFirst = false;

        // Use the same selection rule for live samples and cached-sample replay.
        [[nodiscard]] constexpr bool takeNull(float draw) const {
            return reflectionFirst ? draw >= qReflect : draw < qNull;
        }
    };

    // Proposal probabilities are detached from the differentiated transport.
    // A positive uniform mixture preserves derivative support at opacity 0/1.
    [[nodiscard]] constexpr AdjointSlabSampling opacitySlabSampling(
        float opacity, float transmission, float uniformMix) {
        opacity = std::clamp(opacity, 0.0f, 1.0f);
        transmission = std::clamp(transmission, 0.0f, 1.0f);
        uniformMix = std::clamp(uniformMix, 0.0f, 1.0f);
        if (uniformMix == 1.0f) {
            // Preserve the fixed 0.5/0.5 sampler's seed mapping exactly.
            return {0.5f, 0.5f};
        }
        // Sample the smaller probability directly, retaining tiny opacity or
        // transmission that would disappear when subtracted from one.
        if (opacity <= transmission) {
            const float reflection = opacity + uniformMix * (0.5f - opacity);
            return {1.0f - reflection, reflection, true};
        }
        const float null = transmission + uniformMix * (0.5f - transmission);
        return {null, 1.0f - null};
    }
}
