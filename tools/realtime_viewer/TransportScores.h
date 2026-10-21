#pragma once
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace viewer {
struct TransportCameraScores {
    std::string name;
    bool training = false;
    std::vector<float> camera, shadow;
};

// -1 means sum one pass over all training cameras. Held-out cameras remain
// inspectable individually, but must not inflate the training pruning score.
inline std::vector<double> selectTransportScores(const std::vector<TransportCameraScores>& views,
                                                int source, int component) {
    if (views.empty()) return {};
    if (source < -1 || source >= static_cast<int>(views.size()) || component < 0 || component > 2)
        throw std::invalid_argument("Invalid contribution score selection");
    const auto count = views.front().camera.size();
    std::vector<double> sum(count, 0.0);
    for (std::size_t v = 0; v < views.size(); ++v) {
        const auto& view = views[v];
        if (view.camera.size() != count || view.shadow.size() != count)
            throw std::invalid_argument("Contribution score buffer sizes differ");
        if (source >= 0 ? v != static_cast<std::size_t>(source) : !view.training) continue;
        for (std::size_t i = 0; i < count; ++i) {
            const double camera = view.camera[i], shadow = view.shadow[i];
            if (!std::isfinite(camera) || !std::isfinite(shadow) || camera < 0 || shadow < 0)
                throw std::invalid_argument("Invalid contribution score");
            sum[i] += component == 1 ? camera : component == 2 ? shadow : camera + shadow;
        }
    }
    return sum;
}
}
