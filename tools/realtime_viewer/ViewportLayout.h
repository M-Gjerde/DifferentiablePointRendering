#pragma once

#include "CameraImageView.h"

namespace viewer {
    struct ViewportCanvas {
        ImVec2 min, size;
    };

    // Both ready and rendering frames reserve the same single-line footer.
    // Work status must not change the image rectangle or render resolution.
    [[nodiscard]] inline ViewportCanvas viewportCanvas() {
        auto size = ImGui::GetContentRegionAvail();
        size.x = std::max(1.f, size.x);
        size.y = std::max(1.f, size.y - ImGui::GetTextLineHeightWithSpacing() - 2.f);
        return {ImGui::GetCursorScreenPos(), size};
    }

    [[nodiscard]] inline CameraImageView::Placement viewportImagePlacement(
        const ViewportCanvas& canvas, bool calibrated, const CameraImageView& image,
        unsigned width, unsigned height) {
        // The orbit camera renders to the canvas aspect. Even an older texture
        // must fill that same rectangle while the next frame is in flight.
        return calibrated ? image.placement(canvas.min, canvas.size, width, height)
                          : CameraImageView::Placement{canvas.min, canvas.size};
    }

    inline void finishViewportCanvas(const ViewportCanvas& canvas) {
        ImGui::SetCursorScreenPos(ImVec2(canvas.min.x,
            canvas.min.y + canvas.size.y + ImGui::GetStyle().ItemSpacing.y));
    }
}
