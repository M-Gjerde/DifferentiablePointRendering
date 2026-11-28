#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include <glm/glm.hpp>
#include <imgui.h>
#include <imgui_internal.h>

#include "ViewportLayout.h"

namespace viewer {
    class NavigationGizmo {
    public:
        enum class Action { None, Orbit, Snap, Zoom, Pan, Camera, Projection };

        struct Input {
            Action action = Action::None;
            ImVec2 delta{};
            glm::vec3 axis{0.0f};
            bool capturesMouse = false;
        };

        Input update(const ViewportCanvas& canvas, const glm::mat3& worldToView,
                     bool visible, bool interactive, bool sceneCamera, bool orthographic,
                     bool hasSceneCamera) {
            visible_ = visible;
            sceneCamera_ = sceneCamera;
            orthographic_ = orthographic;
            hasSceneCamera_ = hasSceneCamera;
            hovered_ = Action::None;
            highlightedAxis_ = -1;
            if (!visible_) {
                pressed_ = Action::None;
                return {};
            }

            canvas_ = canvas;
            scale_ = std::min({ImGui::GetFontSize() / 13.0f, canvas.size.x / 100.0f, canvas.size.y / 210.0f});
            center_ = {std::round(canvas.min.x + canvas.size.x - 50.0f * scale_),
                       std::round(canvas.min.y + 50.0f * scale_)};
            alignedAxis_ = -1;
            for (int axis = 0; axis < 3; ++axis) {
                const glm::vec3 direction = worldToView[axis];
                if (direction.x * direction.x + direction.y * direction.y < 1.0e-6f) alignedAxis_ = axis;
                for (int side = 0; side < 2; ++side) {
                    const float sign = side == 0 ? -1.0f : 1.0f;
                    axes_[2 * axis + side] = {sign * direction, axis, sign};
                }
            }

            Input result;
            const auto& io = ImGui::GetIO();
            if (!interactive || io.AppFocusLost) {
                pressed_ = Action::None;
                return result;
            }

            const auto mouse = io.MousePos;
            const bool otherDrag = pressed_ == Action::None &&
                (ImGui::IsMouseDragging(ImGuiMouseButton_Left) || ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ||
                 ImGui::IsMouseDragging(ImGuiMouseButton_Right));
            if (!otherDrag && ImGui::IsWindowHovered() &&
                ImGui::IsMouseHoveringRect(canvas.min, {canvas.min.x + canvas.size.x, canvas.min.y + canvas.size.y})) {
                if (distanceSquared(mouse, center_) <= radius() * radius()) {
                    hovered_ = Action::Orbit;
                    highlightedAxis_ = nearestAxis(mouse);
                } else {
                    for (std::size_t i = 0; i < buttons.size(); ++i) {
                        if (distanceSquared(mouse, buttonCenter(i)) <= buttonRadius() * buttonRadius()) {
                            hovered_ = buttons[i];
                            break;
                        }
                    }
                }
            }

            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && hovered_ != Action::None &&
                (hovered_ != Action::Camera || hasSceneCamera_)) {
                pressed_ = hovered_;
                pressedAxis_ = {};
                if (highlightedAxis_ >= 0) {
                    const auto& handle = axes_[highlightedAxis_];
                    pressedAxis_[handle.axis] = handle.sign;
                }
                dragging_ = false;
            }
            result.capturesMouse = pressed_ != Action::None || hovered_ != Action::None;
            if (pressed_ == Action::None) return result;

            const bool canDrag = pressed_ == Action::Orbit || pressed_ == Action::Zoom || pressed_ == Action::Pan;
            if (canDrag && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                result.action = pressed_;
                result.delta = dragging_ ? io.MouseDelta : ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
                dragging_ = true;
            }
            if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                if (!dragging_ && hovered_ == pressed_ &&
                    io.MouseDragMaxDistanceSqr[ImGuiMouseButton_Left] < io.MouseDragThreshold * io.MouseDragThreshold) {
                    if (pressed_ == Action::Orbit) {
                        result.action = Action::Snap;
                        result.axis = pressedAxis_;
                    } else if (!canDrag) {
                        result.action = pressed_;
                    }
                }
                pressed_ = Action::None;
            }
            return result;
        }

        void draw(ImDrawList& draw, const ImVec4& background) const {
            if (!visible_) return;
            draw.PushClipRect(canvas_.min, {canvas_.min.x + canvas_.size.x, canvas_.min.y + canvas_.size.y}, true);
            const bool orbitActive = hovered_ == Action::Orbit || pressed_ == Action::Orbit;
            if (orbitActive) draw.AddCircleFilled(center_, radius(), IM_COL32(128, 128, 128, 128), 64);

            std::array<int, 6> order{0, 1, 2, 3, 4, 5};
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                const float nearA = axes_[a].view.z, nearB = axes_[b].view.z;
                return nearA == nearB ? a < b : nearA < nearB;
            });
            const ImVec4 colors[] = {{1.0f, 51 / 255.0f, 82 / 255.0f, 1.0f},
                                     {139 / 255.0f, 220 / 255.0f, 0.0f, 1.0f},
                                     {40 / 255.0f, 144 / 255.0f, 1.0f, 1.0f}};
            for (const int index : order) {
                const auto& handle = axes_[index];
                const bool aligned = alignedAxis_ == handle.axis;
                const bool positive = handle.sign > 0.0f;
                if (aligned && handle.view.z < 0.0f) continue;
                const auto color = colors[handle.axis];
                const auto rim = mix(background, color, 0.75f + 0.25f * handle.view.z);
                const ImVec2 tip = axisPosition(handle, 0.8f * radius());
                if (!aligned && (positive || alignedAxis_ != -1)) {
                    const ImVec2 start = axisPosition(handle, -scale_);
                    const int firstVertex = draw.VtxBuffer.Size;
                    draw.AddLine(start, tip, ImGui::GetColorU32(rim), 2.0f * scale_);
                    ImGui::ShadeVertsLinearColorGradientKeepAlpha(&draw, firstVertex, draw.VtxBuffer.Size,
                        start, tip, ImGui::GetColorU32(mix(background, color, 0.75f)), ImGui::GetColorU32(rim));
                }
                const float size = radius() * 0.2f * (1.0f + 0.08f * handle.view.z);
                ImVec4 fill = positive || aligned ? rim : mix(background, color, 0.25f);
                if (!positive && !aligned) fill.w = std::clamp(1.0f + handle.view.z, 0.0f, 1.0f);
                draw.AddCircleFilled(tip, size, ImGui::GetColorU32(fill), 24);
                if (!positive) {
                    const auto ring = aligned ? mix(ImVec4(1, 1, 1, 1), color, 0.5f) : rim;
                    draw.AddCircle(tip, size, ImGui::GetColorU32(ring), 24, 1.3f * scale_);
                }
                const bool highlighted = highlightedAxis_ == index || (aligned && highlightedAxis_ == (index ^ 1));
                if (positive || highlighted || aligned) drawLabel(draw, tip, handle, highlighted);
            }

            for (std::size_t i = 0; i < buttons.size(); ++i) drawButton(draw, buttons[i], buttonCenter(i));
            draw.PopClipRect();
        }

    private:
        struct AxisHandle {
            glm::vec3 view{0.0f};
            int axis = 0;
            float sign = 1.0f;
        };

        static constexpr std::array buttons{Action::Zoom, Action::Pan, Action::Camera, Action::Projection};
        ViewportCanvas canvas_{};
        std::array<AxisHandle, 6> axes_{};
        ImVec2 center_{};
        float scale_ = 1.0f;
        int alignedAxis_ = -1;
        int highlightedAxis_ = -1;
        Action hovered_ = Action::None;
        Action pressed_ = Action::None;
        glm::vec3 pressedAxis_{0.0f};
        bool visible_ = false;
        bool sceneCamera_ = false;
        bool orthographic_ = false;
        bool hasSceneCamera_ = false;
        bool dragging_ = false;

        float radius() const { return 40.0f * scale_; }
        float buttonRadius() const { return 14.0f * scale_; }

        ImVec2 buttonCenter(std::size_t i) const {
            return {std::round(canvas_.min.x + canvas_.size.x - 22.5f * scale_),
                    std::round(canvas_.min.y + (105.0f + 30.0f * static_cast<float>(i)) * scale_)};
        }

        ImVec2 axisPosition(const AxisHandle& handle, float length) const {
            return {center_.x + length * handle.view.x, center_.y - length * handle.view.y};
        }

        int nearestAxis(ImVec2 mouse) const {
            int nearest = -1;
            float best = 0.0f;
            for (int i = 0; i < static_cast<int>(axes_.size()); ++i) {
                const auto& handle = axes_[i];
                // Clicking the center handle of an aligned view flips to the other side.
                if (alignedAxis_ == handle.axis && handle.view.z > 0.0f) continue;
                const float distance = distanceSquared(mouse, axisPosition(handle, radius()));
                if (nearest < 0 || distance < best) {
                    nearest = i;
                    best = distance;
                }
            }
            return nearest;
        }

        static float distanceSquared(ImVec2 a, ImVec2 b) {
            const float x = a.x - b.x, y = a.y - b.y;
            return x * x + y * y;
        }

        static ImVec4 mix(ImVec4 a, ImVec4 b, float t) {
            return {std::lerp(a.x, b.x, t), std::lerp(a.y, b.y, t), std::lerp(a.z, b.z, t), 1.0f};
        }

        void drawLabel(ImDrawList& draw, ImVec2 center, const AxisHandle& handle, bool highlighted) const {
            const char axis = "XYZ"[handle.axis];
            const char label[] = {handle.sign > 0.0f ? axis : '-', handle.sign > 0.0f ? '\0' : axis, '\0'};
            const float size = 10.0f * scale_;
            auto* font = ImGui::GetFont();
            const auto extent = font->CalcTextSizeA(size, 100.0f, 0.0f, label);
            const ImVec2 origin{std::round(center.x - extent.x * 0.5f), std::round(center.y - extent.y * 0.5f)};
            draw.AddText(font, size, origin, highlighted ? IM_COL32_WHITE : IM_COL32(0, 0, 0, 235), label);
        }

        void drawButton(ImDrawList& draw, Action button, ImVec2 center) const {
            const bool enabled = button != Action::Camera || hasSceneCamera_;
            const bool active = enabled && (button == hovered_ || button == pressed_);
            draw.AddCircleFilled(center, buttonRadius(), active ? IM_COL32(108, 108, 108, 128) : IM_COL32(8, 8, 8, 128), 32);
            draw.AddCircle(center, buttonRadius(), IM_COL32(0, 0, 0, 95), 32, scale_);
            const ImU32 color = enabled ? IM_COL32(230, 230, 230, 255) : IM_COL32(144, 144, 144, 175);
            const auto point = [&](float x, float y) { return ImVec2(center.x + scale_ * x, center.y + scale_ * y); };
            const auto line = [&](float x, float y, float endX, float endY) {
                draw.AddLine(point(x, y), point(endX, endY), color, 1.3f * scale_);
            };

            if (button == Action::Zoom) {
                draw.AddCircle(point(-1.5f, -1.5f), 4.7f * scale_, color, 20, 1.5f * scale_);
                line(2, 2, 6.4f, 6.4f);
                line(-3.6f, -1.5f, 0.6f, -1.5f);
                line(-1.5f, -3.6f, -1.5f, 0.6f);
            } else if (button == Action::Pan) {
                const ImVec2 hand[] = {
                    point(-3, 6), point(-6, 0), point(-5, -1), point(-3, 1), point(-3, -5), point(-1.5f, -5),
                    point(-1.5f, -1.5f), point(-1.5f, -7), point(0, -7), point(0, -1.5f), point(0, -6),
                    point(1.5f, -6), point(1.5f, -1.5f), point(1.5f, -4.5f), point(3, -4.5f),
                    point(3, -0.5f), point(4.5f, -2.5f), point(5.5f, -1.5f), point(4, 6),
                };
                draw.AddPolyline(hand, IM_ARRAYSIZE(hand), color, ImDrawFlags_Closed, 1.15f * scale_);
            } else if (button == Action::Camera) {
                if (sceneCamera_) draw.AddRectFilled(point(-7, -4), point(2, 4), color, scale_);
                else draw.AddRect(point(-7, -4), point(2, 4), color, scale_, 0, 1.3f * scale_);
                const ImVec2 lens[] = {point(2, -1.5f), point(6.5f, -4), point(6.5f, 4), point(2, 1.5f)};
                if (sceneCamera_) draw.AddConvexPolyFilled(lens, IM_ARRAYSIZE(lens), color);
                else draw.AddPolyline(lens, IM_ARRAYSIZE(lens), color, 0, 1.3f * scale_);
            } else {
                const float top = orthographic_ ? 6.0f : 3.5f;
                for (const float column : {-1.0f, -0.34f, 0.34f, 1.0f}) line(column * top, -5.5f, column * 6, 5.5f);
                for (const float row : {0.0f, 0.3f, 0.62f, 1.0f}) {
                    const float halfWidth = std::lerp(top, 6.0f, row);
                    line(-halfWidth, -5.5f + 11 * row, halfWidth, -5.5f + 11 * row);
                }
            }
        }
    };
}
