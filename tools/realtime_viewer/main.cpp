#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <vector>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <ImGuizmo.h>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <sycl/sycl.hpp>
#include <stb_image.h>
#include <yaml-cpp/yaml.h>
#include "TrainingDebugDefaults.h"
#include "OptimizationSnapshots.h"
#include "TargetLossProjection.h"
#include "TransportScores.h"

#include "Renderer/GPUDataStructures.h"
#include "Renderer/RenderPackage.h"
#include "Renderer/Kernels/IntersectionKernels.h"
#include "Core/ScopedTimer.h"
#include "Core/DeviceAllocation.h"
#include "SurfelDensity.h"
#include "TerminalTransmittance.h"
#include "SurfaceOverlap.h"
#include "Renderer/SurfaceOverlapScene.h"
#include "ResponsiveWork.h"
#include "ViewportScreenshot.h"
#include "ViewportTexture.h"
#include "RawLinearImage.h"
#include "LinearCompositing.h"
#include "ViewportPicking.h"
#include "ViewportGrid.h"
#include "TrainingControls.h"
#include "MeshExportControls.h"
#include "SceneTree.h"
#include "ViewportLayout.h"
#include "ViewportOverlayCache.h"
#include "InspectorInputHandoff.h"
#include "LivePacketLease.h"
#include "spdlog/spdlog.h"

import Pale.Assets;
import Pale.Assets.Core;
import Pale.DeviceSelector;
import Pale.Log;
import Pale.Render.PathTracer;
import Pale.Render.SceneBuild;
import Pale.Render.SceneUpload;
import Pale.Render.Sensors;
import Pale.Scene;
import Pale.Scene.Components;
import Pale.SceneSerializer;
import Pale.Utils.ImageIO;

#ifndef PALE_DEFAULT_ASSET_DIR
#define PALE_DEFAULT_ASSET_DIR "../Assets"
#endif

namespace {
    using viewer::PointCloudSnapshot;
    using viewer::listOptimizationPointCloudSnapshots;
    using viewer::findLatestOptimizationPointCloud;

    // Walk mode drains input each frame instead of replaying old key states.
    // Preserve one-shot actions even when a press and release both arrived
    // during a slow render and the final held state is already released.
    bool viewerKeyPressed(ImGuiKey key) {
        if (ImGui::IsKeyPressed(key, false)) {
            return true;
        }
        if (!ImGui::GetIO().ConfigInputTrickleEventQueue) {
            for (const ImGuiInputEvent& event : ImGui::GetCurrentContext()->InputEventsTrail) {
                if (event.Type == ImGuiInputEventType_Key && event.Key.Key == key && event.Key.Down) {
                    return true;
                }
            }
        }
        return false;
    }

    bool walkMouseClicked(ImGuiMouseButton button) {
        if (ImGui::IsMouseClicked(button)) {
            return true;
        }
        if (!ImGui::GetIO().ConfigInputTrickleEventQueue) {
            for (const ImGuiInputEvent& event : ImGui::GetCurrentContext()->InputEventsTrail) {
                if (event.Type == ImGuiInputEventType_MouseButton &&
                    event.MouseButton.Button == button && event.MouseButton.Down) {
                    return true;
                }
            }
        }
        return false;
    }

    const glm::vec3 kWorldUp{0.0f, 0.0f, 1.0f};
    const glm::vec3 kDefaultLookAt{0.0f, 0.0f, 0.2f};
    constexpr ImVec4 kBlenderViewportBackground = viewer::ui::canvas;
    constexpr uint32_t kDefaultViewportImageExtent = 500u;
    constexpr float kSidebarDockFraction = 0.22f;
    constexpr ImVec4 kUiAccent = viewer::ui::accent;

    // Collapsing headers do not indent their contents automatically. Keep each
    // section's controls and nested headers one level below its heading.
    struct ScopedSectionIndent {
        ScopedSectionIndent() {
            ImGui::Indent(14.0f);
            ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.52f);
        }
        ~ScopedSectionIndent() {
            ImGui::PopItemWidth();
            ImGui::Unindent(14.0f);
        }
        ScopedSectionIndent(const ScopedSectionIndent&) = delete;
        ScopedSectionIndent& operator=(const ScopedSectionIndent&) = delete;
    };

    constexpr std::size_t kFrameTimeHistoryCapacity = 240u;
    constexpr uint64_t kSnapshotIterationStep = 1000u;

    struct AppArgs {
        std::filesystem::path assetsDir = PALE_DEFAULT_ASSET_DIR;
        std::filesystem::path pointCloudPath = "OptimizerTests/initial_points.ply";
        std::filesystem::path scenePath = "OptimizerTests/scene.xml";
        uint32_t width = 0;
        uint32_t height = 0;
        bool sharedSlabs = false;
        bool sceneInputSpecified = false;
    };

    struct SceneBounds {
        glm::vec3 center{0.0f};
        float radius = 1.0f;
    };

    enum class CameraSource {
        Viewport,
        SceneXml,
    };

    enum class ViewImageMode {
        Rendered,
        MeanDepth,
        MedianDepth,
        VisibleNormal,
        MedianDepthNormal,
        MeanDepthNormal,
        DepthDistortion,
        IntraSlabDepth,
        IntraSlabRayDepth,
        PositionPrimitiveScore,
        SurfelDensity,
        SurfaceOverlap,
        TargetLoss,
        DensificationOrigin,
        PrimitiveAge,
        DepthPositionGradient,
        NormalPositionGradient,
        IntraSlabPositionGradient,
        AdjointDerivative,
        TransportScore,
        TerminalTransmittance,
        RawLinear,
    };

    constexpr std::array<ViewImageMode, 22> kViewImageModeCycleOrder = {
        ViewImageMode::Rendered,
        ViewImageMode::MedianDepth,
        ViewImageMode::DepthDistortion,
        ViewImageMode::MeanDepth,
        ViewImageMode::VisibleNormal,
        ViewImageMode::MeanDepthNormal,
        ViewImageMode::MedianDepthNormal,
        ViewImageMode::SurfaceOverlap,
        ViewImageMode::PositionPrimitiveScore,
        ViewImageMode::IntraSlabRayDepth,
        ViewImageMode::TargetLoss,
        ViewImageMode::SurfelDensity,
        ViewImageMode::DensificationOrigin,
        ViewImageMode::PrimitiveAge,
        ViewImageMode::DepthPositionGradient,
        ViewImageMode::NormalPositionGradient,
        ViewImageMode::IntraSlabPositionGradient,
        ViewImageMode::AdjointDerivative,
        ViewImageMode::TransportScore,
        ViewImageMode::IntraSlabDepth,
        ViewImageMode::TerminalTransmittance,
        ViewImageMode::RawLinear,
    };

    constexpr std::array<const char*, 22> kViewImageModeLabels = {
        "Prediction",
        "Median depth",
        "3 Depth distortion",
        "4 Mean depth",
        "Surface normals",
        "6 Mean depth normal",
        "7 Median depth normal",
        "8 Surface overlap (world space)",
        "9 Position primitive score (saved)",
        "0 Intra-slab depth (mean ray depth)",
        "Target RGB loss (projected into 3D)",
        "Surfel density (projected centers)",
        "Densification split origin",
        "Primitive age",
        "Depth distortion |grad position|",
        "Normal consistency |grad position|",
        "Intra-slab consensus |grad position|",
        "Adjoint radiance derivative (signed)",
        "Transport contribution score",
        "Intra-slab depth (plane distance)",
        "Terminal transmittance loss",
        "Raw linear RGB",
    };

    [[nodiscard]] const char* viewImageModeLabel(ViewImageMode mode) {
        for (std::size_t index = 0; index < kViewImageModeCycleOrder.size(); ++index) {
            if (kViewImageModeCycleOrder[index] == mode) {
                return kViewImageModeLabels[index];
            }
        }

        return "Rendered";
    }

    [[nodiscard]] bool isRegularizerGradientView(ViewImageMode mode) {
        return mode == ViewImageMode::DepthPositionGradient ||
               mode == ViewImageMode::NormalPositionGradient ||
               mode == ViewImageMode::IntraSlabPositionGradient;
    }

    [[nodiscard]] bool requiresVisibleSlabSearch(ViewImageMode mode) {
        // Per-primitive maps use the dominant member of the visible slab.
        return mode == ViewImageMode::DensificationOrigin ||
               mode == ViewImageMode::PrimitiveAge ||
               isRegularizerGradientView(mode);
    }


    enum class ScalarColorMap {
        Viridis,
        Jet,
    };

    struct DebugDisplayBuffers {
        uint32_t width = 0;
        uint32_t height = 0;
        bool terminalTransmittanceValid = false;
        std::vector<float> terminalTransmittance;
        bool meanDepthValid = false;
        bool medianDepthValid = false;
        bool visibleNormalValid = false;
        bool depthNormalValid = false;
        bool depthDistortionValid = false;
        bool intraSlabDepthValid = false;
        float slabLossColorMaximum = 0.0f;
        double slabPlaneLossMean = 0.0;
        double slabRayLossMean = 0.0;
        bool positionPrimitiveScoreValid = false;
        bool surfelDensityValid = false;
        viewer::SurfelDensity surfelDensity{64};
        bool surfaceOverlapValid = false;
        std::vector<float> surfaceOverlap;
        std::vector<float> targetLoss;
        bool positionPrimitiveIndicesValid = false;
        bool densificationOriginValid = false;
        bool primitiveAgeValid = false;
        bool visiblePrimitiveIndicesValid = false;
        bool depthPositionGradientValid = false;
        bool normalPositionGradientValid = false;
        bool intraSlabPositionGradientValid = false;
        std::vector<float> meanDepth;
        std::vector<float> medianDepth;
        std::vector<float> visibleNormal;
        std::vector<float> depthNormal;
        std::vector<float> depthDistortion;
        std::vector<float> intraSlabDepth;
        std::vector<float> intraSlabRayDepth;
        std::vector<float> positionPrimitiveScore;
        std::vector<float> positionObservedPrimitiveScores;
        std::vector<uint32_t> positionPrimitiveIndices;
        std::vector<uint32_t> positionInstanceIndices;
        std::vector<std::uint8_t> densificationOrigin;
        std::vector<std::uint32_t> primitiveAge;
        std::vector<uint32_t> visiblePrimitiveIndices;
        std::vector<float> depthPositionGradient;
        std::vector<float> normalPositionGradient;
        std::vector<float> intraSlabPositionGradient;
        std::size_t positionObservedPrimitiveCount = 0u;
        std::size_t positionUnsampledPrimitiveCount = 0u;
        float positionPrimitiveScoreMax = 0.0f;
        float positionPrimitiveSplitThreshold = 0.0f;
        bool positionPrimitiveMetadataAvailable = false;

        void invalidate() {
            width = 0;
            height = 0;
            terminalTransmittanceValid = false;
            meanDepthValid = false;
            medianDepthValid = false;
            visibleNormalValid = false;
            depthNormalValid = false;
            depthDistortionValid = false;
            intraSlabDepthValid = false;
            slabLossColorMaximum = 0.0f;
            slabPlaneLossMean = 0.0;
            slabRayLossMean = 0.0;
            positionPrimitiveScoreValid = false;
            surfelDensityValid = false;
            surfaceOverlapValid = false;
            surfaceOverlap.clear();
            targetLoss.clear();
            positionPrimitiveIndicesValid = false;
            densificationOriginValid = false;
            primitiveAgeValid = false;
            visiblePrimitiveIndicesValid = false;
            depthPositionGradientValid = false;
            normalPositionGradientValid = false;
            intraSlabPositionGradientValid = false;
            meanDepth.clear();
            medianDepth.clear();
            visibleNormal.clear();
            depthNormal.clear();
            depthDistortion.clear();
            intraSlabDepth.clear();
            intraSlabRayDepth.clear();
            positionPrimitiveScore.clear();
            positionObservedPrimitiveScores.clear();
            positionPrimitiveIndices.clear();
            positionInstanceIndices.clear();
            densificationOrigin.clear();
            primitiveAge.clear();
            visiblePrimitiveIndices.clear();
            depthPositionGradient.clear();
            normalPositionGradient.clear();
            intraSlabPositionGradient.clear();
            positionObservedPrimitiveCount = 0u;
            positionUnsampledPrimitiveCount = 0u;
            positionPrimitiveScoreMax = 0.0f;
            positionPrimitiveSplitThreshold = 0.0f;
            positionPrimitiveMetadataAvailable = false;
        }

        void prepareFor(uint32_t nextWidth, uint32_t nextHeight) {
            if (width == nextWidth && height == nextHeight) {
                return;
            }

            invalidate();
            width = nextWidth;
            height = nextHeight;
        }

    };


    struct OrbitCamera {
        glm::vec3 target{0.0f};
        float distance = 3.0f;
        float yaw = 0.0f;
        float pitch = 0.0f;
        float fovyDegrees = 30.0f;
        float nearClip = 0.01f;
        float farClip = 1000.0f;

        [[nodiscard]] glm::vec3 position() const {
            const float cosPitch = std::cos(pitch);
            const glm::vec3 offset{
                distance * cosPitch * std::cos(yaw),
                distance * cosPitch * std::sin(yaw),
                distance * std::sin(pitch),
            };
            return target + offset;
        }

        [[nodiscard]] glm::mat4 viewMatrix() const {
            return glm::lookAt(position(), target, kWorldUp);
        }

        [[nodiscard]] glm::mat4 projectionMatrix(uint32_t width, uint32_t height) const {
            const float h = static_cast<float>(std::max(width, 1u));
            const float v = static_cast<float>(std::max(height, 1u));
            const float fovyRadians = glm::radians(fovyDegrees);
            const float fy = 0.5f * v / std::tan(0.5f * fovyRadians);
            const float fx = fy;
            const float cx = 0.5f * h;
            const float cy = 0.5f * v;

            const float left = -cx * nearClip / fx;
            const float right = (h - cx) * nearClip / fx;
            const float top = cy * nearClip / fy;
            const float bottom = -(v - cy) * nearClip / fy;
            return perspectiveOffCenterRhZo(left, right, bottom, top, nearClip, farClip);
        }

        void orbit(const ImVec2 delta) {
            yaw -= delta.x * 0.005f;
            pitch += delta.y * 0.005f;
            pitch = std::clamp(pitch, -1.50f, 1.50f);
        }

        void lookAround(const ImVec2 delta) {
            const glm::vec3 eye = position();
            constexpr float walkLookSensitivity = 0.5f;
            orbit({delta.x * walkLookSensitivity, delta.y * walkLookSensitivity});
            target += eye - position();
        }

        void walk(const glm::vec3& localDirection, float travelDistance) {
            const glm::vec3 forward = glm::normalize(target - position());
            const glm::vec3 right = glm::normalize(glm::cross(forward, kWorldUp));
            const glm::vec3 direction =
                right * localDirection.x + forward * localDirection.y + kWorldUp * localDirection.z;
            const float length = glm::length(direction);
            if (length > 0.0f) {
                target += direction * (travelDistance / length);
            }
        }

        void pan(const ImVec2 delta) {
            const glm::vec3 pos = position();
            const glm::vec3 forward = glm::normalize(target - pos);
            glm::vec3 right = glm::normalize(glm::cross(forward, kWorldUp));
            if (!std::isfinite(right.x) || !std::isfinite(right.y) || !std::isfinite(right.z)) {
                right = glm::vec3(1.0f, 0.0f, 0.0f);
            }
            const glm::vec3 up = glm::normalize(glm::cross(right, forward));
            const float scale = distance * 0.0015f;
            target -= right * (delta.x * scale);
            target += up * (delta.y * scale);
        }

        void setPositionKeepingTarget(const glm::vec3& newPosition) {
            const glm::vec3 offset = newPosition - target;
            distance = std::max(glm::length(offset), 0.001f);
            pitch = std::clamp(
                std::asin(std::clamp(offset.z / distance, -1.0f, 1.0f)),
                -1.50f,
                1.50f);
            yaw = std::atan2(offset.y, offset.x);
        }

        void zoom(float wheelDelta) {
            distance *= std::exp(-wheelDelta * 0.12f);
            distance = std::clamp(distance, 0.001f, 10000.0f);
        }

        [[nodiscard]] Pale::CameraGPU makeGpuCamera(uint32_t width, uint32_t height) const {
            Pale::CameraGPU camera{};
            const glm::vec3 pos = position();
            const glm::mat4 view = viewMatrix();
            const glm::mat4 worldFromCamera = glm::inverse(view);

            const float h = static_cast<float>(std::max(width, 1u));
            const float v = static_cast<float>(std::max(height, 1u));
            const float fovyRadians = glm::radians(fovyDegrees);
            const float fy = 0.5f * v / std::tan(0.5f * fovyRadians);
            const float fx = fy;
            const float cx = 0.5f * h;
            const float cy = 0.5f * v;

            const glm::mat4 proj = projectionMatrix(width, height);

            camera.view = Pale::glm2sycl(view);
            camera.proj = Pale::glm2sycl(proj);
            camera.invView = Pale::glm2sycl(worldFromCamera);
            camera.invProj = Pale::glm2sycl(glm::inverse(proj));
            camera.pos = Pale::float3{pos.x, pos.y, pos.z};

            const glm::vec3 forward = glm::normalize(target - pos);
            camera.forward = Pale::float3{forward.x, forward.y, forward.z};
            camera.width = width;
            camera.height = height;
            camera.fovy = fovyDegrees;
            camera.fx = fx;
            camera.fy = fy;
            camera.cx = cx;
            camera.cy = cy;
            camera.hasPinholeIntrinsics = 1u;
            camera.useForAdjointPass = 1u;
            Pale::copyName(camera.name, "RealtimeCamera");
            return camera;
        }

    private:
        static glm::mat4 perspectiveOffCenterRhZo(
            float left,
            float right,
            float bottom,
            float top,
            float nearClip,
            float farClip) {
            glm::mat4 m(0.0f);
            m[0][0] = 2.0f * nearClip / (right - left);
            m[1][1] = 2.0f * nearClip / (top - bottom);
            m[2][0] = (right + left) / (right - left);
            m[2][1] = (top + bottom) / (top - bottom);
            m[2][2] = farClip / (nearClip - farClip);
            m[2][3] = -1.0f;
            m[3][2] = (farClip * nearClip) / (nearClip - farClip);
            return m;
        }
    };

    using Texture2D = viewer::Texture2D;

    struct DropState {
        std::filesystem::path pendingPath;
        bool hasPendingPath = false;
    };

    struct FileBrowserEntry {
        std::filesystem::path path;
        bool directory = false;
    };

    struct TimerAggregate {
        std::string name;
        uint32_t count = 0u;
        double totalMs = 0.0;
        double lastMs = 0.0;
        double maxMs = 0.0;
    };

    [[nodiscard]] double averageTimerMs(const TimerAggregate& aggregate) {
        return aggregate.count > 0u
                   ? aggregate.totalMs / static_cast<double>(aggregate.count)
                   : 0.0;
    }

    [[nodiscard]] double renderTimeShare(double timerMs, double lastRenderMs) {
        return lastRenderMs > 0.0 ? 100.0 * timerMs / lastRenderMs : 0.0;
    }

    [[nodiscard]] bool isOuterTimerRow(const std::string& name) {
        return name == "Rendering time" ||
               name == "Forward Pass Total" ||
               name == "Traced forward pass" ||
               name == "Forward photon mapping: camera gather total" ||
               name == "Forward photon mapping: gather pass launch+wait" ||
               name.rfind("Forward submit:", 0) == 0;
    }

    [[nodiscard]] const TimerAggregate* firstDetailedTimer(
        const std::vector<TimerAggregate>& timerAggregates) {
        const auto iterator = std::find_if(
            timerAggregates.begin(),
            timerAggregates.end(),
            [](const TimerAggregate& aggregate) {
                return !isOuterTimerRow(aggregate.name);
            });
        return iterator != timerAggregates.end() ? &*iterator : nullptr;
    }

    [[nodiscard]] std::vector<TimerAggregate> aggregateTimerRecords(
        const std::vector<Pale::ScopedTimerRecord>& records) {
        std::vector<TimerAggregate> aggregates;
        aggregates.reserve(records.size());
        for (const Pale::ScopedTimerRecord& record : records) {
            auto aggregateIterator = std::find_if(
                aggregates.begin(),
                aggregates.end(),
                [&](const TimerAggregate& aggregate) {
                    return aggregate.name == record.name;
                });
            if (aggregateIterator == aggregates.end()) {
                TimerAggregate aggregate{};
                aggregate.name = record.name;
                aggregate.count = 1u;
                aggregate.totalMs = record.durationMs;
                aggregate.lastMs = record.durationMs;
                aggregate.maxMs = record.durationMs;
                aggregates.push_back(std::move(aggregate));
            } else {
                aggregateIterator->count += 1u;
                aggregateIterator->totalMs += record.durationMs;
                aggregateIterator->lastMs = record.durationMs;
                aggregateIterator->maxMs = std::max(aggregateIterator->maxMs, record.durationMs);
            }
        }

        std::sort(aggregates.begin(), aggregates.end(), [](const TimerAggregate& lhs, const TimerAggregate& rhs) {
            return lhs.totalMs > rhs.totalMs;
        });
        return aggregates;
    }

    [[nodiscard]] double percent(uint64_t numerator, uint64_t denominator) {
        if (denominator == 0u) {
            return 0.0;
        }
        return 100.0 * static_cast<double>(numerator) / static_cast<double>(denominator);
    }

    void drawCounterRow(const char* label, uint64_t value, double pixelCount) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(label);
        ImGui::TableNextColumn();
        ImGui::Text("%llu", static_cast<unsigned long long>(value));
        ImGui::TableNextColumn();
        const double perPixel = pixelCount > 0.0 ? static_cast<double>(value) / pixelCount : 0.0;
        ImGui::Text("%.3f", perPixel);
    }

    void appendCounterTextRow(
        std::ostringstream& stream,
        const char* label,
        uint64_t value,
        double pixelCount) {
        const double perPixel = pixelCount > 0.0 ? static_cast<double>(value) / pixelCount : 0.0;
        stream << label << '\t' << value << '\t' << perPixel << '\n';
    }

    [[nodiscard]] std::string buildTimerProfilingText(
        double lastRenderMs,
        uint32_t renderWidth,
        uint32_t renderHeight,
        const Pale::GPUSceneBuffers& sceneGpu,
        const std::vector<Pale::ScopedTimerRecord>& timerRecords) {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(3);
        const double renderFps = lastRenderMs > 0.0 ? 1000.0 / lastRenderMs : 0.0;

        stream << "Render timing summary\n";
        stream << "Last render ms\t" << lastRenderMs << '\n';
        stream << "Last render FPS\t" << renderFps << '\n';
        stream << "Resolution\t" << renderWidth << 'x' << renderHeight << '\n';
        stream << "Surfels\t" << sceneGpu.pointCount << '\n';
        stream << "Triangles\t" << sceneGpu.triangleCount << '\n';
        stream << "BLAS nodes\t" << sceneGpu.blasNodeCount << '\n';
        stream << "TLAS nodes\t" << sceneGpu.tlasNodeCount << "\n\n";

        stream << "Render stages\n";
        stream << "Stage\tCount\tTotal ms\tAvg ms\tLast ms\tMax ms\t% render\n";
        for (const TimerAggregate& aggregate : aggregateTimerRecords(timerRecords)) {
            stream << aggregate.name << '\t'
                   << aggregate.count << '\t'
                   << aggregate.totalMs << '\t'
                   << averageTimerMs(aggregate) << '\t'
                   << aggregate.lastMs << '\t'
                   << aggregate.maxMs << '\t'
                   << renderTimeShare(aggregate.totalMs, lastRenderMs) << '\n';
        }

        stream << "\nRaw timer events\n";
        stream << "Sequence\tName\tms\n";
        for (const Pale::ScopedTimerRecord& record : timerRecords) {
            stream << record.sequence << '\t'
                   << record.name << '\t'
                   << record.durationMs << '\n';
        }
        return stream.str();
    }

    [[nodiscard]] std::string buildBvhCounterProfilingText(
        uint32_t renderWidth,
        uint32_t renderHeight,
        const Pale::RenderProfilingCounters& counters) {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(3);
        const double pixelCount = static_cast<double>(renderWidth) * static_cast<double>(renderHeight);

        stream << "BVH and primitive tests\n";
        stream << "Counter\tTotal\tPer pixel\n";
        appendCounterTextRow(stream, "Scene ray queries", counters.sceneRayQueries, pixelCount);
        appendCounterTextRow(stream, "TLAS AABB tests", counters.tlasNodeTests, pixelCount);
        appendCounterTextRow(stream, "TLAS AABB hits", counters.tlasNodeHits, pixelCount);
        appendCounterTextRow(stream, "TLAS leaf instances", counters.tlasLeafInstances, pixelCount);
        appendCounterTextRow(stream, "Point BLAS AABB tests", counters.blasPointNodeTests, pixelCount);
        appendCounterTextRow(stream, "Point BLAS AABB hits", counters.blasPointNodeHits, pixelCount);
        appendCounterTextRow(stream, "Point leaf primitive tests", counters.pointLeafPrimitiveTests, pixelCount);
        appendCounterTextRow(stream, "Surfel plane tests", counters.surfelPlaneTests, pixelCount);
        appendCounterTextRow(stream, "Surfel profile tests", counters.surfelProfileTests, pixelCount);
        appendCounterTextRow(stream, "Accepted surfel candidates", counters.surfelAcceptedHits, pixelCount);
        appendCounterTextRow(stream, "Mesh BLAS AABB tests", counters.blasMeshNodeTests, pixelCount);
        appendCounterTextRow(stream, "Mesh BLAS AABB hits", counters.blasMeshNodeHits, pixelCount);
        appendCounterTextRow(stream, "Triangle tests", counters.triangleTests, pixelCount);
        appendCounterTextRow(stream, "Triangle hits", counters.triangleHits, pixelCount);

        stream << "\nRates\n";
        stream << "Metric\tPercent\n";
        stream << "TLAS hit rate\t" << percent(counters.tlasNodeHits, counters.tlasNodeTests) << '\n';
        stream << "Point BLAS hit rate\t"
               << percent(counters.blasPointNodeHits, counters.blasPointNodeTests) << '\n';
        stream << "Point primitive acceptance\t"
               << percent(counters.surfelAcceptedHits, counters.pointLeafPrimitiveTests) << '\n';

        stream << "\nForward gather work\n";
        stream << "Counter\tTotal\tPer pixel\n";
        appendCounterTextRow(stream, "Pixels", counters.forwardGatherPixels, pixelCount);
        appendCounterTextRow(stream, "Point-hit queries", counters.forwardGatherPointHitQueries, pixelCount);
        appendCounterTextRow(stream, "Point-hit candidates", counters.forwardGatherPointHitCandidates, pixelCount);
        appendCounterTextRow(stream, "Local layers", counters.forwardGatherLocalLayers, pixelCount);
        appendCounterTextRow(stream, "Local layer hits", counters.forwardGatherLocalLayerHits, pixelCount);
        appendCounterTextRow(stream, "Object-profile local hits", counters.forwardGatherObjectProfileHits, pixelCount);
        appendCounterTextRow(stream, "Regularizer hits", counters.forwardGatherRegularizerHits, pixelCount);
        appendCounterTextRow(stream, "Photon gather calls", counters.forwardGatherPhotonGatherCalls, pixelCount);
        appendCounterTextRow(stream, "Direct-light calls", counters.forwardGatherDirectLightCalls, pixelCount);
        appendCounterTextRow(stream, "Direct-light light visits", counters.forwardGatherDirectLightLightVisits, pixelCount);
        appendCounterTextRow(stream, "Depth-pair iterations", counters.forwardGatherDepthPairIterations, pixelCount);
        appendCounterTextRow(stream, "Mesh hits", counters.forwardGatherMeshHits, pixelCount);
        appendCounterTextRow(stream, "No-hit terminations", counters.forwardGatherNoHitTerminations, pixelCount);
        appendCounterTextRow(stream, "Opacity terminations", counters.forwardGatherOpacityTerminations, pixelCount);
        appendCounterTextRow(stream, "Max-splat terminations", counters.forwardGatherMaxSplatTerminations, pixelCount);

        stream << "\nAdjoint intersection work\nCounter\tTotal\tPer ray\n";
        const double adjointRayCount = static_cast<double>(counters.adjointIntersectionRays);
        appendCounterTextRow(stream, "Rays", counters.adjointIntersectionRays, adjointRayCount);
        appendCounterTextRow(stream, "Local layers", counters.adjointLocalLayers, adjointRayCount);
        appendCounterTextRow(stream, "Null events", counters.adjointNullEvents, adjointRayCount);
        appendCounterTextRow(stream, "Reflection events", counters.adjointReflectEvents, adjointRayCount);
        appendCounterTextRow(stream, "No-hit terminations", counters.adjointNoHitTerminations, adjointRayCount);
        appendCounterTextRow(stream, "Max-splat terminations", counters.adjointMaxSplatTerminations, adjointRayCount);

        stream << "\nForward gather rates\n";
        stream << "Metric\tValue\n";
        stream << "Candidates per point-hit query\t"
               << (counters.forwardGatherPointHitQueries > 0u
                       ? static_cast<double>(counters.forwardGatherPointHitCandidates) /
                         static_cast<double>(counters.forwardGatherPointHitQueries)
                       : 0.0) << '\n';
        stream << "Hits per local layer\t"
               << (counters.forwardGatherLocalLayers > 0u
                       ? static_cast<double>(counters.forwardGatherLocalLayerHits) /
                         static_cast<double>(counters.forwardGatherLocalLayers)
                       : 0.0) << '\n';
        return stream.str();
    }

    [[nodiscard]] std::string buildProfilingClipboardText(
        double lastRenderMs,
        uint32_t renderWidth,
        uint32_t renderHeight,
        const Pale::GPUSceneBuffers& sceneGpu,
        const Pale::RenderProfilingCounters& counters,
        const std::vector<Pale::ScopedTimerRecord>& timerRecords) {
        return buildTimerProfilingText(lastRenderMs, renderWidth, renderHeight, sceneGpu, timerRecords) +
               "\n" +
               buildBvhCounterProfilingText(renderWidth, renderHeight, counters);
    }

    bool drawProfilingWindow(
        bool& open,
        bool& timerProfilingEnabled,
        bool& gpuCounterProfilingEnabled,
        double lastRenderMs,
        const std::vector<float>& frameTimeHistory,
        uint32_t renderWidth,
        uint32_t renderHeight,
        const Pale::GPUSceneBuffers& sceneGpu,
        const Pale::RenderProfilingCounters& counters,
        const std::vector<Pale::ScopedTimerRecord>& timerRecords) {
        if (!open) {
            return false;
        }

        bool settingsChanged = false;
        if (ImGui::Begin(
                "Profiling",
                &open,
                ImGuiWindowFlags_NoCollapse)) {
            settingsChanged |= ImGui::Checkbox("Timers", &timerProfilingEnabled);
            ImGui::SameLine();
            settingsChanged |= ImGui::Checkbox("GPU counters", &gpuCounterProfilingEnabled);
            ImGui::Separator();

            const double renderFps = lastRenderMs > 0.0 ? 1000.0 / lastRenderMs : 0.0;
            ImGui::Text("Last render: %.3f ms  %.2f FPS", lastRenderMs, renderFps);
            ImGui::Text("Viewer frame: %.2f FPS", static_cast<double>(ImGui::GetIO().Framerate));
            ImGui::Text("Resolution: %u x %u", renderWidth, renderHeight);
            ImGui::Text(
                "Scene: %u surfels, %u triangles, %u BLAS nodes, %u TLAS nodes",
                sceneGpu.pointCount,
                sceneGpu.triangleCount,
                sceneGpu.blasNodeCount,
                sceneGpu.tlasNodeCount);

            if (!frameTimeHistory.empty()) {
                float maxFrameTimeMs = 1.0f;
                float totalFrameTimeMs = 0.0f;
                for (const float frameTimeMs : frameTimeHistory) {
                    maxFrameTimeMs = std::max(maxFrameTimeMs, frameTimeMs);
                    totalFrameTimeMs += frameTimeMs;
                }
                const float averageFrameTimeMs =
                    totalFrameTimeMs / static_cast<float>(frameTimeHistory.size());
                ImGui::Text(
                    "Total render time history: avg %.3f ms  max %.3f ms",
                    averageFrameTimeMs,
                    maxFrameTimeMs);
                ImGui::PlotLines(
                    "##TotalRenderTimeHistory",
                    frameTimeHistory.data(),
                    static_cast<int>(frameTimeHistory.size()),
                    0,
                    nullptr,
                    0.0f,
                    maxFrameTimeMs * 1.05f,
                    ImVec2(-1.0f, 96.0f));
            }

            if (ImGui::Button("Copy timers")) {
                const std::string text =
                    buildTimerProfilingText(lastRenderMs, renderWidth, renderHeight, sceneGpu, timerRecords);
                ImGui::SetClipboardText(text.c_str());
            }
            ImGui::SameLine();
            if (ImGui::Button("Copy BVH tests")) {
                const std::string text =
                    buildBvhCounterProfilingText(renderWidth, renderHeight, counters);
                ImGui::SetClipboardText(text.c_str());
            }
            ImGui::SameLine();
            if (ImGui::Button("Copy all")) {
                const std::string text =
                    buildProfilingClipboardText(
                        lastRenderMs,
                        renderWidth,
                        renderHeight,
                        sceneGpu,
                        counters,
                        timerRecords);
                ImGui::SetClipboardText(text.c_str());
            }

            ImGui::Separator();
            if (ImGui::CollapsingHeader("Render stages", ImGuiTreeNodeFlags_DefaultOpen)) {
                const std::vector<TimerAggregate> timerAggregates = aggregateTimerRecords(timerRecords);
                if (timerAggregates.empty()) {
                    ImGui::TextUnformatted("No timer data for the last render");
                } else if (const TimerAggregate* topTimer = firstDetailedTimer(timerAggregates)) {
                    ImGui::Text(
                        "Top detailed timer: %s  %.3f ms total  %.3f ms avg",
                        topTimer->name.c_str(),
                        topTimer->totalMs,
                        averageTimerMs(*topTimer));
                }

                if (!timerAggregates.empty() && ImGui::BeginTable(
                        "TimerAggregateTable",
                        7,
                        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
                    ImGui::TableSetupColumn("Stage");
                    ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthFixed, 56.0f);
                    ImGui::TableSetupColumn("Total ms", ImGuiTableColumnFlags_WidthFixed, 86.0f);
                    ImGui::TableSetupColumn("Avg ms", ImGuiTableColumnFlags_WidthFixed, 86.0f);
                    ImGui::TableSetupColumn("Last ms", ImGuiTableColumnFlags_WidthFixed, 86.0f);
                    ImGui::TableSetupColumn("Max ms", ImGuiTableColumnFlags_WidthFixed, 86.0f);
                    ImGui::TableSetupColumn("% render", ImGuiTableColumnFlags_WidthFixed, 78.0f);
                    ImGui::TableHeadersRow();

                    for (const TimerAggregate& aggregate : timerAggregates) {
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted(aggregate.name.c_str());
                        ImGui::TableNextColumn();
                        ImGui::Text("%u", aggregate.count);
                        ImGui::TableNextColumn();
                        ImGui::Text("%.3f", aggregate.totalMs);
                        ImGui::TableNextColumn();
                        ImGui::Text("%.3f", averageTimerMs(aggregate));
                        ImGui::TableNextColumn();
                        ImGui::Text("%.3f", aggregate.lastMs);
                        ImGui::TableNextColumn();
                        ImGui::Text("%.3f", aggregate.maxMs);
                        ImGui::TableNextColumn();
                        ImGui::Text("%.1f%%", renderTimeShare(aggregate.totalMs, lastRenderMs));
                    }
                    ImGui::EndTable();
                }
            }

            if (ImGui::CollapsingHeader("BVH and primitive tests", ImGuiTreeNodeFlags_DefaultOpen)) {
                const double pixelCount =
                    static_cast<double>(renderWidth) * static_cast<double>(renderHeight);
                if (ImGui::BeginTable(
                        "BvhCounterTable",
                        3,
                        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
                    ImGui::TableSetupColumn("Counter");
                    ImGui::TableSetupColumn("Total", ImGuiTableColumnFlags_WidthFixed, 132.0f);
                    ImGui::TableSetupColumn("Per pixel", ImGuiTableColumnFlags_WidthFixed, 92.0f);
                    ImGui::TableHeadersRow();
                    drawCounterRow("Scene ray queries", counters.sceneRayQueries, pixelCount);
                    drawCounterRow("TLAS AABB tests", counters.tlasNodeTests, pixelCount);
                    drawCounterRow("TLAS AABB hits", counters.tlasNodeHits, pixelCount);
                    drawCounterRow("TLAS leaf instances", counters.tlasLeafInstances, pixelCount);
                    drawCounterRow("Point BLAS AABB tests", counters.blasPointNodeTests, pixelCount);
                    drawCounterRow("Point BLAS AABB hits", counters.blasPointNodeHits, pixelCount);
                    drawCounterRow("Point leaf primitive tests", counters.pointLeafPrimitiveTests, pixelCount);
                    drawCounterRow("Surfel plane tests", counters.surfelPlaneTests, pixelCount);
                    drawCounterRow("Surfel profile tests", counters.surfelProfileTests, pixelCount);
                    drawCounterRow("Accepted surfel candidates", counters.surfelAcceptedHits, pixelCount);
                    drawCounterRow("Mesh BLAS AABB tests", counters.blasMeshNodeTests, pixelCount);
                    drawCounterRow("Mesh BLAS AABB hits", counters.blasMeshNodeHits, pixelCount);
                    drawCounterRow("Triangle tests", counters.triangleTests, pixelCount);
                    drawCounterRow("Triangle hits", counters.triangleHits, pixelCount);
                    ImGui::EndTable();
                }

                ImGui::Text(
                    "TLAS hit rate: %.2f%%",
                    percent(counters.tlasNodeHits, counters.tlasNodeTests));
                ImGui::Text(
                    "Point BLAS hit rate: %.2f%%",
                    percent(counters.blasPointNodeHits, counters.blasPointNodeTests));
                ImGui::Text(
                    "Point primitive acceptance: %.2f%%",
                    percent(counters.surfelAcceptedHits, counters.pointLeafPrimitiveTests));
            }

            if (ImGui::CollapsingHeader("Forward gather work", ImGuiTreeNodeFlags_DefaultOpen)) {
                const double pixelCount =
                    static_cast<double>(renderWidth) * static_cast<double>(renderHeight);
                if (ImGui::BeginTable(
                        "ForwardGatherCounterTable",
                        3,
                        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
                    ImGui::TableSetupColumn("Counter");
                    ImGui::TableSetupColumn("Total", ImGuiTableColumnFlags_WidthFixed, 132.0f);
                    ImGui::TableSetupColumn("Per pixel", ImGuiTableColumnFlags_WidthFixed, 92.0f);
                    ImGui::TableHeadersRow();
                    drawCounterRow("Pixels", counters.forwardGatherPixels, pixelCount);
                    drawCounterRow("Point-hit queries", counters.forwardGatherPointHitQueries, pixelCount);
                    drawCounterRow("Point-hit candidates", counters.forwardGatherPointHitCandidates, pixelCount);
                    drawCounterRow("Local layers", counters.forwardGatherLocalLayers, pixelCount);
                    drawCounterRow("Local layer hits", counters.forwardGatherLocalLayerHits, pixelCount);
                    drawCounterRow("Object-profile local hits", counters.forwardGatherObjectProfileHits, pixelCount);
                    drawCounterRow("Regularizer hits", counters.forwardGatherRegularizerHits, pixelCount);
                    drawCounterRow("Photon gather calls", counters.forwardGatherPhotonGatherCalls, pixelCount);
                    drawCounterRow("Direct-light calls", counters.forwardGatherDirectLightCalls, pixelCount);
                    drawCounterRow("Direct-light light visits", counters.forwardGatherDirectLightLightVisits, pixelCount);
                    drawCounterRow("Depth-pair iterations", counters.forwardGatherDepthPairIterations, pixelCount);
                    drawCounterRow("Mesh hits", counters.forwardGatherMeshHits, pixelCount);
                    drawCounterRow("No-hit terminations", counters.forwardGatherNoHitTerminations, pixelCount);
                    drawCounterRow("Opacity terminations", counters.forwardGatherOpacityTerminations, pixelCount);
                    drawCounterRow("Max-splat terminations", counters.forwardGatherMaxSplatTerminations, pixelCount);
                    ImGui::EndTable();
                }

                const double candidatesPerQuery =
                    counters.forwardGatherPointHitQueries > 0u
                        ? static_cast<double>(counters.forwardGatherPointHitCandidates) /
                          static_cast<double>(counters.forwardGatherPointHitQueries)
                        : 0.0;
                const double hitsPerLayer =
                    counters.forwardGatherLocalLayers > 0u
                        ? static_cast<double>(counters.forwardGatherLocalLayerHits) /
                          static_cast<double>(counters.forwardGatherLocalLayers)
                        : 0.0;
                ImGui::Text("Candidates/query: %.3f", candidatesPerQuery);
                ImGui::Text("Hits/layer: %.3f", hitsPerLayer);
            }

            if (ImGui::CollapsingHeader("Adjoint intersection work", ImGuiTreeNodeFlags_DefaultOpen)) {
                const double rayCount = static_cast<double>(counters.adjointIntersectionRays);
                if (ImGui::BeginTable("AdjointIntersectionCounterTable", 3,
                        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
                    ImGui::TableSetupColumn("Counter");
                    ImGui::TableSetupColumn("Total");
                    ImGui::TableSetupColumn("Per ray");
                    ImGui::TableHeadersRow();
                    drawCounterRow("Rays", counters.adjointIntersectionRays, rayCount);
                    drawCounterRow("Local layers", counters.adjointLocalLayers, rayCount);
                    drawCounterRow("Null events", counters.adjointNullEvents, rayCount);
                    drawCounterRow("Reflection events", counters.adjointReflectEvents, rayCount);
                    drawCounterRow("No-hit terminations", counters.adjointNoHitTerminations, rayCount);
                    drawCounterRow("Max-splat terminations", counters.adjointMaxSplatTerminations, rayCount);
                    ImGui::EndTable();
                }
            }

            if (ImGui::CollapsingHeader("Raw timer events")) {
                if (timerRecords.empty()) {
                    ImGui::TextUnformatted("No timer events");
                } else if (ImGui::BeginTable(
                               "TimerEventTable",
                               3,
                               ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
                    ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 44.0f);
                    ImGui::TableSetupColumn("Name");
                    ImGui::TableSetupColumn("ms", ImGuiTableColumnFlags_WidthFixed, 86.0f);
                    ImGui::TableHeadersRow();

                    for (const Pale::ScopedTimerRecord& record : timerRecords) {
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text("%llu", static_cast<unsigned long long>(record.sequence));
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted(record.name.c_str());
                        ImGui::TableNextColumn();
                        ImGui::Text("%.3f", record.durationMs);
                    }
                    ImGui::EndTable();
                }
            }
        }
        ImGui::End();

        return settingsChanged;
    }

    [[nodiscard]] bool isPlyPath(const std::filesystem::path& path) {
        std::string extension = path.extension().string();
        std::transform(
            extension.begin(),
            extension.end(),
            extension.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return extension == ".ply";
    }

    template<std::size_t N>
    void copyPathToBuffer(const std::filesystem::path& path, std::array<char, N>& buffer) {
        std::snprintf(buffer.data(), buffer.size(), "%s", path.string().c_str());
    }

    [[nodiscard]] Pale::AssetMeta makeAssetMeta(
        const std::filesystem::path& path,
        Pale::AssetType assetType) {
        Pale::AssetMeta meta{};
        meta.type = assetType;
        meta.path = path;
        std::error_code error;
        if (std::filesystem::exists(path, error) && !error) {
            meta.lastWrite = std::filesystem::last_write_time(path, error);
        }
        return meta;
    }

    [[nodiscard]] Pale::AssetHandle importPathAsType(
        Pale::AssetRegistry& registry,
        const std::filesystem::path& path,
        Pale::AssetType assetType) {
        if (const std::optional<Pale::AssetHandle> existing = registry.findByPath(path)) {
            const Pale::AssetMeta* meta = registry.meta(*existing);
            if (meta && meta->type == assetType) {
                return *existing;
            }
        }
        return registry.import(path, assetType);
    }

    [[nodiscard]] glm::quat normalizeQuaternionOrIdentity(glm::quat quaternion);

    [[nodiscard]] std::size_t countSurfels(const Pale::PointAsset& pointAsset) {
        std::size_t surfelCount = 0;
        for (const Pale::PointGeometry& pointGeometry : pointAsset.points) {
            surfelCount += pointGeometry.positions.size();
        }
        return surfelCount;
    }

    [[nodiscard]] std::filesystem::path savedViewerDefaultPath(
        const std::filesystem::path& originalPath) {
        return originalPath.parent_path() /
               (originalPath.stem().string() + ".viewer_default" + originalPath.extension().string());
    }

    [[nodiscard]] bool writePointCloudPly(
        const Pale::PointAsset& pointAsset,
        const std::filesystem::path& destination,
        std::string& reason) {
        const std::size_t pointCount = countSurfels(pointAsset);
        if (pointCount == 0u) {
            reason = "the point cloud is empty";
            return false;
        }

        for (const Pale::PointGeometry& geometry : pointAsset.points) {
            const std::size_t count = geometry.positions.size();
            if (geometry.quat.size() != count || geometry.scales.size() != count ||
                geometry.albedos.size() != count || geometry.opacities.size() != count ||
                geometry.betas.size() != count || geometry.shapes.size() != count ||
                geometry.powers.size() != count) {
                reason = "a point block has inconsistent attribute counts";
                return false;
            }
        }

        std::filesystem::path temporary = destination;
        temporary += ".tmp";
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            reason = "could not open " + temporary.string();
            return false;
        }

        output << "ply\n"
               << "format binary_little_endian 1.0\n"
               << "comment Saved by Pale Reconstruction Viewer\n"
               << "element vertex " << pointCount << "\n"
               << "property float x\nproperty float y\nproperty float z\n"
               << "property float rot_w\nproperty float rot_x\nproperty float rot_y\nproperty float rot_z\n"
               << "property float su\nproperty float sv\n"
               << "property float albedo_r\nproperty float albedo_g\nproperty float albedo_b\n"
               << "property float opacity\nproperty float beta\nproperty float shape\nproperty float power\n"
               << "property float densification_origin\nproperty float primitive_age\n"
               << "property float densification_position_signal\nproperty float densification_position_sample_count\nproperty float densification_position_threshold\n"
               << "end_header\n";
        std::vector<float> block;
        block.reserve(8192*21);
        const auto flush=[&] {
            output.write(reinterpret_cast<const char*>(block.data()),static_cast<std::streamsize>(block.size()*sizeof(float)));
            block.clear();
        };
        for (const Pale::PointGeometry& geometry : pointAsset.points) {
            for (std::size_t index = 0u; index < geometry.positions.size(); ++index) {
                const glm::vec3& position = geometry.positions[index];
                const glm::quat rotation = normalizeQuaternionOrIdentity(geometry.quat[index]);
                const glm::vec2& scale = geometry.scales[index];
                const glm::vec3& albedo = geometry.albedos[index];
                block.insert(block.end(),{position.x,position.y,position.z,rotation.w,rotation.x,rotation.y,rotation.z,
                    scale.x,scale.y,albedo.r,albedo.g,albedo.b,geometry.opacities[index],geometry.betas[index],geometry.shapes[index],geometry.powers[index],
                    index<geometry.densificationOrigins.size()?static_cast<float>(geometry.densificationOrigins[index]):0.f,
                    index<geometry.primitiveAges.size()?static_cast<float>(geometry.primitiveAges[index]):0.f,
                    index<geometry.densificationPositionSignals.size()?geometry.densificationPositionSignals[index]:0.f,
                    index<geometry.densificationPositionSampleCounts.size()?static_cast<float>(geometry.densificationPositionSampleCounts[index]):0.f,
                    index<geometry.densificationPositionThresholds.size()?geometry.densificationPositionThresholds[index]:0.f});
                if(block.size()>=8192*21)flush();
            }
        }
        if(!block.empty())flush();
        output.close();
        if (!output) {
            reason = "writing " + temporary.string() + " failed";
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return false;
        }

        std::error_code error;
        std::filesystem::rename(temporary, destination, error);
        if (error) {
            error.clear();
            std::filesystem::copy_file(
                temporary,
                destination,
                std::filesystem::copy_options::overwrite_existing,
                error);
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
        }
        if (error) {
            reason = "could not replace " + destination.string() + ": " + error.message();
            return false;
        }
        return true;
    }

    [[nodiscard]] std::size_t countMeshTriangles(const Pale::Mesh& mesh) {
        std::size_t triangleCount = 0;
        for (const Pale::Submesh& submesh : mesh.submeshes) {
            triangleCount += submesh.indices.size() / 3u;
        }
        return triangleCount;
    }

    [[nodiscard]] std::shared_ptr<Pale::Mesh> loadMeshPlyForValidation(
        const std::filesystem::path& path) {
        Pale::AssimpMeshLoader loader;
        return loader.load(Pale::AssetHandle{}, makeAssetMeta(path, Pale::AssetType::Mesh));
    }

    [[nodiscard]] std::shared_ptr<Pale::PointAsset> loadPointCloudPlyForValidation(
        const std::filesystem::path& path) {
        Pale::PLYPointLoader loader;
        return loader.load(Pale::AssetHandle{}, makeAssetMeta(path, Pale::AssetType::PointCloud));
    }

    [[nodiscard]] std::vector<FileBrowserEntry> listBrowserEntries(const std::filesystem::path& directory) {
        std::vector<FileBrowserEntry> entries;
        std::error_code error;
        for (const auto& entry : std::filesystem::directory_iterator(directory, error)) {
            const bool isDirectory = entry.is_directory(error);
            if (!isDirectory && !isPlyPath(entry.path())) {
                continue;
            }
            entries.push_back({entry.path(), isDirectory});
        }

        std::sort(entries.begin(), entries.end(), [](const FileBrowserEntry& lhs, const FileBrowserEntry& rhs) {
            if (lhs.directory != rhs.directory) {
                return lhs.directory;
            }
            return lhs.path.filename().string() < rhs.path.filename().string();
        });
        return entries;
    }

    struct PlyFileSnapshot {
        std::uintmax_t size = 0u;
        std::filesystem::file_time_type writeTime = std::filesystem::file_time_type::min();
    };

    [[nodiscard]] std::optional<PlyFileSnapshot> readPlyFileSnapshot(
        const std::filesystem::path& path) {
        std::error_code error;
        if (!std::filesystem::is_regular_file(path, error) || error) {
            return std::nullopt;
        }
        const std::uintmax_t size = std::filesystem::file_size(path, error);
        if (error) {
            return std::nullopt;
        }
        const std::filesystem::file_time_type writeTime =
            std::filesystem::last_write_time(path, error);
        if (error) {
            return std::nullopt;
        }
        return PlyFileSnapshot{.size = size, .writeTime = writeTime};
    }

    [[nodiscard]] bool samePlyFileSnapshot(
        const PlyFileSnapshot& lhs,
        const PlyFileSnapshot& rhs) {
        return lhs.size == rhs.size && lhs.writeTime == rhs.writeTime;
    }

    [[nodiscard]] bool validatePlyReadyForLoad(
        const std::filesystem::path& path,
        PlyFileSnapshot& validatedSnapshot,
        std::string& reason) {
        const std::optional<PlyFileSnapshot> before = readPlyFileSnapshot(path);
        if (!before || before->size == 0u) {
            reason = "file is missing or empty";
            return false;
        }

        std::ifstream input(path, std::ios::binary);
        if (!input) {
            reason = "file could not be opened";
            return false;
        }

        std::string line;
        if (!std::getline(input, line) || (line != "ply" && line != "ply\r")) {
            reason = "PLY header is incomplete";
            return false;
        }

        bool isAscii = false;
        bool foundFormat = false;
        bool foundEndHeader = false;
        bool inVertexElement = false;
        bool foundVertexElement = false;
        std::uint64_t vertexCount = 0u;
        std::size_t vertexPropertyCount = 0u;

        while (std::getline(input, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            std::istringstream tokens(line);
            std::string keyword;
            tokens >> keyword;
            if (keyword == "format") {
                std::string format;
                tokens >> format;
                foundFormat = !format.empty();
                isAscii = format == "ascii";
            } else if (keyword == "element") {
                std::string elementName;
                std::uint64_t elementCount = 0u;
                if (!(tokens >> elementName >> elementCount)) {
                    reason = "PLY element declaration is incomplete";
                    return false;
                }
                inVertexElement = elementName == "vertex";
                if (inVertexElement) {
                    foundVertexElement = true;
                    vertexCount = elementCount;
                    vertexPropertyCount = 0u;
                }
            } else if (keyword == "property" && inVertexElement) {
                ++vertexPropertyCount;
            } else if (keyword == "end_header") {
                foundEndHeader = true;
                break;
            }
        }

        if (!foundFormat || !foundVertexElement || vertexPropertyCount == 0u || !foundEndHeader) {
            reason = "PLY header is incomplete";
            return false;
        }

        // Optimization snapshots are ASCII. Count and minimally validate every
        // declared vertex before tinyply sees the file; an in-progress writer
        // otherwise exposes the final filename after only the header exists.
        if (isAscii) {
            for (std::uint64_t vertexIndex = 0u; vertexIndex < vertexCount; ++vertexIndex) {
                if (!std::getline(input, line)) {
                    reason =
                        "only " + std::to_string(vertexIndex) + " of " +
                        std::to_string(vertexCount) + " declared vertices are present";
                    return false;
                }
                std::istringstream vertexTokens(line);
                std::string token;
                std::size_t tokenCount = 0u;
                while (vertexTokens >> token) {
                    ++tokenCount;
                }
                if (tokenCount < vertexPropertyCount) {
                    reason =
                        "vertex " + std::to_string(vertexIndex) + " is incomplete";
                    return false;
                }
            }
        }

        input.close();
        const std::optional<PlyFileSnapshot> after = readPlyFileSnapshot(path);
        if (!after || !samePlyFileSnapshot(*before, *after)) {
            reason = "file changed while it was being validated";
            return false;
        }

        validatedSnapshot = *after;
        return true;
    }

    [[nodiscard]] bool equivalentPaths(const std::filesystem::path& lhs, const std::filesystem::path& rhs) {
        std::error_code error;
        if (std::filesystem::equivalent(lhs, rhs, error) && !error) {
            return true;
        }
        return std::filesystem::absolute(lhs).lexically_normal() ==
               std::filesystem::absolute(rhs).lexically_normal();
    }

    [[nodiscard]] std::optional<std::filesystem::path> sceneXmlForOptimizationPointCloud(
        const std::filesystem::path& pointCloudPath) {
        const std::filesystem::path pointsDirectory = pointCloudPath.parent_path();
        if (pointsDirectory.empty() || pointsDirectory.filename() != "points") {
            return std::nullopt;
        }

        const std::filesystem::path runDirectory = pointsDirectory.parent_path();
        if (runDirectory.empty()) {
            return std::nullopt;
        }

        const std::filesystem::path scenePath = runDirectory / "scene.xml";
        std::error_code error;
        if (!std::filesystem::is_regular_file(scenePath, error) || error) {
            return std::nullopt;
        }
        return scenePath;
    }



    bool drawPlyBrowser(
        const char* popupTitle,
        const char* childId,
        bool& browserOpen,
        std::filesystem::path& browserDirectory,
        std::filesystem::path& selectedPath) {
        bool selected = false;
        if (browserOpen) {
            ImGui::OpenPopup(popupTitle);
        }

        if (ImGui::BeginPopupModal(popupTitle, &browserOpen, ImGuiWindowFlags_AlwaysAutoResize)) {
            std::error_code error;
            if (!std::filesystem::exists(browserDirectory, error)) {
                browserDirectory = std::filesystem::current_path(error);
            }
            if (!std::filesystem::is_directory(browserDirectory, error)) {
                browserDirectory = browserDirectory.parent_path();
            }

            ImGui::TextWrapped("%s", browserDirectory.string().c_str());
            if (ImGui::Button("Up")) {
                const std::filesystem::path parent = browserDirectory.parent_path();
                if (!parent.empty()) {
                    browserDirectory = parent;
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Close")) {
                browserOpen = false;
                ImGui::CloseCurrentPopup();
            }

            ImGui::Separator();
            if (ImGui::BeginChild(childId, ImVec2(560.0f, 380.0f), true)) {
                for (const FileBrowserEntry& entry : listBrowserEntries(browserDirectory)) {
                    const std::string label =
                        entry.directory ? "[dir] " + entry.path.filename().string()
                                        : entry.path.filename().string();
                    if (ImGui::Selectable(label.c_str(), false)) {
                        if (entry.directory) {
                            browserDirectory = entry.path;
                        } else {
                            selectedPath = entry.path;
                            selected = true;
                            browserOpen = false;
                            ImGui::CloseCurrentPopup();
                        }
                    }
                }
            }
            ImGui::EndChild();
            ImGui::EndPopup();
        }

        return selected;
    }

    [[nodiscard]] std::vector<Pale::Entity> collectAreaLights(const std::shared_ptr<Pale::Scene>& scene) {
        std::vector<Pale::Entity> lights;
        for (auto [entityHandle, areaLight] : scene->getAllEntitiesWith<Pale::AreaLightComponent>().each()) {
            (void)areaLight;
            lights.emplace_back(entityHandle, scene.get());
        }
        return lights;
    }

    [[nodiscard]] std::string entityName(Pale::Entity entity) {
        if (!entity || !entity.hasComponent<Pale::TagComponent>()) {
            return "Entity";
        }
        return entity.getName();
    }

    [[nodiscard]] std::optional<Pale::AssetHandle> firstPointCloudHandle(
        const std::shared_ptr<Pale::Scene>& scene) {
        for (auto [entityHandle, pointCloud] : scene->getAllEntitiesWith<Pale::PointCloudComponent>().each()) {
            (void)entityHandle;
            return pointCloud.pointCloudID;
        }
        return std::nullopt;
    }

    [[nodiscard]] Pale::Entity firstPointCloudEntity(const std::shared_ptr<Pale::Scene>& scene) {
        for (auto [entityHandle, pointCloud] : scene->getAllEntitiesWith<Pale::PointCloudComponent>().each()) {
            (void)pointCloud;
            return Pale::Entity(entityHandle, scene.get());
        }
        return {};
    }

    [[nodiscard]] std::vector<std::size_t> collectSurfelPowerIndices(
        const Pale::PointGeometry& pointGeometry,
        bool nonZeroPower) {
        std::vector<std::size_t> indices;
        const std::size_t count = pointGeometry.powers.size();
        for (std::size_t index = 0; index < count; ++index) {
            const bool isEmitter = pointGeometry.powers[index] > 0.0f;
            if (isEmitter == nonZeroPower) {
                indices.push_back(index);
            }
        }
        return indices;
    }

    [[nodiscard]] glm::quat normalizeQuaternionOrIdentity(glm::quat quaternion);

    using viewer::PickRay;

    struct PickResult {
        int surfelIndex = -1;
        float t = std::numeric_limits<float>::infinity();
    };

    struct EditableSurfelRef {
        Pale::PointGeometry* pointGeometry = nullptr;
        std::size_t localIndex = 0u;
    };

    [[nodiscard]] glm::mat4 syclMatrixToGlm(const Pale::float4x4& matrix) {
        glm::mat4 result{1.0f};
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                result[c][r] = matrix.row[r][c];
            }
        }
        return result;
    }

    [[nodiscard]] bool intersectSurfelForPick(
        const PickRay& ray,
        const glm::mat4& pointCloudTransform,
        const Pale::PointGeometry& pointGeometry,
        std::size_t localIndex,
        int editorIndex,
        PickResult& bestHit) {
        if (localIndex >= pointGeometry.positions.size() || localIndex >= pointGeometry.quat.size() || localIndex >= pointGeometry.scales.size()) {
            return false;
        }
        if (localIndex < pointGeometry.opacities.size() && pointGeometry.opacities[localIndex] <= 0.0f) {
            return false;
        }

        glm::vec3 tangentU;
        glm::vec3 tangentV;
        const glm::mat3 rotation = glm::mat3_cast(normalizeQuaternionOrIdentity(pointGeometry.quat[localIndex]));
        tangentU = glm::normalize(glm::vec3(rotation[0]));
        tangentV = glm::normalize(glm::vec3(rotation[1]) - tangentU * glm::dot(glm::vec3(rotation[1]), tangentU));
        const glm::mat3 pointCloudLinear{pointCloudTransform};
        const glm::vec3 surfelCenter = glm::vec3(pointCloudTransform * glm::vec4(pointGeometry.positions[localIndex], 1.0f));
        const glm::vec2 surfelScale = glm::max(pointGeometry.scales[localIndex], glm::vec2(1.0e-8f));
        const glm::vec3 basisU = pointCloudLinear * (tangentU * surfelScale.x);
        const glm::vec3 basisV = pointCloudLinear * (tangentV * surfelScale.y);
        const glm::vec3 normal = glm::cross(basisU, basisV);
        const float normalLengthSquared = glm::dot(normal, normal);
        if (normalLengthSquared <= 1.0e-20f || !std::isfinite(normalLengthSquared)) {
            return false;
        }

        const glm::vec3 unitNormal = normal / std::sqrt(normalLengthSquared);
        const float denom = glm::dot(unitNormal, ray.direction);
        if (std::abs(denom) <= 1.0e-8f) {
            return false;
        }
        const float t = glm::dot(unitNormal, surfelCenter - ray.origin) / denom;
        if (t <= 0.0f || t >= bestHit.t) {
            return false;
        }

        const glm::vec3 rel = ray.origin + ray.direction * t - surfelCenter;
        const float a = glm::dot(basisU, basisU);
        const float b = glm::dot(basisU, basisV);
        const float c = glm::dot(basisV, basisV);
        const float d = glm::dot(rel, basisU);
        const float e = glm::dot(rel, basisV);
        const float determinant = a * c - b * b;
        if (std::abs(determinant) <= 1.0e-20f) {
            return false;
        }

        const float u = (d * c - e * b) / determinant;
        const float v = (a * e - b * d) / determinant;
        if (u * u + v * v > 1.0f) {
            return false;
        }

        bestHit = {.surfelIndex = editorIndex, .t = t};
        return true;
    }

    [[nodiscard]] std::optional<int> pickEditableSurfel(
        const std::shared_ptr<Pale::Scene>& scene,
        Pale::AssetAccessFromManager& assetAccessor,
        const PickRay& ray) {
        const std::optional<Pale::AssetHandle> pointCloudHandle = firstPointCloudHandle(scene);
        const std::shared_ptr<Pale::PointAsset> pointCloudAsset =
            pointCloudHandle ? assetAccessor.getPointCloud(*pointCloudHandle) : nullptr;
        if (!pointCloudAsset || countSurfels(*pointCloudAsset) == 0u) {
            return std::nullopt;
        }

        Pale::Entity pointCloudEntity = firstPointCloudEntity(scene);
        glm::mat4 pointCloudTransform{1.0f};
        if (pointCloudEntity && pointCloudEntity.hasComponent<Pale::TransformComponent>()) {
            pointCloudTransform = pointCloudEntity.getComponent<Pale::TransformComponent>().getTransform();
        }

        PickResult bestHit;
        int editorIndex = 0;
        for (const Pale::PointGeometry& pointGeometry : pointCloudAsset->points) {
            for (std::size_t localIndex = 0; localIndex < pointGeometry.positions.size(); ++localIndex) {
                (void)intersectSurfelForPick(ray, pointCloudTransform, pointGeometry, localIndex, editorIndex, bestHit);
                ++editorIndex;
            }
        }
        return bestHit.surfelIndex >= 0 ? std::optional<int>{bestHit.surfelIndex} : std::nullopt;
    }

    [[nodiscard]] std::optional<EditableSurfelRef> resolveEditableSurfel(
        Pale::PointAsset& pointAsset,
        int editorIndex) {
        if (editorIndex < 0) {
            return std::nullopt;
        }

        std::size_t localIndex = static_cast<std::size_t>(editorIndex);
        for (Pale::PointGeometry& pointGeometry : pointAsset.points) {
            if (localIndex < pointGeometry.positions.size()) {
                return EditableSurfelRef{.pointGeometry = &pointGeometry, .localIndex = localIndex};
            }
            localIndex -= pointGeometry.positions.size();
        }
        return std::nullopt;
    }

    [[nodiscard]] glm::quat normalizeQuaternionOrIdentity(glm::quat quaternion) {
        const float lengthSquared = glm::dot(quaternion, quaternion);
        if (lengthSquared <= 1.0e-20f || !std::isfinite(lengthSquared)) {
            return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        }
        quaternion = glm::normalize(quaternion);
        return quaternion.w < 0.0f ? -quaternion : quaternion;
    }

    [[nodiscard]] glm::quat extractRotationQuaternion(const glm::mat4& transform) {
        glm::vec3 tangentU = glm::vec3(transform[0]);
        glm::vec3 tangentV = glm::vec3(transform[1]);

        if (glm::dot(tangentU, tangentU) <= 1.0e-20f || glm::dot(tangentV, tangentV) <= 1.0e-20f) {
            return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        }

        tangentU = glm::normalize(tangentU);
        tangentV = tangentV - glm::dot(tangentV, tangentU) * tangentU;

        if (glm::dot(tangentV, tangentV) <= 1.0e-20f) {
            const glm::vec3 fallback =
                glm::abs(tangentU.y) < 0.9f
                    ? glm::vec3(0.0f, 1.0f, 0.0f)
                    : glm::vec3(1.0f, 0.0f, 0.0f);
            tangentV = fallback - glm::dot(fallback, tangentU) * tangentU;
        }

        tangentV = glm::normalize(tangentV);
        const glm::vec3 normal = glm::normalize(glm::cross(tangentU, tangentV));
        return normalizeQuaternionOrIdentity(glm::quat_cast(glm::mat3(tangentU, tangentV, normal)));
    }

    [[nodiscard]] std::vector<Pale::Entity> collectRuntimeMeshEntities(
        const std::shared_ptr<Pale::Scene>& scene) {
        std::vector<Pale::Entity> meshes;
        for (auto [entityHandle, meshComponent, tagComponent] :
             scene->getAllEntitiesWith<Pale::MeshComponent, Pale::TagComponent>().each()) {
            (void)meshComponent;
            if (tagComponent.tag.rfind("RuntimeMesh", 0u) == 0u) {
                meshes.emplace_back(entityHandle, scene.get());
            }
        }
        return meshes;
    }

    AppArgs parseArgs(int argc, char** argv) {
        AppArgs args;
        std::vector<std::string> positional;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto requireValue = [&](const char* option) -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error(std::string("Missing value for ") + option);
                }
                return argv[++i];
            };

            if (arg == "--assets") {
                args.assetsDir = requireValue("--assets");
            } else if (arg == "--pointcloud") {
                args.pointCloudPath = requireValue("--pointcloud");
                args.sceneInputSpecified = true;
            } else if (arg == "--scene") {
                args.scenePath = requireValue("--scene");
                args.sceneInputSpecified = true;
            } else if (arg == "--width") {
                args.width = static_cast<uint32_t>(std::max(1, std::stoi(requireValue("--width"))));
            } else if (arg == "--height") {
                args.height = static_cast<uint32_t>(std::max(1, std::stoi(requireValue("--height"))));
            } else if (arg == "--shared-slabs") {
                args.sharedSlabs = true;
            } else {
                positional.push_back(arg);
            }
        }

        if (!positional.empty()) {
            args.pointCloudPath = positional[0];
            args.sceneInputSpecified = true;
        }
        if (positional.size() > 1) {
            args.scenePath = positional[1];
        }
        if (!args.scenePath.has_extension()) {
            args.scenePath.replace_extension(".xml");
        }

        return args;
    }

    SceneBounds computeSceneBounds(const Pale::SceneBuild::BuildProducts& buildProducts) {
        if (buildProducts.topLevelNodes.empty()) {
            return {};
        }

        const Pale::TLASNode& root = buildProducts.topLevelNodes.front();
        const glm::vec3 minP = Pale::sycl2glm(root.aabbMin);
        const glm::vec3 maxP = Pale::sycl2glm(root.aabbMax);
        const glm::vec3 center = 0.5f * (minP + maxP);
        const float radius = std::max(glm::length(maxP - minP) * 0.5f, 0.001f);
        return {.center = center, .radius = radius};
    }

    OrbitCamera makeInitialOrbitCamera(
        const Pale::SceneBuild::BuildProducts& buildProducts,
        const SceneBounds& bounds) {
        OrbitCamera orbit;
        orbit.target = kDefaultLookAt;
        orbit.distance = bounds.radius * 2.5f;
        orbit.farClip = std::max(1000.0f, bounds.radius * 20.0f);

        glm::vec3 cameraOffset{orbit.distance, 0.0f, 0.0f};
        if (!buildProducts.cameraGPUs.empty()) {
            const Pale::CameraGPU& firstCamera = buildProducts.cameraGPUs.front();
            const glm::vec3 cameraPosition = Pale::sycl2glm(firstCamera.pos);
            cameraOffset = cameraPosition - orbit.target;
            orbit.distance = std::max(glm::length(cameraOffset), bounds.radius * 0.5f);
            // Keep the viewport's default FOV when borrowing the scene camera position.
        }

        const float safeDistance = std::max(glm::length(cameraOffset), 0.001f);
        orbit.pitch = std::clamp(
            std::asin(std::clamp(cameraOffset.z / safeDistance, -1.0f, 1.0f)),
            -1.50f,
            1.50f);
        orbit.yaw = std::atan2(cameraOffset.y, cameraOffset.x);
        return orbit;
    }

    OrbitCamera makeOrbitCameraFromSceneCamera(
        const Pale::CameraGPU& camera,
        const SceneBounds& bounds,
        const OrbitCamera& fallback) {
        OrbitCamera orbit = fallback;
        const glm::vec3 cameraPosition = Pale::sycl2glm(camera.pos);
        glm::vec3 cameraForward = Pale::sycl2glm(camera.forward);
        if (!std::isfinite(cameraForward.x) ||
            !std::isfinite(cameraForward.y) ||
            !std::isfinite(cameraForward.z) ||
            glm::dot(cameraForward, cameraForward) <= 1.0e-20f) {
            const glm::vec3 centerDirection = bounds.center - cameraPosition;
            cameraForward =
                glm::dot(centerDirection, centerDirection) > 1.0e-20f
                    ? glm::normalize(centerDirection)
                    : glm::vec3(0.0f, 0.0f, -1.0f);
        } else {
            cameraForward = glm::normalize(cameraForward);
        }

        float targetDistance = glm::dot(bounds.center - cameraPosition, cameraForward);
        if (!std::isfinite(targetDistance) || targetDistance <= 0.001f) {
            targetDistance = std::max(bounds.radius * 2.5f, 0.001f);
        }

        orbit.target = cameraPosition + cameraForward * targetDistance;
        orbit.distance = targetDistance;
        orbit.farClip = std::max(orbit.farClip, bounds.radius * 20.0f);
        if (camera.hasPinholeIntrinsics != 0u && camera.fy > 0.0f && camera.height > 0) {
            orbit.fovyDegrees = glm::degrees(
                2.0f * std::atan(static_cast<float>(camera.height) / (2.0f * camera.fy)));
        } else if (camera.fovy > 0.0f) {
            orbit.fovyDegrees = camera.fovy;
        }
        orbit.setPositionKeepingTarget(cameraPosition);
        return orbit;
    }

    void registerAssetLoaders(Pale::AssetManager& assetManager) {
        assetManager.enableHotReload(true);
        assetManager.registerLoader<Pale::Mesh>(
            Pale::AssetType::Mesh,
            std::make_shared<Pale::AssimpMeshLoader>());
        assetManager.registerLoader<Pale::Material>(
            Pale::AssetType::Material,
            std::make_shared<Pale::YamlMaterialLoader>());
        assetManager.registerLoader<Pale::PointAsset>(
            Pale::AssetType::PointCloud,
            std::make_shared<Pale::PLYPointLoader>());
    }

    std::shared_ptr<Pale::Scene> loadSceneWithPointCloud(
        Pale::AssetManager& assetManager,
        const std::filesystem::path& scenePath,
        const std::filesystem::path& pointCloudPath) {
        PlyFileSnapshot validatedSnapshot{};
        std::string readinessFailure;
        if (!validatePlyReadyForLoad(pointCloudPath, validatedSnapshot, readinessFailure)) {
            throw std::runtime_error(
                "Point cloud is not ready to load (" + readinessFailure + "): " +
                pointCloudPath.string());
        }

        assetManager.registry().load("asset_registry.yaml");

        auto scene = std::make_shared<Pale::Scene>();
        Pale::AssetIndexFromRegistry assetIndexer(assetManager.registry());
        Pale::SceneSerializer serializer(scene, assetIndexer);
        if (!serializer.deserialize(scenePath)) {
            throw std::runtime_error("Failed to load scene XML: " + scenePath.string());
        }

        const Pale::AssetHandle pointCloudAssetHandle =
            importPathAsType(assetManager.registry(), pointCloudPath, Pale::AssetType::PointCloud);
        Pale::Entity pointCloudEntity = scene->createEntity("RealtimePointCloud");
        pointCloudEntity.addComponent<Pale::PointCloudComponent>().pointCloudID = pointCloudAssetHandle;

        Pale::AssetAccessFromManager assetAccessor(assetManager);
        assetManager.invalidate(pointCloudAssetHandle);
        const auto pointCloudAsset = assetAccessor.getPointCloud(pointCloudAssetHandle);
        if (!pointCloudAsset || pointCloudAsset->points.empty()) {
            throw std::runtime_error("Point cloud failed to load or contains no point blocks: " + pointCloudPath.string());
        }
        const std::optional<PlyFileSnapshot> loadedSnapshot = readPlyFileSnapshot(pointCloudPath);
        if (!loadedSnapshot || !samePlyFileSnapshot(validatedSnapshot, *loadedSnapshot)) {
            assetManager.invalidate(pointCloudAssetHandle);
            throw std::runtime_error(
                "Point cloud changed while it was being loaded; retry after the writer finishes: " +
                pointCloudPath.string());
        }

        Pale::Log::PA_INFO(
            "Loaded point cloud with {} surfels",
            pointCloudAsset->points.front().positions.size());
        return scene;
    }

    Pale::SensorGPU createSensor(sycl::queue queue, const Pale::CameraGPU& camera) {
        Pale::SensorGPU sensor{};
        sensor.camera = camera;
        sensor.width = camera.width;
        sensor.height = camera.height;
        sensor.cameraSlotIndex = 0u;
        Pale::copyName(sensor.name, "RealtimeCamera");

        const std::size_t pixelCount =
            static_cast<std::size_t>(sensor.width) * static_cast<std::size_t>(sensor.height);
        if(pixelCount == 0 || pixelCount > std::numeric_limits<std::size_t>::max()/sizeof(Pale::float4))
            throw std::runtime_error("Invalid realtime sensor resolution");
        const auto release=[&](void* pointer) { sycl::free(pointer, queue); };
        Pale::detail::AllocationBatch<18, decltype(release)> pending(release);
        const auto allocate=[&](auto*& pointer, std::size_t count) {
            using Element=std::remove_pointer_t<std::remove_reference_t<decltype(pointer)>>;
            pointer=pending.add(sycl::malloc_device<Element>(count, queue));
        };
        allocate(sensor.framebuffer, pixelCount);
        allocate(sensor.outputFramebuffer, pixelCount);
        allocate(sensor.ldrFramebuffer, pixelCount * 4u);
        allocate(sensor.depthDistortionBuffer, pixelCount);
        allocate(sensor.depthDistortionAdjointBuffer, pixelCount);
        allocate(sensor.intraSlabDepthBuffer, pixelCount);
        allocate(sensor.intraSlabRayDepthBuffer, pixelCount);
        allocate(sensor.intraSlabDepthAdjointBuffer, pixelCount);
        allocate(sensor.intraSlabDepthActiveSlabCountBuffer, pixelCount);
        allocate(sensor.visiblePrimitiveIndexBuffer, pixelCount);
        allocate(sensor.medianDepthBuffer, pixelCount);
        allocate(sensor.meanDepthBuffer, pixelCount);
        allocate(sensor.medianDepthAdjointBuffer, pixelCount);
        allocate(sensor.medianWorldPositionBuffer, pixelCount);
        allocate(sensor.visibleNormalBuffer, pixelCount);
        allocate(sensor.normalFromDepthBuffer, pixelCount);
        allocate(sensor.normalFromDepthAdjointBuffer, pixelCount);
        allocate(sensor.visibleNormalAdjointBuffer, pixelCount);
        pending.commit();
        return sensor;
    }

    void clearSensor(sycl::queue queue, Pale::SensorGPU& sensor) {
        const std::size_t pixelCount =
            static_cast<std::size_t>(sensor.width) * static_cast<std::size_t>(sensor.height);
        queue.fill(sensor.framebuffer, Pale::float4{0.0f}, pixelCount);
        queue.memset(sensor.outputFramebuffer, 0, pixelCount * sizeof(sycl::uchar4));
        queue.memset(sensor.ldrFramebuffer, 0, pixelCount * 4u * sizeof(float));
        queue.memset(sensor.depthDistortionBuffer, 0, pixelCount * sizeof(float));
        queue.memset(sensor.depthDistortionAdjointBuffer, 0, pixelCount * sizeof(float));
        queue.memset(sensor.intraSlabDepthBuffer, 0, pixelCount * sizeof(float));
        queue.memset(sensor.intraSlabRayDepthBuffer, 0, pixelCount * sizeof(float));
        queue.memset(sensor.intraSlabDepthAdjointBuffer, 0, pixelCount * sizeof(float));
        queue.memset(sensor.intraSlabDepthActiveSlabCountBuffer, 0, pixelCount * sizeof(std::uint32_t));
        queue.fill(sensor.visiblePrimitiveIndexBuffer, UINT32_MAX, pixelCount);
        queue.memset(sensor.medianDepthBuffer, 0, pixelCount * sizeof(float));
        queue.memset(sensor.meanDepthBuffer, 0, pixelCount * sizeof(float));
        queue.memset(sensor.medianDepthAdjointBuffer, 0, pixelCount * sizeof(float));
        queue.memset(sensor.medianWorldPositionBuffer, 0, pixelCount * 4u * sizeof(float));
        queue.memset(sensor.visibleNormalBuffer, 0, pixelCount * 4u * sizeof(float));
        queue.memset(sensor.normalFromDepthBuffer, 0, pixelCount * 4u * sizeof(float));
        queue.memset(sensor.normalFromDepthAdjointBuffer, 0, pixelCount * 4u * sizeof(float));
        queue.memset(sensor.visibleNormalAdjointBuffer, 0, pixelCount * 4u * sizeof(float));
        queue.wait();
    }

    template<typename T>
    void freeDevicePtr(sycl::queue queue, T*& ptr) {
        if (ptr) {
            sycl::free(ptr, queue);
            ptr = nullptr;
        }
    }

    void destroySensor(sycl::queue queue, Pale::SensorGPU& sensor) {
        freeDevicePtr(queue, sensor.framebuffer);
        freeDevicePtr(queue, sensor.relativeDensificationWeightBuffer);
        freeDevicePtr(queue, sensor.outputFramebuffer);
        freeDevicePtr(queue, sensor.ldrFramebuffer);
        freeDevicePtr(queue, sensor.depthDistortionBuffer);
        freeDevicePtr(queue, sensor.depthDistortionAdjointBuffer);
        freeDevicePtr(queue, sensor.intraSlabDepthBuffer);
        freeDevicePtr(queue, sensor.intraSlabRayDepthBuffer);
        freeDevicePtr(queue, sensor.intraSlabDepthAdjointBuffer);
        freeDevicePtr(queue, sensor.intraSlabDepthActiveSlabCountBuffer);
        freeDevicePtr(queue, sensor.visiblePrimitiveIndexBuffer);
        freeDevicePtr(queue, sensor.medianDepthBuffer);
        freeDevicePtr(queue, sensor.meanDepthBuffer);
        freeDevicePtr(queue, sensor.medianDepthAdjointBuffer);
        freeDevicePtr(queue, sensor.medianWorldPositionBuffer);
        freeDevicePtr(queue, sensor.visibleNormalBuffer);
        freeDevicePtr(queue, sensor.normalFromDepthBuffer);
        freeDevicePtr(queue, sensor.normalFromDepthAdjointBuffer);
        freeDevicePtr(queue, sensor.visibleNormalAdjointBuffer);
        sensor = Pale::SensorGPU{};
        queue.wait();
    }

    void prepareRealtimeViewerRgbLossAdjointSource(
        sycl::queue queue,
        const Pale::SensorGPU& sensor,
        float& lossOut) {
        if (!sensor.framebuffer) {
            throw std::runtime_error("prepareRealtimeViewerRgbLossAdjointSource: missing framebuffer");
        }

        const std::size_t pixelCount =
            static_cast<std::size_t>(sensor.width) * static_cast<std::size_t>(sensor.height);
        const float invElementCount =
            pixelCount > 0u ? 1.0f / (static_cast<float>(pixelCount) * 3.0f) : 0.0f;

        std::vector<float> framebuffer = Pale::downloadSensorRGBARAW(queue, sensor);
        lossOut = 0.0f;
        for (std::size_t pixelIndex = 0; pixelIndex < pixelCount; ++pixelIndex) {
            const std::size_t rgbaIndex = pixelIndex * 4u;
            const float diffR = framebuffer[rgbaIndex + 0u];
            const float diffG = framebuffer[rgbaIndex + 1u];
            const float diffB = framebuffer[rgbaIndex + 2u];
            lossOut += 0.5f * (diffR * diffR + diffG * diffG + diffB * diffB) * invElementCount;

            framebuffer[rgbaIndex + 0u] = diffR * invElementCount;
            framebuffer[rgbaIndex + 1u] = diffG * invElementCount;
            framebuffer[rgbaIndex + 2u] = diffB * invElementCount;
            framebuffer[rgbaIndex + 3u] = 0.0f;
        }
        Pale::uploadSensorRGBA(queue, sensor, std::move(framebuffer));
    }

    void prepareRealtimeViewerSurfaceRegularizerAdjoints(
        sycl::queue queue,
        const Pale::SensorGPU& sensor,
        ViewImageMode mode) {
        const std::size_t pixelCount =
            static_cast<std::size_t>(sensor.width) * static_cast<std::size_t>(sensor.height);
        if (pixelCount == 0u) {
            return;
        }

        // Unit loss weight with the same normalization as training.
        if (mode == ViewImageMode::DepthPositionGradient) {
            queue.fill(sensor.depthDistortionAdjointBuffer,
                1.0f / static_cast<float>(pixelCount), pixelCount).wait();
            return;
        }

        if (mode == ViewImageMode::IntraSlabPositionGradient) {
            const std::vector<uint32_t> intraSlabCounts = Pale::downloadUint32Buffer(
                queue, sensor.intraSlabDepthActiveSlabCountBuffer, pixelCount);
            uint64_t totalActiveSlabs = 0u;
            for (const uint32_t count : intraSlabCounts) {
                totalActiveSlabs += count;
            }
            const float inverseActiveSlabCount =
                1.0f / static_cast<float>(std::max<uint64_t>(totalActiveSlabs, 1u));
            std::vector<float> intraSlabAdjoints(pixelCount, 0.0f);
            for (std::size_t pixelIndex = 0u; pixelIndex < pixelCount; ++pixelIndex) {
                if (intraSlabCounts[pixelIndex] > 0u) {
                    intraSlabAdjoints[pixelIndex] = inverseActiveSlabCount;
                }
            }
            queue.memcpy(
                sensor.intraSlabDepthAdjointBuffer,
                intraSlabAdjoints.data(),
                pixelCount * sizeof(float));
            queue.wait();
            return;
        }

        const std::vector<float> visibleNormals = Pale::downloadFloat4Buffer(
            queue, sensor.visibleNormalBuffer, pixelCount);
        const std::vector<float> depthNormals = Pale::downloadFloat4Buffer(
            queue, sensor.normalFromDepthBuffer, pixelCount);
        uint32_t validNormalCount = 0u;
        for (std::size_t pixelIndex = 0u; pixelIndex < pixelCount; ++pixelIndex) {
            const std::size_t baseIndex = pixelIndex * 4u;
            if (visibleNormals[baseIndex + 3u] > 0.0f &&
                depthNormals[baseIndex + 3u] > 0.0f) {
                ++validNormalCount;
            }
        }
        const float inverseValidNormalCount =
            1.0f / static_cast<float>(std::max(validNormalCount, 1u));
        std::vector<Pale::float4> visibleNormalAdjoints(pixelCount, Pale::float4{0.0f});
        std::vector<Pale::float4> depthNormalAdjoints(pixelCount, Pale::float4{0.0f});
        for (std::size_t pixelIndex = 0u; pixelIndex < pixelCount; ++pixelIndex) {
            const std::size_t baseIndex = pixelIndex * 4u;
            if (visibleNormals[baseIndex + 3u] <= 0.0f ||
                depthNormals[baseIndex + 3u] <= 0.0f) {
                continue;
            }
            visibleNormalAdjoints[pixelIndex] = Pale::float4{
                -inverseValidNormalCount * depthNormals[baseIndex + 0u],
                -inverseValidNormalCount * depthNormals[baseIndex + 1u],
                -inverseValidNormalCount * depthNormals[baseIndex + 2u],
                0.0f};
            depthNormalAdjoints[pixelIndex] = Pale::float4{
                -inverseValidNormalCount * visibleNormals[baseIndex + 0u],
                -inverseValidNormalCount * visibleNormals[baseIndex + 1u],
                -inverseValidNormalCount * visibleNormals[baseIndex + 2u],
                0.0f};
        }
        queue.memcpy(
            sensor.visibleNormalAdjointBuffer,
            visibleNormalAdjoints.data(),
            pixelCount * sizeof(Pale::float4));
        queue.memcpy(
            sensor.normalFromDepthAdjointBuffer,
            depthNormalAdjoints.data(),
            pixelCount * sizeof(Pale::float4));
        queue.wait();
    }

    std::string toLower(std::string value) {
        for (auto& c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return value;
    }

    std::vector<float> loadDebugTarget(const std::filesystem::path& path,
            const Pale::CameraGPU& camera, bool srgb) {
        std::vector<float> rgba;
        uint32_t width = 0, height = 0;
        if (toLower(path.extension().string()) == ".exr") {
            Pale::Utils::loadEXRAsRGBAFloat(path, rgba, width, height);
        } else {
            int w = 0, h = 0, channels = 0;
            const auto filename = path.string();
            auto copy = [&](auto* data, float scale) {
                if (!data) throw std::runtime_error("Cannot load target " + filename + ": " + stbi_failure_reason());
                rgba.resize(static_cast<std::size_t>(w) * h * 4);
                for (std::size_t i = 0; i < rgba.size(); ++i) rgba[i] = float(data[i]) * scale;
                stbi_image_free(data);
                width = static_cast<uint32_t>(w);
                height = static_cast<uint32_t>(h);
            };
            if (stbi_is_hdr(filename.c_str())) copy(stbi_loadf(filename.c_str(), &w, &h, &channels, 4), 1.0f);
            else if (stbi_is_16_bit(filename.c_str())) copy(stbi_load_16(filename.c_str(), &w, &h, &channels, 4), 1.0f / 65535.0f);
            else copy(stbi_load(filename.c_str(), &w, &h, &channels, 4), 1.0f / 255.0f);
        }
        if (width != camera.width || height != camera.height)
            throw std::runtime_error("Target dimensions disagree with scene camera: " + path.string());
        if (srgb) {
            for (std::size_t i = 0; i < rgba.size(); ++i) {
                if (i % 4 == 3) continue; // Training ignores alpha.
                const float value = rgba[i];
                rgba[i] = value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
            }
        }
        return rgba;
    }

    Pale::PathTracerSettings makeDefaultSettings() {
        Pale::PathTracerSettings settings{};
        settings.integratorKind = Pale::IntegratorKind::photonMapping;
        settings.photonsPerLaunch = 65536u;
        settings.maxBounces = 0;
        settings.maxAdjointBounces = 0;
        settings.numForwardPasses = 0;
        settings.numShadowRays = 1u;
        settings.numAdjointShadowRays = 0;
        settings.adjointSamplesPerPixel = 0;
        settings.numGatherPasses = 1u;
        settings.renderDebugGradientImages = false;
        settings.depthDistortionWorldSpace = true;
        settings.medianDepthRetainLastHit = false;
        settings.enableAdjointDirectLight = true;
        settings.rendererDebugShareLocalLayerDirectLighting = true;
        settings.pointGeometrySupportRadius = 0.00f;
        settings.pointGeometryReconstructionLength = 0.0f;
        settings.pointGeometryRayOffsetMultiplier = 1.0f;
        settings.pointGeometryCoverageScale = 1.0f;
        settings.pointGeometryMinimumContributors = 1u;
        settings.pointGeometryDebugShowAlbedo = false;
        return settings;
    }

    Pale::PathTracerSettings makeViewerDebugSettings(
        Pale::PathTracerSettings settings, ViewImageMode mode) {
        if (settings.sharedHeightEnabled) {
            settings.integratorKind = Pale::IntegratorKind::photonMapping;
        }
        settings.computeSurfaceDiagnostics =
            mode != ViewImageMode::Rendered &&
            mode != ViewImageMode::RawLinear &&
            mode != ViewImageMode::TerminalTransmittance &&
            mode != ViewImageMode::SurfelDensity &&
            mode != ViewImageMode::SurfaceOverlap &&
            mode != ViewImageMode::TargetLoss &&
            mode != ViewImageMode::TransportScore &&
            mode != ViewImageMode::PositionPrimitiveScore &&
            mode != ViewImageMode::AdjointDerivative;
        settings.computeDepthNormalDiagnostics =
            mode == ViewImageMode::MedianDepthNormal ||
            mode == ViewImageMode::MeanDepthNormal ||
            mode == ViewImageMode::NormalPositionGradient;
        if (mode == ViewImageMode::MedianDepthNormal || mode == ViewImageMode::MeanDepthNormal) {
            settings.normalFromDepthUseMeanDepth = mode == ViewImageMode::MeanDepthNormal;
        }
        settings.computeVisiblePrimitiveDiagnostics =
            requiresVisibleSlabSearch(mode);
        // The legacy gather kernel requests depth normals through this weight.
        settings.normalConsistencyWeight = settings.computeDepthNormalDiagnostics ? 1.0f : 0.0f;
        return settings;
    }

    [[nodiscard]] uint32_t renderExtentFromAvailable(float available) {
        return static_cast<uint32_t>(std::clamp(
            static_cast<int>(std::floor(std::max(available, 1.0f))),
            16,
            4096));
    }

    [[nodiscard]] std::array<int, 2> initialViewerWindowSize(
        uint32_t targetRenderWidth,
        uint32_t targetRenderHeight) {
        constexpr float renderDockFraction = 1.0f - kSidebarDockFraction;
        constexpr float renderDockHorizontalChrome = 17.0f;
        constexpr int renderDockVerticalChrome = 54;
        const int windowWidth = static_cast<int>(std::ceil(
            (static_cast<float>(targetRenderWidth) + renderDockHorizontalChrome) /
            renderDockFraction));
        const int windowHeight =
            static_cast<int>(targetRenderHeight) + renderDockVerticalChrome;
        return {
            std::clamp(windowWidth, 1120, 1920),
            std::clamp(windowHeight, 720, 1200),
        };
    }

    using viewer::channelToByte;

    void composeViewportPixels(
        const std::vector<float>& rawRenderPixels,
        const viewer::DisplayTransform& displayTransform,
        const OrbitCamera& orbit,
        const Pale::CameraGPU& displayedCamera,
        const SceneBounds& bounds,
        uint32_t renderWidth,
        uint32_t renderHeight,
        bool showViewportGrid,
        bool forceRenderOpacity,
        const ImVec4& backgroundColor,
        viewer::ViewportBackgroundCache& backgroundCache,
        std::vector<uint8_t>& displayPixels) {
        const std::size_t pixelCount =
            static_cast<std::size_t>(renderWidth) * static_cast<std::size_t>(renderHeight);
        if (rawRenderPixels.size() != pixelCount * 4u) return;
        const bool intrinsics=displayedCamera.hasPinholeIntrinsics && displayedCamera.fx>0 && displayedCamera.fy>0;
        const float focal=0.5f*displayedCamera.height/std::tan(0.5f*glm::radians(displayedCamera.fovy));
        const viewer::GridProjection projection{syclMatrixToGlm(displayedCamera.invView),
            intrinsics?displayedCamera.fx:focal,intrinsics?displayedCamera.fy:focal,
            intrinsics?displayedCamera.cx:0.5f*displayedCamera.width,
            intrinsics?displayedCamera.cy:0.5f*displayedCamera.height};
        displayPixels = backgroundCache.get(orbit, bounds, renderWidth, renderHeight,
                                            showViewportGrid, backgroundColor, projection);
        viewer::compositeLinearOverBackground(rawRenderPixels, displayPixels,
                                             displayTransform, forceRenderOpacity);
    }

    [[nodiscard]] glm::vec3 lerpColor(const glm::vec3& a, const glm::vec3& b, float t) {
        return a * (1.0f - t) + b * t;
    }

    [[nodiscard]] glm::vec3 viridisColor(float value) {
        const float t = std::clamp(value, 0.0f, 1.0f);

        static const std::array<glm::vec3, 6> stops{{
            {0.267004f, 0.004874f, 0.329415f},
            {0.253935f, 0.265254f, 0.529983f},
            {0.163625f, 0.471133f, 0.558148f},
            {0.134692f, 0.658636f, 0.517649f},
            {0.477504f, 0.821444f, 0.318195f},
            {0.993248f, 0.906157f, 0.143936f},
        }};
        const float scaled = t * static_cast<float>(stops.size() - 1u);
        const std::size_t lowerIndex =
            std::min(static_cast<std::size_t>(scaled), stops.size() - 2u);
        const float localT = scaled - static_cast<float>(lowerIndex);
        return lerpColor(stops[lowerIndex], stops[lowerIndex + 1u], localT);
    }

    [[nodiscard]] glm::vec3 jetColor(float value) {
        const float t = std::clamp(value, 0.0f, 1.0f);
        return {
            std::clamp(1.5f - std::abs(4.0f * t - 3.0f), 0.0f, 1.0f),
            std::clamp(1.5f - std::abs(4.0f * t - 2.0f), 0.0f, 1.0f),
            std::clamp(1.5f - std::abs(4.0f * t - 1.0f), 0.0f, 1.0f),
        };
    }

    [[nodiscard]] glm::vec3 scalarColor(float value, ScalarColorMap colorMap) {
        switch (colorMap) {
            case ScalarColorMap::Viridis:
                return viridisColor(value);
            case ScalarColorMap::Jet:
                return jetColor(value);
        }
        return viridisColor(value);
    }

    void colorizeScalarBuffer(
        const std::vector<float>& values,
        uint32_t renderWidth,
        uint32_t renderHeight,
        bool invert,
        bool logScale,
        ScalarColorMap colorMap,
        std::vector<uint8_t>& displayPixels,
        float sharedMaximum = 0.0f,
        bool includeZero = false) {
        const std::size_t pixelCount =
            static_cast<std::size_t>(renderWidth) * static_cast<std::size_t>(renderHeight);
        displayPixels.assign(pixelCount * 4u, 0u);
        if (values.size() < pixelCount) {
            return;
        }

        float minValue = 0.0f;
        float maxValue = 0.0f;
        bool hasValue = false;
        for (std::size_t pixelIndex = 0; pixelIndex < pixelCount; ++pixelIndex) {
            const float value = values[pixelIndex];
            if (!std::isfinite(value) || value < 0.0f || (!includeZero && value == 0.0f)) {
                continue;
            }

            const float mappedValue = logScale ? std::log1p(value) : value;
            if (!hasValue) {
                minValue = mappedValue;
                maxValue = mappedValue;
                hasValue = true;
            } else {
                minValue = std::min(minValue, mappedValue);
                maxValue = std::max(maxValue, mappedValue);
            }
        }

        if (sharedMaximum > 0.0f && std::isfinite(sharedMaximum)) {
            minValue = 0.0f;
            maxValue = logScale ? std::log1p(sharedMaximum) : sharedMaximum;
            hasValue = true;
        }
        if (!hasValue) {
            for (std::size_t pixelIndex = 0; pixelIndex < pixelCount; ++pixelIndex) {
                displayPixels[pixelIndex * 4u + 3u] = 255u;
            }
            return;
        }

        const float range = maxValue - minValue;
        const float inverseRange = range > 1.0e-12f ? 1.0f / range : 0.0f;
        for (std::size_t pixelIndex = 0; pixelIndex < pixelCount; ++pixelIndex) {
            const std::size_t baseIndex = pixelIndex * 4u;
            const float value = values[pixelIndex];
            displayPixels[baseIndex + 3u] = 255u;
            if (!std::isfinite(value) || value < 0.0f || (!includeZero && value == 0.0f)) {
                continue;
            }

            const float mappedValue = logScale ? std::log1p(value) : value;
            float normalized = range > 1.0e-12f ? (mappedValue - minValue) * inverseRange : 1.0f;
            normalized = std::clamp(normalized, 0.0f, 1.0f);
            if (invert) {
                normalized = 1.0f - normalized;
            }

            const glm::vec3 color = scalarColor(normalized, colorMap);
            displayPixels[baseIndex + 0u] = channelToByte(color.r);
            displayPixels[baseIndex + 1u] = channelToByte(color.g);
            displayPixels[baseIndex + 2u] = channelToByte(color.b);
        }
    }

    void colorizeScalarBufferFixedRange(
        const std::vector<float>& values,
        uint32_t renderWidth,
        uint32_t renderHeight,
        float displayMinimum,
        float displayMaximum,
        ScalarColorMap colorMap,
        std::vector<uint8_t>& displayPixels) {
        const std::size_t pixelCount =
            static_cast<std::size_t>(renderWidth) * static_cast<std::size_t>(renderHeight);
        displayPixels.assign(pixelCount * 4u, 0u);
        if (values.size() < pixelCount) {
            return;
        }

        const float inverseRange =
            1.0f / std::max(displayMaximum - displayMinimum, 1.0e-12f);
        for (std::size_t pixelIndex = 0u; pixelIndex < pixelCount; ++pixelIndex) {
            const std::size_t baseIndex = pixelIndex * 4u;
            displayPixels[baseIndex + 3u] = 255u;
            if (!std::isfinite(values[pixelIndex])) {
                continue;
            }
            const float normalized = std::clamp(
                (values[pixelIndex] - displayMinimum) * inverseRange, 0.0f, 1.0f);
            const glm::vec3 color = scalarColor(normalized, colorMap);
            displayPixels[baseIndex + 0u] = channelToByte(color.r);
            displayPixels[baseIndex + 1u] = channelToByte(color.g);
            displayPixels[baseIndex + 2u] = channelToByte(color.b);
        }
    }

    [[nodiscard]] float linearToSrgb(float value) {
        const float clamped = std::clamp(value, 0.0f, 1.0f);
        return clamped <= 0.0031308f
                   ? 12.92f * clamped
                   : 1.055f * std::pow(clamped, 1.0f / 2.4f) - 0.055f;
    }


    void colorizeThresholdedPrimitiveScores(
        const std::vector<float>& values,
        uint32_t renderWidth,
        uint32_t renderHeight,
        float splitThreshold,
        ScalarColorMap colorMap,
        std::vector<uint8_t>& displayPixels,
        bool showMissingSamples = false) {
        const std::size_t pixelCount =
            static_cast<std::size_t>(renderWidth) * static_cast<std::size_t>(renderHeight);
        displayPixels.assign(pixelCount * 4u, 0u);
        if (values.size() < pixelCount) {
            return;
        }

        const bool thresholdEnabled = std::isfinite(splitThreshold) && splitThreshold > 0.0f;
        float safeThreshold = thresholdEnabled ? splitThreshold : 1.0f;
        if (!thresholdEnabled) {
            // Disabled densification still shows scores, without marking split candidates.
            for (const float value : values) {
                if (std::isfinite(value)) {
                    safeThreshold = std::max(safeThreshold, value);
                }
            }
        }
        safeThreshold = std::max(safeThreshold, 1.0e-12f);
        constexpr glm::vec3 kThresholdCrossedColor{1.0f, 0.0f, 1.0f};
        for (std::size_t pixelIndex = 0; pixelIndex < pixelCount; ++pixelIndex) {
            const std::size_t baseIndex = pixelIndex * 4u;
            displayPixels[baseIndex + 3u] = 255u;
            const float value = values[pixelIndex];
            if (showMissingSamples && value == -2.0f) {
                // A visible primitive without interval samples is distinct
                // from the background (-1) and a measured zero score (0).
                displayPixels[baseIndex + 0u] = 128u;
                displayPixels[baseIndex + 1u] = 128u;
                displayPixels[baseIndex + 2u] = 128u;
                continue;
            }
            if (!std::isfinite(value) || value < 0.0f) {
                continue;
            }
            const float normalized = std::clamp(value / safeThreshold, 0.0f, 1.0f);
            const glm::vec3 color = thresholdEnabled && value >= safeThreshold
                ? kThresholdCrossedColor
                : scalarColor(normalized, colorMap);
            displayPixels[baseIndex + 0u] = channelToByte(color.r);
            displayPixels[baseIndex + 1u] = channelToByte(color.g);
            displayPixels[baseIndex + 2u] = channelToByte(color.b);
        }
    }

    void colorizeDensificationOrigins(
        const std::vector<std::uint8_t>& origins,
        uint32_t renderWidth,
        uint32_t renderHeight,
        std::vector<uint8_t>& displayPixels) {
        constexpr std::uint8_t kNoVisibleSurfel = 255u;
        const std::size_t pixelCount =
            static_cast<std::size_t>(renderWidth) * static_cast<std::size_t>(renderHeight);
        displayPixels.assign(pixelCount * 4u, 0u);
        if (origins.size() < pixelCount) {
            return;
        }

        for (std::size_t pixelIndex = 0u; pixelIndex < pixelCount; ++pixelIndex) {
            const std::uint8_t origin = origins[pixelIndex];
            if (origin == kNoVisibleSurfel) {
                continue;
            }

            // Provenance is categorical: 0=initial/unknown, 1=clone,
            // 2=position-gradient split; historical origin 3 is shown as a split.
            glm::vec3 color{0.40f};
            if (origin == 1u) {
                color = glm::vec3(0.15f, 0.78f, 0.30f);
            } else if (origin == 2u || origin == 3u) {
                color = glm::vec3(0.18f, 0.48f, 1.00f);

            }

            const std::size_t baseIndex = pixelIndex * 4u;
            displayPixels[baseIndex + 0u] = channelToByte(color.r);
            displayPixels[baseIndex + 1u] = channelToByte(color.g);
            displayPixels[baseIndex + 2u] = channelToByte(color.b);
            displayPixels[baseIndex + 3u] = 255u;
        }
    }

    void colorizePrimitiveAges(
        const std::vector<std::uint32_t>& ages,
        uint32_t renderWidth,
        uint32_t renderHeight,
        std::uint32_t coldAfterIterations,
        std::vector<uint8_t>& displayPixels) {
        constexpr std::uint32_t kNoVisibleSurfel =
            std::numeric_limits<std::uint32_t>::max();
        const std::size_t pixelCount =
            static_cast<std::size_t>(renderWidth) * static_cast<std::size_t>(renderHeight);
        displayPixels.assign(pixelCount * 4u, 0u);
        if (ages.size() < pixelCount) {
            return;
        }

        const float safeRange = static_cast<float>(std::max(coldAfterIterations, 1u));
        for (std::size_t pixelIndex = 0u; pixelIndex < pixelCount; ++pixelIndex) {
            const std::uint32_t age = ages[pixelIndex];
            if (age == kNoVisibleSurfel) {
                continue;
            }

            // Age zero is hot/red. Primitives at or beyond the selected range
            // are cold/blue; Jet makes this convention visually explicit.
            const float hotness = 1.0f - std::clamp(
                static_cast<float>(age) / safeRange, 0.0f, 1.0f);
            const glm::vec3 color = scalarColor(hotness, ScalarColorMap::Jet);
            const std::size_t baseIndex = pixelIndex * 4u;
            displayPixels[baseIndex + 0u] = channelToByte(color.r);
            displayPixels[baseIndex + 1u] = channelToByte(color.g);
            displayPixels[baseIndex + 2u] = channelToByte(color.b);
            displayPixels[baseIndex + 3u] = 255u;
        }
    }

    void colorizeNormalBuffer(
        const std::vector<float>& values,
        uint32_t renderWidth,
        uint32_t renderHeight,
        std::vector<uint8_t>& displayPixels) {
        const std::size_t pixelCount =
            static_cast<std::size_t>(renderWidth) * static_cast<std::size_t>(renderHeight);
        displayPixels.assign(pixelCount * 4u, 0u);
        if (values.size() < pixelCount * 4u) {
            return;
        }

        for (std::size_t pixelIndex = 0; pixelIndex < pixelCount; ++pixelIndex) {
            const std::size_t normalIndex = pixelIndex * 4u;
            const std::size_t baseIndex = pixelIndex * 4u;
            const float nx = values[normalIndex + 0u];
            const float ny = values[normalIndex + 1u];
            const float nz = values[normalIndex + 2u];
            const float validity = values[normalIndex + 3u];
            displayPixels[baseIndex + 3u] = 255u;
            if (!std::isfinite(nx) || !std::isfinite(ny) || !std::isfinite(nz) ||
                !std::isfinite(validity) || validity <= 0.0f) {
                continue;
            }

            const float lengthSquared = nx * nx + ny * ny + nz * nz;
            if (lengthSquared <= 1.0e-12f) {
                continue;
            }

            const float inverseLength = 1.0f / std::sqrt(lengthSquared);
            displayPixels[baseIndex + 0u] = channelToByte(nx * inverseLength * 0.5f + 0.5f);
            displayPixels[baseIndex + 1u] = channelToByte(ny * inverseLength * 0.5f + 0.5f);
            displayPixels[baseIndex + 2u] = channelToByte(nz * inverseLength * 0.5f + 0.5f);
        }
    }

    void glfwErrorCallback(int error, const char* description) {
        Pale::Log::PA_ERROR("GLFW error {}: {}", error, description);
    }

    void glfwDropCallback(GLFWwindow* window, int count, const char** paths) {
        auto* dropState = static_cast<DropState*>(glfwGetWindowUserPointer(window));
        if (!dropState) {
            return;
        }

        for (int pathIndex = 0; pathIndex < count; ++pathIndex) {
            const std::filesystem::path path{paths[pathIndex]};
            std::error_code error;
            if (isPlyPath(path) || std::filesystem::is_directory(path, error)) {
                dropState->pendingPath = path;
                dropState->hasPendingPath = true;
                return;
            }
        }
    }

    ImTextureID toImTextureId(GLuint textureId) {
        return (ImTextureID)(std::uintptr_t)textureId;
    }
}

int main(int argc, char** argv) {
    try {
        AppArgs args = parseArgs(argc, argv);
        const std::filesystem::path assetsDirectory =
            std::filesystem::absolute(args.assetsDir).lexically_normal();
        const std::filesystem::path repositoryRoot = std::filesystem::path(PALE_DEFAULT_ASSET_DIR).parent_path();
        viewer::TrainingControls training(repositoryRoot);
        viewer::MeshExportControls meshExport(repositoryRoot);
        bool showMeshExport=false,meshExportRequested=false;
        YAML::Node displayedExportMetadata;
        std::filesystem::path displayedExportRunConfig;
        if (!args.sceneInputSpecified && !training.datasets.empty()) {
            args.scenePath = std::filesystem::path(training.dataset.data()) / "scene.xml";
            args.pointCloudPath = std::filesystem::path(training.dataset.data()) / "points.ply";
        }
        training.datasetChanged = false; // The startup scene is loaded below.
        const std::filesystem::path optimizationOutputDirectory =
            repositoryRoot / "python" / "OptimizationOutput";
        std::filesystem::current_path(assetsDirectory);

        const std::filesystem::path originalInitialPointCloudPath =
            std::filesystem::absolute(args.pointCloudPath).lexically_normal();
        const std::filesystem::path savedInitialPointCloudPath =
            savedViewerDefaultPath(originalInitialPointCloudPath);

        Pale::Log::init(spdlog::level::level_enum::info);
        glfwSetErrorCallback(glfwErrorCallback);

        Pale::AssetManager assetManager{256};
        registerAssetLoaders(assetManager);

        std::filesystem::path startupPointCloudPath = originalInitialPointCloudPath;
        PlyFileSnapshot savedDefaultSnapshot{};
        std::string savedDefaultFailure;
        if (validatePlyReadyForLoad(
                savedInitialPointCloudPath,
                savedDefaultSnapshot,
                savedDefaultFailure)) {
            startupPointCloudPath = savedInitialPointCloudPath;
            Pale::Log::PA_INFO(
                "Using saved viewer default point cloud: {}",
                startupPointCloudPath.string());
        }

        std::shared_ptr<Pale::Scene> scene =
            loadSceneWithPointCloud(assetManager, args.scenePath, startupPointCloudPath);

        Pale::DeviceSelector deviceSelector;
        sycl::queue queue = deviceSelector.getQueue();
        Pale::AssetAccessFromManager assetAccessor(assetManager);

        Pale::SceneBuild::BuildOptions buildOptions{};
        buildOptions.bvhMaxLeafPoints = 4u;
        buildOptions.pointBvhUseBinnedSah = true;

        Pale::SceneBuild::BuildProducts buildProducts =
            Pale::SceneBuild::build(scene, assetAccessor, buildOptions);
        Pale::GPUSceneBuffers sceneGpu =
            Pale::SceneUpload::allocateAndUpload(buildProducts, queue);

        SceneBounds bounds = computeSceneBounds(buildProducts);
        OrbitCamera orbit = makeInitialOrbitCamera(buildProducts, bounds);
        OrbitCamera initialOrbit = orbit;
        std::filesystem::path currentScenePath = args.scenePath;
        std::filesystem::path currentPointCloudPath = startupPointCloudPath;
        std::array<char, 1024> pointCloudPathBuffer{};
        copyPathToBuffer(currentPointCloudPath, pointCloudPathBuffer);
        std::string pointCloudStatus =
            "Loaded " + std::to_string(buildProducts.points.size()) + " surfels";
        viewer::OptimizationFolderAnchor optimizationFolderAnchor;
        bool latestOptimizationMode = false;
        std::filesystem::path latestOptimizationPointsDirectory;
        std::vector<PointCloudSnapshot> latestOptimizationSnapshots;
        std::size_t latestOptimizationSnapshotIndex = 0u;
        bool plyBrowserOpen = false;
        std::filesystem::path plyBrowserDirectory =
            currentPointCloudPath.has_parent_path() ? currentPointCloudPath.parent_path()
                                                    : std::filesystem::current_path();
        std::filesystem::path currentRuntimeMeshPath;
        std::array<char, 1024> meshPathBuffer{};
        std::string meshStatus = "Drop a triangle .ply file into the render view or load one here";
        bool meshBrowserOpen = false;
        std::filesystem::path meshBrowserDirectory =
            currentPointCloudPath.has_parent_path() ? currentPointCloudPath.parent_path()
                                                    : std::filesystem::current_path();
        DropState dropState;
        std::filesystem::path workingDirectory, workingSourceScene;
        uint64_t workingIteration = 0;
        bool hasWorkingSnapshot = false;
        auto modelReadOnly = [&]() { return training.process.launched() && training.view != viewer::TrainingView::Snapshot; };

        uint32_t renderWidth = args.width;
        uint32_t renderHeight = args.height;
        if (renderWidth == 0 || renderHeight == 0) {
            if (args.width == 0 && args.height == 0) {
                renderWidth = kDefaultViewportImageExtent;
                renderHeight = kDefaultViewportImageExtent;
            } else if (!buildProducts.cameraGPUs.empty()) {
                renderWidth = buildProducts.cameraGPUs.front().width;
                renderHeight = buildProducts.cameraGPUs.front().height;
            } else {
                renderWidth = args.width == 0 ? kDefaultViewportImageExtent : args.width;
                renderHeight = args.height == 0 ? kDefaultViewportImageExtent : args.height;
            }
        }

        Pale::PathTracerSettings settings = makeDefaultSettings();
        if (args.sharedSlabs) {
            settings.sharedHeightEnabled = true;
            settings.sharedHeightTwoSlabs = true;
            settings.sharedHeightShading = 2;
        }
        Pale::PathTracer tracer(queue, settings);
        Pale::SceneBuild::BuildProducts renderBuildProducts = buildProducts;
        renderBuildProducts.cameraGPUs.clear();
        Pale::SensorGPU sensor{};
        bool hasSensor = false;
        bool cameraDirty = true;
        bool tracerDirty = true;
        bool autoRender = true;
        bool renderRequested = true;
        CameraSource cameraSource = CameraSource::Viewport;
        int selectedSceneCameraIndex = 0;
        bool walkNavigationActive = false;
        OrbitCamera orbitBeforeWalk = orbit;
        CameraSource cameraSourceBeforeWalk = cameraSource;
        float walkSpeed = 0.5f;
        double walkCursorX = 0.0;
        double walkCursorY = 0.0;
        bool forceRenderOpacity = false;
        bool showViewportGrid = true;
        ImVec4 viewportBackgroundColor = kBlenderViewportBackground;
        ViewImageMode viewImageMode = ViewImageMode::Rendered;
        ScalarColorMap scalarColorMap = ScalarColorMap::Viridis;
        int densityGridSize = 64;
        float densityColorMaximum = 100.0f;
        bool densityLogScale = true;
        viewer::SurfaceOverlapSettings surfaceOverlapSettings;
        std::vector<viewer::SurfaceOverlapView> surfaceOverlapViews;
        bool surfaceOverlapScoresValid = false;
        std::vector<viewer::SurfaceOverlapScore> surfaceOverlapScores;
        std::vector<std::size_t> surfaceOverlapInstanceOffsets;
        constexpr int kMeanSlabMembersMetric = 0;
        constexpr int kSlabOverflowMetric = 1;
        int surfaceOverlapMetric = kMeanSlabMembersMetric;
        float surfaceOverlapColorMaximum = 4.0f;
        double surfaceOverlapComputeMs = 0.0;
        float surfaceOverlapPeak = 0.0f;
        std::size_t surfaceOverlapEligibleCount = 0u;
        // Training debug defaults mirror python/config.py (OptimizationConfig).
        constexpr float kPositionDefaultThreshold = viewer::defaults::positionThreshold;
        float positionWhatIfDisplayThreshold = kPositionDefaultThreshold;
        float positionWhatIfReferenceThreshold = 0.0f;
        std::filesystem::path positionWhatIfThresholdPointCloudPath;
        std::array<char, 2048> targetDatasetBuffer{};
        bool targetDatasetManual = false;
        int targetColorSpace = std::string(viewer::defaults::targetColorSpace) == "srgb" ? 1 : 0;
        bool transportDirty = true;
        bool transportUseSavedSettings = true;
        bool transportXray = true;
        float transportThreshold = viewer::defaults::transportMinScore;
        int transportSource = -1; // -1 training cameras; otherwise one saved camera
        int transportComponent = 0; // combined, camera, shadow
        std::vector<viewer::TransportCameraScores> transportViews;
        std::vector<double> transportScores;
        std::vector<std::size_t> transportAffected;
        std::string transportStatus = "Enable contribution scores to analyze the loaded scene cameras.";
        struct TransportMarker { glm::vec3 center; std::size_t primitive; int editableIndex; };
        std::vector<TransportMarker> transportMarkers;
        auto refreshTransportSelection = [&]() {
            transportScores = viewer::selectTransportScores(transportViews, transportSource, transportComponent);
            transportAffected.clear();
            for (std::size_t i = 0; i < transportScores.size(); ++i)
                if (!buildProducts.points[i].isEmissive() && transportScores[i] < transportThreshold)
                    transportAffected.push_back(i);
        };
        bool targetLossDirty = true;
        std::vector<viewer::ProjectedTargetLoss> targetLossViews;
        std::vector<std::size_t> targetLossInstanceOffsets;
        std::vector<float> targetLossMean, targetLossWorst;
        int targetLossCamera = -1; // -1: all target cameras
        int targetLossAggregation = 0;
        float targetLossColorMaximum = 0.01f;
        bool targetLossLogColors = true;
        double targetLossComputeMs = 0;
        std::string targetLossStatus = "Select Target RGB loss to compute the projection";
        int primitiveAgeColdAfterIterations = 1000;
        int selectedLightIndex = 0;
        bool showLightGizmo = true;
        bool gizmosEnabled = true;
        int viewportTool = 1;
        bool showLightMarkers = true;
        bool openLightControls = false;
        bool openSurfelControls = false;
        std::vector<glm::vec3> lightLocalCenters;
        struct SurfelLightMarker { glm::vec3 worldCenter; int editorIndex; };
        std::vector<SurfelLightMarker> surfelLightMarkers;
        bool lightMarkersDirty = true;
        struct LightMarkerSelection {
            int areaIndex = -1;
            int surfelIndex = -1;
            bool operator==(const LightMarkerSelection&) const = default;
        };
        std::optional<LightMarkerSelection> pressedLightMarker;
        ImGuizmo::OPERATION lightGizmoOperation = ImGuizmo::TRANSLATE;
        ImGuizmo::MODE lightGizmoMode = ImGuizmo::WORLD;
        int selectedSurfelLightIndex = 0;
        int candidateZeroPowerSurfelIndex = 0;
        float candidateSurfelPower = 1.0f;
        int selectedSurfelEditorIndex = -1;
        bool showSurfelGizmo = true;
        ImGuizmo::OPERATION surfelGizmoOperation = ImGuizmo::TRANSLATE;
        ImGuizmo::MODE surfelGizmoMode = ImGuizmo::WORLD;
        bool viewportGizmoMouseCapture = false;
        bool viewportPickArmed = false;
        std::string surfelLightStatus;
        std::string surfelEditorStatus;
        float exposure = 1.0f;
        float gamma = 1.0f;
        bool useSrgbEncoding = true;
        double lastRenderMs = 0.0;
        std::vector<float> renderFrameTimeHistory;
        renderFrameTimeHistory.reserve(kFrameTimeHistoryCapacity);
        bool runAdjointEveryRender = false;
        bool runAdjointNextRender = false;
        bool viewerAdjointDirectLight = true;
        int viewerAdjointSamplesPerPixel = 1;
        int viewerAdjointBounces = 1;
        bool viewerAdjointAllSurfels = true;
        int viewerAdjointSurfelIndex = 1;
        int viewerAdjointProperty = 8;
        int viewerAdjointSeed = 42;
        float viewerAdjointColorMaximum = 1.0f;
        std::vector<float> viewerAdjointImage;
        bool viewerAdjointImageValid = false;
        constexpr const char* adjointProperties[] = {
            "Position X", "Position Y", "Position Z",
            "Local rotation X", "Local rotation Y", "Local rotation Z",
            "Scale U", "Scale V", "Opacity", "Albedo R", "Beta"};
        double lastViewerAdjointMs = 0.0;
        float lastViewerAdjointLoss = 0.0f;
        std::string viewerAdjointStatus = "Adjoint profiling is off";
        double lastRegularizerGradientMapMs = 0.0;
        std::vector<uint8_t> renderPixels;
        std::vector<uint8_t> rawLinearPixels;
        std::vector<float> rawRenderPixels;
        viewer::DisplayTransform displayedTransform;
        std::vector<uint8_t> pixels;
        viewer::ViewportBackgroundCache viewportBackgroundCache;
        std::vector<uint8_t> screenshotPixels;
        std::string screenshotStatus;
        uint64_t screenshotSequence = 0;
        DebugDisplayBuffers debugDisplayBuffers;
        float terminalTargetAttenuation = 0.9f;
        OrbitCamera displayedOrbit = orbit;
        SceneBounds displayedBounds = bounds;
        CameraSource displayedCameraSource = cameraSource;
        Pale::CameraGPU displayedCamera = orbit.makeGpuCamera(renderWidth, renderHeight);
        uint32_t displayedRenderWidth = renderWidth;
        uint32_t displayedRenderHeight = renderHeight;
        struct PublishedPreviewState {
            std::vector<uint8_t> render,linear,pixels,screenshot;
            std::vector<float> raw,adjoint;
            bool adjointValid;
            DebugDisplayBuffers debug;
            viewer::DisplayTransform transform;
            OrbitCamera orbit;
            SceneBounds bounds;
            CameraSource source;
            Pale::CameraGPU camera;
            uint32_t width,height;
        };
        viewer::PreviewPublication<PublishedPreviewState> previewPublication;
        const auto restorePublishedPreview=[&](PublishedPreviewState prior) {
            renderPixels=std::move(prior.render);rawLinearPixels=std::move(prior.linear);
            pixels=std::move(prior.pixels);screenshotPixels=std::move(prior.screenshot);
            rawRenderPixels=std::move(prior.raw);viewerAdjointImage=std::move(prior.adjoint);viewerAdjointImageValid=prior.adjointValid;
            debugDisplayBuffers=std::move(prior.debug);
            displayedTransform=prior.transform;displayedOrbit=prior.orbit;displayedBounds=prior.bounds;
            displayedCameraSource=prior.source;displayedCamera=prior.camera;
            displayedRenderWidth=prior.width;displayedRenderHeight=prior.height;
        };
        Texture2D texture;
        bool showTrainingProgress = true;
        bool selectTrainingTab = false;
        bool showSceneTree = true;
        constexpr bool showAdvancedTools = true;
        bool launchTrainingRequested = false;
        viewer::ui::Inspector inspectorPage = viewer::ui::Inspector::Reconstruction;
        viewer::SceneTree sceneTree;
        viewer::SceneItemKind sceneSelectionKind = viewer::SceneItemKind::Geometry;
        std::string selectedTreeEntity;
        bool sceneTreeDirty = true;
        bool sceneUploadPending = false;
        viewer::CameraImageView cameraImageView;
        std::string cameraImageKey;
        std::chrono::steady_clock::time_point lastNavigationInput{};
        bool interactivePreview=false,lastPreviewInteractive=false;
        bool fastEmitterEditing=true,fastSurfelEditing=false;
        viewer::PreviewWork previewWork;
        viewer::PreviewRecovery previewRecovery;
        viewer::PreviewWatchdog previewWatchdog;
        const auto previewFailureLog=repositoryRoot/"output"/"viewer_diagnostics"/
            ("preview_"+std::to_string(std::chrono::system_clock::now().time_since_epoch().count())+".jsonl");
        std::function<void()> previewCompletion;
        std::function<void(std::function<void()>,std::function<void()>,const char*)> startPreviewJob;
        std::vector<viewer::SceneTreeGroup> cachedSceneGroups;
        std::optional<viewer::SceneTreeItem> queuedScenePick;
        std::string cachedSceneName,previewActivity;
        std::filesystem::path previewJobScene,previewJobPointCloud;
        std::string previewJobView;
        OrbitCamera queuedOrbit=orbit;
        SceneBounds busyBounds=bounds;
        Pale::CameraGPU busyCamera=displayedCamera;
        CameraSource busyCameraSource=cameraSource;
        bool queuedOrbitChanged=false,queuedRefresh=false,followLaunchAfterPreview=false;
        bool busyHasWorkspace=false;
        std::filesystem::path loadedTrainingSceneRun;
        std::optional<ViewImageMode> queuedView;
        std::optional<viewer::TrainingView> queuedSceneSource;
        bool editLatestRequested=false,returnToEditsRequested=false;
        std::optional<int> queuedCamera;
        std::optional<std::pair<std::filesystem::path,uint64_t>> pendingViewedIteration;
        std::optional<bool> queuedWalkFinish;
        uint32_t busyPointCount=0;
        double busyRenderMs=0;
        std::vector<std::string> busyCameraNames;
        std::string busySelectionName;
        glm::vec3 busySelectionPosition{},busySelectionColor{};
        bool busyHasSelection=false,queuedSelectionChanged=false;
        viewer::SceneItemKind busySelectionKind=viewer::SceneItemKind::Geometry;
        int busySelectionIndex=-1;
        glm::vec3 busySelectionRotation{},busySelectionScale{1},busyLightCenter{};
        glm::mat4 busyPointCloudTransform{1};
        float busySelectionPower=0,busySelectionOpacity=1,busySelectionBeta=0;
        struct DisplayedLightMarker {glm::vec3 center;LightMarkerSelection selection;std::string name;};
        std::vector<DisplayedLightMarker> displayedLightMarkers;
        std::filesystem::path displayedOverlayPointCloud,displayedOverlayScene;
        viewer::LivePacketLeaseCache livePacketLeases;
        std::shared_ptr<const viewer::LivePacketLease> loadedLivePacket,displayedLivePacket;
        viewer::ViewportOverlayCache displayedSelectedHandles;
        bool busyOverlayMatchesScene=false;
        int busySelectedLightIndex=0;
        std::optional<std::pair<LightMarkerSelection,std::filesystem::path>> queuedLightMarkerPick;
        struct InspectorBodyCache {
            viewer::ViewportOverlayCache geometry;
            ImVec2 position{},size{};float scroll=0,contentHeight=0;
            std::string key;bool valid=false;
        };
        std::array<InspectorBodyCache,6> inspectorBodies;
        viewer::InspectorInputHandoff inspectorInput;
        viewer::ui::Inspector lastDrawnInspector=inspectorPage;
        bool holdPreviewForInspector=false;
        int inspectorHandoffFrames=0;
        std::vector<std::size_t> sceneTreeEmitterIndices;
        bool showProfilingWindow = false;
        bool timerProfilingEnabled = true;
        bool gpuCounterProfilingEnabled = false;
        bool dockLayoutInitialized = false;
        Pale::RenderProfilingCounters lastProfilingCounters{};
        std::vector<Pale::ScopedTimerRecord> lastTimerRecords;
        Pale::GPUSceneBuffers busyProfilingScene{};
        Pale::RenderProfilingCounters busyProfilingCounters{};
        std::vector<Pale::ScopedTimerRecord> busyTimerRecords;
        std::vector<float> busyFrameTimes;
        bool busyTimerProfilingEnabled=true,busyGpuCounterProfilingEnabled=false,queuedProfilingSettings=false;
        Pale::RenderProfilingCounters* deviceProfilingCounters =
            sycl::malloc_device<Pale::RenderProfilingCounters>(1, queue);
        if (deviceProfilingCounters == nullptr) {
            throw std::runtime_error("Failed to allocate render profiling counters");
        }
        queue.memset(deviceProfilingCounters, 0, sizeof(Pale::RenderProfilingCounters)).wait();
        Pale::PointGradients viewerAdjointGradients{};
        Pale::DebugImages viewerAdjointDebugImages{};
        Pale::PointGradients viewerRegularizerGradients{};

        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }
#if GLFW_VERSION_MAJOR > 3 || (GLFW_VERSION_MAJOR == 3 && GLFW_VERSION_MINOR >= 4)
        Pale::Log::PA_INFO("Viewer window backend: {}",
            glfwGetPlatform() == GLFW_PLATFORM_WAYLAND ? "Wayland" :
            glfwGetPlatform() == GLFW_PLATFORM_X11 ? "X11" : "other");
#else
        // GLFW before 3.4 exposes build information instead of a platform query.
        Pale::Log::PA_INFO("Viewer GLFW build: {}", glfwGetVersionString());
#endif

        const char* glslVersion = "#version 130";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        const std::array<int, 2> initialWindowSize =
            initialViewerWindowSize(renderWidth, renderHeight);
        GLFWwindow* window = glfwCreateWindow(
            initialWindowSize[0],
            initialWindowSize[1],
            "Pale Reconstruction Viewer",
            nullptr,
            nullptr);
        if (!window) {
            glfwTerminate();
            throw std::runtime_error("Failed to create GLFW window");
        }
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);
        glfwSetWindowUserPointer(window, &dropState);
        glfwSetDropCallback(window, glfwDropCallback);
        glfwSetWindowCloseCallback(window, [](GLFWwindow*) {
            Pale::Log::PA_WARN("Viewer received a window-system close request (user close or display disconnect)");
        });

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        viewer::ui::applyStyle();
        viewer::ui::loadFont();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init(glslVersion);

        auto rebuildSceneGpu = [&]() {
            sceneTreeDirty = true;
            lightLocalCenters.clear();
            surfelLightMarkers.clear();
            lightMarkersDirty = true;
            pressedLightMarker.reset();
            transportDirty = true;
            transportViews.clear();
            transportScores.clear();
            transportAffected.clear();
            transportMarkers.clear();
            targetLossDirty = true;
            surfaceOverlapScoresValid = false;
            debugDisplayBuffers.surfaceOverlapValid = false;
            buildProducts = Pale::SceneBuild::build(scene, assetAccessor, buildOptions);
            // GPU uploads can JIT compile kernels. The preview worker owns that
            // handoff; rebuilding the CPU scene must not block on a GPU queue.
            sceneUploadPending = true;
            renderBuildProducts = buildProducts;
            renderBuildProducts.cameraGPUs.clear();
            bounds = computeSceneBounds(buildProducts);
            orbit.farClip = std::max(1000.0f, bounds.radius * 20.0f);
            tracerDirty = true;
            renderRequested = true;
        };

        auto replacePointCloud = [&](const std::filesystem::path& requestedPath, bool keepLatestOptimizationMode = false) -> bool {
            if (requestedPath.empty()) {
                pointCloudStatus = "No PLY path selected";
                return false;
            }
            if (!isPlyPath(requestedPath)) {
                pointCloudStatus = "Selected file is not a .ply file";
                return false;
            }

            std::error_code error;
            if (!std::filesystem::exists(requestedPath, error) || error) {
                pointCloudStatus = "PLY file does not exist: " + requestedPath.string();
                return false;
            }

            PlyFileSnapshot validatedSnapshot{};
            std::string readinessFailure;
            if (!validatePlyReadyForLoad(requestedPath, validatedSnapshot, readinessFailure)) {
                pointCloudStatus =
                    "PLY is not ready yet (" + readinessFailure + "): " + requestedPath.string();
                return false;
            }

            Pale::AssetHandle pointCloudAssetHandle{};
            Pale::AssetPtr<Pale::PointAsset> pointCloudAsset;
            try {
                pointCloudAssetHandle =
                    importPathAsType(assetManager.registry(), requestedPath, Pale::AssetType::PointCloud);
                assetManager.invalidate(pointCloudAssetHandle);
                pointCloudAsset = assetAccessor.getPointCloud(pointCloudAssetHandle);
            } catch (const std::exception& exception) {
                pointCloudStatus =
                    "Failed to load PLY without replacing the current scene: " +
                    std::string(exception.what());
                return false;
            }
            if (!pointCloudAsset) {
                pointCloudStatus = "Failed to load PLY: " + requestedPath.string();
                return false;
            }

            const std::optional<PlyFileSnapshot> loadedSnapshot = readPlyFileSnapshot(requestedPath);
            if (!loadedSnapshot || !samePlyFileSnapshot(validatedSnapshot, *loadedSnapshot)) {
                assetManager.invalidate(pointCloudAssetHandle);
                pointCloudStatus =
                    "PLY changed while it was loading; kept the current scene. Try again: " +
                    requestedPath.string();
                return false;
            }

            const std::size_t surfelCount = countSurfels(*pointCloudAsset);
            if (surfelCount == 0) {
                pointCloudStatus = "PLY contained no surfels: " + requestedPath.string();
                return false;
            }
            if(selectedSurfelEditorIndex>=0&&static_cast<std::size_t>(selectedSurfelEditorIndex)>=surfelCount)selectedSurfelEditorIndex=-1;

            std::vector<Pale::Entity> pointCloudEntities;
            const auto previousPointCloudHandle=firstPointCloudHandle(scene);
            for (auto [entityHandle, pointCloudComponent] :
                 scene->getAllEntitiesWith<Pale::PointCloudComponent>().each()) {
                (void)pointCloudComponent;
                pointCloudEntities.emplace_back(entityHandle, scene.get());
            }
            for (Pale::Entity entity : pointCloudEntities) {
                scene->destroyEntity(entity);
            }

            Pale::Entity pointCloudEntity = scene->createEntity("RealtimePointCloud");
            pointCloudEntity.addComponent<Pale::PointCloudComponent>().pointCloudID = pointCloudAssetHandle;

            rebuildSceneGpu();
            // Live generations have distinct immutable paths. Do not let the
            // asset LRU retain hundreds of full historical point-cloud arrays.
            if(previousPointCloudHandle && *previousPointCloudHandle!=pointCloudAssetHandle)
                assetManager.invalidate(*previousPointCloudHandle);
            currentPointCloudPath = requestedPath;
            copyPathToBuffer(currentPointCloudPath, pointCloudPathBuffer);
            if (currentPointCloudPath.has_parent_path()) {
                plyBrowserDirectory = currentPointCloudPath.parent_path();
            }

            pointCloudStatus = "Loaded " + std::to_string(surfelCount) + " surfels";
            cameraDirty = true;
            if (!keepLatestOptimizationMode) {
                optimizationFolderAnchor.clear();
                latestOptimizationMode = false;
                latestOptimizationPointsDirectory.clear();
                latestOptimizationSnapshots.clear();
                latestOptimizationSnapshotIndex = 0u;
            }
            return true;
        };

        auto replaceSceneAndPointCloud = [&](
            const std::filesystem::path& requestedScenePath,
            const std::filesystem::path& requestedPointCloudPath,
            bool keepLatestOptimizationMode = false) -> bool {
            if (requestedScenePath.empty()) {
                pointCloudStatus = "No scene XML path selected";
                return false;
            }
            if (requestedPointCloudPath.empty()) {
                pointCloudStatus = "No PLY path selected";
                return false;
            }
            if (!isPlyPath(requestedPointCloudPath)) {
                pointCloudStatus = "Selected file is not a .ply file";
                return false;
            }

            std::error_code error;
            if (!std::filesystem::is_regular_file(requestedScenePath, error) || error) {
                pointCloudStatus = "Scene XML does not exist: " + requestedScenePath.string();
                return false;
            }
            error.clear();
            if (!std::filesystem::exists(requestedPointCloudPath, error) || error) {
                pointCloudStatus = "PLY file does not exist: " + requestedPointCloudPath.string();
                return false;
            }

            std::shared_ptr<Pale::Scene> nextScene;
            try {
                nextScene = loadSceneWithPointCloud(assetManager, requestedScenePath, requestedPointCloudPath);
            } catch (const std::exception& exception) {
                pointCloudStatus = "Failed to load optimization scene: " + std::string(exception.what());
                return false;
            }

            OrbitCamera preservedOrbit = orbit;
            CameraSource preservedCameraSource = cameraSource;
            if (cameraSource == CameraSource::SceneXml && !buildProducts.cameraGPUs.empty()) {
                const int cameraIndex = std::clamp(
                    selectedSceneCameraIndex,
                    0,
                    static_cast<int>(buildProducts.cameraGPUs.size() - 1u));
                preservedOrbit = makeOrbitCameraFromSceneCamera(
                    buildProducts.cameraGPUs[static_cast<std::size_t>(cameraIndex)],
                    bounds,
                    orbit);
                preservedCameraSource = CameraSource::Viewport;
            }

            const auto previousPointCloudHandle=firstPointCloudHandle(scene);
            scene = std::move(nextScene);
            currentScenePath = requestedScenePath;
            currentPointCloudPath = requestedPointCloudPath;
            copyPathToBuffer(currentPointCloudPath, pointCloudPathBuffer);
            if (currentPointCloudPath.has_parent_path()) {
                plyBrowserDirectory = currentPointCloudPath.parent_path();
            }

            selectedSceneCameraIndex = 0;
            selectedLightIndex = 0;
            selectedSurfelLightIndex = 0;
            selectedSurfelEditorIndex = -1;
            surfelLightStatus.clear();
            surfelEditorStatus.clear();
            rebuildSceneGpu();
            if(previousPointCloudHandle && previousPointCloudHandle!=firstPointCloudHandle(scene))
                assetManager.invalidate(*previousPointCloudHandle);
            initialOrbit = makeInitialOrbitCamera(buildProducts, bounds);
            orbit = preservedOrbit;
            orbit.farClip = std::max(orbit.farClip, std::max(1000.0f, bounds.radius * 20.0f));
            cameraSource = preservedCameraSource;
            cameraDirty = true;

            if (!keepLatestOptimizationMode) {
                optimizationFolderAnchor.clear();
                latestOptimizationMode = false;
                latestOptimizationPointsDirectory.clear();
                latestOptimizationSnapshots.clear();
                latestOptimizationSnapshotIndex = 0u;
            }

            pointCloudStatus =
                "Loaded optimization scene " + currentScenePath.filename().string() +
                " with " + std::to_string(buildProducts.points.size()) + " surfels";
            return true;
        };

        auto replaceOptimizationPointCloud = [&](
            const std::filesystem::path& requestedPath,
            bool keepLatestOptimizationMode = false) -> bool {
            if (const std::optional<std::filesystem::path> optimizationScenePath =
                    sceneXmlForOptimizationPointCloud(requestedPath)) {
                return replaceSceneAndPointCloud(*optimizationScenePath, requestedPath, keepLatestOptimizationMode);
            }

            const bool loaded = replacePointCloud(requestedPath, keepLatestOptimizationMode);
            if (loaded) {
                pointCloudStatus += "; no scene.xml found beside optimization run";
            }
            return loaded;
        };

        bool snapshotEditPending=false;
        auto beginSnapshotEdit=[&] {
            const bool emitter=sceneSelectionKind==viewer::SceneItemKind::Emitter||sceneSelectionKind==viewer::SceneItemKind::SurfelEmitter;
            if(emitter?fastEmitterEditing:fastSurfelEditing)
                lastNavigationInput=std::chrono::steady_clock::now();
            followLaunchAfterPreview=false;
            if(!modelReadOnly())return;
            training.view=viewer::TrainingView::Snapshot;
            workingDirectory.clear();workingSourceScene=std::filesystem::absolute(currentScenePath);
            workingIteration=training.viewedIteration;hasWorkingSnapshot=false;snapshotEditPending=true;
            training.message="Editing a private copy of iteration "+std::to_string(workingIteration);
        };
        auto saveWorkingSnapshot = [&]() -> bool {
            if (modelReadOnly()) return hasWorkingSnapshot;
            const auto handle = firstPointCloudHandle(scene);
            const auto points = handle ? assetAccessor.getPointCloud(*handle) : nullptr;
            if (!points) { training.message = "No editable point cloud to save"; return false; }
            try {
                if (workingDirectory.empty()) workingDirectory = viewer::reserveDirectory(repositoryRoot / "output/viewer_snapshots", "snapshot_");
                if (workingSourceScene.empty()) workingSourceScene = std::filesystem::absolute(currentScenePath);
                std::string failure;
                bool written=false;
                viewer::runWithEventPump([&] { written=writePointCloudPly(*points, workingDirectory / "points.ply", failure); },[] { glfwPollEvents(); });
                if (!written) throw std::runtime_error(failure);
                YAML::Node state;
                state["source_scene"] = workingSourceScene.string();
                state["pointcloud"] = (workingDirectory / "points.ply").string();
                state["asset_root"] = assetsDirectory.string();
                state["iteration"] = viewer::jsonNumber(workingIteration);
                for (auto [id, mesh, transform, tag] : scene->getAllEntitiesWith<Pale::MeshComponent, Pale::TransformComponent, Pale::TagComponent>().each()) {
                    Pale::Entity entity(id, scene.get());
                    YAML::Node item;
                    item["name"] = tag.tag;
                    for (int axis = 0; axis < 3; ++axis) {
                        item["translation"].push_back(viewer::jsonNumber(transform.translation[axis]));
                        item["scale"].push_back(viewer::jsonNumber(transform.scale[axis]));
                    }
                    for (float value : {transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w}) item["rotation_xyzw"].push_back(viewer::jsonNumber(value));
                    if (auto meta = assetManager.registry().meta(mesh.meshID)) item["mesh_file"] = std::filesystem::absolute(meta->path).string();
                    if (entity.hasComponent<Pale::AreaLightComponent>()) {
                        auto& light = entity.getComponent<Pale::AreaLightComponent>();
                        glm::vec3 color = light.radiance; float power = light.flux;
                        if (entity.hasComponent<Pale::MaterialComponent>()) {
                            if (auto material = assetManager.get<Pale::Material>(entity.getComponent<Pale::MaterialComponent>().materialID)) {
                                color = material->baseColor; power = material->power;
                            }
                        }
                        for (int axis = 0; axis < 3; ++axis) item["emitter"]["color"].push_back(viewer::jsonNumber(color[axis]));
                        item["emitter"]["power"] = viewer::jsonNumber(power);
                    }
                    state["entities"].push_back(item);
                }
                if (!state["entities"]) state["entities"] = YAML::Node(YAML::NodeType::Sequence);
                viewer::writeJsonAtomic(workingDirectory / "workspace.json", state);
                hasWorkingSnapshot = true;
                training.message = "Saved editable snapshot: " + workingDirectory.string();
                return true;
            } catch (const std::exception& error) { training.message = error.what(); return false; }
        };
        const auto captureExportMetadata=[&] {
            YAML::Node state;
            state["source_scene"]=std::filesystem::absolute(currentScenePath).string();
            state["asset_root"]=assetsDirectory.string();
            state["pointcloud"]=std::filesystem::absolute(currentPointCloudPath).string();
            state["entities"]=YAML::Node(YAML::NodeType::Sequence);
            for(auto [id,mesh,transform,tag]:scene->getAllEntitiesWith<Pale::MeshComponent,Pale::TransformComponent,Pale::TagComponent>().each()) {
                Pale::Entity entity(id,scene.get());YAML::Node item;
                item["name"]=tag.tag;
                for(int axis=0;axis<3;++axis) {
                    item["translation"].push_back(viewer::jsonNumber(transform.translation[axis]));
                    item["scale"].push_back(viewer::jsonNumber(transform.scale[axis]));
                }
                for(float value:{transform.rotation.x,transform.rotation.y,transform.rotation.z,transform.rotation.w})item["rotation_xyzw"].push_back(viewer::jsonNumber(value));
                if(auto meta=assetManager.registry().meta(mesh.meshID))item["mesh_file"]=std::filesystem::absolute(meta->path).string();
                if(entity.hasComponent<Pale::AreaLightComponent>()) {
                    const auto& light=entity.getComponent<Pale::AreaLightComponent>();
                    auto color=light.radiance;float power=light.flux;
                    if(entity.hasComponent<Pale::MaterialComponent>())
                        if(auto material=assetManager.get<Pale::Material>(entity.getComponent<Pale::MaterialComponent>().materialID)) {color=material->baseColor;power=material->power;}
                    for(int axis=0;axis<3;++axis)item["emitter"]["color"].push_back(viewer::jsonNumber(color[axis]));
                    item["emitter"]["power"]=viewer::jsonNumber(power);
                }
                state["entities"].push_back(item);
            }
            return state;
        };
        const auto launchMeshExport=[&](bool busy) {
            if(!showMeshExport)meshExportRequested=false;
            if(!meshExportRequested||meshExport.process.active()||(busy&&!modelReadOnly()))return;
            meshExportRequested=false;
            try {
                std::filesystem::path workspace,runConfig=displayedExportRunConfig;
                if(modelReadOnly()) {
                    if(!displayedExportMetadata.IsMap())throw std::runtime_error("Wait for the first scene preview before exporting");
                    const auto input=viewer::reserveDirectory(repositoryRoot/"output/viewer_snapshots","export_");
                    workspace=input/"workspace.json";
                    viewer::writeJsonAtomic(workspace,displayedExportMetadata);
                } else {
                    if(!saveWorkingSnapshot())throw std::runtime_error(training.message);
                    workspace=workingDirectory/"workspace.json";
                }
                meshExport.launch(training,workspace,runConfig);
            } catch(const std::exception& error) {meshExport.message=error.what();}
        };
        auto invalidateSceneMaterials = [&]() {
            for (auto [id, material] : scene->getAllEntitiesWith<Pale::MaterialComponent>().each()) {
                (void)id; assetManager.invalidate(material.materialID);
            }
        };
        auto restoreWorkingSnapshot = [&]() -> bool {
            if (!hasWorkingSnapshot) return false;
            const auto state = viewer::readJson(workingDirectory / "workspace.json");
            if (!state.IsMap()) return false;
            invalidateSceneMaterials();
            if (!replaceSceneAndPointCloud(state["source_scene"].as<std::string>(), state["pointcloud"].as<std::string>(), true)) return false;
            for (const auto& item : state["entities"]) {
                Pale::Entity entity;
                for (auto [id, tag] : scene->getAllEntitiesWith<Pale::TagComponent>().each()) {
                    if (tag.tag == item["name"].as<std::string>()) { entity = Pale::Entity(id, scene.get()); break; }
                }
                if (!entity && item["mesh_file"]) {
                    entity = scene->createEntity(item["name"].as<std::string>());
                    entity.addComponent<Pale::MeshComponent>().meshID = importPathAsType(assetManager.registry(), item["mesh_file"].as<std::string>(), Pale::AssetType::Mesh);
                }
                if (!entity) continue;
                auto& transform = entity.getComponent<Pale::TransformComponent>();
                for (int axis = 0; axis < 3; ++axis) {
                    transform.translation[axis] = item["translation"][axis].as<float>();
                    transform.scale[axis] = item["scale"][axis].as<float>();
                }
                transform.setRotationQuaternion(glm::quat(item["rotation_xyzw"][3].as<float>(), item["rotation_xyzw"][0].as<float>(), item["rotation_xyzw"][1].as<float>(), item["rotation_xyzw"][2].as<float>()));
                if (item["emitter"] && entity.hasComponent<Pale::AreaLightComponent>()) {
                    auto& light = entity.getComponent<Pale::AreaLightComponent>();
                    for (int axis = 0; axis < 3; ++axis) light.radiance[axis] = item["emitter"]["color"][axis].as<float>();
                    light.flux = item["emitter"]["power"].as<float>();
                    if (entity.hasComponent<Pale::MaterialComponent>()) if (auto material = assetManager.get<Pale::Material>(entity.getComponent<Pale::MaterialComponent>().materialID)) {
                        material->baseColor = light.radiance; material->power = light.flux;
                    }
                }
            }
            rebuildSceneGpu();
            training.view = viewer::TrainingView::Snapshot;
            training.viewedIteration = workingIteration;
            return true;
        };
        auto enterTrainingView = [&](viewer::TrainingView view) {
            if (!training.process.launched()) return;
            if (!modelReadOnly()) {
                if(!saveWorkingSnapshot())return;
                loadedTrainingSceneRun.clear();
            }
            invalidateSceneMaterials();
            // Keep the published image and its iteration until the requested
            // scene has actually rendered. Loading a checkpoint is not an empty
            // viewport state.
            training.view = view; training.loadedGeneration = 0;
            plyBrowserOpen = meshBrowserOpen = false;
        };
        auto editLatestSavedSnapshot = [&]() {
            if (!training.process.launched()) return;
            const auto snapshot = training.latestSavedSnapshot();
            if (!snapshot) { training.message = "Waiting for the first saved snapshot"; return; }
            if (!modelReadOnly() && !saveWorkingSnapshot()) return;
            const auto latest = *snapshot;
            if (!training.status["scene"].IsScalar()) return;
            invalidateSceneMaterials();
            if (!replaceSceneAndPointCloud(training.status["scene"].as<std::string>(), latest.path, true)) return;
            training.view = viewer::TrainingView::Snapshot;
            workingDirectory.clear(); workingSourceScene = std::filesystem::absolute(currentScenePath);
            workingIteration = latest.iteration; training.viewedIteration = workingIteration;
            hasWorkingSnapshot = false;
            if (saveWorkingSnapshot()) restoreWorkingSnapshot();
        };
        const auto navigateCheckpoints=[&] {
            if(!training.process.launched()) {training.checkpointNavigation.reset();training.checkpointSelection.reset();return;}
            const auto direction=training.checkpointNavigation;
            const auto selection=training.checkpointSelection;
            training.checkpointNavigation.reset();training.checkpointSelection.reset();
            if(direction&&*direction==0) {enterTrainingView(viewer::TrainingView::Live);return;}
            const auto saved=training.checkpoints();
            if(saved.empty()) {training.message="Waiting for the first saved checkpoint";return;}
            const uint64_t current=training.view==viewer::TrainingView::History?training.historyIteration:training.viewedIteration;
            std::size_t index=0;
            if(selection)index=std::min(*selection,saved.size()-1);
            else if(direction)index=viewer::checkpointNavigationIndex(saved,current,*direction);
            else return;
            enterTrainingView(viewer::TrainingView::History);
            training.historyPath=saved[index].path;training.historyIteration=saved[index].iteration;
        };
        std::filesystem::path previewSettingsRun, framedTrainingRun;
        std::chrono::steady_clock::time_point lastTrainingViewPoll{};
        auto pollTrainingScene = [&]() {
            if (training.process.launched() && previewSettingsRun != training.process.directory && training.status["scene"].IsScalar()) {
                const auto savedRun = viewer::readJson(training.process.runDirectory / "run_config.json");
                const auto saved = savedRun["renderer_settings"];
                if (saved.IsMap()) {
                    settings = Pale::PathTracerSettings{};
                    settings.integratorKind = Pale::IntegratorKind::photonMapping;
                    settings.cameraGatherKernelKind = Pale::CameraGatherKernelKind::CameraGatherKernel2;
                    settings.pointGeometrySupportRadius = 0.1f;
                    settings.pointGeometryReconstructionLength = 0.0f;
                    settings.pointGeometryCoverageScale = 1.01f;
#define VIEWER_RUN_SETTING(key, member, type) if (saved[key]) settings.member = saved[key].as<type>()
                    VIEWER_RUN_SETTING("local_layer_depth_epsilon", rendererDebugLocalLayerDepthEpsilon, float);
                    VIEWER_RUN_SETTING("local_layer_normal_cosine_threshold", rendererDebugLocalLayerNormalCosineThreshold, float);
                    VIEWER_RUN_SETTING("share_local_layer_direct_lighting", rendererDebugShareLocalLayerDirectLighting, bool);
                    VIEWER_RUN_SETTING("primal_shadow_rays", numShadowRays, uint32_t);
                    VIEWER_RUN_SETTING("bounces", maxBounces, uint32_t);
                    VIEWER_RUN_SETTING("forward_passes", numForwardPasses, uint32_t);
                    VIEWER_RUN_SETTING("max_splat_events_per_ray", rendererDebugMaxSplatEventsPerRay, int);
                    VIEWER_RUN_SETTING("max_local_surfel_hits", rendererDebugMaxLocalSurfelHits, int);
                    VIEWER_RUN_SETTING("point_hit_batch_size", rendererDebugPointHitBatchSize, int);
                    VIEWER_RUN_SETTING("point_hit_batch_lookahead", rendererDebugPointHitBatchLookahead, bool);
                    VIEWER_RUN_SETTING("median_depth_retain_last_hit", medianDepthRetainLastHit, bool);
#undef VIEWER_RUN_SETTING
                    if (saved["photons"]) settings.photonsPerLaunch = static_cast<uint32_t>(saved["photons"].as<double>());
                    if (saved["local_layer_depth_mode"])
                        settings.rendererDebugLocalLayerDepthMode = saved["local_layer_depth_mode"].as<std::string>() == "normal_distance"
                            ? Pale::LocalLayerDepthMode::NormalDistance : Pale::LocalLayerDepthMode::SymmetricRayDepth;
                    previewSettingsRun = training.process.directory;
                    tracerDirty = renderRequested = true;
                }
            }
            if (!modelReadOnly()) return;
            // Do not replace the model beneath a transform gesture that began
            // against the displayed snapshot. The first real edit forks it.
            if(ImGuizmo::IsUsingAny()||viewportGizmoMouseCapture)return;
            const auto now = std::chrono::steady_clock::now();
            if (now - lastTrainingViewPoll < std::chrono::milliseconds(500)) return;
            lastTrainingViewPoll = now;
            try {
                std::filesystem::path path, scenePath;
                uint64_t generation = 0, step = 0;
                if (training.view == viewer::TrainingView::Live && training.live.IsMap()) {
                    generation = training.live["generation"].as<uint64_t>();
                    step = training.live["iteration"].as<uint64_t>();
                    path = training.live["pointcloud"].as<std::string>(); scenePath = training.live["scene"].as<std::string>();
                } else if (training.view == viewer::TrainingView::Saved && training.status["scene"].IsScalar()) {
                    const auto latest = training.latestSavedSnapshot();
                    if (!latest) return;
                    step = latest->iteration; generation = step + 1;
                    path = latest->path; scenePath = training.status["scene"].as<std::string>();
                }
                if(training.view==viewer::TrainingView::History&&!training.historyPath.empty()&&training.status["scene"].IsScalar()) {
                    path=training.historyPath;step=training.historyIteration;generation=step+1;scenePath=training.status["scene"].as<std::string>();
                }
                if (path.empty() || generation == training.loadedGeneration) return;
                const auto run=training.process.directory;
                std::shared_ptr<const viewer::LivePacketLease> packetLease;
                if(training.view==viewer::TrainingView::Live) {
                    // Pin the owned immutable packet before slow scene loading.
                    // The publisher may remove its original name meanwhile.
                    packetLease=livePacketLeases.acquire(path,run);
                    path=packetLease->path();
                }
                // Checkpoint navigation invalidates the requested generation,
                // not the run's scene/camera. Reload XML only when the run or an
                // editable workspace changes, and otherwise replace its PLY.
                const bool onlyPoints=currentScenePath==scenePath && loadedTrainingSceneRun==run;
                auto loaded=std::make_shared<bool>(false);
                startPreviewJob([&,loaded,path,scenePath,onlyPoints,run,packetLease] {
                    *loaded=onlyPoints?replacePointCloud(path,true):replaceSceneAndPointCloud(scenePath,path,true);
                    if(*loaded && framedTrainingRun!=run) {
                        orbit=makeInitialOrbitCamera(buildProducts,bounds);initialOrbit=orbit;
                        cameraSource=CameraSource::Viewport;cameraDirty=true;framedTrainingRun=run;
                    }
                },[&,loaded,generation,step,run,packetLease] {
                    if(*loaded)loadedLivePacket=packetLease;
                    if(training.process.directory!=run)return;
                    if(*loaded) {loadedTrainingSceneRun=run;training.loadedGeneration=generation;pendingViewedIteration=std::make_pair(run,step);}
                    else throw std::runtime_error("Scene update failed: "+pointCloudStatus);
                },"Updating training scene");
            } catch (const std::exception& error) { training.message = error.what(); }
        };

        auto refreshLatestOptimizationPointCloud = [&]() {
            std::optional<std::filesystem::path> latestPointCloud;
            if (optimizationFolderAnchor.active()) {
                if (const auto pointsDirectory = optimizationFolderAnchor.resolve()) {
                    const auto snapshots = listOptimizationPointCloudSnapshots(*pointsDirectory);
                    if (!snapshots.empty()) latestPointCloud = snapshots.back().path;
                }
            } else {
                latestPointCloud = findLatestOptimizationPointCloud(optimizationOutputDirectory);
            }
            if (!latestPointCloud) {
                const auto searchDirectory = optimizationFolderAnchor.active()
                    ? optimizationFolderAnchor.pointsDirectory.value_or(optimizationFolderAnchor.folder)
                    : optimizationOutputDirectory;
                pointCloudStatus = "No optimization point snapshot found in " + searchDirectory.string();
                return;
            }

            std::filesystem::path pointsDirectory = latestPointCloud->parent_path();
            std::vector<PointCloudSnapshot> snapshots = listOptimizationPointCloudSnapshots(pointsDirectory);
            const auto snapshotIterator = std::find_if(
                snapshots.begin(),
                snapshots.end(),
                [&](const PointCloudSnapshot& snapshot) {
                    return equivalentPaths(snapshot.path, *latestPointCloud);
                });
            if (snapshotIterator == snapshots.end()) {
                pointCloudStatus = "Latest optimization snapshot disappeared: " + latestPointCloud->string();
                return;
            }
            const std::size_t latestSnapshotIndex =
                static_cast<std::size_t>(std::distance(snapshots.begin(), snapshotIterator));

            if (replaceOptimizationPointCloud(*latestPointCloud, true)) {
                latestOptimizationMode = true;
                latestOptimizationPointsDirectory = pointsDirectory;
                latestOptimizationSnapshots = std::move(snapshots);
                latestOptimizationSnapshotIndex = latestSnapshotIndex;
                pointCloudStatus =
                    "Latest optimization snapshot " +
                    std::to_string(latestOptimizationSnapshotIndex + 1u) + "/" +
                    std::to_string(latestOptimizationSnapshots.size()) + ": " +
                    latestPointCloud->filename().string();
            }
        };

        auto stepLatestOptimizationSnapshot = [&](int offset, std::optional<uint64_t> iterationStep = std::nullopt) {
            if (offset == 0) {
                return;
            }
            const int direction = offset < 0 ? -1 : 1;
            const std::size_t stepCount =
                offset < 0 ? static_cast<std::size_t>(-offset) : static_cast<std::size_t>(offset);

            if (!latestOptimizationMode || latestOptimizationPointsDirectory.empty()) {
                if (optimizationFolderAnchor.active()) {
                    refreshLatestOptimizationPointCloud();
                } else {
                    pointCloudStatus = "Press R, drop a training folder, or Load latest run PLY before using snapshot navigation";
                }
                return;
            }

            // Refresh only this run so newly written snapshots remain visible.
            // Folder anchors stay in their selected run until explicitly reset.
            std::vector<PointCloudSnapshot> snapshots =
                listOptimizationPointCloudSnapshots(latestOptimizationPointsDirectory);
            if (snapshots.empty()) {
                pointCloudStatus = "No optimization snapshots found in " + latestOptimizationPointsDirectory.string();
                return;
            }

            auto currentIterator = std::find_if(
                snapshots.begin(),
                snapshots.end(),
                [&](const PointCloudSnapshot& snapshot) {
                    return equivalentPaths(snapshot.path, currentPointCloudPath);
                });
            std::size_t currentIndex = currentIterator == snapshots.end()
                                           ? std::min(
                                               latestOptimizationSnapshotIndex,
                                               snapshots.size() - 1u)
                                           : static_cast<std::size_t>(std::distance(snapshots.begin(), currentIterator));

            if (direction < 0 && currentIndex == 0u) {
                latestOptimizationSnapshots = std::move(snapshots);
                latestOptimizationSnapshotIndex = 0u;
                pointCloudStatus = "Already at earliest optimization snapshot";
                return;
            }
            if (direction > 0 && currentIndex + 1u >= snapshots.size()) {
                latestOptimizationSnapshotIndex = snapshots.size() - 1u;
                latestOptimizationSnapshots = std::move(snapshots);
                pointCloudStatus = "Already at latest optimization snapshot";
                return;
            }

            std::size_t nextIndex = currentIndex;
            if (iterationStep.has_value() && *iterationStep > 0u) {
                const uint64_t currentIteration = snapshots[currentIndex].iteration;
                uint64_t targetIteration = currentIteration;
                if (direction < 0) {
                    targetIteration = currentIteration - (currentIteration % *iterationStep);
                    if (targetIteration == currentIteration && targetIteration >= *iterationStep) {
                        targetIteration -= *iterationStep;
                    }
                } else {
                    targetIteration =
                        ((currentIteration / *iterationStep) + 1u) * *iterationStep;
                }

                const auto nextIterator = std::lower_bound(
                    snapshots.begin(),
                    snapshots.end(),
                    targetIteration,
                    [](const PointCloudSnapshot& snapshot, uint64_t target) {
                        return snapshot.iteration < target;
                    });
                if (nextIterator == snapshots.begin()) {
                    nextIndex = 0u;
                } else if (nextIterator == snapshots.end()) {
                    nextIndex = snapshots.size() - 1u;
                } else {
                    const std::size_t afterIndex =
                        static_cast<std::size_t>(std::distance(snapshots.begin(), nextIterator));
                    const std::size_t beforeIndex = afterIndex - 1u;
                    const uint64_t beforeDelta = targetIteration - snapshots[beforeIndex].iteration;
                    const uint64_t afterDelta = snapshots[afterIndex].iteration - targetIteration;
                    nextIndex = beforeDelta <= afterDelta ? beforeIndex : afterIndex;
                }

                if (nextIndex == currentIndex) {
                    nextIndex = direction < 0
                                    ? (currentIndex > 0u ? currentIndex - 1u : 0u)
                                    : std::min(currentIndex + 1u, snapshots.size() - 1u);
                }
            } else {
                nextIndex = direction < 0
                                ? (currentIndex > stepCount ? currentIndex - stepCount : 0u)
                                : std::min(currentIndex + stepCount, snapshots.size() - 1u);
            }
            const std::filesystem::path nextPath = snapshots[nextIndex].path;
            if (replacePointCloud(nextPath, true)) {
                latestOptimizationMode = true;
                latestOptimizationSnapshots = std::move(snapshots);
                latestOptimizationSnapshotIndex = nextIndex;
                pointCloudStatus =
                    "Optimization snapshot " +
                    std::to_string(latestOptimizationSnapshotIndex + 1u) + "/" +
                    std::to_string(latestOptimizationSnapshots.size()) + ": " +
                    nextPath.filename().string();
            }
        };

        auto jumpToOptimizationSnapshotBoundary = [&](bool latestSnapshot) {
            std::filesystem::path pointsDirectory;
            if (optimizationFolderAnchor.active()) {
                pointsDirectory = optimizationFolderAnchor.resolve().value_or(std::filesystem::path{});
                if (pointsDirectory.empty()) {
                    pointCloudStatus = "No optimization snapshots found in " + optimizationFolderAnchor.folder.string();
                    return;
                }
            } else if (latestOptimizationMode && !latestOptimizationPointsDirectory.empty()) {
                pointsDirectory = latestOptimizationPointsDirectory;
            } else if (currentPointCloudPath.has_parent_path()) {
                const std::filesystem::path candidateDirectory = currentPointCloudPath.parent_path();
                if (!listOptimizationPointCloudSnapshots(candidateDirectory).empty()) {
                    pointsDirectory = candidateDirectory;
                }
            }

            if (pointsDirectory.empty()) {
                pointCloudStatus =
                    "Press R, Load latest run PLY, or load an iter_*_points.ply from an optimization points folder first";
                return;
            }

            std::vector<PointCloudSnapshot> snapshots = listOptimizationPointCloudSnapshots(pointsDirectory);
            if (snapshots.empty()) {
                pointCloudStatus = "No optimization snapshots found in " + pointsDirectory.string();
                return;
            }

            const std::size_t targetIndex = latestSnapshot ? snapshots.size() - 1u : 0u;
            const std::filesystem::path targetPath = snapshots[targetIndex].path;
            if (replaceOptimizationPointCloud(targetPath, true)) {
                latestOptimizationMode = true;
                latestOptimizationPointsDirectory = pointsDirectory;
                latestOptimizationSnapshots = std::move(snapshots);
                latestOptimizationSnapshotIndex = targetIndex;
                pointCloudStatus =
                    std::string(latestSnapshot ? "Latest" : "Earliest") +
                    " optimization snapshot " +
                    std::to_string(latestOptimizationSnapshotIndex + 1u) + "/" +
                    std::to_string(latestOptimizationSnapshots.size()) + ": " +
                    targetPath.filename().string();
            }
        };

        auto removeRuntimeMeshes = [&]() {
            const std::vector<Pale::Entity> runtimeMeshes = collectRuntimeMeshEntities(scene);
            for (Pale::Entity entity : runtimeMeshes) {
                scene->destroyEntity(entity);
            }
            if (!runtimeMeshes.empty()) {
                rebuildSceneGpu();
                currentRuntimeMeshPath.clear();
                meshPathBuffer.fill('\0');
                meshStatus = "Removed runtime mesh";
                cameraDirty = true;
            }
        };

        auto loadRuntimeMesh = [&](const std::filesystem::path& requestedPath) -> bool {
            if (requestedPath.empty()) {
                meshStatus = "No mesh PLY path selected";
                return false;
            }
            if (!isPlyPath(requestedPath)) {
                meshStatus = "Selected mesh file is not a .ply file";
                return false;
            }

            std::error_code error;
            if (!std::filesystem::exists(requestedPath, error) || error) {
                meshStatus = "Mesh PLY file does not exist: " + requestedPath.string();
                return false;
            }

            const std::shared_ptr<Pale::Mesh> validationMesh = loadMeshPlyForValidation(requestedPath);
            if (!validationMesh) {
                meshStatus = "Failed to load mesh PLY: " + requestedPath.string();
                return false;
            }
            const std::size_t triangleCount = countMeshTriangles(*validationMesh);
            if (triangleCount == 0) {
                meshStatus = "PLY contains no triangles: " + requestedPath.string();
                return false;
            }

            const Pale::AssetHandle meshAssetHandle =
                importPathAsType(assetManager.registry(), requestedPath, Pale::AssetType::Mesh);
            const auto meshAsset = assetAccessor.getMesh(meshAssetHandle);
            if (!meshAsset || countMeshTriangles(*meshAsset) == 0) {
                meshStatus = "Imported mesh asset contains no triangles: " + requestedPath.string();
                return false;
            }

            for (Pale::Entity entity : collectRuntimeMeshEntities(scene)) {
                scene->destroyEntity(entity);
            }

            Pale::Entity meshEntity =
                scene->createEntity("RuntimeMesh: " + requestedPath.filename().string());
            meshEntity.addComponent<Pale::MeshComponent>().meshID = meshAssetHandle;

            rebuildSceneGpu();
            currentRuntimeMeshPath = requestedPath;
            copyPathToBuffer(currentRuntimeMeshPath, meshPathBuffer);
            if (currentRuntimeMeshPath.has_parent_path()) {
                meshBrowserDirectory = currentRuntimeMeshPath.parent_path();
            }

            meshStatus = "Loaded mesh with " + std::to_string(triangleCount) + " triangles";
            cameraDirty = true;
            return true;
        };

        auto handleDroppedPly = [&](const std::filesystem::path& droppedPath) {
            std::error_code error;
            if (!std::filesystem::exists(droppedPath, error) || error) {
                meshStatus = "Dropped PLY file does not exist: " + droppedPath.string();
                return;
            }

            const std::shared_ptr<Pale::Mesh> validationMesh = loadMeshPlyForValidation(droppedPath);
            if (validationMesh && countMeshTriangles(*validationMesh) > 0) {
                loadRuntimeMesh(droppedPath);
                return;
            }

            const std::shared_ptr<Pale::PointAsset> validationPointCloud =
                loadPointCloudPlyForValidation(droppedPath);
            if (validationPointCloud && countSurfels(*validationPointCloud) > 0) {
                replacePointCloud(droppedPath);
                return;
            }

            meshStatus = "Dropped PLY is neither a triangle mesh nor a surfel point cloud";
            pointCloudStatus = meshStatus;
        };

        auto computeTargetLoss = [&]() {
            targetLossDirty = false;
            targetLossViews.clear();
            targetLossMean.clear();
            targetLossWorst.clear();
            const auto start = std::chrono::steady_clock::now();
            try {
                if (settings.sharedHeightEnabled ||
                    settings.integratorKind != Pale::IntegratorKind::photonMapping ||
                    settings.cameraGatherKernelKind != Pale::CameraGatherKernelKind::CameraGatherKernel2)
                    throw std::runtime_error("Target projection requires photon mapping / CameraGatherKernel2 with Shared surface experiment off");
                if (buildProducts.cameraGPUs.empty()) throw std::runtime_error("No scene cameras to match to target images");
                if (!targetDatasetManual) {
                    std::filesystem::path dataset;
                    auto directory = currentPointCloudPath.parent_path();
                    for (int level = 0; level < 3 && !directory.empty(); ++level, directory = directory.parent_path()) {
                        const auto configPath = directory / "run_config.json";
                        if (!std::filesystem::is_regular_file(configPath)) continue;
                        const auto config = YAML::LoadFile(configPath.string());
                        if (config["dataset_path"]) {
                            dataset = config["dataset_path"].as<std::string>();
                            if (dataset.is_relative()) {
                                // Training normally resolves paths; also accept older configs.
                                for (const auto& base : {repositoryRoot / "python", directory, repositoryRoot}) {
                                    if (std::filesystem::is_directory(base / dataset)) { dataset = base / dataset; break; }
                                }
                            }
                        }
                        if (!dataset.empty()) copyPathToBuffer(dataset, targetDatasetBuffer);
                        const auto optimization = config["optimization_config"];
                        if (optimization && optimization["target_color_space"]) {
                            const auto space = optimization["target_color_space"].as<std::string>();
                            if (space != "linear" && space != "srgb")
                                throw std::runtime_error("Saved target color space is auto; choose an explicit Linear or sRGB interpretation and Apply targets");
                            targetColorSpace = space == "srgb" ? 1 : 0;
                        }
                        break;
                    }
                    if (dataset.empty()) {
                        auto scenePath = currentScenePath;
                        if (scenePath.is_relative()) scenePath = assetsDirectory / scenePath;
                        dataset = scenePath.parent_path();
                    }
                    copyPathToBuffer(dataset, targetDatasetBuffer);
                }
                std::filesystem::path images = targetDatasetBuffer.data();
                if (std::filesystem::is_directory(images / "images")) images /= "images";
                if (!std::filesystem::is_directory(images)) throw std::runtime_error("Set the target dataset or images directory, then Apply targets");
                const auto footprints = Pale::surfaceOverlapFootprints(buildProducts, targetLossInstanceOffsets);
                const auto count = footprints.size();
                if (count == 0) throw std::runtime_error("No surfels to project onto");
                std::vector<viewer::LossInstanceRange> lossRanges(buildProducts.instances.size());
                for (std::size_t i = 0; i < buildProducts.instances.size(); ++i) {
                    const auto& instance = buildProducts.instances[i];
                    if (instance.geometryType == Pale::GeometryType::PointCloud)
                        lossRanges[i] = {targetLossInstanceOffsets[i], buildProducts.pointCloudRanges.at(instance.geometryIndex).firstPoint};
                }
                std::vector<double> totalLoss(count, 0), totalWeight(count, 0);
                targetLossWorst.assign(count, std::numeric_limits<float>::quiet_NaN());
                auto lossSettings = makeViewerDebugSettings(settings, ViewImageMode::Rendered);
                Pale::PathTracer lossTracer(queue, lossSettings);
                std::size_t missing = 0;
                for (const auto& camera : buildProducts.cameraGPUs) {
                    std::vector<std::filesystem::path> matches;
                    for (const auto& entry : std::filesystem::directory_iterator(images)) {
                        const auto suffix = toLower(entry.path().extension().string());
                        if (entry.is_regular_file() && entry.path().stem() == camera.name &&
                            (suffix == ".exr" || suffix == ".hdr" || suffix == ".png" || suffix == ".jpg" ||
                             suffix == ".jpeg" || suffix == ".bmp" || suffix == ".tga" || suffix == ".tif" || suffix == ".tiff"))
                            matches.push_back(entry.path());
                    }
                    if (matches.empty()) { ++missing; continue; }
                    if (matches.size() != 1) throw std::runtime_error("Ambiguous target files for camera " + std::string(camera.name));
                    const auto target = loadDebugTarget(matches.front(), camera, targetColorSpace == 1);
                    auto lossBuild = buildProducts;
                    lossBuild.cameraGPUs = {camera};
                    lossTracer.setScene(sceneGpu, lossBuild);
                    auto lossSensor = createSensor(queue, camera);
                    std::vector<float> losses;
                    try {
                        clearSensor(queue, lossSensor);
                        std::vector<Pale::SensorGPU> sensors{lossSensor};
                        lossTracer.renderForward(sensors);
                        losses = viewer::rgbHalfSquaredError(Pale::downloadSensorRGBARAW(queue, lossSensor), target);
                    } catch (...) { queue.wait(); destroySensor(queue, lossSensor); throw; }
                    destroySensor(queue, lossSensor);
                    const auto projected = viewer::projectTargetLoss(queue, camera, sceneGpu, lossSettings,
                        lossRanges, count, losses);
                    viewer::ProjectedTargetLoss view;
                    view.cameraName = camera.name;
                    view.pixelCount = losses.size();
                    view.mean.assign(count, std::numeric_limits<float>::quiet_NaN());
                    for (std::size_t i = 0; i < count; ++i) {
                        const auto sum = projected.sums[i];
                        if (sum.y() <= 0) continue;
                        view.mean[i] = sum.x() / sum.y();
                        totalLoss[i] += double(sum.x()) / losses.size();
                        totalWeight[i] += double(sum.y()) / losses.size();
                        targetLossWorst[i] = std::isfinite(targetLossWorst[i])
                            ? std::max(targetLossWorst[i], view.mean[i]) : view.mean[i];
                    }
                    for (std::size_t p = 0; p < losses.size(); ++p) {
                        view.imageMean += double(losses[p]) / losses.size();
                        if (projected.coverage[p] <= 0) {
                            ++view.unassignedPixels;
                            view.unassignedMean += double(losses[p]) / losses.size();
                        }
                    }
                    targetLossViews.push_back(std::move(view));
                }
                if (targetLossViews.empty()) throw std::runtime_error("No target image filenames match the scene camera names in " + images.string());
                targetLossMean.assign(count, std::numeric_limits<float>::quiet_NaN());
                for (std::size_t i = 0; i < count; ++i)
                    if (totalWeight[i] > 0) targetLossMean[i] = static_cast<float>(totalLoss[i] / totalWeight[i]);
                targetLossCamera = std::clamp(targetLossCamera, -1, static_cast<int>(targetLossViews.size()) - 1);
                targetLossStatus = "Projected " + std::to_string(targetLossViews.size()) + " target images; " +
                    std::to_string(missing) + " scene cameras without targets. Uses current renderer settings.";
            } catch (const std::exception& exception) {
                targetLossViews.clear();
                targetLossMean.clear();
                targetLossWorst.clear();
                targetLossStatus = exception.what();
            }
            targetLossComputeMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        };

        auto computeTransportScores = [&]() {
            transportDirty = false;
            transportViews.clear();
            transportScores.clear();
            transportAffected.clear();
            transportMarkers.clear();
            const auto start = std::chrono::steady_clock::now();
            try {
                if (buildProducts.cameraGPUs.empty()) throw std::runtime_error("No saved scene cameras; load an optimization snapshot with R first.");
                auto scoreSettings = makeViewerDebugSettings(settings, ViewImageMode::Rendered);
                if (transportUseSavedSettings) {
                    YAML::Node saved;
                    auto directory = currentPointCloudPath.parent_path();
                    for (int level = 0; level < 3 && !directory.empty(); ++level, directory = directory.parent_path()) {
                        const auto path = directory / "run_config.json";
                        if (std::filesystem::is_regular_file(path)) { saved = YAML::LoadFile(path.string())["renderer_settings"]; break; }
                    }
                    if (!saved || !saved.IsMap()) throw std::runtime_error("No saved renderer settings found. Load a run with R, or disable Use saved run settings.");
                    // Start from the same native defaults as the Python renderer.
                    scoreSettings = makeViewerDebugSettings(Pale::PathTracerSettings{}, ViewImageMode::Rendered);
                    scoreSettings.pointGeometrySupportRadius = 0.1f;
                    scoreSettings.pointGeometryReconstructionLength = 0.0f;
                    scoreSettings.pointGeometryRayOffsetMultiplier = 1.0f;
                    scoreSettings.pointGeometryMinimumContributors = 1u;
                    scoreSettings.pointGeometryCoverageScale = 1.01f;
#define LOAD_SCORE_SETTING(key, member, type) if (saved[key]) scoreSettings.member = saved[key].as<type>()
                    LOAD_SCORE_SETTING("local_layer_depth_epsilon", rendererDebugLocalLayerDepthEpsilon, float);
                    LOAD_SCORE_SETTING("local_layer_normal_cosine_threshold", rendererDebugLocalLayerNormalCosineThreshold, float);
                    LOAD_SCORE_SETTING("max_splat_events_per_ray", rendererDebugMaxSplatEventsPerRay, int);
                    LOAD_SCORE_SETTING("max_local_surfel_hits", rendererDebugMaxLocalSurfelHits, int);
                    LOAD_SCORE_SETTING("point_hit_batch_size", rendererDebugPointHitBatchSize, int);
                    LOAD_SCORE_SETTING("point_hit_batch_lookahead", rendererDebugPointHitBatchLookahead, bool);
                    LOAD_SCORE_SETTING("share_local_layer_direct_lighting", rendererDebugShareLocalLayerDirectLighting, bool);
                    LOAD_SCORE_SETTING("primal_shadow_rays", numShadowRays, uint32_t);
                    LOAD_SCORE_SETTING("seed", random.seed, uint32_t);
                    // PythonRenderer currently uses the native one-pass default;
                    // its gather_passes dictionary entry is not applied.
#undef LOAD_SCORE_SETTING
                    if (saved["local_layer_depth_mode"]) {
                        const auto mode = saved["local_layer_depth_mode"].as<std::string>();
                        if (mode == "normal_distance") scoreSettings.rendererDebugLocalLayerDepthMode = Pale::LocalLayerDepthMode::NormalDistance;
                        else if (mode == "symmetric_ray_depth") scoreSettings.rendererDebugLocalLayerDepthMode = Pale::LocalLayerDepthMode::SymmetricRayDepth;
                        else throw std::runtime_error("Unsupported saved local layer depth mode");
                    }
                }
                if (scoreSettings.sharedHeightEnabled) throw std::runtime_error("Contribution analysis requires Shared surface experiment off.");
                scoreSettings.integratorKind = Pale::IntegratorKind::photonMapping;
                scoreSettings.cameraGatherKernelKind = Pale::CameraGatherKernelKind::CameraGatherKernel2;
                scoreSettings.photonsPerLaunch = 1;
                scoreSettings.maxBounces = 0;
                scoreSettings.numForwardPasses = 0;

                Pale::PathTracer scoreTracer(queue, scoreSettings);
                auto activity = Pale::makePrimalActivityStatsForScene(queue, buildProducts);
                scoreTracer.setPrimalActivityStats(&activity);
                try {
                    for (const auto& camera : buildProducts.cameraGPUs) {
                        auto scoreBuild = buildProducts;
                        scoreBuild.cameraGPUs = {camera};
                        auto scoreScene = sceneGpu;
                        scoreScene.profileCounters = nullptr;
                        scoreTracer.setScene(scoreScene, scoreBuild);
                        auto scoreSensor = createSensor(queue, camera);
                        viewer::TransportCameraScores view;
                        view.name = camera.name;
                        view.training = camera.useForAdjointPass;
                        view.camera.resize(activity.numPoints);
                        view.shadow.resize(activity.numPoints);
                        try {
                            clearSensor(queue, scoreSensor);
                            std::vector<Pale::SensorGPU> sensors{scoreSensor};
                            scoreTracer.renderForward(sensors);
                            if (activity.numPoints) {
                                queue.memcpy(view.camera.data(), activity.cameraSurfaceScore, activity.numPoints * sizeof(float)).wait_and_throw();
                                queue.memcpy(view.shadow.data(), activity.shadowOccluderScore, activity.numPoints * sizeof(float)).wait_and_throw();
                            }
                        } catch (...) { queue.wait(); destroySensor(queue, scoreSensor); throw; }
                        destroySensor(queue, scoreSensor);
                        transportViews.push_back(std::move(view));
                    }
                } catch (...) {
                    queue.wait(); scoreTracer.setPrimalActivityStats(nullptr);
                    Pale::freePrimalActivityStats(queue, activity); throw;
                }
                scoreTracer.setPrimalActivityStats(nullptr);
                Pale::freePrimalActivityStats(queue, activity);
                transportSource = std::clamp(transportSource, -1, int(transportViews.size()) - 1);
                refreshTransportSelection();
                std::vector<std::size_t> offsets;
                const auto footprints = Pale::surfaceOverlapFootprints(buildProducts, offsets);
                const auto editableHandle = firstPointCloudHandle(scene);
                const auto editable = editableHandle ? buildProducts.pointCloudIndexById.find(*editableHandle) : buildProducts.pointCloudIndexById.end();
                for (std::size_t instanceIndex = 0; instanceIndex < buildProducts.instances.size(); ++instanceIndex) {
                    const auto& instance = buildProducts.instances[instanceIndex];
                    if (instance.geometryType != Pale::GeometryType::PointCloud) continue;
                    const auto& range = buildProducts.pointCloudRanges.at(instance.geometryIndex);
                    for (std::size_t i = 0; i < range.pointCount; ++i) {
                        const auto primitive = range.firstPoint + i;
                        if (!buildProducts.points[primitive].isEmissive())
                            transportMarkers.push_back({footprints[offsets[instanceIndex] + i].center, primitive, editable != buildProducts.pointCloudIndexById.end() && editable->second == instance.geometryIndex ? static_cast<int>(i) : -1});
                    }
                }
                const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                transportStatus = "Computed " + std::to_string(transportViews.size()) + " cameras at native resolution in " +
                    std::to_string(seconds) + " s. " + (transportUseSavedSettings ? "Saved run traversal settings." : "Current viewer traversal settings.");
            } catch (const std::exception& error) {
                transportViews.clear(); transportScores.clear(); transportAffected.clear(); transportMarkers.clear();
                transportStatus = error.what();
            }
        };

        auto ensureDebugDisplayBuffer = [&](ViewImageMode mode, bool prepare) -> bool {
            if (!hasSensor || displayedRenderWidth == 0u || displayedRenderHeight == 0u) {
                return false;
            }
            if (sensor.width != displayedRenderWidth || sensor.height != displayedRenderHeight) {
                return false;
            }

            debugDisplayBuffers.prepareFor(displayedRenderWidth, displayedRenderHeight);
            const std::size_t pixelCount =
                static_cast<std::size_t>(displayedRenderWidth) * static_cast<std::size_t>(displayedRenderHeight);

            // UI recoloring may reuse host data, but a missing map must never
            // compile a diagnostic kernel or download device memory here.
            if (!prepare) {
                bool ready = false;
                switch (mode) {
                    case ViewImageMode::Rendered: ready = true; break;
                    case ViewImageMode::RawLinear: ready = !rawLinearPixels.empty(); break;
                    case ViewImageMode::AdjointDerivative: ready = viewerAdjointImageValid; break;
                    case ViewImageMode::TransportScore:
                    case ViewImageMode::TargetLoss: ready = debugDisplayBuffers.positionPrimitiveIndicesValid; break;
                    case ViewImageMode::SurfaceOverlap: ready = debugDisplayBuffers.surfaceOverlapValid; break;
                    case ViewImageMode::TerminalTransmittance: ready = debugDisplayBuffers.terminalTransmittanceValid; break;
                    case ViewImageMode::SurfelDensity: ready = debugDisplayBuffers.surfelDensityValid; break;
                    case ViewImageMode::MeanDepth: ready = debugDisplayBuffers.meanDepthValid; break;
                    case ViewImageMode::MedianDepth: ready = debugDisplayBuffers.medianDepthValid; break;
                    case ViewImageMode::VisibleNormal: ready = debugDisplayBuffers.visibleNormalValid; break;
                    case ViewImageMode::MedianDepthNormal:
                    case ViewImageMode::MeanDepthNormal: ready = debugDisplayBuffers.depthNormalValid; break;
                    case ViewImageMode::DepthDistortion: ready = debugDisplayBuffers.depthDistortionValid; break;
                    case ViewImageMode::IntraSlabDepth:
                    case ViewImageMode::IntraSlabRayDepth: ready = debugDisplayBuffers.intraSlabDepthValid; break;
                    case ViewImageMode::PositionPrimitiveScore: ready = debugDisplayBuffers.positionPrimitiveScoreValid; break;
                    case ViewImageMode::DensificationOrigin: ready = debugDisplayBuffers.densificationOriginValid; break;
                    case ViewImageMode::PrimitiveAge: ready = debugDisplayBuffers.primitiveAgeValid; break;
                    case ViewImageMode::DepthPositionGradient: ready = debugDisplayBuffers.depthPositionGradientValid; break;
                    case ViewImageMode::NormalPositionGradient: ready = debugDisplayBuffers.normalPositionGradientValid; break;
                    case ViewImageMode::IntraSlabPositionGradient: ready = debugDisplayBuffers.intraSlabPositionGradientValid; break;
                }
                if (!ready) { renderRequested = true; return false; }
                // Target loss can change its aggregation without retracing.
                if (mode != ViewImageMode::TargetLoss) return true;
            }

            const auto ensureVisiblePrimitiveIndices = [&]() -> bool {
                if (!sensor.visiblePrimitiveIndexBuffer) {
                    return false;
                }
                if (!debugDisplayBuffers.visiblePrimitiveIndicesValid) {
                    debugDisplayBuffers.visiblePrimitiveIndices =
                        Pale::downloadUint32Buffer(
                            queue, sensor.visiblePrimitiveIndexBuffer, pixelCount);
                    debugDisplayBuffers.visiblePrimitiveIndicesValid = true;
                }
                return true;
            };

            const auto ensurePositionPrimitiveIndices = [&]() {
                if (debugDisplayBuffers.positionPrimitiveIndicesValid) {
                    return;
                }
                // The position score is defined per primitive, independently
                // of neighboring depth/normal validity. Trace the frontmost
                // surfel directly instead of borrowing the visible-slab map.
                const auto camera = sensor.camera;
                auto previewScene = sceneGpu;
                previewScene.profileCounters = nullptr;
                auto& hostIndices = debugDisplayBuffers.positionPrimitiveIndices;
                hostIndices.resize(pixelCount);
                auto& hostInstances = debugDisplayBuffers.positionInstanceIndices;
                hostInstances.resize(pixelCount);
                std::vector<sycl::uint2> hits(pixelCount);
                sycl::uint2* output = sycl::malloc_device<sycl::uint2>(pixelCount, queue);
                if (!output) {
                    throw std::runtime_error("Failed to allocate position preview primitive indices");
                }
                try {
                    queue.parallel_for(sycl::range<1>(pixelCount), [=](sycl::id<1> id) {
                        const uint32_t pixel = static_cast<uint32_t>(id[0]);
                        const auto ray = Pale::makePrimaryRayFromPixelJitteredFov(
                            camera, static_cast<float>(pixel % camera.width),
                            static_cast<float>(pixel / camera.width), 0.0f, 0.0f);
                        Pale::WorldHit hit{};
                        Pale::intersectScene(ray, &hit, previewScene, Pale::SurfelIntersectMode::FirstHit);
                        output[id[0]] = hit.hit &&
                            previewScene.instances[hit.instanceIndex].geometryType == Pale::GeometryType::PointCloud
                            ? sycl::uint2{hit.primitiveIndex, hit.instanceIndex}
                            : sycl::uint2{UINT32_MAX, UINT32_MAX};
                    }).wait_and_throw();
                    queue.memcpy(hits.data(), output, pixelCount * sizeof(sycl::uint2)).wait_and_throw();
                    for (std::size_t i = 0; i < pixelCount; ++i) {
                        hostIndices[i] = hits[i].x();
                        hostInstances[i] = hits[i].y();
                    }
                } catch (...) {
                    // Complete any submitted work before releasing its storage.
                    queue.wait();
                    sycl::free(output, queue);
                    throw;
                }
                sycl::free(output, queue);
                debugDisplayBuffers.positionPrimitiveIndicesValid = true;
            };

            const auto makePositionGradientMap = [&](const Pale::PointGradients& gradients) {
                std::vector<float> map(pixelCount, -1.0f);
                if (!ensureVisiblePrimitiveIndices() ||
                    !gradients.gradPosition ||
                    gradients.numPoints != sceneGpu.pointCount) {
                    return map;
                }
                std::vector<Pale::float3> hostGradients(gradients.numPoints);
                queue.memcpy(
                    hostGradients.data(),
                    gradients.gradPosition,
                    gradients.numPoints * sizeof(Pale::float3)).wait();
                std::vector<float> gradientNorms(gradients.numPoints, 0.0f);
                for (std::size_t primitiveIndex = 0u;
                     primitiveIndex < gradients.numPoints;
                     ++primitiveIndex) {
                    const Pale::float3 gradient = hostGradients[primitiveIndex];
                    const float normSquared =
                        gradient.x() * gradient.x() +
                        gradient.y() * gradient.y() +
                        gradient.z() * gradient.z();
                    gradientNorms[primitiveIndex] =
                        std::isfinite(normSquared) && normSquared >= 0.0f
                            ? std::sqrt(normSquared)
                            : 0.0f;
                }
                for (std::size_t pixelIndex = 0u; pixelIndex < pixelCount; ++pixelIndex) {
                    const uint32_t primitiveIndex =
                        debugDisplayBuffers.visiblePrimitiveIndices[pixelIndex];
                    if (primitiveIndex < gradientNorms.size()) {
                        map[pixelIndex] = gradientNorms[primitiveIndex];
                    }
                }
                return map;
            };

            const auto makeDensificationOriginMap = [&]() {
                constexpr std::uint8_t kNoVisibleSurfel = 255u;
                std::vector<std::uint8_t> map(pixelCount, kNoVisibleSurfel);
                if (!ensureVisiblePrimitiveIndices()) {
                    return map;
                }

                const std::optional<Pale::AssetHandle> pointCloudHandle =
                    firstPointCloudHandle(scene);
                const std::shared_ptr<Pale::PointAsset> pointCloudAsset =
                    pointCloudHandle ? assetAccessor.getPointCloud(*pointCloudHandle) : nullptr;
                if (!pointCloudAsset) {
                    return map;
                }

                std::vector<std::uint8_t> primitiveOrigins;
                primitiveOrigins.reserve(sceneGpu.pointCount);
                for (const Pale::PointGeometry& geometry : pointCloudAsset->points) {
                    if (geometry.densificationOrigins.size() == geometry.positions.size()) {
                        primitiveOrigins.insert(
                            primitiveOrigins.end(),
                            geometry.densificationOrigins.begin(),
                            geometry.densificationOrigins.end());
                    } else {
                        // PLYs predating this field are intentionally shown as initial/unknown.
                        primitiveOrigins.insert(
                            primitiveOrigins.end(), geometry.positions.size(), 0u);
                    }
                }

                // Visible primitive indices are scene-GPU indices.  Only color when this
                // point asset is exactly the rendered point stream.
                if (primitiveOrigins.size() != sceneGpu.pointCount) {
                    return map;
                }
                for (std::size_t pixelIndex = 0u; pixelIndex < pixelCount; ++pixelIndex) {
                    const uint32_t primitiveIndex =
                        debugDisplayBuffers.visiblePrimitiveIndices[pixelIndex];
                    if (primitiveIndex < primitiveOrigins.size()) {
                        map[pixelIndex] = primitiveOrigins[primitiveIndex];
                    }
                }
                return map;
            };

            const auto makePrimitiveAgeMap = [&]() {
                constexpr std::uint32_t kNoVisibleSurfel =
                    std::numeric_limits<std::uint32_t>::max();
                constexpr std::uint32_t kUnknownAge = kNoVisibleSurfel - 1u;
                std::vector<std::uint32_t> map(pixelCount, kNoVisibleSurfel);
                if (!ensureVisiblePrimitiveIndices()) {
                    return map;
                }

                const std::optional<Pale::AssetHandle> pointCloudHandle =
                    firstPointCloudHandle(scene);
                const std::shared_ptr<Pale::PointAsset> pointCloudAsset =
                    pointCloudHandle ? assetAccessor.getPointCloud(*pointCloudHandle) : nullptr;
                if (!pointCloudAsset) {
                    return map;
                }

                std::vector<std::uint32_t> primitiveAges;
                primitiveAges.reserve(sceneGpu.pointCount);
                for (const Pale::PointGeometry& geometry : pointCloudAsset->points) {
                    if (geometry.primitiveAges.size() == geometry.positions.size()) {
                        primitiveAges.insert(
                            primitiveAges.end(),
                            geometry.primitiveAges.begin(),
                            geometry.primitiveAges.end());
                    } else {
                        // A PLY predating primitive_age is treated as fully cold.
                        primitiveAges.insert(
                            primitiveAges.end(), geometry.positions.size(), kUnknownAge);
                    }
                }

                if (primitiveAges.size() != sceneGpu.pointCount) {
                    return map;
                }
                for (std::size_t pixelIndex = 0u; pixelIndex < pixelCount; ++pixelIndex) {
                    const std::uint32_t primitiveIndex =
                        debugDisplayBuffers.visiblePrimitiveIndices[pixelIndex];
                    if (primitiveIndex < primitiveAges.size()) {
                        map[pixelIndex] = primitiveAges[primitiveIndex];
                    }
                }
                return map;
            };

            switch (mode) {
                case ViewImageMode::AdjointDerivative:
                    return viewerAdjointImageValid && viewerAdjointImage.size() == pixelCount;
                case ViewImageMode::TransportScore:
                    ensurePositionPrimitiveIndices();
                    return true;
                case ViewImageMode::TargetLoss: {
                    ensurePositionPrimitiveIndices();
                    auto& map = debugDisplayBuffers.targetLoss;
                    map.assign(pixelCount, -1.0f);
                    const auto& scores = targetLossCamera >= 0 && targetLossCamera < int(targetLossViews.size())
                        ? targetLossViews[targetLossCamera].mean
                        : (targetLossAggregation == 0 ? targetLossMean : targetLossWorst);
                    for (std::size_t p = 0; p < pixelCount; ++p) {
                        const auto instanceIndex = debugDisplayBuffers.positionInstanceIndices[p];
                        if (instanceIndex >= buildProducts.instances.size()) continue;
                        const auto& instance = buildProducts.instances[instanceIndex];
                        const auto& range = buildProducts.pointCloudRanges.at(instance.geometryIndex);
                        const auto primitive = debugDisplayBuffers.positionPrimitiveIndices[p];
                        if (primitive < range.firstPoint || primitive - range.firstPoint >= range.pointCount) continue;
                        map[p] = -2.0f;
                        if (instanceIndex >= targetLossInstanceOffsets.size()) continue;
                        const auto index = targetLossInstanceOffsets[instanceIndex] + primitive - range.firstPoint;
                        if (index < scores.size() && std::isfinite(scores[index])) map[p] = scores[index];
                    }
                    return true;
                }
                case ViewImageMode::SurfaceOverlap: {
                    const auto inherited = Pale::surfaceOverlapSettings(settings);
                    std::vector<viewer::SurfaceOverlapView> views;
                    const auto addView = [&](const Pale::CameraGPU& camera) {
                        views.push_back(Pale::surfaceOverlapView(camera));
                    };
                    for (const auto& camera : buildProducts.cameraGPUs) addView(camera);
                    if (views.empty()) addView(sensor.camera);
                    if (inherited != surfaceOverlapSettings || views != surfaceOverlapViews) {
                        surfaceOverlapSettings = inherited;
                        surfaceOverlapViews = std::move(views);
                        surfaceOverlapScoresValid = false;
                        debugDisplayBuffers.surfaceOverlapValid = false;
                    }
                    if (!surfaceOverlapScoresValid) {
                        const auto start = std::chrono::steady_clock::now();
                        const auto footprints = Pale::surfaceOverlapFootprints(buildProducts, surfaceOverlapInstanceOffsets);
                        surfaceOverlapScores = viewer::SurfaceOverlap(footprints, surfaceOverlapSettings).evaluate(surfaceOverlapViews);
                        surfaceOverlapPeak = 0.0f;
                        surfaceOverlapEligibleCount = 0u;
                        for (const auto& score : surfaceOverlapScores) {
                            if (!std::isfinite(score.meanMembers)) continue;
                            ++surfaceOverlapEligibleCount;
                            surfaceOverlapPeak = std::max(surfaceOverlapPeak, score.meanMembers);
                        }
                        surfaceOverlapComputeMs = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start).count();
                        surfaceOverlapScoresValid = true;
                    }
                    if (!debugDisplayBuffers.surfaceOverlapValid) {
                        ensurePositionPrimitiveIndices();
                        auto& map = debugDisplayBuffers.surfaceOverlap;
                        map.assign(pixelCount, std::numeric_limits<float>::quiet_NaN());
                        for (std::size_t pixel = 0; pixel < pixelCount; ++pixel) {
                            const auto instanceIndex = debugDisplayBuffers.positionInstanceIndices[pixel];
                            if (instanceIndex >= buildProducts.instances.size()) continue;
                            const auto& instance = buildProducts.instances[instanceIndex];
                            if (instance.geometryType != Pale::GeometryType::PointCloud) continue;
                            const auto& range = buildProducts.pointCloudRanges.at(instance.geometryIndex);
                            const auto primitive = debugDisplayBuffers.positionPrimitiveIndices[pixel];
                            if (primitive < range.firstPoint || primitive - range.firstPoint >= range.pointCount) continue;
                            const auto index = surfaceOverlapInstanceOffsets[instanceIndex] + primitive - range.firstPoint;
                            const auto& score = surfaceOverlapScores.at(index);
                            map[pixel] = surfaceOverlapMetric == kMeanSlabMembersMetric ? score.meanMembers
                                : (surfaceOverlapMetric == kSlabOverflowMetric ? score.crowdedPercent : score.centerMembers);
                        }
                        debugDisplayBuffers.surfaceOverlapValid = true;
                    }
                    return true;
                }
                case ViewImageMode::TerminalTransmittance:
                    if (!debugDisplayBuffers.terminalTransmittanceValid) {
                        const auto camera = sensor.camera;
                        auto previewScene = sceneGpu;
                        previewScene.profileCounters = nullptr;
                        const auto previewSettings = settings;
                        const float target = terminalTargetAttenuation;
                        auto& values = debugDisplayBuffers.terminalTransmittance;
                        values.resize(pixelCount);
                        float* output = sycl::malloc_device<float>(pixelCount, queue);
                        if (!output) throw std::runtime_error("Failed to allocate transmittance preview");
                        try {
                                queue.parallel_for(sycl::range<1>(pixelCount), [=](sycl::id<1> id) {
                                    const auto pixel = static_cast<uint32_t>(id[0]);
                                    const auto ray = Pale::makePrimaryRayFromPixelJitteredFov(
                                        camera, static_cast<float>(pixel % camera.width),
                                        static_cast<float>(pixel / camera.width), 0.0f, 0.0f);
                                    output[id[0]] = viewer::traceTerminalTransmittanceLoss(
                                        ray, previewScene, previewSettings, target);
                                }).wait_and_throw();
                                queue.memcpy(values.data(), output, pixelCount * sizeof(float)).wait_and_throw();
                        } catch (...) {
                            queue.wait();
                            sycl::free(output, queue);
                            throw;
                        }
                        sycl::free(output, queue);
                        debugDisplayBuffers.terminalTransmittanceValid = true;
                    }
                    return true;
                case ViewImageMode::SurfelDensity:
                    if (!debugDisplayBuffers.surfelDensityValid) {
                        auto& density = debugDisplayBuffers.surfelDensity;
                        density = viewer::SurfelDensity(densityGridSize);
                        const auto& camera = sensor.camera;
                        const bool intrinsics = camera.hasPinholeIntrinsics != 0u &&
                            camera.fx > 0.0f && camera.fy > 0.0f;
                        const float focal = 0.5f * camera.height /
                            std::tan(0.5f * glm::radians(camera.fovy));
                        const float fx = intrinsics ? camera.fx : focal;
                        const float fy = intrinsics ? camera.fy : focal;
                        const float cx = intrinsics ? camera.cx : 0.5f * camera.width;
                        const float cy = intrinsics ? camera.cy : 0.5f * camera.height;
                        // Scene rebuilds keep these host points and instance transforms
                        // synchronized with viewer edits. Count each placed instance.
                        for (const auto& instance : buildProducts.instances) {
                            if (instance.geometryType != Pale::GeometryType::PointCloud) continue;
                            const auto& range = buildProducts.pointCloudRanges.at(instance.geometryIndex);
                            const auto& transform = buildProducts.transforms.at(instance.transformIndex);
                            for (std::size_t i = range.firstPoint;
                                 i < static_cast<std::size_t>(range.firstPoint) + range.pointCount; ++i) {
                                const auto& point = buildProducts.points.at(i);
                                if (point.isEmissive()) continue;
                                const auto world = Pale::transformPoint(transform.objectToWorld, point.position);
                                const auto local = Pale::transformPoint(camera.view, world);
                                density.addCameraCenter(local.x(), local.y(), local.z(),
                                                        fx, fy, cx, cy, camera.width, camera.height);
                            }
                        }
                        debugDisplayBuffers.surfelDensityValid = true;
                    }
                    return true;
                case ViewImageMode::MeanDepth:
                    if (!debugDisplayBuffers.meanDepthValid) {
                        debugDisplayBuffers.meanDepth =
                            Pale::downloadFloatBuffer(queue, sensor.meanDepthBuffer, pixelCount);
                        debugDisplayBuffers.meanDepthValid = true;
                    }
                    return true;
                case ViewImageMode::MedianDepth:
                    if (!debugDisplayBuffers.medianDepthValid) {
                        debugDisplayBuffers.medianDepth =
                            Pale::downloadFloatBuffer(queue, sensor.medianDepthBuffer, pixelCount);
                        debugDisplayBuffers.medianDepthValid = true;
                    }
                    return true;
                case ViewImageMode::VisibleNormal:
                    if (!debugDisplayBuffers.visibleNormalValid) {
                        debugDisplayBuffers.visibleNormal =
                            Pale::downloadFloat4Buffer(queue, sensor.visibleNormalBuffer, pixelCount);
                        debugDisplayBuffers.visibleNormalValid = true;
                    }
                    return true;
                case ViewImageMode::MedianDepthNormal:
                case ViewImageMode::MeanDepthNormal:
                    if (!debugDisplayBuffers.depthNormalValid) {
                        debugDisplayBuffers.depthNormal =
                            Pale::downloadFloat4Buffer(queue, sensor.normalFromDepthBuffer, pixelCount);
                        debugDisplayBuffers.depthNormalValid = true;
                    }
                    return true;
                case ViewImageMode::DepthDistortion:
                    if (!debugDisplayBuffers.depthDistortionValid) {
                        debugDisplayBuffers.depthDistortion =
                            Pale::downloadFloatBuffer(queue, sensor.depthDistortionBuffer, pixelCount);
                        debugDisplayBuffers.depthDistortionValid = true;
                    }
                    return true;
                case ViewImageMode::IntraSlabDepth:
                case ViewImageMode::IntraSlabRayDepth:
                    if (!debugDisplayBuffers.intraSlabDepthValid) {
                        debugDisplayBuffers.intraSlabDepth =
                            Pale::downloadFloatBuffer(queue, sensor.intraSlabDepthBuffer, pixelCount);
                        debugDisplayBuffers.intraSlabRayDepth =
                            Pale::downloadFloatBuffer(queue, sensor.intraSlabRayDepthBuffer, pixelCount);
                        float maximum = 0.0f;
                        double planeSum = 0.0;
                        double raySum = 0.0;
                        for (std::size_t i = 0u; i < pixelCount; ++i) {
                            const float planeLoss = debugDisplayBuffers.intraSlabDepth[i];
                            const float rayLoss = debugDisplayBuffers.intraSlabRayDepth[i];
                            if (std::isfinite(planeLoss)) {
                                maximum = std::max(maximum, planeLoss);
                                planeSum += planeLoss;
                            }
                            if (std::isfinite(rayLoss)) {
                                maximum = std::max(maximum, rayLoss);
                                raySum += rayLoss;
                            }
                        }
                        debugDisplayBuffers.slabLossColorMaximum = maximum;
                        debugDisplayBuffers.slabPlaneLossMean = planeSum / std::max(pixelCount, std::size_t{1});
                        debugDisplayBuffers.slabRayLossMean = raySum / std::max(pixelCount, std::size_t{1});
                        debugDisplayBuffers.intraSlabDepthValid = true;
                    }
                    return true;
                case ViewImageMode::PositionPrimitiveScore:
                    if (!debugDisplayBuffers.positionPrimitiveScoreValid) {
                        ensurePositionPrimitiveIndices();

                        debugDisplayBuffers.positionPrimitiveScore.assign(pixelCount, -1.0f);
                        debugDisplayBuffers.positionObservedPrimitiveScores.clear();
                        debugDisplayBuffers.positionObservedPrimitiveCount = 0u;
                        debugDisplayBuffers.positionUnsampledPrimitiveCount = 0u;
                        debugDisplayBuffers.positionPrimitiveScoreMax = 0.0f;
                        debugDisplayBuffers.positionPrimitiveSplitThreshold = 0.0f;
                        debugDisplayBuffers.positionPrimitiveMetadataAvailable = false;

                        const std::optional<Pale::AssetHandle> pointCloudHandle =
                            firstPointCloudHandle(scene);
                        const std::shared_ptr<Pale::PointAsset> pointCloudAsset =
                            pointCloudHandle ? assetAccessor.getPointCloud(*pointCloudHandle) : nullptr;
                        if (pointCloudAsset) {
                            std::vector<float> primitiveSignals;
                            std::vector<std::uint32_t> primitiveSampleCounts;
                            std::vector<float> primitiveThresholds;
                            primitiveSignals.reserve(sceneGpu.pointCount);
                            primitiveSampleCounts.reserve(sceneGpu.pointCount);
                            primitiveThresholds.reserve(sceneGpu.pointCount);

                            bool completeMetadata = true;
                            for (const Pale::PointGeometry& geometry : pointCloudAsset->points) {
                                const std::size_t pointCount = geometry.positions.size();
                                if (geometry.densificationPositionSignals.size() != pointCount ||
                                    geometry.densificationPositionSampleCounts.size() != pointCount ||
                                    geometry.densificationPositionThresholds.size() != pointCount) {
                                    completeMetadata = false;
                                    break;
                                }
                                primitiveSignals.insert(
                                    primitiveSignals.end(),
                                    geometry.densificationPositionSignals.begin(),
                                    geometry.densificationPositionSignals.end());
                                primitiveSampleCounts.insert(
                                    primitiveSampleCounts.end(),
                                    geometry.densificationPositionSampleCounts.begin(),
                                    geometry.densificationPositionSampleCounts.end());
                                primitiveThresholds.insert(
                                    primitiveThresholds.end(),
                                    geometry.densificationPositionThresholds.begin(),
                                    geometry.densificationPositionThresholds.end());
                            }

                            completeMetadata = completeMetadata &&
                                primitiveSignals.size() == sceneGpu.pointCount;
                            if (completeMetadata && !primitiveThresholds.empty()) {
                                completeMetadata = std::all_of(
                                        primitiveThresholds.begin(),
                                        primitiveThresholds.end(),
                                        [](float threshold) {
                                            return std::isfinite(threshold) && threshold > 0.0f;
                                        });
                                if (completeMetadata) {
                                    // Preserve per-primitive saved thresholds when previewing another value.
                                    auto sortedThresholds = primitiveThresholds;
                                    std::sort(sortedThresholds.begin(), sortedThresholds.end());
                                    const float savedThreshold = sortedThresholds[sortedThresholds.size() / 2u];
                                    debugDisplayBuffers.positionPrimitiveSplitThreshold =
                                        savedThreshold;
                                    debugDisplayBuffers.positionPrimitiveMetadataAvailable = true;
                                    if (positionWhatIfThresholdPointCloudPath !=
                                            currentPointCloudPath ||
                                        positionWhatIfReferenceThreshold != savedThreshold ||
                                        !std::isfinite(positionWhatIfDisplayThreshold) ||
                                        positionWhatIfDisplayThreshold <= 0.0f) {
                                        positionWhatIfDisplayThreshold = kPositionDefaultThreshold;
                                        positionWhatIfReferenceThreshold = savedThreshold;
                                        positionWhatIfThresholdPointCloudPath =
                                            currentPointCloudPath;
                                    }
                                    debugDisplayBuffers.positionObservedPrimitiveScores.reserve(
                                        primitiveSignals.size());
                                    for (std::size_t primitiveIndex = 0u;
                                         primitiveIndex < primitiveSignals.size();
                                         ++primitiveIndex) {
                                        float signal = primitiveSignals[primitiveIndex];
                                        if (primitiveSampleCounts[primitiveIndex] == 0u) {
                                            primitiveSignals[primitiveIndex] = -2.0f;
                                            ++debugDisplayBuffers.positionUnsampledPrimitiveCount;
                                            continue;
                                        }
                                        if (!std::isfinite(signal) || signal < 0.0f) {
                                            primitiveSignals[primitiveIndex] = -1.0f;
                                            continue;
                                        }
                                        signal *= savedThreshold / primitiveThresholds[primitiveIndex];
                                        primitiveSignals[primitiveIndex] = signal;
                                        debugDisplayBuffers.positionObservedPrimitiveScores.push_back(
                                            signal);
                                        ++debugDisplayBuffers.positionObservedPrimitiveCount;
                                        debugDisplayBuffers.positionPrimitiveScoreMax = std::max(
                                            debugDisplayBuffers.positionPrimitiveScoreMax, signal);
                                    }
                                    for (std::size_t pixelIndex = 0u;
                                         pixelIndex < pixelCount;
                                         ++pixelIndex) {
                                        const uint32_t primitiveIndex =
                                            debugDisplayBuffers.positionPrimitiveIndices[pixelIndex];
                                        if (primitiveIndex < primitiveSignals.size()) {
                                            debugDisplayBuffers.positionPrimitiveScore[pixelIndex] =
                                                primitiveSignals[primitiveIndex];
                                        }
                                    }
                                }
                            }
                        }
                        debugDisplayBuffers.positionPrimitiveScoreValid = true;
                    }
                    return true;
                case ViewImageMode::DensificationOrigin:
                    if (!debugDisplayBuffers.densificationOriginValid) {
                        debugDisplayBuffers.densificationOrigin =
                            makeDensificationOriginMap();
                        debugDisplayBuffers.densificationOriginValid = true;
                    }
                    return true;
                case ViewImageMode::PrimitiveAge:
                    if (!debugDisplayBuffers.primitiveAgeValid) {
                        debugDisplayBuffers.primitiveAge = makePrimitiveAgeMap();
                        debugDisplayBuffers.primitiveAgeValid = true;
                    }
                    return true;
                case ViewImageMode::DepthPositionGradient:
                    if (!debugDisplayBuffers.depthPositionGradientValid) {
                        debugDisplayBuffers.depthPositionGradient =
                            makePositionGradientMap(viewerRegularizerGradients);
                        debugDisplayBuffers.depthPositionGradientValid = true;
                    }
                    return true;
                case ViewImageMode::NormalPositionGradient:
                    if (!debugDisplayBuffers.normalPositionGradientValid) {
                        debugDisplayBuffers.normalPositionGradient =
                            makePositionGradientMap(viewerRegularizerGradients);
                        debugDisplayBuffers.normalPositionGradientValid = true;
                    }
                    return true;
                case ViewImageMode::IntraSlabPositionGradient:
                    if (!debugDisplayBuffers.intraSlabPositionGradientValid) {
                        debugDisplayBuffers.intraSlabPositionGradient =
                            makePositionGradientMap(viewerRegularizerGradients);
                        debugDisplayBuffers.intraSlabPositionGradientValid = true;
                    }
                    return true;
                case ViewImageMode::RawLinear:
                    return !rawLinearPixels.empty();
                case ViewImageMode::Rendered:
                    return true;
            }

            return false;
        };

        auto updateDisplayTexture = [&]() {
            if (renderPixels.empty()) {
                return;
            }

            if (viewImageMode == ViewImageMode::RawLinear) {
                if (rawLinearPixels.empty()) return;
                pixels = rawLinearPixels;
                screenshotPixels = pixels; // Opaque PNG: do not attenuate RGB a second time.
                texture.update(pixels, displayedRenderWidth, displayedRenderHeight);
                return;
            }

            if (viewImageMode != ViewImageMode::Rendered) {
                if (!ensureDebugDisplayBuffer(viewImageMode, false)) {
                    if (viewImageMode == ViewImageMode::AdjointDerivative) {
                        // A disabled/failed derivative pass must not show an old map.
                        texture.update(renderPixels, displayedRenderWidth, displayedRenderHeight);
                        screenshotPixels = renderPixels;
                    }
                    return;
                }

                switch (viewImageMode) {
                    case ViewImageMode::TransportScore: {
                        const auto& indices = debugDisplayBuffers.positionPrimitiveIndices;
                        pixels.assign(indices.size() * 4u, 0u);
                        for (std::size_t p = 0; p < indices.size(); ++p) {
                            pixels[4*p+3] = 255;
                            const auto index = indices[p];
                            if (index >= transportScores.size() || buildProducts.points[index].isEmissive()) continue;
                            const bool affected = transportScores[index] < transportThreshold;
                            pixels[4*p] = affected ? 255 : 45;
                            pixels[4*p+1] = affected ? 45 : 150;
                            pixels[4*p+2] = affected ? 140 : 100;
                        }
                        break;
                    }
                    case ViewImageMode::TargetLoss: {
                        auto values = debugDisplayBuffers.targetLoss;
                        float maximum = targetLossColorMaximum;
                        if (targetLossLogColors) {
                            for (auto& value : values) if (value >= 0) value = std::log1p(value / maximum * 100.0f);
                            maximum = std::log1p(100.0f);
                        }
                        colorizeScalarBufferFixedRange(values, displayedRenderWidth, displayedRenderHeight,
                            0.0f, maximum, scalarColorMap, pixels);
                        // Unknown surfels are gray; background stays black.
                        for (std::size_t p = 0; p < values.size(); ++p) {
                            if (values[p] == -1.0f || values[p] == -2.0f) {
                                const uint8_t value = values[p] == -2.0f ? 128u : 0u;
                                pixels[4*p] = pixels[4*p+1] = pixels[4*p+2] = value;
                            }
                        }
                        break;
                    }
                    case ViewImageMode::SurfaceOverlap:
                        colorizeScalarBufferFixedRange(
                            debugDisplayBuffers.surfaceOverlap, displayedRenderWidth, displayedRenderHeight,
                            0.0f, surfaceOverlapMetric == kSlabOverflowMetric ? 100.0f : surfaceOverlapColorMaximum,
                            scalarColorMap, pixels);
                        break;
                    case ViewImageMode::TerminalTransmittance: {
                        const auto& values = debugDisplayBuffers.terminalTransmittance;
                        colorizeScalarBufferFixedRange(values, displayedRenderWidth, displayedRenderHeight,
                            0.0f, terminalTargetAttenuation * terminalTargetAttenuation, scalarColorMap, pixels);
                        for (std::size_t p = 0; p < values.size(); ++p) {
                            if (values[p] == 0.0f) {
                                pixels[4*p] = pixels[4*p+1] = pixels[4*p+2] = 0;
                            } else if (values[p] < 0.0f) {
                                pixels[4*p] = pixels[4*p+2] = 255;
                                pixels[4*p+1] = 0;
                            }
                        }
                        break;
                    }
                    case ViewImageMode::SurfelDensity: {
                        const auto& density = debugDisplayBuffers.surfelDensity;
                        const std::size_t grid = static_cast<std::size_t>(density.gridSize);
                        pixels.assign(static_cast<std::size_t>(displayedRenderWidth) * displayedRenderHeight * 4u, 0u);
                        const float maximum = std::max(densityColorMaximum, 1.0f);
                        for (std::size_t y = 0; y < displayedRenderHeight; ++y) {
                            for (std::size_t x = 0; x < displayedRenderWidth; ++x) {
                                const auto count = density.counts[(y * grid / displayedRenderHeight) * grid +
                                                                  x * grid / displayedRenderWidth];
                                const std::size_t offset = (y * displayedRenderWidth + x) * 4u;
                                pixels[offset + 3u] = 255u;
                                if (count == 0u) continue;
                                const float value = densityLogScale
                                    ? std::log1p(static_cast<float>(count)) / std::log1p(maximum)
                                    : static_cast<float>(count) / maximum;
                                const auto color = scalarColor(std::clamp(value, 0.0f, 1.0f), scalarColorMap);
                                pixels[offset] = channelToByte(color.r);
                                pixels[offset + 1u] = channelToByte(color.g);
                                pixels[offset + 2u] = channelToByte(color.b);
                            }
                        }
                        break;
                    }
                    case ViewImageMode::MeanDepth:
                        colorizeScalarBuffer(
                            debugDisplayBuffers.meanDepth,
                            displayedRenderWidth,
                            displayedRenderHeight,
                            true,
                            false,
                            scalarColorMap,
                            pixels);
                        break;
                    case ViewImageMode::MedianDepth:
                        colorizeScalarBuffer(
                            debugDisplayBuffers.medianDepth,
                            displayedRenderWidth,
                            displayedRenderHeight,
                            true,
                            false,
                            scalarColorMap,
                            pixels);
                        break;
                    case ViewImageMode::VisibleNormal:
                        colorizeNormalBuffer(
                            debugDisplayBuffers.visibleNormal,
                            displayedRenderWidth,
                            displayedRenderHeight,
                            pixels);
                        break;
                    case ViewImageMode::MedianDepthNormal:
                    case ViewImageMode::MeanDepthNormal:
                        colorizeNormalBuffer(
                            debugDisplayBuffers.depthNormal,
                            displayedRenderWidth,
                            displayedRenderHeight,
                            pixels);
                        break;
                    case ViewImageMode::DepthDistortion:
                        colorizeScalarBuffer(
                            debugDisplayBuffers.depthDistortion,
                            displayedRenderWidth,
                            displayedRenderHeight,
                            false,
                            true,
                            scalarColorMap,
                            pixels);
                        break;
                    case ViewImageMode::IntraSlabDepth:
                    case ViewImageMode::IntraSlabRayDepth:
                        colorizeScalarBuffer(
                            viewImageMode == ViewImageMode::IntraSlabDepth
                                ? debugDisplayBuffers.intraSlabDepth : debugDisplayBuffers.intraSlabRayDepth,
                            displayedRenderWidth,
                            displayedRenderHeight,
                            false,
                            true,
                            scalarColorMap,
                            pixels,
                            debugDisplayBuffers.slabLossColorMaximum);
                        break;
                    case ViewImageMode::PositionPrimitiveScore:
                        colorizeThresholdedPrimitiveScores(
                            debugDisplayBuffers.positionPrimitiveScore,
                            displayedRenderWidth,
                            displayedRenderHeight,
                            positionWhatIfDisplayThreshold,
                            scalarColorMap,
                            pixels,
                            true);
                        break;
                    case ViewImageMode::DensificationOrigin:
                        colorizeDensificationOrigins(
                            debugDisplayBuffers.densificationOrigin,
                            displayedRenderWidth,
                            displayedRenderHeight,
                            pixels);
                        break;
                    case ViewImageMode::PrimitiveAge:
                        colorizePrimitiveAges(
                            debugDisplayBuffers.primitiveAge,
                            displayedRenderWidth,
                            displayedRenderHeight,
                            static_cast<std::uint32_t>(
                                std::max(primitiveAgeColdAfterIterations, 1)),
                            pixels);
                        break;
                    case ViewImageMode::DepthPositionGradient:
                        colorizeScalarBuffer(
                            debugDisplayBuffers.depthPositionGradient,
                            displayedRenderWidth,
                            displayedRenderHeight,
                            false,
                            true,
                            scalarColorMap,
                            pixels);
                        break;
                    case ViewImageMode::NormalPositionGradient:
                        colorizeScalarBuffer(
                            debugDisplayBuffers.normalPositionGradient,
                            displayedRenderWidth,
                            displayedRenderHeight,
                            false,
                            true,
                            scalarColorMap,
                            pixels);
                        break;
                    case ViewImageMode::IntraSlabPositionGradient:
                        colorizeScalarBuffer(
                            debugDisplayBuffers.intraSlabPositionGradient,
                            displayedRenderWidth,
                            displayedRenderHeight,
                            false,
                            true,
                            scalarColorMap,
                            pixels);
                        break;
                    case ViewImageMode::AdjointDerivative:
                        pixels.resize(viewerAdjointImage.size() * 4u);
                        for (std::size_t p = 0; p < viewerAdjointImage.size(); ++p) {
                            const float g = viewerAdjointImage[p];
                            const auto value = std::isfinite(g) ? static_cast<uint8_t>(255.0f * std::clamp(
                                std::abs(g) / viewerAdjointColorMaximum, 0.0f, 1.0f)) : uint8_t{0};
                            pixels[4*p] = std::isfinite(g) && g < 0.0f ? 255u - value : 255u;
                            pixels[4*p+1] = std::isfinite(g) ? 255u - value : 0u;
                            pixels[4*p+2] = std::isfinite(g) && g > 0.0f ? 255u - value : 255u;
                            pixels[4*p+3] = 255u;
                        }
                        break;
                    case ViewImageMode::RawLinear: // Handled before the diagnostic switch.
                    case ViewImageMode::Rendered:
                        break;
                }
                screenshotPixels = viewImageMode == ViewImageMode::AdjointDerivative ? pixels :
                    viewer::transparentScreenshotPixels(
                        pixels, renderPixels, displayedRenderWidth, displayedRenderHeight);
                texture.update(pixels, displayedRenderWidth, displayedRenderHeight);
                return;
            }

            if (rawRenderPixels.empty()) return;
            screenshotPixels = viewer::straightAlphaScreenshot(rawRenderPixels, displayedTransform);
            composeViewportPixels(
                rawRenderPixels,
                displayedTransform,
                displayedOrbit,
                displayedCamera,
                displayedBounds,
                displayedRenderWidth,
                displayedRenderHeight,
                displayedCameraSource == CameraSource::Viewport && showViewportGrid,
                forceRenderOpacity,
                viewportBackgroundColor,
                viewportBackgroundCache,
                pixels);
            texture.update(pixels, displayedRenderWidth, displayedRenderHeight);
        };

        auto setViewImageMode = [&](ViewImageMode nextMode) {
            if (viewImageMode == nextMode) {
                return;
            }

            // A newly selected diagnostic needs fresh buffers. Never display
            viewImageMode = nextMode;
            previewRecovery.manualRetry();
            renderRequested = true;
            if (nextMode == ViewImageMode::Rendered) {
                updateDisplayTexture();
            }
        };

        const auto cycleViewImageMode = [&](int direction) {
            const std::size_t modeCount = kViewImageModeCycleOrder.size();
            std::size_t currentIndex = 0u;
            for (std::size_t index = 0u; index < modeCount; ++index) {
                if (kViewImageModeCycleOrder[index] == viewImageMode) {
                    currentIndex = index;
                    break;
                }
            }

            const std::size_t nextIndex = direction > 0
                ? (currentIndex + 1u) % modeCount
                : (currentIndex + modeCount - 1u) % modeCount;
            setViewImageMode(kViewImageModeCycleOrder[nextIndex]);
        };

        auto ensureViewerAdjointGradients = [&]() {
            const bool hasUsableGradients =
                viewerAdjointGradients.gradPosition != nullptr &&
                viewerAdjointGradients.numPoints == sceneGpu.pointCount &&
                viewerAdjointGradients.cameraSlotCount == 1u;
            if (hasUsableGradients) {
                return;
            }

            if (viewerAdjointGradients.gradPosition != nullptr ||
                viewerAdjointGradients.numPoints != 0u) {
                Pale::freeGradientsForScene(queue, viewerAdjointGradients);
            }

            Pale::SceneBuild::BuildProducts adjointBuildProducts = renderBuildProducts;
            adjointBuildProducts.cameraGPUs.clear();
            adjointBuildProducts.cameraGPUs.push_back(sensor.camera);
            viewerAdjointGradients =
                Pale::makeGradientsForScene(queue, adjointBuildProducts, nullptr);
        };

        auto freeViewerRegularizerGradients = [&]() {
            Pale::freeGradientsForScene(queue, viewerRegularizerGradients);
        };

        auto ensureViewerRegularizerGradients = [&]() {
            if (viewerRegularizerGradients.gradPosition != nullptr &&
                viewerRegularizerGradients.numPoints == sceneGpu.pointCount) {
                return;
            }

            freeViewerRegularizerGradients();
            Pale::SceneBuild::BuildProducts gradientBuildProducts = renderBuildProducts;
            gradientBuildProducts.cameraGPUs.clear();
            gradientBuildProducts.cameraGPUs.push_back(sensor.camera);
            viewerRegularizerGradients =
                Pale::makeGradientsForScene(queue, gradientBuildProducts, nullptr);
        };

        auto runViewerRegularizerGradientPass = [&](std::vector<Pale::SensorGPU>& renderSensors) {
            lastRegularizerGradientMapMs = 0.0;
            if (settings.sharedHeightEnabled) return;
            if (!isRegularizerGradientView(viewImageMode) ||
                !hasSensor || sceneGpu.pointCount == 0u) {
                return;
            }

            const auto start = std::chrono::steady_clock::now();
            ensureViewerRegularizerGradients();
            prepareRealtimeViewerSurfaceRegularizerAdjoints(queue, sensor, viewImageMode);

            Pale::PathTracerSettings regularizerSettings = settings;
            regularizerSettings.depthDistortionWeight =
                viewImageMode == ViewImageMode::DepthPositionGradient ? 1.0f : 0.0f;
            regularizerSettings.normalConsistencyWeight =
                viewImageMode == ViewImageMode::NormalPositionGradient ? 1.0f : 0.0f;
            regularizerSettings.intraSlabDepthRegularizerWeight =
                viewImageMode == ViewImageMode::IntraSlabPositionGradient ? 1.0f : 0.0f;
            tracer.getSettings() = regularizerSettings;
            // The backward kernel reads the point count from the depth slot,
            // even when only another regularizer is enabled. No buffers needed.
            Pale::PointGradients unusedDepthGradients{};
            unusedDepthGradients.numPoints = sceneGpu.pointCount;
            Pale::PointGradients unusedNormalGradients{};
            Pale::PointGradients unusedIntraSlabGradients{};
            tracer.renderSurfaceRegularizersBackward(
                renderSensors,
                regularizerSettings.depthDistortionWeight != 0.0f
                    ? viewerRegularizerGradients : unusedDepthGradients,
                regularizerSettings.normalConsistencyWeight != 0.0f
                    ? viewerRegularizerGradients : unusedNormalGradients,
                regularizerSettings.intraSlabDepthRegularizerWeight != 0.0f
                    ? viewerRegularizerGradients : unusedIntraSlabGradients,
                nullptr);
            tracer.getSettings() = settings;
            queue.wait();

            const auto stop = std::chrono::steady_clock::now();
            lastRegularizerGradientMapMs =
                std::chrono::duration<double, std::milli>(stop - start).count();
        };

        auto runViewerAdjointPass = [&](std::vector<Pale::SensorGPU>& renderSensors) {
            viewerAdjointImageValid = false;
            if (settings.sharedHeightEnabled) {
                viewerAdjointImage.clear();
                runAdjointNextRender = false;
                viewerAdjointStatus = "Shared height is forward-only; adjoint disabled";
                return;
            }
            const bool derivativeView = viewImageMode == ViewImageMode::AdjointDerivative;
            if (!runAdjointEveryRender && !runAdjointNextRender && !derivativeView) {
                return;
            }
            runAdjointNextRender = false;

            if (!hasSensor || sceneGpu.pointCount == 0u) {
                viewerAdjointStatus = "Adjoint skipped: no active sensor or surfels";
                lastViewerAdjointMs = 0.0;
                lastViewerAdjointLoss = 0.0f;
                return;
            }

            viewerAdjointSamplesPerPixel = std::clamp(viewerAdjointSamplesPerPixel, 1, 256);
            viewerAdjointBounces = std::clamp(viewerAdjointBounces, 1, 8);

            const auto start = std::chrono::steady_clock::now();
            {
                Pale::ScopedTimer timer("Viewer adjoint source setup", spdlog::level::debug);
                if (derivativeView) {
                    // d(sum of linear RGB)/dRGB = (1,1,1), with no pixel-count
                    // normalization, exposure, tone mapping, or regularizer terms.
                    queue.fill(sensor.framebuffer, Pale::float4{1.0f, 1.0f, 1.0f, 0.0f},
                        static_cast<std::size_t>(sensor.width) * sensor.height).wait();
                    lastViewerAdjointLoss = 0.0f;
                } else {
                    prepareRealtimeViewerRgbLossAdjointSource(
                        queue, sensor, lastViewerAdjointLoss);
                }
            }
            {
                Pale::ScopedTimer timer("Viewer adjoint gradient buffer setup", spdlog::level::debug);
                ensureViewerAdjointGradients();
                if (derivativeView) {
                    const auto pixelCount = static_cast<std::size_t>(sensor.width) * sensor.height;
                    if (viewerAdjointDebugImages.numPixels != pixelCount) {
                        Pale::freeDebugImagesForScene(queue, &viewerAdjointDebugImages, 1u);
                    }
                    auto debugBuildProducts = renderBuildProducts;
                    debugBuildProducts.cameraGPUs = {sensor.camera};
                    debugBuildProducts.cameraGPUs[0].useForAdjointPass = 1u;
                    Pale::ensureDebugImagesForScene(queue, debugBuildProducts, &viewerAdjointDebugImages);
                }
            }

            Pale::PathTracerSettings adjointSettings = settings;
            adjointSettings.maxAdjointBounces = static_cast<uint32_t>(viewerAdjointBounces);
            adjointSettings.adjointSamplesPerPixel =
                static_cast<uint32_t>(viewerAdjointSamplesPerPixel);
            adjointSettings.enableAdjointDirectLight = viewerAdjointDirectLight;
            adjointSettings.renderDebugGradientImages = derivativeView;
            adjointSettings.surfelIndexForDebugImages = viewerAdjointAllSurfels
                ? Pale::kDebugGradientAllSurfels
                : static_cast<uint32_t>(std::clamp(viewerAdjointSurfelIndex,
                    0, static_cast<int>(sceneGpu.pointCount) - 1));
            adjointSettings.random.seed = static_cast<uint64_t>(viewerAdjointSeed);
            adjointSettings.numAdjointPathShadowRays =
                std::max(adjointSettings.numAdjointPathShadowRays, 1u);

            tracer.getSettings() = adjointSettings;
            {
                Pale::ScopedTimer timer("Viewer adjoint pass total", spdlog::level::debug);
                tracer.renderBackward(renderSensors, viewerAdjointGradients,
                    derivativeView ? &viewerAdjointDebugImages : nullptr);
            }
            tracer.getSettings() = settings;
            queue.wait();

            const auto stop = std::chrono::steady_clock::now();
            lastViewerAdjointMs =
                std::chrono::duration<double, std::milli>(stop - start).count();
            if (derivativeView) {
                const float* columns[] = {
                    viewerAdjointDebugImages.framebufferPosX, viewerAdjointDebugImages.framebufferPosY,
                    viewerAdjointDebugImages.framebufferPosZ, viewerAdjointDebugImages.framebufferRotX,
                    viewerAdjointDebugImages.framebufferRotY, viewerAdjointDebugImages.framebufferRotZ,
                    viewerAdjointDebugImages.framebufferScaleU, viewerAdjointDebugImages.framebufferScaleV,
                    viewerAdjointDebugImages.framebufferOpacity, viewerAdjointDebugImages.framebufferAlbedo,
                    viewerAdjointDebugImages.framebufferBeta};
                viewerAdjointImage = Pale::downloadFloatBuffer(queue,
                    columns[viewerAdjointProperty], viewerAdjointDebugImages.numPixels);
                viewerAdjointImageValid = true;
                viewerAdjointStatus = "Unit RGB source; samples averaged, fixed color scale";
            } else {
                viewerAdjointStatus = "Last adjoint: " + std::to_string(lastViewerAdjointMs) +
                    " ms, loss " + std::to_string(lastViewerAdjointLoss);
            }
        };

        auto renderNow = [&]() {
            // Until every render/diagnostic pass succeeds, the old published
            // image and camera remain a complete, consistent presentation.
            previewPublication.begin(PublishedPreviewState{std::move(renderPixels),std::move(rawLinearPixels),
                std::move(pixels),std::move(screenshotPixels),std::move(rawRenderPixels),std::move(viewerAdjointImage),viewerAdjointImageValid,std::move(debugDisplayBuffers),
                displayedTransform,displayedOrbit,displayedBounds,displayedCameraSource,displayedCamera,
                displayedRenderWidth,displayedRenderHeight});
            debugDisplayBuffers=DebugDisplayBuffers{};
            viewerAdjointImageValid=false;
            try {
            if(sceneUploadPending) {
                Pale::SceneUpload::uploadOrReallocate(buildProducts,sceneGpu,queue);
                sceneGpu.profileCounters=gpuCounterProfilingEnabled?deviceProfilingCounters:nullptr;
                sceneUploadPending=false;
            }
            if (viewImageMode == ViewImageMode::TransportScore && transportDirty) computeTransportScores();
            if (viewImageMode == ViewImageMode::TargetLoss && targetLossDirty) computeTargetLoss();
            renderWidth = std::clamp(renderWidth, 16u, 4096u);
            renderHeight = std::clamp(renderHeight, 16u, 4096u);

            if (buildProducts.cameraGPUs.empty()) {
                cameraSource = CameraSource::Viewport;
                selectedSceneCameraIndex = 0;
            } else {
                selectedSceneCameraIndex = std::clamp(
                    selectedSceneCameraIndex,
                    0,
                    static_cast<int>(buildProducts.cameraGPUs.size() - 1u));
            }

            Pale::CameraGPU camera =
                cameraSource == CameraSource::SceneXml && !buildProducts.cameraGPUs.empty()
                    ? buildProducts.cameraGPUs[static_cast<std::size_t>(selectedSceneCameraIndex)]
                    : orbit.makeGpuCamera(renderWidth, renderHeight);

            if (cameraSource == CameraSource::SceneXml) {
                renderWidth = std::max(camera.width, 1u);
                renderHeight = std::max(camera.height, 1u);
                camera.width = renderWidth;
                camera.height = renderHeight;
            }

            renderBuildProducts.cameraGPUs.clear();
            renderBuildProducts.cameraGPUs.push_back(camera);

            if (!hasSensor || sensor.width != renderWidth || sensor.height != renderHeight) {
                if (hasSensor) {
                    destroySensor(queue, sensor);
                    hasSensor = false;
                }
                sensor = createSensor(queue, camera);
                hasSensor = true;
                tracerDirty = true;
            }

            sensor.camera = camera;
            sensor.exposureCorrection = exposure;
            sensor.gammaCorrection = gamma;
            sensor.useSrgbEncoding = useSrgbEncoding;
            clearSensor(queue, sensor);

            Pale::ScopedTimerDetail::setProfilingEnabled(timerProfilingEnabled);
            Pale::ScopedTimerDetail::clearProfilingRecords();

            Pale::RenderProfilingCounters* desiredProfilingCounters =
                gpuCounterProfilingEnabled ? deviceProfilingCounters : nullptr;
            if (sceneGpu.profileCounters != desiredProfilingCounters) {
                sceneGpu.profileCounters = desiredProfilingCounters;
                tracerDirty = true;
            }
            if (desiredProfilingCounters != nullptr) {
                queue.memset(
                    desiredProfilingCounters,
                    0,
                    sizeof(Pale::RenderProfilingCounters)).wait();
            }

            Pale::PathTracerSettings activeTracerSettings =
                makeViewerDebugSettings(settings, viewImageMode);
            if(interactivePreview) {
                activeTracerSettings.rendererDebugMaxSplatEventsPerRay=2;
                activeTracerSettings.rendererDebugShareLocalLayerDirectLighting=true;
            }
            if (runAdjointEveryRender || runAdjointNextRender ||
                viewImageMode == ViewImageMode::AdjointDerivative) {
                viewerAdjointSamplesPerPixel = std::clamp(viewerAdjointSamplesPerPixel, 1, 256);
                viewerAdjointBounces = std::clamp(viewerAdjointBounces, 1, 8);
                activeTracerSettings.maxAdjointBounces =
                    static_cast<uint32_t>(viewerAdjointBounces);
                activeTracerSettings.adjointSamplesPerPixel =
                    static_cast<uint32_t>(viewerAdjointSamplesPerPixel);
                activeTracerSettings.enableAdjointDirectLight = viewerAdjointDirectLight;
                activeTracerSettings.numAdjointPathShadowRays =
                    std::max(activeTracerSettings.numAdjointPathShadowRays, 1u);
            }

            if (tracerDirty) {
                tracer.getSettings() = activeTracerSettings;
                tracer.setScene(sceneGpu, renderBuildProducts);
                tracerDirty = false;
            } else {
                tracer.getSettings() = activeTracerSettings;
            }

            std::vector<Pale::SensorGPU> renderSensors{sensor};
            const auto start = std::chrono::steady_clock::now();
            tracer.renderForward(renderSensors);
            // Loss/adjoint passes overwrite this buffer. Keep the completed primal
            // and its display settings together, even during diagnostic views.
            rawRenderPixels = Pale::downloadSensorRGBARAW(queue, sensor);
            displayedTransform = {exposure, gamma, useSrgbEncoding};
            if (viewImageMode == ViewImageMode::RawLinear) {
                rawLinearPixels = viewer::rawLinearPreview(rawRenderPixels);
            } else {
                rawLinearPixels.clear();
            }
            runViewerRegularizerGradientPass(renderSensors);
            runViewerAdjointPass(renderSensors);
            const auto stop = std::chrono::steady_clock::now();
            lastRenderMs =
                std::chrono::duration<double, std::milli>(stop - start).count();
            renderFrameTimeHistory.push_back(static_cast<float>(lastRenderMs));
            if (renderFrameTimeHistory.size() > kFrameTimeHistoryCapacity) {
                renderFrameTimeHistory.erase(renderFrameTimeHistory.begin());
            }
            lastTimerRecords = Pale::ScopedTimerDetail::snapshotProfilingRecords();
            if (desiredProfilingCounters != nullptr) {
                queue.memcpy(
                    &lastProfilingCounters,
                    desiredProfilingCounters,
                    sizeof(Pale::RenderProfilingCounters)).wait();
            } else {
                lastProfilingCounters = {};
            }

            renderPixels = Pale::downloadSensorRGBA(queue, sensor);
            debugDisplayBuffers.invalidate();
            displayedOrbit = orbit;
            displayedBounds = bounds;
            displayedCameraSource = cameraSource;
            displayedCamera = camera;
            displayedRenderWidth = renderWidth;
            displayedRenderHeight = renderHeight;
            // Prepare diagnostic data on the renderer thread too: some views
            // launch their own SYCL kernels. OpenGL upload stays on the UI thread.
            if (viewImageMode != ViewImageMode::Rendered) {
                ensureDebugDisplayBuffer(viewImageMode, true);
            }
            cameraDirty = false;
            renderRequested = false;
            } catch(...) {
                previewPublication.finish(false,restorePublishedPreview);
                throw;
            }
        };

        const auto finishWalkNavigation = [&](bool cancel) {
            if (!walkNavigationActive) {
                return;
            }
            if (cancel) {
                orbit = orbitBeforeWalk;
                cameraSource = cameraSourceBeforeWalk;
            }
            walkNavigationActive = false;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            if (glfwRawMouseMotionSupported()) {
                glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
            }
            viewportPickArmed = false;
            pressedLightMarker.reset();
            viewportGizmoMouseCapture = false;
            cameraDirty = true;
            tracerDirty = true;
            renderRequested = true;
        };

        const auto resetOrbitView = [&]() {
            finishWalkNavigation(false);
            orbit = initialOrbit;
            cameraSource = CameraSource::Viewport;
            cameraDirty = true;
            tracerDirty = true;
            renderRequested = true;
        };

        const auto makeSceneGroups=[&] {
            const auto areaLights=collectAreaLights(scene);
            if (sceneTreeDirty) {
                sceneTreeEmitterIndices.clear();
                if (auto handle = firstPointCloudHandle(scene)) {
                    if (auto cloud = assetAccessor.getPointCloud(*handle); cloud && !cloud->points.empty())
                        sceneTreeEmitterIndices = collectSurfelPowerIndices(cloud->points.front(), true);
                }
                sceneTreeDirty = false;
            }
            std::vector<viewer::SceneTreeGroup> sceneGroups{{"Geometry",{}},{"Materials",{}},{"Emitters",{}},{"Cameras",{}}};
            sceneGroups[0].items.push_back({viewer::SceneItemKind::Geometry,0,"Surfels · "+std::to_string(buildProducts.points.size()),"Reconstruction primitives",viewer::ui::Icon::Cube,sceneSelectionKind==viewer::SceneItemKind::Geometry});
            sceneGroups[1].items.push_back({viewer::SceneItemKind::Material,-1,"Surfel albedo","Select a surfel in the viewport to inspect its material",viewer::ui::Icon::Material,false});
            int meshOrdinal=0;
            for (auto [id, mesh, tag] : scene->getAllEntitiesWith<Pale::MeshComponent,Pale::TagComponent>().each()) {
                Pale::Entity entity(id,scene.get());
                if (!entity.hasComponent<Pale::AreaLightComponent>())
                    sceneGroups[0].items.push_back({viewer::SceneItemKind::Mesh,meshOrdinal,tag.tag,"Source triangle mesh",viewer::ui::Icon::Cube,sceneSelectionKind==viewer::SceneItemKind::Mesh && selectedTreeEntity==tag.tag});
                if (entity.hasComponent<Pale::MaterialComponent>())
                    sceneGroups[1].items.push_back({viewer::SceneItemKind::Material,meshOrdinal,tag.tag,"Mesh material",viewer::ui::Icon::Material,sceneSelectionKind==viewer::SceneItemKind::Material && selectedTreeEntity==tag.tag});
                ++meshOrdinal;
            }
            for (std::size_t i=0;i<areaLights.size();++i)
                sceneGroups[2].items.push_back({viewer::SceneItemKind::Emitter,static_cast<int>(i),entityName(areaLights[i]),"Mesh area emitter",viewer::ui::Icon::Light,sceneSelectionKind==viewer::SceneItemKind::Emitter && selectedLightIndex==static_cast<int>(i)});
            const auto addSurfelEmitterRow=[&](std::size_t listIndex) {
                const auto index=sceneTreeEmitterIndices[listIndex];
                sceneGroups[2].items.push_back({viewer::SceneItemKind::SurfelEmitter,static_cast<int>(listIndex),"Emitter surfel "+std::to_string(index),"Select to inspect emission, position and material",viewer::ui::Icon::Light,sceneSelectionKind==viewer::SceneItemKind::SurfelEmitter && selectedSurfelEditorIndex==static_cast<int>(index)});
            };
            for (std::size_t i=0;i<std::min<std::size_t>(sceneTreeEmitterIndices.size(),256);++i)addSurfelEmitterRow(i);
            if (selectedSurfelLightIndex>=256 && static_cast<std::size_t>(selectedSurfelLightIndex)<sceneTreeEmitterIndices.size())addSurfelEmitterRow(selectedSurfelLightIndex);
            sceneGroups[2].total=areaLights.size()+sceneTreeEmitterIndices.size();
            for (std::size_t i=0;i<buildProducts.cameraGPUs.size();++i)
                sceneGroups[3].items.push_back({viewer::SceneItemKind::Camera,static_cast<int>(i),buildProducts.cameraGPUs[i].name,"Calibrated camera · fixed pose and intrinsics",viewer::ui::Icon::Camera,cameraSource==CameraSource::SceneXml && selectedSceneCameraIndex==static_cast<int>(i)});
            return sceneGroups;
        };

        const auto recordPreviewFailure=[&](const std::string& error) {
            previewRecovery.fail(error);
            tracerDirty=true;
            // Preserve the request, rather than permanently starving the
            // preview after a failed upload, render, or diagnostic pass.
            renderRequested=true;
            Pale::Log::PA_ERROR("Preview failed during {} (failure {}): {}",previewActivity,previewRecovery.failures(),error);
            try {
                std::filesystem::create_directories(previewFailureLog.parent_path());
                YAML::Node event;
                event["event"]="preview_failure";event["error"]=error;event["activity"]=previewActivity;
                event["unix_ms"]=viewer::jsonNumber(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
                event["failure"]=viewer::jsonNumber(previewRecovery.failures());
                event["scene"]=previewJobScene.string();event["pointcloud"]=previewJobPointCloud.string();
                event["view"]=previewJobView;
                event["width"]=viewer::jsonNumber(renderWidth);event["height"]=viewer::jsonNumber(renderHeight);
                event["scene_upload_pending"]=sceneUploadPending;
                event["loaded_generation"]=viewer::jsonNumber(training.loadedGeneration);
                std::ofstream output(previewFailureLog,std::ios::app);
                viewer::writeJsonValue(output,event);output<<'\n';
            } catch(const std::exception& loggingError) {
                Pale::Log::PA_WARN("Cannot write preview diagnostics: {}",loggingError.what());
            }
        };
        const auto recordLongPreview=[&] {
            Pale::Log::PA_WARN("Preview has not completed after 15 seconds during {}; UI/training remain active",previewActivity);
            try {
                std::filesystem::create_directories(previewFailureLog.parent_path());
                YAML::Node event;
                event["event"]="preview_long_work";event["activity"]=previewActivity;
                event["unix_ms"]=viewer::jsonNumber(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
                event["elapsed_ms"]=viewer::jsonNumber(previewWatchdog.elapsedMilliseconds());
                event["scene"]=previewJobScene.string();event["pointcloud"]=previewJobPointCloud.string();
                event["view"]=previewJobView;
                std::ofstream output(previewFailureLog,std::ios::app);viewer::writeJsonValue(output,event);output<<'\n';
            } catch(const std::exception& error) {Pale::Log::PA_WARN("Cannot write long-preview diagnostics: {}",error.what());}
        };
        const auto drawPreviewFailureNotice=[&](const viewer::ViewportCanvas& canvas,bool busy) {
            const bool longWork=previewWatchdog.longRunning();
            if(previewRecovery.error().empty()&&!longWork)return false;
            const auto cursor=ImGui::GetCursorScreenPos();
            ImGui::SetCursorScreenPos(ImVec2(canvas.min.x+50,canvas.min.y+8));
            ImGui::PushStyleColor(ImGuiCol_ChildBg,ImVec4(.13f,.14f,.16f,.98f));
            ImGui::BeginChild("Preview recovery",ImVec2(std::min(620.f,std::max(120.f,canvas.size.x-65)),std::min(180.f,std::max(75.f,canvas.size.y-20))),ImGuiChildFlags_Borders);
            ImGui::TextColored(previewRecovery.failed()?ImVec4(1,.65f,.35f,1):viewer::ui::green,"%s",
                longWork?"Preview is still working":previewRecovery.failed()?(previewRecovery.paused()?"Preview paused after repeated failures":"Preview update failed; retrying"):"Preview recovered");
            if(longWork)ImGui::TextWrapped("Training and UI remain active. Kernel compilation or rendering has not completed yet.");
            else ImGui::TextWrapped("%s",previewRecovery.error().c_str());
            if(previewRecovery.failed()||longWork) {
                if(busy)ImGui::BeginDisabled();
                if(ImGui::Button("Retry preview")) {
                    previewRecovery.manualRetry();
                    if(busy)queuedRefresh=true;else renderRequested=true;
                }
                if(busy)ImGui::EndDisabled();
                if(busy&&ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))ImGui::SetTooltip("The current preview must finish before retrying.");
            } else if(ImGui::Button("Dismiss"))previewRecovery.dismiss();
            ImGui::SameLine();if(ImGui::Button(longWork?"Copy diagnostics path":"Copy error"))
                ImGui::SetClipboardText(longWork?previewFailureLog.string().c_str():previewRecovery.error().c_str());
            ImGui::TextWrapped("Details: %s",previewFailureLog.string().c_str());
            const bool hovered=ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows);
            ImGui::EndChild();ImGui::PopStyleColor();ImGui::SetCursorScreenPos(cursor);
            return hovered;
        };

        const auto drawApplicationMenu=[&](bool busy) {
            bool start=false;
            if (ImGui::BeginMainMenuBar()) {
                ImGui::TextColored(kUiAccent,"pale"); ImGui::Spacing();
                if (ImGui::BeginMenu("File")) {
                    if (ImGui::MenuItem("Save editable snapshot", nullptr, false, !busy && !modelReadOnly())) saveWorkingSnapshot();
                    if (ImGui::MenuItem("Export mesh..."))showMeshExport=true;
                    ImGui::EndMenu();
                }
                if (ImGui::BeginMenu("Reconstruction")) {
                    if (ImGui::MenuItem("Training setup")) {
                        if(busy)inspectorPage=viewer::ui::Inspector::Reconstruction;else selectTrainingTab=true;
                    }
                    ImGui::EndMenu();
                }
                if (ImGui::BeginMenu("Window")) {
                    ImGui::MenuItem("Scene Collection", nullptr, &showSceneTree);
                    ImGui::MenuItem("Run monitor", nullptr, &showTrainingProgress);
                    if (ImGui::MenuItem("Restore workspace layout")) dockLayoutInitialized=false;
                    ImGui::Separator();
                    ImGui::MenuItem("Profiling", nullptr, &showProfilingWindow);
                    ImGui::EndMenu();
                }
                ImGui::SameLine(); ImGui::TextDisabled("JOINT PBdR");
                ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX()+20,ImGui::GetWindowWidth()-380));
                ImGui::Checkbox("Auto preview",&autoRender); ImGui::SameLine();
                if(viewer::ui::iconButton("Refresh preview",viewer::ui::Icon::Refresh,false,ImVec2(25,ImGui::GetFrameHeight()))) {
                    previewRecovery.manualRetry();
                    if(busy)queuedRefresh=true;else renderRequested=true;
                }
                ImGui::SameLine();
                if(training.drawStartButton(ImVec2(182,0),busy&&training.useWorkingSnapshot))start=true;
                ImGui::EndMainMenuBar();
            }
            return start;
        };
        const auto drawViewportStatus=[&](bool busy) {
            // Work begins and ends many times during a single gesture. Keep the
            // visible guidance unchanged and expose transient details on hover.
            ImGui::TextDisabled("%s  ·  %u surfels  ·  %s",
                modelReadOnly()?"Read-only training view":"Editable snapshot",
                busy?busyPointCount:sceneGpu.pointCount,
                walkNavigationActive?"WASD walk · Enter to finish":"Drag orbit · RMB pan · Wheel zoom");
            if(ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("Last preview: %.1f ms",busy?busyRenderMs:lastRenderMs);
                ImGui::TextUnformatted("Fast preview uses 2 splat events and shared lighting; full quality returns when idle.");
                ImGui::TextUnformatted("Preview updates run in the background; navigation and edits remain available.");
                ImGui::EndTooltip();
            }
        };

        // Keep the same controls and geometry while the renderer owns the scene.
        // Scene-source changes wait for handoff; display choices can be queued.
        const auto drawViewportToolbar=[&](bool busy) {
            const auto mode=busy?queuedView.value_or(viewImageMode):viewImageMode;
            ImGui::SetNextItemWidth(175);
            if (ImGui::BeginCombo("##ViewportMode",viewImageModeLabel(mode))) {
                for(auto candidate:kViewImageModeCycleOrder)
                    if(ImGui::Selectable(viewImageModeLabel(candidate),mode==candidate)) {
                        if(busy)queuedView=candidate;else setViewImageMode(candidate);
                    }
                ImGui::EndCombo();
            }
            ImGui::SameLine();
            if (training.process.launched()) {
                ImGui::SetNextItemWidth(168);
                const char* source = training.view==viewer::TrainingView::Live ? "Live training" : training.view==viewer::TrainingView::Saved ? "Follow saved" : training.view==viewer::TrainingView::History ? "Checkpoint history" : "Editable snapshot";
                if(ImGui::BeginCombo("##SceneSource",source)) {
                    if(ImGui::Selectable("Live training",training.view==viewer::TrainingView::Live)) {
                        if(busy)queuedSceneSource=viewer::TrainingView::Live;else enterTrainingView(viewer::TrainingView::Live);
                    }
                    if(ImGui::Selectable("Follow saved",training.view==viewer::TrainingView::Saved)) {
                        if(busy)queuedSceneSource=viewer::TrainingView::Saved;else enterTrainingView(viewer::TrainingView::Saved);
                    }
                    if(ImGui::Selectable("Browse checkpoints...")) {training.requestedMonitorTab=2;showTrainingProgress=true;}
                    ImGui::Separator();
                    if(ImGui::Selectable("Edit latest saved snapshot")) {
                        if(busy)editLatestRequested=true;else editLatestSavedSnapshot();
                    }
                    if(hasWorkingSnapshot && modelReadOnly() && ImGui::Selectable("Return to edits")) {
                        if(busy)returnToEditsRequested=true;else restoreWorkingSnapshot();
                    }
                    ImGui::EndCombo();
                }
                ImGui::SameLine(); ImGui::TextDisabled("%s",modelReadOnly()?"Read-only":"Editable");
                ImGui::SameLine();
            }
            const float overlayWidth = 164;
            ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(),ImGui::GetWindowWidth()-overlayWidth-12));
            if(ImGui::Button("Gizmos & overlays",ImVec2(overlayWidth,0)))ImGui::OpenPopup("ViewportAids");
            if(ImGui::BeginPopup("ViewportAids")) {
                ImGui::SeparatorText("Viewport overlays");
                if(ImGui::Checkbox("Ground grid",&showViewportGrid)) {
                    if(busy)queuedRefresh=true;else updateDisplayTexture();
                }
                ImGui::SeparatorText("Gizmos");
                ImGui::Checkbox("Show gizmos",&gizmosEnabled);
                if(!gizmosEnabled)ImGui::BeginDisabled();
                ImGui::Checkbox("Emitter markers",&showLightMarkers);
                ImGui::Checkbox("Emitter transform handles",&showLightGizmo);
                ImGui::Checkbox("Surfel transform handles",&showSurfelGizmo);
                if(!gizmosEnabled)ImGui::EndDisabled();
                ImGui::SeparatorText("Transform coordinates");
                bool local=lightGizmoMode==ImGuizmo::LOCAL;
                if(ImGui::RadioButton("World",!local))lightGizmoMode=surfelGizmoMode=ImGuizmo::WORLD;
                ImGui::SameLine(); if(ImGui::RadioButton("Local",local))lightGizmoMode=surfelGizmoMode=ImGuizmo::LOCAL;
                ImGui::EndPopup();
            }
            ImGui::Separator();
        };

        const auto refreshLightMarkerMetadata=[&] {
            auto lights=collectAreaLights(scene);
            if(!lightMarkersDirty&&lightLocalCenters.size()==lights.size())return;
            lightLocalCenters.clear();
            for(auto light:lights) {
                glm::vec3 low(std::numeric_limits<float>::infinity()),high(-std::numeric_limits<float>::infinity());
                if(light.hasComponent<Pale::MeshComponent>())if(const auto mesh=assetAccessor.getMesh(light.getComponent<Pale::MeshComponent>().meshID))
                    for(const auto& part:mesh->submeshes)for(const auto& position:part.positions) {low=glm::min(low,position);high=glm::max(high,position);}
                lightLocalCenters.push_back(std::isfinite(low.x)?(low+high)*.5f:glm::vec3(0));
            }
            surfelLightMarkers.clear();
            auto entity=firstPointCloudEntity(scene);const auto handle=firstPointCloudHandle(scene);
            const auto asset=handle?assetAccessor.getPointCloud(*handle):nullptr;
            const auto transform=entity&&entity.hasComponent<Pale::TransformComponent>()?entity.getComponent<Pale::TransformComponent>().getTransform():glm::mat4(1);
            int index=0;
            if(asset)for(const auto& geometry:asset->points)for(std::size_t i=0;i<geometry.positions.size();++i,++index)
                if(i<geometry.powers.size()&&geometry.powers[i]>0)surfelLightMarkers.push_back({glm::vec3(transform*glm::vec4(geometry.positions[i],1)),index});
            lightMarkersDirty=false;
        };
        const auto publishDisplayedOverlays=[&] {
            displayedExportMetadata=captureExportMetadata();
            displayedExportRunConfig.clear();
            if(training.status["scene"].IsScalar()&&currentScenePath==std::filesystem::path(training.status["scene"].as<std::string>()))
                displayedExportRunConfig=training.process.runDirectory/"run_config.json";
            displayedLivePacket=livePacketLeases.retain(currentPointCloudPath);
            refreshLightMarkerMetadata();displayedLightMarkers.clear();
            auto lights=collectAreaLights(scene);
            for(std::size_t i=0;i<lights.size();++i)
                displayedLightMarkers.push_back({glm::vec3(lights[i].getComponent<Pale::TransformComponent>().getTransform()*glm::vec4(lightLocalCenters[i],1)),{static_cast<int>(i),-1},entityName(lights[i])});
            for(const auto& marker:surfelLightMarkers)displayedLightMarkers.push_back({marker.worldCenter,{-1,marker.editorIndex},"Surfel light "+std::to_string(marker.editorIndex)});
            displayedOverlayPointCloud=currentPointCloudPath;displayedOverlayScene=currentScenePath;
        };
        const auto drawViewportToolRail=[&](ImVec2 canvas,bool surfelSelected) {
            const auto after=ImGui::GetCursorScreenPos();ImGui::SetCursorScreenPos(ImVec2(canvas.x+8,canvas.y+42));ImGui::BeginGroup();
            for(const auto& [tool,icon,label]:std::array<std::tuple<int,viewer::ui::Icon,const char*>,4>{{
                {0,viewer::ui::Icon::Select,"Select"},{1,viewer::ui::Icon::Move,"Move selection"},
                {2,viewer::ui::Icon::Rotate,"Rotate selection"},{3,viewer::ui::Icon::Scale,"Scale mesh emitter"}}}) {
                const bool disabled=tool==3&&surfelSelected;if(disabled)ImGui::BeginDisabled();
                if(viewer::ui::iconButton(label,icon,viewportTool==tool,ImVec2(30,30))) {
                    viewportTool=tool;
                    if(tool==1)lightGizmoOperation=surfelGizmoOperation=ImGuizmo::TRANSLATE;
                    if(tool==2)lightGizmoOperation=surfelGizmoOperation=ImGuizmo::ROTATE;
                    if(tool==3)lightGizmoOperation=ImGuizmo::SCALE;
                }
                if(disabled)ImGui::EndDisabled();
            }
            ImGui::EndGroup();ImGui::SetCursorScreenPos(after);
        };
        const auto drawDisplayedLightMarkers=[&](const Pale::CameraGPU& camera,ImVec2 origin,ImVec2 size,bool hoverable,
                viewer::SceneItemKind selectedKind,int selectedSurfel,int selectedArea) {
            std::optional<LightMarkerSelection> hovered;std::string name;float nearest=14.f*14.f;
            if(!gizmosEnabled||!showLightMarkers||walkNavigationActive)return hovered;
            auto* draw=ImGui::GetWindowDrawList();draw->PushClipRect(origin,ImVec2(origin.x+size.x,origin.y+size.y),true);
            const auto mouse=ImGui::GetMousePos();
            for(const auto& marker:displayedLightMarkers) {
                const auto screen=viewer::projectViewportPoint(camera,marker.center,{origin.x,origin.y},{size.x,size.y});if(!screen)continue;
                const ImVec2 point(screen->x+18,screen->y-18);
                if(point.x<origin.x||point.x>=origin.x+size.x||point.y<origin.y||point.y>=origin.y+size.y)continue;
                const bool selected=marker.selection.areaIndex>=0
                    ? selectedKind==viewer::SceneItemKind::Emitter&&selectedSurfel<0&&selectedArea==marker.selection.areaIndex
                    : selectedSurfel==marker.selection.surfelIndex;
                const float distance=(point.x-mouse.x)*(point.x-mouse.x)+(point.y-mouse.y)*(point.y-mouse.y);
                if(hoverable&&distance<nearest) {nearest=distance;hovered=marker.selection;name=marker.name;}
                const auto color=selected?IM_COL32(255,255,255,255):IM_COL32(255,201,75,255);
                draw->AddLine(ImVec2(screen->x,screen->y),point,IM_COL32(255,201,75,140));
                draw->AddCircleFilled(point,9,IM_COL32(25,23,18,235),16);draw->AddCircle(point,selected?11:7,color,16,1.5f);
                draw->AddCircleFilled(point,3,IM_COL32(255,201,75,255),12);
                for(int axis=0;axis<4;++axis) {const float a=axis*1.5707963f,x=std::cos(a),y=std::sin(a);draw->AddLine(ImVec2(point.x+x*5,point.y+y*5),ImVec2(point.x+x*9,point.y+y*9),color,1.5f);}
            }
            draw->PopClipRect();if(hovered) {ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);ImGui::SetTooltip("%s\nClick to select and edit",name.c_str());}
            return hovered;
        };
        const auto inspectorContextKey=[&](viewer::ui::Inspector page) {
            std::string key=std::to_string(static_cast<int>(page))+"/"+std::to_string(static_cast<int>(viewImageMode));
            if(const auto* window=ImGui::FindWindowByName("Properties"))key+="/"+std::to_string(window->Size.x)+"/"+std::to_string(window->Size.y)+"/"+std::to_string(window->Pos.x)+"/"+std::to_string(window->Pos.y);
            if(page==viewer::ui::Inspector::Selection)key+="/"+currentPointCloudPath.string()+"/"+std::to_string(static_cast<int>(sceneSelectionKind))+"/"+std::to_string(selectedSurfelEditorIndex)+"/"+std::to_string(selectedLightIndex);
            if(page==viewer::ui::Inspector::Camera||page==viewer::ui::Inspector::Files||page==viewer::ui::Inspector::Advanced)key+="/"+currentScenePath.string();
            return key;
        };
        const auto capturePreviewUi=[&] {
            busyProfilingScene=sceneGpu;busyProfilingCounters=lastProfilingCounters;
            busyTimerRecords=lastTimerRecords;busyFrameTimes=renderFrameTimeHistory;
            busyTimerProfilingEnabled=timerProfilingEnabled;busyGpuCounterProfilingEnabled=gpuCounterProfilingEnabled;
            busyOverlayMatchesScene=displayedOverlayPointCloud==currentPointCloudPath&&displayedOverlayScene==currentScenePath;
            busySelectedLightIndex=selectedLightIndex;
            busyHasWorkspace=hasWorkingSnapshot||(!modelReadOnly()&&firstPointCloudHandle(scene).has_value());
            cachedSceneGroups=makeSceneGroups();cachedSceneName=currentScenePath.stem().string();
            queuedOrbit=orbit;busyBounds=bounds;busyCamera=displayedCamera;busyCameraSource=cameraSource;
            busyPointCount=static_cast<uint32_t>(buildProducts.points.size());busyRenderMs=lastRenderMs;
            busyCameraNames.clear();for(const auto& camera:buildProducts.cameraGPUs)busyCameraNames.emplace_back(camera.name);
            busyHasSelection=false;busySelectionName=selectedTreeEntity;busySelectionKind=sceneSelectionKind;
            busySelectionIndex=selectedSurfelEditorIndex;busyLightCenter=glm::vec3(0);busyPointCloudTransform=glm::mat4(1);
            if(auto entity=firstPointCloudEntity(scene);entity&&entity.hasComponent<Pale::TransformComponent>())busyPointCloudTransform=entity.getComponent<Pale::TransformComponent>().getTransform();
            if(sceneSelectionKind==viewer::SceneItemKind::Emitter) {
                const auto lights=collectAreaLights(scene);
                if(selectedLightIndex>=0&&static_cast<std::size_t>(selectedLightIndex)<lights.size()) {
                    auto light=lights[selectedLightIndex];busySelectionName=entityName(light);
                    if(static_cast<std::size_t>(selectedLightIndex)<lightLocalCenters.size())busyLightCenter=lightLocalCenters[selectedLightIndex];
                    const auto& transform=light.getComponent<Pale::TransformComponent>();
                    busySelectionPosition=transform.translation;busySelectionRotation=transform.rotationEuler;busySelectionScale=transform.scale;
                    const auto& emitter=light.getComponent<Pale::AreaLightComponent>();busySelectionColor=emitter.radiance;busySelectionPower=emitter.flux;
                    if(light.hasComponent<Pale::MaterialComponent>())if(const auto material=assetManager.get<Pale::Material>(light.getComponent<Pale::MaterialComponent>().materialID)) {busySelectionColor=material->baseColor;busySelectionPower=material->power;}
                    busyHasSelection=true;
                }
            } else if((sceneSelectionKind==viewer::SceneItemKind::Geometry||sceneSelectionKind==viewer::SceneItemKind::SurfelEmitter)&&selectedSurfelEditorIndex>=0) {
                if(const auto handle=firstPointCloudHandle(scene)) {
                    const auto cloud=assetAccessor.getPointCloud(*handle);
                    if(cloud)if(const auto point=resolveEditableSurfel(*cloud,selectedSurfelEditorIndex);point&&point->pointGeometry) {
                        busySelectionName="Surfel "+std::to_string(selectedSurfelEditorIndex);
                        busySelectionPosition=point->pointGeometry->positions[point->localIndex];
                        const auto& geometry=*point->pointGeometry;const auto i=point->localIndex;
                        busySelectionColor=geometry.albedos[i];busySelectionRotation=glm::degrees(glm::eulerAngles(geometry.quat[i]));
                        busySelectionScale=glm::vec3(geometry.scales[i],1);busySelectionPower=geometry.powers[i];busySelectionOpacity=geometry.opacities[i];busySelectionBeta=geometry.betas[i];busyHasSelection=true;
                    }
                }
            }
        };
        startPreviewJob=[&](std::function<void()> work,std::function<void()> completion,const char* activity) {
            capturePreviewUi();previewActivity=activity;previewCompletion=std::move(completion);
            previewJobScene=currentScenePath;previewJobPointCloud=currentPointCloudPath;previewJobView=viewImageModeLabel(viewImageMode);
            interactivePreview=std::chrono::steady_clock::now()-lastNavigationInput<std::chrono::milliseconds(180);
            previewWatchdog.begin();
            try {previewWork.start(std::move(work));}
            catch(const std::exception& error) {previewWatchdog.finish();previewCompletion={};recordPreviewFailure(error.what());}
        };
        const auto loadRequestedInitializer=[&] {
            const auto directory=*training.initializedPointCloudRequest;
            training.initializedPointCloudRequest.reset();
            if(!modelReadOnly()&&!saveWorkingSnapshot())return;
            training.useWorkingSnapshot=false;
            followLaunchAfterPreview=false;
            invalidateSceneMaterials();
            auto loaded=std::make_shared<bool>(false);
            startPreviewJob([&,directory,loaded] {
                *loaded=replaceSceneAndPointCloud(directory/"scene.xml",directory/"points.ply",true);
                if(*loaded) {
                    orbit=makeInitialOrbitCamera(buildProducts,bounds);initialOrbit=orbit;
                    cameraSource=CameraSource::Viewport;cameraDirty=renderRequested=true;
                }
            },[&,directory,loaded] {
                if(!*loaded) {training.message=pointCloudStatus;return;}
                training.view=viewer::TrainingView::Snapshot;
                workingDirectory.clear();workingSourceScene=std::filesystem::absolute(currentScenePath);
                hasWorkingSnapshot=false;workingIteration=training.viewedIteration=0;
                training.loadedGeneration=0;loadedTrainingSceneRun.clear();framedTrainingRun.clear();
                pendingViewedIteration.reset();loadedLivePacket.reset();
                sceneSelectionKind=viewer::SceneItemKind::Geometry;
                training.message="Initialized point cloud loaded · "+directory.filename().string();
            },"Updating training scene");
        };
        const auto drawBusyFrame=[&] {
            // The worker exclusively owns scene/renderer state. This frame reads
            // CPU snapshots and UI-owned training state, never in-flight buffers.
            ImGui_ImplOpenGL3_NewFrame();ImGui_ImplGlfw_NewFrame();
            io.ConfigInputTrickleEventQueue=true;ImGui::NewFrame();
            bool start=launchTrainingRequested;
            launchTrainingRequested=false;
            if(!io.WantTextInput&&!io.KeyCtrl&&!io.KeyAlt&&!io.KeySuper) {
                if(ImGui::IsKeyPressed(ImGuiKey_R))training.checkpointNavigation=0;
                if(ImGui::IsKeyPressed(ImGuiKey_N))training.checkpointNavigation=-2;
                if(ImGui::IsKeyPressed(ImGuiKey_M))training.checkpointNavigation=2;
            }
            start=drawApplicationMenu(true)||start;
            auto* viewport=ImGui::GetMainViewport();const auto dock=ImHashStr("PaleRealtimeViewerDockSpace");
            if(!dockLayoutInitialized) {viewer::ui::defaultDockLayout(dock,viewport);dockLayoutInitialized=true;}
            ImGui::DockSpaceOverViewport(dock,viewport,ImGuiDockNodeFlags_PassthruCentralNode);ImGuizmo::BeginFrame();ImGuizmo::Enable(true);
            if(auto pick=sceneTree.draw(cachedSceneName,cachedSceneGroups,showSceneTree)) {
                // A scene replacement may renumber rows; preserve selection only
                // for a render of the same immutable scene.
                if(previewActivity!="Updating training scene")queuedScenePick=pick;
                inspectorPage=viewer::ui::Inspector::Selection;
            }
            ImGui::Begin("Properties",nullptr,ImGuiWindowFlags_NoScrollbar|ImGuiWindowFlags_NoScrollWithMouse);
            ImGui::BeginChild("InspectorRail",ImVec2(32,0),ImGuiChildFlags_None,ImGuiWindowFlags_NoScrollbar);
            viewer::ui::inspectorRail(inspectorPage,showAdvancedTools);ImGui::EndChild();ImGui::SameLine(0,8);
            const auto cachedPage=inspectorBodies[static_cast<int>(inspectorPage)].valid?inspectorPage:lastDrawnInspector;
            auto& cachedInspector=inspectorBodies[static_cast<int>(cachedPage)];
            if(inspectorPage!=viewer::ui::Inspector::Reconstruction&&cachedInspector.valid)
                ImGui::SetNextWindowScroll(ImVec2(-1,cachedInspector.scroll));
            ImGui::BeginChild("InspectorContents",ImVec2(0,0));
            if(inspectorPage==viewer::ui::Inspector::Reconstruction)start=training.drawSetup(busyHasWorkspace,true)||start;
            else if(cachedInspector.valid) {
                cachedInspector.geometry.draw(*ImGui::GetWindowDrawList(),ImGui::GetWindowPos(),cachedInspector.size);
                ImGui::Dummy(ImVec2(1,std::max(1.f,cachedInspector.contentHeight)));
                if(cachedPage==inspectorPage&&ImGui::IsWindowHovered()&&
                   (ImGui::IsMouseClicked(ImGuiMouseButton_Left)||io.MouseWheel!=0))
                    inspectorInput.queue(static_cast<int>(inspectorPage),cachedInspector.key,io.MousePos,io.MouseWheel,ImGui::IsMouseClicked(ImGuiMouseButton_Left));
            }
            ImGui::EndChild();ImGui::End();
            if(start&&!training.process.active()) {
                if(training.useWorkingSnapshot) {training.message="Wait for the current preview before starting from an edited snapshot";}
                else try {
                    const bool preserve=!modelReadOnly();training.launch();showTrainingProgress=true;
                    if(preserve) {training.view=viewer::TrainingView::Snapshot;followLaunchAfterPreview=true;}
                } catch(const std::exception& error) {training.message=error.what();}
            }
            ImGui::Begin("Viewport",nullptr,ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoScrollbar|ImGuiWindowFlags_NoScrollWithMouse);
            drawViewportToolbar(true);
            const auto viewportCanvas=viewer::viewportCanvas();
            const auto available=viewportCanvas.size;
            const bool previewNoticeHovered=drawPreviewFailureNotice(viewportCanvas,true);
            if(texture.id) {
                const auto canvas=viewportCanvas.min;
                const ImVec2 canvasMax(canvas.x+available.x,canvas.y+available.y);
                const auto placement=viewer::viewportImagePlacement(viewportCanvas,busyCameraSource==CameraSource::SceneXml,cameraImageView,texture.width,texture.height);
                const auto size=placement.size,origin=placement.min;
                ImGui::SetCursorScreenPos(origin);ImGui::PushClipRect(canvas,canvasMax,true);
                ImGui::Image(toImTextureId(texture.id),size);
                const ImVec2 toolsMin(canvas.x+8,canvas.y+42),toolsMax(canvas.x+42,canvas.y+182);
                const bool toolsHovered=ImGui::IsMouseHoveringRect(toolsMin,toolsMax);
                const bool imageHovered=ImGui::IsItemHovered()&&!toolsHovered;
                drawViewportToolRail(canvas,busySelectionIndex>=0);
                ImGui::GetWindowDrawList()->AddText(ImVec2(origin.x+12,origin.y+10),ImGui::GetColorU32(viewer::ui::muted),busyCameraSource==CameraSource::SceneXml?busyCamera.name:"User perspective");
                const bool replacingScene=previewActivity=="Updating training scene"||!busyOverlayMatchesScene;
                const auto hoveredMarker=drawDisplayedLightMarkers(busyCamera,origin,size,imageHovered&&!replacingScene,busySelectionKind,busySelectionIndex,busySelectedLightIndex);
                if(!replacingScene) {
                    if(ImGui::IsMouseClicked(ImGuiMouseButton_Left))pressedLightMarker=hoveredMarker;
                    if(pressedLightMarker&&ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                        if(hoveredMarker==pressedLightMarker&&io.MouseDragMaxDistanceSqr[ImGuiMouseButton_Left]<io.MouseDragThreshold*io.MouseDragThreshold)
                            queuedLightMarkerPick=std::make_pair(*hoveredMarker,displayedOverlayPointCloud);
                        pressedLightMarker.reset();
                    }
                }
                bool gizmoCapturing=viewportGizmoMouseCapture;
                if(replacingScene&&gizmosEnabled&&viewportTool!=0) {
                    displayedSelectedHandles.draw(*ImGui::GetWindowDrawList(),origin,size);
                    if(displayedSelectedHandles.contains(io.MousePos,origin,size)) {
                        gizmoCapturing=true;
                        if(ImGui::IsMouseDown(ImGuiMouseButton_Left))viewportGizmoMouseCapture=true;
                    }
                }
                if(busyHasSelection&&!replacingScene&&gizmosEnabled&&viewportTool!=0&&
                   (busySelectionKind==viewer::SceneItemKind::Emitter?showLightGizmo:showSurfelGizmo)) {
                    const bool emitter=busySelectionKind==viewer::SceneItemKind::Emitter;
                    glm::mat4 matrix=glm::translate(glm::mat4(1),busySelectionPosition)*glm::mat4_cast(glm::quat(glm::radians(busySelectionRotation)));
                    if(emitter)matrix=glm::translate(glm::scale(matrix,busySelectionScale),busyLightCenter);
                    else matrix=busyPointCloudTransform*matrix;
                    auto view=syclMatrixToGlm(busyCamera.view),projection=syclMatrixToGlm(busyCamera.proj);
                    ImGuizmo::SetDrawlist(ImGui::GetWindowDrawList());ImGuizmo::SetOrthographic(false);
                    ImGuizmo::SetRect(origin.x,origin.y,size.x,size.y);ImGuizmo::PushID(emitter?"area-light-gizmo":"surfel-gizmo");
                    if(ImGuizmo::Manipulate(glm::value_ptr(view),glm::value_ptr(projection),emitter?lightGizmoOperation:surfelGizmoOperation,emitter?(lightGizmoOperation==ImGuizmo::SCALE?ImGuizmo::LOCAL:lightGizmoMode):surfelGizmoMode,glm::value_ptr(matrix))) {
                        if(emitter) {
                            Pale::TransformComponent transform;transform.setTransform(glm::translate(matrix,-busyLightCenter));
                            busySelectionPosition=transform.translation;busySelectionRotation=transform.rotationEuler;busySelectionScale=transform.scale;
                        } else {
                            const auto local=glm::inverse(busyPointCloudTransform)*matrix;
                            busySelectionPosition=glm::vec3(local[3]);busySelectionRotation=glm::degrees(glm::eulerAngles(extractRotationQuaternion(local)));
                        }
                        queuedSelectionChanged=true;
                    }
                    gizmoCapturing=gizmoCapturing||ImGuizmo::IsOver()||ImGuizmo::IsUsingAny();ImGuizmo::PopID();
                }
                if(!walkNavigationActive&&busyCameraSource==CameraSource::Viewport) {
                    auto view=replacingScene?syclMatrixToGlm(busyCamera.view):queuedOrbit.viewMatrix();
                    const ImVec2 gizmoSize(92,92),gizmoPosition(origin.x+size.x-104,origin.y+12);
                    ImGuizmo::ViewManipulate(glm::value_ptr(view),queuedOrbit.distance,gizmoPosition,gizmoSize,IM_COL32(16,16,16,96));
                    if(ImGuizmo::IsUsingViewManipulate()) {
                        const auto cameraFromWorld=glm::inverse(view);
                        const auto forward=-glm::normalize(glm::vec3(cameraFromWorld[2]));
                        if(std::isfinite(forward.x)&&std::isfinite(forward.y)&&std::isfinite(forward.z)) {
                            queuedOrbit.setPositionKeepingTarget(queuedOrbit.target-forward*queuedOrbit.distance);
                            queuedOrbitChanged=true;lastNavigationInput=std::chrono::steady_clock::now();
                        }
                    }
                    gizmoCapturing=gizmoCapturing||ImGuizmo::IsViewManipulateHovered()||ImGuizmo::IsUsingViewManipulate();
                }
                if(!ImGui::IsMouseDown(ImGuiMouseButton_Left))viewportGizmoMouseCapture=false;
                if((imageHovered||(busyCameraSource==CameraSource::SceneXml&&ImGui::IsMouseHoveringRect(canvas,canvasMax)))&&!previewNoticeHovered&&!toolsHovered&&!hoveredMarker&&!gizmoCapturing&&!io.WantTextInput&&!walkNavigationActive) {
                    const bool orbitInput=ImGui::IsMouseDragging(ImGuiMouseButton_Left);
                    const bool panInput=ImGui::IsMouseDragging(ImGuiMouseButton_Right)||ImGui::IsMouseDragging(ImGuiMouseButton_Middle);
                    if(busyCameraSource==CameraSource::SceneXml&&!orbitInput) {
                        if(panInput)cameraImageView.drag(io.MouseDelta);
                        if(io.MouseWheel!=0)cameraImageView.zoomAt(io.MousePos,canvas,available,io.MouseWheel);
                    } else if(orbitInput||panInput||io.MouseWheel!=0) {
                        lastNavigationInput=std::chrono::steady_clock::now();queuedCamera.reset();
                        if(busyCameraSource==CameraSource::SceneXml&&!queuedOrbitChanged)queuedOrbit=makeOrbitCameraFromSceneCamera(busyCamera,busyBounds,queuedOrbit);
                        if(orbitInput)queuedOrbit.orbit(io.MouseDelta);
                        if(panInput)queuedOrbit.pan(io.MouseDelta);
                        if(io.MouseWheel!=0)queuedOrbit.zoom(io.MouseWheel);
                        queuedOrbitChanged=true;
                    }
                }
                ImGui::PopClipRect();viewer::finishViewportCanvas(viewportCanvas);
            } else ImGui::TextWrapped("The first preview will appear when compilation finishes.");
            if(walkNavigationActive&&!queuedWalkFinish) {
                queuedOrbit.lookAround(io.MouseDelta);
                const glm::vec3 direction{float(ImGui::IsKeyDown(ImGuiKey_D))-float(ImGui::IsKeyDown(ImGuiKey_A)),float(ImGui::IsKeyDown(ImGuiKey_W))-float(ImGui::IsKeyDown(ImGuiKey_S)),float(ImGui::IsKeyDown(ImGuiKey_E))-float(ImGui::IsKeyDown(ImGuiKey_Q))};
                if(glm::dot(direction,direction)>0||io.MouseDelta.x||io.MouseDelta.y) {
                    lastNavigationInput=std::chrono::steady_clock::now();
                    queuedOrbit.walk(direction,walkSpeed*(io.KeyShift?4.f:io.KeyAlt?.25f:1.f)*std::min(io.DeltaTime,.1f));queuedOrbitChanged=true;
                }
                if(ImGui::IsKeyPressed(ImGuiKey_Enter)||ImGui::IsKeyPressed(ImGuiKey_Escape)||!glfwGetWindowAttrib(window,GLFW_FOCUSED)) {
                    queuedWalkFinish=ImGui::IsKeyPressed(ImGuiKey_Escape)||!glfwGetWindowAttrib(window,GLFW_FOCUSED);
                    glfwSetInputMode(window,GLFW_CURSOR,GLFW_CURSOR_NORMAL);
                }
            }
            drawViewportStatus(true);
            ImGui::End();training.drawProgress(showTrainingProgress);
            if(drawProfilingWindow(showProfilingWindow,busyTimerProfilingEnabled,busyGpuCounterProfilingEnabled,
                busyRenderMs,busyFrameTimes,busyCamera.width,busyCamera.height,busyProfilingScene,busyProfilingCounters,busyTimerRecords))
                queuedProfilingSettings=true;
            if(meshExport.draw(showMeshExport,displayedExportMetadata.IsMap(),displayedOverlayPointCloud.filename().string()))meshExportRequested=true;
            launchMeshExport(true);
            ImGui::Render();int width=0,height=0;glfwGetFramebufferSize(window,&width,&height);
            glViewport(0,0,width,height);glClearColor(kBlenderViewportBackground.x,kBlenderViewportBackground.y,kBlenderViewportBackground.z,1);
            glClear(GL_COLOR_BUFFER_BIT);ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());glfwSwapBuffers(window);
        };

        // This guard is declared after all worker-referenced lambdas, so an
        // exception in the UI joins the worker before those captures unwind.
        struct ViewerWorkLifetime {
            viewer::TrainingSession& training;
            viewer::TrainingSession& exporter;
            viewer::PreviewWork& preview;
            ~ViewerWorkLifetime() {
                training.shutdown();exporter.shutdown();
                try {preview.finish();}catch(...){}
            }
        } viewerWorkLifetime{training.process,meshExport.process,previewWork};

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            training.poll();
            meshExport.poll();
            if(previewWork.ready()) {
                previewWatchdog.finish();
                const auto priorTextureRevision=texture.revision;
                const auto completionFailed=[&](const std::string& error) {
                    previewPublication.finish(texture.revision!=priorTextureRevision,restorePublishedPreview);
                    recordPreviewFailure(error);
                };
                try {
                    previewWork.finish();if(previewCompletion)previewCompletion();
                    previewPublication.finish(true,restorePublishedPreview);
                }
                catch(const std::bad_alloc&) {completionFailed("Preview memory allocation failed. Reduce preview resolution or diagnostic cost, then retry.");}
                catch(const std::exception& error) {completionFailed(error.what());}
                catch(...) {completionFailed("Unknown preview worker failure");}
                previewCompletion={};
                if(queuedProfilingSettings) {
                    if(gpuCounterProfilingEnabled!=busyGpuCounterProfilingEnabled)tracerDirty=true;
                    timerProfilingEnabled=busyTimerProfilingEnabled;gpuCounterProfilingEnabled=busyGpuCounterProfilingEnabled;
                    queuedProfilingSettings=false;renderRequested=true;
                }
                if(queuedLightMarkerPick) {
                    if(queuedLightMarkerPick->second==currentPointCloudPath) {
                        const auto pick=queuedLightMarkerPick->first;
                        if(pick.areaIndex>=0) {selectedLightIndex=pick.areaIndex;selectedSurfelEditorIndex=-1;sceneSelectionKind=viewer::SceneItemKind::Emitter;}
                        else {selectedSurfelEditorIndex=pick.surfelIndex;sceneSelectionKind=viewer::SceneItemKind::SurfelEmitter;}
                        openLightControls=true;inspectorPage=viewer::ui::Inspector::Selection;
                    }
                    queuedLightMarkerPick.reset();
                }
                if(queuedSelectionChanged) {
                    queuedSelectionChanged=false;beginSnapshotEdit();
                    if(busySelectionKind==viewer::SceneItemKind::Emitter) {
                        for(auto light:collectAreaLights(scene))if(entityName(light)==busySelectionName) {
                            auto& transform=light.getComponent<Pale::TransformComponent>();transform.translation=busySelectionPosition;transform.setRotationEuler(busySelectionRotation);transform.scale=busySelectionScale;
                            auto& emitter=light.getComponent<Pale::AreaLightComponent>();emitter.radiance=busySelectionColor;emitter.flux=busySelectionPower;
                            if(light.hasComponent<Pale::MaterialComponent>())if(auto material=assetManager.get<Pale::Material>(light.getComponent<Pale::MaterialComponent>().materialID)) {material->baseColor=busySelectionColor;material->power=busySelectionPower;}
                            openLightControls=true;break;
                        }
                    } else if(const auto handle=firstPointCloudHandle(scene)) {
                        const auto cloud=assetAccessor.getPointCloud(*handle);
                        if(cloud)if(auto point=resolveEditableSurfel(*cloud,busySelectionIndex);point&&point->pointGeometry) {
                            auto& geometry=*point->pointGeometry;const auto i=point->localIndex;
                            geometry.positions[i]=busySelectionPosition;geometry.quat[i]=glm::quat(glm::radians(busySelectionRotation));geometry.scales[i]=glm::vec2(busySelectionScale);
                            geometry.albedos[i]=busySelectionColor;geometry.powers[i]=busySelectionPower;geometry.opacities[i]=busySelectionOpacity;geometry.betas[i]=busySelectionBeta;
                            selectedSurfelEditorIndex=busySelectionIndex;openSurfelControls=true;
                        }
                    }
                    rebuildSceneGpu();
                }
                if(queuedOrbitChanged) {orbit=queuedOrbit;cameraSource=CameraSource::Viewport;cameraDirty=renderRequested=true;queuedOrbitChanged=false;}
                if(queuedCamera && *queuedCamera>=0 && static_cast<std::size_t>(*queuedCamera)<buildProducts.cameraGPUs.size()) {
                    selectedSceneCameraIndex=*queuedCamera;cameraSource=CameraSource::SceneXml;cameraDirty=renderRequested=true;
                }
                queuedCamera.reset();
                if(queuedWalkFinish) {finishWalkNavigation(*queuedWalkFinish);queuedWalkFinish.reset();}
                if(queuedView) {setViewImageMode(*queuedView);queuedView.reset();}
                if(queuedSceneSource) {enterTrainingView(*queuedSceneSource);queuedSceneSource.reset();}
                if(editLatestRequested) {editLatestRequested=false;editLatestSavedSnapshot();}
                if(returnToEditsRequested) {returnToEditsRequested=false;restoreWorkingSnapshot();}
                if(queuedRefresh) {renderRequested=true;queuedRefresh=false;}
                if(followLaunchAfterPreview) {
                    if(saveWorkingSnapshot()) {invalidateSceneMaterials();training.view=viewer::TrainingView::Live;training.loadedGeneration=0;}
                    followLaunchAfterPreview=false;
                }
            }
            if(previewWork.busy()&&previewWatchdog.reportDue())recordLongPreview();
            if(previewWork.busy()) {drawBusyFrame();continue;}
            if (walkNavigationActive && !glfwGetWindowAttrib(window, GLFW_FOCUSED)) {
                finishWalkNavigation(true);
            }
            if (glfwWindowShouldClose(window)) { break; }

            if (modelReadOnly() && dropState.hasPendingPath) {
                dropState.hasPendingPath = false;
                training.message = "Switch to an editable snapshot before loading other geometry";
            }
            if (dropState.hasPendingPath) {
                const std::filesystem::path droppedPath = dropState.pendingPath;
                dropState.hasPendingPath = false;
                finishWalkNavigation(false);
                std::error_code error;
                if (std::filesystem::is_directory(droppedPath, error)) {
                    optimizationFolderAnchor.set(droppedPath);
                    latestOptimizationMode = false;
                    latestOptimizationPointsDirectory.clear();
                    latestOptimizationSnapshots.clear();
                    latestOptimizationSnapshotIndex = 0u;
                    refreshLatestOptimizationPointCloud();
                } else {
                    handleDroppedPly(droppedPath);
                }
            }

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            // Avoid replaying stale movement over several expensive render frames.
            io.ConfigInputTrickleEventQueue = !walkNavigationActive;
            if(inspectorInput.pending())inspectorHandoffFrames=std::max(inspectorHandoffFrames,2);
            inspectorInput.deliver(static_cast<int>(inspectorPage),inspectorContextKey(inspectorPage),glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
            ImGui::NewFrame();
            // Snapshot the actual object representation to detect every settings
            // control without coupling cache invalidation to individual widgets.
            std::array<unsigned char, sizeof(settings)> settingsBeforeUi;
            std::memcpy(settingsBeforeUi.data(), &settings, sizeof(settings));

            bool navigationInputCaptured = walkNavigationActive;
            const bool navigationShortcutAllowed =
                !io.WantTextInput && !io.KeyCtrl && !io.KeyAlt && !io.KeySuper;
            if ((walkNavigationActive || navigationShortcutAllowed) &&
                viewerKeyPressed(ImGuiKey_Space)) {
                resetOrbitView();
                navigationInputCaptured = true;
            } else if (walkNavigationActive) {
                if (viewerKeyPressed(ImGuiKey_Escape) ||
                    walkMouseClicked(ImGuiMouseButton_Right)) {
                    finishWalkNavigation(true);
                } else if (viewerKeyPressed(ImGuiKey_Enter) ||
                           viewerKeyPressed(ImGuiKey_Backslash) ||
                           walkMouseClicked(ImGuiMouseButton_Left)) {
                    finishWalkNavigation(false);
                } else {
                    double cursorX = 0.0, cursorY = 0.0;
                    glfwGetCursorPos(window, &cursorX, &cursorY);
                    const ImVec2 lookDelta{
                        static_cast<float>(cursorX - walkCursorX),
                        static_cast<float>(cursorY - walkCursorY)};
                    walkCursorX = cursorX;
                    walkCursorY = cursorY;
                    walkSpeed = std::clamp(walkSpeed * std::exp(io.MouseWheel * 0.2f), 0.01f, 1000.0f);
                    const glm::vec3 direction{
                        static_cast<float>(ImGui::IsKeyDown(ImGuiKey_D)) - static_cast<float>(ImGui::IsKeyDown(ImGuiKey_A)),
                        static_cast<float>(ImGui::IsKeyDown(ImGuiKey_W)) - static_cast<float>(ImGui::IsKeyDown(ImGuiKey_S)),
                        static_cast<float>(ImGui::IsKeyDown(ImGuiKey_E)) - static_cast<float>(ImGui::IsKeyDown(ImGuiKey_Q))};
                    if (lookDelta.x != 0.0f || lookDelta.y != 0.0f || glm::dot(direction, direction) > 0.0f) {
                        lastNavigationInput = std::chrono::steady_clock::now();
                        orbit.lookAround(lookDelta);
                        const float speedMultiplier = io.KeyShift ? 4.0f : (io.KeyAlt ? 0.25f : 1.0f);
                        orbit.walk(direction, walkSpeed * speedMultiplier * std::min(io.DeltaTime, 0.1f));
                        cameraDirty = true;
                        renderRequested = true;
                    }
                }
            } else if (navigationShortcutAllowed && viewerKeyPressed(ImGuiKey_Backslash)) {
                orbitBeforeWalk = orbit;
                cameraSourceBeforeWalk = cameraSource;
                if (cameraSource == CameraSource::SceneXml && !buildProducts.cameraGPUs.empty()) {
                    orbit = makeOrbitCameraFromSceneCamera(
                        buildProducts.cameraGPUs[static_cast<std::size_t>(selectedSceneCameraIndex)], bounds, orbit);
                }
                cameraSource = CameraSource::Viewport;
                walkNavigationActive = true;
                navigationInputCaptured = true;
                walkSpeed = std::clamp(bounds.radius * 0.5f, 0.1f, 50.0f);
                viewportPickArmed = false;
                viewportGizmoMouseCapture = false;
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                if (glfwRawMouseMotionSupported()) {
                    glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
                }
                glfwGetCursorPos(window, &walkCursorX, &walkCursorY);
                cameraDirty = true;
                tracerDirty = true;
                renderRequested = true;
            } else if (viewerKeyPressed(ImGuiKey_Escape)) {
                Pale::Log::PA_INFO("Viewer close requested by Escape");
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }

            // Capture walk input without dimming the UI or the rendered image.
            // This ImGui version makes color pickers transparent inside
            // BeginDisabled(false), so only open a disabled scope when needed.
            if (navigationInputCaptured) {
                ImGui::PushStyleVar(ImGuiStyleVar_DisabledAlpha, 1.0f);
                ImGui::BeginDisabled();
                ImGui::PopStyleVar();
            }
            if(drawApplicationMenu(false))launchTrainingRequested=true;

            ImGuiViewport* mainViewport = ImGui::GetMainViewport();
            const ImGuiID dockspaceId = ImHashStr("PaleRealtimeViewerDockSpace");
            if (!dockLayoutInitialized) {
                dockLayoutInitialized = true;
                viewer::ui::defaultDockLayout(dockspaceId, mainViewport);
            }
            ImGui::DockSpaceOverViewport(
                dockspaceId,
                mainViewport,
                ImGuiDockNodeFlags_PassthruCentralNode);
            ImGuizmo::BeginFrame();

            // Walk navigation owns mouse movement, but viewer shortcuts remain available.
            if (!io.WantTextInput &&
                !io.KeyCtrl &&
                !io.KeyAlt &&
                !io.KeySuper) {
                if (viewerKeyPressed(ImGuiKey_R)) {
                    if (training.process.launched()) {
                        enterTrainingView(viewer::TrainingView::Live);
                    } else refreshLatestOptimizationPointCloud();
                }

                if (!training.process.launched() && viewerKeyPressed(ImGuiKey_F)) {
                    jumpToOptimizationSnapshotBoundary(false);
                }

                if (!training.process.launched() && viewerKeyPressed(ImGuiKey_L)) {
                    jumpToOptimizationSnapshotBoundary(true);
                }

                if (viewerKeyPressed(ImGuiKey_N)) {
                    if(training.process.launched())training.checkpointNavigation=-2;
                    else stepLatestOptimizationSnapshot(-1,kSnapshotIterationStep);
                }

                if (viewerKeyPressed(ImGuiKey_M)) {
                    if(training.process.launched())training.checkpointNavigation=2;
                    else stepLatestOptimizationSnapshot(1,kSnapshotIterationStep);
                }

                if (!training.process.launched() && ImGui::IsKeyDown(ImGuiKey_LeftArrow)) {
                    stepLatestOptimizationSnapshot(-1);
                }

                if (!training.process.launched() && ImGui::IsKeyDown(ImGuiKey_RightArrow)) {
                    stepLatestOptimizationSnapshot(1);
                }

                // Scene cameras cycle once per frame while held, like the
                // horizontal snapshot arrows. Keep one-shot snapshot behavior
                // for the viewport camera and preserve short camera-key taps.
                const bool cycleSceneCameras = cameraSource == CameraSource::SceneXml;
                const int verticalArrowStep =
                    static_cast<int>(viewerKeyPressed(ImGuiKey_UpArrow) ||
                        (cycleSceneCameras && ImGui::IsKeyDown(ImGuiKey_UpArrow))) -
                    static_cast<int>(viewerKeyPressed(ImGuiKey_DownArrow) ||
                        (cycleSceneCameras && ImGui::IsKeyDown(ImGuiKey_DownArrow)));
                if (verticalArrowStep != 0) {
                    if (cameraSource == CameraSource::SceneXml) {
                        const int cameraCount = static_cast<int>(buildProducts.cameraGPUs.size());
                        if (cameraCount > 1) {
                            selectedSceneCameraIndex =
                                (selectedSceneCameraIndex + verticalArrowStep + cameraCount) % cameraCount;
                            cameraDirty = true;
                            tracerDirty = true;
                            renderRequested = true;
                        }
                    } else if (!training.process.launched()) {
                        stepLatestOptimizationSnapshot(verticalArrowStep);
                    }
                }

                // The first ten menu/cycle entries map to 1..9, then 0.
                constexpr std::array<ImGuiKey, 10> viewImageModeShortcutKeys = {
                    ImGuiKey_1,
                    ImGuiKey_2,
                    ImGuiKey_3,
                    ImGuiKey_4,
                    ImGuiKey_5,
                    ImGuiKey_6,
                    ImGuiKey_7,
                    ImGuiKey_8,
                    ImGuiKey_9,
                    ImGuiKey_0,
                };

                for (std::size_t shortcutIndex = 0;
                     shortcutIndex < viewImageModeShortcutKeys.size();
                     ++shortcutIndex) {
                    if (viewerKeyPressed(viewImageModeShortcutKeys[shortcutIndex])) {
                        setViewImageMode(kViewImageModeCycleOrder[shortcutIndex]);
                        break;
                    }
                }

                const bool nextViewModePressed =
                    viewerKeyPressed(ImGuiKey_KeypadAdd) ||
                    (io.KeyShift && viewerKeyPressed(ImGuiKey_Equal));
                const bool previousViewModePressed =
                    viewerKeyPressed(ImGuiKey_KeypadSubtract) ||
                    viewerKeyPressed(ImGuiKey_Minus);
                if (nextViewModePressed) {
                    cycleViewImageMode(1);
                } else if (previousViewModePressed) {
                    cycleViewImageMode(-1);
                }
            }
            std::vector<Pale::Entity> areaLights = collectAreaLights(scene);
            if (areaLights.empty()) {
                selectedLightIndex = 0;
            } else {
                selectedLightIndex = std::clamp(
                    selectedLightIndex,
                    0,
                    static_cast<int>(areaLights.size() - 1u));
            }
            Pale::Entity selectedLight =
                areaLights.empty() ? Pale::Entity{} : areaLights[static_cast<std::size_t>(selectedLightIndex)];

            auto sceneGroups=makeSceneGroups();
            cachedSceneGroups=sceneGroups;cachedSceneName=currentScenePath.stem().string();
            auto treePick=sceneTree.draw(cachedSceneName,sceneGroups,showSceneTree);
            if(!treePick&&queuedScenePick) {treePick=queuedScenePick;queuedScenePick.reset();}
            if (auto picked=treePick) {
                if(picked->kind==viewer::SceneItemKind::Emitter && (picked->index<0 || static_cast<std::size_t>(picked->index)>=areaLights.size()))picked.reset();
                if(picked && picked->kind==viewer::SceneItemKind::SurfelEmitter && (picked->index<0 || static_cast<std::size_t>(picked->index)>=sceneTreeEmitterIndices.size()))picked.reset();
                if(picked && picked->kind==viewer::SceneItemKind::Camera && (picked->index<0 || static_cast<std::size_t>(picked->index)>=buildProducts.cameraGPUs.size()))picked.reset();
                if(picked) {
                sceneSelectionKind=picked->kind; selectedTreeEntity=picked->name;
                inspectorPage=viewer::ui::Inspector::Selection;
                if (picked->kind==viewer::SceneItemKind::Emitter) {
                    selectedLightIndex=picked->index; selectedLight=areaLights[selectedLightIndex]; selectedSurfelEditorIndex=-1; openLightControls=true;
                } else if (picked->kind==viewer::SceneItemKind::SurfelEmitter) {
                    selectedSurfelLightIndex=picked->index; selectedSurfelEditorIndex=static_cast<int>(sceneTreeEmitterIndices[picked->index]); openLightControls=true;
                } else if (picked->kind==viewer::SceneItemKind::Camera) {
                    selectedSceneCameraIndex=picked->index; cameraSource=CameraSource::SceneXml;
                    inspectorPage=viewer::ui::Inspector::Camera; cameraDirty=tracerDirty=renderRequested=true;
                } else if (picked->kind==viewer::SceneItemKind::Geometry || (picked->kind==viewer::SceneItemKind::Material && picked->index==-1)) {
                    sceneSelectionKind=viewer::SceneItemKind::Geometry; openSurfelControls=true;
                }
            }
            }
            if (selectTrainingTab) { inspectorPage = viewer::ui::Inspector::Reconstruction; selectTrainingTab = false; }
            if (openLightControls || openSurfelControls) {
                inspectorPage = viewer::ui::Inspector::Selection;
                sceneSelectionKind = openLightControls ? (selectedSurfelEditorIndex >= 0 ? viewer::SceneItemKind::SurfelEmitter : viewer::SceneItemKind::Emitter) : viewer::SceneItemKind::Geometry;
            }
            ImGui::Begin("Properties", nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
            ImGui::BeginChild("InspectorRail", ImVec2(32,0), ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar);
            viewer::ui::inspectorRail(inspectorPage, showAdvancedTools);
            ImGui::EndChild(); ImGui::SameLine(0,8);
            ImGui::BeginChild("InspectorContents", ImVec2(0,0));
            ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * .52f);
            if (inspectorPage == viewer::ui::Inspector::Reconstruction) {
                try {
                    if (training.drawSetup(hasWorkingSnapshot || (!modelReadOnly() && firstPointCloudHandle(scene).has_value()))) launchTrainingRequested = true;
                } catch (const std::exception& error) { training.message = error.what(); }
            }
            if (inspectorPage == viewer::ui::Inspector::View) {
            if (ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_DefaultOpen)) {
                const ScopedSectionIndent sectionIndent;
                if (ImGui::BeginCombo("View", viewImageModeLabel(viewImageMode))) {
                    for (const ViewImageMode candidateMode : kViewImageModeCycleOrder) {
                        if (!showAdvancedTools && candidateMode != ViewImageMode::Rendered && candidateMode != ViewImageMode::MedianDepth && candidateMode != ViewImageMode::VisibleNormal && candidateMode != ViewImageMode::TargetLoss && candidateMode != ViewImageMode::SurfelDensity && candidateMode != ViewImageMode::RawLinear) continue;
                        const bool selected = viewImageMode == candidateMode;
                        if (ImGui::Selectable(viewImageModeLabel(candidateMode), selected)) {
                            setViewImageMode(candidateMode);
                        }
                        if (selected) {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }
                if (viewImageMode == ViewImageMode::RawLinear) {
                    ImGui::TextWrapped("Raw primal RGB: no exposure, sRGB/gamma, normalization, or alpha blending. Display clips to [0,1]; non-finite RGB is magenta.");
                }
                if (viewImageMode != ViewImageMode::Rendered && viewImageMode != ViewImageMode::RawLinear &&
                        ImGui::TreeNodeEx("View options", ImGuiTreeNodeFlags_DefaultOpen)) {
                    const ScopedSectionIndent sectionIndent;
                    if (viewImageMode == ViewImageMode::DepthDistortion ||
                        viewImageMode == ViewImageMode::DepthPositionGradient) {
                        ImGui::TextWrapped("Preview uses unit loss weight; colors rescale to each frame.");
                    }
                    if (viewImageMode == ViewImageMode::TerminalTransmittance) {
                        if (ImGui::SliderFloat("Target attenuation", &terminalTargetAttenuation, 0.8f, 1.0f, "%.2f")) {
                            debugDisplayBuffers.terminalTransmittanceValid = false;
                            updateDisplayTexture();
                        }
                        ImGui::TextWrapped("Camera rays only. Loss = max(0, target - attenuation)^2. Empty rays and rays meeting the target are black. Magenta means traversal limit reached; escape is unknown.");
                        ImGui::Text("Fixed loss color range: 0 to %.3f", terminalTargetAttenuation * terminalTargetAttenuation);
                        ImGui::TextWrapped("Preview only: no opacity or training updates. Uses surfel footprints and slab compositing.");
                    }
                    if (viewImageMode == ViewImageMode::DepthDistortion || viewImageMode == ViewImageMode::DepthPositionGradient) {
                        int distortionDepthMode = settings.depthDistortionWorldSpace ? (settings.depthDistortionSquared ? 1 : 0) : 2;
                        const char* distortionDepthModes[] = {"World distance (absolute)", "World distance (squared)", "Normalized depth (squared)"};
                        if (ImGui::Combo("Depth distortion", &distortionDepthMode, distortionDepthModes, 3)) {
                            settings.depthDistortionWorldSpace = distortionDepthMode != 2;
                            settings.depthDistortionSquared = distortionDepthMode == 1;
                            renderRequested = true;
                        }
                    }

                    if (viewImageMode == ViewImageMode::AdjointDerivative) {
                        bool changed = ImGui::Combo("Derivative property", &viewerAdjointProperty,
                            adjointProperties, static_cast<int>(std::size(adjointProperties)));
                        changed |= ImGui::Checkbox("All surfels (common change)", &viewerAdjointAllSurfels);
                        if (!viewerAdjointAllSurfels) {
                            changed |= ImGui::InputInt("Global surfel index", &viewerAdjointSurfelIndex);
                            viewerAdjointSurfelIndex = std::clamp(viewerAdjointSurfelIndex, 0,
                                std::max(0, static_cast<int>(sceneGpu.pointCount) - 1));
                            std::optional<uint32_t> pickedGlobalIndex;
                            if (const auto handle = firstPointCloudHandle(scene);
                                handle && selectedSurfelEditorIndex >= 0) {
                                const auto found = buildProducts.pointCloudIndexById.find(*handle);
                                if (found != buildProducts.pointCloudIndexById.end()) {
                                    const auto& range = buildProducts.pointCloudRanges.at(found->second);
                                    if (static_cast<uint32_t>(selectedSurfelEditorIndex) < range.pointCount)
                                        pickedGlobalIndex = range.firstPoint + selectedSurfelEditorIndex;
                                }
                            }
                            ImGui::BeginDisabled(!pickedGlobalIndex);
                            if (ImGui::Button("Use picked surfel") && pickedGlobalIndex) {
                                viewerAdjointSurfelIndex = static_cast<int>(*pickedGlobalIndex);
                                changed = true;
                            }
                            ImGui::EndDisabled();
                            ImGui::TextDisabled("Click the viewport to pick, then use the picked surfel");
                        }
                        changed |= ImGui::SliderInt("Derivative adjoint SPP", &viewerAdjointSamplesPerPixel,
                            1, 256, "%d", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
                        ImGui::Text("Fixed: qNull %.3f, qReflect %.3f",
                            settings.sampling.qNull, settings.sampling.qReflect);
                        changed |= ImGui::InputInt("Adjoint seed", &viewerAdjointSeed);
                        viewerAdjointSeed = std::max(0, viewerAdjointSeed);
                        if (ImGui::Button("Resample adjoint")) {
                            viewerAdjointSeed = (viewerAdjointSeed + 1u) % 2147483647u;
                            changed = true;
                        }
                        if (changed) renderRequested = true;
                        if (ImGui::DragFloat("Full color at |derivative|", &viewerAdjointColorMaximum,
                            0.001f, 1.0e-10f, 1.0e10f, "%.6g", ImGuiSliderFlags_Logarithmic)) {
                            viewerAdjointColorMaximum = std::max(1.0e-10f, viewerAdjointColorMaximum);
                            updateDisplayTexture();
                        }
                        if (ImGui::Button("Fit derivative colors") && viewerAdjointImageValid) {
                            viewerAdjointColorMaximum = 1.0e-10f;
                            for (float g : viewerAdjointImage) if (std::isfinite(g))
                                viewerAdjointColorMaximum = std::max(viewerAdjointColorMaximum, std::abs(g));
                            updateDisplayTexture();
                        }
                        ImGui::TextWrapped("Red: increasing this property increases linear R+G+B. "
                            "Blue: decreases it. White: zero. Colors stay fixed across SPP and seed changes.");
                        ImGui::TextWrapped("All surfels sums the derivative for the same property change everywhere. "
                            "Rotation uses local tangent-frame radians; albedo changes the red channel only.");
                        ImGui::Text("Adjoint: %.3f ms", lastViewerAdjointMs);
                        ImGui::TextWrapped("%s", viewerAdjointStatus.c_str());
                        if (settings.rendererDebugShareLocalLayerDirectLighting)
                            ImGui::TextWrapped("Shared lighting is enabled: this displays its existing detached-lighting VJP.");
                    }
                    if ((viewImageMode == ViewImageMode::MedianDepth || viewImageMode == ViewImageMode::MedianDepthNormal ||
                         viewImageMode == ViewImageMode::NormalPositionGradient) &&
                        ImGui::TreeNodeEx("Median depth settings", ImGuiTreeNodeFlags_DefaultOpen)) {
                        const ScopedSectionIndent sectionIndent;
                        if (ImGui::SliderFloat("Median depth threshold", &settings.medianDepthThreshold,
                                               0.001f, 0.999f, "%.3f", ImGuiSliderFlags_AlwaysClamp)) {
                            settings.medianDepthThreshold = std::clamp(settings.medianDepthThreshold, 0.001f, 0.999f);
                            renderRequested = true;
                        }
                        ImGui::TextWrapped("Accumulated opacity needed to select depth. Lower values select earlier hits.");
                        if (ImGui::Checkbox("Retain last accepted depth", &settings.medianDepthRetainLastHit)) {
                            renderRequested = true;
                        }
                        ImGui::TextWrapped("If the threshold is never reached, use the last accepted surface hit. Empty rays still have no depth.");
                        ImGui::TextDisabled("The existing traversal-limit fallback remains active.");
                        ImGui::TreePop();
                    }
                    if (viewImageMode == ViewImageMode::IntraSlabDepth ||
                        viewImageMode == ViewImageMode::IntraSlabRayDepth) {
                        ImGui::TextWrapped("Both slab losses use the same logarithmic color scale and unit weight.");
                        ImGui::TextWrapped("Mean ray depth: equal-weight squared hit-depth differences from the slab mean.");
                        if (settings.sharedHeightEnabled) {
                            ImGui::TextWrapped("Slab loss previews are unavailable while Shared surface experiment is enabled.");
                        } else if (debugDisplayBuffers.intraSlabDepthValid) {
                            ImGui::Text("Mean per pixel: plane %.6g | ray %.6g",
                                debugDisplayBuffers.slabPlaneLossMean, debugDisplayBuffers.slabRayLossMean);
                            ImGui::Text("Shared color range: 0 to %.6g", debugDisplayBuffers.slabLossColorMaximum);
                        }
                    }
                    if (viewImageMode == ViewImageMode::MeanDepth ||
                        viewImageMode == ViewImageMode::MedianDepth ||
                        viewImageMode == ViewImageMode::DepthDistortion ||
                        viewImageMode == ViewImageMode::IntraSlabDepth ||
                        viewImageMode == ViewImageMode::IntraSlabRayDepth ||
                        viewImageMode == ViewImageMode::PositionPrimitiveScore ||
                        viewImageMode == ViewImageMode::TerminalTransmittance ||
                        viewImageMode == ViewImageMode::SurfelDensity ||
                        viewImageMode == ViewImageMode::SurfaceOverlap ||
                        viewImageMode == ViewImageMode::TargetLoss ||
                        viewImageMode == ViewImageMode::DepthPositionGradient ||
                        viewImageMode == ViewImageMode::NormalPositionGradient ||
                        viewImageMode == ViewImageMode::IntraSlabPositionGradient) {
                        int scalarColorMapIndex = static_cast<int>(scalarColorMap);
                        const char* scalarColorMaps[] = {"Viridis", "Jet"};
                        if (ImGui::Combo(
                                "Scalar colormap",
                                &scalarColorMapIndex,
                                scalarColorMaps,
                                IM_ARRAYSIZE(scalarColorMaps))) {
                            scalarColorMap = static_cast<ScalarColorMap>(scalarColorMapIndex);
                            updateDisplayTexture();
                        }
                    }
                    if (viewImageMode == ViewImageMode::SurfaceOverlap) {
                        const char* metrics[] = {"Mean slab members", "Slab overflow footprint (%)", "Center slab members"};
                        bool refreshMap = ImGui::Combo("Overlap metric", &surfaceOverlapMetric, metrics, IM_ARRAYSIZE(metrics));
                        refreshMap |= ImGui::Checkbox("Normal-distance overlap", &settings.surfaceOverlapNormalDistance);
                        if (ImGui::DragFloat("Overlap distance tolerance", &settings.surfaceOverlapDepthTolerance,
                                0.0001f, 1.0e-5f, 10.0f, "%.6f", ImGuiSliderFlags_AlwaysClamp)) {
                            settings.surfaceOverlapDepthTolerance = std::max(settings.surfaceOverlapDepthTolerance, 1.0e-5f);
                            refreshMap = true;
                        }
                        ImGui::TextWrapped(settings.surfaceOverlapNormalDistance
                            ? "Symmetric distance along each surfel's normal. Includes neighbors in front and behind; ray intervals widen at grazing angles."
                            : "Symmetric distance along each camera ray. At grazing angles, the same normal separation can fall outside the interval.");
                        const float normalCosine = Pale::rendererDebugLocalLayerNormalCosineThreshold(settings);
                        const auto memberLimit = Pale::rendererDebugMaxLocalSurfelHits(settings);
                        ImGui::Text("Inherited normal cosine: %.3f (%.1f degrees)",
                            normalCosine, glm::degrees(std::acos(normalCosine)));
                        ImGui::Text("Overflow: more than %u total members (including anchor)", memberLimit);
                        ImGui::TextDisabled("Overlap distance is independent of Surfel traversal. Alpha filtering is disabled.");
                        if (viewImageMode == ViewImageMode::SurfaceOverlap && surfaceOverlapMetric != kSlabOverflowMetric) {
                            refreshMap |= ImGui::SliderFloat("Overlap color maximum (members)",
                                &surfaceOverlapColorMaximum, 1.0f, 128.0f, "%.1f");
                            surfaceOverlapColorMaximum = std::max(surfaceOverlapColorMaximum, 1.0f);
                        }
                        if (refreshMap) {
                            debugDisplayBuffers.surfaceOverlapValid = false;
                            renderRequested = true;
                        }
                        ImGui::Text("Observed surfels: %zu   peak mean slab members: %.2f",
                            surfaceOverlapEligibleCount, surfaceOverlapPeak);
                        ImGui::Text("64 samples/surfel x %zu cameras; last calculation: %.1f ms",
                            surfaceOverlapViews.size(), surfaceOverlapComputeMs);
                        ImGui::TextWrapped("Counts full-footprint intersections in a slab anchored at each sample, using the renderer's "
                            "ray-facing normal test and the overlap distance above. Counts include the anchor, exclude lights, and continue beyond the member limit.");
                        ImGui::TextWrapped("Overflow is the percentage of in-frame sample/camera pairs exceeding the slab capacity. "
                            "Color range: 0%% to 100%%. Member counts are averaged over those pairs.");
                        ImGui::TextWrapped(buildProducts.cameraGPUs.empty()
                            ? "No saved scene cameras: using the displayed camera; scores change when it moves."
                            : "Uses saved scene cameras; orbiting the viewer only changes which surfel scores are displayed.");
                        ImGui::TextWrapped("Potential membership only: hidden surfaces are included; occlusion, transmission and ray-event limits "
                            "are not applied. Black means no observed surfel. Training uses the same scoring rule with its own overlap settings; viewer changes are preview-only.");
                    }
                    if (viewImageMode == ViewImageMode::TransportScore) {
                        if (ImGui::Checkbox("Use saved run settings", &transportUseSavedSettings)) {
                            transportDirty = true; renderRequested = true;
                        }
                        if (ImGui::Button("Recompute contributions")) { transportDirty = true; renderRequested = true; }
                        ImGui::TextWrapped("%s", transportStatus.c_str());
                        bool recolor = ImGui::Combo("Score component", &transportComponent, "Camera + shadow\0Camera only\0Shadow only\0");
                        const char* sourceLabel = transportSource < 0 || transportSource >= int(transportViews.size()) ? "All training cameras (one cycle)" : transportViews[transportSource].name.c_str();
                        if (ImGui::BeginCombo("Score source", sourceLabel)) {
                            if (ImGui::Selectable("All training cameras (one cycle)", transportSource < 0)) { transportSource = -1; recolor = true; }
                            for (std::size_t i = 0; i < transportViews.size(); ++i) {
                                ImGui::PushID(static_cast<int>(i));
                                const auto label = transportViews[i].name + (transportViews[i].training ? " [training]" : " [held out]");
                                if (ImGui::Selectable(label.c_str(), transportSource == int(i))) { transportSource = int(i); recolor = true; }
                                ImGui::PopID();
                            }
                            ImGui::EndCombo();
                        }
                        recolor |= ImGui::DragFloat("Passing score", &transportThreshold, 0.1f, 0.001f, 1e9f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
                        if (ImGui::Button("Reset passing score to 10")) { transportThreshold = 10.0f; recolor = true; }
                        if (recolor) { refreshTransportSelection(); updateDisplayTexture(); }
                        ImGui::Checkbox("X-ray affected centers", &transportXray);
                        ImGui::Text("Below threshold: %zu surfels", transportAffected.size());
                        const auto trainingCameras = std::count_if(transportViews.begin(), transportViews.end(),
                            [](const auto& view) { return view.training; });
                        ImGui::Text("Training cameras: %zu / %zu", static_cast<std::size_t>(trainingCameras), transportViews.size());
                        if (!transportViews.empty() && trainingCameras == 0)
                            ImGui::TextWrapped("No cameras are marked for training: the training total is zero. Select an individual camera to inspect its contributions.");
                        ImGui::TextWrapped("Pink: score < threshold. Green: score >= threshold. Camera + shadow summed over training cameras matches the pruning score. Preview only; no surfels are removed.");
                        ImGui::TextWrapped("Scores use incoming transmission and effective opacity. Hidden surfels can have zero support. X-ray shows up to 20,000 affected centers, including occluded ones; click a marker to inspect it.");
                        if (ImGui::Button("Select next affected surfel")) {
                            const auto handle = firstPointCloudHandle(scene);
                            const auto found = handle ? buildProducts.pointCloudIndexById.find(*handle) : buildProducts.pointCloudIndexById.end();
                            if (found != buildProducts.pointCloudIndexById.end()) {
                                const auto& range = buildProducts.pointCloudRanges.at(found->second);
                                int first = -1, next = -1;
                                for (const auto primitive : transportAffected) {
                                    if (primitive < range.firstPoint || primitive - range.firstPoint >= range.pointCount) continue;
                                    const int local = static_cast<int>(primitive - range.firstPoint);
                                    if (first < 0) first = local;
                                    if (local > selectedSurfelEditorIndex) { next = local; break; }
                                }
                                selectedSurfelEditorIndex = next >= 0 ? next : first;
                            }
                        }
                        if (selectedSurfelEditorIndex >= 0) {
                            const auto id = static_cast<std::size_t>(selectedSurfelEditorIndex);
                            // Optimization snapshots contain one editable point cloud; the score buffers use global point indices.
                            const auto handle = firstPointCloudHandle(scene);
                            const auto found = handle ? buildProducts.pointCloudIndexById.find(*handle) : buildProducts.pointCloudIndexById.end();
                            if (found != buildProducts.pointCloudIndexById.end()) {
                                const auto& range = buildProducts.pointCloudRanges.at(found->second);
                                const auto primitive = range.firstPoint + id;
                                if (id < range.pointCount && primitive < transportScores.size()) {
                                    ImGui::Text("Selected surfel %d: %.6g", selectedSurfelEditorIndex, transportScores[primitive]);
                                    if (ImGui::BeginTable("Contribution breakdown", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY, ImVec2(0, 160))) {
                                        ImGui::TableSetupColumn("Camera"); ImGui::TableSetupColumn("Camera score"); ImGui::TableSetupColumn("Shadow score"); ImGui::TableHeadersRow();
                                        for (const auto& view : transportViews) {
                                            ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::TextUnformatted(view.name.c_str());
                                            ImGui::TableNextColumn(); ImGui::Text("%.6g", view.camera[primitive]);
                                            ImGui::TableNextColumn(); ImGui::Text("%.6g", view.shadow[primitive]);
                                        }
                                        ImGui::EndTable();
                                    }
                                }
                            }
                        }
                    }
                    if (viewImageMode == ViewImageMode::TargetLoss) {
                        ImGui::InputText("Target dataset / images", targetDatasetBuffer.data(), targetDatasetBuffer.size());
                        ImGui::Combo("Target color space", &targetColorSpace, "Linear\0sRGB\0");
                        if (ImGui::Button("Apply targets")) {
                            targetDatasetManual = true;
                            targetLossDirty = true;
                            renderRequested = true;
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Discover from loaded run")) {
                            targetDatasetManual = false;
                            targetLossDirty = true;
                            renderRequested = true;
                        }
                        if (ImGui::Button("Reproject target loss")) {
                            targetLossDirty = true;
                            renderRequested = true;
                        }
                        const char* cameraLabel = targetLossCamera >= 0 && targetLossCamera < int(targetLossViews.size())
                            ? targetLossViews[targetLossCamera].cameraName.c_str() : "All target images";
                        bool recolor = false;
                        if (ImGui::BeginCombo("Loss source camera", cameraLabel)) {
                            if (ImGui::Selectable("All target images", targetLossCamera == -1)) { targetLossCamera = -1; recolor = true; }
                            for (std::size_t i = 0; i < targetLossViews.size(); ++i) {
                                if (ImGui::Selectable(targetLossViews[i].cameraName.c_str(), targetLossCamera == int(i))) {
                                    targetLossCamera = int(i);
                                    recolor = true;
                                }
                            }
                            ImGui::EndCombo();
                        }
                        if (targetLossCamera == -1)
                            recolor |= ImGui::Combo("Across cameras", &targetLossAggregation, "Visibility-weighted mean\0Worst camera mean\0");
                        recolor |= ImGui::DragFloat("Loss color maximum", &targetLossColorMaximum, 0.0001f,
                            1.0e-8f, 1.0e8f, "%.6g", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
                        recolor |= ImGui::Checkbox("Logarithmic loss colors", &targetLossLogColors);
                        if (ImGui::Button("Fit loss colors to peak")) {
                            const auto& scores = targetLossCamera >= 0 && targetLossCamera < int(targetLossViews.size())
                                ? targetLossViews[targetLossCamera].mean : (targetLossAggregation == 0 ? targetLossMean : targetLossWorst);
                            targetLossColorMaximum = 1.0e-8f;
                            for (float value : scores) if (std::isfinite(value)) targetLossColorMaximum = std::max(targetLossColorMaximum, value);
                            recolor = true;
                        }
                        if (recolor) updateDisplayTexture();
                        double loss = 0, unassigned = 0;
                        std::size_t pixelsWithoutSurfels = 0, sourcePixels = 0, sourceViews = 0;
                        for (std::size_t i = 0; i < targetLossViews.size(); ++i) {
                            if (targetLossCamera >= 0 && int(i) != targetLossCamera) continue;
                            const auto& view = targetLossViews[i];
                            loss += view.imageMean;
                            unassigned += view.unassignedMean;
                            pixelsWithoutSurfels += view.unassignedPixels;
                            sourcePixels += view.pixelCount;
                            ++sourceViews;
                        }
                        if (sourceViews) {
                            ImGui::Text("Image half-MSE: %.6g   unassigned: %.6g", loss / sourceViews, unassigned / sourceViews);
                            ImGui::Text("Pixels without a surfel contribution: %zu / %zu", pixelsWithoutSurfels, sourcePixels);
                        }
                        ImGui::TextWrapped("%s", targetLossStatus.c_str());
                        ImGui::Text("Last projection: %.1f ms", targetLossComputeMs);
                        ImGui::TextWrapped("Linear RGB half-squared error, weighted by slab opacity and transmission onto each surfel. "
                            "Gray: no target observation. Black: background. Colors remain fixed while orbiting.");
                        ImGui::TextWrapped("Completely missing geometry has no surfel to color; its image loss is included in unassigned above. "
                            "This is image-error attribution, not a geometry gradient or a geometric regularizer.");
                        ImGui::TextWrapped("Projection refreshes after geometry or renderer changes. Exposure and display gamma do not affect the linear loss. "
                            "Targets must match scene camera names and resolution; alpha is ignored, as in training.");
                    }
                    if (viewImageMode == ViewImageMode::SurfelDensity) {
                        if (ImGui::SliderInt("Density grid (cells per axis)", &densityGridSize, 8, 256)) {
                            densityGridSize = std::clamp(densityGridSize, 8, 256);
                            debugDisplayBuffers.surfelDensityValid = false;
                            updateDisplayTexture();
                        }
                        bool recolorDensity = ImGui::SliderFloat(
                            "Density color maximum (surfels/cell)", &densityColorMaximum,
                            1.0f, 10000.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
                        recolorDensity |= ImGui::Checkbox("Logarithmic density colors", &densityLogScale);
                        const auto& density = debugDisplayBuffers.surfelDensity;
                        if (ImGui::Button("Fit density colors to current peak")) {
                            densityColorMaximum = static_cast<float>(std::max(density.peak, 1u));
                            recolorDensity = true;
                        }
                        if (recolorDensity) {
                            densityColorMaximum = std::max(densityColorMaximum, 1.0f);
                            updateDisplayTexture();
                        }
                        ImGui::Text("Non-emissive surfels: %zu   centers in view: %zu", density.total, density.inView);
                        ImGui::Text("Peak: %u surfels/cell   grid: %d x %d", density.peak, density.gridSize, density.gridSize);
                        ImGui::TextWrapped(
                            "Counts all projected centers, including hidden and transparent surfels; lights are excluded. "
                            "Black means an empty cell. Colors saturate at the selected maximum.");
                        ImGui::TextWrapped(
                            "Compare snapshots with the same camera, grid and color maximum. "
                            "The scale stays fixed until you change it. This is screen-space concentration, not world-space density.");
                    }
                    if (viewImageMode == ViewImageMode::PositionPrimitiveScore) {
                        if (!debugDisplayBuffers.positionPrimitiveMetadataAvailable) {
                            ImGui::TextWrapped(
                                "This PLY does not contain saved position densification statistics. "
                                "Run optimization again to generate a trustworthy visualization.");
                        } else {
                            const float savedThreshold =
                                debugDisplayBuffers.positionPrimitiveSplitThreshold;
                            ImGui::Text(
                                "Saved threshold: %.9g",
                                savedThreshold);
                            const float sliderMinimum = static_cast<float>(std::max(
                                static_cast<double>(std::numeric_limits<float>::min()),
                                static_cast<double>(std::min(savedThreshold, kPositionDefaultThreshold)) * 1.0e-3));
                            const float sliderMaximum = static_cast<float>(std::min(
                                static_cast<double>(std::numeric_limits<float>::max()),
                                static_cast<double>(std::max(savedThreshold, kPositionDefaultThreshold)) * 1.0e3));
                            if (ImGui::SliderFloat(
                                    "What-if position threshold",
                                    &positionWhatIfDisplayThreshold,
                                    sliderMinimum,
                                    sliderMaximum,
                                    "%.9g",
                                    ImGuiSliderFlags_Logarithmic)) {
                                positionWhatIfDisplayThreshold = std::clamp(
                                    positionWhatIfDisplayThreshold,
                                    sliderMinimum,
                                    sliderMaximum);
                                updateDisplayTexture();
                            }
                            if (ImGui::Button("Use saved##position-threshold")) {
                                positionWhatIfDisplayThreshold = savedThreshold;
                                updateDisplayTexture();
                            }
                            ImGui::SameLine();
                            if (ImGui::Button("Use config default##position-threshold")) {
                                positionWhatIfDisplayThreshold = kPositionDefaultThreshold;
                                updateDisplayTexture();
                            }
                            ImGui::TextDisabled(
                                "Below: saved G_i / selected threshold; magenta: at/above boundary");

                            const std::size_t splitCandidateCount = static_cast<std::size_t>(
                                std::count_if(
                                    debugDisplayBuffers.positionObservedPrimitiveScores.begin(),
                                    debugDisplayBuffers.positionObservedPrimitiveScores.end(),
                                    [&](float signal) {
                                        return signal >= positionWhatIfDisplayThreshold;
                                    }));
                            ImGui::Text(
                                "Observed: %zu   above selected: %zu   max G_i: %.9g",
                                debugDisplayBuffers.positionObservedPrimitiveCount,
                                splitCandidateCount,
                                debugDisplayBuffers.positionPrimitiveScoreMax);
                            ImGui::TextDisabled(
                                "Values are the optimizer's saved interval-average statistics");
                            ImGui::Text("Without interval samples: %zu (gray)",
                                debugDisplayBuffers.positionUnsampledPrimitiveCount);
                            ImGui::TextDisabled("Frontmost surfel per pixel; black: background");
                            ImGui::TextWrapped(
                                "New surfels and intervals with no accumulated position signal have no score yet. "
                                "Training can skip the first half of each densification interval.");
                        }
                    }
                    if (viewImageMode == ViewImageMode::DensificationOrigin) {
                        ImGui::TextDisabled(
                            "gray: initial/unknown  green: clone  blue: position split");
                    }
                    if (viewImageMode == ViewImageMode::PrimitiveAge) {
                        if (ImGui::SliderInt(
                                "Cold after iterations",
                                &primitiveAgeColdAfterIterations,
                                1,
                                100000,
                                "%d",
                                ImGuiSliderFlags_Logarithmic)) {
                            primitiveAgeColdAfterIterations = std::max(
                                primitiveAgeColdAfterIterations, 1);
                            updateDisplayTexture();
                        }
                        ImGui::TextDisabled(
                            "red: created/split now  blue: age >= selected range");
                    }
                    ImGui::TreePop();
                }
                ImGui::BeginDisabled(screenshotPixels.empty());
                if (ImGui::Button("Save screenshot (PNG)")) {
                    try {
                        const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count();
                        const auto directory = repositoryRoot / "output" / "screenshots";
                        std::filesystem::path path;
                        do {
                            path = directory / ("viewport_" + std::to_string(milliseconds) + "_" +
                                                std::to_string(screenshotSequence++) + ".png");
                        } while (std::filesystem::exists(path));
                        if (!Pale::Utils::savePNG(path, screenshotPixels,
                                                  displayedRenderWidth, displayedRenderHeight)) {
                            throw std::runtime_error("PNG writer could not save " + path.string());
                        }
                        screenshotStatus = "Saved " + path.string();
                        Pale::Log::PA_INFO("{}", screenshotStatus);
                    } catch (const std::exception& error) {
                        screenshotStatus = "Screenshot failed: " + std::string(error.what());
                    }
                }
                ImGui::EndDisabled();
                if (ImGui::IsItemHovered()) {
                    if (viewImageMode == ViewImageMode::RawLinear) {
                        ImGui::SetTooltip("Save the raw-linear preview as opaque 8-bit PNG. Values are clipped to [0,1].");
                    } else {
                        ImGui::SetTooltip("Save the displayed render at its native resolution with surface opacity.\n"
                                          "Transparent background; no grid, controls, or gizmos.");
                    }
                }
                if (!screenshotStatus.empty()) {
                    ImGui::TextWrapped("%s", screenshotStatus.c_str());
                }
                if (ImGui::TreeNode("Appearance")) {
                    const ScopedSectionIndent sectionIndent;
                    // Avoid BeginDisabled(false): it makes the color picker transparent
                    // in this ImGui version (see the walk-navigation scope above).
                    const bool appearanceDisabled = viewImageMode == ViewImageMode::RawLinear;
                    if (appearanceDisabled) {
                        ImGui::BeginDisabled();
                    }
                    if (ImGui::Checkbox("Force opacity = 1", &forceRenderOpacity)) {
                        updateDisplayTexture();
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Hide the background behind contributing rays. Does not change surfel opacity or primal RGB.");
                    }
                    if (ImGui::ColorEdit3("Background color", &viewportBackgroundColor.x)) {
                        updateDisplayTexture();
                    }
                    if (ImGui::DragFloat("Exposure", &exposure, 0.01f, 0.001f, 100.0f, "%.3f")) {
                        renderRequested = true;
                    }
                    if (ImGui::Checkbox("sRGB output", &useSrgbEncoding)) {
                        renderRequested = true;
                    }
                    if (!useSrgbEncoding &&
                        ImGui::DragFloat("Gamma", &gamma, 0.01f, 0.1f, 5.0f, "%.3f")) {
                        renderRequested = true;
                    }
                    if (appearanceDisabled) {
                        ImGui::EndDisabled();
                    }

                    ImGui::TreePop();
                }
            }

            }
            if (inspectorPage == viewer::ui::Inspector::View) {
                if(ImGui::CollapsingHeader("Interaction preview")) {
                    ImGui::Checkbox("Fast preview for emitter edits",&fastEmitterEditing);
                    ImGui::Checkbox("Fast preview for surfel edits",&fastSurfelEditing);
                    ImGui::TextWrapped("Lower preview quality while editing for faster feedback. Full quality returns when interaction stops.");
                }
            }
            if (inspectorPage == viewer::ui::Inspector::Camera) {
            if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
                const ScopedSectionIndent sectionIndent;
                int cameraSourceIndex = cameraSource == CameraSource::SceneXml ? 1 : 0;
                const char* cameraSources[] = {"Viewport camera", "scene.xml camera"};
                if (ImGui::Combo("Camera source", &cameraSourceIndex, cameraSources, 2)) {
                    cameraSource =
                        cameraSourceIndex == 1 && !buildProducts.cameraGPUs.empty()
                            ? CameraSource::SceneXml
                            : CameraSource::Viewport;
                    cameraDirty = true;
                    tracerDirty = true;
                    renderRequested = true;
                }

                if (cameraSource == CameraSource::SceneXml) {
                    ImGui::TextDisabled("Up/Down: next/previous scene camera (wraps)");
                    if (buildProducts.cameraGPUs.empty()) {
                        ImGui::Text("No scene.xml cameras loaded");
                    } else if (ImGui::BeginCombo(
                                   "Scene camera",
                                   buildProducts.cameraGPUs[static_cast<std::size_t>(selectedSceneCameraIndex)].name)) {
                        for (std::size_t cameraIndex = 0; cameraIndex < buildProducts.cameraGPUs.size(); ++cameraIndex) {
                            const bool selected =
                                static_cast<int>(cameraIndex) == selectedSceneCameraIndex;
                            const char* cameraName = buildProducts.cameraGPUs[cameraIndex].name;
                            if (ImGui::Selectable(cameraName, selected)) {
                                selectedSceneCameraIndex = static_cast<int>(cameraIndex);
                                cameraDirty = true;
                                tracerDirty = true;
                                renderRequested = true;
                            }
                            if (selected) {
                                ImGui::SetItemDefaultFocus();
                            }
                        }
                        ImGui::EndCombo();
                    }
                }

                if(cameraSource==CameraSource::SceneXml&&ImGui::Button("Fit camera image"))cameraImageView.reset();
                if (ImGui::Button("Reset view")) {
                    resetOrbitView();
                }
                if (cameraSource == CameraSource::Viewport) {
                    if (ImGui::DragFloat("FOV", &orbit.fovyDegrees, 0.25f, 5.0f, 160.0f, "%.1f deg")) {
                        cameraDirty = true;
                    }
                }
            }

            }
            if (inspectorPage == viewer::ui::Inspector::Files) {
            if (modelReadOnly()) ImGui::BeginDisabled();
            if (ImGui::CollapsingHeader("Scene files", ImGuiTreeNodeFlags_DefaultOpen)) {
                const ScopedSectionIndent sectionIndent;
                ImGui::TextWrapped("Scene: %s", currentScenePath.string().c_str());
                ImGui::TextWrapped("Point cloud: %s", currentPointCloudPath.string().c_str());
                ImGui::Text("Point cloud PLY path");
                ImGui::PushItemWidth(-1.0f);
                ImGui::InputText("##PlyPath", pointCloudPathBuffer.data(), pointCloudPathBuffer.size());
                ImGui::PopItemWidth();
                if (ImGui::Button("Load point cloud PLY")) {
                    replacePointCloud(std::filesystem::path(pointCloudPathBuffer.data()));
                }
                ImGui::SameLine();
                if (ImGui::Button("Browse##PointCloud")) {
                    plyBrowserOpen = true;
                }
                if (optimizationFolderAnchor.active()) {
                    ImGui::TextWrapped("Training folder: %s", optimizationFolderAnchor.folder.string().c_str());
                    if (optimizationFolderAnchor.pointsDirectory) {
                        ImGui::TextWrapped("Anchored run: %s",
                            optimizationFolderAnchor.pointsDirectory->parent_path().string().c_str());
                    }
                }
                ImGui::TextDisabled("Drop a training folder to anchor checkpoint navigation.");
                if (ImGui::Button("Reset to OptimizationOutput")) {
                    optimizationFolderAnchor.clear();
                    latestOptimizationMode = false;
                    latestOptimizationPointsDirectory.clear();
                    latestOptimizationSnapshots.clear();
                    latestOptimizationSnapshotIndex = 0u;
                    refreshLatestOptimizationPointCloud();
                }
                if (ImGui::Button(optimizationFolderAnchor.active() ? "Refresh latest checkpoint (R)" : "Load latest run PLY")) {
                    refreshLatestOptimizationPointCloud();
                }
                ImGui::SameLine();
                if (ImGui::Button("First")) {
                    jumpToOptimizationSnapshotBoundary(false);
                }
                ImGui::SameLine();
                if (ImGui::Button("Last")) {
                    jumpToOptimizationSnapshotBoundary(true);
                }
                if (!pointCloudStatus.empty()) {
                    ImGui::TextWrapped("%s", pointCloudStatus.c_str());
                }
                ImGui::Separator();
                ImGui::TextWrapped(
                    "Runtime mesh: %s",
                    currentRuntimeMeshPath.empty() ? "(none)" : currentRuntimeMeshPath.string().c_str());
                ImGui::Text("Mesh PLY path");
                ImGui::PushItemWidth(-1.0f);
                ImGui::InputText("##MeshPlyPath", meshPathBuffer.data(), meshPathBuffer.size());
                ImGui::PopItemWidth();
                if (ImGui::Button("Load mesh PLY")) {
                    loadRuntimeMesh(std::filesystem::path(meshPathBuffer.data()));
                }
                ImGui::SameLine();
                if (ImGui::Button("Browse##Mesh")) {
                    meshBrowserOpen = true;
                }
                if (!currentRuntimeMeshPath.empty()) {
                    ImGui::SameLine();
                    if (ImGui::Button("Remove mesh")) {
                        removeRuntimeMeshes();
                    }
                }
                if (!meshStatus.empty()) {
                    ImGui::TextWrapped("%s", meshStatus.c_str());
                }
                ImGui::Separator();

            }

            std::filesystem::path selectedPlyPath;
            if (drawPlyBrowser(
                    "Select Point Cloud PLY",
                    "PointCloudPlyFiles",
                    plyBrowserOpen,
                    plyBrowserDirectory,
                    selectedPlyPath)) {
                replacePointCloud(selectedPlyPath);
            }

            std::filesystem::path selectedMeshPath;
            if (drawPlyBrowser(
                    "Select Mesh PLY",
                    "MeshPlyFiles",
                    meshBrowserOpen,
                    meshBrowserDirectory,
                    selectedMeshPath)) {
                loadRuntimeMesh(selectedMeshPath);
            }
            if (modelReadOnly()) ImGui::EndDisabled();

            }
            if (inspectorPage == viewer::ui::Inspector::Selection) {
            ImGui::TextColored(kUiAccent, "%s", modelReadOnly() ? "INSPECT SELECTION" : "EDIT SNAPSHOT");
            if (sceneSelectionKind == viewer::SceneItemKind::Camera && !buildProducts.cameraGPUs.empty()) {
                const auto& camera=buildProducts.cameraGPUs[static_cast<std::size_t>(selectedSceneCameraIndex)];
                ImGui::TextWrapped("%s",camera.name); ImGui::SeparatorText("Calibration");
                ImGui::Text("Image  %u × %u",camera.width,camera.height);
                ImGui::Text("Focal length  %.3f  %.3f",camera.fx,camera.fy);
                ImGui::Text("Principal point  %.3f  %.3f",camera.cx,camera.cy);
                ImGui::TextWrapped("Dataset pose and intrinsics are fixed.");
                if(ImGui::Button("View through camera")) {cameraSource=CameraSource::SceneXml; cameraDirty=tracerDirty=renderRequested=true;}
            }
            if (sceneSelectionKind == viewer::SceneItemKind::Mesh || sceneSelectionKind == viewer::SceneItemKind::Material) {
                ImGui::TextWrapped("%s", selectedTreeEntity.c_str());
                for (auto [id, mesh, transform, tag] : scene->getAllEntitiesWith<Pale::MeshComponent, Pale::TransformComponent, Pale::TagComponent>().each()) {
                    if (tag.tag != selectedTreeEntity) continue;
                    ImGui::SeparatorText("Transform");
                    ImGui::Text("Location  %.3f  %.3f  %.3f", transform.translation.x, transform.translation.y, transform.translation.z);
                    ImGui::Text("Rotation  %.1f  %.1f  %.1f", transform.rotationEuler.x, transform.rotationEuler.y, transform.rotationEuler.z);
                    ImGui::Text("Scale     %.3f  %.3f  %.3f", transform.scale.x, transform.scale.y, transform.scale.z);
                    Pale::Entity entity(id, scene.get());
                    if (entity.hasComponent<Pale::MaterialComponent>()) {
                        auto material = assetManager.get<Pale::Material>(entity.getComponent<Pale::MaterialComponent>().materialID);
                        if (material) { ImGui::SeparatorText("Material"); ImGui::Text("Albedo  %.3f  %.3f  %.3f", material->baseColor.x, material->baseColor.y, material->baseColor.z); }
                    }
                }
                ImGui::TextDisabled("Source mesh properties");
            }
            if (openLightControls) ImGui::SetNextItemOpen(true);
            if ((sceneSelectionKind == viewer::SceneItemKind::Emitter || sceneSelectionKind == viewer::SceneItemKind::SurfelEmitter) && ImGui::CollapsingHeader("Emitters", ImGuiTreeNodeFlags_DefaultOpen)) {
                const ScopedSectionIndent sectionIndent;
                if (openLightControls) ImGui::SetScrollHereY(0.15f);

                ImGui::TextWrapped("%s",modelReadOnly() ? "Changing a property creates a private editable snapshot." : "Snapshot edits are independent of training.");
                if (sceneSelectionKind == viewer::SceneItemKind::Emitter) {
                ImGui::TextDisabled("AREA EMITTER");
                if (areaLights.empty()) {
                    ImGui::TextWrapped("No editable mesh area lights in the scene");
                } else {
                    const std::string selectedLightName = entityName(selectedLight);
                    ImGui::SetNextItemWidth(-1);
                    if (ImGui::BeginCombo("##SelectedLight", selectedLightName.c_str())) {
                        for (std::size_t lightIndex = 0; lightIndex < areaLights.size(); ++lightIndex) {
                            const bool selected = static_cast<int>(lightIndex) == selectedLightIndex;
                            const std::string label =
                                entityName(areaLights[lightIndex]) + "##Light" + std::to_string(lightIndex);
                            if (ImGui::Selectable(label.c_str(), selected)) {
                                selectedLightIndex = static_cast<int>(lightIndex);
                                selectedLight = areaLights[lightIndex];
                                selectedSurfelEditorIndex = -1;
                            }
                            if (selected) {
                                ImGui::SetItemDefaultFocus();
                            }
                        }
                        ImGui::EndCombo();
                    }


                    bool lightChanged = false;
                    auto& transform = selectedLight.getComponent<Pale::TransformComponent>();
                    if (viewer::ui::float3("Location", &transform.translation.x, 0.01f, -10000.0f, 10000.0f, "%.3f")) {
                        lightChanged = true;
                    }
                    glm::vec3 rotationEuler = transform.rotationEuler;
                    if (viewer::ui::float3("Rotation", &rotationEuler.x, 0.25f, -360.0f, 360.0f, "%.1f")) {
                        transform.setRotationEuler(rotationEuler);
                        lightChanged = true;
                    }
                    if (viewer::ui::float3("Scale", &transform.scale.x, 0.01f, 0.001f, 10000.0f, "%.3f")) {
                        transform.scale = glm::max(transform.scale, glm::vec3(0.001f));
                        lightChanged = true;
                    }

                    auto& areaLight = selectedLight.getComponent<Pale::AreaLightComponent>();
                    std::shared_ptr<Pale::Material> lightMaterial;
                    if (selectedLight.hasComponent<Pale::MaterialComponent>()) {
                        const auto& materialComponent = selectedLight.getComponent<Pale::MaterialComponent>();
                        lightMaterial = assetManager.get<Pale::Material>(materialComponent.materialID);
                    }

                    glm::vec3 emissionColor = lightMaterial ? lightMaterial->baseColor : areaLight.radiance;
                    float emissionPower = lightMaterial ? lightMaterial->power : areaLight.flux;
                    if (viewer::ui::color3("Color", &emissionColor.x)) {
                        areaLight.radiance = emissionColor;
                        if (lightMaterial) {
                            lightMaterial->baseColor = emissionColor;
                        }
                        lightChanged = true;
                    }
                    if (viewer::ui::scalar("Power", &emissionPower, 0.1f, 0.0f, 1000000.0f, "%.3f")) {
                        emissionPower = std::max(emissionPower, 0.0f);
                        areaLight.flux = emissionPower;
                        if (lightMaterial) {
                            lightMaterial->power = emissionPower;
                        }
                        lightChanged = true;
                    }
                    if (!lightMaterial) {
                        ImGui::TextWrapped("Selected light has no editable emissive material");
                    }


                    if (lightChanged) {
                        beginSnapshotEdit();rebuildSceneGpu();
                    }
                }

                }
                if (sceneSelectionKind == viewer::SceneItemKind::SurfelEmitter) {
                ImGui::SeparatorText("Surfel emitter");
                const std::optional<Pale::AssetHandle> pointCloudHandle = firstPointCloudHandle(scene);
                if (!pointCloudHandle) {
                    ImGui::TextWrapped("No point cloud loaded");
                } else {
                    const std::shared_ptr<Pale::PointAsset> pointCloudAsset =
                        assetAccessor.getPointCloud(*pointCloudHandle);
                    if (!pointCloudAsset || pointCloudAsset->points.empty()) {
                        ImGui::TextWrapped("Point cloud asset is not loaded");
                    } else {
                        Pale::PointGeometry& pointGeometry = pointCloudAsset->points.front();
                        const std::vector<std::size_t> emittingSurfels =
                            collectSurfelPowerIndices(pointGeometry, true);
                        const std::vector<std::size_t> zeroPowerSurfels =
                            collectSurfelPowerIndices(pointGeometry, false);

                        ImGui::Text(
                            "Surfels: %zu, emitting: %zu, zero power: %zu",
                            pointGeometry.powers.size(),
                            emittingSurfels.size(),
                            zeroPowerSurfels.size());

                        bool surfelChanged = false;
                        if (emittingSurfels.empty()) {
                            ImGui::TextWrapped("No non-zero surfel powers");
                            selectedSurfelLightIndex = 0;
                        } else {
                            selectedSurfelLightIndex = std::clamp(
                                selectedSurfelLightIndex,
                                0,
                                static_cast<int>(emittingSurfels.size() - 1u));
                            const std::size_t selectedPowerIndex =
                                emittingSurfels[static_cast<std::size_t>(selectedSurfelLightIndex)];
                            const std::string selectedSurfelLabel =
                                "surfel " + std::to_string(selectedPowerIndex);
                            if (ImGui::BeginCombo("Selected surfel light", selectedSurfelLabel.c_str())) {
                                for (std::size_t listIndex = 0; listIndex < emittingSurfels.size(); ++listIndex) {
                                    const std::size_t surfelIndex = emittingSurfels[listIndex];
                                    const bool selected =
                                        static_cast<int>(listIndex) == selectedSurfelLightIndex;
	                                    const std::string label =
	                                        "surfel " + std::to_string(surfelIndex) +
	                                        " power " + std::to_string(pointGeometry.powers[surfelIndex]);
	                                    if (ImGui::Selectable(label.c_str(), selected)) {
	                                        selectedSurfelLightIndex = static_cast<int>(listIndex);
	                                        selectedSurfelEditorIndex = static_cast<int>(surfelIndex);
	                                        surfelEditorStatus = "Selected surfel light " + std::to_string(selectedSurfelEditorIndex);
	                                    }
                                    if (selected) {
                                        ImGui::SetItemDefaultFocus();
                                    }
                                }
                                ImGui::EndCombo();
                            }


	                            if (selectedPowerIndex < pointGeometry.positions.size()) {
	                                glm::vec3& position = pointGeometry.positions[selectedPowerIndex];
	                                if (ImGui::DragFloat3("Surfel position", &position.x, 0.01f, -10000.0f, 10000.0f, "%.3f")) {
	                                    surfelChanged = true;
	                                    selectedSurfelEditorIndex = static_cast<int>(selectedPowerIndex);
	                                    surfelLightStatus =
	                                        "Moved surfel " + std::to_string(selectedPowerIndex);
                                }
                            }
                            if (selectedPowerIndex < pointGeometry.albedos.size()) {
                                const glm::vec3& albedo = pointGeometry.albedos[selectedPowerIndex];
                                ImGui::Text(
                                    "Albedo: %.3f %.3f %.3f",
                                    albedo.x,
                                    albedo.y,
                                    albedo.z);
                            }

                            float selectedPower = pointGeometry.powers[selectedPowerIndex];
	                            if (ImGui::DragFloat(
	                                    "Selected power",
                                    &selectedPower,
                                    0.1f,
                                    0.0f,
                                    1000000.0f,
                                    "%.3f")) {
	                                pointGeometry.powers[selectedPowerIndex] = std::max(selectedPower, 0.0f);
	                                surfelChanged = true;
	                                selectedSurfelEditorIndex = static_cast<int>(selectedPowerIndex);
	                                surfelLightStatus =
	                                    "Updated surfel " + std::to_string(selectedPowerIndex);
	                            }
	                            if (ImGui::Button("Remove surfel light")) {
	                                pointGeometry.powers[selectedPowerIndex] = 0.0f;
	                                surfelChanged = true;
	                                selectedSurfelEditorIndex = static_cast<int>(selectedPowerIndex);
	                                surfelLightStatus =
	                                    "Removed surfel " + std::to_string(selectedPowerIndex) + " from lights";
                            }


                        }

                        ImGui::Separator();

                        if (pointGeometry.powers.empty()) {
                            ImGui::TextWrapped("Point cloud has no power attribute values");
                        } else if (zeroPowerSurfels.empty()) {
                            ImGui::TextWrapped("No zero-power surfels available to add");
                        } else {
                            candidateZeroPowerSurfelIndex = std::clamp(
                                candidateZeroPowerSurfelIndex,
                                0,
                                static_cast<int>(pointGeometry.powers.size() - 1u));
                            if (ImGui::Button("Find zero-power surfel")) {
                                auto iterator = std::lower_bound(
                                    zeroPowerSurfels.begin(),
                                    zeroPowerSurfels.end(),
                                    static_cast<std::size_t>(candidateZeroPowerSurfelIndex));
                                if (iterator == zeroPowerSurfels.end()) {
                                    iterator = zeroPowerSurfels.begin();
                                }
                                candidateZeroPowerSurfelIndex = static_cast<int>(*iterator);
                            }
                            ImGui::InputInt("Zero-power surfel index", &candidateZeroPowerSurfelIndex);
                            candidateZeroPowerSurfelIndex = std::clamp(
                                candidateZeroPowerSurfelIndex,
                                0,
                                static_cast<int>(pointGeometry.powers.size() - 1u));

                            const std::size_t candidateIndex =
                                static_cast<std::size_t>(candidateZeroPowerSurfelIndex);
                            if (candidateIndex < pointGeometry.positions.size()) {
                                const glm::vec3& position = pointGeometry.positions[candidateIndex];
                                ImGui::Text(
                                    "Candidate: %.3f %.3f %.3f",
                                    position.x,
                                    position.y,
                                    position.z);
                            }
                            ImGui::DragFloat(
                                "New surfel power",
                                &candidateSurfelPower,
                                0.1f,
                                0.001f,
                                1000000.0f,
                                "%.3f");
                            candidateSurfelPower = std::max(candidateSurfelPower, 0.001f);

                            if (pointGeometry.powers[candidateIndex] > 0.0f) {
                                ImGui::TextWrapped("Selected surfel already has non-zero power");
	                            } else if (ImGui::Button("Add surfel light")) {
	                                pointGeometry.powers[candidateIndex] = candidateSurfelPower;
	                                selectedSurfelLightIndex = static_cast<int>(emittingSurfels.size());
	                                selectedSurfelEditorIndex = static_cast<int>(candidateIndex);
	                                surfelChanged = true;
                                surfelLightStatus =
                                    "Added surfel " + std::to_string(candidateIndex) + " as light";
                            }
                        }

                        if (!surfelLightStatus.empty()) {
                            ImGui::TextWrapped("%s", surfelLightStatus.c_str());
                        }

                        if (surfelChanged) {
                            beginSnapshotEdit();rebuildSceneGpu();
                        }
                    }
                }
                }
            }

            openLightControls = false;
            if (openSurfelControls) ImGui::SetNextItemOpen(true);
            if ((sceneSelectionKind == viewer::SceneItemKind::Geometry || sceneSelectionKind == viewer::SceneItemKind::SurfelEmitter) && ImGui::CollapsingHeader("Surfel properties", ImGuiTreeNodeFlags_DefaultOpen)) {
                const ScopedSectionIndent sectionIndent;
                const std::optional<Pale::AssetHandle> pointCloudHandle = firstPointCloudHandle(scene);
                const std::shared_ptr<Pale::PointAsset> pointCloudAsset =
                    pointCloudHandle ? assetAccessor.getPointCloud(*pointCloudHandle) : nullptr;

                if (openSurfelControls) ImGui::SetScrollHereY(0.15f);
                ImGui::TextWrapped("Select a surfel in the viewport to inspect geometry and material.");
                if(modelReadOnly())ImGui::TextWrapped("Changing a property creates a private editable snapshot; training continues.");

                if (!pointCloudAsset || countSurfels(*pointCloudAsset) == 0u) {
                    ImGui::TextWrapped("No editable surfels are loaded");
                    selectedSurfelEditorIndex = -1;
                } else {
                    std::size_t surfelCount = countSurfels(*pointCloudAsset);

                    if (ImGui::Button("Add surfel")) {
                        beginSnapshotEdit();
                        Pale::PointGeometry& pointGeometry = pointCloudAsset->points.back();
                        const std::size_t previousBlockCount = pointGeometry.positions.size();
                        const float defaultScale = std::clamp(bounds.radius * 0.12f, 0.05f, 1.0f);
                        const glm::quat cameraFacingRotation =
                            extractRotationQuaternion(glm::inverse(orbit.viewMatrix()));

                        pointGeometry.positions.push_back(orbit.target);
                        pointGeometry.quat.push_back(cameraFacingRotation);
                        pointGeometry.scales.emplace_back(defaultScale, defaultScale);
                        pointGeometry.albedos.emplace_back(0.8f, 0.8f, 0.8f);
                        pointGeometry.opacities.push_back(0.9f);
                        pointGeometry.betas.push_back(-1.0f);
                        pointGeometry.shapes.push_back(0.0f);
                        pointGeometry.powers.push_back(0.0f);

                        const auto appendOptionalMetadata =
                            [previousBlockCount]<typename T>(std::vector<T>& values, T defaultValue) {
                                if (!values.empty()) {
                                    values.resize(previousBlockCount, defaultValue);
                                    values.push_back(defaultValue);
                                }
                            };
                        appendOptionalMetadata(pointGeometry.densificationOrigins, std::uint8_t{0u});
                        appendOptionalMetadata(pointGeometry.primitiveAges, std::uint32_t{0u});
                        appendOptionalMetadata(pointGeometry.densificationPositionSignals, 0.0f);
                        appendOptionalMetadata(
                            pointGeometry.densificationPositionSampleCounts,
                            std::uint32_t{0u});
                        appendOptionalMetadata(pointGeometry.densificationPositionThresholds, 0.0f);

                        selectedSurfelEditorIndex = static_cast<int>(surfelCount);
                        ++surfelCount;
                        surfelEditorStatus =
                            "Added surfel " + std::to_string(selectedSurfelEditorIndex) +
                            " at the viewport focus; save it to keep it for the next run";
                        rebuildSceneGpu();
                    }
                    ImGui::SameLine();
                    ImGui::TextDisabled("New surfels face the current viewport");


                    selectedSurfelEditorIndex = std::clamp(
                        selectedSurfelEditorIndex,
                        -1,
                        static_cast<int>(surfelCount - 1u));

                    ImGui::SetNextItemWidth(-1.0f);
                    if (ImGui::InputInt("Surfel index", &selectedSurfelEditorIndex)) {
                        selectedSurfelEditorIndex = std::clamp(
                            selectedSurfelEditorIndex,
                            -1,
                            static_cast<int>(surfelCount - 1u));
                    }
                    ImGui::Text("Valid range: -1 (none), 0 - %zu", surfelCount - 1u);

                    std::size_t localSurfelIndex = selectedSurfelEditorIndex >= 0 ? static_cast<std::size_t>(selectedSurfelEditorIndex) : surfelCount;
                    Pale::PointGeometry* selectedPointGeometry = nullptr;
                    for (Pale::PointGeometry& pointGeometry : pointCloudAsset->points) {
                        if (localSurfelIndex < pointGeometry.positions.size()) {
                            selectedPointGeometry = &pointGeometry;
                            break;
                        }
                        localSurfelIndex -= pointGeometry.positions.size();
                    }

                    bool surfelChanged = false;
                    if (!selectedPointGeometry) {
                        ImGui::TextWrapped(selectedSurfelEditorIndex < 0 ? "No surfel selected" : "The selected surfel could not be resolved");
                    } else {
                        Pale::PointGeometry& pointGeometry = *selectedPointGeometry;


                        if (localSurfelIndex < pointGeometry.positions.size()) {
                            glm::vec3& position = pointGeometry.positions[localSurfelIndex];
                            surfelChanged |= ImGui::DragFloat3(
                                "Position",
                                &position.x,
                                0.001f,
                                -10000.0f,
                                10000.0f,
                                "%.6f");
                        }

                        if (localSurfelIndex < pointGeometry.quat.size()) {
                            glm::quat& quaternion = pointGeometry.quat[localSurfelIndex];
                            if (ImGui::DragFloat4(
                                    "Rotation quaternion (x, y, z, w)",
                                    &quaternion.x,
                                    0.001f,
                                    -1.0f,
                                    1.0f,
                                    "%.6f")) {
                                quaternion = normalizeQuaternionOrIdentity(quaternion);
                                surfelChanged = true;
                            }
                        }

                        if (localSurfelIndex < pointGeometry.scales.size()) {
                            glm::vec2& scale = pointGeometry.scales[localSurfelIndex];
                            if (ImGui::DragFloat2(
                                    "Scale (u, v)",
                                    &scale.x,
                                    0.001f,
                                    0.000001f,
                                    10000.0f,
                                    "%.6f")) {
                                scale = glm::max(scale, glm::vec2(0.000001f));
                                surfelChanged = true;
                            }
                        }

                        if (localSurfelIndex < pointGeometry.albedos.size()) {
                            glm::vec3& albedo = pointGeometry.albedos[localSurfelIndex];
                            if (ImGui::ColorEdit3("Albedo", &albedo.x, ImGuiColorEditFlags_Float)) {
                                albedo = glm::clamp(albedo, glm::vec3(0.0f), glm::vec3(1.0f));
                                surfelChanged = true;
                            }
                        }

                        auto drawAdditionalSurfelProperties = [&]<typename PointGeometryType>(
                                                                  PointGeometryType& editablePointGeometry) {
                            if constexpr (requires { editablePointGeometry.opacities; }) {
                                if (localSurfelIndex < editablePointGeometry.opacities.size()) {
                                    float& opacity = editablePointGeometry.opacities[localSurfelIndex];
                                    if (ImGui::DragFloat("Opacity", &opacity, 0.001f, 0.0f, 1.0f, "%.6f")) {
                                        opacity = std::clamp(opacity, 0.0f, 1.0f);
                                        surfelChanged = true;
                                    }
                                }
                            } else if constexpr (requires { editablePointGeometry.opacity; }) {
                                if (localSurfelIndex < editablePointGeometry.opacity.size()) {
                                    float& opacity = editablePointGeometry.opacity[localSurfelIndex];
                                    if (ImGui::DragFloat("Opacity", &opacity, 0.001f, 0.0f, 1.0f, "%.6f")) {
                                        opacity = std::clamp(opacity, 0.0f, 1.0f);
                                        surfelChanged = true;
                                    }
                                }
                            }

                            if constexpr (requires { editablePointGeometry.betas; }) {
                                if (localSurfelIndex < editablePointGeometry.betas.size()) {
                                    float& beta = editablePointGeometry.betas[localSurfelIndex];
                                    surfelChanged |= ImGui::DragFloat(
                                        "Beta",
                                        &beta,
                                        0.001f,
                                        -100.0f,
                                        100.0f,
                                        "%.6f");
                                }
                            } else if constexpr (requires { editablePointGeometry.beta; }) {
                                if (localSurfelIndex < editablePointGeometry.beta.size()) {
                                    float& beta = editablePointGeometry.beta[localSurfelIndex];
                                    surfelChanged |= ImGui::DragFloat(
                                        "Beta",
                                        &beta,
                                        0.001f,
                                        -100.0f,
                                        100.0f,
                                        "%.6f");
                                }
                            }
                        };
                        drawAdditionalSurfelProperties(pointGeometry);

                        if (localSurfelIndex < pointGeometry.powers.size()) {
                            float& power = pointGeometry.powers[localSurfelIndex];
                            if (ImGui::DragFloat("Power", &power, 0.01f, 0.0f, 1000000.0f, "%.6f")) {
                                power = std::max(power, 0.0f);
                                surfelChanged = true;
                            }
                        }

                        if (surfelChanged) {
                            beginSnapshotEdit();
                            surfelEditorStatus =
                                "Updated surfel " + std::to_string(selectedSurfelEditorIndex);
                            rebuildSceneGpu();
                        }

                    }

                    if (!surfelEditorStatus.empty()) {
                        ImGui::TextWrapped("%s", surfelEditorStatus.c_str());
                    }
                }
            }

            ImGui::Spacing();
            openSurfelControls = false;
            }
            if (inspectorPage == viewer::ui::Inspector::Advanced) {
            ImGui::SeparatorText("Advanced");
            if (ImGui::CollapsingHeader("Renderer debug")) {
                const ScopedSectionIndent sectionIndent;
                int integrator = settings.integratorKind == Pale::IntegratorKind::photonMapping
                                     ? 2
                                     : settings.integratorKind == Pale::IntegratorKind::lightTracing
                                           ? 1
                                           : 0;
                const char* integrators[] = {"Cylinder ray", "Light tracing", "Photon mapping"};
                if (ImGui::Combo("Integrator", &integrator, integrators, 3)) {
                    settings.integratorKind =
                        integrator == 2 ? Pale::IntegratorKind::photonMapping
                                        : integrator == 1 ? Pale::IntegratorKind::lightTracing
                                                          : Pale::IntegratorKind::lightTracingCylinderRay;
                    tracerDirty = true;
                    renderRequested = true;
                }

                if (ImGui::Checkbox("Shared surface experiment", &settings.sharedHeightEnabled)) {
                    if (settings.sharedHeightEnabled) {
                        settings.integratorKind = Pale::IntegratorKind::photonMapping;
                        settings.cameraGatherKernelKind = Pale::CameraGatherKernelKind::CameraGatherKernel2;
                        viewImageMode = ViewImageMode::Rendered;
                    }
                    tracerDirty = true;
                    renderRequested = true;
                }
                if (settings.sharedHeightEnabled) {
                    bool changed = false;
                    changed |= ImGui::Checkbox("Two overlapping slabs", &settings.sharedHeightTwoSlabs);
                    if (settings.sharedHeightTwoSlabs) {
                        changed |= ImGui::InputInt("Slab A first surfel", &settings.sharedHeightSlabStartA);
                        changed |= ImGui::SliderInt("Slab A surfel count", &settings.sharedHeightSlabCountA, 1, 8);
                        changed |= ImGui::InputInt("Slab B first surfel", &settings.sharedHeightSlabStartB);
                        changed |= ImGui::SliderInt("Slab B surfel count", &settings.sharedHeightSlabCountB, 1, 8);
                        changed |= ImGui::DragFloat("Slab half width", &settings.sharedHeightSlabHalfWidth, 0.01f, 0.001f, 100.0f);
                        changed |= ImGui::DragFloat("Slab half depth h", &settings.sharedHeightSlabHalfDepth, 0.005f, 0.001f, 100.0f);
                        changed |= ImGui::DragFloat("Slab A lateral shift", &settings.sharedHeightSlabLateralShiftA, 0.01f);
                        changed |= ImGui::DragFloat("Slab B lateral shift", &settings.sharedHeightSlabLateralShiftB, 0.01f);
                        changed |= ImGui::DragFloat("Slab A depth shift", &settings.sharedHeightSlabDepthShiftA, 0.005f);
                        changed |= ImGui::DragFloat("Slab B depth shift", &settings.sharedHeightSlabDepthShiftB, 0.005f);
                        changed |= ImGui::DragFloat("Slab coverage", &settings.sharedHeightSlabCoverage, 0.1f, 0.01f, 100.0f);
                        ImGui::TextWrapped("Ranges are fixed GPU indices, up to eight surfels per slab. Invalid ranges or emissive members render blank. Offsets move support boundaries without changing membership. Excessive offsets can remove coverage.");
                    } else {
                        changed |= ImGui::InputInt("Height member A (-1: auto)", &settings.sharedHeightMemberA);
                        changed |= ImGui::InputInt("Height member B (-1: auto)", &settings.sharedHeightMemberB);
                    }
                    changed |= ImGui::Combo("Height shading", &settings.sharedHeightShading,
                        "Point lights (unshadowed)\0Albedo\0Surface normals\0Slab weights (A red, B blue)\0");
                    if (changed) {
                        settings.sharedHeightMemberA = std::max(-1, settings.sharedHeightMemberA);
                        settings.sharedHeightMemberB = std::max(-1, settings.sharedHeightMemberB);
                        renderRequested = true;
                    }
                    if (!settings.sharedHeightTwoSlabs) {
                        int members[2]{settings.sharedHeightMemberA, settings.sharedHeightMemberB};
                        for (int m = 0; m < 2; ++m) {
                            if (members[m] < 0) {
                                for (std::size_t i = 0; i < renderBuildProducts.points.size(); ++i) {
                                    if (int(i) != members[1 - m] && !renderBuildProducts.points[i].isEmissive()) {
                                        members[m] = int(i); break;
                                    }
                                }
                            }
                        }
                        ImGui::Text("GPU surfel indices: %d, %d", members[0], members[1]);
                        bool validPair = members[0] >= 0 && members[1] >= 0 && members[0] != members[1];
                        for (int member : members)
                            validPair = validPair && member >= 0 &&
                                std::size_t(member) < renderBuildProducts.points.size() &&
                                !renderBuildProducts.points[member].isEmissive();
                        if (!validPair)
                            ImGui::TextWrapped("Select two distinct, non-emissive surfels. Invalid selections render blank.");
                    }
                    if (renderBuildProducts.instances.size() != 1 ||
                        renderBuildProducts.instances.front().geometryType != Pale::GeometryType::PointCloud)
                        ImGui::TextWrapped("This experiment requires one point-cloud instance and no meshes; this scene will render blank.");
                    ImGui::TextWrapped("Forward experiment: cubic footprint blend. Two-slab mode blends geometry before intersection; h controls smooth depth support. Beta is unused for joined members. No shadows, indirect lighting, or adjoint. Use Rendered and Height shading to inspect it. Other surfels remain separate.");
                    ImGui::Separator();
                }
                int cameraGatherKernelIndex =
                    settings.cameraGatherKernelKind == Pale::CameraGatherKernelKind::CameraGatherKernel2 ? 1 : 0;

                if (ImGui::RadioButton("launchCameraGatherKernel", &cameraGatherKernelIndex, 0) ||
                    ImGui::RadioButton("launchCameraGatherKernel2", &cameraGatherKernelIndex, 1)) {
                    settings.cameraGatherKernelKind =
                        cameraGatherKernelIndex == 1
                            ? Pale::CameraGatherKernelKind::CameraGatherKernel2
                            : Pale::CameraGatherKernelKind::CameraGatherKernel;

                    renderRequested = true;
                    }

                ImGui::TextDisabled("Debug maps are computed only for the selected Display view");
                if (isRegularizerGradientView(viewImageMode)) {
                    ImGui::Text(
                        "Selected regularizer gradient map: %.3f ms (unit loss weight)",
                        lastRegularizerGradientMapMs);
                }


                if (ImGui::CollapsingHeader("Adjoint profiling")) {
                    const ScopedSectionIndent sectionIndent;
                    if (ImGui::Button("Run adjoint once")) {
                        runAdjointNextRender = true;
                        tracerDirty = true;
                        renderRequested = true;
                    }
                    ImGui::SameLine();
                    if (ImGui::Checkbox("Every render", &runAdjointEveryRender)) {
                        tracerDirty = true;
                        renderRequested = true;
                    }

                    if (ImGui::SliderInt("Adjoint SPP", &viewerAdjointSamplesPerPixel, 1, 256,
                                         "%d", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp)) {
                        viewerAdjointSamplesPerPixel = std::clamp(viewerAdjointSamplesPerPixel, 1, 256);
                        tracerDirty = true;
                        if (runAdjointEveryRender || viewImageMode == ViewImageMode::AdjointDerivative) {
                            renderRequested = true;
                        }
                    }
                    if (ImGui::DragInt("Adjoint bounces", &viewerAdjointBounces, 0.1f, 1, 8)) {
                        viewerAdjointBounces = std::clamp(viewerAdjointBounces, 1, 8);
                        tracerDirty = true;
                        if (runAdjointEveryRender || viewImageMode == ViewImageMode::AdjointDerivative) {
                            renderRequested = true;
                        }
                    }
                    if (ImGui::Checkbox("Adjoint direct light", &viewerAdjointDirectLight) &&
                        (runAdjointEveryRender || viewImageMode == ViewImageMode::AdjointDerivative)) {
                        tracerDirty = true;
                        renderRequested = true;
                    }
                    ImGui::Text("Last adjoint: %.3f ms", lastViewerAdjointMs);
                    ImGui::Text("Adjoint RGB objective: %.6g", static_cast<double>(lastViewerAdjointLoss));
                    ImGui::TextWrapped("%s", viewerAdjointStatus.c_str());
                }

                if (ImGui::CollapsingHeader("Point BVH")) {
                    const ScopedSectionIndent sectionIndent;
                    bool pointBvhBuildChanged = false;

                    int pointBvhMaxLeafPoints = static_cast<int>(buildOptions.bvhMaxLeafPoints);
                    if (ImGui::SliderInt("Leaf size", &pointBvhMaxLeafPoints, 1, 32)) {
                        buildOptions.bvhMaxLeafPoints =
                            static_cast<uint32_t>(std::clamp(pointBvhMaxLeafPoints, 1, 32));
                        pointBvhBuildChanged = true;
                    }

                    pointBvhBuildChanged |= ImGui::Checkbox(
                        "Binned SAH",
                        &buildOptions.pointBvhUseBinnedSah);

                    if (pointBvhBuildChanged) {
                        rebuildSceneGpu();
                        renderRequested = true;
                    }
                }

                if (ImGui::CollapsingHeader("Surfel traversal")) {
                    const ScopedSectionIndent sectionIndent;
                    int slabDepthMode = static_cast<int>(settings.rendererDebugLocalLayerDepthMode);
                    const char* slabDepthModes[] = {
                        "Along surface normal", "Symmetric along ray"
                    };
                    if (ImGui::Combo("Slab distance", &slabDepthMode, slabDepthModes, 2)) {
                        settings.rendererDebugLocalLayerDepthMode =
                            static_cast<Pale::LocalLayerDepthMode>(slabDepthMode);
                        tracerDirty = true;
                        renderRequested = true;
                    }
                    const bool symmetricSlabDepth = settings.rendererDebugLocalLayerDepthMode ==
                        Pale::LocalLayerDepthMode::SymmetricRayDepth;
                    ImGui::TextWrapped(symmetricSlabDepth
                        ? "Fixed distance before and after the first hit along the ray."
                        : "Distance measured along the first hit's surface normal; the ray interval widens at grazing angles.");
                    ImGui::TextWrapped("Normal alignment and member limits apply in both modes.");
                    float localLayerDepthEpsilon = settings.rendererDebugLocalLayerDepthEpsilon;
                    if (ImGui::DragFloat(
                            symmetricSlabDepth
                                ? "Ray depth half-width##LocalLayerDepthEpsilon"
                                : "Normal distance tolerance##LocalLayerDepthEpsilon",
                            &localLayerDepthEpsilon,
                            0.0005f,
                            0.0f,
                            10.0f,
                            "%.6f")) {
                        settings.rendererDebugLocalLayerDepthEpsilon =
                            std::max(localLayerDepthEpsilon, 0.0f);
                        renderRequested = true;
                    }

                    float localLayerNormalCosineThreshold =
                        settings.rendererDebugLocalLayerNormalCosineThreshold;
                    if (ImGui::SliderFloat(
                            "Local layer normal cosine",
                            &localLayerNormalCosineThreshold,
                            -1.0f,
                            1.0f,
                            "%.3f")) {
                        settings.rendererDebugLocalLayerNormalCosineThreshold =
                            std::clamp(localLayerNormalCosineThreshold, -1.0f, 1.0f);
                        renderRequested = true;
                    }

                    int maxSplatEventsPerRay =
                        static_cast<int>(settings.rendererDebugMaxSplatEventsPerRay);
                    if (ImGui::SliderInt(
                            "Max splat events",
                            &maxSplatEventsPerRay,
                            1,
                            static_cast<int>(Pale::kMaxSplatEventsPerRay))) {
                        settings.rendererDebugMaxSplatEventsPerRay =
                            static_cast<uint32_t>(std::clamp(
                                maxSplatEventsPerRay,
                                1,
                                static_cast<int>(Pale::kMaxSplatEventsPerRay)));
                        renderRequested = true;
                    }

                    int maxLocalSurfelHits =
                        static_cast<int>(settings.rendererDebugMaxLocalSurfelHits);
                    if (ImGui::SliderInt(
                            "Max local surfel hits",
                            &maxLocalSurfelHits,
                            1,
                            static_cast<int>(Pale::kMaxLocalSurfelHits))) {
                        settings.rendererDebugMaxLocalSurfelHits =
                            static_cast<uint32_t>(std::clamp(
                                maxLocalSurfelHits,
                                1,
                                static_cast<int>(Pale::kMaxLocalSurfelHits)));
                        renderRequested = true;
                    }

                    int pointHitBatchSize =
                        static_cast<int>(settings.rendererDebugPointHitBatchSize);
                    if (ImGui::SliderInt(
                            "Point hit batch size",
                            &pointHitBatchSize,
                            1,
                            static_cast<int>(Pale::kMaxPointHitBatch))) {
                        settings.rendererDebugPointHitBatchSize =
                            static_cast<uint32_t>(std::clamp(
                                pointHitBatchSize,
                                1,
                                static_cast<int>(Pale::kMaxPointHitBatch)));
                        renderRequested = true;
                    }

                    if (ImGui::Checkbox(
                            "Point hit batch look-ahead",
                            &settings.rendererDebugPointHitBatchLookahead)) {
                        renderRequested = true;
                    }

                    if (ImGui::Checkbox(
                            "Share layer direct light",
                            &settings.rendererDebugShareLocalLayerDirectLighting)) {
                        renderRequested = true;
                    }


                }

                if (ImGui::Checkbox("Show point albedo", &settings.pointGeometryDebugShowAlbedo)) {
                    renderRequested = true;
                }
            }

            if (ImGui::CollapsingHeader("Scene statistics & profiling")) {
                const ScopedSectionIndent sectionIndent;
                ImGui::Text("Surfels: %u", sceneGpu.pointCount);
                ImGui::Text("Triangles: %u", sceneGpu.triangleCount);
                ImGui::Text("BLAS nodes: %u", sceneGpu.blasNodeCount);
                ImGui::Text("TLAS nodes: %u", sceneGpu.tlasNodeCount);
                ImGui::Checkbox("Show profiling window", &showProfilingWindow);
                ImGui::Text("Camera: %.3f %.3f %.3f",
                            orbit.position().x,
                            orbit.position().y,
                            orbit.position().z);
            }

            if (ImGui::CollapsingHeader("Navigation help")) {
                const ScopedSectionIndent sectionIndent;
                ImGui::TextDisabled("Space: reset orbit | Backslash: walk navigation");
                if (walkNavigationActive) {
                    ImGui::TextWrapped("Walk: mouse look, WASD move, Q/E down/up; Shift fast, Alt slow; wheel changes speed.");
                    ImGui::TextWrapped("Enter / left click / Backslash: keep view. Escape / right click: cancel.");
                    ImGui::Text("Walk speed: %.2f units/s", walkSpeed);
                }
                ImGui::TextWrapped("1-9, 0: select display mode. +/-: cycle display modes.");
                ImGui::TextWrapped("Drag to orbit; right or middle drag to pan; scroll to zoom.");
            }

            }
            ImGui::PopItemWidth();
            {
                auto& cached=inspectorBodies[static_cast<int>(inspectorPage)];
                cached.position=ImGui::GetWindowPos();cached.size=ImGui::GetWindowSize();
                cached.scroll=ImGui::GetScrollY();
                cached.contentHeight=std::max(1.f,ImGui::GetCurrentWindow()->DC.CursorMaxPos.y-ImGui::GetCurrentWindow()->DC.CursorStartPos.y);
                cached.key=inspectorContextKey(inspectorPage);cached.geometry.capture(*ImGui::GetWindowDrawList(),0,cached.position,cached.size);cached.valid=true;
                lastDrawnInspector=inspectorPage;
                auto* properties=ImGui::FindWindowByName("Properties");auto& context=*ImGui::GetCurrentContext();
                holdPreviewForInspector=inspectorInput.pending()||inspectorHandoffFrames>0||
                    (context.ActiveIdWindow&&properties&&ImGui::IsWindowWithinBeginStackOf(context.ActiveIdWindow,properties)&&
                     (ImGui::GetIO().WantTextInput||ImGui::IsMouseDown(ImGuiMouseButton_Left)));
                for(const auto& popup:context.OpenPopupStack)
                    if(popup.Window&&properties&&ImGui::IsWindowWithinBeginStackOf(popup.Window,properties))holdPreviewForInspector=true;
                if(!context.OpenPopupStack.empty()&&context.NavWindow&&properties&&
                   ImGui::IsWindowWithinBeginStackOf(context.NavWindow,properties))holdPreviewForInspector=true;
                if(ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows)&&ImGui::IsMouseDown(ImGuiMouseButton_Left))holdPreviewForInspector=true;
                if(inspectorHandoffFrames>0)--inspectorHandoffFrames;
            }
            ImGui::EndChild();
            ImGui::End();
            openLightControls = openSurfelControls = false;
            if (training.datasetChanged) {
                training.datasetChanged=false;
                training.message="Dataset selected for the next reconstruction";
            }
            if (launchTrainingRequested && training.process.active()) {
                launchTrainingRequested=false;
            }
            if (launchTrainingRequested) {
                launchTrainingRequested = false;
                try {
                    if (!modelReadOnly() && !saveWorkingSnapshot()) training.message = "Could not preserve the editable scene";
                    else { training.launch(training.useWorkingSnapshot ? workingDirectory / "workspace.json" : std::filesystem::path{}); showTrainingProgress = true; }
                } catch (const std::exception& error) { training.message = error.what(); }
            }

            ImGui::Begin(
                "Viewport",
                nullptr,
                ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoScrollbar|ImGuiWindowFlags_NoScrollWithMouse);

            drawViewportToolbar(false);
            const auto viewportCanvas=viewer::viewportCanvas();
            const auto available=viewportCanvas.size;
            const bool previewNoticeHovered=drawPreviewFailureNotice(viewportCanvas,false);
            if (cameraSource == CameraSource::Viewport) {
                const uint32_t targetRenderWidth = renderExtentFromAvailable(available.x);
                const uint32_t targetRenderHeight = renderExtentFromAvailable(available.y);
                if (renderWidth != targetRenderWidth || renderHeight != targetRenderHeight) {
                    renderWidth = targetRenderWidth;
                    renderHeight = targetRenderHeight;
                    cameraDirty = true;
                    renderRequested = true;
                }
            }
            if (texture.id != 0) {
                const ImVec2 imageCanvasMin=viewportCanvas.min;
                const ImVec2 imageCanvasMax(imageCanvasMin.x+available.x,imageCanvasMin.y+available.y);
                if(cameraSource==CameraSource::SceneXml) {
                    const std::string key=currentScenePath.string()+"/"+buildProducts.cameraGPUs[static_cast<std::size_t>(selectedSceneCameraIndex)].name;
                    if(key!=cameraImageKey) {cameraImageView.reset();cameraImageKey=key;}
                }
                const auto placement=viewer::viewportImagePlacement(viewportCanvas,cameraSource==CameraSource::SceneXml,cameraImageView,texture.width,texture.height);
                const ImVec2 imageSize=placement.size;
                ImGui::SetCursorScreenPos(placement.min);
                ImGui::PushClipRect(imageCanvasMin,imageCanvasMax,true);
                ImGui::Image(toImTextureId(texture.id), imageSize);
                const ImVec2 imageMin = ImGui::GetItemRectMin();
                const bool cameraCanvasHovered=ImGui::IsMouseHoveringRect(imageCanvasMin,imageCanvasMax);
                const ImVec2 toolsMin(imageCanvasMin.x+8,imageCanvasMin.y+42), toolsMax(imageCanvasMin.x+42,imageCanvasMin.y+182);
                const bool toolsHovered = ImGui::IsMouseHoveringRect(toolsMin,toolsMax);
                const bool imageHovered = ImGui::IsItemHovered() && !toolsHovered;
                drawViewportToolRail(imageCanvasMin,selectedSurfelEditorIndex>=0);
                auto* overlay=ImGui::GetWindowDrawList();
                overlay->AddText(ImVec2(imageMin.x+12,imageMin.y+10),ImGui::GetColorU32(viewer::ui::muted),cameraSource==CameraSource::SceneXml ? displayedCamera.name : "User perspective");

                const bool overlayMatchesScene=displayedOverlayPointCloud==currentPointCloudPath&&displayedOverlayScene==currentScenePath;
                if(overlayMatchesScene)refreshLightMarkerMetadata();
                const auto hoveredLightMarker=drawDisplayedLightMarkers(displayedCamera,imageMin,imageSize,
                    imageHovered&&!navigationInputCaptured&&overlayMatchesScene,sceneSelectionKind,selectedSurfelEditorIndex,selectedLightIndex);

                std::optional<int> transportMarkerPick;
                if (viewImageMode == ViewImageMode::TransportScore && transportXray && !transportScores.empty()) {
                    const auto cameraView = Pale::surfaceOverlapView(displayedCamera);
                    auto* draw = ImGui::GetWindowDrawList();
                    draw->PushClipRect(imageMin, ImVec2(imageMin.x + imageSize.x, imageMin.y + imageSize.y), true);
                    const auto mouse = ImGui::GetMousePos();
                    float nearest = 36.0f;
                    std::size_t shown = 0;
                    for (const auto& marker : transportMarkers) {
                        if (marker.primitive >= transportScores.size() || transportScores[marker.primitive] >= transportThreshold) continue;
                        const auto center = cameraView.worldToCamera * glm::vec4(marker.center, 1);
                        if (center.z >= 0) continue;
                        const float u = (cameraView.cx + cameraView.fx * center.x / -center.z) / cameraView.width;
                        const float v = (cameraView.cy - cameraView.fy * center.y / -center.z) / cameraView.height;
                        if (!(u >= 0 && u < 1 && v >= 0 && v < 1)) continue;
                        if (++shown > 20000) break;
                        const ImVec2 screen(imageMin.x + u * imageSize.x, imageMin.y + v * imageSize.y);
                        draw->AddCircleFilled(screen, 2.5f, IM_COL32(255, 45, 140, 200));
                        if (marker.editableIndex >= 0 && marker.editableIndex == selectedSurfelEditorIndex)
                            draw->AddCircle(screen, 6.0f, IM_COL32(255, 255, 255, 255), 12, 2.0f);
                        const float distance = (screen.x - mouse.x) * (screen.x - mouse.x) + (screen.y - mouse.y) * (screen.y - mouse.y);
                        if (imageHovered && marker.editableIndex >= 0 && distance < nearest) { nearest = distance; transportMarkerPick = marker.editableIndex; }
                    }
                    draw->PopClipRect();
                }
                const bool lightMarkerInput = (hoveredLightMarker || pressedLightMarker) && !ImGuizmo::IsUsingAny();
                ImGuizmo::Enable(!lightMarkerInput);
                bool viewportGizmoHovered = false;
                bool viewportGizmoUsing = false;
                const int firstHandleIndex=ImGui::GetWindowDrawList()->IdxBuffer.Size;
                if(overlayMatchesScene) {
                if (gizmosEnabled && viewportTool != 0 && sceneSelectionKind == viewer::SceneItemKind::Emitter && !navigationInputCaptured && showLightGizmo && selectedLight && selectedSurfelEditorIndex < 0) {
                    auto& transform = selectedLight.getComponent<Pale::TransformComponent>();
                    const glm::vec3 localCenter = lightLocalCenters.size() == areaLights.size()
                        ? lightLocalCenters[static_cast<std::size_t>(selectedLightIndex)] : glm::vec3(0);
                    glm::mat4 lightTransform = glm::translate(transform.getTransform(), localCenter);
                    glm::mat4 view = syclMatrixToGlm(displayedCamera.view);
                    glm::mat4 projection = syclMatrixToGlm(displayedCamera.proj);

                    ImGuizmo::SetOrthographic(false);
                    ImGuizmo::SetDrawlist(ImGui::GetWindowDrawList());
                    ImGuizmo::SetRect(imageMin.x, imageMin.y, imageSize.x, imageSize.y);
                    ImGuizmo::PushID("area-light-gizmo");
                    if (ImGuizmo::Manipulate(
                            glm::value_ptr(view),
                            glm::value_ptr(projection),
                            lightGizmoOperation,
                            lightGizmoOperation == ImGuizmo::SCALE ? ImGuizmo::LOCAL : lightGizmoMode,
                            glm::value_ptr(lightTransform))) {
                        beginSnapshotEdit();transform.setTransform(glm::translate(lightTransform, -localCenter));
                        rebuildSceneGpu();
                    }
                    viewportGizmoHovered = viewportGizmoHovered || ImGuizmo::IsOver();
                    viewportGizmoUsing = viewportGizmoUsing || ImGuizmo::IsUsing() || ImGuizmo::IsUsingAny();
                    ImGuizmo::PopID();
                }

	                if (gizmosEnabled && viewportTool != 0 && !navigationInputCaptured && showSurfelGizmo) {
	                    const std::optional<Pale::AssetHandle> pointCloudHandle = firstPointCloudHandle(scene);
	                    const std::shared_ptr<Pale::PointAsset> pointCloudAsset =
	                        pointCloudHandle ? assetAccessor.getPointCloud(*pointCloudHandle) : nullptr;
	                    const std::size_t editableSurfelCount = pointCloudAsset ? countSurfels(*pointCloudAsset) : 0u;
		                    if (pointCloudAsset && editableSurfelCount > 0u && selectedSurfelEditorIndex >= 0) {
		                        selectedSurfelEditorIndex = std::clamp(
		                            selectedSurfelEditorIndex,
		                            0,
	                            static_cast<int>(editableSurfelCount - 1u));
	                        const std::optional<EditableSurfelRef> editableSurfel =
	                            resolveEditableSurfel(*pointCloudAsset, selectedSurfelEditorIndex);
	                        if (editableSurfel && editableSurfel->pointGeometry) {
	                            Pale::PointGeometry& pointGeometry = *editableSurfel->pointGeometry;
	                            const std::size_t surfelIndex = editableSurfel->localIndex;
	                            if (surfelIndex < pointGeometry.positions.size() && surfelIndex < pointGeometry.quat.size()) {
	                                Pale::Entity pointCloudEntity = firstPointCloudEntity(scene);
	                                glm::mat4 pointCloudTransform{1.0f};
                                if (pointCloudEntity && pointCloudEntity.hasComponent<Pale::TransformComponent>()) {
                                    pointCloudTransform =
                                        pointCloudEntity.getComponent<Pale::TransformComponent>().getTransform();
                                }

                                const glm::mat4 localSurfelTransform =
                                    glm::translate(glm::mat4(1.0f), pointGeometry.positions[surfelIndex]) *
                                    glm::mat4_cast(normalizeQuaternionOrIdentity(pointGeometry.quat[surfelIndex]));
                                glm::mat4 surfelTransform =
                                    pointCloudTransform * localSurfelTransform;
                                glm::mat4 view = syclMatrixToGlm(displayedCamera.view);
                                glm::mat4 projection = syclMatrixToGlm(displayedCamera.proj);

                                ImGuizmo::SetOrthographic(false);
                                ImGuizmo::SetDrawlist(ImGui::GetWindowDrawList());
                                ImGuizmo::SetRect(imageMin.x, imageMin.y, imageSize.x, imageSize.y);
                                ImGuizmo::PushID("surfel-gizmo");
                                if (ImGuizmo::Manipulate(
                                        glm::value_ptr(view),
                                        glm::value_ptr(projection),
                                        surfelGizmoOperation,
                                        surfelGizmoMode,
                                        glm::value_ptr(surfelTransform))) {
	                                    const glm::mat4 localEditedTransform =
	                                        glm::inverse(pointCloudTransform) * surfelTransform;
	                                    pointGeometry.positions[surfelIndex] = glm::vec3(localEditedTransform[3]);
	                                    if (surfelGizmoOperation == ImGuizmo::ROTATE) {
	                                        pointGeometry.quat[surfelIndex] = extractRotationQuaternion(localEditedTransform);
	                                    }
	                                    surfelEditorStatus =
	                                        (surfelGizmoOperation == ImGuizmo::ROTATE ? "Rotated surfel " : "Moved surfel ") +
	                                        std::to_string(selectedSurfelEditorIndex);
	                                    beginSnapshotEdit();rebuildSceneGpu();
	                                }
                                viewportGizmoHovered = viewportGizmoHovered || ImGuizmo::IsOver();
                                viewportGizmoUsing = viewportGizmoUsing || ImGuizmo::IsUsing() || ImGuizmo::IsUsingAny();
                                ImGuizmo::PopID();
	                        }
	                    }
	                }
                }

                } else if(gizmosEnabled&&viewportTool!=0) {
                    displayedSelectedHandles.draw(*ImGui::GetWindowDrawList(),imageMin,imageSize);
                    viewportGizmoHovered=displayedSelectedHandles.contains(io.MousePos,imageMin,imageSize);
                }
                if(overlayMatchesScene)displayedSelectedHandles.capture(*ImGui::GetWindowDrawList(),firstHandleIndex,imageMin,imageSize);
                ImGuizmo::Enable(true);
                if (!navigationInputCaptured && cameraSource == CameraSource::Viewport) {
                    glm::mat4 viewGizmoMatrix = orbit.viewMatrix();
                    const ImVec2 viewGizmoSize{92.0f, 92.0f};
                    const ImVec2 viewGizmoPosition{
                        imageMin.x + imageSize.x - viewGizmoSize.x - 12.0f,
                        imageMin.y + 12.0f,
                    };
                    ImGuizmo::ViewManipulate(
                        glm::value_ptr(viewGizmoMatrix),
                        orbit.distance,
                        viewGizmoPosition,
                        viewGizmoSize,
                        IM_COL32(16, 16, 16, 96));
                    if (ImGuizmo::IsUsingViewManipulate()) {
                        const glm::mat4 cameraFromWorld = glm::inverse(viewGizmoMatrix);
                        const glm::vec3 forward = -glm::normalize(glm::vec3(cameraFromWorld[2]));
                        if (std::isfinite(forward.x) && std::isfinite(forward.y) && std::isfinite(forward.z)) {
                            orbit.setPositionKeepingTarget(orbit.target - forward * orbit.distance);
                            cameraDirty = true;
                        }
                    }
                    viewportGizmoHovered = viewportGizmoHovered || ImGuizmo::IsViewManipulateHovered();
                    viewportGizmoUsing = viewportGizmoUsing || ImGuizmo::IsUsingViewManipulate();
                }

                if (!ImGui::IsMouseDown(ImGuiMouseButton_Left) &&
                    !ImGui::IsMouseDown(ImGuiMouseButton_Middle) &&
                    !ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
                    viewportGizmoMouseCapture = false;
                }
                if (viewportGizmoUsing ||
                    (viewportGizmoHovered &&
                     (ImGui::IsMouseClicked(ImGuiMouseButton_Left) ||
                      ImGui::IsMouseClicked(ImGuiMouseButton_Middle) ||
                      ImGui::IsMouseClicked(ImGuiMouseButton_Right)))) {
                    viewportGizmoMouseCapture = true;
                }

                const bool viewportCameraInputBlocked =
                    previewNoticeHovered || navigationInputCaptured || toolsHovered || viewportGizmoUsing ||
                    (!lightMarkerInput && (viewportGizmoHovered || viewportGizmoMouseCapture));
                const ImVec2 imageMax{imageMin.x + imageSize.x, imageMin.y + imageSize.y};
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    pressedLightMarker = !viewportCameraInputBlocked ? hoveredLightMarker : std::nullopt;
                    viewportPickArmed = imageHovered && !viewportCameraInputBlocked && !pressedLightMarker;
                }
                if (pressedLightMarker && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                    if (hoveredLightMarker == pressedLightMarker && !navigationInputCaptured &&
                        io.MouseDragMaxDistanceSqr[ImGuiMouseButton_Left] < io.MouseDragThreshold * io.MouseDragThreshold) {
                        if (pressedLightMarker->areaIndex >= 0) {
                            selectedLightIndex = pressedLightMarker->areaIndex;
                            selectedSurfelEditorIndex = -1;
                            openLightControls = true;
                        } else {
                            selectedSurfelEditorIndex = pressedLightMarker->surfelIndex;
                            surfelEditorStatus = "Selected surfel light " + std::to_string(selectedSurfelEditorIndex);
                            const auto found=std::find(sceneTreeEmitterIndices.begin(),sceneTreeEmitterIndices.end(),static_cast<std::size_t>(selectedSurfelEditorIndex));
                            if(found!=sceneTreeEmitterIndices.end())selectedSurfelLightIndex=static_cast<int>(found-sceneTreeEmitterIndices.begin());
                            openLightControls = true;
                        }
                    }
                }
                if (viewportPickArmed && ImGui::IsMouseDragging(ImGuiMouseButton_Left, io.MouseDragThreshold)) {
                    viewportPickArmed = false;
                }
                if (viewportPickArmed && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                    const bool releaseInsideImage = ImGui::IsMouseHoveringRect(imageMin, imageMax, false);
                    if (releaseInsideImage && !viewportCameraInputBlocked && displayedRenderWidth > 0u && displayedRenderHeight > 0u) {
                        const ImVec2 mouse = ImGui::GetMousePos();
                        const auto pickRay = viewer::makePickRay(
                            displayedCamera, {mouse.x, mouse.y},
                            {imageMin.x, imageMin.y}, {imageSize.x, imageSize.y});
                        const std::optional<int> pickedSurfelIndex = transportMarkerPick ? transportMarkerPick : pickRay
                            ? pickEditableSurfel(scene, assetAccessor, *pickRay) : std::nullopt;
                        if (pickedSurfelIndex) {
                            selectedSurfelEditorIndex = *pickedSurfelIndex;
                            surfelEditorStatus = "Picked surfel " + std::to_string(selectedSurfelEditorIndex);
                            openSurfelControls=true;
                        } else {
                            selectedSurfelEditorIndex = -1;
                            surfelEditorStatus = "No surfel selected";
                        }
                    }
                    viewportPickArmed = false;
                }
                if ((imageHovered || (cameraSource==CameraSource::SceneXml&&cameraCanvasHovered)) && !viewportCameraInputBlocked && !hoveredLightMarker && !pressedLightMarker) {
                    const bool orbitDragging = ImGui::IsMouseDragging(ImGuiMouseButton_Left);
                    const bool panDragging =
                        ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ||
                        ImGui::IsMouseDragging(ImGuiMouseButton_Right);
                    const bool zooming = io.MouseWheel != 0.0f;
                    if(cameraSource==CameraSource::SceneXml&&!orbitDragging) {
                        if(panDragging)cameraImageView.drag(io.MouseDelta);
                        if(zooming)cameraImageView.zoomAt(io.MousePos,imageCanvasMin,available,io.MouseWheel);
                    }
                    if (cameraSource == CameraSource::SceneXml && orbitDragging) {
                        orbit = makeOrbitCameraFromSceneCamera(displayedCamera, bounds, orbit);
                        cameraSource = CameraSource::Viewport;
                        cameraDirty = true;
                        renderRequested = true;
                    }
                    if (cameraSource == CameraSource::Viewport) {
                        if(orbitDragging||panDragging||zooming)lastNavigationInput=std::chrono::steady_clock::now();
                        if (orbitDragging) {
                            orbit.orbit(io.MouseDelta);
                            cameraDirty = true;
                        }
                        if (panDragging) {
                            orbit.pan(io.MouseDelta);
                            cameraDirty = true;
                        }
                        if (zooming) {
                            orbit.zoom(io.MouseWheel);
                            cameraDirty = true;
                        }
                    }
                }
                if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) pressedLightMarker.reset();
                ImGui::PopClipRect();viewer::finishViewportCanvas(viewportCanvas);
            } else {
                ImGui::TextWrapped("%s", modelReadOnly() ? "Waiting for the launched training run to publish a scene. See Run monitor for initialization or errors." : "No preview yet");
            }
            drawViewportStatus(false);
            ImGui::End();

            if(snapshotEditPending) {
                snapshotEditPending=false;
                const auto source=cameraSource;const auto cameraIndex=selectedSceneCameraIndex;const auto pose=orbit;
                const int surfelIndex=selectedSurfelEditorIndex,lightIndex=selectedLightIndex,surfelLightIndex=selectedSurfelLightIndex;
                const auto kind=sceneSelectionKind;
                if(saveWorkingSnapshot()&&restoreWorkingSnapshot()) {
                    orbit=pose;selectedSurfelEditorIndex=surfelIndex;selectedLightIndex=lightIndex;selectedSurfelLightIndex=surfelLightIndex;sceneSelectionKind=kind;
                    if(source==CameraSource::SceneXml&&static_cast<std::size_t>(cameraIndex)<buildProducts.cameraGPUs.size()) {cameraSource=source;selectedSceneCameraIndex=cameraIndex;}
                    cameraDirty=renderRequested=true;
                }
            }
            training.drawProgress(showTrainingProgress);
            if(meshExport.draw(showMeshExport,displayedExportMetadata.IsMap(),displayedOverlayPointCloud.filename().string()))meshExportRequested=true;
            launchMeshExport(false);

            const bool previousGpuCounterProfilingEnabled = gpuCounterProfilingEnabled;
            const bool profilingSettingsChanged = drawProfilingWindow(
                showProfilingWindow,
                timerProfilingEnabled,
                gpuCounterProfilingEnabled,
                lastRenderMs,
                renderFrameTimeHistory,
                displayedRenderWidth,
                displayedRenderHeight,
                sceneGpu,
                lastProfilingCounters,
                lastTimerRecords);
            if (profilingSettingsChanged) {
                if (previousGpuCounterProfilingEnabled != gpuCounterProfilingEnabled) {
                    tracerDirty = true;
                }
                renderRequested = true;
            }

            if (navigationInputCaptured) {
                ImGui::EndDisabled();
            }
            if (std::memcmp(settingsBeforeUi.data(), &settings, sizeof(settings)) != 0) {
                targetLossDirty = true;
                transportDirty = true;
                if (viewImageMode == ViewImageMode::TransportScore) renderRequested = true;
                if (viewImageMode == ViewImageMode::TargetLoss) renderRequested = true;
            }
            ImGui::Render();
            int displayW = 0;
            int displayH = 0;
            glfwGetFramebufferSize(window, &displayW, &displayH);
            glViewport(0, 0, displayW, displayH);
            glClearColor(
                kBlenderViewportBackground.x,
                kBlenderViewportBackground.y,
                kBlenderViewportBackground.z,
                kBlenderViewportBackground.w);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window);
            if(!glfwWindowShouldClose(window)) {
                const auto now=viewer::PreviewRecovery::Clock::now();
                if(previewRecovery.retryDue(now))renderRequested=true;
                if(lastPreviewInteractive&&std::chrono::steady_clock::now()-lastNavigationInput>=std::chrono::milliseconds(180))renderRequested=true;
                if(training.checkpointNavigation||training.checkpointSelection)navigateCheckpoints();
                // Render each accepted scene before accepting another live
                // generation, even if training publishes faster than we draw.
                if(training.initializedPointCloudRequest&&!holdPreviewForInspector)loadRequestedInitializer();
                if(!previewWork.busy()&&viewer::canPollTrainingScene(sceneUploadPending,training.loadedGeneration)&&!holdPreviewForInspector&&previewRecovery.dispatchAllowed(now))pollTrainingScene();
                if(!holdPreviewForInspector&&!previewWork.busy() && previewRecovery.dispatchAllowed(now) && (!modelReadOnly() || training.loadedGeneration!=0) && ((autoRender&&cameraDirty)||renderRequested))
                    startPreviewJob(renderNow,[&] {
                        updateDisplayTexture();publishDisplayedOverlays();lastPreviewInteractive=interactivePreview;
                        previewRecovery.succeeded();
                        if(pendingViewedIteration) {
                            if(pendingViewedIteration->first==training.process.directory&&modelReadOnly())training.viewedIteration=pendingViewedIteration->second;
                            pendingViewedIteration.reset();
                        }
                    },"Rendering / compiling preview kernels");
            }
        }
        // Stop the owned optimizer immediately, before waiting for any long
        // in-flight preview/JIT work to release renderer resources.
        training.process.shutdown();
        meshExport.shutdown();
        glfwHideWindow(window);
        try {previewWork.finish();}catch(const std::exception& error) {Pale::Log::PA_WARN("Preview shutdown: {}",error.what());}

        Pale::Log::PA_INFO("Viewer event loop ended; releasing renderer resources");
        if (hasSensor) {
            destroySensor(queue, sensor);
        }
        Pale::SceneUpload::freeBuffers(sceneGpu, queue);
        if (deviceProfilingCounters != nullptr) {
            sycl::free(deviceProfilingCounters, queue);
            deviceProfilingCounters = nullptr;
        }
        Pale::freeGradientsForScene(queue, viewerAdjointGradients);
        Pale::freeDebugImagesForScene(queue, &viewerAdjointDebugImages, 1u);
        freeViewerRegularizerGradients();
        texture.destroy();

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 0;
    } catch (const std::exception& exception) {
        spdlog::critical("{}", exception.what());
        return 1;
    }
}
