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
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
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

#include "Renderer/GPUDataStructures.h"
#include "Renderer/RenderPackage.h"
#include "Renderer/Kernels/IntersectionKernels.h"
#include "Core/ScopedTimer.h"
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

#ifndef PALE_DEFAULT_ASSET_DIR
#define PALE_DEFAULT_ASSET_DIR "../Assets"
#endif

namespace {
    const glm::vec3 kWorldUp{0.0f, 0.0f, 1.0f};
    const glm::vec3 kDefaultLookAt{0.0f, 0.0f, 0.2f};
    constexpr ImVec4 kBlenderViewportBackground{0.215f, 0.215f, 0.215f, 1.0f};
    constexpr uint32_t kDefaultViewportImageExtent = 500u;
    constexpr float kSidebarDockFraction = 0.50f;
    constexpr std::size_t kFrameTimeHistoryCapacity = 240u;
    constexpr uint64_t kSnapshotIterationStep = 1000u;

    struct AppArgs {
        std::filesystem::path assetsDir = PALE_DEFAULT_ASSET_DIR;
        std::filesystem::path pointCloudPath = "OptimizerTests/initial_points.ply";
        std::filesystem::path scenePath = "OptimizerTests/scene.xml";
        uint32_t width = 0;
        uint32_t height = 0;
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
        DepthNormal,
        DepthDistortion,
        IntraSlabDepth,
        CurvatureScale,
        CurvaturePrimitiveScore,
        PositionPrimitiveScore,
        DensificationOrigin,
        PrimitiveAge,
        DepthPositionGradient,
        NormalPositionGradient,
        IntraSlabPositionGradient,
        SsimTarget,
        RgbHalfMse,
        SsimIndex,
        Dssim,
        RgbObjectiveGradient,
    };

    constexpr std::array<ViewImageMode, 20> kViewImageModeShortcutOrder = {
        ViewImageMode::Rendered,
        ViewImageMode::MedianDepth,
        ViewImageMode::DepthDistortion,
        ViewImageMode::MeanDepth,
        ViewImageMode::VisibleNormal,
        ViewImageMode::DepthNormal,
        ViewImageMode::IntraSlabDepth,
        ViewImageMode::CurvatureScale,
        ViewImageMode::PositionPrimitiveScore,
        ViewImageMode::CurvaturePrimitiveScore,
        ViewImageMode::DensificationOrigin,
        ViewImageMode::PrimitiveAge,
        ViewImageMode::DepthPositionGradient,
        ViewImageMode::NormalPositionGradient,
        ViewImageMode::IntraSlabPositionGradient,
        ViewImageMode::SsimTarget,
        ViewImageMode::RgbHalfMse,
        ViewImageMode::SsimIndex,
        ViewImageMode::Dssim,
        ViewImageMode::RgbObjectiveGradient,
    };

    constexpr std::array<const char*, 20> kViewImageModeLabels = {
        "1 Rendered",
        "2 Median depth",
        "3 Depth distortion",
        "4 Mean depth",
        "5 Visible normal",
        "6 Depth normal",
        "7 Intra-slab depth",
        "8 Curvature scale",
        "9 Position primitive score (saved)",
        "Curvature primitive score",
        "Densification split origin",
        "Primitive age",
        "Depth distortion |grad position|",
        "Normal consistency |grad position|",
        "Intra-slab consensus |grad position|",
        "SSIM target RGB",
        "RGB half-MSE per pixel",
        "SSIM index per pixel",
        "DSSIM per pixel",
        "RGB objective |dL/dRGB|",
    };

    [[nodiscard]] const char* viewImageModeLabel(ViewImageMode mode) {
        for (std::size_t index = 0; index < kViewImageModeShortcutOrder.size(); ++index) {
            if (kViewImageModeShortcutOrder[index] == mode) {
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
        // The per-primitive maps use the identity selected by the curvature
        // pass, even when the displayed quantity itself is not curvature.
        return mode == ViewImageMode::CurvatureScale ||
               mode == ViewImageMode::CurvaturePrimitiveScore ||
               mode == ViewImageMode::PositionPrimitiveScore ||
               mode == ViewImageMode::DensificationOrigin ||
               mode == ViewImageMode::PrimitiveAge ||
               isRegularizerGradientView(mode);
    }

    [[nodiscard]] bool isSsimDebugView(ViewImageMode mode) {
        return mode == ViewImageMode::SsimTarget ||
               mode == ViewImageMode::RgbHalfMse ||
               mode == ViewImageMode::SsimIndex ||
               mode == ViewImageMode::Dssim ||
               mode == ViewImageMode::RgbObjectiveGradient;
    }

    enum class ScalarColorMap {
        Viridis,
        Jet,
    };

    struct DebugDisplayBuffers {
        uint32_t width = 0;
        uint32_t height = 0;
        bool meanDepthValid = false;
        bool medianDepthValid = false;
        bool visibleNormalValid = false;
        bool depthNormalValid = false;
        bool depthDistortionValid = false;
        bool intraSlabDepthValid = false;
        bool curvatureScaleValid = false;
        bool curvaturePrimitiveScoreValid = false;
        bool positionPrimitiveScoreValid = false;
        bool positionPrimitiveRadianceBiasAvailable = false;
        bool positionPrimitiveIndicesValid = false;
        bool densificationOriginValid = false;
        bool primitiveAgeValid = false;
        bool visiblePrimitiveIndicesValid = false;
        bool depthPositionGradientValid = false;
        bool normalPositionGradientValid = false;
        bool intraSlabPositionGradientValid = false;
        bool ssimDebugValid = false;
        std::vector<float> meanDepth;
        std::vector<float> medianDepth;
        std::vector<float> visibleNormal;
        std::vector<float> depthNormal;
        std::vector<float> depthDistortion;
        std::vector<float> intraSlabDepth;
        std::vector<float> curvatureScale;
        std::vector<float> curvaturePrimitiveScore;
        std::vector<float> curvatureObservedPrimitiveScores;
        std::vector<float> positionPrimitiveScore;
        std::vector<float> positionObservedPrimitiveScores;
        std::vector<uint32_t> positionPrimitiveIndices;
        std::vector<std::uint8_t> densificationOrigin;
        std::vector<std::uint32_t> primitiveAge;
        std::vector<uint32_t> visiblePrimitiveIndices;
        std::vector<float> depthPositionGradient;
        std::vector<float> normalPositionGradient;
        std::vector<float> intraSlabPositionGradient;
        std::vector<float> ssimTargetLinearRgba;
        std::vector<float> rgbHalfMse;
        std::vector<float> ssimIndex;
        std::vector<float> dssim;
        std::vector<float> rgbObjectiveGradient;
        std::size_t curvatureObservedPrimitiveCount = 0u;
        float curvaturePrimitiveScoreMax = 0.0f;
        std::size_t positionObservedPrimitiveCount = 0u;
        std::size_t positionUnsampledPrimitiveCount = 0u;
        float positionPrimitiveScoreMax = 0.0f;
        float positionPrimitiveSplitThreshold = 0.0f;
        bool positionPrimitiveMetadataAvailable = false;
        float rgbHalfMseMean = 0.0f;
        float ssimMean = 0.0f;
        float dssimMean = 0.0f;
        float rgbObjectiveMean = 0.0f;

        void invalidate() {
            width = 0;
            height = 0;
            meanDepthValid = false;
            medianDepthValid = false;
            visibleNormalValid = false;
            depthNormalValid = false;
            depthDistortionValid = false;
            intraSlabDepthValid = false;
            curvatureScaleValid = false;
            curvaturePrimitiveScoreValid = false;
            positionPrimitiveScoreValid = false;
            positionPrimitiveRadianceBiasAvailable = false;
            positionPrimitiveIndicesValid = false;
            densificationOriginValid = false;
            primitiveAgeValid = false;
            visiblePrimitiveIndicesValid = false;
            depthPositionGradientValid = false;
            normalPositionGradientValid = false;
            intraSlabPositionGradientValid = false;
            ssimDebugValid = false;
            meanDepth.clear();
            medianDepth.clear();
            visibleNormal.clear();
            depthNormal.clear();
            depthDistortion.clear();
            intraSlabDepth.clear();
            curvatureScale.clear();
            curvaturePrimitiveScore.clear();
            curvatureObservedPrimitiveScores.clear();
            positionPrimitiveScore.clear();
            positionObservedPrimitiveScores.clear();
            positionPrimitiveIndices.clear();
            densificationOrigin.clear();
            primitiveAge.clear();
            visiblePrimitiveIndices.clear();
            depthPositionGradient.clear();
            normalPositionGradient.clear();
            intraSlabPositionGradient.clear();
            ssimTargetLinearRgba.clear();
            rgbHalfMse.clear();
            ssimIndex.clear();
            dssim.clear();
            rgbObjectiveGradient.clear();
            curvatureObservedPrimitiveCount = 0u;
            curvaturePrimitiveScoreMax = 0.0f;
            positionObservedPrimitiveCount = 0u;
            positionUnsampledPrimitiveCount = 0u;
            positionPrimitiveScoreMax = 0.0f;
            positionPrimitiveSplitThreshold = 0.0f;
            positionPrimitiveMetadataAvailable = false;
            rgbHalfMseMean = 0.0f;
            ssimMean = 0.0f;
            dssimMean = 0.0f;
            rgbObjectiveMean = 0.0f;
        }

        void prepareFor(uint32_t nextWidth, uint32_t nextHeight) {
            if (width == nextWidth && height == nextHeight) {
                return;
            }

            invalidate();
            width = nextWidth;
            height = nextHeight;
        }

        void releaseSsimDebug() {
            ssimDebugValid = false;
            std::vector<float>().swap(ssimTargetLinearRgba);
            std::vector<float>().swap(rgbHalfMse);
            std::vector<float>().swap(ssimIndex);
            std::vector<float>().swap(dssim);
            std::vector<float>().swap(rgbObjectiveGradient);
            rgbHalfMseMean = 0.0f;
            ssimMean = 0.0f;
            dssimMean = 0.0f;
            rgbObjectiveMean = 0.0f;
        }
    };

    struct SsimTargetCache {
        std::filesystem::path path;
        uint32_t width = 0u;
        uint32_t height = 0u;
        std::vector<float> linearRgba;
        std::string status = "SSIM debug maps are disabled";

        void invalidate() {
            path.clear();
            width = 0u;
            height = 0u;
            std::vector<float>().swap(linearRgba);
        }
    };

    struct OrbitCamera {
        glm::vec3 target{0.0f};
        float distance = 3.0f;
        float yaw = 0.0f;
        float pitch = 0.0f;
        float fovyDegrees = 45.0f;
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

    struct Texture2D {
        GLuint id = 0;
        uint32_t width = 0;
        uint32_t height = 0;

        void update(const std::vector<uint8_t>& rgba, uint32_t newWidth, uint32_t newHeight) {
            if (id == 0) {
                glGenTextures(1, &id);
            }

            glBindTexture(GL_TEXTURE_2D, id);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

            if (width != newWidth || height != newHeight) {
                width = newWidth;
                height = newHeight;
                glTexImage2D(
                    GL_TEXTURE_2D,
                    0,
                    GL_RGBA8,
                    static_cast<GLsizei>(width),
                    static_cast<GLsizei>(height),
                    0,
                    GL_RGBA,
                    GL_UNSIGNED_BYTE,
                    rgba.data());
            } else {
                glTexSubImage2D(
                    GL_TEXTURE_2D,
                    0,
                    0,
                    0,
                    static_cast<GLsizei>(width),
                    static_cast<GLsizei>(height),
                    GL_RGBA,
                    GL_UNSIGNED_BYTE,
                    rgba.data());
            }
        }

        void destroy() {
            if (id != 0) {
                glDeleteTextures(1, &id);
                id = 0;
            }
            width = 0;
            height = 0;
        }
    };

    struct DropState {
        std::filesystem::path pendingPlyPath;
        bool hasPendingPlyPath = false;
    };

    struct FileBrowserEntry {
        std::filesystem::path path;
        bool directory = false;
    };

    struct PointCloudSnapshot {
        std::filesystem::path path;
        uint64_t iteration = 0u;
        std::filesystem::file_time_type writeTime = std::filesystem::file_time_type::min();
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

    [[nodiscard]] std::size_t countSurfels(const Pale::PointAsset& pointAsset) {
        std::size_t surfelCount = 0;
        for (const Pale::PointGeometry& pointGeometry : pointAsset.points) {
            surfelCount += pointGeometry.positions.size();
        }
        return surfelCount;
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

    [[nodiscard]] std::filesystem::file_time_type lastWriteTimeOrMin(const std::filesystem::path& path) {
        std::error_code error;
        const std::filesystem::file_time_type time = std::filesystem::last_write_time(path, error);
        return error ? std::filesystem::file_time_type::min() : time;
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

    [[nodiscard]] std::optional<uint64_t> parseIterationPointCloudFilename(const std::filesystem::path& path) {
        const std::string filename = path.filename().string();
        const std::string prefix = "iter_";
        const std::string suffix = "_points.ply";
        if (filename.rfind(prefix, 0u) != 0u ||
            filename.size() <= prefix.size() + suffix.size() ||
            filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) != 0) {
            return std::nullopt;
        }

        uint64_t iteration = 0u;
        for (std::size_t index = prefix.size(); index < filename.size() - suffix.size(); ++index) {
            const char c = filename[index];
            if (c < '0' || c > '9') {
                return std::nullopt;
            }
            iteration = iteration * 10u + static_cast<uint64_t>(c - '0');
        }
        return iteration;
    }

    [[nodiscard]] std::vector<PointCloudSnapshot> listOptimizationPointCloudSnapshots(
        const std::filesystem::path& pointsDirectory) {
        std::vector<PointCloudSnapshot> snapshots;
        std::error_code error;
        if (!std::filesystem::is_directory(pointsDirectory, error)) {
            return snapshots;
        }

        for (const auto& pointEntry : std::filesystem::directory_iterator(pointsDirectory, error)) {
            if (error) {
                break;
            }
            if (!pointEntry.is_regular_file(error) || error || !isPlyPath(pointEntry.path())) {
                error.clear();
                continue;
            }

            const std::optional<uint64_t> iteration = parseIterationPointCloudFilename(pointEntry.path());
            if (!iteration) {
                continue;
            }
            snapshots.push_back({
                .path = pointEntry.path(),
                .iteration = *iteration,
                .writeTime = lastWriteTimeOrMin(pointEntry.path()),
            });
        }

        std::sort(snapshots.begin(), snapshots.end(), [](const PointCloudSnapshot& lhs, const PointCloudSnapshot& rhs) {
            if (lhs.iteration != rhs.iteration) {
                return lhs.iteration < rhs.iteration;
            }
            if (lhs.writeTime != rhs.writeTime) {
                return lhs.writeTime < rhs.writeTime;
            }
            return lhs.path.filename().string() < rhs.path.filename().string();
        });
        return snapshots;
    }

    [[nodiscard]] std::optional<std::filesystem::path> findLatestOptimizationPointCloud(
        const std::filesystem::path& optimizationOutputDirectory) {
        struct RunCandidate {
            std::filesystem::path runDirectory;
            std::filesystem::path metricsPath;
            PointCloudSnapshot pointCloud;
            std::filesystem::file_time_type metricsWriteTime = std::filesystem::file_time_type::min();
        };

        std::error_code error;
        if (!std::filesystem::is_directory(optimizationOutputDirectory, error)) {
            return std::nullopt;
        }

        std::optional<RunCandidate> bestRun;
        std::filesystem::recursive_directory_iterator runEntry(
            optimizationOutputDirectory,
            std::filesystem::directory_options::skip_permission_denied,
            error);
        const std::filesystem::recursive_directory_iterator end;
        while (runEntry != end) {
            std::error_code entryError;
            const std::filesystem::directory_entry entry = *runEntry;
            const bool isMetricsFile =
                entry.path().filename() == "metrics.csv" &&
                entry.is_regular_file(entryError) &&
                !entryError;

            runEntry.increment(error);
            if (error) {
                error.clear();
            }
            if (!isMetricsFile) {
                continue;
            }

            const std::filesystem::path metricsPath = entry.path();
            const std::filesystem::path runDirectory = metricsPath.parent_path();
            const std::filesystem::path pointsDirectory = runDirectory / "points";
            std::error_code pointsError;
            if (!std::filesystem::is_directory(pointsDirectory, pointsError) || pointsError) {
                continue;
            }

            const std::vector<PointCloudSnapshot> snapshots = listOptimizationPointCloudSnapshots(pointsDirectory);
            if (snapshots.empty()) {
                continue;
            }
            const PointCloudSnapshot& bestPointCloud = snapshots.back();

            RunCandidate candidate{
                .runDirectory = runDirectory,
                .metricsPath = metricsPath,
                .pointCloud = bestPointCloud,
                .metricsWriteTime = lastWriteTimeOrMin(metricsPath),
            };
            if (!bestRun ||
                candidate.metricsWriteTime > bestRun->metricsWriteTime ||
                (candidate.metricsWriteTime == bestRun->metricsWriteTime &&
                 candidate.metricsPath.string() > bestRun->metricsPath.string())) {
                bestRun = candidate;
            }
        }

        if (!bestRun) {
            return std::nullopt;
        }
        return bestRun->pointCloud.path;
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

    [[nodiscard]] std::optional<std::filesystem::path> targetPngForOptimizationPointCloud(
        const std::filesystem::path& pointCloudPath,
        const std::string& cameraName) {
        std::filesystem::path runDirectory = pointCloudPath.parent_path();
        if (runDirectory.filename() == "points") {
            runDirectory = runDirectory.parent_path();
        }
        if (runDirectory.empty() || cameraName.empty()) {
            return std::nullopt;
        }

        const std::filesystem::path targetPath =
            runDirectory / ("render_target_" + cameraName + ".png");
        std::error_code error;
        if (!std::filesystem::is_regular_file(targetPath, error) || error) {
            return std::nullopt;
        }
        return targetPath;
    }

    [[nodiscard]] float srgbToLinear(float value) {
        const float clamped = std::clamp(value, 0.0f, 1.0f);
        return clamped <= 0.04045f
                   ? clamped / 12.92f
                   : std::pow((clamped + 0.055f) / 1.055f, 2.4f);
    }

    bool ensureLinearSsimTarget(
        const std::filesystem::path& targetPath,
        uint32_t expectedWidth,
        uint32_t expectedHeight,
        SsimTargetCache& cache) {
        if (cache.path == targetPath &&
            cache.width == expectedWidth &&
            cache.height == expectedHeight &&
            cache.linearRgba.size() ==
                static_cast<std::size_t>(expectedWidth) * expectedHeight * 4u) {
            return true;
        }

        cache.invalidate();
        int imageWidth = 0;
        int imageHeight = 0;
        int sourceChannels = 0;
        stbi_uc* image = stbi_load(
            targetPath.string().c_str(), &imageWidth, &imageHeight, &sourceChannels, 4);
        if (image == nullptr) {
            cache.status = "Could not load target PNG: " + targetPath.string();
            return false;
        }

        if (imageWidth != static_cast<int>(expectedWidth) ||
            imageHeight != static_cast<int>(expectedHeight)) {
            cache.status =
                "Target resolution " + std::to_string(imageWidth) + "x" +
                std::to_string(imageHeight) + " does not match render " +
                std::to_string(expectedWidth) + "x" + std::to_string(expectedHeight);
            stbi_image_free(image);
            return false;
        }

        const std::size_t pixelCount =
            static_cast<std::size_t>(expectedWidth) * expectedHeight;
        cache.linearRgba.resize(pixelCount * 4u);
        for (std::size_t pixelIndex = 0u; pixelIndex < pixelCount; ++pixelIndex) {
            const std::size_t baseIndex = pixelIndex * 4u;
            cache.linearRgba[baseIndex + 0u] =
                srgbToLinear(static_cast<float>(image[baseIndex + 0u]) / 255.0f);
            cache.linearRgba[baseIndex + 1u] =
                srgbToLinear(static_cast<float>(image[baseIndex + 1u]) / 255.0f);
            cache.linearRgba[baseIndex + 2u] =
                srgbToLinear(static_cast<float>(image[baseIndex + 2u]) / 255.0f);
            cache.linearRgba[baseIndex + 3u] = 1.0f;
        }
        stbi_image_free(image);

        cache.path = targetPath;
        cache.width = expectedWidth;
        cache.height = expectedHeight;
        cache.status = "Loaded " + targetPath.string();
        return true;
    }

    [[nodiscard]] std::vector<float> makeGaussianKernel1d(int windowSize, float sigma) {
        const int radius = windowSize / 2;
        const float exponentScale = -0.5f / (sigma * sigma);
        std::vector<float> kernel(static_cast<std::size_t>(windowSize), 0.0f);
        float sum = 0.0f;
        for (int offset = -radius; offset <= radius; ++offset) {
            const float value = std::exp(static_cast<float>(offset * offset) * exponentScale);
            kernel[static_cast<std::size_t>(offset + radius)] = value;
            sum += value;
        }
        const float inverseSum = 1.0f / std::max(sum, 1.0e-12f);
        for (float& value : kernel) {
            value *= inverseSum;
        }
        return kernel;
    }

    // Symmetric zero-padded Gaussian convolution, matching the training SSIM window.
    [[nodiscard]] std::vector<float> convolveRgbZeroPadded(
        const std::vector<float>& source,
        uint32_t width,
        uint32_t height,
        const std::vector<float>& kernel) {
        const std::size_t pixelCount = static_cast<std::size_t>(width) * height;
        std::vector<float> temporary(pixelCount * 3u, 0.0f);
        std::vector<float> result(pixelCount * 3u, 0.0f);
        const int radius = static_cast<int>(kernel.size() / 2u);

        for (uint32_t y = 0u; y < height; ++y) {
            for (uint32_t x = 0u; x < width; ++x) {
                const std::size_t outputBase =
                    (static_cast<std::size_t>(y) * width + x) * 3u;
                for (int offset = -radius; offset <= radius; ++offset) {
                    const int sourceX = static_cast<int>(x) + offset;
                    if (sourceX < 0 || sourceX >= static_cast<int>(width)) {
                        continue;
                    }
                    const float weight = kernel[static_cast<std::size_t>(offset + radius)];
                    const std::size_t sourceBase =
                        (static_cast<std::size_t>(y) * width +
                         static_cast<uint32_t>(sourceX)) * 3u;
                    for (std::size_t channel = 0u; channel < 3u; ++channel) {
                        temporary[outputBase + channel] += weight * source[sourceBase + channel];
                    }
                }
            }
        }

        for (uint32_t y = 0u; y < height; ++y) {
            for (uint32_t x = 0u; x < width; ++x) {
                const std::size_t outputBase =
                    (static_cast<std::size_t>(y) * width + x) * 3u;
                for (int offset = -radius; offset <= radius; ++offset) {
                    const int sourceY = static_cast<int>(y) + offset;
                    if (sourceY < 0 || sourceY >= static_cast<int>(height)) {
                        continue;
                    }
                    const float weight = kernel[static_cast<std::size_t>(offset + radius)];
                    const std::size_t sourceBase =
                        (static_cast<std::size_t>(sourceY) * width + x) * 3u;
                    for (std::size_t channel = 0u; channel < 3u; ++channel) {
                        result[outputBase + channel] += weight * temporary[sourceBase + channel];
                    }
                }
            }
        }
        return result;
    }

    void computeSsimDebugBuffers(
        const std::vector<float>& renderedRgba,
        const std::vector<float>& targetRgba,
        uint32_t width,
        uint32_t height,
        float ssimWeight,
        int windowSize,
        float sigma,
        DebugDisplayBuffers& buffers) {
        const std::size_t pixelCount = static_cast<std::size_t>(width) * height;
        if (pixelCount == 0u || renderedRgba.size() != pixelCount * 4u ||
            targetRgba.size() != pixelCount * 4u) {
            return;
        }

        std::vector<float> rendered(pixelCount * 3u);
        std::vector<float> target(pixelCount * 3u);
        std::vector<float> renderedSquared(pixelCount * 3u);
        std::vector<float> targetSquared(pixelCount * 3u);
        std::vector<float> renderedTarget(pixelCount * 3u);
        for (std::size_t pixelIndex = 0u; pixelIndex < pixelCount; ++pixelIndex) {
            for (std::size_t channel = 0u; channel < 3u; ++channel) {
                const std::size_t rgbIndex = pixelIndex * 3u + channel;
                const std::size_t rgbaIndex = pixelIndex * 4u + channel;
                const float x = renderedRgba[rgbaIndex];
                const float y = targetRgba[rgbaIndex];
                rendered[rgbIndex] = x;
                target[rgbIndex] = y;
                renderedSquared[rgbIndex] = x * x;
                targetSquared[rgbIndex] = y * y;
                renderedTarget[rgbIndex] = x * y;
            }
        }

        const std::vector<float> kernel = makeGaussianKernel1d(windowSize, sigma);
        const std::vector<float> muX = convolveRgbZeroPadded(rendered, width, height, kernel);
        const std::vector<float> muY = convolveRgbZeroPadded(target, width, height, kernel);
        const std::vector<float> expectedX2 =
            convolveRgbZeroPadded(renderedSquared, width, height, kernel);
        const std::vector<float> expectedY2 =
            convolveRgbZeroPadded(targetSquared, width, height, kernel);
        const std::vector<float> expectedXY =
            convolveRgbZeroPadded(renderedTarget, width, height, kernel);

        constexpr float c1 = 0.01f * 0.01f;
        constexpr float c2 = 0.03f * 0.03f;
        std::vector<float> dMean(pixelCount * 3u);
        std::vector<float> dVariance(pixelCount * 3u);
        std::vector<float> dCovariance(pixelCount * 3u);
        std::vector<float> muXDVariance(pixelCount * 3u);
        std::vector<float> muYDCovariance(pixelCount * 3u);
        std::vector<float> channelSsim(pixelCount * 3u);

        for (std::size_t index = 0u; index < pixelCount * 3u; ++index) {
            const float varianceX = expectedX2[index] - muX[index] * muX[index];
            const float varianceY = expectedY2[index] - muY[index] * muY[index];
            const float covariance = expectedXY[index] - muX[index] * muY[index];
            const float a = 2.0f * muX[index] * muY[index] + c1;
            const float b = std::max(
                muX[index] * muX[index] + muY[index] * muY[index] + c1,
                1.0e-12f);
            const float c = 2.0f * covariance + c2;
            const float d = std::max(varianceX + varianceY + c2, 1.0e-12f);
            const float luminance = a / b;
            const float contrast = c / d;
            channelSsim[index] = luminance * contrast;
            dMean[index] = contrast *
                (2.0f * muY[index] * b - 2.0f * muX[index] * a) / (b * b);
            dVariance[index] = -luminance * c / (d * d);
            dCovariance[index] = luminance * 2.0f / d;
            muXDVariance[index] = muX[index] * dVariance[index];
            muYDCovariance[index] = muY[index] * dCovariance[index];
        }

        const std::vector<float> convolvedDMean =
            convolveRgbZeroPadded(dMean, width, height, kernel);
        const std::vector<float> convolvedDVariance =
            convolveRgbZeroPadded(dVariance, width, height, kernel);
        const std::vector<float> convolvedMuXDVariance =
            convolveRgbZeroPadded(muXDVariance, width, height, kernel);
        const std::vector<float> convolvedDCovariance =
            convolveRgbZeroPadded(dCovariance, width, height, kernel);
        const std::vector<float> convolvedMuYDCovariance =
            convolveRgbZeroPadded(muYDCovariance, width, height, kernel);

        buffers.ssimTargetLinearRgba = targetRgba;
        buffers.rgbHalfMse.assign(pixelCount, 0.0f);
        buffers.ssimIndex.assign(pixelCount, 0.0f);
        buffers.dssim.assign(pixelCount, 0.0f);
        buffers.rgbObjectiveGradient.assign(pixelCount, 0.0f);
        buffers.rgbHalfMseMean = 0.0f;
        buffers.ssimMean = 0.0f;
        buffers.dssimMean = 0.0f;
        buffers.rgbObjectiveMean = 0.0f;
        const float inverseRgbElementCount =
            1.0f / static_cast<float>(pixelCount * 3u);
        for (std::size_t pixelIndex = 0u; pixelIndex < pixelCount; ++pixelIndex) {
            float squaredError = 0.0f;
            float ssimSum = 0.0f;
            float gradientSquaredNorm = 0.0f;
            for (std::size_t channel = 0u; channel < 3u; ++channel) {
                const std::size_t index = pixelIndex * 3u + channel;
                const float difference = rendered[index] - target[index];
                squaredError += difference * difference;
                ssimSum += channelSsim[index];
                const float ssimGradient =
                    convolvedDMean[index] +
                    2.0f * rendered[index] * convolvedDVariance[index] -
                    2.0f * convolvedMuXDVariance[index] +
                    target[index] * convolvedDCovariance[index] -
                    convolvedMuYDCovariance[index];
                const float objectiveGradient =
                    ((1.0f - ssimWeight) * difference - ssimWeight * ssimGradient) *
                    inverseRgbElementCount;
                gradientSquaredNorm += objectiveGradient * objectiveGradient;
            }

            buffers.rgbHalfMse[pixelIndex] = squaredError / 6.0f;
            buffers.ssimIndex[pixelIndex] = ssimSum / 3.0f;
            buffers.dssim[pixelIndex] = 1.0f - buffers.ssimIndex[pixelIndex];
            buffers.rgbObjectiveGradient[pixelIndex] = std::sqrt(
                std::max(gradientSquaredNorm, 0.0f));
            buffers.rgbHalfMseMean += buffers.rgbHalfMse[pixelIndex];
            buffers.ssimMean += buffers.ssimIndex[pixelIndex];
            buffers.dssimMean += buffers.dssim[pixelIndex];
        }

        const float inversePixelCount = 1.0f / static_cast<float>(pixelCount);
        buffers.rgbHalfMseMean *= inversePixelCount;
        buffers.ssimMean *= inversePixelCount;
        buffers.dssimMean *= inversePixelCount;
        buffers.rgbObjectiveMean =
            (1.0f - ssimWeight) * buffers.rgbHalfMseMean +
            ssimWeight * buffers.dssimMean;
        buffers.ssimDebugValid = true;
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

    struct PickRay {
        glm::vec3 origin{0.0f};
        glm::vec3 direction{0.0f, 0.0f, -1.0f};
    };

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

    [[nodiscard]] PickRay makePickRay(const Pale::CameraGPU& camera, float pixelX, float pixelY) {
        const float width = static_cast<float>(std::max(camera.width, 1u));
        const float height = static_cast<float>(std::max(camera.height, 1u));
        const float vFlipped = height - pixelY;
        const float ndcX = 2.0f * pixelX / width - 1.0f;
        const float ndcY = 2.0f * vFlipped / height - 1.0f;
        const float fy = 0.5f * height / std::tan(0.5f * glm::radians(camera.fovy));
        const float fx = fy * (width / height);
        const glm::vec3 cameraDirection = glm::normalize(glm::vec3{
            ndcX * (0.5f * width) / fx,
            ndcY * (0.5f * height) / fy,
            -1.0f,
        });
        const glm::mat4 worldFromCamera = syclMatrixToGlm(camera.invView);
        return {.origin = Pale::sycl2glm(camera.pos), .direction = glm::normalize(glm::mat3(worldFromCamera) * cameraDirection)};
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
        const Pale::CameraGPU& camera,
        float pixelX,
        float pixelY) {
        const std::optional<Pale::AssetHandle> pointCloudHandle = firstPointCloudHandle(scene);
        const std::shared_ptr<Pale::PointAsset> pointCloudAsset =
            pointCloudHandle ? assetAccessor.getPointCloud(*pointCloudHandle) : nullptr;
        if (!pointCloudAsset || countSurfels(*pointCloudAsset) == 0u) {
            return std::nullopt;
        }

        const PickRay ray = makePickRay(camera, pixelX, pixelY);
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
            } else if (arg == "--scene") {
                args.scenePath = requireValue("--scene");
            } else if (arg == "--width") {
                args.width = static_cast<uint32_t>(std::max(1, std::stoi(requireValue("--width"))));
            } else if (arg == "--height") {
                args.height = static_cast<uint32_t>(std::max(1, std::stoi(requireValue("--height"))));
            } else {
                positional.push_back(arg);
            }
        }

        if (!positional.empty()) {
            args.pointCloudPath = positional[0];
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
            if (firstCamera.hasPinholeIntrinsics != 0u && firstCamera.fy > 0.0f && firstCamera.height > 0) {
                orbit.fovyDegrees = glm::degrees(
                    2.0f * std::atan(static_cast<float>(firstCamera.height) / (2.0f * firstCamera.fy)));
            } else {
                orbit.fovyDegrees = firstCamera.fovy;
            }
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
        sensor.framebuffer = sycl::malloc_device<Pale::float4>(pixelCount, queue);
        sensor.outputFramebuffer = sycl::malloc_device<sycl::uchar4>(pixelCount, queue);
        sensor.ldrFramebuffer = sycl::malloc_device<float>(pixelCount * 4u, queue);
        sensor.depthDistortionBuffer = sycl::malloc_device<float>(pixelCount, queue);
        sensor.depthDistortionAdjointBuffer = sycl::malloc_device<float>(pixelCount, queue);
        sensor.visibilityWeightedOpacityBuffer = sycl::malloc_device<float>(pixelCount, queue);
        sensor.intraSlabDepthBuffer = sycl::malloc_device<float>(pixelCount, queue);
        sensor.intraSlabDepthAdjointBuffer = sycl::malloc_device<float>(pixelCount, queue);
        sensor.intraSlabDepthActiveSlabCountBuffer = sycl::malloc_device<std::uint32_t>(pixelCount, queue);
        sensor.curvatureScaleBuffer = sycl::malloc_device<float>(pixelCount, queue);
        sensor.curvatureScaleAdjointBuffer = sycl::malloc_device<float>(pixelCount, queue);
        sensor.curvatureScaleActiveSlabCountBuffer = sycl::malloc_device<std::uint32_t>(pixelCount, queue);
        sensor.curvaturePrimitiveIndexBuffer = sycl::malloc_device<std::uint32_t>(pixelCount, queue);
        sensor.medianDepthBuffer = sycl::malloc_device<float>(pixelCount, queue);
        sensor.meanDepthBuffer = sycl::malloc_device<float>(pixelCount, queue);
        sensor.medianDepthAdjointBuffer = sycl::malloc_device<float>(pixelCount, queue);
        sensor.medianWorldPositionBuffer = sycl::malloc_device<Pale::float4>(pixelCount, queue);
        sensor.visibleNormalBuffer = sycl::malloc_device<Pale::float4>(pixelCount, queue);
        sensor.normalFromDepthBuffer = sycl::malloc_device<Pale::float4>(pixelCount, queue);
        sensor.normalFromDepthAdjointBuffer = sycl::malloc_device<Pale::float4>(pixelCount, queue);
        sensor.visibleNormalAdjointBuffer = sycl::malloc_device<Pale::float4>(pixelCount, queue);

        if (!sensor.framebuffer || !sensor.outputFramebuffer || !sensor.ldrFramebuffer ||
            !sensor.intraSlabDepthBuffer || !sensor.intraSlabDepthAdjointBuffer ||
            !sensor.intraSlabDepthActiveSlabCountBuffer || !sensor.curvatureScaleBuffer ||
            !sensor.curvatureScaleAdjointBuffer || !sensor.curvatureScaleActiveSlabCountBuffer ||
            !sensor.curvaturePrimitiveIndexBuffer) {
            throw std::runtime_error("Failed to allocate realtime sensor framebuffers");
        }
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
        queue.memset(sensor.visibilityWeightedOpacityBuffer, 0, pixelCount * sizeof(float));
        queue.memset(sensor.intraSlabDepthBuffer, 0, pixelCount * sizeof(float));
        queue.memset(sensor.intraSlabDepthAdjointBuffer, 0, pixelCount * sizeof(float));
        queue.memset(sensor.intraSlabDepthActiveSlabCountBuffer, 0, pixelCount * sizeof(std::uint32_t));
        queue.memset(sensor.curvatureScaleBuffer, 0, pixelCount * sizeof(float));
        queue.memset(sensor.curvatureScaleAdjointBuffer, 0, pixelCount * sizeof(float));
        queue.memset(sensor.curvatureScaleActiveSlabCountBuffer, 0, pixelCount * sizeof(std::uint32_t));
        queue.fill(sensor.curvaturePrimitiveIndexBuffer, UINT32_MAX, pixelCount);
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
        freeDevicePtr(queue, sensor.outputFramebuffer);
        freeDevicePtr(queue, sensor.ldrFramebuffer);
        freeDevicePtr(queue, sensor.depthDistortionBuffer);
        freeDevicePtr(queue, sensor.depthDistortionAdjointBuffer);
        freeDevicePtr(queue, sensor.visibilityWeightedOpacityBuffer);
        freeDevicePtr(queue, sensor.intraSlabDepthBuffer);
        freeDevicePtr(queue, sensor.intraSlabDepthAdjointBuffer);
        freeDevicePtr(queue, sensor.intraSlabDepthActiveSlabCountBuffer);
        freeDevicePtr(queue, sensor.curvatureScaleBuffer);
        freeDevicePtr(queue, sensor.curvatureScaleAdjointBuffer);
        freeDevicePtr(queue, sensor.curvatureScaleActiveSlabCountBuffer);
        freeDevicePtr(queue, sensor.curvaturePrimitiveIndexBuffer);
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
        const Pale::SensorGPU& sensor) {
        const std::size_t pixelCount =
            static_cast<std::size_t>(sensor.width) * static_cast<std::size_t>(sensor.height);
        if (pixelCount == 0u) {
            return;
        }

        // Debug each source with unit loss weight while retaining the exact
        // production normalizations used by training.
        queue.fill(
            sensor.depthDistortionAdjointBuffer,
            1.0f / static_cast<float>(pixelCount),
            pixelCount);
        queue.fill(sensor.curvatureScaleAdjointBuffer, 0.0f, pixelCount);
        queue.fill(sensor.medianDepthAdjointBuffer, 0.0f, pixelCount);

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
        settings.enableAdjointDirectLight = true;
        settings.pointGeometrySupportRadius = 0.00f;
        settings.pointGeometryReconstructionLength = 0.0f;
        settings.pointGeometryRayOffsetMultiplier = 1.0f;
        settings.pointGeometryCoverageScale = 1.0f;
        settings.pointGeometryMinimumContributors = 1u;
        settings.pointGeometryDebugShowAlbedo = false;
        return settings;
    }

    ImVec2 fitImageSize(ImVec2 available, uint32_t width, uint32_t height) {
        if (width == 0 || height == 0 || available.x <= 1.0f || available.y <= 1.0f) {
            return ImVec2(1.0f, 1.0f);
        }
        const float imageAspect = static_cast<float>(width) / static_cast<float>(height);
        float drawWidth = available.x;
        float drawHeight = drawWidth / imageAspect;
        if (drawHeight > available.y) {
            drawHeight = available.y;
            drawWidth = drawHeight * imageAspect;
        }
        return ImVec2(std::max(drawWidth, 1.0f), std::max(drawHeight, 1.0f));
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
            std::clamp(windowWidth, 640, 1920),
            std::clamp(windowHeight, 480, 1200),
        };
    }

    [[nodiscard]] float chooseGridSpacing(float extent) {
        const float rawSpacing = std::max(extent / 80.0f, 0.001f);
        const float magnitude = std::pow(10.0f, std::floor(std::log10(rawSpacing)));
        const float normalized = rawSpacing / magnitude;
        if (normalized <= 1.0f) {
            return magnitude;
        }
        if (normalized <= 2.0f) {
            return 2.0f * magnitude;
        }
        if (normalized <= 5.0f) {
            return 5.0f * magnitude;
        }
        return 10.0f * magnitude;
    }

    struct GridLineStyle {
        glm::vec3 color{1.0f};
        float alpha = 1.0f;
        float thickness = 1.0f;
    };

    [[nodiscard]] uint8_t channelToByte(float value) {
        return static_cast<uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
    }

    void blendPixel(std::vector<uint8_t>& rgba, std::size_t pixelIndex, const GridLineStyle& style, float coverage) {
        const float alpha = std::clamp(style.alpha * coverage, 0.0f, 1.0f);
        if (alpha <= 0.0f) {
            return;
        }

        const std::size_t baseIndex = pixelIndex * 4u;
        const float invAlpha = 1.0f - alpha;
        rgba[baseIndex + 0u] = channelToByte(
            static_cast<float>(rgba[baseIndex + 0u]) / 255.0f * invAlpha + style.color.r * alpha);
        rgba[baseIndex + 1u] = channelToByte(
            static_cast<float>(rgba[baseIndex + 1u]) / 255.0f * invAlpha + style.color.g * alpha);
        rgba[baseIndex + 2u] = channelToByte(
            static_cast<float>(rgba[baseIndex + 2u]) / 255.0f * invAlpha + style.color.b * alpha);
        rgba[baseIndex + 3u] = 255u;
    }

    void blendGridSample(
        std::vector<uint8_t>& rgba,
        std::size_t pixelIndex,
        float distanceToLine,
        float worldPerPixel,
        const GridLineStyle& style) {
        const float antialiasWidth = std::max(worldPerPixel, 1.0e-6f);
        const float halfWidth = std::max(style.thickness * 0.5f, 0.5f) * antialiasWidth;
        const float coverage = std::clamp((halfWidth + antialiasWidth - distanceToLine) / antialiasWidth, 0.0f, 1.0f);
        blendPixel(rgba, pixelIndex, style, coverage);
    }

    [[nodiscard]] const GridLineStyle& gridLineStyleForIndex(
        int index,
        const GridLineStyle& axisLine,
        const GridLineStyle& majorLine,
        const GridLineStyle& minorLine) {
        if (index == 0) {
            return axisLine;
        }
        if (index % 10 == 0) {
            return majorLine;
        }
        return minorLine;
    }

    void shadeViewportBackground(
        std::vector<uint8_t>& rgba,
        const OrbitCamera& orbit,
        const SceneBounds& bounds,
        uint32_t renderWidth,
        uint32_t renderHeight,
        bool showViewportGrid) {
        const float extent = std::max(
            10.0f,
            std::max(bounds.radius * 2.5f, orbit.distance * 2.5f));
        const float spacing = chooseGridSpacing(extent);
        const GridLineStyle minorLine{{0.29f, 0.29f, 0.29f}, 0.42f, 1.0f};
        const GridLineStyle majorLine{{0.38f, 0.38f, 0.38f}, 0.57f, 1.15f};
        const GridLineStyle xAxisLine{{0.75f, 0.29f, 0.29f}, 0.82f, 1.8f};
        const GridLineStyle yAxisLine{{0.32f, 0.59f, 0.32f}, 0.82f, 1.8f};
        const glm::mat4 cameraFromWorld = glm::inverse(orbit.viewMatrix());
        const glm::mat3 worldFromCameraDirection{cameraFromWorld};
        const glm::vec3 cameraPosition = glm::vec3(cameraFromWorld[3]);
        const glm::vec3 cameraForward = glm::normalize(orbit.target - cameraPosition);
        const float width = static_cast<float>(std::max(renderWidth, 1u));
        const float height = static_cast<float>(std::max(renderHeight, 1u));
        const float fy = 0.5f * height / std::tan(0.5f * glm::radians(orbit.fovyDegrees));
        const float fx = fy * (width / height);
        const std::size_t pixelCount =
            static_cast<std::size_t>(renderWidth) * static_cast<std::size_t>(renderHeight);
        const uint8_t backgroundR = channelToByte(kBlenderViewportBackground.x);
        const uint8_t backgroundG = channelToByte(kBlenderViewportBackground.y);
        const uint8_t backgroundB = channelToByte(kBlenderViewportBackground.z);

        for (int y = 0; y < static_cast<int>(renderHeight); ++y) {
            for (uint32_t x = 0; x < renderWidth; ++x) {
                const std::size_t pixelIndex =
                    static_cast<std::size_t>(y) * static_cast<std::size_t>(renderWidth) + static_cast<std::size_t>(x);
                if (pixelIndex >= pixelCount) {
                    continue;
                }

                const std::size_t baseIndex = pixelIndex * 4u;
                rgba[baseIndex + 0u] = backgroundR;
                rgba[baseIndex + 1u] = backgroundG;
                rgba[baseIndex + 2u] = backgroundB;
                rgba[baseIndex + 3u] = 255u;
                if (!showViewportGrid) {
                    continue;
                }

                const float u = static_cast<float>(x) + 0.5f;
                const float v = static_cast<float>(y) + 0.5f;
                const float vFlipped = height - v;
                const float ndcX = 2.0f * u / width - 1.0f;
                const float ndcY = 2.0f * vFlipped / height - 1.0f;
                const glm::vec3 cameraDirection = glm::normalize(glm::vec3{
                    ndcX * (0.5f * width) / fx,
                    ndcY * (0.5f * height) / fy,
                    -1.0f,
                });
                const glm::vec3 rayDirection = glm::normalize(worldFromCameraDirection * cameraDirection);
                if (std::abs(rayDirection.z) <= 1.0e-6f) {
                    continue;
                }

                const float t = -cameraPosition.z / rayDirection.z;
                if (t <= orbit.nearClip) {
                    continue;
                }

                const glm::vec3 hit = cameraPosition + rayDirection * t;
                const float forwardDepth = glm::dot(hit - cameraPosition, cameraForward);
                if (forwardDepth <= orbit.nearClip) {
                    continue;
                }

                const float worldPerPixel =
                    std::max(2.0f * std::tan(0.5f * glm::radians(orbit.fovyDegrees)) * forwardDepth / height, 1.0e-6f);
                const int xIndex = static_cast<int>(std::round(hit.x / spacing));
                const int yIndex = static_cast<int>(std::round(hit.y / spacing));
                const float xDistance = std::abs(hit.x - static_cast<float>(xIndex) * spacing);
                const float yDistance = std::abs(hit.y - static_cast<float>(yIndex) * spacing);

                blendGridSample(
                    rgba,
                    pixelIndex,
                    xDistance,
                    worldPerPixel,
                    gridLineStyleForIndex(xIndex, yAxisLine, majorLine, minorLine));
                blendGridSample(
                    rgba,
                    pixelIndex,
                    yDistance,
                    worldPerPixel,
                    gridLineStyleForIndex(yIndex, xAxisLine, majorLine, minorLine));
            }
        }
    }

    void composeViewportPixels(
        const std::vector<uint8_t>& renderPixels,
        const OrbitCamera& orbit,
        const SceneBounds& bounds,
        uint32_t renderWidth,
        uint32_t renderHeight,
        bool showViewportGrid,
        bool forceRenderOpacity,
        std::vector<uint8_t>& displayPixels) {
        const std::size_t pixelCount =
            static_cast<std::size_t>(renderWidth) * static_cast<std::size_t>(renderHeight);
        if (renderPixels.size() < pixelCount * 4u) {
            return;
        }

        displayPixels.assign(pixelCount * 4u, 0u);
        shadeViewportBackground(
            displayPixels,
            orbit,
            bounds,
            renderWidth,
            renderHeight,
            showViewportGrid);

        for (std::size_t pixelIndex = 0; pixelIndex < pixelCount; ++pixelIndex) {
            const std::size_t baseIndex = pixelIndex * 4u;
            const bool hasRenderContribution =
                renderPixels[baseIndex + 3u] > 0u ||
                renderPixels[baseIndex + 0u] > 0u ||
                renderPixels[baseIndex + 1u] > 0u ||
                renderPixels[baseIndex + 2u] > 0u;
            if (!hasRenderContribution) {
                continue;
            }

            const float alpha =
                forceRenderOpacity ? 1.0f : static_cast<float>(renderPixels[baseIndex + 3u]) / 255.0f;
            const float invAlpha = 1.0f - alpha;
            displayPixels[baseIndex + 0u] = channelToByte(
                static_cast<float>(renderPixels[baseIndex + 0u]) / 255.0f * alpha +
                static_cast<float>(displayPixels[baseIndex + 0u]) / 255.0f * invAlpha);
            displayPixels[baseIndex + 1u] = channelToByte(
                static_cast<float>(renderPixels[baseIndex + 1u]) / 255.0f * alpha +
                static_cast<float>(displayPixels[baseIndex + 1u]) / 255.0f * invAlpha);
            displayPixels[baseIndex + 2u] = channelToByte(
                static_cast<float>(renderPixels[baseIndex + 2u]) / 255.0f * alpha +
                static_cast<float>(displayPixels[baseIndex + 2u]) / 255.0f * invAlpha);
            displayPixels[baseIndex + 3u] = 255u;
        }
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
        std::vector<uint8_t>& displayPixels) {
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
            if (!std::isfinite(value) || value <= 0.0f) {
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
            if (!std::isfinite(value) || value <= 0.0f) {
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

    void displayLinearRgbaAsSrgb(
        const std::vector<float>& linearRgba,
        uint32_t renderWidth,
        uint32_t renderHeight,
        std::vector<uint8_t>& displayPixels) {
        const std::size_t pixelCount =
            static_cast<std::size_t>(renderWidth) * static_cast<std::size_t>(renderHeight);
        displayPixels.assign(pixelCount * 4u, 0u);
        if (linearRgba.size() < pixelCount * 4u) {
            return;
        }
        for (std::size_t pixelIndex = 0u; pixelIndex < pixelCount; ++pixelIndex) {
            const std::size_t baseIndex = pixelIndex * 4u;
            displayPixels[baseIndex + 0u] = channelToByte(linearToSrgb(linearRgba[baseIndex + 0u]));
            displayPixels[baseIndex + 1u] = channelToByte(linearToSrgb(linearRgba[baseIndex + 1u]));
            displayPixels[baseIndex + 2u] = channelToByte(linearToSrgb(linearRgba[baseIndex + 2u]));
            displayPixels[baseIndex + 3u] = 255u;
        }
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

        const float safeThreshold =
            std::isfinite(splitThreshold) && splitThreshold > 1.0e-12f
                ? splitThreshold
                : 1.0f;
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
            const glm::vec3 color = value >= safeThreshold
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
            // 2=position-gradient split, 3=curvature-violation split.
            glm::vec3 color{0.40f};
            if (origin == 1u) {
                color = glm::vec3(0.15f, 0.78f, 0.30f);
            } else if (origin == 2u) {
                color = glm::vec3(0.18f, 0.48f, 1.00f);
            } else if (origin == 3u) {
                color = glm::vec3(0.82f, 0.26f, 0.92f);
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
            if (isPlyPath(path)) {
                dropState->pendingPlyPath = path;
                dropState->hasPendingPlyPath = true;
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
        const AppArgs args = parseArgs(argc, argv);
        const std::filesystem::path assetsDirectory =
            std::filesystem::absolute(args.assetsDir).lexically_normal();
        const std::filesystem::path repositoryRoot = assetsDirectory.parent_path();
        const std::filesystem::path optimizationOutputDirectory =
            repositoryRoot / "python" / "OptimizationOutput";
        std::filesystem::current_path(assetsDirectory);

        Pale::Log::init(spdlog::level::level_enum::info);
        glfwSetErrorCallback(glfwErrorCallback);

        Pale::AssetManager assetManager{256};
        registerAssetLoaders(assetManager);

        std::shared_ptr<Pale::Scene> scene =
            loadSceneWithPointCloud(assetManager, args.scenePath, args.pointCloudPath);

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
        std::filesystem::path currentScenePath = args.scenePath;
        std::filesystem::path currentPointCloudPath = args.pointCloudPath;
        std::array<char, 1024> pointCloudPathBuffer{};
        copyPathToBuffer(currentPointCloudPath, pointCloudPathBuffer);
        std::string pointCloudStatus =
            "Loaded " + std::to_string(buildProducts.points.size()) + " surfels";
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
        Pale::PathTracer tracer(queue, settings);
        Pale::CurvatureDensificationStats curvatureDensificationStats =
            Pale::makeCurvatureDensificationStatsForScene(queue, buildProducts);
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
        bool forceRenderOpacity = false;
        bool showViewportGrid = false;
        ViewImageMode viewImageMode = ViewImageMode::Rendered;
        ScalarColorMap scalarColorMap = ScalarColorMap::Viridis;
        float curvatureViolationDisplayThreshold = 5.0f;
        float positionWhatIfDisplayThreshold = 0.0f;
        float positionWhatIfReferenceThreshold = 0.0f;
        std::filesystem::path positionWhatIfThresholdPointCloudPath;
        float positionRadianceBiasStrength = 1.0f;
        float positionRadianceBiasMinWeight = 0.5f;
        float positionRadianceBiasMaxWeight = 10.0f;
        constexpr float kPositionRadianceBiasFloor = 1.0e-3f;
        int primitiveAgeColdAfterIterations = 1000;
        // Viewer-only diagnostic default. Optimization keeps its renderer-side
        // debug allocations disabled unless explicitly requested there.
        bool regularizerPrimitiveGradientMapsEnabled = true;
        bool ssimDebugMapsEnabled = false;
        float viewerSsimWeight = 0.2f;
        int viewerSsimWindowSize = 11;
        float viewerSsimSigma = 1.5f;
        int selectedLightIndex = 0;
        bool showLightGizmo = true;
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
        double lastViewerAdjointMs = 0.0;
        float lastViewerAdjointLoss = 0.0f;
        std::string viewerAdjointStatus = "Adjoint profiling is off";
        double lastRegularizerGradientMapMs = 0.0;
        double lastSsimDebugMapMs = 0.0;
        std::vector<uint8_t> renderPixels;
        std::vector<uint8_t> pixels;
        DebugDisplayBuffers debugDisplayBuffers;
        SsimTargetCache ssimTargetCache;
        OrbitCamera displayedOrbit = orbit;
        SceneBounds displayedBounds = bounds;
        CameraSource displayedCameraSource = cameraSource;
        Pale::CameraGPU displayedCamera = orbit.makeGpuCamera(renderWidth, renderHeight);
        uint32_t displayedRenderWidth = renderWidth;
        uint32_t displayedRenderHeight = renderHeight;
        Texture2D texture;
        bool showProfilingWindow = true;
        bool timerProfilingEnabled = true;
        bool gpuCounterProfilingEnabled = false;
        bool dockLayoutInitialized = false;
        Pale::RenderProfilingCounters lastProfilingCounters{};
        std::vector<Pale::ScopedTimerRecord> lastTimerRecords;
        Pale::RenderProfilingCounters* deviceProfilingCounters =
            sycl::malloc_device<Pale::RenderProfilingCounters>(1, queue);
        if (deviceProfilingCounters == nullptr) {
            throw std::runtime_error("Failed to allocate render profiling counters");
        }
        queue.memset(deviceProfilingCounters, 0, sizeof(Pale::RenderProfilingCounters)).wait();
        Pale::PointGradients viewerAdjointGradients{};
        Pale::PointGradients viewerDepthRegularizerGradients{};
        Pale::PointGradients viewerNormalRegularizerGradients{};
        Pale::PointGradients viewerIntraSlabRegularizerGradients{};

        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }

        const char* glslVersion = "#version 130";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        const std::array<int, 2> initialWindowSize =
            initialViewerWindowSize(renderWidth, renderHeight);
        GLFWwindow* window = glfwCreateWindow(
            initialWindowSize[0],
            initialWindowSize[1],
            "Pale Realtime Viewer",
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

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        ImGui::StyleColorsDark();
        ImGui::GetStyle().Colors[ImGuiCol_WindowBg] = kBlenderViewportBackground;
        ImGui::GetStyle().Colors[ImGuiCol_ChildBg] = kBlenderViewportBackground;
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init(glslVersion);

        auto rebuildSceneGpu = [&]() {
            buildProducts = Pale::SceneBuild::build(scene, assetAccessor, buildOptions);
            Pale::SceneUpload::uploadOrReallocate(buildProducts, sceneGpu, queue);
            if (curvatureDensificationStats.numPoints != buildProducts.points.size()) {
                Pale::freeCurvatureDensificationStats(queue, curvatureDensificationStats);
                curvatureDensificationStats =
                    Pale::makeCurvatureDensificationStatsForScene(queue, buildProducts);
            }
            sceneGpu.profileCounters =
                gpuCounterProfilingEnabled ? deviceProfilingCounters : nullptr;
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

            std::vector<Pale::Entity> pointCloudEntities;
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
            currentPointCloudPath = requestedPath;
            copyPathToBuffer(currentPointCloudPath, pointCloudPathBuffer);
            if (currentPointCloudPath.has_parent_path()) {
                plyBrowserDirectory = currentPointCloudPath.parent_path();
            }

            pointCloudStatus = "Loaded " + std::to_string(surfelCount) + " surfels";
            cameraDirty = true;
            if (!keepLatestOptimizationMode) {
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
            orbit = preservedOrbit;
            orbit.farClip = std::max(orbit.farClip, std::max(1000.0f, bounds.radius * 20.0f));
            cameraSource = preservedCameraSource;
            cameraDirty = true;

            if (!keepLatestOptimizationMode) {
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

        auto refreshLatestOptimizationPointCloud = [&]() {
            const std::optional<std::filesystem::path> latestPointCloud =
                findLatestOptimizationPointCloud(optimizationOutputDirectory);
            if (!latestPointCloud) {
                pointCloudStatus =
                    "No optimization point snapshot found in " + optimizationOutputDirectory.string();
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
                pointCloudStatus = "Press R or Load latest run PLY before using snapshot navigation";
                return;
            }

            bool switchedToNewLatestDirectory = false;
            const std::optional<std::filesystem::path> latestPointCloud =
                findLatestOptimizationPointCloud(optimizationOutputDirectory);
            if (latestPointCloud) {
                const std::filesystem::path latestPointsDirectory = latestPointCloud->parent_path();
                if (!equivalentPaths(latestPointsDirectory, latestOptimizationPointsDirectory)) {
                    latestOptimizationPointsDirectory = latestPointsDirectory;
                    switchedToNewLatestDirectory = true;
                }
            }

            std::vector<PointCloudSnapshot> snapshots =
                listOptimizationPointCloudSnapshots(latestOptimizationPointsDirectory);
            if (snapshots.empty()) {
                pointCloudStatus = "No optimization snapshots found in " + latestOptimizationPointsDirectory.string();
                return;
            }

            std::optional<std::size_t> newLatestSnapshotIndex;
            if (switchedToNewLatestDirectory && latestPointCloud) {
                const auto latestIterator = std::find_if(
                    snapshots.begin(),
                    snapshots.end(),
                    [&](const PointCloudSnapshot& snapshot) {
                        return equivalentPaths(snapshot.path, *latestPointCloud);
                    });
                newLatestSnapshotIndex = latestIterator == snapshots.end()
                                             ? snapshots.size() - 1u
                                             : static_cast<std::size_t>(
                                                 std::distance(snapshots.begin(), latestIterator));
                latestOptimizationSnapshotIndex = *newLatestSnapshotIndex;
            }

            if (switchedToNewLatestDirectory && direction > 0 && latestPointCloud) {
                const std::size_t latestIndex = newLatestSnapshotIndex.value_or(snapshots.size() - 1u);
                if (replaceOptimizationPointCloud(snapshots[latestIndex].path, true)) {
                    latestOptimizationMode = true;
                    latestOptimizationSnapshots = std::move(snapshots);
                    latestOptimizationSnapshotIndex = latestIndex;
                    pointCloudStatus =
                        "Latest optimization snapshot " +
                        std::to_string(latestOptimizationSnapshotIndex + 1u) + "/" +
                        std::to_string(latestOptimizationSnapshots.size()) + ": " +
                        currentPointCloudPath.filename().string();
                }
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
                                               newLatestSnapshotIndex.value_or(latestOptimizationSnapshotIndex),
                                               snapshots.size() - 1u)
                                           : static_cast<std::size_t>(std::distance(snapshots.begin(), currentIterator));

            if (direction < 0 && currentIndex == 0u) {
                if (switchedToNewLatestDirectory && replaceOptimizationPointCloud(snapshots[currentIndex].path, true)) {
                    latestOptimizationMode = true;
                    pointCloudStatus = "Already at earliest optimization snapshot";
                }
                latestOptimizationSnapshots = std::move(snapshots);
                latestOptimizationSnapshotIndex = 0u;
                pointCloudStatus = "Already at earliest optimization snapshot";
                return;
            }
            if (direction > 0 && currentIndex + 1u >= snapshots.size()) {
                latestOptimizationSnapshots = std::move(snapshots);
                latestOptimizationSnapshotIndex = snapshots.size() - 1u;
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
            if (latestOptimizationMode && !latestOptimizationPointsDirectory.empty()) {
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

        auto ensureDebugDisplayBuffer = [&](ViewImageMode mode) -> bool {
            if (!hasSensor || displayedRenderWidth == 0u || displayedRenderHeight == 0u) {
                return false;
            }
            if (sensor.width != displayedRenderWidth || sensor.height != displayedRenderHeight) {
                return false;
            }

            debugDisplayBuffers.prepareFor(displayedRenderWidth, displayedRenderHeight);
            const std::size_t pixelCount =
                static_cast<std::size_t>(displayedRenderWidth) * static_cast<std::size_t>(displayedRenderHeight);

            const auto ensureVisiblePrimitiveIndices = [&]() -> bool {
                if (!sensor.curvaturePrimitiveIndexBuffer) {
                    return false;
                }
                if (!debugDisplayBuffers.visiblePrimitiveIndicesValid) {
                    debugDisplayBuffers.visiblePrimitiveIndices =
                        Pale::downloadUint32Buffer(
                            queue, sensor.curvaturePrimitiveIndexBuffer, pixelCount);
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
                // surfel directly instead of borrowing the curvature map.
                const auto camera = sensor.camera;
                auto previewScene = sceneGpu;
                previewScene.profileCounters = nullptr;
                auto& hostIndices = debugDisplayBuffers.positionPrimitiveIndices;
                hostIndices.resize(pixelCount);
                uint32_t* output = sycl::malloc_device<uint32_t>(pixelCount, queue);
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
                            ? hit.primitiveIndex : UINT32_MAX;
                    }).wait_and_throw();
                    queue.memcpy(hostIndices.data(), output, pixelCount * sizeof(uint32_t)).wait_and_throw();
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
                case ViewImageMode::DepthNormal:
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
                    if (!debugDisplayBuffers.intraSlabDepthValid) {
                        debugDisplayBuffers.intraSlabDepth =
                            Pale::downloadFloatBuffer(queue, sensor.intraSlabDepthBuffer, pixelCount);
                        debugDisplayBuffers.intraSlabDepthValid = true;
                    }
                    return true;
                case ViewImageMode::CurvatureScale:
                    if (!debugDisplayBuffers.curvatureScaleValid) {
                        debugDisplayBuffers.curvatureScale =
                            Pale::downloadFloatBuffer(queue, sensor.curvatureScaleBuffer, pixelCount);
                        debugDisplayBuffers.curvatureScaleValid = true;
                    }
                    return true;
                case ViewImageMode::CurvaturePrimitiveScore:
                    if (!debugDisplayBuffers.curvaturePrimitiveScoreValid) {
                        if (!ensureVisiblePrimitiveIndices()) {
                            return false;
                        }
                        const std::vector<float> violationSums =
                            Pale::downloadFloatBuffer(
                                queue,
                                curvatureDensificationStats.violationSum,
                                curvatureDensificationStats.numPoints);
                        const std::vector<uint32_t> violationCounts =
                            Pale::downloadUint32Buffer(
                                queue,
                                curvatureDensificationStats.violationCount,
                                curvatureDensificationStats.numPoints);

                        std::vector<float> primitiveScores(
                            curvatureDensificationStats.numPoints, -1.0f);
                        debugDisplayBuffers.curvatureObservedPrimitiveScores.clear();
                        debugDisplayBuffers.curvatureObservedPrimitiveScores.reserve(
                            curvatureDensificationStats.numPoints);
                        debugDisplayBuffers.curvatureObservedPrimitiveCount = 0u;
                        debugDisplayBuffers.curvaturePrimitiveScoreMax = 0.0f;
                        for (std::size_t primitiveIndex = 0u;
                             primitiveIndex < curvatureDensificationStats.numPoints;
                             ++primitiveIndex) {
                            const uint32_t observationCount = violationCounts[primitiveIndex];
                            if (observationCount == 0u ||
                                !std::isfinite(violationSums[primitiveIndex])) {
                                continue;
                            }
                            const float score = violationSums[primitiveIndex] /
                                static_cast<float>(observationCount);
                            if (!std::isfinite(score) || score < 0.0f) {
                                continue;
                            }
                            primitiveScores[primitiveIndex] = score;
                            debugDisplayBuffers.curvatureObservedPrimitiveScores.push_back(score);
                            ++debugDisplayBuffers.curvatureObservedPrimitiveCount;
                            debugDisplayBuffers.curvaturePrimitiveScoreMax = std::max(
                                debugDisplayBuffers.curvaturePrimitiveScoreMax, score);
                        }

                        debugDisplayBuffers.curvaturePrimitiveScore.assign(pixelCount, -1.0f);
                        for (std::size_t pixelIndex = 0u;
                             pixelIndex < pixelCount;
                             ++pixelIndex) {
                            const uint32_t primitiveIndex =
                                debugDisplayBuffers.visiblePrimitiveIndices[pixelIndex];
                            if (primitiveIndex < primitiveScores.size()) {
                                debugDisplayBuffers.curvaturePrimitiveScore[pixelIndex] =
                                    primitiveScores[primitiveIndex];
                            }
                        }
                        debugDisplayBuffers.curvaturePrimitiveScoreValid = true;
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
                        debugDisplayBuffers.positionPrimitiveRadianceBiasAvailable = false;

                        const std::optional<Pale::AssetHandle> pointCloudHandle =
                            firstPointCloudHandle(scene);
                        const std::shared_ptr<Pale::PointAsset> pointCloudAsset =
                            pointCloudHandle ? assetAccessor.getPointCloud(*pointCloudHandle) : nullptr;
                        if (pointCloudAsset) {
                            std::vector<float> primitiveSignals;
                            std::vector<std::uint32_t> primitiveSampleCounts;
                            std::vector<float> primitiveThresholds;
                            std::vector<float> primitiveRadianceRms;
                            std::vector<float> primitiveBaseThresholds;
                            primitiveSignals.reserve(sceneGpu.pointCount);
                            primitiveSampleCounts.reserve(sceneGpu.pointCount);
                            primitiveThresholds.reserve(sceneGpu.pointCount);
                            primitiveRadianceRms.reserve(sceneGpu.pointCount);
                            primitiveBaseThresholds.reserve(sceneGpu.pointCount);

                            bool completeMetadata = true;
                            bool completeRadianceBiasMetadata = true;
                            for (const Pale::PointGeometry& geometry : pointCloudAsset->points) {
                                const std::size_t pointCount = geometry.positions.size();
                                if (geometry.densificationPositionSignals.size() != pointCount ||
                                    geometry.densificationPositionSampleCounts.size() != pointCount ||
                                    geometry.densificationPositionThresholds.size() != pointCount) {
                                    completeMetadata = false;
                                    break;
                                }
                                if (geometry.densificationPositionRadianceRms.size() != pointCount ||
                                    geometry.densificationPositionBaseThresholds.size() != pointCount) {
                                    completeRadianceBiasMetadata = false;
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
                                if (completeRadianceBiasMetadata) {
                                    primitiveRadianceRms.insert(
                                        primitiveRadianceRms.end(),
                                        geometry.densificationPositionRadianceRms.begin(),
                                        geometry.densificationPositionRadianceRms.end());
                                    primitiveBaseThresholds.insert(
                                        primitiveBaseThresholds.end(),
                                        geometry.densificationPositionBaseThresholds.begin(),
                                        geometry.densificationPositionBaseThresholds.end());
                                }
                            }

                            completeMetadata = completeMetadata &&
                                primitiveSignals.size() == sceneGpu.pointCount;
                            if (completeMetadata && !primitiveThresholds.empty()) {
                                completeRadianceBiasMetadata = completeRadianceBiasMetadata &&
                                    primitiveRadianceRms.size() == sceneGpu.pointCount &&
                                    primitiveBaseThresholds.size() == sceneGpu.pointCount;
                                const std::vector<float>& selectedThresholds =
                                    completeRadianceBiasMetadata
                                        ? primitiveBaseThresholds
                                        : primitiveThresholds;
                                completeMetadata = std::all_of(
                                        selectedThresholds.begin(),
                                        selectedThresholds.end(),
                                        [](float threshold) {
                                            return std::isfinite(threshold) && threshold > 0.0f;
                                        });
                                if (completeMetadata) {
                                    // Legacy PLYs can store per-primitive biased
                                    // thresholds too. Preserve their ratios with
                                    // a common reference for the display slider.
                                    auto sortedThresholds = selectedThresholds;
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
                                        positionWhatIfDisplayThreshold = savedThreshold;
                                        positionWhatIfReferenceThreshold = savedThreshold;
                                        positionWhatIfThresholdPointCloudPath =
                                            currentPointCloudPath;
                                    }
                                    debugDisplayBuffers.positionObservedPrimitiveScores.reserve(
                                        primitiveSignals.size());
                                    float medianRadiance = kPositionRadianceBiasFloor;
                                    if (completeRadianceBiasMetadata) {
                                        std::vector<float> observedRadiance;
                                        observedRadiance.reserve(primitiveRadianceRms.size());
                                        for (std::size_t primitiveIndex = 0u;
                                             primitiveIndex < primitiveRadianceRms.size();
                                             ++primitiveIndex) {
                                            const float radiance = primitiveRadianceRms[primitiveIndex];
                                            if (primitiveSampleCounts[primitiveIndex] > 0u &&
                                                std::isfinite(radiance) && radiance > 0.0f) {
                                                observedRadiance.push_back(std::max(
                                                    radiance, kPositionRadianceBiasFloor));
                                            }
                                        }
                                        if (!observedRadiance.empty()) {
                                            debugDisplayBuffers.positionPrimitiveRadianceBiasAvailable = true;
                                            std::sort(observedRadiance.begin(), observedRadiance.end());
                                            const std::size_t middle = observedRadiance.size() / 2u;
                                            medianRadiance = observedRadiance[middle];
                                            if (observedRadiance.size() % 2u == 0u) {
                                                medianRadiance = 0.5f * (
                                                    observedRadiance[middle - 1u] +
                                                    observedRadiance[middle]);
                                            }
                                        }
                                    }
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
                                        signal *= savedThreshold / selectedThresholds[primitiveIndex];
                                        if (completeRadianceBiasMetadata &&
                                            primitiveRadianceRms[primitiveIndex] > 0.0f) {
                                            const float brightness = std::max(
                                                primitiveRadianceRms[primitiveIndex],
                                                kPositionRadianceBiasFloor);
                                            const float weight = std::clamp(
                                                std::pow(
                                                    medianRadiance / brightness,
                                                    positionRadianceBiasStrength),
                                                positionRadianceBiasMinWeight,
                                                positionRadianceBiasMaxWeight);
                                            signal *= weight;
                                        }
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
                    if (!regularizerPrimitiveGradientMapsEnabled) {
                        return false;
                    }
                    if (!debugDisplayBuffers.depthPositionGradientValid) {
                        debugDisplayBuffers.depthPositionGradient =
                            makePositionGradientMap(viewerDepthRegularizerGradients);
                        debugDisplayBuffers.depthPositionGradientValid = true;
                    }
                    return true;
                case ViewImageMode::NormalPositionGradient:
                    if (!regularizerPrimitiveGradientMapsEnabled) {
                        return false;
                    }
                    if (!debugDisplayBuffers.normalPositionGradientValid) {
                        debugDisplayBuffers.normalPositionGradient =
                            makePositionGradientMap(viewerNormalRegularizerGradients);
                        debugDisplayBuffers.normalPositionGradientValid = true;
                    }
                    return true;
                case ViewImageMode::IntraSlabPositionGradient:
                    if (!regularizerPrimitiveGradientMapsEnabled) {
                        return false;
                    }
                    if (!debugDisplayBuffers.intraSlabPositionGradientValid) {
                        debugDisplayBuffers.intraSlabPositionGradient =
                            makePositionGradientMap(viewerIntraSlabRegularizerGradients);
                        debugDisplayBuffers.intraSlabPositionGradientValid = true;
                    }
                    return true;
                case ViewImageMode::SsimTarget:
                case ViewImageMode::RgbHalfMse:
                case ViewImageMode::SsimIndex:
                case ViewImageMode::Dssim:
                case ViewImageMode::RgbObjectiveGradient:
                    return ssimDebugMapsEnabled && debugDisplayBuffers.ssimDebugValid;
                case ViewImageMode::Rendered:
                    return true;
            }

            return false;
        };

        auto updateDisplayTexture = [&]() {
            if (renderPixels.empty()) {
                return;
            }

            if (viewImageMode != ViewImageMode::Rendered) {
                if (!ensureDebugDisplayBuffer(viewImageMode)) {
                    return;
                }

                switch (viewImageMode) {
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
                    case ViewImageMode::DepthNormal:
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
                        colorizeScalarBuffer(
                            debugDisplayBuffers.intraSlabDepth,
                            displayedRenderWidth,
                            displayedRenderHeight,
                            false,
                            true,
                            scalarColorMap,
                            pixels);
                        break;
                    case ViewImageMode::CurvatureScale:
                        colorizeScalarBuffer(
                            debugDisplayBuffers.curvatureScale,
                            displayedRenderWidth,
                            displayedRenderHeight,
                            false,
                            true,
                            scalarColorMap,
                            pixels);
                        break;
                    case ViewImageMode::CurvaturePrimitiveScore:
                        colorizeThresholdedPrimitiveScores(
                            debugDisplayBuffers.curvaturePrimitiveScore,
                            displayedRenderWidth,
                            displayedRenderHeight,
                            curvatureViolationDisplayThreshold,
                            scalarColorMap,
                            pixels);
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
                    case ViewImageMode::SsimTarget:
                        displayLinearRgbaAsSrgb(
                            debugDisplayBuffers.ssimTargetLinearRgba,
                            displayedRenderWidth,
                            displayedRenderHeight,
                            pixels);
                        break;
                    case ViewImageMode::RgbHalfMse:
                        colorizeScalarBuffer(
                            debugDisplayBuffers.rgbHalfMse,
                            displayedRenderWidth,
                            displayedRenderHeight,
                            false,
                            true,
                            scalarColorMap,
                            pixels);
                        break;
                    case ViewImageMode::SsimIndex:
                        colorizeScalarBufferFixedRange(
                            debugDisplayBuffers.ssimIndex,
                            displayedRenderWidth,
                            displayedRenderHeight,
                            0.0f,
                            1.0f,
                            scalarColorMap,
                            pixels);
                        break;
                    case ViewImageMode::Dssim:
                        colorizeScalarBuffer(
                            debugDisplayBuffers.dssim,
                            displayedRenderWidth,
                            displayedRenderHeight,
                            false,
                            true,
                            scalarColorMap,
                            pixels);
                        break;
                    case ViewImageMode::RgbObjectiveGradient:
                        colorizeScalarBuffer(
                            debugDisplayBuffers.rgbObjectiveGradient,
                            displayedRenderWidth,
                            displayedRenderHeight,
                            false,
                            true,
                            scalarColorMap,
                            pixels);
                        break;
                    case ViewImageMode::Rendered:
                        break;
                }
                texture.update(pixels, displayedRenderWidth, displayedRenderHeight);
                return;
            }

            if (displayedCameraSource == CameraSource::Viewport) {
                composeViewportPixels(
                    renderPixels,
                    displayedOrbit,
                    displayedBounds,
                    displayedRenderWidth,
                    displayedRenderHeight,
                    showViewportGrid,
                    forceRenderOpacity,
                    pixels);
            } else {
                pixels = renderPixels;
            }
            texture.update(pixels, displayedRenderWidth, displayedRenderHeight);
        };

        auto setViewImageMode = [&](ViewImageMode nextMode) {
            if (viewImageMode == nextMode) {
                return;
            }

            // These maps need a fresh visible-slab identity/curvature output.
            const bool needsSlabSearch = requiresVisibleSlabSearch(nextMode);
            if (needsSlabSearch || requiresVisibleSlabSearch(viewImageMode)) {
                renderRequested = true;
            }
            viewImageMode = nextMode;
            if (needsSlabSearch) {
                return;
            }
            updateDisplayTexture();
        };

        const auto isViewImageModeAvailable = [&](ViewImageMode mode) {
            return !(isRegularizerGradientView(mode) &&
                     !regularizerPrimitiveGradientMapsEnabled) &&
                   !(isSsimDebugView(mode) && !ssimDebugMapsEnabled);
        };

        const auto cycleViewImageMode = [&](int direction) {
            const std::size_t modeCount = kViewImageModeShortcutOrder.size();
            std::size_t currentIndex = 0u;
            for (std::size_t index = 0u; index < modeCount; ++index) {
                if (kViewImageModeShortcutOrder[index] == viewImageMode) {
                    currentIndex = index;
                    break;
                }
            }

            for (std::size_t offset = 1u; offset <= modeCount; ++offset) {
                const std::size_t nextIndex = direction > 0
                    ? (currentIndex + offset) % modeCount
                    : (currentIndex + modeCount - (offset % modeCount)) % modeCount;
                const ViewImageMode nextMode = kViewImageModeShortcutOrder[nextIndex];
                if (isViewImageModeAvailable(nextMode)) {
                    setViewImageMode(nextMode);
                    return;
                }
            }
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
            Pale::freeGradientsForScene(queue, viewerDepthRegularizerGradients);
            Pale::freeGradientsForScene(queue, viewerNormalRegularizerGradients);
            Pale::freeGradientsForScene(queue, viewerIntraSlabRegularizerGradients);
        };

        auto ensureViewerRegularizerGradients = [&]() {
            const bool hasUsableGradients =
                viewerDepthRegularizerGradients.gradPosition != nullptr &&
                viewerNormalRegularizerGradients.gradPosition != nullptr &&
                viewerIntraSlabRegularizerGradients.gradPosition != nullptr &&
                viewerDepthRegularizerGradients.numPoints == sceneGpu.pointCount &&
                viewerNormalRegularizerGradients.numPoints == sceneGpu.pointCount &&
                viewerIntraSlabRegularizerGradients.numPoints == sceneGpu.pointCount;
            if (hasUsableGradients) {
                return;
            }

            freeViewerRegularizerGradients();
            Pale::SceneBuild::BuildProducts gradientBuildProducts = renderBuildProducts;
            gradientBuildProducts.cameraGPUs.clear();
            gradientBuildProducts.cameraGPUs.push_back(sensor.camera);
            viewerDepthRegularizerGradients =
                Pale::makeGradientsForScene(queue, gradientBuildProducts, nullptr);
            viewerNormalRegularizerGradients =
                Pale::makeGradientsForScene(queue, gradientBuildProducts, nullptr);
            viewerIntraSlabRegularizerGradients =
                Pale::makeGradientsForScene(queue, gradientBuildProducts, nullptr);
        };

        auto runViewerRegularizerGradientPass = [&](std::vector<Pale::SensorGPU>& renderSensors) {
            if (!regularizerPrimitiveGradientMapsEnabled ||
                !hasSensor || sceneGpu.pointCount == 0u) {
                return;
            }

            const auto start = std::chrono::steady_clock::now();
            ensureViewerRegularizerGradients();
            prepareRealtimeViewerSurfaceRegularizerAdjoints(queue, sensor);

            Pale::PathTracerSettings regularizerSettings = settings;
            regularizerSettings.depthDistortionWeight = 1.0f;
            regularizerSettings.normalConsistencyWeight = 1.0f;
            regularizerSettings.visibilityWeightedOpacityRegularizerWeight = 0.0f;
            regularizerSettings.intraSlabDepthRegularizerWeight = 1.0f;
            regularizerSettings.curvatureScaleRegularizerWeight = 0.0f;
            tracer.getSettings() = regularizerSettings;
            Pale::PointGradients unusedOpacityGradients{};
            Pale::PointGradients unusedCurvatureGradients{};
            tracer.renderSurfaceRegularizersBackward(
                renderSensors,
                viewerDepthRegularizerGradients,
                viewerNormalRegularizerGradients,
                unusedOpacityGradients,
                viewerIntraSlabRegularizerGradients,
                unusedCurvatureGradients,
                nullptr);
            tracer.getSettings() = settings;
            queue.wait();

            const auto stop = std::chrono::steady_clock::now();
            lastRegularizerGradientMapMs =
                std::chrono::duration<double, std::milli>(stop - start).count();
        };

        auto runViewerAdjointPass = [&](std::vector<Pale::SensorGPU>& renderSensors) {
            if (!runAdjointEveryRender && !runAdjointNextRender) {
                return;
            }
            runAdjointNextRender = false;

            if (!hasSensor || sceneGpu.pointCount == 0u) {
                viewerAdjointStatus = "Adjoint skipped: no active sensor or surfels";
                lastViewerAdjointMs = 0.0;
                lastViewerAdjointLoss = 0.0f;
                return;
            }

            viewerAdjointSamplesPerPixel = std::clamp(viewerAdjointSamplesPerPixel, 1, 64);
            viewerAdjointBounces = std::clamp(viewerAdjointBounces, 1, 8);

            const auto start = std::chrono::steady_clock::now();
            {
                Pale::ScopedTimer timer("Viewer adjoint source setup", spdlog::level::debug);
                prepareRealtimeViewerRgbLossAdjointSource(
                    queue, sensor, lastViewerAdjointLoss);
            }
            {
                Pale::ScopedTimer timer("Viewer adjoint gradient buffer setup", spdlog::level::debug);
                ensureViewerAdjointGradients();
            }

            Pale::PathTracerSettings adjointSettings = settings;
            adjointSettings.maxAdjointBounces = static_cast<uint32_t>(viewerAdjointBounces);
            adjointSettings.adjointSamplesPerPixel =
                static_cast<uint32_t>(viewerAdjointSamplesPerPixel);
            adjointSettings.enableAdjointDirectLight = viewerAdjointDirectLight;
            adjointSettings.numAdjointPathShadowRays =
                std::max(adjointSettings.numAdjointPathShadowRays, 1u);

            tracer.getSettings() = adjointSettings;
            {
                Pale::ScopedTimer timer("Viewer adjoint pass total", spdlog::level::debug);
                tracer.renderBackward(renderSensors, viewerAdjointGradients, nullptr);
            }
            tracer.getSettings() = settings;
            queue.wait();

            const auto stop = std::chrono::steady_clock::now();
            lastViewerAdjointMs =
                std::chrono::duration<double, std::milli>(stop - start).count();
            viewerAdjointStatus =
                "Last adjoint: " + std::to_string(lastViewerAdjointMs) +
                " ms, loss " + std::to_string(lastViewerAdjointLoss);
        };

        auto renderNow = [&]() {
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

            Pale::PathTracerSettings activeTracerSettings = settings;
            activeTracerSettings.computeCurvatureDiagnostics =
                requiresVisibleSlabSearch(viewImageMode);
            tracer.setCurvatureDensificationStats(
                viewImageMode == ViewImageMode::CurvaturePrimitiveScore
                    ? &curvatureDensificationStats : nullptr);
            if (runAdjointEveryRender || runAdjointNextRender) {
                viewerAdjointSamplesPerPixel = std::clamp(viewerAdjointSamplesPerPixel, 1, 64);
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
            // Adjoint profiling reuses the raw framebuffer as its source. Capture
            // the primal linear RGB first so SSIM diagnostics describe the forward render.
            std::vector<float> ssimRenderedLinearRgba;
            if (ssimDebugMapsEnabled && cameraSource == CameraSource::SceneXml) {
                ssimRenderedLinearRgba = Pale::downloadSensorRGBARAW(queue, sensor);
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
            if (ssimDebugMapsEnabled) {
                const auto ssimStart = std::chrono::steady_clock::now();
                if (cameraSource != CameraSource::SceneXml) {
                    ssimTargetCache.status =
                        "SSIM maps require a Scene XML camera aligned with its training target";
                } else {
                    const std::string cameraName(
                        camera.name, strnlen(camera.name, sizeof(camera.name)));
                    const auto targetPath = targetPngForOptimizationPointCloud(
                        currentPointCloudPath, cameraName);
                    if (!targetPath) {
                        ssimTargetCache.status =
                            "No render_target_" + cameraName +
                            ".png beside this optimization point cloud";
                    } else if (ensureLinearSsimTarget(
                                   *targetPath,
                                   renderWidth,
                                   renderHeight,
                                   ssimTargetCache)) {
                        debugDisplayBuffers.prepareFor(renderWidth, renderHeight);
                        computeSsimDebugBuffers(
                            ssimRenderedLinearRgba,
                            ssimTargetCache.linearRgba,
                            renderWidth,
                            renderHeight,
                            viewerSsimWeight,
                            viewerSsimWindowSize,
                            viewerSsimSigma,
                            debugDisplayBuffers);
                    }
                }
                const auto ssimStop = std::chrono::steady_clock::now();
                lastSsimDebugMapMs =
                    std::chrono::duration<double, std::milli>(ssimStop - ssimStart).count();
            }
            updateDisplayTexture();
            cameraDirty = false;
            renderRequested = false;
        };

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }

            if (dropState.hasPendingPlyPath) {
                const std::filesystem::path droppedPath = dropState.pendingPlyPath;
                dropState.hasPendingPlyPath = false;
                handleDroppedPly(droppedPath);
            }

            if ((autoRender && cameraDirty) || renderRequested) {
                renderNow();
            }

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            if (ImGui::BeginMainMenuBar()) {
                if (ImGui::BeginMenu("View")) {
                    ImGui::MenuItem("Profiling", nullptr, &showProfilingWindow);
                    ImGui::EndMenu();
                }
                ImGui::EndMainMenuBar();
            }

            ImGuiViewport* mainViewport = ImGui::GetMainViewport();
            const ImGuiID dockspaceId = ImHashStr("PaleRealtimeViewerDockSpace");
            if (!dockLayoutInitialized) {
                dockLayoutInitialized = true;
                ImGui::DockBuilderRemoveNode(dockspaceId);
                ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace);
                ImGui::DockBuilderSetNodePos(dockspaceId, mainViewport->WorkPos);
                ImGui::DockBuilderSetNodeSize(dockspaceId, mainViewport->WorkSize);

                ImGuiID leftDockId = 0;
                ImGuiID renderDockId = 0;
                ImGui::DockBuilderSplitNode(
                    dockspaceId,
                    ImGuiDir_Left,
                    kSidebarDockFraction,
                    &leftDockId,
                    &renderDockId);

                ImGuiID profilingDockId = 0;
                ImGuiID rendererSettingsDockId = 0;
                ImGui::DockBuilderSplitNode(
                    leftDockId,
                    ImGuiDir_Down,
                    0.40f,
                    &profilingDockId,
                    &rendererSettingsDockId);

                ImGui::DockBuilderDockWindow("Renderer settings", rendererSettingsDockId);
                ImGui::DockBuilderDockWindow("Profiling", profilingDockId);
                ImGui::DockBuilderDockWindow("Render", renderDockId);
                ImGui::DockBuilderFinish(dockspaceId);
            }
            ImGui::DockSpaceOverViewport(
                dockspaceId,
                mainViewport,
                ImGuiDockNodeFlags_PassthruCentralNode);
            ImGuizmo::BeginFrame();

            if (!io.WantTextInput &&
                !io.KeyCtrl &&
                !io.KeyAlt &&
                !io.KeySuper) {
                if (ImGui::IsKeyPressed(ImGuiKey_R, false)) {
                    refreshLatestOptimizationPointCloud();
                }

                if (ImGui::IsKeyPressed(ImGuiKey_F, false)) {
                    jumpToOptimizationSnapshotBoundary(false);
                }

                if (ImGui::IsKeyPressed(ImGuiKey_L, false)) {
                    jumpToOptimizationSnapshotBoundary(true);
                }

                if (ImGui::IsKeyPressed(ImGuiKey_N, false)) {
                    stepLatestOptimizationSnapshot(-1, kSnapshotIterationStep);
                }

                if (ImGui::IsKeyPressed(ImGuiKey_M, false)) {
                    stepLatestOptimizationSnapshot(1, kSnapshotIterationStep);
                }

                if (ImGui::IsKeyDown(ImGuiKey_LeftArrow)) {
                    stepLatestOptimizationSnapshot(-1);
                }

                if (ImGui::IsKeyDown(ImGuiKey_RightArrow)) {
                    stepLatestOptimizationSnapshot(1);
                }

                if (ImGui::IsKeyPressed(ImGuiKey_UpArrow, false)) {
                    stepLatestOptimizationSnapshot(1);
                }

                if (ImGui::IsKeyPressed(ImGuiKey_DownArrow, false)) {
                    stepLatestOptimizationSnapshot(-1);
                }

                constexpr std::array<ImGuiKey, 9> viewImageModeShortcutKeys = {
                    ImGuiKey_1,
                    ImGuiKey_2,
                    ImGuiKey_3,
                    ImGuiKey_4,
                    ImGuiKey_5,
                    ImGuiKey_6,
                    ImGuiKey_7,
                    ImGuiKey_8,
                    ImGuiKey_9,
                };

                for (std::size_t shortcutIndex = 0;
                     shortcutIndex < viewImageModeShortcutKeys.size();
                     ++shortcutIndex) {
                    if (ImGui::IsKeyPressed(viewImageModeShortcutKeys[shortcutIndex], false)) {
                        setViewImageMode(kViewImageModeShortcutOrder[shortcutIndex]);
                        break;
                    }
                }

                const bool nextViewModePressed =
                    ImGui::IsKeyPressed(ImGuiKey_KeypadAdd, false) ||
                    (io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_Equal, false));
                const bool previousViewModePressed =
                    ImGui::IsKeyPressed(ImGuiKey_KeypadSubtract, false) ||
                    ImGui::IsKeyPressed(ImGuiKey_Minus, false);
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

            ImGui::Begin("Renderer settings");
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
            if (ImGui::Button("Load latest run PLY")) {
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
            std::filesystem::path selectedPlyPath;
            if (drawPlyBrowser(
                    "Select Point Cloud PLY",
                    "PointCloudPlyFiles",
                    plyBrowserOpen,
                    plyBrowserDirectory,
                    selectedPlyPath)) {
                replacePointCloud(selectedPlyPath);
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
            std::filesystem::path selectedMeshPath;
            if (drawPlyBrowser(
                    "Select Mesh PLY",
                    "MeshPlyFiles",
                    meshBrowserOpen,
                    meshBrowserDirectory,
                    selectedMeshPath)) {
                loadRuntimeMesh(selectedMeshPath);
            }
            ImGui::Separator();

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

            int distortionDepthMode = settings.depthDistortionWorldSpace ? 0 : 1;
            const char* distortionDepthModes[] = {"World distance", "Normalized depth (legacy)"};
            if (ImGui::Combo("Depth distortion", &distortionDepthMode, distortionDepthModes, 2)) {
                settings.depthDistortionWorldSpace = distortionDepthMode == 0;
                renderRequested = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Use the same depth mode as the training run. World distance preserves "
                    "the contribution of distant surfaces in both loss and gradient previews.");
            }

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

            if (viewImageMode == ViewImageMode::DepthDistortion ||
                viewImageMode == ViewImageMode::DepthPositionGradient) {
                ImGui::TextDisabled("Preview uses unit loss weight; colors rescale to each frame");
            }

            ImGui::Text("Resolution: %u x %u", renderWidth, renderHeight);
            if (ImGui::BeginCombo("Display", viewImageModeLabel(viewImageMode))) {
                for (const ViewImageMode candidateMode : kViewImageModeShortcutOrder) {
                    if (!isViewImageModeAvailable(candidateMode)) {
                        continue;
                    }
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
            ImGui::TextDisabled("1-9 direct   +/- cycle display modes");
            if (viewImageMode == ViewImageMode::MeanDepth ||
                viewImageMode == ViewImageMode::MedianDepth ||
                viewImageMode == ViewImageMode::DepthDistortion ||
                viewImageMode == ViewImageMode::IntraSlabDepth ||
                viewImageMode == ViewImageMode::CurvatureScale ||
                viewImageMode == ViewImageMode::CurvaturePrimitiveScore ||
                viewImageMode == ViewImageMode::PositionPrimitiveScore ||
                viewImageMode == ViewImageMode::DepthPositionGradient ||
                viewImageMode == ViewImageMode::NormalPositionGradient ||
                viewImageMode == ViewImageMode::IntraSlabPositionGradient ||
                viewImageMode == ViewImageMode::RgbHalfMse ||
                viewImageMode == ViewImageMode::SsimIndex ||
                viewImageMode == ViewImageMode::Dssim ||
                viewImageMode == ViewImageMode::RgbObjectiveGradient) {
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
            if (viewImageMode == ViewImageMode::CurvaturePrimitiveScore) {
                if (ImGui::SliderFloat(
                        "Curvature split threshold",
                        &curvatureViolationDisplayThreshold,
                        1.0e-6f,
                        1.0e6f,
                        "%.4g",
                        ImGuiSliderFlags_Logarithmic)) {
                    curvatureViolationDisplayThreshold = std::max(
                        curvatureViolationDisplayThreshold, 1.0e-6f);
                    updateDisplayTexture();
                }

                ImGui::TextDisabled(
                    "Below: C_i / threshold colormap; magenta: at/above split boundary");

                const std::size_t splitCandidateCount = static_cast<std::size_t>(
                    std::count_if(
                        debugDisplayBuffers.curvatureObservedPrimitiveScores.begin(),
                        debugDisplayBuffers.curvatureObservedPrimitiveScores.end(),
                        [&](float score) {
                            return score >= curvatureViolationDisplayThreshold;
                        }));
                ImGui::Text(
                    "Observed: %zu   above threshold: %zu   max C_i: %.4g",
                    debugDisplayBuffers.curvatureObservedPrimitiveCount,
                    splitCandidateCount,
                    debugDisplayBuffers.curvaturePrimitiveScoreMax);
            }
            if (viewImageMode == ViewImageMode::PositionPrimitiveScore) {
                ImGui::SeparatorText("Densification radiance bias");
                ImGui::BeginDisabled(!debugDisplayBuffers.positionPrimitiveRadianceBiasAvailable);
                bool radianceBiasChanged = false;
                radianceBiasChanged |= ImGui::SliderFloat(
                    "Bias strength", &positionRadianceBiasStrength, 0.0f, 2.0f, "%.3f");
                radianceBiasChanged |= ImGui::SliderFloat(
                    "Minimum weight", &positionRadianceBiasMinWeight, 0.05f, 1.0f,
                    "%.3f", ImGuiSliderFlags_Logarithmic);
                radianceBiasChanged |= ImGui::SliderFloat(
                    "Maximum weight", &positionRadianceBiasMaxWeight, 1.0f, 20.0f,
                    "%.3f", ImGuiSliderFlags_Logarithmic);
                ImGui::EndDisabled();
                if (radianceBiasChanged) {
                    positionRadianceBiasStrength = std::max(positionRadianceBiasStrength, 0.0f);
                    positionRadianceBiasMinWeight = std::clamp(positionRadianceBiasMinWeight, 0.05f, 1.0f);
                    positionRadianceBiasMaxWeight = std::max(positionRadianceBiasMaxWeight, 1.0f);
                    debugDisplayBuffers.positionPrimitiveScoreValid = false;
                    updateDisplayTexture();
                }
                if (debugDisplayBuffers.positionPrimitiveRadianceBiasAvailable) {
                    ImGui::TextDisabled(
                        "Preview only; radiance floor: %.4g", kPositionRadianceBiasFloor);
                } else {
                    ImGui::TextWrapped(
                        "Bias adjustment is unavailable: saved radiance statistics are missing "
                        "or contain no positive values for sampled surfels. "
                        "Restart training with relative densification enabled and load a new snapshot. "
                        "Increasing bias cannot recover missing radiance measurements.");
                }
                if (!debugDisplayBuffers.positionPrimitiveMetadataAvailable) {
                    ImGui::TextWrapped(
                        "This PLY does not contain saved position densification statistics. "
                        "Run optimization again to generate a trustworthy visualization.");
                } else {
                    const float savedThreshold =
                        debugDisplayBuffers.positionPrimitiveSplitThreshold;
                    ImGui::Text(
                        debugDisplayBuffers.positionPrimitiveRadianceBiasAvailable
                            ? "Saved base threshold: %.9g"
                            : "Saved effective threshold: %.9g",
                        savedThreshold);
                    const float sliderMinimum = static_cast<float>(std::max(
                        static_cast<double>(std::numeric_limits<float>::min()),
                        static_cast<double>(savedThreshold) * 1.0e-3));
                    const float sliderMaximum = static_cast<float>(std::min(
                        static_cast<double>(std::numeric_limits<float>::max()),
                        static_cast<double>(savedThreshold) * 1.0e3));
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
                    ImGui::TextDisabled(
                        debugDisplayBuffers.positionPrimitiveRadianceBiasAvailable
                            ? "Below: biased G_i / selected base threshold; magenta: at/above boundary"
                            : "Below: saved G_i / selected threshold; magenta: at/above boundary");

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
                    "gray: initial/unknown  green: clone  blue: position split  purple: curvature split");
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
            if (cameraSource == CameraSource::Viewport) {
                if (ImGui::DragFloat("FOV", &orbit.fovyDegrees, 0.25f, 5.0f, 160.0f, "%.1f deg")) {
                    cameraDirty = true;
                }
                if (ImGui::Checkbox("Force opacity = 1", &forceRenderOpacity)) {
                    updateDisplayTexture();
                }
                if (ImGui::Checkbox("Show grid", &showViewportGrid)) {
                    updateDisplayTexture();
                }
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

            ImGui::Separator();
            if (ImGui::CollapsingHeader("Lights")) {
                ImGui::Text("Area lights: %zu", areaLights.size());
                if (areaLights.empty()) {
                    ImGui::TextWrapped("No editable mesh area lights in the scene");
                } else {
                    const std::string selectedLightName = entityName(selectedLight);
                    if (ImGui::BeginCombo("Selected light", selectedLightName.c_str())) {
                        for (std::size_t lightIndex = 0; lightIndex < areaLights.size(); ++lightIndex) {
                            const bool selected = static_cast<int>(lightIndex) == selectedLightIndex;
                            const std::string label =
                                entityName(areaLights[lightIndex]) + "##Light" + std::to_string(lightIndex);
                            if (ImGui::Selectable(label.c_str(), selected)) {
                                selectedLightIndex = static_cast<int>(lightIndex);
                                selectedLight = areaLights[lightIndex];
                            }
                            if (selected) {
                                ImGui::SetItemDefaultFocus();
                            }
                        }
                        ImGui::EndCombo();
                    }

                    bool lightChanged = false;
                    auto& transform = selectedLight.getComponent<Pale::TransformComponent>();
                    if (ImGui::DragFloat3("Position", &transform.translation.x, 0.01f, -10000.0f, 10000.0f, "%.3f")) {
                        lightChanged = true;
                    }
                    glm::vec3 rotationEuler = transform.rotationEuler;
                    if (ImGui::DragFloat3("Rotation", &rotationEuler.x, 0.25f, -360.0f, 360.0f, "%.2f deg")) {
                        transform.setRotationEuler(rotationEuler);
                        lightChanged = true;
                    }
                    if (ImGui::DragFloat3("Scale", &transform.scale.x, 0.01f, 0.001f, 10000.0f, "%.3f")) {
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
                    if (ImGui::ColorEdit3("Emission color", &emissionColor.x)) {
                        areaLight.radiance = emissionColor;
                        if (lightMaterial) {
                            lightMaterial->baseColor = emissionColor;
                        }
                        lightChanged = true;
                    }
                    if (ImGui::DragFloat("Emission power", &emissionPower, 0.1f, 0.0f, 1000000.0f, "%.3f")) {
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

                    ImGui::Checkbox("Show light gizmo", &showLightGizmo);
                    if (ImGui::RadioButton("Move", lightGizmoOperation == ImGuizmo::TRANSLATE)) {
                        lightGizmoOperation = ImGuizmo::TRANSLATE;
                    }
                    ImGui::SameLine();
                    if (ImGui::RadioButton("Rotate", lightGizmoOperation == ImGuizmo::ROTATE)) {
                        lightGizmoOperation = ImGuizmo::ROTATE;
                    }
                    ImGui::SameLine();
                    if (ImGui::RadioButton("Scale", lightGizmoOperation == ImGuizmo::SCALE)) {
                        lightGizmoOperation = ImGuizmo::SCALE;
                    }
                    if (lightGizmoOperation != ImGuizmo::SCALE) {
                        if (ImGui::RadioButton("Local", lightGizmoMode == ImGuizmo::LOCAL)) {
                            lightGizmoMode = ImGuizmo::LOCAL;
                        }
                        ImGui::SameLine();
                        if (ImGui::RadioButton("World", lightGizmoMode == ImGuizmo::WORLD)) {
                            lightGizmoMode = ImGuizmo::WORLD;
                        }
                    }

                    if (lightChanged) {
                        rebuildSceneGpu();
                    }
                }

                ImGui::Separator();
                ImGui::Text("Surfel lights");
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
                            ImGui::Checkbox("Show surfel gizmo", &showSurfelGizmo);
                            if (showSurfelGizmo) {
                                if (ImGui::RadioButton("Move##surfel_gizmo", surfelGizmoOperation == ImGuizmo::TRANSLATE)) {
                                    surfelGizmoOperation = ImGuizmo::TRANSLATE;
                                }
                                ImGui::SameLine();
                                if (ImGui::RadioButton("Rotate##surfel_gizmo", surfelGizmoOperation == ImGuizmo::ROTATE)) {
                                    surfelGizmoOperation = ImGuizmo::ROTATE;
                                }
                                if (ImGui::RadioButton("Local##surfel_gizmo", surfelGizmoMode == ImGuizmo::LOCAL)) {
                                    surfelGizmoMode = ImGuizmo::LOCAL;
                                }
                                ImGui::SameLine();
                                if (ImGui::RadioButton("World##surfel_gizmo", surfelGizmoMode == ImGuizmo::WORLD)) {
                                    surfelGizmoMode = ImGuizmo::WORLD;
                                }
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
                            rebuildSceneGpu();
                        }
                    }
                }
            }

            ImGui::Separator();
            if (ImGui::CollapsingHeader("Renderer debug")) {
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

                if (ImGui::Checkbox(
                        "Regularizer primitive gradient maps",
                        &regularizerPrimitiveGradientMapsEnabled)) {
                    if (!regularizerPrimitiveGradientMapsEnabled) {
                        freeViewerRegularizerGradients();
                        if (viewImageMode == ViewImageMode::DepthPositionGradient ||
                            viewImageMode == ViewImageMode::NormalPositionGradient ||
                            viewImageMode == ViewImageMode::IntraSlabPositionGradient) {
                            setViewImageMode(ViewImageMode::Rendered);
                        }
                    }
                    renderRequested = true;
                }
                ImGui::TextDisabled(
                    "Viewer default; allocates 3 gradient sets and runs the surface adjoint");
                if (regularizerPrimitiveGradientMapsEnabled) {
                    ImGui::Text(
                        "Last regularizer gradient maps: %.3f ms (unit loss weights)",
                        lastRegularizerGradientMapMs);
                }

                ImGui::SeparatorText("SSIM diagnostics");
                if (ImGui::Checkbox("SSIM debug maps", &ssimDebugMapsEnabled)) {
                    if (!ssimDebugMapsEnabled) {
                        debugDisplayBuffers.releaseSsimDebug();
                        ssimTargetCache.invalidate();
                        ssimTargetCache.status = "SSIM debug maps are disabled";
                        if (isSsimDebugView(viewImageMode)) {
                            setViewImageMode(ViewImageMode::Rendered);
                        }
                    }
                    renderRequested = true;
                }
                ImGui::TextDisabled(
                    "Off by default; loads the saved target and computes CPU SSIM maps");
                if (ssimDebugMapsEnabled) {
                    bool ssimSettingsChanged = false;
                    ssimSettingsChanged |= ImGui::SliderFloat(
                        "SSIM weight", &viewerSsimWeight, 0.0f, 1.0f, "%.3f");
                    if (ImGui::SliderInt(
                            "SSIM window", &viewerSsimWindowSize, 1, 31)) {
                        if ((viewerSsimWindowSize & 1) == 0) {
                            viewerSsimWindowSize = std::min(viewerSsimWindowSize + 1, 31);
                        }
                        ssimSettingsChanged = true;
                    }
                    ssimSettingsChanged |= ImGui::SliderFloat(
                        "SSIM sigma", &viewerSsimSigma, 0.1f, 5.0f, "%.3f");
                    if (ssimSettingsChanged) {
                        viewerSsimWeight = std::clamp(viewerSsimWeight, 0.0f, 1.0f);
                        viewerSsimSigma = std::max(viewerSsimSigma, 0.1f);
                        renderRequested = true;
                    }
                    if (ImGui::Button("Reload SSIM target")) {
                        ssimTargetCache.invalidate();
                        renderRequested = true;
                    }
                    ImGui::TextWrapped("%s", ssimTargetCache.status.c_str());
                    if (debugDisplayBuffers.ssimDebugValid) {
                        ImGui::Text(
                            "RGB: combined %.6g   half-MSE %.6g",
                            debugDisplayBuffers.rgbObjectiveMean,
                            debugDisplayBuffers.rgbHalfMseMean);
                        ImGui::Text(
                            "SSIM %.6g   DSSIM %.6g   maps %.3f ms",
                            debugDisplayBuffers.ssimMean,
                            debugDisplayBuffers.dssimMean,
                            lastSsimDebugMapMs);
                    }
                }

                if (ImGui::CollapsingHeader("Adjoint profiling")) {
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

                    if (ImGui::DragInt("Adjoint SPP", &viewerAdjointSamplesPerPixel, 0.1f, 1, 64)) {
                        viewerAdjointSamplesPerPixel = std::clamp(viewerAdjointSamplesPerPixel, 1, 64);
                        tracerDirty = true;
                        if (runAdjointEveryRender) {
                            renderRequested = true;
                        }
                    }
                    if (ImGui::DragInt("Adjoint bounces", &viewerAdjointBounces, 0.1f, 1, 8)) {
                        viewerAdjointBounces = std::clamp(viewerAdjointBounces, 1, 8);
                        tracerDirty = true;
                        if (runAdjointEveryRender) {
                            renderRequested = true;
                        }
                    }
                    if (ImGui::Checkbox("Adjoint direct light", &viewerAdjointDirectLight) &&
                        runAdjointEveryRender) {
                        tracerDirty = true;
                        renderRequested = true;
                    }
                    ImGui::Text("Last adjoint: %.3f ms", lastViewerAdjointMs);
                    ImGui::Text("Adjoint RGB objective: %.6g", static_cast<double>(lastViewerAdjointLoss));
                    ImGui::TextWrapped("%s", viewerAdjointStatus.c_str());
                }

                if (ImGui::CollapsingHeader("Point BVH")) {
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
                    float localLayerDepthEpsilon = settings.rendererDebugLocalLayerDepthEpsilon;
                    if (ImGui::DragFloat(
                            "Local layer depth epsilon",
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

            ImGui::Separator();
            if (ImGui::CollapsingHeader("Surfel editor")) {
                const std::optional<Pale::AssetHandle> pointCloudHandle = firstPointCloudHandle(scene);
                const std::shared_ptr<Pale::PointAsset> pointCloudAsset =
                    pointCloudHandle ? assetAccessor.getPointCloud(*pointCloudHandle) : nullptr;

                if (!pointCloudAsset || countSurfels(*pointCloudAsset) == 0u) {
                    ImGui::TextWrapped("No editable surfels are loaded");
                    selectedSurfelEditorIndex = -1;
                } else {
                    const std::size_t surfelCount = countSurfels(*pointCloudAsset);
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

            ImGui::Separator();
            ImGui::Checkbox("Auto render", &autoRender);
            if (ImGui::Button("Render")) {
                renderRequested = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset view")) {
                orbit = makeInitialOrbitCamera(buildProducts, bounds);
                cameraDirty = true;
                renderRequested = true;
            }
            ImGui::Text("Last render: %.2f ms", lastRenderMs);
            ImGui::Text("Camera: %.3f %.3f %.3f",
                        orbit.position().x,
                        orbit.position().y,
                        orbit.position().z);
            ImGui::End();

            ImGui::Begin(
                "Render",
                nullptr,
                ImGuiWindowFlags_NoCollapse);

            const ImVec2 available = ImGui::GetContentRegionAvail();
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
                const ImVec2 imageSize =
                    cameraSource == CameraSource::Viewport
                        ? ImVec2(
                              std::max(available.x, 1.0f),
                              std::max(available.y, 1.0f))
                        : fitImageSize(available, texture.width, texture.height);
                ImGui::Image(toImTextureId(texture.id), imageSize);
                const bool imageHovered = ImGui::IsItemHovered();
                const ImVec2 imageMin = ImGui::GetItemRectMin();

                bool viewportGizmoHovered = false;
                bool viewportGizmoUsing = false;
                if (showLightGizmo && selectedLight && cameraSource == CameraSource::Viewport) {
                    auto& transform = selectedLight.getComponent<Pale::TransformComponent>();
                    glm::mat4 lightTransform = transform.getTransform();
                    glm::mat4 view = orbit.viewMatrix();
                    glm::mat4 projection = orbit.projectionMatrix(renderWidth, renderHeight);

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
                        transform.setTransform(lightTransform);
                        rebuildSceneGpu();
                    }
                    viewportGizmoHovered = viewportGizmoHovered || ImGuizmo::IsOver();
                    viewportGizmoUsing = viewportGizmoUsing || ImGuizmo::IsUsing() || ImGuizmo::IsUsingAny();
                    ImGuizmo::PopID();
                }

	                if (showSurfelGizmo && cameraSource == CameraSource::Viewport) {
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
                                glm::mat4 view = orbit.viewMatrix();
                                glm::mat4 projection = orbit.projectionMatrix(renderWidth, renderHeight);

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
	                                    rebuildSceneGpu();
	                                }
                                viewportGizmoHovered = viewportGizmoHovered || ImGuizmo::IsOver();
                                viewportGizmoUsing = viewportGizmoUsing || ImGuizmo::IsUsing() || ImGuizmo::IsUsingAny();
                                ImGuizmo::PopID();
	                        }
	                    }
	                }
                }

                if (cameraSource == CameraSource::Viewport) {
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
	                    viewportGizmoHovered || viewportGizmoUsing || viewportGizmoMouseCapture;
	                const ImVec2 imageMax{imageMin.x + imageSize.x, imageMin.y + imageSize.y};
	                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
	                    viewportPickArmed = imageHovered && !viewportCameraInputBlocked;
	                }
	                if (viewportPickArmed && ImGui::IsMouseDragging(ImGuiMouseButton_Left, io.MouseDragThreshold)) {
	                    viewportPickArmed = false;
	                }
	                if (viewportPickArmed && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
	                    const bool releaseInsideImage = ImGui::IsMouseHoveringRect(imageMin, imageMax, false);
	                    if (releaseInsideImage && !viewportCameraInputBlocked && displayedRenderWidth > 0u && displayedRenderHeight > 0u) {
	                        const ImVec2 mouse = ImGui::GetMousePos();
	                        const float normalizedX = std::clamp((mouse.x - imageMin.x) / std::max(imageSize.x, 1.0f), 0.0f, 0.999999f);
	                        const float normalizedY = std::clamp((mouse.y - imageMin.y) / std::max(imageSize.y, 1.0f), 0.0f, 0.999999f);
	                        const float pixelX = normalizedX * static_cast<float>(displayedRenderWidth);
	                        const float pixelY = normalizedY * static_cast<float>(displayedRenderHeight);
		                        if (const std::optional<int> pickedSurfelIndex = pickEditableSurfel(scene, assetAccessor, displayedCamera, pixelX, pixelY)) {
		                            selectedSurfelEditorIndex = *pickedSurfelIndex;
		                            surfelEditorStatus = "Picked surfel " + std::to_string(selectedSurfelEditorIndex);
		                        } else {
		                            selectedSurfelEditorIndex = -1;
		                            surfelEditorStatus = "No surfel selected";
		                        }
	                    }
	                    viewportPickArmed = false;
	                }
	                if (imageHovered && !viewportCameraInputBlocked) {
	                    const bool orbitDragging = ImGui::IsMouseDragging(ImGuiMouseButton_Left);
	                    const bool panDragging =
	                        ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ||
	                        ImGui::IsMouseDragging(ImGuiMouseButton_Right);
	                    const bool zooming = io.MouseWheel != 0.0f;
	                    if (cameraSource == CameraSource::SceneXml && (orbitDragging || panDragging || zooming)) {
	                        orbit = makeOrbitCameraFromSceneCamera(displayedCamera, bounds, orbit);
	                        cameraSource = CameraSource::Viewport;
	                        cameraDirty = true;
	                        renderRequested = true;
	                    }
	                    if (cameraSource == CameraSource::Viewport) {
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
            } else {
                ImGui::Text("No render yet");
            }
            ImGui::End();

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
        }

        if (hasSensor) {
            destroySensor(queue, sensor);
        }
        tracer.setCurvatureDensificationStats(nullptr);
        Pale::freeCurvatureDensificationStats(queue, curvatureDensificationStats);
        Pale::SceneUpload::freeBuffers(sceneGpu, queue);
        if (deviceProfilingCounters != nullptr) {
            sycl::free(deviceProfilingCounters, queue);
            deviceProfilingCounters = nullptr;
        }
        Pale::freeGradientsForScene(queue, viewerAdjointGradients);
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
