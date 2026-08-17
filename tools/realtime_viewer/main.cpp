#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cctype>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <ImGuizmo.h>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <sycl/sycl.hpp>

#include "Renderer/GPUDataStructures.h"
#include "Renderer/RenderPackage.h"
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
    };

    constexpr std::array<ViewImageMode, 6> kViewImageModeShortcutOrder = {
        ViewImageMode::Rendered,
        ViewImageMode::MedianDepth,
        ViewImageMode::DepthDistortion,
        ViewImageMode::MeanDepth,
        ViewImageMode::VisibleNormal,
        ViewImageMode::DepthNormal,
    };

    constexpr std::array<const char*, 6> kViewImageModeLabels = {
        "1 Rendered",
        "2 Median depth",
        "3 Depth distortion",
        "4 Mean depth",
        "5 Visible normal",
        "6 Depth normal",
    };

    int viewImageModeShortcutIndex(ViewImageMode mode) {
        for (std::size_t index = 0; index < kViewImageModeShortcutOrder.size(); ++index) {
            if (kViewImageModeShortcutOrder[index] == mode) {
                return static_cast<int>(index);
            }
        }

        return 0;
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
        std::vector<float> meanDepth;
        std::vector<float> medianDepth;
        std::vector<float> visibleNormal;
        std::vector<float> depthNormal;
        std::vector<float> depthDistortion;

        void invalidate() {
            width = 0;
            height = 0;
            meanDepthValid = false;
            medianDepthValid = false;
            visibleNormalValid = false;
            depthNormalValid = false;
            depthDistortionValid = false;
            meanDepth.clear();
            medianDepth.clear();
            visibleNormal.clear();
            depthNormal.clear();
            depthDistortion.clear();
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
            pitch -= delta.y * 0.005f;
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
            PointCloudSnapshot pointCloud;
            std::filesystem::file_time_type writeTime = std::filesystem::file_time_type::min();
        };

        std::error_code error;
        if (!std::filesystem::is_directory(optimizationOutputDirectory, error)) {
            return std::nullopt;
        }

        std::optional<RunCandidate> bestRun;
        for (const auto& runEntry : std::filesystem::directory_iterator(optimizationOutputDirectory, error)) {
            if (error) {
                break;
            }
            if (!runEntry.is_directory(error) || error) {
                error.clear();
                continue;
            }

            const std::filesystem::path runDirectory = runEntry.path();
            const std::filesystem::path pointsDirectory = runDirectory / "points";
            if (!std::filesystem::is_directory(pointsDirectory, error)) {
                error.clear();
                continue;
            }

            const std::vector<PointCloudSnapshot> snapshots = listOptimizationPointCloudSnapshots(pointsDirectory);
            if (snapshots.empty()) {
                continue;
            }
            const PointCloudSnapshot& bestPointCloud = snapshots.back();

            const std::filesystem::file_time_type runWriteTime = std::max(
                std::max(lastWriteTimeOrMin(runDirectory), lastWriteTimeOrMin(pointsDirectory)),
                bestPointCloud.writeTime);
            RunCandidate candidate{.pointCloud = bestPointCloud, .writeTime = runWriteTime};
            if (!bestRun ||
                candidate.writeTime > bestRun->writeTime ||
                (candidate.writeTime == bestRun->writeTime &&
                 candidate.pointCloud.path.filename().string() > bestRun->pointCloud.path.filename().string())) {
                bestRun = candidate;
            }
        }

        if (!bestRun) {
            return std::nullopt;
        }
        return bestRun->pointCloud.path;
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
        const auto pointCloudAsset = assetAccessor.getPointCloud(pointCloudAssetHandle);
        if (!pointCloudAsset || pointCloudAsset->points.empty()) {
            throw std::runtime_error("Point cloud failed to load or contains no point blocks: " + pointCloudPath.string());
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
        sensor.medianDepthBuffer = sycl::malloc_device<float>(pixelCount, queue);
        sensor.meanDepthBuffer = sycl::malloc_device<float>(pixelCount, queue);
        sensor.medianDepthAdjointBuffer = sycl::malloc_device<float>(pixelCount, queue);
        sensor.medianWorldPositionBuffer = sycl::malloc_device<Pale::float4>(pixelCount, queue);
        sensor.visibleNormalBuffer = sycl::malloc_device<Pale::float4>(pixelCount, queue);
        sensor.normalFromDepthBuffer = sycl::malloc_device<Pale::float4>(pixelCount, queue);
        sensor.normalFromDepthAdjointBuffer = sycl::malloc_device<Pale::float4>(pixelCount, queue);
        sensor.visibleNormalAdjointBuffer = sycl::malloc_device<Pale::float4>(pixelCount, queue);

        if (!sensor.framebuffer || !sensor.outputFramebuffer || !sensor.ldrFramebuffer) {
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
            static_cast<int>(std::lround(std::max(available, 1.0f))),
            16,
            4096));
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

        Pale::SceneBuild::BuildProducts buildProducts =
            Pale::SceneBuild::build(scene, assetAccessor, buildOptions);
        Pale::GPUSceneBuffers sceneGpu =
            Pale::SceneUpload::allocateAndUpload(buildProducts, queue);

        SceneBounds bounds = computeSceneBounds(buildProducts);
        OrbitCamera orbit = makeInitialOrbitCamera(buildProducts, bounds);
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
            if (!buildProducts.cameraGPUs.empty()) {
                renderWidth = buildProducts.cameraGPUs.front().width;
                renderHeight = buildProducts.cameraGPUs.front().height;
            } else {
                renderWidth = 800u;
                renderHeight = 600u;
            }
        }

        Pale::PathTracerSettings settings = makeDefaultSettings();
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
        bool forceRenderOpacity = false;
        bool showViewportGrid = false;
        ViewImageMode viewImageMode = ViewImageMode::Rendered;
        ScalarColorMap scalarColorMap = ScalarColorMap::Viridis;
        int selectedLightIndex = 0;
        bool showLightGizmo = true;
        ImGuizmo::OPERATION lightGizmoOperation = ImGuizmo::TRANSLATE;
        ImGuizmo::MODE lightGizmoMode = ImGuizmo::WORLD;
        int selectedSurfelLightIndex = 0;
        int candidateZeroPowerSurfelIndex = 0;
        float candidateSurfelPower = 1.0f;
        int selectedSurfelEditorIndex = 0;
        bool showSurfelGizmo = true;
        ImGuizmo::OPERATION surfelGizmoOperation = ImGuizmo::TRANSLATE;
        ImGuizmo::MODE surfelGizmoMode = ImGuizmo::WORLD;
        bool viewportGizmoMouseCapture = false;
        bool viewportPickArmed = false;
        std::string surfelLightStatus;
        std::string surfelEditorStatus;
        float exposure = 1.0f;
        float gamma = 1.0f;
        double lastRenderMs = 0.0;
        std::vector<uint8_t> renderPixels;
        std::vector<uint8_t> pixels;
        DebugDisplayBuffers debugDisplayBuffers;
        OrbitCamera displayedOrbit = orbit;
        SceneBounds displayedBounds = bounds;
        CameraSource displayedCameraSource = cameraSource;
        Pale::CameraGPU displayedCamera = orbit.makeGpuCamera(renderWidth, renderHeight);
        uint32_t displayedRenderWidth = renderWidth;
        uint32_t displayedRenderHeight = renderHeight;
        Texture2D texture;

        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }

        const char* glslVersion = "#version 130";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        GLFWwindow* window = glfwCreateWindow(1280, 720, "Pale Realtime Viewer", nullptr, nullptr);
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
        ImGui::StyleColorsDark();
        ImGui::GetStyle().Colors[ImGuiCol_WindowBg] = kBlenderViewportBackground;
        ImGui::GetStyle().Colors[ImGuiCol_ChildBg] = kBlenderViewportBackground;
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init(glslVersion);

        auto rebuildSceneGpu = [&]() {
            buildProducts = Pale::SceneBuild::build(scene, assetAccessor, buildOptions);
            Pale::SceneUpload::uploadOrReallocate(buildProducts, sceneGpu, queue);
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

            const Pale::AssetHandle pointCloudAssetHandle =
                importPathAsType(assetManager.registry(), requestedPath, Pale::AssetType::PointCloud);
            assetManager.invalidate(pointCloudAssetHandle);
            const auto pointCloudAsset = assetAccessor.getPointCloud(pointCloudAssetHandle);
            if (!pointCloudAsset) {
                pointCloudStatus = "Failed to load PLY: " + requestedPath.string();
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

            if (replacePointCloud(*latestPointCloud, true)) {
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

        auto stepLatestOptimizationSnapshot = [&](int direction) {
            if (!latestOptimizationMode || latestOptimizationPointsDirectory.empty()) {
                pointCloudStatus = "Press R or Load latest run PLY before using left/right snapshot navigation";
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
                if (replacePointCloud(snapshots[latestIndex].path, true)) {
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
                if (switchedToNewLatestDirectory && replacePointCloud(snapshots[currentIndex].path, true)) {
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

            const std::size_t nextIndex =
                direction < 0 ? currentIndex - 1u : currentIndex + 1u;
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

            viewImageMode = nextMode;
            updateDisplayTexture();
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
            clearSensor(queue, sensor);

            if (tracerDirty) {
                tracer.getSettings() = settings;
                tracer.setScene(sceneGpu, renderBuildProducts);
                tracerDirty = false;
            } else {
                tracer.getSettings() = settings;
            }

            std::vector<Pale::SensorGPU> renderSensors{sensor};
            const auto start = std::chrono::steady_clock::now();
            tracer.renderForward(renderSensors);
            const auto stop = std::chrono::steady_clock::now();
            lastRenderMs =
                std::chrono::duration<double, std::milli>(stop - start).count();

            renderPixels = Pale::downloadSensorRGBA(queue, sensor);
            debugDisplayBuffers.invalidate();
            displayedOrbit = orbit;
            displayedBounds = bounds;
            displayedCameraSource = cameraSource;
            displayedCamera = camera;
            displayedRenderWidth = renderWidth;
            displayedRenderHeight = renderHeight;
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
            ImGuizmo::BeginFrame();

            if (!io.WantTextInput &&
                !io.KeyCtrl &&
                !io.KeyAlt &&
                !io.KeySuper) {
                if (ImGui::IsKeyPressed(ImGuiKey_R, false)) {
                    refreshLatestOptimizationPointCloud();
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

                constexpr std::array<ImGuiKey, 6> viewImageModeShortcutKeys = {
                    ImGuiKey_1,
                    ImGuiKey_2,
                    ImGuiKey_3,
                    ImGuiKey_4,
                    ImGuiKey_5,
                    ImGuiKey_6,
                };

                for (std::size_t shortcutIndex = 0;
                     shortcutIndex < viewImageModeShortcutKeys.size();
                     ++shortcutIndex) {
                    if (ImGui::IsKeyPressed(viewImageModeShortcutKeys[shortcutIndex], false)) {
                        setViewImageMode(kViewImageModeShortcutOrder[shortcutIndex]);
                        break;
                    }
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

            ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(330.0f, static_cast<float>(io.DisplaySize.y)), ImGuiCond_Always);
            ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
            ImGui::TextWrapped("Scene: %s", args.scenePath.string().c_str());
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

            ImGui::Text("Resolution: %u x %u", renderWidth, renderHeight);
            int viewImageModeIndex = viewImageModeShortcutIndex(viewImageMode);
            if (ImGui::Combo(
                    "Display",
                    &viewImageModeIndex,
                    kViewImageModeLabels.data(),
                    static_cast<int>(kViewImageModeLabels.size()))) {
                setViewImageMode(kViewImageModeShortcutOrder[static_cast<std::size_t>(viewImageModeIndex)]);
            }
            if (viewImageMode == ViewImageMode::MeanDepth ||
                viewImageMode == ViewImageMode::MedianDepth ||
                viewImageMode == ViewImageMode::DepthDistortion) {
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
            if (ImGui::DragFloat("Gamma", &gamma, 0.01f, 0.1f, 5.0f, "%.3f")) {
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
            if (ImGui::CollapsingHeader("Renderer debug", ImGuiTreeNodeFlags_DefaultOpen)) {
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

                float localLayerDepthEpsilon = settings.rendererDebugLocalLayerDepthEpsilon;
                if (ImGui::DragFloat(
                        "LocalLayerDepthEpsilon",
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
                        "LocalLayerNormalCosineThreshold",
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
                        "kMaxSplatEventsPerRay",
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
                        "kMaxLocalSurfelHits",
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
                    selectedSurfelEditorIndex = 0;
                } else {
                    const std::size_t surfelCount = countSurfels(*pointCloudAsset);
                    selectedSurfelEditorIndex = std::clamp(
                        selectedSurfelEditorIndex,
                        0,
                        static_cast<int>(surfelCount - 1u));

                    ImGui::SetNextItemWidth(-1.0f);
                    if (ImGui::InputInt("Surfel index", &selectedSurfelEditorIndex)) {
                        selectedSurfelEditorIndex = std::clamp(
                            selectedSurfelEditorIndex,
                            0,
                            static_cast<int>(surfelCount - 1u));
                    }
                    ImGui::Text("Valid range: 0 - %zu", surfelCount - 1u);

                    std::size_t localSurfelIndex = static_cast<std::size_t>(selectedSurfelEditorIndex);
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
                        ImGui::TextWrapped("The selected surfel could not be resolved");
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

            ImGui::SetNextWindowPos(ImVec2(330.0f, 0.0f), ImGuiCond_Always);
            ImGui::SetNextWindowSize(
                ImVec2(std::max(1.0f, io.DisplaySize.x - 330.0f), static_cast<float>(io.DisplaySize.y)),
                ImGuiCond_Always);
            ImGui::Begin(
                "Render",
                nullptr,
                ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

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
	                    if (pointCloudAsset && editableSurfelCount > 0u) {
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

                                const glm::vec2 surfelScale =
                                    surfelIndex < pointGeometry.scales.size()
                                        ? pointGeometry.scales[surfelIndex]
                                        : glm::vec2(settings.pointGeometrySupportRadius);
                                const float gizmoScale = std::max(
                                    settings.pointGeometrySupportRadius,
                                    std::max(surfelScale.x, surfelScale.y));
                                const glm::mat4 localSurfelTransform =
                                    glm::translate(glm::mat4(1.0f), pointGeometry.positions[surfelIndex]) *
                                    glm::mat4_cast(normalizeQuaternionOrIdentity(pointGeometry.quat[surfelIndex])) *
                                    glm::scale(glm::mat4(1.0f), glm::vec3(std::max(gizmoScale, 0.001f)));
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
	                            surfelEditorStatus = "No surfel under cursor";
	                        }
	                    }
	                    viewportPickArmed = false;
	                }
	                if (cameraSource == CameraSource::Viewport && imageHovered && !viewportCameraInputBlocked) {
	                    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                        orbit.orbit(io.MouseDelta);
                        cameraDirty = true;
                    }
                    if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ||
                        ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
                        orbit.pan(io.MouseDelta);
                        cameraDirty = true;
                    }
                    if (io.MouseWheel != 0.0f) {
                        orbit.zoom(io.MouseWheel);
                        cameraDirty = true;
                    }
                }
            } else {
                ImGui::Text("No render yet");
            }
            ImGui::End();

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
        Pale::SceneUpload::freeBuffers(sceneGpu, queue);
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
