// pale_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "Renderer/RenderPackage.h"
#include "Renderer/Kernels/KernelHelpers.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdint>
#include <memory>
#include <filesystem>
#include <entt/entt.hpp>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/string_cast.hpp"
#include <glm/gtc/quaternion.hpp>

import Pale.DeviceSelector;
import Pale.Scene.Components;
import Pale.SceneSerializer;
import Pale.Log;
import Pale.Utils.ImageIO;
import Pale.Assets;
import Pale.Assets.Core;
import Pale.Render.SceneBuild;
import Pale.Render.SceneUpload;
import Pale.Render.PathTracer;
import Pale.Render.Sensors;
import Pale.Scene;


namespace py = pybind11;

// replace your get_u64 with this
static inline uint64_t get_u64(const py::dict &d, const char *k, uint64_t def) {
    if (!d.contains(k)) return def;
    py::int_ v = d[k];
    try {
        // Coerce anything numeric (float, numpy scalar) via Python int()
        return py::int_(v).cast<uint64_t>();
    } catch (const py::error_already_set &) {
        return def; // or throw if you prefer strict
    }
}

static inline int get_i(const py::dict &d, const char *k, int def) {
    if (d.contains(k)) return py::cast<int>(d[k]);
    return def;
}

static inline bool get_b(const py::dict &d, const char *k, bool def) {
    if (d.contains(k)) return py::cast<bool>(d[k]);
    return def;
}

static inline float get_f(const py::dict &d, const char *k, float def) {
    if (d.contains(k)) return py::cast<float>(d[k]);
    return def;
}

static inline std::string get_s(const py::dict &d, const char *k, const std::string &def) {
    if (d.contains(k)) return py::cast<std::string>(d[k]);
    return def;
}

static inline glm::quat normalizeQuaternionOrIdentity(glm::quat q) {
    const bool finite = std::isfinite(q.w) && std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z);
    const float lengthSquared = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (!finite || lengthSquared <= 1.0e-20f) return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    const float invLength = 1.0f / std::sqrt(lengthSquared);
    q.w *= invLength; q.x *= invLength; q.y *= invLength; q.z *= invLength;
    if (q.w < 0.0f) { q.w = -q.w; q.x = -q.x; q.y = -q.y; q.z = -q.z; }
    return q;
}
static inline float glmLengthSquared(const glm::vec3 &v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
static inline void frameFromQuaternion(const glm::quat &inputQuaternion, Pale::float3 &tangentUOut, Pale::float3 &tangentVOut) {
    const glm::quat q = normalizeQuaternionOrIdentity(inputQuaternion);
    const glm::mat3 rotation = glm::mat3_cast(q);
    glm::vec3 tangentU = glm::vec3(rotation[0]);
    glm::vec3 tangentV = glm::vec3(rotation[1]);
    if (glmLengthSquared(tangentU) <= 1.0e-20f) tangentU = glm::vec3(1.0f, 0.0f, 0.0f);
    tangentU = glm::normalize(tangentU);
    tangentV -= glm::dot(tangentV, tangentU) * tangentU;
    if (glmLengthSquared(tangentV) <= 1.0e-20f) {
        const glm::vec3 fallback = std::abs(tangentU.y) < 0.9f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
        tangentV = fallback - glm::dot(fallback, tangentU) * tangentU;
    }
    tangentV = glm::normalize(tangentV);
    tangentUOut = tangentU;
    tangentVOut = tangentV;
}

static inline glm::quat quaternionFromFrame(const Pale::float3 &tangentUIn,
                                            const Pale::float3 &tangentVIn) {
    glm::vec3 tangentU(tangentUIn.x(), tangentUIn.y(), tangentUIn.z());
    glm::vec3 tangentV(tangentVIn.x(), tangentVIn.y(), tangentVIn.z());

    if (glmLengthSquared(tangentU) <= 1.0e-20f) {
        tangentU = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    tangentU = glm::normalize(tangentU);

    tangentV -= glm::dot(tangentV, tangentU) * tangentU;
    if (glmLengthSquared(tangentV) <= 1.0e-20f) {
        const glm::vec3 fallback =
                std::abs(tangentU.y) < 0.9f
                    ? glm::vec3(0.0f, 1.0f, 0.0f)
                    : glm::vec3(1.0f, 0.0f, 0.0f);
        tangentV = fallback - glm::dot(fallback, tangentU) * tangentU;
    }
    tangentV = glm::normalize(tangentV);

    const glm::vec3 normal = glm::normalize(glm::cross(tangentU, tangentV));
    glm::mat3 frame(1.0f);
    frame[0] = tangentU;
    frame[1] = tangentV;
    frame[2] = normal;

    return normalizeQuaternionOrIdentity(glm::quat_cast(frame));
}


class PythonRenderer {
    struct TrainingTargetDevice {
        std::string cameraName;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        Pale::float4 *rgba = nullptr;
        // [combined RGB objective, half-MSE, DSSIM].
        float *loss = nullptr;
    };

    struct RgbLossOptions {
        float ssimWeight = 0.0f;
        int ssimWindowSize = 11;
        float ssimSigma = 1.5f;
    };

    struct RgbSsimScratch {
        std::size_t pixelCapacity = 0;
        Pale::float4 *renderedMean = nullptr;
        Pale::float4 *targetMean = nullptr;
        Pale::float4 *derivativeMean = nullptr;
        Pale::float4 *derivativeVariance = nullptr;
        Pale::float4 *derivativeCovariance = nullptr;
    };

    struct DeviceTrainingStepOptions {
        std::string optimizer = "adam";
        float learningRatePosition = 0.0f;
        float learningRateRotation = 0.0f;
        float learningRateScale = 0.0f;
        float learningRateAlbedo = 0.0f;
        float learningRateOpacity = 0.0f;
        float learningRateBeta = 0.0f;
        float cameraBatchScale = 1.0f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float epsilon = 1.0e-8f;
        float maxRotationStepRadians = 0.01f;
    };

    struct DeviceAdamState {
        std::size_t pointCount = 0;
        std::uint32_t step = 0;

        Pale::float3 *positionM = nullptr;
        Pale::float3 *positionV = nullptr;
        Pale::float3 *rotationM = nullptr;
        Pale::float3 *rotationV = nullptr;
        Pale::float2 *scaleM = nullptr;
        Pale::float2 *scaleV = nullptr;
        Pale::float3 *albedoM = nullptr;
        Pale::float3 *albedoV = nullptr;
        float *opacityM = nullptr;
        float *opacityV = nullptr;
        float *betaM = nullptr;
        float *betaV = nullptr;
    };

    struct SelectedTrainingBatch {
        std::vector<Pale::SensorGPU> sensors;
        std::vector<Pale::DebugImages> debugImages;
        std::vector<TrainingTargetDevice *> targets;
    };

public:
    PythonRenderer(const std::string &assetRootDir,
                   const std::string &sceneXml,
                   const std::string &pointCloudFile,
                   const py::dict &settingsDict // <-- accept dict
    ) {
        std::filesystem::current_path(assetRootDir);
        int level = 2;
        if (!settingsDict.is_none()) {
            // use integer types consistent with your struct
            level = (get_i(settingsDict, "logging", 2));
        }

        Pale::Log::init(level);

        m_settings.integratorKind = Pale::IntegratorKind::photonMapping;
        m_settings.pointGeometrySupportRadius = 0.1f;
        m_settings.pointGeometryReconstructionLength = 0.0f;
        m_settings.pointGeometryRayOffsetMultiplier = 1.0f;
        m_settings.pointGeometryMinimumContributors = 1u;
        m_settings.pointGeometryCoverageScale = 1.01f;

        assetManager = std::make_unique<Pale::AssetManager>(256);
        assetManager->enableHotReload(true);
        assetManager->registerLoader<Pale::Mesh>(Pale::AssetType::Mesh, std::make_shared<Pale::AssimpMeshLoader>());
        assetManager->registerLoader<Pale::Material>(Pale::AssetType::Material,
                                                     std::make_shared<Pale::YamlMaterialLoader>());
        assetManager->registerLoader<Pale::PointAsset>(Pale::AssetType::PointCloud,
                                                       std::make_shared<Pale::PLYPointLoader>());

        assetManager->registry().load("");

        scene = std::make_shared<Pale::Scene>();
        Pale::AssetIndexFromRegistry assetIndexer(assetManager->registry());
        Pale::SceneSerializer sceneSerializer(scene, assetIndexer);
        sceneSerializer.deserialize(sceneXml);

        std::filesystem::path pointCloudPath = pointCloudFile.empty()
                                                   ? std::filesystem::path("initial.ply")
                                                   : std::filesystem::path(pointCloudFile);
        auto pointCloudHandle = assetIndexer.importPath(
            pointCloudPath,
            Pale::AssetType::PointCloud
        );
        auto gaussianEntity = scene->createEntity("Gaussian");
        gaussianEntity.addComponent<Pale::PointCloudComponent>().pointCloudID = pointCloudHandle;

        // Store for later use in set_point_parameters
        pointCloudAssetHandle = pointCloudHandle;
        deviceSelector = std::make_unique<Pale::DeviceSelector>();
        Pale::AssetAccessFromManager assetAccessor(*assetManager);

        // Map python keys -> engine m_settings. Adjust names to your struct.
        // Example mappings based on your dict:
        //   "photons": 1e6, "bounces": 6, "gather_passes": 6,
        //   "adjoint_bounces": 1, "adjoint_passes": 6
        if (!settingsDict.is_none()) {
            // use integer types consistent with your struct
            m_settings.photonsPerLaunch = get_u64(settingsDict, "photons", m_settings.photonsPerLaunch);
            m_settings.maxBounces = get_i(settingsDict, "bounces", m_settings.maxBounces);
            m_settings.numForwardPasses = get_i(settingsDict, "forward_passes", m_settings.numForwardPasses);
            m_settings.numShadowRays = get_i(settingsDict, "primal_shadow_rays", m_settings.numShadowRays);
            m_settings.numAdjointShadowRays = get_i(settingsDict, "adjoint_shadow_rays", m_settings.numAdjointShadowRays);
            m_settings.maxAdjointBounces = get_i(settingsDict, "adjoint_bounces", m_settings.maxAdjointBounces);
            m_settings.adjointSamplesPerPixel = get_i(settingsDict, "adjoint_passes", m_settings.adjointSamplesPerPixel);
            m_settings.random.seed = get_i(settingsDict, "seed", m_settings.random.seed);
            m_settings.renderDebugGradientImages =
                    get_b(settingsDict, "debug_images", m_settings.renderDebugGradientImages);
            m_settings.enableAdjointDirectLight =
                    get_b(settingsDict, "enable_adjoint_shadow_rays", m_settings.enableAdjointDirectLight);
            m_settings.numAdjointPathShadowRays =
                    get_i(settingsDict, "adjoint_shadow_path_rays", m_settings.numAdjointPathShadowRays);
            m_settings.sampling.qNull =
                    get_f(settingsDict, "adjoint_q_null", m_settings.sampling.qNull);
            m_settings.sampling.qReflect =
                    get_f(settingsDict, "adjoint_q_reflect", m_settings.sampling.qReflect);
            m_settings.depthDistortionWeight =
                    get_f(settingsDict, "depth_distort_weight", m_settings.depthDistortionWeight);

            m_settings.normalConsistencyWeight =
                    get_f(settingsDict, "normal_consistency_weight", m_settings.normalConsistencyWeight);
            m_settings.normalFromDepthUseMeanDepth =
                    get_b(settingsDict, "normal_from_depth_use_mean_depth", m_settings.normalFromDepthUseMeanDepth);

            m_settings.visibilityWeightedOpacityRegularizerWeight =
                    get_f(settingsDict,
                          "opacity_prior_weight",
                          m_settings.visibilityWeightedOpacityRegularizerWeight);
            m_settings.intraSlabDepthRegularizerWeight =
                    get_f(settingsDict,
                          "intra_slab_depth_weight",
                          m_settings.intraSlabDepthRegularizerWeight);
            m_settings.curvatureScaleRegularizerWeight =
                    get_f(settingsDict,
                          "curvature_scale_weight",
                          m_settings.curvatureScaleRegularizerWeight);
            m_settings.rendererDebugShareLocalLayerDirectLighting =
                    get_b(settingsDict,
                          "share_local_layer_direct_lighting",
                          m_settings.rendererDebugShareLocalLayerDirectLighting);
            curvatureDensificationEnabled =
                    get_b(settingsDict, "enable_curvature_densification", false);
            primalActivityTrackingEnabled =
                    get_b(settingsDict, "enable_primal_activity_tracking", false);
            m_settings.rendererDebugMinimumProjectedFootprint =
                    get_b(settingsDict,
                          "minimum_projected_footprint",
                          m_settings.rendererDebugMinimumProjectedFootprint);
            m_settings.rendererDebugMinimumProjectedFootprintPixels =
                    get_f(settingsDict,
                          "minimum_projected_footprint_pixels",
                          m_settings.rendererDebugMinimumProjectedFootprintPixels);
            // Finite-difference/debug controls. Exposing these through the
            // regular settings dictionary lets tests exercise both the batched
            // and scalar intersection paths with identical scene data.
            m_settings.rendererDebugLocalLayerDepthEpsilon =
                    get_f(settingsDict,
                          "local_layer_depth_epsilon",
                          m_settings.rendererDebugLocalLayerDepthEpsilon);
            m_settings.rendererDebugLocalLayerNormalCosineThreshold =
                    get_f(settingsDict,
                          "local_layer_normal_cosine_threshold",
                          m_settings.rendererDebugLocalLayerNormalCosineThreshold);
            m_settings.rendererDebugMaxSplatEventsPerRay =
                    get_i(settingsDict,
                          "max_splat_events_per_ray",
                          m_settings.rendererDebugMaxSplatEventsPerRay);
            m_settings.rendererDebugMaxLocalSurfelHits =
                    get_i(settingsDict,
                          "max_local_surfel_hits",
                          m_settings.rendererDebugMaxLocalSurfelHits);
            m_settings.rendererDebugPointHitBatchSize =
                    get_i(settingsDict,
                          "point_hit_batch_size",
                          m_settings.rendererDebugPointHitBatchSize);
            m_settings.rendererDebugPointHitBatchLookahead =
                    get_b(settingsDict,
                          "point_hit_batch_lookahead",
                          m_settings.rendererDebugPointHitBatchLookahead);
            // add other keys as needed, e.g., samplesPerPixel, exposure, etc.
        }

        buildProducts = Pale::SceneBuild::build(scene, assetAccessor, Pale::SceneBuild::BuildOptions());
        sceneGpu = Pale::SceneUpload::allocateAndUpload(buildProducts, deviceSelector->getQueue());
        sensorsForward = Pale::makeSensorsForScene(deviceSelector->getQueue(), buildProducts);
        //Pale::float4 color = {0.025, 0.075, 0.165, 1.0f};
        //Pale::setBackgroundColor(deviceSelector->getQueue(), sensorsForward, color);

        debugImages.resize(sensorsForward.size());
        gradients = Pale::makeGradientsForScene(deviceSelector->getQueue(), buildProducts, debugImages.data());
        depthDistortionGradients = Pale::makeGradientsForScene(deviceSelector->getQueue(), buildProducts, nullptr);
        normalConsistencyGradients = Pale::makeGradientsForScene(deviceSelector->getQueue(), buildProducts, nullptr);
        visibilityOpacityGradients = Pale::makeGradientsForScene(deviceSelector->getQueue(), buildProducts, nullptr);
        intraSlabDepthGradients = Pale::makeGradientsForScene(deviceSelector->getQueue(), buildProducts, nullptr);
        curvatureScaleGradients = Pale::makeGradientsForScene(deviceSelector->getQueue(), buildProducts, nullptr);
        if (curvatureDensificationEnabled) {
            curvatureDensificationStats = Pale::makeCurvatureDensificationStatsForScene(
                deviceSelector->getQueue(), buildProducts);
        }
        if (primalActivityTrackingEnabled) {
            primalActivityStats = Pale::makePrimalActivityStatsForScene(
                deviceSelector->getQueue(), buildProducts);
        }

        // Print summary
        Pale::Log::PA_WARN("=== Renderer Settings ===");
        Pale::Log::PA_WARN("  Photons per launch        : {}", m_settings.photonsPerLaunch);
        Pale::Log::PA_WARN("  Max bounces               : {}", m_settings.maxBounces);
        Pale::Log::PA_WARN("  Forward passes            : {}", m_settings.numForwardPasses);
        Pale::Log::PA_WARN("  Shadow Rays               : {}", m_settings.numShadowRays);
        Pale::Log::PA_WARN("  Adjoint Shadow Rays       : {}", m_settings.numAdjointShadowRays);
        Pale::Log::PA_WARN("  Adjoint bounces           : {}", m_settings.maxAdjointBounces);
        Pale::Log::PA_WARN("  Adjoint samples per pixel : {}", m_settings.adjointSamplesPerPixel);
        Pale::Log::PA_WARN("  Using Adjoint Shadow rays : {}", m_settings.enableAdjointDirectLight);
        Pale::Log::PA_WARN("  Adjoint Shadow ray count  : {}", m_settings.numAdjointPathShadowRays);
        Pale::Log::PA_WARN("  Adjoint q(null/reflect)   : {}/{}",
                           m_settings.sampling.qNull,
                           m_settings.sampling.qReflect);
        Pale::Log::PA_WARN("  Visibility opacity weight : {}", m_settings.visibilityWeightedOpacityRegularizerWeight);
        Pale::Log::PA_WARN("  Depth Distortion Weight   : {}", m_settings.depthDistortionWeight);
        Pale::Log::PA_WARN("  Normal Consistency Weight : {}", m_settings.normalConsistencyWeight);
        Pale::Log::PA_WARN("  Intra-slab depth weight   : {}", m_settings.intraSlabDepthRegularizerWeight);
        Pale::Log::PA_WARN("  Curvature scale weight    : {}", m_settings.curvatureScaleRegularizerWeight);
        Pale::Log::PA_WARN("  Shared slab direct light  : {}", m_settings.rendererDebugShareLocalLayerDirectLighting);
        Pale::Log::PA_WARN("  Curvature densification   : {}", curvatureDensificationEnabled);
        Pale::Log::PA_WARN("  Primal activity tracking  : {}", primalActivityTrackingEnabled);
        Pale::Log::PA_WARN("  Minimum footprint enabled : {}", m_settings.rendererDebugMinimumProjectedFootprint);
        Pale::Log::PA_WARN("  Minimum footprint sigma px: {}", m_settings.rendererDebugMinimumProjectedFootprintPixels);
        Pale::Log::PA_WARN("  Local layer depth epsilon : {}", m_settings.rendererDebugLocalLayerDepthEpsilon);
        Pale::Log::PA_WARN("  Local layer normal cosine : {}", m_settings.rendererDebugLocalLayerNormalCosineThreshold);
        Pale::Log::PA_WARN("  Max splat events per ray  : {}", m_settings.rendererDebugMaxSplatEventsPerRay);
        Pale::Log::PA_WARN("  Max local surfel hits     : {}", m_settings.rendererDebugMaxLocalSurfelHits);
        Pale::Log::PA_WARN("  Point-hit batch size      : {}", m_settings.rendererDebugPointHitBatchSize);
        Pale::Log::PA_WARN("  Point-hit lookahead       : {}", m_settings.rendererDebugPointHitBatchLookahead);
        Pale::Log::PA_WARN("=== Sensors (Forward) ===");
        for (size_t i = 0; i < sensorsForward.size(); ++i) {
            const auto &s = sensorsForward[i];

            Pale::Log::PA_WARN("  --- Sensor {} ---", i);
            Pale::Log::PA_WARN("      Name                : {}", s.name);
            Pale::Log::PA_WARN("      Resolution          : {} x {}", s.width, s.height);

            Pale::Log::PA_WARN("      Camera Position     : ({}, {}, {})",
                               s.camera.pos.x(), s.camera.pos.y(), s.camera.pos.z());
            Pale::Log::PA_WARN("      Exposure / Gamma    : {} / {}",
                               s.exposureCorrection,
                               s.gammaCorrection);
            Pale::Log::PA_WARN("      Output encoding     : {}",
                               s.useSrgbEncoding ? "sRGB" : "power gamma");
        }


        pathTracer = std::make_unique<Pale::PathTracer>(deviceSelector->getQueue(), m_settings);
        pathTracer->setScene(sceneGpu, buildProducts);
        pathTracer->setCurvatureDensificationStats(
            curvatureDensificationEnabled ? &curvatureDensificationStats : nullptr);
        pathTracer->setPrimalActivityStats(
            primalActivityTrackingEnabled ? &primalActivityStats : nullptr);
    }

    ~PythonRenderer() {
        if (deviceSelector) {
            auto queue = deviceSelector->getQueue();

            Pale::freeGradientsForScene(queue, gradients);
            Pale::freeGradientsForScene(queue, depthDistortionGradients);
            Pale::freeGradientsForScene(queue, normalConsistencyGradients);
            Pale::freeGradientsForScene(queue, visibilityOpacityGradients);
            Pale::freeGradientsForScene(queue, intraSlabDepthGradients);
            Pale::freeGradientsForScene(queue, curvatureScaleGradients);
            Pale::freeCurvatureDensificationStats(queue, curvatureDensificationStats);
            Pale::freePrimalActivityStats(queue, primalActivityStats);

            Pale::freeDebugImagesForScene(queue, debugImages.data(), debugImages.size());
            freeTrainingTargets(queue);
            freeRgbSsimScratch(queue);
            freeDeviceTrainingState(queue);
            queue.wait();
        }

        if (assetManager) {
            assetManager->registry().save("asset_registry.yaml");
        }
    }

    py::dict render_forward(std::string cameraName) {
        py::gil_scoped_release release;

        std::vector<Pale::SensorGPU> selectedSensors =
            selectSensorsByName(cameraName.empty()
                ? std::optional<std::string>{}
                : std::optional<std::string>{cameraName});

        pathTracer->renderForward(selectedSensors);

        auto queue = deviceSelector->getQueue();

        struct HostImage {
            std::string cameraName;
            std::uint32_t imageWidth;
            std::uint32_t imageHeight;

            std::vector<float> imageData;
            std::vector<float> imageDataRAW;
            std::vector<float> depthDistortionData;
            std::vector<float> visibilityWeightedOpacityData;
            std::vector<float> intraSlabDepthData;
            std::vector<std::uint32_t> intraSlabDepthActiveSlabCountData;
            std::vector<float> curvatureScaleData;
            std::vector<std::uint32_t> curvatureScaleActiveSlabCountData;

            std::vector<float> medianDepthData;
            std::vector<float> meanDepthData;
            std::vector<float> medianWorldPositionData;
            std::vector<float> visibleNormalData;
            std::vector<float> normalFromDepthData;
        };

        std::vector<HostImage> hostImages;
        hostImages.reserve(selectedSensors.size());

        for (const auto &sensor: selectedSensors) {
            HostImage hostImage;

            hostImage.cameraName =
                    std::string(sensor.name, strnlen(sensor.name, sizeof(sensor.name)));

            hostImage.imageWidth = sensor.width;
            hostImage.imageHeight = sensor.height;

            const std::size_t pixelCount =
                    static_cast<std::size_t>(hostImage.imageWidth) *
                    static_cast<std::size_t>(hostImage.imageHeight);

            hostImage.imageData = Pale::downloadSensorLDR(queue, sensor);
            hostImage.imageDataRAW = Pale::downloadSensorRGBARAW(queue, sensor);

            hostImage.depthDistortionData =
                    Pale::downloadFloatBuffer(queue, sensor.depthDistortionBuffer, pixelCount);

            hostImage.visibilityWeightedOpacityData =
                    Pale::downloadFloatBuffer(queue, sensor.visibilityWeightedOpacityBuffer, pixelCount);

            hostImage.intraSlabDepthData =
                    Pale::downloadFloatBuffer(queue, sensor.intraSlabDepthBuffer, pixelCount);
            hostImage.intraSlabDepthActiveSlabCountData =
                    Pale::downloadUint32Buffer(
                        queue, sensor.intraSlabDepthActiveSlabCountBuffer, pixelCount);
            hostImage.curvatureScaleData =
                    Pale::downloadFloatBuffer(queue, sensor.curvatureScaleBuffer, pixelCount);
            hostImage.curvatureScaleActiveSlabCountData =
                    Pale::downloadUint32Buffer(
                        queue, sensor.curvatureScaleActiveSlabCountBuffer, pixelCount);

            hostImage.medianDepthData =
                    Pale::downloadFloatBuffer(queue, sensor.medianDepthBuffer, pixelCount);

            hostImage.meanDepthData =
                    Pale::downloadFloatBuffer(queue, sensor.meanDepthBuffer, pixelCount);

            hostImage.medianWorldPositionData =
                    Pale::downloadFloat4Buffer(queue, sensor.medianWorldPositionBuffer, pixelCount);

            hostImage.visibleNormalData =
                    Pale::downloadFloat4Buffer(queue, sensor.visibleNormalBuffer, pixelCount);

            hostImage.normalFromDepthData =
                    Pale::downloadFloat4Buffer(queue, sensor.normalFromDepthBuffer, pixelCount);


            hostImages.push_back(std::move(hostImage));
        }

        py::gil_scoped_acquire acquire;

        py::dict result;

        for (auto &hostImage: hostImages) {
            const std::uint32_t imageWidth = hostImage.imageWidth;
            const std::uint32_t imageHeight = hostImage.imageHeight;

            std::vector<ssize_t> rgbaShape{
                static_cast<ssize_t>(imageHeight),
                static_cast<ssize_t>(imageWidth),
                static_cast<ssize_t>(4)
            };

            std::vector<ssize_t> rgbaStrides{
                static_cast<ssize_t>(imageWidth * 4 * sizeof(float)),
                static_cast<ssize_t>(4 * sizeof(float)),
                static_cast<ssize_t>(sizeof(float))
            };

            std::vector<ssize_t> scalarShape{
                static_cast<ssize_t>(imageHeight),
                static_cast<ssize_t>(imageWidth)
            };

            std::vector<ssize_t> scalarStrides{
                static_cast<ssize_t>(imageWidth * sizeof(float)),
                static_cast<ssize_t>(sizeof(float))
            };

            auto makeRGBAArray = [&](std::vector<float> &buffer) -> py::array_t<float> {
                auto *ownedBuffer = new std::vector<float>(std::move(buffer));
                return py::array_t<float>(
                    rgbaShape,
                    rgbaStrides,
                    ownedBuffer->data(),
                    py::capsule(ownedBuffer, [](void *ptr) {
                        delete static_cast<std::vector<float> *>(ptr);
                    })
                );
            };

            auto makeScalarArray = [&](std::vector<float> &buffer) -> py::array_t<float> {
                auto *ownedBuffer = new std::vector<float>(std::move(buffer));
                return py::array_t<float>(
                    scalarShape,
                    scalarStrides,
                    ownedBuffer->data(),
                    py::capsule(ownedBuffer, [](void *ptr) {
                        delete static_cast<std::vector<float> *>(ptr);
                    })
                );
            };

            auto makeUintScalarArray = [&](std::vector<std::uint32_t> &buffer)
                    -> py::array_t<std::uint32_t> {
                auto *ownedBuffer = new std::vector<std::uint32_t>(std::move(buffer));
                std::vector<ssize_t> uintScalarStrides{
                    static_cast<ssize_t>(imageWidth * sizeof(std::uint32_t)),
                    static_cast<ssize_t>(sizeof(std::uint32_t))
                };
                return py::array_t<std::uint32_t>(
                    scalarShape,
                    uintScalarStrides,
                    ownedBuffer->data(),
                    py::capsule(ownedBuffer, [](void *ptr) {
                        delete static_cast<std::vector<std::uint32_t> *>(ptr);
                    })
                );
            };

            py::dict cameraResult;
            cameraResult[py::str("image")] = makeRGBAArray(hostImage.imageData);
            cameraResult[py::str("raw")] = makeRGBAArray(hostImage.imageDataRAW);

            cameraResult[py::str("depth_distortion")] =
                    makeScalarArray(hostImage.depthDistortionData);

            cameraResult[py::str("opacity_prior")] =
                    makeScalarArray(hostImage.visibilityWeightedOpacityData);

            cameraResult[py::str("intra_slab_depth")] =
                    makeScalarArray(hostImage.intraSlabDepthData);
            cameraResult[py::str("intra_slab_depth_active_slab_count")] =
                    makeUintScalarArray(hostImage.intraSlabDepthActiveSlabCountData);
            cameraResult[py::str("curvature_scale")] =
                    makeScalarArray(hostImage.curvatureScaleData);
            cameraResult[py::str("curvature_scale_active_slab_count")] =
                    makeUintScalarArray(hostImage.curvatureScaleActiveSlabCountData);

            cameraResult[py::str("median_depth")] =
                    makeScalarArray(hostImage.medianDepthData);
            cameraResult[py::str("mean_depth")] =
                    makeScalarArray(hostImage.meanDepthData);
            cameraResult[py::str("median_world_position")] =
                    makeRGBAArray(hostImage.medianWorldPositionData);

            cameraResult[py::str("visible_normal")] =
                    makeRGBAArray(hostImage.visibleNormalData);

            cameraResult[py::str("normal_from_depth")] =
                    makeRGBAArray(hostImage.normalFromDepthData);

            result[py::str(hostImage.cameraName)] = std::move(cameraResult);
        }

        return result;
    }

    void upload_training_targets(const py::dict &targetImagesDictionary) {
        auto syclQueue = deviceSelector->getQueue();

        for (const auto &sensor: sensorsForward) {
            const std::string cameraName(sensor.name, strnlen(sensor.name, sizeof(sensor.name)));
            if (!targetImagesDictionary.contains(py::str(cameraName))) {
                continue;
            }

            py::array targetRgbArray = targetImagesDictionary[py::str(cameraName)].cast<py::array>();
            py::buffer_info bufferInfo = targetRgbArray.request();
            if (bufferInfo.ndim != 3 || bufferInfo.shape[2] != 3) {
                throw std::runtime_error(
                    "upload_training_targets: target image for camera '" + cameraName +
                    "' must be HxWx3 float32");
            }
            if (bufferInfo.itemsize != sizeof(float)) {
                throw std::runtime_error(
                    "upload_training_targets: target image for camera '" + cameraName +
                    "' must have dtype float32");
            }

            const std::uint32_t height = static_cast<std::uint32_t>(bufferInfo.shape[0]);
            const std::uint32_t width = static_cast<std::uint32_t>(bufferInfo.shape[1]);
            if (width != sensor.width || height != sensor.height) {
                throw std::runtime_error(
                    "upload_training_targets: resolution mismatch for camera '" + cameraName +
                    "': target image is " + std::to_string(width) + "x" + std::to_string(height) +
                    ", but sensor is " + std::to_string(sensor.width) + "x" +
                    std::to_string(sensor.height));
            }

            const auto *rgbPointer = static_cast<const float *>(bufferInfo.ptr);
            const std::size_t pixelCount = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
            std::vector<Pale::float4> targetRgba(pixelCount);
            for (std::size_t pixelIndex = 0; pixelIndex < pixelCount; ++pixelIndex) {
                const std::size_t rgbIndex = pixelIndex * 3u;
                targetRgba[pixelIndex] = Pale::float4{
                    rgbPointer[rgbIndex + 0u],
                    rgbPointer[rgbIndex + 1u],
                    rgbPointer[rgbIndex + 2u],
                    1.0f
                };
            }

            TrainingTargetDevice &target = trainingTargets[cameraName];
            ensureTrainingTargetCapacity(target, cameraName, width, height, syclQueue);
            syclQueue.memcpy(
                target.rgba,
                targetRgba.data(),
                pixelCount * sizeof(Pale::float4));
        }

        syclQueue.wait_and_throw();
    }

    py::tuple render_rgb_loss_backward(const py::list &cameraNamesList,
                                       const py::dict &optionsDictionary = py::dict()) {
        auto syclQueue = deviceSelector->getQueue();

        SelectedTrainingBatch selectedBatch =
                selectTrainingBatch(cameraNamesList, "render_rgb_loss_backward");
        const RgbLossOptions rgbLossOptions = parseRgbLossOptions(optionsDictionary);

        {
            py::gil_scoped_release release;
            pathTracer->renderForward(selectedBatch.sensors);

            for (std::size_t cameraIndex = 0; cameraIndex < selectedBatch.sensors.size(); ++cameraIndex) {
                launchRgbLossAdjointKernel(
                    syclQueue,
                    selectedBatch.sensors[cameraIndex],
                    *selectedBatch.targets[cameraIndex],
                    rgbLossOptions);
            }
            syclQueue.wait_and_throw();

            pathTracer->renderBackward(
                selectedBatch.sensors,
                gradients,
                selectedBatch.debugImages.data());
        }

        py::dict lossValues;
        py::dict l2LossValues;
        py::dict dssimLossValues;
        for (std::size_t cameraIndex = 0; cameraIndex < selectedBatch.sensors.size(); ++cameraIndex) {
            std::array<float, 3> lossComponents{};
            syclQueue.memcpy(
                lossComponents.data(),
                selectedBatch.targets[cameraIndex]->loss,
                lossComponents.size() * sizeof(float)).wait();
            const std::string cameraName(
                selectedBatch.sensors[cameraIndex].name,
                strnlen(selectedBatch.sensors[cameraIndex].name, sizeof(selectedBatch.sensors[cameraIndex].name)));
            lossValues[py::str(cameraName)] = lossComponents[0];
            l2LossValues[py::str(cameraName)] = lossComponents[1];
            dssimLossValues[py::str(cameraName)] = lossComponents[2];
        }

        py::dict adjointImages;
        adjointImages["loss_values"] = std::move(lossValues);
        adjointImages["l2_loss_values"] = std::move(l2LossValues);
        adjointImages["dssim_loss_values"] = std::move(dssimLossValues);
        adjointImages["gradient_stats"] = makeGradientStatsDictionary(gradients);

        return py::make_tuple(makeGradientDictionary(gradients), adjointImages);
    }

    py::dict render_rgb_training_step(const py::list &cameraNamesList,
                                      const py::dict &optionsDictionary = py::dict()) {
        auto syclQueue = deviceSelector->getQueue();

        SelectedTrainingBatch selectedBatch =
                selectTrainingBatch(cameraNamesList, "render_rgb_training_step");
        DeviceTrainingStepOptions options = parseDeviceTrainingStepOptions(optionsDictionary);
        const RgbLossOptions rgbLossOptions = parseRgbLossOptions(optionsDictionary);
        const bool returnGradientStats =
                get_b(optionsDictionary, "return_gradient_stats", false);

        {
            py::gil_scoped_release release;

            pathTracer->renderForward(selectedBatch.sensors);

            for (std::size_t cameraIndex = 0; cameraIndex < selectedBatch.sensors.size(); ++cameraIndex) {
                launchRgbLossAdjointKernel(
                    syclQueue,
                    selectedBatch.sensors[cameraIndex],
                    *selectedBatch.targets[cameraIndex],
                    rgbLossOptions);
            }
            syclQueue.wait_and_throw();

            pathTracer->renderBackward(
                selectedBatch.sensors,
                gradients,
                selectedBatch.debugImages.data());

            ensureDeviceTrainingState(gradients.numPoints, syclQueue);
            launchDeviceTrainingStepKernel(
                syclQueue, gradients, nullptr, nullptr, nullptr, nullptr, nullptr, options);
            launchPointBvhRefitKernel(syclQueue);
            syclQueue.wait_and_throw();
            devicePointParametersDirty = true;
        }

        py::dict lossValues;
        py::dict l2LossValues;
        py::dict dssimLossValues;
        for (std::size_t cameraIndex = 0; cameraIndex < selectedBatch.sensors.size(); ++cameraIndex) {
            std::array<float, 3> lossComponents{};
            syclQueue.memcpy(
                lossComponents.data(),
                selectedBatch.targets[cameraIndex]->loss,
                lossComponents.size() * sizeof(float)).wait();
            const std::string cameraName(
                selectedBatch.sensors[cameraIndex].name,
                strnlen(selectedBatch.sensors[cameraIndex].name, sizeof(selectedBatch.sensors[cameraIndex].name)));
            lossValues[py::str(cameraName)] = lossComponents[0];
            l2LossValues[py::str(cameraName)] = lossComponents[1];
            dssimLossValues[py::str(cameraName)] = lossComponents[2];
        }

        py::dict result;
        result["loss_values"] = std::move(lossValues);
        result["l2_loss_values"] = std::move(l2LossValues);
        result["dssim_loss_values"] = std::move(dssimLossValues);
        result["point_count"] = static_cast<std::uint64_t>(gradients.numPoints);
        result["optimizer_step"] = static_cast<std::uint64_t>(deviceTrainingState.step);
        if (returnGradientStats) {
            result["gradient_stats"] = makeGradientStatsDictionary(gradients);
        }
        return result;
    }

    py::dict render_rgb_backward_from_current_forward(const py::list &cameraNamesList,
                                                      const py::dict &optionsDictionary = py::dict()) {
        auto syclQueue = deviceSelector->getQueue();

        SelectedTrainingBatch selectedBatch =
                selectTrainingBatch(cameraNamesList, "render_rgb_backward_from_current_forward");
        const RgbLossOptions rgbLossOptions = parseRgbLossOptions(optionsDictionary);
        const bool returnGradientStats =
                get_b(optionsDictionary, "return_gradient_stats", false);

        {
            py::gil_scoped_release release;

            for (std::size_t cameraIndex = 0; cameraIndex < selectedBatch.sensors.size(); ++cameraIndex) {
                launchRgbLossAdjointKernel(
                    syclQueue,
                    selectedBatch.sensors[cameraIndex],
                    *selectedBatch.targets[cameraIndex],
                    rgbLossOptions);
            }
            syclQueue.wait_and_throw();

            pathTracer->renderBackward(
                selectedBatch.sensors,
                gradients,
                selectedBatch.debugImages.data());
        }

        py::dict lossValues;
        py::dict l2LossValues;
        py::dict dssimLossValues;
        for (std::size_t cameraIndex = 0; cameraIndex < selectedBatch.sensors.size(); ++cameraIndex) {
            std::array<float, 3> lossComponents{};
            syclQueue.memcpy(
                lossComponents.data(),
                selectedBatch.targets[cameraIndex]->loss,
                lossComponents.size() * sizeof(float)).wait();
            const std::string cameraName(
                selectedBatch.sensors[cameraIndex].name,
                strnlen(selectedBatch.sensors[cameraIndex].name, sizeof(selectedBatch.sensors[cameraIndex].name)));
            lossValues[py::str(cameraName)] = lossComponents[0];
            l2LossValues[py::str(cameraName)] = lossComponents[1];
            dssimLossValues[py::str(cameraName)] = lossComponents[2];
        }

        py::dict result;
        result["loss_values"] = std::move(lossValues);
        result["l2_loss_values"] = std::move(l2LossValues);
        result["dssim_loss_values"] = std::move(dssimLossValues);
        result["point_count"] = static_cast<std::uint64_t>(gradients.numPoints);
        if (returnGradientStats) {
            result["gradient_stats"] = makeGradientStatsDictionary(gradients);
        }
        return result;
    }

    py::dict apply_device_training_step(const py::dict &optionsDictionary = py::dict()) {
        auto syclQueue = deviceSelector->getQueue();
        DeviceTrainingStepOptions options = parseDeviceTrainingStepOptions(optionsDictionary);
        const bool includeDepthDistortion =
                get_b(optionsDictionary, "include_depth_distortion", false);
        const bool includeNormalConsistency =
                get_b(optionsDictionary, "include_normal_consistency", false);
        const bool includeVisibilityWeightedOpacity =
                get_b(optionsDictionary, "include_opacity_prior", false);
        const bool includeIntraSlabDepth =
                get_b(optionsDictionary, "include_intra_slab_depth", false);
        const bool includeCurvatureScale =
                get_b(optionsDictionary, "include_curvature_scale", false);

        const Pale::PointGradients *depthGradients =
                includeDepthDistortion ? &depthDistortionGradients : nullptr;
        const Pale::PointGradients *normalGradients =
                includeNormalConsistency ? &normalConsistencyGradients : nullptr;
        const Pale::PointGradients *visibilityGradients =
                includeVisibilityWeightedOpacity ? &visibilityOpacityGradients : nullptr;
        const Pale::PointGradients *intraSlabGradients =
                includeIntraSlabDepth ? &intraSlabDepthGradients : nullptr;
        const Pale::PointGradients *curvatureGradients =
                includeCurvatureScale ? &curvatureScaleGradients : nullptr;

        {
            py::gil_scoped_release release;
            ensureDeviceTrainingState(gradients.numPoints, syclQueue);
            launchDeviceTrainingStepKernel(
                syclQueue,
                gradients,
                depthGradients,
                normalGradients,
                visibilityGradients,
                intraSlabGradients,
                curvatureGradients,
                options);
            launchPointBvhRefitKernel(syclQueue);
            syclQueue.wait_and_throw();
            devicePointParametersDirty = true;
        }

        py::dict result;
        result["point_count"] = static_cast<std::uint64_t>(gradients.numPoints);
        result["optimizer_step"] = static_cast<std::uint64_t>(deviceTrainingState.step);
        return result;
    }

    py::dict render_forward_surface_regularizer_loss_and_adjoint(
        const py::list &cameraNamesList,
        const py::dict &optionsDictionary = py::dict()) {
        auto syclQueue = deviceSelector->getQueue();
        SelectedTrainingBatch selectedBatch =
                selectTrainingBatch(cameraNamesList, "render_forward_surface_regularizer_loss_and_adjoint");

        const bool useDepthDistortion =
                get_b(optionsDictionary, "use_depth_distortion", false);
        const bool useNormalConsistency =
                get_b(optionsDictionary, "use_normal_consistency", false);
        const bool useVisibilityWeightedOpacity =
                get_b(optionsDictionary, "use_opacity_prior", false);
        const bool useIntraSlabDepth =
                get_b(optionsDictionary, "use_intra_slab_depth", false);
        const bool useCurvatureScale =
                get_b(optionsDictionary, "use_curvature_scale", false);
        const float depthDistortionWeight =
                get_f(optionsDictionary, "depth_distortion_weight", 0.0f);
        const float normalConsistencyWeight =
                get_f(optionsDictionary, "normal_consistency_weight", 0.0f);
        const float visibilityWeightedOpacityWeight =
                get_f(optionsDictionary, "opacity_prior_weight", 0.0f);
        const float intraSlabDepthWeight =
                get_f(optionsDictionary, "intra_slab_depth_weight", 0.0f);
        const float curvatureScaleWeight =
                get_f(optionsDictionary, "curvature_scale_weight", 0.0f);

        const std::size_t cameraCount = selectedBatch.sensors.size();
        std::vector<float> depthDistortionSums(cameraCount, 0.0f);
        std::vector<float> normalConsistencySums(cameraCount, 0.0f);
        std::vector<float> visibilityWeightedOpacitySums(cameraCount, 0.0f);
        std::vector<float> intraSlabDepthSums(cameraCount, 0.0f);
        std::vector<float> curvatureScaleSums(cameraCount, 0.0f);
        std::vector<std::uint32_t> normalConsistencyValidCounts(cameraCount, 0u);
        std::vector<std::uint32_t> intraSlabDepthActiveSlabCounts(cameraCount, 0u);
        std::vector<std::uint32_t> curvatureScaleActiveSlabCounts(cameraCount, 0u);

        float *depthDistortionSumsDevice = nullptr;
        float *normalConsistencySumsDevice = nullptr;
        float *visibilityWeightedOpacitySumsDevice = nullptr;
        float *intraSlabDepthSumsDevice = nullptr;
        float *curvatureScaleSumsDevice = nullptr;
        std::uint32_t *normalConsistencyValidCountsDevice = nullptr;
        std::uint32_t *intraSlabDepthActiveSlabCountsDevice = nullptr;
        std::uint32_t *curvatureScaleActiveSlabCountsDevice = nullptr;

        auto freeTempBuffers = [&]() {
            if (depthDistortionSumsDevice) {
                sycl::free(depthDistortionSumsDevice, syclQueue);
                depthDistortionSumsDevice = nullptr;
            }
            if (normalConsistencySumsDevice) {
                sycl::free(normalConsistencySumsDevice, syclQueue);
                normalConsistencySumsDevice = nullptr;
            }
            if (visibilityWeightedOpacitySumsDevice) {
                sycl::free(visibilityWeightedOpacitySumsDevice, syclQueue);
                visibilityWeightedOpacitySumsDevice = nullptr;
            }
            if (intraSlabDepthSumsDevice) {
                sycl::free(intraSlabDepthSumsDevice, syclQueue);
                intraSlabDepthSumsDevice = nullptr;
            }
            if (curvatureScaleSumsDevice) {
                sycl::free(curvatureScaleSumsDevice, syclQueue);
                curvatureScaleSumsDevice = nullptr;
            }
            if (normalConsistencyValidCountsDevice) {
                sycl::free(normalConsistencyValidCountsDevice, syclQueue);
                normalConsistencyValidCountsDevice = nullptr;
            }
            if (intraSlabDepthActiveSlabCountsDevice) {
                sycl::free(intraSlabDepthActiveSlabCountsDevice, syclQueue);
                intraSlabDepthActiveSlabCountsDevice = nullptr;
            }
            if (curvatureScaleActiveSlabCountsDevice) {
                sycl::free(curvatureScaleActiveSlabCountsDevice, syclQueue);
                curvatureScaleActiveSlabCountsDevice = nullptr;
            }
        };

        try {
            py::gil_scoped_release release;

            m_settings.depthDistortionWeight = depthDistortionWeight;
            m_settings.normalConsistencyWeight = normalConsistencyWeight;
            m_settings.visibilityWeightedOpacityRegularizerWeight = visibilityWeightedOpacityWeight;
            m_settings.intraSlabDepthRegularizerWeight = intraSlabDepthWeight;
            m_settings.curvatureScaleRegularizerWeight = curvatureScaleWeight;
            auto &pathSettings = pathTracer->getSettings();
            pathSettings.depthDistortionWeight = depthDistortionWeight;
            pathSettings.normalConsistencyWeight = normalConsistencyWeight;
            pathSettings.visibilityWeightedOpacityRegularizerWeight = visibilityWeightedOpacityWeight;
            pathSettings.intraSlabDepthRegularizerWeight = intraSlabDepthWeight;
            pathSettings.curvatureScaleRegularizerWeight = curvatureScaleWeight;

            pathTracer->renderForward(selectedBatch.sensors);

            depthDistortionSumsDevice = sycl::malloc_device<float>(cameraCount, syclQueue);
            normalConsistencySumsDevice = sycl::malloc_device<float>(cameraCount, syclQueue);
            visibilityWeightedOpacitySumsDevice = sycl::malloc_device<float>(cameraCount, syclQueue);
            intraSlabDepthSumsDevice = sycl::malloc_device<float>(cameraCount, syclQueue);
            curvatureScaleSumsDevice = sycl::malloc_device<float>(cameraCount, syclQueue);
            normalConsistencyValidCountsDevice =
                    sycl::malloc_device<std::uint32_t>(cameraCount, syclQueue);
            intraSlabDepthActiveSlabCountsDevice =
                    sycl::malloc_device<std::uint32_t>(cameraCount, syclQueue);
            curvatureScaleActiveSlabCountsDevice =
                    sycl::malloc_device<std::uint32_t>(cameraCount, syclQueue);
            if (!depthDistortionSumsDevice ||
                !normalConsistencySumsDevice ||
                !visibilityWeightedOpacitySumsDevice ||
                !intraSlabDepthSumsDevice ||
                !curvatureScaleSumsDevice ||
                !normalConsistencyValidCountsDevice ||
                !intraSlabDepthActiveSlabCountsDevice ||
                !curvatureScaleActiveSlabCountsDevice) {
                throw std::runtime_error(
                    "render_forward_surface_regularizer_loss_and_adjoint: failed to allocate temporary loss buffers");
            }

            syclQueue.fill(depthDistortionSumsDevice, 0.0f, cameraCount);
            syclQueue.fill(normalConsistencySumsDevice, 0.0f, cameraCount);
            syclQueue.fill(visibilityWeightedOpacitySumsDevice, 0.0f, cameraCount);
            syclQueue.fill(intraSlabDepthSumsDevice, 0.0f, cameraCount);
            syclQueue.fill(curvatureScaleSumsDevice, 0.0f, cameraCount);
            syclQueue.fill(normalConsistencyValidCountsDevice, 0u, cameraCount);
            syclQueue.fill(intraSlabDepthActiveSlabCountsDevice, 0u, cameraCount);
            syclQueue.fill(curvatureScaleActiveSlabCountsDevice, 0u, cameraCount);

            for (std::size_t cameraIndex = 0; cameraIndex < cameraCount; ++cameraIndex) {
                launchSurfaceRegularizerLossAccumulationKernel(
                    syclQueue,
                    selectedBatch.sensors[cameraIndex],
                    depthDistortionSumsDevice + cameraIndex,
                    normalConsistencySumsDevice + cameraIndex,
                    visibilityWeightedOpacitySumsDevice + cameraIndex,
                    intraSlabDepthSumsDevice + cameraIndex,
                    curvatureScaleSumsDevice + cameraIndex,
                    normalConsistencyValidCountsDevice + cameraIndex,
                    intraSlabDepthActiveSlabCountsDevice + cameraIndex,
                    curvatureScaleActiveSlabCountsDevice + cameraIndex,
                    useDepthDistortion,
                    useNormalConsistency,
                    useVisibilityWeightedOpacity,
                    useIntraSlabDepth,
                    useCurvatureScale);
            }

            syclQueue.memcpy(
                depthDistortionSums.data(),
                depthDistortionSumsDevice,
                cameraCount * sizeof(float));
            syclQueue.memcpy(
                normalConsistencySums.data(),
                normalConsistencySumsDevice,
                cameraCount * sizeof(float));
            syclQueue.memcpy(
                visibilityWeightedOpacitySums.data(),
                visibilityWeightedOpacitySumsDevice,
                cameraCount * sizeof(float));
            syclQueue.memcpy(
                intraSlabDepthSums.data(),
                intraSlabDepthSumsDevice,
                cameraCount * sizeof(float));
            syclQueue.memcpy(
                curvatureScaleSums.data(),
                curvatureScaleSumsDevice,
                cameraCount * sizeof(float));
            syclQueue.memcpy(
                normalConsistencyValidCounts.data(),
                normalConsistencyValidCountsDevice,
                cameraCount * sizeof(std::uint32_t));
            syclQueue.memcpy(
                intraSlabDepthActiveSlabCounts.data(),
                intraSlabDepthActiveSlabCountsDevice,
                cameraCount * sizeof(std::uint32_t));
            syclQueue.memcpy(
                curvatureScaleActiveSlabCounts.data(),
                curvatureScaleActiveSlabCountsDevice,
                cameraCount * sizeof(std::uint32_t));
            syclQueue.wait_and_throw();

            for (std::size_t cameraIndex = 0; cameraIndex < cameraCount; ++cameraIndex) {
                launchSurfaceRegularizerAdjointFillKernel(
                    syclQueue,
                    selectedBatch.sensors[cameraIndex],
                    depthDistortionWeight,
                    normalConsistencyWeight,
                    intraSlabDepthWeight,
                    curvatureScaleWeight,
                    normalConsistencyValidCounts[cameraIndex],
                    intraSlabDepthActiveSlabCounts[cameraIndex],
                    curvatureScaleActiveSlabCounts[cameraIndex],
                    useDepthDistortion,
                    useNormalConsistency,
                    useIntraSlabDepth,
                    useCurvatureScale);
            }
            syclQueue.wait_and_throw();
        } catch (...) {
            freeTempBuffers();
            throw;
        }
        freeTempBuffers();

        py::dict result = makeZeroLossValuesDictionary();
        result["depth_distortion_grad_images"] = py::dict();
        result["visible_normal_adjoints"] = py::dict();
        result["depth_normal_adjoints"] = py::dict();
        result["depth_distortion_maps_for_logging"] = py::dict();
        result["intra_slab_depth_maps_for_logging"] = py::dict();
        result["curvature_scale_maps_for_logging"] = py::dict();
        py::dict perCameraLossValues;

        float totalDepthRaw = 0.0f;
        float totalDepthWeighted = 0.0f;
        float totalNormalRaw = 0.0f;
        float totalNormalWeighted = 0.0f;
        float totalVisibilityOpacityRaw = 0.0f;
        float totalVisibilityOpacityWeighted = 0.0f;
        float totalIntraSlabDepthRaw = 0.0f;
        float totalIntraSlabDepthWeighted = 0.0f;
        float totalCurvatureScaleRaw = 0.0f;
        float totalCurvatureScaleWeighted = 0.0f;

        for (std::size_t cameraIndex = 0; cameraIndex < cameraCount; ++cameraIndex) {
            const Pale::SensorGPU &sensor = selectedBatch.sensors[cameraIndex];
            const std::string cameraName(
                sensor.name,
                strnlen(sensor.name, sizeof(sensor.name)));
            const float pixelCount =
                    std::max(1.0f, static_cast<float>(sensor.width) * static_cast<float>(sensor.height));
            const float validNormalCount =
                    std::max(1.0f, static_cast<float>(normalConsistencyValidCounts[cameraIndex]));

            const float depthRaw = useDepthDistortion
                                       ? depthDistortionSums[cameraIndex] / pixelCount
                                       : 0.0f;
            const float depthWeighted = depthRaw * depthDistortionWeight;
            const float normalRaw = useNormalConsistency
                                        ? normalConsistencySums[cameraIndex] / validNormalCount
                                        : 0.0f;
            const float normalWeighted = normalRaw * normalConsistencyWeight;
            const float visibilityOpacityRaw = useVisibilityWeightedOpacity
                                                   ? visibilityWeightedOpacitySums[cameraIndex] / pixelCount
                                                   : 0.0f;
            const float visibilityOpacityWeighted = visibilityOpacityRaw * visibilityWeightedOpacityWeight;
            const float activeIntraSlabCount = std::max(
                1.0f,
                static_cast<float>(intraSlabDepthActiveSlabCounts[cameraIndex]));
            const float activeCurvatureSlabCount = std::max(
                1.0f,
                static_cast<float>(curvatureScaleActiveSlabCounts[cameraIndex]));
            const float intraSlabDepthRaw = useIntraSlabDepth
                ? intraSlabDepthSums[cameraIndex] / activeIntraSlabCount
                : 0.0f;
            const float intraSlabDepthWeighted = intraSlabDepthRaw * intraSlabDepthWeight;
            const float curvatureScaleRaw = useCurvatureScale
                ? curvatureScaleSums[cameraIndex] / activeCurvatureSlabCount
                : 0.0f;
            const float curvatureScaleWeighted = curvatureScaleRaw * curvatureScaleWeight;

            py::dict cameraLossValues = makeZeroLossValuesDictionary();
            cameraLossValues["total_depth_distortion_loss_raw"] = depthRaw;
            cameraLossValues["total_depth_distortion_loss_weighted"] = depthWeighted;
            cameraLossValues["total_normal_loss_raw"] = normalRaw;
            cameraLossValues["total_normal_loss_weighted"] = normalWeighted;
            cameraLossValues["total_opacity_prior_loss_raw"] = visibilityOpacityRaw;
            cameraLossValues["total_opacity_prior_loss_weighted"] = visibilityOpacityWeighted;
            cameraLossValues["total_intra_slab_depth_loss_raw"] = intraSlabDepthRaw;
            cameraLossValues["total_intra_slab_depth_loss_weighted"] = intraSlabDepthWeighted;
            cameraLossValues["total_curvature_scale_loss_raw"] = curvatureScaleRaw;
            cameraLossValues["total_curvature_scale_loss_weighted"] = curvatureScaleWeighted;
            cameraLossValues["total_loss_value"] =
                depthWeighted + normalWeighted + visibilityOpacityWeighted +
                intraSlabDepthWeighted + curvatureScaleWeighted;
            perCameraLossValues[py::str(cameraName)] = cameraLossValues;

            totalDepthRaw += depthRaw;
            totalDepthWeighted += depthWeighted;
            totalNormalRaw += normalRaw;
            totalNormalWeighted += normalWeighted;
            totalVisibilityOpacityRaw += visibilityOpacityRaw;
            totalVisibilityOpacityWeighted += visibilityOpacityWeighted;
            totalIntraSlabDepthRaw += intraSlabDepthRaw;
            totalIntraSlabDepthWeighted += intraSlabDepthWeighted;
            totalCurvatureScaleRaw += curvatureScaleRaw;
            totalCurvatureScaleWeighted += curvatureScaleWeighted;
        }

        result["total_depth_distortion_loss_raw"] = totalDepthRaw;
        result["total_depth_distortion_loss_weighted"] = totalDepthWeighted;
        result["total_normal_loss_raw"] = totalNormalRaw;
        result["total_normal_loss_weighted"] = totalNormalWeighted;
        result["total_opacity_prior_loss_raw"] = totalVisibilityOpacityRaw;
        result["total_opacity_prior_loss_weighted"] = totalVisibilityOpacityWeighted;
        result["total_intra_slab_depth_loss_raw"] = totalIntraSlabDepthRaw;
        result["total_intra_slab_depth_loss_weighted"] = totalIntraSlabDepthWeighted;
        result["total_curvature_scale_loss_raw"] = totalCurvatureScaleRaw;
        result["total_curvature_scale_loss_weighted"] = totalCurvatureScaleWeighted;
        result["total_loss_value"] =
            totalDepthWeighted + totalNormalWeighted + totalVisibilityOpacityWeighted +
            totalIntraSlabDepthWeighted + totalCurvatureScaleWeighted;
        result["per_camera_loss_values"] = std::move(perCameraLossValues);
        return result;
    }

    py::dict render_surface_regularizers_backward_from_current_adjoint(
        const py::list &cameraNamesList,
        bool returnGradients = false) {
        SelectedTrainingBatch selectedBatch =
                selectTrainingBatch(cameraNamesList, "render_surface_regularizers_backward_from_current_adjoint");

        {
            py::gil_scoped_release release;
            pathTracer->renderSurfaceRegularizersBackward(
                selectedBatch.sensors,
                depthDistortionGradients,
                normalConsistencyGradients,
                visibilityOpacityGradients,
                intraSlabDepthGradients,
                curvatureScaleGradients,
                selectedBatch.debugImages.data());
        }

        if (!returnGradients) {
            return py::dict{};
        }

        py::dict result;
        result["depth_distortion"] = makeGradientDictionary(depthDistortionGradients);
        result["normal_consistency"] = makeGradientDictionary(normalConsistencyGradients);
        result["opacity_prior"] = makeGradientDictionary(visibilityOpacityGradients);
        result["intra_slab_depth"] = makeGradientDictionary(intraSlabDepthGradients);
        result["curvature_scale"] = makeGradientDictionary(curvatureScaleGradients);
        return result;
    }

    void reset_trainable_opacity_on_gpu(float opacityValue) {
        if (!sceneGpu.points || sceneGpu.pointCount == 0u) {
            return;
        }

        auto syclQueue = deviceSelector->getQueue();
        {
            py::gil_scoped_release release;
            syclQueue.parallel_for<class ResetTrainableOpacityKernelTag>(
                sycl::range<1>(sceneGpu.pointCount),
                [points = sceneGpu.points, opacityValue](sycl::id<1> itemId) {
                    Pale::Point &point = points[static_cast<std::uint32_t>(itemId[0])];
                    if (!point.isEmissive()) {
                        point.opacity = sycl::fmin(sycl::fmax(opacityValue, 0.0f), 1.0f);
                    }
                });
            syclQueue.wait_and_throw();
            devicePointParametersDirty = true;
        }
    }

    py::tuple render_backward(const py::dict &targetImagesDictionary) {
        using std::int64_t;
        using std::size_t;

        auto syclQueue = deviceSelector->getQueue();

        struct HostAdjointImage {
            std::string cameraName;
            std::uint32_t imageWidth{};
            std::uint32_t imageHeight{};
            std::vector<float> imageRgbaData; // H * W * 4
        };

        std::vector<Pale::SensorGPU> availableAdjointSensors;
        std::vector<Pale::DebugImages> availableDebugImages;
        std::vector<HostAdjointImage> hostAdjointImages;

        // Map cameraName -> RGBA target buffer (HxWx4 float)
        std::unordered_map<std::string, std::vector<float> > targetRgbaPerCamera;

        // 1. WITH GIL: read Python dict, convert to RGBA buffers
        // ------------------------------------------------------------
        for (std::size_t sensorIndex = 0; sensorIndex < sensorsForward.size(); ++sensorIndex) {
            const auto &sensor = sensorsForward[sensorIndex];

            if (!sensor.camera.useForAdjointPass) {
                continue;
            }

            std::string cameraName(sensor.name, strnlen(sensor.name, sizeof(sensor.name)));

            if (!targetImagesDictionary.contains(py::str(cameraName))) {
                continue;
            }

            py::array targetRgbArray =
                    targetImagesDictionary[py::str(cameraName)].cast<py::array>();

            py::buffer_info bufferInfo = targetRgbArray.request();
            if (bufferInfo.ndim != 3 || bufferInfo.shape[2] != 3) {
                throw std::runtime_error(
                    "render_backward: target image for camera '" + cameraName +
                    "' must be HxWx3 float32"
                );
            }
            if (bufferInfo.itemsize != sizeof(float)) {
                throw std::runtime_error(
                    "render_backward: target image for camera '" + cameraName +
                    "' must have dtype float32"
                );
            }

            const int64_t height = static_cast<int64_t>(bufferInfo.shape[0]);
            const int64_t width = static_cast<int64_t>(bufferInfo.shape[1]);

            if (static_cast<std::uint32_t>(width) != sensor.width ||
                static_cast<std::uint32_t>(height) != sensor.height) {
                throw std::runtime_error(
                    "render_backward: resolution mismatch for camera '" +
                    cameraName + "': target image is " + std::to_string(width) +
                    "x" + std::to_string(height) + ", but sensor is " +
                    std::to_string(sensor.width) + "x" +
                    std::to_string(sensor.height)
                );
            }

            const auto *rgbPointer = static_cast<const float *>(bufferInfo.ptr);

            std::vector<float> rgbaTarget;
            rgbaTarget.resize(
                static_cast<size_t>(height) *
                static_cast<size_t>(width) * 4u
            );

            for (int64_t pixelY = 0; pixelY < height; ++pixelY) {
                for (int64_t pixelX = 0; pixelX < width; ++pixelX) {
                    const size_t rgbIndex =
                            static_cast<size_t>((pixelY * width + pixelX) * 3);
                    const size_t rgbaIndex =
                            static_cast<size_t>((pixelY * width + pixelX) * 4);

                    rgbaTarget[rgbaIndex + 0] = rgbPointer[rgbIndex + 0];
                    rgbaTarget[rgbaIndex + 1] = rgbPointer[rgbIndex + 1];
                    rgbaTarget[rgbaIndex + 2] = rgbPointer[rgbIndex + 2];
                    rgbaTarget[rgbaIndex + 3] = 1.0f;
                }
            }

            targetRgbaPerCamera.emplace(std::move(cameraName), std::move(rgbaTarget));
            availableAdjointSensors.push_back(sensor);
            availableDebugImages.push_back(debugImages[sensorIndex]);
        }

        // ------------------------------------------------------------
        // 2. WITHOUT GIL: upload targets, run adjoint, download gradients
        // ------------------------------------------------------------
        py::gil_scoped_release release;

        // 2a. Upload RGBA targets per sensor
        for (auto &sensor: availableAdjointSensors) {
            std::string cameraName(
                sensor.name,
                strnlen(sensor.name, sizeof(sensor.name))
            );
            auto it = targetRgbaPerCamera.find(cameraName);
            if (it == targetRgbaPerCamera.end()) {
                continue; // should not happen given checks above
            }
            uploadSensorRGBA(syclQueue, sensor, it->second);
        }

        // 2b. Run backward pass (re-enable when ready)
        pathTracer->renderBackward(
            availableAdjointSensors,
            gradients,
            availableDebugImages.data());

        const std::size_t pointCount = gradients.numPoints;
        const std::size_t cameraSlotCount = gradients.cameraSlotCount;
        const std::size_t primitiveCameraCount = pointCount * cameraSlotCount;

        std::vector<Pale::float3> gradPositionHost(pointCount);
        std::vector<Pale::float3> cloneSignalHost(pointCount);
        std::vector<Pale::float3> gradRotationHost(pointCount);
        std::vector<Pale::float2> gradScaleHost(pointCount);
        std::vector<Pale::float3> gradColorHost(pointCount);
        std::vector<float> gradOpacityHost(pointCount);
        std::vector<float> gradBetaHost(pointCount);
        std::vector<float> gradShapeHost(pointCount);
        std::vector<float> gradPowerHost(pointCount);

        std::vector<float> cloneSignalMeanNormHost(pointCount);
        std::vector<float> cloneSignalStdHost(pointCount);
        std::vector<float> cloneSignalCoherenceHost(pointCount);
        std::vector<float> cloneSignalDisagreementHost(pointCount);
        std::vector<uint32_t> cloneSignalActiveCameraCountHost(pointCount);

        std::vector<Pale::float3> gradPositionPerPrimitivePerCameraHost(primitiveCameraCount);
        std::vector<uint32_t> gradPositionRecordCountPerPrimitivePerCameraHost(primitiveCameraCount);
        std::vector<Pale::float3> cloneSignalPerPrimitivePerCameraHost(primitiveCameraCount);
        std::vector<uint32_t> cloneSignalRecordCountPerPrimitivePerCameraHost(primitiveCameraCount);

        if (pointCount > 0) {
            if (gradients.gradPosition) {
                syclQueue.memcpy(
                    gradPositionHost.data(),
                    gradients.gradPosition,
                    pointCount * sizeof(Pale::float3)
                );
            }
            if (gradients.cloneSignal) {
                syclQueue.memcpy(
                    cloneSignalHost.data(),
                    gradients.cloneSignal,
                    pointCount * sizeof(Pale::float3)
                );
            }
            if (gradients.gradRotation) {
                syclQueue.memcpy(
                    gradRotationHost.data(),
                    gradients.gradRotation,
                    pointCount * sizeof(Pale::float3)
                );
            }

            if (gradients.gradScale) {
                syclQueue.memcpy(
                    gradScaleHost.data(),
                    gradients.gradScale,
                    pointCount * sizeof(Pale::float2)
                );
            }
            if (gradients.gradAlbedo) {
                syclQueue.memcpy(
                    gradColorHost.data(),
                    gradients.gradAlbedo,
                    pointCount * sizeof(Pale::float3)
                );
            }
            if (gradients.gradOpacity) {
                syclQueue.memcpy(
                    gradOpacityHost.data(),
                    gradients.gradOpacity,
                    pointCount * sizeof(float)
                );
            }
            if (gradients.gradBeta) {
                syclQueue.memcpy(
                    gradBetaHost.data(),
                    gradients.gradBeta,
                    pointCount * sizeof(float)
                );
            }
            if (gradients.gradShape) {
                syclQueue.memcpy(
                    gradShapeHost.data(),
                    gradients.gradShape,
                    pointCount * sizeof(float)
                );
            }

            if (gradients.cloneSignalMeanNorm) {
                syclQueue.memcpy(
                    cloneSignalMeanNormHost.data(),
                    gradients.cloneSignalMeanNorm,
                    pointCount * sizeof(float));
            }

            if (gradients.cloneSignalStd) {
                syclQueue.memcpy(
                    cloneSignalStdHost.data(),
                    gradients.cloneSignalStd,
                    pointCount * sizeof(float));
            }

            if (gradients.cloneSignalCoherence) {
                syclQueue.memcpy(
                    cloneSignalCoherenceHost.data(),
                    gradients.cloneSignalCoherence,
                    pointCount * sizeof(float));
            }

            if (gradients.cloneSignalDisagreement) {
                syclQueue.memcpy(
                    cloneSignalDisagreementHost.data(),
                    gradients.cloneSignalDisagreement,
                    pointCount * sizeof(float));
            }

            if (gradients.cloneSignalActiveCameraCount) {
                syclQueue.memcpy(
                    cloneSignalActiveCameraCountHost.data(),
                    gradients.cloneSignalActiveCameraCount,
                    pointCount * sizeof(uint32_t));
            }

            if (primitiveCameraCount > 0 &&
                gradients.gradPositionPerPrimitivePerCamera) {
                syclQueue.memcpy(
                    gradPositionPerPrimitivePerCameraHost.data(),
                    gradients.gradPositionPerPrimitivePerCamera,
                    primitiveCameraCount * sizeof(Pale::float3));
            }

            if (primitiveCameraCount > 0 &&
                gradients.gradPositionRecordCountPerPrimitivePerCamera) {
                syclQueue.memcpy(
                    gradPositionRecordCountPerPrimitivePerCameraHost.data(),
                    gradients.gradPositionRecordCountPerPrimitivePerCamera,
                    primitiveCameraCount * sizeof(uint32_t));
            }

            if (primitiveCameraCount > 0 &&
                gradients.cloneSignalPerPrimitivePerCamera) {
                syclQueue.memcpy(
                    cloneSignalPerPrimitivePerCameraHost.data(),
                    gradients.cloneSignalPerPrimitivePerCamera,
                    primitiveCameraCount * sizeof(Pale::float3));
            }

            if (primitiveCameraCount > 0 &&
                gradients.cloneSignalRecordCountPerPrimitivePerCamera) {
                syclQueue.memcpy(
                    cloneSignalRecordCountPerPrimitivePerCameraHost.data(),
                    gradients.cloneSignalRecordCountPerPrimitivePerCamera,
                    primitiveCameraCount * sizeof(uint32_t));
            }

            syclQueue.wait_and_throw();
        }

        // 2c. Download adjoint images per sensor
        for (auto &sensor: availableAdjointSensors) {
            HostAdjointImage hostImage;
            hostImage.cameraName = std::string(
                sensor.name,
                strnlen(sensor.name, sizeof(sensor.name))
            );
            hostImage.imageWidth = sensor.width;
            hostImage.imageHeight = sensor.height;
            hostImage.imageRgbaData =
                    Pale::downloadSensorRGBARAW(syclQueue, sensor);

            hostAdjointImages.push_back(std::move(hostImage));
        }

        // ------------------------------------------------------------
        // 3. WITH GIL: wrap gradients and images into NumPy/Python objects
        // ------------------------------------------------------------
        py::gil_scoped_acquire gilAcquire;

        auto makeFloat3Array =
                [](std::vector<Pale::float3> &hostVector, std::size_t elementCount) -> py::array {
            auto *ownedVector = new std::vector<Pale::float3>(std::move(hostVector));
            std::vector<ssize_t> arrayShape{
                static_cast<ssize_t>(elementCount),
                3
            };
            std::vector<ssize_t> arrayStrides{
                static_cast<ssize_t>(sizeof(Pale::float3)),
                static_cast<ssize_t>(sizeof(float))
            };

            return py::array(
                py::buffer_info(
                    ownedVector->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    2,
                    arrayShape,
                    arrayStrides
                ),
                py::capsule(ownedVector, [](void *pointer) {
                    delete static_cast<std::vector<Pale::float3> *>(pointer);
                })
            );
        };

        auto makeFloat2Array =
                [](std::vector<Pale::float2> &hostVector, std::size_t elementCount) -> py::array {
            auto *ownedVector = new std::vector<Pale::float2>(std::move(hostVector));
            std::vector<ssize_t> arrayShape{
                static_cast<ssize_t>(elementCount),
                2
            };
            std::vector<ssize_t> arrayStrides{
                static_cast<ssize_t>(sizeof(Pale::float2)),
                static_cast<ssize_t>(sizeof(float))
            };

            return py::array(
                py::buffer_info(
                    ownedVector->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    2,
                    arrayShape,
                    arrayStrides
                ),
                py::capsule(ownedVector, [](void *pointer) {
                    delete static_cast<std::vector<Pale::float2> *>(pointer);
                })
            );
        };

        auto makeFloat1Array =
                [](std::vector<float> &hostVector, std::size_t elementCount) -> py::array {
            auto *ownedVector = new std::vector<float>(std::move(hostVector));
            std::vector<ssize_t> arrayShape{
                static_cast<ssize_t>(elementCount)
            };
            std::vector<ssize_t> arrayStrides{
                static_cast<ssize_t>(sizeof(float))
            };

            return py::array(
                py::buffer_info(
                    ownedVector->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    1,
                    arrayShape,
                    arrayStrides
                ),
                py::capsule(ownedVector, [](void *pointer) {
                    delete static_cast<std::vector<float> *>(pointer);
                })
            );
        };

        auto makeUint1Array =
                [](std::vector<uint32_t> &hostVector, std::size_t elementCount) -> py::array {
            auto *ownedVector = new std::vector<uint32_t>(std::move(hostVector));

            std::vector<ssize_t> arrayShape{
                static_cast<ssize_t>(elementCount)
            };

            std::vector<ssize_t> arrayStrides{
                static_cast<ssize_t>(sizeof(uint32_t))
            };

            return py::array(
                py::buffer_info(
                    ownedVector->data(),
                    sizeof(uint32_t),
                    py::format_descriptor<uint32_t>::format(),
                    1,
                    arrayShape,
                    arrayStrides
                ),
                py::capsule(ownedVector, [](void *pointer) {
                    delete static_cast<std::vector<uint32_t> *>(pointer);
                })
            );
        };

        auto makeFloat3CameraArray =
                [](std::vector<Pale::float3> &hostVector,
                   std::size_t pointCount,
                   std::size_t cameraSlotCount) -> py::array {
            auto *ownedVector = new std::vector<Pale::float3>(std::move(hostVector));

            std::vector<ssize_t> arrayShape{
                static_cast<ssize_t>(pointCount),
                static_cast<ssize_t>(cameraSlotCount),
                3
            };

            std::vector<ssize_t> arrayStrides{
                static_cast<ssize_t>(cameraSlotCount * sizeof(Pale::float3)),
                static_cast<ssize_t>(sizeof(Pale::float3)),
                static_cast<ssize_t>(sizeof(float))
            };

            return py::array(
                py::buffer_info(
                    ownedVector->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    3,
                    arrayShape,
                    arrayStrides
                ),
                py::capsule(ownedVector, [](void *pointer) {
                    delete static_cast<std::vector<Pale::float3> *>(pointer);
                })
            );
        };

        auto makeUintCameraArray =
                [](std::vector<uint32_t> &hostVector,
                   std::size_t pointCount,
                   std::size_t cameraSlotCount) -> py::array {
            auto *ownedVector = new std::vector<uint32_t>(std::move(hostVector));

            std::vector<ssize_t> arrayShape{
                static_cast<ssize_t>(pointCount),
                static_cast<ssize_t>(cameraSlotCount)
            };

            std::vector<ssize_t> arrayStrides{
                static_cast<ssize_t>(cameraSlotCount * sizeof(uint32_t)),
                static_cast<ssize_t>(sizeof(uint32_t))
            };

            return py::array(
                py::buffer_info(
                    ownedVector->data(),
                    sizeof(uint32_t),
                    py::format_descriptor<uint32_t>::format(),
                    2,
                    arrayShape,
                    arrayStrides
                ),
                py::capsule(ownedVector, [](void *pointer) {
                    delete static_cast<std::vector<uint32_t> *>(pointer);
                })
            );
        };

        py::dict gradientDictionary;
        gradientDictionary["position"] = makeFloat3Array(gradPositionHost, pointCount);
        gradientDictionary["clone_signal"] = makeFloat3Array(cloneSignalHost, pointCount);
        gradientDictionary["rotation"] = makeFloat3Array(gradRotationHost, pointCount);
        gradientDictionary["scale"] = makeFloat2Array(gradScaleHost, pointCount);
        gradientDictionary["albedo"] = makeFloat3Array(gradColorHost, pointCount);
        gradientDictionary["opacity"] = makeFloat1Array(gradOpacityHost, pointCount);
        gradientDictionary["beta"] = makeFloat1Array(gradBetaHost, pointCount);
        gradientDictionary["shape"] = makeFloat1Array(gradShapeHost, pointCount);
        gradientDictionary["power"] = makeFloat1Array(gradPowerHost, pointCount);

        // Top-level container for all images
        py::dict adjointImagesDictionary;

        auto makeRgbaImageArray =
                [](std::vector<float> &imageBuffer,
                   std::uint32_t imageWidth,
                   std::uint32_t imageHeight) -> py::array {
            auto *ownedImageBuffer =
                    new std::vector<float>(std::move(imageBuffer));

            std::vector<ssize_t> rgbaShape{
                static_cast<ssize_t>(imageHeight),
                static_cast<ssize_t>(imageWidth),
                4
            };
            std::vector<ssize_t> rgbaStrides{
                static_cast<ssize_t>(imageWidth * 4 * sizeof(float)),
                static_cast<ssize_t>(4 * sizeof(float)),
                static_cast<ssize_t>(sizeof(float))
            };

            return py::array(
                py::buffer_info(
                    ownedImageBuffer->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    3,
                    rgbaShape,
                    rgbaStrides
                ),
                py::capsule(ownedImageBuffer, [](void *pointer) {
                    delete static_cast<std::vector<float> *>(pointer);
                })
            );
        };


        // 3a. Main adjoint source images per camera
        py::dict adjointSourceDict;
        for (auto &hostImage: hostAdjointImages) {
            py::array rgbaArray = makeRgbaImageArray(
                hostImage.imageRgbaData,
                hostImage.imageWidth,
                hostImage.imageHeight
            );

            adjointSourceDict[py::str(hostImage.cameraName)] = std::move(rgbaArray);
        }
        adjointImagesDictionary["adjoint_source"] = std::move(adjointSourceDict);


        if (pathTracer->getSettings().renderDebugGradientImages) {
            py::dict debugPerCameraDict;

            for (std::size_t i = 0; i < availableAdjointSensors.size(); ++i) {
                const auto &sensor = availableAdjointSensors[i];

                Pale::DebugGradientImagesHost debugImagesHost =
                        Pale::downloadDebugGradientImages(
                            deviceSelector->getQueue(),
                            sensor,
                            availableDebugImages[i]
                        );

                const std::uint32_t imageWidth = sensor.width;
                const std::uint32_t imageHeight = sensor.height;

                // Per-camera dict: {"position": img, "tangent_u": img, ...}
                py::dict cameraDebugDict;

                // Position gradient image
                if (!debugImagesHost.positionX.empty()) {
                    cameraDebugDict["position_x"] = makeRgbaImageArray(
                        debugImagesHost.positionX,
                        imageWidth,
                        imageHeight
                    );
                }
                // Position gradient image
                if (!debugImagesHost.positionY.empty()) {
                    cameraDebugDict["position_y"] = makeRgbaImageArray(
                        debugImagesHost.positionY,
                        imageWidth,
                        imageHeight
                    );
                }
                // Position gradient image
                if (!debugImagesHost.positionZ.empty()) {
                    cameraDebugDict["position_z"] = makeRgbaImageArray(
                        debugImagesHost.positionZ,
                        imageWidth,
                        imageHeight
                    );
                }

                // Tangent U gradient image
                if (!debugImagesHost.rotationX.empty()) {
                    cameraDebugDict["rotation_x"] = makeRgbaImageArray(
                        debugImagesHost.rotationX,
                        imageWidth,
                        imageHeight
                    );
                }

                // Tangent U gradient image
                if (!debugImagesHost.rotationY.empty()) {
                    cameraDebugDict["rotation_y"] = makeRgbaImageArray(
                        debugImagesHost.rotationY,
                        imageWidth,
                        imageHeight
                    );
                }

                // Tangent U gradient image
                if (!debugImagesHost.rotationZ.empty()) {
                    cameraDebugDict["rotation_z"] = makeRgbaImageArray(
                        debugImagesHost.rotationZ,
                        imageWidth,
                        imageHeight
                    );
                }


                // Scale gradient image
                if (!debugImagesHost.scaleU.empty()) {
                    cameraDebugDict["scale_u"] = makeRgbaImageArray(
                        debugImagesHost.scaleU,
                        imageWidth,
                        imageHeight
                    );
                }

                // Scale gradient image
                if (!debugImagesHost.scaleV.empty()) {
                    cameraDebugDict["scale_v"] = makeRgbaImageArray(
                        debugImagesHost.scaleV,
                        imageWidth,
                        imageHeight
                    );
                }

                // Color gradient image
                if (!debugImagesHost.albedo.empty()) {
                    cameraDebugDict["albedo"] = makeRgbaImageArray(
                        debugImagesHost.albedo,
                        imageWidth,
                        imageHeight
                    );
                }

                // Opacity gradient image
                if (!debugImagesHost.opacity.empty()) {
                    cameraDebugDict["opacity"] = makeRgbaImageArray(
                        debugImagesHost.opacity,
                        imageWidth,
                        imageHeight
                    );
                }

                // Beta gradient image
                if (!debugImagesHost.beta.empty()) {
                    cameraDebugDict["beta"] = makeRgbaImageArray(
                        debugImagesHost.beta,
                        imageWidth,
                        imageHeight
                    );
                }

                if (cameraDebugDict.size() > 0) {
                    std::string cameraName(sensor.name,
                                           strnlen(sensor.name, sizeof(sensor.name)));
                    debugPerCameraDict[py::str(cameraName)] = std::move(cameraDebugDict);
                }
            }

            adjointImagesDictionary["debug"] = std::move(debugPerCameraDict);
        }

        py::dict gradientStatsDictionary;
        gradientStatsDictionary["position_per_camera"] = makeFloat3CameraArray(
            gradPositionPerPrimitivePerCameraHost, pointCount, cameraSlotCount);
        gradientStatsDictionary["position_record_count_per_camera"] = makeUintCameraArray(
            gradPositionRecordCountPerPrimitivePerCameraHost, pointCount, cameraSlotCount);
        gradientStatsDictionary["clone_signal_mean_norm"] =
                makeFloat1Array(cloneSignalMeanNormHost, pointCount);
        gradientStatsDictionary["clone_signal_std"] =
                makeFloat1Array(cloneSignalStdHost, pointCount);
        gradientStatsDictionary["clone_signal_coherence"] =
                makeFloat1Array(cloneSignalCoherenceHost, pointCount);
        gradientStatsDictionary["clone_signal_disagreement"] =
                makeFloat1Array(cloneSignalDisagreementHost, pointCount);
        gradientStatsDictionary["clone_signal_active_camera_count"] =
                makeUint1Array(cloneSignalActiveCameraCountHost, pointCount);
        gradientStatsDictionary["clone_signal_per_camera"] = makeFloat3CameraArray(
            cloneSignalPerPrimitivePerCameraHost, pointCount, cameraSlotCount);
        gradientStatsDictionary["clone_signal_record_count_per_camera"] = makeUintCameraArray(
            cloneSignalRecordCountPerPrimitivePerCameraHost, pointCount, cameraSlotCount);
        adjointImagesDictionary["gradient_stats"] = std::move(gradientStatsDictionary);

        return py::make_tuple(gradientDictionary, adjointImagesDictionary);
    }

    py::dict render_depth_distortion_backward(const py::dict &distortionGradImagesDictionary) {
        using std::int64_t;
        using std::size_t;

        auto syclQueue = deviceSelector->getQueue();


        std::vector<Pale::SensorGPU> selectedCameras;
        selectedCameras.reserve(sensorsForward.size());

        std::unordered_map<std::string, std::vector<float> > distortionAdjointPerCamera;
        distortionAdjointPerCamera.reserve(sensorsForward.size());

        // ------------------------------------------------------------
        // 1. WITH GIL: read Python dict, validate HxW float32 images
        // ------------------------------------------------------------
        for (std::size_t i = 0; i < sensorsForward.size(); ++i) {
            const auto &sensor = sensorsForward[i];
            std::string cameraName(
                sensor.name,
                strnlen(sensor.name, sizeof(sensor.name))
            );

            if (!distortionGradImagesDictionary.contains(py::str(cameraName))) {
                continue;
            }

            py::array adjointArray =
                    distortionGradImagesDictionary[py::str(cameraName)].cast<py::array>();

            py::buffer_info bufferInfo = adjointArray.request();

            if (bufferInfo.ndim != 2) {
                throw std::runtime_error(
                    "render_depth_distortion_backward: adjoint image for camera '" +
                    cameraName + "' must be HxW float32"
                );
            }
            if (bufferInfo.itemsize != sizeof(float)) {
                throw std::runtime_error(
                    "render_depth_distortion_backward: adjoint image for camera '" +
                    cameraName + "' must have dtype float32"
                );
            }

            const int64_t height = static_cast<int64_t>(bufferInfo.shape[0]);
            const int64_t width = static_cast<int64_t>(bufferInfo.shape[1]);

            if (static_cast<std::uint32_t>(width) != sensor.width ||
                static_cast<std::uint32_t>(height) != sensor.height) {
                throw std::runtime_error(
                    "render_depth_distortion_backward: resolution mismatch for camera '" +
                    cameraName + "': adjoint image is " + std::to_string(width) +
                    "x" + std::to_string(height) + ", but sensor is " +
                    std::to_string(sensor.width) + "x" +
                    std::to_string(sensor.height)
                );
            }

            const auto *src = static_cast<const float *>(bufferInfo.ptr);
            const std::size_t pixelCount =
                    static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

            std::vector<float> hostAdjoint(pixelCount);
            std::memcpy(hostAdjoint.data(), src, pixelCount * sizeof(float));

            distortionAdjointPerCamera.emplace(cameraName, std::move(hostAdjoint));

            selectedCameras.push_back(sensor);
        }

        // Nothing to do
        if (selectedCameras.empty()) {
            py::dict emptyGradientDictionary;
            return emptyGradientDictionary;
        }

        // ------------------------------------------------------------
        // 2. WITHOUT GIL: upload adjoints, run backward pass, download gradients
        // ------------------------------------------------------------
        py::gil_scoped_release release;

        // 2a. Upload per-camera HxW float adjoint images into storage buffers
        for (auto &entry: selectedCameras) {
            const std::string cameraName(
                entry.name,
                strnlen(entry.name, sizeof(entry.name))
            );

            auto it = distortionAdjointPerCamera.find(cameraName);
            if (it == distortionAdjointPerCamera.end()) {
                continue;
            }
            Pale::uploadFloatImage(
                syclQueue,
                entry.depthDistortionAdjointBuffer,
                it->second
            );
        }

        // 2c. Run regularizer backward pass
        // Replace with your actual path tracer entry point if the signature differs.
        pathTracer->renderDepthDistortionBackward(selectedCameras, gradients);

        const std::size_t pointCount = gradients.numPoints;

        std::vector<Pale::float3> gradPositionHost(pointCount);
        std::vector<Pale::float3> gradRotationHost(pointCount);
        std::vector<Pale::float2> gradScaleHost(pointCount);
        std::vector<Pale::float3> gradColorHost(pointCount); // should remain zero
        std::vector<float> gradOpacityHost(pointCount);
        std::vector<float> gradBetaHost(pointCount);
        std::vector<float> gradShapeHost(pointCount); // should remain zero
        std::vector<float> gradPowerHost(pointCount); // should remain zero

        if (pointCount > 0) {
            if (gradients.gradPosition) {
                syclQueue.memcpy(
                    gradPositionHost.data(),
                    gradients.gradPosition,
                    pointCount * sizeof(Pale::float3)
                );
            }
            if (gradients.gradRotation) {
                syclQueue.memcpy(
                    gradRotationHost.data(),
                    gradients.gradRotation,
                    pointCount * sizeof(Pale::float3)
                );
            }

            if (gradients.gradScale) {
                syclQueue.memcpy(
                    gradScaleHost.data(),
                    gradients.gradScale,
                    pointCount * sizeof(Pale::float2)
                );
            }
            if (gradients.gradAlbedo) {
                syclQueue.memcpy(
                    gradColorHost.data(),
                    gradients.gradAlbedo,
                    pointCount * sizeof(Pale::float3)
                );
            }
            if (gradients.gradOpacity) {
                syclQueue.memcpy(
                    gradOpacityHost.data(),
                    gradients.gradOpacity,
                    pointCount * sizeof(float)
                );
            }
            if (gradients.gradBeta) {
                syclQueue.memcpy(
                    gradBetaHost.data(),
                    gradients.gradBeta,
                    pointCount * sizeof(float)
                );
            }

            syclQueue.wait_and_throw();
        }

        // ------------------------------------------------------------
        // 3. WITH GIL: wrap gradients into NumPy arrays
        // ------------------------------------------------------------
        py::gil_scoped_acquire gilAcquire;

        auto makeFloat3Array =
                [](std::vector<Pale::float3> &hostVector, std::size_t elementCount) -> py::array {
            auto *ownedVector = new std::vector<Pale::float3>(std::move(hostVector));
            std::vector<ssize_t> shape{
                static_cast<ssize_t>(elementCount),
                3
            };
            std::vector<ssize_t> strides{
                static_cast<ssize_t>(sizeof(Pale::float3)),
                static_cast<ssize_t>(sizeof(float))
            };

            return py::array(
                py::buffer_info(
                    ownedVector->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    2,
                    shape,
                    strides
                ),
                py::capsule(ownedVector, [](void *pointer) {
                    delete static_cast<std::vector<Pale::float3> *>(pointer);
                })
            );
        };

        auto makeFloat2Array =
                [](std::vector<Pale::float2> &hostVector, std::size_t elementCount) -> py::array {
            auto *ownedVector = new std::vector<Pale::float2>(std::move(hostVector));
            std::vector<ssize_t> shape{
                static_cast<ssize_t>(elementCount),
                2
            };
            std::vector<ssize_t> strides{
                static_cast<ssize_t>(sizeof(Pale::float2)),
                static_cast<ssize_t>(sizeof(float))
            };

            return py::array(
                py::buffer_info(
                    ownedVector->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    2,
                    shape,
                    strides
                ),
                py::capsule(ownedVector, [](void *pointer) {
                    delete static_cast<std::vector<Pale::float2> *>(pointer);
                })
            );
        };

        auto makeFloat1Array =
                [](std::vector<float> &hostVector, std::size_t elementCount) -> py::array {
            auto *ownedVector = new std::vector<float>(std::move(hostVector));
            std::vector<ssize_t> shape{
                static_cast<ssize_t>(elementCount)
            };
            std::vector<ssize_t> strides{
                static_cast<ssize_t>(sizeof(float))
            };

            return py::array(
                py::buffer_info(
                    ownedVector->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    1,
                    shape,
                    strides
                ),
                py::capsule(ownedVector, [](void *pointer) {
                    delete static_cast<std::vector<float> *>(pointer);
                })
            );
        };

        py::dict gradientDictionary;
        gradientDictionary["position"] = makeFloat3Array(gradPositionHost, pointCount);
        gradientDictionary["rotation"] = makeFloat3Array(gradRotationHost, pointCount);
        gradientDictionary["scale"] = makeFloat2Array(gradScaleHost, pointCount);
        gradientDictionary["albedo"] = makeFloat3Array(gradColorHost, pointCount);
        gradientDictionary["opacity"] = makeFloat1Array(gradOpacityHost, pointCount);
        gradientDictionary["beta"] = makeFloat1Array(gradBetaHost, pointCount);
        gradientDictionary["shape"] = makeFloat1Array(gradShapeHost, pointCount);
        gradientDictionary["power"] = makeFloat1Array(gradPowerHost, pointCount);

        return gradientDictionary;
    }

    py::dict render_normal_consistency_backward(
        const py::dict &visibleNormalGradImagesDictionary,
        const py::dict &normalFromDepthGradImagesDictionary) {
        using std::int64_t;
        using std::size_t;

        auto syclQueue = deviceSelector->getQueue();


        std::vector<Pale::SensorGPU> selectedCameras;
        selectedCameras.reserve(sensorsForward.size());

        std::unordered_map<std::string, std::vector<float> > visibleNormalAdjointPerCamera;
        std::unordered_map<std::string, std::vector<float> > normalFromDepthAdjointPerCamera;

        auto packNormalAdjointToRGBA =
                [](const py::array &adjointArray,
                   std::uint32_t expectedWidth,
                   std::uint32_t expectedHeight,
                   const std::string &cameraName,
                   const char *fieldName) -> std::vector<float> {
            py::buffer_info info = adjointArray.request();

            if (info.itemsize != sizeof(float)) {
                throw std::runtime_error(
                    std::string("render_normal_consistency_backward: '") +
                    fieldName + "' for camera '" + cameraName + "' must have dtype float32");
            }

            if (info.ndim != 3) {
                throw std::runtime_error(
                    std::string("render_normal_consistency_backward: '") +
                    fieldName + "' for camera '" + cameraName +
                    "' must have shape HxWx3 or HxWx4");
            }

            const int64_t height = static_cast<int64_t>(info.shape[0]);
            const int64_t width = static_cast<int64_t>(info.shape[1]);
            const int64_t channels = static_cast<int64_t>(info.shape[2]);

            if (channels != 3 && channels != 4) {
                throw std::runtime_error(
                    std::string("render_normal_consistency_backward: '") +
                    fieldName + "' for camera '" + cameraName +
                    "' must have shape HxWx3 or HxWx4");
            }

            if (static_cast<std::uint32_t>(width) != expectedWidth ||
                static_cast<std::uint32_t>(height) != expectedHeight) {
                throw std::runtime_error(
                    std::string("render_normal_consistency_backward: resolution mismatch for '") +
                    fieldName + "' camera '" + cameraName + "'");
            }

            const float *src = static_cast<const float *>(info.ptr);

            std::vector<float> rgba(
                static_cast<size_t>(width) * static_cast<size_t>(height) * 4u, 0.0f);

            for (int64_t y = 0; y < height; ++y) {
                for (int64_t x = 0; x < width; ++x) {
                    const size_t srcBase = static_cast<size_t>((y * width + x) * channels);
                    const size_t dstBase = static_cast<size_t>((y * width + x) * 4);

                    rgba[dstBase + 0] = src[srcBase + 0];
                    rgba[dstBase + 1] = src[srcBase + 1];
                    rgba[dstBase + 2] = src[srcBase + 2];
                    rgba[dstBase + 3] = 0.0f;
                }
            }

            return rgba;
        };

        for (const auto &sensor: sensorsForward) {
            std::string cameraName(
                sensor.name,
                strnlen(sensor.name, sizeof(sensor.name)));

            const bool hasVisible =
                    visibleNormalGradImagesDictionary.contains(py::str(cameraName));
            const bool hasDepth =
                    normalFromDepthGradImagesDictionary.contains(py::str(cameraName));

            if (!hasVisible && !hasDepth) {
                continue;
            }

            if (!hasVisible || !hasDepth) {
                throw std::runtime_error(
                    "render_normal_consistency_backward: both visible_normal and "
                    "normal_from_depth adjoints must be provided for camera '" + cameraName + "'");
            }

            py::array visibleAdjointArray =
                    visibleNormalGradImagesDictionary[py::str(cameraName)].cast<py::array>();

            py::array depthAdjointArray =
                    normalFromDepthGradImagesDictionary[py::str(cameraName)].cast<py::array>();

            visibleNormalAdjointPerCamera.emplace(
                cameraName,
                packNormalAdjointToRGBA(
                    visibleAdjointArray,
                    sensor.width,
                    sensor.height,
                    cameraName,
                    "visible_normal"));

            normalFromDepthAdjointPerCamera.emplace(
                cameraName,
                packNormalAdjointToRGBA(
                    depthAdjointArray,
                    sensor.width,
                    sensor.height,
                    cameraName,
                    "normal_from_depth"));

            selectedCameras.push_back(sensor);
        }

        if (selectedCameras.empty()) {
            py::dict emptyGradientDictionary;
            return emptyGradientDictionary;
        }

        py::gil_scoped_release release;

        for (auto &sensor: selectedCameras) {
            const std::string cameraName(
                sensor.name,
                strnlen(sensor.name, sizeof(sensor.name)));

            auto visIt = visibleNormalAdjointPerCamera.find(cameraName);
            auto depIt = normalFromDepthAdjointPerCamera.find(cameraName);

            if (visIt == visibleNormalAdjointPerCamera.end() ||
                depIt == normalFromDepthAdjointPerCamera.end()) {
                continue;
            }

            const std::size_t pixelCount =
                    static_cast<std::size_t>(sensor.width) *
                    static_cast<std::size_t>(sensor.height);

            syclQueue.memcpy(
                sensor.visibleNormalAdjointBuffer,
                visIt->second.data(),
                pixelCount * 4 * sizeof(float));

            syclQueue.memcpy(
                sensor.normalFromDepthAdjointBuffer,
                depIt->second.data(),
                pixelCount * 4 * sizeof(float));
        }

        syclQueue.wait_and_throw();

        pathTracer->renderNormalConsistencyBackward(selectedCameras, gradients);

        const std::size_t pointCount = gradients.numPoints;

        std::vector<Pale::float3> gradPositionHost(pointCount);
        std::vector<Pale::float3> gradRotationHost(pointCount);
        std::vector<Pale::float2> gradScaleHost(pointCount);
        std::vector<Pale::float3> gradColorHost(pointCount);
        std::vector<float> gradOpacityHost(pointCount);
        std::vector<float> gradBetaHost(pointCount);
        std::vector<float> gradShapeHost(pointCount);
        std::vector<float> gradPowerHost(pointCount);

        if (pointCount > 0) {
            if (gradients.gradPosition) {
                syclQueue.memcpy(
                    gradPositionHost.data(),
                    gradients.gradPosition,
                    pointCount * sizeof(Pale::float3));
            }
            if (gradients.gradRotation) {
                syclQueue.memcpy(
                    gradRotationHost.data(),
                    gradients.gradRotation,
                    pointCount * sizeof(Pale::float3));
            }

            if (gradients.gradScale) {
                syclQueue.memcpy(
                    gradScaleHost.data(),
                    gradients.gradScale,
                    pointCount * sizeof(Pale::float2));
            }
            if (gradients.gradAlbedo) {
                syclQueue.memcpy(
                    gradColorHost.data(),
                    gradients.gradAlbedo,
                    pointCount * sizeof(Pale::float3));
            }
            if (gradients.gradOpacity) {
                syclQueue.memcpy(
                    gradOpacityHost.data(),
                    gradients.gradOpacity,
                    pointCount * sizeof(float));
            }
            if (gradients.gradBeta) {
                syclQueue.memcpy(
                    gradBetaHost.data(),
                    gradients.gradBeta,
                    pointCount * sizeof(float));
            }

            syclQueue.wait_and_throw();
        }

        py::gil_scoped_acquire gilAcquire;

        auto makeFloat3Array =
                [](std::vector<Pale::float3> &hostVector, std::size_t elementCount) -> py::array {
            auto *ownedVector = new std::vector<Pale::float3>(std::move(hostVector));
            std::vector<ssize_t> shape{
                static_cast<ssize_t>(elementCount),
                3
            };
            std::vector<ssize_t> strides{
                static_cast<ssize_t>(sizeof(Pale::float3)),
                static_cast<ssize_t>(sizeof(float))
            };

            return py::array(
                py::buffer_info(
                    ownedVector->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    2,
                    shape,
                    strides
                ),
                py::capsule(ownedVector, [](void *pointer) {
                    delete static_cast<std::vector<Pale::float3> *>(pointer);
                })
            );
        };

        auto makeFloat2Array =
                [](std::vector<Pale::float2> &hostVector, std::size_t elementCount) -> py::array {
            auto *ownedVector = new std::vector<Pale::float2>(std::move(hostVector));
            std::vector<ssize_t> shape{
                static_cast<ssize_t>(elementCount),
                2
            };
            std::vector<ssize_t> strides{
                static_cast<ssize_t>(sizeof(Pale::float2)),
                static_cast<ssize_t>(sizeof(float))
            };

            return py::array(
                py::buffer_info(
                    ownedVector->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    2,
                    shape,
                    strides
                ),
                py::capsule(ownedVector, [](void *pointer) {
                    delete static_cast<std::vector<Pale::float2> *>(pointer);
                })
            );
        };

        auto makeFloat1Array =
                [](std::vector<float> &hostVector, std::size_t elementCount) -> py::array {
            auto *ownedVector = new std::vector<float>(std::move(hostVector));
            std::vector<ssize_t> shape{
                static_cast<ssize_t>(elementCount)
            };
            std::vector<ssize_t> strides{
                static_cast<ssize_t>(sizeof(float))
            };

            return py::array(
                py::buffer_info(
                    ownedVector->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    1,
                    shape,
                    strides
                ),
                py::capsule(ownedVector, [](void *pointer) {
                    delete static_cast<std::vector<float> *>(pointer);
                })
            );
        };

        py::dict gradientDictionary;
        gradientDictionary["position"] = makeFloat3Array(gradPositionHost, pointCount);
        gradientDictionary["rotation"] = makeFloat3Array(gradRotationHost, pointCount);
        gradientDictionary["scale"] = makeFloat2Array(gradScaleHost, pointCount);
        gradientDictionary["albedo"] = makeFloat3Array(gradColorHost, pointCount);
        gradientDictionary["opacity"] = makeFloat1Array(gradOpacityHost, pointCount);
        gradientDictionary["beta"] = makeFloat1Array(gradBetaHost, pointCount);
        gradientDictionary["shape"] = makeFloat1Array(gradShapeHost, pointCount);
        gradientDictionary["power"] = makeFloat1Array(gradPowerHost, pointCount);

        return gradientDictionary;
    }

    py::dict makeGradientDictionary(Pale::PointGradients &sourceGradients) {
        auto syclQueue = deviceSelector->getQueue();
        const std::size_t pointCount = sourceGradients.numPoints;

        std::vector<Pale::float3> gradPositionHost(pointCount);
        std::vector<Pale::float3> cloneSignalHost(pointCount);
        std::vector<Pale::float3> gradRotationHost(pointCount);
        std::vector<Pale::float2> gradScaleHost(pointCount);
        std::vector<Pale::float3> gradAlbedoHost(pointCount);
        std::vector<float> gradOpacityHost(pointCount, 0.0f);
        std::vector<float> gradBetaHost(pointCount, 0.0f);
        std::vector<float> gradShapeHost(pointCount, 0.0f);
        std::vector<float> gradPowerHost(pointCount, 0.0f);

        if (pointCount > 0) {
            if (sourceGradients.gradPosition) {
                syclQueue.memcpy(gradPositionHost.data(), sourceGradients.gradPosition,
                                 pointCount * sizeof(Pale::float3));
            }
            if (sourceGradients.cloneSignal) {
                syclQueue.memcpy(cloneSignalHost.data(), sourceGradients.cloneSignal,
                                 pointCount * sizeof(Pale::float3));
            }
            if (sourceGradients.gradRotation) {
                syclQueue.memcpy(gradRotationHost.data(), sourceGradients.gradRotation,
                                 pointCount * sizeof(Pale::float3));
            }
            if (sourceGradients.gradScale) {
                syclQueue.memcpy(gradScaleHost.data(), sourceGradients.gradScale, pointCount * sizeof(Pale::float2));
            }
            if (sourceGradients.gradAlbedo) {
                syclQueue.memcpy(gradAlbedoHost.data(), sourceGradients.gradAlbedo, pointCount * sizeof(Pale::float3));
            }
            if (sourceGradients.gradOpacity) {
                syclQueue.memcpy(gradOpacityHost.data(), sourceGradients.gradOpacity, pointCount * sizeof(float));
            }
            if (sourceGradients.gradBeta) {
                syclQueue.memcpy(gradBetaHost.data(), sourceGradients.gradBeta, pointCount * sizeof(float));
            }
            if (sourceGradients.gradShape) {
                syclQueue.memcpy(gradShapeHost.data(), sourceGradients.gradShape, pointCount * sizeof(float));
            }

            syclQueue.wait_and_throw();
        }

        auto makeFloat3Array = [](std::vector<Pale::float3> &hostVector, std::size_t elementCount) -> py::array {
            auto *ownedVector = new std::vector<Pale::float3>(std::move(hostVector));
            std::vector<ssize_t> arrayShape{static_cast<ssize_t>(elementCount), 3};
            std::vector<ssize_t> arrayStrides{
                static_cast<ssize_t>(sizeof(Pale::float3)),
                static_cast<ssize_t>(sizeof(float))
            };

            return py::array(
                py::buffer_info(
                    ownedVector->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    2,
                    arrayShape,
                    arrayStrides),
                py::capsule(ownedVector, [](void *pointer) {
                    delete static_cast<std::vector<Pale::float3> *>(pointer);
                }));
        };

        auto makeFloat2Array = [](std::vector<Pale::float2> &hostVector, std::size_t elementCount) -> py::array {
            auto *ownedVector = new std::vector<Pale::float2>(std::move(hostVector));
            std::vector<ssize_t> arrayShape{static_cast<ssize_t>(elementCount), 2};
            std::vector<ssize_t> arrayStrides{
                static_cast<ssize_t>(sizeof(Pale::float2)),
                static_cast<ssize_t>(sizeof(float))
            };

            return py::array(
                py::buffer_info(
                    ownedVector->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    2,
                    arrayShape,
                    arrayStrides),
                py::capsule(ownedVector, [](void *pointer) {
                    delete static_cast<std::vector<Pale::float2> *>(pointer);
                }));
        };

        auto makeFloat1Array = [](std::vector<float> &hostVector, std::size_t elementCount) -> py::array {
            auto *ownedVector = new std::vector<float>(std::move(hostVector));
            std::vector<ssize_t> arrayShape{static_cast<ssize_t>(elementCount)};
            std::vector<ssize_t> arrayStrides{static_cast<ssize_t>(sizeof(float))};

            return py::array(
                py::buffer_info(
                    ownedVector->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    1,
                    arrayShape,
                    arrayStrides),
                py::capsule(ownedVector, [](void *pointer) {
                    delete static_cast<std::vector<float> *>(pointer);
                }));
        };

        py::dict gradientDictionary;
        gradientDictionary["position"] = makeFloat3Array(gradPositionHost, pointCount);
        gradientDictionary["clone_signal"] = makeFloat3Array(cloneSignalHost, pointCount);
        gradientDictionary["rotation"] = makeFloat3Array(gradRotationHost, pointCount);
        gradientDictionary["scale"] = makeFloat2Array(gradScaleHost, pointCount);
        gradientDictionary["albedo"] = makeFloat3Array(gradAlbedoHost, pointCount);
        gradientDictionary["opacity"] = makeFloat1Array(gradOpacityHost, pointCount);
        gradientDictionary["beta"] = makeFloat1Array(gradBetaHost, pointCount);
        gradientDictionary["shape"] = makeFloat1Array(gradShapeHost, pointCount);
        gradientDictionary["power"] = makeFloat1Array(gradPowerHost, pointCount);
        return gradientDictionary;
    }

    py::dict makeGradientStatsDictionary(Pale::PointGradients &sourceGradients) {
        auto syclQueue = deviceSelector->getQueue();
        const std::size_t pointCount = sourceGradients.numPoints;
        const std::size_t cameraSlotCount = sourceGradients.cameraSlotCount;
        const std::size_t primitiveCameraCount = pointCount * cameraSlotCount;

        std::vector<Pale::float3> gradPositionPerPrimitivePerCameraHost(primitiveCameraCount);
        std::vector<uint32_t> gradPositionRecordCountPerPrimitivePerCameraHost(primitiveCameraCount, 0u);

        std::vector<float> cloneSignalMeanNormHost(pointCount, 0.0f);
        std::vector<float> cloneSignalStdHost(pointCount, 0.0f);
        std::vector<float> cloneSignalCoherenceHost(pointCount, 0.0f);
        std::vector<float> cloneSignalDisagreementHost(pointCount, 0.0f);
        std::vector<uint32_t> cloneSignalActiveCameraCountHost(pointCount, 0u);
        std::vector<Pale::float3> cloneSignalPerPrimitivePerCameraHost(primitiveCameraCount);
        std::vector<uint32_t> cloneSignalRecordCountPerPrimitivePerCameraHost(primitiveCameraCount, 0u);

        if (pointCount > 0) {
            if (sourceGradients.cloneSignalMeanNorm) {
                syclQueue.memcpy(cloneSignalMeanNormHost.data(), sourceGradients.cloneSignalMeanNorm,
                                 pointCount * sizeof(float));
            }
            if (sourceGradients.cloneSignalStd) {
                syclQueue.memcpy(cloneSignalStdHost.data(), sourceGradients.cloneSignalStd,
                                 pointCount * sizeof(float));
            }
            if (sourceGradients.cloneSignalCoherence) {
                syclQueue.memcpy(cloneSignalCoherenceHost.data(), sourceGradients.cloneSignalCoherence,
                                 pointCount * sizeof(float));
            }
            if (sourceGradients.cloneSignalDisagreement) {
                syclQueue.memcpy(cloneSignalDisagreementHost.data(), sourceGradients.cloneSignalDisagreement,
                                 pointCount * sizeof(float));
            }
            if (sourceGradients.cloneSignalActiveCameraCount) {
                syclQueue.memcpy(cloneSignalActiveCameraCountHost.data(), sourceGradients.cloneSignalActiveCameraCount,
                                 pointCount * sizeof(uint32_t));
            }
        }

        if (primitiveCameraCount > 0) {
            if (sourceGradients.gradPositionPerPrimitivePerCamera) {
                syclQueue.memcpy(gradPositionPerPrimitivePerCameraHost.data(),
                                 sourceGradients.gradPositionPerPrimitivePerCamera,
                                 primitiveCameraCount * sizeof(Pale::float3));
            }
            if (sourceGradients.gradPositionRecordCountPerPrimitivePerCamera) {
                syclQueue.memcpy(gradPositionRecordCountPerPrimitivePerCameraHost.data(),
                                 sourceGradients.gradPositionRecordCountPerPrimitivePerCamera,
                                 primitiveCameraCount * sizeof(uint32_t));
            }
            if (sourceGradients.cloneSignalPerPrimitivePerCamera) {
                syclQueue.memcpy(cloneSignalPerPrimitivePerCameraHost.data(),
                                 sourceGradients.cloneSignalPerPrimitivePerCamera,
                                 primitiveCameraCount * sizeof(Pale::float3));
            }
            if (sourceGradients.cloneSignalRecordCountPerPrimitivePerCamera) {
                syclQueue.memcpy(cloneSignalRecordCountPerPrimitivePerCameraHost.data(),
                                 sourceGradients.cloneSignalRecordCountPerPrimitivePerCamera,
                                 primitiveCameraCount * sizeof(uint32_t));
            }
        }
        syclQueue.wait_and_throw();

        auto makeFloat1Array = [](std::vector<float> &hostVector, std::size_t elementCount) -> py::array {
            auto *ownedVector = new std::vector<float>(std::move(hostVector));
            std::vector<ssize_t> arrayShape{static_cast<ssize_t>(elementCount)};
            std::vector<ssize_t> arrayStrides{static_cast<ssize_t>(sizeof(float))};
            return py::array(
                py::buffer_info(ownedVector->data(), sizeof(float), py::format_descriptor<float>::format(), 1,
                                arrayShape, arrayStrides),
                py::capsule(ownedVector, [](void *pointer) {
                    delete static_cast<std::vector<float> *>(pointer);
                }));
        };

        auto makeUint1Array = [](std::vector<uint32_t> &hostVector, std::size_t elementCount) -> py::array {
            auto *ownedVector = new std::vector<uint32_t>(std::move(hostVector));
            std::vector<ssize_t> arrayShape{static_cast<ssize_t>(elementCount)};
            std::vector<ssize_t> arrayStrides{static_cast<ssize_t>(sizeof(uint32_t))};
            return py::array(
                py::buffer_info(ownedVector->data(), sizeof(uint32_t), py::format_descriptor<uint32_t>::format(), 1,
                                arrayShape, arrayStrides),
                py::capsule(ownedVector, [](void *pointer) {
                    delete static_cast<std::vector<uint32_t> *>(pointer);
                }));
        };

        auto makeFloat3CameraArray =
                [](std::vector<Pale::float3> &hostVector, std::size_t pointCount,
                   std::size_t cameraSlotCount) -> py::array {
            auto *ownedVector = new std::vector<Pale::float3>(std::move(hostVector));
            std::vector<ssize_t> arrayShape{
                static_cast<ssize_t>(pointCount),
                static_cast<ssize_t>(cameraSlotCount),
                3
            };
            std::vector<ssize_t> arrayStrides{
                static_cast<ssize_t>(cameraSlotCount * sizeof(Pale::float3)),
                static_cast<ssize_t>(sizeof(Pale::float3)),
                static_cast<ssize_t>(sizeof(float))
            };
            return py::array(
                py::buffer_info(ownedVector->data(), sizeof(float), py::format_descriptor<float>::format(), 3,
                                arrayShape, arrayStrides),
                py::capsule(ownedVector, [](void *pointer) {
                    delete static_cast<std::vector<Pale::float3> *>(pointer);
                }));
        };

        auto makeUintCameraArray =
                [](std::vector<uint32_t> &hostVector, std::size_t pointCount,
                   std::size_t cameraSlotCount) -> py::array {
            auto *ownedVector = new std::vector<uint32_t>(std::move(hostVector));
            std::vector<ssize_t> arrayShape{
                static_cast<ssize_t>(pointCount),
                static_cast<ssize_t>(cameraSlotCount)
            };
            std::vector<ssize_t> arrayStrides{
                static_cast<ssize_t>(cameraSlotCount * sizeof(uint32_t)),
                static_cast<ssize_t>(sizeof(uint32_t))
            };
            return py::array(
                py::buffer_info(ownedVector->data(), sizeof(uint32_t), py::format_descriptor<uint32_t>::format(), 2,
                                arrayShape, arrayStrides),
                py::capsule(ownedVector, [](void *pointer) {
                    delete static_cast<std::vector<uint32_t> *>(pointer);
                }));
        };

        py::dict gradientStatsDictionary;
        gradientStatsDictionary["position_per_camera"] =
                makeFloat3CameraArray(gradPositionPerPrimitivePerCameraHost, pointCount, cameraSlotCount);
        gradientStatsDictionary["position_record_count_per_camera"] =
                makeUintCameraArray(gradPositionRecordCountPerPrimitivePerCameraHost, pointCount, cameraSlotCount);
        gradientStatsDictionary["clone_signal_mean_norm"] = makeFloat1Array(cloneSignalMeanNormHost, pointCount);
        gradientStatsDictionary["clone_signal_std"] = makeFloat1Array(cloneSignalStdHost, pointCount);
        gradientStatsDictionary["clone_signal_coherence"] = makeFloat1Array(cloneSignalCoherenceHost, pointCount);
        gradientStatsDictionary["clone_signal_disagreement"] = makeFloat1Array(cloneSignalDisagreementHost, pointCount);
        gradientStatsDictionary["clone_signal_active_camera_count"] =
                makeUint1Array(cloneSignalActiveCameraCountHost, pointCount);
        gradientStatsDictionary["clone_signal_per_camera"] =
                makeFloat3CameraArray(cloneSignalPerPrimitivePerCameraHost, pointCount, cameraSlotCount);
        gradientStatsDictionary["clone_signal_record_count_per_camera"] =
                makeUintCameraArray(cloneSignalRecordCountPerPrimitivePerCameraHost, pointCount, cameraSlotCount);
        return gradientStatsDictionary;
    }

    py::dict runSurfaceRegularizersBackward(
        const py::list &cameraNamesList,
        const py::dict &depthDistortionGradImagesDictionary,
        const py::dict &visibleNormalGradImagesDictionary,
        const py::dict &normalFromDepthGradImagesDictionary,
        const py::dict &intraSlabDepthGradImagesDictionary,
        const py::dict &curvatureScaleGradImagesDictionary,
        bool returnGradients) {
        using std::int64_t;
        using std::size_t;

        auto syclQueue = deviceSelector->getQueue();

        std::unordered_set<std::string> selectedCameraNames;
        for (const py::handle item: cameraNamesList) {
            selectedCameraNames.insert(py::cast<std::string>(item));
        }

        std::vector<Pale::SensorGPU> selectedCameras;
        std::vector<Pale::DebugImages> selectedDebugImages;

        selectedCameras.reserve(sensorsForward.size());
        selectedDebugImages.reserve(sensorsForward.size());

        std::unordered_map<std::string, std::vector<float> > depthAdjointPerCamera;
        std::unordered_map<std::string, std::vector<float> > visibleNormalAdjointPerCamera;
        std::unordered_map<std::string, std::vector<float> > normalFromDepthAdjointPerCamera;
        std::unordered_map<std::string, std::vector<float> > intraSlabDepthAdjointPerCamera;
        std::unordered_map<std::string, std::vector<float> > curvatureScaleAdjointPerCamera;

        auto packFloatImage = [](const py::array &array,
                                 std::uint32_t expectedWidth,
                                 std::uint32_t expectedHeight,
                                 const std::string &cameraName,
                                 const char *fieldName) -> std::vector<float> {
            py::buffer_info info = array.request();

            if (info.ndim != 2 || info.itemsize != sizeof(float)) {
                throw std::runtime_error(
                    std::string("render_surface_regularizers_backward: '") +
                    fieldName + "' for camera '" + cameraName + "' must be HxW float32");
            }

            const int64_t height = static_cast<int64_t>(info.shape[0]);
            const int64_t width = static_cast<int64_t>(info.shape[1]);

            if (static_cast<std::uint32_t>(width) != expectedWidth ||
                static_cast<std::uint32_t>(height) != expectedHeight) {
                throw std::runtime_error(
                    std::string("render_surface_regularizers_backward: resolution mismatch for '") +
                    fieldName + "' camera '" + cameraName + "'");
            }

            const auto *src = static_cast<const float *>(info.ptr);
            const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);

            std::vector<float> hostImage(pixelCount);
            std::memcpy(hostImage.data(), src, pixelCount * sizeof(float));
            return hostImage;
        };

        auto packNormalAdjointToRGBA = [](const py::array &array,
                                          std::uint32_t expectedWidth,
                                          std::uint32_t expectedHeight,
                                          const std::string &cameraName,
                                          const char *fieldName) -> std::vector<float> {
            py::buffer_info info = array.request();

            if (info.ndim != 3 || info.itemsize != sizeof(float)) {
                throw std::runtime_error(
                    std::string("render_surface_regularizers_backward: '") +
                    fieldName + "' for camera '" + cameraName + "' must be HxWx3 or HxWx4 float32");
            }

            const int64_t height = static_cast<int64_t>(info.shape[0]);
            const int64_t width = static_cast<int64_t>(info.shape[1]);
            const int64_t channels = static_cast<int64_t>(info.shape[2]);

            if (channels != 3 && channels != 4) {
                throw std::runtime_error(
                    std::string("render_surface_regularizers_backward: '") +
                    fieldName + "' for camera '" + cameraName + "' must be HxWx3 or HxWx4");
            }

            if (static_cast<std::uint32_t>(width) != expectedWidth ||
                static_cast<std::uint32_t>(height) != expectedHeight) {
                throw std::runtime_error(
                    std::string("render_surface_regularizers_backward: resolution mismatch for '") +
                    fieldName + "' camera '" + cameraName + "'");
            }

            const float *src = static_cast<const float *>(info.ptr);
            std::vector<float> rgba(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u, 0.0f);

            for (int64_t y = 0; y < height; ++y) {
                for (int64_t x = 0; x < width; ++x) {
                    const size_t srcBase = static_cast<size_t>((y * width + x) * channels);
                    const size_t dstBase = static_cast<size_t>((y * width + x) * 4);

                    rgba[dstBase + 0] = src[srcBase + 0];
                    rgba[dstBase + 1] = src[srcBase + 1];
                    rgba[dstBase + 2] = src[srcBase + 2];
                    rgba[dstBase + 3] = 0.0f;
                }
            }

            return rgba;
        };

        for (std::size_t sensorIndex = 0; sensorIndex < sensorsForward.size(); ++sensorIndex) {
            const auto &sensor = sensorsForward[sensorIndex];
            const std::string cameraName(sensor.name, strnlen(sensor.name, sizeof(sensor.name)));

            if (!sensor.camera.useForAdjointPass) {
                continue;
            }

            if (!selectedCameraNames.contains(cameraName)) {
                continue;
            }

            selectedCameras.push_back(sensor);
            selectedDebugImages.push_back(debugImages[sensorIndex]);

            if (depthDistortionGradImagesDictionary.contains(py::str(cameraName))) {
                depthAdjointPerCamera.emplace(
                    cameraName,
                    packFloatImage(
                        depthDistortionGradImagesDictionary[py::str(cameraName)].cast<py::array>(),
                        sensor.width,
                        sensor.height,
                        cameraName,
                        "depth_distortion"));
            }

            if (intraSlabDepthGradImagesDictionary.contains(py::str(cameraName))) {
                intraSlabDepthAdjointPerCamera.emplace(
                    cameraName,
                    packFloatImage(
                        intraSlabDepthGradImagesDictionary[py::str(cameraName)].cast<py::array>(),
                        sensor.width,
                        sensor.height,
                        cameraName,
                        "intra_slab_depth"));
            }

            if (curvatureScaleGradImagesDictionary.contains(py::str(cameraName))) {
                curvatureScaleAdjointPerCamera.emplace(
                    cameraName,
                    packFloatImage(
                        curvatureScaleGradImagesDictionary[py::str(cameraName)].cast<py::array>(),
                        sensor.width,
                        sensor.height,
                        cameraName,
                        "curvature_scale"));
            }

            const bool hasVisibleNormal = visibleNormalGradImagesDictionary.contains(py::str(cameraName));
            const bool hasDepthNormal = normalFromDepthGradImagesDictionary.contains(py::str(cameraName));

            if (hasVisibleNormal != hasDepthNormal) {
                throw std::runtime_error(
                    "render_surface_regularizers_backward: visible_normal and normal_from_depth adjoints must either both "
                    "be present or both be absent for camera '" + cameraName + "'");
            }

            if (hasVisibleNormal) {
                visibleNormalAdjointPerCamera.emplace(
                    cameraName,
                    packNormalAdjointToRGBA(
                        visibleNormalGradImagesDictionary[py::str(cameraName)].cast<py::array>(),
                        sensor.width,
                        sensor.height,
                        cameraName,
                        "visible_normal"));

                normalFromDepthAdjointPerCamera.emplace(
                    cameraName,
                    packNormalAdjointToRGBA(
                        normalFromDepthGradImagesDictionary[py::str(cameraName)].cast<py::array>(),
                        sensor.width,
                        sensor.height,
                        cameraName,
                        "normal_from_depth"));
            }
        }

        if (selectedCameras.empty()) {
            return py::dict{};
        }

        py::gil_scoped_release release;

        for (auto &sensor: selectedCameras) {
            const std::string cameraName(sensor.name, strnlen(sensor.name, sizeof(sensor.name)));
            const size_t pixelCount = static_cast<size_t>(sensor.width) * static_cast<size_t>(sensor.height);

            syclQueue.fill(sensor.depthDistortionAdjointBuffer, 0.0f, pixelCount);
            syclQueue.fill(sensor.intraSlabDepthAdjointBuffer, 0.0f, pixelCount);
            syclQueue.fill(sensor.curvatureScaleAdjointBuffer, 0.0f, pixelCount);
            syclQueue.fill(sensor.visibleNormalAdjointBuffer, Pale::float4{0.0f}, pixelCount);
            syclQueue.fill(sensor.normalFromDepthAdjointBuffer, Pale::float4{0.0f}, pixelCount);
            syclQueue.fill(sensor.medianDepthAdjointBuffer, 0.0f, pixelCount);

            auto depthIt = depthAdjointPerCamera.find(cameraName);
            if (depthIt != depthAdjointPerCamera.end()) {
                Pale::uploadFloatImage(
                    syclQueue,
                    sensor.depthDistortionAdjointBuffer,
                    depthIt->second);
            }

            auto intraSlabIt = intraSlabDepthAdjointPerCamera.find(cameraName);
            if (intraSlabIt != intraSlabDepthAdjointPerCamera.end()) {
                Pale::uploadFloatImage(
                    syclQueue,
                    sensor.intraSlabDepthAdjointBuffer,
                    intraSlabIt->second);
            }

            auto curvatureScaleIt = curvatureScaleAdjointPerCamera.find(cameraName);
            if (curvatureScaleIt != curvatureScaleAdjointPerCamera.end()) {
                Pale::uploadFloatImage(
                    syclQueue,
                    sensor.curvatureScaleAdjointBuffer,
                    curvatureScaleIt->second);
            }

            auto visibleIt = visibleNormalAdjointPerCamera.find(cameraName);
            auto depthNormalIt = normalFromDepthAdjointPerCamera.find(cameraName);

            if (visibleIt != visibleNormalAdjointPerCamera.end() &&
                depthNormalIt != normalFromDepthAdjointPerCamera.end()) {
                syclQueue.memcpy(
                    sensor.visibleNormalAdjointBuffer,
                    visibleIt->second.data(),
                    pixelCount * 4u * sizeof(float));

                syclQueue.memcpy(
                    sensor.normalFromDepthAdjointBuffer,
                    depthNormalIt->second.data(),
                    pixelCount * 4u * sizeof(float));
            }
        }

        syclQueue.wait_and_throw();

        pathTracer->renderSurfaceRegularizersBackward(
            selectedCameras,
            depthDistortionGradients,
            normalConsistencyGradients,
            visibilityOpacityGradients,
            intraSlabDepthGradients,
            curvatureScaleGradients,
            selectedDebugImages.data());

        py::gil_scoped_acquire gilAcquire;

        if (!returnGradients) {
            return py::dict{};
        }

        py::dict result;
        result["depth_distortion"] = makeGradientDictionary(depthDistortionGradients);
        result["normal_consistency"] = makeGradientDictionary(normalConsistencyGradients);
        result["opacity_prior"] = makeGradientDictionary(visibilityOpacityGradients);
        result["intra_slab_depth"] = makeGradientDictionary(intraSlabDepthGradients);
        result["curvature_scale"] = makeGradientDictionary(curvatureScaleGradients);
        return result;
    }

    py::dict render_surface_regularizers_backward(
        const py::list &cameraNamesList,
        const py::dict &depthDistortionGradImagesDictionary,
        const py::dict &visibleNormalGradImagesDictionary,
        const py::dict &normalFromDepthGradImagesDictionary,
        const py::dict &intraSlabDepthGradImagesDictionary,
        const py::dict &curvatureScaleGradImagesDictionary) {
        return runSurfaceRegularizersBackward(
            cameraNamesList,
            depthDistortionGradImagesDictionary,
            visibleNormalGradImagesDictionary,
            normalFromDepthGradImagesDictionary,
            intraSlabDepthGradImagesDictionary,
            curvatureScaleGradImagesDictionary,
            true);
    }

    void render_surface_regularizers_backward_no_gradients(
        const py::list &cameraNamesList,
        const py::dict &depthDistortionGradImagesDictionary,
        const py::dict &visibleNormalGradImagesDictionary,
        const py::dict &normalFromDepthGradImagesDictionary,
        const py::dict &intraSlabDepthGradImagesDictionary,
        const py::dict &curvatureScaleGradImagesDictionary) {
        (void) runSurfaceRegularizersBackward(
            cameraNamesList,
            depthDistortionGradImagesDictionary,
            visibleNormalGradImagesDictionary,
            normalFromDepthGradImagesDictionary,
            intraSlabDepthGradImagesDictionary,
            curvatureScaleGradImagesDictionary,
            false);
    }

    bool sync_point_parameters_from_gpu() {
        return syncPointParametersFromGpu(true);
    }

    py::dict capture_device_adam_state() {
        py::dict state;
        state["point_count"] = static_cast<std::uint64_t>(0u);
        state["step"] = static_cast<std::uint64_t>(0u);

        if (!isDeviceTrainingStateAllocated()) {
            return state;
        }

        auto syclQueue = deviceSelector->getQueue();
        const std::size_t pointCount = deviceTrainingState.pointCount;

        std::vector<Pale::float3> positionM(pointCount);
        std::vector<Pale::float3> positionV(pointCount);
        std::vector<Pale::float3> rotationM(pointCount);
        std::vector<Pale::float3> rotationV(pointCount);
        std::vector<Pale::float2> scaleM(pointCount);
        std::vector<Pale::float2> scaleV(pointCount);
        std::vector<Pale::float3> albedoM(pointCount);
        std::vector<Pale::float3> albedoV(pointCount);
        std::vector<float> opacityM(pointCount);
        std::vector<float> opacityV(pointCount);
        std::vector<float> betaM(pointCount);
        std::vector<float> betaV(pointCount);

        {
            py::gil_scoped_release release;
            syclQueue.memcpy(positionM.data(), deviceTrainingState.positionM, pointCount * sizeof(Pale::float3));
            syclQueue.memcpy(positionV.data(), deviceTrainingState.positionV, pointCount * sizeof(Pale::float3));
            syclQueue.memcpy(rotationM.data(), deviceTrainingState.rotationM, pointCount * sizeof(Pale::float3));
            syclQueue.memcpy(rotationV.data(), deviceTrainingState.rotationV, pointCount * sizeof(Pale::float3));
            syclQueue.memcpy(scaleM.data(), deviceTrainingState.scaleM, pointCount * sizeof(Pale::float2));
            syclQueue.memcpy(scaleV.data(), deviceTrainingState.scaleV, pointCount * sizeof(Pale::float2));
            syclQueue.memcpy(albedoM.data(), deviceTrainingState.albedoM, pointCount * sizeof(Pale::float3));
            syclQueue.memcpy(albedoV.data(), deviceTrainingState.albedoV, pointCount * sizeof(Pale::float3));
            syclQueue.memcpy(opacityM.data(), deviceTrainingState.opacityM, pointCount * sizeof(float));
            syclQueue.memcpy(opacityV.data(), deviceTrainingState.opacityV, pointCount * sizeof(float));
            syclQueue.memcpy(betaM.data(), deviceTrainingState.betaM, pointCount * sizeof(float));
            syclQueue.memcpy(betaV.data(), deviceTrainingState.betaV, pointCount * sizeof(float));
            syclQueue.wait_and_throw();
        }

        state["point_count"] = static_cast<std::uint64_t>(pointCount);
        state["step"] = static_cast<std::uint64_t>(deviceTrainingState.step);
        state["position_m"] = makeDeviceAdamFloat3Array(std::move(positionM), pointCount);
        state["position_v"] = makeDeviceAdamFloat3Array(std::move(positionV), pointCount);
        state["rotation_m"] = makeDeviceAdamFloat3Array(std::move(rotationM), pointCount);
        state["rotation_v"] = makeDeviceAdamFloat3Array(std::move(rotationV), pointCount);
        state["scale_m"] = makeDeviceAdamFloat2Array(std::move(scaleM), pointCount);
        state["scale_v"] = makeDeviceAdamFloat2Array(std::move(scaleV), pointCount);
        state["albedo_m"] = makeDeviceAdamFloat3Array(std::move(albedoM), pointCount);
        state["albedo_v"] = makeDeviceAdamFloat3Array(std::move(albedoV), pointCount);
        state["opacity_m"] = makeDeviceAdamFloat1Array(std::move(opacityM), pointCount);
        state["opacity_v"] = makeDeviceAdamFloat1Array(std::move(opacityV), pointCount);
        state["beta_m"] = makeDeviceAdamFloat1Array(std::move(betaM), pointCount);
        state["beta_v"] = makeDeviceAdamFloat1Array(std::move(betaV), pointCount);
        return state;
    }

    void upload_device_adam_state(const py::dict &state) {
        auto syclQueue = deviceSelector->getQueue();
        const std::size_t pointCount =
                static_cast<std::size_t>(get_u64(state, "point_count", 0u));
        const std::uint32_t step =
                static_cast<std::uint32_t>(get_u64(state, "step", 0u));

        if (pointCount == 0u) {
            freeDeviceTrainingState(syclQueue);
            return;
        }

        if (!sceneGpu.points || sceneGpu.pointCount != pointCount) {
            throw std::runtime_error(
                "upload_device_adam_state: state point count does not match current device scene");
        }

        std::vector<Pale::float3> positionM =
                readDeviceAdamFloat3Array(state, "position_m", pointCount);
        std::vector<Pale::float3> positionV =
                readDeviceAdamFloat3Array(state, "position_v", pointCount);
        std::vector<Pale::float3> rotationM =
                readDeviceAdamFloat3Array(state, "rotation_m", pointCount);
        std::vector<Pale::float3> rotationV =
                readDeviceAdamFloat3Array(state, "rotation_v", pointCount);
        std::vector<Pale::float2> scaleM =
                readDeviceAdamFloat2Array(state, "scale_m", pointCount);
        std::vector<Pale::float2> scaleV =
                readDeviceAdamFloat2Array(state, "scale_v", pointCount);
        std::vector<Pale::float3> albedoM =
                readDeviceAdamFloat3Array(state, "albedo_m", pointCount);
        std::vector<Pale::float3> albedoV =
                readDeviceAdamFloat3Array(state, "albedo_v", pointCount);
        std::vector<float> opacityM =
                readDeviceAdamFloat1Array(state, "opacity_m", pointCount);
        std::vector<float> opacityV =
                readDeviceAdamFloat1Array(state, "opacity_v", pointCount);
        std::vector<float> betaM =
                readDeviceAdamFloat1Array(state, "beta_m", pointCount);
        std::vector<float> betaV =
                readDeviceAdamFloat1Array(state, "beta_v", pointCount);

        {
            py::gil_scoped_release release;
            freeDeviceTrainingState(syclQueue);
            ensureDeviceTrainingState(pointCount, syclQueue);
            syclQueue.memcpy(deviceTrainingState.positionM, positionM.data(), pointCount * sizeof(Pale::float3));
            syclQueue.memcpy(deviceTrainingState.positionV, positionV.data(), pointCount * sizeof(Pale::float3));
            syclQueue.memcpy(deviceTrainingState.rotationM, rotationM.data(), pointCount * sizeof(Pale::float3));
            syclQueue.memcpy(deviceTrainingState.rotationV, rotationV.data(), pointCount * sizeof(Pale::float3));
            syclQueue.memcpy(deviceTrainingState.scaleM, scaleM.data(), pointCount * sizeof(Pale::float2));
            syclQueue.memcpy(deviceTrainingState.scaleV, scaleV.data(), pointCount * sizeof(Pale::float2));
            syclQueue.memcpy(deviceTrainingState.albedoM, albedoM.data(), pointCount * sizeof(Pale::float3));
            syclQueue.memcpy(deviceTrainingState.albedoV, albedoV.data(), pointCount * sizeof(Pale::float3));
            syclQueue.memcpy(deviceTrainingState.opacityM, opacityM.data(), pointCount * sizeof(float));
            syclQueue.memcpy(deviceTrainingState.opacityV, opacityV.data(), pointCount * sizeof(float));
            syclQueue.memcpy(deviceTrainingState.betaM, betaM.data(), pointCount * sizeof(float));
            syclQueue.memcpy(deviceTrainingState.betaV, betaV.data(), pointCount * sizeof(float));
            syclQueue.wait_and_throw();
        }
        deviceTrainingState.step = step;
    }

    py::dict get_point_parameters() {
        syncPointParametersFromGpuIfDirty();
        if (!assetManager) throw std::runtime_error("get_point_parameters: assetManager is null");
        auto pointAssetSharedPtr = assetManager->get<Pale::PointAsset>(pointCloudAssetHandle);
        if (!pointAssetSharedPtr) throw std::runtime_error("get_point_parameters: failed to get PointAsset for dynamic point cloud");
        Pale::PointAsset &pointAsset = *pointAssetSharedPtr;
        if (pointAsset.points.empty()) throw std::runtime_error("get_point_parameters: PointAsset has no PointGeometry blocks");
        Pale::PointGeometry &pointGeometry = pointAsset.points.front();
        const std::size_t pointCount = pointGeometry.positions.size();
        if (pointGeometry.quat.size() != pointCount || pointGeometry.scales.size() != pointCount || pointGeometry.albedos.size() != pointCount || pointGeometry.opacities.size() != pointCount || pointGeometry.betas.size() != pointCount || pointGeometry.powers.size() != pointCount) {
            throw std::runtime_error("get_point_parameters: PointGeometry size mismatch");
        }
        std::vector<Pale::float3> positionHost(pointCount);
        std::vector<float> rotationHost(pointCount * 4u);
        std::vector<Pale::float2> scaleHost(pointCount);
        std::vector<Pale::float3> albedoHost(pointCount);
        std::vector<float> opacityHost(pointCount);
        std::vector<float> betaHost(pointCount);
        std::vector<float> shapeHost(pointCount);
        std::vector<float> powerHost(pointCount);
        std::vector<float> densificationOriginHost(pointCount, 0.0f);
        const bool hasPrimitiveAgeMetadata =
            pointGeometry.primitiveAges.size() == pointCount;
        std::vector<float> primitiveAgeHost;
        if (hasPrimitiveAgeMetadata) {
            primitiveAgeHost.resize(pointCount, 0.0f);
        }
        for (std::size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
            const glm::quat q = normalizeQuaternionOrIdentity(pointGeometry.quat[pointIndex]);
            positionHost[pointIndex] = pointGeometry.positions[pointIndex];
            rotationHost[pointIndex * 4u + 0u] = q.w;
            rotationHost[pointIndex * 4u + 1u] = q.x;
            rotationHost[pointIndex * 4u + 2u] = q.y;
            rotationHost[pointIndex * 4u + 3u] = q.z;
            scaleHost[pointIndex].x() = pointGeometry.scales[pointIndex].x;
            scaleHost[pointIndex].y() = pointGeometry.scales[pointIndex].y;
            albedoHost[pointIndex] = pointGeometry.albedos[pointIndex];
            opacityHost[pointIndex] = pointGeometry.opacities[pointIndex];
            betaHost[pointIndex] = pointGeometry.betas[pointIndex];
            shapeHost[pointIndex] = pointGeometry.shapes.size() == pointCount ? pointGeometry.shapes[pointIndex] : 0.0f;
            powerHost[pointIndex] = pointGeometry.powers[pointIndex];
            if (pointGeometry.densificationOrigins.size() == pointCount) {
                densificationOriginHost[pointIndex] = static_cast<float>(
                    pointGeometry.densificationOrigins[pointIndex]);
            }
            if (hasPrimitiveAgeMetadata) {
                primitiveAgeHost[pointIndex] = static_cast<float>(
                    pointGeometry.primitiveAges[pointIndex]);
            }
        }
        auto makeFloat3Array = [](std::vector<Pale::float3> &hostVector, std::size_t count) -> py::array {
            auto *owner = new std::vector<Pale::float3>(std::move(hostVector));
            std::vector<ssize_t> shape{static_cast<ssize_t>(count), 3};
            std::vector<ssize_t> strides{static_cast<ssize_t>(sizeof(Pale::float3)), static_cast<ssize_t>(sizeof(float))};
            return py::array(py::buffer_info(owner->data(), sizeof(float), py::format_descriptor<float>::format(), 2, shape, strides), py::capsule(owner, [](void *pointer) { delete static_cast<std::vector<Pale::float3> *>(pointer); }));
        };
        auto makeFloat4Array = [](std::vector<float> &hostVector, std::size_t count) -> py::array {
            auto *owner = new std::vector<float>(std::move(hostVector));
            std::vector<ssize_t> shape{static_cast<ssize_t>(count), 4};
            std::vector<ssize_t> strides{static_cast<ssize_t>(4 * sizeof(float)), static_cast<ssize_t>(sizeof(float))};
            return py::array(py::buffer_info(owner->data(), sizeof(float), py::format_descriptor<float>::format(), 2, shape, strides), py::capsule(owner, [](void *pointer) { delete static_cast<std::vector<float> *>(pointer); }));
        };
        auto makeFloat2Array = [](std::vector<Pale::float2> &hostVector, std::size_t count) -> py::array {
            auto *owner = new std::vector<Pale::float2>(std::move(hostVector));
            std::vector<ssize_t> shape{static_cast<ssize_t>(count), 2};
            std::vector<ssize_t> strides{static_cast<ssize_t>(sizeof(Pale::float2)), static_cast<ssize_t>(sizeof(float))};
            return py::array(py::buffer_info(owner->data(), sizeof(float), py::format_descriptor<float>::format(), 2, shape, strides), py::capsule(owner, [](void *pointer) { delete static_cast<std::vector<Pale::float2> *>(pointer); }));
        };
        auto makeFloat1Array = [](std::vector<float> &hostVector, std::size_t count) -> py::array {
            auto *owner = new std::vector<float>(std::move(hostVector));
            std::vector<ssize_t> shape{static_cast<ssize_t>(count)};
            std::vector<ssize_t> strides{static_cast<ssize_t>(sizeof(float))};
            return py::array(py::buffer_info(owner->data(), sizeof(float), py::format_descriptor<float>::format(), 1, shape, strides), py::capsule(owner, [](void *pointer) { delete static_cast<std::vector<float> *>(pointer); }));
        };
        py::dict parameterDictionary;
        parameterDictionary["position"] = makeFloat3Array(positionHost, pointCount);
        parameterDictionary["rotation"] = makeFloat4Array(rotationHost, pointCount);
        parameterDictionary["scale"] = makeFloat2Array(scaleHost, pointCount);
        parameterDictionary["albedo"] = makeFloat3Array(albedoHost, pointCount);
        parameterDictionary["opacity"] = makeFloat1Array(opacityHost, pointCount);
        parameterDictionary["beta"] = makeFloat1Array(betaHost, pointCount);
        parameterDictionary["shape"] = makeFloat1Array(shapeHost, pointCount);
        parameterDictionary["power"] = makeFloat1Array(powerHost, pointCount);
        parameterDictionary["densification_origin"] =
            makeFloat1Array(densificationOriginHost, pointCount);
        if (hasPrimitiveAgeMetadata) {
            parameterDictionary["primitive_age"] =
                makeFloat1Array(primitiveAgeHost, pointCount);
        }
        return parameterDictionary;
    }

    py::dict get_curvature_densification_stats() {
        const std::size_t pointCount = curvatureDensificationStats.numPoints;
        py::array_t<float> violationSum(pointCount);
        py::array_t<std::uint32_t> violationCount(pointCount);
        py::array_t<float> directionTensorUu(pointCount);
        py::array_t<float> directionTensorUv(pointCount);
        py::array_t<float> directionTensorVv(pointCount);

        if (pointCount > 0u) {
            if (!curvatureDensificationStats.violationSum ||
                !curvatureDensificationStats.violationCount ||
                !curvatureDensificationStats.directionTensorUu ||
                !curvatureDensificationStats.directionTensorUv ||
                !curvatureDensificationStats.directionTensorVv) {
                throw std::runtime_error(
                    "get_curvature_densification_stats: enabled buffers are incomplete");
            }

            auto syclQueue = deviceSelector->getQueue();
            float *violationSumHost = violationSum.mutable_data();
            std::uint32_t *violationCountHost = violationCount.mutable_data();
            float *directionTensorUuHost = directionTensorUu.mutable_data();
            float *directionTensorUvHost = directionTensorUv.mutable_data();
            float *directionTensorVvHost = directionTensorVv.mutable_data();
            {
                py::gil_scoped_release release;
                syclQueue.memcpy(
                    violationSumHost,
                    curvatureDensificationStats.violationSum,
                    pointCount * sizeof(float));
                syclQueue.memcpy(
                    violationCountHost,
                    curvatureDensificationStats.violationCount,
                    pointCount * sizeof(std::uint32_t));
                syclQueue.memcpy(
                    directionTensorUuHost,
                    curvatureDensificationStats.directionTensorUu,
                    pointCount * sizeof(float));
                syclQueue.memcpy(
                    directionTensorUvHost,
                    curvatureDensificationStats.directionTensorUv,
                    pointCount * sizeof(float));
                syclQueue.memcpy(
                    directionTensorVvHost,
                    curvatureDensificationStats.directionTensorVv,
                    pointCount * sizeof(float));
                syclQueue.wait_and_throw();
            }
        }

        py::dict result;
        result["violation_sum"] = std::move(violationSum);
        result["violation_count"] = std::move(violationCount);
        result["direction_tensor_uu"] = std::move(directionTensorUu);
        result["direction_tensor_uv"] = std::move(directionTensorUv);
        result["direction_tensor_vv"] = std::move(directionTensorVv);
        return result;
    }

    py::dict get_primal_activity_stats() {
        const std::size_t pointCount = primalActivityStats.numPoints;
        py::array_t<std::uint32_t> cameraSurfaceHitCount(pointCount);
        py::array_t<std::uint32_t> shadowOccluderHitCount(pointCount);

        if (pointCount > 0u) {
            if (!primalActivityStats.cameraSurfaceHitCount ||
                !primalActivityStats.shadowOccluderHitCount) {
                throw std::runtime_error(
                    "get_primal_activity_stats: buffers are incomplete");
            }
            auto syclQueue = deviceSelector->getQueue();
            {
                py::gil_scoped_release release;
                syclQueue.memcpy(
                    cameraSurfaceHitCount.mutable_data(),
                    primalActivityStats.cameraSurfaceHitCount,
                    pointCount * sizeof(std::uint32_t));
                syclQueue.memcpy(
                    shadowOccluderHitCount.mutable_data(),
                    primalActivityStats.shadowOccluderHitCount,
                    pointCount * sizeof(std::uint32_t));
                syclQueue.wait_and_throw();
            }
        }

        py::dict result;
        result["camera_surface_hit_count"] = std::move(cameraSurfaceHitCount);
        result["shadow_occluder_hit_count"] = std::move(shadowOccluderHitCount);
        return result;
    }


    void apply_point_optimization(const py::dict &parameterDictionary) {
        if (!parameterDictionary.contains("position")) return;
        devicePointParametersDirty = false;
        if (!parameterDictionary.contains("rotation")) throw std::runtime_error("apply_point_optimization: expected key 'rotation' with shape (N,4)");
        py::array positionArray = parameterDictionary["position"].cast<py::array>();
        py::buffer_info positionInfo = positionArray.request();
        if (positionInfo.ndim != 2 || positionInfo.shape[1] != 3) throw std::runtime_error("Expected 'position' to have shape (N,3)");
        const std::size_t pointCount = static_cast<std::size_t>(positionInfo.shape[0]);
        const std::size_t currentPointCount = buildProducts.points.size();
        if (pointCount == 0) return;
        if (pointCount != currentPointCount) {
            Pale::Log::PA_ERROR("apply_point_optimization: incoming point count {} does not match current buildProducts.points size {}. This function does not handle topology changes.", pointCount, currentPointCount);
            throw std::runtime_error("apply_point_optimization expects consistent point count; use densification API for adding/removing points.");
        }
        if (deviceTrainingState.pointCount != 0u && deviceTrainingState.pointCount != pointCount) {
            freeDeviceTrainingState(deviceSelector->getQueue());
        }
        auto requireArray = [&](const char *key) -> py::array {
            if (!parameterDictionary.contains(key)) throw std::runtime_error("apply_point_optimization: missing key: " + std::string(key));
            return parameterDictionary[key].cast<py::array>();
        };
        auto checkMatrix = [&](const py::buffer_info &info, const char *key, std::size_t dim) {
            if (info.ndim != 2 || info.shape[0] != static_cast<ssize_t>(pointCount) || info.shape[1] != static_cast<ssize_t>(dim)) throw std::runtime_error(std::string("Expected '") + key + "' to have shape (N," + std::to_string(dim) + ")");
            if (info.itemsize != sizeof(float)) throw std::runtime_error(std::string("Expected '") + key + "' to be float32");
        };
        auto checkVector = [&](const py::buffer_info &info, const char *key) {
            if (info.ndim != 1 || info.shape[0] != static_cast<ssize_t>(pointCount)) throw std::runtime_error(std::string("Expected '") + key + "' to have shape (N,)");
            if (info.itemsize != sizeof(float)) throw std::runtime_error(std::string("Expected '") + key + "' to be float32");
        };
        py::array rotationArray = requireArray("rotation");
        py::array scaleArray = requireArray("scale");
        py::array albedoArray = requireArray("albedo");
        py::array opacityArray = requireArray("opacity");
        py::array betaArray = requireArray("beta");
        py::array powerArray = requireArray("power");
        py::buffer_info rotationInfo = rotationArray.request();
        py::buffer_info scaleInfo = scaleArray.request();
        py::buffer_info albedoInfo = albedoArray.request();
        py::buffer_info opacityInfo = opacityArray.request();
        py::buffer_info betaInfo = betaArray.request();
        py::buffer_info powerInfo = powerArray.request();
        checkMatrix(positionInfo, "position", 3);
        checkMatrix(rotationInfo, "rotation", 4);
        checkMatrix(scaleInfo, "scale", 2);
        checkMatrix(albedoInfo, "albedo", 3);
        checkVector(opacityInfo, "opacity");
        checkVector(betaInfo, "beta");
        checkVector(powerInfo, "power");
        const float *positionData = static_cast<const float *>(positionInfo.ptr);
        const float *rotationData = static_cast<const float *>(rotationInfo.ptr);
        const float *scaleData = static_cast<const float *>(scaleInfo.ptr);
        const float *albedoData = static_cast<const float *>(albedoInfo.ptr);
        const float *opacityData = static_cast<const float *>(opacityInfo.ptr);
        const float *betaData = static_cast<const float *>(betaInfo.ptr);
        const float *powerData = static_cast<const float *>(powerInfo.ptr);
        for (std::size_t i = 0; i < pointCount; ++i) {
            const std::size_t i3 = i * 3u;
            const std::size_t i4 = i * 4u;
            const std::size_t i2 = i * 2u;
            Pale::Point &point = buildProducts.points[i];
            point.position = glm::vec3(positionData[i3 + 0u], positionData[i3 + 1u], positionData[i3 + 2u]);
            const glm::quat q = normalizeQuaternionOrIdentity(glm::quat(rotationData[i4 + 0u], rotationData[i4 + 1u], rotationData[i4 + 2u], rotationData[i4 + 3u]));
            frameFromQuaternion(q, point.tanU, point.tanV);
            point.scale.x() = scaleData[i2 + 0u];
            point.scale.y() = scaleData[i2 + 1u];
            point.albedo = glm::vec3(albedoData[i3 + 0u], albedoData[i3 + 1u], albedoData[i3 + 2u]);
            point.opacity = opacityData[i];
            point.beta = betaData[i];
            point.flux = powerData[i];
        }
        if (!assetManager) {
            Pale::Log::PA_WARN("apply_point_optimization: assetManager is null, skipping asset point cloud update.");
        } else {
            auto pointAssetSharedPtr = assetManager->get<Pale::PointAsset>(pointCloudAssetHandle);
            if (!pointAssetSharedPtr) {
                Pale::Log::PA_ERROR("apply_point_optimization: failed to get PointAsset for handle {}", std::string(pointCloudAssetHandle));
            } else if (!pointAssetSharedPtr->points.empty()) {
                Pale::PointGeometry &pointGeometry = pointAssetSharedPtr->points.front();
                if (pointGeometry.positions.size() != pointCount || pointGeometry.quat.size() != pointCount || pointGeometry.scales.size() != pointCount || pointGeometry.albedos.size() != pointCount || pointGeometry.betas.size() != pointCount || pointGeometry.opacities.size() != pointCount || pointGeometry.powers.size() != pointCount) {
                    throw std::runtime_error("apply_point_optimization: PointGeometry size mismatch");
                }
                for (std::size_t i = 0; i < pointCount; ++i) {
                    const std::size_t i3 = i * 3u;
                    const std::size_t i4 = i * 4u;
                    const std::size_t i2 = i * 2u;
                    pointGeometry.positions[i] = glm::vec3(positionData[i3 + 0u], positionData[i3 + 1u], positionData[i3 + 2u]);
                    pointGeometry.quat[i] = normalizeQuaternionOrIdentity(glm::quat(rotationData[i4 + 0u], rotationData[i4 + 1u], rotationData[i4 + 2u], rotationData[i4 + 3u]));
                    pointGeometry.scales[i] = glm::vec2(scaleData[i2 + 0u], scaleData[i2 + 1u]);
                    pointGeometry.albedos[i] = glm::vec3(albedoData[i3 + 0u], albedoData[i3 + 1u], albedoData[i3 + 2u]);
                    pointGeometry.opacities[i] = opacityData[i];
                    pointGeometry.betas[i] = betaData[i];
                    pointGeometry.powers[i] = powerData[i];
                }
            }
        }
        Pale::SceneUpload::upload(buildProducts, sceneGpu, deviceSelector->getQueue());
        pathTracer->setScene(sceneGpu, buildProducts);
        devicePointParametersDirty = false;
    }

    void rebuild_bvh() {
        syncPointParametersFromGpuIfDirty();
        const std::size_t previousDeviceOptimizerPointCount = deviceTrainingState.pointCount;
        Pale::AssetAccessFromManager assetAccessor(*assetManager);
        buildProducts = Pale::SceneBuild::build(scene, assetAccessor, Pale::SceneBuild::BuildOptions());
        Pale::SceneUpload::uploadOrReallocate(buildProducts, sceneGpu, deviceSelector->getQueue());
        if (previousDeviceOptimizerPointCount != 0u &&
            previousDeviceOptimizerPointCount != buildProducts.points.size()) {
            freeDeviceTrainingState(deviceSelector->getQueue());
        }
        Pale::freeGradientsForScene(deviceSelector->getQueue(), gradients);
        Pale::freeGradientsForScene(deviceSelector->getQueue(), depthDistortionGradients);
        Pale::freeGradientsForScene(deviceSelector->getQueue(), normalConsistencyGradients);
        Pale::freeGradientsForScene(deviceSelector->getQueue(), visibilityOpacityGradients);
        Pale::freeGradientsForScene(deviceSelector->getQueue(), intraSlabDepthGradients);
        Pale::freeGradientsForScene(deviceSelector->getQueue(), curvatureScaleGradients);
        Pale::freeCurvatureDensificationStats(
            deviceSelector->getQueue(), curvatureDensificationStats);
        Pale::freePrimalActivityStats(
            deviceSelector->getQueue(), primalActivityStats);
        Pale::freeDebugImagesForScene(deviceSelector->getQueue(), debugImages.data(), debugImages.size());
        debugImages.clear();
        debugImages.resize(sensorsForward.size());
        gradients = Pale::makeGradientsForScene(deviceSelector->getQueue(), buildProducts, debugImages.data());
        depthDistortionGradients = Pale::makeGradientsForScene(deviceSelector->getQueue(), buildProducts, nullptr);
        normalConsistencyGradients = Pale::makeGradientsForScene(deviceSelector->getQueue(), buildProducts, nullptr);
        visibilityOpacityGradients = Pale::makeGradientsForScene(deviceSelector->getQueue(), buildProducts, nullptr);
        intraSlabDepthGradients = Pale::makeGradientsForScene(deviceSelector->getQueue(), buildProducts, nullptr);
        curvatureScaleGradients = Pale::makeGradientsForScene(deviceSelector->getQueue(), buildProducts, nullptr);
        if (curvatureDensificationEnabled) {
            curvatureDensificationStats = Pale::makeCurvatureDensificationStatsForScene(
                deviceSelector->getQueue(), buildProducts);
        }
        if (primalActivityTrackingEnabled) {
            primalActivityStats = Pale::makePrimalActivityStatsForScene(
                deviceSelector->getQueue(), buildProducts);
        }
        pathTracer->setScene(sceneGpu, buildProducts);
        pathTracer->setCurvatureDensificationStats(
            curvatureDensificationEnabled ? &curvatureDensificationStats : nullptr);
        pathTracer->setPrimalActivityStats(
            primalActivityTrackingEnabled ? &primalActivityStats : nullptr);
        devicePointParametersDirty = false;
    }

    void remove_points(const py::dict &parameterDictionary) {
        syncPointParametersFromGpuIfDirty();
        // -----------------------------------------------------------------
        // 0) Check required input
        // -----------------------------------------------------------------
        if (!parameterDictionary.contains("indices")) {
            throw std::runtime_error("remove_points: expected key 'indices' (1D int32 / int64 array)");
        }

        py::array indicesArray = parameterDictionary["indices"].cast<py::array>();
        py::buffer_info indicesInfo = indicesArray.request();

        if (indicesInfo.ndim != 1) {
            throw std::runtime_error("remove_points: 'indices' must be a 1D array");
        }

        const std::size_t removeCount = static_cast<std::size_t>(indicesInfo.shape[0]);
        if (removeCount == 0) {
            Pale::Log::PA_INFO("remove_points: no indices provided, nothing to remove.");
            return;
        }

        const void *indicesVoidPointer = indicesInfo.ptr;
        const bool indicesAreInt64 = (indicesInfo.itemsize == sizeof(std::int64_t));
        const bool indicesAreInt32 = (indicesInfo.itemsize == sizeof(std::int32_t));
        if (!indicesAreInt32 && !indicesAreInt64) {
            throw std::runtime_error("remove_points: 'indices' must have dtype int32 or int64");
        }

        // -----------------------------------------------------------------
        // 1) Get point cloud asset
        // -----------------------------------------------------------------
        if (!assetManager) {
            throw std::runtime_error("remove_points: assetManager is null");
        }

        auto pointAssetSharedPtr = assetManager->get<Pale::PointAsset>(pointCloudAssetHandle);
        if (!pointAssetSharedPtr) {
            throw std::runtime_error("remove_points: failed to get PointAsset for dynamic point cloud");
        }

        Pale::PointAsset &pointAsset = *pointAssetSharedPtr;
        if (pointAsset.points.empty()) {
            throw std::runtime_error("remove_points: PointAsset has no PointGeometry blocks");
        }

        Pale::PointGeometry &pointGeometry = pointAsset.points.front();
        const std::size_t currentPointCount = pointGeometry.positions.size();

        if (currentPointCount == 0) {
            Pale::Log::PA_INFO("remove_points: current point cloud is empty, nothing to remove.");
            return;
        }

        // -----------------------------------------------------------------
        // 2) Build keep-mask from indices to remove
        // -----------------------------------------------------------------
        std::vector<char> keepMask(currentPointCount, 1);

        auto markIndexForRemoval = [&](std::size_t removalIndex) {
            if (removalIndex >= currentPointCount) {
                throw std::out_of_range("remove_points: index out of range");
            }
            keepMask[removalIndex] = 0;
        };

        if (indicesAreInt64) {
            const auto *indexData = static_cast<const std::int64_t *>(indicesVoidPointer);
            for (std::size_t removeIndex = 0; removeIndex < removeCount; ++removeIndex) {
                const std::int64_t value = indexData[removeIndex];
                if (value < 0) {
                    throw std::out_of_range("remove_points: negative index is not allowed");
                }
                markIndexForRemoval(static_cast<std::size_t>(value));
            }
        } else {
            const auto *indexData = static_cast<const std::int32_t *>(indicesVoidPointer);
            for (std::size_t removeIndex = 0; removeIndex < removeCount; ++removeIndex) {
                const std::int32_t value = indexData[removeIndex];
                if (value < 0) {
                    throw std::out_of_range("remove_points: negative index is not allowed");
                }
                markIndexForRemoval(static_cast<std::size_t>(value));
            }
        }

        std::size_t newPointCount = 0;
        for (char keepFlag: keepMask) {
            if (keepFlag) {
                ++newPointCount;
            }
        }

        if (newPointCount == 0) {
            Pale::Log::PA_WARN(
                "remove_points: all points would be removed ({} total). "
                "Proceeding, but make sure your pipeline handles the empty case.",
                currentPointCount
            );
        }

        // -----------------------------------------------------------------
        // 3) Filter all attribute arrays in PointGeometry
        // -----------------------------------------------------------------
        auto filterVectorInPlace = [&](auto &vectorAttribute) {
            using AttributeType = typename std::decay_t<decltype(vectorAttribute)>::value_type;
            std::vector<AttributeType> filteredVector;
            filteredVector.reserve(newPointCount);

            for (std::size_t pointIndex = 0; pointIndex < currentPointCount; ++pointIndex) {
                if (keepMask[pointIndex]) {
                    filteredVector.push_back(vectorAttribute[pointIndex]);
                }
            }

            vectorAttribute.swap(filteredVector);
        };

        filterVectorInPlace(pointGeometry.positions);
        filterVectorInPlace(pointGeometry.quat);
        filterVectorInPlace(pointGeometry.scales);
        filterVectorInPlace(pointGeometry.albedos);
        filterVectorInPlace(pointGeometry.opacities);
        filterVectorInPlace(pointGeometry.shapes);
        filterVectorInPlace(pointGeometry.betas);
        filterVectorInPlace(pointGeometry.powers);
        if (pointGeometry.densificationOrigins.size() == currentPointCount) {
            filterVectorInPlace(pointGeometry.densificationOrigins);
        }
        if (pointGeometry.primitiveAges.size() == currentPointCount) {
            filterVectorInPlace(pointGeometry.primitiveAges);
        }

        Pale::Log::PA_INFO(
            "remove_points: removed {} points, new point count = {}",
            currentPointCount - newPointCount,
            newPointCount
        );

        // -----------------------------------------------------------------
        // 4) Rebuild BVH and GPU buffers from updated asset (renderer is ground truth)
        // -----------------------------------------------------------------
        rebuild_bvh();
    }

    void add_new_points(const py::dict &parameterDictionary) {
        syncPointParametersFromGpuIfDirty();
        auto pointAssetSharedPtr = assetManager->get<Pale::PointAsset>(pointCloudAssetHandle);
        if (!pointAssetSharedPtr) throw std::runtime_error("add_new_points: failed to get PointAsset for dynamic point cloud");
        Pale::PointAsset &pointAsset = *pointAssetSharedPtr;
        if (pointAsset.points.empty()) throw std::runtime_error("add_new_points: PointAsset has no PointGeometry blocks");
        Pale::PointGeometry &pointGeometry = pointAsset.points.front();
        if (!parameterDictionary.contains("new")) { Pale::Log::PA_INFO("add_new_points: no 'new' block provided, nothing to append."); return; }
        py::dict newDict = parameterDictionary["new"].cast<py::dict>();
        auto getFloatArray = [&](const char *key) -> py::array_t<float, py::array::c_style | py::array::forcecast> {
            if (!newDict.contains(key)) throw std::runtime_error(std::string("add_new_points: missing key 'new.") + key + "'");
            return newDict[key].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        };
        auto getOptionalFloatArray = [&](const char *key) -> std::optional<py::array_t<float, py::array::c_style | py::array::forcecast>> {
            if (!newDict.contains(key)) return std::nullopt;
            return newDict[key].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        };
        py::array_t<float, py::array::c_style | py::array::forcecast> positionArray = getFloatArray("position");
        py::array_t<float, py::array::c_style | py::array::forcecast> rotationArray = getFloatArray("rotation");
        py::array_t<float, py::array::c_style | py::array::forcecast> scaleArray = getFloatArray("scale");
        py::array_t<float, py::array::c_style | py::array::forcecast> albedoArray = getFloatArray("albedo");
        py::array_t<float, py::array::c_style | py::array::forcecast> opacityArray = getFloatArray("opacity");
        py::array_t<float, py::array::c_style | py::array::forcecast> betaArray = getFloatArray("beta");
        std::optional<py::array_t<float, py::array::c_style | py::array::forcecast>> powerArrayOpt = getOptionalFloatArray("power");
        py::buffer_info positionInfo = positionArray.request();
        py::buffer_info rotationInfo = rotationArray.request();
        py::buffer_info scaleInfo = scaleArray.request();
        py::buffer_info albedoInfo = albedoArray.request();
        py::buffer_info opacityInfo = opacityArray.request();
        py::buffer_info betaInfo = betaArray.request();
        auto checkMatrixShape = [](const py::buffer_info &bufferInfo, std::size_t expectedCount, std::size_t expectedDim, const char *name) {
            if (bufferInfo.ndim != 2 || bufferInfo.shape[0] != static_cast<ssize_t>(expectedCount) || bufferInfo.shape[1] != static_cast<ssize_t>(expectedDim)) throw std::runtime_error(std::string("add_new_points: 'new.") + name + "' must have shape (N," + std::to_string(expectedDim) + ")");
            if (bufferInfo.itemsize != sizeof(float)) throw std::runtime_error(std::string("add_new_points: 'new.") + name + "' must be float32");
        };
        auto checkVectorShape = [](const py::buffer_info &bufferInfo, std::size_t expectedCount, const char *name) {
            const bool validShape = (bufferInfo.ndim == 1 && bufferInfo.shape[0] == static_cast<ssize_t>(expectedCount)) || (bufferInfo.ndim == 2 && bufferInfo.shape[0] == static_cast<ssize_t>(expectedCount) && bufferInfo.shape[1] == 1);
            if (!validShape) throw std::runtime_error(std::string("add_new_points: 'new.") + name + "' must have shape (N,) or (N,1)");
            if (bufferInfo.itemsize != sizeof(float)) throw std::runtime_error(std::string("add_new_points: 'new.") + name + "' must be float32");
        };
        if (positionInfo.ndim != 2 || positionInfo.shape[1] != 3) throw std::runtime_error("add_new_points: 'new.position' must have shape (N,3)");
        if (positionInfo.itemsize != sizeof(float)) throw std::runtime_error("add_new_points: 'new.position' must be float32");
        const std::size_t newPointCount = static_cast<std::size_t>(positionInfo.shape[0]);
        if (newPointCount == 0) { Pale::Log::PA_INFO("add_new_points: 'new' block has zero points, nothing to append."); return; }
        checkMatrixShape(rotationInfo, newPointCount, 4, "rotation");
        checkMatrixShape(scaleInfo, newPointCount, 2, "scale");
        checkMatrixShape(albedoInfo, newPointCount, 3, "albedo");
        checkVectorShape(opacityInfo, newPointCount, "opacity");
        checkVectorShape(betaInfo, newPointCount, "beta");
        py::buffer_info powerInfo{};
        bool hasPower = false;
        if (powerArrayOpt.has_value()) { powerInfo = powerArrayOpt.value().request(); checkVectorShape(powerInfo, newPointCount, "power"); hasPower = true; }
        const float *positionData = static_cast<const float *>(positionInfo.ptr);
        const float *rotationData = static_cast<const float *>(rotationInfo.ptr);
        const float *scaleData = static_cast<const float *>(scaleInfo.ptr);
        const float *albedoData = static_cast<const float *>(albedoInfo.ptr);
        const float *opacityData = static_cast<const float *>(opacityInfo.ptr);
        const float *betaData = static_cast<const float *>(betaInfo.ptr);
        const float *powerData = hasPower ? static_cast<const float *>(powerInfo.ptr) : nullptr;
        const std::size_t currentPointCount = pointGeometry.positions.size();
        const std::size_t newTotalPointCount = currentPointCount + newPointCount;
        auto reserveAttribute = [newTotalPointCount](auto &vectorAttribute) { vectorAttribute.reserve(newTotalPointCount); };
        reserveAttribute(pointGeometry.positions);
        reserveAttribute(pointGeometry.quat);
        reserveAttribute(pointGeometry.scales);
        reserveAttribute(pointGeometry.albedos);
        reserveAttribute(pointGeometry.opacities);
        reserveAttribute(pointGeometry.shapes);
        reserveAttribute(pointGeometry.betas);
        reserveAttribute(pointGeometry.powers);
        if (!pointGeometry.densificationOrigins.empty()) {
            reserveAttribute(pointGeometry.densificationOrigins);
        }
        if (!pointGeometry.primitiveAges.empty()) {
            reserveAttribute(pointGeometry.primitiveAges);
        }
        for (std::size_t pointIndex = 0; pointIndex < newPointCount; ++pointIndex) {
            const std::size_t i3 = pointIndex * 3u;
            const std::size_t i4 = pointIndex * 4u;
            const std::size_t i2 = pointIndex * 2u;
            pointGeometry.positions.push_back(glm::vec3(positionData[i3 + 0u], positionData[i3 + 1u], positionData[i3 + 2u]));
            pointGeometry.quat.push_back(normalizeQuaternionOrIdentity(glm::quat(rotationData[i4 + 0u], rotationData[i4 + 1u], rotationData[i4 + 2u], rotationData[i4 + 3u])));
            pointGeometry.scales.push_back(glm::vec2(scaleData[i2 + 0u], scaleData[i2 + 1u]));
            pointGeometry.albedos.push_back(glm::vec3(albedoData[i3 + 0u], albedoData[i3 + 1u], albedoData[i3 + 2u]));
            pointGeometry.opacities.push_back(opacityData[pointIndex]);
            pointGeometry.betas.push_back(betaData[pointIndex]);
            pointGeometry.powers.push_back(hasPower ? powerData[pointIndex] : 0.0f);
            pointGeometry.shapes.push_back(0.0f);
            if (!pointGeometry.densificationOrigins.empty()) {
                pointGeometry.densificationOrigins.push_back(0u);
            }
            if (!pointGeometry.primitiveAges.empty()) {
                pointGeometry.primitiveAges.push_back(0u);
            }
        }
        Pale::Log::PA_INFO("add_new_points: final point count in geometry = {} (added {} new points)", pointGeometry.positions.size(), newPointCount);
        rebuild_bvh();
    }

    std::vector<std::string> getCameraNames() {
        std::vector<std::string> names;
        for (const auto &camera: buildProducts.cameras()) {
            names.emplace_back(camera.name);
        }
        return names;
    }

    std::vector<std::string> getTrainingCameras() {
        std::vector<std::string> names;
        for (const auto &camera: buildProducts.cameras()) {
            if (camera.useForAdjointPass)
                names.emplace_back(camera.name);
        }
        return names;
    }

    void set_point_opacity(float newOpacity, int index) {
        syncPointParametersFromGpuIfDirty();
        if (!assetManager) {
            throw std::runtime_error("set_gaussian_opacity: assetManager is null");
        }

        auto pointAssetSharedPtr = assetManager->get<Pale::PointAsset>(pointCloudAssetHandle);
        if (!pointAssetSharedPtr) {
            throw std::runtime_error("set_gaussian_opacity: failed to get PointAsset for dynamic point cloud");
        }

        Pale::PointAsset &pointAsset = *pointAssetSharedPtr;
        if (pointAsset.points.empty()) {
            throw std::runtime_error("set_gaussian_opacity: PointAsset has no PointGeometry blocks");
        }

        Pale::PointGeometry &pointGeometry = pointAsset.points.front();

        const int pointCount = static_cast<int>(pointGeometry.opacities.size());
        if (index < 0 || index >= pointCount) {
            throw std::runtime_error("set_gaussian_opacity: index out of range");
        }

        pointGeometry.opacities[index] = newOpacity;

        // Only needed if BVH / acceleration depends on opacity (often it doesn't).
        // If you can skip it, do so for performance.
        rebuild_bvh();
        //Pale::Log::PA_ERROR("Opacity: {}/{}", pointGeometry.opacities[index], buildProducts.points[index].opacity);
    }

    void set_point_translation(float newPosition, float axis, int index) {
        syncPointParametersFromGpuIfDirty();
        if (!assetManager) {
            throw std::runtime_error("set_gaussian_opacity: assetManager is null");
        }

        auto pointAssetSharedPtr = assetManager->get<Pale::PointAsset>(pointCloudAssetHandle);
        if (!pointAssetSharedPtr) {
            throw std::runtime_error("set_gaussian_opacity: failed to get PointAsset for dynamic point cloud");
        }

        Pale::PointAsset &pointAsset = *pointAssetSharedPtr;
        if (pointAsset.points.empty()) {
            throw std::runtime_error("set_gaussian_opacity: PointAsset has no PointGeometry blocks");
        }

        Pale::PointGeometry &pointGeometry = pointAsset.points.front();

        const int pointCount = static_cast<int>(pointGeometry.positions.size());
        if (index < 0 || index >= pointCount) {
            throw std::runtime_error("set_gaussian_opacity: index out of range");
        }

        pointGeometry.positions[index][axis] = newPosition;

        // Only needed if BVH / acceleration depends on opacity (often it doesn't).
        // If you can skip it, do so for performance.
        rebuild_bvh();
        //Pale::Log::PA_ERROR("Opacity: {}/{}", pointGeometry.opacities[index], buildProducts.points[index].opacity);
    }

    void set_point_albedo(float newIntensity, float axis, int index) {
        syncPointParametersFromGpuIfDirty();
        if (!assetManager) {
            throw std::runtime_error("set_point_albedo: assetManager is null");
        }

        auto pointAssetSharedPtr = assetManager->get<Pale::PointAsset>(pointCloudAssetHandle);
        if (!pointAssetSharedPtr) {
            throw std::runtime_error("set_point_albedo: failed to get PointAsset for dynamic point cloud");
        }

        Pale::PointAsset &pointAsset = *pointAssetSharedPtr;
        if (pointAsset.points.empty()) {
            throw std::runtime_error("set_point_albedo: PointAsset has no PointGeometry blocks");
        }

        Pale::PointGeometry &pointGeometry = pointAsset.points.front();

        const int pointCount = static_cast<int>(pointGeometry.positions.size());
        if (index < 0 || index >= pointCount) {
            throw std::runtime_error("set_point_albedo: index out of range");
        }

        pointGeometry.albedos[index][axis] = newIntensity;
        rebuild_bvh();
    }

    static inline void orthonormalizeFrame(glm::vec3 &tanU, glm::vec3 &tanV) {
        tanU = normalize(tanU);

        tanV = tanV - tanU * dot(tanV, tanU);
        tanV = normalize(tanV);

        // Optional: keep a right-handed frame if needed
        const glm::vec3 n = normalize(cross(tanU, tanV));
        tanV = normalize(cross(n, tanU));
    }

    static inline glm::vec3 rotateAxisAngle(
        const glm::vec3 &v,
        const glm::vec3 &axisUnit,
        float angleRadians) {
        const float c = std::cos(angleRadians);
        const float s = std::sin(angleRadians);

        return v * c
               + cross(axisUnit, v) * s
               + axisUnit * (dot(axisUnit, v) * (1.0f - c));
    }

    void set_point_rotation_degrees(float angleDegrees, int axisIndex, int index) {
        syncPointParametersFromGpuIfDirty();
        if (!assetManager) throw std::runtime_error("set_point_rotation_degrees: assetManager is null");
        auto pointAssetSharedPtr = assetManager->get<Pale::PointAsset>(pointCloudAssetHandle);
        if (!pointAssetSharedPtr) throw std::runtime_error("set_point_rotation_degrees: failed to get PointAsset");
        Pale::PointAsset &pointAsset = *pointAssetSharedPtr;
        if (pointAsset.points.empty()) throw std::runtime_error("set_point_rotation_degrees: no PointGeometry blocks");
        Pale::PointGeometry &pointGeometry = pointAsset.points.front();
        const int pointCount = static_cast<int>(pointGeometry.positions.size());
        if (index < 0 || index >= pointCount) throw std::runtime_error("set_point_rotation_degrees: index out of range");
        glm::vec3 axis(0.0f, 0.0f, 0.0f);
        switch (axisIndex) {
            case 0: axis = glm::vec3(1.0f, 0.0f, 0.0f); break;
            case 1: axis = glm::vec3(0.0f, 1.0f, 0.0f); break;
            case 2: axis = glm::vec3(0.0f, 0.0f, 1.0f); break;
            default: throw std::runtime_error("set_point_rotation_degrees: invalid axisIndex");
        }
        pointGeometry.quat[index] = normalizeQuaternionOrIdentity(glm::angleAxis(glm::radians(angleDegrees), axis));
        rebuild_bvh();
    }

    std::vector<Pale::SensorGPU> selectSensorsByName(const std::optional<std::string>& cameraName) const {
        if (!cameraName.has_value() || cameraName->empty()) {
            return sensorsForward;
        }

        std::vector<Pale::SensorGPU> selectedSensors;

        for (const Pale::SensorGPU& sensor : sensorsForward) {
            const std::string sensorName(sensor.name, strnlen(sensor.name, sizeof(sensor.name)));

            if (sensorName == cameraName.value()) {
                selectedSensors.push_back(sensor);
                break;
            }
        }

        if (selectedSensors.empty()) {
            throw std::runtime_error("Camera not found: " + cameraName.value());
        }

        return selectedSensors;
    }

    void set_point_scale(float newScale, float axis, int index) {
        syncPointParametersFromGpuIfDirty();
        if (!assetManager) {
            throw std::runtime_error("set_gaussian_opacity: assetManager is null");
        }

        auto pointAssetSharedPtr = assetManager->get<Pale::PointAsset>(pointCloudAssetHandle);
        if (!pointAssetSharedPtr) {
            throw std::runtime_error("set_gaussian_opacity: failed to get PointAsset for dynamic point cloud");
        }

        Pale::PointAsset &pointAsset = *pointAssetSharedPtr;
        if (pointAsset.points.empty()) {
            throw std::runtime_error("set_gaussian_opacity: PointAsset has no PointGeometry blocks");
        }

        Pale::PointGeometry &pointGeometry = pointAsset.points.front();

        const int pointCount = static_cast<int>(pointGeometry.positions.size());
        if (index < 0 || index >= pointCount) {
            throw std::runtime_error("set_gaussian_opacity: index out of range");
        }

        pointGeometry.scales[index][axis] = newScale;

        // Only needed if BVH / acceleration depends on opacity (often it doesn't).
        // If you can skip it, do so for performance.
        rebuild_bvh();
        //Pale::Log::PA_ERROR("Opacity: {}/{}", pointGeometry.opacities[index], buildProducts.points[index].opacity);
    }


    void set_point_beta(float newBeta, int index) {
        syncPointParametersFromGpuIfDirty();
        if (!assetManager) {
            throw std::runtime_error("set_point_beta: assetManager is null");
        }

        auto pointAssetSharedPtr = assetManager->get<Pale::PointAsset>(pointCloudAssetHandle);
        if (!pointAssetSharedPtr) {
            throw std::runtime_error("set_point_beta: failed to get PointAsset for dynamic point cloud");
        }

        Pale::PointAsset &pointAsset = *pointAssetSharedPtr;
        if (pointAsset.points.empty()) {
            throw std::runtime_error("set_point_beta: PointAsset has no PointGeometry blocks");
        }

        Pale::PointGeometry &pointGeometry = pointAsset.points.front();

        const int pointCount = static_cast<int>(pointGeometry.opacities.size());
        if (index < 0 || index >= pointCount) {
            throw std::runtime_error("set_point_beta: index out of range");
        }

        pointGeometry.betas[index] = newBeta;

        // Only needed if BVH / acceleration depends on opacity (often it doesn't).
        // If you can skip it, do so for performance.
        rebuild_bvh();
        //Pale::Log::PA_ERROR("Beta: {}/{}", pointGeometry.betas[index], buildProducts.points[index].beta);
    }


    void set_point_properties(py::tuple translation3, py::tuple rotationQuat4, py::tuple scale3, py::tuple albedo3, float opacity, float beta, int index = -1) {
        syncPointParametersFromGpuIfDirty();
        if (translation3.size() != 3 || rotationQuat4.size() != 4 || scale3.size() != 3 || albedo3.size() != 3) throw std::runtime_error("Expected translation(3), rotation_quat_wxyz(4), scale(3), albedo(3)");
        const glm::vec3 newTranslation{py::cast<float>(translation3[0]), py::cast<float>(translation3[1]), py::cast<float>(translation3[2])};
        const glm::quat rotationDelta = normalizeQuaternionOrIdentity(glm::quat(py::cast<float>(rotationQuat4[0]), py::cast<float>(rotationQuat4[1]), py::cast<float>(rotationQuat4[2]), py::cast<float>(rotationQuat4[3])));
        const glm::vec2 newScale{py::cast<float>(scale3[0]), py::cast<float>(scale3[1])};
        const glm::vec3 newColor{py::cast<float>(albedo3[0]), py::cast<float>(albedo3[1]), py::cast<float>(albedo3[2])};
        auto pointAssetSharedPtr = assetManager->get<Pale::PointAsset>(pointCloudAssetHandle);
        if (!pointAssetSharedPtr) throw std::runtime_error("set_point_properties: failed to get PointAsset for dynamic point cloud");
        Pale::PointAsset &pointAsset = *pointAssetSharedPtr;
        if (pointAsset.points.empty()) throw std::runtime_error("set_point_properties: PointAsset has no PointGeometry blocks");
        Pale::PointGeometry &pointGeometry = pointAsset.points.front();
        if (index < 0) {
            for (std::size_t i = 0; i < pointGeometry.positions.size(); ++i) {
                pointGeometry.positions[i] += newTranslation;
                pointGeometry.quat[i] = normalizeQuaternionOrIdentity(rotationDelta * pointGeometry.quat[i]);
                pointGeometry.albedos[i] += newColor;
                pointGeometry.opacities[i] += opacity;
                pointGeometry.betas[i] += beta;
                pointGeometry.scales[i] *= newScale;
            }
        } else {
            if (index >= static_cast<int>(pointGeometry.positions.size())) throw std::runtime_error("set_point_properties: index out of range");
            pointGeometry.positions[index] += newTranslation;
            pointGeometry.quat[index] = normalizeQuaternionOrIdentity(rotationDelta * pointGeometry.quat[index]);
            pointGeometry.albedos[index] = newColor;
            pointGeometry.opacities[index] = opacity;
            pointGeometry.betas[index] = beta;
            pointGeometry.scales[index] = newScale;
        }
        rebuild_bvh();
    }

private:
    SelectedTrainingBatch selectTrainingBatch(const py::list &cameraNamesList,
                                              const char *callerName) {
        SelectedTrainingBatch selectedBatch;
        selectedBatch.sensors.reserve(py::len(cameraNamesList));
        selectedBatch.debugImages.reserve(py::len(cameraNamesList));
        selectedBatch.targets.reserve(py::len(cameraNamesList));

        for (const auto &cameraNameObject: cameraNamesList) {
            const std::string cameraName = py::cast<std::string>(cameraNameObject);
            bool foundSensor = false;

            for (std::size_t sensorIndex = 0; sensorIndex < sensorsForward.size(); ++sensorIndex) {
                const Pale::SensorGPU &sensor = sensorsForward[sensorIndex];
                const std::string sensorName(sensor.name, strnlen(sensor.name, sizeof(sensor.name)));
                if (sensorName != cameraName) {
                    continue;
                }

                auto targetIt = trainingTargets.find(cameraName);
                if (targetIt == trainingTargets.end() || targetIt->second.rgba == nullptr) {
                    throw std::runtime_error(
                        std::string(callerName) + ": no uploaded target image for camera '" + cameraName + "'");
                }

                selectedBatch.sensors.push_back(sensor);
                selectedBatch.debugImages.push_back(debugImages[sensorIndex]);
                selectedBatch.targets.push_back(&targetIt->second);
                foundSensor = true;
                break;
            }

            if (!foundSensor) {
                throw std::runtime_error(std::string(callerName) + ": camera not found: " + cameraName);
            }
        }

        if (selectedBatch.sensors.empty()) {
            throw std::runtime_error(std::string(callerName) + ": camera list is empty");
        }

        return selectedBatch;
    }

    static DeviceTrainingStepOptions parseDeviceTrainingStepOptions(const py::dict &optionsDictionary) {
        DeviceTrainingStepOptions options;
        options.optimizer = get_s(optionsDictionary, "optimizer", options.optimizer);
        if (options.optimizer != "adam" && options.optimizer != "sgd") {
            throw std::runtime_error(
                "render_rgb_training_step: expected optimizer to be 'adam' or 'sgd', got '" +
                options.optimizer + "'");
        }

        options.learningRatePosition =
                get_f(optionsDictionary, "learning_rate_position", options.learningRatePosition);
        options.learningRateRotation =
                get_f(optionsDictionary, "learning_rate_rotation", options.learningRateRotation);
        options.learningRateScale =
                get_f(optionsDictionary, "learning_rate_scale", options.learningRateScale);
        options.learningRateAlbedo =
                get_f(optionsDictionary, "learning_rate_albedo", options.learningRateAlbedo);
        options.learningRateOpacity =
                get_f(optionsDictionary, "learning_rate_opacity", options.learningRateOpacity);
        options.learningRateBeta =
                get_f(optionsDictionary, "learning_rate_beta", options.learningRateBeta);
        options.cameraBatchScale =
                get_f(optionsDictionary, "camera_batch_scale", options.cameraBatchScale);
        options.beta1 = get_f(optionsDictionary, "adam_beta1", options.beta1);
        options.beta2 = get_f(optionsDictionary, "adam_beta2", options.beta2);
        options.epsilon = get_f(optionsDictionary, "adam_epsilon", options.epsilon);
        options.maxRotationStepRadians =
                get_f(optionsDictionary, "max_rotation_step_radians", options.maxRotationStepRadians);
        return options;
    }

    static RgbLossOptions parseRgbLossOptions(const py::dict &optionsDictionary) {
        RgbLossOptions options;
        options.ssimWeight = get_f(optionsDictionary, "ssim_weight", options.ssimWeight);
        options.ssimWindowSize = get_i(
            optionsDictionary, "ssim_window_size", options.ssimWindowSize);
        options.ssimSigma = get_f(optionsDictionary, "ssim_sigma", options.ssimSigma);

        if (!std::isfinite(options.ssimWeight) ||
            options.ssimWeight < 0.0f || options.ssimWeight > 1.0f) {
            throw std::runtime_error("ssim_weight must be finite and in [0, 1]");
        }
        if (options.ssimWindowSize <= 0 ||
            options.ssimWindowSize > 31 ||
            (options.ssimWindowSize & 1) == 0) {
            throw std::runtime_error("ssim_window_size must be odd and in [1, 31]");
        }
        if (!std::isfinite(options.ssimSigma) || options.ssimSigma <= 0.0f) {
            throw std::runtime_error("ssim_sigma must be finite and positive");
        }
        return options;
    }

    bool isDeviceTrainingStateAllocated() const {
        return deviceTrainingState.pointCount != 0u &&
               deviceTrainingState.positionM != nullptr &&
               deviceTrainingState.positionV != nullptr &&
               deviceTrainingState.rotationM != nullptr &&
               deviceTrainingState.rotationV != nullptr &&
               deviceTrainingState.scaleM != nullptr &&
               deviceTrainingState.scaleV != nullptr &&
               deviceTrainingState.albedoM != nullptr &&
               deviceTrainingState.albedoV != nullptr &&
               deviceTrainingState.opacityM != nullptr &&
               deviceTrainingState.opacityV != nullptr &&
               deviceTrainingState.betaM != nullptr &&
               deviceTrainingState.betaV != nullptr;
    }

    static py::array makeDeviceAdamFloat3Array(
        std::vector<Pale::float3> &&hostVector,
        std::size_t elementCount) {
        auto *ownedVector = new std::vector<Pale::float3>(std::move(hostVector));
        std::vector<ssize_t> arrayShape{static_cast<ssize_t>(elementCount), 3};
        std::vector<ssize_t> arrayStrides{
            static_cast<ssize_t>(sizeof(Pale::float3)),
            static_cast<ssize_t>(sizeof(float))
        };

        return py::array(
            py::buffer_info(
                ownedVector->data(),
                sizeof(float),
                py::format_descriptor<float>::format(),
                2,
                arrayShape,
                arrayStrides),
            py::capsule(ownedVector, [](void *pointer) {
                delete static_cast<std::vector<Pale::float3> *>(pointer);
            }));
    }

    static py::array makeDeviceAdamFloat2Array(
        std::vector<Pale::float2> &&hostVector,
        std::size_t elementCount) {
        auto *ownedVector = new std::vector<Pale::float2>(std::move(hostVector));
        std::vector<ssize_t> arrayShape{static_cast<ssize_t>(elementCount), 2};
        std::vector<ssize_t> arrayStrides{
            static_cast<ssize_t>(sizeof(Pale::float2)),
            static_cast<ssize_t>(sizeof(float))
        };

        return py::array(
            py::buffer_info(
                ownedVector->data(),
                sizeof(float),
                py::format_descriptor<float>::format(),
                2,
                arrayShape,
                arrayStrides),
            py::capsule(ownedVector, [](void *pointer) {
                delete static_cast<std::vector<Pale::float2> *>(pointer);
            }));
    }

    static py::array makeDeviceAdamFloat1Array(
        std::vector<float> &&hostVector,
        std::size_t elementCount) {
        auto *ownedVector = new std::vector<float>(std::move(hostVector));
        std::vector<ssize_t> arrayShape{static_cast<ssize_t>(elementCount)};
        std::vector<ssize_t> arrayStrides{static_cast<ssize_t>(sizeof(float))};

        return py::array(
            py::buffer_info(
                ownedVector->data(),
                sizeof(float),
                py::format_descriptor<float>::format(),
                1,
                arrayShape,
                arrayStrides),
            py::capsule(ownedVector, [](void *pointer) {
                delete static_cast<std::vector<float> *>(pointer);
            }));
    }

    static std::vector<Pale::float3> readDeviceAdamFloat3Array(
        const py::dict &state,
        const char *key,
        std::size_t pointCount) {
        if (!state.contains(key)) {
            throw std::runtime_error(
                std::string("upload_device_adam_state: missing key '") + key + "'");
        }

        py::array_t<float, py::array::c_style | py::array::forcecast> array =
                state[py::str(key)].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        py::buffer_info info = array.request();
        if (info.ndim != 2 ||
            info.shape[0] != static_cast<ssize_t>(pointCount) ||
            info.shape[1] != 3) {
            throw std::runtime_error(
                std::string("upload_device_adam_state: expected '") +
                key +
                "' to have shape (N,3)");
        }

        const float *data = static_cast<const float *>(info.ptr);
        std::vector<Pale::float3> result(pointCount);
        for (std::size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
            const std::size_t base = pointIndex * 3u;
            result[pointIndex] = Pale::float3{
                data[base + 0u],
                data[base + 1u],
                data[base + 2u]
            };
        }
        return result;
    }

    static std::vector<Pale::float2> readDeviceAdamFloat2Array(
        const py::dict &state,
        const char *key,
        std::size_t pointCount) {
        if (!state.contains(key)) {
            throw std::runtime_error(
                std::string("upload_device_adam_state: missing key '") + key + "'");
        }

        py::array_t<float, py::array::c_style | py::array::forcecast> array =
                state[py::str(key)].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        py::buffer_info info = array.request();
        if (info.ndim != 2 ||
            info.shape[0] != static_cast<ssize_t>(pointCount) ||
            info.shape[1] != 2) {
            throw std::runtime_error(
                std::string("upload_device_adam_state: expected '") +
                key +
                "' to have shape (N,2)");
        }

        const float *data = static_cast<const float *>(info.ptr);
        std::vector<Pale::float2> result(pointCount);
        for (std::size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
            const std::size_t base = pointIndex * 2u;
            result[pointIndex] = Pale::float2{data[base + 0u], data[base + 1u]};
        }
        return result;
    }

    static std::vector<float> readDeviceAdamFloat1Array(
        const py::dict &state,
        const char *key,
        std::size_t pointCount) {
        if (!state.contains(key)) {
            throw std::runtime_error(
                std::string("upload_device_adam_state: missing key '") + key + "'");
        }

        py::array_t<float, py::array::c_style | py::array::forcecast> array =
                state[py::str(key)].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        py::buffer_info info = array.request();
        const bool isFlatVector =
                info.ndim == 1 &&
                info.shape[0] == static_cast<ssize_t>(pointCount);
        const bool isColumnVector =
                info.ndim == 2 &&
                info.shape[0] == static_cast<ssize_t>(pointCount) &&
                info.shape[1] == 1;
        if (!isFlatVector && !isColumnVector) {
            throw std::runtime_error(
                std::string("upload_device_adam_state: expected '") +
                key +
                "' to have shape (N,) or (N,1)");
        }

        const float *data = static_cast<const float *>(info.ptr);
        return std::vector<float>(data, data + pointCount);
    }

    void freeDeviceTrainingState(sycl::queue queue) {
        auto releasePointer = [&queue](auto *&pointer) {
            if (pointer) {
                sycl::free(pointer, queue);
                pointer = nullptr;
            }
        };

        releasePointer(deviceTrainingState.positionM);
        releasePointer(deviceTrainingState.positionV);
        releasePointer(deviceTrainingState.rotationM);
        releasePointer(deviceTrainingState.rotationV);
        releasePointer(deviceTrainingState.scaleM);
        releasePointer(deviceTrainingState.scaleV);
        releasePointer(deviceTrainingState.albedoM);
        releasePointer(deviceTrainingState.albedoV);
        releasePointer(deviceTrainingState.opacityM);
        releasePointer(deviceTrainingState.opacityV);
        releasePointer(deviceTrainingState.betaM);
        releasePointer(deviceTrainingState.betaV);
        deviceTrainingState.pointCount = 0;
        deviceTrainingState.step = 0;
    }

    void ensureDeviceTrainingState(std::size_t pointCount, sycl::queue queue) {
        if (pointCount == 0u) {
            throw std::runtime_error("ensureDeviceTrainingState: point count is zero");
        }

        const bool alreadyAllocated =
                deviceTrainingState.pointCount == pointCount &&
                deviceTrainingState.positionM != nullptr &&
                deviceTrainingState.positionV != nullptr &&
                deviceTrainingState.rotationM != nullptr &&
                deviceTrainingState.rotationV != nullptr &&
                deviceTrainingState.scaleM != nullptr &&
                deviceTrainingState.scaleV != nullptr &&
                deviceTrainingState.albedoM != nullptr &&
                deviceTrainingState.albedoV != nullptr &&
                deviceTrainingState.opacityM != nullptr &&
                deviceTrainingState.opacityV != nullptr &&
                deviceTrainingState.betaM != nullptr &&
                deviceTrainingState.betaV != nullptr;

        if (alreadyAllocated) {
            return;
        }

        freeDeviceTrainingState(queue);
        deviceTrainingState.pointCount = pointCount;

        deviceTrainingState.positionM = sycl::malloc_device<Pale::float3>(pointCount, queue);
        deviceTrainingState.positionV = sycl::malloc_device<Pale::float3>(pointCount, queue);
        deviceTrainingState.rotationM = sycl::malloc_device<Pale::float3>(pointCount, queue);
        deviceTrainingState.rotationV = sycl::malloc_device<Pale::float3>(pointCount, queue);
        deviceTrainingState.scaleM = sycl::malloc_device<Pale::float2>(pointCount, queue);
        deviceTrainingState.scaleV = sycl::malloc_device<Pale::float2>(pointCount, queue);
        deviceTrainingState.albedoM = sycl::malloc_device<Pale::float3>(pointCount, queue);
        deviceTrainingState.albedoV = sycl::malloc_device<Pale::float3>(pointCount, queue);
        deviceTrainingState.opacityM = sycl::malloc_device<float>(pointCount, queue);
        deviceTrainingState.opacityV = sycl::malloc_device<float>(pointCount, queue);
        deviceTrainingState.betaM = sycl::malloc_device<float>(pointCount, queue);
        deviceTrainingState.betaV = sycl::malloc_device<float>(pointCount, queue);

        if (!deviceTrainingState.positionM || !deviceTrainingState.positionV ||
            !deviceTrainingState.rotationM || !deviceTrainingState.rotationV ||
            !deviceTrainingState.scaleM || !deviceTrainingState.scaleV ||
            !deviceTrainingState.albedoM || !deviceTrainingState.albedoV ||
            !deviceTrainingState.opacityM || !deviceTrainingState.opacityV ||
            !deviceTrainingState.betaM || !deviceTrainingState.betaV) {
            freeDeviceTrainingState(queue);
            throw std::runtime_error("ensureDeviceTrainingState: failed to allocate Adam state on device");
        }

        queue.memset(deviceTrainingState.positionM, 0, pointCount * sizeof(Pale::float3));
        queue.memset(deviceTrainingState.positionV, 0, pointCount * sizeof(Pale::float3));
        queue.memset(deviceTrainingState.rotationM, 0, pointCount * sizeof(Pale::float3));
        queue.memset(deviceTrainingState.rotationV, 0, pointCount * sizeof(Pale::float3));
        queue.memset(deviceTrainingState.scaleM, 0, pointCount * sizeof(Pale::float2));
        queue.memset(deviceTrainingState.scaleV, 0, pointCount * sizeof(Pale::float2));
        queue.memset(deviceTrainingState.albedoM, 0, pointCount * sizeof(Pale::float3));
        queue.memset(deviceTrainingState.albedoV, 0, pointCount * sizeof(Pale::float3));
        queue.memset(deviceTrainingState.opacityM, 0, pointCount * sizeof(float));
        queue.memset(deviceTrainingState.opacityV, 0, pointCount * sizeof(float));
        queue.memset(deviceTrainingState.betaM, 0, pointCount * sizeof(float));
        queue.memset(deviceTrainingState.betaV, 0, pointCount * sizeof(float));
        queue.wait_and_throw();
    }

    void launchDeviceTrainingStepKernel(sycl::queue queue,
                                        const Pale::PointGradients &pointGradients,
                                        const Pale::PointGradients *depthGradients,
                                        const Pale::PointGradients *normalGradients,
                                        const Pale::PointGradients *visibilityGradients,
                                        const Pale::PointGradients *intraSlabGradients,
                                        const Pale::PointGradients *curvatureGradients,
                                        const DeviceTrainingStepOptions &options) {
        if (!sceneGpu.points || sceneGpu.pointCount == 0u) {
            throw std::runtime_error("launchDeviceTrainingStepKernel: scene has no device points");
        }
        if (pointGradients.numPoints != sceneGpu.pointCount ||
            deviceTrainingState.pointCount != sceneGpu.pointCount) {
            throw std::runtime_error("launchDeviceTrainingStepKernel: point count mismatch");
        }
        auto validateOptionalGradientSource = [&](const Pale::PointGradients *gradientSource,
                                                  const char *sourceName) {
            if (gradientSource && gradientSource->numPoints != sceneGpu.pointCount) {
                throw std::runtime_error(
                    std::string("launchDeviceTrainingStepKernel: ") +
                    sourceName +
                    " point count mismatch");
            }
        };
        validateOptionalGradientSource(depthGradients, "depthDistortionGradients");
        validateOptionalGradientSource(normalGradients, "normalConsistencyGradients");
        validateOptionalGradientSource(visibilityGradients, "visibilityOpacityGradients");
        validateOptionalGradientSource(intraSlabGradients, "intraSlabDepthGradients");
        validateOptionalGradientSource(curvatureGradients, "curvatureScaleGradients");

        const bool useAdam = options.optimizer == "adam";
        const std::uint32_t adamStep = ++deviceTrainingState.step;
        const float biasCorrection1 =
                useAdam ? 1.0f - std::pow(options.beta1, static_cast<float>(adamStep)) : 1.0f;
        const float biasCorrection2 =
                useAdam ? 1.0f - std::pow(options.beta2, static_cast<float>(adamStep)) : 1.0f;

        queue.parallel_for<class DeviceTrainingStepKernelTag>(
            sycl::range<1>(sceneGpu.pointCount),
            [points = sceneGpu.points,
             gradPosition = pointGradients.gradPosition,
             gradRotation = pointGradients.gradRotation,
             gradScale = pointGradients.gradScale,
             gradAlbedo = pointGradients.gradAlbedo,
             gradOpacity = pointGradients.gradOpacity,
             gradBeta = pointGradients.gradBeta,
             depthGradPosition = depthGradients ? depthGradients->gradPosition : nullptr,
             depthGradRotation = depthGradients ? depthGradients->gradRotation : nullptr,
             depthGradScale = depthGradients ? depthGradients->gradScale : nullptr,
             depthGradAlbedo = depthGradients ? depthGradients->gradAlbedo : nullptr,
             depthGradOpacity = depthGradients ? depthGradients->gradOpacity : nullptr,
             depthGradBeta = depthGradients ? depthGradients->gradBeta : nullptr,
             normalGradPosition = normalGradients ? normalGradients->gradPosition : nullptr,
             normalGradRotation = normalGradients ? normalGradients->gradRotation : nullptr,
             normalGradScale = normalGradients ? normalGradients->gradScale : nullptr,
             normalGradAlbedo = normalGradients ? normalGradients->gradAlbedo : nullptr,
             normalGradOpacity = normalGradients ? normalGradients->gradOpacity : nullptr,
             normalGradBeta = normalGradients ? normalGradients->gradBeta : nullptr,
             visibilityGradPosition = visibilityGradients ? visibilityGradients->gradPosition : nullptr,
             visibilityGradRotation = visibilityGradients ? visibilityGradients->gradRotation : nullptr,
             visibilityGradScale = visibilityGradients ? visibilityGradients->gradScale : nullptr,
             visibilityGradAlbedo = visibilityGradients ? visibilityGradients->gradAlbedo : nullptr,
             visibilityGradOpacity = visibilityGradients ? visibilityGradients->gradOpacity : nullptr,
             visibilityGradBeta = visibilityGradients ? visibilityGradients->gradBeta : nullptr,
             intraSlabGradPosition = intraSlabGradients ? intraSlabGradients->gradPosition : nullptr,
             intraSlabGradRotation = intraSlabGradients ? intraSlabGradients->gradRotation : nullptr,
             intraSlabGradScale = intraSlabGradients ? intraSlabGradients->gradScale : nullptr,
             intraSlabGradAlbedo = intraSlabGradients ? intraSlabGradients->gradAlbedo : nullptr,
             intraSlabGradOpacity = intraSlabGradients ? intraSlabGradients->gradOpacity : nullptr,
             intraSlabGradBeta = intraSlabGradients ? intraSlabGradients->gradBeta : nullptr,
             curvatureGradPosition = curvatureGradients ? curvatureGradients->gradPosition : nullptr,
             curvatureGradRotation = curvatureGradients ? curvatureGradients->gradRotation : nullptr,
             curvatureGradScale = curvatureGradients ? curvatureGradients->gradScale : nullptr,
             curvatureGradAlbedo = curvatureGradients ? curvatureGradients->gradAlbedo : nullptr,
             curvatureGradOpacity = curvatureGradients ? curvatureGradients->gradOpacity : nullptr,
             curvatureGradBeta = curvatureGradients ? curvatureGradients->gradBeta : nullptr,
             positionM = deviceTrainingState.positionM,
             positionV = deviceTrainingState.positionV,
             rotationM = deviceTrainingState.rotationM,
             rotationV = deviceTrainingState.rotationV,
             scaleM = deviceTrainingState.scaleM,
             scaleV = deviceTrainingState.scaleV,
             albedoM = deviceTrainingState.albedoM,
             albedoV = deviceTrainingState.albedoV,
             opacityM = deviceTrainingState.opacityM,
             opacityV = deviceTrainingState.opacityV,
             betaM = deviceTrainingState.betaM,
             betaV = deviceTrainingState.betaV,
             useAdam,
             beta1 = options.beta1,
             beta2 = options.beta2,
             epsilon = options.epsilon,
             biasCorrection1,
             biasCorrection2,
             lrPosition = options.learningRatePosition,
             lrRotation = options.learningRateRotation,
             lrScale = options.learningRateScale,
             lrAlbedo = options.learningRateAlbedo,
             lrOpacity = options.learningRateOpacity,
             lrBeta = options.learningRateBeta,
             cameraBatchScale = options.cameraBatchScale,
             maxRotationStepRadians = options.maxRotationStepRadians](sycl::id<1> itemId) {
                const std::uint32_t primitiveIndex = static_cast<std::uint32_t>(itemId[0]);
                Pale::Point &point = points[primitiveIndex];

                if (point.isEmissive()) {
                    return;
                }

                auto cleanGradient = [](float value) -> float {
                    constexpr float maxAbsGradientComponent = 1.0e6f;
                    if (!sycl::isfinite(value)) {
                        return 0.0f;
                    }
                    return sycl::fmin(
                        sycl::fmax(value, -maxAbsGradientComponent),
                        maxAbsGradientComponent);
                };
                auto cleanParameter = [](float value, float fallback) -> float {
                    return sycl::isfinite(value) ? value : fallback;
                };
                auto clampValue = [](float value, float minValue, float maxValue) -> float {
                    if (!sycl::isfinite(value)) {
                        return minValue;
                    }
                    return sycl::fmin(sycl::fmax(value, minValue), maxValue);
                };
                auto adamUpdate = [&](float gradientValue, float &m, float &v, float learningRate) -> float {
                    if (learningRate == 0.0f) {
                        return 0.0f;
                    }
                    if (!useAdam) {
                        return learningRate * gradientValue;
                    }
                    if (!sycl::isfinite(m)) {
                        m = 0.0f;
                    }
                    if (!sycl::isfinite(v)) {
                        v = 0.0f;
                    }
                    m = beta1 * m + (1.0f - beta1) * gradientValue;
                    v = beta2 * v + (1.0f - beta2) * gradientValue * gradientValue;
                    const float mHat = m / sycl::fmax(biasCorrection1, 1.0e-20f);
                    const float vHat = v / sycl::fmax(biasCorrection2, 1.0e-20f);
                    const float update = learningRate * mHat / (sycl::sqrt(vHat) + epsilon);
                    return sycl::isfinite(update) ? update : 0.0f;
                };
                auto dot3 = [](const Pale::float3 &a, const Pale::float3 &b) -> float {
                    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
                };
                auto cross3 = [](const Pale::float3 &a, const Pale::float3 &b) -> Pale::float3 {
                    return Pale::float3{
                        a.y() * b.z() - a.z() * b.y(),
                        a.z() * b.x() - a.x() * b.z(),
                        a.x() * b.y() - a.y() * b.x()
                    };
                };
                auto normalizeOrFallback = [&](const Pale::float3 &value,
                                               const Pale::float3 &fallback) -> Pale::float3 {
                    const float lengthSquared = dot3(value, value);
                    if (!sycl::isfinite(lengthSquared) || lengthSquared <= 1.0e-20f) {
                        return fallback;
                    }
                    const float invLength = sycl::rsqrt(lengthSquared);
                    return value * invLength;
                };
                auto sumFloat3Gradient = [&](Pale::float3 *depthPointer,
                                             Pale::float3 *normalPointer,
                                             Pale::float3 *visibilityPointer,
                                             Pale::float3 *intraSlabPointer,
                                             Pale::float3 *curvaturePointer,
                                             Pale::float3 baseGradient) -> Pale::float3 {
                    Pale::float3 gradient = baseGradient;
                    if (depthPointer) {
                        gradient += depthPointer[primitiveIndex];
                    }
                    if (normalPointer) {
                        gradient += normalPointer[primitiveIndex];
                    }
                    if (visibilityPointer) {
                        gradient += visibilityPointer[primitiveIndex];
                    }
                    if (intraSlabPointer) {
                        gradient += intraSlabPointer[primitiveIndex];
                    }
                    if (curvaturePointer) {
                        gradient += curvaturePointer[primitiveIndex];
                    }
                    return gradient;
                };
                auto sumFloatGradient = [&](float *depthPointer,
                                            float *normalPointer,
                                            float *visibilityPointer,
                                            float *intraSlabPointer,
                                            float *curvaturePointer,
                                            float baseGradient) -> float {
                    float gradient = baseGradient;
                    if (depthPointer) {
                        gradient += depthPointer[primitiveIndex];
                    }
                    if (normalPointer) {
                        gradient += normalPointer[primitiveIndex];
                    }
                    if (visibilityPointer) {
                        gradient += visibilityPointer[primitiveIndex];
                    }
                    if (intraSlabPointer) {
                        gradient += intraSlabPointer[primitiveIndex];
                    }
                    if (curvaturePointer) {
                        gradient += curvaturePointer[primitiveIndex];
                    }
                    return gradient;
                };

                const Pale::float3 positionGradient = sumFloat3Gradient(
                    depthGradPosition,
                    normalGradPosition,
                    visibilityGradPosition,
                    intraSlabGradPosition,
                    curvatureGradPosition,
                    gradPosition[primitiveIndex]);
                const Pale::float3 rotationGradient = sumFloat3Gradient(
                    depthGradRotation,
                    normalGradRotation,
                    visibilityGradRotation,
                    intraSlabGradRotation,
                    curvatureGradRotation,
                    gradRotation[primitiveIndex]);
                const Pale::float2 baseScaleGradient = gradScale[primitiveIndex];
                float scaleGradientX = baseScaleGradient.x();
                float scaleGradientY = baseScaleGradient.y();
                if (depthGradScale) {
                    scaleGradientX += depthGradScale[primitiveIndex].x();
                    scaleGradientY += depthGradScale[primitiveIndex].y();
                }
                if (normalGradScale) {
                    scaleGradientX += normalGradScale[primitiveIndex].x();
                    scaleGradientY += normalGradScale[primitiveIndex].y();
                }
                if (visibilityGradScale) {
                    scaleGradientX += visibilityGradScale[primitiveIndex].x();
                    scaleGradientY += visibilityGradScale[primitiveIndex].y();
                }
                if (intraSlabGradScale) {
                    scaleGradientX += intraSlabGradScale[primitiveIndex].x();
                    scaleGradientY += intraSlabGradScale[primitiveIndex].y();
                }
                if (curvatureGradScale) {
                    scaleGradientX += curvatureGradScale[primitiveIndex].x();
                    scaleGradientY += curvatureGradScale[primitiveIndex].y();
                }
                const Pale::float3 albedoGradient = sumFloat3Gradient(
                    depthGradAlbedo,
                    normalGradAlbedo,
                    visibilityGradAlbedo,
                    intraSlabGradAlbedo,
                    curvatureGradAlbedo,
                    gradAlbedo[primitiveIndex]);
                const float opacityGradient = sumFloatGradient(
                    depthGradOpacity,
                    normalGradOpacity,
                    visibilityGradOpacity,
                    intraSlabGradOpacity,
                    curvatureGradOpacity,
                    gradOpacity[primitiveIndex]);
                const float betaGradient = sumFloatGradient(
                    depthGradBeta,
                    normalGradBeta,
                    visibilityGradBeta,
                    intraSlabGradBeta,
                    curvatureGradBeta,
                    gradBeta[primitiveIndex]);

                const float positionUpdateX = adamUpdate(
                    cleanGradient(positionGradient.x() * cameraBatchScale),
                    positionM[primitiveIndex].x(),
                    positionV[primitiveIndex].x(),
                    lrPosition);
                const float positionUpdateY = adamUpdate(
                    cleanGradient(positionGradient.y() * cameraBatchScale),
                    positionM[primitiveIndex].y(),
                    positionV[primitiveIndex].y(),
                    lrPosition);
                const float positionUpdateZ = adamUpdate(
                    cleanGradient(positionGradient.z() * cameraBatchScale),
                    positionM[primitiveIndex].z(),
                    positionV[primitiveIndex].z(),
                    lrPosition);
                point.position.x() = clampValue(cleanParameter(point.position.x(), 0.0f) - positionUpdateX, -5.0f, 5.0f);
                point.position.y() = clampValue(cleanParameter(point.position.y(), 0.0f) - positionUpdateY, -5.0f, 5.0f);
                point.position.z() = clampValue(cleanParameter(point.position.z(), 0.0f) - positionUpdateZ, -5.0f, 5.0f);

                float rotationDeltaX = -adamUpdate(
                    cleanGradient(rotationGradient.x() * cameraBatchScale),
                    rotationM[primitiveIndex].x(),
                    rotationV[primitiveIndex].x(),
                    lrRotation);
                float rotationDeltaY = -adamUpdate(
                    cleanGradient(rotationGradient.y() * cameraBatchScale),
                    rotationM[primitiveIndex].y(),
                    rotationV[primitiveIndex].y(),
                    lrRotation);
                float rotationDeltaZ = -adamUpdate(
                    cleanGradient(rotationGradient.z() * cameraBatchScale),
                    rotationM[primitiveIndex].z(),
                    rotationV[primitiveIndex].z(),
                    lrRotation);

                const float rotationLength = sycl::sqrt(
                    rotationDeltaX * rotationDeltaX +
                    rotationDeltaY * rotationDeltaY +
                    rotationDeltaZ * rotationDeltaZ);
                if (rotationLength > 1.0e-12f && sycl::isfinite(rotationLength)) {
                    if (maxRotationStepRadians > 0.0f && rotationLength > maxRotationStepRadians) {
                        const float clampScale = maxRotationStepRadians / rotationLength;
                        rotationDeltaX *= clampScale;
                        rotationDeltaY *= clampScale;
                        rotationDeltaZ *= clampScale;
                    }

                    const float clampedRotationLength = sycl::sqrt(
                        rotationDeltaX * rotationDeltaX +
                        rotationDeltaY * rotationDeltaY +
                        rotationDeltaZ * rotationDeltaZ);
                    const float invRotationLength = 1.0f / sycl::fmax(clampedRotationLength, 1.0e-12f);
                    const float axisX = rotationDeltaX * invRotationLength;
                    const float axisY = rotationDeltaY * invRotationLength;
                    const float axisZ = rotationDeltaZ * invRotationLength;

                    const float c = sycl::cos(clampedRotationLength);
                    const float s = sycl::sin(clampedRotationLength);
                    const float t = 1.0f - c;

                    const float r00 = t * axisX * axisX + c;
                    const float r01 = t * axisX * axisY - s * axisZ;
                    const float r10 = t * axisX * axisY + s * axisZ;
                    const float r11 = t * axisY * axisY + c;
                    const float r20 = t * axisX * axisZ - s * axisY;
                    const float r21 = t * axisY * axisZ + s * axisX;

                    Pale::float3 tangentU = normalizeOrFallback(point.tanU, Pale::float3{1.0f, 0.0f, 0.0f});
                    Pale::float3 tangentV = point.tanV - tangentU * dot3(tangentU, point.tanV);
                    const Pale::float3 fallbackV =
                            sycl::fabs(tangentU.y()) < 0.9f
                                ? Pale::float3{0.0f, 1.0f, 0.0f}
                                : Pale::float3{1.0f, 0.0f, 0.0f};
                    tangentV = normalizeOrFallback(tangentV, fallbackV - tangentU * dot3(tangentU, fallbackV));
                    const Pale::float3 tangentW = normalizeOrFallback(
                        cross3(tangentU, tangentV),
                        Pale::float3{0.0f, 0.0f, 1.0f});

                    Pale::float3 updatedTangentU =
                            tangentU * r00 +
                            tangentV * r10 +
                            tangentW * r20;
                    Pale::float3 updatedTangentV =
                            tangentU * r01 +
                            tangentV * r11 +
                            tangentW * r21;

                    updatedTangentU = normalizeOrFallback(updatedTangentU, tangentU);
                    updatedTangentV = updatedTangentV - updatedTangentU * dot3(updatedTangentU, updatedTangentV);
                    updatedTangentV = normalizeOrFallback(updatedTangentV, tangentV);

                    point.tanU = updatedTangentU;
                    point.tanV = updatedTangentV;
                }

                const float scaleUpdateX = adamUpdate(
                    cleanGradient(scaleGradientX * cameraBatchScale),
                    scaleM[primitiveIndex].x(),
                    scaleV[primitiveIndex].x(),
                    lrScale);
                const float scaleUpdateY = adamUpdate(
                    cleanGradient(scaleGradientY * cameraBatchScale),
                    scaleM[primitiveIndex].y(),
                    scaleV[primitiveIndex].y(),
                    lrScale);
                constexpr float minSurfelScale = 1.0e-6f;
                point.scale.x() = clampValue(cleanParameter(point.scale.x(), minSurfelScale) - scaleUpdateX, minSurfelScale, 1.0f);
                point.scale.y() = clampValue(cleanParameter(point.scale.y(), minSurfelScale) - scaleUpdateY, minSurfelScale, 1.0f);

                const float albedoUpdateX = adamUpdate(
                    cleanGradient(albedoGradient.x() * cameraBatchScale),
                    albedoM[primitiveIndex].x(),
                    albedoV[primitiveIndex].x(),
                    lrAlbedo);
                const float albedoUpdateY = adamUpdate(
                    cleanGradient(albedoGradient.y() * cameraBatchScale),
                    albedoM[primitiveIndex].y(),
                    albedoV[primitiveIndex].y(),
                    lrAlbedo);
                const float albedoUpdateZ = adamUpdate(
                    cleanGradient(albedoGradient.z() * cameraBatchScale),
                    albedoM[primitiveIndex].z(),
                    albedoV[primitiveIndex].z(),
                    lrAlbedo);
                point.albedo.x() = clampValue(cleanParameter(point.albedo.x(), 0.0f) - albedoUpdateX, 0.0f, 1.0f);
                point.albedo.y() = clampValue(cleanParameter(point.albedo.y(), 0.0f) - albedoUpdateY, 0.0f, 1.0f);
                point.albedo.z() = clampValue(cleanParameter(point.albedo.z(), 0.0f) - albedoUpdateZ, 0.0f, 1.0f);

                const float opacityUpdate = adamUpdate(
                    cleanGradient(opacityGradient * cameraBatchScale),
                    opacityM[primitiveIndex],
                    opacityV[primitiveIndex],
                    lrOpacity);
                point.opacity = clampValue(cleanParameter(point.opacity, 0.0f) - opacityUpdate, 0.0f, 1.0f);

                const float betaUpdate = adamUpdate(
                    cleanGradient(betaGradient * cameraBatchScale),
                    betaM[primitiveIndex],
                    betaV[primitiveIndex],
                    lrBeta);
                point.beta = clampValue(cleanParameter(point.beta, 1.0f) - betaUpdate, -2.0f, 5.0f);
            });
    }

    void launchPointBvhRefitKernel(sycl::queue queue) {
        if (!sceneGpu.points || !sceneGpu.blasNodes || !sceneGpu.blasRanges ||
            !sceneGpu.tlasNodes || !sceneGpu.instances || !sceneGpu.transforms ||
            !sceneGpu.pointPermutation || !sceneGpu.pointTraversalData) {
            return;
        }

        const std::uint32_t instanceCount =
                static_cast<std::uint32_t>(buildProducts.instances.size());
        if (instanceCount == 0u || sceneGpu.blasNodeCount == 0u || sceneGpu.tlasNodeCount == 0u) {
            return;
        }

        queue.single_task<class PointBvhRefitKernelTag>(
            [scene = sceneGpu, instanceCount]() {
                auto dot3 = [](const Pale::float3 &a, const Pale::float3 &b) -> float {
                    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
                };
                auto cross3 = [](const Pale::float3 &a, const Pale::float3 &b) -> Pale::float3 {
                    return Pale::float3{
                        a.y() * b.z() - a.z() * b.y(),
                        a.z() * b.x() - a.x() * b.z(),
                        a.x() * b.y() - a.y() * b.x()
                    };
                };
                auto normalizeOrFallback = [&](const Pale::float3 &value,
                                               const Pale::float3 &fallback) -> Pale::float3 {
                    const float lengthSquared = dot3(value, value);
                    if (!sycl::isfinite(lengthSquared) || lengthSquared <= 1.0e-20f) {
                        return fallback;
                    }
                    return value * sycl::rsqrt(lengthSquared);
                };
                auto min3 = [](const Pale::float3 &a, const Pale::float3 &b) -> Pale::float3 {
                    return Pale::float3{
                        sycl::fmin(a.x(), b.x()),
                        sycl::fmin(a.y(), b.y()),
                        sycl::fmin(a.z(), b.z())
                    };
                };
                auto max3 = [](const Pale::float3 &a, const Pale::float3 &b) -> Pale::float3 {
                    return Pale::float3{
                        sycl::fmax(a.x(), b.x()),
                        sycl::fmax(a.y(), b.y()),
                        sycl::fmax(a.z(), b.z())
                    };
                };
                auto makeSurfelAabbBeta = [&](const Pale::Point &surfel,
                                               Pale::float3 &aabbMin,
                                               Pale::float3 &aabbMax) {
                    const Pale::float3 tangentU =
                            normalizeOrFallback(surfel.tanU, Pale::float3{1.0f, 0.0f, 0.0f});
                    const Pale::float3 tangentV =
                            normalizeOrFallback(surfel.tanV, Pale::float3{0.0f, 1.0f, 0.0f});
                    const Pale::float3 normalDirection =
                            normalizeOrFallback(cross3(tangentU, tangentV), Pale::float3{0.0f, 0.0f, 1.0f});

                    const float supportRadiusU = sycl::fmax(surfel.scale.x(), 0.0f);
                    const float supportRadiusV = sycl::fmax(surfel.scale.y(), 0.0f);
                    constexpr float normalThickness = 0.001f;

                    auto computeAxisExtent = [&](int axisIndex) -> float {
                        const float tangentUComponent =
                                axisIndex == 0
                                    ? tangentU.x()
                                    : (axisIndex == 1 ? tangentU.y() : tangentU.z());
                        const float tangentVComponent =
                                axisIndex == 0
                                    ? tangentV.x()
                                    : (axisIndex == 1 ? tangentV.y() : tangentV.z());
                        const float normalComponent =
                                sycl::fabs(axisIndex == 0
                                               ? normalDirection.x()
                                               : (axisIndex == 1 ? normalDirection.y() : normalDirection.z()));
                        const float projectedInPlane =
                                sycl::sqrt((supportRadiusU * tangentUComponent) *
                                           (supportRadiusU * tangentUComponent) +
                                           (supportRadiusV * tangentVComponent) *
                                           (supportRadiusV * tangentVComponent));
                        return projectedInPlane + normalThickness * normalComponent;
                    };

                    const Pale::float3 halfExtent{
                        computeAxisExtent(0),
                        computeAxisExtent(1),
                        computeAxisExtent(2)
                    };
                    aabbMin = surfel.position - halfExtent;
                    aabbMax = surfel.position + halfExtent;
                };
                auto writePackedPointBvhChild = [](Pale::PackedPointBVHNode &packedNode,
                                                   bool writeLeftChild,
                                                   const Pale::BVHNode *nodes,
                                                   std::uint32_t childNodeIndex) {
                    const Pale::BVHNode &childNode = nodes[childNodeIndex];
                    const std::uint32_t childIndex =
                            childNode.triCount > 0u ? childNode.leftFirst : childNodeIndex;
                    const std::uint32_t childCount =
                            childNode.triCount > 0u ? childNode.triCount : 0u;

                    if (writeLeftChild) {
                        packedNode.leftAabbMin = childNode.aabbMin;
                        packedNode.leftAabbMax = childNode.aabbMax;
                        packedNode.leftIndex = childIndex;
                        packedNode.leftCount = childCount;
                    } else {
                        packedNode.rightAabbMin = childNode.aabbMin;
                        packedNode.rightAabbMax = childNode.aabbMax;
                        packedNode.rightIndex = childIndex;
                        packedNode.rightCount = childCount;
                    }
                };
                auto writePackedPointBvhNode = [&](Pale::PackedPointBVHNode *packedNodes,
                                                   const Pale::BVHNode *nodes,
                                                   std::uint32_t localNodeIndex) {
                    const Pale::BVHNode &node = nodes[localNodeIndex];
                    Pale::PackedPointBVHNode packedNode{};
                    if (node.triCount > 0u) {
                        writePackedPointBvhChild(packedNode, true, nodes, localNodeIndex);
                    } else {
                        writePackedPointBvhChild(packedNode, true, nodes, node.leftFirst);
                        writePackedPointBvhChild(packedNode, false, nodes, node.leftFirst + 1u);
                    }
                    packedNodes[localNodeIndex] = packedNode;
                };
                auto updatePackedPointQbvhNode = [](Pale::PackedPointQBVHNode &qbvhNode,
                                                    const Pale::BVHNode *nodes) {
                    for (std::uint32_t slot = 0u; slot < 4u; ++slot) {
                        const std::uint32_t sourceNodeIndex = qbvhNode.childSourceNodeIndex[slot];
                        if (sourceNodeIndex == UINT32_MAX) {
                            continue;
                        }

                        const Pale::BVHNode &sourceNode = nodes[sourceNodeIndex];
                        qbvhNode.minX[slot] = sourceNode.aabbMin.x();
                        qbvhNode.minY[slot] = sourceNode.aabbMin.y();
                        qbvhNode.minZ[slot] = sourceNode.aabbMin.z();
                        qbvhNode.maxX[slot] = sourceNode.aabbMax.x();
                        qbvhNode.maxY[slot] = sourceNode.aabbMax.y();
                        qbvhNode.maxZ[slot] = sourceNode.aabbMax.z();

                        if (sourceNode.triCount > 0u) {
                            qbvhNode.childIndex[slot] = sourceNode.leftFirst;
                            qbvhNode.childCount[slot] = sourceNode.triCount;
                        } else {
                            qbvhNode.childCount[slot] = 0u;
                        }
                    }
                };

                for (std::uint32_t instanceIndex = 0u; instanceIndex < instanceCount; ++instanceIndex) {
                    const Pale::InstanceRecord &instance = scene.instances[instanceIndex];
                    if (instance.geometryType != Pale::GeometryType::PointCloud) {
                        continue;
                    }

                    const Pale::BLASRange blasRange = scene.blasRanges[instance.blasRangeIndex];
                    if (blasRange.nodeCount == 0u) {
                        continue;
                    }

                    Pale::BVHNode *nodes = scene.blasNodes + blasRange.firstNode;
                    Pale::PackedPointBVHNode *packedNodes = nullptr;
                    if (scene.pointPackedBvhNodes != nullptr &&
                        scene.pointPackedBvhRanges != nullptr &&
                        instance.blasRangeIndex < scene.pointPackedBvhRangeCount) {
                        const Pale::BLASRange packedRange = scene.pointPackedBvhRanges[instance.blasRangeIndex];
                        if (packedRange.nodeCount >= blasRange.nodeCount &&
                            packedRange.firstNode + packedRange.nodeCount <= scene.pointPackedBvhNodeCount) {
                            packedNodes = scene.pointPackedBvhNodes + packedRange.firstNode;
                        }
                    }
                    Pale::PackedPointQBVHNode *qbvhNodes = nullptr;
                    std::uint32_t qbvhNodeCount = 0u;
                    if (scene.pointQbvhNodes != nullptr &&
                        scene.pointQbvhRanges != nullptr &&
                        instance.blasRangeIndex < scene.pointQbvhRangeCount) {
                        const Pale::BLASRange qbvhRange = scene.pointQbvhRanges[instance.blasRangeIndex];
                        if (qbvhRange.nodeCount > 0u &&
                            qbvhRange.firstNode + qbvhRange.nodeCount <= scene.pointQbvhNodeCount) {
                            qbvhNodes = scene.pointQbvhNodes + qbvhRange.firstNode;
                            qbvhNodeCount = qbvhRange.nodeCount;
                        }
                    }

                    for (int localNodeIndex = static_cast<int>(blasRange.nodeCount) - 1;
                         localNodeIndex >= 0;
                         --localNodeIndex) {
                        Pale::BVHNode &node = nodes[localNodeIndex];
                        if (node.triCount > 0u) {
                            Pale::float3 nodeMin{FLT_MAX, FLT_MAX, FLT_MAX};
                            Pale::float3 nodeMax{-FLT_MAX, -FLT_MAX, -FLT_MAX};
                            for (std::uint32_t primitiveOffset = 0u;
                                 primitiveOffset < node.triCount;
                                 ++primitiveOffset) {
                                const std::uint32_t primitiveIndex =
                                        scene.pointPermutation[node.leftFirst + primitiveOffset];
                                scene.pointTraversalData[node.leftFirst + primitiveOffset] =
                                        Pale::makeSurfelTraversalData(scene.points[primitiveIndex], primitiveIndex);
                                Pale::float3 surfelMin{0.0f};
                                Pale::float3 surfelMax{0.0f};
                                makeSurfelAabbBeta(scene.points[primitiveIndex], surfelMin, surfelMax);
                                nodeMin = min3(nodeMin, surfelMin);
                                nodeMax = max3(nodeMax, surfelMax);
                            }
                            node.aabbMin = nodeMin;
                            node.aabbMax = nodeMax;
                        } else {
                            const Pale::BVHNode &leftNode = nodes[node.leftFirst];
                            const Pale::BVHNode &rightNode = nodes[node.leftFirst + 1u];
                            node.aabbMin = min3(leftNode.aabbMin, rightNode.aabbMin);
                            node.aabbMax = max3(leftNode.aabbMax, rightNode.aabbMax);
                        }

                        if (packedNodes != nullptr) {
                            writePackedPointBvhNode(
                                packedNodes,
                                nodes,
                                static_cast<std::uint32_t>(localNodeIndex));
                        }
                    }

                    if (qbvhNodes != nullptr) {
                        for (std::uint32_t qbvhNodeIndex = 0u;
                             qbvhNodeIndex < qbvhNodeCount;
                             ++qbvhNodeIndex) {
                            updatePackedPointQbvhNode(qbvhNodes[qbvhNodeIndex], nodes);
                        }
                    }
                }

                for (int tlasNodeIndex = static_cast<int>(scene.tlasNodeCount) - 1;
                     tlasNodeIndex >= 0;
                     --tlasNodeIndex) {
                    Pale::TLASNode &node = scene.tlasNodes[tlasNodeIndex];
                    if (node.count > 0u) {
                        const std::uint32_t instanceIndex = node.leftChild;
                        if (instanceIndex >= instanceCount) {
                            continue;
                        }
                        const Pale::InstanceRecord &instance = scene.instances[instanceIndex];
                        const Pale::BLASRange blasRange = scene.blasRanges[instance.blasRangeIndex];
                        if (blasRange.nodeCount == 0u) {
                            continue;
                        }
                        const Pale::BVHNode &rootNode = scene.blasNodes[blasRange.firstNode];
                        const Pale::Transform &transform = scene.transforms[instance.transformIndex];

                        Pale::float3 worldMin{FLT_MAX, FLT_MAX, FLT_MAX};
                        Pale::float3 worldMax{-FLT_MAX, -FLT_MAX, -FLT_MAX};
                        for (int cornerIndex = 0; cornerIndex < 8; ++cornerIndex) {
                            const bool bx = (cornerIndex & 4) != 0;
                            const bool by = (cornerIndex & 2) != 0;
                            const bool bz = (cornerIndex & 1) != 0;
                            const Pale::float3 pointObject{
                                bx ? rootNode.aabbMax.x() : rootNode.aabbMin.x(),
                                by ? rootNode.aabbMax.y() : rootNode.aabbMin.y(),
                                bz ? rootNode.aabbMax.z() : rootNode.aabbMin.z()
                            };
                            const Pale::float3 pointWorld = Pale::toWorldPoint(pointObject, transform);
                            worldMin = min3(worldMin, pointWorld);
                            worldMax = max3(worldMax, pointWorld);
                        }
                        node.aabbMin = worldMin;
                        node.aabbMax = worldMax;
                    } else {
                        const Pale::TLASNode &leftNode = scene.tlasNodes[node.leftChild];
                        const Pale::TLASNode &rightNode = scene.tlasNodes[node.rightChild];
                        node.aabbMin = min3(leftNode.aabbMin, rightNode.aabbMin);
                        node.aabbMax = max3(leftNode.aabbMax, rightNode.aabbMax);
                    }
                }
            });
    }

    static py::dict makeZeroLossValuesDictionary() {
        py::dict result;
        result["total_rgb_loss_value"] = 0.0f;
        result["total_depth_distortion_loss_raw"] = 0.0f;
        result["total_depth_distortion_loss_weighted"] = 0.0f;
        result["total_normal_loss_raw"] = 0.0f;
        result["total_normal_loss_weighted"] = 0.0f;
        result["total_opacity_prior_loss_raw"] = 0.0f;
        result["total_opacity_prior_loss_weighted"] = 0.0f;
        result["total_intra_slab_depth_loss_raw"] = 0.0f;
        result["total_intra_slab_depth_loss_weighted"] = 0.0f;
        result["total_curvature_scale_loss_raw"] = 0.0f;
        result["total_curvature_scale_loss_weighted"] = 0.0f;
        result["total_loss_value"] = 0.0f;
        return result;
    }

    static void launchSurfaceRegularizerLossAccumulationKernel(
        sycl::queue queue,
        const Pale::SensorGPU &sensor,
        float *depthDistortionSum,
        float *normalConsistencySum,
        float *visibilityWeightedOpacitySum,
        float *intraSlabDepthSum,
        float *curvatureScaleSum,
        std::uint32_t *normalConsistencyValidCount,
        std::uint32_t *intraSlabDepthActiveSlabCount,
        std::uint32_t *curvatureScaleActiveSlabCount,
        bool useDepthDistortion,
        bool useNormalConsistency,
        bool useVisibilityWeightedOpacity,
        bool useIntraSlabDepth,
        bool useCurvatureScale) {
        const std::uint32_t pixelCount = sensor.width * sensor.height;
        if (pixelCount == 0u) {
            return;
        }

        if (useDepthDistortion && (!sensor.depthDistortionBuffer || !depthDistortionSum)) {
            throw std::runtime_error(
                "launchSurfaceRegularizerLossAccumulationKernel: missing depth distortion buffers");
        }
        if (useNormalConsistency &&
            (!sensor.visibleNormalBuffer ||
             !sensor.normalFromDepthBuffer ||
             !normalConsistencySum ||
             !normalConsistencyValidCount)) {
            throw std::runtime_error(
                "launchSurfaceRegularizerLossAccumulationKernel: missing normal consistency buffers");
        }
        if (useVisibilityWeightedOpacity &&
            (!sensor.visibilityWeightedOpacityBuffer || !visibilityWeightedOpacitySum)) {
            throw std::runtime_error(
                "launchSurfaceRegularizerLossAccumulationKernel: missing visibility-weighted opacity buffers");
        }
        if (useIntraSlabDepth &&
            (!sensor.intraSlabDepthBuffer || !sensor.intraSlabDepthActiveSlabCountBuffer ||
             !intraSlabDepthSum || !intraSlabDepthActiveSlabCount)) {
            throw std::runtime_error(
                "launchSurfaceRegularizerLossAccumulationKernel: missing intra-slab depth buffers");
        }
        if (useCurvatureScale &&
            (!sensor.curvatureScaleBuffer || !sensor.curvatureScaleActiveSlabCountBuffer ||
             !curvatureScaleSum || !curvatureScaleActiveSlabCount)) {
            throw std::runtime_error(
                "launchSurfaceRegularizerLossAccumulationKernel: missing curvature-scale buffers");
        }

        queue.parallel_for<class SurfaceRegularizerLossAccumulationKernelTag>(
            sycl::range<1>(pixelCount),
            [depthDistortionBuffer = sensor.depthDistortionBuffer,
             visibleNormalBuffer = sensor.visibleNormalBuffer,
             normalFromDepthBuffer = sensor.normalFromDepthBuffer,
             visibilityWeightedOpacityBuffer = sensor.visibilityWeightedOpacityBuffer,
             intraSlabDepthBuffer = sensor.intraSlabDepthBuffer,
             intraSlabDepthCountBuffer = sensor.intraSlabDepthActiveSlabCountBuffer,
             curvatureScaleBuffer = sensor.curvatureScaleBuffer,
             curvatureScaleCountBuffer = sensor.curvatureScaleActiveSlabCountBuffer,
             depthDistortionSum,
             normalConsistencySum,
             visibilityWeightedOpacitySum,
             intraSlabDepthSum,
             curvatureScaleSum,
             normalConsistencyValidCount,
             intraSlabDepthActiveSlabCount,
             curvatureScaleActiveSlabCount,
             useDepthDistortion,
             useNormalConsistency,
             useVisibilityWeightedOpacity,
             useIntraSlabDepth,
             useCurvatureScale](sycl::id<1> pixelId) {
                const std::uint32_t pixelIndex = static_cast<std::uint32_t>(pixelId[0]);

                auto clean = [](float value) -> float {
                    return sycl::isfinite(value) ? value : 0.0f;
                };

                if (useDepthDistortion) {
                    const float depthValue = clean(depthDistortionBuffer[pixelIndex]);
                    auto depthAtomic = sycl::atomic_ref<
                        float,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(*depthDistortionSum);
                    depthAtomic.fetch_add(depthValue);
                }

                if (useVisibilityWeightedOpacity) {
                    const float opacityValue = clean(visibilityWeightedOpacityBuffer[pixelIndex]);
                    auto opacityAtomic = sycl::atomic_ref<
                        float,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(*visibilityWeightedOpacitySum);
                    opacityAtomic.fetch_add(opacityValue);
                }

                if (useIntraSlabDepth) {
                    const float lossValue = clean(intraSlabDepthBuffer[pixelIndex]);
                    const std::uint32_t activeCount = intraSlabDepthCountBuffer[pixelIndex];
                    auto lossAtomic = sycl::atomic_ref<
                        float,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(*intraSlabDepthSum);
                    lossAtomic.fetch_add(lossValue);
                    auto countAtomic = sycl::atomic_ref<
                        std::uint32_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(*intraSlabDepthActiveSlabCount);
                    countAtomic.fetch_add(activeCount);
                }

                if (useCurvatureScale) {
                    const float lossValue = clean(curvatureScaleBuffer[pixelIndex]);
                    const std::uint32_t activeCount = curvatureScaleCountBuffer[pixelIndex];
                    auto lossAtomic = sycl::atomic_ref<
                        float,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(*curvatureScaleSum);
                    lossAtomic.fetch_add(lossValue);
                    auto countAtomic = sycl::atomic_ref<
                        std::uint32_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(*curvatureScaleActiveSlabCount);
                    countAtomic.fetch_add(activeCount);
                }

                if (useNormalConsistency) {
                    const Pale::float4 visibleNormal = visibleNormalBuffer[pixelIndex];
                    const Pale::float4 depthNormal = normalFromDepthBuffer[pixelIndex];

                    const float visibleW = clean(visibleNormal.w());
                    const float depthW = clean(depthNormal.w());
                    if (visibleW > 0.0f && depthW > 0.0f) {
                        const float visibleX = clean(visibleNormal.x());
                        const float visibleY = clean(visibleNormal.y());
                        const float visibleZ = clean(visibleNormal.z());
                        const float depthX = clean(depthNormal.x());
                        const float depthY = clean(depthNormal.y());
                        const float depthZ = clean(depthNormal.z());
                        const float dotNormal =
                                visibleX * depthX + visibleY * depthY + visibleZ * depthZ;

                        auto normalAtomic = sycl::atomic_ref<
                            float,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(*normalConsistencySum);
                        normalAtomic.fetch_add(1.0f - dotNormal);

                        auto countAtomic = sycl::atomic_ref<
                            std::uint32_t,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(*normalConsistencyValidCount);
                        countAtomic.fetch_add(1u);
                    }
                }
            });
    }

    static void launchSurfaceRegularizerAdjointFillKernel(
        sycl::queue queue,
        const Pale::SensorGPU &sensor,
        float depthDistortionWeight,
        float normalConsistencyWeight,
        float intraSlabDepthWeight,
        float curvatureScaleWeight,
        std::uint32_t normalConsistencyValidCount,
        std::uint32_t intraSlabDepthActiveSlabCount,
        std::uint32_t curvatureScaleActiveSlabCount,
        bool useDepthDistortion,
        bool useNormalConsistency,
        bool useIntraSlabDepth,
        bool useCurvatureScale) {
        const std::uint32_t pixelCount = sensor.width * sensor.height;
        if (pixelCount == 0u) {
            return;
        }

        if (!sensor.depthDistortionAdjointBuffer ||
            !sensor.intraSlabDepthAdjointBuffer ||
            !sensor.curvatureScaleAdjointBuffer ||
            !sensor.visibleNormalAdjointBuffer ||
            !sensor.normalFromDepthAdjointBuffer ||
            !sensor.medianDepthAdjointBuffer) {
            throw std::runtime_error(
                "launchSurfaceRegularizerAdjointFillKernel: missing surface regularizer adjoint buffers");
        }
        if (useNormalConsistency &&
            (!sensor.visibleNormalBuffer || !sensor.normalFromDepthBuffer)) {
            throw std::runtime_error(
                "launchSurfaceRegularizerAdjointFillKernel: missing normal consistency forward buffers");
        }

        const float depthScale = useDepthDistortion
                                     ? depthDistortionWeight / static_cast<float>(pixelCount)
                                     : 0.0f;
        const float normalScale = useNormalConsistency
                                      ? normalConsistencyWeight /
                                        static_cast<float>(std::max(normalConsistencyValidCount, 1u))
                                      : 0.0f;
        const float intraSlabDepthScale = useIntraSlabDepth
            ? intraSlabDepthWeight /
              static_cast<float>(std::max(intraSlabDepthActiveSlabCount, 1u))
            : 0.0f;
        const float curvatureScale = useCurvatureScale
            ? curvatureScaleWeight /
              static_cast<float>(std::max(curvatureScaleActiveSlabCount, 1u))
            : 0.0f;

        queue.parallel_for<class SurfaceRegularizerAdjointFillKernelTag>(
            sycl::range<1>(pixelCount),
            [visibleNormalBuffer = sensor.visibleNormalBuffer,
             normalFromDepthBuffer = sensor.normalFromDepthBuffer,
             depthDistortionAdjointBuffer = sensor.depthDistortionAdjointBuffer,
             intraSlabDepthAdjointBuffer = sensor.intraSlabDepthAdjointBuffer,
             intraSlabDepthCountBuffer = sensor.intraSlabDepthActiveSlabCountBuffer,
             curvatureScaleAdjointBuffer = sensor.curvatureScaleAdjointBuffer,
             curvatureScaleCountBuffer = sensor.curvatureScaleActiveSlabCountBuffer,
             visibleNormalAdjointBuffer = sensor.visibleNormalAdjointBuffer,
             normalFromDepthAdjointBuffer = sensor.normalFromDepthAdjointBuffer,
             medianDepthAdjointBuffer = sensor.medianDepthAdjointBuffer,
             depthScale,
             normalScale,
             intraSlabDepthScale,
             curvatureScale,
             useIntraSlabDepth,
             useCurvatureScale,
             useNormalConsistency](sycl::id<1> pixelId) {
                const std::uint32_t pixelIndex = static_cast<std::uint32_t>(pixelId[0]);

                auto clean = [](float value) -> float {
                    return sycl::isfinite(value) ? value : 0.0f;
                };

                depthDistortionAdjointBuffer[pixelIndex] = depthScale;
                intraSlabDepthAdjointBuffer[pixelIndex] =
                    useIntraSlabDepth && intraSlabDepthCountBuffer[pixelIndex] > 0u
                        ? intraSlabDepthScale
                        : 0.0f;
                curvatureScaleAdjointBuffer[pixelIndex] =
                    useCurvatureScale && curvatureScaleCountBuffer[pixelIndex] > 0u
                        ? curvatureScale
                        : 0.0f;
                medianDepthAdjointBuffer[pixelIndex] = 0.0f;

                Pale::float4 visibleAdjoint{0.0f, 0.0f, 0.0f, 0.0f};
                Pale::float4 depthAdjoint{0.0f, 0.0f, 0.0f, 0.0f};

                if (useNormalConsistency) {
                    const Pale::float4 visibleNormal = visibleNormalBuffer[pixelIndex];
                    const Pale::float4 depthNormal = normalFromDepthBuffer[pixelIndex];
                    const float visibleW = clean(visibleNormal.w());
                    const float depthW = clean(depthNormal.w());

                    if (visibleW > 0.0f && depthW > 0.0f) {
                        const float visibleX = clean(visibleNormal.x());
                        const float visibleY = clean(visibleNormal.y());
                        const float visibleZ = clean(visibleNormal.z());
                        const float depthX = clean(depthNormal.x());
                        const float depthY = clean(depthNormal.y());
                        const float depthZ = clean(depthNormal.z());

                        visibleAdjoint = Pale::float4{
                            -normalScale * depthX,
                            -normalScale * depthY,
                            -normalScale * depthZ,
                            0.0f
                        };
                        depthAdjoint = Pale::float4{
                            -normalScale * visibleX,
                            -normalScale * visibleY,
                            -normalScale * visibleZ,
                            0.0f
                        };
                    }
                }

                visibleNormalAdjointBuffer[pixelIndex] = visibleAdjoint;
                normalFromDepthAdjointBuffer[pixelIndex] = depthAdjoint;
            });
    }

    bool syncPointParametersFromGpuIfDirty() {
        return syncPointParametersFromGpu(false);
    }

    bool syncPointParametersFromGpu(bool forceSync) {
        if (!forceSync && !devicePointParametersDirty) {
            return false;
        }
        if (!sceneGpu.points || sceneGpu.pointCount == 0u) {
            devicePointParametersDirty = false;
            return false;
        }
        if (!assetManager) {
            throw std::runtime_error("sync_point_parameters_from_gpu: assetManager is null");
        }

        const std::size_t pointCount = sceneGpu.pointCount;
        std::vector<Pale::Point> hostPoints(pointCount);
        auto queue = deviceSelector->getQueue();
        queue.memcpy(hostPoints.data(), sceneGpu.points, pointCount * sizeof(Pale::Point)).wait_and_throw();

        buildProducts.points = hostPoints;

        auto pointAssetSharedPtr = assetManager->get<Pale::PointAsset>(pointCloudAssetHandle);
        if (!pointAssetSharedPtr) {
            throw std::runtime_error("sync_point_parameters_from_gpu: failed to get PointAsset for dynamic point cloud");
        }

        Pale::PointAsset &pointAsset = *pointAssetSharedPtr;
        if (pointAsset.points.empty()) {
            throw std::runtime_error("sync_point_parameters_from_gpu: PointAsset has no PointGeometry blocks");
        }

        Pale::PointGeometry &pointGeometry = pointAsset.points.front();
        pointGeometry.positions.resize(pointCount);
        pointGeometry.quat.resize(pointCount);
        pointGeometry.scales.resize(pointCount);
        pointGeometry.albedos.resize(pointCount);
        pointGeometry.opacities.resize(pointCount);
        pointGeometry.shapes.resize(pointCount);
        pointGeometry.betas.resize(pointCount);
        pointGeometry.powers.resize(pointCount);

        for (std::size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
            const Pale::Point &point = hostPoints[pointIndex];
            pointGeometry.positions[pointIndex] =
                    glm::vec3(point.position.x(), point.position.y(), point.position.z());
            pointGeometry.quat[pointIndex] = quaternionFromFrame(point.tanU, point.tanV);
            pointGeometry.scales[pointIndex] = glm::vec2(point.scale.x(), point.scale.y());
            pointGeometry.albedos[pointIndex] =
                    glm::vec3(point.albedo.x(), point.albedo.y(), point.albedo.z());
            pointGeometry.opacities[pointIndex] = point.opacity;
            pointGeometry.shapes[pointIndex] = point.shape;
            pointGeometry.betas[pointIndex] = point.beta;
            pointGeometry.powers[pointIndex] = point.flux;
        }

        devicePointParametersDirty = false;
        return true;
    }

    static void freeTrainingTarget(TrainingTargetDevice &target, sycl::queue queue) {
        if (target.rgba) {
            sycl::free(target.rgba, queue);
            target.rgba = nullptr;
        }
        if (target.loss) {
            sycl::free(target.loss, queue);
            target.loss = nullptr;
        }
        target.width = 0;
        target.height = 0;
    }

    void freeTrainingTargets(sycl::queue queue) {
        for (auto &[cameraName, target]: trainingTargets) {
            freeTrainingTarget(target, queue);
        }
        trainingTargets.clear();
    }

    static void ensureTrainingTargetCapacity(TrainingTargetDevice &target,
                                             const std::string &cameraName,
                                             std::uint32_t width,
                                             std::uint32_t height,
                                             sycl::queue queue) {
        const bool sameShape =
                target.rgba != nullptr &&
                target.loss != nullptr &&
                target.width == width &&
                target.height == height;
        if (sameShape) {
            return;
        }

        freeTrainingTarget(target, queue);
        target.cameraName = cameraName;
        target.width = width;
        target.height = height;

        const std::size_t pixelCount = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        target.rgba = sycl::malloc_device<Pale::float4>(pixelCount, queue);
        target.loss = sycl::malloc_device<float>(3u, queue);
        if (!target.rgba || !target.loss) {
            throw std::runtime_error("ensureTrainingTargetCapacity: failed to allocate device target buffers");
        }
    }

    void freeRgbSsimScratch(sycl::queue queue) {
        auto release = [&queue](Pale::float4 *&pointer) {
            if (pointer) {
                sycl::free(pointer, queue);
                pointer = nullptr;
            }
        };
        release(rgbSsimScratch.renderedMean);
        release(rgbSsimScratch.targetMean);
        release(rgbSsimScratch.derivativeMean);
        release(rgbSsimScratch.derivativeVariance);
        release(rgbSsimScratch.derivativeCovariance);
        rgbSsimScratch.pixelCapacity = 0;
    }

    void ensureRgbSsimScratchCapacity(std::size_t pixelCount, sycl::queue queue) {
        if (rgbSsimScratch.pixelCapacity >= pixelCount &&
            rgbSsimScratch.renderedMean && rgbSsimScratch.targetMean &&
            rgbSsimScratch.derivativeMean && rgbSsimScratch.derivativeVariance &&
            rgbSsimScratch.derivativeCovariance) {
            return;
        }

        // A multi-camera batch reuses this storage serially. If a later camera is
        // larger, finish the already queued SSIM pass before replacing its storage.
        if (rgbSsimScratch.pixelCapacity > 0u) {
            queue.wait_and_throw();
        }
        freeRgbSsimScratch(queue);
        rgbSsimScratch.renderedMean = sycl::malloc_device<Pale::float4>(pixelCount, queue);
        rgbSsimScratch.targetMean = sycl::malloc_device<Pale::float4>(pixelCount, queue);
        rgbSsimScratch.derivativeMean = sycl::malloc_device<Pale::float4>(pixelCount, queue);
        rgbSsimScratch.derivativeVariance = sycl::malloc_device<Pale::float4>(pixelCount, queue);
        rgbSsimScratch.derivativeCovariance = sycl::malloc_device<Pale::float4>(pixelCount, queue);
        if (!rgbSsimScratch.renderedMean || !rgbSsimScratch.targetMean ||
            !rgbSsimScratch.derivativeMean || !rgbSsimScratch.derivativeVariance ||
            !rgbSsimScratch.derivativeCovariance) {
            freeRgbSsimScratch(queue);
            throw std::runtime_error("ensureRgbSsimScratchCapacity: failed to allocate SSIM scratch buffers");
        }
        rgbSsimScratch.pixelCapacity = pixelCount;
    }

    void launchRgbLossAdjointKernel(sycl::queue queue,
                                    const Pale::SensorGPU &sensor,
                                    TrainingTargetDevice &target,
                                    const RgbLossOptions &options) {
        if (!sensor.framebuffer || !target.rgba || !target.loss) {
            throw std::runtime_error("launchRgbLossAdjointKernel: missing framebuffer or target buffer");
        }
        if (sensor.width != target.width || sensor.height != target.height) {
            throw std::runtime_error("launchRgbLossAdjointKernel: sensor/target resolution mismatch");
        }

        const std::uint32_t pixelCount = sensor.width * sensor.height;
        const float invElementCount = pixelCount > 0u
                                          ? 1.0f / (static_cast<float>(pixelCount) * 3.0f)
                                          : 0.0f;

        queue.fill(target.loss, 0.0f, 3u);
        if (options.ssimWeight <= 0.0f) {
            queue.parallel_for<class RgbLossAdjointKernelTag>(
                sycl::range<1>(pixelCount),
                [framebuffer = sensor.framebuffer,
                 targetRgba = target.rgba,
                 lossOut = target.loss,
                 invElementCount](sycl::id<1> pixelId) {
                    const std::uint32_t pixelIndex = static_cast<std::uint32_t>(pixelId[0]);
                    const Pale::float4 rendered = framebuffer[pixelIndex];
                    const Pale::float4 targetPixel = targetRgba[pixelIndex];

                    const float diffR = rendered.x() - targetPixel.x();
                    const float diffG = rendered.y() - targetPixel.y();
                    const float diffB = rendered.z() - targetPixel.z();
                    const float l2Contribution =
                            0.5f * (diffR * diffR + diffG * diffG + diffB * diffB) * invElementCount;

                    auto combinedAtomic = sycl::atomic_ref<
                        float,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(lossOut[0]);
                    combinedAtomic.fetch_add(l2Contribution);
                    auto l2Atomic = sycl::atomic_ref<
                        float,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(lossOut[1]);
                    l2Atomic.fetch_add(l2Contribution);

                    framebuffer[pixelIndex] = Pale::float4{
                        diffR * invElementCount,
                        diffG * invElementCount,
                        diffB * invElementCount,
                        0.0f
                    };
                });
            return;
        }

        ensureRgbSsimScratchCapacity(pixelCount, queue);

        const int windowRadius = options.ssimWindowSize / 2;
        const float gaussianExponentScale =
                -0.5f / (options.ssimSigma * options.ssimSigma);
        float gaussianSum = 0.0f;
        for (int offset = -windowRadius; offset <= windowRadius; ++offset) {
            gaussianSum += std::exp(
                static_cast<float>(offset * offset) * gaussianExponentScale);
        }
        const float invGaussian2dSum = 1.0f / (gaussianSum * gaussianSum);
        constexpr float C1 = 0.01f * 0.01f;
        constexpr float C2 = 0.03f * 0.03f;

        // Pass 1 computes each SSIM window and the three local partial derivatives
        // needed by the transposed Gaussian convolution in the image adjoint.
        queue.parallel_for<class RgbSsimStatisticsKernelTag>(
            sycl::range<1>(pixelCount),
            [framebuffer = sensor.framebuffer,
             targetRgba = target.rgba,
             lossOut = target.loss,
             renderedMeanOut = rgbSsimScratch.renderedMean,
             targetMeanOut = rgbSsimScratch.targetMean,
             derivativeMeanOut = rgbSsimScratch.derivativeMean,
             derivativeVarianceOut = rgbSsimScratch.derivativeVariance,
             derivativeCovarianceOut = rgbSsimScratch.derivativeCovariance,
             width = sensor.width,
             height = sensor.height,
             windowRadius,
             gaussianExponentScale,
             invGaussian2dSum,
             invElementCount,
             ssimWeight = options.ssimWeight](sycl::id<1> pixelId) {
                const std::uint32_t pixelIndex = static_cast<std::uint32_t>(pixelId[0]);
                const int centerX = static_cast<int>(pixelIndex % width);
                const int centerY = static_cast<int>(pixelIndex / width);

                Pale::float4 renderedMean{0.0f, 0.0f, 0.0f, 0.0f};
                Pale::float4 targetMean{0.0f, 0.0f, 0.0f, 0.0f};
                Pale::float4 renderedSecondMoment{0.0f, 0.0f, 0.0f, 0.0f};
                Pale::float4 targetSecondMoment{0.0f, 0.0f, 0.0f, 0.0f};
                Pale::float4 crossMoment{0.0f, 0.0f, 0.0f, 0.0f};

                for (int offsetY = -windowRadius; offsetY <= windowRadius; ++offsetY) {
                    const int sampleY = centerY + offsetY;
                    if (sampleY < 0 || sampleY >= static_cast<int>(height)) continue;
                    for (int offsetX = -windowRadius; offsetX <= windowRadius; ++offsetX) {
                        const int sampleX = centerX + offsetX;
                        if (sampleX < 0 || sampleX >= static_cast<int>(width)) continue;
                        const float radiusSquared = static_cast<float>(
                            offsetX * offsetX + offsetY * offsetY);
                        const float weight =
                            sycl::exp(radiusSquared * gaussianExponentScale) * invGaussian2dSum;
                        const std::uint32_t sampleIndex =
                            static_cast<std::uint32_t>(sampleY) * width +
                            static_cast<std::uint32_t>(sampleX);
                        const Pale::float4 renderedSample = framebuffer[sampleIndex];
                        const Pale::float4 targetSample = targetRgba[sampleIndex];
                        renderedMean += renderedSample * weight;
                        targetMean += targetSample * weight;
                        renderedSecondMoment += Pale::float4{
                            renderedSample.x() * renderedSample.x(),
                            renderedSample.y() * renderedSample.y(),
                            renderedSample.z() * renderedSample.z(),
                            0.0f
                        } * weight;
                        targetSecondMoment += Pale::float4{
                            targetSample.x() * targetSample.x(),
                            targetSample.y() * targetSample.y(),
                            targetSample.z() * targetSample.z(),
                            0.0f
                        } * weight;
                        crossMoment += Pale::float4{
                            renderedSample.x() * targetSample.x(),
                            renderedSample.y() * targetSample.y(),
                            renderedSample.z() * targetSample.z(),
                            0.0f
                        } * weight;
                    }
                }

                const Pale::float4 renderedVariance{
                    renderedSecondMoment.x() - renderedMean.x() * renderedMean.x(),
                    renderedSecondMoment.y() - renderedMean.y() * renderedMean.y(),
                    renderedSecondMoment.z() - renderedMean.z() * renderedMean.z(),
                    0.0f
                };
                const Pale::float4 targetVariance{
                    targetSecondMoment.x() - targetMean.x() * targetMean.x(),
                    targetSecondMoment.y() - targetMean.y() * targetMean.y(),
                    targetSecondMoment.z() - targetMean.z() * targetMean.z(),
                    0.0f
                };
                const Pale::float4 covariance{
                    crossMoment.x() - renderedMean.x() * targetMean.x(),
                    crossMoment.y() - renderedMean.y() * targetMean.y(),
                    crossMoment.z() - renderedMean.z() * targetMean.z(),
                    0.0f
                };
                Pale::float4 derivativeMean{0.0f, 0.0f, 0.0f, 0.0f};
                Pale::float4 derivativeVariance{0.0f, 0.0f, 0.0f, 0.0f};
                Pale::float4 derivativeCovariance{0.0f, 0.0f, 0.0f, 0.0f};
                float ssimSum = 0.0f;

                for (int channel = 0; channel < 3; ++channel) {
                    const float meanRendered = renderedMean[channel];
                    const float meanTarget = targetMean[channel];
                    const float luminanceNumerator =
                        2.0f * meanRendered * meanTarget + C1;
                    const float luminanceDenominator = sycl::fmax(
                        meanRendered * meanRendered + meanTarget * meanTarget + C1,
                        1.0e-12f);
                    const float contrastNumerator = 2.0f * covariance[channel] + C2;
                    const float contrastDenominator = sycl::fmax(
                        renderedVariance[channel] + targetVariance[channel] + C2,
                        1.0e-12f);
                    const float luminance = luminanceNumerator / luminanceDenominator;
                    const float contrast = contrastNumerator / contrastDenominator;
                    ssimSum += luminance * contrast;

                    derivativeMean[channel] = contrast * (
                        2.0f * meanTarget * luminanceDenominator -
                        2.0f * meanRendered * luminanceNumerator) /
                        (luminanceDenominator * luminanceDenominator);
                    derivativeVariance[channel] =
                        -luminance * contrastNumerator /
                        (contrastDenominator * contrastDenominator);
                    derivativeCovariance[channel] =
                        2.0f * luminance / contrastDenominator;
                }

                renderedMeanOut[pixelIndex] = renderedMean;
                targetMeanOut[pixelIndex] = targetMean;
                derivativeMeanOut[pixelIndex] = derivativeMean;
                derivativeVarianceOut[pixelIndex] = derivativeVariance;
                derivativeCovarianceOut[pixelIndex] = derivativeCovariance;

                const Pale::float4 rendered = framebuffer[pixelIndex];
                const Pale::float4 targetPixel = targetRgba[pixelIndex];
                const float diffR = rendered.x() - targetPixel.x();
                const float diffG = rendered.y() - targetPixel.y();
                const float diffB = rendered.z() - targetPixel.z();
                const float l2Contribution =
                    0.5f * (diffR * diffR + diffG * diffG + diffB * diffB) *
                    invElementCount;
                const float dssimContribution =
                    (3.0f - ssimSum) * invElementCount;
                const float combinedContribution =
                    (1.0f - ssimWeight) * l2Contribution +
                    ssimWeight * dssimContribution;

                auto combinedAtomic = sycl::atomic_ref<
                    float,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(lossOut[0]);
                combinedAtomic.fetch_add(combinedContribution);
                auto l2Atomic = sycl::atomic_ref<
                    float,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(lossOut[1]);
                l2Atomic.fetch_add(l2Contribution);
                auto dssimAtomic = sycl::atomic_ref<
                    float,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(lossOut[2]);
                dssimAtomic.fetch_add(dssimContribution);
            });

        // Pass 2 applies the transpose of the Gaussian window. For each image sample i:
        // dSSIM/dx_i = sum_j w_ji [S_mu + 2(x_i-mu_j)S_var + (y_i-muy_j)S_cov].
        queue.parallel_for<class RgbSsimAdjointKernelTag>(
            sycl::range<1>(pixelCount),
            [framebuffer = sensor.framebuffer,
             targetRgba = target.rgba,
             renderedMean = rgbSsimScratch.renderedMean,
             targetMean = rgbSsimScratch.targetMean,
             derivativeMean = rgbSsimScratch.derivativeMean,
             derivativeVariance = rgbSsimScratch.derivativeVariance,
             derivativeCovariance = rgbSsimScratch.derivativeCovariance,
             width = sensor.width,
             height = sensor.height,
             windowRadius,
             gaussianExponentScale,
             invGaussian2dSum,
             invElementCount,
             ssimWeight = options.ssimWeight](sycl::id<1> pixelId) {
                const std::uint32_t pixelIndex = static_cast<std::uint32_t>(pixelId[0]);
                const int sampleX = static_cast<int>(pixelIndex % width);
                const int sampleY = static_cast<int>(pixelIndex / width);
                const Pale::float4 rendered = framebuffer[pixelIndex];
                const Pale::float4 targetPixel = targetRgba[pixelIndex];
                Pale::float4 ssimGradient{0.0f, 0.0f, 0.0f, 0.0f};

                for (int offsetY = -windowRadius; offsetY <= windowRadius; ++offsetY) {
                    const int centerY = sampleY + offsetY;
                    if (centerY < 0 || centerY >= static_cast<int>(height)) continue;
                    for (int offsetX = -windowRadius; offsetX <= windowRadius; ++offsetX) {
                        const int centerX = sampleX + offsetX;
                        if (centerX < 0 || centerX >= static_cast<int>(width)) continue;
                        const float radiusSquared = static_cast<float>(
                            offsetX * offsetX + offsetY * offsetY);
                        const float weight =
                            sycl::exp(radiusSquared * gaussianExponentScale) * invGaussian2dSum;
                        const std::uint32_t centerIndex =
                            static_cast<std::uint32_t>(centerY) * width +
                            static_cast<std::uint32_t>(centerX);
                        for (int channel = 0; channel < 3; ++channel) {
                            ssimGradient[channel] += weight * (
                                derivativeMean[centerIndex][channel] +
                                2.0f * (rendered[channel] - renderedMean[centerIndex][channel]) *
                                    derivativeVariance[centerIndex][channel] +
                                (targetPixel[channel] - targetMean[centerIndex][channel]) *
                                    derivativeCovariance[centerIndex][channel]);
                        }
                    }
                }

                framebuffer[pixelIndex] = Pale::float4{
                    ((1.0f - ssimWeight) * (rendered.x() - targetPixel.x()) -
                     ssimWeight * ssimGradient.x()) * invElementCount,
                    ((1.0f - ssimWeight) * (rendered.y() - targetPixel.y()) -
                     ssimWeight * ssimGradient.y()) * invElementCount,
                    ((1.0f - ssimWeight) * (rendered.z() - targetPixel.z()) -
                     ssimWeight * ssimGradient.z()) * invElementCount,
                    0.0f
                };
            });
    }

    std::unique_ptr<Pale::AssetManager> assetManager{};
    std::shared_ptr<Pale::Scene> scene{};
    std::unique_ptr<Pale::DeviceSelector> deviceSelector{};
    Pale::PathTracerSettings m_settings{};

    std::vector<Pale::SensorGPU> sensorsForward{};
    std::unique_ptr<Pale::PathTracer> pathTracer{};
    std::vector<Pale::DebugImages> debugImages;

    Pale::AssetHandle pointCloudAssetHandle{};

    Pale::SceneBuild::BuildProducts buildProducts{};
    Pale::GPUSceneBuffers sceneGpu{};
    Pale::PointGradients gradients{};
    Pale::PointGradients depthDistortionGradients{};
    Pale::PointGradients normalConsistencyGradients{};
    Pale::PointGradients visibilityOpacityGradients{};
    Pale::PointGradients intraSlabDepthGradients{};
    Pale::PointGradients curvatureScaleGradients{};
    Pale::CurvatureDensificationStats curvatureDensificationStats{};
    Pale::PrimalActivityStats primalActivityStats{};
    bool primalActivityTrackingEnabled{false};
    bool curvatureDensificationEnabled{false};
    std::unordered_map<std::string, TrainingTargetDevice> trainingTargets{};
    RgbSsimScratch rgbSsimScratch{};
    DeviceAdamState deviceTrainingState{};
    bool devicePointParametersDirty{false};

    // Adjoint buffers
    bool adjointBuffersAllocated{false};
    float *adjointFramebuffer{nullptr};
    float *adjointFramebufferGrad{nullptr};
    Pale::float3 *gradientPkBuffer{nullptr};
    size_t gradCount{1024}; // set to your point count or resize after build
};

// ---- pybind11 module ----
PYBIND11_MODULE(pale, m) {
    py::class_<PythonRenderer>(m, "Renderer")
            .def(py::init<
                     const std::string &,
                     const std::string &,
                     const std::string &,
                     const py::dict &>(),
                 py::arg("assetRootDir"),
                 py::arg("sceneXml") = "cbox_custom.xml",
                 py::arg("pointCloudFile") = "initial.ply",
                 py::arg("settings") = py::dict()
            )
            .def("render_forward", &PythonRenderer::render_forward, py::arg("camera_name") = "")
            .def("upload_training_targets",
                 &PythonRenderer::upload_training_targets,
                 py::arg("target_images"))
            .def("render_rgb_loss_backward",
                 &PythonRenderer::render_rgb_loss_backward,
                 py::arg("camera_names"),
                 py::arg("options") = py::dict())
            .def("render_rgb_training_step",
                 &PythonRenderer::render_rgb_training_step,
                 py::arg("camera_names"),
                 py::arg("options") = py::dict())
            .def("render_rgb_backward_from_current_forward",
                 &PythonRenderer::render_rgb_backward_from_current_forward,
                 py::arg("camera_names"),
                 py::arg("options") = py::dict())
            .def("render_forward_surface_regularizer_loss_and_adjoint",
                 &PythonRenderer::render_forward_surface_regularizer_loss_and_adjoint,
                 py::arg("camera_names"),
                 py::arg("options") = py::dict())
            .def("render_surface_regularizers_backward_from_current_adjoint",
                 &PythonRenderer::render_surface_regularizers_backward_from_current_adjoint,
                 py::arg("camera_names"),
                 py::arg("return_gradients") = false)
            .def("apply_device_training_step",
                 &PythonRenderer::apply_device_training_step,
                 py::arg("options") = py::dict())
            .def("reset_trainable_opacity_on_gpu",
                 &PythonRenderer::reset_trainable_opacity_on_gpu,
                 py::arg("opacity"))
            .def("get_training_camera_names", &PythonRenderer::getTrainingCameras)
            .def("get_camera_names", &PythonRenderer::getCameraNames)
            .def("render_backward", &PythonRenderer::render_backward, py::arg("targetRgb32f"))
            .def("render_depth_distortion_backward",
                 &PythonRenderer::render_depth_distortion_backward,
                 py::arg("depthDistortionGrad32f"))
            .def("render_normal_consistency_backward",
                 &PythonRenderer::render_normal_consistency_backward,
                 py::arg("visibleNormalGrad32f"),
                 py::arg("normalFromDepthGrad32f"))
            .def("get_point_parameters", &PythonRenderer::get_point_parameters)
            .def("get_curvature_densification_stats",
                 &PythonRenderer::get_curvature_densification_stats)
            .def("get_primal_activity_stats",
                 &PythonRenderer::get_primal_activity_stats)
            .def("sync_point_parameters_from_gpu", &PythonRenderer::sync_point_parameters_from_gpu)
            .def("capture_device_adam_state", &PythonRenderer::capture_device_adam_state)
            .def("upload_device_adam_state",
                 &PythonRenderer::upload_device_adam_state,
                 py::arg("state"))
            .def("apply_point_optimization", &PythonRenderer::apply_point_optimization, py::arg("parameters"))
            .def("add_points", &PythonRenderer::add_new_points, py::arg("parameters"))
            .def("remove_points", &PythonRenderer::remove_points, py::arg("parameters"))
            .def("rebuild_bvh", &PythonRenderer::rebuild_bvh)
            .def("set_point_properties",
                 &PythonRenderer::set_point_properties,
                 py::arg("translation3"), py::arg("rotation_quat4"),
                 py::arg("scale3"), py::arg("albedo3"),
                 py::arg("opacity"), py::arg("beta"),
                 py::arg("index") = -1)
            .def("set_point_opacity", &PythonRenderer::set_point_opacity, py::arg("opacity"), py::arg("index"))
            .def("set_point_translation", &PythonRenderer::set_point_translation, py::arg("translation"),
                 py::arg("axis"), py::arg("index"))
            .def("set_point_albedo", &PythonRenderer::set_point_albedo, py::arg("intensity"), py::arg("axis"),
                 py::arg("index"))
            .def("set_point_rotation_degrees", &PythonRenderer::set_point_rotation_degrees, py::arg("rotation_deg"),
                 py::arg("axis"), py::arg("index"))
            .def("set_point_scale", &PythonRenderer::set_point_scale, py::arg("scale"), py::arg("axis"),
                 py::arg("index"))
            .def("set_point_beta", &PythonRenderer::set_point_beta, py::arg("beta"), py::arg("index")).def(
                "render_surface_regularizers_backward",
                &PythonRenderer::render_surface_regularizers_backward,
                py::arg("camera_names"),
                py::arg("depth_distortion_grad_images"),
                py::arg("visible_normal_grad_images"),
                py::arg("normal_from_depth_grad_images"),
                py::arg("intra_slab_depth_grad_images"),
                py::arg("curvature_scale_grad_images"))
            .def("render_surface_regularizers_backward_no_gradients",
                 &PythonRenderer::render_surface_regularizers_backward_no_gradients,
                 py::arg("camera_names"),
                 py::arg("depth_distortion_grad_images"),
                 py::arg("visible_normal_grad_images"),
                 py::arg("normal_from_depth_grad_images"),
                 py::arg("intra_slab_depth_grad_images"),
                 py::arg("curvature_scale_grad_images"));
}
