// Main.cpp
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <entt/entt.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/string_cast.hpp"
#include "Renderer/RenderPackage.h"
#include "spdlog/spdlog.h"

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

static std::string assetPathOrId(const Pale::AssetRegistry &registry, const Pale::AssetHandle &assetHandle) {
    if (auto meta = registry.meta(assetHandle)) {
        return meta->path.string();
    }
    return std::string(assetHandle);
}

static bool isNonzeroWeight(float weight) {
    return std::isfinite(weight) && std::fabs(weight) > 0.0f;
}

static void logSceneSummary(std::shared_ptr<Pale::Scene> &scene, Pale::AssetManager &assetManager) {
    auto &registry = assetManager.registry();

    Pale::Log::PA_INFO("===== Scene Summary =====");
    size_t entityCount = 0;
    size_t meshCount = 0;
    size_t emissiveCount = 0;

    auto view = scene->getAllEntitiesWith<Pale::IDComponent>();
    for (entt::entity entity: view) {
        Pale::Entity sceneEntity(entity, scene.get());
        ++entityCount;

        const char *name = sceneEntity.getName().c_str();
        const bool hasMesh = sceneEntity.hasComponent<Pale::MeshComponent>();
        const bool hasMaterial = sceneEntity.hasComponent<Pale::MaterialComponent>();
        const bool hasEmitter = sceneEntity.hasComponent<Pale::AreaLightComponent>();

        Pale::Log::PA_INFO("[Entity] {}", name);

        if (hasMesh) {
            auto &meshComponent = sceneEntity.getComponent<Pale::MeshComponent>();
            ++meshCount;
            std::string meshLabel = assetPathOrId(registry, meshComponent.meshID);

            size_t submeshCount = 0;
            if (auto mesh = assetManager.get<Pale::Mesh>(meshComponent.meshID)) {
                submeshCount = mesh->submeshes.size();
            } else {
                Pale::Log::PA_WARN("  Mesh: {} (FAILED to load)", meshLabel);
            }

            Pale::Log::PA_INFO("  Mesh: {}  (submeshes: {})", meshLabel, submeshCount);
        }

        if (hasMaterial) {
            auto &materialComponent = sceneEntity.getComponent<Pale::MaterialComponent>();
            std::string materialLabel = assetPathOrId(registry, materialComponent.materialID);

            if (auto material = assetManager.get<Pale::Material>(materialComponent.materialID)) {
                Pale::Log::PA_INFO(
                    "  Material: {}  [baseColor=({:.3f},{:.3f},{:.3f}) roughness={:.3f} metallic={:.3f}]",
                    materialLabel,
                    material->baseColor.x,
                    material->baseColor.y,
                    material->baseColor.z,
                    material->roughness,
                    material->metallic);
            } else {
                Pale::Log::PA_INFO("  Material: {}  (pending load)", materialLabel);
            }
        }

        if (hasEmitter) {
            ++emissiveCount;
            auto &emitterComponent = sceneEntity.getComponent<Pale::AreaLightComponent>();
            Pale::Log::PA_INFO(
                "  Emissive radiance=({:.3f},{:.3f},{:.3f})",
                emitterComponent.radiance.x,
                emitterComponent.radiance.y,
                emitterComponent.radiance.z);
        }
    }

    Pale::Log::PA_INFO(
        "===== Totals: entities={} meshes={} emissives={} =====",
        entityCount,
        meshCount,
        emissiveCount);
}

void rebuild_bvh(Pale::PathTracer *pathTracer,
                 std::shared_ptr<Pale::Scene> &scene,
                 Pale::SceneBuild::BuildProducts &buildProducts,
                 Pale::AssetManager *assetManager,
                 Pale::DeviceSelector &deviceSelector,
                 Pale::GPUSceneBuffers &sceneGpu) {
    Pale::AssetAccessFromManager assetAccessor(*assetManager);

    auto options = Pale::SceneBuild::BuildOptions();
    options.bvhMaxLeafPoints = 4;

    buildProducts = Pale::SceneBuild::build(scene, assetAccessor, options);

    Pale::SceneUpload::uploadOrReallocate(
        buildProducts,
        sceneGpu,
        deviceSelector.getQueue());

    pathTracer->setScene(sceneGpu, buildProducts);
}

static std::vector<float> packScalarToRGBA(
    const std::vector<float> &scalarValues,
    uint32_t pixelCount,
    bool zeroIsValid) {
    std::vector<float> rgbaValues(pixelCount * 4u, 0.0f);

    for (uint32_t pixelIndex = 0; pixelIndex < pixelCount; ++pixelIndex) {
        float value = scalarValues[pixelIndex];
        const bool finite = std::isfinite(value);
        if (!finite) {
            value = 0.0f;
        }

        const bool valid = finite && (zeroIsValid || value > 0.0f);
        rgbaValues[4u * pixelIndex + 0u] = value;
        rgbaValues[4u * pixelIndex + 1u] = value;
        rgbaValues[4u * pixelIndex + 2u] = value;
        rgbaValues[4u * pixelIndex + 3u] = valid ? 1.0f : 0.0f;
    }

    return rgbaValues;
}

static void saveScalarFloatBufferAsEXR(
    Pale::DeviceSelector &deviceSelector,
    float *deviceBuffer,
    uint32_t imageWidth,
    uint32_t imageHeight,
    const std::filesystem::path &path,
    bool zeroIsValid) {
    const uint32_t pixelCount = imageWidth * imageHeight;

    std::vector<float> scalarValues =
            Pale::downloadFloatBuffer(
                deviceSelector.getQueue(),
                deviceBuffer,
                pixelCount);

    std::vector<float> rgbaValues =
            packScalarToRGBA(scalarValues, pixelCount, zeroIsValid);

    Pale::Utils::saveRGBAFloatAsEXR(
        path,
        rgbaValues,
        imageWidth,
        imageHeight);
}

static void saveCameraAuxiliaryBuffers(
    Pale::DeviceSelector &deviceSelector,
    const Pale::SensorGPU &sensor,
    const std::filesystem::path &baseDir,
    const std::string &fileName) {
    const uint32_t imageWidth = sensor.camera.width;
    const uint32_t imageHeight = sensor.camera.height;
    const uint32_t pixelCount = imageWidth * imageHeight;

    const std::filesystem::path imageDir = baseDir / "images";
    std::filesystem::create_directories(imageDir);

    saveScalarFloatBufferAsEXR(
        deviceSelector,
        sensor.medianDepthBuffer,
        imageWidth,
        imageHeight,
        imageDir / (fileName + "_median_depth.exr"),
        false);

    saveScalarFloatBufferAsEXR(
        deviceSelector,
        sensor.meanDepthBuffer,
        imageWidth,
        imageHeight,
        imageDir / (fileName + "_mean_depth.exr"),
        false);

    saveScalarFloatBufferAsEXR(
        deviceSelector,
        sensor.depthDistortionBuffer,
        imageWidth,
        imageHeight,
        imageDir / (fileName + "_depth_distortion.exr"),
        true);

    saveScalarFloatBufferAsEXR(
        deviceSelector,
        sensor.visibilityWeightedOpacityBuffer,
        imageWidth,
        imageHeight,
        imageDir / (fileName + "_visibility_weighted_opacity.exr"),
        true); {
        std::vector<float> medianWorldPositionRGBA =
                Pale::downloadFloat4Buffer(
                    deviceSelector.getQueue(),
                    sensor.medianWorldPositionBuffer,
                    pixelCount);

        Pale::Utils::saveRGBAFloatAsEXR(
            imageDir / (fileName + "_median_world_position.exr"),
            medianWorldPositionRGBA,
            imageWidth,
            imageHeight);
    } {
        std::vector<float> visibleNormalRGBA =
                Pale::downloadFloat4Buffer(
                    deviceSelector.getQueue(),
                    sensor.visibleNormalBuffer,
                    pixelCount);

        Pale::Utils::saveRGBAFloatAsEXR(
            imageDir / (fileName + "_visible_normal.exr"),
            visibleNormalRGBA,
            imageWidth,
            imageHeight);
    } {
        std::vector<float> normalFromDepthRGBA =
                Pale::downloadFloat4Buffer(
                    deviceSelector.getQueue(),
                    sensor.normalFromDepthBuffer,
                    pixelCount);

        Pale::Utils::saveRGBAFloatAsEXR(
            imageDir / (fileName + "_normal_from_depth.exr"),
            normalFromDepthRGBA,
            imageWidth,
            imageHeight);
    }
}

static void saveForwardImagesAndAuxiliaryBuffers(
    Pale::DeviceSelector &deviceSelector,
    const std::vector<Pale::SensorGPU> &sensors,
    const std::filesystem::path &baseDir,
    const std::string &fileNameSuffix = "") {
    const std::filesystem::path imageDir = baseDir / "images";
    std::filesystem::create_directories(imageDir);

    for (const auto &sensor: sensors) {
        const uint32_t imageWidth = sensor.width;
        const uint32_t imageHeight = sensor.height;
        const std::string fileName = std::string(sensor.name) + fileNameSuffix;

        std::vector<uint8_t> rgba =
                Pale::downloadSensorRGBA(deviceSelector.getQueue(), sensor);

        std::vector<float> rgbaRaw =
                Pale::downloadSensorRGBARAW(deviceSelector.getQueue(), sensor);

        const std::filesystem::path pngPath =
                imageDir / (fileName + ".png");

        const std::filesystem::path rawPath =
                imageDir / (fileName + "_raw.exr");

        Pale::Utils::savePNG(pngPath, rgba, imageWidth, imageHeight);

        Pale::Log::PA_INFO(
            "Saving raw image to: {}",
            (std::filesystem::current_path() / rawPath).string());

        Pale::Utils::saveRGBAFloatAsEXR(
            rawPath,
            rgbaRaw,
            imageWidth,
            imageHeight);

        saveCameraAuxiliaryBuffers(
            deviceSelector,
            sensor,
            baseDir,
            fileName);
    }
}

static void uploadSurfaceRegularizerAdjoints(
    Pale::DeviceSelector &deviceSelector,
    Pale::SensorGPU &sensor,
    float depthDistortionWeight,
    float normalConsistencyWeight) {
    auto queue = deviceSelector.getQueue();

    const uint32_t imageWidth = sensor.width;
    const uint32_t imageHeight = sensor.height;
    const uint32_t pixelCount = imageWidth * imageHeight;

    queue.fill(sensor.depthDistortionAdjointBuffer, 0.0f, pixelCount).wait();
    queue.fill(sensor.visibleNormalAdjointBuffer, Pale::float4{0.0f}, pixelCount).wait();
    queue.fill(sensor.normalFromDepthAdjointBuffer, Pale::float4{0.0f}, pixelCount).wait();
    queue.fill(sensor.medianDepthAdjointBuffer, 0.0f, pixelCount).wait();

    if (isNonzeroWeight(depthDistortionWeight)) {
        const float depthDistortionAdjoint =
                depthDistortionWeight / static_cast<float>(std::max(pixelCount, 1u));

        std::vector<float> depthDistortionAdjointHost(pixelCount, depthDistortionAdjoint);

        Pale::uploadFloatImage(
            queue,
            sensor.depthDistortionAdjointBuffer,
            depthDistortionAdjointHost);
    }

    if (!isNonzeroWeight(normalConsistencyWeight)) {
        return;
    }

    std::vector<float> visibleNormalHost =
            Pale::downloadFloat4Buffer(
                queue,
                sensor.visibleNormalBuffer,
                pixelCount);

    std::vector<float> normalFromDepthHost =
            Pale::downloadFloat4Buffer(
                queue,
                sensor.normalFromDepthBuffer,
                pixelCount);

    uint32_t validCount = 0u;

    for (uint32_t pixelIndex = 0; pixelIndex < pixelCount; ++pixelIndex) {
        const uint32_t baseIndex = 4u * pixelIndex;

        const bool visibleValid = visibleNormalHost[baseIndex + 3u] > 0.0f;
        const bool depthValid = normalFromDepthHost[baseIndex + 3u] > 0.0f;

        const bool visibleFinite =
                std::isfinite(visibleNormalHost[baseIndex + 0u]) &&
                std::isfinite(visibleNormalHost[baseIndex + 1u]) &&
                std::isfinite(visibleNormalHost[baseIndex + 2u]);

        const bool depthFinite =
                std::isfinite(normalFromDepthHost[baseIndex + 0u]) &&
                std::isfinite(normalFromDepthHost[baseIndex + 1u]) &&
                std::isfinite(normalFromDepthHost[baseIndex + 2u]);

        if (visibleValid && depthValid && visibleFinite && depthFinite) {
            ++validCount;
        }
    }

    const float normalAdjointScale =
            normalConsistencyWeight / static_cast<float>(std::max(validCount, 1u));

    std::vector<float> visibleNormalAdjointHost(pixelCount * 4u, 0.0f);
    std::vector<float> normalFromDepthAdjointHost(pixelCount * 4u, 0.0f);

    for (uint32_t pixelIndex = 0; pixelIndex < pixelCount; ++pixelIndex) {
        const uint32_t baseIndex = 4u * pixelIndex;

        const bool visibleValid = visibleNormalHost[baseIndex + 3u] > 0.0f;
        const bool depthValid = normalFromDepthHost[baseIndex + 3u] > 0.0f;

        const bool visibleFinite =
                std::isfinite(visibleNormalHost[baseIndex + 0u]) &&
                std::isfinite(visibleNormalHost[baseIndex + 1u]) &&
                std::isfinite(visibleNormalHost[baseIndex + 2u]);

        const bool depthFinite =
                std::isfinite(normalFromDepthHost[baseIndex + 0u]) &&
                std::isfinite(normalFromDepthHost[baseIndex + 1u]) &&
                std::isfinite(normalFromDepthHost[baseIndex + 2u]);

        if (!(visibleValid && depthValid && visibleFinite && depthFinite)) {
            continue;
        }

        const float visibleNormalX = visibleNormalHost[baseIndex + 0u];
        const float visibleNormalY = visibleNormalHost[baseIndex + 1u];
        const float visibleNormalZ = visibleNormalHost[baseIndex + 2u];

        const float depthNormalX = normalFromDepthHost[baseIndex + 0u];
        const float depthNormalY = normalFromDepthHost[baseIndex + 1u];
        const float depthNormalZ = normalFromDepthHost[baseIndex + 2u];

        visibleNormalAdjointHost[baseIndex + 0u] = -normalAdjointScale * depthNormalX;
        visibleNormalAdjointHost[baseIndex + 1u] = -normalAdjointScale * depthNormalY;
        visibleNormalAdjointHost[baseIndex + 2u] = -normalAdjointScale * depthNormalZ;
        visibleNormalAdjointHost[baseIndex + 3u] = 0.0f;

        normalFromDepthAdjointHost[baseIndex + 0u] = -normalAdjointScale * visibleNormalX;
        normalFromDepthAdjointHost[baseIndex + 1u] = -normalAdjointScale * visibleNormalY;
        normalFromDepthAdjointHost[baseIndex + 2u] = -normalAdjointScale * visibleNormalZ;
        normalFromDepthAdjointHost[baseIndex + 3u] = 0.0f;
    }

    queue.memcpy(
        sensor.visibleNormalAdjointBuffer,
        visibleNormalAdjointHost.data(),
        pixelCount * 4u * sizeof(float)).wait();

    queue.memcpy(
        sensor.normalFromDepthAdjointBuffer,
        normalFromDepthAdjointHost.data(),
        pixelCount * 4u * sizeof(float)).wait();
}

static std::vector<Pale::SensorGPU> makeAdjointSensorSubset(
    const std::vector<Pale::SensorGPU> &allSensors) {
    std::vector<Pale::SensorGPU> adjointSensors;
    for (const Pale::SensorGPU &sensor : allSensors) {
        if (sensor.camera.useForAdjointPass) {
            adjointSensors.push_back(sensor);
        }
    }
    return adjointSensors;
}

static std::vector<Pale::DebugImages> makeDebugImageSubsetForSensors(
    const std::vector<Pale::SensorGPU> &allSensors,
    const std::vector<Pale::DebugImages> &debugImages) {
    std::vector<Pale::DebugImages> selectedDebugImages;
    for (std::size_t sensorIndex = 0; sensorIndex < allSensors.size(); ++sensorIndex) {
        if (allSensors[sensorIndex].camera.useForAdjointPass) {
            selectedDebugImages.push_back(debugImages[sensorIndex]);
        }
    }
    return selectedDebugImages;
}

static void saveGradientSet(
    const std::vector<float> &rgbaBuffer,
    const std::filesystem::path &cameraDebugDir,
    const std::string &baseName,
    uint32_t imageWidth,
    uint32_t imageHeight,
    float adjointSamplesPerPixel) {
    if (rgbaBuffer.empty()) {
        return;
    }

    const std::filesystem::path exrPath = cameraDebugDir / (baseName + ".exr");
    const std::filesystem::path pngPath = cameraDebugDir / (baseName + "_seismic.png");
    const std::filesystem::path pngQ99Path = cameraDebugDir / (baseName + "_seismic_q099.png");

    Pale::Utils::saveRGBAFloatAsEXR(
        exrPath,
        rgbaBuffer,
        imageWidth,
        imageHeight);

    if (Pale::Utils::saveGradientSignPNG(
        pngPath,
        rgbaBuffer,
        imageWidth,
        imageHeight,
        adjointSamplesPerPixel,
        1.0f,
        false,
        true)) {
        Pale::Log::PA_INFO("Wrote PNG image to: {}", pngPath.string());
    }

    Pale::Utils::saveGradientSignPNG(
        pngQ99Path,
        rgbaBuffer,
        imageWidth,
        imageHeight,
        adjointSamplesPerPixel,
        0.99f,
        false,
        true);
}

static void saveDebugGradientImagesForSensors(
    Pale::DeviceSelector &deviceSelector,
    const std::vector<Pale::SensorGPU> &sensors,
    const std::vector<Pale::DebugImages> &debugImages,
    const std::filesystem::path &outputRoot,
    const std::string &prefix,
    float adjointSamplesPerPixel) {
    for (std::size_t sensorIndex = 0; sensorIndex < sensors.size(); ++sensorIndex) {
        const Pale::SensorGPU &sensor = sensors[sensorIndex];
        const std::string sensorName = sensor.name;
        const uint32_t imageWidth = sensor.width;
        const uint32_t imageHeight = sensor.height;

        const std::filesystem::path cameraDebugDir = outputRoot / sensorName;
        std::filesystem::create_directories(cameraDebugDir);

        Pale::DebugGradientImagesHost debugImagesHost =
                Pale::downloadDebugGradientImages(
                    deviceSelector.getQueue(),
                    sensor,
                    debugImages[sensorIndex]);

        saveGradientSet(
            debugImagesHost.positionX,
            cameraDebugDir,
            prefix + "_position_x",
            imageWidth,
            imageHeight,
            adjointSamplesPerPixel);

        saveGradientSet(
            debugImagesHost.positionY,
            cameraDebugDir,
            prefix + "_position_y",
            imageWidth,
            imageHeight,
            adjointSamplesPerPixel);

        saveGradientSet(
            debugImagesHost.positionZ,
            cameraDebugDir,
            prefix + "_position_z",
            imageWidth,
            imageHeight,
            adjointSamplesPerPixel);


        /*
        saveGradientSet(
            debugImagesHost.rotation,
            cameraDebugDir,
            prefix + "_rotation",
            imageWidth,
            imageHeight,
            adjointSamplesPerPixel);
        */


        saveGradientSet(
            debugImagesHost.scale,
            cameraDebugDir,
            prefix + "_scale",
            imageWidth,
            imageHeight,
            adjointSamplesPerPixel);

        saveGradientSet(
            debugImagesHost.albedo,
            cameraDebugDir,
            prefix + "_albedo",
            imageWidth,
            imageHeight,
            adjointSamplesPerPixel);

        saveGradientSet(
            debugImagesHost.opacity,
            cameraDebugDir,
            prefix + "_opacity",
            imageWidth,
            imageHeight,
            adjointSamplesPerPixel);

        saveGradientSet(
            debugImagesHost.beta,
            cameraDebugDir,
            prefix + "_beta",
            imageWidth,
            imageHeight,
            adjointSamplesPerPixel);
    }
}

int main(int argc, char **argv) {
    std::filesystem::path workingDirectory = "../Assets";
    std::filesystem::current_path(workingDirectory);

    Pale::Log::init(spdlog::level::level_enum::debug);

    Pale::AssetManager assetManager{256};
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

    assetManager.registry().load("asset_registry.yaml");

    std::filesystem::path pointCloudPath = "initial.ply";
    std::filesystem::path sceneName = "cbox_custom.xml";

    if (argc > 1) {
        pointCloudPath = argv[1];
    }

    if (argc > 2) {
        sceneName = argv[2];
        sceneName.replace_extension(".xml");
    }

    Pale::Log::PA_INFO("Scene file path: {}", sceneName.string());
    Pale::Log::PA_INFO("Pointcloud file path: {}", pointCloudPath.string());

    std::shared_ptr<Pale::Scene> scene = std::make_shared<Pale::Scene>();
    Pale::AssetIndexFromRegistry assetIndexer(assetManager.registry());
    Pale::SceneSerializer serializer(scene, assetIndexer);
    serializer.deserialize(sceneName);

    const bool addPoints = true;
    const bool addModel = false;

    if (addPoints) {
        auto pointCloudAssetHandle =
                assetIndexer.importPath(pointCloudPath, Pale::AssetType::PointCloud);

        auto pointCloudEntity = scene->createEntity("PointCloud");
        pointCloudEntity.addComponent<Pale::PointCloudComponent>().pointCloudID = pointCloudAssetHandle;

        Pale::AssetAccessFromManager assetAccessor(assetManager);
        const auto pointCloudAsset = assetAccessor.getPointCloud(pointCloudAssetHandle);

        if (!pointCloudAsset || pointCloudAsset->points.empty()) {
            throw std::runtime_error("Point cloud asset failed to load or contains no PointGeometry blocks.");
        }

        const Pale::PointGeometry &pointGeometry = pointCloudAsset->points.front();
        Pale::Log::PA_INFO("Loaded point cloud with {} surfels.", pointGeometry.positions.size());
    }

    if (addModel) {
        Pale::Entity bunnyEntity = scene->createEntity("Model");

        auto &bunnyTransformComponent = bunnyEntity.getComponent<Pale::TransformComponent>();
        bunnyTransformComponent.setPosition(glm::vec3(0.3f, 0.4f, 0.3f));
        bunnyTransformComponent.setRotationEuler(glm::vec3(0.0f, 0.0f, 0.0f));
        bunnyTransformComponent.setScale(glm::vec3(0.7f, 0.7f, 0.7f));

        Pale::AssetHandle bunnyMeshAssetHandle =
                assetIndexer.importPath("meshes/bunny.ply", Pale::AssetType::Mesh);

        auto &bunnyMeshComponent = bunnyEntity.addComponent<Pale::MeshComponent>();
        bunnyMeshComponent.meshID = bunnyMeshAssetHandle;

        Pale::AssetHandle bunnyMaterialAssetHandle =
                assetIndexer.importPath(
                    "Materials/cbox/bsdf_blue_0.mat.yaml",
                    Pale::AssetType::Material);

        auto &bunnyMaterialComponent = bunnyEntity.addComponent<Pale::MaterialComponent>();
        bunnyMaterialComponent.materialID = bunnyMaterialAssetHandle;
    }

    logSceneSummary(scene, assetManager);

    Pale::DeviceSelector deviceSelector;
    Pale::AssetAccessFromManager assetAccessor(assetManager);

    auto options = Pale::SceneBuild::BuildOptions();
    options.bvhMaxLeafPoints = 4;

    auto buildProducts = Pale::SceneBuild::build(scene, assetAccessor, options);
    auto sceneGpu = Pale::SceneUpload::allocateAndUpload(buildProducts, deviceSelector.getQueue());

    const bool renderPhotonMapping = true;

    if (renderPhotonMapping) {
        const float depthDistortionWeight = 500.0f;
        const float normalConsistencyWeight = 0.05f;
        const float visibilityWeightedOpacityWeight = 0.01f;

        Pale::PathTracerSettings settings{};
        settings.integratorKind = Pale::IntegratorKind::photonMapping;
        settings.photonsPerLaunch = 1e6;
        settings.maxBounces = 1;
        settings.maxAdjointBounces = 1;

        settings.numForwardPasses = 1;
        settings.numShadowRays = 1;
        settings.numAdjointShadowRays = 1;
        settings.adjointSamplesPerPixel = 4;

        settings.renderDebugGradientImages = true;
        settings.enableAdjointDirectLight = true;
        settings.surfelIndexForDebugImages = 2;

        // New regularizer structure:
        // 0.0 means disabled. Nonzero means enabled.
        settings.depthDistortionWeight = depthDistortionWeight;
        settings.normalConsistencyWeight = normalConsistencyWeight;
        settings.visibilityWeightedOpacityRegularizerWeight = visibilityWeightedOpacityWeight;

        Pale::PathTracer tracer(deviceSelector.getQueue(), settings);
        tracer.setScene(sceneGpu, buildProducts);

        const std::filesystem::path outputRoot =
                std::filesystem::path("Output") / sceneName.parent_path();

        Pale::Log::PA_INFO("Forward Render Pass...");
        std::vector<Pale::SensorGPU> sensors =
                Pale::makeSensorsForScene(deviceSelector.getQueue(), buildProducts);


        tracer.renderForward(sensors);
        saveForwardImagesAndAuxiliaryBuffers(deviceSelector, sensors, outputRoot);


        {
            auto entities = scene->getAllEntitiesWith<Pale::PointCloudComponent>();
            if (entities.empty()) {
                throw std::runtime_error("debug gradients: scene has no PointCloudComponent");
            }

            Pale::Entity entity(entities.front(), scene.get());

            auto pointAssetSharedPtr =
                    assetManager.get<Pale::PointAsset>(
                        entity.getComponent<Pale::PointCloudComponent>().pointCloudID);

            if (!pointAssetSharedPtr) {
                throw std::runtime_error("debug gradients: failed to get PointAsset for dynamic point cloud");
            }

            Pale::PointAsset &pointAsset = *pointAssetSharedPtr;
            if (pointAsset.points.empty()) {
                throw std::runtime_error("debug gradients: PointAsset has no PointGeometry blocks");
            }

            Pale::PointGeometry &pointGeometry = pointAsset.points.front();
            const uint32_t debugSurfelIndex = settings.surfelIndexForDebugImages;

            if (debugSurfelIndex >= pointGeometry.positions.size()) {
                throw std::runtime_error("debug gradients: surfelIndexForDebugImages is out of range");
            }

            pointGeometry.positions[debugSurfelIndex].x = -0.2f;
            pointGeometry.positions[debugSurfelIndex].y = 0.2f;
            pointGeometry.positions[debugSurfelIndex].z = 0.6f;
            rebuild_bvh(&tracer, scene, buildProducts, &assetManager, deviceSelector, sceneGpu);


            Pale::Log::PA_INFO("Forward Render Pass after debug perturbation...");
            sensors = Pale::makeSensorsForScene(deviceSelector.getQueue(), buildProducts);
            tracer.renderForward(sensors);
            saveForwardImagesAndAuxiliaryBuffers(deviceSelector, sensors, outputRoot, "_debug");

            std::vector<Pale::SensorGPU> adjointSensors = makeAdjointSensorSubset(sensors);

            Pale::Log::PA_INFO("Photometric Adjoint Render Pass...");


            std::vector<Pale::DebugImages> photoDebugImages(sensors.size());

            Pale::PointGradients photoGradients =
                    Pale::makeGradientsForScene(
                        deviceSelector.getQueue(),
                        buildProducts,
                        photoDebugImages.data());

            const bool useConstantDebugTarget = true;
            const float constantDebugTargetValue = -100.0f;
            const float adjointSamplesPerPixel =
                    static_cast<float>(tracer.getSettings().adjointSamplesPerPixel);

            std::vector<std::vector<float> > adjointSourceImages(adjointSensors.size());

            auto sensorNameString = [](const Pale::SensorGPU &sensor) {
                return std::string(sensor.name);
            };

            auto findForwardSensorByName =
                    [&](const std::string &sensorName) -> const Pale::SensorGPU * {
                for (const auto &forwardSensor: sensors) {
                    if (sensorNameString(forwardSensor) == sensorName) {
                        return &forwardSensor;
                    }
                }
                return nullptr;
            };

            for (std::size_t sensorIndex = 0; sensorIndex < adjointSensors.size(); ++sensorIndex) {
                Pale::SensorGPU &adjointSensor = adjointSensors[sensorIndex];
                const std::string sensorName = sensorNameString(adjointSensor);

                const Pale::SensorGPU *forwardSensor =
                        findForwardSensorByName(sensorName);

                if (!forwardSensor) {
                    throw std::runtime_error(
                        "debug gradients: no matching forward sensor for adjoint sensor " + sensorName);
                }

                const uint32_t imageWidth = adjointSensor.width;
                const uint32_t imageHeight = adjointSensor.height;

                if (forwardSensor->width != imageWidth || forwardSensor->height != imageHeight) {
                    throw std::runtime_error(
                        "debug gradients: forward/adjoint resolution mismatch for sensor " + sensorName);
                }

                std::vector<float> rgbaHostRendered =
                        Pale::downloadSensorRGBARAW(
                            deviceSelector.getQueue(),
                            *forwardSensor);

                std::vector<float> rgbaHostAdjointTarget;
                uint32_t targetWidth = 0;
                uint32_t targetHeight = 0;

                const std::filesystem::path targetImagePath =
                        outputRoot / "images" / (sensorName + "_raw.exr");

                Pale::Utils::loadEXRAsRGBAFloat(
                    targetImagePath,
                    rgbaHostAdjointTarget,
                    targetWidth,
                    targetHeight);

                if (targetWidth != imageWidth || targetHeight != imageHeight) {
                    throw std::runtime_error(
                        "debug gradients: target resolution mismatch for sensor " + sensorName);
                }

                if (useConstantDebugTarget) {
                    const std::size_t pixelCount =
                            static_cast<std::size_t>(imageWidth) * imageHeight;

                    for (std::size_t pixelIndex = 0; pixelIndex < pixelCount; ++pixelIndex) {
                        const std::size_t dstIndex = pixelIndex * 4ull;
                        rgbaHostAdjointTarget[dstIndex + 0u] = constantDebugTargetValue;
                        rgbaHostAdjointTarget[dstIndex + 1u] = constantDebugTargetValue;
                        rgbaHostAdjointTarget[dstIndex + 2u] = constantDebugTargetValue;
                        rgbaHostAdjointTarget[dstIndex + 3u] = constantDebugTargetValue;
                    }
                }

                std::vector<float> rgbaHostAdjointSource =
                        Pale::Utils::computeL2ImageGradientRGBA(
                            rgbaHostRendered,
                            rgbaHostAdjointTarget,
                            imageWidth,
                            imageHeight);

                Pale::uploadSensorRGBA(
                    deviceSelector.getQueue(),
                    adjointSensor,
                    rgbaHostAdjointSource);

                adjointSourceImages[sensorIndex] = std::move(rgbaHostAdjointSource);
            }

            std::vector<Pale::DebugImages> photoDebugImagesSelected =
                makeDebugImageSubsetForSensors(sensors, photoDebugImages);

            tracer.renderBackward(
                adjointSensors,
                photoGradients,
                photoDebugImagesSelected.data());

            Pale::Log::PA_INFO("Surface Regularizer Backward Pass...");

            std::vector<Pale::DebugImages> surfaceDebugImages(sensors.size());

            Pale::PointGradients surfaceGradients =
                    Pale::makeGradientsForScene(
                        deviceSelector.getQueue(),
                        buildProducts,
                        surfaceDebugImages.data());

            for (Pale::SensorGPU &sensor: adjointSensors) {
                uploadSurfaceRegularizerAdjoints(
                    deviceSelector,
                    sensor,
                    depthDistortionWeight,
                    normalConsistencyWeight);
            }

            std::vector<Pale::DebugImages> surfaceDebugImagesSelected =
                makeDebugImageSubsetForSensors(sensors, surfaceDebugImages);

            tracer.renderSurfaceRegularizersBackward(
                adjointSensors,
                surfaceGradients,
                surfaceDebugImagesSelected.data());

            if (debugSurfelIndex < photoGradients.numPoints) {
                float photoGradientBeta{};
                float photoGradientOpacity{};
                Pale::float3 photoGradientPosition{};

                deviceSelector.getQueue().memcpy(
                    &photoGradientBeta,
                    &photoGradients.gradBeta[debugSurfelIndex],
                    sizeof(float)).wait();

                deviceSelector.getQueue().memcpy(
                    &photoGradientOpacity,
                    &photoGradients.gradOpacity[debugSurfelIndex],
                    sizeof(float)).wait();

                deviceSelector.getQueue().memcpy(
                    &photoGradientPosition,
                    &photoGradients.gradPosition[debugSurfelIndex],
                    sizeof(Pale::float3)).wait();

                Pale::Log::PA_INFO("photo debug surfel index = {}", debugSurfelIndex);
                Pale::Log::PA_INFO("photo grad Beta = ({})", photoGradientBeta);
                Pale::Log::PA_INFO("photo grad Opacity = ({})", photoGradientOpacity);
                Pale::Log::PA_INFO(
                    "photo grad Position = ({}, {}, {})",
                    photoGradientPosition.x(),
                    photoGradientPosition.y(),
                    photoGradientPosition.z());
            }

            if (debugSurfelIndex < surfaceGradients.numPoints) {
                float surfaceGradientBeta{};
                float surfaceGradientOpacity{};
                Pale::float3 surfaceGradientPosition{};

                deviceSelector.getQueue().memcpy(
                    &surfaceGradientBeta,
                    &surfaceGradients.gradBeta[debugSurfelIndex],
                    sizeof(float)).wait();

                deviceSelector.getQueue().memcpy(
                    &surfaceGradientOpacity,
                    &surfaceGradients.gradOpacity[debugSurfelIndex],
                    sizeof(float)).wait();

                deviceSelector.getQueue().memcpy(
                    &surfaceGradientPosition,
                    &surfaceGradients.gradPosition[debugSurfelIndex],
                    sizeof(Pale::float3)).wait();

                Pale::Log::PA_INFO("surface regularizer debug surfel index = {}", debugSurfelIndex);
                Pale::Log::PA_INFO("surface grad Beta = ({})", surfaceGradientBeta);
                Pale::Log::PA_INFO("surface grad Opacity = ({})", surfaceGradientOpacity);
                Pale::Log::PA_INFO(
                    "surface grad Position = ({}, {}, {})",
                    surfaceGradientPosition.x(),
                    surfaceGradientPosition.y(),
                    surfaceGradientPosition.z());
            }

            for (std::size_t sensorIndex = 0; sensorIndex < adjointSensors.size(); ++sensorIndex) {
                const Pale::SensorGPU &adjointSensor = adjointSensors[sensorIndex];
                const std::string sensorName = adjointSensor.name;
                const uint32_t imageWidth = adjointSensor.width;
                const uint32_t imageHeight = adjointSensor.height;

                const std::filesystem::path cameraDebugDir = outputRoot / sensorName;
                std::filesystem::create_directories(cameraDebugDir);

                saveGradientSet(
                    adjointSourceImages[sensorIndex],
                    cameraDebugDir,
                    "photo_adjoint_source_l2_gradient",
                    imageWidth,
                    imageHeight,
                    adjointSamplesPerPixel);
            }

            if (settings.renderDebugGradientImages) {
                saveDebugGradientImagesForSensors(
                    deviceSelector,
                    adjointSensors,
                    photoDebugImagesSelected,
                    outputRoot,
                    "photo",
                    adjointSamplesPerPixel);

                saveDebugGradientImagesForSensors(
                    deviceSelector,
                    adjointSensors,
                    surfaceDebugImagesSelected,
                    outputRoot,
                    "surface",
                    adjointSamplesPerPixel);
            }
        }
    }

    assetManager.registry().save("asset_registry.yaml");
    deviceSelector.getQueue().wait();
    return 0;
}
