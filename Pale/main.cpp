// Main.cpp
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
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
        true);

    {
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
    }

    {
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
    }

    {
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
    for (const Pale::SensorGPU &sensor: allSensors) {
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

static void savePointGradientsAsCsv(
    Pale::DeviceSelector &deviceSelector,
    const Pale::PointGradients &gradients,
    const std::filesystem::path &path) {
    auto queue = deviceSelector.getQueue();
    const std::size_t pointCount = gradients.numPoints;

    std::vector<Pale::float3> gradPositionHost(pointCount);
    std::vector<Pale::float3> gradTangentUHost(pointCount);
    std::vector<Pale::float3> gradTangentVHost(pointCount);
    std::vector<Pale::float2> gradScaleHost(pointCount);
    std::vector<Pale::float3> gradAlbedoHost(pointCount);
    std::vector<float> gradOpacityHost(pointCount, 0.0f);
    std::vector<float> gradBetaHost(pointCount, 0.0f);
    std::vector<float> gradShapeHost(pointCount, 0.0f);

    if (pointCount > 0) {
        if (gradients.gradPosition) {
            queue.memcpy(gradPositionHost.data(), gradients.gradPosition, pointCount * sizeof(Pale::float3));
        }
        if (gradients.gradRotation) {
            queue.memcpy(gradTangentUHost.data(), gradients.gradRotation, pointCount * sizeof(Pale::float3));
        }
        if (gradients.gradScale) {
            queue.memcpy(gradScaleHost.data(), gradients.gradScale, pointCount * sizeof(Pale::float2));
        }
        if (gradients.gradAlbedo) {
            queue.memcpy(gradAlbedoHost.data(), gradients.gradAlbedo, pointCount * sizeof(Pale::float3));
        }
        if (gradients.gradOpacity) {
            queue.memcpy(gradOpacityHost.data(), gradients.gradOpacity, pointCount * sizeof(float));
        }
        if (gradients.gradBeta) {
            queue.memcpy(gradBetaHost.data(), gradients.gradBeta, pointCount * sizeof(float));
        }
        if (gradients.gradShape) {
            queue.memcpy(gradShapeHost.data(), gradients.gradShape, pointCount * sizeof(float));
        }
        queue.wait();
    }

    std::filesystem::create_directories(path.parent_path());

    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open gradient CSV for writing: " + path.string());
    }

    file << "index,"
         << "grad_position_x,grad_position_y,grad_position_z,"
         << "grad_tangent_u_x,grad_tangent_u_y,grad_tangent_u_z,"
         << "grad_tangent_v_x,grad_tangent_v_y,grad_tangent_v_z,"
         << "grad_scale_u,grad_scale_v,"
         << "grad_albedo_r,grad_albedo_g,grad_albedo_b,"
         << "grad_opacity,grad_beta,grad_shape\n";

    for (std::size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
        const Pale::float3 &gradPosition = gradPositionHost[pointIndex];
        const Pale::float3 &gradTangentU = gradTangentUHost[pointIndex];
        const Pale::float3 &gradTangentV = gradTangentVHost[pointIndex];
        const Pale::float2 &gradScale = gradScaleHost[pointIndex];
        const Pale::float3 &gradAlbedo = gradAlbedoHost[pointIndex];

        file << pointIndex << ","
             << gradPosition.x() << "," << gradPosition.y() << "," << gradPosition.z() << ","
             << gradTangentU.x() << "," << gradTangentU.y() << "," << gradTangentU.z() << ","
             << gradTangentV.x() << "," << gradTangentV.y() << "," << gradTangentV.z() << ","
             << gradScale.x() << "," << gradScale.y() << ","
             << gradAlbedo.x() << "," << gradAlbedo.y() << "," << gradAlbedo.z() << ","
             << gradOpacityHost[pointIndex] << ","
             << gradBetaHost[pointIndex] << ","
             << gradShapeHost[pointIndex] << "\n";
    }

    Pale::Log::PA_INFO("Saved point gradients: {}", path.string());
}

static void logSinglePointGradient(
    Pale::DeviceSelector &deviceSelector,
    const Pale::PointGradients &gradients,
    uint32_t pointIndex,
    const std::string &label) {
    if (pointIndex >= gradients.numPoints) {
        return;
    }

    float gradientBeta{};
    float gradientOpacity{};
    Pale::float3 gradientPosition{};

    if (gradients.gradBeta) {
        deviceSelector.getQueue().memcpy(
            &gradientBeta,
            &gradients.gradBeta[pointIndex],
            sizeof(float)).wait();
    }

    if (gradients.gradOpacity) {
        deviceSelector.getQueue().memcpy(
            &gradientOpacity,
            &gradients.gradOpacity[pointIndex],
            sizeof(float)).wait();
    }

    if (gradients.gradPosition) {
        deviceSelector.getQueue().memcpy(
            &gradientPosition,
            &gradients.gradPosition[pointIndex],
            sizeof(Pale::float3)).wait();
    }

    Pale::Log::PA_INFO("{} debug surfel index = {}", label, pointIndex);
    Pale::Log::PA_INFO("{} grad Beta = ({})", label, gradientBeta);
    Pale::Log::PA_INFO("{} grad Opacity = ({})", label, gradientOpacity);
    Pale::Log::PA_INFO(
        "{} grad Position = ({}, {}, {})",
        label,
        gradientPosition.x(),
        gradientPosition.y(),
        gradientPosition.z());
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

        saveGradientSet(
            debugImagesHost.scaleU,
            cameraDebugDir,
            prefix + "_scaleU",
            imageWidth,
            imageHeight,
            adjointSamplesPerPixel);


        saveGradientSet(
            debugImagesHost.scaleV,
            cameraDebugDir,
            prefix + "_scaleV",
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


static float sanitizeScalar(float value) {
    if (!std::isfinite(value)) {
        return 0.0f;
    }
    return value;
}

static std::vector<float> robustNormalizeScalars(
    const std::vector<float> &values,
    float lowerPercentile = 1.0f,
    float upperPercentile = 99.0f) {
    std::vector<float> finiteValues;
    finiteValues.reserve(values.size());

    for (float value: values) {
        if (std::isfinite(value)) {
            finiteValues.push_back(value);
        }
    }

    std::vector<float> normalized(values.size(), 0.0f);

    if (finiteValues.empty()) {
        return normalized;
    }

    std::sort(finiteValues.begin(), finiteValues.end());

    const auto percentileValue = [&](float percentile) -> float {
        const float clampedPercentile = std::clamp(percentile, 0.0f, 100.0f);
        const float position =
            (clampedPercentile / 100.0f) * static_cast<float>(finiteValues.size() - 1u);

        const std::size_t lowerIndex = static_cast<std::size_t>(std::floor(position));
        const std::size_t upperIndex = std::min(lowerIndex + 1u, finiteValues.size() - 1u);
        const float fraction = position - static_cast<float>(lowerIndex);

        return finiteValues[lowerIndex] * (1.0f - fraction) + finiteValues[upperIndex] * fraction;
    };

    const float valueMin = percentileValue(lowerPercentile);
    const float valueMax = percentileValue(upperPercentile);
    const float denominator = valueMax - valueMin;

    if (!(denominator > 1.0e-12f)) {
        return normalized;
    }

    for (std::size_t index = 0; index < values.size(); ++index) {
        const float value = sanitizeScalar(values[index]);
        normalized[index] = std::clamp((value - valueMin) / denominator, 0.0f, 1.0f);
    }

    return normalized;
}

static std::vector<float> normalizeByMaximumValue(
    const std::vector<float> &values,
    float maximumValue) {
    std::vector<float> normalized(values.size(), 0.0f);

    const float safeMaximumValue = std::max(maximumValue, 1.0f);
    for (std::size_t index = 0; index < values.size(); ++index) {
        normalized[index] = std::clamp(sanitizeScalar(values[index]) / safeMaximumValue, 0.0f, 1.0f);
    }

    return normalized;
}

static std::array<uint8_t, 3> blackRedYellowWhiteColor(float normalizedValue) {
    const float t = std::clamp(normalizedValue, 0.0f, 1.0f);

    const float red = std::clamp(3.0f * t, 0.0f, 1.0f);
    const float green = std::clamp(3.0f * t - 1.0f, 0.0f, 1.0f);
    const float blue = std::clamp(3.0f * t - 2.0f, 0.0f, 1.0f);

    return {
        static_cast<uint8_t>(std::clamp(red * 255.0f, 0.0f, 255.0f)),
        static_cast<uint8_t>(std::clamp(green * 255.0f, 0.0f, 255.0f)),
        static_cast<uint8_t>(std::clamp(blue * 255.0f, 0.0f, 255.0f)),
    };
}

static void saveColoredGradientStatsPly(
    const Pale::PointGeometry &pointGeometry,
    const std::vector<float> &scalarValues,
    const std::vector<float> &normalizedValues,
    const std::filesystem::path &path,
    const std::string &scalarName) {
    const std::size_t pointCount =
        std::min(pointGeometry.positions.size(), scalarValues.size());

    if (pointCount == 0u) {
        Pale::Log::PA_WARN("saveColoredGradientStatsPly: no points for {}", path.string());
        return;
    }

    std::filesystem::create_directories(path.parent_path());

    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open gradient-stat PLY for writing: " + path.string());
    }

    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << pointCount << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "property float " << scalarName << "\n";
    file << "end_header\n";

    for (std::size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
        const auto &position = pointGeometry.positions[pointIndex];
        const auto color = blackRedYellowWhiteColor(normalizedValues[pointIndex]);
        const float scalarValue = sanitizeScalar(scalarValues[pointIndex]);

        file << position.x << " "
             << position.y << " "
             << position.z << " "
             << static_cast<int>(color[0]) << " "
             << static_cast<int>(color[1]) << " "
             << static_cast<int>(color[2]) << " "
             << scalarValue << "\n";
    }

    Pale::Log::PA_INFO("Saved gradient-stat PLY: {}", path.string());
}

static void savePointGradientStatsAsColoredPlys(
    Pale::DeviceSelector &deviceSelector,
    const Pale::PointGradients &gradients,
    const Pale::PointGeometry &pointGeometry,
    const std::filesystem::path &outputDir,
    uint32_t activeCameraCountMax) {
    auto queue = deviceSelector.getQueue();
    const std::size_t pointCount = gradients.numPoints;

    if (pointCount == 0u) {
        Pale::Log::PA_WARN("savePointGradientStatsAsColoredPlys: no gradients to save");
        return;
    }

    if (!gradients.gradPosition ||
        !gradients.gradPositionMeanNorm ||
        !gradients.gradPositionStd ||
        !gradients.gradPositionActiveCameraCount) {
        Pale::Log::PA_WARN("savePointGradientStatsAsColoredPlys: gradient stats buffers are missing");
        return;
    }

    std::vector<Pale::float3> gradPositionHost(pointCount);
    std::vector<float> positionMeanNormHost(pointCount, 0.0f);
    std::vector<float> positionStdHost(pointCount, 0.0f);
    std::vector<uint32_t> activeCameraCountHost(pointCount, 0u);

    queue.memcpy(
        gradPositionHost.data(),
        gradients.gradPosition,
        pointCount * sizeof(Pale::float3));

    queue.memcpy(
        positionMeanNormHost.data(),
        gradients.gradPositionMeanNorm,
        pointCount * sizeof(float));

    queue.memcpy(
        positionStdHost.data(),
        gradients.gradPositionStd,
        pointCount * sizeof(float));

    queue.memcpy(
        activeCameraCountHost.data(),
        gradients.gradPositionActiveCameraCount,
        pointCount * sizeof(uint32_t));

    queue.wait();

    std::vector<float> gradientNormHost(pointCount, 0.0f);
    std::vector<float> activeCameraCountFloatHost(pointCount, 0.0f);

    for (std::size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
        const Pale::float3 &gradient = gradPositionHost[pointIndex];

        const float gradientSquaredNorm =
            gradient.x() * gradient.x() +
            gradient.y() * gradient.y() +
            gradient.z() * gradient.z();

        gradientNormHost[pointIndex] = std::sqrt(std::max(gradientSquaredNorm, 0.0f));
        activeCameraCountFloatHost[pointIndex] = static_cast<float>(activeCameraCountHost[pointIndex]);
    }

    std::filesystem::create_directories(outputDir);

    saveColoredGradientStatsPly(
        pointGeometry,
        positionStdHost,
        robustNormalizeScalars(positionStdHost),
        outputDir / "gradient_position_std.ply",
        "gradient_position_std");

    saveColoredGradientStatsPly(
        pointGeometry,
        positionMeanNormHost,
        robustNormalizeScalars(positionMeanNormHost),
        outputDir / "gradient_geometric_pressure.ply",
        "gradient_geometric_pressure");

    saveColoredGradientStatsPly(
        pointGeometry,
        gradientNormHost,
        robustNormalizeScalars(gradientNormHost),
        outputDir / "gradient_position_norm.ply",
        "gradient_position_norm");

    saveColoredGradientStatsPly(
        pointGeometry,
        activeCameraCountFloatHost,
        normalizeByMaximumValue(activeCameraCountFloatHost, static_cast<float>(std::max(activeCameraCountMax, 1u))),
        outputDir / "gradient_active_camera_count.ply",
        "gradient_active_camera_count");
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
        settings.adjointSamplesPerPixel = 64;

        settings.renderDebugGradientImages = true;
        settings.enableAdjointDirectLight = true;
        settings.surfelIndexForDebugImages = 2;

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

            std::vector<std::vector<float>> adjointSourceImages(adjointSensors.size());

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

            if (settings.renderDebugGradientImages) {
                savePointGradientStatsAsColoredPlys(
                    deviceSelector,
                    photoGradients,
                    pointGeometry,
                    outputRoot / "gradient_stats",
                    static_cast<uint32_t>(adjointSensors.size()));
            }

            std::vector<Pale::DebugImages> surfaceDebugImages(sensors.size());

            Pale::PointGradients depthDistortionGradients =
                Pale::makeGradientsForScene(
                    deviceSelector.getQueue(),
                    buildProducts,
                    surfaceDebugImages.data());

            Pale::PointGradients normalConsistencyGradients =
                Pale::makeGradientsForScene(
                    deviceSelector.getQueue(),
                    buildProducts,
                    nullptr);

            Pale::PointGradients visibilityOpacityGradients =
                Pale::makeGradientsForScene(
                    deviceSelector.getQueue(),
                    buildProducts,
                    nullptr);

            for (Pale::SensorGPU &sensor: adjointSensors) {
                uploadSurfaceRegularizerAdjoints(
                    deviceSelector,
                    sensor,
                    settings.depthDistortionWeight,
                    settings.normalConsistencyWeight);
            }

            std::vector<Pale::DebugImages> surfaceDebugImagesSelected =
                makeDebugImageSubsetForSensors(sensors, surfaceDebugImages);

            tracer.renderSurfaceRegularizersBackward(
                adjointSensors,
                depthDistortionGradients,
                normalConsistencyGradients,
                visibilityOpacityGradients,
                surfaceDebugImagesSelected.data());

            /*
            const std::filesystem::path surfaceGradientDir = outputRoot / "surface_gradients";
            savePointGradientsAsCsv(
                deviceSelector,
                depthDistortionGradients,
                surfaceGradientDir / "depth_distortion_gradients.csv");

            savePointGradientsAsCsv(
                deviceSelector,
                normalConsistencyGradients,
                surfaceGradientDir / "normal_consistency_gradients.csv");

            savePointGradientsAsCsv(
                deviceSelector,
                visibilityOpacityGradients,
                surfaceGradientDir / "visibility_weighted_opacity_gradients.csv");
            */

            logSinglePointGradient(
                deviceSelector,
                photoGradients,
                debugSurfelIndex,
                "photo");

            logSinglePointGradient(
                deviceSelector,
                depthDistortionGradients,
                debugSurfelIndex,
                "depth distortion");

            logSinglePointGradient(
                deviceSelector,
                normalConsistencyGradients,
                debugSurfelIndex,
                "normal consistency");

            logSinglePointGradient(
                deviceSelector,
                visibilityOpacityGradients,
                debugSurfelIndex,
                "visibility opacity");

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