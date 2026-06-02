// Main.cpp
#include <memory>
#include <filesystem>
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


static std::string assetPathOrId(const Pale::AssetRegistry &reg, const Pale::AssetHandle &id) {
    if (auto m = reg.meta(id)) return m->path.string();
    return std::string(id); // fallback if it's not in the registry
}

static void logSceneSummary(std::shared_ptr<Pale::Scene> &scene,
                            Pale::AssetManager &am) {
    auto &reg = am.registry();

    Pale::Log::PA_INFO("===== Scene Summary =====");
    size_t entityCount = 0;
    size_t meshCount = 0;
    size_t emissiveCount = 0;

    auto view = scene->getAllEntitiesWith<Pale::IDComponent>();
    for (entt::entity entity: view) {
        Pale::Entity e(entity, scene.get());
        ++entityCount;

        const char *name = e.getName().c_str();
        bool hasMesh = e.hasComponent<Pale::MeshComponent>();
        bool hasMat = e.hasComponent<Pale::MaterialComponent>();
        bool hasEm = e.hasComponent<Pale::AreaLightComponent>();

        Pale::Log::PA_INFO("[Entity] {}", name);

        // Mesh
        if (hasMesh) {
            auto &mc = e.getComponent<Pale::MeshComponent>();
            ++meshCount;
            std::string meshLabel = assetPathOrId(reg, mc.meshID);

            size_t submeshCount = 0;
            if (auto mesh = am.get<Pale::Mesh>(mc.meshID)) {
                submeshCount = mesh->submeshes.size();
            } else {
                Pale::Log::PA_WARN("  Mesh: {} (FAILED to load)", meshLabel);
            }

            Pale::Log::PA_INFO("  Mesh: {}  (submeshes: {})", meshLabel, submeshCount);
        }

        // Material
        if (hasMat) {
            auto &matc = e.getComponent<Pale::MaterialComponent>();
            std::string matLabel = assetPathOrId(reg, matc.materialID);

            if (auto mat = am.get<Pale::Material>(matc.materialID)) {
                Pale::Log::PA_INFO(
                    "  Material: {}  [baseColor=({:.3f},{:.3f},{:.3f}) roughness={:.3f} metallic={:.3f}]",
                    matLabel,
                    mat->baseColor.x, mat->baseColor.y, mat->baseColor.z,
                    mat->roughness, mat->metallic
                );
            } else {
                Pale::Log::PA_INFO("  Material: {}  (pending load)", matLabel);
            }
        }

        // Emissive
        if (hasEm) {
            ++emissiveCount;
            auto &em = e.getComponent<Pale::AreaLightComponent>();
            Pale::Log::PA_INFO("  Emissive radiance=({:.3f},{:.3f},{:.3f})",
                               em.radiance.x, em.radiance.y, em.radiance.z);
        }
    }

    Pale::Log::PA_INFO("===== Totals: entities={} meshes={} emissives={} =====",
                       entityCount, meshCount, emissiveCount);
}


void rebuild_bvh(Pale::PathTracer *pathTracer, std::shared_ptr<Pale::Scene> &scene,
                 Pale::SceneBuild::BuildProducts &buildProducts, Pale::AssetManager *assetManager,
                 Pale::DeviceSelector &deviceSelector, Pale::GPUSceneBuffers &sceneGpu) {
    Pale::AssetAccessFromManager assetAccessor(*assetManager);

    buildProducts = Pale::SceneBuild::build(
        scene,
        assetAccessor,
        Pale::SceneBuild::BuildOptions()
    );


    Pale::SceneUpload::uploadOrReallocate(
        buildProducts,
        sceneGpu,
        deviceSelector.getQueue()
    );

    pathTracer->setScene(sceneGpu, buildProducts);
}


static std::vector<float> packScalarToRGBA(
    const std::vector<float> &src,
    uint32_t pixelCount) {
    std::vector<float> out(pixelCount * 4, 0.0f);
    for (uint32_t i = 0; i < pixelCount; ++i) {
        const float v = src[i];
        const bool valid = std::isfinite(v) && v > 0.0f;

        const float stored = valid ? v : 0.0f;
        out[4 * i + 0] = stored;
        out[4 * i + 1] = stored;
        out[4 * i + 2] = stored;
        out[4 * i + 3] = valid ? 1.0f : 0.0f;
    }
    return out;
}

void saveCameraAuxiliaryBuffers(
    Pale::DeviceSelector &deviceSelector,
    const Pale::SensorGPU &sensor,
    const std::filesystem::path &baseDir,
    const std::string &fileName) {
    const uint32_t imageWidth = sensor.camera.width;
    const uint32_t imageHeight = sensor.camera.height;
    const uint32_t pixelCount = imageWidth * imageHeight;

    const std::filesystem::path imageDir = baseDir / "images";
    std::filesystem::create_directories(imageDir);

    // -------------------------------------------------------------------------
    // 1) Median depth: scalar -> replicate to RGB, alpha = validity
    // -------------------------------------------------------------------------
    {
        std::vector<float> medianDepthRaw =
                Pale::downloadFloatBuffer(
                    deviceSelector.getQueue(),
                    sensor.medianDepthBuffer,
                    pixelCount);

        std::vector<float> medianDepthRGBA =
                packScalarToRGBA(medianDepthRaw, pixelCount);

        const std::filesystem::path path =
                imageDir / (fileName + "_median_depth.exr");

        Pale::Utils::saveRGBAFloatAsEXR(
            path,
            medianDepthRGBA,
            imageWidth,
            imageHeight);

        std::vector<float> meanDepthRaw =
                Pale::downloadFloatBuffer(
                    deviceSelector.getQueue(),
                    sensor.meanDepthBuffer,
                    pixelCount);

        std::vector<float> meanDepthRGBA =
                packScalarToRGBA(meanDepthRaw, pixelCount);

        const std::filesystem::path path_mean =
                imageDir / (fileName + "_mean_depth.exr");

        Pale::Utils::saveRGBAFloatAsEXR(
            path_mean,
            meanDepthRGBA,
            imageWidth,
            imageHeight);
    }

    // -------------------------------------------------------------------------
    // 2) Median world position: RGB = world xyz, A = validity/mask if you stored it
    // -------------------------------------------------------------------------
    {
        std::vector<float> medianWorldPositionRGBA =
                Pale::downloadFloat4Buffer(
                    deviceSelector.getQueue(),
                    sensor.medianWorldPositionBuffer,
                    pixelCount);

        const std::filesystem::path path =
                imageDir / (fileName + "_median_world_position.exr");

        Pale::Utils::saveRGBAFloatAsEXR(
            path,
            medianWorldPositionRGBA,
            imageWidth,
            imageHeight);
    }

    // -------------------------------------------------------------------------
    // 3) Visible normal: RGB = raw normal in [-1,1], A = validity/mask
    // -------------------------------------------------------------------------
    {
        std::vector<float> visibleNormalRGBA =
                Pale::downloadFloat4Buffer(
                    deviceSelector.getQueue(),
                    sensor.visibleNormalBuffer,
                    pixelCount);

        const std::filesystem::path path =
                imageDir / (fileName + "_visible_normal.exr");

        Pale::Utils::saveRGBAFloatAsEXR(
            path,
            visibleNormalRGBA,
            imageWidth,
            imageHeight);
    }

    // -------------------------------------------------------------------------
    // 4) Normal from median depth: RGB = raw normal in [-1,1], A = validity/mask
    // -------------------------------------------------------------------------
    {
        std::vector<float> normalFromDepthRGBA =
                Pale::downloadFloat4Buffer(
                    deviceSelector.getQueue(),
                    sensor.normalFromDepthBuffer,
                    pixelCount);

        const std::filesystem::path path =
                imageDir / (fileName + "_normal_from_depth.exr");

        Pale::Utils::saveRGBAFloatAsEXR(
            path,
            normalFromDepthRGBA,
            imageWidth,
            imageHeight);
    }
}


int main(int argc, char **argv) {
    std::filesystem::path workingDirectory = "../Assets";
    std::filesystem::current_path(workingDirectory);

    Pale::Log::init(spdlog::level::level_enum::debug);

    Pale::AssetManager assetManager{256};
    assetManager.enableHotReload(true);
    assetManager.registerLoader<Pale::Mesh>(Pale::AssetType::Mesh,
                                            std::make_shared<Pale::AssimpMeshLoader>());
    assetManager.registerLoader<Pale::Material>(Pale::AssetType::Material,
                                                std::make_shared<Pale::YamlMaterialLoader>());

    assetManager.registerLoader<Pale::PointAsset>(Pale::AssetType::PointCloud,
                                                  std::make_shared<Pale::PLYPointLoader>());

    assetManager.registry().load("asset_registry.yaml");

    std::filesystem::path sceneName;
    std::filesystem::path pointCloudPath;
    pointCloudPath = "initial.ply"; // default
    sceneName = "cbox_custom.xml";
    if (argc > 1) {
        pointCloudPath = argv[1];
    }
    if (argc > 2) {
        sceneName = argv[2];
        sceneName.replace_extension(".xml");
    }
    Pale::Log::PA_INFO("Scene file path: {}", sceneName.string());
    Pale::Log::PA_INFO("Pointcloud file path: {}", pointCloudPath.string());
    // Load in xml file and Create Scene from xml
    std::shared_ptr<Pale::Scene> scene = std::make_shared<Pale::Scene>();
    Pale::AssetIndexFromRegistry assetIndexer(assetManager.registry());
    Pale::SceneSerializer serializer(scene, assetIndexer);
    //serializer.deserialize("scene_blender_30.xml");
    //serializer.deserialize("scene_blender_1.xml");
    //serializer.deserialize("scene.xml");
    //serializer.deserialize("scene_blender_debug.xml");
    serializer.deserialize(sceneName);
    //serializer.deserialize("empty.xml");

    // Add Single Gaussian
    // Check CLI input for point cloud file


    bool addPoints = true;
    bool addModel = !true;
    if (addPoints) {
        auto assetHandle = assetIndexer.importPath(pointCloudPath, Pale::AssetType::PointCloud);
        auto entityPointCloud = scene->createEntity("PointCloud");
        entityPointCloud.addComponent<Pale::PointCloudComponent>().pointCloudID = assetHandle;
        auto &transform = entityPointCloud.getComponent<Pale::PointCloudComponent>();

        Pale::AssetAccessFromManager assetAccessor(assetManager);

        const auto pointCloudAsset = assetAccessor.getPointCloud(assetHandle);
        // assuming one geometry block per asset for now
        const Pale::PointGeometry &pointGeometry = pointCloudAsset->points.front();
    }

    if (addModel) {
        Pale::Entity bunnyEntity = scene->createEntity("Model");
        // 1) Transform
        auto &bunnyTransformComponent = bunnyEntity.getComponent<Pale::TransformComponent>();
        bunnyTransformComponent.setPosition(glm::vec3(0.3f, 0.4f, 0.3f));
        bunnyTransformComponent.setRotationEuler(glm::vec3(0.0f, 0.0f, 0.0f));
        bunnyTransformComponent.setScale(glm::vec3(0.7f, 0.7f, 0.7f));

        // 2) Mesh
        Pale::AssetHandle bunnyMeshAssetHandle =
                assetIndexer.importPath("meshes/bunny.ply", Pale::AssetType::Mesh);

        auto &bunnyMeshComponent = bunnyEntity.addComponent<Pale::MeshComponent>();
        bunnyMeshComponent.meshID = bunnyMeshAssetHandle;

        // 3) Material
        Pale::AssetHandle bunnyMaterialAssetHandle =
                assetIndexer.importPath("Materials/cbox/bsdf_blue_0.mat.yaml",
                                        Pale::AssetType::Material);

        auto &bunnyMaterialComponent = bunnyEntity.addComponent<Pale::MaterialComponent>();
        bunnyMaterialComponent.materialID = bunnyMaterialAssetHandle;
    }

    logSceneSummary(scene, assetManager);

    //FInd Sycl Device
    Pale::DeviceSelector deviceSelector;

    // Build rendering products (BLAS. TLAS, Emissive lists, etc..)
    Pale::AssetAccessFromManager assetAccessor(assetManager);

    auto options = Pale::SceneBuild::BuildOptions();
    options.bvhMaxLeafPoints = 4;
    auto buildProducts = Pale::SceneBuild::build(scene, assetAccessor, options);
    // Upload Scene to GPU
    auto gpu = Pale::SceneUpload::allocateAndUpload(buildProducts, deviceSelector.getQueue()); // scene only

    bool renderPhotonMapping = true;

    if (renderPhotonMapping) {
        Pale::PathTracerSettings settings;
        settings.integratorKind = Pale::IntegratorKind::photonMapping;
        settings.photonsPerLaunch = 1e6;
        settings.maxBounces = 2;
        settings.maxAdjointBounces = 2; // 1 == First surfel intersection gradients, 2 = Second surfel gradients

        settings.numForwardPasses = 10;
        settings.numShadowRays = 1;
        settings.numAdjointShadowRays = 1;
        settings.adjointSamplesPerPixel = 16;

        settings.renderDebugGradientImages = true;
        settings.enableAdjointDirectLight = true;
        settings.useDepthDistortion = true;
        settings.useNormalConsistency = true;
        settings.surfelIndexForDebugImages = 2;

        Pale::PathTracer tracer(deviceSelector.getQueue(), settings);

        tracer.setScene(gpu, buildProducts);

        Pale::Log::PA_INFO("Forward Render Pass...");
        std::vector<Pale::SensorGPU> sensors = Pale::makeSensorsForScene(deviceSelector.getQueue(), buildProducts);
        tracer.renderForward(sensors);

        // Save target image
        for (const auto &sensor: sensors) {
            std::vector<uint8_t> rgba =
                    Pale::downloadSensorRGBA(deviceSelector.getQueue(), sensor);
            const uint32_t imageWidth = sensor.width;
            const uint32_t imageHeight = sensor.height;
            std::vector<float> rgbaRaw = Pale::downloadSensorRGBARAW(deviceSelector.getQueue(), sensor);
            // Per-camera output directory: Output/<pointcloud>/<camera_name>/
            std::filesystem::path baseDir =
                    std::filesystem::path("Output") / sceneName.parent_path(); // assumes sensor.name is std::string
            std::filesystem::create_directories(baseDir);
            std::string fileName = sensor.name;
            //fileName += "_photonmap";
            std::filesystem::path filePath = baseDir / "images" / (fileName + ".png");
            Pale::Utils::savePNG(filePath, rgba, imageWidth, imageHeight);
            std::filesystem::path rawFilePath =
                    baseDir / "images" / (fileName + "_raw.exr");
            Pale::Log::PA_INFO("Saving image to: {}", (std::filesystem::current_path() / rawFilePath).string());
            Pale::Utils::saveRGBAFloatAsEXR(
                rawFilePath,
                rgbaRaw,
                imageWidth,
                imageHeight
            );


            std::vector<float> depthDistortionRaw =
                    Pale::downloadSensorDepthDistortionRAW(deviceSelector.getQueue(), sensor);
            std::filesystem::path depthDistortionfilePath = baseDir / "images" / (fileName + "depth_distortion.exr");
            std::vector<float> distortionRGBA(imageWidth * imageHeight * 4, 1.0f);
            for (uint32_t i = 0; i < imageWidth * imageHeight; ++i) {
                float v = depthDistortionRaw[i];
                if (!std::isfinite(v)) v = 0.0f;

                distortionRGBA[4 * i + 0] = v;
                distortionRGBA[4 * i + 1] = v;
                distortionRGBA[4 * i + 2] = v;
                distortionRGBA[4 * i + 3] = 1.0f;
            }

            Pale::Utils::saveRGBAFloatAsEXR(
                depthDistortionfilePath,
                distortionRGBA,
                imageWidth,
                imageHeight
            );

            saveCameraAuxiliaryBuffers(deviceSelector, sensor, baseDir, sensor.name);
        }


        if (settings.renderDebugGradientImages) {
            {
                auto entities = scene->getAllEntitiesWith<Pale::PointCloudComponent>();
                if (entities.empty()) {
                    throw std::runtime_error("debug gradients: scene has no PointCloudComponent");
                }
                Pale::Entity entity(entities.front(), scene.get());
                auto pointAssetSharedPtr = assetManager.get<Pale::PointAsset>(
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
                rebuild_bvh(&tracer, scene, buildProducts, &assetManager, deviceSelector, gpu);
            }

            Pale::Log::PA_INFO("Forward Render Pass after debug perturbation...");
            sensors = Pale::makeSensorsForScene(deviceSelector.getQueue(), buildProducts);
            tracer.renderForward(sensors);

            Pale::Log::PA_INFO("Adjoint Render Pass...");
            std::vector<Pale::SensorGPU> adjointSensors = Pale::makeSensorsForScene(
                deviceSelector.getQueue(), buildProducts, true, true);
            std::vector<Pale::DebugImages> debugImages(adjointSensors.size());
            Pale::PointGradients gradients = Pale::makeGradientsForScene(
                deviceSelector.getQueue(), buildProducts, debugImages.data());

            const bool useConstantDebugTarget = true;
            const float constantDebugTargetValue = -100.0f;
            const float adjointSamplesPerPixel = static_cast<float>(tracer.getSettings().adjointSamplesPerPixel);
            const std::filesystem::path outputRoot = std::filesystem::path("Output") / sceneName.parent_path();
            std::vector<std::vector<float>> adjointSourceImages(adjointSensors.size());

            auto sensorNameString = [](const Pale::SensorGPU &sensor) {
                return std::string(sensor.name);
            };
            auto findForwardSensorByName = [&](const std::string &sensorName) -> const Pale::SensorGPU * {
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
                const Pale::SensorGPU *forwardSensor = findForwardSensorByName(sensorName);
                if (!forwardSensor) {
                    throw std::runtime_error("debug gradients: no matching forward sensor for adjoint sensor " + sensorName);
                }
                const uint32_t imageWidth = adjointSensor.width;
                const uint32_t imageHeight = adjointSensor.height;
                if (forwardSensor->width != imageWidth || forwardSensor->height != imageHeight) {
                    throw std::runtime_error("debug gradients: forward/adjoint resolution mismatch for sensor " + sensorName);
                }

                std::vector<float> rgbaHostRendered =
                        Pale::downloadSensorRGBARAW(deviceSelector.getQueue(), *forwardSensor);
                std::vector<float> rgbaHostAdjointTarget;
                uint32_t targetWidth = 0;
                uint32_t targetHeight = 0;

                const std::filesystem::path targetImagePath = outputRoot / "images" / (sensorName + "_raw.exr");
                Pale::Utils::loadEXRAsRGBAFloat(targetImagePath, rgbaHostAdjointTarget, targetWidth, targetHeight);

                if (targetWidth != imageWidth || targetHeight != imageHeight) {
                    throw std::runtime_error("debug gradients: target resolution mismatch for sensor " + sensorName);
                }

                if (useConstantDebugTarget) {
                    const std::size_t pixelCount = static_cast<std::size_t>(imageWidth) * imageHeight;
                    for (std::size_t pixelIndex = 0; pixelIndex < pixelCount; ++pixelIndex) {
                        const std::size_t dstIndex = pixelIndex * 4ull;
                        rgbaHostAdjointTarget[dstIndex + 0] = constantDebugTargetValue;
                        rgbaHostAdjointTarget[dstIndex + 1] = constantDebugTargetValue;
                        rgbaHostAdjointTarget[dstIndex + 2] = constantDebugTargetValue;
                        rgbaHostAdjointTarget[dstIndex + 3] = constantDebugTargetValue;
                    }
                }

                std::vector<float> rgbaHostAdjointSource = Pale::Utils::computeL2ImageGradientRGBA(
                    rgbaHostRendered, rgbaHostAdjointTarget, imageWidth, imageHeight);
                Pale::uploadSensorRGBA(deviceSelector.getQueue(), adjointSensor, rgbaHostAdjointSource);
                adjointSourceImages[sensorIndex] = std::move(rgbaHostAdjointSource);
            }

            tracer.renderBackward(adjointSensors, gradients, debugImages.data());
            if (settings.useDepthDistortion) {
                tracer.renderDepthDistortionBackward(adjointSensors, gradients);
            }
            if (settings.useNormalConsistency) {
                tracer.renderNormalConsistencyBackward(adjointSensors, gradients);
            }

            const uint32_t debugSurfelIndex = settings.surfelIndexForDebugImages;
            if (debugSurfelIndex < gradients.numPoints) {
                float hostGradientBeta{};
                float hostGradientOpacity{};
                Pale::float3 hostGradientPosition{};
                deviceSelector.getQueue().memcpy(&hostGradientBeta, &gradients.gradBeta[debugSurfelIndex], sizeof(float)).wait();
                deviceSelector.getQueue().memcpy(&hostGradientOpacity, &gradients.gradOpacity[debugSurfelIndex], sizeof(float)).wait();
                deviceSelector.getQueue().memcpy(&hostGradientPosition, &gradients.gradPosition[debugSurfelIndex], sizeof(Pale::float3)).wait();

                Pale::Log::PA_INFO("debug surfel index = {}", debugSurfelIndex);
                Pale::Log::PA_INFO("grad Beta = ({})", hostGradientBeta);
                Pale::Log::PA_INFO("grad Opacity = ({})", hostGradientOpacity);
                Pale::Log::PA_INFO(
                    "grad Position = ({}, {}, {})",
                    hostGradientPosition.x(),
                    hostGradientPosition.y(),
                    hostGradientPosition.z());
            }

            for (std::size_t sensorIndex = 0; sensorIndex < adjointSensors.size(); ++sensorIndex) {
                const Pale::SensorGPU &adjointSensor = adjointSensors[sensorIndex];
                const std::string sensorName = adjointSensor.name;
                const uint32_t imageWidth = adjointSensor.width;
                const uint32_t imageHeight = adjointSensor.height;
                const std::filesystem::path cameraDebugDir = outputRoot / sensorName;
                std::filesystem::create_directories(cameraDebugDir);

                auto saveGradientSet = [&](const std::vector<float> &rgbaBuffer, const std::string &baseName) {
                    if (rgbaBuffer.empty()) {
                        return;
                    }
                    const std::filesystem::path exrPath = cameraDebugDir / (baseName + ".exr");
                    const std::filesystem::path pngPath = cameraDebugDir / (baseName + "_seismic.png");
                    const std::filesystem::path pngQ99Path = cameraDebugDir / (baseName + "_seismic_q099.png");

                    Pale::Utils::saveRGBAFloatAsEXR(exrPath, rgbaBuffer, imageWidth, imageHeight);
                    if (Pale::Utils::saveGradientSignPNG(
                        pngPath, rgbaBuffer, imageWidth, imageHeight, adjointSamplesPerPixel, 1.0f, false, true)) {
                        Pale::Log::PA_INFO("Wrote PNG image to: {}", pngPath.string());
                    }
                    Pale::Utils::saveGradientSignPNG(
                        pngQ99Path, rgbaBuffer, imageWidth, imageHeight, adjointSamplesPerPixel, 0.99f, false, true);
                };

                saveGradientSet(adjointSourceImages[sensorIndex], "adjoint_source_l2_gradient");

                Pale::DebugGradientImagesHost debugImagesHost = Pale::downloadDebugGradientImages(
                    deviceSelector.getQueue(), adjointSensor, debugImages[sensorIndex]);

                saveGradientSet(debugImagesHost.positionX, "position_x");
                saveGradientSet(debugImagesHost.positionY, "position_y");
                saveGradientSet(debugImagesHost.positionZ, "position_z");
                saveGradientSet(debugImagesHost.rotation, "rotation");
                saveGradientSet(debugImagesHost.scale, "scale");
                saveGradientSet(debugImagesHost.albedo, "albedo");
                saveGradientSet(debugImagesHost.opacity, "opacity");
                saveGradientSet(debugImagesHost.beta, "beta");
            }
        }
        // Write Registry:
    }
    assetManager.registry().save("asset_registry.yaml");
    deviceSelector.getQueue().wait();
    return 0;
}