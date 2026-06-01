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
        settings.maxBounces = 1;
        settings.maxAdjointBounces = 1; // 2 == First surfel intersection gradients, 3 = Second surfel gradients

        settings.numForwardPasses = 1;
        settings.numShadowRays = 1;
        settings.numAdjointShadowRays = 1;
        settings.adjointSamplesPerPixel = 4;

        settings.renderDebugGradientImages = !true;
        settings.enableAdjointDirectLight = true;
        settings.useDepthDistortion = true;
        settings.useNormalConsistency = true;
        settings.surfelIndexForDebugImages = UINT32_MAX;

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


        for (int i = 0; i < 1; ++i) {
            if (settings.renderDebugGradientImages) {
                {
                    auto entities = scene->getAllEntitiesWith<Pale::PointCloudComponent>();
                    Pale::Entity entity(entities.front(), (scene.get()));
                    auto pointAssetSharedPtr = assetManager.get<Pale::PointAsset>(
                        entity.getComponent<Pale::PointCloudComponent>().pointCloudID);
                    if (!pointAssetSharedPtr) {
                        throw std::runtime_error(
                            "set_gaussian_opacity: failed to get PointAsset for dynamic point cloud");
                    }
                    Pale::PointAsset &pointAsset = *pointAssetSharedPtr;
                    Pale::PointGeometry &pointGeometry = pointAsset.points.front();
                    //pointGeometry.positions[5].z = 0.3f;
                    rebuild_bvh(&tracer, scene, buildProducts, &assetManager, deviceSelector, gpu);
                }
                Pale::Log::PA_INFO("Forward Render Pass...");
                tracer.renderForward(sensors); // films is span/array

                // Render with pertubation
                // Save target image
                for (const auto &sensor: sensors) {
                    std::vector<uint8_t> rgba = Pale::downloadSensorRGBA(deviceSelector.getQueue(), sensor);
                    const uint32_t imageWidth = sensor.width;
                    const uint32_t imageHeight = sensor.height;
                    std::vector<float> rgbaRaw = Pale::downloadSensorRGBARAW(deviceSelector.getQueue(), sensor);
                    std::filesystem::path baseDir = std::filesystem::path("Output") / sceneName.parent_path();
                    // assumes sensor.name is std::string
                    std::filesystem::create_directories(baseDir);
                    std::string fileName = sensor.name;
                    //fileName += "_photonmap";
                    std::filesystem::path filePath = baseDir / "images" / (fileName + ".png");
                    Pale::Utils::savePNG(filePath, rgba, imageWidth, imageHeight);
                    std::filesystem::path rawFilePath = baseDir / "images" / (fileName + "_raw.exr");
                    Pale::Utils::saveRGBAFloatAsEXR(
                        rawFilePath,
                        rgbaRaw,
                        imageWidth,
                        imageHeight
                    );
                }
                Pale::Log::PA_INFO("Adjoint Render Pass...");
                std::vector<Pale::SensorGPU> availableSensors = Pale::makeSensorsForScene(
                    deviceSelector.getQueue(), buildProducts, true, true);
                std::vector<Pale::SensorGPU> selectedAdjointSensors;
                Pale::SensorGPU selectedSensor;
                for (int i = 0; const auto &sensor: availableSensors) {
                    if (sensor.camera.useForAdjointPass) {
                        selectedAdjointSensors.push_back(sensor);
                        for (const auto &forwardSensor: sensors) {
                            if (std::string(sensor.camera.name) == std::string(forwardSensor.camera.name))
                                selectedSensor = forwardSensor;
                        }
                        break;
                    }
                    i++;
                }
                std::vector<Pale::DebugImages> debugImages(selectedAdjointSensors.size());
                Pale::PointGradients gradients = Pale::makeGradientsForScene(
                    deviceSelector.getQueue(), buildProducts, debugImages.data());
                std::vector<float> rgbaHostAdjointTarget;
                std::filesystem::path baseDir =
                        std::filesystem::path("Output") / sceneName.parent_path(); // assumes sensor.name is std::string


                std::string fileName = selectedSensor.name;
                std::filesystem::path targetImagePath = baseDir / "images" / (fileName + "_raw.exr");

                uint32_t width, height;
                Pale::Utils::loadEXRAsRGBAFloat(targetImagePath, rgbaHostAdjointTarget, width, height);

                bool filledTarget = true;
                if (filledTarget)
                    for (std::uint32_t y = 0; y < height; ++y) {
                        for (std::uint32_t x = 0; x < width; ++x) {
                            const std::size_t pixelIndex = static_cast<std::size_t>(y) * width + x;
                            const std::size_t dstIndex = pixelIndex * 4ull;
                            // Imf::Rgba stores half by default; implicit conversion to float is fine.
                            float value = -100.0f;
                            rgbaHostAdjointTarget[dstIndex + 0] = value;
                            rgbaHostAdjointTarget[dstIndex + 1] = value;
                            rgbaHostAdjointTarget[dstIndex + 2] = value;
                            rgbaHostAdjointTarget[dstIndex + 3] = value;
                        }
                    }
                std::vector<float> rgbaHostRendered =
                        Pale::downloadSensorRGBARAW(deviceSelector.getQueue(), selectedSensor);

                std::vector<float> rgbaHostAdjointSource =
                        Pale::Utils::computeL2ImageGradientRGBA(rgbaHostRendered, rgbaHostAdjointTarget, width, height);


                std::vector<float> rgba =
                        Pale::uploadSensorRGBA(deviceSelector.getQueue(), selectedAdjointSensors.front(),
                                               rgbaHostAdjointSource);


                tracer.renderBackward(selectedAdjointSensors, gradients, debugImages.data()); // PRNG replay adjoint
                tracer.renderDepthDistortionBackward(selectedAdjointSensors, gradients); // PRNG replay adjoint
                tracer.renderNormalConsistencyBackward(selectedAdjointSensors, gradients); // PRNG replay adjoint
                float hostGradientBeta{};
                deviceSelector.getQueue().memcpy(&hostGradientBeta, gradients.gradBeta, sizeof(float)).wait();
                Pale::Log::PA_INFO(
                    "grad Beta = ({})",
                    hostGradientBeta
                );
                float hostGradientOpacity{};
                deviceSelector.getQueue().memcpy(&hostGradientOpacity, gradients.gradOpacity, sizeof(float)).wait();
                Pale::Log::PA_INFO("grad Opacity = ({})", hostGradientOpacity);
                Pale::float3 hostPosition{};
                deviceSelector.getQueue().memcpy(&hostPosition, &gradients.gradPosition[1], 3 * sizeof(float)).wait();

                Pale::Log::PA_INFO("grad Position = ({}, {}, {})", hostPosition.x(), hostPosition.y(), hostPosition.z()
                );

                for (size_t i = 0; const auto &adjointSensor: selectedAdjointSensors) {
                    auto debugImagesHost = Pale::downloadDebugGradientImages(
                        deviceSelector.getQueue(), adjointSensor, debugImages[i]);
                    i++;
                    const uint32_t imageWidth = adjointSensor.width;
                    const uint32_t imageHeight = adjointSensor.height;
                    const float adjointSamplesPerPixel = static_cast<float>(tracer.getSettings().
                        adjointSamplesPerPixel);
                    // Per-camera base directory: Output/<pointcloud>/<camera_name>/
                    std::filesystem::path baseDir =
                            std::filesystem::path("Output") / sceneName.parent_path() / adjointSensor.name; {
                        std::filesystem::path pngPath = baseDir / "adjoint_source_l2_gradient_seismic.png";
                        Pale::Utils::saveGradientSignPNG(pngPath, rgbaHostAdjointSource, width, height,
                                                         adjointSamplesPerPixel, 1.0f, false, true);
                        std::filesystem::path pngQ99Path = baseDir / "adjoint_source_l2_gradient_seismic_q099.png";
                        Pale::Utils::saveGradientSignPNG(pngQ99Path, rgbaHostAdjointSource, width, height,
                                                         adjointSamplesPerPixel, 0.95f, false, true);
                            }


                    std::filesystem::create_directories(baseDir);

                    auto saveGradientSet = [&](const std::vector<float> &rgbaBuffer,
                                               const std::string &prefixBaseName) {
                        // Full-range (absQuantile = 1.0)
                        {
                            std::string fileName = prefixBaseName + "_seismic.png";
                            std::filesystem::path filePath = baseDir / fileName;
                            if (Pale::Utils::saveGradientSignPNG(filePath, rgbaBuffer, imageWidth, imageHeight,
                                                                 adjointSamplesPerPixel, 1.0f, false, true)) {
                                Pale::Log::PA_INFO("Wrote PNG image to: {}", filePath.string());
                                                                 }
                            // q=0.99
                            std::string fileNameQuantile = prefixBaseName + "_seismic_q099.png";
                            std::filesystem::path filePathQuantile = baseDir / fileNameQuantile;
                            Pale::Utils::saveGradientSignPNG(filePathQuantile, rgbaBuffer, imageWidth, imageHeight,
                                                             adjointSamplesPerPixel, 0.99f, false, true);
                        }
                    };

                    saveGradientSet(debugImagesHost.positionX, "posX");
                    saveGradientSet(debugImagesHost.positionY, "posY");
                    saveGradientSet(debugImagesHost.positionZ, "posZ");
                    saveGradientSet(debugImagesHost.rotation, "rot");
                    saveGradientSet(debugImagesHost.scale, "scale");
                    saveGradientSet(debugImagesHost.opacity, "opacity");
                    saveGradientSet(debugImagesHost.albedo, "albedo");
                    saveGradientSet(debugImagesHost.beta, "beta");
                }
            }
        }

        if (settings.renderDebugGradientImages) {
            std::vector<Pale::DebugImages> debugImages(sensors.size());
            Pale::PointGradients gradients = Pale::makeGradientsForScene(deviceSelector.getQueue(), buildProducts,
                                                                         debugImages.data());

            std::vector<Pale::SensorGPU> adjointSensors =
                    Pale::makeSensorsForScene(deviceSelector.getQueue(), buildProducts, true, true);

            Pale::Log::PA_INFO("Adjoint Render Pass...");
            tracer.renderBackward(adjointSensors, gradients, debugImages.data()); // PRNG replay adjoint

            for (size_t i = 0; const auto &adjointSensor: adjointSensors) {
                auto debugImagesHost = Pale::downloadDebugGradientImages(
                    deviceSelector.getQueue(), adjointSensor, debugImages[i]);

                i++;
                const uint32_t imageWidth = adjointSensor.width;
                const uint32_t imageHeight = adjointSensor.height;
                const float adjointSamplesPerPixel =
                        static_cast<float>(tracer.getSettings().adjointSamplesPerPixel);

                // Per-camera base directory: Output/<pointcloud>/<camera_name>/
                std::filesystem::path baseDir =
                        std::filesystem::path("Output")

                        / adjointSensor.name;

                std::filesystem::create_directories(baseDir);

                auto saveGradientSet = [&](const std::vector<float> &rgbaBuffer,
                                           const std::string &prefixBaseName) {
                    const char channelNames[3] = {'R', 'G', 'B'};

                    for (int channelIndex = 0; channelIndex < 3; ++channelIndex) {
                        const char channelChar = channelNames[channelIndex];

                        // Full-range (absQuantile = 1.0)
                        {
                            std::string fileName =
                                    prefixBaseName + "_" + channelChar + "_seismic.png";
                            std::filesystem::path filePath = baseDir / fileName;

                            if (Pale::Utils::saveGradientSingleChannelPNG(
                                filePath,
                                rgbaBuffer,
                                imageWidth,
                                imageHeight,
                                channelIndex,
                                adjointSamplesPerPixel,
                                1.0f,
                                false,
                                true)) {
                                Pale::Log::PA_INFO("Wrote PNG image to: {}", filePath.string());
                                }

                            // q=0.99
                            std::string fileNameQuantile =
                                    prefixBaseName + "_" + channelChar + "_seismic_q099.png";
                            std::filesystem::path filePathQuantile = baseDir / fileNameQuantile;

                            Pale::Utils::saveGradientSingleChannelPNG(
                                filePathQuantile,
                                rgbaBuffer,
                                imageWidth,
                                imageHeight,
                                channelIndex,
                                adjointSamplesPerPixel,
                                0.99f,
                                false,
                                true);
                        }
                    }
                };

                saveGradientSet(debugImagesHost.positionX, "posX");
                saveGradientSet(debugImagesHost.positionY, "posY");
                saveGradientSet(debugImagesHost.positionZ, "posZ");
                saveGradientSet(debugImagesHost.rotation, "rot");
                saveGradientSet(debugImagesHost.scale, "scale");
                saveGradientSet(debugImagesHost.opacity, "opacity");
                saveGradientSet(debugImagesHost.albedo, "albedo");
                saveGradientSet(debugImagesHost.beta, "beta");
                //saveGradientSet(debugImagesHost.depthLoss, "DepthLoss");
                //saveGradientSet(debugImagesHost.depthLossPos, "DepthLossPos");
                //saveGradientSet(debugImagesHost.normalLoss, "NormalLoss");
            }

            // Download and log gradPosition[0]
            if (gradients.numPoints > 0 && gradients.gradPosition != nullptr) {
                Pale::float3 hostGradientPosition0{};
                deviceSelector.getQueue()
                        .memcpy(&hostGradientPosition0,
                                gradients.gradPosition,
                                sizeof(Pale::float3))
                        .wait();

                const float gradientMagnitude =
                        std::sqrt(hostGradientPosition0.x() * hostGradientPosition0.x() +
                                  hostGradientPosition0.y() * hostGradientPosition0.y() +
                                  hostGradientPosition0.z() * hostGradientPosition0.z());

                Pale::Log::PA_INFO(
                    "gradPosition[0] = ({}, {}, {}), |g| = {}",
                    hostGradientPosition0.x(),
                    hostGradientPosition0.y(),
                    hostGradientPosition0.z(),
                    gradientMagnitude
                );
            }
        }
    }

    // Write Registry:
    assetManager.registry().save("asset_registry.yaml");
    deviceSelector.getQueue().wait();
    return 0;
}
