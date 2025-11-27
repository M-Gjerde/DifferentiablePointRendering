// Main.cpp
#include <memory>
#include <filesystem>
#include <entt/entt.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/string_cast.hpp"
#include "Renderer/RenderPackage.h"

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


static void debugDensifyPointAsset(
    Pale::AssetManager &assetManager,
    const Pale::AssetHandle &pointCloudAssetHandle,
    uint32_t numberOfDebugCopies) {
    auto pointAssetSharedPtr = assetManager.get<Pale::PointAsset>(pointCloudAssetHandle);
    if (!pointAssetSharedPtr) {
        Pale::Log::PA_ERROR("debugDensifyPointAsset: failed to get PointAsset for handle {}",
                            std::string(pointCloudAssetHandle));
        return;
    }

    Pale::PointAsset &pointAsset = *pointAssetSharedPtr;
    if (pointAsset.points.empty()) {
        Pale::Log::PA_WARN("debugDensifyPointAsset: PointAsset has no PointGeometry blocks");
        return;
    }

    Pale::PointGeometry &pointGeometry = pointAsset.points.front();
    const std::size_t originalPointCount = pointGeometry.positions.size();

    if (originalPointCount == 0) {
        Pale::Log::PA_WARN("debugDensifyPointAsset: original point count is zero, nothing to duplicate");
        return;
    }

    Pale::Log::PA_INFO(
        "debugDensifyPointAsset: original point count = {}, creating {} debug copies",
        originalPointCount,
        numberOfDebugCopies
    );

    // Reserve space for all attribute arrays
    const std::size_t newTotalPointCount = originalPointCount + numberOfDebugCopies;
    auto reserveAttribute = [newTotalPointCount](auto &vectorAttribute) {
        vectorAttribute.reserve(newTotalPointCount);
    };

    reserveAttribute(pointGeometry.positions);
    reserveAttribute(pointGeometry.tanU);
    reserveAttribute(pointGeometry.tanV);
    reserveAttribute(pointGeometry.scales);
    reserveAttribute(pointGeometry.colors);
    reserveAttribute(pointGeometry.opacities);
    reserveAttribute(pointGeometry.shapes);
    reserveAttribute(pointGeometry.betas);

    // Base point to duplicate
    const std::size_t baseIndex = 0; // duplicate first point for simplicity
    const glm::vec3 basePosition = pointGeometry.positions[baseIndex];
    const glm::vec3 baseTanU = pointGeometry.tanU[baseIndex];
    const glm::vec3 baseTanV = pointGeometry.tanV[baseIndex];
    const glm::vec2 baseScale = pointGeometry.scales[baseIndex];
    const glm::vec3 baseColor = pointGeometry.colors[baseIndex];
    const float baseOpacity = pointGeometry.opacities[baseIndex];
    const float baseShape = pointGeometry.shapes[baseIndex];
    const float baseBeta = pointGeometry.betas[baseIndex];


    std::mt19937_64 rng;

    for (uint32_t debugCopyIndex = 0; debugCopyIndex < numberOfDebugCopies; ++debugCopyIndex) {
        glm::vec3 debugPosition = basePosition;

        std::uniform_real_distribution<float> unif(-0.66, 0.66);
        std::uniform_real_distribution<float> unif2(0, 1);

        debugPosition.x += unif(rng);
        debugPosition.y += unif(rng);
        debugPosition.z += unif(rng);

        glm::vec3 debugColor = baseColor;
        debugColor.x = unif2(rng);
        debugColor.y = unif2(rng);
        debugColor.z = unif2(rng);

        glm::vec2 debugScale = baseScale * (1.0f + 0.05f * static_cast<float>(debugCopyIndex));

        pointGeometry.positions.push_back(debugPosition);
        pointGeometry.tanU.push_back(baseTanU);
        pointGeometry.tanV.push_back(baseTanV);
        pointGeometry.scales.push_back(debugScale);
        pointGeometry.colors.push_back(debugColor);
        pointGeometry.opacities.push_back(baseOpacity);
        pointGeometry.shapes.push_back(baseShape);
        pointGeometry.betas.push_back(baseBeta);

        Pale::Log::PA_INFO(
            "debugDensifyPointAsset: created debug point {} at position=({}, {}, {}), color=({}, {}, {})",
            originalPointCount + debugCopyIndex,
            debugPosition.x, debugPosition.y, debugPosition.z,
            debugColor.x, debugColor.y, debugColor.z
        );
    }

    Pale::Log::PA_INFO("debugDensifyPointAsset: new point count = {}",
                       pointGeometry.positions.size());
}


int main(int argc, char **argv) {
    std::filesystem::path workingDirectory = "../Assets";
    std::filesystem::current_path(workingDirectory);

    Pale::Log::init();

    Pale::AssetManager assetManager{256};
    assetManager.enableHotReload(true);
    assetManager.registerLoader<Pale::Mesh>(Pale::AssetType::Mesh,
                                            std::make_shared<Pale::AssimpMeshLoader>());
    assetManager.registerLoader<Pale::Material>(Pale::AssetType::Material,
                                                std::make_shared<Pale::YamlMaterialLoader>());

    assetManager.registerLoader<Pale::PointAsset>(Pale::AssetType::PointCloud,
                                                  std::make_shared<Pale::PLYPointLoader>());

    assetManager.registry().load("asset_registry.yaml");

    // Load in xml file and Create Scene from xml
    std::shared_ptr<Pale::Scene> scene = std::make_shared<Pale::Scene>();
    Pale::AssetIndexFromRegistry assetIndexer(assetManager.registry());
    Pale::SceneSerializer serializer(scene, assetIndexer);
    serializer.deserialize("cbox_custom.xml");

    // Add Single Gaussian
    // Check CLI input for point cloud file
    std::filesystem::path pointCloudPath;
    if (argc > 1) {
        pointCloudPath = argv[1];
    } else {
        pointCloudPath = "initial.ply"; // default
    }

    bool addPoints = true;
    if (addPoints) {
        auto assetHandle = assetIndexer.importPath("PointClouds" / pointCloudPath, Pale::AssetType::PointCloud);
        auto entityGaussian = scene->createEntity("Gaussian");
        entityGaussian.addComponent<Pale::PointCloudComponent>().pointCloudID = assetHandle;
        auto &transform = entityGaussian.getComponent<Pale::TransformComponent>();
    }

    if (!addPoints) {
        Pale::Entity bunnyEntity = scene->createEntity("Bunny");
        // 1) Transform
        auto &bunnyTransformComponent = bunnyEntity.getComponent<Pale::TransformComponent>();
        bunnyTransformComponent.setPosition(glm::vec3(0.0f, 0.0f, 0.8f));
        bunnyTransformComponent.setRotationEuler(glm::vec3(0.0f, -25.0f, 45.0f));
        bunnyTransformComponent.setScale(glm::vec3(1.0f));

        // 2) Mesh
        Pale::AssetHandle bunnyMeshAssetHandle =
                assetIndexer.importPath("meshes/cube.ply", Pale::AssetType::Mesh);

        auto &bunnyMeshComponent = bunnyEntity.addComponent<Pale::MeshComponent>();
        bunnyMeshComponent.meshID = bunnyMeshAssetHandle;

        // 3) Material
        Pale::AssetHandle bunnyMaterialAssetHandle =
                assetIndexer.importPath("Materials/cbox_custom/bsdf_light_gray_0.mat.yaml", Pale::AssetType::Material);

        auto &bunnyMaterialComponent = bunnyEntity.addComponent<Pale::MaterialComponent>();
        bunnyMaterialComponent.materialID = bunnyMaterialAssetHandle;
    }

    logSceneSummary(scene, assetManager);

    //FInd Sycl Device
    Pale::DeviceSelector deviceSelector;
    // Build rendering products (BLAS. TLAS, Emissive lists, etc..)
    Pale::AssetAccessFromManager assetAccessor(assetManager);

    auto buildProducts = Pale::SceneBuild::build(scene, assetAccessor, Pale::SceneBuild::BuildOptions());
    // Upload Scene to GPU
    auto gpu = Pale::SceneUpload::allocateAndUpload(buildProducts, deviceSelector.getQueue()); // scene only

    //  cuda/rocm
    Pale::PathTracerSettings settings;
    settings.photonsPerLaunch = 1e5;
    settings.maxBounces = 4;
    settings.numForwardPasses = 20;
    settings.numGatherPasses = 1;
    settings.maxAdjointBounces = 2;
    settings.adjointSamplesPerPixel = 4;
    settings.renderDebugGradientImages = true;

    Pale::PathTracer tracer(deviceSelector.getQueue(), settings);
    tracer.setScene(gpu, buildProducts);
    Pale::Log::PA_INFO("Forward Render Pass...");

    Pale::SensorGPU sensor = Pale::makeSensorsForScene(deviceSelector.getQueue(), buildProducts, true);


    tracer.renderForward(sensor); // films is span/array

    {
        // // Save each sensor image
        std::vector<uint8_t> rgba = Pale::downloadSensorRGBA(deviceSelector.getQueue(), sensor);
        const uint32_t W = sensor.width, H = sensor.height;
        std::filesystem::path filePath = "Output" / pointCloudPath.filename().replace_extension("") /
                                         "out_photonmap.png";
        Pale::Utils::savePNG(filePath, rgba, W, H);
    }

    // -------------------------------------------------------
    // DEBUG: modify topology via asset, then rebuild scene
    // -------------------------------------------------------

    /*
    const uint32_t numberOfDebugCopies = 10;
    debugDensifyPointAsset(assetManager, assetHandle, numberOfDebugCopies);

    buildProducts = Pale::SceneBuild::build(scene, assetAccessor, Pale::SceneBuild::BuildOptions());
    Pale::SceneUpload::uploadOrReallocate(buildProducts, gpu, deviceSelector.getQueue()); // scene only
    tracer.setScene(gpu, buildProducts);

    // Forward render with new topology
    tracer.renderForward(sensor);

    {
        // // Save each sensor image
        std::vector<float> rgb = Pale::downloadSensorLDR(deviceSelector.getQueue(), sensor);
        const uint32_t W = sensor.width, H = sensor.height;
        std::filesystem::path filePath = "Output" / pointCloudPath.filename().replace_extension("") /
            "out_photonmap_ldr_modified.png";
        Pale::Utils::savePNGWith3Channel(filePath, rgb, W, H);
    }
    */

    // outputs derived from cameras
    // Start a Tracer
    // Register the scene with the Tracer

    // Render
    Pale::PointGradients gradients = Pale::makeGradientsForScene(deviceSelector.getQueue(), buildProducts);


    // 4) (Optional) load or compute residuals on host, upload pointer
    //auto adjoint = calculateAdjointImage("Output/target/out_photonmap.pfm", deviceSelector.getQueue(), sensor, true);
    Pale::SensorGPU adjointSensor = Pale::makeSensorsForScene(deviceSelector.getQueue(), buildProducts, true);

    Pale::Log::PA_INFO("Adjoint Render Pass...");
    tracer.renderBackward(adjointSensor, gradients); // PRNG replay adjoint

    {
        auto debugImages = Pale::downloadDebugGradientImages(
            deviceSelector.getQueue(), adjointSensor, gradients);

        const uint32_t imageWidth = adjointSensor.width;
        const uint32_t imageHeight = adjointSensor.height;
        const float adjointSamplesPerPixel =
                static_cast<float>(tracer.getSettings().adjointSamplesPerPixel);

        std::filesystem::path baseDir =
                "Output" / pointCloudPath.filename().replace_extension("");
        std::filesystem::create_directories(baseDir);

        auto saveGradientSet = [&](const std::vector<float> &rgbaBuffer,
                                   const std::string &prefixBaseName) {
            // channelIndex: 0 = R, 1 = G, 2 = B
            const char channelNames[3] = {'R', 'G', 'B'};

            for (int channelIndex = 0; channelIndex < 3; ++channelIndex) {
                const char channelChar = channelNames[channelIndex];

                // Full-range (absQuantile = 1.0)
                {
                    std::string fileName =
                            "adjoint_" + prefixBaseName + "_" + channelChar + "_seismic.png";
                    std::filesystem::path filePath = baseDir / fileName;

                    if (Pale::Utils::saveGradientSingleChannelPNG(
                        filePath,
                        rgbaBuffer,
                        imageWidth,
                        imageHeight,
                        /*channelIndex=*/channelIndex,
                        adjointSamplesPerPixel,
                        /*absQuantile=*/1.0f,
                        /*flipY=*/false,
                        /*useSeismic=*/true)) {
                        Pale::Log::PA_INFO("Wrote PNG image to: {}", filePath.string());
                    }

                    // q=0.99
                    std::string fileNameQuantile =
                            "adjoint_" + prefixBaseName + "_" + channelChar + "_seismic_q099.png";
                    std::filesystem::path filePathQuantile = baseDir / fileNameQuantile;

                    Pale::Utils::saveGradientSingleChannelPNG(
                        filePathQuantile,
                        rgbaBuffer,
                        imageWidth,
                        imageHeight,
                        /*channelIndex=*/channelIndex,
                        adjointSamplesPerPixel,
                        /*absQuantile=*/0.99f,
                        /*flipY=*/false,
                        /*useSeismic=*/true);
                }
            }
        };

        // Save all debug gradient images:
        //  - position   → framebuffer_pos
        //  - rotation   → framebuffer_rot
        //  - scale      → framebuffer_scale
        //  - opacity    → framebuffer_opacity
        //  - albedo     → framebuffer_albedo
        saveGradientSet(debugImages.position, "pos");
        saveGradientSet(debugImages.rotation, "rot");
        saveGradientSet(debugImages.scale, "scale");
        saveGradientSet(debugImages.opacity, "opacity");
        saveGradientSet(debugImages.albedo, "albedo");
        saveGradientSet(debugImages.beta, "beta");
    }


    // Write Registry:
    assetManager.registry().save("asset_registry.yaml");
    deviceSelector.getQueue().wait();
    return 0;
}
