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


inline void applyPointTopologyChange(
std::shared_ptr<Pale::Scene> scene,
Pale::AssetAccessFromManager assetAccessor,
    Pale::SceneBuild::BuildProducts &buildProducts,
    Pale::GPUSceneBuffers &sceneGpu,
    Pale::PointGradients &gradients,
    sycl::queue queue)
{
    // Assume single dynamic point cloud for now
    if (!buildProducts.pointCloudRanges.empty()) {
        auto &range = buildProducts.pointCloudRanges[0];
        range.pointCount = static_cast<uint32_t>(buildProducts.points.size());
        Pale::Log::PA_INFO(
            "applyPointTopologyChange: updated pointCloudRanges[0] to firstPoint = {}, pointCount = {}",
            range.firstPoint,
            range.pointCount
        );
    }

    Pale::Log::PA_INFO("applyPointTopologyChange: rebuilding BVHs and reallocating GPU buffers");
    Pale::SceneBuild::rebuildBVHs(buildProducts, Pale::SceneBuild::BuildOptions());
    Pale::SceneUpload::allocateOrReallocate(buildProducts, sceneGpu, queue);

    Pale::freeGradientsForScene(queue, gradients);
    gradients = Pale::makeGradientsForScene(queue, buildProducts);
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

    auto assetHandle = assetIndexer.importPath("PointClouds" / pointCloudPath, Pale::AssetType::PointCloud);
    auto entityGaussian = scene->createEntity("Gaussian");
    entityGaussian.addComponent<Pale::PointCloudComponent>().pointCloudID = assetHandle;
    auto& transform = entityGaussian.getComponent<Pale::TransformComponent>();


    //transform.setRotationEuler(glm::vec3(0.0f, 0.0f, 165.0f));
    //transform.setScale(glm::vec3(0.5f, 0.5f, 0.5f));
    //transform.setPosition(glm::vec3(0.05f, 0.0f, 0.0f));
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
    settings.photonsPerLaunch = 1e4; // 1e6
    settings.maxBounces = 4;
    settings.numForwardPasses = 30;
    settings.numGatherPasses = 16;
    settings.maxAdjointBounces = 2;
    settings.adjointSamplesPerPixel = 12;

    Pale::PathTracer tracer(deviceSelector.getQueue(), settings);
    tracer.setScene(gpu, buildProducts);
    Pale::Log::PA_INFO("Forward Render Pass...");

    Pale::SensorGPU sensor = Pale::makeSensorsForScene(deviceSelector.getQueue(), buildProducts);

    tracer.renderForward(sensor); // films is span/array


    // outputs derived from cameras
    // Start a Tracer
    // Register the scene with the Tracer

    // Render
    Pale::PointGradients gradients = Pale::makeGradientsForScene(deviceSelector.getQueue(), buildProducts);
    /*
    // --- DEBUG: modify point topology and rebuild BVH exactly like Python path ---
if (!buildProducts.pointCloudRanges.empty()) {
    auto &pointCloudRange = buildProducts.pointCloudRanges[0];

    Pale::Log::PA_INFO("DEBUG: before change, pointCloudRanges[0] : firstPoint={}, pointCount={}",
                       pointCloudRange.firstPoint, pointCloudRange.pointCount);

    if (pointCloudRange.pointCount >= 1) {
        const uint32_t originalPointCount = pointCloudRange.pointCount;
        constexpr uint32_t numberOfDebugCopies = 100; // add extra points

        buildProducts.points.front().scale *= 0.3f;
        // Resize the global point array to hold the new copies
        const uint32_t newTotalPointCount = originalPointCount + numberOfDebugCopies;
        buildProducts.points.resize(newTotalPointCount);

        // Base point to duplicate
        const Pale::Point basePoint = buildProducts.points[pointCloudRange.firstPoint];

        for (uint32_t debugCopyIndex = 0; debugCopyIndex < numberOfDebugCopies; ++debugCopyIndex) {
            const uint32_t newPointIndex = originalPointCount + debugCopyIndex;
            Pale::Point &newPoint = buildProducts.points[newPointIndex];

            // Start from the base point
            newPoint = basePoint;

            // Simple deterministic perturbation: small grid around base position
            const float offsetScale = 0.2f; // tweak as desired
            const float offsetX = -0.3 + offsetScale * static_cast<float>(debugCopyIndex);
            const float offsetY = offsetScale * static_cast<float>(debugCopyIndex % 2 ? 1 : -1);
            const float offsetZ = 0.0f;

            newPoint.position.x() += offsetX;
            newPoint.position.y() += offsetY;
            newPoint.position.z() += offsetZ;

            // Optional: slightly vary color and scale so you can see them visually
            newPoint.color.x() = sycl::clamp(basePoint.color.x() + 0.1f * debugCopyIndex, 0.0f, 1.0f);
            newPoint.color.y() = sycl::clamp(basePoint.color.y() - 0.05f * debugCopyIndex, 0.0f, 1.0f);
            newPoint.color.z() = basePoint.color.z();

            newPoint.scale.x() *= (1.0f + 0.05f * debugCopyIndex);
            newPoint.scale.y() *= (1.0f + 0.05f * debugCopyIndex);

            Pale::Log::PA_INFO(
                "DEBUG: created debug point {} at position=({}, {}, {}), color=({}, {}, {})",
                newPointIndex,
                newPoint.position.x(), newPoint.position.y(), newPoint.position.z(),
                newPoint.color.x(), newPoint.color.y(), newPoint.color.z()
            );
        }

        // Update the point cloud range’s logical pointCount to match
        pointCloudRange.pointCount = newTotalPointCount;

        Pale::Log::PA_INFO("DEBUG: resized buildProducts.points to {}, pointCloudRange.pointCount={}",
                           buildProducts.points.size(), pointCloudRange.pointCount);
    }


        // Now apply the same pipeline as Python
        applyPointTopologyChange(scene, assetAccessor, buildProducts, gpu, gradients, deviceSelector.getQueue());

        // Make sure tracer sees the updated scene
        Pale::SceneUpload::upload(buildProducts, gpu, deviceSelector.getQueue());
        tracer.setScene(gpu, buildProducts);
    }
    */

    //tracer.renderForward(sensor); // films is span/array

    {
        // // Save each sensor image
        std::vector<uint8_t> rgba = Pale::downloadSensorRGBA(deviceSelector.getQueue(), sensor);
        const uint32_t W = sensor.width, H = sensor.height;
        std::filesystem::path filePath = "Output" / pointCloudPath.filename().replace_extension("") / "out_photonmap.png";
        Pale::Utils::savePNG(filePath, rgba, W, H);

        //Pale::Utils::savePFM(filePath.replace_extension(".pfm"), rgba, W, H); // writes RGB, drops A
    }
    {
        // // Save each sensor image
        std::vector<float> rgb = Pale::downloadSensorLDR(deviceSelector.getQueue(), sensor);
        const uint32_t W = sensor.width, H = sensor.height;
        std::filesystem::path filePath = "Output" / pointCloudPath.filename().replace_extension("") / "out_photonmap_ldr.png";
        Pale::Utils::savePNGWith3Channel(filePath, rgb, W, H);
    }

    //if (pointCloudPath.filename() != "initial.ply") {
    //    Pale::Log::PA_INFO("TARGET RENDERED exiting...");
    //    return 0;
    //}


    // 4) (Optional) load or compute residuals on host, upload pointer
    //auto adjoint = calculateAdjointImage("Output/target/out_photonmap.pfm", deviceSelector.getQueue(), sensor, true);
    Pale::SensorGPU adjointSensor = Pale::makeSensorsForScene(deviceSelector.getQueue(), buildProducts, true);

    Pale::Log::PA_INFO("Adjoint Render Pass...");
    tracer.renderBackward(adjointSensor, gradients); // PRNG replay adjoint


        {
        // // Save each sensor image
        auto rgba = Pale::downloadDebugGradientImage(deviceSelector.getQueue(), adjointSensor, gradients);
        const uint32_t W = adjointSensor.width, H = adjointSensor.height;
        std::filesystem::path filePath = "Output" / pointCloudPath.filename().replace_extension("") / "adjoint_out.png";
        if (Pale::Utils::saveGradientSignPNG(
            filePath, rgba, W, H, tracer.getSettings().adjointSamplesPerPixel, 1.0, false, true)) {
            Pale::Log::PA_INFO("Wrote PNG image to: {}", filePath.string());
        };

        Pale::Utils::savePFM(filePath.replace_extension(".pfm"), rgba, W, H); // writes RGB, drops A}

        filePath.replace_filename("adjoint_out_099_quantile.png");
        if (Pale::Utils::saveGradientSignPNG(
            filePath, rgba, W, H, tracer.getSettings().adjointSamplesPerPixel, 0.99, false, true)) {
            Pale::Log::PA_INFO("Wrote PNG image to: {}", filePath.string());
        };
    }


    // Write Registry:
    assetManager.registry().save("asset_registry.yaml");
    deviceSelector.getQueue().wait();
    return 0;
}
