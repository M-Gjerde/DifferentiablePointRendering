// Main.cpp
#include <memory>
#include <filesystem>
#include <entt/entt.hpp>
#include "Renderer/GPUDataStructures.h"

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


Pale::AdjointGPU calculateAdjointImage(std::filesystem::path targetImagePath, sycl::queue queue, Pale::SensorGPU &sensor) {
    // 1) Download current predicted image (RGBA, linear)
    Pale::AdjointGPU adjoint;
    std::vector<float> predictedRgba; // size = W*H*4
    uint32_t predictedWidth = 0, predictedHeight = 0; {
        predictedRgba = Pale::downloadSensorRGBA(queue, sensor);
        predictedWidth = sensor.width;
        predictedHeight = sensor.height;
    }

    // 2) Load target reference image (RGB, linear)
    std::vector<float> targetRgb; // size = W*H*3
    uint32_t targetWidth = 0, targetHeight = 0;
    if (!Pale::Utils::loadPFM_RGB(targetImagePath, targetRgb, targetWidth, targetHeight)) {
        Pale::Log::PA_ERROR("Failed to find target image at {}", targetImagePath.string());
        return adjoint;
    };

    // 3) Validate dimensions
    if (predictedWidth != targetWidth || predictedHeight != targetHeight) {
        Pale::Log::PA_ERROR("setResiduals(): predicted and target image sizes differ");
        return adjoint;
    }
    const uint32_t imageWidth = predictedWidth;
    const uint32_t imageHeight = predictedHeight;
    const size_t pixelCount = static_cast<size_t>(imageWidth) * imageHeight;

    // 4) Extract predicted RGB from RGBA
    std::vector<float> predictedRgb(pixelCount * 3u);
    for (size_t i = 0, j = 0; i < pixelCount; ++i, j += 4) {
        predictedRgb[i * 3 + 0] = predictedRgba[j + 0];
        predictedRgb[i * 3 + 1] = predictedRgba[j + 1];
        predictedRgb[i * 3 + 2] = predictedRgba[j + 2];
    }

    // 5) Compute per-pixel residuals and adjoint (for 0.5 * L2)
    //     residual = predicted - target
    //     adjoint  = residual  (∂L/∂predicted)
    std::vector<float> residualRgb(pixelCount * 3u);
    for (size_t k = 0; k < residualRgb.size(); ++k) {
        residualRgb[k] = predictedRgb[k] - targetRgb[k];
    }
    std::vector<Pale::float4> residualRgba(pixelCount);
    for (size_t i = 0; i < pixelCount; ++i) {
        const size_t k = i * 3u;
        residualRgba[i] = Pale::float4{residualRgb[k+0], residualRgb[k+1], residualRgb[k+2], 0.0f};
    }



    // Optional: if you use mean squared error over N pixels, scale by 1/N
    // const float invPixelCount = 1.0f / static_cast<float>(pixelCount);
    // for (float& v : residualRgb) v *= invPixelCount;

    // 6) Save adjoint image to disk (PFM, RGB)
    const std::filesystem::path adjointPath = "Output/initial/adjoint_rgb.pfm";
    Pale::Utils::savePFM(adjointPath, residualRgb, imageWidth, imageHeight);

    // 6b) Also save each RGB component separately
    std::vector<float> residualR(pixelCount);
    std::vector<float> residualG(pixelCount);
    std::vector<float> residualB(pixelCount);

    for (size_t i = 0; i < pixelCount; ++i) {
        residualR[i] = residualRgb[i * 3 + 0];
        residualG[i] = residualRgb[i * 3 + 1];
        residualB[i] = residualRgb[i * 3 + 2];
    }

    Pale::Utils::savePFM("Output/initial/adjoint_r.pfm", residualR, imageWidth, imageHeight);
    Pale::Utils::savePFM("Output/initial/adjoint_g.pfm", residualG, imageWidth, imageHeight);
    Pale::Utils::savePFM("Output/initial/adjoint_b.pfm", residualB, imageWidth, imageHeight);

    // 7) (Optional) also save residual magnitude for debugging
    {
        std::vector<float> residualLuminance(pixelCount);
        for (size_t i = 0; i < pixelCount; ++i) {
            const float r = residualRgb[i * 3 + 0];
            const float g = residualRgb[i * 3 + 1];
            const float b = residualRgb[i * 3 + 2];
            residualLuminance[i] = std::sqrt(r * r + g * g + b * b);
        }
        Pale::Utils::savePFM("Output/initial/adjoint_mag.pfm", residualLuminance, imageWidth, imageHeight,
                             1,
                             true);
    }


    auto* deviceResidualRgba = sycl::malloc_device<Pale::float4>(pixelCount, queue);
    auto* gradient_pk = sycl::malloc_device<Pale::float3>(10, queue);
    queue.memcpy(deviceResidualRgba, residualRgba.data(),
                 sizeof(Pale::float4) * pixelCount).wait();

    queue.fill(gradient_pk, Pale::float3(0.0f), 10);

    adjoint.framebuffer = deviceResidualRgba;
    adjoint.width = imageWidth;
    adjoint.height = imageHeight;
    adjoint.gradient_pk = gradient_pk;

    return adjoint;

}

int main(int argc, char** argv) {
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

    logSceneSummary(scene, assetManager);

    //FInd Sycl Device
    Pale::DeviceSelector deviceSelector;
    // Build rendering products (BLAS. TLAS, Emissive lists, etc..)
    Pale::AssetAccessFromManager assetAccessor(assetManager);

    auto buildProducts = Pale::SceneBuild::build(scene, assetAccessor, Pale::SceneBuild::BuildOptions());
    // Upload Scene to GPU
    auto gpu = Pale::SceneUpload::upload(buildProducts, deviceSelector.getQueue()); // scene only

    Pale::SensorGPU sensor = Pale::makeSensorsForScene(deviceSelector.getQueue(), buildProducts);
    // outputs derived from cameras
    // Start a Tracer
    Pale::PathTracer tracer(deviceSelector.getQueue());
    // Register the scene with the Tracer
    tracer.setScene(gpu);

    // Render
    tracer.renderForward(sensor); // films is span/array

    {
        // // Save each sensor image
        auto rgba = Pale::downloadSensorRGBA(deviceSelector.getQueue(), sensor);
        const uint32_t W = sensor.width, H = sensor.height;
        float gamma = 3.2f;
        float exposure = 1.5f;
        std::filesystem::path filePath = "Output" / pointCloudPath.filename().replace_extension("")  /"out.png";
        if (Pale::Utils::savePNGWithToneMap(
                filePath, rgba, W, H,
                exposure,
                gamma,
                false)) {
            Pale::Log::PA_INFO("Wrote PNG image to: {}", filePath.string());
                };


    Pale::Utils::savePFM(filePath.replace_extension(".pfm"), rgba, W, H); // writes RGB, drops A
    }

    // 4) (Optional) load or compute residuals on host, upload pointer
    auto adjoint = calculateAdjointImage("Output/target/out.pfm", deviceSelector.getQueue(), sensor);
    Pale::SensorGPU adjointSensor = Pale::makeSensorsForScene(deviceSelector.getQueue(), buildProducts);
    tracer.renderBackward(adjointSensor, adjoint);                      // PRNG replay adjoint

    std::vector<Pale::float3> gradients(10);

    deviceSelector.getQueue().memcpy(gradients.data(), adjoint.gradient_pk, 10 * sizeof(Pale::float3)).wait();;

    for (size_t instanceIndex = 0; instanceIndex < 1; ++instanceIndex)
        Pale::Log::PA_INFO("Point Gradient: x: {}, y: {}, z: {}, contributions: {}", gradients.at(instanceIndex).x(), gradients.at(instanceIndex).y(), gradients.at(instanceIndex).z(), gradients.at(1).x());

    {
        // // Save each sensor image
        auto rgba = Pale::downloadSensorRGBA(deviceSelector.getQueue(), adjointSensor);
        const uint32_t W = adjointSensor.width, H = adjointSensor.height;
        float gamma = 2.2f;
        float exposure = 0.0f;
        std::filesystem::path filePath = "Output" / pointCloudPath.filename().replace_extension("")  /"adjoint_out.png";
        if (Pale::Utils::savePNGWithToneMap(
                filePath, rgba, W, H,
                exposure,
                gamma,
                false)) {
            Pale::Log::PA_INFO("Wrote PNG image to: {}", filePath.string());
                };
        Pale::Utils::savePFM(filePath.replace_extension(".pfm"), rgba, W, H); // writes RGB, drops A}
    }
    // Write Registry:
    assetManager.registry().save("asset_registry.yaml");

    return 0;
}
