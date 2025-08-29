import Pale.DeviceSelector;
import Pale.Scene;
import Pale.SceneSerializer;
import Pale.Log;
import Pale.Utils.ImageIO;

import Pale.Assets;

import Pale.Render.SceneGPU;
import Pale.Render.PathTracer;
import Pale.Render.PathTracerConfig;
import Pale.Render.Sensors;

#include <memory>
#include <filesystem>
#include <entt/entt.hpp>

#include "Scene/Components.h"

static std::string assetPathOrId(const Pale::AssetRegistry& reg, const Pale::AssetHandle& id) {
    if (auto m = reg.meta(id)) return m->path.string();
    return std::string(id); // fallback if it's not in the registry
}

static void logSceneSummary(Pale::Scene& scene,
                            Pale::AssetManager& am)
{
    auto& reg = am.registry();

    Pale::Log::PA_INFO("===== Scene Summary =====");
    size_t entityCount = 0, meshCount = 0, emissiveCount = 0;

    auto view = scene.getAllEntitiesWith<Pale::IDComponent>();
    for (entt::entity entity : view){
        Pale::Entity e(entity, &scene);
        ++entityCount;

        const char* name = e.getName().c_str();
        bool hasMesh = e.hasComponent<Pale::MeshComponent>();
        bool hasMat  = e.hasComponent<Pale::MaterialComponent>();
        bool hasEm   = e.hasComponent<Pale::AreaLightComponent>();

        Pale::Log::PA_INFO("[Entity] {}", name);

        // Mesh
        if (hasMesh) {
            auto& mc = e.getComponent<Pale::MeshComponent>();
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
            auto& matc = e.getComponent<Pale::MaterialComponent>();
            std::string matLabel = assetPathOrId(reg, matc.material);

            if (auto mat = am.get<Pale::Material>(matc.material)) {
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
            auto& em = e.getComponent<Pale::AreaLightComponent>();
            Pale::Log::PA_INFO("  Emissive radiance=({:.3f},{:.3f},{:.3f})",
                               em.radiance.x, em.radiance.y, em.radiance.z);
        }
    }

    Pale::Log::PA_INFO("===== Totals: entities={} meshes={} emissives={} =====",
                       entityCount, meshCount, emissiveCount);
}


int main() {
    std::filesystem::path workingDirectory = "../Assets";
    std::filesystem::current_path(workingDirectory);

    Pale::Log::init();

    Pale::AssetManager am{256};
    am.enableHotReload(true);
    am.registerLoader<Pale::Mesh>(Pale::AssetType::Mesh,
        std::make_shared<Pale::AssimpMeshLoader>());

    am.registerLoader<Pale::Material>(Pale::AssetType::Material,
        std::make_shared<Pale::YamlMaterialLoader>());

    am.registry().load("asset_registry.yaml");

    // Load in xml file and Create Scene from xml
    std::shared_ptr<Pale::Scene> scene = std::make_shared<Pale::Scene>();
    Pale::AssetProviderFromRegistry assets(am.registry());
    Pale::SceneSerializer serializer(scene, assets);
    serializer.deserialize("cbox.xml");
    logSceneSummary(*scene, am);

    //FInd Sycl Device
    Pale::DeviceSelector deviceSelector;
    // Upload Scene to GPU
    Pale::SceneGPU gpu = Pale::SceneGPU::upload(scene, deviceSelector.getQueue());   // scene only
    auto sensors   = Pale::makeSensorsForScene(deviceSelector.getQueue(), gpu);  // outputs derived from cameras
    // Start a Tracer
    Pale::PathTracer tracer(deviceSelector.getQueue());
    // Register the scene with the Tracer
    tracer.setScene(gpu);

    // Render
    Pale::RenderBatch batch{ .samples = 500'000, .maxBounces = 6, .seed = 0 };
    tracer.renderForward(batch, sensors);          // films is span/array

    // 4) (Optional) load or compute residuals on host, upload pointer
    //    tracer.setResidualsDevice(d_residuals, W*H);  // if you have them
    //    tracer.renderBackward();                      // PRNG replay adjoint


    // Save each sensor image
    for (size_t i = 0; i < sensors.size(); ++i) {
        auto rgba = Pale::downloadSensorRGBA(deviceSelector.getQueue(), sensors[i]);
        const uint32_t W = sensors[i].width, H = sensors[i].height;
        Pale::Utils::savePNG(("out_" + std::to_string(i) + ".png").c_str(), rgba, W, H);
        Pale::Utils::savePFM(("out_" + std::to_string(i) + ".pfm").c_str(), rgba, W, H);
    }
    // Write Registry:
    am.registry().save("asset_registry.yaml");

    return 0;
}
