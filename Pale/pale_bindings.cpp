// pale_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "Renderer/RenderPackage.h"

#include <memory>
#include <filesystem>
#include <entt/entt.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/string_cast.hpp"

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

#include <filesystem>
#include <memory>

namespace py = pybind11;

// replace your get_u64 with this
static inline uint64_t get_u64(const py::dict& d, const char* k, uint64_t def) {
    if (!d.contains(k)) return def;
    py::int_ v = d[k];
    try {
        // Coerce anything numeric (float, numpy scalar) via Python int()
        return py::int_(v).cast<uint64_t>();
    } catch (const py::error_already_set&) {
        return def; // or throw if you prefer strict
    }
}

static inline int get_i(const py::dict& d, const char* k, int def){
    if (d.contains(k)) return py::cast<int>(d[k]);
    return def;
}
static inline float get_f(const py::dict& d, const char* k, float def){
    if (d.contains(k)) return py::cast<float>(d[k]);
    return def;
}

class PythonRenderer {
public:
    PythonRenderer(const std::string& assetRootDir,
                   const std::string& sceneXml,
                   const std::string& pointCloudFile,
                   const py::dict& settingsDict // <-- accept dict
    ) {
        std::filesystem::current_path(assetRootDir);
        Pale::Log::init();

        assetManager = std::make_unique<Pale::AssetManager>(256);
        assetManager->enableHotReload(true);
        assetManager->registerLoader<Pale::Mesh>(Pale::AssetType::Mesh, std::make_shared<Pale::AssimpMeshLoader>());
        assetManager->registerLoader<Pale::Material>(Pale::AssetType::Material, std::make_shared<Pale::YamlMaterialLoader>());
        assetManager->registerLoader<Pale::PointAsset>(Pale::AssetType::PointCloud, std::make_shared<Pale::PLYPointLoader>());

        assetManager->registry().load("");

        scene = std::make_shared<Pale::Scene>();
        Pale::AssetIndexFromRegistry assetIndexer(assetManager->registry());
        Pale::SceneSerializer sceneSerializer(scene, assetIndexer);
        sceneSerializer.deserialize(sceneXml);

        std::filesystem::path pointCloudPath = pointCloudFile.empty()
            ? std::filesystem::path("initial.ply")
            : std::filesystem::path(pointCloudFile);
        auto pointCloudHandle = assetIndexer.importPath("PointClouds" / pointCloudPath, Pale::AssetType::PointCloud);

        auto gaussianEntity = scene->createEntity("Gaussian");
        gaussianEntity.addComponent<Pale::PointCloudComponent>().pointCloudID = pointCloudHandle;

        deviceSelector = std::make_unique<Pale::DeviceSelector>();
        Pale::AssetAccessFromManager assetAccessor(*assetManager);

        auto buildProducts = Pale::SceneBuild::build(scene, assetAccessor, Pale::SceneBuild::BuildOptions());
        auto sceneGpu = Pale::SceneUpload::upload(buildProducts, deviceSelector->getQueue());

        sensorForward = Pale::makeSensorsForScene(deviceSelector->getQueue(), buildProducts);
        sensorAdjoint = Pale::makeSensorsForScene(deviceSelector->getQueue(), buildProducts);

        Pale::PathTracerSettings settings{};  // defaults from engine

        // Map python keys -> engine settings. Adjust names to your struct.
        // Example mappings based on your dict:
        //   "photons": 1e6, "bounces": 6, "gather_passes": 6,
        //   "adjoint_bounces": 1, "adjoint_passes": 6
        if (!settingsDict.is_none()) {
            // use integer types consistent with your struct
            settings.photonsPerLaunch       = get_u64(settingsDict, "photons",         settings.photonsPerLaunch);
            settings.maxBounces        = get_i  (settingsDict, "bounces",         settings.maxBounces);
            settings.numForwardPasses      = get_i  (settingsDict, "forward_passes",   settings.numForwardPasses);
            settings.numGatherPasses      = get_i  (settingsDict, "gather_passes",   settings.numGatherPasses);
            settings.maxAdjointBounces    = get_i  (settingsDict, "adjoint_bounces", settings.maxAdjointBounces);
            settings.adjointSamplesPerPixel     = get_i  (settingsDict, "adjoint_passes",  settings.adjointSamplesPerPixel);
            // add other keys as needed, e.g., samplesPerPixel, exposure, etc.
        }

        pathTracer = std::make_unique<Pale::PathTracer>(deviceSelector->getQueue(), settings);
        pathTracer->setScene(sceneGpu, buildProducts);
    }

    ~PythonRenderer() {
        if (assetManager) {
            assetManager->registry().save("asset_registry.yaml");
        }
        if (deviceSelector) deviceSelector->getQueue().wait();
    }

    // Forward render -> returns HxWx3 float32 NumPy
    py::array_t<float> render_forward(float exposure, float gamma, bool flipY) {
        py::gil_scoped_release release; // long GPU section

        pathTracer->renderForward(sensorForward);

        auto rgbaHost = Pale::downloadSensorRGBA(deviceSelector->getQueue(), sensorForward);
        const uint32_t imageWidth = sensorForward.width;
        const uint32_t imageHeight = sensorForward.height;

        // Tone-map to linear RGB in [0,1] as float32 HxWx3
        std::vector<float> rgb(imageWidth * imageHeight * 3u);
        //Pale::Utils::toneMapInPlace(rgbaHost, imageWidth, imageHeight, exposure, gamma, flipY);
        for (size_t i = 0, j = 0; i < rgbaHost.size(); i += 4, j += 3) {
            rgb[j + 0] = rgbaHost[i + 0];
            rgb[j + 1] = rgbaHost[i + 1];
            rgb[j + 2] = rgbaHost[i + 2];
        } {
            // // Save each sensor image
            auto rgba = Pale::downloadSensorRGBA(deviceSelector->getQueue(), sensorForward);
            const uint32_t W = sensorForward.width, H = sensorForward.height;
            float gamma = 2.8f;
            float exposure = 5.8f;
            std::filesystem::path filePath = "Output/out_photonmap.png";
            if (Pale::Utils::savePNGWithToneMap(
                filePath, rgba, W, H,
                exposure,
                gamma,
                false)) {
                Pale::Log::PA_INFO("Wrote PNG image to: {}", filePath.string());
            };

            Pale::Utils::savePFM(filePath.replace_extension(".pfm"), rgba, W, H); // writes RGB, drops A
        }

        py::gil_scoped_acquire acquire;
        std::vector<ssize_t> shape{static_cast<ssize_t>(imageHeight), static_cast<ssize_t>(imageWidth), 3};
        std::vector<ssize_t> strides{
            static_cast<ssize_t>(imageWidth * 3 * sizeof(float)),
            static_cast<ssize_t>(3 * sizeof(float)),
            static_cast<ssize_t>(sizeof(float))
        };
        return py::array_t<float>(shape, strides, rgb.data(),
                                  py::capsule(new std::vector<float>(std::move(rgb)),
                                              [](void *p) { delete static_cast<std::vector<float> *>(p); }));
    }


    // Backward render: targetRgb32f is HxWx3 float32; returns Nx3 float32 gradients
    py::array_t<float> render_backward(py::array targetRgb32f, bool flipY) {
        // Upload target to device and compute residuals inside calculateAdjointImage
        py::buffer_info targetInfo = targetRgb32f.request();
        if (targetInfo.ndim != 3 || targetInfo.shape[2] != 3)
            throw std::runtime_error("target must be HxWx3 float32");
        if (targetInfo.itemsize != sizeof(float))
            throw std::runtime_error("target dtype must be float32");

        auto q = deviceSelector->getQueue();

        py::gil_scoped_release release;


        std::vector<Pale::float3> gradientHost(gradCount);
        q.memcpy(gradientHost.data(), gradientPkBuffer, gradCount * sizeof(Pale::float3)).wait();
    }

    // Optional: set renderer settings from Python
    void set_adjoint_spp(float samplesPerPixel) {
        auto settings = pathTracer->getSettings();
        settings.adjointSamplesPerPixel = samplesPerPixel;
        //pathTracer->setSettings(settings);
    }

    std::pair<int, int> get_image_size() const {
        return {static_cast<int>(sensorForward.width), static_cast<int>(sensorForward.height)};
    }

private:
    std::unique_ptr<Pale::AssetManager> assetManager{};
    std::shared_ptr<Pale::Scene> scene{};
    std::unique_ptr<Pale::DeviceSelector> deviceSelector{};

    Pale::SensorGPU sensorForward{};
    Pale::SensorGPU sensorAdjoint{};
    std::unique_ptr<Pale::PathTracer> pathTracer{};

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
              const std::string&, // assetRootDir
              const std::string&, // sceneXml
              const std::string&, // pointCloudFile
              const py::dict&     // settingsDict
            >(),
            py::arg("assetRootDir"),
            py::arg("sceneXml")       = "cbox_custom.xml",
            py::arg("pointCloudFile") = "initial.ply",
            py::arg("settings")       = py::dict()   // default empty
        )
        .def("render_forward",  &PythonRenderer::render_forward,
             py::arg("exposure") = 5.8f, py::arg("gamma") = 2.8f, py::arg("flipY") = true)
        .def("render_backward", &PythonRenderer::render_backward,
             py::arg("targetRgb32f"), py::arg("flipY") = true)
        .def("set_adjoint_spp", &PythonRenderer::set_adjoint_spp)
        .def("get_image_size",  &PythonRenderer::get_image_size);
}
