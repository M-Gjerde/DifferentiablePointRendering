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

static inline float get_f(const py::dict &d, const char *k, float def) {
    if (d.contains(k)) return py::cast<float>(d[k]);
    return def;
}

class PythonRenderer {
public:
    PythonRenderer(const std::string &assetRootDir,
                   const std::string &sceneXml,
                   const std::string &pointCloudFile,
                   const py::dict &settingsDict // <-- accept dict
    ) {
        std::filesystem::current_path(assetRootDir);
        Pale::Log::init();

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
        auto pointCloudHandle = assetIndexer.importPath("PointClouds" / pointCloudPath, Pale::AssetType::PointCloud);

        auto gaussianEntity = scene->createEntity("Gaussian");
        gaussianEntity.addComponent<Pale::PointCloudComponent>().pointCloudID = pointCloudHandle;

        deviceSelector = std::make_unique<Pale::DeviceSelector>();
        Pale::AssetAccessFromManager assetAccessor(*assetManager);

        buildProducts = Pale::SceneBuild::build(scene, assetAccessor, Pale::SceneBuild::BuildOptions());
        sceneGpu = Pale::SceneUpload::upload(buildProducts, deviceSelector->getQueue());

        sensorForward = Pale::makeSensorsForScene(deviceSelector->getQueue(), buildProducts);
        sensorAdjoint = Pale::makeSensorsForScene(deviceSelector->getQueue(), buildProducts);

        Pale::PathTracerSettings settings{}; // defaults from engine

        // Map python keys -> engine settings. Adjust names to your struct.
        // Example mappings based on your dict:
        //   "photons": 1e6, "bounces": 6, "gather_passes": 6,
        //   "adjoint_bounces": 1, "adjoint_passes": 6
        if (!settingsDict.is_none()) {
            // use integer types consistent with your struct
            settings.photonsPerLaunch = get_u64(settingsDict, "photons", settings.photonsPerLaunch);
            settings.maxBounces = get_i(settingsDict, "bounces", settings.maxBounces);
            settings.numForwardPasses = get_i(settingsDict, "forward_passes", settings.numForwardPasses);
            settings.numGatherPasses = get_i(settingsDict, "gather_passes", settings.numGatherPasses);
            settings.maxAdjointBounces = get_i(settingsDict, "adjoint_bounces", settings.maxAdjointBounces);
            settings.adjointSamplesPerPixel = get_i(settingsDict, "adjoint_passes", settings.adjointSamplesPerPixel);
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
    py::array_t<float> render_forward() {
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


   // Replace your current render_backward with this:
py::tuple render_backward(const py::array& targetRgb32f) {
    // --- validate target ---
    py::buffer_info info = targetRgb32f.request();
    if (info.ndim != 3 || info.shape[2] != 3) {
        throw std::runtime_error("target must be HxWx3 float32");
    }
    if (info.itemsize != sizeof(float)) {
        throw std::runtime_error("target dtype must be float32");
    }
    const int64_t Ht = static_cast<int64_t>(info.shape[0]);
    const int64_t Wt = static_cast<int64_t>(info.shape[1]);

    auto q = deviceSelector->getQueue();

    // --- GPU work without the GIL ---
    py::gil_scoped_release release;

    // Optional: upload target to device if your tracer expects it.
    // Example (adjust to your API):
    // pathTracer->setAdjointTarget(static_cast<float*>(info.ptr), Wt, Ht);

    // Run adjoint
    pathTracer->renderBackward(sensorAdjoint);

    // Download gradients (Nx3)
    std::vector<Pale::float3> gradHost(gradCount);
    //q.memcpy(gradHost.data(), gradientPkBuffer, gradCount * sizeof(Pale::float3)).wait();

    // Download adjoint framebuffer RGBA (HxWx4 float)
    auto rgbaHost = Pale::downloadSensorRGBA(q, sensorAdjoint);
    const uint32_t W = sensorAdjoint.width;
    const uint32_t H = sensorAdjoint.height;

    // --- back to Python world ---
    py::gil_scoped_acquire acquire;

    // Wrap gradients as (N,3) float32
    auto gradOwner = new std::vector<Pale::float3>(std::move(gradHost));
    std::vector<ssize_t> gshape{ static_cast<ssize_t>(gradCount), 3 };
    std::vector<ssize_t> gstrides{ static_cast<ssize_t>(sizeof(Pale::float3)), static_cast<ssize_t>(sizeof(float)) };
    py::array gradArray(
        py::buffer_info(
            gradOwner->data(),                   // ptr
            sizeof(float),                       // itemsize
            py::format_descriptor<float>::format(),
            2,                                   // ndim
            gshape,
            gstrides
        ),
        py::capsule(gradOwner, [](void* p){ delete static_cast<std::vector<Pale::float3>*>(p); })
    );

    // Wrap RGBA as (H,W,4) float32
    auto imgOwner = new std::vector<float>(std::move(rgbaHost));
    std::vector<ssize_t> ishape{ static_cast<ssize_t>(H), static_cast<ssize_t>(W), 4 };
    std::vector<ssize_t> istrides{
        static_cast<ssize_t>(W) * 4 * static_cast<ssize_t>(sizeof(float)),
        static_cast<ssize_t>(4) * static_cast<ssize_t>(sizeof(float)),
        static_cast<ssize_t>(sizeof(float))
    };
    py::array imgArray(
        py::buffer_info(
            imgOwner->data(),
            sizeof(float),
            py::format_descriptor<float>::format(),
            3,
            ishape,
            istrides
        ),
        py::capsule(imgOwner, [](void* p){ delete static_cast<std::vector<float>*>(p); })
    );

    return py::make_tuple(gradArray, imgArray);
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

    void set_gaussian_transform(py::tuple translation3, py::tuple rotationQuat4, py::tuple scale3) {
        if (translation3.size() != 3 || rotationQuat4.size() != 4 || scale3.size() != 3) {
            throw std::runtime_error("Expected translation(3), rotation_quat(4), scale(3)");
        }
        const glm::vec3 newTranslation{
            py::cast<float>(translation3[0]),
            py::cast<float>(translation3[1]),
            py::cast<float>(translation3[2])
        };
        // quaternion as (x, y, z, w)
        const glm::quat newRotation{
            py::cast<float>(rotationQuat4[3]),
            py::cast<float>(rotationQuat4[0]),
            py::cast<float>(rotationQuat4[1]),
            py::cast<float>(rotationQuat4[2])
        };
        const glm::vec3 newScale{
            py::cast<float>(scale3[0]),
            py::cast<float>(scale3[1]),
            py::cast<float>(scale3[2])
        };

        // 1) mutate the CPU-side scene
        bool found = false;
        auto view = scene->getAllEntitiesWith<Pale::TransformComponent, Pale::TagComponent>();
        for (auto [entityId, transformComponent, tagComponent]: view.each()) {
            if (tagComponent.tag == std::string("Gaussian")) {
                transformComponent.translation = newTranslation;
                transformComponent.rotation = newRotation;
                transformComponent.scale = newScale;
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("Entity with tag 'Gaussian' not found");
        }

        // 2) push the change to GPU. Easiest safe path: rebuild + re-upload.
        Pale::AssetAccessFromManager assetAccessor(*assetManager);
        buildProducts = Pale::SceneBuild::build(scene, assetAccessor, Pale::SceneBuild::BuildOptions());
        sceneGpu = Pale::SceneUpload::upload(buildProducts, deviceSelector->getQueue());
        sensorForward = Pale::makeSensorsForScene(deviceSelector->getQueue(), buildProducts);
        sensorAdjoint = Pale::makeSensorsForScene(deviceSelector->getQueue(), buildProducts);

        // 3) rebind the scene in the path tracer if needed
        pathTracer->setScene(sceneGpu, buildProducts);
    }

private:
    std::unique_ptr<Pale::AssetManager> assetManager{};
    std::shared_ptr<Pale::Scene> scene{};
    std::unique_ptr<Pale::DeviceSelector> deviceSelector{};

    Pale::SensorGPU sensorForward{};
    Pale::SensorGPU sensorAdjoint{};
    std::unique_ptr<Pale::PathTracer> pathTracer{};

    Pale::SceneBuild::BuildProducts buildProducts{};
    Pale::GPUSceneBuffers sceneGpu{};
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
                     const std::string &, // assetRootDir
                     const std::string &, // sceneXml
                     const std::string &, // pointCloudFile
                     const py::dict & // settingsDict
                 >(),
                 py::arg("assetRootDir"),
                 py::arg("sceneXml") = "cbox_custom.xml",
                 py::arg("pointCloudFile") = "initial.ply",
                 py::arg("settings") = py::dict() // default empty
            )
            .def("render_forward", &PythonRenderer::render_forward)
            .def("render_backward", &PythonRenderer::render_backward,
                 py::arg("targetRgb32f"))
            .def("set_adjoint_spp", &PythonRenderer::set_adjoint_spp)
            .def("get_image_size", &PythonRenderer::get_image_size)
            .def("set_gaussian_transform", &PythonRenderer::set_gaussian_transform,
                 py::arg("translation3"), py::arg("rotation_quat4"), py::arg("scale3"));
}
