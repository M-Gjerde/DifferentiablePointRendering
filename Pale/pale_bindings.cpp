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
        int level = 2;
        if (!settingsDict.is_none()) {
            // use integer types consistent with your struct
            level = (get_i(settingsDict, "logging", 2));
        }

        Pale::Log::init(level);

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
        sceneGpu = Pale::SceneUpload::allocateAndUpload(buildProducts, deviceSelector->getQueue());

        sensorForward = Pale::makeSensorsForScene(deviceSelector->getQueue(), buildProducts);
        sensorAdjoint = Pale::makeSensorsForScene(deviceSelector->getQueue(), buildProducts);
        gradients = Pale::makeGradientsForScene(deviceSelector->getQueue(), buildProducts);

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

        // Print summary
        Pale::Log::PA_WARN("=== Renderer Settings ===");
        Pale::Log::PA_WARN("  Photons per launch        : {}", settings.photonsPerLaunch);
        Pale::Log::PA_WARN("  Max bounces               : {}", settings.maxBounces);
        Pale::Log::PA_WARN("  Forward passes            : {}", settings.numForwardPasses);
        Pale::Log::PA_WARN("  Gather passes             : {}", settings.numGatherPasses);
        Pale::Log::PA_WARN("  Adjoint bounces           : {}", settings.maxAdjointBounces);
        Pale::Log::PA_WARN("  Adjoint samples per pixel : {}", settings.adjointSamplesPerPixel);
        Pale::Log::PA_WARN("  Camera Tonemapping : Exposure: {}, Gamma: {}", sensorForward.exposureCorrection,
                           sensorForward.gammaCorrection);
        Pale::Log::PA_WARN("  Adjoint samples per pixel : {}", settings.adjointSamplesPerPixel);

        pathTracer = std::make_unique<Pale::PathTracer>(deviceSelector->getQueue(), settings);
        pathTracer->setScene(sceneGpu, buildProducts);
    }

    ~PythonRenderer() {
        if (assetManager) {
            assetManager->registry().save("asset_registry.yaml");
        }
        if (deviceSelector) deviceSelector->getQueue().wait();
    }


    py::array_t<float> render_forward() {
        py::gil_scoped_release release;

        pathTracer->renderForward(sensorForward);

        std::vector<float> rgbHost = Pale::downloadSensorLDR(deviceSelector->getQueue(), sensorForward);
        const uint32_t W = sensorForward.width;
        const uint32_t H = sensorForward.height;

        // wrap NumPy buffer
        py::gil_scoped_acquire acquire;
        std::vector<ssize_t> shape{(ssize_t) H, (ssize_t) W, 3};
        std::vector<ssize_t> strides{
            (ssize_t) (W * 3 * sizeof(float)),
            (ssize_t) (3 * sizeof(float)),
            (ssize_t) (sizeof(float))
        };

        return py::array_t<float>(
            shape, strides, rgbHost.data(),
            py::capsule(new std::vector<float>(std::move(rgbHost)),
                        [](void *p) { delete static_cast<std::vector<float> *>(p); })
        );
    }


    // Replace your current render_backward with this:
    py::tuple render_backward(const py::array &targetRgb32f) {
        // --- validate target ---
        py::buffer_info info = targetRgb32f.request();
        if (info.ndim != 3 || info.shape[2] != 3) {
            throw std::runtime_error("target must be HxWx3 float32");
        }
        if (info.itemsize != sizeof(float)) {
            throw std::runtime_error("target dtype must be float32");
        }
        const int64_t H = static_cast<int64_t>(info.shape[0]);
        const int64_t W = static_cast<int64_t>(info.shape[1]);
        auto *rgbPtr = static_cast<const float *>(info.ptr);

        auto q = deviceSelector->getQueue();

        // --- GPU work without the GIL ---
        py::gil_scoped_release release;
        // ------------------------------------------------------------
        // 1. Convert HxWx3 → HxWx4 into std::vector<float>
        // ------------------------------------------------------------
        std::vector<float> rgbaTarget;
        rgbaTarget.resize(H * W * 4);

        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                size_t idx3 = (y * W + x) * 3;
                size_t idx4 = (y * W + x) * 4;

                rgbaTarget[idx4 + 0] = rgbPtr[idx3 + 0];
                rgbaTarget[idx4 + 1] = rgbPtr[idx3 + 1];
                rgbaTarget[idx4 + 2] = rgbPtr[idx3 + 2];
                rgbaTarget[idx4 + 3] = 1.0f; // alpha channel
            }
        }

        // ------------------------------------------------------------
        // 2. Upload to the GPU adjoint sensor target buffer
        // ------------------------------------------------------------
        uploadSensorRGBA(q, sensorAdjoint, rgbaTarget);

        pathTracer->renderBackward(sensorAdjoint, gradients);

        const std::size_t pointCount = gradients.numPoints;

        // --- Download all gradient arrays from GPU to host ---

        std::vector<Pale::float3> gradPositionHost(pointCount);
        std::vector<Pale::float3> gradTangentUHost(pointCount);
        std::vector<Pale::float3> gradTangentVHost(pointCount);
        std::vector<Pale::float2> gradScaleHost(pointCount);
        std::vector<Pale::float3> gradColorHost(pointCount);
        std::vector<float> gradOpacityHost(pointCount);
        std::vector<float> gradBetaHost(pointCount);
        std::vector<float> gradShapeHost(pointCount);

        sycl::queue syclQueue = deviceSelector->getQueue();
        if (pointCount > 0) {
            // Only memcpy if the device pointers are non-null
            if (gradients.gradPosition) {
                syclQueue.memcpy(
                    gradPositionHost.data(),
                    gradients.gradPosition,
                    pointCount * sizeof(Pale::float3));
            }
            if (gradients.gradTanU) {
                syclQueue.memcpy(
                    gradTangentUHost.data(),
                    gradients.gradTanU,
                    pointCount * sizeof(Pale::float3));
            }
            if (gradients.gradTanV) {
                syclQueue.memcpy(
                    gradTangentVHost.data(),
                    gradients.gradTanV,
                    pointCount * sizeof(Pale::float3));
            }
            if (gradients.gradScale) {
                syclQueue.memcpy(
                    gradScaleHost.data(),
                    gradients.gradScale,
                    pointCount * sizeof(Pale::float2));
            }
            if (gradients.gradColor) {
                syclQueue.memcpy(
                    gradColorHost.data(),
                    gradients.gradColor,
                    pointCount * sizeof(Pale::float3));
            }
            if (gradients.gradOpacity) {
                syclQueue.memcpy(
                    gradOpacityHost.data(),
                    gradients.gradOpacity,
                    pointCount * sizeof(float));
            }
            if (gradients.gradBeta) {
                syclQueue.memcpy(
                    gradBetaHost.data(),
                    gradients.gradBeta,
                    pointCount * sizeof(float));
            }
            if (gradients.gradShape) {
                syclQueue.memcpy(
                    gradShapeHost.data(),
                    gradients.gradShape,
                    pointCount * sizeof(float));
            }

            syclQueue.wait_and_throw();
        }

        // Download adjoint framebuffer RGBA (HxWx4 float)
        auto rgbaHost = Pale::downloadSensorRGBARaw(syclQueue, sensorAdjoint);
        const std::uint32_t imageWidth = sensorAdjoint.width;
        const std::uint32_t imageHeight = sensorAdjoint.height;

        // --- back to Python world ---
        py::gil_scoped_acquire gilAcquire;

        auto makeFloat3Array = [](std::vector<Pale::float3> &hostVector,
                                  std::size_t count) -> py::array {
            auto *owner = new std::vector<Pale::float3>(std::move(hostVector));
            std::vector<ssize_t> shape{
                static_cast<ssize_t>(count),
                3
            };
            std::vector<ssize_t> strides{
                static_cast<ssize_t>(sizeof(Pale::float3)),
                static_cast<ssize_t>(sizeof(float))
            };

            return py::array(
                py::buffer_info(
                    owner->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    2,
                    shape,
                    strides
                ),
                py::capsule(owner, [](void *pointer) {
                    delete static_cast<std::vector<Pale::float3> *>(pointer);
                })
            );
        };

        auto makeFloat2Array = [](std::vector<Pale::float2> &hostVector,
                                  std::size_t count) -> py::array {
            auto *owner = new std::vector<Pale::float2>(std::move(hostVector));
            std::vector<ssize_t> shape{
                static_cast<ssize_t>(count),
                2
            };
            std::vector<ssize_t> strides{
                static_cast<ssize_t>(sizeof(Pale::float2)),
                static_cast<ssize_t>(sizeof(float))
            };

            return py::array(
                py::buffer_info(
                    owner->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    2,
                    shape,
                    strides
                ),
                py::capsule(owner, [](void *pointer) {
                    delete static_cast<std::vector<Pale::float2> *>(pointer);
                })
            );
        };

        auto makeFloat1Array = [](std::vector<float> &hostVector,
                                  std::size_t count) -> py::array {
            auto *owner = new std::vector<float>(std::move(hostVector));
            std::vector<ssize_t> shape{
                static_cast<ssize_t>(count)
            };
            std::vector<ssize_t> strides{
                static_cast<ssize_t>(sizeof(float))
            };

            return py::array(
                py::buffer_info(
                    owner->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    1,
                    shape,
                    strides
                ),
                py::capsule(owner, [](void *pointer) {
                    delete static_cast<std::vector<float> *>(pointer);
                })
            );
        };

        // Wrap gradients in a Python dict
        py::dict gradientDictionary;
        gradientDictionary["position"] = makeFloat3Array(gradPositionHost, pointCount);
        gradientDictionary["tangent_u"] = makeFloat3Array(gradTangentUHost, pointCount);
        gradientDictionary["tangent_v"] = makeFloat3Array(gradTangentVHost, pointCount);
        gradientDictionary["scale"] = makeFloat2Array(gradScaleHost, pointCount);
        gradientDictionary["color"] = makeFloat3Array(gradColorHost, pointCount);
        gradientDictionary["opacity"] = makeFloat1Array(gradOpacityHost, pointCount);
        gradientDictionary["beta"] = makeFloat1Array(gradBetaHost, pointCount);
        gradientDictionary["shape"] = makeFloat1Array(gradShapeHost, pointCount);

        // Wrap RGBA buffer as (H, W, 4) float32 NumPy array
        auto *rgbaOwner = new std::vector<float>(std::move(rgbaHost));
        std::vector<ssize_t> rgbaShape{
            static_cast<ssize_t>(imageHeight),
            static_cast<ssize_t>(imageWidth),
            4
        };
        std::vector<ssize_t> rgbaStrides{
            static_cast<ssize_t>(imageWidth * 4 * sizeof(float)),
            static_cast<ssize_t>(4 * sizeof(float)),
            static_cast<ssize_t>(sizeof(float))
        };

        py::array rgbaArray(
            py::buffer_info(
                rgbaOwner->data(),
                sizeof(float),
                py::format_descriptor<float>::format(),
                3,
                rgbaShape,
                rgbaStrides
            ),
            py::capsule(rgbaOwner, [](void *pointer) {
                delete static_cast<std::vector<float> *>(pointer);
            })
        );

        // Return (gradients, adjoint_image)
        return py::make_tuple(gradientDictionary, rgbaArray);
    }

    py::dict get_point_parameters() {
        const std::size_t pointCount = buildProducts.points.size();

        // Host copies of all parameters
        std::vector<Pale::float3> positionHost(pointCount);
        std::vector<Pale::float3> tangentUHost(pointCount);
        std::vector<Pale::float3> tangentVHost(pointCount);
        std::vector<Pale::float2> scaleHost(pointCount);
        std::vector<Pale::float3> colorHost(pointCount);
        std::vector<float> opacityHost(pointCount);
        std::vector<float> betaHost(pointCount);
        std::vector<float> shapeHost(pointCount);

        for (std::size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
            const auto &point = buildProducts.points[pointIndex];
            positionHost[pointIndex] = point.position;
            tangentUHost[pointIndex] = point.tanU;
            tangentVHost[pointIndex] = point.tanV;
            scaleHost[pointIndex] = point.scale;
            colorHost[pointIndex] = point.color;
            opacityHost[pointIndex] = point.opacity;
            betaHost[pointIndex] = point.beta;
            shapeHost[pointIndex] = point.shape;
        }

        // Reuse the same makers as in render_backward (or define them once)
        auto makeFloat3Array = [](std::vector<Pale::float3> &hostVector,
                                  std::size_t count) -> py::array {
            auto *owner = new std::vector<Pale::float3>(std::move(hostVector));
            std::vector<ssize_t> shape{
                static_cast<ssize_t>(count),
                3
            };
            std::vector<ssize_t> strides{
                static_cast<ssize_t>(sizeof(Pale::float3)),
                static_cast<ssize_t>(sizeof(float))
            };

            return py::array(
                py::buffer_info(
                    owner->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    2,
                    shape,
                    strides
                ),
                py::capsule(owner, [](void *pointer) {
                    delete static_cast<std::vector<Pale::float3> *>(pointer);
                })
            );
        };

        auto makeFloat2Array = [](std::vector<Pale::float2> &hostVector,
                                  std::size_t count) -> py::array {
            auto *owner = new std::vector<Pale::float2>(std::move(hostVector));
            std::vector<ssize_t> shape{
                static_cast<ssize_t>(count),
                2
            };
            std::vector<ssize_t> strides{
                static_cast<ssize_t>(sizeof(Pale::float2)),
                static_cast<ssize_t>(sizeof(float))
            };

            return py::array(
                py::buffer_info(
                    owner->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    2,
                    shape,
                    strides
                ),
                py::capsule(owner, [](void *pointer) {
                    delete static_cast<std::vector<Pale::float2> *>(pointer);
                })
            );
        };

        auto makeFloat1Array = [](std::vector<float> &hostVector,
                                  std::size_t count) -> py::array {
            auto *owner = new std::vector<float>(std::move(hostVector));
            std::vector<ssize_t> shape{
                static_cast<ssize_t>(count)
            };
            std::vector<ssize_t> strides{
                static_cast<ssize_t>(sizeof(float))
            };

            return py::array(
                py::buffer_info(
                    owner->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    1,
                    shape,
                    strides
                ),
                py::capsule(owner, [](void *pointer) {
                    delete static_cast<std::vector<float> *>(pointer);
                })
            );
        };

        py::dict parameterDictionary;
        parameterDictionary["position"] = makeFloat3Array(positionHost, pointCount);
        parameterDictionary["tangent_u"] = makeFloat3Array(tangentUHost, pointCount);
        parameterDictionary["tangent_v"] = makeFloat3Array(tangentVHost, pointCount);
        parameterDictionary["scale"] = makeFloat2Array(scaleHost, pointCount);
        parameterDictionary["color"] = makeFloat3Array(colorHost, pointCount);
        parameterDictionary["opacity"] = makeFloat1Array(opacityHost, pointCount);
        parameterDictionary["beta"] = makeFloat1Array(betaHost, pointCount);
        parameterDictionary["shape"] = makeFloat1Array(shapeHost, pointCount);

        return parameterDictionary;
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

    void set_point_parameters(const py::dict &parameterDictionary, bool rebuild) {
        if (!parameterDictionary.contains("position")) {
            // nothing to do if we do not get positions/point count
            return;
        }

        // 1) Inspect incoming 'position' to determine desired point count
        py::array positionArray = parameterDictionary["position"].cast<py::array>();
        py::buffer_info positionInfo = positionArray.request();

        if (positionInfo.ndim != 2 || positionInfo.shape[1] != 3) {
            throw std::runtime_error("Expected 'position' to have shape (N,3)");
        }

        std::size_t incomingPointCount =
                static_cast<std::size_t>(positionInfo.shape[0]);

        std::size_t currentPointCount = buildProducts.points.size();

        if (incomingPointCount == 0) {
            // Nothing to render; you can decide whether to clear points or ignore.
            return;
        }

        // 2) If counts differ, resize buildProducts.points to match Python
        if (incomingPointCount != currentPointCount) {
            Pale::Log::PA_INFO(
                "set_point_parameters: resizing buildProducts.points from {} to {}",
                currentPointCount,
                incomingPointCount
            );
            buildProducts.points.resize(incomingPointCount);
        }

        const std::size_t pointCount = buildProducts.points.size();

        // 3) Helpers now use the updated pointCount
        auto assignFloat3FieldFromArray =
                [&](const char *key, Pale::float3 Pale::Point::*memberPointer) {
            if (!parameterDictionary.contains(key)) {
                throw std::runtime_error("New points does not contain key");
            }
            py::array arrayObject = parameterDictionary[key].cast<py::array>();

            py::buffer_info bufferInfo = arrayObject.request();
            if (bufferInfo.ndim != 2 ||
                bufferInfo.shape[0] != static_cast<ssize_t>(pointCount) ||
                bufferInfo.shape[1] != 3) {
                throw std::runtime_error(
                    std::string("Expected '") + key + "' to have shape (N,3)");
            }
            if (bufferInfo.itemsize != sizeof(float)) {
                throw std::runtime_error(
                    std::string("Expected '") + key + "' to be float32");
            }

            auto *dataPointer = static_cast<float *>(bufferInfo.ptr);
            for (std::size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
                const std::size_t baseIndex = pointIndex * 3;
                Pale::float3 value;
                value.x() = dataPointer[baseIndex + 0];
                value.y() = dataPointer[baseIndex + 1];
                value.z() = dataPointer[baseIndex + 2];
                buildProducts.points[pointIndex].*memberPointer = value;
            }
        };

        auto assignFloat2FieldFromArray = [&](const char *key,
                                              auto memberPointer) {
            if (!parameterDictionary.contains(key)) {
                throw std::runtime_error("New points does not contain key");
            }
            py::array arrayObject = parameterDictionary[key].cast<py::array>();

            py::buffer_info bufferInfo = arrayObject.request();
            if (bufferInfo.ndim != 2 ||
                bufferInfo.shape[0] != static_cast<ssize_t>(pointCount) ||
                bufferInfo.shape[1] != 2) {
                throw std::runtime_error(std::string("Expected '") + key +
                                         "' to have shape (N,2)");
            }
            if (bufferInfo.itemsize != sizeof(float)) {
                throw std::runtime_error(std::string("Expected '") + key +
                                         "' to be float32");
            }

            auto *dataPointer = static_cast<float *>(bufferInfo.ptr);
            for (std::size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
                const std::size_t baseIndex = pointIndex * 2;
                Pale::float2 value;
                value.x() = dataPointer[baseIndex + 0];
                value.y() = dataPointer[baseIndex + 1];
                buildProducts.points[pointIndex].*memberPointer = value;
            }
        };

        auto assignFloat1FieldFromArray = [&](const char *key,
                                              auto memberPointer) {
            if (!parameterDictionary.contains(key)) {
                throw std::runtime_error("New points does not contain key");
            }
            py::array arrayObject = parameterDictionary[key].cast<py::array>();

            py::buffer_info bufferInfo = arrayObject.request();
            if (bufferInfo.ndim != 1 ||
                bufferInfo.shape[0] != static_cast<ssize_t>(pointCount)) {
                throw std::runtime_error(std::string("Expected '") + key +
                                         "' to have shape (N,)");
            }
            if (bufferInfo.itemsize != sizeof(float)) {
                throw std::runtime_error(std::string("Expected '") + key +
                                         "' to be float32");
            }

            auto *dataPointer = static_cast<float *>(bufferInfo.ptr);
            for (std::size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
                buildProducts.points[pointIndex].*memberPointer = 1.0f;
            }
        };

        // 4) Assign provided fields
        assignFloat3FieldFromArray("position", &Pale::Point::position);
        assignFloat3FieldFromArray("tangent_u", &Pale::Point::tanU);
        assignFloat3FieldFromArray("tangent_v", &Pale::Point::tanV);
        assignFloat2FieldFromArray("scale", &Pale::Point::scale);
        assignFloat3FieldFromArray("color", &Pale::Point::color);
        assignFloat1FieldFromArray("opacity", &Pale::Point::opacity);

        // 5) Rebuild BVH + GPU buffers if requested
        if (true) {
            // Keep pointCloudRanges coherent with the updated points array.
            // For now we assume a single dynamic point cloud.
            if (!buildProducts.pointCloudRanges.empty()) {
                auto &range = buildProducts.pointCloudRanges[0];

                range.pointCount = static_cast<uint32_t>(buildProducts.points.size());
                Pale::Log::PA_INFO(
                    "set_point_parameters: updated pointCloudRanges[0] to firstPoint = {}, pointCount = {}",
                    range.firstPoint,
                    range.pointCount
                );
            }

            Pale::Log::PA_INFO("Rebuilding BVH and reallocating after topology change");
            Pale::AssetAccessFromManager assetAccessor(*assetManager);

            Pale::SceneBuild::rebuildBVHs(scene, assetAccessor, buildProducts, Pale::SceneBuild::BuildOptions());
            Pale::SceneUpload::allocateOrReallocate(buildProducts, sceneGpu, deviceSelector->getQueue());

            // Recreate gradients to match new point count and resolution
            if (rebuild) {
                Pale::freeGradientsForScene(deviceSelector->getQueue(), gradients);
                gradients = Pale::makeGradientsForScene(deviceSelector->getQueue(), buildProducts);
            }
        }
        reuploadSceneGpu();
    }

    void reuploadSceneGpu() {
        Pale::SceneUpload::upload(buildProducts, sceneGpu, deviceSelector->getQueue());
        pathTracer->setScene(sceneGpu, buildProducts);
    }

    void set_gaussian_transform(py::tuple translation3,
                                py::tuple rotationQuat4,
                                py::tuple scale3,
                                py::tuple color3,
                                float opacity,
                                int index = -1) {
        if (translation3.size() != 3 || rotationQuat4.size() != 4 || scale3.size() != 3) {
            throw std::runtime_error("Expected translation(3), rotation_quat(4), scale(3)");
        }

        const glm::vec3 newTranslation{
            py::cast<float>(translation3[0]), py::cast<float>(translation3[1]),
            py::cast<float>(translation3[2])
        };
        const glm::vec3 newColor{
            py::cast<float>(color3[0]),
            py::cast<float>(color3[1]),
            py::cast<float>(color3[2])
        };

        // quaternion as (x, y, z, w)
        const glm::quat rotationDelta{
            py::cast<float>(rotationQuat4[3]), // w
            py::cast<float>(rotationQuat4[0]), // x
            py::cast<float>(rotationQuat4[1]), // y
            py::cast<float>(rotationQuat4[2]) // z
        };

        const glm::vec2 newScale{
            py::cast<float>(scale3[0]),
            py::cast<float>(scale3[1]),
        };

        Pale::AssetAccessFromManager assetAccessor(*assetManager);
        buildProducts = Pale::SceneBuild::build(
            scene,
            assetAccessor,
            Pale::SceneBuild::BuildOptions()
        );

        // --- apply translation to either one point or all points ---
        if (index < 0) {
            // perturb all points
            const sycl::float3 translationDelta = Pale::glm2sycl(newTranslation);
            for (auto &point: buildProducts.points) {
                point.position += translationDelta;

                // 3) scale vector on the Gaussian (if you store it per point)
                const sycl::float2 scaleDelta = Pale::glm2sycl(newScale);

                // 2) orientation: rotate tanU / tanV by the rotation delta
                glm::vec3 tanUGlm = Pale::sycl2glm(point.tanU);
                glm::vec3 tanVGlm = Pale::sycl2glm(point.tanV);
                tanUGlm = rotationDelta * tanUGlm;
                tanVGlm = rotationDelta * tanVGlm;
                point.tanU = Pale::glm2sycl(tanUGlm);
                point.tanV = Pale::glm2sycl(tanVGlm);

                point.color += Pale::glm2sycl(newColor);
                point.opacity += opacity;
                point.scale *= scaleDelta; // component-wise
            }
        } else {
            if (static_cast<std::size_t>(index) >= buildProducts.points.size()) {
                throw std::out_of_range("set_gaussian_transform: index out of range");
            }
            buildProducts.points[index].position += Pale::glm2sycl(newTranslation);
        }

        Pale::SceneUpload::upload(buildProducts, sceneGpu, deviceSelector->getQueue());
        sensorForward = Pale::makeSensorsForScene(deviceSelector->getQueue(), buildProducts);
        sensorAdjoint = Pale::makeSensorsForScene(deviceSelector->getQueue(), buildProducts);

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
    Pale::PointGradients gradients{};
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
                 py::arg("translation3"), py::arg("rotation_quat4"), py::arg("scale3"), py::arg("color3"),
                 py::arg("opacity"), py::arg("index") = -1).def(
                "get_point_parameters", &PythonRenderer::get_point_parameters)
            .def("set_point_parameters", &PythonRenderer::set_point_parameters, py::arg("parameters"),
                 py::arg("rebuild"));
}
