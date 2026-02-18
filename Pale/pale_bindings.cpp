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
static inline uint64_t get_u64(const py::dict& d, const char* k, uint64_t def) {
    if (!d.contains(k)) return def;
    py::int_ v = d[k];
    try {
        // Coerce anything numeric (float, numpy scalar) via Python int()
        return py::int_(v).cast<uint64_t>();
    }
    catch (const py::error_already_set&) {
        return def; // or throw if you prefer strict
    }
}

static inline int get_i(const py::dict& d, const char* k, int def) {
    if (d.contains(k)) return py::cast<int>(d[k]);
    return def;
}

static inline bool get_b(const py::dict& d, const char* k, bool def) {
    if (d.contains(k)) return py::cast<bool>(d[k]);
    return def;
}

static inline float get_f(const py::dict& d, const char* k, float def) {
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
        auto pointCloudHandle = assetIndexer.importPath(
            "PointClouds" / pointCloudPath,
            Pale::AssetType::PointCloud
        );
        auto gaussianEntity = scene->createEntity("Gaussian");
        gaussianEntity.addComponent<Pale::PointCloudComponent>().pointCloudID = pointCloudHandle;

        // Store for later use in set_point_parameters
        pointCloudAssetHandle = pointCloudHandle;

        deviceSelector = std::make_unique<Pale::DeviceSelector>();
        Pale::AssetAccessFromManager assetAccessor(*assetManager);

        buildProducts = Pale::SceneBuild::build(scene, assetAccessor, Pale::SceneBuild::BuildOptions());
        sceneGpu = Pale::SceneUpload::allocateAndUpload(buildProducts, deviceSelector->getQueue());

        sensorsForward = Pale::makeSensorsForScene(deviceSelector->getQueue(), buildProducts);
        //Pale::float4 color = {0.025, 0.075, 0.165, 1.0f};
        //Pale::setBackgroundColor(deviceSelector->getQueue(), sensorsForward, color);

        sensorsAdjoint = Pale::makeSensorsForScene(deviceSelector->getQueue(), buildProducts, true, true);
        debugImages.resize(sensorsForward.size());
        gradients = Pale::makeGradientsForScene(deviceSelector->getQueue(), buildProducts, debugImages.data());

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
            settings.renderDebugGradientImages =
                get_b(settingsDict, "debug_images", settings.renderDebugGradientImages);
            settings.depthDistortionWeight =
                get_f(settingsDict, "depth_distort_weight", settings.depthDistortionWeight);
            settings.normalConsistencyWeight =
                get_f(settingsDict, "normal_consistency_weight", settings.normalConsistencyWeight);
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
        Pale::Log::PA_WARN("  Depth Distortion Weight   : {}", settings.depthDistortionWeight);
        Pale::Log::PA_WARN("  Normal Consistency Weight : {}", settings.normalConsistencyWeight);

        Pale::Log::PA_WARN("=== Sensors (Forward) ===");
        for (size_t i = 0; i < sensorsForward.size(); ++i) {
            const auto& s = sensorsForward[i];

            Pale::Log::PA_WARN("  --- Sensor {} ---", i);
            Pale::Log::PA_WARN("      Name                : {}", s.name);
            Pale::Log::PA_WARN("      Resolution          : {} x {}", s.width, s.height);

            Pale::Log::PA_WARN("      Camera Position     : ({}, {}, {})",
                               s.camera.pos.x(), s.camera.pos.y(), s.camera.pos.z());
            Pale::Log::PA_WARN("      Exposure / Gamma    : {} / {}",
                               s.exposureCorrection,
                               s.gammaCorrection);
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


    py::dict render_forward(std::string cameraName) {
        // Release GIL while doing GPU work and host copies
        py::gil_scoped_release release;

        std::vector<Pale::SensorGPU> selectedSensors;
        for (const auto& sensor : sensorsForward) {
            if (cameraName == sensor.name) {
                selectedSensors.push_back(sensor);
            }
        }
        if (cameraName.empty())
            selectedSensors = sensorsForward;

        // Render all forward sensors
        pathTracer->renderForward(selectedSensors);

        auto queue = deviceSelector->getQueue();

        struct HostImage {
            std::string cameraName;
            std::uint32_t imageWidth;
            std::uint32_t imageHeight;
            std::vector<float> imageData; // H * W * 4, row-major RGB
            std::vector<float> imageDataRAW; // H * W * 4, row-major RGB
        };

        std::vector<HostImage> hostImages;
        hostImages.reserve(selectedSensors.size());

        for (const auto& sensor : selectedSensors) {
            HostImage hostImage;

            // Safely build a std::string from char[16] (ensure zero-terminated on creation)
            hostImage.cameraName = std::string(sensor.name,
                                               strnlen(sensor.name, sizeof(sensor.name)));

            // Prefer the sensor fields you actually use for allocation
            hostImage.imageWidth = sensor.width;
            hostImage.imageHeight = sensor.height;

            hostImage.imageData = Pale::downloadSensorLDR(queue, sensor);
            hostImage.imageDataRAW = Pale::downloadSensorRGBARAW(queue, sensor);
            hostImages.push_back(std::move(hostImage));
        }

        // Re-acquire GIL to create Python objects
        py::gil_scoped_acquire acquire;

        py::dict result;

        for (auto& hostImage : hostImages) {
            const std::uint32_t imageWidth = hostImage.imageWidth;
            const std::uint32_t imageHeight = hostImage.imageHeight;

            // Shape: (H, W, 4)
            std::vector<ssize_t> shape{
                static_cast<ssize_t>(imageHeight),
                static_cast<ssize_t>(imageWidth),
                static_cast<ssize_t>(4)
            };

            // Strides: row, col, channel (float32)
            std::vector<ssize_t> strides{
                static_cast<ssize_t>(imageWidth * 4 * sizeof(float)),
                static_cast<ssize_t>(4 * sizeof(float)),
                static_cast<ssize_t>(sizeof(float))
            };

            // Move imageData into a heap-allocated vector so NumPy can own it
            auto* ownedBuffer =
                new std::vector<float>(std::move(hostImage.imageData));
            // Move imageData into a heap-allocated vector so NumPy can own it
            auto* ownedBuffer2 =
                new std::vector<float>(std::move(hostImage.imageDataRAW));

            py::array_t<float> numpyImage(
                shape,
                strides,
                ownedBuffer->data(),
                py::capsule(ownedBuffer, [](void* ptr) {
                    delete static_cast<std::vector<float>*>(ptr);
                })
            );
            py::array_t<float> numpyImage2(
                shape,
                strides,
                ownedBuffer2->data(),
                py::capsule(ownedBuffer2, [](void* ptr) {
                    delete static_cast<std::vector<float>*>(ptr);
                })
            );

            result[py::str(hostImage.cameraName)] = std::move(numpyImage);
            result[py::str(hostImage.cameraName + "_raw")] = std::move(numpyImage2);
        }

        return result;
    }

    py::tuple render_backward(const py::dict& targetImagesDictionary) {
        using std::int64_t;
        using std::size_t;

        auto syclQueue = deviceSelector->getQueue();


        struct HostAdjointImage {
            std::string cameraName;
            std::uint32_t imageWidth{};
            std::uint32_t imageHeight{};
            std::vector<float> imageRgbaData; // H * W * 4
        };

        std::vector<HostAdjointImage> hostAdjointImages;
        hostAdjointImages.reserve(sensorsAdjoint.size());

        // Map cameraName -> RGBA target buffer (HxWx4 float)
        std::unordered_map<std::string, std::vector<float>> targetRgbaPerCamera;
        targetRgbaPerCamera.reserve(sensorsAdjoint.size());

        std::vector<Pale::SensorGPU> availableAdjointSensors;
        // ------------------------------------------------------------
        // 1. WITH GIL: read Python dict, convert to RGBA buffers
        // ------------------------------------------------------------
        for (auto& sensor : sensorsAdjoint) {
            // Safe string construction
            std::string cameraName(
                sensor.name,
                strnlen(sensor.name, sizeof(sensor.name))
            );

            if (!targetImagesDictionary.contains(py::str(cameraName))) {
                //Pale::Log::PA_WARN(
                //    "render_backward: missing target image for camera '" +
                //    cameraName + "'"
                //);
                continue;
            }

            py::array targetRgbArray =
                targetImagesDictionary[py::str(cameraName)].cast<py::array>();

            py::buffer_info bufferInfo = targetRgbArray.request();
            if (bufferInfo.ndim != 3 || bufferInfo.shape[2] != 3) {
                throw std::runtime_error(
                    "render_backward: target image for camera '" + cameraName +
                    "' must be HxWx3 float32"
                );
            }
            if (bufferInfo.itemsize != sizeof(float)) {
                throw std::runtime_error(
                    "render_backward: target image for camera '" + cameraName +
                    "' must have dtype float32"
                );
            }

            const int64_t height = static_cast<int64_t>(bufferInfo.shape[0]);
            const int64_t width = static_cast<int64_t>(bufferInfo.shape[1]);

            if (static_cast<std::uint32_t>(width) != sensor.width ||
                static_cast<std::uint32_t>(height) != sensor.height) {
                throw std::runtime_error(
                    "render_backward: resolution mismatch for camera '" +
                    cameraName + "': target image is " + std::to_string(width) +
                    "x" + std::to_string(height) + ", but sensor is " +
                    std::to_string(sensor.width) + "x" +
                    std::to_string(sensor.height)
                );
            }

            const auto* rgbPointer = static_cast<const float*>(bufferInfo.ptr);

            std::vector<float> rgbaTarget;
            rgbaTarget.resize(
                static_cast<size_t>(height) *
                static_cast<size_t>(width) * 4u
            );

            for (int64_t pixelY = 0; pixelY < height; ++pixelY) {
                for (int64_t pixelX = 0; pixelX < width; ++pixelX) {
                    const size_t rgbIndex =
                        static_cast<size_t>((pixelY * width + pixelX) * 3);
                    const size_t rgbaIndex =
                        static_cast<size_t>((pixelY * width + pixelX) * 4);

                    rgbaTarget[rgbaIndex + 0] = rgbPointer[rgbIndex + 0];
                    rgbaTarget[rgbaIndex + 1] = rgbPointer[rgbIndex + 1];
                    rgbaTarget[rgbaIndex + 2] = rgbPointer[rgbIndex + 2];
                    rgbaTarget[rgbaIndex + 3] = 1.0f;
                }
            }

            targetRgbaPerCamera.emplace(std::move(cameraName), std::move(rgbaTarget));
            availableAdjointSensors.emplace_back(sensor);
        }

        // ------------------------------------------------------------
        // 2. WITHOUT GIL: upload targets, run adjoint, download gradients
        // ------------------------------------------------------------
        py::gil_scoped_release release;

        // 2a. Upload RGBA targets per sensor
        for (auto& sensor : availableAdjointSensors) {
            std::string cameraName(
                sensor.name,
                strnlen(sensor.name, sizeof(sensor.name))
            );
            auto it = targetRgbaPerCamera.find(cameraName);
            if (it == targetRgbaPerCamera.end()) {
                continue; // should not happen given checks above
            }
            uploadSensorRGBA(syclQueue, sensor, it->second);
        }

        // 2b. Run backward pass (re-enable when ready)
        pathTracer->renderBackward(availableAdjointSensors, gradients, debugImages.data());

        const std::size_t pointCount = gradients.numPoints;

        std::vector<Pale::float3> gradPositionHost(pointCount);
        std::vector<Pale::float3> gradTangentUHost(pointCount);
        std::vector<Pale::float3> gradTangentVHost(pointCount);
        std::vector<Pale::float2> gradScaleHost(pointCount);
        std::vector<Pale::float3> gradColorHost(pointCount);
        std::vector<float> gradOpacityHost(pointCount);
        std::vector<float> gradBetaHost(pointCount);
        std::vector<float> gradShapeHost(pointCount);

        if (pointCount > 0) {
            if (gradients.gradPosition) {
                syclQueue.memcpy(
                    gradPositionHost.data(),
                    gradients.gradPosition,
                    pointCount * sizeof(Pale::float3)
                );
            }
            if (gradients.gradTanU) {
                syclQueue.memcpy(
                    gradTangentUHost.data(),
                    gradients.gradTanU,
                    pointCount * sizeof(Pale::float3)
                );
            }
            if (gradients.gradTanV) {
                syclQueue.memcpy(
                    gradTangentVHost.data(),
                    gradients.gradTanV,
                    pointCount * sizeof(Pale::float3)
                );
            }
            if (gradients.gradScale) {
                syclQueue.memcpy(
                    gradScaleHost.data(),
                    gradients.gradScale,
                    pointCount * sizeof(Pale::float2)
                );
            }
            if (gradients.gradAlbedo) {
                syclQueue.memcpy(
                    gradColorHost.data(),
                    gradients.gradAlbedo,
                    pointCount * sizeof(Pale::float3)
                );
            }
            if (gradients.gradOpacity) {
                syclQueue.memcpy(
                    gradOpacityHost.data(),
                    gradients.gradOpacity,
                    pointCount * sizeof(float)
                );
            }
            if (gradients.gradBeta) {
                syclQueue.memcpy(
                    gradBetaHost.data(),
                    gradients.gradBeta,
                    pointCount * sizeof(float)
                );
            }
            if (gradients.gradShape) {
                syclQueue.memcpy(
                    gradShapeHost.data(),
                    gradients.gradShape,
                    pointCount * sizeof(float)
                );
            }

            syclQueue.wait_and_throw();
        }

        // 2c. Download adjoint images per sensor
        for (auto& sensor : availableAdjointSensors) {
            HostAdjointImage hostImage;
            hostImage.cameraName = std::string(
                sensor.name,
                strnlen(sensor.name, sizeof(sensor.name))
            );
            hostImage.imageWidth = sensor.width;
            hostImage.imageHeight = sensor.height;
            hostImage.imageRgbaData =
                Pale::downloadSensorRGBARAW(syclQueue, sensor);

            hostAdjointImages.push_back(std::move(hostImage));
        }

        // ------------------------------------------------------------
        // 3. WITH GIL: wrap gradients and images into NumPy/Python objects
        // ------------------------------------------------------------
        py::gil_scoped_acquire gilAcquire;

        auto makeFloat3Array =
            [](std::vector<Pale::float3>& hostVector, std::size_t elementCount) -> py::array {
            auto* ownedVector = new std::vector<Pale::float3>(std::move(hostVector));
            std::vector<ssize_t> arrayShape{
                static_cast<ssize_t>(elementCount),
                3
            };
            std::vector<ssize_t> arrayStrides{
                static_cast<ssize_t>(sizeof(Pale::float3)),
                static_cast<ssize_t>(sizeof(float))
            };

            return py::array(
                py::buffer_info(
                    ownedVector->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    2,
                    arrayShape,
                    arrayStrides
                ),
                py::capsule(ownedVector, [](void* pointer) {
                    delete static_cast<std::vector<Pale::float3>*>(pointer);
                })
            );
        };

        auto makeFloat2Array =
            [](std::vector<Pale::float2>& hostVector, std::size_t elementCount) -> py::array {
            auto* ownedVector = new std::vector<Pale::float2>(std::move(hostVector));
            std::vector<ssize_t> arrayShape{
                static_cast<ssize_t>(elementCount),
                2
            };
            std::vector<ssize_t> arrayStrides{
                static_cast<ssize_t>(sizeof(Pale::float2)),
                static_cast<ssize_t>(sizeof(float))
            };

            return py::array(
                py::buffer_info(
                    ownedVector->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    2,
                    arrayShape,
                    arrayStrides
                ),
                py::capsule(ownedVector, [](void* pointer) {
                    delete static_cast<std::vector<Pale::float2>*>(pointer);
                })
            );
        };

        auto makeFloat1Array =
            [](std::vector<float>& hostVector, std::size_t elementCount) -> py::array {
            auto* ownedVector = new std::vector<float>(std::move(hostVector));
            std::vector<ssize_t> arrayShape{
                static_cast<ssize_t>(elementCount)
            };
            std::vector<ssize_t> arrayStrides{
                static_cast<ssize_t>(sizeof(float))
            };

            return py::array(
                py::buffer_info(
                    ownedVector->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    1,
                    arrayShape,
                    arrayStrides
                ),
                py::capsule(ownedVector, [](void* pointer) {
                    delete static_cast<std::vector<float>*>(pointer);
                })
            );
        };

        py::dict gradientDictionary;
        gradientDictionary["position"] = makeFloat3Array(gradPositionHost, pointCount);
        gradientDictionary["tangent_u"] = makeFloat3Array(gradTangentUHost, pointCount);
        gradientDictionary["tangent_v"] = makeFloat3Array(gradTangentVHost, pointCount);
        gradientDictionary["scale"] = makeFloat2Array(gradScaleHost, pointCount);
        gradientDictionary["albedo"] = makeFloat3Array(gradColorHost, pointCount);
        gradientDictionary["opacity"] = makeFloat1Array(gradOpacityHost, pointCount);
        gradientDictionary["beta"] = makeFloat1Array(gradBetaHost, pointCount);
        gradientDictionary["shape"] = makeFloat1Array(gradShapeHost, pointCount);

        // Top-level container for all images
        py::dict adjointImagesDictionary;

        auto makeRgbaImageArray =
            [](std::vector<float>& imageBuffer,
               std::uint32_t imageWidth,
               std::uint32_t imageHeight) -> py::array {
            auto* ownedImageBuffer =
                new std::vector<float>(std::move(imageBuffer));

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

            return py::array(
                py::buffer_info(
                    ownedImageBuffer->data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    3,
                    rgbaShape,
                    rgbaStrides
                ),
                py::capsule(ownedImageBuffer, [](void* pointer) {
                    delete static_cast<std::vector<float>*>(pointer);
                })
            );
        };


        // 3a. Main adjoint source images per camera
        py::dict adjointSourceDict;
        for (auto& hostImage : hostAdjointImages) {
            py::array rgbaArray = makeRgbaImageArray(
                hostImage.imageRgbaData,
                hostImage.imageWidth,
                hostImage.imageHeight
            );

            adjointSourceDict[py::str(hostImage.cameraName)] = std::move(rgbaArray);
        }
        adjointImagesDictionary["adjoint_source"] = std::move(adjointSourceDict);


        if (pathTracer->getSettings().renderDebugGradientImages) {
            py::dict debugPerCameraDict;

            for (std::size_t i = 0; i < sensorsAdjoint.size(); ++i) {
                const auto& sensor = sensorsAdjoint[i];

                Pale::DebugGradientImagesHost debugImagesHost =
                    Pale::downloadDebugGradientImages(
                        deviceSelector->getQueue(),
                        sensor,
                        debugImages[i]
                    );

                const std::uint32_t imageWidth = sensor.width;
                const std::uint32_t imageHeight = sensor.height;

                // Per-camera dict: {"position": img, "tangent_u": img, ...}
                py::dict cameraDebugDict;

                // Position gradient image
                if (!debugImagesHost.positionX.empty()) {
                    cameraDebugDict["position_x"] = makeRgbaImageArray(
                        debugImagesHost.positionX,
                        imageWidth,
                        imageHeight
                    );
                }
                // Position gradient image
                if (!debugImagesHost.positionY.empty()) {
                    cameraDebugDict["position_y"] = makeRgbaImageArray(
                        debugImagesHost.positionY,
                        imageWidth,
                        imageHeight
                    );
                }
                // Position gradient image
                if (!debugImagesHost.positionZ.empty()) {
                    cameraDebugDict["position_z"] = makeRgbaImageArray(
                        debugImagesHost.positionZ,
                        imageWidth,
                        imageHeight
                    );
                }

                // Tangent U gradient image
                if (!debugImagesHost.rotation.empty()) {
                    cameraDebugDict["rotation"] = makeRgbaImageArray(
                        debugImagesHost.rotation,
                        imageWidth,
                        imageHeight
                    );
                }


                // Scale gradient image
                if (!debugImagesHost.scale.empty()) {
                    cameraDebugDict["scale"] = makeRgbaImageArray(
                        debugImagesHost.scale,
                        imageWidth,
                        imageHeight
                    );
                }

                // Color gradient image
                if (!debugImagesHost.albedo.empty()) {
                    cameraDebugDict["albedo"] = makeRgbaImageArray(
                        debugImagesHost.albedo,
                        imageWidth,
                        imageHeight
                    );
                }

                // Opacity gradient image
                if (!debugImagesHost.opacity.empty()) {
                    cameraDebugDict["opacity"] = makeRgbaImageArray(
                        debugImagesHost.opacity,
                        imageWidth,
                        imageHeight
                    );
                }

                // Beta gradient image
                if (!debugImagesHost.beta.empty()) {
                    cameraDebugDict["beta"] = makeRgbaImageArray(
                        debugImagesHost.beta,
                        imageWidth,
                        imageHeight
                    );
                }

                if (cameraDebugDict.size() > 0) {
                    std::string cameraName(sensor.name,
                                           strnlen(sensor.name, sizeof(sensor.name)));
                    debugPerCameraDict[py::str(cameraName)] = std::move(cameraDebugDict);
                }
            }

            adjointImagesDictionary["debug"] = std::move(debugPerCameraDict);
        }


        return py::make_tuple(gradientDictionary, adjointImagesDictionary);
    }


    py::dict get_point_parameters() {
        const std::size_t pointCount = buildProducts.points.size();

        // Host copies of all parameters
        std::vector<Pale::float3> positionHost(pointCount);
        std::vector<Pale::float3> tangentUHost(pointCount);
        std::vector<Pale::float3> tangentVHost(pointCount);
        std::vector<Pale::float2> scaleHost(pointCount);
        std::vector<Pale::float3> albedoHost(pointCount);
        std::vector<float> opacityHost(pointCount);
        std::vector<float> betaHost(pointCount);
        std::vector<float> shapeHost(pointCount);

        for (std::size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
            const auto& point = buildProducts.points[pointIndex];
            positionHost[pointIndex] = point.position;
            tangentUHost[pointIndex] = point.tanU;
            tangentVHost[pointIndex] = point.tanV;
            scaleHost[pointIndex] = point.scale;
            albedoHost[pointIndex] = point.albedo;
            opacityHost[pointIndex] = point.opacity;
            betaHost[pointIndex] = point.beta;
            shapeHost[pointIndex] = point.shape;
        }

        // Reuse the same makers as in render_backward (or define them once)
        auto makeFloat3Array = [](std::vector<Pale::float3>& hostVector,
                                  std::size_t count) -> py::array {
            auto* owner = new std::vector<Pale::float3>(std::move(hostVector));
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
                py::capsule(owner, [](void* pointer) {
                    delete static_cast<std::vector<Pale::float3>*>(pointer);
                })
            );
        };

        auto makeFloat2Array = [](std::vector<Pale::float2>& hostVector,
                                  std::size_t count) -> py::array {
            auto* owner = new std::vector<Pale::float2>(std::move(hostVector));
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
                py::capsule(owner, [](void* pointer) {
                    delete static_cast<std::vector<Pale::float2>*>(pointer);
                })
            );
        };

        auto makeFloat1Array = [](std::vector<float>& hostVector,
                                  std::size_t count) -> py::array {
            auto* owner = new std::vector<float>(std::move(hostVector));
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
                py::capsule(owner, [](void* pointer) {
                    delete static_cast<std::vector<float>*>(pointer);
                })
            );
        };

        py::dict parameterDictionary;
        parameterDictionary["position"] = makeFloat3Array(positionHost, pointCount);
        parameterDictionary["tangent_u"] = makeFloat3Array(tangentUHost, pointCount);
        parameterDictionary["tangent_v"] = makeFloat3Array(tangentVHost, pointCount);
        parameterDictionary["scale"] = makeFloat2Array(scaleHost, pointCount);
        parameterDictionary["albedo"] = makeFloat3Array(albedoHost, pointCount);
        parameterDictionary["opacity"] = makeFloat1Array(opacityHost, pointCount);
        parameterDictionary["beta"] = makeFloat1Array(betaHost, pointCount);
        parameterDictionary["shape"] = makeFloat1Array(shapeHost, pointCount);

        return parameterDictionary;
    }


    void apply_point_optimization(const py::dict& parameterDictionary) {
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

        const std::size_t incomingPointCount =
            static_cast<std::size_t>(positionInfo.shape[0]);

        const std::size_t currentPointCount = buildProducts.points.size();

        if (incomingPointCount == 0) {
            // Nothing to render; ignore.
            return;
        }

        // IMPORTANT: this function is for optimization only, not topology changes.
        if (incomingPointCount != currentPointCount) {
            Pale::Log::PA_ERROR(
                "apply_point_optimization: incoming point count {} does not "
                "match current buildProducts.points size {}. This function does "
                "not handle topology changes.",
                incomingPointCount,
                currentPointCount
            );
            throw std::runtime_error(
                "apply_point_optimization expects consistent point count; "
                "use densification API for adding/removing points."
            );
        }

        const std::size_t pointCount = currentPointCount;

        // 2) Helpers for writing into buildProducts.points
        auto assignFloat3FieldFromArray =
            [&](const char* key, Pale::float3 Pale::Point::* memberPointer) {
            if (!parameterDictionary.contains(key)) {
                throw std::runtime_error("New points dictionary does not contain key: " + std::string(key));
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

            auto* dataPointer = static_cast<float*>(bufferInfo.ptr);
            for (std::size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
                const std::size_t baseIndex = pointIndex * 3;
                Pale::float3 value;
                value.x() = dataPointer[baseIndex + 0];
                value.y() = dataPointer[baseIndex + 1];
                value.z() = dataPointer[baseIndex + 2];
                buildProducts.points[pointIndex].*memberPointer = value;
            }
        };

        auto assignFloat2FieldFromArray =
            [&](const char* key, Pale::float2 Pale::Point::* memberPointer) {
            if (!parameterDictionary.contains(key)) {
                throw std::runtime_error("New points dictionary does not contain key: " + std::string(key));
            }
            py::array arrayObject = parameterDictionary[key].cast<py::array>();

            py::buffer_info bufferInfo = arrayObject.request();
            if (bufferInfo.ndim != 2 ||
                bufferInfo.shape[0] != static_cast<ssize_t>(pointCount) ||
                bufferInfo.shape[1] != 2) {
                throw std::runtime_error(
                    std::string("Expected '") + key + "' to have shape (N,2)");
            }
            if (bufferInfo.itemsize != sizeof(float)) {
                throw std::runtime_error(
                    std::string("Expected '") + key + "' to be float32");
            }

            auto* dataPointer = static_cast<float*>(bufferInfo.ptr);
            for (std::size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
                const std::size_t baseIndex = pointIndex * 2;
                Pale::float2 value;
                value.x() = dataPointer[baseIndex + 0];
                value.y() = dataPointer[baseIndex + 1];
                buildProducts.points[pointIndex].*memberPointer = value;
            }
        };

        auto assignFloat1FieldFromArray =
            [&](const char* key, float Pale::Point::* memberPointer) {
            if (!parameterDictionary.contains(key)) {
                throw std::runtime_error("New points dictionary does not contain key: " + std::string(key));
            }
            py::array arrayObject = parameterDictionary[key].cast<py::array>();

            py::buffer_info bufferInfo = arrayObject.request();
            if (bufferInfo.ndim != 1 ||
                bufferInfo.shape[0] != static_cast<ssize_t>(pointCount)) {
                throw std::runtime_error(
                    std::string("Expected '") + key + "' to have shape (N,)");
            }
            if (bufferInfo.itemsize != sizeof(float)) {
                throw std::runtime_error(
                    std::string("Expected '") + key + "' to be float32");
            }

            auto* dataPointer = static_cast<float*>(bufferInfo.ptr);
            for (std::size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
                buildProducts.points[pointIndex].*memberPointer = dataPointer[pointIndex];
            }
        };

        // 3) Assign provided fields into buildProducts
        assignFloat3FieldFromArray("position", &Pale::Point::position);
        assignFloat3FieldFromArray("tangent_u", &Pale::Point::tanU);
        assignFloat3FieldFromArray("tangent_v", &Pale::Point::tanV);
        assignFloat2FieldFromArray("scale", &Pale::Point::scale);
        assignFloat3FieldFromArray("albedo", &Pale::Point::albedo);
        assignFloat1FieldFromArray("opacity", &Pale::Point::opacity);
        assignFloat1FieldFromArray("beta", &Pale::Point::beta);

        // 4) Mirror changes back to the underlying point cloud asset
        if (!assetManager) {
            Pale::Log::PA_WARN("apply_point_optimization: assetManager is null, "
                "skipping asset point cloud update.");
        }
        else {
            auto pointAssetSharedPtr = assetManager->get<Pale::PointAsset>(pointCloudAssetHandle);
            if (!pointAssetSharedPtr) {
                Pale::Log::PA_ERROR("apply_point_optimization: failed to get PointAsset for handle {}",
                                    std::string(pointCloudAssetHandle));
            }
            else {
                Pale::PointAsset& pointAsset = *pointAssetSharedPtr;
                if (pointAsset.points.empty()) {
                    Pale::Log::PA_WARN("apply_point_optimization: PointAsset has no PointGeometry blocks");
                }
                else {
                    Pale::PointGeometry& pointGeometry = pointAsset.points.front();

                    // Ensure the asset geometry has at least pointCount entries
                    if (pointGeometry.positions.size() != pointCount ||
                        pointGeometry.tanU.size() != pointCount ||
                        pointGeometry.tanV.size() != pointCount ||
                        pointGeometry.scales.size() != pointCount ||
                        pointGeometry.albedos.size() != pointCount ||
                        pointGeometry.betas.size() != pointCount ||
                        pointGeometry.opacities.size() != pointCount) {
                        Pale::Log::PA_ERROR(
                            "apply_point_optimization: PointGeometry size mismatch. "
                            "positions={}, tanU={}, tanV={}, scales={}, albedos={}, opacities={}, betas={}, expected={}",
                            pointGeometry.positions.size(),
                            pointGeometry.tanU.size(),
                            pointGeometry.tanV.size(),
                            pointGeometry.scales.size(),
                            pointGeometry.albedos.size(),
                            pointGeometry.opacities.size(),
                            pointGeometry.betas.size(),
                            pointCount
                        );
                        throw std::runtime_error(
                            "PointGeometry size mismatch when applying point optimization.");
                    }

                    for (std::size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
                        const Pale::Point& optimizedPoint = buildProducts.points[pointIndex];

                        pointGeometry.positions[pointIndex] = Pale::sycl2glm(optimizedPoint.position);
                        pointGeometry.tanU[pointIndex] = Pale::sycl2glm(optimizedPoint.tanU);
                        pointGeometry.tanV[pointIndex] = Pale::sycl2glm(optimizedPoint.tanV);
                        pointGeometry.scales[pointIndex] = Pale::sycl2glm(optimizedPoint.scale);
                        pointGeometry.albedos[pointIndex] = Pale::sycl2glm(optimizedPoint.albedo);
                        pointGeometry.opacities[pointIndex] = optimizedPoint.opacity;
                        // If you also keep beta/shape in the asset, mirror them here as well:
                        pointGeometry.betas[pointIndex] = optimizedPoint.beta;
                        // pointGeometry.shapes[pointIndex]    = optimizedPoint.shape;
                    }

                    Pale::Log::PA_INFO(
                        "apply_point_optimization: synchronized {} optimized points back into PointAsset.",
                        pointCount
                    );
                }
            }
        }

        //rebuild_bvh();
        //// 5) Upload updated buildProducts to GPU, no BVH rebuild
        Pale::SceneUpload::upload(buildProducts, sceneGpu, deviceSelector->getQueue());
        pathTracer->setScene(sceneGpu, buildProducts);
    }

    void rebuild_bvh() {
        Pale::AssetAccessFromManager assetAccessor(*assetManager);

        buildProducts = Pale::SceneBuild::build(
            scene,
            assetAccessor,
            Pale::SceneBuild::BuildOptions()
        );


        Pale::SceneUpload::uploadOrReallocate(
            buildProducts,
            sceneGpu,
            deviceSelector->getQueue()
        );

        Pale::freeGradientsForScene(deviceSelector->getQueue(), gradients);
        Pale::freeDebugImagesForScene(deviceSelector->getQueue(), debugImages.data(), debugImages.size());
        gradients = Pale::makeGradientsForScene(deviceSelector->getQueue(), buildProducts, debugImages.data());

        pathTracer->setScene(sceneGpu, buildProducts);
    }

    void remove_points(const py::dict& parameterDictionary) {
        // -----------------------------------------------------------------
        // 0) Check required input
        // -----------------------------------------------------------------
        if (!parameterDictionary.contains("indices")) {
            throw std::runtime_error("remove_points: expected key 'indices' (1D int32 / int64 array)");
        }

        py::array indicesArray = parameterDictionary["indices"].cast<py::array>();
        py::buffer_info indicesInfo = indicesArray.request();

        if (indicesInfo.ndim != 1) {
            throw std::runtime_error("remove_points: 'indices' must be a 1D array");
        }

        const std::size_t removeCount = static_cast<std::size_t>(indicesInfo.shape[0]);
        if (removeCount == 0) {
            Pale::Log::PA_INFO("remove_points: no indices provided, nothing to remove.");
            return;
        }

        const void* indicesVoidPointer = indicesInfo.ptr;
        const bool indicesAreInt64 = (indicesInfo.itemsize == sizeof(std::int64_t));
        const bool indicesAreInt32 = (indicesInfo.itemsize == sizeof(std::int32_t));
        if (!indicesAreInt32 && !indicesAreInt64) {
            throw std::runtime_error("remove_points: 'indices' must have dtype int32 or int64");
        }

        // -----------------------------------------------------------------
        // 1) Get point cloud asset
        // -----------------------------------------------------------------
        if (!assetManager) {
            throw std::runtime_error("remove_points: assetManager is null");
        }

        auto pointAssetSharedPtr = assetManager->get<Pale::PointAsset>(pointCloudAssetHandle);
        if (!pointAssetSharedPtr) {
            throw std::runtime_error("remove_points: failed to get PointAsset for dynamic point cloud");
        }

        Pale::PointAsset& pointAsset = *pointAssetSharedPtr;
        if (pointAsset.points.empty()) {
            throw std::runtime_error("remove_points: PointAsset has no PointGeometry blocks");
        }

        Pale::PointGeometry& pointGeometry = pointAsset.points.front();
        const std::size_t currentPointCount = pointGeometry.positions.size();

        if (currentPointCount == 0) {
            Pale::Log::PA_INFO("remove_points: current point cloud is empty, nothing to remove.");
            return;
        }

        // -----------------------------------------------------------------
        // 2) Build keep-mask from indices to remove
        // -----------------------------------------------------------------
        std::vector<char> keepMask(currentPointCount, 1);

        auto markIndexForRemoval = [&](std::size_t removalIndex) {
            if (removalIndex >= currentPointCount) {
                throw std::out_of_range("remove_points: index out of range");
            }
            keepMask[removalIndex] = 0;
        };

        if (indicesAreInt64) {
            const auto* indexData = static_cast<const std::int64_t*>(indicesVoidPointer);
            for (std::size_t removeIndex = 0; removeIndex < removeCount; ++removeIndex) {
                const std::int64_t value = indexData[removeIndex];
                if (value < 0) {
                    throw std::out_of_range("remove_points: negative index is not allowed");
                }
                markIndexForRemoval(static_cast<std::size_t>(value));
            }
        }
        else {
            const auto* indexData = static_cast<const std::int32_t*>(indicesVoidPointer);
            for (std::size_t removeIndex = 0; removeIndex < removeCount; ++removeIndex) {
                const std::int32_t value = indexData[removeIndex];
                if (value < 0) {
                    throw std::out_of_range("remove_points: negative index is not allowed");
                }
                markIndexForRemoval(static_cast<std::size_t>(value));
            }
        }

        std::size_t newPointCount = 0;
        for (char keepFlag : keepMask) {
            if (keepFlag) {
                ++newPointCount;
            }
        }

        if (newPointCount == 0) {
            Pale::Log::PA_WARN(
                "remove_points: all points would be removed ({} total). "
                "Proceeding, but make sure your pipeline handles the empty case.",
                currentPointCount
            );
        }

        // -----------------------------------------------------------------
        // 3) Filter all attribute arrays in PointGeometry
        // -----------------------------------------------------------------
        auto filterVectorInPlace = [&](auto& vectorAttribute) {
            using AttributeType = typename std::decay_t<decltype(vectorAttribute)>::value_type;
            std::vector<AttributeType> filteredVector;
            filteredVector.reserve(newPointCount);

            for (std::size_t pointIndex = 0; pointIndex < currentPointCount; ++pointIndex) {
                if (keepMask[pointIndex]) {
                    filteredVector.push_back(vectorAttribute[pointIndex]);
                }
            }

            vectorAttribute.swap(filteredVector);
        };

        filterVectorInPlace(pointGeometry.positions);
        filterVectorInPlace(pointGeometry.tanU);
        filterVectorInPlace(pointGeometry.tanV);
        filterVectorInPlace(pointGeometry.scales);
        filterVectorInPlace(pointGeometry.albedos);
        filterVectorInPlace(pointGeometry.opacities);
        filterVectorInPlace(pointGeometry.shapes);
        filterVectorInPlace(pointGeometry.betas);

        Pale::Log::PA_INFO(
            "remove_points: removed {} points, new point count = {}",
            currentPointCount - newPointCount,
            newPointCount
        );

        // -----------------------------------------------------------------
        // 4) Rebuild BVH and GPU buffers from updated asset (renderer is ground truth)
        // -----------------------------------------------------------------
        rebuild_bvh();
    }


    void add_new_points(const py::dict& parameterDictionary) {
        // ---------------------------------------------------------------------
        // 0) Get point cloud asset
        // ---------------------------------------------------------------------
        auto pointAssetSharedPtr = assetManager->get<Pale::PointAsset>(pointCloudAssetHandle);
        if (!pointAssetSharedPtr) {
            throw std::runtime_error("add_new_points: failed to get PointAsset for dynamic point cloud");
        }

        Pale::PointAsset& pointAsset = *pointAssetSharedPtr;
        if (pointAsset.points.empty()) {
            throw std::runtime_error("add_new_points: PointAsset has no PointGeometry blocks");
        }

        Pale::PointGeometry& pointGeometry = pointAsset.points.front();

        // ---------------------------------------------------------------------
        // 1) Read "new" points only (position / tangent_u / tangent_v / scale / albedo)
        // ---------------------------------------------------------------------
        if (!parameterDictionary.contains("new")) {
            Pale::Log::PA_INFO("add_new_points: no 'new' block provided, nothing to append.");
            return;
        }

        py::dict newDict = parameterDictionary["new"].cast<py::dict>();

        auto getArray = [&](const char* key) -> py::array {
            if (!newDict.contains(key)) {
                throw std::runtime_error(std::string("add_new_points: missing key 'new.") + key + "'");
            }
            return newDict[key].cast<py::array>();
        };

        py::array positionArray = getArray("position");
        py::array tangentUArray = getArray("tangent_u");
        py::array tangentVArray = getArray("tangent_v");
        py::array scaleArray = getArray("scale");
        py::array albedoArray = getArray("albedo");
        py::array opacityArray = getArray("opacity");
        py::array betaArray = getArray("beta");

        py::buffer_info positionInfo = positionArray.request();
        py::buffer_info tangentUInfo = tangentUArray.request();
        py::buffer_info tangentVInfo = tangentVArray.request();
        py::buffer_info scaleInfo = scaleArray.request();
        py::buffer_info albedoInfo = albedoArray.request();
        py::buffer_info opacityInfo = opacityArray.request();
        py::buffer_info betaInfo = betaArray.request();

        auto checkShape = [](const py::buffer_info& bufferInfo,
                             std::size_t expectedCount,
                             std::size_t expectedDim,
                             const char* name) {
            if (bufferInfo.ndim != 2 ||
                bufferInfo.shape[0] != static_cast<ssize_t>(expectedCount) ||
                bufferInfo.shape[1] != static_cast<ssize_t>(expectedDim)) {
                throw std::runtime_error(
                    std::string("add_new_points: 'new.") + name +
                    "' must have shape (N," + std::to_string(expectedDim) + ")"
                );
            }
            if (bufferInfo.itemsize != sizeof(float)) {
                throw std::runtime_error(
                    std::string("add_new_points: 'new.") + name + "' must be float32"
                );
            }
        };

        if (positionInfo.ndim != 2 || positionInfo.shape[1] != 3) {
            throw std::runtime_error("add_new_points: 'new.position' must have shape (N,3)");
        }
        if (positionInfo.itemsize != sizeof(float)) {
            throw std::runtime_error("add_new_points: 'new.position' must be float32");
        }

        std::size_t newPointCount = static_cast<std::size_t>(positionInfo.shape[0]);
        if (newPointCount == 0) {
            Pale::Log::PA_INFO("add_new_points: 'new' block has zero points, nothing to append.");
            return;
        }

        checkShape(tangentUInfo, newPointCount, 3, "tangent_u");
        checkShape(tangentVInfo, newPointCount, 3, "tangent_v");
        checkShape(scaleInfo, newPointCount, 2, "scale");
        checkShape(albedoInfo, newPointCount, 3, "albedo");

        if (opacityInfo.ndim != 1 ||
            opacityInfo.shape[0] != static_cast<ssize_t>(newPointCount)) {
            throw std::runtime_error(
                "add_new_points: 'new.opacity' must have shape (N,)");
        }
        if (opacityInfo.itemsize != sizeof(float)) {
            throw std::runtime_error(
                "add_new_points: 'new.opacity' must be float32");
        }
        if (betaInfo.ndim != 1 ||
            betaInfo.shape[0] != static_cast<ssize_t>(newPointCount)) {
            throw std::runtime_error(
                "add_new_points: 'new.opacity' must have shape (N,)");
        }
        if (betaInfo.itemsize != sizeof(float)) {
            throw std::runtime_error(
                "add_new_points: 'new.opacity' must be float32");
        }

        const float* positionData = static_cast<float*>(positionInfo.ptr);
        const float* tangentUData = static_cast<float*>(tangentUInfo.ptr);
        const float* tangentVData = static_cast<float*>(tangentVInfo.ptr);
        const float* scaleData = static_cast<float*>(scaleInfo.ptr);
        const float* albedoData = static_cast<float*>(albedoInfo.ptr);
        const float* opacityData = static_cast<float*>(opacityInfo.ptr);
        const float* betaData = static_cast<float*>(betaInfo.ptr);

        // ---------------------------------------------------------------------
        // 2) Append new points at the bottom (no modification of existing points)
        // ---------------------------------------------------------------------
        const std::size_t currentPointCount = pointGeometry.positions.size();
        const std::size_t newTotalPointCount = currentPointCount + newPointCount;

        auto reserveAttribute = [newTotalPointCount](auto& vectorAttribute) {
            vectorAttribute.reserve(newTotalPointCount);
        };

        reserveAttribute(pointGeometry.positions);
        reserveAttribute(pointGeometry.tanU);
        reserveAttribute(pointGeometry.tanV);
        reserveAttribute(pointGeometry.scales);
        reserveAttribute(pointGeometry.albedos);
        reserveAttribute(pointGeometry.opacities);
        reserveAttribute(pointGeometry.shapes);
        reserveAttribute(pointGeometry.betas);

        for (std::size_t pointIndex = 0; pointIndex < newPointCount; ++pointIndex) {
            const std::size_t basePositionIndex = pointIndex * 3;
            const std::size_t baseTangentUIndex = pointIndex * 3;
            const std::size_t baseTangentVIndex = pointIndex * 3;
            const std::size_t baseScaleIndex = pointIndex * 2;
            const std::size_t baseColorIndex = pointIndex * 3;
            const std::size_t baseOpacityIndex = pointIndex * 1;
            const std::size_t baseBetaIndex = pointIndex * 1;

            glm::vec3 positionValue;
            positionValue.x = positionData[basePositionIndex + 0];
            positionValue.y = positionData[basePositionIndex + 1];
            positionValue.z = positionData[basePositionIndex + 2];

            glm::vec3 tangentUValue;
            tangentUValue.x = tangentUData[baseTangentUIndex + 0];
            tangentUValue.y = tangentUData[baseTangentUIndex + 1];
            tangentUValue.z = tangentUData[baseTangentUIndex + 2];

            glm::vec3 tangentVValue;
            tangentVValue.x = tangentVData[baseTangentVIndex + 0];
            tangentVValue.y = tangentVData[baseTangentVIndex + 1];
            tangentVValue.z = tangentVData[baseTangentVIndex + 2];

            glm::vec2 scaleValue;
            scaleValue.x = scaleData[baseScaleIndex + 0];
            scaleValue.y = scaleData[baseScaleIndex + 1];

            glm::vec3 albedoValue;
            albedoValue.x = albedoData[baseColorIndex + 0];
            albedoValue.y = albedoData[baseColorIndex + 1];
            albedoValue.z = albedoData[baseColorIndex + 2];

            float opacityValue = opacityData[baseOpacityIndex];
            float betaValue = betaData[baseBetaIndex];

            pointGeometry.positions.push_back(positionValue);
            pointGeometry.tanU.push_back(tangentUValue);
            pointGeometry.tanV.push_back(tangentVValue);
            pointGeometry.scales.push_back(scaleValue);
            pointGeometry.albedos.push_back(albedoValue);
            pointGeometry.opacities.push_back(opacityValue);

            pointGeometry.betas.push_back(betaValue);

            // Defaults for other attributes of new Gaussians
            pointGeometry.shapes.push_back(0.0f);
        }

        Pale::Log::PA_INFO(
            "add_new_points: final point count in geometry = {} (added {} new points)",
            pointGeometry.positions.size(),
            newPointCount
        );

        // ---------------------------------------------------------------------
        // 3) Rebuild buildProducts and GPU resources
        // ---------------------------------------------------------------------
        rebuild_bvh();
    }

    std::vector<std::string> getCameraNames() {
        std::vector<std::string> names;
        for (const auto& camera : buildProducts.cameras()) {
            names.emplace_back(camera.name);
        }
        return names;
    }

    void set_point_opacity(float newOpacity, int index) {
        if (!assetManager) {
            throw std::runtime_error("set_gaussian_opacity: assetManager is null");
        }

        auto pointAssetSharedPtr = assetManager->get<Pale::PointAsset>(pointCloudAssetHandle);
        if (!pointAssetSharedPtr) {
            throw std::runtime_error("set_gaussian_opacity: failed to get PointAsset for dynamic point cloud");
        }

        Pale::PointAsset& pointAsset = *pointAssetSharedPtr;
        if (pointAsset.points.empty()) {
            throw std::runtime_error("set_gaussian_opacity: PointAsset has no PointGeometry blocks");
        }

        Pale::PointGeometry& pointGeometry = pointAsset.points.front();

        const int pointCount = static_cast<int>(pointGeometry.opacities.size());
        if (index < 0 || index >= pointCount) {
            throw std::runtime_error("set_gaussian_opacity: index out of range");
        }

        pointGeometry.opacities[index] = newOpacity;

        // Only needed if BVH / acceleration depends on opacity (often it doesn't).
        // If you can skip it, do so for performance.
        rebuild_bvh();
        //Pale::Log::PA_ERROR("Opacity: {}/{}", pointGeometry.opacities[index], buildProducts.points[index].opacity);

    }


    void set_point_beta(float newBeta, int index) {
        if (!assetManager) {
            throw std::runtime_error("set_point_beta: assetManager is null");
        }

        auto pointAssetSharedPtr = assetManager->get<Pale::PointAsset>(pointCloudAssetHandle);
        if (!pointAssetSharedPtr) {
            throw std::runtime_error("set_point_beta: failed to get PointAsset for dynamic point cloud");
        }

        Pale::PointAsset& pointAsset = *pointAssetSharedPtr;
        if (pointAsset.points.empty()) {
            throw std::runtime_error("set_point_beta: PointAsset has no PointGeometry blocks");
        }

        Pale::PointGeometry& pointGeometry = pointAsset.points.front();

        const int pointCount = static_cast<int>(pointGeometry.opacities.size());
        if (index < 0 || index >= pointCount) {
            throw std::runtime_error("set_point_beta: index out of range");
        }

        pointGeometry.betas[index] = newBeta;

        // Only needed if BVH / acceleration depends on opacity (often it doesn't).
        // If you can skip it, do so for performance.
        rebuild_bvh();
        //Pale::Log::PA_ERROR("Beta: {}/{}", pointGeometry.betas[index], buildProducts.points[index].beta);

    }


    void set_point_properties(py::tuple translation3,
                              py::tuple rotationQuat4,
                              py::tuple scale3,
                              py::tuple albedo3,
                              float opacity,
                              float beta,
                              int index = -1) {
        if (translation3.size() != 3 || rotationQuat4.size() != 4 || scale3.size() != 3) {
            throw std::runtime_error("Expected translation(3), rotation_quat(4), scale(3)");
        }

        const glm::vec3 newTranslation{
            py::cast<float>(translation3[0]),
            py::cast<float>(translation3[1]),
            py::cast<float>(translation3[2])
        };
        const glm::vec3 newColor{
            py::cast<float>(albedo3[0]),
            py::cast<float>(albedo3[1]),
            py::cast<float>(albedo3[2])
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

        auto pointAssetSharedPtr = assetManager->get<Pale::PointAsset>(pointCloudAssetHandle);
        if (!pointAssetSharedPtr) {
            throw std::runtime_error("add_new_points: failed to get PointAsset for dynamic point cloud");
        }

        Pale::PointAsset& pointAsset = *pointAssetSharedPtr;
        if (pointAsset.points.empty()) {
            throw std::runtime_error("add_new_points: PointAsset has no PointGeometry blocks");
        }

        Pale::PointGeometry& pointGeometry = pointAsset.points.front();

        // --- apply translation to either one point or all points ---
        if (index < 0) {
            // perturb all points
            const sycl::float3 translationDelta = Pale::glm2sycl(newTranslation);
            for (int i = 0; i < pointGeometry.positions.size(); ++i) {
                pointGeometry.positions[i] += newTranslation;

                // 2) orientation: rotate tanU / tanV by the rotation delta
                glm::vec3 tanUGlm = (pointGeometry.tanU[i]);
                glm::vec3 tanVGlm = (pointGeometry.tanV[i]);

                tanUGlm = rotationDelta * tanUGlm;
                tanVGlm = rotationDelta * tanVGlm;

                pointGeometry.tanU[i] = tanUGlm;
                pointGeometry.tanV[i] = tanVGlm;

                pointGeometry.albedos[i] += newColor;
                pointGeometry.opacities[i] += opacity;
                pointGeometry.betas[i] += beta;
                pointGeometry.scales[i] *= newScale; // component-wise
                i++;
            }
        }
        else {
            if (index > pointGeometry.positions.size() - 1)
                throw std::runtime_error("add_new_points: index out of range");
            pointGeometry.positions[index] += newTranslation;

            // 2) orientation: rotate tanU / tanV by the rotation delta
            glm::vec3 tanUGlm = (pointGeometry.tanU[index]);
            glm::vec3 tanVGlm = (pointGeometry.tanV[index]);

            tanUGlm = rotationDelta * tanUGlm;
            tanVGlm = rotationDelta * tanVGlm;

            pointGeometry.tanU[index] = (tanUGlm);
            pointGeometry.tanV[index] = (tanVGlm);

            pointGeometry.albedos[index] = newColor;
            pointGeometry.opacities[index] = opacity;
            pointGeometry.betas[index] = beta;
            pointGeometry.scales[index] = newScale; // component-wise
        }


        rebuild_bvh();
    }

private:
    std::unique_ptr<Pale::AssetManager> assetManager{};
    std::shared_ptr<Pale::Scene> scene{};
    std::unique_ptr<Pale::DeviceSelector> deviceSelector{};

    std::vector<Pale::SensorGPU> sensorsForward{};
    std::vector<Pale::SensorGPU> sensorsAdjoint{};
    std::unique_ptr<Pale::PathTracer> pathTracer{};
    std::vector<Pale::DebugImages> debugImages;

    Pale::AssetHandle pointCloudAssetHandle{};

    Pale::SceneBuild::BuildProducts buildProducts{};
    Pale::GPUSceneBuffers sceneGpu{};
    Pale::PointGradients gradients{};
    // Adjoint buffers
    bool adjointBuffersAllocated{false};
    float* adjointFramebuffer{nullptr};
    float* adjointFramebufferGrad{nullptr};
    Pale::float3* gradientPkBuffer{nullptr};
    size_t gradCount{1024}; // set to your point count or resize after build
};

// ---- pybind11 module ----
PYBIND11_MODULE(pale, m) {
    py::class_<PythonRenderer>(m, "Renderer")
        .def(py::init<
                 const std::string&, // assetRootDir
                 const std::string&, // sceneXml
                 const std::string&, // pointCloudFile
                 const py::dict& // settingsDict
             >(),
             py::arg("assetRootDir"),
             py::arg("sceneXml") = "cbox_custom.xml",
             py::arg("pointCloudFile") = "initial.ply",
             py::arg("settings") = py::dict() // default empty
        )
        .def("render_forward", &PythonRenderer::render_forward, py::arg("camera_name") = "")
        .def("get_camera_names", &PythonRenderer::getCameraNames)
        .def("render_backward", &PythonRenderer::render_backward,
             py::arg("targetRgb32f"))
        .def(
            "get_point_parameters", &PythonRenderer::get_point_parameters)
        .def("apply_point_optimization", &PythonRenderer::apply_point_optimization, py::arg("parameters"))
        .def("add_points", &PythonRenderer::add_new_points,
             py::arg("parameters"))
        .def("remove_points", &PythonRenderer::remove_points,
             py::arg("parameters"))
        .def("rebuild_bvh", &PythonRenderer::rebuild_bvh)
        .def("set_point_properties",
             &PythonRenderer::set_point_properties,
             py::arg("translation3"), py::arg("rotation_quat4"),
             py::arg("scale3"), py::arg("albedo3"),
             py::arg("opacity"), py::arg("beta"),
             py::arg("index") = -1)
        .def("set_point_opacity",
             &PythonRenderer::set_point_opacity, py::arg("opacity"),
             py::arg("index"))
        .def("set_point_beta",
             &PythonRenderer::set_point_beta, py::arg("beta"),
             py::arg("index"));
}
