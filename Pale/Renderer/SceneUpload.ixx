//
// Created by magnus on 8/29/25.
//
module;

#include <cstdint>
#include <memory>
#include <sycl/sycl.hpp>

#include "Renderer/GPUDataStructures.h"

export module Pale.Render.SceneUpload;

import Pale.Scene;
import Pale.Render.SceneBuild;
import Pale.Log;

export namespace Pale {
    class SceneUpload {
    public:
        // ---------------------------------------------------------------------
        // Helper utilities (no lambdas)
        // ---------------------------------------------------------------------
        template<typename T>
        static void freeDeviceArray(T *&devicePointer, sycl::queue queue) {
            if (devicePointer != nullptr) {
                sycl::free(devicePointer, queue);
                devicePointer = nullptr;
            }
        }

        template<typename T>
        static void reallocCountedDeviceArray(T *&devicePointer,
                                              std::uint32_t &currentCount,
                                              std::uint32_t newCount,
                                              sycl::queue queue) {
            if (currentCount == newCount) {
                return;
            }

            // Free old memory (if any)
            freeDeviceArray(devicePointer, queue);

            currentCount = newCount;
            if (newCount > 0) {
                const std::size_t bytes = static_cast<std::size_t>(newCount) * sizeof(T);
                devicePointer = static_cast<T *>(sycl::malloc_device(bytes, queue));
            }
        }

        template<typename T>
        static void reallocVectorBackedArray(T *&devicePointer,
                                             std::size_t newElementCount,
                                             sycl::queue queue) {
            if (newElementCount == 0) {
                // Nothing needed – just free if allocated
                freeDeviceArray(devicePointer, queue);
                return;
            }

            // Always free and reallocate (simple and safe;
            // can be optimized by tracking old element count if needed)
            freeDeviceArray(devicePointer, queue);

            const std::size_t bytes = newElementCount * sizeof(T);
            devicePointer = static_cast<T *>(sycl::malloc_device(bytes, queue));
        }

        // ---------------------------------------------------------------------
        // 0) Free: release all device memory owned by GPUSceneBuffers
        // ---------------------------------------------------------------------
        static void freeBuffers(GPUSceneBuffers &gpuSceneBuffers, sycl::queue queue) {
            // geometry
            freeDeviceArray(gpuSceneBuffers.vertices, queue);
            freeDeviceArray(gpuSceneBuffers.triangles, queue);
            freeDeviceArray(gpuSceneBuffers.points, queue);
            freeDeviceArray(gpuSceneBuffers.blasNodes, queue);
            freeDeviceArray(gpuSceneBuffers.blasRanges, queue);
            freeDeviceArray(gpuSceneBuffers.tlasNodes, queue);
            freeDeviceArray(gpuSceneBuffers.transforms, queue);
            freeDeviceArray(gpuSceneBuffers.materials, queue);
            freeDeviceArray(gpuSceneBuffers.instances, queue);

            // lights
            freeDeviceArray(gpuSceneBuffers.lights, queue);
            freeDeviceArray(gpuSceneBuffers.emissiveTriangles, queue);

            // reset counts
            gpuSceneBuffers.vertexCount = 0;
            gpuSceneBuffers.pointCount = 0;
            gpuSceneBuffers.triangleCount = 0;
            gpuSceneBuffers.blasNodeCount = 0;
            gpuSceneBuffers.tlasNodeCount = 0;
            gpuSceneBuffers.lightCount = 0;
            gpuSceneBuffers.emissiveTriangleCount = 0;
        }

        // ---------------------------------------------------------------------
        // 1) Allocate: device memory based on BuildProducts, no memcpy
        // ---------------------------------------------------------------------
        static GPUSceneBuffers allocate(const SceneBuild::BuildProducts &buildProducts,
                                        sycl::queue queue) {
            GPUSceneBuffers gpuSceneBuffers{};

            // geometry sizes
            gpuSceneBuffers.vertexCount = static_cast<std::uint32_t>(buildProducts.vertices.size());
            gpuSceneBuffers.pointCount = static_cast<std::uint32_t>(buildProducts.points.size());
            gpuSceneBuffers.triangleCount = static_cast<std::uint32_t>(buildProducts.triangles.size());
            gpuSceneBuffers.blasNodeCount = static_cast<std::uint32_t>(buildProducts.bottomLevelNodes.size());
            gpuSceneBuffers.tlasNodeCount = static_cast<std::uint32_t>(buildProducts.topLevelNodes.size());

            // light sizes
            gpuSceneBuffers.lightCount =
                    static_cast<std::uint32_t>(buildProducts.lights.size());
            gpuSceneBuffers.emissiveTriangleCount =
                    static_cast<std::uint32_t>(buildProducts.emissiveTriangles.size());

            // allocate geometry
            if (gpuSceneBuffers.vertexCount > 0) {
                const std::size_t bytes =
                        static_cast<std::size_t>(gpuSceneBuffers.vertexCount) * sizeof(Vertex);
                gpuSceneBuffers.vertices =
                        static_cast<Vertex *>(sycl::malloc_device(bytes, queue));
            }
            if (gpuSceneBuffers.triangleCount > 0) {
                const std::size_t bytes =
                        static_cast<std::size_t>(gpuSceneBuffers.triangleCount) * sizeof(Triangle);
                gpuSceneBuffers.triangles =
                        static_cast<Triangle *>(sycl::malloc_device(bytes, queue));
            }
            if (gpuSceneBuffers.pointCount > 0) {
                const std::size_t bytes =
                        static_cast<std::size_t>(gpuSceneBuffers.pointCount) * sizeof(Point);
                gpuSceneBuffers.points =
                        static_cast<Point *>(sycl::malloc_device(bytes, queue));
            }
            if (gpuSceneBuffers.blasNodeCount > 0) {
                const std::size_t bytes =
                        static_cast<std::size_t>(gpuSceneBuffers.blasNodeCount) * sizeof(BVHNode);
                gpuSceneBuffers.blasNodes =
                        static_cast<BVHNode *>(sycl::malloc_device(bytes, queue));
            }
            if (!buildProducts.bottomLevelRanges.empty()) {
                const std::size_t bytes =
                        buildProducts.bottomLevelRanges.size() * sizeof(BLASRange);
                gpuSceneBuffers.blasRanges =
                        static_cast<BLASRange *>(sycl::malloc_device(bytes, queue));
            }
            if (gpuSceneBuffers.tlasNodeCount > 0) {
                const std::size_t bytes =
                        static_cast<std::size_t>(gpuSceneBuffers.tlasNodeCount) * sizeof(TLASNode);
                gpuSceneBuffers.tlasNodes =
                        static_cast<TLASNode *>(sycl::malloc_device(bytes, queue));
            }
            if (!buildProducts.transforms.empty()) {
                const std::size_t bytes =
                        buildProducts.transforms.size() * sizeof(Transform);
                gpuSceneBuffers.transforms =
                        static_cast<Transform *>(sycl::malloc_device(bytes, queue));
            }
            if (!buildProducts.materials.empty()) {
                const std::size_t bytes =
                        buildProducts.materials.size() * sizeof(GPUMaterial);
                gpuSceneBuffers.materials =
                        static_cast<GPUMaterial *>(sycl::malloc_device(bytes, queue));
            }
            if (!buildProducts.instances.empty()) {
                const std::size_t bytes =
                        buildProducts.instances.size() * sizeof(InstanceRecord);
                gpuSceneBuffers.instances =
                        static_cast<InstanceRecord *>(sycl::malloc_device(bytes, queue));
            }

            // allocate lights
            if (gpuSceneBuffers.lightCount > 0) {
                const std::size_t bytes =
                        static_cast<std::size_t>(gpuSceneBuffers.lightCount) * sizeof(GPULightRecord);
                gpuSceneBuffers.lights =
                        static_cast<GPULightRecord *>(sycl::malloc_device(bytes, queue));
            }
            if (gpuSceneBuffers.emissiveTriangleCount > 0) {
                const std::size_t bytes =
                        static_cast<std::size_t>(gpuSceneBuffers.emissiveTriangleCount) *
                        sizeof(GPUEmissiveTriangle);
                gpuSceneBuffers.emissiveTriangles =
                        static_cast<GPUEmissiveTriangle *>(sycl::malloc_device(bytes, queue));
            }

            return gpuSceneBuffers;
        }

        // ---------------------------------------------------------------------
        // 1b) Allocate or reallocate in-place based on BuildProducts
        //     (used by uploadOrReallocate)
        // ---------------------------------------------------------------------
        static void allocateOrReallocate(const SceneBuild::BuildProducts &buildProducts,
                                         GPUSceneBuffers &gpuSceneBuffers,
                                         sycl::queue queue) {
            // Geometry sizes
            const std::uint32_t newVertexCount =
                    static_cast<std::uint32_t>(buildProducts.vertices.size());
            const std::uint32_t newPointCount =
                    static_cast<std::uint32_t>(buildProducts.points.size());
            const std::uint32_t newTriangleCount =
                    static_cast<std::uint32_t>(buildProducts.triangles.size());
            const std::uint32_t newBlasNodeCount =
                    static_cast<std::uint32_t>(buildProducts.bottomLevelNodes.size());
            const std::uint32_t newTlasNodeCount =
                    static_cast<std::uint32_t>(buildProducts.topLevelNodes.size());

            const std::uint32_t newLightCount =
                    static_cast<std::uint32_t>(buildProducts.lights.size());
            const std::uint32_t newEmissiveTriangleCount =
                    static_cast<std::uint32_t>(buildProducts.emissiveTriangles.size());

            auto logReallocCounted = [](const char *name,
                                        std::uint32_t oldCount,
                                        std::uint32_t newCount) {
                if (oldCount != newCount) {
                    Pale::Log::PA_INFO(
                        "allocateOrReallocate: {} count changed {} → {} (reallocating)",
                        name, oldCount, newCount
                    );
                } else {
                    Pale::Log::PA_INFO(
                        "allocateOrReallocate: {} count unchanged ({})",
                        name, oldCount
                    );
                }
            };

            // ---- Counted device arrays ----------------------------------------

            logReallocCounted("vertices", gpuSceneBuffers.vertexCount, newVertexCount);
            reallocCountedDeviceArray(gpuSceneBuffers.vertices,
                                      gpuSceneBuffers.vertexCount,
                                      newVertexCount,
                                      queue);

            logReallocCounted("points", gpuSceneBuffers.pointCount, newPointCount);
            reallocCountedDeviceArray(gpuSceneBuffers.points,
                                      gpuSceneBuffers.pointCount,
                                      newPointCount,
                                      queue);

            logReallocCounted("triangles", gpuSceneBuffers.triangleCount, newTriangleCount);
            reallocCountedDeviceArray(gpuSceneBuffers.triangles,
                                      gpuSceneBuffers.triangleCount,
                                      newTriangleCount,
                                      queue);

            logReallocCounted("blasNodes", gpuSceneBuffers.blasNodeCount, newBlasNodeCount);
            reallocCountedDeviceArray(gpuSceneBuffers.blasNodes,
                                      gpuSceneBuffers.blasNodeCount,
                                      newBlasNodeCount,
                                      queue);

            logReallocCounted("tlasNodes", gpuSceneBuffers.tlasNodeCount, newTlasNodeCount);
            reallocCountedDeviceArray(gpuSceneBuffers.tlasNodes,
                                      gpuSceneBuffers.tlasNodeCount,
                                      newTlasNodeCount,
                                      queue);

            logReallocCounted("lights", gpuSceneBuffers.lightCount, newLightCount);
            reallocCountedDeviceArray(gpuSceneBuffers.lights,
                                      gpuSceneBuffers.lightCount,
                                      newLightCount,
                                      queue);

            logReallocCounted("emissiveTriangles",
                              gpuSceneBuffers.emissiveTriangleCount,
                              newEmissiveTriangleCount);
            reallocCountedDeviceArray(gpuSceneBuffers.emissiveTriangles,
                                      gpuSceneBuffers.emissiveTriangleCount,
                                      newEmissiveTriangleCount,
                                      queue);

            // ---- Vector-backed arrays (no count stored in GPUSceneBuffers) ----

            const std::size_t newBlasRangeCount = buildProducts.bottomLevelRanges.size();
            Pale::Log::PA_INFO(
                "allocateOrReallocate: blasRanges resized to {} (vector-backed)",
                newBlasRangeCount
            );
            reallocVectorBackedArray(gpuSceneBuffers.blasRanges,
                                     newBlasRangeCount,
                                     queue);

            const std::size_t newTransformCount = buildProducts.transforms.size();
            Pale::Log::PA_INFO(
                "allocateOrReallocate: transforms resized to {} (vector-backed)",
                newTransformCount
            );
            reallocVectorBackedArray(gpuSceneBuffers.transforms,
                                     newTransformCount,
                                     queue);

            const std::size_t newMaterialCount = buildProducts.materials.size();
            Pale::Log::PA_INFO(
                "allocateOrReallocate: materials resized to {} (vector-backed)",
                newMaterialCount
            );
            reallocVectorBackedArray(gpuSceneBuffers.materials,
                                     newMaterialCount,
                                     queue);

            const std::size_t newInstanceCount = buildProducts.instances.size();
            Pale::Log::PA_INFO(
                "allocateOrReallocate: instances resized to {} (vector-backed)",
                newInstanceCount
            );
            reallocVectorBackedArray(gpuSceneBuffers.instances,
                                     newInstanceCount,
                                     queue);
        }


        // ---------------------------------------------------------------------
        // 2) Upload: memcpy CPU BuildProducts into already-allocated GPU buffers
        // ---------------------------------------------------------------------
        static void upload(const SceneBuild::BuildProducts &buildProducts,
                           GPUSceneBuffers &gpuSceneBuffers,
                           sycl::queue queue) {
            // Optional: sanity checks – you can tighten/relax as you wish
            if (gpuSceneBuffers.vertexCount != buildProducts.vertices.size() ||
                gpuSceneBuffers.pointCount != buildProducts.points.size() ||
                gpuSceneBuffers.triangleCount != buildProducts.triangles.size() ||
                gpuSceneBuffers.blasNodeCount != buildProducts.bottomLevelNodes.size() ||
                gpuSceneBuffers.tlasNodeCount != buildProducts.topLevelNodes.size()) {
                throw std::runtime_error(
                    "SceneUpload::upload: GPU buffer sizes do not match BuildProducts sizes");
            }

            if (gpuSceneBuffers.vertexCount > 0 &&
                gpuSceneBuffers.vertices != nullptr &&
                !buildProducts.vertices.empty()) {
                queue.memcpy(
                    gpuSceneBuffers.vertices,
                    buildProducts.vertices.data(),
                    gpuSceneBuffers.vertexCount * sizeof(Vertex));
            }

            if (gpuSceneBuffers.triangleCount > 0 &&
                gpuSceneBuffers.triangles != nullptr &&
                !buildProducts.triangles.empty()) {
                queue.memcpy(
                    gpuSceneBuffers.triangles,
                    buildProducts.triangles.data(),
                    gpuSceneBuffers.triangleCount * sizeof(Triangle));
            }

            if (gpuSceneBuffers.pointCount > 0 &&
                gpuSceneBuffers.points != nullptr &&
                !buildProducts.points.empty()) {
                queue.memcpy(
                    gpuSceneBuffers.points,
                    buildProducts.points.data(),
                    gpuSceneBuffers.pointCount * sizeof(Point));
            }

            if (gpuSceneBuffers.blasNodeCount > 0 &&
                gpuSceneBuffers.blasNodes != nullptr &&
                !buildProducts.bottomLevelNodes.empty()) {
                queue.memcpy(
                    gpuSceneBuffers.blasNodes,
                    buildProducts.bottomLevelNodes.data(),
                    gpuSceneBuffers.blasNodeCount * sizeof(BVHNode));
            }

            if (gpuSceneBuffers.blasRanges != nullptr &&
                !buildProducts.bottomLevelRanges.empty()) {
                queue.memcpy(
                    gpuSceneBuffers.blasRanges,
                    buildProducts.bottomLevelRanges.data(),
                    buildProducts.bottomLevelRanges.size() * sizeof(BLASRange));
            }

            if (gpuSceneBuffers.tlasNodeCount > 0 &&
                gpuSceneBuffers.tlasNodes != nullptr &&
                !buildProducts.topLevelNodes.empty()) {
                queue.memcpy(
                    gpuSceneBuffers.tlasNodes,
                    buildProducts.topLevelNodes.data(),
                    gpuSceneBuffers.tlasNodeCount * sizeof(TLASNode));
            }

            if (gpuSceneBuffers.transforms != nullptr &&
                !buildProducts.transforms.empty()) {
                queue.memcpy(
                    gpuSceneBuffers.transforms,
                    buildProducts.transforms.data(),
                    buildProducts.transforms.size() * sizeof(Transform));
            }

            if (gpuSceneBuffers.materials != nullptr &&
                !buildProducts.materials.empty()) {
                queue.memcpy(
                    gpuSceneBuffers.materials,
                    buildProducts.materials.data(),
                    buildProducts.materials.size() * sizeof(GPUMaterial));
            }

            if (gpuSceneBuffers.instances != nullptr &&
                !buildProducts.instances.empty()) {
                queue.memcpy(
                    gpuSceneBuffers.instances,
                    buildProducts.instances.data(),
                    buildProducts.instances.size() * sizeof(InstanceRecord));
            }

            if (gpuSceneBuffers.lightCount > 0 &&
                gpuSceneBuffers.lights != nullptr &&
                !buildProducts.lights.empty()) {
                queue.memcpy(
                    gpuSceneBuffers.lights,
                    buildProducts.lights.data(),
                    gpuSceneBuffers.lightCount * sizeof(GPULightRecord));
            }

            if (gpuSceneBuffers.emissiveTriangleCount > 0 &&
                gpuSceneBuffers.emissiveTriangles != nullptr &&
                !buildProducts.emissiveTriangles.empty()) {
                queue.memcpy(
                    gpuSceneBuffers.emissiveTriangles,
                    buildProducts.emissiveTriangles.data(),
                    gpuSceneBuffers.emissiveTriangleCount * sizeof(GPUEmissiveTriangle));
            }

            queue.wait();
        }

        // ---------------------------------------------------------------------
        // 2b) Upload with optional reallocation
        //
        // If sizes match: behaves like upload().
        // If sizes differ: frees and reallocates buffers, then uploads.
        // ---------------------------------------------------------------------
        static void uploadOrReallocate(const SceneBuild::BuildProducts &buildProducts,
                                       GPUSceneBuffers &gpuSceneBuffers,
                                       sycl::queue queue) {
            const bool sameSizes =
                    gpuSceneBuffers.vertexCount == buildProducts.vertices.size() &&
                    gpuSceneBuffers.pointCount == buildProducts.points.size() &&
                    gpuSceneBuffers.triangleCount == buildProducts.triangles.size() &&
                    gpuSceneBuffers.blasNodeCount == buildProducts.bottomLevelNodes.size() &&
                    gpuSceneBuffers.tlasNodeCount == buildProducts.topLevelNodes.size() &&
                    gpuSceneBuffers.lightCount == buildProducts.lights.size() &&
                    gpuSceneBuffers.emissiveTriangleCount == buildProducts.emissiveTriangles.size();


            Pale::Log::PA_INFO("Renderer Registered new points: {}", sameSizes);
            if (!sameSizes) {
                // Reallocate in-place
                allocateOrReallocate(buildProducts, gpuSceneBuffers, queue);
            }

            upload(buildProducts, gpuSceneBuffers, queue);
        }

        // ---------------------------------------------------------------------
        // 3) Convenience: old behavior (allocate + upload)
        // ---------------------------------------------------------------------
        static GPUSceneBuffers allocateAndUpload(const SceneBuild::BuildProducts &buildProducts,
                                                 sycl::queue queue) {
            GPUSceneBuffers gpuSceneBuffers = allocate(buildProducts, queue);
            upload(buildProducts, gpuSceneBuffers, queue);
            return gpuSceneBuffers;
        }
    };
}
