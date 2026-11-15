//
// Created by magnus on 8/29/25.
//
module;

#include <cstdint>
#include <memory>
#include <sycl/sycl.hpp>

#include "Renderer/GPUDataStructures.h"
#include "Renderer/Kernels/KernelHelpers.h"
#include "Core/DeviceAllocation.h"

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
            if (currentCount == newCount && (newCount == 0 || devicePointer != nullptr)) {
                return;
            }
            detail::replaceDeviceAllocation(devicePointer, newCount,
                [&](std::size_t count) { return sycl::malloc_device<T>(count, queue); },
                [&](void* pointer) { sycl::free(pointer, queue); });
            currentCount = newCount;
        }

        template<typename T>
        static void reallocVectorBackedArray(T *&devicePointer,
                                             std::size_t newElementCount,
                                             sycl::queue queue) {
            detail::replaceDeviceAllocation(devicePointer, newElementCount,
                [&](std::size_t count) { return sycl::malloc_device<T>(count, queue); },
                [&](void* pointer) { sycl::free(pointer, queue); });
        }

        // ---------------------------------------------------------------------
        // 0) Free: release all device memory owned by GPUSceneBuffers
        // ---------------------------------------------------------------------
        static void freeBuffers(GPUSceneBuffers &gpuSceneBuffers, sycl::queue queue) {
            // geometry
            freeDeviceArray(gpuSceneBuffers.vertices, queue);
            freeDeviceArray(gpuSceneBuffers.triangles, queue);
            freeDeviceArray(gpuSceneBuffers.points, queue);
            freeDeviceArray(gpuSceneBuffers.pointTraversalData, queue);
            freeDeviceArray(gpuSceneBuffers.pointPackedBvhNodes, queue);
            freeDeviceArray(gpuSceneBuffers.pointPackedBvhRanges, queue);
            freeDeviceArray(gpuSceneBuffers.pointQbvhNodes, queue);
            freeDeviceArray(gpuSceneBuffers.pointQbvhRanges, queue);
            freeDeviceArray(gpuSceneBuffers.blasNodes, queue);
            freeDeviceArray(gpuSceneBuffers.blasRanges, queue);
            freeDeviceArray(gpuSceneBuffers.tlasNodes, queue);
            freeDeviceArray(gpuSceneBuffers.transforms, queue);
            freeDeviceArray(gpuSceneBuffers.materials, queue);
            freeDeviceArray(gpuSceneBuffers.instances, queue);
            freeDeviceArray(gpuSceneBuffers.pointPermutation, queue);
            // lights
            freeDeviceArray(gpuSceneBuffers.lights, queue);
            freeDeviceArray(gpuSceneBuffers.emissiveTriangles, queue);

            // reset counts
            gpuSceneBuffers.vertexCount = 0;
            gpuSceneBuffers.pointCount = 0;
            gpuSceneBuffers.pointTraversalDataCount = 0;
            gpuSceneBuffers.pointPackedBvhNodeCount = 0;
            gpuSceneBuffers.pointPackedBvhRangeCount = 0;
            gpuSceneBuffers.pointQbvhNodeCount = 0;
            gpuSceneBuffers.pointQbvhRangeCount = 0;
            gpuSceneBuffers.triangleCount = 0;
            gpuSceneBuffers.blasNodeCount = 0;
            gpuSceneBuffers.tlasNodeCount = 0;
            gpuSceneBuffers.lightCount = 0;
            gpuSceneBuffers.emissiveTriangleCount = 0;
            gpuSceneBuffers.pointPermutationCount = 0;
        }

        // ---------------------------------------------------------------------
        // 1) Allocate: device memory based on BuildProducts, no memcpy
        // ---------------------------------------------------------------------
        static GPUSceneBuffers allocate(const SceneBuild::BuildProducts &buildProducts,
                                        sycl::queue queue) {
            GPUSceneBuffers gpuSceneBuffers{};
            try {
                allocateOrReallocate(buildProducts, gpuSceneBuffers, queue);
            } catch(...) {
                freeBuffers(gpuSceneBuffers, queue);
                throw;
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
            const std::uint32_t newPointPackedBvhNodeCount =
                    static_cast<std::uint32_t>(buildProducts.pointPackedBvhNodes.size());
            const std::uint32_t newPointPackedBvhRangeCount =
                    static_cast<std::uint32_t>(buildProducts.pointPackedBvhRanges.size());
            const std::uint32_t newPointQbvhNodeCount =
                    static_cast<std::uint32_t>(buildProducts.pointQbvhNodes.size());
            const std::uint32_t newPointQbvhRangeCount =
                    static_cast<std::uint32_t>(buildProducts.pointQbvhRanges.size());
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

            const std::uint32_t newPointTraversalDataCount =
                    static_cast<std::uint32_t>(buildProducts.pointTraversalData.size());
            logReallocCounted("pointTraversalData",
                              gpuSceneBuffers.pointTraversalDataCount,
                              newPointTraversalDataCount);
            reallocCountedDeviceArray(gpuSceneBuffers.pointTraversalData,
                                      gpuSceneBuffers.pointTraversalDataCount,
                                      newPointTraversalDataCount,
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

            logReallocCounted("pointPackedBvhNodes",
                              gpuSceneBuffers.pointPackedBvhNodeCount,
                              newPointPackedBvhNodeCount);
            reallocCountedDeviceArray(gpuSceneBuffers.pointPackedBvhNodes,
                                      gpuSceneBuffers.pointPackedBvhNodeCount,
                                      newPointPackedBvhNodeCount,
                                      queue);

            logReallocCounted("pointPackedBvhRanges",
                              gpuSceneBuffers.pointPackedBvhRangeCount,
                              newPointPackedBvhRangeCount);
            reallocCountedDeviceArray(gpuSceneBuffers.pointPackedBvhRanges,
                                      gpuSceneBuffers.pointPackedBvhRangeCount,
                                      newPointPackedBvhRangeCount,
                                      queue);

            logReallocCounted("pointQbvhNodes",
                              gpuSceneBuffers.pointQbvhNodeCount,
                              newPointQbvhNodeCount);
            reallocCountedDeviceArray(gpuSceneBuffers.pointQbvhNodes,
                                      gpuSceneBuffers.pointQbvhNodeCount,
                                      newPointQbvhNodeCount,
                                      queue);

            logReallocCounted("pointQbvhRanges",
                              gpuSceneBuffers.pointQbvhRangeCount,
                              newPointQbvhRangeCount);
            reallocCountedDeviceArray(gpuSceneBuffers.pointQbvhRanges,
                                      gpuSceneBuffers.pointQbvhRangeCount,
                                      newPointQbvhRangeCount,
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

            const std::uint32_t newPointPermutationCount =
                    static_cast<std::uint32_t>(buildProducts.pointPermutation.size());

            logReallocCounted("pointPermutation",
                              gpuSceneBuffers.pointPermutationCount,
                              newPointPermutationCount);
            reallocCountedDeviceArray(gpuSceneBuffers.pointPermutation,
                                      gpuSceneBuffers.pointPermutationCount,
                                      newPointPermutationCount,
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
                gpuSceneBuffers.pointTraversalDataCount != buildProducts.pointTraversalData.size() ||
                gpuSceneBuffers.triangleCount != buildProducts.triangles.size() ||
                gpuSceneBuffers.blasNodeCount != buildProducts.bottomLevelNodes.size() ||
                gpuSceneBuffers.pointPackedBvhNodeCount != buildProducts.pointPackedBvhNodes.size() ||
                gpuSceneBuffers.pointPackedBvhRangeCount != buildProducts.pointPackedBvhRanges.size() ||
                gpuSceneBuffers.pointQbvhNodeCount != buildProducts.pointQbvhNodes.size() ||
                gpuSceneBuffers.pointQbvhRangeCount != buildProducts.pointQbvhRanges.size() ||
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

                for (int i = 0; const auto &it: buildProducts.points) {
                    Log::PA_TRACE("Point After Upload [{}]: {}, {}, {}", i, it.position.x(), it.position.y(),
                                  it.position.z());
                    i++;
                    if (i == 4)
                        break;
                }
            }

            if (gpuSceneBuffers.blasNodeCount > 0 &&
                gpuSceneBuffers.blasNodes != nullptr &&
                !buildProducts.bottomLevelNodes.empty()) {
                queue.memcpy(
                    gpuSceneBuffers.blasNodes,
                    buildProducts.bottomLevelNodes.data(),
                    gpuSceneBuffers.blasNodeCount * sizeof(BVHNode));
            }

            if (gpuSceneBuffers.pointPackedBvhNodeCount > 0 &&
                gpuSceneBuffers.pointPackedBvhNodes != nullptr &&
                !buildProducts.pointPackedBvhNodes.empty()) {
                queue.memcpy(
                    gpuSceneBuffers.pointPackedBvhNodes,
                    buildProducts.pointPackedBvhNodes.data(),
                    gpuSceneBuffers.pointPackedBvhNodeCount * sizeof(PackedPointBVHNode));
            }

            if (gpuSceneBuffers.pointPackedBvhRangeCount > 0 &&
                gpuSceneBuffers.pointPackedBvhRanges != nullptr &&
                !buildProducts.pointPackedBvhRanges.empty()) {
                queue.memcpy(
                    gpuSceneBuffers.pointPackedBvhRanges,
                    buildProducts.pointPackedBvhRanges.data(),
                    gpuSceneBuffers.pointPackedBvhRangeCount * sizeof(BLASRange));
            }

            if (gpuSceneBuffers.pointQbvhNodeCount > 0 &&
                gpuSceneBuffers.pointQbvhNodes != nullptr &&
                !buildProducts.pointQbvhNodes.empty()) {
                queue.memcpy(
                    gpuSceneBuffers.pointQbvhNodes,
                    buildProducts.pointQbvhNodes.data(),
                    gpuSceneBuffers.pointQbvhNodeCount * sizeof(PackedPointQBVHNode));
            }

            if (gpuSceneBuffers.pointQbvhRangeCount > 0 &&
                gpuSceneBuffers.pointQbvhRanges != nullptr &&
                !buildProducts.pointQbvhRanges.empty()) {
                queue.memcpy(
                    gpuSceneBuffers.pointQbvhRanges,
                    buildProducts.pointQbvhRanges.data(),
                    gpuSceneBuffers.pointQbvhRangeCount * sizeof(BLASRange));
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

            if (gpuSceneBuffers.pointPermutationCount > 0 &&
    gpuSceneBuffers.pointPermutation != nullptr &&
    !buildProducts.pointPermutation.empty()) {
                queue.memcpy(
                    gpuSceneBuffers.pointPermutation,
                    buildProducts.pointPermutation.data(),
                    gpuSceneBuffers.pointPermutationCount * sizeof(uint32_t));
    }

            // Use the same device arithmetic as refitNode<true>. Uploading the
            // host-computed normals/exponents makes the first no-op refit change
            // intersection depths and, at discrete slab boundaries, radiance.
            // The in-order queue has uploaded both points and permutation here.
            if (gpuSceneBuffers.pointTraversalDataCount > 0) {
                const auto scene = gpuSceneBuffers;
                queue.parallel_for<class InitializeSurfelTraversalData>(
                    sycl::range<1>(scene.pointTraversalDataCount), [=](sycl::id<1> index) {
                        const auto primitiveIndex = scene.pointPermutation[index[0]];
                        scene.pointTraversalData[index[0]] =
                            makeSurfelTraversalData(scene.points[primitiveIndex], primitiveIndex);
                    });
            }
            queue.wait_and_throw();
        }

        // ---------------------------------------------------------------------
        // 2b) Upload with optional reallocation
        //
        // Counted arrays retain their storage when sizes match. Metadata arrays
        // without stored capacities are refreshed before copying. Allocation
        // failures preserve each array's previous pointer/count for recovery.
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
                gpuSceneBuffers.pointTraversalDataCount == buildProducts.pointTraversalData.size() &&
                gpuSceneBuffers.pointPackedBvhNodeCount == buildProducts.pointPackedBvhNodes.size() &&
                gpuSceneBuffers.pointPackedBvhRangeCount == buildProducts.pointPackedBvhRanges.size() &&
                gpuSceneBuffers.pointQbvhNodeCount == buildProducts.pointQbvhNodes.size() &&
                gpuSceneBuffers.pointQbvhRangeCount == buildProducts.pointQbvhRanges.size() &&
                gpuSceneBuffers.pointPermutationCount == buildProducts.pointPermutation.size() &&
                gpuSceneBuffers.lightCount == buildProducts.lights.size() &&
                gpuSceneBuffers.emissiveTriangleCount == buildProducts.emissiveTriangles.size();

            Pale::Log::PA_INFO("Renderer Registered new points: {}", sameSizes);
            if (!sameSizes) {
                // Reallocate in-place
                allocateOrReallocate(buildProducts, gpuSceneBuffers, queue);
            } else {
                // These arrays have no capacity fields in GPUSceneBuffers.
                // Equal geometry counts do not imply equal scene metadata;
                // copying a larger instance/material array into old storage
                // would otherwise overrun its allocation.
                reallocVectorBackedArray(gpuSceneBuffers.blasRanges, buildProducts.bottomLevelRanges.size(), queue);
                reallocVectorBackedArray(gpuSceneBuffers.transforms, buildProducts.transforms.size(), queue);
                reallocVectorBackedArray(gpuSceneBuffers.materials, buildProducts.materials.size(), queue);
                reallocVectorBackedArray(gpuSceneBuffers.instances, buildProducts.instances.size(), queue);
            }

            upload(buildProducts, gpuSceneBuffers, queue);
        }

        // ---------------------------------------------------------------------
        // 3) Convenience: old behavior (allocate + upload)
        // ---------------------------------------------------------------------
        static GPUSceneBuffers allocateAndUpload(const SceneBuild::BuildProducts &buildProducts,
                                                 sycl::queue queue) {
            GPUSceneBuffers gpuSceneBuffers = allocate(buildProducts, queue);
            try {
                upload(buildProducts, gpuSceneBuffers, queue);
            } catch(...) {
                // Complete any copies already submitted before releasing their
                // destination buffers, including a failed diagnostic upload.
                queue.wait();
                freeBuffers(gpuSceneBuffers, queue);
                throw;
            }
            return gpuSceneBuffers;
        }
    };
}
