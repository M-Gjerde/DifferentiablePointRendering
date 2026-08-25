// IntersectionKernels.h
#pragma once

#include <sycl/sycl.hpp>
#include <limits>

#include <Renderer/GPUDataStructures.h>
#include "KernelHelpers.h"
#include "Utils.h"

namespace Pale {
    // -----------------------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------------------

    SYCL_EXTERNAL inline void addRenderProfileCounter(uint64_t *counter, uint64_t amount) {
        if (amount == 0u) {
            return;
        }

        sycl::atomic_ref<
            uint64_t,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space> atomicCounter(*counter);
        atomicCounter.fetch_add(amount);
    }

    SYCL_EXTERNAL inline void flushMeshBvhProfile(
        const GPUSceneBuffers &scene,
        uint64_t nodeTests,
        uint64_t nodeHits,
        uint64_t triangleTests,
        uint64_t triangleHits) {
        RenderProfilingCounters *counters = scene.profileCounters;
        if (counters == nullptr) {
            return;
        }

        addRenderProfileCounter(&counters->blasMeshNodeTests, nodeTests);
        addRenderProfileCounter(&counters->blasMeshNodeHits, nodeHits);
        addRenderProfileCounter(&counters->triangleTests, triangleTests);
        addRenderProfileCounter(&counters->triangleHits, triangleHits);
    }

    SYCL_EXTERNAL inline void flushPointBvhProfile(
        const GPUSceneBuffers &scene,
        uint64_t nodeTests,
        uint64_t nodeHits,
        uint64_t primitiveTests,
        uint64_t planeTests,
        uint64_t profileTests,
        uint64_t acceptedHits) {
        RenderProfilingCounters *counters = scene.profileCounters;
        if (counters == nullptr) {
            return;
        }

        addRenderProfileCounter(&counters->blasPointNodeTests, nodeTests);
        addRenderProfileCounter(&counters->blasPointNodeHits, nodeHits);
        addRenderProfileCounter(&counters->pointLeafPrimitiveTests, primitiveTests);
        addRenderProfileCounter(&counters->surfelPlaneTests, planeTests);
        addRenderProfileCounter(&counters->surfelProfileTests, profileTests);
        addRenderProfileCounter(&counters->surfelAcceptedHits, acceptedHits);
    }

    SYCL_EXTERNAL inline void flushTlasProfile(
        const GPUSceneBuffers &scene,
        uint64_t rayQueries,
        uint64_t nodeTests,
        uint64_t nodeHits,
        uint64_t leafInstances) {
        RenderProfilingCounters *counters = scene.profileCounters;
        if (counters == nullptr) {
            return;
        }

        addRenderProfileCounter(&counters->sceneRayQueries, rayQueries);
        addRenderProfileCounter(&counters->tlasNodeTests, nodeTests);
        addRenderProfileCounter(&counters->tlasNodeHits, nodeHits);
        addRenderProfileCounter(&counters->tlasLeafInstances, leafInstances);
    }


    struct ChildEntry {
        uint32_t nodeIndex;
        float tEntry;
    };

    template<int MaxN = 256>
    struct TraversalEntryStack {
        ChildEntry data[MaxN];
        int sp = 0;

        bool push(uint32_t nodeIndex, float tEntry) {
            if (sp >= MaxN) return false;
            data[sp++] = ChildEntry{nodeIndex, tEntry};
            return true;
        }

        ChildEntry pop() {
            if (sp <= 0) return ChildEntry{0u, 0.0f};
            return data[--sp];
        }

        bool empty() const { return sp == 0; }
    };

    template<typename StackT>
    SYCL_EXTERNAL inline void pushNearFar(StackT &traversalStack,
                                          int leftIndex, float leftTEntry,
                                          int rightIndex, float rightTEntry) {
        if (leftTEntry <= rightTEntry) {
            traversalStack.push(rightIndex);
            traversalStack.push(leftIndex);
        } else {
            traversalStack.push(leftIndex);
            traversalStack.push(rightIndex);
        }
    }

    template<int MaxN>
    SYCL_EXTERNAL inline void pushNearFarEntries(TraversalEntryStack<MaxN> &traversalStack,
                                                 uint32_t leftIndex, float leftTEntry,
                                                 uint32_t rightIndex, float rightTEntry) {
        if (leftTEntry <= rightTEntry) {
            traversalStack.push(rightIndex, rightTEntry);
            traversalStack.push(leftIndex, leftTEntry);
        } else {
            traversalStack.push(leftIndex, leftTEntry);
            traversalStack.push(rightIndex, rightTEntry);
        }
    }

    SYCL_EXTERNAL inline bool tryGetSinglePointCloudInstance(
        const GPUSceneBuffers &scene,
        uint32_t &instanceIndexOut) {
        if (scene.triangleCount != 0u || scene.tlasNodeCount != 1u) {
            return false;
        }

        const TLASNode &root = scene.tlasNodes[0];
        if (root.count != 1u) {
            return false;
        }

        const uint32_t instanceIndex = root.leftChild;
        if (scene.instances[instanceIndex].geometryType != GeometryType::PointCloud) {
            return false;
        }

        instanceIndexOut = instanceIndex;
        return true;
    }

    SYCL_EXTERNAL inline bool tryGetPackedPointBvhRange(
        const GPUSceneBuffers &scene,
        uint32_t blasRangeIndex,
        BLASRange &packedRangeOut) {
        if (scene.pointPackedBvhNodes == nullptr || scene.pointPackedBvhRanges == nullptr) {
            return false;
        }
        if (blasRangeIndex >= scene.pointPackedBvhRangeCount) {
            return false;
        }

        const BLASRange packedRange = scene.pointPackedBvhRanges[blasRangeIndex];
        if (packedRange.nodeCount == 0u) {
            return false;
        }
        if (packedRange.firstNode + packedRange.nodeCount > scene.pointPackedBvhNodeCount) {
            return false;
        }

        packedRangeOut = packedRange;
        return true;
    }

    SYCL_EXTERNAL inline bool packedPointBvhSideValid(uint32_t childIndex, uint32_t childCount) {
        return childIndex != UINT32_MAX || childCount > 0u;
    }

    SYCL_EXTERNAL inline bool tryGetPointQbvhRange(
        const GPUSceneBuffers &scene,
        uint32_t blasRangeIndex,
        BLASRange &qbvhRangeOut) {
        if (scene.pointQbvhNodes == nullptr || scene.pointQbvhRanges == nullptr) {
            return false;
        }
        if (blasRangeIndex >= scene.pointQbvhRangeCount) {
            return false;
        }

        const BLASRange qbvhRange = scene.pointQbvhRanges[blasRangeIndex];
        if (qbvhRange.nodeCount == 0u) {
            return false;
        }
        if (qbvhRange.firstNode + qbvhRange.nodeCount > scene.pointQbvhNodeCount) {
            return false;
        }

        qbvhRangeOut = qbvhRange;
        return true;
    }

    SYCL_EXTERNAL inline bool pointQbvhChildValid(const PackedPointQBVHNode &node, uint32_t slot) {
        return node.childSourceNodeIndex[slot] != UINT32_MAX;
    }

    struct PointQBVHTraversalEntry {
        uint32_t childIndex;
        uint32_t childCount;
        float tEntry;
    };

    template<int MaxN = 64>
    struct PointQBVHTraversalStack {
        PointQBVHTraversalEntry data[MaxN];
        int sp = 0;

        bool push(uint32_t childIndex, uint32_t childCount, float tEntry) {
            if (sp >= MaxN) return false;
            data[sp++] = PointQBVHTraversalEntry{childIndex, childCount, tEntry};
            return true;
        }

        PointQBVHTraversalEntry pop() {
            if (sp <= 0) return PointQBVHTraversalEntry{0u, 0u, 0.0f};
            return data[--sp];
        }

        bool empty() const { return sp == 0; }
    };

    SYCL_EXTERNAL inline void insertPointQbvhHitSorted(
        uint32_t *hitChildIndices,
        uint32_t *hitChildCounts,
        float *hitTEntries,
        uint32_t &hitCount,
        uint32_t childIndex,
        uint32_t childCount,
        float tEntry) {
        uint32_t insertIndex = hitCount;
        while (insertIndex > 0u && tEntry < hitTEntries[insertIndex - 1u]) {
            hitChildIndices[insertIndex] = hitChildIndices[insertIndex - 1u];
            hitChildCounts[insertIndex] = hitChildCounts[insertIndex - 1u];
            hitTEntries[insertIndex] = hitTEntries[insertIndex - 1u];
            --insertIndex;
        }

        hitChildIndices[insertIndex] = childIndex;
        hitChildCounts[insertIndex] = childCount;
        hitTEntries[insertIndex] = tEntry;
        ++hitCount;
    }

    template<int MaxN>
    SYCL_EXTERNAL inline void pushPointQbvhHitsNearFirst(
        PointQBVHTraversalStack<MaxN> &traversalStack,
        const uint32_t *hitChildIndices,
        const uint32_t *hitChildCounts,
        const float *hitTEntries,
        uint32_t hitCount) {
        for (int hitIndex = static_cast<int>(hitCount) - 1; hitIndex >= 0; --hitIndex) {
            traversalStack.push(
                hitChildIndices[hitIndex],
                hitChildCounts[hitIndex],
                hitTEntries[hitIndex]);
        }
    }

    // -----------------------------------------------------------------------------
    // Triangle BLAS (unchanged except near-to-far child push)
    // -----------------------------------------------------------------------------
    SYCL_EXTERNAL static bool intersectBLASMesh(const Ray &rayObject,
                                                uint32_t geometryIndex,
                                                LocalHit &localHitOut,
                                                const GPUSceneBuffers &scene,
                                                const Transform &transform,
                                                float tMin = 0.0f) {
        const BLASRange &blasRange = scene.blasRanges[geometryIndex];
        const BVHNode *bvhNodes = scene.blasNodes + blasRange.firstNode;
        const Triangle *triangles = scene.triangles;
        const Vertex *vertices = scene.vertices;

        float bestTHit = std::numeric_limits<float>::infinity();
        bool hitAnyTriangle = false;
        const float3 inverseDirection = safeInvDir(rayObject.direction);
        const bool profileEnabled = scene.profileCounters != nullptr;
        uint64_t profileNodeTests = 0u;
        uint64_t profileNodeHits = 0u;
        uint64_t profileTriangleTests = 0u;
        uint64_t profileTriangleHits = 0u;

        TraversalEntryStack<64> traversalStack;
        float rootTEntry = 0.0f;
        if (profileEnabled) ++profileNodeTests;
        if (slabIntersectAABB(rayObject, bvhNodes[0], inverseDirection, bestTHit, rootTEntry)) {
            if (profileEnabled) ++profileNodeHits;
            traversalStack.push(0u, rootTEntry);
        }

        while (!traversalStack.empty()) {
            const ChildEntry stackEntry = traversalStack.pop();
            if (stackEntry.tEntry > bestTHit) {
                continue;
            }
            const uint32_t nodeIndex = stackEntry.nodeIndex;
            const BVHNode &node = bvhNodes[nodeIndex];

            if (node.triCount == 0) {
                // Internal: left child is node.leftFirst, right child is node.leftFirst + 1
                const uint32_t leftIndex = node.leftFirst;
                const uint32_t rightIndex = node.leftFirst + 1;

                float leftTEntry = std::numeric_limits<float>::infinity();
                float rightTEntry = std::numeric_limits<float>::infinity();

                if (profileEnabled) profileNodeTests += 2u;
                const bool hitLeft = slabIntersectAABB(rayObject, bvhNodes[leftIndex], inverseDirection, bestTHit,
                                                       leftTEntry);
                const bool hitRight = slabIntersectAABB(rayObject, bvhNodes[rightIndex], inverseDirection, bestTHit,
                                                        rightTEntry);
                if (profileEnabled) {
                    profileNodeHits += static_cast<uint64_t>(hitLeft) + static_cast<uint64_t>(hitRight);
                }

                if (hitLeft && hitRight) {
                    pushNearFarEntries(traversalStack, leftIndex, leftTEntry, rightIndex, rightTEntry);
                } else if (hitLeft) {
                    traversalStack.push(leftIndex, leftTEntry);
                } else if (hitRight) {
                    traversalStack.push(rightIndex, rightTEntry);
                }
                continue;
            }

            // Leaf: test triangles
            for (uint32_t i = 0; i < node.triCount; ++i) {
                uint32_t triangleIndex = node.leftFirst + i; // global index
                const Triangle &tri = triangles[triangleIndex];

                const float3 A = vertices[tri.v0].pos;
                const float3 B = vertices[tri.v1].pos;
                const float3 C = vertices[tri.v2].pos;

                float t = FLT_MAX, u = 0.0f, v = 0.0f;
                if (profileEnabled) ++profileTriangleTests;
                if (intersectTriangle(rayObject, A, B, C, t, u, v, 1e-4f) && t < bestTHit && t > tMin) {
                    if (profileEnabled) ++profileTriangleHits;
                    bestTHit = t;
                    hitAnyTriangle = true;
                    localHitOut.t = t;
                    localHitOut.primitiveIndex = triangleIndex;
                    localHitOut.transmissivity = 0.0f; // opaque triangle

                    localHitOut.worldHit = toWorldPoint(rayObject.origin + t * rayObject.direction, transform);
                }
            }
        }

        flushMeshBvhProfile(scene, profileNodeTests, profileNodeHits, profileTriangleTests, profileTriangleHits);
        return hitAnyTriangle;
    }


    // ----------------------------------------------------------------------------
    // Point-cloud BLAS: single closest-hit query (no event list, no sorting).
    // Returns the first (nearest) surfel intersection in (ray.tMin, ray.tMax).
    // ----------------------------------------------------------------------------
    SYCL_EXTERNAL static bool intersectBLASPointCloudFirstHit(
        const Ray &rayObject,
        uint32_t blasRangeIndex,
        LocalHit &localHitOut,
        const GPUSceneBuffers &scene) {
        const BLASRange &blasRange = scene.blasRanges[blasRangeIndex];
        const BVHNode *bvhNodes = scene.blasNodes + blasRange.firstNode;
        if (scene.pointTraversalData == nullptr) {
            return false;
        }


        bool hitAny = false;
        float bestTHit = std::numeric_limits<float>::infinity();
        uint32_t bestSurfelIndex = 0u;

        float bestAlphaGeomAtHit = 0.0f;
        float3 bestHitLocal{0.0f};

        const float3 inverseDirection = safeInvDir(rayObject.direction);
        const bool profileEnabled = scene.profileCounters != nullptr;
        uint64_t profileNodeTests = 0u;
        uint64_t profileNodeHits = 0u;
        uint64_t profilePrimitiveTests = 0u;
        uint64_t profilePlaneTests = 0u;
        uint64_t profileProfileTests = 0u;
        uint64_t profileAcceptedHits = 0u;

        BLASRange qbvhRange{};
        if (tryGetPointQbvhRange(scene, blasRangeIndex, qbvhRange)) {
            const PackedPointQBVHNode *qbvhNodes = scene.pointQbvhNodes + qbvhRange.firstNode;

            auto processQbvhLeaf = [&](uint32_t firstTraversalIndex, uint32_t traversalCount) {
                for (uint32_t primitiveOffset = 0; primitiveOffset < traversalCount; ++primitiveOffset) {
                    const uint32_t traversalIndex = firstTraversalIndex + primitiveOffset;
                    const SurfelTraversalData &surfel = scene.pointTraversalData[traversalIndex];
                    const uint32_t primitiveIndex = surfel.primitiveIndex;
                    if (profileEnabled) ++profilePrimitiveTests;

                    float tHitLocal = 0.0f;
                    float alphaGeom = 0.0f;

                    if (profileEnabled && !surfel.isEmissive()) {
                        ++profilePlaneTests;
                    }
                    if (surfel.isEmissive() || !intersectSurfel(rayObject, surfel, RayEpsilon2, bestTHit, tHitLocal,
                                                                RayEpsilon2)) {
                        continue;
                    }

                    const float3 hitLocal = rayObject.origin + tHitLocal * rayObject.direction;
                    const float2 uv = phiInverse(hitLocal, surfel);
                    if (profileEnabled) ++profileProfileTests;
                    if (!opacityBeta(uv[0], uv[1], surfel, &alphaGeom) || alphaGeom <= 0.0f) {
                        continue;
                    }

                    if (profileEnabled) ++profileAcceptedHits;
                    hitAny = true;
                    bestTHit = tHitLocal;
                    bestSurfelIndex = primitiveIndex;
                    bestHitLocal = hitLocal;
                    bestAlphaGeomAtHit = alphaGeom;
                }
            };

            PointQBVHTraversalStack<64> traversalStack;
            traversalStack.push(0u, 0u, 0.0f);

            while (!traversalStack.empty()) {
                const PointQBVHTraversalEntry stackEntry = traversalStack.pop();
                if (stackEntry.tEntry > bestTHit) {
                    continue;
                }

                if (stackEntry.childCount > 0u) {
                    processQbvhLeaf(stackEntry.childIndex, stackEntry.childCount);
                    continue;
                }

                const PackedPointQBVHNode &node = qbvhNodes[stackEntry.childIndex];
                uint32_t hitChildIndices[4]{0u, 0u, 0u, 0u};
                uint32_t hitChildCounts[4]{0u, 0u, 0u, 0u};
                float hitTEntries[4]{
                    std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::infinity()
                };
                uint32_t hitCount = 0u;

                for (uint32_t slot = 0u; slot < 4u; ++slot) {
                    if (!pointQbvhChildValid(node, slot)) {
                        continue;
                    }
                    if (profileEnabled) ++profileNodeTests;

                    float childTEntry = std::numeric_limits<float>::infinity();
                    const bool hitChild = slabIntersectAABB(
                        rayObject,
                        float3{node.minX[slot], node.minY[slot], node.minZ[slot]},
                        float3{node.maxX[slot], node.maxY[slot], node.maxZ[slot]},
                        inverseDirection,
                        bestTHit,
                        childTEntry);

                    if (!hitChild) {
                        continue;
                    }

                    if (profileEnabled) ++profileNodeHits;
                    insertPointQbvhHitSorted(
                        hitChildIndices,
                        hitChildCounts,
                        hitTEntries,
                        hitCount,
                        node.childIndex[slot],
                        node.childCount[slot],
                        childTEntry);
                }

                pushPointQbvhHitsNearFirst(
                    traversalStack,
                    hitChildIndices,
                    hitChildCounts,
                    hitTEntries,
                    hitCount);
            }

            if (!hitAny) {
                flushPointBvhProfile(
                    scene,
                    profileNodeTests,
                    profileNodeHits,
                    profilePrimitiveTests,
                    profilePlaneTests,
                    profileProfileTests,
                    profileAcceptedHits);
                return false;
            }

            localHitOut.t = bestTHit;
            localHitOut.primitiveIndex = bestSurfelIndex;
            localHitOut.transmissivity = 1.0f;
            localHitOut.alpha = bestAlphaGeomAtHit;
            localHitOut.worldHit = bestHitLocal;

            flushPointBvhProfile(
                scene,
                profileNodeTests,
                profileNodeHits,
                profilePrimitiveTests,
                profilePlaneTests,
                profileProfileTests,
                profileAcceptedHits);
            return true;
        }

        BLASRange packedRange{};
        if (tryGetPackedPointBvhRange(scene, blasRangeIndex, packedRange)) {
            const PackedPointBVHNode *packedNodes = scene.pointPackedBvhNodes + packedRange.firstNode;

            auto processPackedLeaf = [&](uint32_t firstTraversalIndex, uint32_t traversalCount) {
                for (uint32_t primitiveOffset = 0; primitiveOffset < traversalCount; ++primitiveOffset) {
                    const uint32_t traversalIndex = firstTraversalIndex + primitiveOffset;
                    const SurfelTraversalData &surfel = scene.pointTraversalData[traversalIndex];
                    const uint32_t primitiveIndex = surfel.primitiveIndex;
                    if (profileEnabled) ++profilePrimitiveTests;

                    float tHitLocal = 0.0f;
                    float alphaGeom = 0.0f;

                    if (profileEnabled && !surfel.isEmissive()) {
                        ++profilePlaneTests;
                    }
                    if (surfel.isEmissive() || !intersectSurfel(rayObject, surfel, RayEpsilon2, bestTHit, tHitLocal,
                                                                RayEpsilon2)) {
                        continue;
                    }

                    const float3 hitLocal = rayObject.origin + tHitLocal * rayObject.direction;
                    const float2 uv = phiInverse(hitLocal, surfel);
                    if (profileEnabled) ++profileProfileTests;
                    if (!opacityBeta(uv[0], uv[1], surfel, &alphaGeom) || alphaGeom <= 0.0f) {
                        continue;
                    }

                    if (profileEnabled) ++profileAcceptedHits;
                    hitAny = true;
                    bestTHit = tHitLocal;
                    bestSurfelIndex = primitiveIndex;
                    bestHitLocal = hitLocal;
                    bestAlphaGeomAtHit = alphaGeom;
                }
            };

            TraversalEntryStack<64> traversalStack;
            traversalStack.push(0u, 0.0f);

            while (!traversalStack.empty()) {
                const ChildEntry stackEntry = traversalStack.pop();
                if (stackEntry.tEntry > bestTHit) {
                    continue;
                }

                const PackedPointBVHNode &node = packedNodes[stackEntry.nodeIndex];
                const bool leftValid = packedPointBvhSideValid(node.leftIndex, node.leftCount);
                const bool rightValid = packedPointBvhSideValid(node.rightIndex, node.rightCount);

                float leftTEntry = std::numeric_limits<float>::infinity();
                float rightTEntry = std::numeric_limits<float>::infinity();
                if (profileEnabled) {
                    profileNodeTests += static_cast<uint64_t>(leftValid) + static_cast<uint64_t>(rightValid);
                }

                const bool hitLeft = leftValid && slabIntersectAABB(
                    rayObject, node.leftAabbMin, node.leftAabbMax, inverseDirection, bestTHit, leftTEntry);
                const bool hitRight = rightValid && slabIntersectAABB(
                    rayObject, node.rightAabbMin, node.rightAabbMax, inverseDirection, bestTHit, rightTEntry);

                if (profileEnabled) {
                    profileNodeHits += static_cast<uint64_t>(hitLeft) + static_cast<uint64_t>(hitRight);
                }

                if (hitLeft && node.leftCount > 0u) {
                    processPackedLeaf(node.leftIndex, node.leftCount);
                }
                if (hitRight && node.rightCount > 0u) {
                    processPackedLeaf(node.rightIndex, node.rightCount);
                }

                const bool pushLeft = hitLeft && node.leftCount == 0u;
                const bool pushRight = hitRight && node.rightCount == 0u;
                if (pushLeft && pushRight) {
                    pushNearFarEntries(traversalStack, node.leftIndex, leftTEntry, node.rightIndex, rightTEntry);
                } else if (pushLeft) {
                    traversalStack.push(node.leftIndex, leftTEntry);
                } else if (pushRight) {
                    traversalStack.push(node.rightIndex, rightTEntry);
                }
            }

            if (!hitAny) {
                flushPointBvhProfile(
                    scene,
                    profileNodeTests,
                    profileNodeHits,
                    profilePrimitiveTests,
                    profilePlaneTests,
                    profileProfileTests,
                    profileAcceptedHits);
                return false;
            }

            localHitOut.t = bestTHit;
            localHitOut.primitiveIndex = bestSurfelIndex;
            localHitOut.transmissivity = 1.0f;
            localHitOut.alpha = bestAlphaGeomAtHit;
            localHitOut.worldHit = bestHitLocal;

            flushPointBvhProfile(
                scene,
                profileNodeTests,
                profileNodeHits,
                profilePrimitiveTests,
                profilePlaneTests,
                profileProfileTests,
                profileAcceptedHits);
            return true;
        }

        TraversalEntryStack<64> traversalStack;
        float rootTEntry = 0.0f;
        if (profileEnabled) ++profileNodeTests;
        if (slabIntersectAABB(rayObject, bvhNodes[0], inverseDirection, bestTHit, rootTEntry)) {
            if (profileEnabled) ++profileNodeHits;
            traversalStack.push(0u, rootTEntry);
        }

        while (!traversalStack.empty()) {
            const ChildEntry stackEntry = traversalStack.pop();
            if (stackEntry.tEntry > bestTHit) {
                continue;
            }
            const uint32_t nodeIndex = stackEntry.nodeIndex;
            const BVHNode &node = bvhNodes[nodeIndex];

            if (node.triCount == 0) {
                const uint32_t leftIndex = node.leftFirst;
                const uint32_t rightIndex = node.leftFirst + 1;

                float leftTEntry = std::numeric_limits<float>::infinity();
                float rightTEntry = std::numeric_limits<float>::infinity();

                if (profileEnabled) profileNodeTests += 2u;
                const bool hitLeft = slabIntersectAABB(rayObject, bvhNodes[leftIndex], inverseDirection, bestTHit,
                                                       leftTEntry);
                const bool hitRight = slabIntersectAABB(rayObject, bvhNodes[rightIndex], inverseDirection, bestTHit,
                                                        rightTEntry);
                if (profileEnabled) {
                    profileNodeHits += static_cast<uint64_t>(hitLeft) + static_cast<uint64_t>(hitRight);
                }

                if (hitLeft && hitRight) {
                    pushNearFarEntries(traversalStack, leftIndex, leftTEntry, rightIndex, rightTEntry);
                } else if (hitLeft) {
                    traversalStack.push(leftIndex, leftTEntry);
                } else if (hitRight) {
                    traversalStack.push(rightIndex, rightTEntry);
                }
                continue;
            }

            // Leaf: test surfels
            for (uint32_t primitiveOffset = 0; primitiveOffset < node.triCount; ++primitiveOffset) {
                const uint32_t traversalIndex = node.leftFirst + primitiveOffset;
                const SurfelTraversalData &surfel = scene.pointTraversalData[traversalIndex];
                const uint32_t primitiveIndex = surfel.primitiveIndex;
                if (profileEnabled) ++profilePrimitiveTests;

                float tHitLocal = 0.0f;
                float alphaGeom = 0.0f;

                if (profileEnabled && !surfel.isEmissive()) {
                    ++profilePlaneTests;
                }
                if (surfel.isEmissive() || !intersectSurfel(rayObject, surfel, RayEpsilon2, bestTHit, tHitLocal,
                                                            RayEpsilon2))
                    continue;

                float3 hitLocal = rayObject.origin + tHitLocal * rayObject.direction;
                const float2 uv = phiInverse(hitLocal, surfel);
                if (profileEnabled) ++profileProfileTests;
                if (!opacityBeta(uv[0], uv[1], surfel, &alphaGeom) || alphaGeom <= 0.0f) {
                    continue;
                }

                // Keep closest
                if (profileEnabled) ++profileAcceptedHits;
                hitAny = true;
                bestTHit = tHitLocal;
                bestSurfelIndex = primitiveIndex;
                bestHitLocal = hitLocal;
                bestAlphaGeomAtHit = alphaGeom;
            }
        }

        if (!hitAny) {
            flushPointBvhProfile(
                scene,
                profileNodeTests,
                profileNodeHits,
                profilePrimitiveTests,
                profilePlaneTests,
                profileProfileTests,
                profileAcceptedHits);
            return false;
        }

        // Populate output hit.
        // Note: exact field names depend on your LocalHit definition.
        localHitOut.t = bestTHit;
        localHitOut.primitiveIndex = bestSurfelIndex;

        // For this "first hit" query there is no prior absorption handled here.
        // If you want per-ray compositing, use alphaGeom at the caller.
        localHitOut.transmissivity = 1.0f;
        localHitOut.alpha = bestAlphaGeomAtHit;

        localHitOut.worldHit = bestHitLocal;


        flushPointBvhProfile(
            scene,
            profileNodeTests,
            profileNodeHits,
            profilePrimitiveTests,
            profilePlaneTests,
            profileProfileTests,
            profileAcceptedHits);
        return true;
    }

    SYCL_EXTERNAL static uint32_t collectBLASPointCloudLocalLayer(
        const Ray &rayWorld,
        const Ray &rayObject,
        uint32_t blasRangeIndex,
        const Transform &transform,
        float tMinWorld,
        float tMaxWorld,
        LocalSurfelLayerHit *localHits,
        uint32_t maxLocalHitCount,
        const GPUSceneBuffers &scene) {
        if (maxLocalHitCount == 0u) {
            return 0u;
        }
        if (scene.pointTraversalData == nullptr) {
            return 0u;
        }

        const BLASRange &blasRange = scene.blasRanges[blasRangeIndex];
        const BVHNode *bvhNodes = scene.blasNodes + blasRange.firstNode;
        const float3 inverseDirectionObject = safeInvDir(rayObject.direction);
        const float4 objectDirection4 = transform.worldToObject * float4{rayWorld.direction, 0.0f};
        const float3 objectDirection{objectDirection4.x(), objectDirection4.y(), objectDirection4.z()};
        const float objectTPerWorldT = sycl::fmax(
            sycl::sqrt(dot(objectDirection, objectDirection)),
            RayEpsilon);
        const float infinity = std::numeric_limits<float>::infinity();
        const bool hasFiniteWorldTMax = tMaxWorld < infinity;
        float objectTMaxLimit = hasFiniteWorldTMax ? tMaxWorld * objectTPerWorldT : infinity;
        uint32_t localHitCount = 0u;
        const bool profileEnabled = scene.profileCounters != nullptr;
        uint64_t profileNodeTests = 0u;
        uint64_t profileNodeHits = 0u;
        uint64_t profilePrimitiveTests = 0u;
        uint64_t profilePlaneTests = 0u;
        uint64_t profileProfileTests = 0u;
        uint64_t profileAcceptedHits = 0u;

        BLASRange qbvhRange{};
        if (tryGetPointQbvhRange(scene, blasRangeIndex, qbvhRange)) {
            const PackedPointQBVHNode *qbvhNodes = scene.pointQbvhNodes + qbvhRange.firstNode;

            auto processQbvhLeaf = [&](uint32_t firstTraversalIndex, uint32_t traversalCount) {
                for (uint32_t primitiveOffset = 0u; primitiveOffset < traversalCount; ++primitiveOffset) {
                    const uint32_t traversalIndex = firstTraversalIndex + primitiveOffset;
                    const SurfelTraversalData &surfel = scene.pointTraversalData[traversalIndex];
                    const uint32_t primitiveIndex = surfel.primitiveIndex;
                    if (profileEnabled) ++profilePrimitiveTests;

                    if (surfel.isEmissive()) {
                        continue;
                    }

                    float tHitObject = 0.0f;
                    float alphaGeom = 0.0f;
                    if (profileEnabled) ++profilePlaneTests;
                    if (!intersectSurfel(rayObject, surfel, RayEpsilon2, objectTMaxLimit, tHitObject,
                                         RayEpsilon2)) {
                        continue;
                    }

                    const float3 hitPositionObject = rayObject.origin + tHitObject * rayObject.direction;
                    const float2 uv = phiInverse(hitPositionObject, surfel);
                    if (profileEnabled) ++profileProfileTests;
                    if (!opacityBeta(uv[0], uv[1], surfel, &alphaGeom) || alphaGeom <= 0.0f) {
                        continue;
                    }

                    const float3 hitPositionW = toWorldPoint(hitPositionObject, transform);
                    const float tHitWorld = dot(hitPositionW - rayWorld.origin, rayWorld.direction);
                    if (tHitWorld < tMinWorld || tHitWorld > tMaxWorld) {
                        continue;
                    }

                    LocalSurfelLayerHit candidateHit{};
                    candidateHit.tWorld = tHitWorld;
                    candidateHit.primitiveIndex = primitiveIndex;
                    candidateHit.alphaGeom = alphaGeom;
                    candidateHit.hitPositionW = hitPositionW;
                    if (profileEnabled) ++profileAcceptedHits;
                    insertLocalSurfelLayerHit(
                        localHits,
                        localHitCount,
                        maxLocalHitCount,
                        candidateHit);

                    if (localHitCount == maxLocalHitCount) {
                        const float farthestBufferedTWorld =
                            sycl::fmin(tMaxWorld, localHits[maxLocalHitCount - 1u].tWorld);
                        objectTMaxLimit = sycl::fmin(
                            objectTMaxLimit,
                            farthestBufferedTWorld * objectTPerWorldT);
                    }
                }
            };

            PointQBVHTraversalStack<64> traversalStack;
            traversalStack.push(0u, 0u, 0.0f);

            while (!traversalStack.empty()) {
                const PointQBVHTraversalEntry stackEntry = traversalStack.pop();
                if (stackEntry.tEntry > objectTMaxLimit) {
                    continue;
                }

                if (stackEntry.childCount > 0u) {
                    processQbvhLeaf(stackEntry.childIndex, stackEntry.childCount);
                    continue;
                }

                const PackedPointQBVHNode &node = qbvhNodes[stackEntry.childIndex];
                uint32_t hitChildIndices[4]{0u, 0u, 0u, 0u};
                uint32_t hitChildCounts[4]{0u, 0u, 0u, 0u};
                float hitTEntries[4]{
                    std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::infinity()
                };
                uint32_t hitCount = 0u;

                for (uint32_t slot = 0u; slot < 4u; ++slot) {
                    if (!pointQbvhChildValid(node, slot)) {
                        continue;
                    }
                    if (profileEnabled) ++profileNodeTests;

                    float childTEntry = std::numeric_limits<float>::infinity();
                    const bool hitChild = slabIntersectAABB(
                        rayObject,
                        float3{node.minX[slot], node.minY[slot], node.minZ[slot]},
                        float3{node.maxX[slot], node.maxY[slot], node.maxZ[slot]},
                        inverseDirectionObject,
                        objectTMaxLimit,
                        childTEntry);

                    if (!hitChild) {
                        continue;
                    }

                    if (profileEnabled) ++profileNodeHits;
                    insertPointQbvhHitSorted(
                        hitChildIndices,
                        hitChildCounts,
                        hitTEntries,
                        hitCount,
                        node.childIndex[slot],
                        node.childCount[slot],
                        childTEntry);
                }

                pushPointQbvhHitsNearFirst(
                    traversalStack,
                    hitChildIndices,
                    hitChildCounts,
                    hitTEntries,
                    hitCount);
            }

            flushPointBvhProfile(
                scene,
                profileNodeTests,
                profileNodeHits,
                profilePrimitiveTests,
                profilePlaneTests,
                profileProfileTests,
                profileAcceptedHits);
            return localHitCount;
        }

        BLASRange packedRange{};
        if (tryGetPackedPointBvhRange(scene, blasRangeIndex, packedRange)) {
            const PackedPointBVHNode *packedNodes = scene.pointPackedBvhNodes + packedRange.firstNode;

            auto processPackedLeaf = [&](uint32_t firstTraversalIndex, uint32_t traversalCount) {
                for (uint32_t primitiveOffset = 0u; primitiveOffset < traversalCount; ++primitiveOffset) {
                    const uint32_t traversalIndex = firstTraversalIndex + primitiveOffset;
                    const SurfelTraversalData &surfel = scene.pointTraversalData[traversalIndex];
                    const uint32_t primitiveIndex = surfel.primitiveIndex;
                    if (profileEnabled) ++profilePrimitiveTests;

                    if (surfel.isEmissive()) {
                        continue;
                    }

                    float tHitObject = 0.0f;
                    float alphaGeom = 0.0f;
                    if (profileEnabled) ++profilePlaneTests;
                    if (!intersectSurfel(rayObject, surfel, RayEpsilon2, objectTMaxLimit, tHitObject,
                                         RayEpsilon2)) {
                        continue;
                    }

                    const float3 hitPositionObject = rayObject.origin + tHitObject * rayObject.direction;
                    const float2 uv = phiInverse(hitPositionObject, surfel);
                    if (profileEnabled) ++profileProfileTests;
                    if (!opacityBeta(uv[0], uv[1], surfel, &alphaGeom) || alphaGeom <= 0.0f) {
                        continue;
                    }

                    const float3 hitPositionW = toWorldPoint(hitPositionObject, transform);
                    const float tHitWorld = dot(hitPositionW - rayWorld.origin, rayWorld.direction);
                    if (tHitWorld < tMinWorld || tHitWorld > tMaxWorld) {
                        continue;
                    }

                    LocalSurfelLayerHit candidateHit{};
                    candidateHit.tWorld = tHitWorld;
                    candidateHit.primitiveIndex = primitiveIndex;
                    candidateHit.alphaGeom = alphaGeom;
                    candidateHit.hitPositionW = hitPositionW;
                    if (profileEnabled) ++profileAcceptedHits;
                    insertLocalSurfelLayerHit(
                        localHits,
                        localHitCount,
                        maxLocalHitCount,
                        candidateHit);

                    if (localHitCount == maxLocalHitCount) {
                        const float farthestBufferedTWorld =
                            sycl::fmin(tMaxWorld, localHits[maxLocalHitCount - 1u].tWorld);
                        objectTMaxLimit = sycl::fmin(
                            objectTMaxLimit,
                            farthestBufferedTWorld * objectTPerWorldT);
                    }
                }
            };

            TraversalEntryStack<64> traversalStack;
            traversalStack.push(0u, 0.0f);

            while (!traversalStack.empty()) {
                const ChildEntry stackEntry = traversalStack.pop();
                if (stackEntry.tEntry > objectTMaxLimit) {
                    continue;
                }

                const PackedPointBVHNode &node = packedNodes[stackEntry.nodeIndex];
                const bool leftValid = packedPointBvhSideValid(node.leftIndex, node.leftCount);
                const bool rightValid = packedPointBvhSideValid(node.rightIndex, node.rightCount);

                float leftTEntry = std::numeric_limits<float>::infinity();
                float rightTEntry = std::numeric_limits<float>::infinity();
                if (profileEnabled) {
                    profileNodeTests += static_cast<uint64_t>(leftValid) + static_cast<uint64_t>(rightValid);
                }

                const bool hitLeft = leftValid && slabIntersectAABB(
                    rayObject, node.leftAabbMin, node.leftAabbMax, inverseDirectionObject, objectTMaxLimit,
                    leftTEntry);
                const bool hitRight = rightValid && slabIntersectAABB(
                    rayObject, node.rightAabbMin, node.rightAabbMax, inverseDirectionObject, objectTMaxLimit,
                    rightTEntry);

                if (profileEnabled) {
                    profileNodeHits += static_cast<uint64_t>(hitLeft) + static_cast<uint64_t>(hitRight);
                }

                if (hitLeft && node.leftCount > 0u) {
                    processPackedLeaf(node.leftIndex, node.leftCount);
                }
                if (hitRight && node.rightCount > 0u) {
                    processPackedLeaf(node.rightIndex, node.rightCount);
                }

                const bool pushLeft = hitLeft && node.leftCount == 0u;
                const bool pushRight = hitRight && node.rightCount == 0u;
                if (pushLeft && pushRight) {
                    pushNearFarEntries(traversalStack, node.leftIndex, leftTEntry, node.rightIndex, rightTEntry);
                } else if (pushLeft) {
                    traversalStack.push(node.leftIndex, leftTEntry);
                } else if (pushRight) {
                    traversalStack.push(node.rightIndex, rightTEntry);
                }
            }

            flushPointBvhProfile(
                scene,
                profileNodeTests,
                profileNodeHits,
                profilePrimitiveTests,
                profilePlaneTests,
                profileProfileTests,
                profileAcceptedHits);
            return localHitCount;
        }

        TraversalEntryStack<64> traversalStack;
        float rootTEntry = 0.0f;
        if (profileEnabled) ++profileNodeTests;
        if (slabIntersectAABB(
                rayObject,
                bvhNodes[0],
                inverseDirectionObject,
                objectTMaxLimit,
                rootTEntry)) {
            if (profileEnabled) ++profileNodeHits;
            traversalStack.push(0u, rootTEntry);
        }
        while (!traversalStack.empty()) {
            const ChildEntry stackEntry = traversalStack.pop();
            if (stackEntry.tEntry > objectTMaxLimit) {
                continue;
            }
            const uint32_t nodeIndex = stackEntry.nodeIndex;
            const BVHNode &node = bvhNodes[nodeIndex];
            if (node.triCount == 0u) {
                const uint32_t leftIndex = node.leftFirst;
                const uint32_t rightIndex = node.leftFirst + 1;
                float leftTEntry = std::numeric_limits<float>::infinity();
                float rightTEntry = std::numeric_limits<float>::infinity();
                if (profileEnabled) profileNodeTests += 2u;
                const bool hitLeft = slabIntersectAABB(rayObject, bvhNodes[leftIndex], inverseDirectionObject,
                                                       objectTMaxLimit, leftTEntry);
                const bool hitRight = slabIntersectAABB(rayObject, bvhNodes[rightIndex], inverseDirectionObject,
                                                        objectTMaxLimit, rightTEntry);
                if (profileEnabled) {
                    profileNodeHits += static_cast<uint64_t>(hitLeft) + static_cast<uint64_t>(hitRight);
                }
                if (hitLeft && hitRight) {
                    pushNearFarEntries(traversalStack, leftIndex, leftTEntry, rightIndex, rightTEntry);
                } else if (hitLeft) {
                    traversalStack.push(leftIndex, leftTEntry);
                } else if (hitRight) {
                    traversalStack.push(rightIndex, rightTEntry);
                }
                continue;
            }

            for (uint32_t primitiveOffset = 0u; primitiveOffset < node.triCount; ++primitiveOffset) {
                const uint32_t traversalIndex = node.leftFirst + primitiveOffset;
                const SurfelTraversalData &surfel = scene.pointTraversalData[traversalIndex];
                const uint32_t primitiveIndex = surfel.primitiveIndex;
                if (profileEnabled) ++profilePrimitiveTests;
                // Preserve your current FirstHit behavior.
                if (surfel.isEmissive()) {
                    continue;
                }
                float tHitObject = 0.0f;
                float alphaGeom = 0.0f;
                float3 hitPositionObject(0.0f);
                if (profileEnabled) ++profilePlaneTests;
                if (!intersectSurfel(rayObject, surfel, RayEpsilon2, objectTMaxLimit, tHitObject,
                                     RayEpsilon2)) {
                    continue;
                }

                hitPositionObject = rayObject.origin + tHitObject * rayObject.direction;
                const float2 uv = phiInverse(hitPositionObject, surfel);
                if (profileEnabled) ++profileProfileTests;
                if (!opacityBeta(uv[0], uv[1], surfel, &alphaGeom) || alphaGeom <= 0.0f) {
                    continue;
                }

                const float3 hitPositionW = toWorldPoint(hitPositionObject, transform);
                const float tHitWorld = dot(hitPositionW - rayWorld.origin, rayWorld.direction);
                if (tHitWorld < tMinWorld || tHitWorld > tMaxWorld) {
                    continue;
                }
                LocalSurfelLayerHit candidateHit{};
                candidateHit.tWorld = tHitWorld;
                candidateHit.primitiveIndex = primitiveIndex;
                candidateHit.alphaGeom = alphaGeom;
                candidateHit.hitPositionW = hitPositionW;
                if (profileEnabled) ++profileAcceptedHits;
                insertLocalSurfelLayerHit(
                    localHits,
                    localHitCount,
                    maxLocalHitCount,
                    candidateHit);

                if (localHitCount == maxLocalHitCount) {
                    const float farthestBufferedTWorld =
                        sycl::fmin(tMaxWorld, localHits[maxLocalHitCount - 1u].tWorld);
                    objectTMaxLimit = sycl::fmin(
                        objectTMaxLimit,
                        farthestBufferedTWorld * objectTPerWorldT);
                }
            }
        }

        flushPointBvhProfile(
            scene,
            profileNodeTests,
            profileNodeHits,
            profilePrimitiveTests,
            profilePlaneTests,
            profileProfileTests,
            profileAcceptedHits);
        return localHitCount;
    }

    SYCL_EXTERNAL static uint32_t collectScenePointHitsDirect(
        const Ray &rayWorld,
        const GPUSceneBuffers &scene,
        float tMinWorld,
        float tMaxWorld,
        LocalSurfelLayerHit *hits,
        uint32_t maxHitCount,
        uint32_t &instanceIndexOut) {
        flushTlasProfile(scene, 1u, 0u, 0u, 0u);
        if (!tryGetSinglePointCloudInstance(scene, instanceIndexOut)) {
            return 0u;
        }

        const InstanceRecord &instance = scene.instances[instanceIndexOut];
        const Transform &transform = scene.transforms[instance.transformIndex];
        const Ray rayObject = toObjectSpace(rayWorld, transform);
        return collectBLASPointCloudLocalLayer(
            rayWorld,
            rayObject,
            instance.blasRangeIndex,
            transform,
            tMinWorld,
            tMaxWorld,
            hits,
            maxHitCount,
            scene);
    }

    struct PointCloudLocalLayer {
        uint32_t hitCount = 0u;
        float furthestT = 0.0f;
        float transmission = 1.0f;
        float opacity = 0.0f;
        float3 referenceNormalW{0.0f};
        LocalSurfelLayerHit hits[kMaxLocalSurfelHits];
        float alphaEff[kMaxLocalSurfelHits];
        float weight[kMaxLocalSurfelHits];
        float directLightEpsilon[kMaxLocalSurfelHits] = {RayEpsilon};
    };

    SYCL_EXTERNAL static PointCloudLocalLayer buildPointCloudLocalLayerFromHits(
        const Ray &rayWorld,
        const LocalSurfelLayerHit &firstHit,
        const LocalSurfelLayerHit *candidateHits,
        uint32_t candidateCount,
        const GPUSceneBuffers &scene,
        float localLayerDepthEpsilon,
        uint32_t maxLocalSurfelHits,
        float localLayerNormalCosineThreshold) {
        PointCloudLocalLayer layer{};
        layer.hitCount = 0u;
        layer.furthestT = firstHit.tWorld;
        layer.transmission = 1.0f;
        layer.opacity = 0.0f;

        if (maxLocalSurfelHits == 0u) return layer;
        if (firstHit.primitiveIndex == kInvalidIndex) return layer;

        const Point &referenceSurfel = scene.points[firstHit.primitiveIndex];

        layer.referenceNormalW = normalize(cross(referenceSurfel.tanU, referenceSurfel.tanV));
        if (dot(layer.referenceNormalW, -rayWorld.direction) < 0.0f) layer.referenceNormalW = -layer.referenceNormalW;

        // localLayerDepthEpsilon now represents physical slab thickness along
        // the anchor surface normal rather than distance along the viewing ray.
        //
        // For two parallel surfaces separated by normal distance delta:
        //
        //     delta = Delta_t * |n . d|
        //
        // hence:
        //
        //     Delta_t = delta / |n . d|.
        //
        // Clamp the denominator so a nearly tangent ray cannot generate an
        // arbitrarily long search interval.
        static constexpr float kMinLocalLayerViewCosine = 0.05f;

        const float viewCosine = sycl::fabs(dot(layer.referenceNormalW, rayWorld.direction));
        const float effectiveViewCosine = sycl::fmax(viewCosine, kMinLocalLayerViewCosine);
        const float raySearchDepth = localLayerDepthEpsilon / effectiveViewCosine;
        const float localTMax = firstHit.tWorld + raySearchDepth;

        // Then determine actual slab membership using physical normal distance
        // from the anchor rather than ray-depth distance.
        for (uint32_t candidateIndex = 0u;
             candidateIndex < candidateCount && layer.hitCount < maxLocalSurfelHits;
             ++candidateIndex) {
            const LocalSurfelLayerHit &candidate = candidateHits[candidateIndex];

            if (candidate.primitiveIndex == kInvalidIndex) continue;
            if (candidate.tWorld + RayEpsilon < firstHit.tWorld) continue;
            if (candidate.tWorld > localTMax) continue;

            const Point &candidateSurfel = scene.points[candidate.primitiveIndex];

            float3 candidateNormalW = normalize(cross(candidateSurfel.tanU, candidateSurfel.tanV));
            if (dot(candidateNormalW, -rayWorld.direction) < 0.0f) candidateNormalW = -candidateNormalW;

            // Keep the normal-consistency criterion. This prevents nearby but
            // differently oriented surfaces from becoming one unresolved slab.
            const float normalAgreement = dot(layer.referenceNormalW, candidateNormalW);
            if (normalAgreement < localLayerNormalCosineThreshold) continue;

            const float3 anchorToCandidate = candidate.hitPositionW - firstHit.hitPositionW;

            // View-independent slab depth:
            //
            //     d_perp = |(x_i - x_1) . n_1|
            //
            // rather than:
            //
            //     d_ray = |t_i - t_1|.
            const float normalDistance = sycl::fabs(dot(anchorToCandidate, layer.referenceNormalW));
            if (normalDistance > localLayerDepthEpsilon) continue;

            layer.hits[layer.hitCount] = candidate;
            ++layer.hitCount;
        }

        // Numerical fallback: the already established FirstHit must always
        // remain a constituent of its own slab.
        if (layer.hitCount == 0u) {
            layer.hits[0].tWorld = firstHit.tWorld;
            layer.hits[0].primitiveIndex = firstHit.primitiveIndex;
            layer.hits[0].alphaGeom = firstHit.alphaGeom;
            layer.hits[0].hitPositionW = firstHit.hitPositionW;
            layer.hitCount = 1u;
        }

        // -------------------------------------------------------------------------
        // Construct opacity/transmission for accepted slab constituents only.
        // -------------------------------------------------------------------------
        for (uint32_t localHitIndex = 0u; localHitIndex < layer.hitCount; ++localHitIndex) {
            const LocalSurfelLayerHit &localHit = layer.hits[localHitIndex];
            const Point &surfel = scene.points[localHit.primitiveIndex];

            layer.alphaEff[localHitIndex] = 0.0f;
            layer.weight[localHitIndex] = 0.0f;

            const float3 distanceFromLayerAnchorW = localHit.hitPositionW - firstHit.hitPositionW;

            const float directLightingEpsilon = sycl::fmax(
                RayEpsilon,
                sycl::sqrt(dot(distanceFromLayerAnchorW, distanceFromLayerAnchorW)));

            // Keep this per-constituent. At grazing incidence two constituents
            // can now be far apart along the ray even though they are physically
            // close in normal distance.
            layer.directLightEpsilon[localHitIndex] = localLayerDepthEpsilon;

            layer.furthestT = sycl::fmax(layer.furthestT, localHit.tWorld);

            const float alphaEff = sycl::clamp(surfel.opacity * localHit.alphaGeom, 0.0f, 1.0f);
            layer.alphaEff[localHitIndex] = alphaEff;

            layer.transmission *= sycl::fmax(0.0f, 1.0f - alphaEff);
        }

        layer.opacity = 1.0f - layer.transmission;

        // -------------------------------------------------------------------------
        // Average over all unresolved depth orders.
        // -------------------------------------------------------------------------
        float localLayerWeightSum = 0.0f;

        if (layer.opacity > 0.0f) {
            for (uint32_t localHitIndex = 0u; localHitIndex < layer.hitCount; ++localHitIndex) {
                const float alphaEff = layer.alphaEff[localHitIndex];
                if (alphaEff <= 0.0f) continue;

                float transmittancePolynomial[kMaxLocalSurfelHits];

                for (uint32_t coefficientIndex = 0u; coefficientIndex < maxLocalSurfelHits; ++coefficientIndex) {
                    transmittancePolynomial[coefficientIndex] = 0.0f;
                }

                transmittancePolynomial[0] = 1.0f;
                uint32_t polynomialDegree = 0u;

                for (uint32_t otherHitIndex = 0u; otherHitIndex < layer.hitCount; ++otherHitIndex) {
                    if (otherHitIndex == localHitIndex) continue;

                    const float otherAlphaEff = layer.alphaEff[otherHitIndex];
                    if (otherAlphaEff <= 0.0f) continue;

                    for (int32_t coefficientIndex = static_cast<int32_t>(polynomialDegree);
                         coefficientIndex >= 0;
                         --coefficientIndex) {
                        transmittancePolynomial[coefficientIndex + 1] -=
                                otherAlphaEff * transmittancePolynomial[coefficientIndex];
                    }

                    ++polynomialDegree;
                }

                float expectedPreviousTransmittance = 0.0f;

                for (uint32_t coefficientIndex = 0u; coefficientIndex <= polynomialDegree; ++coefficientIndex) {
                    expectedPreviousTransmittance +=
                            transmittancePolynomial[coefficientIndex] / static_cast<float>(coefficientIndex + 1u);
                }

                const float layerWeight =
                        alphaEff * sycl::clamp(expectedPreviousTransmittance, 0.0f, 1.0f);

                layer.weight[localHitIndex] = layerWeight;
                localLayerWeightSum += layerWeight;
            }
        }

        if (localLayerWeightSum > 1.0e-8f) {
            const float weightNormalization = layer.opacity / localLayerWeightSum;

            for (uint32_t localHitIndex = 0u; localHitIndex < layer.hitCount; ++localHitIndex) {
                layer.weight[localHitIndex] *= weightNormalization;
            }
        }

        return layer;
    }

    SYCL_EXTERNAL static PointCloudLocalLayer collectPointCloudLocalLayer(
        const Ray &rayWorld,
        const WorldHit &firstHit,
        const InstanceRecord &instance,
        const GPUSceneBuffers &scene,
        float localLayerDepthEpsilon,
        uint32_t maxLocalSurfelHits,
        float localLayerNormalCosineThreshold) {
        if (maxLocalSurfelHits == 0u) {
            return PointCloudLocalLayer{};
        }

        const Transform &transform = scene.transforms[instance.transformIndex];
        const Point &referenceSurfel = scene.points[firstHit.primitiveIndex];
        float3 referenceNormalW = normalize(cross(referenceSurfel.tanU, referenceSurfel.tanV));
        if (dot(referenceNormalW, -rayWorld.direction) < 0.0f) referenceNormalW = -referenceNormalW;

        static constexpr float kMinLocalLayerViewCosine = 0.05f;
        const float viewCosine = sycl::fabs(dot(referenceNormalW, rayWorld.direction));
        const float effectiveViewCosine = sycl::fmax(viewCosine, kMinLocalLayerViewCosine);
        const float raySearchDepth = localLayerDepthEpsilon / effectiveViewCosine;

        const float localTMin = firstHit.t;
        const float localTMax = firstHit.t + raySearchDepth;

        const Ray rayObject = toObjectSpace(rayWorld, transform);

        LocalSurfelLayerHit candidateHits[kMaxLocalSurfelHits];

        const uint32_t candidateCount = collectBLASPointCloudLocalLayer(
            rayWorld,
            rayObject,
            instance.blasRangeIndex,
            transform,
            localTMin,
            localTMax,
            candidateHits,
            maxLocalSurfelHits,
            scene);

        LocalSurfelLayerHit firstLocalHit{};
        firstLocalHit.tWorld = firstHit.t;
        firstLocalHit.primitiveIndex = firstHit.primitiveIndex;
        firstLocalHit.alphaGeom = firstHit.alphaGeom;
        firstLocalHit.hitPositionW = firstHit.hitPositionW;

        return buildPointCloudLocalLayerFromHits(
            rayWorld,
            firstLocalHit,
            candidateHits,
            candidateCount,
            scene,
            localLayerDepthEpsilon,
            maxLocalSurfelHits,
            localLayerNormalCosineThreshold);
    }

    // Transmit and only attenuate the ray.
    SYCL_EXTERNAL static bool intersectBLASPointCloudTransmit(
        const Ray &rayObject,
        uint32_t blasRangeIndex,
        LocalHit &localHitOut,
        const GPUSceneBuffers &scene) {
        const BLASRange &blasRange = scene.blasRanges[blasRangeIndex];
        const BVHNode *bvhNodes = scene.blasNodes + blasRange.firstNode;
        if (scene.pointTraversalData == nullptr) {
            return false;
        }

        float cumulativeTransmittance = 1.0f;

        // Find next closest surfel hit with t in (tMin, tMax).
        auto findNextClosestSurfel = [&](float tMin,
                                         float tMax,
                                         float &outTHit,
                                         uint32_t &outSurfelIndex,
                                         float &outAlphaGeomAtHit) -> bool {
            bool hitAny = false;
            float bestTHit = tMax;

            const float3 inverseDirection = safeInvDir(rayObject.direction);
            const bool profileEnabled = scene.profileCounters != nullptr;
            uint64_t profileNodeTests = 0u;
            uint64_t profileNodeHits = 0u;
            uint64_t profilePrimitiveTests = 0u;
            uint64_t profilePlaneTests = 0u;
            uint64_t profileProfileTests = 0u;
            uint64_t profileAcceptedHits = 0u;

            BLASRange qbvhRange{};
            if (tryGetPointQbvhRange(scene, blasRangeIndex, qbvhRange)) {
                const PackedPointQBVHNode *qbvhNodes = scene.pointQbvhNodes + qbvhRange.firstNode;

                auto processQbvhLeaf = [&](uint32_t firstTraversalIndex, uint32_t traversalCount) {
                    for (uint32_t primitiveOffset = 0; primitiveOffset < traversalCount; ++primitiveOffset) {
                        const uint32_t traversalIndex = firstTraversalIndex + primitiveOffset;
                        const SurfelTraversalData &surfel = scene.pointTraversalData[traversalIndex];
                        const uint32_t primitiveIndex = surfel.primitiveIndex;

                        if (profileEnabled) ++profilePrimitiveTests;

                        float tHitLocal = 0.0f;
                        float alphaGeom = 0.0f;
                        if (profileEnabled) ++profilePlaneTests;
                        if (!intersectSurfel(rayObject, surfel, RayEpsilon2, bestTHit, tHitLocal,
                                             RayEpsilon2)) {
                            continue;
                        }

                        const float3 hitLocal = rayObject.origin + tHitLocal * rayObject.direction;
                        const float2 uv = phiInverse(hitLocal, surfel);
                        if (profileEnabled) ++profileProfileTests;
                        if (!opacityBeta(uv[0], uv[1], surfel, &alphaGeom) || alphaGeom <= 0.0f) {
                            continue;
                        }

                        if (tHitLocal <= tMin) {
                            continue;
                        }

                        if (profileEnabled) ++profileAcceptedHits;
                        bestTHit = tHitLocal;
                        outSurfelIndex = primitiveIndex;
                        outAlphaGeomAtHit = alphaGeom;
                        hitAny = true;
                    }
                };

                PointQBVHTraversalStack<64> traversalStack;
                traversalStack.push(0u, 0u, 0.0f);

                while (!traversalStack.empty()) {
                    const PointQBVHTraversalEntry stackEntry = traversalStack.pop();
                    if (stackEntry.tEntry > bestTHit) {
                        continue;
                    }

                    if (stackEntry.childCount > 0u) {
                        processQbvhLeaf(stackEntry.childIndex, stackEntry.childCount);
                        continue;
                    }

                    const PackedPointQBVHNode &node = qbvhNodes[stackEntry.childIndex];
                    uint32_t hitChildIndices[4]{0u, 0u, 0u, 0u};
                    uint32_t hitChildCounts[4]{0u, 0u, 0u, 0u};
                    float hitTEntries[4]{
                        std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::infinity()
                    };
                    uint32_t hitCount = 0u;

                    for (uint32_t slot = 0u; slot < 4u; ++slot) {
                        if (!pointQbvhChildValid(node, slot)) {
                            continue;
                        }
                        if (profileEnabled) ++profileNodeTests;

                        float childTEntry = std::numeric_limits<float>::infinity();
                        const bool hitChild = slabIntersectAABB(
                            rayObject,
                            float3{node.minX[slot], node.minY[slot], node.minZ[slot]},
                            float3{node.maxX[slot], node.maxY[slot], node.maxZ[slot]},
                            inverseDirection,
                            bestTHit,
                            childTEntry);

                        if (!hitChild) {
                            continue;
                        }

                        if (profileEnabled) ++profileNodeHits;
                        insertPointQbvhHitSorted(
                            hitChildIndices,
                            hitChildCounts,
                            hitTEntries,
                            hitCount,
                            node.childIndex[slot],
                            node.childCount[slot],
                            childTEntry);
                    }

                    pushPointQbvhHitsNearFirst(
                        traversalStack,
                        hitChildIndices,
                        hitChildCounts,
                        hitTEntries,
                        hitCount);
                }

                flushPointBvhProfile(
                    scene,
                    profileNodeTests,
                    profileNodeHits,
                    profilePrimitiveTests,
                    profilePlaneTests,
                    profileProfileTests,
                    profileAcceptedHits);

                if (!hitAny) {
                    return false;
                }

                outTHit = bestTHit;
                return true;
            }

            BLASRange packedRange{};
            if (tryGetPackedPointBvhRange(scene, blasRangeIndex, packedRange)) {
                const PackedPointBVHNode *packedNodes = scene.pointPackedBvhNodes + packedRange.firstNode;

                auto processPackedLeaf = [&](uint32_t firstTraversalIndex, uint32_t traversalCount) {
                    for (uint32_t primitiveOffset = 0; primitiveOffset < traversalCount; ++primitiveOffset) {
                        const uint32_t traversalIndex = firstTraversalIndex + primitiveOffset;
                        const SurfelTraversalData &surfel = scene.pointTraversalData[traversalIndex];
                        const uint32_t primitiveIndex = surfel.primitiveIndex;

                        if (profileEnabled) ++profilePrimitiveTests;

                        float tHitLocal = 0.0f;
                        float alphaGeom = 0.0f;
                        if (profileEnabled) ++profilePlaneTests;
                        if (!intersectSurfel(rayObject, surfel, RayEpsilon2, bestTHit, tHitLocal,
                                             RayEpsilon2)) {
                            continue;
                        }

                        const float3 hitLocal = rayObject.origin + tHitLocal * rayObject.direction;
                        const float2 uv = phiInverse(hitLocal, surfel);
                        if (profileEnabled) ++profileProfileTests;
                        if (!opacityBeta(uv[0], uv[1], surfel, &alphaGeom) || alphaGeom <= 0.0f) {
                            continue;
                        }

                        if (tHitLocal <= tMin) {
                            continue;
                        }

                        if (profileEnabled) ++profileAcceptedHits;
                        bestTHit = tHitLocal;
                        outSurfelIndex = primitiveIndex;
                        outAlphaGeomAtHit = alphaGeom;
                        hitAny = true;
                    }
                };

                TraversalEntryStack<64> traversalStack;
                traversalStack.push(0u, 0.0f);

                while (!traversalStack.empty()) {
                    const ChildEntry stackEntry = traversalStack.pop();
                    if (stackEntry.tEntry > bestTHit) {
                        continue;
                    }

                    const PackedPointBVHNode &node = packedNodes[stackEntry.nodeIndex];
                    const bool leftValid = packedPointBvhSideValid(node.leftIndex, node.leftCount);
                    const bool rightValid = packedPointBvhSideValid(node.rightIndex, node.rightCount);

                    float leftTEntry = std::numeric_limits<float>::infinity();
                    float rightTEntry = std::numeric_limits<float>::infinity();
                    if (profileEnabled) {
                        profileNodeTests += static_cast<uint64_t>(leftValid) + static_cast<uint64_t>(rightValid);
                    }

                    const bool hitLeft = leftValid && slabIntersectAABB(
                        rayObject, node.leftAabbMin, node.leftAabbMax, inverseDirection, bestTHit, leftTEntry);
                    const bool hitRight = rightValid && slabIntersectAABB(
                        rayObject, node.rightAabbMin, node.rightAabbMax, inverseDirection, bestTHit, rightTEntry);

                    if (profileEnabled) {
                        profileNodeHits += static_cast<uint64_t>(hitLeft) + static_cast<uint64_t>(hitRight);
                    }

                    if (hitLeft && node.leftCount > 0u) {
                        processPackedLeaf(node.leftIndex, node.leftCount);
                    }
                    if (hitRight && node.rightCount > 0u) {
                        processPackedLeaf(node.rightIndex, node.rightCount);
                    }

                    const bool pushLeft = hitLeft && node.leftCount == 0u;
                    const bool pushRight = hitRight && node.rightCount == 0u;
                    if (pushLeft && pushRight) {
                        pushNearFarEntries(traversalStack, node.leftIndex, leftTEntry, node.rightIndex, rightTEntry);
                    } else if (pushLeft) {
                        traversalStack.push(node.leftIndex, leftTEntry);
                    } else if (pushRight) {
                        traversalStack.push(node.rightIndex, rightTEntry);
                    }
                }

                flushPointBvhProfile(
                    scene,
                    profileNodeTests,
                    profileNodeHits,
                    profilePrimitiveTests,
                    profilePlaneTests,
                    profileProfileTests,
                    profileAcceptedHits);

                if (!hitAny) {
                    return false;
                }

                outTHit = bestTHit;
                return true;
            }

            TraversalEntryStack<64> traversalStack;
            float rootTEntry = 0.0f;
            if (profileEnabled) ++profileNodeTests;
            if (slabIntersectAABB(rayObject, bvhNodes[0], inverseDirection, bestTHit, rootTEntry)) {
                if (profileEnabled) ++profileNodeHits;
                traversalStack.push(0u, rootTEntry);
            }

            while (!traversalStack.empty()) {
                const ChildEntry stackEntry = traversalStack.pop();
                if (stackEntry.tEntry > bestTHit) {
                    continue;
                }
                const uint32_t nodeIndex = stackEntry.nodeIndex;
                const BVHNode &node = bvhNodes[nodeIndex];

                if (node.triCount == 0) {
                    const uint32_t leftIndex = node.leftFirst;
                    const uint32_t rightIndex = node.leftFirst + 1;

                    float leftTEntry = std::numeric_limits<float>::infinity();
                    float rightTEntry = std::numeric_limits<float>::infinity();

                    if (profileEnabled) profileNodeTests += 2u;
                    const bool hitLeft = slabIntersectAABB(rayObject, bvhNodes[leftIndex], inverseDirection, bestTHit,
                                                           leftTEntry);
                    const bool hitRight = slabIntersectAABB(rayObject, bvhNodes[rightIndex], inverseDirection, bestTHit,
                                                            rightTEntry);
                    if (profileEnabled) {
                        profileNodeHits += static_cast<uint64_t>(hitLeft) + static_cast<uint64_t>(hitRight);
                    }

                    if (hitLeft && hitRight)
                        pushNearFarEntries(traversalStack, leftIndex, leftTEntry, rightIndex, rightTEntry);
                    else if (hitLeft) traversalStack.push(leftIndex, leftTEntry);
                    else if (hitRight) traversalStack.push(rightIndex, rightTEntry);
                    continue;
                }

                // Leaf: test surfels
                for (uint32_t primitiveOffset = 0; primitiveOffset < node.triCount; ++primitiveOffset) {
                    const uint32_t traversalIndex = node.leftFirst + primitiveOffset;
                    const SurfelTraversalData &surfel = scene.pointTraversalData[traversalIndex];
                    const uint32_t primitiveIndex = surfel.primitiveIndex;

                    if (profileEnabled) ++profilePrimitiveTests;

                    float tHitLocal = 0.0f;
                    float alphaGeom = 0.0f;
                    float3 hitLocal{};
                    if (profileEnabled) ++profilePlaneTests;
                    if (!intersectSurfel(rayObject, surfel, RayEpsilon2, bestTHit, tHitLocal,
                                         RayEpsilon2))
                        continue;

                    hitLocal = rayObject.origin + tHitLocal * rayObject.direction;
                    const float2 uv = phiInverse(hitLocal, surfel);
                    if (profileEnabled) ++profileProfileTests;
                    if (!opacityBeta(uv[0], uv[1], surfel, &alphaGeom) || alphaGeom <= 0.0f) {
                        continue;
                    }

                    if (tHitLocal <= tMin)
                        continue;

                    // Keep closest
                    if (profileEnabled) ++profileAcceptedHits;
                    bestTHit = tHitLocal;
                    outSurfelIndex = primitiveIndex;
                    outAlphaGeomAtHit = alphaGeom;
                    hitAny = true;
                }
            }

            flushPointBvhProfile(
                scene,
                profileNodeTests,
                profileNodeHits,
                profilePrimitiveTests,
                profilePlaneTests,
                profileProfileTests,
                profileAcceptedHits);

            if (!hitAny)
                return false;

            outTHit = bestTHit;
            return true;
        };


        // Stochastic accept/reject loop over successive closest hits
        float tMin = RayEpsilon2;
        while (true) {
            float tHit = 0.0f;
            uint32_t surfelIndex = UINT32_MAX;
            float alphaGeomAtHit = 0.0f;

            if (!findNextClosestSurfel(tMin, std::numeric_limits<float>::infinity(), tHit, surfelIndex,
                                       alphaGeomAtHit)) {
                // No more candidates: pure transmission through this BLAS
                localHitOut.transmissivity = cumulativeTransmittance;
                break;
            }

            float alphaEff = scene.points[surfelIndex].opacity * alphaGeomAtHit;
            float tau = 1.0f - alphaEff;
            cumulativeTransmittance *= tau;
            tMin = tHit + RayEpsilon2;

            // stop early if we reach 0 transmittance
            if ((cumulativeTransmittance) <= 0.001f) {
                break;
            }
        }

        localHitOut.transmissivity = cumulativeTransmittance;
        return false;
    }


    // -----------------------------------------------------------------------------
    // TLAS traversal with near-to-far ordering and multiplicative transmittance
    // -----------------------------------------------------------------------------
    SYCL_EXTERNAL static bool intersectScene(const Ray &rayWorld,
                                             WorldHit *worldHitOut,
                                             const GPUSceneBuffers &scene,
                                             SurfelIntersectMode rayIntersectMode = SurfelIntersectMode::FirstHit) {
        const TLASNode *tlasNodes = scene.tlasNodes;
        const InstanceRecord *instanceRecords = scene.instances;
        const Transform *transforms = scene.transforms;

        bool foundAnySurfaceHit = false;
        const float3 inverseDirectionWorld = safeInvDir(rayWorld.direction);

        worldHitOut->t = FLT_MAX;

        float bestWorldTHit = std::numeric_limits<float>::infinity();
        float transmittanceProduct = 1.0f; // accumulate product over visited splat instances in front of the first hit
        const bool profileEnabled = scene.profileCounters != nullptr;
        uint64_t profileNodeTests = 0u;
        uint64_t profileNodeHits = 0u;
        uint64_t profileLeafInstances = 0u;

        TraversalEntryStack<64> traversalStack;
        float rootTEntry = 0.0f;
        if (profileEnabled) ++profileNodeTests;
        if (slabIntersectAABB(rayWorld, tlasNodes[0], inverseDirectionWorld, bestWorldTHit, rootTEntry)) {
            if (profileEnabled) ++profileNodeHits;
            traversalStack.push(0u, rootTEntry);
        }

        while (!traversalStack.empty()) {
            const ChildEntry stackEntry = traversalStack.pop();
            if (stackEntry.tEntry > bestWorldTHit) {
                continue;
            }
            const uint32_t nodeIndex = stackEntry.nodeIndex;
            const TLASNode &node = tlasNodes[nodeIndex];
            if (node.count == 0) {
                // Internal TLAS node: near-to-far push
                const uint32_t leftIndex = node.leftChild;
                const uint32_t rightIndex = node.rightChild;
                float leftTEntry = std::numeric_limits<float>::infinity();
                float rightTEntry = std::numeric_limits<float>::infinity();
                if (profileEnabled) profileNodeTests += 2u;
                const bool hitLeft = slabIntersectAABB(rayWorld, tlasNodes[leftIndex], inverseDirectionWorld,
                                                       bestWorldTHit, leftTEntry);
                const bool hitRight = slabIntersectAABB(rayWorld, tlasNodes[rightIndex], inverseDirectionWorld,
                                                        bestWorldTHit, rightTEntry);
                if (profileEnabled) {
                    profileNodeHits += static_cast<uint64_t>(hitLeft) + static_cast<uint64_t>(hitRight);
                }
                if (hitLeft && hitRight) {
                    pushNearFarEntries(traversalStack, leftIndex, leftTEntry, rightIndex, rightTEntry);
                } else if (hitLeft) {
                    traversalStack.push(leftIndex, leftTEntry);
                } else if (hitRight) {
                    traversalStack.push(rightIndex, rightTEntry);
                }
                continue;
            }

            // Leaf: exactly one instance
            if (profileEnabled) ++profileLeafInstances;
            const uint32_t instanceIndex = node.leftChild;
            const InstanceRecord &instance = instanceRecords[instanceIndex];
            const Transform &transform = transforms[instance.transformIndex];
            Ray rayObject = toObjectSpace(rayWorld, transform);
            LocalHit localHit{};
            bool acceptedHitInInstance = false;
            if (instance.geometryType == GeometryType::Mesh) {
                acceptedHitInInstance = intersectBLASMesh(rayObject, instance.blasRangeIndex, localHit, scene,
                                                          transform);
            } else {
                switch (rayIntersectMode) {
                    case SurfelIntersectMode::Transmit:
                        acceptedHitInInstance = intersectBLASPointCloudTransmit(
                            rayObject, instance.blasRangeIndex, localHit, scene);
                        break;
                    case SurfelIntersectMode::FirstHit:
                        acceptedHitInInstance = intersectBLASPointCloudFirstHit(
                            rayObject,
                            instance.blasRangeIndex,
                            localHit,
                            scene);
                        break;
                    default: ;
                }
            }

            if (acceptedHitInInstance) {
                // Convert to world, test depth
                const float3 hitWorld = localHit.worldHit; // you already compute this in BLAS
                const float3 toHitWorld = hitWorld - rayWorld.origin;
                const float tWorld = dot(toHitWorld, rayWorld.direction);
                if (tWorld > 0.0f && tWorld < bestWorldTHit) {
                    bestWorldTHit = tWorld;
                    foundAnySurfaceHit = true;
                    worldHitOut->hit = true;
                    worldHitOut->t = tWorld;
                    worldHitOut->hitPositionW = hitWorld;
                    worldHitOut->instanceIndex = instanceIndex;
                    worldHitOut->primitiveIndex = localHit.primitiveIndex;
                    worldHitOut->alphaGeom = localHit.alpha;
                    if (instance.geometryType == GeometryType::PointCloud) {
                        transmittanceProduct *= localHit.transmissivity;
                    }
                }
            }
            // No accepted hit, but if this was a splat field we may have partial transmission through it
            if (!acceptedHitInInstance && instance.geometryType == GeometryType::PointCloud) {
                transmittanceProduct *= localHit.transmissivity;
            }
        }
        // If no surface hit at all, expose total transmission accumulated
        if (!foundAnySurfaceHit) {
            worldHitOut->hit = false;
        }
        worldHitOut->transmissivity = transmittanceProduct;
        flushTlasProfile(scene, 1u, profileNodeTests, profileNodeHits, profileLeafInstances);
        return foundAnySurfaceHit;
    }

    SYCL_EXTERNAL inline float traceShadowTransmissionToPoint(
        const GPUSceneBuffers &scene,
        const PathTracerSettings &settings,
        const float3 &shadingPositionW,
        const float3 &shadingNormalW,
        const float3 &lightPositionW,
        const float eps) {
        const uint32_t maxSplatEventsPerRay =
                rendererDebugMaxSplatEventsPerRay(settings);
        const uint32_t maxLocalSurfelHits =
                rendererDebugMaxLocalSurfelHits(settings);
        const float3 lightVector = lightPositionW - shadingPositionW;
        const float lightDistanceSquared = dot(lightVector, lightVector);

        if (lightDistanceSquared <= 1.0e-12f) {
            return 0.0f;
        }

        const float lightDistance = sycl::sqrt(lightDistanceSquared);
        const float3 lightDirection = lightVector / lightDistance;

        Ray shadowRay{};
        shadowRay.origin = shadingPositionW + shadingNormalW * eps;
        shadowRay.direction = lightDirection;
        shadowRay.normal = shadingNormalW;

        float shadowTransmission = 1.0f;
        const uint32_t pointHitBatchSize = rendererDebugPointHitBatchSize(settings);
        uint32_t directPointInstanceIndex = kInvalidIndex;

        if (pointHitBatchSize > 1u &&
            tryGetSinglePointCloudInstance(scene, directPointInstanceIndex)) {
            for (uint32_t hitIndex = 0u; hitIndex < maxSplatEventsPerRay;) {
                const float remainingLightDistance = dot(
                    lightPositionW - shadowRay.origin,
                    shadowRay.direction);

                if (remainingLightDistance <= eps) {
                    break;
                }

                LocalSurfelLayerHit pointHits[kMaxPointHitBatch];
                uint32_t pointInstanceIndex = kInvalidIndex;
                const uint32_t remainingHitBudget = maxSplatEventsPerRay - hitIndex;
                const uint32_t batchCapacity =
                    pointHitBatchSize < remainingHitBudget ? pointHitBatchSize : remainingHitBudget;
                const uint32_t hitCount = collectScenePointHitsDirect(
                    shadowRay,
                    scene,
                    eps,
                    remainingLightDistance - eps,
                    pointHits,
                    batchCapacity,
                    pointInstanceIndex);

                if (hitCount == 0u) {
                    break;
                }

                float furthestConsumedT = 0.0f;
                for (uint32_t batchIndex = 0u; batchIndex < hitCount; ++batchIndex) {
                    const LocalSurfelLayerHit &pointHit = pointHits[batchIndex];
                    if (pointHit.tWorld >= remainingLightDistance - eps) {
                        break;
                    }

                    furthestConsumedT = sycl::fmax(furthestConsumedT, pointHit.tWorld);
                    const Point &surfel = scene.points[pointHit.primitiveIndex];
                    const float alphaEff = sycl::clamp(
                        surfel.opacity * pointHit.alphaGeom,
                        0.0f,
                        1.0f - 1.0e-6f);

                    if (alphaEff > 0.0f) {
                        shadowTransmission *= sycl::fmax(0.0f, 1.0f - alphaEff);
                    }

                    ++hitIndex;
                    if (shadowTransmission <= 1.0e-6f) {
                        return shadowTransmission;
                    }
                }

                if (furthestConsumedT <= 0.0f) {
                    break;
                }

                shadowRay.origin += shadowRay.direction * (furthestConsumedT + eps);
                if (hitCount < batchCapacity) {
                    break;
                }
            }

            return shadowTransmission;
        }

        for (uint32_t shadowTraversalIndex = 0u;
             shadowTraversalIndex < maxSplatEventsPerRay;
             ++shadowTraversalIndex) {
            const float remainingLightDistance = dot(
                lightPositionW - shadowRay.origin,
                shadowRay.direction);

            if (remainingLightDistance <= eps) {
                break;
            }

            WorldHit shadowHit{};
            intersectScene(
                shadowRay,
                &shadowHit,
                scene,
                SurfelIntersectMode::FirstHit);

            if (!shadowHit.hit) {
                break;
            }

            // The nearest hit lies at or beyond the sampled light point.
            if (shadowHit.t >= remainingLightDistance - eps) {
                break;
            }

            const InstanceRecord &hitInstance =
                    scene.instances[shadowHit.instanceIndex];

            if (hitInstance.geometryType == GeometryType::Mesh) {
                return 0.0f;
            }

            if (hitInstance.geometryType != GeometryType::PointCloud) {
                return 0.0f;
            }

            const Transform &transform =
                    scene.transforms[hitInstance.transformIndex];

            const Ray shadowRayObject =
                    toObjectSpace(shadowRay, transform);

            const float localLayerStartSlack = 4.0f * eps;
            const float localTMin = sycl::fmax(eps, shadowHit.t - localLayerStartSlack);
            const float localTMax = sycl::fmin(shadowHit.t + eps,
                                               remainingLightDistance - eps);
            LocalSurfelLayerHit localHits[kMaxLocalSurfelHits];

            uint32_t localHitCount =
                    collectBLASPointCloudLocalLayer(
                        shadowRay,
                        shadowRayObject,
                        hitInstance.blasRangeIndex,
                        transform,
                        localTMin,
                        localTMax,
                        localHits,
                        maxLocalSurfelHits,
                        scene);

            // Preserve the already-found closest hit if local collection fails.
            if (localHitCount == 0u) {
                localHits[0].tWorld = shadowHit.t;
                localHits[0].primitiveIndex = shadowHit.primitiveIndex;
                localHits[0].alphaGeom = shadowHit.alphaGeom;
                localHits[0].hitPositionW = shadowHit.hitPositionW;
                localHitCount = 1u;
            }

            float combinedOpticalDepth = 0.0f;
            float furthestLayerT = shadowHit.t;

            for (uint32_t localHitIndex = 0u;
                 localHitIndex < localHitCount;
                 ++localHitIndex) {
                const LocalSurfelLayerHit &localHit =
                        localHits[localHitIndex];

                // Never let hits at or behind the light sample block it.
                if (localHit.tWorld >= remainingLightDistance - eps) {
                    continue;
                }

                furthestLayerT = sycl::fmax(
                    furthestLayerT,
                    localHit.tWorld);

                const Point &surfel =
                        scene.points[localHit.primitiveIndex];

                const float alphaEff = sycl::clamp(
                    surfel.opacity * localHit.alphaGeom,
                    0.0f,
                    1.0f - 1.0e-6f);

                if (alphaEff <= 0.0f) {
                    continue;
                }

                combinedOpticalDepth +=
                        -sycl::log(1.0f - alphaEff);
            }

            const float localLayerTransmission =
                    sycl::exp(-combinedOpticalDepth);

            shadowTransmission *= localLayerTransmission;

            if (shadowTransmission <= 1.0e-6f) {
                return shadowTransmission;
            }

            // Advance past every surfel already absorbed in this local layer.
            shadowRay.origin +=
                    shadowRay.direction *
                    (furthestLayerT + eps);
        }

        return shadowTransmission;
    }

    /*
    SYCL_EXTERNAL inline float traceShadowTransmissionToPoint(
        const GPUSceneBuffers &scene,
        const float3 &shadingPositionW,
        const float3 &shadingNormalW,
        const float3 &lightPositionW) {
        const float3 lightVector = lightPositionW - shadingPositionW;
        const float lightDistanceSquared = dot(lightVector, lightVector);
        if (lightDistanceSquared <= 1e-12f) {
            return 0.0f;
        }

        const float lightDistance = sycl::sqrt(lightDistanceSquared);
        const float3 lightDirection = lightVector / lightDistance;


        Ray shadowRay{};
        shadowRay.origin = shadingPositionW + shadingNormalW * RayEpsilon2;
        shadowRay.direction = lightDirection;
        shadowRay.normal = shadingNormalW;

        float shadowTransmission = 1.0f;

        for (uint32_t shadowTraversalIndex = 0u;
             shadowTraversalIndex < kMaxSplatEventsPerRay;
             ++shadowTraversalIndex) {
            WorldHit shadowHit{};
            intersectScene(
                shadowRay,
                &shadowHit,
                scene,
                SurfelIntersectMode::FirstHit);

            if (!shadowHit.hit) {
                break;
            }

            const float3 hitVector = shadowHit.hitPositionW - shadingPositionW;
            const float hitDistance = sycl::sqrt(dot(hitVector, hitVector));

            if (hitDistance >= lightDistance - RayEpsilon2) {
                break;
            }

            const InstanceRecord &hitInstance = scene.instances[shadowHit.instanceIndex];

            if (hitInstance.geometryType == GeometryType::Mesh) {
                return 0.0f;
            }

            if (hitInstance.geometryType == GeometryType::PointCloud) {
                const Point &surfel = scene.points[shadowHit.primitiveIndex];
                const float oneMinusAlpha = 1.0f - surfel.opacity * shadowHit.alphaGeom;
                shadowTransmission *= sycl::fmax(0.0f, oneMinusAlpha);

                if (shadowTransmission <= 1e-6f) {
                    return shadowTransmission;
                }

                shadowRay.origin = shadowHit.hitPositionW + shadowRay.direction * RayEpsilon2;
                continue;
            }

            return 0.0f;
        }

        return shadowTransmission;
    }
    */

    SYCL_EXTERNAL inline float3 estimateDirectPointSampledPointLights(
        const GPUSceneBuffers &scene,
        const PathTracerSettings &settings,
        const float3 &surfacePositionW,
        const float3 &surfaceNormalW,
        const float3 &diffuseAlbedo,
        const float eps = RayEpsilon) {
        float3 accumulatedRadiance(0.0f);

        const float3 diffuseBrdf = diffuseAlbedo * M_1_PIf;

        for (uint32_t lightIndex = 0u;
             lightIndex < scene.lightCount;
             ++lightIndex) {
            const GPULightRecord &light = scene.lights[lightIndex];

            // Your current light records appear to use emissive surfels as
            // light-position carriers. We interpret each as an isotropic point light.
            if (light.lightType != LightType::Surfel) {
                continue;
            }

            const Point &lightSurfel = scene.points[light.primitiveIndex];
            const float3 lightPositionW = lightSurfel.position;

            const float3 toLight = lightPositionW - surfacePositionW;
            const float distanceSquared = dot(toLight, toLight);

            if (distanceSquared <= 1.0e-12f) {
                continue;
            }

            const float distance = sycl::sqrt(distanceSquared);
            const float3 lightDirection = toLight / distance;

            const float surfaceCosine =
                    sycl::fmax(0.0f, dot(surfaceNormalW, lightDirection));

            if (surfaceCosine <= 0.0f) {
                continue;
            }

            const float shadowTransmission =
                    traceShadowTransmissionToPoint(
                        scene,
                        settings,
                        surfacePositionW,
                        surfaceNormalW,
                        lightPositionW, eps);


            if (shadowTransmission <= 0.0f) {
                continue;
            }

            // Treat light.flux as total radiant flux Phi [W].
            // An isotropic point light has radiant intensity I = Phi / (4 pi).
            const float3 radiantIntensity =
                    light.flux * light.color * (1.0f / (4.0f * M_PIf));

            accumulatedRadiance +=
                    diffuseBrdf *
                    radiantIntensity *
                    shadowTransmission *
                    (surfaceCosine / distanceSquared);
        }

        return accumulatedRadiance;
    }

    SYCL_EXTERNAL inline float3 computeIncidentRadianceFromPointLights(
        const GPUSceneBuffers &scene,
        const PathTracerSettings &settings,
        const float3 &surfacePositionW,
        const float3 &surfaceNormalW,
        const float eps = RayEpsilon) {
        float3 accumulatedRadiance(0.0f);

        for (uint32_t lightIndex = 0u;
             lightIndex < scene.lightCount;
             ++lightIndex) {
            const GPULightRecord &light = scene.lights[lightIndex];

            // Your current light records appear to use emissive surfels as
            // light-position carriers. We interpret each as an isotropic point light.
            if (light.lightType != LightType::Surfel) {
                continue;
            }

            const Point &lightSurfel = scene.points[light.primitiveIndex];
            const float3 lightPositionW = lightSurfel.position;

            const float3 toLight = lightPositionW - surfacePositionW;
            const float distanceSquared = dot(toLight, toLight);

            if (distanceSquared <= 1.0e-12f) {
                continue;
            }

            const float distance = sycl::sqrt(distanceSquared);
            const float3 lightDirection = toLight / distance;

            const float surfaceCosine =
                    sycl::fmax(0.0f, dot(surfaceNormalW, lightDirection));

            if (surfaceCosine <= 0.0f) {
                continue;
            }

            const float shadowTransmission =
                    traceShadowTransmissionToPoint(
                        scene,
                        settings,
                        surfacePositionW,
                        surfaceNormalW,
                        lightPositionW, eps);


            if (shadowTransmission <= 0.0f) {
                continue;
            }

            // Treat light.flux as total radiant flux Phi [W].
            // An isotropic point light has radiant intensity I = Phi / (4 pi).
            const float3 radiantIntensity =
                    light.flux * light.color * (1.0f / (4.0f * M_PIf));

            accumulatedRadiance +=
                    radiantIntensity *
                    shadowTransmission *
                    (surfaceCosine / distanceSquared);
        }

        return accumulatedRadiance;
    }

    /*
    SYCL_EXTERNAL inline float3 estimateDirectAreaLightAtDiffuseSurface(
        const GPUSceneBuffers& scene,
        const float3& shadingPositionW,
        const float3& shadingNormalW,
        const float3& diffuseAlbedo,
        const PathTracerSettings& settings,
        rng::Xorshift128& rng128) {
        const uint32_t samplesPerLight = settings.numShadowRays;
        if (samplesPerLight == 0u) {
            return float3(0.0f);
        }
        float3 accumulatedDirectRadiance(0.0f);
        const float invSamplesPerLight = 1.0f / static_cast<float>(samplesPerLight);

        for (uint32_t lightIndex = 0u;
             lightIndex < scene.lightCount;
             ++lightIndex) {
            const GPULightRecord light = scene.lights[lightIndex];
            // Keep only finite area/surfel emitters.
            if (light.lightType != LightType::Surfel) {
                continue;
            }
            for (uint32_t shadowSampleIndex = 0u;
                 shadowSampleIndex < samplesPerLight;
                 ++shadowSampleIndex) {
                // Important:
                // This function should sample a point on the already-selected light.
                // It must NOT randomly choose another light internally.
                const AreaLightSample lightSample =
                    sampleMeshAreaLightByIndex(scene, lightIndex, rng128);
                if (!lightSample.valid) {
                    continue;
                }
                // Deterministic loop over lights:
                // no p_L(k), only area pdf p_A(y | k).
                const float pdfArea = lightSample.pdfArea;
                if (pdfArea <= 0.0f) {
                    continue;
                }

                const float3 lightVector =
                    lightSample.positionW - shadingPositionW;

                const float lightDistanceSquared =
                    dot(lightVector, lightVector);

                if (lightDistanceSquared <= 1.0e-12f) {
                    continue;
                }
                const float lightDistance = sycl::sqrt(lightDistanceSquared);
                const float3 lightDirection = lightVector / lightDistance;

                // Incoming direction at the shading point: x -> y.
                const float shadingCosine = sycl::fmax(0.0f, dot(shadingNormalW, lightDirection));
                if (shadingCosine <= 0.0f) {
                    continue;
                }
                // One-sided emitter cosine at y.
                const float lightCosine = sycl::fmax(0.0f, dot(lightSample.normalW, -lightDirection));
                if (lightCosine <= 0.0f) {
                    continue;
                }
                // Accumulated transmittance to sampled emitter point.
                const float shadowTransmission =
                    traceShadowTransmissionToPoint(
                        scene,
                        settings,
                        shadingPositionW,
                        shadingNormalW,
                        lightSample.positionW);

                if (shadowTransmission <= 0.0f) {
                    continue;
                }
                const float geometricTerm = (shadingCosine * lightCosine) / (lightDistanceSquared + 1.0e-8f);
                const float3 diffuseBrdf = diffuseAlbedo * M_1_PIf;
                // For one-sided Lambertian area emitter:
                // L_e = Phi / (pi A).
                const float3 emittedRadiance = lightSample.flux / (M_PIf * lightSample.totalAreaWorld);
                const float3 sampleContribution = diffuseBrdf * emittedRadiance * shadowTransmission * geometricTerm * (
                    1.0f / pdfArea) * invSamplesPerLight;
                accumulatedDirectRadiance += sampleContribution;
            }
        }

        return accumulatedDirectRadiance;
    }


    struct PointSampledSceneHit {
        bool hit = false;
        bool isEmissive = false;
        float tWorld = std::numeric_limits<float>::infinity();
        float3 hitPositionW{0.0f};
        float3 geometricNormalW{0.0f};
        float3 albedo{0.0f};
        float3 emittedRadiance{0.0f};
        float opacity = 1.0f;
        uint32_t instanceIndex = UINT32_MAX;
        uint32_t primitiveIndex = UINT32_MAX;
        GeometryType geometryType{};
        uint32_t contributorCount = 0u;
    };

    SYCL_EXTERNAL inline float3 pointSampledNormalObject(const Point &surfel) {
        const float3 tangentU = normalize(surfel.tanU);
        const float3 tangentV = normalize(surfel.tanV - tangentU * dot(tangentU, surfel.tanV));
        return normalize(cross(tangentU, tangentV));
    }

    SYCL_EXTERNAL inline float3 pointSampledNormalWorld(const Point &surfel, const Transform &transform) {
        const float3 tangentUWorld = transformDirection(transform.objectToWorld, surfel.tanU);
        const float3 tangentVWorld = transformDirection(transform.objectToWorld, surfel.tanV);
        return normalize(cross(tangentUWorld, tangentVWorld));
    }

    SYCL_EXTERNAL inline bool intersectPointSampledDisk(
        const Ray &rayObject, const Point &surfel, float supportRadius,
        float tMin, float tMax, float &outT, float &outDistanceToRayPlaneHit) {
        const float3 normalObject = pointSampledNormalObject(surfel);
        const float normalDirectionDot = dot(normalObject, rayObject.direction);

        if (sycl::fabs(normalDirectionDot) <= RayEpsilon2) {
            return false;
        }
        const float tPlane = dot(normalObject, surfel.position - rayObject.origin) / normalDirectionDot;
        if (tPlane <= tMin || tPlane >= tMax) {
            return false;
        }
        const float3 rayPlaneIntersection = rayObject.origin + tPlane * rayObject.direction;
        const float3 tangentOffset = rayPlaneIntersection - surfel.position;
        const float distanceSquared = dot(tangentOffset, tangentOffset);
        if (distanceSquared >= supportRadius * supportRadius) {
            return false;
        }
        outT = tPlane;
        outDistanceToRayPlaneHit = sycl::sqrt(distanceSquared);
        return true;
    }


    SYCL_EXTERNAL inline bool intersectBLASPointSampledGeometry(
        const Ray &rayWorld, const Ray &rayObject, uint32_t blasRangeIndex,
        const Transform &transform, const GPUSceneBuffers &scene,
        const PathTracerSettings &settings, PointSampledSceneHit &outHit) {
        const float supportRadius = settings.pointGeometrySupportRadius;
        if (supportRadius <= 0.0f) {
            return false;
        }
        const BLASRange &blasRange = scene.blasRanges[blasRangeIndex];
        const BVHNode *bvhNodes = scene.blasNodes + blasRange.firstNode;
        const float3 inverseDirection = safeInvDir(rayObject.direction);
        bool foundSeed = false;
        float seedTObject = std::numeric_limits<float>::infinity();
        float3 seedHitObject{0.0f};
        uint32_t seedPrimitiveIndex = UINT32_MAX;
        SmallStack<256> seedTraversalStack;
        seedTraversalStack.push(0);
        while (!seedTraversalStack.empty()) {
            const int nodeIndex = seedTraversalStack.pop();
            const BVHNode &node = bvhNodes[nodeIndex];
            float nodeTEntry = 0.0f;
            if (!slabIntersectAABB(
                rayObject, node, inverseDirection,
                seedTObject, nodeTEntry)) {
                continue;
            }
            if (node.triCount == 0u) {
                const int leftIndex = node.leftFirst;
                const int rightIndex = node.leftFirst + 1;
                float leftTEntry = 0.0f;
                float rightTEntry = 0.0f;
                const bool hitLeft = slabIntersectAABB(
                    rayObject, bvhNodes[leftIndex], inverseDirection,
                    seedTObject, leftTEntry);
                const bool hitRight = slabIntersectAABB(
                    rayObject, bvhNodes[rightIndex], inverseDirection,
                    seedTObject, rightTEntry);

                if (hitLeft && hitRight) {
                    pushNearFar(seedTraversalStack, leftIndex, leftTEntry, rightIndex, rightTEntry);
                } else if (hitLeft) {
                    seedTraversalStack.push(leftIndex);
                } else if (hitRight) {
                    seedTraversalStack.push(rightIndex);
                }
                continue;
            }

            float alphaGeom = 0.0f;
            for (uint32_t primitiveOffset = 0u; primitiveOffset < node.triCount; ++primitiveOffset) {
                const uint32_t primitiveIndex = scene.pointPermutation[node.leftFirst + primitiveOffset];
                const Point &surfel = scene.points[primitiveIndex];
                float tDiskObject = 0.0f;
                float3 hitLocal;

                if (!intersectSurfel(rayObject, surfel, RayEpsilon2, seedTObject,
                                     tDiskObject, RayEpsilon2)) {
                    continue;
                }

                hitLocal = rayObject.origin + tDiskObject * rayObject.direction;
                const float2 uv = phiInverse(hitLocal, surfel);
                if (!opacityBeta(uv[0], uv[1], surfel, &alphaGeom) || alphaGeom <= 0.0f) {
                    continue;
                }
                const float seedEffectiveOpacity = sycl::clamp(surfel.opacity * alphaGeom, 0.0f, 1.0f);
                if (seedEffectiveOpacity <= 0.0f) {
                    continue;
                }

                foundSeed = true;
                seedTObject = tDiskObject;
                seedHitObject = hitLocal;
                seedPrimitiveIndex = primitiveIndex;
            }
        }
        if (!foundSeed) {
            return false;
        }
        const float reconstructionLength = sycl::fmax(settings.pointGeometryReconstructionLength, 2.0f * supportRadius);
        const float pointGeometryMinimumT = 2.0f * settings.pointGeometrySupportRadius;
        const float cylinderTMin = sycl::fmax(pointGeometryMinimumT, seedTObject - RayEpsilon2);
        const float cylinderTMax = seedTObject + reconstructionLength;
        const float3 referenceNormalWorld = pointSampledNormalWorld(scene.points[seedPrimitiveIndex], transform);
        float totalWeight = 0.0f;
        float closestTWorld = std::numeric_limits<float>::infinity();
        float3 weightedNormalWorld{0.0f};
        float3 weightedAlbedo{0.0f};
        float3 weightedEmission{0.0f};
        float weightedOpacity = 0.0f;
        float emissiveWeight = 0.0f;
        uint32_t representativePrimitiveIndex = seedPrimitiveIndex;
        uint32_t contributorCount = 0u;
        SmallStack<256> reconstructionTraversalStack;
        reconstructionTraversalStack.push(0);
        while (!reconstructionTraversalStack.empty()) {
            const int nodeIndex = reconstructionTraversalStack.pop();
            const BVHNode &node = bvhNodes[nodeIndex];
            float nodeTEntry = 0.0f;
            if (!slabIntersectAABB(rayObject, node, inverseDirection, cylinderTMax, nodeTEntry)) {
                continue;
            }
            if (node.triCount == 0u) {
                const int leftIndex = node.leftFirst;
                const int rightIndex = node.leftFirst + 1;
                float leftTEntry = 0.0f;
                float rightTEntry = 0.0f;
                const bool hitLeft = slabIntersectAABB(rayObject, bvhNodes[leftIndex], inverseDirection, cylinderTMax,
                                                       leftTEntry);
                const bool hitRight = slabIntersectAABB(rayObject, bvhNodes[rightIndex], inverseDirection, cylinderTMax,
                                                        rightTEntry);
                if (hitLeft && hitRight) {
                    pushNearFar(reconstructionTraversalStack, leftIndex, leftTEntry, rightIndex, rightTEntry);
                } else if (hitLeft) {
                    reconstructionTraversalStack.push(leftIndex);
                } else if (hitRight) {
                    reconstructionTraversalStack.push(rightIndex);
                }
                continue;
            }
            for (uint32_t primitiveOffset = 0u; primitiveOffset < node.triCount; ++primitiveOffset) {
                const uint32_t primitiveIndex = scene.pointPermutation[node.leftFirst + primitiveOffset];
                const Point &surfel = scene.points[primitiveIndex];
                float tPlaneObject = 0.0f;
                float3 hitLocal;
                float alphaGeom;

                if (!intersectSurfel(rayObject, surfel, cylinderTMin, cylinderTMax,
                                     tPlaneObject, RayEpsilon2)) {
                    continue;
                }
                hitLocal = rayObject.origin + tPlaneObject * rayObject.direction;
                const float2 uv = phiInverse(hitLocal, surfel);
                if (!opacityBeta(uv[0], uv[1], surfel, &alphaGeom)) {
                    continue;
                }
                const float effectiveOpacity = sycl::clamp(surfel.opacity * alphaGeom, 0.0f, 1.0f);
                if (effectiveOpacity <= 0.0f) {
                    continue;
                }

                constexpr float weight = 1.0f;
                const float3 localRayPoint = rayObject.origin + tPlaneObject * rayObject.direction;
                const float3 worldRayPoint = toWorldPoint(localRayPoint, transform);
                const float tWorld = dot(worldRayPoint - rayWorld.origin, rayWorld.direction);
                float3 normalWorld = pointSampledNormalWorld(surfel, transform);
                if (dot(normalWorld, referenceNormalWorld) < 0.0f) {
                    normalWorld = -normalWorld;
                }
                totalWeight += weight;
                weightedNormalWorld += weight * normalWorld;
                weightedAlbedo += weight * (surfel.alpha_r * surfel.albedo);
                weightedOpacity += effectiveOpacity;
                if (surfel.isEmissive()) {
                    const float surfelArea = M_PIf * surfel.scale.x() * surfel.scale.y();
                    if (surfelArea > 1.0e-10f) {
                        const float3 emittedRadiance = surfel.albedo * (surfel.flux / (M_PIf * surfelArea));
                        weightedEmission += weight * emittedRadiance;
                        emissiveWeight += weight;
                    }
                }
                if (tWorld < closestTWorld) {
                    closestTWorld = tWorld;
                    representativePrimitiveIndex = primitiveIndex;
                }
                ++contributorCount;
            }
        }
        if (contributorCount < settings.pointGeometryMinimumContributors ||
            totalWeight <= 1.0e-8f) {
            return false;
        }
        const float weightedNormalLengthSquared = dot(weightedNormalWorld, weightedNormalWorld);
        outHit.hit = true;
        outHit.tWorld = closestTWorld;
        outHit.hitPositionW = rayWorld.origin + rayWorld.direction * outHit.tWorld;
        outHit.geometricNormalW = weightedNormalLengthSquared > 1.0e-12f
                                      ? normalize(weightedNormalWorld)
                                      : referenceNormalWorld;
        outHit.albedo = weightedAlbedo / totalWeight;
        outHit.emittedRadiance = weightedEmission / totalWeight;
        outHit.opacity = sycl::clamp(weightedOpacity, 0.0f, 1.0f);
        outHit.isEmissive = emissiveWeight > 0.0f;
        outHit.primitiveIndex = representativePrimitiveIndex;
        outHit.contributorCount = contributorCount;
        return true;
    }


    SYCL_EXTERNAL inline bool intersectScenePointSampledGeometry(
        const Ray &rayWorld, const GPUSceneBuffers &scene,
        const PathTracerSettings &settings, PointSampledSceneHit &outHit) {
        outHit = PointSampledSceneHit{};

        const TLASNode *tlasNodes = scene.tlasNodes;
        const float3 inverseDirectionWorld = safeInvDir(rayWorld.direction);
        float bestTWorld = std::numeric_limits<float>::infinity();

        SmallStack<256> traversalStack;
        traversalStack.push(0);

        while (!traversalStack.empty()) {
            const uint32_t nodeIndex = traversalStack.pop();
            const TLASNode &node = tlasNodes[nodeIndex];

            float nodeTEntry = 0.0f;
            if (!slabIntersectAABB(rayWorld, node, inverseDirectionWorld, bestTWorld, nodeTEntry)) {
                continue;
            }

            if (node.count == 0u) {
                const uint32_t leftIndex = node.leftChild;
                const uint32_t rightIndex = node.rightChild;

                float leftTEntry = 0.0f;
                float rightTEntry = 0.0f;

                const bool hitLeft = slabIntersectAABB(
                    rayWorld, tlasNodes[leftIndex], inverseDirectionWorld, bestTWorld, leftTEntry);

                const bool hitRight = slabIntersectAABB(
                    rayWorld, tlasNodes[rightIndex], inverseDirectionWorld, bestTWorld, rightTEntry);

                if (hitLeft && hitRight) {
                    pushNearFar(
                        traversalStack, static_cast<int>(leftIndex), leftTEntry,
                        static_cast<int>(rightIndex), rightTEntry);
                } else if (hitLeft) {
                    traversalStack.push(static_cast<int>(leftIndex));
                } else if (hitRight) {
                    traversalStack.push(static_cast<int>(rightIndex));
                }

                continue;
            }

            const uint32_t instanceIndex = node.leftChild;
            const InstanceRecord &instance = scene.instances[instanceIndex];
            const Transform &transform = scene.transforms[instance.transformIndex];

            if (instance.geometryType == GeometryType::PointCloud) {
                const Ray rayObject = toObjectSpace(rayWorld, transform);
                PointSampledSceneHit pointCloudHit{};

                if (!intersectBLASPointSampledGeometry(
                    rayWorld, rayObject, instance.blasRangeIndex,
                    transform, scene, settings, pointCloudHit)) {
                    continue;
                }

                if (pointCloudHit.tWorld <= RayEpsilon2 || pointCloudHit.tWorld >= bestTWorld) {
                    continue;
                }

                pointCloudHit.instanceIndex = instanceIndex;
                pointCloudHit.geometryType = GeometryType::PointCloud;

                bestTWorld = pointCloudHit.tWorld;
                outHit = pointCloudHit;
                continue;
            }

            if (instance.geometryType == GeometryType::Mesh) {
                const Ray rayObject = toObjectSpace(rayWorld, transform);
                LocalHit localHit{};

                if (!intersectBLASMesh(rayObject, instance.blasRangeIndex, localHit, scene, transform)) {
                    continue;
                }

                const float3 hitPositionW = localHit.worldHit;
                const float tWorld = dot(hitPositionW - rayWorld.origin, rayWorld.direction);

                if (tWorld <= RayEpsilon2 || tWorld >= bestTWorld) {
                    continue;
                }

                WorldHit meshHit{};
                meshHit.hit = true;
                meshHit.hitPositionW = hitPositionW;
                meshHit.t = tWorld;
                meshHit.instanceIndex = instanceIndex;
                meshHit.primitiveIndex = localHit.primitiveIndex;

                buildIntersectionNormal(scene, meshHit);

                const GPUMaterial &material = scene.materials[instance.materialIndex];

                outHit.hit = true;
                outHit.tWorld = tWorld;
                outHit.hitPositionW = hitPositionW;
                outHit.geometricNormalW = meshHit.geometricNormalW;
                outHit.albedo = material.baseColor;
                outHit.isEmissive = material.isEmissive();
                outHit.emittedRadiance = material.power * material.baseColor;
                outHit.opacity = 1.0f;
                outHit.instanceIndex = instanceIndex;
                outHit.primitiveIndex = localHit.primitiveIndex;
                outHit.geometryType = GeometryType::Mesh;

                bestTWorld = tWorld;
            }
        }

        return outHit.hit;
    }


    SYCL_EXTERNAL inline float tracePointSampledShadowTransmissionToPoint(
        const GPUSceneBuffers &scene,
        const PathTracerSettings &settings,
        const float3 &surfacePositionW,
        const float3 &surfaceNormalW,
        const float3 &lightPositionW) {
        const float rayOffset = sycl::fmax(
            RayEpsilon,
            settings.pointGeometryRayOffsetMultiplier * settings.pointGeometrySupportRadius);
        Ray shadowRay{};
        shadowRay.origin = surfacePositionW + surfaceNormalW * rayOffset;
        shadowRay.normal = surfaceNormalW;

        float transmission = 1.0f;
        for (uint32_t traversalIndex = 0u; traversalIndex < kMaxSplatEventsPerRay; ++traversalIndex) {
            const float3 lightVector = lightPositionW - shadowRay.origin;
            const float lightDistanceSquared = dot(lightVector, lightVector);
            if (lightDistanceSquared <= 1.0e-10f) {
                return transmission;
            }

            const float lightDistance = sycl::sqrt(lightDistanceSquared);
            shadowRay.direction = lightVector / lightDistance;

            PointSampledSceneHit shadowHit{};
            if (!intersectScenePointSampledGeometry(shadowRay, scene, settings, shadowHit)) {
                return transmission;
            }

            if (shadowHit.tWorld >= lightDistance - rayOffset) {
                return transmission;
            }

            if (shadowHit.geometryType == GeometryType::Mesh) {
                return 0.0f;
            }

            const float alphaEff = sycl::clamp(shadowHit.opacity, 0.0f, 1.0f);
            transmission *= 1.0f - alphaEff;
            if (transmission <= 1.0e-6f) {
                return 0.0f;
            }

            shadowRay.origin = shadowRay.origin + shadowRay.direction * (shadowHit.tWorld + rayOffset);
        }

        return transmission;
    }


    SYCL_EXTERNAL inline float3 estimateDirectPointSampledAreaLight(
        const GPUSceneBuffers &scene,
        const PathTracerSettings &settings,
        const float3 &surfacePositionW,
        const float3 &surfaceNormalW,
        const float3 &diffuseAlbedo,
        rng::Xorshift128 &rng128) {
        if (settings.numShadowRays == 0u) {
            return float3{0.0f};
        }

        float3 radiance{0.0f};
        const float inverseSampleCount = 1.0f / static_cast<float>(settings.numShadowRays);
        for (uint32_t lightIndex = 0u; lightIndex < scene.lightCount; ++lightIndex) {
            const GPULightRecord &light = scene.lights[lightIndex];

            if (light.lightType != LightType::Surfel) {
                continue;
            }
            for (uint32_t shadowSampleIndex = 0u; shadowSampleIndex < settings.numShadowRays; ++shadowSampleIndex) {
                const AreaLightSample lightSample = sampleMeshAreaLightByIndex(scene, lightIndex, rng128);
                if (!lightSample.valid ||
                    lightSample.pdfArea <= 0.0f ||
                    lightSample.totalAreaWorld <= 1.0e-10f) {
                    continue;
                }
                const float3 toLight = lightSample.positionW - surfacePositionW;
                const float distanceSquared = dot(toLight, toLight);
                if (distanceSquared <= 1.0e-10f) {
                    continue;
                }
                const float distance = sycl::sqrt(distanceSquared);
                const float3 lightDirection = toLight / distance;
                const float surfaceCosine = sycl::fmax(0.0f, dot(surfaceNormalW, lightDirection));
                const float lightCosine = sycl::fmax(0.0f, dot(lightSample.normalW, -lightDirection));
                if (surfaceCosine <= 0.0f ||
                    lightCosine <= 0.0f) {
                    continue;
                }
                const float shadowTransmission = tracePointSampledShadowTransmissionToPoint(
                    scene, settings, surfacePositionW, surfaceNormalW, lightSample.positionW);
                if (shadowTransmission <= 0.0f) {
                    continue;
                }
                const float3 brdf = diffuseAlbedo * M_1_PIf;
                const float3 emittedRadiance = lightSample.flux / (M_PIf * lightSample.totalAreaWorld);
                radiance += brdf * emittedRadiance * shadowTransmission *
                    (surfaceCosine * lightCosine / (distanceSquared + 1.0e-8f)) *
                    (1.0f / lightSample.pdfArea) * inverseSampleCount;
            }
        }

        return radiance;
    }
    */
} // namespace Pale
