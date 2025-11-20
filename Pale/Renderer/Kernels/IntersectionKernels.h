// SplatIntersection.hpp
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

    struct ChildEntry {
        int nodeIndex;
        float tEntry;
    };

    SYCL_EXTERNAL inline bool computeAabbEntry(const Ray &ray,
                                               const BVHNode &node,
                                               const float3 &inverseDirection,
                                               float currentBestTHit,
                                               float &outTEntry) {
        return slabIntersectAABB(ray, node, inverseDirection, currentBestTHit, outTEntry);
    }

    SYCL_EXTERNAL inline bool computeAabbEntry(const Ray &ray,
                                               const TLASNode &node,
                                               const float3 &inverseDirection,
                                               float currentBestTHit,
                                               float &outTEntry) {
        return slabIntersectAABB(ray, node, inverseDirection, currentBestTHit, outTEntry);
    }

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

    // -----------------------------------------------------------------------------
    // Triangle BLAS (unchanged except near-to-far child push)
    // -----------------------------------------------------------------------------
    SYCL_EXTERNAL static bool intersectBLASMesh(const Ray &rayObject,
                                                uint32_t geometryIndex,
                                                LocalHit &localHitOut,
                                                const GPUSceneBuffers &scene) {
        const BLASRange &blasRange = scene.blasRanges[geometryIndex];
        const BVHNode *bvhNodes = scene.blasNodes + blasRange.firstNode;
        const Triangle *triangles = scene.triangles;
        const Vertex *vertices = scene.vertices;

        float bestTHit = std::numeric_limits<float>::infinity();
        bool hitAnyTriangle = false;
        const float3 inverseDirection = safeInvDir(rayObject.direction);

        SmallStack<512> traversalStack;
        traversalStack.push(0); // root

        while (!traversalStack.empty()) {
            const int nodeIndex = traversalStack.pop();
            const BVHNode &node = bvhNodes[nodeIndex];

            float nodeTEntry = 0.0f;
            if (!slabIntersectAABB(rayObject, node, inverseDirection, bestTHit, nodeTEntry))
                continue;

            if (node.triCount == 0) {
                // Internal: left child is node.leftFirst, right child is node.leftFirst + 1
                const int leftIndex = node.leftFirst;
                const int rightIndex = node.leftFirst + 1;

                float leftTEntry = std::numeric_limits<float>::infinity();
                float rightTEntry = std::numeric_limits<float>::infinity();

                const bool hitLeft = computeAabbEntry(rayObject, bvhNodes[leftIndex], inverseDirection, bestTHit,
                                                      leftTEntry);
                const bool hitRight = computeAabbEntry(rayObject, bvhNodes[rightIndex], inverseDirection, bestTHit,
                                                       rightTEntry);

                if (hitLeft && hitRight) pushNearFar(traversalStack, leftIndex, leftTEntry, rightIndex, rightTEntry);
                else if (hitLeft) traversalStack.push(leftIndex);
                else if (hitRight) traversalStack.push(rightIndex);
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
                if (intersectTriangle(rayObject, A, B, C, t, u, v, 1e-4f) && t < bestTHit) {
                    bestTHit = t;
                    hitAnyTriangle = true;
                    localHitOut.t = t;
                    localHitOut.primitiveIndex = triangleIndex;
                    localHitOut.transmissivity = 0.0f; // opaque triangle
                }
            }
        }

        return hitAnyTriangle;
    }

    // -----------------------------------------------------------------------------
    // Point-cloud BLAS with ordered Bernoulli thinning and depth grouping
    // -----------------------------------------------------------------------------
    SYCL_EXTERNAL static bool intersectBLASPointCloud(const Ray &rayObject,
                                                      uint32_t blasRangeIndex,
                                                      LocalHit &localHitOut,
                                                      const GPUSceneBuffers &scene,
                                                      rng::Xorshift128 &rng128,
                                                      const Transform &transform,
                                                      const Ray &rayWorld,
                                                      RayIntersectMode rayIntersectMode,
                                                      uint32_t scatterOnPrimitiveIndex) {
        const BLASRange &blasRange = scene.blasRanges[blasRangeIndex];
        const BVHNode *bvhNodes = scene.blasNodes + blasRange.firstNode;

        float bestAcceptedTHit = std::numeric_limits<float>::infinity();
        bool foundAcceptedScatter = false;
        const float3 inverseDirection = safeInvDir(rayObject.direction);
        constexpr float rayEpsilon = 1e-4f;
        constexpr float sameDepthEpsilon = 1e-3f;

        float cumulativeTransmittanceBefore = 1.0f;

        SmallStack<64> traversalStack;
        traversalStack.push(0);

        float currentGroupDepthKey = -std::numeric_limits<float>::infinity();
        BoundedVector<float, kMaxSplatEvents> groupDepthKeys;
        BoundedVector<float, kMaxSplatEvents> groupLocalTs;
        BoundedVector<float, kMaxSplatEvents> groupAlphas;
        BoundedVector<uint32_t, kMaxSplatEvents> groupIndices;

        auto clearCurrentGroup = [&]() {
            groupDepthKeys.clear();
            groupLocalTs.clear();
            groupAlphas.clear();
            groupIndices.clear();
        };

        auto scatterCurrentGroup = [&](rng::Xorshift128 &randomNumberGenerator) -> bool {
            if (groupLocalTs.empty()) return false;

            // 1) Push this slice’s events into LocalHit (depth-sorted already)
            for (size_t groupIndex = 0; groupIndex < groupLocalTs.size(); ++groupIndex) {
                if (localHitOut.splatEventCount >= kMaxSplatEvents)
                    continue;

                const int eventIndex = localHitOut.splatEventCount++;
                localHitOut.splatEvents[eventIndex].t = groupLocalTs[groupIndex];
                localHitOut.splatEvents[eventIndex].alpha = groupAlphas[groupIndex];
                localHitOut.splatEvents[eventIndex].primitiveIndex = groupIndices[groupIndex];
            }

            // 2) Composite alpha at this depth slice
            float productOneMinusAlpha = 1.0f;
            for (size_t groupIndex = 0; groupIndex < groupAlphas.size(); ++groupIndex) {
                productOneMinusAlpha *= (1.0f - groupAlphas[groupIndex]);
            }

            float compositeAlpha = 1.0f - productOneMinusAlpha;
            compositeAlpha = sycl::clamp(compositeAlpha, 0.0f, 1.0f);

            if (compositeAlpha <= 0.0f) {
                // No opacity contribution at this slice
                clearCurrentGroup();
                return false;
            }

            // 3) Forced-scatter debug: check if this slice contains the target surfel
            const bool forceScatterOnSpecificSurfel =
                (rayIntersectMode == RayIntersectMode::Scatter) &&
                (scatterOnPrimitiveIndex != UINT32_MAX);

            int forcedGroupIndex = -1;
            if (forceScatterOnSpecificSurfel) {
                for (size_t groupIndex = 0; groupIndex < groupIndices.size(); ++groupIndex) {
                    if (groupIndices[groupIndex] == scatterOnPrimitiveIndex) {
                        forcedGroupIndex = static_cast<int>(groupIndex);
                        break;
                    }
                }
            }

            // 4) Handle mode-specific logic
            bool mustScatter = false;
            bool mustTransmit = false;

            switch (rayIntersectMode) {
                case RayIntersectMode::Transmit: {
                    mustTransmit = true;
                    break;
                }
                case RayIntersectMode::Random: {
                    // Original behavior: stochastic choice between transmit and scatter
                    const float uniformSample = randomNumberGenerator.nextFloat();
                    if (uniformSample < compositeAlpha) {
                        mustScatter = true;
                    } else {
                        mustTransmit = true;
                    }
                    break;
                }
                case RayIntersectMode::Scatter: {
                    if (forceScatterOnSpecificSurfel) {
                        if (forcedGroupIndex >= 0) {
                            // This slice contains the requested surfel: we must scatter on it
                            mustScatter = true;
                            mustTransmit = false;
                        } else {
                            // This slice does *not* contain the requested surfel: pure transmit
                            mustScatter = false;
                            mustTransmit = true;
                        }
                    } else {
                        // Debug / forced-scatter mode without a specific primitive:
                        // if there is any opacity in this slice, we must scatter on one event.
                        mustScatter = true;
                    }
                    break;
                }
            }

            if (mustTransmit) {
                // Pure transmission through this slice
                cumulativeTransmittanceBefore *= (1.0f - compositeAlpha);
                clearCurrentGroup();
                return false;
            }

            if (!mustScatter) {
                // Defensive, but logically we should always be either scatter or transmit
                clearCurrentGroup();
                return false;
            }

            // 5) We are in a "scatter" mode
            if (forceScatterOnSpecificSurfel && forcedGroupIndex >= 0) {
                // Deterministic scatter on the requested surfel
                localHitOut.t = groupLocalTs[forcedGroupIndex];
                localHitOut.primitiveIndex = groupIndices[forcedGroupIndex];
                localHitOut.transmissivity = cumulativeTransmittanceBefore;
                // Do not count this as an extra splat event; it is the main hit.

                clearCurrentGroup();
                return true;
            }

            // 6) Default scatter behavior: pick one event in this group via sequential thinning
            float survivalInsideGroup = 1.0f;
            const float safeCompositeAlpha = sycl::fmax(compositeAlpha, 1e-8f);

            for (size_t groupIndex = 0; groupIndex < groupAlphas.size(); ++groupIndex) {
                const float eventAlpha = groupAlphas[groupIndex];
                const float probabilityFirstHere =
                        eventAlpha * survivalInsideGroup / safeCompositeAlpha;

                if (randomNumberGenerator.nextFloat() < probabilityFirstHere) {
                    localHitOut.t = groupLocalTs[groupIndex];
                    localHitOut.primitiveIndex = groupIndices[groupIndex];
                    localHitOut.transmissivity = cumulativeTransmittanceBefore;
                    // Do not count this as a splat event, only as the main hit.

                    clearCurrentGroup();
                    return true; // accepted scatter at this depth
                }

                survivalInsideGroup *= (1.0f - eventAlpha);
            }

            // 7) Fallback: numerics might occasionally skip everything; enforce one event
            localHitOut.t = groupLocalTs.back();
            localHitOut.primitiveIndex = groupIndices.back();
            localHitOut.transmissivity = cumulativeTransmittanceBefore;

            clearCurrentGroup();
            return true;
        };



        auto transmitCurrentGroup = [&](rng::Xorshift128 &rng)-> bool {
            if (groupLocalTs.empty()) return false;

            // 1) push this slice’s events into LocalHit (depth-sorted already)
            for (size_t i = 0; i < groupLocalTs.size(); ++i) {
                if (localHitOut.splatEventCount >= kMaxSplatEvents - 1)
                    continue;
                int index = localHitOut.splatEventCount++;

                localHitOut.splatEvents[index].t = groupLocalTs[i];
                localHitOut.splatEvents[index].alpha = groupAlphas[i];
                localHitOut.splatEvents[index].primitiveIndex = groupIndices[i];
            }

            // 2) composite transmit through the slice
            float oneMinus = 1.0f;
            for (size_t i = 0; i < groupAlphas.size(); ++i) oneMinus *= (1.0f - groupAlphas[i]);
            float compositeAlpha = 1.0f - oneMinus;

            cumulativeTransmittanceBefore *= (1.0f - compositeAlpha);
            clearCurrentGroup();

            // 3) never accept; keep traversing
            return false;
        };

        while (!traversalStack.empty()) {
            const int nodeIndex = traversalStack.pop();
            const BVHNode &node = bvhNodes[nodeIndex];

            float nodeTEntry = 0.0f;
            if (!slabIntersectAABB(rayObject, node, inverseDirection, bestAcceptedTHit, nodeTEntry))
                continue;

            if (node.triCount == 0) {
                const int leftIndex = node.leftFirst;
                const int rightIndex = node.leftFirst + 1;

                float leftTEntry = std::numeric_limits<float>::infinity();
                float rightTEntry = std::numeric_limits<float>::infinity();

                const bool hitLeft = computeAabbEntry(rayObject, bvhNodes[leftIndex], inverseDirection,
                                                      bestAcceptedTHit, leftTEntry);
                const bool hitRight = computeAabbEntry(rayObject, bvhNodes[rightIndex], inverseDirection,
                                                       bestAcceptedTHit, rightTEntry);

                if (hitLeft && hitRight) {
                    pushNearFar(traversalStack, leftIndex, leftTEntry, rightIndex, rightTEntry);
                } else if (hitLeft) {
                    traversalStack.push(leftIndex);
                } else if (hitRight) {
                    traversalStack.push(rightIndex);
                }
                continue;
            }

            // Leaf: collect surfel events, sort by tHit
            BoundedVector<float, kMaxSplatEvents> leafLocalTHits;
            // Should match the maxleafpoints in the BVH construction
            BoundedVector<float, kMaxSplatEvents> leafAlphas;
            BoundedVector<float, kMaxSplatEvents> leafDepthKeys;
            BoundedVector<uint32_t, kMaxSplatEvents> leafIndices;
            leafLocalTHits.clear();
            leafAlphas.clear();
            leafIndices.clear();
            leafDepthKeys.clear();

            for (uint32_t local = 0; local < node.triCount; ++local) {
                if (leafLocalTHits.size() >= kMaxSplatEvents)
                    continue;


                const uint32_t surfelIndex = node.leftFirst + local;
                const Point &surfel = scene.points[surfelIndex];

                float tHitLocal = 0.0f;
                float opacity = 0.0f;
                float3 outHitLocal;
                if (!intersectSurfel(rayObject, surfel, rayEpsilon, bestAcceptedTHit, tHitLocal, outHitLocal, opacity))
                    continue;

                float3 worldPoint = toWorldPoint(outHitLocal, transform);
                const float depthKey = dot(worldPoint - rayWorld.origin, rayWorld.direction);

                const float alphaAtHit = sycl::clamp(opacity * surfel.opacity, 0.0f, 1.0f);

                leafLocalTHits.pushBack(tHitLocal);
                leafAlphas.pushBack(alphaAtHit);
                leafIndices.pushBack(surfelIndex);
                leafDepthKeys.pushBack(depthKey);
            }

            // Sort leaf events by t using insertionSortByKey (no STL)
            const int leafCount = leafLocalTHits.size();
            BoundedVector<int, kMaxSplatEvents> order;
            BoundedVector<float, kMaxSplatEvents> tempKeys;
            order.clear();
            tempKeys.clear();
            for (int i = 0; i < leafCount; ++i) {
                order.pushBack(i);
                tempKeys.pushBack(leafDepthKeys[i]);
            }
            insertionSortByKeyTie(tempKeys.data(), leafAlphas.data(), order.data(), leafCount, sameDepthEpsilon);

            // Stream into running depth group
            for (int k = 0; k < leafCount; ++k) {
                const int i = order[k];
                const float alphaAtHit = leafAlphas[i];
                const float depthKey = leafDepthKeys[i]; // world-ray key
                const float localTHit = leafLocalTHits[i]; // object-space t
                const uint32_t surfelIndex = leafIndices[i];

                if (groupDepthKeys.size() == 0 || sycl::fabs(depthKey - currentGroupDepthKey) <= sameDepthEpsilon) {
                    if (groupDepthKeys.size() == 0) currentGroupDepthKey = depthKey;
                    (void) groupDepthKeys.pushBack(depthKey);
                    (void) groupLocalTs.pushBack(localTHit);
                    (void) groupAlphas.pushBack(alphaAtHit);
                    (void) groupIndices.pushBack(surfelIndex);
                } else {
                    switch (rayIntersectMode) {
                        case RayIntersectMode::Transmit:
                            transmitCurrentGroup(rng128); // must write groupLocalTs[i] to out events
                            break;
                        default:
                            if (scatterCurrentGroup(rng128)) {
                                foundAcceptedScatter = true;
                                bestAcceptedTHit = localHitOut.t; // set from groupLocalTs in scatterCurrentGroup
                                break;
                            }
                    }
                    if (foundAcceptedScatter) break;

                    currentGroupDepthKey = depthKey;
                    groupDepthKeys.clear();
                    groupLocalTs.clear();
                    groupAlphas.clear();
                    groupIndices.clear();

                    (void) groupDepthKeys.pushBack(depthKey);
                    (void) groupLocalTs.pushBack(localTHit);
                    (void) groupAlphas.pushBack(alphaAtHit);
                    (void) groupIndices.pushBack(surfelIndex);
                }
            }
            if (foundAcceptedScatter) break;
        }

        // Tail group
        if (!foundAcceptedScatter && groupLocalTs.size() > 0) {
            if (rayIntersectMode == RayIntersectMode::Transmit) {
                (void) transmitCurrentGroup(rng128);
            } else {
                if (scatterCurrentGroup(rng128)) {
                    foundAcceptedScatter = true;
                    bestAcceptedTHit = localHitOut.t;
                }
            }
        }

        // Pure transmission through this BLAS
        if (!foundAcceptedScatter) {
            localHitOut.transmissivity = cumulativeTransmittanceBefore;
            // do not set t / primitiveIndex
        }

        return foundAcceptedScatter;
    }

    // -----------------------------------------------------------------------------
    // TLAS traversal with near-to-far ordering and multiplicative transmittance
    // -----------------------------------------------------------------------------
    SYCL_EXTERNAL static bool intersectScene(const Ray &rayWorld,
                                             WorldHit *worldHitOut,
                                             const GPUSceneBuffers &scene,
                                             rng::Xorshift128 &rng128,
                                             RayIntersectMode rayIntersectMode = RayIntersectMode::Random,
                                             uint32_t scatterOnPrimitiveIndex = UINT32_MAX) {
        const TLASNode *tlasNodes = scene.tlasNodes;
        const InstanceRecord *instanceRecords = scene.instances;
        const Transform *transforms = scene.transforms;

        bool foundAnySurfaceHit = false;
        const float3 inverseDirectionWorld = safeInvDir(rayWorld.direction);

        worldHitOut->t = FLT_MAX;

        SmallStack<256> traversalStack;
        traversalStack.push(0); // root

        float bestWorldTHit = std::numeric_limits<float>::infinity();
        float transmittanceProduct = 1.0f; // accumulate product over visited splat instances in front of the first hit

        while (!traversalStack.empty()) {
            const int nodeIndex = traversalStack.pop();
            const TLASNode &node = tlasNodes[nodeIndex];

            float nodeTEntry = 0.0f;
            if (!slabIntersectAABB(rayWorld, node, inverseDirectionWorld, bestWorldTHit, nodeTEntry))
                continue;

            if (node.count == 0) {
                // Internal TLAS node: near-to-far push
                const int leftIndex = node.leftChild;
                const int rightIndex = node.rightChild;

                float leftTEntry = std::numeric_limits<float>::infinity();
                float rightTEntry = std::numeric_limits<float>::infinity();

                const bool hitLeft = computeAabbEntry(rayWorld, tlasNodes[leftIndex], inverseDirectionWorld,
                                                      bestWorldTHit, leftTEntry);
                const bool hitRight = computeAabbEntry(rayWorld, tlasNodes[rightIndex], inverseDirectionWorld,
                                                       bestWorldTHit, rightTEntry);

                if (hitLeft && hitRight) {
                    pushNearFar(traversalStack, leftIndex, leftTEntry, rightIndex, rightTEntry);
                } else if (hitLeft) {
                    traversalStack.push(leftIndex);
                } else if (hitRight) {
                    traversalStack.push(rightIndex);
                }
                continue;
            }

            // Leaf: exactly one instance
            const uint32_t instanceIndex = node.leftChild;
            const InstanceRecord &instance = instanceRecords[instanceIndex];
            const Transform &transform = transforms[instance.transformIndex];
            Ray rayObject = toObjectSpace(rayWorld, transform);

            LocalHit localHit{};
            bool acceptedHitInInstance = false;

            if (instance.geometryType == GeometryType::Mesh) {
                acceptedHitInInstance = intersectBLASMesh(rayObject, instance.blasRangeIndex, localHit, scene);
            } else {
                acceptedHitInInstance = intersectBLASPointCloud(rayObject, instance.blasRangeIndex, localHit, scene,
                                                                rng128, transform, rayWorld, rayIntersectMode, scatterOnPrimitiveIndex);
                for (size_t i = 0; i < localHit.splatEventCount; ++i) {
                    worldHitOut->splatEvents[i].alpha = localHit.splatEvents[i].alpha;
                    worldHitOut->splatEvents[i].primitiveIndex = localHit.splatEvents[i].primitiveIndex;

                    const float3 hitPointWorld = toWorldPoint(
                        rayObject.origin + localHit.splatEvents[i].t * rayObject.direction,
                        transform);
                    const float tWorld = dot(hitPointWorld - rayWorld.origin, rayWorld.direction);
                    worldHitOut->splatEvents[i].hitWorld = hitPointWorld;
                    worldHitOut->splatEvents[i].t = tWorld;
                }
                worldHitOut->splatEventCount = localHit.splatEventCount;
            }

            if (acceptedHitInInstance) {
                // Convert to world, test depth
                const float3 hitPointWorld = toWorldPoint(rayObject.origin + localHit.t * rayObject.direction,
                                                          transform);
                const float tWorld = dot(hitPointWorld - rayWorld.origin, rayWorld.direction);
                // assumes normalized direction
                if (tWorld < 0.0f || tWorld >= bestWorldTHit) continue;

                bestWorldTHit = tWorld;
                foundAnySurfaceHit = true;

                worldHitOut->hit = true;
                worldHitOut->t = tWorld;
                worldHitOut->hitPositionW = hitPointWorld;
                worldHitOut->instanceIndex = instanceIndex;
                worldHitOut->primitiveIndex = localHit.primitiveIndex;

                // Multiply transmissions seen before entering this instance with transmission before the accepted event inside it
                worldHitOut->transmissivity = transmittanceProduct * (instance.geometryType == GeometryType::PointCloud
                                                                          ? localHit.transmissivity
                                                                          : 1.0f);
                // Stop traversal because we found the nearest accepted hit
                continue;
            }

            // No accepted hit, but if this was a splat field we may have partial transmission through it
            if (!acceptedHitInInstance && instance.geometryType == GeometryType::PointCloud) {
                transmittanceProduct *= localHit.transmissivity;
            }
        }

        // If no surface hit at all, expose total transmission accumulated
        if (!foundAnySurfaceHit) {
            worldHitOut->hit = false;
            worldHitOut->transmissivity = transmittanceProduct;
        }

        return foundAnySurfaceHit;
    }

    // -----------------------------------------------------------------------------
    // Visibility wrapper
    // -----------------------------------------------------------------------------
    SYCL_EXTERNAL static WorldHit traceVisibility(const Ray &rayIn,
                                                  float /*tMax*/,
                                                  // not used here; early-exit is driven by BVH and bestTHit
                                                  const GPUSceneBuffers &scene,
                                                  rng::Xorshift128 &rng128) {
        WorldHit worldHit{};
        intersectScene(rayIn, &worldHit, scene, rng128);
        return worldHit;
    }
} // namespace Pale
