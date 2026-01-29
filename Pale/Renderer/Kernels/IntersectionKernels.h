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
                                                const GPUSceneBuffers &scene,
                                                const Transform &transform) {
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

                const bool hitLeft = slabIntersectAABB(rayObject, bvhNodes[leftIndex], inverseDirection, bestTHit, leftTEntry);
                const bool hitRight = slabIntersectAABB(rayObject, bvhNodes[rightIndex], inverseDirection, bestTHit, rightTEntry);

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

                    localHitOut.worldHit = toWorldPoint(rayObject.origin + t * rayObject.direction, transform);

                }
            }
        }

        return hitAnyTriangle;
    }

    // -----------------------------------------------------------------------------
    // Point-cloud BLAS with ordered Bernoulli thinning and depth grouping
    // -----------------------------------------------------------------------------
    /*
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
        constexpr float rayEpsilon = 1e-5f;
        constexpr float sameDepthEpsilon = 1e-3f;

        float cumulativeTransmittanceBefore = 1.0f;

        SmallStack<256> traversalStack;
        traversalStack.push(0);

        BoundedVector<float, kMaxSplatEventsPerRay> groupDepthKeys;
        BoundedVector<float, kMaxSplatEventsPerRay> groupLocalTs;
        BoundedVector<float, kMaxSplatEventsPerRay> groupAlphas;
        BoundedVector<uint32_t, kMaxSplatEventsPerRay> groupIndices;

        auto clearCurrentGroup = [&]() {
            groupDepthKeys.clear();
            groupLocalTs.clear();
            groupAlphas.clear();
            groupIndices.clear();
        };

        auto scatterCurrentGroup = [&](rng::Xorshift128 &randomNumberGenerator) -> bool {
            if (groupLocalTs.empty()) return false;

            // 1) Push this slice’s events into LocalHit (depth-sorted already)
            float runningTransmittanceWithinSlice = 1.0f;

            for (size_t groupIndex = 0; groupIndex < groupLocalTs.size(); ++groupIndex) {
                if (localHitOut.splatEventCount >= kMaxSplatEventsPerRay)
                    continue;

                const uint32_t surfelIndex = groupIndices[groupIndex];

                // This matches how you compute eventAlpha later in the thinning step
                const float surfelOpacity = scene.points[surfelIndex].opacity;
                const float eventAlphaEff = groupAlphas[groupIndex] * surfelOpacity; // α_i^eff

                const int eventIndex = localHitOut.splatEventCount++;
                localHitOut.splatEvents[eventIndex].t = groupLocalTs[groupIndex];
                localHitOut.splatEvents[eventIndex].alpha = groupAlphas[groupIndex];
                localHitOut.splatEvents[eventIndex].primitiveIndex = surfelIndex;

                // Global τ_front for this event:
                // τ_before_event = cumulativeTransmittanceBefore * Π_{prev events in this slice}(1 - α_eff)
                const float tauBeforeThisEvent =
                        cumulativeTransmittanceBefore * runningTransmittanceWithinSlice;
                localHitOut.splatEvents[eventIndex].tau = tauBeforeThisEvent;

                // Update within-slice transmittance for the *next* event
                runningTransmittanceWithinSlice *= (1.0f - eventAlphaEff);
            }


            // 2) Composite alpha at this depth slice
            float productOneMinusAlphaEff = 1.0f;
            for (size_t groupIndex = 0; groupIndex < groupAlphas.size(); ++groupIndex) {
                productOneMinusAlphaEff *= (1.0f - groupAlphas[groupIndex] * scene.points[groupIndices[groupIndex]].
                                            opacity);
            }

            float compositeAlphaEff = 1.0f - productOneMinusAlphaEff;
            compositeAlphaEff = sycl::clamp(compositeAlphaEff, 0.0f, 1.0f);

            if (compositeAlphaEff <= 0.0f) {
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
                    if (uniformSample < compositeAlphaEff) {
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
                cumulativeTransmittanceBefore *= (1.0f - compositeAlphaEff);
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
            const float safeCompositeAlpha = sycl::fmax(compositeAlphaEff, 1e-8f);

            for (size_t groupIndex = 0; groupIndex < groupAlphas.size(); ++groupIndex) {
                const float opacity = scene.points[groupIndices[groupIndex]].opacity;

                const float eventAlpha = groupAlphas[groupIndex] * opacity;
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

        BoundedVector<float,    kMaxSplatEventsPerRay> candidateLocalTs;
        BoundedVector<float,    kMaxSplatEventsPerRay> candidateAlphas;
        BoundedVector<float,    kMaxSplatEventsPerRay> candidateDepthKeys;
        BoundedVector<uint32_t, kMaxSplatEventsPerRay> candidateIndices;


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

            // Leaf: just collect candidates, DO NOT clear the global arrays here
            for (uint32_t local = 0; local < node.triCount; ++local) {
                if (candidateLocalTs.size() >= kMaxSplatEventsPerRay)
                    break;

                const uint32_t surfelIndex = node.leftFirst + local;
                const Point& surfel = scene.points[surfelIndex];

                float tHitLocal = 0.0f;
                float opacity = 0.0f;
                float3 outHitLocal;
                if (!intersectSurfel(rayObject, surfel, rayEpsilon, FLT_MAX, tHitLocal, outHitLocal, opacity))
                    continue;

                const float3 worldPoint = toWorldPoint(outHitLocal, transform);
                const float depthKey = dot(worldPoint - rayWorld.origin, rayWorld.direction);
                const float alphaAtHit = sycl::clamp(opacity, 0.0f, 1.0f);

                candidateLocalTs.pushBack(tHitLocal);
                candidateAlphas.pushBack(alphaAtHit);
                candidateDepthKeys.pushBack(depthKey);
                candidateIndices.pushBack(surfelIndex);
            }
        }

        const int candidateCount = candidateLocalTs.size();
        BoundedVector<int,   kMaxSplatEventsPerRay> order;
        BoundedVector<float, kMaxSplatEventsPerRay> tempKeys;

        order.clear();
        tempKeys.clear();
        for (int i = 0; i < candidateCount; ++i) {
            order.pushBack(i);
            tempKeys.pushBack(candidateDepthKeys[i]);
        }

        // Same insertionSortByKeyTie you already have
        insertionSortByKeyTie(tempKeys.data(),
                              candidateAlphas.data(),
                              order.data(),
                              candidateCount,
                              sameDepthEpsilon);

        float currentGroupDepthKey = -std::numeric_limits<float>::infinity();
        groupDepthKeys.clear();
        groupLocalTs.clear();
        groupAlphas.clear();
        groupIndices.clear();

        for (int k = 0; k < candidateCount; ++k) {
            const int i = order[k];
            const float depthKey   = candidateDepthKeys[i];
            const float localTHit  = candidateLocalTs[i];
            const float alphaAtHit = candidateAlphas[i];
            const uint32_t surfelIndex = candidateIndices[i];

            if (groupDepthKeys.size() == 0 ||
                sycl::fabs(depthKey - currentGroupDepthKey) <= sameDepthEpsilon) {

                if (groupDepthKeys.size() == 0)
                    currentGroupDepthKey = depthKey;

                groupDepthKeys.pushBack(depthKey);
                groupLocalTs.pushBack(localTHit);
                groupAlphas.pushBack(alphaAtHit);
                groupIndices.pushBack(surfelIndex);
                } else {
                    // finalize previous slice
                    if (scatterCurrentGroup(rng128)) {
                        foundAcceptedScatter = true;
                        bestAcceptedTHit = localHitOut.t;
                        break;
                    }

                    // start new slice
                    currentGroupDepthKey = depthKey;
                    groupDepthKeys.clear();
                    groupLocalTs.clear();
                    groupAlphas.clear();
                    groupIndices.clear();

                    groupDepthKeys.pushBack(depthKey);
                    groupLocalTs.pushBack(localTHit);
                    groupAlphas.pushBack(alphaAtHit);
                    groupIndices.pushBack(surfelIndex);
                }
        }

        // tail slice
        if (!foundAcceptedScatter && groupLocalTs.size() > 0) {
            if (scatterCurrentGroup(rng128)) {
                foundAcceptedScatter = true;
                bestAcceptedTHit = localHitOut.t;
            }
        }


        // Pure transmission through this BLAS
        if (!foundAcceptedScatter) {
            localHitOut.transmissivity = cumulativeTransmittanceBefore;
            // do not set t / primitiveIndex
        }

        return foundAcceptedScatter;
    }
    */

    // ----------------------------------------------------------------------------
// Point-cloud BLAS without per-ray event list:
// Repeated closest-hit queries + stochastic accept/reject.
// - No candidate arrays
// - No sorting
// - Correct front-to-back behavior by construction (tMin advances)
// ----------------------------------------------------------------------------
SYCL_EXTERNAL static bool intersectBLASPointCloudStochastic(const Ray &rayObject,
                                                            uint32_t blasRangeIndex,
                                                            LocalHit &localHitOut,
                                                            const GPUSceneBuffers &scene,
                                                            rng::Xorshift128 &rng128,
                                                            RayIntersectMode rayIntersectMode,
                                                            uint32_t scatterOnPrimitiveIndex) {
    const BLASRange &blasRange = scene.blasRanges[blasRangeIndex];
    const BVHNode *bvhNodes = scene.blasNodes + blasRange.firstNode;

    constexpr float rayEpsilon = 1e-5f;
    constexpr float tAdvanceEpsilon = 1e-4f; // advance after a rejected hit to avoid re-hitting same surfel
    constexpr uint32_t maxRejections = 256;  // cap work per ray (tune)

    float cumulativeTransmittanceBefore = 1.0f;

    // Find next closest surfel hit with t in (tMin, tMax).
    auto findNextClosestSurfel = [&](float tMin,
                                     float tMax,
                                     float &outTHit,
                                     uint32_t &outSurfelIndex,
                                     float &outAlphaGeomAtHit) -> bool {
        bool hitAny = false;
        float bestTHit = tMax;

        const float3 inverseDirection = safeInvDir(rayObject.direction);

        SmallStack<256> traversalStack;
        traversalStack.push(0);

        while (!traversalStack.empty()) {
            const int nodeIndex = traversalStack.pop();
            const BVHNode &node = bvhNodes[nodeIndex];

            float nodeTEntry = 0.0f;
            if (!slabIntersectAABB(rayObject, node, inverseDirection, bestTHit, nodeTEntry))
                continue;

            if (node.triCount == 0) {
                const int leftIndex = node.leftFirst;
                const int rightIndex = node.leftFirst + 1;

                float leftTEntry = std::numeric_limits<float>::infinity();
                float rightTEntry = std::numeric_limits<float>::infinity();

                const bool hitLeft = slabIntersectAABB(rayObject, bvhNodes[leftIndex], inverseDirection, bestTHit, leftTEntry);
                const bool hitRight = slabIntersectAABB(rayObject, bvhNodes[rightIndex], inverseDirection, bestTHit, rightTEntry);

                if (hitLeft && hitRight) pushNearFar(traversalStack, leftIndex, leftTEntry, rightIndex, rightTEntry);
                else if (hitLeft) traversalStack.push(leftIndex);
                else if (hitRight) traversalStack.push(rightIndex);
                continue;
            }

            // Leaf: test surfels
            for (uint32_t local = 0; local < node.triCount; ++local) {
                const uint32_t surfelIndex = node.leftFirst + local;
                const Point &surfel = scene.points[surfelIndex];

                float tHitLocal = 0.0f;
                float alphaGeom = 0.0f;
                float3 hitLocal{};
                if (!intersectSurfel(rayObject, surfel, rayEpsilon, bestTHit, tHitLocal, hitLocal, alphaGeom))
                    continue;

                if (tHitLocal <= tMin)
                    continue;

                // Keep closest
                bestTHit = tHitLocal;
                outSurfelIndex = surfelIndex;
                outAlphaGeomAtHit = sycl::clamp(alphaGeom, 0.0f, 1.0f);
                hitAny = true;
            }
        }

        if (!hitAny)
            return false;

        outTHit = bestTHit;
        return true;
    };

    // Stochastic accept/reject loop over successive closest hits
    float tMin = rayEpsilon;

    for (uint32_t rejectionCount = 0; rejectionCount < maxRejections; ++rejectionCount) {
        float tHit = 0.0f;
        uint32_t surfelIndex = 0;
        float alphaGeomAtHit = 0.0f;

        if (!findNextClosestSurfel(tMin, std::numeric_limits<float>::infinity(), tHit, surfelIndex, alphaGeomAtHit)) {
            // No more candidates: pure transmission through this BLAS
            localHitOut.transmissivity = cumulativeTransmittanceBefore;
            return false;
        }

        const Point &surfel = scene.points[surfelIndex];

        // Effective interaction probability at this candidate
        const float alphaEff = sycl::clamp(alphaGeomAtHit * surfel.opacity, 0.0f, 1.0f);

        // Random mode: accept with probability alphaEff, otherwise transmit and continue
         const float u = rng128.nextFloat();
            if (u < alphaEff) {
                localHitOut.t = tHit;
                localHitOut.primitiveIndex = surfelIndex;
                localHitOut.transmissivity = cumulativeTransmittanceBefore;
                return true;
            }

            cumulativeTransmittanceBefore *= (1.0f - alphaEff);
            tMin = tHit + tAdvanceEpsilon;
    }

    // Work cap reached: return whatever transmittance we accumulated (biases slightly if cap triggers often)
    localHitOut.transmissivity = cumulativeTransmittanceBefore;
    return false;
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

                const bool hitLeft = slabIntersectAABB(rayWorld, tlasNodes[leftIndex], inverseDirectionWorld,
                                                      bestWorldTHit, leftTEntry);
                const bool hitRight = slabIntersectAABB(rayWorld, tlasNodes[rightIndex], inverseDirectionWorld,
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
                acceptedHitInInstance = intersectBLASMesh(rayObject, instance.blasRangeIndex, localHit, scene, transform);
            } else {
                acceptedHitInInstance = intersectBLASPointCloudStochastic(
                                         rayObject,
                                         instance.blasRangeIndex,
                                         localHit,
                                         scene,
                                         rng128,
                                         rayIntersectMode,
                                         scatterOnPrimitiveIndex);

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
