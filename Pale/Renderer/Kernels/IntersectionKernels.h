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
                                                      RayIntersectMode rayIntersectMode = RayIntersectMode::Random) {
        const BLASRange &blasRange = scene.blasRanges[blasRangeIndex];
        const BVHNode *bvhNodes = scene.blasNodes + blasRange.firstNode;

        float bestAcceptedTHit = std::numeric_limits<float>::infinity();
        bool foundAcceptedScatter = false;
        const float3 inverseDirection = safeInvDir(rayObject.direction);
        constexpr float rayEpsilon = 1e-4f;
        constexpr float sameDepthEpsilon = 1e-5f;

        float cumulativeTransmittanceBefore = 1.0f;

        SmallStack<32> traversalStack;
        traversalStack.push(0);

        float currentGroupDepthKey = -std::numeric_limits<float>::infinity();
        BoundedVector<float, 32> groupDepthKeys;
        BoundedVector<float, 32> groupLocalTs;
        BoundedVector<float, 32> groupAlphas;
        BoundedVector<uint32_t, 32> groupIndices;
        BoundedVector<uint32_t, 32> splatEvents;

        auto clearCurrentGroup = [&]() {
            groupDepthKeys.clear();
            groupLocalTs.clear();
            groupAlphas.clear();
            groupIndices.clear();
            splatEvents.clear();
        };

        // Flushing a group: (ordered Bernoulli thinning) for a group of splats. In the equations this is a S_slice subset S of all the splats along a ray
        auto scatterCurrentGroup = [&](rng::Xorshift128 &rng)-> bool {
            if (groupLocalTs.empty()) return false;

            // Composite acceptance at this depth
            float productOneMinusAlpha = 1.0f;
            for (size_t i = 0; i < groupAlphas.size(); ++i) productOneMinusAlpha *= (1.0f - groupAlphas[i]);
            float compositeAlpha = 1.0f - productOneMinusAlpha;

            if (compositeAlpha <= 0.0f) {
                // No effect
                clearCurrentGroup();
                return false;
            }

            float uniformSample = rng.nextFloat();
            if (uniformSample >= compositeAlpha) {
                // Full transmission through this depth slice
                cumulativeTransmittanceBefore *= (1.0f - compositeAlpha);
                clearCurrentGroup();
                return false;
            }

            // Scatter inside this group using sequential thinning (unbiased)
            float survivalInsideGroup = 1.0f;
            for (size_t i = 0; i < groupAlphas.size(); ++i) {
                float probabilityFirstHere = groupAlphas[i] * survivalInsideGroup / sycl::fmax(compositeAlpha, 1e-8f);
                if (rng.nextFloat() < probabilityFirstHere) {
                    localHitOut.t = groupLocalTs[i];
                    localHitOut.primitiveIndex = groupIndices[i];
                    localHitOut.transmissivity = cumulativeTransmittanceBefore;
                    // transmittance before the accepted event

                    clearCurrentGroup();
                    return true; // accepted scatter at this depth
                }
                survivalInsideGroup *= (1.0f - groupAlphas[i]);
            }

            // Fallback: pick last
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

                if (hitLeft && hitRight) pushNearFar(traversalStack, leftIndex, leftTEntry, rightIndex, rightTEntry);
                else if (hitLeft) traversalStack.push(leftIndex);
                else if (hitRight) traversalStack.push(rightIndex);
                continue;
            }

            // Leaf: collect surfel events, sort by tHit
            BoundedVector<float, 32> leafLocalTHits; // Should match the maxleafpoints in the BVH construction
            BoundedVector<float, 32> leafAlphas;
            BoundedVector<float, 32> leafDepthKeys;
            BoundedVector<uint32_t, 32> leafIndices;
            leafLocalTHits.clear();
            leafAlphas.clear();
            leafIndices.clear();
            leafDepthKeys.clear();

            for (uint32_t local = 0; local < node.triCount; ++local) {
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
            BoundedVector<int, 32> order;
            BoundedVector<float, 32> tempKeys;
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
                const float depthKey        = leafDepthKeys[i];     // world-ray key
                const float localTHit       = leafLocalTHits[i];    // object-space t
                const float alphaAtHit      = leafAlphas[i];
                const uint32_t surfelIndex  = leafIndices[i];

                if (groupDepthKeys.size() == 0 || sycl::fabs(depthKey - currentGroupDepthKey) <= sameDepthEpsilon) {
                    if (groupDepthKeys.size() == 0) currentGroupDepthKey = depthKey;
                    (void)groupDepthKeys.pushBack(depthKey);
                    (void)groupLocalTs.pushBack(localTHit);
                    (void)groupAlphas.pushBack(alphaAtHit);
                    (void)groupIndices.pushBack(surfelIndex);
                } else {
                    switch (rayIntersectMode) {
                        case RayIntersectMode::Transmit:
                            transmitCurrentGroup(rng128);    // must write groupLocalTs[i] to out events
                            break;
                        default:
                            if (scatterCurrentGroup(rng128)) {
                                foundAcceptedScatter = true;
                                bestAcceptedTHit = localHitOut.t;   // set from groupLocalTs in scatterCurrentGroup
                                break;
                            }
                    }
                    if (foundAcceptedScatter) break;

                    currentGroupDepthKey = depthKey;
                    groupDepthKeys.clear();
                    groupLocalTs.clear();
                    groupAlphas.clear();
                    groupIndices.clear();

                    (void)groupDepthKeys.pushBack(depthKey);
                    (void)groupLocalTs.pushBack(localTHit);
                    (void)groupAlphas.pushBack(alphaAtHit);
                    (void)groupIndices.pushBack(surfelIndex);
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
                                             RayIntersectMode rayIntersectMode = RayIntersectMode::Random) {
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

                if (hitLeft && hitRight) pushNearFar(traversalStack, leftIndex, leftTEntry, rightIndex, rightTEntry);
                else if (hitLeft) traversalStack.push(leftIndex);
                else if (hitRight) traversalStack.push(rightIndex);
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
                                                                rng128, transform, rayWorld, rayIntersectMode);
                if (rayIntersectMode == RayIntersectMode::Transmit) {
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
