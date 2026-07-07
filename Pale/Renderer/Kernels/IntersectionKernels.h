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


    struct ChildEntry {
        int nodeIndex;
        float tEntry;
    };

    template <typename StackT>
    SYCL_EXTERNAL inline void pushNearFar(StackT& traversalStack,
                                          int leftIndex, float leftTEntry,
                                          int rightIndex, float rightTEntry) {
        if (leftTEntry <= rightTEntry) {
            traversalStack.push(rightIndex);
            traversalStack.push(leftIndex);
        }
        else {
            traversalStack.push(leftIndex);
            traversalStack.push(rightIndex);
        }
    }

    // -----------------------------------------------------------------------------
    // Triangle BLAS (unchanged except near-to-far child push)
    // -----------------------------------------------------------------------------
    SYCL_EXTERNAL static bool intersectBLASMesh(const Ray& rayObject,
                                                uint32_t geometryIndex,
                                                LocalHit& localHitOut,
                                                const GPUSceneBuffers& scene,
                                                const Transform& transform,
                                                float tMin = 0.0f) {
        const BLASRange& blasRange = scene.blasRanges[geometryIndex];
        const BVHNode* bvhNodes = scene.blasNodes + blasRange.firstNode;
        const Triangle* triangles = scene.triangles;
        const Vertex* vertices = scene.vertices;

        float bestTHit = std::numeric_limits<float>::infinity();
        bool hitAnyTriangle = false;
        const float3 inverseDirection = safeInvDir(rayObject.direction);

        SmallStack<256> traversalStack;
        traversalStack.push(0); // root

        while (!traversalStack.empty()) {
            const int nodeIndex = traversalStack.pop();
            const BVHNode& node = bvhNodes[nodeIndex];

            float nodeTEntry = 0.0f;
            if (!slabIntersectAABB(rayObject, node, inverseDirection, bestTHit, nodeTEntry))
                continue;

            if (node.triCount == 0) {
                // Internal: left child is node.leftFirst, right child is node.leftFirst + 1
                const int leftIndex = node.leftFirst;
                const int rightIndex = node.leftFirst + 1;

                float leftTEntry = std::numeric_limits<float>::infinity();
                float rightTEntry = std::numeric_limits<float>::infinity();

                const bool hitLeft = slabIntersectAABB(rayObject, bvhNodes[leftIndex], inverseDirection, bestTHit,
                                                       leftTEntry);
                const bool hitRight = slabIntersectAABB(rayObject, bvhNodes[rightIndex], inverseDirection, bestTHit,
                                                        rightTEntry);

                if (hitLeft && hitRight) pushNearFar(traversalStack, leftIndex, leftTEntry, rightIndex, rightTEntry);
                else if (hitLeft) traversalStack.push(leftIndex);
                else if (hitRight) traversalStack.push(rightIndex);
                continue;
            }

            // Leaf: test triangles
            for (uint32_t i = 0; i < node.triCount; ++i) {
                uint32_t triangleIndex = node.leftFirst + i; // global index
                const Triangle& tri = triangles[triangleIndex];

                const float3 A = vertices[tri.v0].pos;
                const float3 B = vertices[tri.v1].pos;
                const float3 C = vertices[tri.v2].pos;

                float t = FLT_MAX, u = 0.0f, v = 0.0f;
                if (intersectTriangle(rayObject, A, B, C, t, u, v, 1e-4f) && t < bestTHit && t > tMin) {
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


    // ----------------------------------------------------------------------------
    // Point-cloud BLAS: single closest-hit query (no event list, no sorting).
    // Returns the first (nearest) surfel intersection in (ray.tMin, ray.tMax).
    // ----------------------------------------------------------------------------
    SYCL_EXTERNAL static bool intersectBLASPointCloudFirstHit(
        const Ray& rayObject,
        uint32_t blasRangeIndex,
        LocalHit& localHitOut,
        const GPUSceneBuffers& scene) {
        const BLASRange& blasRange = scene.blasRanges[blasRangeIndex];
        const BVHNode* bvhNodes = scene.blasNodes + blasRange.firstNode;


        bool hitAny = false;
        float bestTHit = std::numeric_limits<float>::infinity();
        uint32_t bestSurfelIndex = 0u;

        float bestAlphaGeomAtHit = 0.0f;
        float3 bestHitLocal{0.0f};

        const float3 inverseDirection = safeInvDir(rayObject.direction);

        SmallStack<256> traversalStack;
        traversalStack.push(0);

        while (!traversalStack.empty()) {
            const int nodeIndex = traversalStack.pop();
            const BVHNode& node = bvhNodes[nodeIndex];

            float nodeTEntry = 0.0f;
            if (!slabIntersectAABB(rayObject, node, inverseDirection, bestTHit, nodeTEntry))
                continue;

            if (node.triCount == 0) {
                const int leftIndex = node.leftFirst;
                const int rightIndex = node.leftFirst + 1;

                float leftTEntry = std::numeric_limits<float>::infinity();
                float rightTEntry = std::numeric_limits<float>::infinity();

                const bool hitLeft = slabIntersectAABB(rayObject, bvhNodes[leftIndex], inverseDirection, bestTHit,
                                                       leftTEntry);
                const bool hitRight = slabIntersectAABB(rayObject, bvhNodes[rightIndex], inverseDirection, bestTHit,
                                                        rightTEntry);

                if (hitLeft && hitRight) {
                    pushNearFar(traversalStack, leftIndex, leftTEntry, rightIndex, rightTEntry);
                }
                else if (hitLeft) {
                    traversalStack.push(leftIndex);
                }
                else if (hitRight) {
                    traversalStack.push(rightIndex);
                }
                continue;
            }

            // Leaf: test surfels
            for (uint32_t primitiveOffset = 0; primitiveOffset < node.triCount; ++primitiveOffset) {
                const uint32_t primitiveIndex =
                    scene.pointPermutation[node.leftFirst + primitiveOffset];

                const Point& surfel = scene.points[primitiveIndex];

                float tHitLocal = 0.0f;
                float alphaGeom = 0.0f;

                if (surfel.isEmissive() || !intersectSurfel(rayObject, surfel, RayEpsilon2, bestTHit, tHitLocal,
                                                            RayEpsilon2))
                    continue;

                float3 hitLocal = rayObject.origin + tHitLocal * rayObject.direction;
                const float2 uv = phiInverse(hitLocal, surfel);
                if (!opacityBeta(uv[0], uv[1], surfel, &alphaGeom) || alphaGeom <= 0.0f) {
                    continue;
                }

                // Keep closest
                hitAny = true;
                bestTHit = tHitLocal;
                bestSurfelIndex = primitiveIndex;
                bestHitLocal = hitLocal;
                bestAlphaGeomAtHit = alphaGeom;
            }
        }

        if (!hitAny)
            return false;

        // Populate output hit.
        // Note: exact field names depend on your LocalHit definition.
        localHitOut.t = bestTHit;
        localHitOut.primitiveIndex = bestSurfelIndex;

        // For this "first hit" query there is no prior absorption handled here.
        // If you want per-ray compositing, use alphaGeom at the caller.
        localHitOut.transmissivity = 1.0f;
        localHitOut.alpha = bestAlphaGeomAtHit;

        localHitOut.worldHit = bestHitLocal;


        return true;
    }

    SYCL_EXTERNAL static uint32_t collectBLASPointCloudLocalLayer(
        const Ray& rayWorld,
        const Ray& rayObject,
        uint32_t blasRangeIndex,
        const Transform& transform,
        float tMinWorld,
        float tMaxWorld,
        LocalSurfelLayerHit* localHits,
        uint32_t maxLocalHitCount,
        const GPUSceneBuffers& scene) {
        const BLASRange& blasRange = scene.blasRanges[blasRangeIndex];
        const BVHNode* bvhNodes = scene.blasNodes + blasRange.firstNode;
        const float3 inverseDirectionObject = safeInvDir(rayObject.direction);
        uint32_t localHitCount = 0u;
        SmallStack<256> traversalStack;
        traversalStack.push(0);
        while (!traversalStack.empty()) {
            const int nodeIndex = traversalStack.pop();
            const BVHNode& node = bvhNodes[nodeIndex];
            float nodeTEntry = 0.0f;
            if (!slabIntersectAABB(rayObject, node, inverseDirectionObject, std::numeric_limits<float>::infinity(),
                                   nodeTEntry)) {
                continue;
            }
            if (node.triCount == 0u) {
                const int leftIndex = node.leftFirst;
                const int rightIndex = node.leftFirst + 1;
                float leftTEntry = std::numeric_limits<float>::infinity();
                float rightTEntry = std::numeric_limits<float>::infinity();
                const bool hitLeft = slabIntersectAABB(rayObject, bvhNodes[leftIndex], inverseDirectionObject,
                                                       std::numeric_limits<float>::infinity(), leftTEntry);
                const bool hitRight = slabIntersectAABB(rayObject, bvhNodes[rightIndex], inverseDirectionObject,
                                                        std::numeric_limits<float>::infinity(), rightTEntry);
                if (hitLeft && hitRight) {
                    pushNearFar(traversalStack, leftIndex, leftTEntry, rightIndex, rightTEntry);
                }
                else if (hitLeft) {
                    traversalStack.push(leftIndex);
                }
                else if (hitRight) {
                    traversalStack.push(rightIndex);
                }
                continue;
            }

            for (uint32_t primitiveOffset = 0u; primitiveOffset < node.triCount; ++primitiveOffset) {
                const uint32_t primitiveIndex = scene.pointPermutation[node.leftFirst + primitiveOffset];
                const Point& surfel = scene.points[primitiveIndex];
                // Preserve your current FirstHit behavior.
                if (surfel.isEmissive()) {
                    continue;
                }
                float tHitObject = 0.0f;
                float alphaGeom = 0.0f;
                float3 hitPositionObject(0.0f);
                if (!intersectSurfel(rayObject, surfel, RayEpsilon2, std::numeric_limits<float>::infinity(), tHitObject,
                                     RayEpsilon2)) {
                    continue;
                }

                hitPositionObject = rayObject.origin + tHitObject * rayObject.direction;
                const float2 uv = phiInverse(hitPositionObject, surfel);
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
                insertLocalSurfelLayerHit(
                    localHits,
                    localHitCount,
                    maxLocalHitCount,
                    candidateHit);
            }
        }

        return localHitCount;
    }

    // Transmit and only attenuate the ray.
    SYCL_EXTERNAL static bool intersectBLASPointCloudTransmit(
        const Ray& rayObject,
        uint32_t blasRangeIndex,
        LocalHit& localHitOut,
        const GPUSceneBuffers& scene) {
        const BLASRange& blasRange = scene.blasRanges[blasRangeIndex];
        const BVHNode* bvhNodes = scene.blasNodes + blasRange.firstNode;

        float cumulativeTransmittance = 1.0f;

        // Find next closest surfel hit with t in (tMin, tMax).
        auto findNextClosestSurfel = [&](float tMin,
                                         float tMax,
                                         float& outTHit,
                                         uint32_t& outSurfelIndex,
                                         float& outAlphaGeomAtHit) -> bool {
            bool hitAny = false;
            float bestTHit = tMax;

            const float3 inverseDirection = safeInvDir(rayObject.direction);

            SmallStack<256> traversalStack;
            traversalStack.push(0);

            while (!traversalStack.empty()) {
                const int nodeIndex = traversalStack.pop();
                const BVHNode& node = bvhNodes[nodeIndex];

                float nodeTEntry = 0.0f;
                if (!slabIntersectAABB(rayObject, node, inverseDirection, bestTHit, nodeTEntry))
                    continue;

                if (node.triCount == 0) {
                    const int leftIndex = node.leftFirst;
                    const int rightIndex = node.leftFirst + 1;

                    float leftTEntry = std::numeric_limits<float>::infinity();
                    float rightTEntry = std::numeric_limits<float>::infinity();

                    const bool hitLeft = slabIntersectAABB(rayObject, bvhNodes[leftIndex], inverseDirection, bestTHit,
                                                           leftTEntry);
                    const bool hitRight = slabIntersectAABB(rayObject, bvhNodes[rightIndex], inverseDirection, bestTHit,
                                                            rightTEntry);

                    if (hitLeft && hitRight)
                        pushNearFar(traversalStack, leftIndex, leftTEntry, rightIndex, rightTEntry);
                    else if (hitLeft) traversalStack.push(leftIndex);
                    else if (hitRight) traversalStack.push(rightIndex);
                    continue;
                }

                // Leaf: test surfels
                for (uint32_t primitiveOffset = 0; primitiveOffset < node.triCount; ++primitiveOffset) {
                    const uint32_t primitiveIndex =
                        scene.pointPermutation[node.leftFirst + primitiveOffset];

                    const Point& surfel = scene.points[primitiveIndex];

                    float tHitLocal = 0.0f;
                    float alphaGeom = 0.0f;
                    float3 hitLocal{};
                    if (!intersectSurfel(rayObject, surfel, RayEpsilon2, bestTHit, tHitLocal,
                                         RayEpsilon2))
                        continue;

                    hitLocal = rayObject.origin + tHitLocal * rayObject.direction;
                    const float2 uv = phiInverse(hitLocal, surfel);
                    if (!opacityBeta(uv[0], uv[1], surfel, &alphaGeom) || alphaGeom <= 0.0f) {
                        continue;
                    }

                    if (tHitLocal <= tMin)
                        continue;

                    // Keep closest
                    bestTHit = tHitLocal;
                    outSurfelIndex = primitiveIndex;
                    outAlphaGeomAtHit = alphaGeom;
                    hitAny = true;
                }
            }

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
    SYCL_EXTERNAL static bool intersectScene(const Ray& rayWorld,
                                             WorldHit* worldHitOut,
                                             const GPUSceneBuffers& scene,
                                             SurfelIntersectMode rayIntersectMode = SurfelIntersectMode::FirstHit) {
        const TLASNode* tlasNodes = scene.tlasNodes;
        const InstanceRecord* instanceRecords = scene.instances;
        const Transform* transforms = scene.transforms;

        bool foundAnySurfaceHit = false;
        const float3 inverseDirectionWorld = safeInvDir(rayWorld.direction);

        worldHitOut->t = FLT_MAX;

        SmallStack<256> traversalStack;
        traversalStack.push(0); // root

        float bestWorldTHit = std::numeric_limits<float>::infinity();
        float transmittanceProduct = 1.0f; // accumulate product over visited splat instances in front of the first hit

        while (!traversalStack.empty()) {
            const uint32_t nodeIndex = traversalStack.pop();
            const TLASNode& node = tlasNodes[nodeIndex];
            float nodeTEntry = 0.0f;
            if (!slabIntersectAABB(rayWorld, node, inverseDirectionWorld, bestWorldTHit, nodeTEntry))
                continue;
            if (node.count == 0) {
                // Internal TLAS node: near-to-far push
                const uint32_t leftIndex = node.leftChild;
                const uint32_t rightIndex = node.rightChild;
                float leftTEntry = std::numeric_limits<float>::infinity();
                float rightTEntry = std::numeric_limits<float>::infinity();
                const bool hitLeft = slabIntersectAABB(rayWorld, tlasNodes[leftIndex], inverseDirectionWorld,
                                                       bestWorldTHit, leftTEntry);
                const bool hitRight = slabIntersectAABB(rayWorld, tlasNodes[rightIndex], inverseDirectionWorld,
                                                        bestWorldTHit, rightTEntry);
                if (hitLeft && hitRight) {
                    pushNearFar(traversalStack, leftIndex, leftTEntry, rightIndex, rightTEntry);
                }
                else if (hitLeft) {
                    traversalStack.push(leftIndex);
                }
                else if (hitRight) {
                    traversalStack.push(rightIndex);
                }
                continue;
            }

            // Leaf: exactly one instance
            const uint32_t instanceIndex = node.leftChild;
            const InstanceRecord& instance = instanceRecords[instanceIndex];
            const Transform& transform = transforms[instance.transformIndex];
            Ray rayObject = toObjectSpace(rayWorld, transform);
            LocalHit localHit{};
            bool acceptedHitInInstance = false;
            if (instance.geometryType == GeometryType::Mesh) {
                acceptedHitInInstance = intersectBLASMesh(rayObject, instance.blasRangeIndex, localHit, scene,
                                                          transform);
            }
            else {
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
        return foundAnySurfaceHit;
    }

    SYCL_EXTERNAL inline float traceShadowTransmissionToPoint(
        const GPUSceneBuffers& scene,
        const PathTracerSettings& settings,
        const float3& shadingPositionW,
        const float3& shadingNormalW,
        const float3& lightPositionW,
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

            const InstanceRecord& hitInstance =
                scene.instances[shadowHit.instanceIndex];

            if (hitInstance.geometryType == GeometryType::Mesh) {
                return 0.0f;
            }

            if (hitInstance.geometryType != GeometryType::PointCloud) {
                return 0.0f;
            }

            const Transform& transform =
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
                const LocalSurfelLayerHit& localHit =
                    localHits[localHitIndex];

                // Never let hits at or behind the light sample block it.
                if (localHit.tWorld >= remainingLightDistance - eps) {
                    continue;
                }

                furthestLayerT = sycl::fmax(
                    furthestLayerT,
                    localHit.tWorld);

                const Point& surfel =
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
        const GPUSceneBuffers& scene,
        const PathTracerSettings& settings,
        const float3& surfacePositionW,
        const float3& surfaceNormalW,
        const float3& diffuseAlbedo,
        const float eps = RayEpsilon) {
        float3 accumulatedRadiance(0.0f);

        const float3 diffuseBrdf = diffuseAlbedo * M_1_PIf;

        for (uint32_t lightIndex = 0u;
             lightIndex < scene.lightCount;
             ++lightIndex) {
            const GPULightRecord& light = scene.lights[lightIndex];

            // Your current light records appear to use emissive surfels as
            // light-position carriers. We interpret each as an isotropic point light.
            if (light.lightType != LightType::Surfel) {
                continue;
            }

            const Point& lightSurfel = scene.points[light.primitiveIndex];
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

    /*
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

    /*
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

    /*
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

    /*
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

    /*
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
