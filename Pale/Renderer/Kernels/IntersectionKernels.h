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
                                                const Transform &transform,
                                                float tMin = 0.0f) {
        const BLASRange &blasRange = scene.blasRanges[geometryIndex];
        const BVHNode *bvhNodes = scene.blasNodes + blasRange.firstNode;
        const Triangle *triangles = scene.triangles;
        const Vertex *vertices = scene.vertices;

        float bestTHit = std::numeric_limits<float>::infinity();
        bool hitAnyTriangle = false;
        const float3 inverseDirection = safeInvDir(rayObject.direction);

        SmallStack<256> traversalStack;
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
                const Triangle &tri = triangles[triangleIndex];

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
        const Ray &rayObject,
        uint32_t blasRangeIndex,
        LocalHit &localHitOut,
        const GPUSceneBuffers &scene) {
        const BLASRange &blasRange = scene.blasRanges[blasRangeIndex];
        const BVHNode *bvhNodes = scene.blasNodes + blasRange.firstNode;


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
            const BVHNode &node = bvhNodes[nodeIndex];

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
                } else if (hitLeft) {
                    traversalStack.push(leftIndex);
                } else if (hitRight) {
                    traversalStack.push(rightIndex);
                }
                continue;
            }

            // Leaf: test surfels
            for (uint32_t primitiveOffset = 0; primitiveOffset < node.triCount; ++primitiveOffset) {
                const uint32_t primitiveIndex =
                        scene.pointPermutation[node.leftFirst + primitiveOffset];

                const Point &surfel = scene.points[primitiveIndex];

                float tHitLocal = 0.0f;
                float alphaGeom = 0.0f;
                float3 hitLocal{0.0f};

                if (surfel.isEmissive() || !intersectSurfel(rayObject, surfel, RayEpsilon, bestTHit, tHitLocal, hitLocal, alphaGeom, RayEpsilon))
                    continue;

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

    // Transmit and only attenuate the ray.
    SYCL_EXTERNAL static bool intersectBLASPointCloudTransmit(
        const Ray &rayObject,
        uint32_t blasRangeIndex,
        LocalHit &localHitOut,
        const GPUSceneBuffers &scene) {
        const BLASRange &blasRange = scene.blasRanges[blasRangeIndex];
        const BVHNode *bvhNodes = scene.blasNodes + blasRange.firstNode;

        constexpr float tAdvanceEpsilon = 1e-8f; // advance after a rejected hit to avoid re-hitting same surfel

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

                    const Point &surfel = scene.points[primitiveIndex];

                    float tHitLocal = 0.0f;
                    float alphaGeom = 0.0f;
                    float3 hitLocal{};
                    if (!intersectSurfel(rayObject, surfel, RayEpsilon, bestTHit, tHitLocal, hitLocal, alphaGeom, RayEpsilon))
                        continue;

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
        float tMin = RayEpsilon;
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
            tMin = tHit + tAdvanceEpsilon;

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

        SmallStack<64> traversalStack;
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

                // If rayWorld.direction normalized:
                const float tWorld = dot(toHitWorld, rayWorld.direction);

                // If NOT normalized:

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

    SYCL_EXTERNAL inline float traceShadowTransmissionToLight(
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

        constexpr uint32_t maxShadowTraversals = 32u;

        Ray shadowRay{};
        shadowRay.origin = shadingPositionW + lightDirection * RayEpsilon;
        shadowRay.direction = lightDirection;
        shadowRay.normal = shadingNormalW;

        float shadowTransmission = 1.0f;

        for (uint32_t shadowTraversalIndex = 0u;
             shadowTraversalIndex < maxShadowTraversals;
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

            if (hitDistance >= lightDistance - RayEpsilon) {
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

                shadowRay.origin = shadowHit.hitPositionW + shadowRay.direction * RayEpsilon;
                continue;
            }

            return 0.0f;
        }

        return shadowTransmission;
    }


    SYCL_EXTERNAL inline float3 estimateDirectLightAtDiffuseSurface(
        const GPUSceneBuffers &scene,
        const float3 &shadingPositionW,
        const float3 &shadingNormalW,
        const float3 &diffuseAlbedo,
        uint32_t numShadowRays,
        rng::Xorshift128 &rng128) {
        if (numShadowRays == 0u) {
            return float3(0.0f);
        }

        float3 accumulatedDirectRadiance(0.0f);

        for (uint32_t shadowSampleIndex = 0u;
             shadowSampleIndex < numShadowRays;
             ++shadowSampleIndex) {
            const AreaLightSample lightSample = sampleMeshAreaLight(scene, rng128);
            if (!lightSample.valid) {
                continue;
            }

            const float fullPdfArea = lightSample.pdfSelectLight * lightSample.pdfArea;
            if (fullPdfArea <= 0.0f) {
                continue;
            }

            const float3 lightVector = lightSample.positionW - shadingPositionW;
            const float lightDistanceSquared = dot(lightVector, lightVector);
            if (lightDistanceSquared <= 1e-12f) {
                continue;
            }

            const float lightDistance = sycl::sqrt(lightDistanceSquared);
            const float3 lightDirection = lightVector / lightDistance;

            const float shadingCosine =
                    sycl::fmax(0.0f, dot(shadingNormalW, lightDirection));
            if (shadingCosine <= 0.0f) {
                continue;
            }

            const float lightCosine =
                    sycl::fmax(0.0f, dot(lightSample.normalW, -lightDirection));
            if (lightCosine <= 0.0f) {
                continue;
            }

            const float shadowTransmission = traceShadowTransmissionToLight(
                scene,
                shadingPositionW,
                shadingNormalW,
                lightSample.positionW);

            if (shadowTransmission <= 0.0f) {
                continue;
            }

            const float geometricTerm =
                    (shadingCosine * lightCosine) / (lightDistanceSquared + 1e-8f);

            const float3 diffuseBrdf = diffuseAlbedo * M_1_PIf;

            float3 radiance = lightSample.flux / (M_PIf * lightSample.totalAreaWorld);
            // Here lightSample.power is treated as emitted radiance, to stay
            // consistent with your existing mesh emissive branch:
            // material.power * material.baseColor
            const float3 sampleContribution =
                diffuseBrdf *
                radiance *
                shadowTransmission *
                (geometricTerm / fullPdfArea);

            accumulatedDirectRadiance += sampleContribution;
        }

        return accumulatedDirectRadiance * (1.0f / static_cast<float>(numShadowRays));
    }
} // namespace Pale
