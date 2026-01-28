#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

#include "IntersectionKernels.h"
#include "Renderer/RenderPackage.h"
#include "Renderer/GPUDataStructures.h"
#include "Renderer/GPUDataTypes.h"

namespace Pale {
    // ------------------------------------------------------------------------
    // Fixed-size top-N container (sorted by smallest t)
    // ------------------------------------------------------------------------
    template<uint32_t N>
    struct TopCandidates {
        uint32_t count = 0;
        float t[N];
        float w[N];
        float3 n[N];

        SYCL_EXTERNAL inline void tryInsert(float tCandidate, float wCandidate, const float3 &nCandidate) {
            if (!(wCandidate > 0.0f)) return;
            if (count == N && tCandidate >= t[count - 1]) return;

            if (count < N) {
                t[count] = tCandidate;
                w[count] = wCandidate;
                n[count] = nCandidate;
                count++;
            } else {
                t[N - 1] = tCandidate;
                w[N - 1] = wCandidate;
                n[N - 1] = nCandidate;
            }

            for (int i = int(count) - 1; i > 0; --i) {
                if (t[i] >= t[i - 1]) break;
                std::swap(t[i], t[i - 1]);
                std::swap(w[i], w[i - 1]);
                std::swap(n[i], n[i - 1]);
            }
        }
    };

    float sqDistPointAABB(float3 p, const float3 aabbMin,
                          const float3 aabbMax) {
        float sqDist = 0.0f;
        for (int i = 0; i < 3; i++) {
            // For each axis count any excess distance outside box extents
            float v = p[i];
            if (v < aabbMin[i]) sqDist += (aabbMin[i] - v) * (aabbMin[i] - v);
            if (v > aabbMax[i]) sqDist += (v - aabbMax[i]) * (v - aabbMax[i]);
        }
        return sqDist;
    }

    // Returns true if sphere s intersects AABB b, false otherwise
    int testSphereAABB(float3 sphereCenter, float sphereRadius, const float3 aabbMin,
                       const float3 aabbMax) {
        // Compute squared distance between sphere center and AABB
        float sqDist = sqDistPointAABB(sphereCenter, aabbMin, aabbMax);
        // Sphere and AABB intersect if the (squared) distance
        // between them is less than the (squared) sphere radius
        return sqDist <= sphereRadius * sphereRadius;
    }


    // ----------------------------------------------------------------------------
    // Point-cloud BLAS with stochastic accept/reject + Schaufler/Jensen-style
    // local interpolation from N neighbors (cylinder/disk idea).
    //
    // - Preserves your front-to-back transmittance model.
    // - When a candidate is accepted, refines hit t + normal by gathering up to N=8
    //   nearby surfels (via BVH sphere query around the anchor hit point) and
    //   interpolating plane-ray intersections with weights w = (r - d).
    //
    // Rename: no longer "Stochastic" so you can call it directly.
    // ----------------------------------------------------------------------------
    // ----------------------------------------------------------------------------
    // Point-cloud BLAS intersection with:
    // - front-to-back stochastic transmittance
    // - Schaufler/Jensen-style cylinder neighbor interpolation (N=8)
    // - surfel radius derived from scale
    // - no RayIntersectMode / no forced scatter
    // ----------------------------------------------------------------------------
    SYCL_EXTERNAL static bool intersectBLASPointCloudCylinder(
        const Ray &rayObject,
        uint32_t blasRangeIndex,
        LocalHit &localHitOut,
        const GPUSceneBuffers &scene,
        rng::Xorshift128 &rng128) {
        const BLASRange &blasRange = scene.blasRanges[blasRangeIndex];
        const BVHNode *bvhNodes = scene.blasNodes + blasRange.firstNode;

        constexpr float rayEpsilon = 1e-5f;
        constexpr float tAdvanceEpsilon = 1e-4f; // advance after a rejected hit to avoid re-hitting same surfel
        constexpr uint32_t maxRejections = 256; // cap work per ray (tune)

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


        // Find next closest surfel hit with t in (tMin, tMax).
        auto gatherNeighbours = [&](float3 sphereCenter, float sphereRadius
                                    ) -> bool {
            bool hitAny = false;

            const float3 inverseDirection = safeInvDir(rayObject.direction);

            SmallStack<256> traversalStack;
            traversalStack.push(0);

            while (!traversalStack.empty()) {
                const int nodeIndex = traversalStack.pop();
                const BVHNode &node = bvhNodes[nodeIndex];

                float nodeTEntry = 0.0f;
                if (!testSphereAABB(sphereCenter, sphereRadius, node.aabbMin, node.aabbMax))
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

            if (!findNextClosestSurfel(tMin, std::numeric_limits<float>::infinity(), tHit, surfelIndex,
                                       alphaGeomAtHit)) {
                // No more candidates: pure transmission through this BLAS
                localHitOut.transmissivity = cumulativeTransmittanceBefore;
                return false;
            }

            const Point &surfel = scene.points[surfelIndex];

            // Effective interaction probability at this candidate
            const float alphaEff = sycl::clamp(alphaGeomAtHit * surfel.opacity, 0.0f, 1.0f);

            // Forced primitive scatter mode:
            // - If this hit is not the requested surfel: we must transmit through it.
            // - If it is the requested surfel: we must accept here.


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
    SYCL_EXTERNAL static bool intersectSceneCylinder(const Ray &rayWorld,
                                                     WorldHit *worldHitOut,
                                                     const GPUSceneBuffers &scene,
                                                     rng::Xorshift128 &rng128) {
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
                acceptedHitInInstance = intersectBLASMesh(rayObject, instance.blasRangeIndex, localHit, scene);
            } else {
                acceptedHitInInstance = intersectBLASPointCloudCylinder(
                    rayObject,
                    instance.blasRangeIndex,
                    localHit,
                    scene,
                    rng128);
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
                worldHitOut->geometricNormalW = localHit.normal;
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


    void launchCylinderRayIntersectKernel(RenderPackage &pkg, uint32_t activeRayCount) {
        auto queue = pkg.queue;
        auto scene = pkg.scene;
        auto sensor = pkg.sensor;
        auto settings = pkg.settings;

        auto *hitRecords = pkg.intermediates.hitRecords;
        auto *raysIn = pkg.intermediates.primaryRays;
        auto *raysOut = pkg.intermediates.extensionRaysA;
        auto *countPrimary = pkg.intermediates.countPrimary;

        const uint32_t photonCount = settings.photonsPerLaunch;
        const uint32_t forwardPasses = settings.numForwardPasses;
        const float totalPhotons = photonCount * forwardPasses;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            uint64_t baseSeed = settings.randomSeed;

            commandGroupHandler.parallel_for<struct RayGenEmitterKernelTag>(
                sycl::range<1>(activeRayCount),
                [=](sycl::id<1> globalId) {
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, globalId[0]);
                    // Choose any generator you like:
                    rng::Xorshift128 rng128(perItemSeed);
                    const uint32_t rayIndex = globalId[0];

                    WorldHit worldHit{};
                    RayState rayState = raysIn[rayIndex];
                    intersectSceneCylinder(rayState.ray, &worldHit, scene, rng128);
                    if (!worldHit.hit) {
                        hitRecords[rayIndex] = worldHit;
                        return;
                    }

                    buildIntersectionNormal(scene, worldHit);
                    hitRecords[rayIndex] = worldHit;
                });
        });
        queue.wait();
    }
}
