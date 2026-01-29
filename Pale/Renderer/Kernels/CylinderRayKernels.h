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
    template<uint32_t maxCount>
    struct TopCandidates {
        uint32_t count = 0;
        float distances[maxCount]{};
        uint32_t indices[maxCount]{};

        SYCL_EXTERNAL inline void tryInsert(float distance, uint32_t index) {
            // reject invalid distances (negative, zero, NaN)
            if (!(distance > 0.0f)) return;

            // early reject if worse than current worst
            if (count == maxCount && distance >= distances[count - 1]) return;

            if (count < maxCount) {
                distances[count] = distance;
                indices[count] = index;
                count++;
            } else {
                distances[maxCount - 1] = distance;
                indices[maxCount - 1] = index;
            }

            // insertion sort step (bubble up)
            for (int i = int(count) - 1; i > 0; --i) {
                if (distances[i] >= distances[i - 1]) break;

                std::swap(distances[i], distances[i - 1]);
                std::swap(indices[i], indices[i - 1]);
            }
        }

        bool hasCandidates() { return count > 0; }

        SYCL_EXTERNAL inline void clear() { count = 0; }
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
    int testSphereAABB(float3 sphereCenter, float sphereRadius, const BVHNode &node) {
        // Compute squared distance between sphere center and AABB
        float sqDist = sqDistPointAABB(sphereCenter, node.aabbMin, node.aabbMax);
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
    // ----------------------------------------------------------------------------
    SYCL_EXTERNAL static bool intersectBLASPointCloudCylinder(
        const Ray &rayObject,
        uint32_t blasRangeIndex,
        LocalHit &localHitOut,
        const GPUSceneBuffers &scene,
        const Transform &transform,
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


        TopCandidates<16> nearestSurfels;

        // Find next closest surfel hit with t in (tMin, tMax).
        auto gatherNeighbours = [&](float3 sphereCenter, float sphereRadius
        ) -> bool {
            bool hitAny = false;
            nearestSurfels.clear();

            const float3 inverseDirection = safeInvDir(rayObject.direction);

            SmallStack<256> traversalStack;
            traversalStack.push(0);

            while (!traversalStack.empty()) {
                const int nodeIndex = traversalStack.pop();
                const BVHNode &node = bvhNodes[nodeIndex];

                float nodeTEntry = 0.0f;
                if (!testSphereAABB(sphereCenter, sphereRadius, node))
                    continue;

                if (node.triCount == 0) {
                    const int leftIndex = node.leftFirst;
                    const int rightIndex = node.leftFirst + 1;

                    const bool hitLeft = testSphereAABB(sphereCenter, sphereRadius, bvhNodes[leftIndex]);
                    const bool hitRight = testSphereAABB(sphereCenter, sphereRadius, bvhNodes[rightIndex]);

                    if (hitLeft) traversalStack.push(leftIndex);
                    if (hitRight) traversalStack.push(rightIndex);

                    continue;
                }

                // Leaf: test surfels
                for (uint32_t local = 0; local < node.triCount; ++local) {
                    const uint32_t surfelIndex = node.leftFirst + local;
                    const Point &surfel = scene.points[surfelIndex];

                    // Check if the surfel center is inside the sphere.
                    float dist = length(surfel.position - sphereCenter);

                    bool isInside = dist < sphereRadius;

                    if (isInside) {
                        nearestSurfels.tryInsert(dist, surfelIndex);
                    }
                }
            }

            return nearestSurfels.hasCandidates();
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


            // Random mode: accept with probability alphaEff, otherwise transmit and continue
            const float u = rng128.nextFloat();
            if (u < alphaEff) {
                localHitOut.t = tHit;
                localHitOut.primitiveIndex = surfelIndex;
                localHitOut.transmissivity = cumulativeTransmittanceBefore;

                float3 worldHit = toWorldPoint(rayObject.origin + tHit * rayObject.direction, transform);
                localHitOut.worldHit = worldHit;

                float gatherSphereRadius = 0.015f;
                bool hasCandidates = gatherNeighbours(worldHit, gatherSphereRadius);

                if (hasCandidates) {
                    const float r = gatherSphereRadius;
                    const float epsDenom = 1.0e-6f;

                    // Use the world ray that produced worldHit.
                    // If you only have rayObject here, ensure it is world-space.
                    const float3 rayOrigin = rayObject.origin;
                    const float3 rayDirection = normalize(rayObject.direction);

                    float weightedTSum = 0.0f;
                    float weightSum = 0.0f;

                    for (uint32_t i = 0; i < nearestSurfels.count; ++i) {
                        const uint32_t neighborIndex = nearestSurfels.indices[i];
                        const Point &neighborSurfel = scene.points[neighborIndex];

                        const float3 p_i = neighborSurfel.position; // world
                        const float3 n_i = normalize(cross(neighborSurfel.tanU, neighborSurfel.tanV));


                        const float denom = dot(rayDirection, n_i);
                        if (sycl::fabs(denom) <= epsDenom) {
                            continue; // nearly parallel -> unstable t_i
                        }

                        // Ray-plane intersection with plane through p_i with normal n_i:
                        // t_i = <(p_i - o), n_i> / <d, n_i>
                        const float t_i = dot(p_i - rayOrigin, n_i) / denom;
                        if (!(t_i > 0.0f)) {
                            continue;
                        }

                        const float3 q_i = rayOrigin + t_i * rayDirection;

                        // Since q_i is on the plane, (p_i - q_i) lies in-plane, so this is the in-plane distance d_i.
                        const float d_i = length(p_i - q_i);
                        if (d_i >= r) {
                            continue; // outside disk radius -> no contribution
                        }

                        // Weight: w_i = (r - d_i)
                        const float w_i = (r - d_i);
                        weightedTSum += w_i * t_i;
                        weightSum += w_i;
                    }

                    if (weightSum > 0.0f) {
                        const float refinedT = weightedTSum / weightSum;
                        localHitOut.worldHit = rayOrigin + refinedT * rayDirection;

                        // If you also want the paper's normal interpolation (recommended):
                        // n = normalize(Σ w_i n_i)
                        // (not shown since you asked for position only)
                    }
                }

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
                acceptedHitInInstance = intersectBLASMesh(rayObject, instance.blasRangeIndex, localHit, scene,
                                                          transform);
            } else {
                acceptedHitInInstance = intersectBLASPointCloudCylinder(
                    rayObject,
                    instance.blasRangeIndex,
                    localHit,
                    scene,
                    transform,
                    rng128);
            }


            if (acceptedHitInInstance) {
                // Convert to world, test depth

                const float tWorld = dot(localHit.worldHit - rayWorld.origin, rayWorld.direction);
                // assumes normalized direction
                if (tWorld < 0.0f || tWorld >= bestWorldTHit) continue;

                bestWorldTHit = tWorld;
                foundAnySurfaceHit = true;

                worldHitOut->hit = true;
                worldHitOut->t = tWorld;
                worldHitOut->hitPositionW = localHit.worldHit;
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

    void launchCylinderContributionKernel(RenderPackage &pkg, uint32_t activeRayCount, uint32_t cameraIndex) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto &photonMap = pkg.intermediates.map; // DeviceSurfacePhotonMapGrid
        uint64_t baseSeed = pkg.settings.randomSeed * (cameraIndex + 5);


        // Host-side (before launching kernel)
        SensorGPU sensor = pkg.sensor[cameraIndex];

        auto &intermediates = pkg.intermediates;
        auto *hitRecords = pkg.intermediates.hitRecords;
        auto *raysIn = pkg.intermediates.primaryRays;
        auto *raysOut = pkg.intermediates.extensionRaysA;
        auto *countExtensionOut = pkg.intermediates.countExtensionOut;

        queue.submit([&](sycl::handler &cgh) {
            uint64_t baseSeed = settings.randomSeed * (static_cast<uint64_t>(cameraIndex) + 5ull);

            cgh.parallel_for<class ShadeKernelTag>(
                sycl::range<1>(activeRayCount),
                // ReSharper disable once CppDFAUnusedValue
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);

                    const WorldHit worldHit = hitRecords[rayIndex];
                    const RayState rayState = raysIn[rayIndex];
                    if (!worldHit.hit)
                        return;

                    const InstanceRecord &instance = scene.instances[worldHit.instanceIndex];

                    const float3 &surfacePointWorld = worldHit.hitPositionW;
                    const float3 &surfaceNormalWorld = worldHit.geometricNormalW; // ensure normalized
                    const float3 &incomingDirectionWorld = -rayState.ray.direction; // direction arriving at surface

                    // Project to pixel and get omega_c (surface -> camera) and distance
                    uint32_t pixelIndex = 0u;
                    float3 omegaSurfaceToCamera;
                    float distanceToCamera = 0.0f;

                    if (!projectToPixelFromFovY(sensor, surfacePointWorld, pixelIndex, omegaSurfaceToCamera,
                                                distanceToCamera))
                        return;

                    // Backface / cosine term at surface
                    const float signedCosineToCamera = dot(surfaceNormalWorld, omegaSurfaceToCamera);
                    const float cosineAbsToCamera = sycl::fabs(signedCosineToCamera);
                    if (cosineAbsToCamera <= 0.0f)
                        return;

                    // Visibility: shadow ray from surface to camera
                    const float3 contributionRayOrigin = surfacePointWorld + surfaceNormalWorld * 1e-6f;
                    const float3 contributionDirection = omegaSurfaceToCamera;
                    const float shadowRayMaxT = distanceToCamera - 2e-4f;
                    Ray ray{contributionRayOrigin, contributionDirection};

                    WorldHit visibilityCheck{};
                    intersectSceneCylinder(ray, &visibilityCheck, scene, rng128);

                    if (visibilityCheck.hit && visibilityCheck.t <= shadowRayMaxT) {
                        return;
                    }
                    // Lambertian BSDF: f = albedo / pi

                    const float tauDiffuse = 0.0f; // diffuse transmission
                    float3 rho{0.0f};
                    switch (instance.geometryType) {
                        case GeometryType::Mesh:
                            rho = scene.materials[instance.materialIndex].baseColor;
                            break;
                        case GeometryType::PointCloud:
                            rho = scene.points[worldHit.primitiveIndex].albedo;
                            break;
                        default:
                            break;
                    }

                    const float3 bsdfValue = rho * M_1_PIf;
                    // Geometry term from pinhole importance (1/r^2 and cosine at surface)
                    const float inverseDistanceSquared = 1.0f / (distanceToCamera * distanceToCamera);

                    const float width = float(sensor.width);
                    const float height = float(sensor.height);

                    const float fovYRad = glm::radians(sensor.camera.fovy);
                    const float tanHalfFovY = sycl::tan(0.5f * fovYRad);
                    const float tanHalfFovX = tanHalfFovY * (width / height);

                    // film plane at z=1 has size: 2*tanHalfFovX by 2*tanHalfFovY
                    const float filmWidth = 2.0f * tanHalfFovX;
                    const float filmHeight = 2.0f * tanHalfFovY;

                    const float pixelArea = (filmWidth / width) * (filmHeight / height);
                    const float invPixelArea = 1.0f / pixelArea;
                    // Contribution (delta sensor, pixel binning)
                    const float3 contribution =
                            rayState.pathThroughput *
                            (bsdfValue + float3{tauDiffuse}) * visibilityCheck.transmissivity *
                            (cosineAbsToCamera * inverseDistanceSquared) * invPixelArea;

                    // Atomic accumulate to framebuffer
                    atomicAddFloat3ToImage(&sensor.framebuffer[pixelIndex], contribution);
                }
            );
        });
    }
}
