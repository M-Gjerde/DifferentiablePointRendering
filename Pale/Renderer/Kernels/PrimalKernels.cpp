//
// Created by magnus on 9/12/25.
//

#include "PrimalKernels.h"

#include <cmath>

#include "KernelHelpers.h"
#include "IntersectionKernels.h"

namespace Pale {
    void launchRayGenEmitterKernel(RenderPackage &pkg, uint32_t forwardPass) {
        auto queue = pkg.queue;
        auto scene = pkg.scene;
        auto settings = pkg.settings;

        auto *raysIn = pkg.intermediates.primaryRays;
        auto *countPrimary = pkg.intermediates.countPrimary;

        const uint32_t emittedCount = settings.photonsPerLaunch * settings.numForwardPasses;
        const float invEmittedCount = 1.0f / float(emittedCount);

        // One invariant seed for the whole render (stable noise across reruns)
        const uint64_t renderSeed = pkg.random.seed;


        queue.submit([&](sycl::handler &commandGroupHandler) {
            const uint64_t forwardPassIndex = uint64_t(forwardPass);

            commandGroupHandler.parallel_for<struct RayGenEmitterKernelTag>(
                sycl::range<1>(settings.photonsPerLaunch),
                [=](sycl::id<1> globalId) {
                    const uint64_t photonIndex = uint64_t(globalId[0]);
                    const uint64_t pathId = forwardPassIndex * uint64_t(settings.photonsPerLaunch) + photonIndex;
                    const uint64_t seed =
                            rng::makeSeed(renderSeed, pathId, 0u, rng::kStreamRayGen, 0u);
                    rng::Xorshift128 rng128(seed);

                    if (scene.lightCount == 0) {
                        return;
                    }

                    AreaLightSample ls = sampleMeshAreaLight(scene, rng128);
                    if (!ls.valid) {
                        return;
                    }

                    const float3 initialThroughput = ls.flux * scene.lightCount * invEmittedCount;

                    RayState ray{};
                    ray.ray.origin = ls.positionW + ls.normalW * 1e-5f;
                    ray.ray.direction = ls.direction;
                    ray.ray.normal = ls.normalW;
                    ray.pathThroughput = initialThroughput;
                    ray.bounceIndex = 0;
                    ray.traversalIndex = 0;
                    ray.lightIndex = ls.lightIndex;
                    ray.pathId = pathId;

                    auto counter = sycl::atomic_ref<uint32_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(*countPrimary);

                    const uint32_t slot = counter.fetch_add(1u);
                    raysIn[slot] = ray;
                }
            );
        });

        queue.wait();
    }


    void launchIntersectKernel(RenderPackage &pkg, uint32_t activeRayCount) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto &intermediates = pkg.intermediates;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            const uint64_t renderSeed = pkg.random.seed;

            commandGroupHandler.parallel_for<class launchIntersectKernel>(
                sycl::range<1>(activeRayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    RayState currentRayState = intermediates.primaryRays[rayIndex];

                    // Guard against pathological transparent stacks or self-intersection loops.
                    constexpr uint32_t maxInlineNullTraversals = 32;

                    for (uint32_t inlineTraversalIndex = 0; inlineTraversalIndex < maxInlineNullTraversals;
                         ++inlineTraversalIndex) {
                        const uint64_t stepSeed =
                                rng::makeSeed(renderSeed,
                                              currentRayState.pathId,
                                              currentRayState.traversalIndex,
                                              rng::kStreamTraversal,
                                              107u);
                        rng::Xorshift128 stepRng(stepSeed);

                        WorldHit worldHit{};
                        intersectScene(currentRayState.ray, &worldHit, scene, SurfelIntersectMode::FirstHit);

                        if (!worldHit.hit) {
                            return;
                        }

                        bool isIndirect = currentRayState.bounceIndex > 0;


                        buildIntersectionNormal(scene, worldHit);
                        const InstanceRecord &instance = scene.instances[worldHit.instanceIndex];
                        // ---------------------------------------------------------------------
                        // Mesh: this is a real scattering event, so it finishes this kernel call.
                        // ---------------------------------------------------------------------
                        if (instance.geometryType == GeometryType::Mesh) {
                            bool isBackfaceHit = dot(currentRayState.ray.direction, worldHit.geometricNormalW) > 0.0f;
                            if (isBackfaceHit) {
                                worldHit.geometricNormalW *= -1.0f;
                            }

                            if (settings.integratorKind == IntegratorKind::lightTracing) {
                                HitInfoContribution contribution{};
                                contribution.geometricNormalW = worldHit.geometricNormalW;
                                contribution.hitPositionW = worldHit.hitPositionW;
                                contribution.instanceIndex = worldHit.instanceIndex;
                                contribution.pathThroughput = currentRayState.pathThroughput;
                                contribution.type = instance.geometryType;
                                contribution.primitiveIndex = worldHit.primitiveIndex;
                                appendContributionAtomic(
                                    intermediates.countContributions,
                                    intermediates.hitContribution,
                                    intermediates.maxHitContributionCount,
                                    contribution);
                            }

                            if (settings.integratorKind == IntegratorKind::photonMapping && isIndirect) {
                                depositPhotonSurface(
                                    worldHit,
                                    currentRayState.ray.direction,
                                    worldHit.geometricNormalW,
                                    currentRayState.pathThroughput,
                                    intermediates.map);
                            }

                            float3 sampledOutgoingDirectionW = currentRayState.ray.direction;
                            float sampledPdf = 0.0f;
                            sampleCosineHemisphere(
                                stepRng,
                                worldHit.geometricNormalW,
                                sampledOutgoingDirectionW,
                                sampledPdf);

                            const GPUMaterial material = scene.materials[instance.materialIndex];
                            const float3 throughputMultiplier = material.baseColor;

                            RayState nextRayState{};
                            nextRayState.ray.origin = worldHit.hitPositionW + (worldHit.geometricNormalW * 1e-6f);
                            nextRayState.ray.direction = sampledOutgoingDirectionW;
                            nextRayState.ray.normal = worldHit.geometricNormalW;
                            nextRayState.bounceIndex = currentRayState.bounceIndex + 1; // real bounce
                            nextRayState.traversalIndex = currentRayState.traversalIndex + 1;
                            nextRayState.pixelIndex = currentRayState.pixelIndex;
                            nextRayState.pathThroughput = currentRayState.pathThroughput * throughputMultiplier;
                            nextRayState.pathId = currentRayState.pathId;
                            nextRayState.lightIndex = currentRayState.lightIndex;

                            if (!applyRussianRoulette(
                                stepRng,
                                nextRayState.bounceIndex,
                                nextRayState.pathThroughput,
                                settings.russianRouletteStart)) {
                                return;
                            }

                            auto extensionCounter = sycl::atomic_ref<uint32_t,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>(
                                *intermediates.countExtensionOut);

                            const uint32_t outIndex = extensionCounter.fetch_add(1u);
                            if (outIndex < intermediates.maxRayQueueCapacity) {
                                intermediates.extensionRaysA[outIndex] = nextRayState;
                            }
                            return;
                        }

                        // ---------------------------------------------------------------------
                        // Point cloud: null events are traversals only, reflect ends this call.
                        // ---------------------------------------------------------------------
                        if (instance.geometryType == GeometryType::PointCloud) {
                            float rayAdvanceEpsilon = 1e-5f;
                            const Point &surfel = scene.points[worldHit.primitiveIndex];
                            const float randomNumber = stepRng.nextFloat();

                            // Null event:
                            //   - does NOT increment bounceIndex
                            //   - does NOT enqueue a ray yet
                            //   - continues traversal immediately in this same kernel call
                            if (randomNumber < settings.sampling.qNull) {
                                const float effectiveOpacity = worldHit.alphaGeom * surfel.opacity;
                                const float attenuation = 1.0f - effectiveOpacity;
                                const float weight = attenuation / settings.sampling.qNull;

                                currentRayState.ray.origin = worldHit.hitPositionW + (currentRayState.ray.direction *
                                                                 rayAdvanceEpsilon);
                                currentRayState.traversalIndex = currentRayState.traversalIndex + 1;
                                currentRayState.pathThroughput = currentRayState.pathThroughput * weight;
                                continue;
                            }

                            // Reflect event:
                            //   - this is the first real interaction after any number of null events
                            //   - increment bounceIndex
                            //   - enqueue and return
                            if (randomNumber < settings.sampling.qNull + settings.sampling.qReflect) {
                                const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));
                                const float signedCosineIncident =
                                        dot(canonicalNormalW, -currentRayState.ray.direction);
                                const int sideSign = signNonZero(signedCosineIncident);
                                const float3 orientedNormal = static_cast<float>(sideSign) * canonicalNormalW;

                                if (settings.integratorKind == IntegratorKind::photonMapping && isIndirect) {
                                    depositPhotonSurface(
                                        worldHit,
                                        currentRayState.ray.direction,
                                        orientedNormal,
                                        currentRayState.pathThroughput / settings.sampling.qReflect,
                                        intermediates.map);
                                }

                                float3 sampledOutgoingDirectionW = currentRayState.ray.direction;
                                float sampledPdf = 0.0f;
                                sampleUniformHemisphereAroundNormal(
                                    stepRng,
                                    orientedNormal,
                                    sampledOutgoingDirectionW,
                                    sampledPdf);

                                const float cosineTheta =
                                        sycl::fmax(0.0f, dot(sampledOutgoingDirectionW, orientedNormal));
                                const float3 scatteringFunction =
                                        surfel.alpha_r * surfel.albedo * M_1_PIf;
                                const float effectiveOpacity = worldHit.alphaGeom * surfel.opacity;
                                const float3 reflectWeight =
                                ((effectiveOpacity / settings.sampling.qReflect) *
                                 scatteringFunction *
                                 cosineTheta / sampledPdf);

                                const float3 nextPathThroughput =
                                        currentRayState.pathThroughput * reflectWeight;

                                if (settings.integratorKind == IntegratorKind::lightTracing) {
                                    HitInfoContribution contribution{};
                                    contribution.hitPositionW = worldHit.hitPositionW;
                                    contribution.geometricNormalW = orientedNormal;
                                    contribution.instanceIndex = worldHit.instanceIndex;
                                    contribution.pathThroughput = nextPathThroughput;
                                    contribution.type = instance.geometryType;
                                    contribution.primitiveIndex = worldHit.primitiveIndex;
                                    contribution.eventType = EventType::Reflect;
                                    appendContributionAtomic(
                                        intermediates.countContributions,
                                        intermediates.hitContribution,
                                        intermediates.maxHitContributionCount,
                                        contribution);
                                }

                                RayState nextRayState{};
                                nextRayState.ray.origin = worldHit.hitPositionW + (orientedNormal * 1e-5f);
                                nextRayState.ray.direction = sampledOutgoingDirectionW;
                                nextRayState.ray.normal = orientedNormal;
                                nextRayState.bounceIndex = currentRayState.bounceIndex + 1; // real bounce
                                nextRayState.traversalIndex = currentRayState.traversalIndex + 1;
                                nextRayState.pixelIndex = currentRayState.pixelIndex;
                                nextRayState.pathThroughput = nextPathThroughput;
                                nextRayState.pathId = currentRayState.pathId;
                                nextRayState.lightIndex = currentRayState.lightIndex;

                                if (!applyRussianRoulette(
                                    stepRng,
                                    nextRayState.bounceIndex,
                                    nextRayState.pathThroughput,
                                    settings.russianRouletteStart)) {
                                    return;
                                }

                                auto extensionCounter = sycl::atomic_ref<uint32_t,
                                    sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(
                                    *intermediates.countExtensionOut);

                                const uint32_t outIndex = extensionCounter.fetch_add(1u);
                                if (outIndex < intermediates.maxRayQueueCapacity) {
                                    intermediates.extensionRaysA[outIndex] = nextRayState;
                                }
                                return;
                            }

                            // Transmission / absorption currently simplified out.
                            // Leave empty / terminate for now.
                            return;
                        }

                        return;
                    }


                    // Traversal guard hit: terminate the ray.
                    return;
                });
        });

        queue.wait();
    }


    /*
    void launchContributionKernel(RenderPackage& pkg, uint32_t contributionCount, uint32_t cameraIndex) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& settings = pkg.settings;
        // Host-side (before launching kernel)
        SensorGPU sensor = pkg.sensors[cameraIndex];
        auto* hitRecords = pkg.intermediates.hitContribution;

        queue.submit([&](sycl::handler& cgh) {
            uint64_t renderSeed = pkg.random.seed;

            cgh.parallel_for<class launchContributionKernel>(
                sycl::range<1>(contributionCount),
                // ReSharper disable once CppDFAUnusedValue
                [=](sycl::id<1> globalId) {
                    const uint32_t contributionIndex = globalId[0];
                    const uint64_t seed =
                        rng::makeSeed(renderSeed, contributionIndex, cameraIndex, rng::kStreamDeposit, 0u);

                    rng::Xorshift128 rng128(seed);
                    const HitInfoContribution& contribution = hitRecords[contributionIndex];
                    const InstanceRecord& instance = scene.instances[contribution.instanceIndex];
                    auto& geometryType = instance.geometryType;
                    const float3& surfacePointWorld = contribution.hitPositionW;
                    const float3& surfaceNormalWorld = contribution.geometricNormalW; // ensure normalized
                    // Project to pixel and get omega_c (surface -> camera) and distance
                    uint32_t pixelIndex = 0u;
                    float3 omegaSurfaceToCamera;
                    float distanceToCamera = 0.0f;
                    bool debug = false;
                    int pixelX = 0;
                    int pixelY = 0;
                    if (!projectToPixelFromPinhole(sensor, surfacePointWorld, pixelIndex, omegaSurfaceToCamera,
                                                   distanceToCamera, pixelX, pixelY, debug))
                        return;
                    // Backface / cosine term at surface
                    const float cosineToCamera = dot(surfaceNormalWorld, omegaSurfaceToCamera);
                    if (cosineToCamera <= 0.0f)
                        return;
                    float cosThetaCamera = dot(sensor.camera.forward, -omegaSurfaceToCamera);
                    if (cosThetaCamera <= 0.0f)
                        return;
                    // Visibility: shadow ray from surface to camera
                    const float shadowRayMaxT = distanceToCamera - 1e-4f;
                    Ray ray{surfacePointWorld, omegaSurfaceToCamera};
                    WorldHit visibilityCheck{};
                    intersectScene(ray, &visibilityCheck, scene, rng128, SurfelIntersectMode::Transmit);
                    if (visibilityCheck.t <= shadowRayMaxT && visibilityCheck.hit) {
                        return;
                    }
                    // DO measurement ray BSDF selection.
                    float3 rho{0.0f};
                    if (geometryType == GeometryType::Mesh) {
                        const GPUMaterial material = scene.materials[instance.materialIndex];
                        // If we hit instance was a mesh do ordinary BRDF stuff.
                        const float3 lambertBrdf = material.baseColor;
                        rho = lambertBrdf * M_1_PIf;
                    }
                    else if (geometryType == GeometryType::PointCloud) {
                        const Point point = scene.points[contribution.primitiveIndex];
                        // Mixture BSDF update simplifies analytically
                        if (contribution.eventType == EventType::Reflect) {
                            rho = (point.albedo) * M_1_PIf;
                        }
                        else if (contribution.eventType == EventType::Transmit) {
                            rho = (point.albedo) * M_1_PIf;
                        }
                    }
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
                    // Convert film pixel area to solid angle of that pixel (pinhole, film plane at z=1):
                    // dω = cos^3(θc) dA  =>  1/Ωpixel = 1/(Apixel cos^3)
                    const float invPixelSolidAngle =
                        invPixelArea / (cosThetaCamera * cosThetaCamera * cosThetaCamera);
                    const float3 flux =
                        contribution.pathThroughput * rho * visibilityCheck.transmissivity *
                        (cosineToCamera * inverseDistanceSquared) * invPixelSolidAngle;
                    // Atomic accumulate to framebuffer
                    atomicAddFloat3ToImage(&sensor.framebuffer[pixelIndex], flux);
                }
            );
        });
    }


    // Check if samples from light sources connect directly to camera
    void launchContributionEmitterVisibleKernel(RenderPackage& pkg, uint32_t activeRayCount, uint32_t cameraIndex) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& settings = pkg.settings;
        // Host-side (before launching kernel)
        SensorGPU sensor = pkg.sensors[cameraIndex];
        const uint32_t photonCount = settings.photonsPerLaunch;
        queue.submit([&](sycl::handler& commandGroupHandler) {
            uint64_t renderSeed = pkg.random.seed;
            float invPhotonCount = 1.f / photonCount; // 1/N

            commandGroupHandler.parallel_for<class ShadeKernelTag>(
                sycl::range<1>(photonCount),
                [=](sycl::id<1> globalId) {
                    const uint64_t seed =
                        rng::makeSeed(renderSeed, globalId[0], cameraIndex, rng::kStreamDeposit, 0u);

                    rng::Xorshift128 rng128(seed);

                    if (scene.lightCount == 0) return;
                    AreaLightSample ls = sampleMeshAreaLight(scene, rng128);
                    if (!ls.valid) return;
                    // Storing radiance as watt simplifies this line:
                    const float3 initialThroughput = ls.flux * scene.lightCount * invPhotonCount;

                    const float3& surfacePointWorld = ls.positionW;
                    const float3& surfaceNormalWorld = ls.normalW; // ensure normalized
                    // Project to pixel and get omega_c (surface -> camera) and distance
                    uint32_t pixelIndex = 0u;
                    float3 omegaSurfaceToCamera;
                    float distanceToCamera = 0.0f;
                    bool debug = false;
                    int pixelX = 0;
                    int pixelY = 0;
                    if (!projectToPixelFromPinhole(sensor, surfacePointWorld, pixelIndex, omegaSurfaceToCamera,
                                                   distanceToCamera, pixelX, pixelY, debug))
                        return;

                    // Backface / cosine term at surface
                    const float signedCosineToCamera = dot(surfaceNormalWorld, omegaSurfaceToCamera);
                    if (signedCosineToCamera <= 0.0f)
                        return;

                    float cosThetaCamera = dot(sensor.camera.forward, -omegaSurfaceToCamera);
                    if (cosThetaCamera <= 0.0f)
                        return;

                    // Visibility: shadow ray from surface to camera
                    const float shadowRayMaxT = distanceToCamera - 1e-4f;
                    Ray ray{surfacePointWorld, omegaSurfaceToCamera};
                    WorldHit visibilityCheck{};
                    intersectScene(ray, &visibilityCheck, scene, rng128, SurfelIntersectMode::Transmit);

                    if (visibilityCheck.t <= shadowRayMaxT && visibilityCheck.hit) {
                        return;
                    }
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

                    // Convert film pixel area to solid angle of that pixel (pinhole, film plane at z=1):
                    // dω = cos^3(θc) dA  =>  1/Ωpixel = 1/(Apixel cos^3)
                    const float cosThetaCameraClamped = sycl::fmax(cosThetaCamera, 1e-6f);
                    const float invPixelSolidAngle =
                        invPixelArea / (cosThetaCameraClamped * cosThetaCameraClamped * cosThetaCameraClamped);

                    const float3 flux =
                        initialThroughput * visibilityCheck.transmissivity *
                        (signedCosineToCamera * inverseDistanceSquared) * invPixelSolidAngle;
                    // Atomic accumulate to framebuffer
                    atomicAddFloat3ToImage(&sensor.framebuffer[pixelIndex], flux);
                }
            );
        });
    }

*/


    void launchCameraGatherKernelOld(RenderPackage &pkg, uint32_t cameraIndex) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto &photonMap = pkg.intermediates.map;

        SensorGPU sensor = pkg.sensors[cameraIndex];

        const std::uint32_t imageWidth = sensor.camera.width;
        const std::uint32_t imageHeight = sensor.camera.height;
        const std::uint32_t pixelCount = imageWidth * imageHeight;
        pkg.queue.fill(sensor.framebuffer, float4{0}, pixelCount).wait();

        if (settings.useDepthDistortion) {
            pkg.queue.fill(sensor.depthDistortionBuffer, 0.0f, pixelCount).wait();
            pkg.queue.fill(sensor.depthDistortionAdjointBuffer, 0.0f, pixelCount).wait();
        }

        queue.submit([&](sycl::handler &cgh) {
            const uint64_t renderSeed = pkg.random.seed;

            cgh.parallel_for<class CameraGatherKernel>(
                sycl::range<1>(pixelCount),
                [=](sycl::id<1> tid) {
                    const std::uint32_t pixelIndex = tid[0];
                    const std::uint32_t pixelX = pixelIndex % imageWidth;
                    const std::uint32_t pixelY = pixelIndex / imageWidth;

                    const uint64_t directionSeed =
                            rng::makeSeed(renderSeed, pixelIndex, cameraIndex, rng::kStreamGather, 0u);
                    rng::Xorshift128 rng(directionSeed);

                    float3 accumulatedRadianceRGB(0.0f);

                    const float jitterX = rng.nextFloat() - 0.5f;
                    const float jitterY = rng.nextFloat() - 0.5f;

                    Ray primaryRay = makePrimaryRayFromPixelJitteredFov(
                        sensor.camera,
                        static_cast<float>(pixelX),
                        static_cast<float>(pixelY),
                        0.5f,
                        0.5f);

                    const float cameraCosine = dot(sensor.camera.forward, primaryRay.direction);

                    float transmittance = 1.0f;
                    const uint32_t maxTraversalRays = 32u;

                    float distortion = 0.f;
                    float prefixW = 0.f;
                    float prefixWZ = 0.f;

                    const float3 rayOrigin0 = primaryRay.origin;
                    const float3 rayDir0 = primaryRay.direction;

                    for (uint32_t traversalIndex = 0u;
                         traversalIndex < maxTraversalRays;
                         ++traversalIndex) {
                        WorldHit worldHit{};
                        intersectScene(
                            primaryRay,
                            &worldHit,
                            scene,
                            SurfelIntersectMode::FirstHit);

                        if (!worldHit.hit) {
                            break;
                        }

                        buildIntersectionNormal(scene, worldHit);
                        const auto &instance = scene.instances[worldHit.instanceIndex];

                        // -------------------------------------------------------------
                        // Visible point-cloud layer
                        // -------------------------------------------------------------
                        if (instance.geometryType == GeometryType::PointCloud) {
                            const Point &surfel = scene.points[worldHit.primitiveIndex];

                            float3 normalW = normalize(cross(surfel.tanU, surfel.tanV));
                            const bool hitBackside =
                                    dot(normalW, -primaryRay.direction) < 0.0f;

                            if (hitBackside) {
                                normalW = -normalW;
                            }

                            const float alphaEff = surfel.opacity * worldHit.alphaGeom;

                            const float3 indirectIrradiance = gatherDiffuseIrradianceAtPoint(
                                worldHit.hitPositionW,
                                normalW,
                                photonMap);

                            const float3 indirectRadiance =
                                    indirectIrradiance *
                                    (surfel.alpha_r * surfel.albedo * M_1_PIf) *
                                    alphaEff;


                            float surfelArea = M_PIf * length(normalW);
                            float3 emittedRadiance =
                                    surfel.albedo * (surfel.flux / (M_PIf * surfelArea));

                            if (surfel.flux > 0.0f && hitBackside) {
                                emittedRadiance = float3(0.0f);
                            }

                            const float3 directRadiance =
                                    estimateDirectLightAtDiffuseSurface(
                                        scene,
                                        worldHit.hitPositionW,
                                        normalW,
                                        surfel.alpha_r * surfel.albedo,
                                        settings.numShadowRays,
                                        rng) * alphaEff;

                            const float3 outgoingRadiance =
                                    emittedRadiance + indirectRadiance + directRadiance;
                            accumulatedRadianceRGB += transmittance * outgoingRadiance;


                            // Depth distortion
                            if (settings.useDepthDistortion) {
                                float ai = surfel.opacity * worldHit.alphaGeom; //a_i
                                float wi = transmittance * ai; // ω_i
                                float zi = dot(worldHit.hitPositionW - sensor.camera.pos, sensor.camera.forward);
                                distortion += 2.0f * wi * (zi * prefixW - prefixWZ);
                                prefixW += wi;
                                prefixWZ += wi * zi;
                            }

                            // update
                            transmittance *= (1.0f - alphaEff);
                            primaryRay.origin =
                                    worldHit.hitPositionW + primaryRay.direction * 1e-8f;
                            continue;
                        }

                        // -------------------------------------------------------------
                        // Terminal mesh hit
                        // -------------------------------------------------------------
                        if (instance.geometryType == GeometryType::Mesh) {
                            const GPUMaterial &material =
                                    scene.materials[instance.materialIndex];

                            if (material.isEmissive()) {
                                const float3 emittedRadiance =
                                        material.power * material.baseColor;

                                accumulatedRadianceRGB +=
                                        transmittance * min(emittedRadiance, 1.0f);
                            } else {
                                const bool isBackfaceHit =
                                        dot(primaryRay.direction, worldHit.geometricNormalW) > 0.0f;

                                const float3 normalW = isBackfaceHit
                                                           ? -worldHit.geometricNormalW
                                                           : worldHit.geometricNormalW;

                                const float3 indirectIrradiance = gatherDiffuseIrradianceAtPoint(
                                    worldHit.hitPositionW,
                                    normalW,
                                    photonMap);

                                const float3 indirectRadiance =
                                        (material.baseColor * M_1_PIf) * indirectIrradiance;

                                const float3 directRadiance =
                                        estimateDirectLightAtDiffuseSurface(
                                            scene,
                                            worldHit.hitPositionW,
                                            normalW,
                                            material.baseColor,
                                            settings.numShadowRays,
                                            rng);

                                const float3 outgoingRadiance =
                                        indirectRadiance + directRadiance;

                                accumulatedRadianceRGB += transmittance * outgoingRadiance;
                            }

                            transmittance = 0.0f;
                            break;
                        }
                    }

                    const std::uint32_t framebufferIndex =
                            pixelY * imageWidth + pixelX;

                    accumulatedRadianceRGB *= cameraCosine;

                    const float4 currentValue(
                        accumulatedRadianceRGB.x(),
                        accumulatedRadianceRGB.y(),
                        accumulatedRadianceRGB.z(),
                        1.0f);


                    if (settings.useDepthDistortion) {
                        sensor.depthDistortionBuffer[pixelIndex] = distortion;
                    }

                    sensor.framebuffer[framebufferIndex] += currentValue;
                });
        });

        queue.wait();
    }

    void launchCameraGatherKernel(RenderPackage &pkg, uint32_t cameraIndex) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto &photonMap = pkg.intermediates.map;

        SensorGPU sensor = pkg.sensors[cameraIndex];

        const std::uint32_t imageWidth = sensor.camera.width;
        const std::uint32_t imageHeight = sensor.camera.height;
        const std::uint32_t pixelCount = imageWidth * imageHeight;

        queue.fill(sensor.framebuffer, float4{0.0f, 0.0f, 0.0f, 0.0f}, pixelCount).wait();

        if (settings.useNormalConsistency) {
            queue.fill(sensor.medianDepthBuffer, 0.0f, pixelCount).wait();
            queue.fill(sensor.medianWorldPositionBuffer, float4{0.0f, 0.0f, 0.0f, 0.0f}, pixelCount).wait();
            queue.fill(sensor.visibleNormalBuffer, float4{0.0f, 0.0f, 0.0f, 0.0f}, pixelCount).wait();
            queue.fill(sensor.normalFromDepthBuffer, float4{0.0f, 0.0f, 0.0f, 0.0f}, pixelCount).wait();
        }

        if (settings.useDepthDistortion) {
            queue.fill(sensor.depthDistortionBuffer, 0.0f, pixelCount).wait();
            queue.fill(sensor.depthDistortionAdjointBuffer, 0.0f, pixelCount).wait();
        }

        // -------------------------------------------------------------------------
        // Pass 1:
        //   - RGB gather
        //   - median depth
        //   - median world position
        //   - visible normal at median surface
        // -------------------------------------------------------------------------
        queue.submit([&](sycl::handler &cgh) {
            const uint64_t renderSeed = pkg.random.seed;

            cgh.parallel_for<class CameraGatherKernel>(
                sycl::range<1>(pixelCount),
                [=](sycl::id<1> tid) {
                    const std::uint32_t pixelIndex = tid[0];
                    const std::uint32_t pixelX = pixelIndex % imageWidth;
                    const std::uint32_t pixelY = pixelIndex / imageWidth;

                    const uint64_t directionSeed =
                            rng::makeSeed(renderSeed, pixelIndex, cameraIndex, rng::kStreamGather, 0u);
                    rng::Xorshift128 rng(directionSeed);

                    float3 accumulatedRadianceRGB(0.0f, 0.0f, 0.0f);

                    // Center-of-pixel sample in your jitter convention
                    Ray primaryRay = makePrimaryRayFromPixelJitteredFov(
                        sensor.camera,
                        static_cast<float>(pixelX),
                        static_cast<float>(pixelY),
                        0.0f,
                        0.0f);

                    const float cameraCosine = dot(sensor.camera.forward, primaryRay.direction);

                    float transmittance = 1.0f;
                    const uint32_t maxTraversalRays = 32u;

                    // Depth distortion accumulation
                    float distortion = 0.0f;
                    float prefixW = 0.0f;
                    float prefixWZ = 0.0f;

                    // Median-depth tracking
                    float accumulatedCompositeWeight = 0.0f;
                    bool medianFound = false;
                    float medianDepth = 0.0f;
                    float3 medianWorldPosition(0.0f, 0.0f, 0.0f);
                    float3 medianNormalW(0.0f, 0.0f, 0.0f);

                    for (uint32_t traversalIndex = 0u;
                         traversalIndex < maxTraversalRays;
                         ++traversalIndex) {
                        WorldHit worldHit{};
                        intersectScene(
                            primaryRay,
                            &worldHit,
                            scene,
                            SurfelIntersectMode::FirstHit);

                        if (!worldHit.hit) {
                            break;
                        }

                        buildIntersectionNormal(scene, worldHit);
                        const auto &instance = scene.instances[worldHit.instanceIndex];

                        // -------------------------------------------------------------
                        // Visible point-cloud layer
                        // -------------------------------------------------------------
                        if (instance.geometryType == GeometryType::PointCloud) {
                            const Point &surfel = scene.points[worldHit.primitiveIndex];

                            float3 normalW = normalize(cross(surfel.tanU, surfel.tanV));
                            const bool hitBackside =
                                    dot(normalW, -primaryRay.direction) < 0.0f;

                            if (hitBackside) {
                                normalW = -normalW;
                            }

                            const float alphaEff = surfel.opacity * worldHit.alphaGeom;

                            const float3 indirectIrradiance = gatherDiffuseIrradianceAtPoint(
                                worldHit.hitPositionW,
                                normalW,
                                photonMap);

                            const float3 indirectRadiance =
                                    indirectIrradiance *
                                    (surfel.alpha_r * surfel.albedo * M_1_PIf) *
                                    alphaEff;

                            float surfelArea = M_PIf * length(normalW);
                            float3 emittedRadiance =
                                    surfel.albedo * (surfel.flux / (M_PIf * surfelArea));

                            if (surfel.flux > 0.0f && hitBackside) {
                                emittedRadiance = float3(0.0f, 0.0f, 0.0f);
                            }

                            const float3 directRadiance =
                                    estimateDirectLightAtDiffuseSurface(
                                        scene,
                                        worldHit.hitPositionW,
                                        normalW,
                                        surfel.alpha_r * surfel.albedo,
                                        settings.numShadowRays,
                                        rng) * alphaEff;

                            const float3 outgoingRadiance =
                                    emittedRadiance + indirectRadiance + directRadiance;

                            accumulatedRadianceRGB += transmittance * outgoingRadiance;

                            // Median depth using compositing weights w_i = T_i * alpha_i
                            const float wi = transmittance * alphaEff;
                            const float zi =
                                    dot(worldHit.hitPositionW - sensor.camera.pos, sensor.camera.forward);

                            if (settings.useNormalConsistency && !medianFound && (accumulatedCompositeWeight + wi) >=
                                0.5f) {
                                medianFound = true;
                                medianDepth = zi;
                                medianWorldPosition = worldHit.hitPositionW;
                                medianNormalW = normalW;
                            }

                            accumulatedCompositeWeight += wi;

                            // Depth distortion
                            if (settings.useDepthDistortion) {
                                const float ai = alphaEff;
                                distortion += 2.0f * wi * (zi * prefixW - prefixWZ);
                                prefixW += wi;
                                prefixWZ += wi * zi;
                            }

                            transmittance *= (1.0f - alphaEff);
                            primaryRay.origin =
                                    worldHit.hitPositionW + primaryRay.direction * 1e-8f;
                            continue;
                        }

                        // -------------------------------------------------------------
                        // Terminal mesh hit
                        // -------------------------------------------------------------
                        if (instance.geometryType == GeometryType::Mesh) {
                            const GPUMaterial &material =
                                    scene.materials[instance.materialIndex];

                            const bool isBackfaceHit =
                                    dot(primaryRay.direction, worldHit.geometricNormalW) > 0.0f;

                            const float3 normalW = isBackfaceHit
                                                       ? -worldHit.geometricNormalW
                                                       : worldHit.geometricNormalW;

                            // Treat terminal mesh as opaque for median-depth purposes.
                            {
                                const float wi = transmittance;
                                const float zi =
                                        dot(worldHit.hitPositionW - sensor.camera.pos,
                                            sensor.camera.forward);

                                if (!medianFound && (accumulatedCompositeWeight + wi) >= 0.5f) {
                                    medianFound = true;
                                    medianDepth = zi;
                                    medianWorldPosition = worldHit.hitPositionW;
                                    medianNormalW = normalW;
                                }

                                accumulatedCompositeWeight += wi;
                            }

                            if (material.isEmissive()) {
                                const float3 emittedRadiance =
                                        material.power * material.baseColor;

                                accumulatedRadianceRGB +=
                                        transmittance * min(emittedRadiance, 1.0f);
                            } else {
                                const float3 indirectIrradiance = gatherDiffuseIrradianceAtPoint(
                                    worldHit.hitPositionW,
                                    normalW,
                                    photonMap);

                                const float3 indirectRadiance =
                                        (material.baseColor * M_1_PIf) * indirectIrradiance;

                                const float3 directRadiance =
                                        estimateDirectLightAtDiffuseSurface(
                                            scene,
                                            worldHit.hitPositionW,
                                            normalW,
                                            material.baseColor,
                                            settings.numShadowRays,
                                            rng);

                                const float3 outgoingRadiance =
                                        indirectRadiance + directRadiance;

                                accumulatedRadianceRGB += transmittance * outgoingRadiance;
                            }

                            transmittance = 0.0f;
                            break;
                        }
                    }

                    const std::uint32_t framebufferIndex =
                            pixelY * imageWidth + pixelX;

                    accumulatedRadianceRGB *= cameraCosine;

                    const float4 currentValue(
                        accumulatedRadianceRGB.x(),
                        accumulatedRadianceRGB.y(),
                        accumulatedRadianceRGB.z(),
                        1.0f);

                    sensor.framebuffer[framebufferIndex] += currentValue;

                    if (settings.useDepthDistortion) {
                        sensor.depthDistortionBuffer[pixelIndex] = distortion;
                    }

                    if (medianFound) {
                        sensor.medianDepthBuffer[pixelIndex] = medianDepth;

                        sensor.medianWorldPositionBuffer[pixelIndex] =
                                float4{
                                    medianWorldPosition.x(),
                                    medianWorldPosition.y(),
                                    medianWorldPosition.z(),
                                    1.0f
                                };

                        sensor.visibleNormalBuffer[pixelIndex] =
                                float4{
                                    medianNormalW.x(),
                                    medianNormalW.y(),
                                    medianNormalW.z(),
                                    1.0f
                                };
                    } else {
                        sensor.medianDepthBuffer[pixelIndex] = 0.0f;
                        sensor.medianWorldPositionBuffer[pixelIndex] =
                                float4{0.0f, 0.0f, 0.0f, 0.0f};
                        sensor.visibleNormalBuffer[pixelIndex] =
                                float4{0.0f, 0.0f, 0.0f, 0.0f};
                    }
                });
        });

        queue.wait();

        // -------------------------------------------------------------------------
        // Pass 2:
        //   Normal from median depth map
        // -------------------------------------------------------------------------
        if (settings.useNormalConsistency) {
            queue.submit([&](sycl::handler &cgh) {
                cgh.parallel_for<class MedianDepthNormalKernel>(
                    sycl::range<1>(pixelCount),
                    [=](sycl::id<1> tid) {
                        const std::uint32_t pixelIndex = tid[0];
                        const std::uint32_t x = pixelIndex % imageWidth;
                        const std::uint32_t y = pixelIndex / imageWidth;

                        if (x == 0u || y == 0u || x + 1u >= imageWidth || y + 1u >= imageHeight) {
                            sensor.normalFromDepthBuffer[pixelIndex] =
                                    float4{0.0f, 0.0f, 0.0f, 0.0f};
                            return;
                        }

                        const uint32_t idxL = y * imageWidth + (x - 1u);
                        const uint32_t idxR = y * imageWidth + (x + 1u);
                        const uint32_t idxU = (y - 1u) * imageWidth + x;
                        const uint32_t idxD = (y + 1u) * imageWidth + x;

                        const float zC = sensor.medianDepthBuffer[pixelIndex];
                        const float zL = sensor.medianDepthBuffer[idxL];
                        const float zR = sensor.medianDepthBuffer[idxR];
                        const float zU = sensor.medianDepthBuffer[idxU];
                        const float zD = sensor.medianDepthBuffer[idxD];

                        if (zC <= 0.0f || zL <= 0.0f || zR <= 0.0f || zU <= 0.0f || zD <= 0.0f) {
                            sensor.normalFromDepthBuffer[pixelIndex] =
                                    float4{0.0f, 0.0f, 0.0f, 0.0f};
                            return;
                        }

                        const float3 pC =
                                reconstructWorldPositionFromDepthCenter(sensor.camera, x, y, zC);
                        const float3 pL =
                                reconstructWorldPositionFromDepthCenter(sensor.camera, x - 1u, y, zL);
                        const float3 pR =
                                reconstructWorldPositionFromDepthCenter(sensor.camera, x + 1u, y, zR);
                        const float3 pU =
                                reconstructWorldPositionFromDepthCenter(sensor.camera, x, y - 1u, zU);
                        const float3 pD =
                                reconstructWorldPositionFromDepthCenter(sensor.camera, x, y + 1u, zD);

                        const float leftDepthDifference = sycl::fabs(zL - zC);
                        const float rightDepthDifference = sycl::fabs(zR - zC);
                        const float upDepthDifference = sycl::fabs(zU - zC);
                        const float downDepthDifference = sycl::fabs(zD - zC);

                        float3 dx;
                        if (rightDepthDifference < leftDepthDifference) {
                            dx = pR - pC;
                        } else {
                            dx = pC - pL;
                        }

                        float3 dy;
                        if (downDepthDifference < upDepthDifference) {
                            dy = pD - pC;
                        } else {
                            dy = pC - pU;
                        }

                        const float dxLen2 = dot(dx, dx);
                        const float dyLen2 = dot(dy, dy);

                        if (dxLen2 <= 1e-16f || dyLen2 <= 1e-16f) {
                            sensor.normalFromDepthBuffer[pixelIndex] =
                                    float4{0.0f, 0.0f, 0.0f, 0.0f};
                            return;
                        }

                        float3 normalW = normalize(cross(dx, dy));

                        const float3 viewDirectionToCamera = normalize(sensor.camera.pos - pC);
                        if (dot(normalW, viewDirectionToCamera) < 0.0f) {
                            normalW = -normalW;
                        }

                        sensor.normalFromDepthBuffer[pixelIndex] =
                                float4{normalW.x(), normalW.y(), normalW.z(), 1.0f};
                    });
            });

            queue.wait();
        }
    }

    /*
    void generateNextRays(RenderPackage& pkg, uint32_t activeRayCount) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& settings = pkg.settings;

        auto& intermediates = pkg.intermediates;
        auto* hitRecords = pkg.intermediates.hitRecords;
        auto* raysIn = pkg.intermediates.primaryRays;
        auto* raysOut = pkg.intermediates.extensionRaysA;
        auto* countExtensionOut = pkg.intermediates.countExtensionOut;

        queue.submit([&](sycl::handler& cgh) {
            uint64_t randomNumber = pkg.random.number;
            cgh.parallel_for<class ShadeKernelTag>(
                sycl::range<1>(activeRayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(randomNumber, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);
                    const WorldHit worldHit = hitRecords[rayIndex];
                    const RayState rayState = raysIn[rayIndex];
                    if (!worldHit.hit) return;

                    const InstanceRecord instance = scene.instances[worldHit.instanceIndex];
                    auto& geometryType = instance.geometryType;
                    float3 throughputMultiplier{0.0f};
                    float3 sampledOutgoingDirectionW = rayState.ray.direction;

                    if (geometryType == GeometryType::Mesh) {
                        const GPUMaterial material = scene.materials[instance.materialIndex];
                        // If we hit instance was a mesh do ordinary BRDF stuff.
                        float sampledPdf = 0.0f;
                        sampleCosineHemisphere(rng128, worldHit.geometricNormalW, sampledOutgoingDirectionW,
                                               sampledPdf);
                        const float3 lambertBrdf = material.baseColor;
                        throughputMultiplier = lambertBrdf * worldHit.transmissivity;
                    }

                    if (geometryType == GeometryType::PointCloud) {
                        const Point point = scene.points[worldHit.primitiveIndex];
                        // Reuse same albedo for scatter/Transmission
                        float3 c = point.albedo;
                        float alpha_r = point.alpha_r;
                        float alpha_t = point.alpha_t;

                        float3 rho_r = c * alpha_r; // diffuse reflectance
                        float3 rho_t = c * alpha_t; // diffuse transmission

                        const float segmentTransmittance = worldHit.transmissivity;
                        // Choose scalar for lobe probability
                        const float alphaSum = alpha_r + alpha_t;
                        if (alphaSum <= 0.0f) {
                            throughputMultiplier = float3{0.0f};
                            return;
                        }

                        const float pReflect = alpha_r / alphaSum;
                        const float uLobe = rng128.nextFloat();
                        const bool chooseReflection = (uLobe < pReflect);
                        const float3 lobeNormalW =
                            chooseReflection
                                ? worldHit.geometricNormalW
                                : (-worldHit.geometricNormalW);

                        float sampledPdf = 0.0f;
                        sampleCosineHemisphere(
                            rng128,
                            lobeNormalW,
                            sampledOutgoingDirectionW,
                            sampledPdf
                        );
                        // Mixture BSDF update simplifies analytically
                        throughputMultiplier = (rho_r + rho_t) * segmentTransmittance;
                    }

                    if (settings.integratorKind == IntegratorKind::photonMapping) {
                        auto& devicePtr = *intermediates.map.photonCountDevicePtr;
                        sycl::atomic_ref<uint32_t,
                                         sycl::memory_order::acq_rel,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            photonCounter(devicePtr);

                        const uint32_t slot = photonCounter.fetch_add(1u);
                        if (slot < intermediates.map.photonCapacity) {
                            DevicePhotonSurface photonEntry{};
                            photonEntry.position = worldHit.hitPositionW;
                            float3 baseNormalW = normalize(worldHit.geometricNormalW);
                            uint32_t primitiveIndexForDeposit = worldHit.instanceIndex;
                            const float signedCosineIncident = dot(baseNormalW, -rayState.ray.direction);
                            const int sideSign = signNonZero(signedCosineIncident);
                            //const float3 orientedNormalW = (sideSign >= 0) ? baseNormalW : (-baseNormalW);
                            photonEntry.power = rayState.pathThroughput * worldHit.transmissivity;
                            // Multiply with opacity to avoid derive an adjoint photon map pass?
                            //photonEntry.normal = orientedNormalW;
                            //photonEntry.sideSign = sideSign;
                            //photonEntry.geometryType = instance.geometryType;
                            photonEntry.isValid = 1u;
                            //photonEntry.incomingDirection = -rayState.ray.direction;
                            intermediates.map.photons[slot] = photonEntry;
                        }
                    }

                    // --- Spawn next ray with offset along *oriented* normal ---
                    RayState nextState{};
                    // Spawn next ray
                    nextState.ray.origin = worldHit.hitPositionW + (worldHit.geometricNormalW * 1e-6f);
                    nextState.ray.direction = sampledOutgoingDirectionW;
                    nextState.ray.normal = worldHit.geometricNormalW;
                    nextState.bounceIndex = rayState.bounceIndex + 1;
                    nextState.pixelIndex = rayState.pixelIndex;
                    nextState.pathThroughput = rayState.pathThroughput * throughputMultiplier;
                    // --- Russian roulette termination (after computing nextState) ---
                    //if (nextState.bounceIndex >= settings.russianRouletteStart) {
                    //    // Luminance-based continuation probability in [pMin, 1]
                    //    const float3 throughputRgb = nextState.pathThroughput;
                    //    const float luminance = luminanceGrayscale(throughputRgb);
                    //    const float pMin = 0.20f; // safety floor to avoid zero-probability bias
                    //    const float continuationProbability = sycl::clamp(luminance, pMin, 1.0f);
                    //
                    //    if (rng128.nextFloat() >= continuationProbability) {
                    //        return; // terminate path, do not enqueue
                    //    }
                    //    nextState.pathThroughput = nextState.pathThroughput / continuationProbability; // unbiased
                    //}
                    // --- Enqueue ---
                    auto extensionCounter = sycl::atomic_ref<uint32_t,
                                                             sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>(
                        *countExtensionOut);
                    const uint32_t outIndex = extensionCounter.fetch_add(1);
                    raysOut[outIndex] = nextState;
                });
        });
        queue.wait();
    }
    */

    void computePhotonCellIdsAndPermutation(
        sycl::queue &queue,
        DeviceSurfacePhotonMapGrid grid,
        std::uint32_t photonCount) {
        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range<1>(photonCount), [=](sycl::id<1> idx) {
                const std::uint32_t photonIndex = static_cast<std::uint32_t>(idx[0]);
                const DevicePhotonSurface photon = grid.photons[photonIndex];

                grid.photonIndex[photonIndex] = photonIndex;

                if (photon.isValid == 0u) {
                    grid.photonCellId[photonIndex] = kInvalidIndex;
                    return;
                }

                const sycl::int3 cell = worldToCell(photon.position, grid);
                const std::uint32_t cellId = linearCellIndex(cell, grid.gridResolution);
                grid.photonCellId[photonIndex] = cellId;
            });
        }).wait();
    }

    void clearCellArrays(sycl::queue &queue, DeviceSurfacePhotonMapGrid grid) {
        static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;

        const std::uint32_t cellCount = grid.totalCellCount;
        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range<1>(cellCount), [=](sycl::id<1> idx) {
                const std::uint32_t c = static_cast<std::uint32_t>(idx[0]);
                grid.cellCount[c] = 0u;
                grid.cellWriteOffset[c] = 0u;
                grid.cellStart[c] = kInvalidIndex;
                grid.cellEnd[c] = kInvalidIndex;
            });
        }).wait();
    }


    void countPhotonsPerCell(
        sycl::queue &queue,
        DeviceSurfacePhotonMapGrid grid,
        std::uint32_t photonCount) {
        static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range<1>(photonCount), [=](sycl::id<1> idx) {
                const std::uint32_t i = static_cast<std::uint32_t>(idx[0]);
                const std::uint32_t cellId = grid.photonCellId[i];
                if (cellId == kInvalidIndex) return;

                auto atomicCount = sycl::atomic_ref<std::uint32_t,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(grid.cellCount[cellId]);

                atomicCount.fetch_add(1u);
            });
        }).wait();
    }


    void scatterPhotonsIntoCells(
        sycl::queue &queue,
        DeviceSurfacePhotonMapGrid grid,
        std::uint32_t photonCount) {
        static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range<1>(photonCount), [=](sycl::id<1> idx) {
                const std::uint32_t i = static_cast<std::uint32_t>(idx[0]);
                const std::uint32_t cellId = grid.photonCellId[i];
                if (cellId == kInvalidIndex) return;

                const std::uint32_t start = grid.cellStart[cellId];
                // start should be valid if count > 0
                if (start == kInvalidIndex) return;

                auto atomicOffset = sycl::atomic_ref<std::uint32_t,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(
                    grid.cellWriteOffset[cellId]);

                const std::uint32_t localOffset = atomicOffset.fetch_add(1u);
                const uint32_t end = grid.cellEnd[cellId];
                const uint32_t writeIndex = start + localOffset;
                if (writeIndex < end) {
                    grid.sortedPhotonIndex[writeIndex] = i;
                }
            });
        });
    }

    static constexpr std::uint32_t kScanBlockSize = 1024;

    void exclusiveScanCellCountsToCellStart(
        sycl::queue &queue,
        DeviceSurfacePhotonMapGrid grid) {
        const std::uint32_t totalCellCount = grid.totalCellCount;
        const std::uint32_t blockSize = kScanBlockSize;
        const std::uint32_t blockCount = (totalCellCount + blockSize - 1u) / blockSize;

        std::uint32_t *cellCount = grid.cellCount;
        std::uint32_t *cellStart = grid.cellStart;
        std::uint32_t *blockSums = grid.blockSums;
        std::uint32_t *blockPrefix = grid.blockPrefix;

        // Pass 1: per-block exclusive scan into cellStart + write block sums
        queue.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<std::uint32_t, 1> localData(sycl::range<1>(blockSize), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(blockCount * blockSize), sycl::range<1>(blockSize)),
                [=](sycl::nd_item<1> item) {
                    const std::uint32_t localIndex = static_cast<std::uint32_t>(item.get_local_id(0));
                    const std::uint32_t blockIndex = static_cast<std::uint32_t>(item.get_group(0));
                    const std::uint32_t globalIndex = blockIndex * blockSize + localIndex;

                    // Load into local memory (out-of-range -> 0)
                    std::uint32_t value = 0u;
                    if (globalIndex < totalCellCount)
                        value = cellCount[globalIndex];

                    localData[localIndex] = value;
                    item.barrier(sycl::access::fence_space::local_space);

                    // Blelloch upsweep
                    for (std::uint32_t offset = 1u; offset < blockSize; offset <<= 1u) {
                        const std::uint32_t index = (localIndex + 1u) * offset * 2u - 1u;
                        if (index < blockSize)
                            localData[index] += localData[index - offset];

                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    // Write total sum for this block, then set root to 0 for exclusive scan
                    if (localIndex == blockSize - 1u) {
                        blockSums[blockIndex] = localData[localIndex];
                        localData[localIndex] = 0u;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    // Blelloch downsweep
                    for (std::uint32_t offset = blockSize >> 1u; offset > 0u; offset >>= 1u) {
                        const std::uint32_t index = (localIndex + 1u) * offset * 2u - 1u;
                        if (index < blockSize) {
                            const std::uint32_t left = localData[index - offset];
                            const std::uint32_t right = localData[index];
                            localData[index - offset] = right;
                            localData[index] = right + left;
                        }

                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    // Store per-element exclusive prefix into cellStart
                    if (globalIndex < totalCellCount)
                        cellStart[globalIndex] = localData[localIndex];
                });
        }).wait();

        // Pass 2: exclusive scan of block sums on CPU (blockCount is small)
        std::vector<std::uint32_t> blockSumsHost(blockCount);
        queue.memcpy(blockSumsHost.data(), blockSums, sizeof(std::uint32_t) * blockCount).wait();

        std::vector<std::uint32_t> blockPrefixHost(blockCount);
        std::uint32_t runningSum = 0u;
        for (std::uint32_t b = 0; b < blockCount; ++b) {
            blockPrefixHost[b] = runningSum;
            runningSum += blockSumsHost[b];
        }

        queue.memcpy(blockPrefix, blockPrefixHost.data(), sizeof(std::uint32_t) * blockCount).wait();

        // Pass 3: add block prefix to each element’s local prefix
        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range<1>(totalCellCount), [=](sycl::id<1> idx) {
                const std::uint32_t globalIndex = static_cast<std::uint32_t>(idx[0]);
                const std::uint32_t blockIndex = globalIndex / blockSize;
                cellStart[globalIndex] += blockPrefix[blockIndex];
            });
        }).wait();
    }

    void finalizeCellRanges(
        sycl::queue &queue,
        DeviceSurfacePhotonMapGrid grid) {
        static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;

        const std::uint32_t totalCellCount = grid.totalCellCount;

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range<1>(totalCellCount), [=](sycl::id<1> idx) {
                const std::uint32_t c = static_cast<std::uint32_t>(idx[0]);
                const std::uint32_t count = grid.cellCount[c];

                grid.cellWriteOffset[c] = 0u;

                if (count == 0u) {
                    grid.cellStart[c] = kInvalidIndex;
                    grid.cellEnd[c] = kInvalidIndex;
                } else {
                    const std::uint32_t start = grid.cellStart[c];
                    grid.cellEnd[c] = start + count;
                }
            });
        }).wait();
    }


    void buildPhotonCellRangesAndOrdering(
        sycl::queue &queue,
        DeviceSurfacePhotonMapGrid grid,
        std::uint32_t photonCount) {
        clearCellArrays(queue, grid); // counts/start/end/offset=0/invalid
        computePhotonCellIdsAndPermutation(queue, grid, photonCount); // keys (optional now)
        countPhotonsPerCell(queue, grid, photonCount); // histogram

        exclusiveScanCellCountsToCellStart(queue, grid); // cellStart from cellCount (implement)
        finalizeCellRanges(queue, grid); // cellEnd = start + count, invalid if count==0, offset=0
        scatterPhotonsIntoCells(queue, grid, photonCount); // sortedPhotonIndex
    }
} // Pale
