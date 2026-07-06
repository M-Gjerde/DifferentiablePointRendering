//
// Created by magnus on 9/12/25.
//

#include "PrimalKernels.h"

#include <cmath>

#include "KernelHelpers.h"
#include "IntersectionKernels.h"

namespace Pale {
    void launchRayGenEmitterKernel(RenderPackage& pkg, uint32_t forwardPass) {
        auto queue = pkg.queue;
        auto scene = pkg.scene;
        auto settings = pkg.settings;

        auto* raysIn = pkg.intermediates.primaryRays;
        auto* countPrimary = pkg.intermediates.countPrimary;

        const uint32_t emittedCount = settings.photonsPerLaunch * settings.numForwardPasses;
        const float invEmittedCount = 1.0f / float(emittedCount);

        // One invariant seed for the whole render (stable noise across reruns)
        const uint64_t renderSeed = pkg.random.seed;


        queue.submit([&](sycl::handler& commandGroupHandler) {
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


    void launchIntersectKernel(RenderPackage& pkg, uint32_t activeRayCount) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& settings = pkg.settings;
        auto& intermediates = pkg.intermediates;

        queue.submit([&](sycl::handler& commandGroupHandler) {
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
                            const InstanceRecord& instance = scene.instances[worldHit.instanceIndex];
                            // ---------------------------------------------------------------------
                            // Mesh: this is a real scattering event, so it finishes this kernel call.
                            // ---------------------------------------------------------------------
                            if (instance.geometryType == GeometryType::Mesh) {
                                bool isBackfaceHit =
                                    dot(currentRayState.ray.direction, worldHit.geometricNormalW) > 0.0f;
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
                                const GPUMaterial material = scene.materials[instance.materialIndex];
                                if (settings.integratorKind == IntegratorKind::photonMapping &&
                                    isIndirect &&
                                    !material.isEmissive()) {
                                    depositPhotonSurface(worldHit, currentRayState.ray.direction,
                                                         worldHit.geometricNormalW, currentRayState.pathThroughput,
                                                         intermediates.map);
                                }
                                float3 sampledOutgoingDirectionW = currentRayState.ray.direction;
                                float sampledPdf = 0.0f;
                                sampleCosineHemisphere(stepRng, worldHit.geometricNormalW, sampledOutgoingDirectionW,
                                                       sampledPdf);
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
                            // Point cloud: alpha controls whether the ray transmits or scatters.
                            // ---------------------------------------------------------------------
                            if (instance.geometryType == GeometryType::PointCloud) {
                                const Point& surfel = scene.points[worldHit.primitiveIndex];
                                const float effectiveOpacity = sycl::fmin(
                                    1.0f, sycl::fmax(0.0f, worldHit.alphaGeom * surfel.opacity));
                                const float scatterProbability = effectiveOpacity;
                                const float transmitProbability = 1.0f - scatterProbability;
                                const float randomNumber = stepRng.nextFloat();

                                // -----------------------------------------------------------------
                                // Transmit / null event:
                                //   - probability: 1 - alpha_eff
                                //   - physical multiplier: 1 - alpha_eff
                                //   - sampled-event weight: (1 - alpha_eff) / p_transmit = 1
                                //   - does NOT increment bounceIndex
                                // -----------------------------------------------------------------
                                if (randomNumber < transmitProbability) {
                                    currentRayState.ray.origin =
                                        worldHit.hitPositionW + currentRayState.ray.direction * RayEpsilon;
                                    currentRayState.traversalIndex += 1;
                                    // No throughput attenuation here because this branch was sampled
                                    // with exactly the physical transmission probability.
                                    continue;
                                }

                                // -----------------------------------------------------------------
                                // Scatter / reflect event:
                                //   - probability: alpha_eff
                                //   - physical multiplier: alpha_eff * diffuse_reflectance
                                //   - sampled-event weight:
                                //       alpha_eff * diffuse_reflectance / p_scatter
                                //       = diffuse_reflectance
                                //   - increments bounceIndex
                                // -----------------------------------------------------------------
                                if (scatterProbability <= 0.0f) {
                                    return;
                                }

                                const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));

                                const float signedCosineIncident =
                                    dot(canonicalNormalW, -currentRayState.ray.direction);

                                const int sideSign = signNonZero(signedCosineIncident);
                                const float3 orientedNormal = static_cast<float>(sideSign) * canonicalNormalW;

                                if (settings.integratorKind == IntegratorKind::photonMapping && isIndirect && !surfel.
                                    isEmissive()) {
                                    depositPhotonSurface(worldHit, currentRayState.ray.direction, orientedNormal,
                                                         currentRayState.pathThroughput / scatterProbability,
                                                         intermediates.map);
                                }

                                float3 sampledOutgoingDirectionW = currentRayState.ray.direction;
                                float sampledPdf = 0.0f;
                                sampleCosineHemisphere(stepRng, orientedNormal, sampledOutgoingDirectionW, sampledPdf);
                                const float3 reflectanceMultiplier = surfel.alpha_r * surfel.albedo;
                                const float3 nextPathThroughput =
                                    currentRayState.pathThroughput * reflectanceMultiplier;
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
                                nextRayState.ray.origin = worldHit.hitPositionW + orientedNormal * RayEpsilon;
                                nextRayState.ray.direction = sampledOutgoingDirectionW;
                                nextRayState.ray.normal = orientedNormal;
                                nextRayState.bounceIndex = currentRayState.bounceIndex + 1;
                                nextRayState.traversalIndex = currentRayState.traversalIndex + 1;
                                nextRayState.pixelIndex = currentRayState.pixelIndex;
                                nextRayState.pathThroughput = nextPathThroughput;
                                nextRayState.pathId = currentRayState.pathId;
                                nextRayState.lightIndex = currentRayState.lightIndex;
                                if (!applyRussianRoulette(stepRng, nextRayState.bounceIndex,
                                                          nextRayState.pathThroughput, settings.russianRouletteStart)) {
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
                                // Transmission / absorption currently simplified out.
                                // Leave empty / terminate for now.
                                return;
                            }
                            return;
                        }
                        // Traversal guard hit: terminate the ray.
                        return;
                    });
            }
        );
        queue.wait();
    }


    void launchCameraGatherKernel2(RenderPackage& pkg, uint32_t cameraIndex, uint32_t gatherPass) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& settings = pkg.settings;
        auto& photonMap = pkg.intermediates.map;
        SensorGPU sensor = pkg.sensors[cameraIndex];
        const std::uint32_t imageWidth = sensor.camera.width;
        const std::uint32_t imageHeight = sensor.camera.height;
        const std::uint32_t pixelCount = imageWidth * imageHeight;
        queue.fill(sensor.framebuffer, float4{0.0f, 0.0f, 0.0f, 0.0f}, pixelCount).wait();
        queue.fill(sensor.medianDepthBuffer, 0.0f, pixelCount).wait();
        queue.fill(sensor.meanDepthBuffer, 0.0f, pixelCount).wait();
        queue.fill(sensor.medianWorldPositionBuffer, float4{0.0f, 0.0f, 0.0f, 0.0f}, pixelCount).wait();
        queue.fill(sensor.visibleNormalBuffer, float4{0.0f, 0.0f, 0.0f, 0.0f}, pixelCount).wait();
        queue.fill(sensor.normalFromDepthBuffer, float4{0.0f, 0.0f, 0.0f, 0.0f}, pixelCount).wait();
        queue.fill(sensor.depthDistortionBuffer, 0.0f, pixelCount).wait();
        queue.fill(sensor.depthDistortionAdjointBuffer, 0.0f, pixelCount).wait();
        queue.fill(sensor.visibilityWeightedOpacityBuffer, 0.0f, pixelCount).wait();

        // -------------------------------------------------------------------------
        // Pass 1:
        //   - RGB gather
        //   - median depth
        //   - median world position
        //   - visible normal at median surface
        // -------------------------------------------------------------------------
        queue.submit([&](sycl::handler& cgh) {
            const uint64_t renderSeed = pkg.random.seed;

            cgh.parallel_for<class CameraGatherKernel>(
                sycl::range<1>(pixelCount),
                [=](sycl::id<1> tid) {
                    const std::uint32_t pixelIndex = tid[0];
                    const std::uint32_t pixelX = pixelIndex % imageWidth;
                    const std::uint32_t pixelY = pixelIndex / imageWidth;
                    const uint64_t directionSeed = rng::makeSeed(renderSeed, pixelIndex, cameraIndex,
                                                                 rng::kStreamGather, gatherPass);
                    rng::Xorshift128 rng(directionSeed);
                    float3 accumulatedRadianceRGB(0.0f, 0.0f, 0.0f);
                    // Center-of-pixel sample in your jitter convention
                    const float jitterX = rng.nextFloat() - 0.5f;
                    const float jitterY = rng.nextFloat() - 0.5f;
                    Ray primaryRay = makePrimaryRayFromPixelJitteredFov(
                        sensor.camera,
                        static_cast<float>(pixelX),
                        static_cast<float>(pixelY),
                        0, 0
                    );
                    const float cameraCosine = dot(sensor.camera.forward, primaryRay.direction);
                    float transmittance = 1.0f;
                    // Depth distortion accumulation
                    float distortion = 0.0f;
                    float prefixWeight = 0.0f;
                    float prefixWeightDepth = 0.0f;
                    float prefixWeightDepthSquared = 0.0f;
                    // Visibility regularizer
                    float visibilityWeightedOpacityLoss = 0.0f;
                    // Median-depth tracking
                    float accumulatedCompositeWeight = 0.0f;
                    bool medianFound = false;
                    float medianDepth = 0.0f;
                    float3 medianWorldPosition(0.0f, 0.0f, 0.0f);
                    float3 medianNormalW(0.0f, 0.0f, 0.0f);
                    float accumulatedMeanDepthWeight = 0.0f;
                    float accumulatedMeanDepth = 0.0f;
                    bool debug = false;
                    if (pixelX == 286 && pixelY == 240) {
                        debug = true;
                    }
                    else {
                        //return;
                    }

                    const float localLayerDepthEpsilon =
                        rendererDebugLocalLayerDepthEpsilon(settings);
                    const uint32_t maxSplatEventsPerRay =
                        rendererDebugMaxSplatEventsPerRay(settings);
                    const uint32_t maxLocalSurfelHits =
                        rendererDebugMaxLocalSurfelHits(settings);

                    for (uint32_t traversalIndex = 0u; traversalIndex < maxSplatEventsPerRay; ++traversalIndex) {
                        WorldHit worldHit{};
                        intersectScene(primaryRay, &worldHit, scene, SurfelIntersectMode::FirstHit);
                        if (!worldHit.hit) {
                            break;
                        }
                        buildIntersectionNormal(scene, worldHit);
                        const auto& instance = scene.instances[worldHit.instanceIndex];
                        // -------------------------------------------------------------
                        // Visible point-cloud layer
                        // -------------------------------------------------------------
                        if (instance.geometryType == GeometryType::PointCloud) {
                            const Transform& transform = scene.transforms[instance.transformIndex];
                            // This should scale with surfel size, not be a global value such as 0.1.
                            const float localTMin = worldHit.t - localLayerDepthEpsilon;
                            const float localTMax = worldHit.t + localLayerDepthEpsilon;

                            LocalSurfelLayerHit localHits[kMaxLocalSurfelHits];
                            const Ray rayObject = toObjectSpace(primaryRay, transform);
                            uint32_t localHitCount = collectBLASPointCloudLocalLayer(
                                primaryRay, rayObject, instance.blasRangeIndex, transform, localTMin, localTMax,
                                localHits, maxLocalSurfelHits, scene);

                            // Numerical fallback: preserve the already found FirstHit.
                            if (localHitCount == 0u) {
                                localHits[0].tWorld = worldHit.t;
                                localHits[0].primitiveIndex = worldHit.primitiveIndex;
                                localHits[0].alphaGeom = worldHit.alphaGeom;
                                localHits[0].hitPositionW = worldHit.hitPositionW;
                                localHitCount = 1u;
                            }

                            float furthestLayerT = worldHit.t;
                            float localAlphaEff[kMaxLocalSurfelHits];
                            float localLayerWeight[kMaxLocalSurfelHits];
                            float localLayerTransmission = 1.0f;

                            for (uint32_t localHitIndex = 0u; localHitIndex < localHitCount; ++localHitIndex) {
                                const LocalSurfelLayerHit& localHit = localHits[localHitIndex];
                                const Point& surfel = scene.points[localHit.primitiveIndex];
                                const float alphaEff = sycl::clamp(surfel.opacity * localHit.alphaGeom, 0.0f, 1.0f);
                                localAlphaEff[localHitIndex] = alphaEff;
                                localLayerWeight[localHitIndex] = 0.0f;
                                furthestLayerT = sycl::fmax(furthestLayerT, localHit.tWorld);
                                localLayerTransmission *= sycl::fmax(0.0f, 1.0f - alphaEff);
                            }

                            const float localLayerOpacity = 1.0f - localLayerTransmission;

                            // Average over all unresolved depth orders in the local layer. Equal opaque
                            // surfels therefore split the layer contribution evenly while the final
                            // transmission remains the product of their transparencies.
                            float localLayerWeightSum = 0.0f;
                            if (localLayerOpacity > 0.0f) {
                                for (uint32_t localHitIndex = 0u; localHitIndex < localHitCount; ++localHitIndex) {
                                    const float alphaEff = localAlphaEff[localHitIndex];
                                    if (alphaEff <= 0.0f) {
                                        continue;
                                    }

                                    float transmittancePolynomial[kMaxLocalSurfelHits];
                                    for (uint32_t coefficientIndex = 0u;
                                         coefficientIndex < maxLocalSurfelHits;
                                         ++coefficientIndex) {
                                        transmittancePolynomial[coefficientIndex] = 0.0f;
                                    }
                                    transmittancePolynomial[0] = 1.0f;

                                    uint32_t polynomialDegree = 0u;
                                    for (uint32_t otherHitIndex = 0u;
                                         otherHitIndex < localHitCount;
                                         ++otherHitIndex) {
                                        if (otherHitIndex == localHitIndex) {
                                            continue;
                                        }

                                        const float otherAlphaEff = localAlphaEff[otherHitIndex];
                                        if (otherAlphaEff <= 0.0f) {
                                            continue;
                                        }

                                        for (int32_t coefficientIndex = static_cast<int32_t>(polynomialDegree);
                                             coefficientIndex >= 0;
                                             --coefficientIndex) {
                                            transmittancePolynomial[coefficientIndex + 1] -=
                                                otherAlphaEff * transmittancePolynomial[coefficientIndex];
                                        }
                                        ++polynomialDegree;
                                    }

                                    float expectedPreviousTransmittance = 0.0f;
                                    for (uint32_t coefficientIndex = 0u;
                                         coefficientIndex <= polynomialDegree;
                                         ++coefficientIndex) {
                                        expectedPreviousTransmittance +=
                                            transmittancePolynomial[coefficientIndex] /
                                            static_cast<float>(coefficientIndex + 1u);
                                    }

                                    const float layerWeight = alphaEff * sycl::clamp(
                                        expectedPreviousTransmittance,
                                        0.0f,
                                        1.0f);
                                    localLayerWeight[localHitIndex] = layerWeight;
                                    localLayerWeightSum += layerWeight;
                                }
                            }

                            if (localLayerWeightSum > 1.0e-8f) {
                                const float weightNormalization = localLayerOpacity / localLayerWeightSum;
                                for (uint32_t localHitIndex = 0u; localHitIndex < localHitCount; ++localHitIndex) {
                                    localLayerWeight[localHitIndex] *= weightNormalization;
                                }
                            }

                            const float3 directLightingPositionW = localHits[0].hitPositionW;
                            for (uint32_t localHitIndex = 0u; localHitIndex < localHitCount; ++localHitIndex) {
                                const float layerWeight = localLayerWeight[localHitIndex];
                                const LocalSurfelLayerHit& localHit = localHits[localHitIndex];
                                const Point& surfel = scene.points[localHit.primitiveIndex];
                                if (layerWeight <= 0.0f) {
                                    continue;
                                }
                                const float compositeWeight = transmittance * layerWeight;
                                float3 normalW = normalize(cross(surfel.tanU, surfel.tanV));
                                const bool hitBackside = dot(normalW, -primaryRay.direction) < 0.0f;
                                if (hitBackside) {
                                    normalW = -normalW;
                                }
                                const float3 indirectIrradiance = gatherDiffuseIrradianceAtPoint(
                                    localHit.hitPositionW, normalW, photonMap);
                                const float3 indirectRadiance =
                                    indirectIrradiance * (surfel.alpha_r * surfel.albedo * M_1_PIf);

                                const float surfelArea = M_PIf * surfel.scale.x() * surfel.scale.y();
                                float3 emittedRadiance =
                                    surfel.albedo * (surfel.flux / (M_PIf * surfelArea));

                                /*
                                const float3 directRadiance =
                                    estimateDirectAreaLightAtDiffuseSurface(
                                        scene,
                                        directLightingPositionW,
                                        normalW,
                                        surfel.alpha_r * surfel.albedo,
                                        settings,
                                        rng);
                                */
                                const float3 directRadiance = estimateDirectPointSampledPointLights(
                                    scene,
                                    settings,
                                    localHit.hitPositionW,
                                    normalW,
                                    surfel.alpha_r * surfel.albedo);

                                const float3 outgoingRadiance = emittedRadiance + indirectRadiance + directRadiance;
                                accumulatedRadianceRGB += compositeWeight * outgoingRadiance;
                                const float depth = dot(localHit.hitPositionW - sensor.camera.pos,
                                                        sensor.camera.forward);
                                accumulatedMeanDepthWeight += compositeWeight;
                                accumulatedMeanDepth += compositeWeight * depth;
                                const float opacityResidual = 1.0f - surfel.opacity;
                                visibilityWeightedOpacityLoss += compositeWeight * opacityResidual * opacityResidual;
                                if (!medianFound &&
                                    (accumulatedCompositeWeight + compositeWeight) >= 0.5f) {
                                    medianFound = true;
                                    medianDepth = depth;
                                    medianWorldPosition = localHit.hitPositionW;
                                    medianNormalW = normalW;
                                }
                                accumulatedCompositeWeight += compositeWeight;
                                const float normalizedDepth = depthDistortionNdc01(depth);
                                distortion += compositeWeight * (
                                    normalizedDepth * normalizedDepth * prefixWeight + prefixWeightDepthSquared - 2.0f *
                                    normalizedDepth * prefixWeightDepth);
                                prefixWeight += compositeWeight;
                                prefixWeightDepth += compositeWeight * normalizedDepth;
                                prefixWeightDepthSquared += compositeWeight * normalizedDepth * normalizedDepth;
                            }
                            transmittance *= localLayerTransmission;

                            // Advance once past the whole local layer.
                            primaryRay.origin = primaryRay.origin + primaryRay.direction * (
                                furthestLayerT + RayEpsilon);
                            continue;
                        }
                        // -------------------------------------------------------------
                        // Terminal mesh hit
                        // -------------------------------------------------------------
                        if (instance.geometryType == GeometryType::Mesh) {
                            const GPUMaterial& material = scene.materials[instance.materialIndex];
                            const bool isBackfaceHit = dot(primaryRay.direction, worldHit.geometricNormalW) > 0.0f;
                            const float3 normalW = isBackfaceHit
                                                       ? -worldHit.geometricNormalW
                                                       : worldHit.geometricNormalW;
                            // Treat terminal mesh as opaque for median-depth purposes.
                            {
                                const float wi = transmittance;
                                const float zi = dot(worldHit.hitPositionW - sensor.camera.pos, sensor.camera.forward);
                                accumulatedMeanDepthWeight += wi;
                                accumulatedMeanDepth += wi * zi;
                                if (!medianFound && (accumulatedCompositeWeight + wi) >= 0.5f) {
                                    medianFound = true;
                                    medianDepth = zi;
                                    medianWorldPosition = worldHit.hitPositionW;
                                    medianNormalW = normalW;
                                }
                                accumulatedCompositeWeight += wi;
                            }
                            if (material.isEmissive()) {
                                const float3 emittedRadiance = material.power * material.baseColor;
                                accumulatedRadianceRGB += transmittance * min(emittedRadiance, 1.0f);
                            }
                            else {
                                const float3 indirectIrradiance = gatherDiffuseIrradianceAtPoint(
                                    worldHit.hitPositionW, normalW, photonMap);
                                const float3 indirectRadiance = (material.baseColor * M_1_PIf) * indirectIrradiance;
                                const float3 directRadiance = estimateDirectAreaLightAtDiffuseSurface(
                                    scene, worldHit.hitPositionW, normalW, material.baseColor, settings, rng);
                                const float3 outgoingRadiance = indirectRadiance + directRadiance;
                                accumulatedRadianceRGB += transmittance * outgoingRadiance;
                            }
                            transmittance = 0.0f;
                            break;
                        }
                    }
                    const std::uint32_t framebufferIndex = pixelY * imageWidth + pixelX;
                    //accumulatedRadianceRGB *= cameraCosine;
                    const float alpha = sycl::clamp(accumulatedCompositeWeight, 0.0f, 1.0f);
                    const float4 currentValue(
                        accumulatedRadianceRGB.x(),
                        accumulatedRadianceRGB.y(),
                        accumulatedRadianceRGB.z(),
                        alpha);
                    sensor.framebuffer[framebufferIndex] += currentValue;
                    sensor.depthDistortionBuffer[pixelIndex] = distortion;
                    sensor.visibilityWeightedOpacityBuffer[pixelIndex] = visibilityWeightedOpacityLoss;
                    if (accumulatedMeanDepthWeight > 1.0e-6f) {
                        sensor.meanDepthBuffer[pixelIndex] = accumulatedMeanDepth / accumulatedMeanDepthWeight;
                    }
                    else {
                        sensor.meanDepthBuffer[pixelIndex] = 0.0f;
                    }

                    if (medianFound) {
                        sensor.medianDepthBuffer[pixelIndex] = medianDepth;
                        sensor.medianWorldPositionBuffer[pixelIndex] = float4{
                            medianWorldPosition.x(), medianWorldPosition.y(), medianWorldPosition.z(), 1.0f
                        };
                        sensor.visibleNormalBuffer[pixelIndex] = float4{
                            medianNormalW.x(), medianNormalW.y(), medianNormalW.z(), 1.0f
                        };
                    }
                    else {
                        sensor.medianDepthBuffer[pixelIndex] = 0.0f;
                        sensor.medianWorldPositionBuffer[pixelIndex] = float4{0.0f};
                        sensor.visibleNormalBuffer[pixelIndex] = float4{0.0f};
                    }
                }
            );
        });

        queue.wait();

        // -------------------------------------------------------------------------
        // Pass 2:
        //   Normal from 2DGS-style pseudo surface depth map
        // -------------------------------------------------------------------------
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<class SurfaceDepthNormalKernel>(
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
                    // Same meaning as 2DGS pipe.depth_ratio:
                    //
                    //   depthRatio = 1 -> use median depth for bounded scenes
                    //   depthRatio = 0 -> use expected / mean depth for unbounded scenes
                    //
                    // Add this setting if you do not already have it.
                    const float depthRatio = 1;
                    const float zMedianC = sensor.medianDepthBuffer[pixelIndex];
                    const float zMedianL = sensor.medianDepthBuffer[idxL];
                    const float zMedianR = sensor.medianDepthBuffer[idxR];
                    const float zMedianU = sensor.medianDepthBuffer[idxU];
                    const float zMedianD = sensor.medianDepthBuffer[idxD];

                    const float zMeanC = sensor.meanDepthBuffer[pixelIndex];
                    const float zMeanL = sensor.meanDepthBuffer[idxL];
                    const float zMeanR = sensor.meanDepthBuffer[idxR];
                    const float zMeanU = sensor.meanDepthBuffer[idxU];
                    const float zMeanD = sensor.meanDepthBuffer[idxD];

                    const float zC = zMeanC * (1.0f - depthRatio) + zMedianC * depthRatio;
                    const float zL = zMeanL * (1.0f - depthRatio) + zMedianL * depthRatio;
                    const float zR = zMeanR * (1.0f - depthRatio) + zMedianR * depthRatio;
                    const float zU = zMeanU * (1.0f - depthRatio) + zMedianU * depthRatio;
                    const float zD = zMeanD * (1.0f - depthRatio) + zMedianD * depthRatio;

                    if (zC <= 0.0f || zL <= 0.0f || zR <= 0.0f || zU <= 0.0f || zD <= 0.0f) {
                        sensor.normalFromDepthBuffer[pixelIndex] = float4{0.0f, 0.0f, 0.0f, 0.0f};
                        return;
                    }
                    const float3 pL = reconstructWorldPositionFromDepthCenter(sensor.camera, x - 1u, y, zL);
                    const float3 pR = reconstructWorldPositionFromDepthCenter(sensor.camera, x + 1u, y, zR);
                    const float3 pU = reconstructWorldPositionFromDepthCenter(sensor.camera, x, y - 1u, zU);
                    const float3 pD = reconstructWorldPositionFromDepthCenter(sensor.camera, x, y + 1u, zD);
                    // 2DGS depth_to_normal:
                    //
                    //   dx = points[2:, 1:-1] - points[:-2, 1:-1]
                    //   dy = points[1:-1, 2:] - points[1:-1, :-2]
                    //   normal = normalize(cross(dx, dy))
                    //
                    // With image coordinates:
                    //
                    //   tangentY = P(y + 1, x) - P(y - 1, x)
                    //   tangentX = P(y, x + 1) - P(y, x - 1)
                    //
                    const float3 tangentY = pD - pU;
                    const float3 tangentX = pR - pL;

                    const float tangentYLengthSquared = dot(tangentY, tangentY);
                    const float tangentXLengthSquared = dot(tangentX, tangentX);
                    if (tangentYLengthSquared <= 1.0e-16f || tangentXLengthSquared <= 1.0e-16f) {
                        sensor.normalFromDepthBuffer[pixelIndex] =
                            float4{0.0f, 0.0f, 0.0f, 0.0f};
                        return;
                    }
                    // Match 2DGS cross-product order exactly:
                    //
                    //   normal = normalize(cross(tangentY, tangentX))
                    //
                    // Do not flip toward the camera if you want exact 2DGS behavior.
                    const float3 normalW = normalize(cross(tangentY, tangentX));
                    sensor.normalFromDepthBuffer[pixelIndex] =
                        float4{normalW.x(), normalW.y(), normalW.z(), 1.0f};
                });
        });

        queue.wait();
    }


    void launchCameraGatherKernel(RenderPackage &pkg, uint32_t cameraIndex, uint32_t gatherPassIdx) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto &photonMap = pkg.intermediates.map;
        SensorGPU sensor = pkg.sensors[cameraIndex];
        const std::uint32_t imageWidth = sensor.camera.width;
        const std::uint32_t imageHeight = sensor.camera.height;
        const std::uint32_t pixelCount = imageWidth * imageHeight;
        queue.fill(sensor.framebuffer, float4{0.0f, 0.0f, 0.0f, 0.0f}, pixelCount).wait();
        queue.fill(sensor.medianDepthBuffer, 0.0f, pixelCount).wait();
        queue.fill(sensor.meanDepthBuffer, 0.0f, pixelCount).wait();
        queue.fill(sensor.medianWorldPositionBuffer, float4{0.0f, 0.0f, 0.0f, 0.0f}, pixelCount).wait();
        queue.fill(sensor.visibleNormalBuffer, float4{0.0f, 0.0f, 0.0f, 0.0f}, pixelCount).wait();
        queue.fill(sensor.normalFromDepthBuffer, float4{0.0f, 0.0f, 0.0f, 0.0f}, pixelCount).wait();
        queue.fill(sensor.depthDistortionBuffer, 0.0f, pixelCount).wait();
        queue.fill(sensor.depthDistortionAdjointBuffer, 0.0f, pixelCount).wait();


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
                    const uint64_t directionSeed = rng::makeSeed(renderSeed, pixelIndex, cameraIndex,
                                                                 rng::kStreamGather, 0u);
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
                    // Depth distortion accumulation
                    float distortion = 0.0f;
                    float prefixWeight = 0.0f;
                    float prefixWeightDepth = 0.0f;
                    float prefixWeightDepthSquared = 0.0f;
                    // Median-depth tracking
                    float accumulatedCompositeWeight = 0.0f;
                    bool medianFound = false;
                    float medianDepth = 0.0f;
                    float3 medianWorldPosition(0.0f, 0.0f, 0.0f);
                    float3 medianNormalW(0.0f, 0.0f, 0.0f);
                    float accumulatedMeanDepthWeight = 0.0f;
                    float accumulatedMeanDepth = 0.0f;

                    for (uint32_t traversalIndex = 0u;
                         traversalIndex < kMaxSplatEventsPerRay;
                         ++traversalIndex) {
                        WorldHit worldHit{};
                        intersectScene(primaryRay, &worldHit, scene, SurfelIntersectMode::FirstHit);
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
                            const bool hitBackside = dot(normalW, -primaryRay.direction) < 0.0f;
                            if (hitBackside) {
                                normalW = -normalW;
                            }
                            const float alphaEff = surfel.opacity * worldHit.alphaGeom;
                            const float3 indirectIrradiance = gatherDiffuseIrradianceAtPoint(
                                worldHit.hitPositionW, normalW, photonMap);
                            const float3 indirectRadiance =
                                    indirectIrradiance * (surfel.alpha_r * surfel.albedo * M_1_PIf) * alphaEff;
                            const float surfelArea = M_PIf * surfel.scale.x() * surfel.scale.y();
                            float3 emittedRadiance = surfel.albedo * (surfel.flux / (M_PIf * surfelArea)) * alphaEff;
                            if (surfel.isEmissive() && hitBackside) {
                                //emittedRadiance = float3(0.0f, 0.0f, 0.0f);
                            }
                            /*
                            const float3 directRadiance =
                                    estimateDirectAreaLightAtDiffuseSurface(scene, worldHit.hitPositionW, normalW,
                                                                            surfel.alpha_r * surfel.albedo, settings,
                                                                            rng) * alphaEff;
                            */

                            const float3 directRadiance = estimateDirectPointSampledPointLights(
                            scene,
                            settings,
                            worldHit.hitPositionW,
                            normalW, surfel.alpha_r * surfel.albedo ) * alphaEff;

                            const float3 outgoingRadiance = emittedRadiance + indirectRadiance + directRadiance;
                            accumulatedRadianceRGB += transmittance * outgoingRadiance;
                            // Median depth using compositing weights w_i = T_i * alpha_i
                            const float wi = transmittance * alphaEff;
                            const float zi = dot(worldHit.hitPositionW - sensor.camera.pos, sensor.camera.forward);

                            accumulatedMeanDepthWeight += wi;
                            accumulatedMeanDepth += wi * zi;

                            if (!medianFound && (accumulatedCompositeWeight + wi) >= 0.5f) {
                                medianFound = true;
                                medianDepth = zi;
                                medianWorldPosition = worldHit.hitPositionW;
                                medianNormalW = normalW;
                            }

                            accumulatedCompositeWeight += wi;
                            // Depth distortion
                            const float mi = depthDistortionNdc01(zi);
                            distortion += wi * (mi * mi * prefixWeight + prefixWeightDepthSquared - 2.0f * mi *
                                                prefixWeightDepth);
                            prefixWeight += wi;
                            prefixWeightDepth += wi * mi;
                            prefixWeightDepthSquared += wi * mi * mi;
                            transmittance *= (1.0f - alphaEff);
                            primaryRay.origin = worldHit.hitPositionW + primaryRay.direction * RayEpsilon;
                            continue;
                        }
                        // -------------------------------------------------------------
                        // Terminal mesh hit
                        // -------------------------------------------------------------
                        if (instance.geometryType == GeometryType::Mesh) {
                            const GPUMaterial &material = scene.materials[instance.materialIndex];
                            const bool isBackfaceHit = dot(primaryRay.direction, worldHit.geometricNormalW) > 0.0f;
                            const float3 normalW = isBackfaceHit
                                                       ? -worldHit.geometricNormalW
                                                       : worldHit.geometricNormalW;
                            // Treat terminal mesh as opaque for median-depth purposes.
                            {
                                const float wi = transmittance;
                                const float zi = dot(worldHit.hitPositionW - sensor.camera.pos, sensor.camera.forward);

                                accumulatedMeanDepthWeight += wi;
                                accumulatedMeanDepth += wi * zi;

                                if (!medianFound && (accumulatedCompositeWeight + wi) >= 0.5f) {
                                    medianFound = true;
                                    medianDepth = zi;
                                    medianWorldPosition = worldHit.hitPositionW;
                                    medianNormalW = normalW;
                                }

                                accumulatedCompositeWeight += wi;
                            }
                            if (material.isEmissive()) {
                                const float3 emittedRadiance = material.power * material.baseColor;
                                accumulatedRadianceRGB +=
                                        transmittance * min(emittedRadiance, 1.0f);
                            } else {
                                const float3 indirectIrradiance = gatherDiffuseIrradianceAtPoint(
                                    worldHit.hitPositionW, normalW, photonMap);
                                const float3 indirectRadiance = (material.baseColor * M_1_PIf) * indirectIrradiance;
                                //const float3 directRadiance =
                                //        estimateDirectAreaLightAtDiffuseSurface(
                                //            scene, worldHit.hitPositionW, normalW, material.baseColor, settings, rng);

                                const float3 directRadiance = estimateDirectPointSampledPointLights(
                                    scene,
                                    settings,
                                    worldHit.hitPositionW,
                                    normalW,
                                    material.baseColor);

                                const float3 outgoingRadiance = indirectRadiance + directRadiance;
                                accumulatedRadianceRGB += transmittance * outgoingRadiance;
                            }
                            transmittance = 0.0f;
                            break;
                        }
                    }
                    const std::uint32_t framebufferIndex = pixelY * imageWidth + pixelX;
                    //accumulatedRadianceRGB *= cameraCosine;
                    const float4 currentValue(accumulatedRadianceRGB.x(), accumulatedRadianceRGB.y(),
                                              accumulatedRadianceRGB.z(), 1.0f);
                    sensor.framebuffer[framebufferIndex] += currentValue;
                    sensor.depthDistortionBuffer[pixelIndex] = distortion;
                    if (accumulatedMeanDepthWeight > 1.0e-6f) {
                        sensor.meanDepthBuffer[pixelIndex] =
                                accumulatedMeanDepth / accumulatedMeanDepthWeight;
                    } else {
                        sensor.meanDepthBuffer[pixelIndex] = 0.0f;
                    }

                    if (medianFound) {
                        sensor.medianDepthBuffer[pixelIndex] = medianDepth;
                        sensor.medianWorldPositionBuffer[pixelIndex] = float4{
                            medianWorldPosition.x(), medianWorldPosition.y(), medianWorldPosition.z(), 1.0f
                        };
                        sensor.visibleNormalBuffer[pixelIndex] = float4{
                            medianNormalW.x(), medianNormalW.y(), medianNormalW.z(), 1.0f
                        };
                    } else {
                        sensor.medianDepthBuffer[pixelIndex] = 0.0f;
                        sensor.medianWorldPositionBuffer[pixelIndex] = float4{0.0f};
                        sensor.visibleNormalBuffer[pixelIndex] = float4{0.0f};
                    }
                });
        });

        queue.wait();

        // -------------------------------------------------------------------------
        // Pass 2:
        //   Normal from 2DGS-style pseudo surface depth map
        // -------------------------------------------------------------------------
        if (settings.normalConsistencyWeight) {
            queue.submit([&](sycl::handler &cgh) {
                cgh.parallel_for<class SurfaceDepthNormalKernel>(
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
                        // Same meaning as 2DGS pipe.depth_ratio:
                        //
                        //   depthRatio = 1 -> use median depth for bounded scenes
                        //   depthRatio = 0 -> use expected / mean depth for unbounded scenes
                        //
                        // Add this setting if you do not already have it.
                        const float depthRatio = 1;
                        const float zMedianC = sensor.medianDepthBuffer[pixelIndex];
                        const float zMedianL = sensor.medianDepthBuffer[idxL];
                        const float zMedianR = sensor.medianDepthBuffer[idxR];
                        const float zMedianU = sensor.medianDepthBuffer[idxU];
                        const float zMedianD = sensor.medianDepthBuffer[idxD];

                        const float zMeanC = sensor.meanDepthBuffer[pixelIndex];
                        const float zMeanL = sensor.meanDepthBuffer[idxL];
                        const float zMeanR = sensor.meanDepthBuffer[idxR];
                        const float zMeanU = sensor.meanDepthBuffer[idxU];
                        const float zMeanD = sensor.meanDepthBuffer[idxD];

                        const float zC = zMeanC * (1.0f - depthRatio) + zMedianC * depthRatio;
                        const float zL = zMeanL * (1.0f - depthRatio) + zMedianL * depthRatio;
                        const float zR = zMeanR * (1.0f - depthRatio) + zMedianR * depthRatio;
                        const float zU = zMeanU * (1.0f - depthRatio) + zMedianU * depthRatio;
                        const float zD = zMeanD * (1.0f - depthRatio) + zMedianD * depthRatio;

                        if (zC <= 0.0f || zL <= 0.0f || zR <= 0.0f || zU <= 0.0f || zD <= 0.0f) {
                            sensor.normalFromDepthBuffer[pixelIndex] = float4{0.0f, 0.0f, 0.0f, 0.0f};
                            return;
                        }
                        const float3 pL = reconstructWorldPositionFromDepthCenter(sensor.camera, x - 1u, y, zL);
                        const float3 pR = reconstructWorldPositionFromDepthCenter(sensor.camera, x + 1u, y, zR);
                        const float3 pU = reconstructWorldPositionFromDepthCenter(sensor.camera, x, y - 1u, zU);
                        const float3 pD = reconstructWorldPositionFromDepthCenter(sensor.camera, x, y + 1u, zD);
                        // 2DGS depth_to_normal:
                        //
                        //   dx = points[2:, 1:-1] - points[:-2, 1:-1]
                        //   dy = points[1:-1, 2:] - points[1:-1, :-2]
                        //   normal = normalize(cross(dx, dy))
                        //
                        // With image coordinates:
                        //
                        //   tangentY = P(y + 1, x) - P(y - 1, x)
                        //   tangentX = P(y, x + 1) - P(y, x - 1)
                        //
                        const float3 tangentY = pD - pU;
                        const float3 tangentX = pR - pL;

                        const float tangentYLengthSquared = dot(tangentY, tangentY);
                        const float tangentXLengthSquared = dot(tangentX, tangentX);
                        if (tangentYLengthSquared <= 1.0e-16f || tangentXLengthSquared <= 1.0e-16f) {
                            sensor.normalFromDepthBuffer[pixelIndex] =
                                    float4{0.0f, 0.0f, 0.0f, 0.0f};
                            return;
                        }
                        // Match 2DGS cross-product order exactly:
                        //
                        //   normal = normalize(cross(tangentY, tangentX))
                        //
                        // Do not flip toward the camera if you want exact 2DGS behavior.
                        const float3 normalW = normalize(cross(tangentY, tangentX));
                        sensor.normalFromDepthBuffer[pixelIndex] =
                                float4{normalW.x(), normalW.y(), normalW.z(), 1.0f};
                    });
            });

            queue.wait();
        }
    }


    /*
    void launchPointSampledPathTracingCameraKernel(
        RenderPackage& pkg,
        uint32_t cameraIndex,
        uint32_t sampleIndex) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& settings = pkg.settings;
        const SensorGPU sensor = pkg.sensors[cameraIndex];
        const uint32_t imageWidth = sensor.camera.width;
        const uint32_t imageHeight = sensor.camera.height;
        const uint32_t pixelCount = imageWidth * imageHeight;
        if (sampleIndex == 0u) {
            queue.fill(sensor.framebuffer, float4{0.0f}, pixelCount);
            queue.fill(sensor.medianDepthBuffer, 0.0f, pixelCount);
            queue.fill(sensor.meanDepthBuffer, 0.0f, pixelCount);
            queue.fill(sensor.medianWorldPositionBuffer, float4{0.0f}, pixelCount);
            queue.fill(sensor.visibleNormalBuffer, float4{0.0f}, pixelCount);
            queue.fill(sensor.normalFromDepthBuffer, float4{0.0f}, pixelCount);
            queue.fill(sensor.depthDistortionBuffer, 0.0f, pixelCount);
            queue.fill(sensor.visibilityWeightedOpacityBuffer, 0.0f, pixelCount);
            queue.wait();
        }

        queue.submit([&](sycl::handler& cgh) {
            const uint64_t renderSeed = pkg.random.seed;
            const uint32_t totalSamples = sycl::max(1u, settings.numGatherPasses);
            const float inverseTotalSamples = 1.0f / static_cast<float>(totalSamples);

            cgh.parallel_for<class PointSampledCylinderCameraGatherKernel>(
                sycl::range<1>(pixelCount),
                [=](sycl::id<1> tid) {
                    const std::uint32_t pixelIndex = tid[0];
                    const std::uint32_t pixelX = pixelIndex % imageWidth;
                    const std::uint32_t pixelY = pixelIndex / imageWidth;
                    const uint64_t directionSeed = rng::makeSeed(renderSeed, pixelIndex, cameraIndex,
                                                                 rng::kStreamGather, sampleIndex);
                    rng::Xorshift128 rng(directionSeed);
                    float3 accumulatedRadianceRGB(0.0f, 0.0f, 0.0f);
                    Ray primaryRay = makePrimaryRayFromPixelJitteredFov(
                        sensor.camera,
                        static_cast<float>(pixelX),
                        static_cast<float>(pixelY),
                        0.0f,
                        0.0f
                    );
                    float transmittance = 1.0f;
                    float distortion = 0.0f;
                    float prefixWeight = 0.0f;
                    float prefixWeightDepth = 0.0f;
                    float prefixWeightDepthSquared = 0.0f;
                    float visibilityWeightedOpacityLoss = 0.0f;
                    float accumulatedCompositeWeight = 0.0f;
                    bool medianFound = false;
                    float medianDepth = 0.0f;
                    float3 medianWorldPosition(0.0f, 0.0f, 0.0f);
                    float3 medianNormalW(0.0f, 0.0f, 0.0f);
                    float accumulatedMeanDepthWeight = 0.0f;
                    float accumulatedMeanDepth = 0.0f;

                    for (uint32_t traversalIndex = 0u; traversalIndex < kMaxSplatEventsPerRay; ++traversalIndex) {
                        PointSampledSceneHit hit{};
                        if (!intersectScenePointSampledGeometry(primaryRay, scene, settings, hit)) {
                            break;
                        }

                        float alphaEff = sycl::clamp(hit.opacity, 0.0f, 1.0f);
                        if (hit.geometryType == GeometryType::Mesh) {
                            alphaEff = 1.0f;
                        }
                        if (alphaEff <= 0.0f) {
                            primaryRay.origin = primaryRay.origin + primaryRay.direction * (hit.tWorld + RayEpsilon);
                            continue;
                        }

                        float3 normalW = hit.geometricNormalW;
                        if (dot(normalW, -primaryRay.direction) < 0.0f) {
                            normalW = -normalW;
                        }

                        float3 outgoingRadiance = hit.emittedRadiance;
                        if (settings.pointGeometryDebugShowAlbedo) {
                            outgoingRadiance = hit.albedo;
                            //alphaEff = 1.0f;
                        } else if (!hit.isEmissive || hit.geometryType == GeometryType::PointCloud) {
                            outgoingRadiance += estimateDirectPointSampledAreaLight(
                                scene,
                                settings,
                                hit.hitPositionW,
                                normalW,
                                hit.albedo,
                                rng);
                        }

                        accumulatedRadianceRGB += transmittance * alphaEff * outgoingRadiance;

                        const float compositeWeight = transmittance * alphaEff;
                        const float depth = dot(hit.hitPositionW - sensor.camera.pos, sensor.camera.forward);
                        accumulatedMeanDepthWeight += compositeWeight;
                        accumulatedMeanDepth += compositeWeight * depth;
                        const float opacityResidual = 1.0f - alphaEff;
                        visibilityWeightedOpacityLoss += compositeWeight * opacityResidual * opacityResidual;
                        if (!medianFound && (accumulatedCompositeWeight + compositeWeight) >= 0.5f) {
                            medianFound = true;
                            medianDepth = depth;
                            medianWorldPosition = hit.hitPositionW;
                            medianNormalW = normalW;
                        }
                        accumulatedCompositeWeight += compositeWeight;

                        const float normalizedDepth = depthDistortionNdc01(depth);
                        distortion += compositeWeight * (
                            normalizedDepth * normalizedDepth * prefixWeight +
                            prefixWeightDepthSquared -
                            2.0f * normalizedDepth * prefixWeightDepth);
                        prefixWeight += compositeWeight;
                        prefixWeightDepth += compositeWeight * normalizedDepth;
                        prefixWeightDepthSquared += compositeWeight * normalizedDepth * normalizedDepth;

                        transmittance *= 1.0f - alphaEff;
                        if (transmittance <= 1.0e-4f || alphaEff >= 1.0f) {
                            break;
                        }

                        const float rayOffset = RayEpsilon;
                        primaryRay.origin = primaryRay.origin + primaryRay.direction * (hit.tWorld + rayOffset);
                    }
                    const std::uint32_t framebufferIndex = pixelY * imageWidth + pixelX;
                    const float4 currentValue(
                        accumulatedRadianceRGB.x() * inverseTotalSamples,
                        accumulatedRadianceRGB.y() * inverseTotalSamples,
                        accumulatedRadianceRGB.z() * inverseTotalSamples,
                        1.0f);
                    sensor.framebuffer[framebufferIndex] += currentValue;
                    sensor.depthDistortionBuffer[pixelIndex] += distortion * inverseTotalSamples;
                    sensor.visibilityWeightedOpacityBuffer[pixelIndex] +=
                        visibilityWeightedOpacityLoss * inverseTotalSamples;
                    if (accumulatedMeanDepthWeight > 1.0e-6f) {
                        sensor.meanDepthBuffer[pixelIndex] = accumulatedMeanDepth / accumulatedMeanDepthWeight;
                    }
                    else {
                        sensor.meanDepthBuffer[pixelIndex] = 0.0f;
                    }

                    if (medianFound) {
                        sensor.medianDepthBuffer[pixelIndex] = medianDepth;
                        sensor.medianWorldPositionBuffer[pixelIndex] = float4{
                            medianWorldPosition.x(), medianWorldPosition.y(), medianWorldPosition.z(), 1.0f
                        };
                        sensor.visibleNormalBuffer[pixelIndex] = float4{
                            medianNormalW.x(), medianNormalW.y(), medianNormalW.z(), 1.0f
                        };
                    }
                    else {
                        sensor.medianDepthBuffer[pixelIndex] = 0.0f;
                        sensor.medianWorldPositionBuffer[pixelIndex] = float4{0.0f};
                        sensor.visibleNormalBuffer[pixelIndex] = float4{0.0f};
                    }
                }
            );
        });

        queue.wait();
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
        sycl::queue& queue,
        DeviceSurfacePhotonMapGrid grid,
        std::uint32_t photonCount) {
        queue.submit([&](sycl::handler& cgh) {
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

    void clearCellArrays(sycl::queue& queue, DeviceSurfacePhotonMapGrid grid) {
        static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;

        const std::uint32_t cellCount = grid.totalCellCount;
        queue.submit([&](sycl::handler& cgh) {
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
        sycl::queue& queue,
        DeviceSurfacePhotonMapGrid grid,
        std::uint32_t photonCount) {
        static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;

        queue.submit([&](sycl::handler& cgh) {
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
        sycl::queue& queue,
        DeviceSurfacePhotonMapGrid grid,
        std::uint32_t photonCount) {
        static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;

        queue.submit([&](sycl::handler& cgh) {
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
        sycl::queue& queue,
        DeviceSurfacePhotonMapGrid grid) {
        const std::uint32_t totalCellCount = grid.totalCellCount;
        const std::uint32_t blockSize = kScanBlockSize;
        const std::uint32_t blockCount = (totalCellCount + blockSize - 1u) / blockSize;

        std::uint32_t* cellCount = grid.cellCount;
        std::uint32_t* cellStart = grid.cellStart;
        std::uint32_t* blockSums = grid.blockSums;
        std::uint32_t* blockPrefix = grid.blockPrefix;

        // Pass 1: per-block exclusive scan into cellStart + write block sums
        queue.submit([&](sycl::handler& cgh) {
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
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(totalCellCount), [=](sycl::id<1> idx) {
                const std::uint32_t globalIndex = static_cast<std::uint32_t>(idx[0]);
                const std::uint32_t blockIndex = globalIndex / blockSize;
                cellStart[globalIndex] += blockPrefix[blockIndex];
            });
        }).wait();
    }

    void finalizeCellRanges(
        sycl::queue& queue,
        DeviceSurfacePhotonMapGrid grid) {
        static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;

        const std::uint32_t totalCellCount = grid.totalCellCount;

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(totalCellCount), [=](sycl::id<1> idx) {
                const std::uint32_t c = static_cast<std::uint32_t>(idx[0]);
                const std::uint32_t count = grid.cellCount[c];

                grid.cellWriteOffset[c] = 0u;

                if (count == 0u) {
                    grid.cellStart[c] = kInvalidIndex;
                    grid.cellEnd[c] = kInvalidIndex;
                }
                else {
                    const std::uint32_t start = grid.cellStart[c];
                    grid.cellEnd[c] = start + count;
                }
            });
        }).wait();
    }


    void buildPhotonCellRangesAndOrdering(
        sycl::queue& queue,
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
