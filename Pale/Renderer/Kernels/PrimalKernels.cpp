//
// Created by magnus on 9/12/25.
//
#include "PrimalKernels.h"
#include "IntersectionKernels.h"
#include "KernelHelpers.h"
#include <cmath>
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
    const uint64_t forwardPassIndex = uint64_t(forwardPass);
    sycl::event kernelEvent1 = queue.parallel_for<struct RayGenEmitterKernelTag>(sycl::range<1>(settings.photonsPerLaunch), [=](sycl::id<1> globalId) {
        const uint64_t photonIndex = uint64_t(globalId[0]);
        const uint64_t pathId = forwardPassIndex * uint64_t(settings.photonsPerLaunch) + photonIndex;
        const uint64_t seed = rng::makeSeed(renderSeed, pathId, 0u, rng::kStreamRayGen, 0u);
        rng::Xorshift128 rng128(seed);
        if (scene.lightCount == 0) { return; }
        AreaLightSample ls = sampleMeshAreaLight(scene, rng128);
        if (!ls.valid) { return; }
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
        auto counter = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(*countPrimary);
        const uint32_t slot = counter.fetch_add(1u);
        raysIn[slot] = ray;
    });
    kernelEvent1.wait();
}
void launchIntersectKernel(RenderPackage &pkg, uint32_t activeRayCount) {
    auto &queue = pkg.queue;
    auto &scene = pkg.scene;
    auto &settings = pkg.settings;
    auto &intermediates = pkg.intermediates;
    const uint64_t renderSeed = pkg.random.seed;
    sycl::event kernelEvent2 = queue.parallel_for<class launchIntersectKernel>(sycl::range<1>(activeRayCount), [=](sycl::id<1> globalId) {
        const uint32_t rayIndex = globalId[0];
        RayState currentRayState = intermediates.primaryRays[rayIndex];
        // Guard against pathological transparent stacks or self-intersection loops.
        constexpr uint32_t maxInlineNullTraversals = 32;
        for (uint32_t inlineTraversalIndex = 0; inlineTraversalIndex < maxInlineNullTraversals; ++inlineTraversalIndex) {
            const uint64_t stepSeed = rng::makeSeed(renderSeed, currentRayState.pathId, currentRayState.traversalIndex, rng::kStreamTraversal, 107u);
            rng::Xorshift128 stepRng(stepSeed);
            WorldHit worldHit{};
            intersectScene(currentRayState.ray, &worldHit, scene, SurfelIntersectMode::FirstHit);
            if (!worldHit.hit) { return; }
            bool isIndirect = currentRayState.bounceIndex > 0;
            buildIntersectionNormal(scene, worldHit);
            const InstanceRecord &instance = scene.instances[worldHit.instanceIndex];
            // ---------------------------------------------------------------------
            // Mesh: this is a real scattering event, so it finishes this kernel call.
            // ---------------------------------------------------------------------
            if (instance.geometryType == GeometryType::Mesh) {
                bool isBackfaceHit = dot(currentRayState.ray.direction, worldHit.geometricNormalW) > 0.0f;
                if (isBackfaceHit) { worldHit.geometricNormalW *= -1.0f; }
                if (settings.integratorKind == IntegratorKind::lightTracing) {
                    HitInfoContribution contribution{};
                    contribution.geometricNormalW = worldHit.geometricNormalW;
                    contribution.hitPositionW = worldHit.hitPositionW;
                    contribution.instanceIndex = worldHit.instanceIndex;
                    contribution.pathThroughput = currentRayState.pathThroughput;
                    contribution.type = instance.geometryType;
                    contribution.primitiveIndex = worldHit.primitiveIndex;
                    appendContributionAtomic(intermediates.countContributions, intermediates.hitContribution, intermediates.maxHitContributionCount, contribution);
                }
                const GPUMaterial material = scene.materials[instance.materialIndex];
                if (settings.integratorKind == IntegratorKind::photonMapping && isIndirect && !material.isEmissive()) {
                    depositPhotonSurface(worldHit, currentRayState.ray.direction, worldHit.geometricNormalW, currentRayState.pathThroughput, intermediates.map);
                }
                float3 sampledOutgoingDirectionW = currentRayState.ray.direction;
                float sampledPdf = 0.0f;
                sampleCosineHemisphere(stepRng, worldHit.geometricNormalW, sampledOutgoingDirectionW, sampledPdf);
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
                if (!applyRussianRoulette(stepRng, nextRayState.bounceIndex, nextRayState.pathThroughput, settings.russianRouletteStart)) { return; }
                auto extensionCounter = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(*intermediates.countExtensionOut);
                const uint32_t outIndex = extensionCounter.fetch_add(1u);
                if (outIndex < intermediates.maxRayQueueCapacity) { intermediates.extensionRaysA[outIndex] = nextRayState; }
                return;
            }
            // ---------------------------------------------------------------------
            // Point cloud: alpha controls whether the ray transmits or scatters.
            // ---------------------------------------------------------------------
            if (instance.geometryType == GeometryType::PointCloud) {
                const Point &surfel = scene.points[worldHit.primitiveIndex];
                const float effectiveOpacity = sycl::fmin(1.0f, sycl::fmax(0.0f, worldHit.alphaGeom * surfel.opacity));
                const float scatterProbability = settings.sampling.qReflect;
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
                    currentRayState.ray.origin = worldHit.hitPositionW + currentRayState.ray.direction * RayEpsilon;
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
                if (scatterProbability <= 0.0f) { return; }
                const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));
                const float signedCosineIncident = dot(canonicalNormalW, -currentRayState.ray.direction);
                const int sideSign = signNonZero(signedCosineIncident);
                const float3 orientedNormal = static_cast<float>(sideSign) * canonicalNormalW;
                if (settings.integratorKind == IntegratorKind::photonMapping && isIndirect && !surfel.isEmissive()) {
                    depositPhotonSurface(worldHit, currentRayState.ray.direction, orientedNormal, currentRayState.pathThroughput / scatterProbability, intermediates.map);
                }
                float3 sampledOutgoingDirectionW = currentRayState.ray.direction;
                float sampledPdf = 0.0f;
                sampleCosineHemisphere(stepRng, orientedNormal, sampledOutgoingDirectionW, sampledPdf);
                const float3 reflectanceMultiplier = surfel.alpha_r * surfel.albedo;
                const float3 nextPathThroughput = currentRayState.pathThroughput * reflectanceMultiplier;
                if (settings.integratorKind == IntegratorKind::lightTracing) {
                    HitInfoContribution contribution{};
                    contribution.hitPositionW = worldHit.hitPositionW;
                    contribution.geometricNormalW = orientedNormal;
                    contribution.instanceIndex = worldHit.instanceIndex;
                    contribution.pathThroughput = nextPathThroughput;
                    contribution.type = instance.geometryType;
                    contribution.primitiveIndex = worldHit.primitiveIndex;
                    contribution.eventType = EventType::Reflect;
                    appendContributionAtomic(intermediates.countContributions, intermediates.hitContribution, intermediates.maxHitContributionCount, contribution);
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
                if (!applyRussianRoulette(stepRng, nextRayState.bounceIndex, nextRayState.pathThroughput, settings.russianRouletteStart)) { return; }
                auto extensionCounter = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(*intermediates.countExtensionOut);
                const uint32_t outIndex = extensionCounter.fetch_add(1u);
                if (outIndex < intermediates.maxRayQueueCapacity) { intermediates.extensionRaysA[outIndex] = nextRayState; }
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
    kernelEvent2.wait();
}
void launchCameraGatherKernel2(RenderPackage &pkg, uint32_t cameraIndex, uint32_t gatherPass) {
    auto &queue = pkg.queue;
    auto &scene = pkg.scene;
    auto &settings = pkg.settings;
    auto &photonMap = pkg.intermediates.map;
    SensorGPU sensor = pkg.sensors[cameraIndex];
    const uint32_t imageWidth = sensor.camera.width;
    const uint32_t imageHeight = sensor.camera.height;
    const uint32_t pixelCount = imageWidth * imageHeight;
    queue.fill(sensor.framebuffer, float4{0.0f}, pixelCount).wait();
    queue.fill(sensor.medianDepthBuffer, 0.0f, pixelCount).wait();
    queue.fill(sensor.meanDepthBuffer, 0.0f, pixelCount).wait();
    queue.fill(sensor.medianWorldPositionBuffer, float4{0.0f}, pixelCount).wait();
    queue.fill(sensor.visibleNormalBuffer, float4{0.0f}, pixelCount).wait();
    queue.fill(sensor.normalFromDepthBuffer, float4{0.0f}, pixelCount).wait();
    queue.fill(sensor.depthDistortionBuffer, 0.0f, pixelCount).wait();
    queue.fill(sensor.depthDistortionAdjointBuffer, 0.0f, pixelCount).wait();
    queue.fill(sensor.intraSlabDepthBuffer, 0.0f, pixelCount).wait();
    queue.fill(sensor.intraSlabDepthAdjointBuffer, 0.0f, pixelCount).wait();
    queue.fill(sensor.intraSlabDepthActiveSlabCountBuffer, 0u, pixelCount).wait();
    queue.fill(sensor.curvatureScaleBuffer, 0.0f, pixelCount).wait();
    queue.fill(sensor.curvatureScaleAdjointBuffer, 0.0f, pixelCount).wait();
    queue.fill(sensor.curvatureScaleActiveSlabCountBuffer, 0u, pixelCount).wait();
    queue.fill(sensor.visibilityWeightedOpacityBuffer, 0.0f, pixelCount).wait();
    sycl::event kernelEvent3 = queue.parallel_for<class CameraGatherKernel>(sycl::range<1>(pixelCount), [=](sycl::id<1> tid) {
        constexpr float kAlphaEpsilon = 1.0e-8f;
        const uint32_t pixelIndex = static_cast<uint32_t>(tid[0]);
        const uint32_t pixelX = pixelIndex % imageWidth;
        const uint32_t pixelY = pixelIndex / imageWidth;
        const Ray originalRay = makePrimaryRayFromPixelJitteredFov(sensor.camera, static_cast<float>(pixelX), static_cast<float>(pixelY), 0.0f, 0.0f);
        const float localLayerDepthEpsilon = rendererDebugLocalLayerDepthEpsilon(settings);
        const uint32_t maxSplatEventsPerRay = rendererDebugMaxSplatEventsPerRay(settings);
        const uint32_t maxLocalSurfelHits = rendererDebugMaxLocalSurfelHits(settings);
        const uint32_t pointHitBatchSize = rendererDebugPointHitBatchSize(settings);
        const uint32_t pointHitBatchLookaheadCapacity =
            rendererDebugPointHitBatchLookaheadCapacity(settings);
        const float localLayerNormalCosineThreshold = rendererDebugLocalLayerNormalCosineThreshold(settings);
        const bool shareLocalLayerDirectLighting = settings.rendererDebugShareLocalLayerDirectLighting;
        const bool profileEnabled = scene.profileCounters != nullptr;
        uint64_t profilePointHitQueries = 0u;
        uint64_t profilePointHitCandidates = 0u;
        uint64_t profileLocalLayers = 0u;
        uint64_t profileLocalLayerHits = 0u;
        uint64_t profileObjectProfileHits = 0u;
        uint64_t profileLowPassProfileHits = 0u;
        uint64_t profileRegularizerHits = 0u;
        uint64_t profilePhotonGatherCalls = 0u;
        uint64_t profileDirectLightCalls = 0u;
        uint64_t profileDirectLightLightVisits = 0u;
        uint64_t profileDepthPairIterations = 0u;
        uint64_t profileMeshHits = 0u;
        uint64_t profileNoHitTerminations = 0u;
        uint64_t profileOpacityTerminations = 0u;
        uint64_t profileMaxSplatTerminations = 0u;
        bool profileStoppedByNoHit = false;
        bool profileStoppedByOpacity = false;
        const MinimumProjectedFootprintFilter minimumFootprintFilter =
            minimumProjectedFootprintFilterFromSettings(
                settings,
                sensor.camera,
                float2{static_cast<float>(pixelX), static_cast<float>(pixelY)});
        uint32_t directPointInstanceIndex = kInvalidIndex;
        const bool canUsePointHitBatches =
            pointHitBatchSize > 1u &&
            tryGetSinglePointCloudInstance(scene, directPointInstanceIndex);
        // =====================================================================
        // PASS A: physical rendering.
        //
        // Keep the existing symmetric slab definition here. The slab is part
        // of the radiance / transmission model, not the surface regularizer.
        // =====================================================================
        Ray renderingRay = originalRay;
        float3 accumulatedRadianceRGB{0.0f};
        float renderingTransmittance = 1.0f;
        float accumulatedRenderingWeight = 0.0f;
        // =====================================================================
        // Individual-surfel surface quantities.
        //
        // Rendering below still uses symmetric slabs. The regularizer consumes
        // individual front-to-back surfel hits from the same traversal stream.
        // =====================================================================
        float regularizerTransmittance = 1.0f;
        float accumulatedRegularizerWeight = 0.0f;
        float accumulatedWeightedDepth = 0.0f;
        float3 accumulatedWeightedNormal{0.0f};
        float visibilityWeightedOpacityLoss = 0.0f;
        float intraSlabDepthLossSum = 0.0f;
        uint32_t intraSlabDepthActiveSlabCount = 0u;
        float previousDepthDistortionWeights[kMaxSplatEventsPerRay];
        float previousDepthDistortionNdcDepths[kMaxSplatEventsPerRay];
        uint32_t previousDepthDistortionHitCount = 0u;
        uint32_t regularizerHitIndex = 0u;
        float distortion = 0.0f;
        float accumulatedCompositeWeight = 0.0f;
        bool medianFound = false;
        float medianDepth = 0.0f;
        float3 medianWorldPosition{0.0f};
        bool keepTracingRegularizer = true;

        auto accumulateRegularizerHit =
            [&](const LocalSurfelLayerHit &regularizerHit) -> bool {
            if (regularizerHit.primitiveIndex == kInvalidIndex) {
                return true;
            }
            const Point &surfel = scene.points[regularizerHit.primitiveIndex];
            const float alphaGeom = regularizerHit.alphaGeom;
            const float alpha = surfel.opacity * alphaGeom;
            if (alphaGeom <= kAlphaEpsilon) {
                return true;
            }
            const float depth = dot(regularizerHit.hitPositionW - sensor.camera.pos, sensor.camera.forward);
            if (depth <= 0.0f) {
                return true;
            }
            const float opacityResidual = 1.0f - surfel.opacity;
            visibilityWeightedOpacityLoss +=
                    regularizerTransmittance * alphaGeom * opacityResidual * opacityResidual;
            if (alpha <= kAlphaEpsilon) {
                return true;
            }

            const float compositeWeight = regularizerTransmittance * alpha;
            accumulatedRegularizerWeight += compositeWeight;
            accumulatedWeightedDepth += compositeWeight * depth;

            if (!medianFound && accumulatedCompositeWeight + compositeWeight >= 0.5f) {
                medianFound = true;
                medianDepth = depth;
                medianWorldPosition = regularizerHit.hitPositionW;
            }
            accumulatedCompositeWeight += compositeWeight;

            const float ndcDepth = depthDistortionNdc01(depth);
            if (profileEnabled) {
                profileRegularizerHits += 1u;
                profileDepthPairIterations += previousDepthDistortionHitCount;
            }
            for (uint32_t previousIndex = 0u; previousIndex < previousDepthDistortionHitCount; ++previousIndex) {
                const float depthDifference = ndcDepth - previousDepthDistortionNdcDepths[previousIndex];
                distortion += previousDepthDistortionWeights[previousIndex] * compositeWeight *
                              depthDifference * depthDifference;
            }
            if (previousDepthDistortionHitCount < kMaxSplatEventsPerRay) {
                previousDepthDistortionWeights[previousDepthDistortionHitCount] = compositeWeight;
                previousDepthDistortionNdcDepths[previousDepthDistortionHitCount] = ndcDepth;
                ++previousDepthDistortionHitCount;
            }

            float3 orientedNormalW = normalize(cross(surfel.tanU, surfel.tanV));
            if (dot(orientedNormalW, -originalRay.direction) < 0.0f) { orientedNormalW = -orientedNormalW; }
            accumulatedWeightedNormal += compositeWeight * orientedNormalW;

            regularizerTransmittance *= 1.0f - alpha;
            return regularizerTransmittance > kAlphaEpsilon;
        };

        auto renderPointLocalLayer = [&](const PointCloudLocalLayer &localLayer, const Ray &layerRay) {
            const float slabOpacity = localLayer.opacity;
            if (profileEnabled) {
                profileLocalLayers += 1u;
                profileLocalLayerHits += localLayer.hitCount;
                for (uint32_t localHitIndex = 0u; localHitIndex < localLayer.hitCount; ++localHitIndex) {
                    if (localLayer.hits[localHitIndex].alphaProfileBranch == kSurfelAlphaProfileLowPass) {
                        profileLowPassProfileHits += 1u;
                    } else {
                        profileObjectProfileHits += 1u;
                    }
                }
            }
            if (localLayer.hitCount == 0u) {
                return;
            }

            const LocalSurfelLayerHit &anchorHit = localLayer.hits[0];
            const float3 anchorPositionW = anchorHit.hitPositionW;
            const PointCloudLocalLayerConsensus slabConsensus =
                    computePointCloudLocalLayerConsensus(localLayer, originalRay, scene);
            const float3 sharedSlabPositionW =
                    slabConsensus.valid != 0u ? slabConsensus.pointW : anchorPositionW;

            // Full, unblended point-to-plane consensus constraint:
            //
            //   x_Q = o + d * B_Q/A_Q
            //   delta_i = n_i . (sg(x_Q) - p_i)
            //   L_Q = 1/|Q| sum_i (delta_i/h)^2.
            //
            // A_Q and B_Q are reconstructed every forward pass, but x_Q is a
            // detached target in the corresponding adjoint.
            if (localLayer.hitCount > 1u && slabConsensus.valid != 0u) {
                const float inverseMemberCount =
                        1.0f / static_cast<float>(localLayer.hitCount);
                const float inverseSlabThicknessSquared =
                        1.0f / (localLayerDepthEpsilon * localLayerDepthEpsilon);
                float slabLoss = 0.0f;
                for (uint32_t localHitIndex = 0u;
                     localHitIndex < localLayer.hitCount;
                     ++localHitIndex) {
                    const LocalSurfelLayerHit &localHit = localLayer.hits[localHitIndex];
                    const Point &memberSurfel = scene.points[localHit.primitiveIndex];
                    const float3 memberNormalW =
                            normalize(cross(memberSurfel.tanU, memberSurfel.tanV));
                    const float signedPlaneDistance =
                            dot(memberNormalW, sharedSlabPositionW - memberSurfel.position);
                    slabLoss += signedPlaneDistance * signedPlaneDistance *
                                inverseSlabThicknessSquared;
                }
                intraSlabDepthLossSum += slabLoss * inverseMemberCount;
                ++intraSlabDepthActiveSlabCount;
            }

            float3 anchorNormalW = localLayer.referenceNormalW;
            if (dot(anchorNormalW, anchorNormalW) <= 1.0e-16f) {
                const Point &anchorSurfel = scene.points[anchorHit.primitiveIndex];
                anchorNormalW = normalize(cross(anchorSurfel.tanU, anchorSurfel.tanV));
                if (dot(anchorNormalW, -layerRay.direction) < 0.0f) anchorNormalW = -anchorNormalW;
            }
            const float sharedDirectLightEpsilon = localLayer.directLightEpsilon[0];

            for (uint32_t localHitIndex = 0u; localHitIndex < localLayer.hitCount; ++localHitIndex) {
                const float layerWeight = localLayer.weight[localHitIndex];
                if (layerWeight <= 0.0f) continue;
                const LocalSurfelLayerHit &localHit = localLayer.hits[localHitIndex];
                const Point &surfel = scene.points[localHit.primitiveIndex];
                float3 normalW = normalize(cross(surfel.tanU, surfel.tanV));
                if (dot(normalW, -layerRay.direction) < 0.0f) normalW = -normalW;
                const float compositeWeight = renderingTransmittance * layerWeight;
                if (profileEnabled) {
                    profilePhotonGatherCalls += 1u;
                }
                const float3 indirectIrradiance = gatherDiffuseIrradianceAtPoint(localHit.hitPositionW, normalW, photonMap);
                const float3 indirectRadiance = indirectIrradiance * (surfel.alpha_r * surfel.albedo * M_1_PIf);
                const float surfelArea = M_PIf * surfel.scale.x() * surfel.scale.y();
                float3 emittedRadiance{0.0f};
                if (surfelArea > 1.0e-12f && sycl::isfinite(surfelArea)) {
                    emittedRadiance = surfel.albedo * (surfel.flux / (M_PIf * surfelArea));
                }
                float3 directRadiance{0.0f};
                if (!shareLocalLayerDirectLighting) {
                    if (profileEnabled) {
                        profileDirectLightCalls += 1u;
                        profileDirectLightLightVisits += scene.lightCount;
                    }
                    directRadiance = estimateDirectPointSampledPointLights(
                        scene,
                        settings,
                        localHit.hitPositionW,
                        normalW,
                        surfel.alpha_r * surfel.albedo,
                        localLayer.directLightEpsilon[localHitIndex]);
                }
                accumulatedRadianceRGB += compositeWeight * (emittedRadiance + indirectRadiance + directRadiance);
            }

            if (shareLocalLayerDirectLighting && profileEnabled) {
                profileDirectLightCalls += 1u;
                profileDirectLightLightVisits += scene.lightCount;
            }
            if (!shareLocalLayerDirectLighting) {
                if (slabOpacity > kAlphaEpsilon) { accumulatedRenderingWeight += renderingTransmittance * slabOpacity; }
                renderingTransmittance *= localLayer.transmission;
                return;
            }
            for (uint32_t lightIndex = 0u; lightIndex < scene.lightCount; ++lightIndex) {
                const GPULightRecord &light = scene.lights[lightIndex];
                if (light.lightType != LightType::Surfel) {
                    continue;
                }

                const Point &lightSurfel = scene.points[light.primitiveIndex];
                const float3 lightPositionW = lightSurfel.position;
                const float3 toLight = lightPositionW - sharedSlabPositionW;
                const float distanceSquared = dot(toLight, toLight);
                if (distanceSquared <= 1.0e-12f) {
                    continue;
                }

                const float distance = sycl::sqrt(distanceSquared);
                const float3 lightDirection = toLight / distance;
                const float shadowTransmission =
                    traceShadowTransmissionToPoint(
                        scene,
                        settings,
                        sharedSlabPositionW,
                        anchorNormalW,
                        lightPositionW,
                        sharedDirectLightEpsilon);
                if (shadowTransmission <= 0.0f) {
                    continue;
                }

                const float3 radiantIntensity =
                    light.flux * light.color * (1.0f / (4.0f * M_PIf));
                const float3 sharedIncident =
                    radiantIntensity * shadowTransmission * (1.0f / distanceSquared);

                for (uint32_t localHitIndex = 0u; localHitIndex < localLayer.hitCount; ++localHitIndex) {
                    const float layerWeight = localLayer.weight[localHitIndex];
                    if (layerWeight <= 0.0f) continue;
                    const LocalSurfelLayerHit &localHit = localLayer.hits[localHitIndex];
                    const Point &surfel = scene.points[localHit.primitiveIndex];
                    float3 normalW = normalize(cross(surfel.tanU, surfel.tanV));
                    if (dot(normalW, -layerRay.direction) < 0.0f) normalW = -normalW;
                    const float surfaceCosine = sycl::fmax(0.0f, dot(normalW, lightDirection));
                    if (surfaceCosine <= 0.0f) {
                        continue;
                    }
                    const float3 diffuseBrdf = surfel.alpha_r * surfel.albedo * M_1_PIf;
                    const float compositeWeight = renderingTransmittance * layerWeight;
                    accumulatedRadianceRGB +=
                        compositeWeight * diffuseBrdf * sharedIncident * surfaceCosine;
                }
            }
            if (slabOpacity > kAlphaEpsilon) { accumulatedRenderingWeight += renderingTransmittance * slabOpacity; }
            renderingTransmittance *= localLayer.transmission;
        };

        if (canUsePointHitBatches) {
            for (uint32_t traversalIndex = 0u; traversalIndex < maxSplatEventsPerRay;) {
                LocalSurfelLayerHit pointHits[kMaxPointHitBatchWithLookahead];
                uint32_t pointInstanceIndex = kInvalidIndex;
                const uint32_t hitCount = collectScenePointHitsDirect(
                    renderingRay,
                    scene,
                    RayEpsilon,
                    std::numeric_limits<float>::infinity(),
                    pointHits,
                    pointHitBatchLookaheadCapacity,
                    pointInstanceIndex,
                    minimumFootprintFilter);
                if (profileEnabled) {
                    profilePointHitQueries += 1u;
                    profilePointHitCandidates += hitCount;
                }

                if (hitCount == 0u) {
                    if (profileEnabled) {
                        profileNoHitTerminations += 1u;
                        profileStoppedByNoHit = true;
                    }
                    break;
                }

                const uint32_t coreHitCount =
                    hitCount < pointHitBatchSize ? hitCount : pointHitBatchSize;
                uint32_t hitCursor = 0u;
                float furthestConsumedT = 0.0f;
                while (hitCursor < coreHitCount && traversalIndex < maxSplatEventsPerRay) {
                    const uint32_t oldHitCursor = hitCursor;
                    const LocalSurfelLayerHit &anchorHit = pointHits[hitCursor];
                    const PointCloudLocalLayer localLayer = buildPointCloudLocalLayerFromHits(
                        renderingRay,
                        anchorHit,
                        pointHits + hitCursor,
                        hitCount - hitCursor,
                        scene,
                        localLayerDepthEpsilon,
                        maxLocalSurfelHits,
                        localLayerNormalCosineThreshold);

                    renderPointLocalLayer(localLayer, renderingRay);
                    ++traversalIndex;
                    furthestConsumedT = sycl::fmax(furthestConsumedT, localLayer.furthestT);

                    while (hitCursor < hitCount && pointHits[hitCursor].tWorld <= localLayer.furthestT + RayEpsilon) {
                        ++hitCursor;
                    }
                    if (hitCursor == oldHitCursor) {
                        ++hitCursor;
                    }
                    if (renderingTransmittance <= kAlphaEpsilon) {
                        if (profileEnabled && !profileStoppedByOpacity) {
                            profileOpacityTerminations += 1u;
                            profileStoppedByOpacity = true;
                        }
                        break;
                    }
                }

                if (keepTracingRegularizer) {
                    for (uint32_t batchIndex = 0u;
                         batchIndex < hitCount && regularizerHitIndex < maxSplatEventsPerRay;
                         ++batchIndex) {
                        const LocalSurfelLayerHit &pointHit = pointHits[batchIndex];
                        if (pointHit.tWorld > furthestConsumedT + RayEpsilon) {
                            break;
                        }

                        keepTracingRegularizer = accumulateRegularizerHit(pointHit);
                        ++regularizerHitIndex;
                        if (!keepTracingRegularizer) {
                            break;
                        }
                    }
                }

                const bool consumedAllFetchedHits = hitCursor >= hitCount;
                if (furthestConsumedT <= 0.0f) {
                    break;
                }

                renderingRay.origin += renderingRay.direction * (furthestConsumedT + RayEpsilon);
                if (renderingTransmittance <= kAlphaEpsilon ||
                    (consumedAllFetchedHits && hitCount < pointHitBatchLookaheadCapacity)) {
                    if (profileEnabled && renderingTransmittance <= kAlphaEpsilon && !profileStoppedByOpacity) {
                        profileOpacityTerminations += 1u;
                        profileStoppedByOpacity = true;
                    }
                    if (profileEnabled &&
                        renderingTransmittance > kAlphaEpsilon &&
                        consumedAllFetchedHits &&
                        hitCount < pointHitBatchLookaheadCapacity &&
                        !profileStoppedByNoHit) {
                        profileNoHitTerminations += 1u;
                        profileStoppedByNoHit = true;
                    }
                    break;
                }
            }
            if (profileEnabled &&
                !profileStoppedByNoHit &&
                !profileStoppedByOpacity &&
                renderingTransmittance > kAlphaEpsilon) {
                profileMaxSplatTerminations += 1u;
            }
        } else {
            for (uint32_t traversalIndex = 0u; traversalIndex < maxSplatEventsPerRay; ++traversalIndex) {
                WorldHit worldHit{};
                if (profileEnabled) {
                    profilePointHitQueries += 1u;
                }
                intersectScene(renderingRay, &worldHit, scene, SurfelIntersectMode::FirstHit);
                if (!worldHit.hit) {
                    if (profileEnabled) {
                        profileNoHitTerminations += 1u;
                        profileStoppedByNoHit = true;
                    }
                    break;
                }
                buildIntersectionNormal(scene, worldHit);
                const InstanceRecord &instance = scene.instances[worldHit.instanceIndex];
                if (instance.geometryType == GeometryType::PointCloud) {
                    const PointCloudLocalLayer localLayer = collectPointCloudLocalLayer(
                        renderingRay,
                        worldHit,
                        instance,
                        scene,
                        localLayerDepthEpsilon,
                        maxLocalSurfelHits,
                        localLayerNormalCosineThreshold,
                        minimumFootprintFilter);
                    if (profileEnabled) {
                        profilePointHitCandidates += localLayer.hitCount;
                    }
                    if (keepTracingRegularizer) {
                        for (uint32_t localHitIndex = 0u;
                             localHitIndex < localLayer.hitCount && regularizerHitIndex < maxSplatEventsPerRay;
                             ++localHitIndex) {
                            keepTracingRegularizer = accumulateRegularizerHit(localLayer.hits[localHitIndex]);
                            ++regularizerHitIndex;
                            if (!keepTracingRegularizer) {
                                break;
                            }
                        }
                    }
                    renderPointLocalLayer(localLayer, renderingRay);
                    renderingRay.origin += renderingRay.direction * (localLayer.furthestT + RayEpsilon);
                    continue;
                }
                if (instance.geometryType == GeometryType::Mesh) {
                    if (profileEnabled) {
                        profileMeshHits += 1u;
                    }
                    const GPUMaterial &material = scene.materials[instance.materialIndex];
                    const bool isBackfaceHit = dot(renderingRay.direction, worldHit.geometricNormalW) > 0.0f;
                    const float3 normalW = isBackfaceHit ? -worldHit.geometricNormalW : worldHit.geometricNormalW;
                    accumulatedRenderingWeight += renderingTransmittance;
                    if (material.isEmissive()) {
                        const float3 emittedRadiance = material.power * material.baseColor;
                        accumulatedRadianceRGB += renderingTransmittance * min(emittedRadiance, 1.0f);
                    } else {
                        if (profileEnabled) {
                            profilePhotonGatherCalls += 1u;
                            profileDirectLightCalls += 1u;
                            profileDirectLightLightVisits += scene.lightCount;
                        }
                        const float3 indirectIrradiance = gatherDiffuseIrradianceAtPoint(worldHit.hitPositionW, normalW, photonMap);
                        const float3 indirectRadiance = (material.baseColor * M_1_PIf) * indirectIrradiance;
                        const float3 directRadiance = estimateDirectPointSampledPointLights(scene, settings, worldHit.hitPositionW, normalW, material.baseColor, localLayerDepthEpsilon);
                        accumulatedRadianceRGB += renderingTransmittance * (indirectRadiance + directRadiance);
                    }
                    renderingTransmittance = 0.0f;
                    if (profileEnabled && !profileStoppedByOpacity) {
                        profileOpacityTerminations += 1u;
                        profileStoppedByOpacity = true;
                    }
                    break;
                }
            }
            if (profileEnabled &&
                !profileStoppedByNoHit &&
                !profileStoppedByOpacity &&
                renderingTransmittance > kAlphaEpsilon) {
                profileMaxSplatTerminations += 1u;
            }
        }
        // =====================================================================
        // Store physical rendering outputs.
        // =====================================================================
        sensor.framebuffer[pixelIndex] = float4{accumulatedRadianceRGB.x(), accumulatedRadianceRGB.y(), accumulatedRadianceRGB.z(), sycl::clamp(accumulatedRenderingWeight, 0.0f, 1.0f)};
        // =====================================================================
        // Store regularizer outputs.
        // =====================================================================
        sensor.depthDistortionBuffer[pixelIndex] = distortion;
        sensor.intraSlabDepthBuffer[pixelIndex] = intraSlabDepthLossSum;
        sensor.intraSlabDepthActiveSlabCountBuffer[pixelIndex] =
                intraSlabDepthActiveSlabCount;
        if (accumulatedRegularizerWeight > 1.0e-6f) {
            sensor.meanDepthBuffer[pixelIndex] = accumulatedWeightedDepth / accumulatedRegularizerWeight;
        } else {
            sensor.meanDepthBuffer[pixelIndex] = 0.0f;
        }
        sensor.visibilityWeightedOpacityBuffer[pixelIndex] = visibilityWeightedOpacityLoss;
        // Old median semantics:
        // no 50% accumulated opacity -> no surface depth.
        if (medianFound) {
            sensor.medianDepthBuffer[pixelIndex] = medianDepth;
            sensor.medianWorldPositionBuffer[pixelIndex] = float4{medianWorldPosition.x(), medianWorldPosition.y(), medianWorldPosition.z(), 1.0f};
        } else {
            sensor.medianDepthBuffer[pixelIndex] = 0.0f;
            sensor.medianWorldPositionBuffer[pixelIndex] = float4{0.0f};
        }
        if (medianFound) {
            sensor.visibleNormalBuffer[pixelIndex] = float4{accumulatedWeightedNormal.x(), accumulatedWeightedNormal.y(), accumulatedWeightedNormal.z(), 1.0f};
        } else {
            sensor.visibleNormalBuffer[pixelIndex] = float4{0.0f};
        }
        if (profileEnabled) {
            RenderProfilingCounters *counters = scene.profileCounters;
            addRenderProfileCounter(&counters->forwardGatherPixels, 1u);
            addRenderProfileCounter(&counters->forwardGatherPointHitQueries, profilePointHitQueries);
            addRenderProfileCounter(&counters->forwardGatherPointHitCandidates, profilePointHitCandidates);
            addRenderProfileCounter(&counters->forwardGatherLocalLayers, profileLocalLayers);
            addRenderProfileCounter(&counters->forwardGatherLocalLayerHits, profileLocalLayerHits);
            addRenderProfileCounter(&counters->forwardGatherObjectProfileHits, profileObjectProfileHits);
            addRenderProfileCounter(&counters->forwardGatherLowPassProfileHits, profileLowPassProfileHits);
            addRenderProfileCounter(&counters->forwardGatherRegularizerHits, profileRegularizerHits);
            addRenderProfileCounter(&counters->forwardGatherPhotonGatherCalls, profilePhotonGatherCalls);
            addRenderProfileCounter(&counters->forwardGatherDirectLightCalls, profileDirectLightCalls);
            addRenderProfileCounter(&counters->forwardGatherDirectLightLightVisits, profileDirectLightLightVisits);
            addRenderProfileCounter(&counters->forwardGatherDepthPairIterations, profileDepthPairIterations);
            addRenderProfileCounter(&counters->forwardGatherMeshHits, profileMeshHits);
            addRenderProfileCounter(&counters->forwardGatherNoHitTerminations, profileNoHitTerminations);
            addRenderProfileCounter(&counters->forwardGatherOpacityTerminations, profileOpacityTerminations);
            addRenderProfileCounter(&counters->forwardGatherMaxSplatTerminations, profileMaxSplatTerminations);
        }
    });
    kernelEvent3.wait();
    // -------------------------------------------------------------------------
    // Pass 2:
    //   Normal from 2DGS-style pseudo surface depth map
    // -------------------------------------------------------------------------
    sycl::event kernelEvent4 = queue.parallel_for<class SurfaceDepthNormalKernel>(sycl::range<1>(pixelCount), [=](sycl::id<1> tid) {
        const std::uint32_t pixelIndex = tid[0];
        const std::uint32_t x = pixelIndex % imageWidth;
        const std::uint32_t y = pixelIndex / imageWidth;
        if (x == 0u || y == 0u || x + 1u >= imageWidth || y + 1u >= imageHeight) {
            sensor.normalFromDepthBuffer[pixelIndex] = float4{0.0f, 0.0f, 0.0f, 0.0f};
            return;
        }
        const uint32_t idxL = y * imageWidth + (x - 1u);
        const uint32_t idxR = y * imageWidth + (x + 1u);
        const uint32_t idxU = (y - 1u) * imageWidth + x;
        const uint32_t idxD = (y + 1u) * imageWidth + x;
        const bool useMeanDepth = settings.normalFromDepthUseMeanDepth;
        const float zC = useMeanDepth ? sensor.meanDepthBuffer[pixelIndex] : sensor.medianDepthBuffer[pixelIndex];
        const float zL = useMeanDepth ? sensor.meanDepthBuffer[idxL] : sensor.medianDepthBuffer[idxL];
        const float zR = useMeanDepth ? sensor.meanDepthBuffer[idxR] : sensor.medianDepthBuffer[idxR];
        const float zU = useMeanDepth ? sensor.meanDepthBuffer[idxU] : sensor.medianDepthBuffer[idxU];
        const float zD = useMeanDepth ? sensor.meanDepthBuffer[idxD] : sensor.medianDepthBuffer[idxD];
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
            sensor.normalFromDepthBuffer[pixelIndex] = float4{0.0f, 0.0f, 0.0f, 0.0f};
            return;
        }
        // Match 2DGS cross-product order exactly:
        //
        //   normal = normalize(cross(tangentY, tangentX))
        //
        // Do not flip toward the camera if you want exact 2DGS behavior.
        const float3 normalW = normalize(cross(tangentY, tangentX));
        sensor.normalFromDepthBuffer[pixelIndex] = float4{normalW.x(), normalW.y(), normalW.z(), 1.0f};
    });
    kernelEvent4.wait();

    // -------------------------------------------------------------------------
    // Pass 3: curvature-aware surfel-scale regularizer.
    //
    // Curvature is measured from the visibility-weighted surfel normal field.
    // The stored visibleNormalBuffer is not assumed to be unit length, so each
    // sample is normalized before finite differencing. World-space positions
    // still come from the same mean/median pseudo-surface depth used elsewhere.
    //
    // The loss is evaluated only on the slab closest to that pseudo-surface:
    //
    //   r_i^2 = 0.5 * (s_u^2 + s_v^2)
    //   v_i   = kappa * r_i^2 / (2 * gamma * h) - 1
    //   e_i   = max(0, v_i)
    //   L_Q   = mean_i(e_i^2)
    //
    // Curvature and the selected slab are treated as stop-gradient quantities
    // by the adjoint pass.  Store the exact loss in a dedicated buffer so RGB
    // rendering is never replaced by a diagnostic visualization.
    // -------------------------------------------------------------------------
    constexpr float visibleNormalLengthSquaredEpsilon = 1.0e-12f;
    const float slabThickness = rendererDebugLocalLayerDepthEpsilon(settings);
    const CurvatureDensificationStats curvatureDensificationStats =
        pkg.curvatureDensificationStats;
    sycl::event scaleCurvatureEvent = queue.parallel_for<class ScaleCurvatureRegularizerKernel2>(sycl::range<1>(pixelCount), [=](sycl::id<1> tid) {
            const uint32_t pixelIndex = static_cast<uint32_t>(tid[0]);
            const uint32_t pixelX = pixelIndex % imageWidth;
            const uint32_t pixelY = pixelIndex / imageWidth;

            sensor.curvatureScaleBuffer[pixelIndex] = 0.0f;
            sensor.curvatureScaleActiveSlabCountBuffer[pixelIndex] = 0u;
            if (sensor.curvaturePrimitiveIndexBuffer != nullptr) {
                sensor.curvaturePrimitiveIndexBuffer[pixelIndex] = UINT32_MAX;
            }

            // N(x+1,y) and N(x,y+1) must exist. Their validity flag also rejects
            // the border invalidated by SurfaceDepthNormalKernel.
            if (pixelX + 1u >= imageWidth || pixelY + 1u >= imageHeight || slabThickness <= 0.0f) {
                return;
            }

            const uint32_t rightPixelIndex = pixelY * imageWidth + (pixelX + 1u);
            const uint32_t downPixelIndex = (pixelY + 1u) * imageWidth + pixelX;
            const float4 centerVisibleNormal = sensor.visibleNormalBuffer[pixelIndex];
            const float4 rightVisibleNormal = sensor.visibleNormalBuffer[rightPixelIndex];
            const float4 downVisibleNormal = sensor.visibleNormalBuffer[downPixelIndex];

            if (centerVisibleNormal.w() <= 0.0f || rightVisibleNormal.w() <= 0.0f || downVisibleNormal.w() <= 0.0f) {
                return;
            }

            const float3 centerVisibleNormalRaw{
                centerVisibleNormal.x(),
                centerVisibleNormal.y(),
                centerVisibleNormal.z()
            };
            const float3 rightVisibleNormalRaw{
                rightVisibleNormal.x(),
                rightVisibleNormal.y(),
                rightVisibleNormal.z()
            };
            const float3 downVisibleNormalRaw{
                downVisibleNormal.x(),
                downVisibleNormal.y(),
                downVisibleNormal.z()
            };

            const float centerVisibleNormalLengthSquared = dot(centerVisibleNormalRaw, centerVisibleNormalRaw);
            const float rightVisibleNormalLengthSquared = dot(rightVisibleNormalRaw, rightVisibleNormalRaw);
            const float downVisibleNormalLengthSquared = dot(downVisibleNormalRaw, downVisibleNormalRaw);

            if (centerVisibleNormalLengthSquared <= visibleNormalLengthSquaredEpsilon ||
                rightVisibleNormalLengthSquared <= visibleNormalLengthSquaredEpsilon ||
                downVisibleNormalLengthSquared <= visibleNormalLengthSquaredEpsilon) {
                return;
            }

            const float3 centerNormalW = centerVisibleNormalRaw / sycl::sqrt(centerVisibleNormalLengthSquared);
            float3 rightNormalW = rightVisibleNormalRaw / sycl::sqrt(rightVisibleNormalLengthSquared);
            float3 downNormalW = downVisibleNormalRaw / sycl::sqrt(downVisibleNormalLengthSquared);

            const bool useMeanDepth = settings.normalFromDepthUseMeanDepth;
            const float centerDepth = useMeanDepth ? sensor.meanDepthBuffer[pixelIndex] : sensor.medianDepthBuffer[pixelIndex];
            const float rightDepth = useMeanDepth ? sensor.meanDepthBuffer[rightPixelIndex] : sensor.medianDepthBuffer[rightPixelIndex];
            const float downDepth = useMeanDepth ? sensor.meanDepthBuffer[downPixelIndex] : sensor.medianDepthBuffer[downPixelIndex];
            if (centerDepth <= 0.0f || rightDepth <= 0.0f || downDepth <= 0.0f) {
                return;
            }

            const float3 centerPositionW = reconstructWorldPositionFromDepthCenter(sensor.camera, pixelX, pixelY, centerDepth);
            const float3 rightPositionW = reconstructWorldPositionFromDepthCenter(sensor.camera, pixelX + 1u, pixelY, rightDepth);
            const float3 downPositionW = reconstructWorldPositionFromDepthCenter(sensor.camera, pixelX, pixelY + 1u, downDepth);

            // Curvature is orientation-independent. Avoid artificial spikes if a
            // neighboring visible surfel normal happens to have opposite sign.
            if (dot(centerNormalW, rightNormalW) < 0.0f) { rightNormalW = -rightNormalW; }
            if (dot(centerNormalW, downNormalW) < 0.0f) { downNormalW = -downNormalW; }

            const float3 rightNormalDifference = rightNormalW - centerNormalW;
            const float3 downNormalDifference = downNormalW - centerNormalW;
            const float3 rightPositionDifference = rightPositionW - centerPositionW;
            const float3 downPositionDifference = downPositionW - centerPositionW;
            const float rightNormalDifferenceLength = sycl::sqrt(dot(rightNormalDifference, rightNormalDifference));
            const float downNormalDifferenceLength = sycl::sqrt(dot(downNormalDifference, downNormalDifference));
            const float rightPositionDifferenceLength = sycl::sqrt(dot(rightPositionDifference, rightPositionDifference));
            const float downPositionDifferenceLength = sycl::sqrt(dot(downPositionDifference, downPositionDifference));
            const float curvatureX = rightNormalDifferenceLength /
                                     (rightPositionDifferenceLength + CurvatureRegularizerDistanceEpsilon);
            const float curvatureY = downNormalDifferenceLength /
                                     (downPositionDifferenceLength + CurvatureRegularizerDistanceEpsilon);
            const float curvature = sycl::fmax(curvatureX, curvatureY);

            // Locate the local slab corresponding to the visible pseudo surface.
            // Retrace the ray and select the slab closest to the pseudo-surface.
            Ray scaleRegularizerRay = makePrimaryRayFromPixelJitteredFov(sensor.camera, static_cast<float>(pixelX), static_cast<float>(pixelY), 0.0f, 0.0f);
            const uint32_t maxSplatEventsPerRay = rendererDebugMaxSplatEventsPerRay(settings);
            const uint32_t maxLocalSurfelHits = rendererDebugMaxLocalSurfelHits(settings);
            const float localLayerNormalCosineThreshold = rendererDebugLocalLayerNormalCosineThreshold(settings);
            const MinimumProjectedFootprintFilter minimumFootprintFilter = minimumProjectedFootprintFilterFromSettings(settings, sensor.camera, float2{static_cast<float>(pixelX), static_cast<float>(pixelY)});

            float closestSurfaceDepthDifference = std::numeric_limits<float>::infinity();
            PointCloudLocalLayer selectedLayer{};
            uint32_t selectedTransformIndex = UINT32_MAX;
            bool foundPointSlab = false;

            for (uint32_t traversalIndex = 0u; traversalIndex < maxSplatEventsPerRay; ++traversalIndex) {
                WorldHit worldHit{};
                intersectScene(scaleRegularizerRay, &worldHit, scene, SurfelIntersectMode::FirstHit);
                if (!worldHit.hit) { break; }

                buildIntersectionNormal(scene, worldHit);
                const InstanceRecord &instance = scene.instances[worldHit.instanceIndex];
                if (instance.geometryType == GeometryType::Mesh) { break; }
                if (instance.geometryType != GeometryType::PointCloud) { break; }

                const PointCloudLocalLayer localLayer = collectPointCloudLocalLayer(scaleRegularizerRay, worldHit, instance, scene, slabThickness, maxLocalSurfelHits, localLayerNormalCosineThreshold, minimumFootprintFilter);
                if (localLayer.hitCount == 0u) { break; }

                const LocalSurfelLayerHit &anchorHit = localLayer.hits[0];
                const float anchorSurfaceDepth = dot(anchorHit.hitPositionW - sensor.camera.pos, sensor.camera.forward);
                const float surfaceDepthDifference = sycl::fabs(anchorSurfaceDepth - centerDepth);

                if (surfaceDepthDifference < closestSurfaceDepthDifference) {
                    selectedLayer = localLayer;
                    selectedTransformIndex = instance.transformIndex;
                    closestSurfaceDepthDifference = surfaceDepthDifference;
                    foundPointSlab = true;
                }

                scaleRegularizerRay.origin += scaleRegularizerRay.direction * (localLayer.furthestT + RayEpsilon);
            }

            if (!foundPointSlab) {
                return;
            }

            if (sensor.curvaturePrimitiveIndexBuffer != nullptr) {
                // Use the strongest renderer contributor as the displayed
                // primitive identity, but only within the exact selected slab
                // used below for the curvature regularizer/statistics.
                uint32_t dominantHitIndex = 0u;
                float dominantWeight = selectedLayer.weight[0];
                for (uint32_t localHitIndex = 1u;
                     localHitIndex < selectedLayer.hitCount;
                     ++localHitIndex) {
                    if (selectedLayer.weight[localHitIndex] > dominantWeight) {
                        dominantWeight = selectedLayer.weight[localHitIndex];
                        dominantHitIndex = localHitIndex;
                    }
                }
                sensor.curvaturePrimitiveIndexBuffer[pixelIndex] =
                    selectedLayer.hits[dominantHitIndex].primitiveIndex;
            }

            float accumulatedScaleLoss = 0.0f;
            const bool accumulateDensificationStats =
                curvatureDensificationStats.numPoints == scene.pointCount &&
                curvatureDensificationStats.violationSum != nullptr &&
                curvatureDensificationStats.violationCount != nullptr &&
                curvatureDensificationStats.directionTensorUu != nullptr &&
                curvatureDensificationStats.directionTensorUv != nullptr &&
                curvatureDensificationStats.directionTensorVv != nullptr &&
                selectedTransformIndex != UINT32_MAX;

            for (uint32_t localHitIndex = 0u;
                 localHitIndex < selectedLayer.hitCount;
                 ++localHitIndex) {
                const LocalSurfelLayerHit &localHit = selectedLayer.hits[localHitIndex];
                const Point &surfel = scene.points[localHit.primitiveIndex];
                const float scaleU = surfel.scale.x();
                const float scaleV = surfel.scale.y();
                const float radiusSquared = 0.5f * (scaleU * scaleU + scaleV * scaleV);
                const float normalizedViolation = curvature * radiusSquared /
                    (2.0f * CurvatureScaleRegularizerGamma * slabThickness) - 1.0f;
                const float residual = sycl::fmax(0.0f, normalizedViolation);
                accumulatedScaleLoss += residual * residual;

                // These are forward-only structural statistics. They use the
                // raw geometric violation and therefore do not depend on the
                // curvature regularizer loss weight.
                if (!accumulateDensificationStats ||
                    localHit.primitiveIndex >= curvatureDensificationStats.numPoints ||
                    !sycl::isfinite(radiusSquared) ||
                    !sycl::isfinite(residual)) {
                    continue;
                }

                const uint32_t primitiveIndex = localHit.primitiveIndex;
                sycl::atomic_ref<float,
                                 sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>(
                    curvatureDensificationStats.violationSum[primitiveIndex]
                ).fetch_add(residual);
                sycl::atomic_ref<uint32_t,
                                 sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>(
                    curvatureDensificationStats.violationCount[primitiveIndex]
                ).fetch_add(1u);

                const float directionalViolationX = sycl::fmax(
                    0.0f,
                    curvatureX * radiusSquared /
                        (2.0f * CurvatureScaleRegularizerGamma * slabThickness) - 1.0f);
                const float directionalViolationY = sycl::fmax(
                    0.0f,
                    curvatureY * radiusSquared /
                        (2.0f * CurvatureScaleRegularizerGamma * slabThickness) - 1.0f);
                if (directionalViolationX <= 0.0f && directionalViolationY <= 0.0f) {
                    continue;
                }

                const Transform &transform = scene.transforms[selectedTransformIndex];
                float3 tangentU = transformDirection(transform.objectToWorld, surfel.tanU);
                float3 tangentV = transformDirection(transform.objectToWorld, surfel.tanV);
                const float tangentULengthSquared = dot(tangentU, tangentU);
                tangentV -= tangentU * dot(tangentU, tangentV);
                const float tangentVLengthSquared = dot(tangentV, tangentV);
                const bool finiteFrame =
                    sycl::isfinite(tangentU.x()) && sycl::isfinite(tangentU.y()) &&
                    sycl::isfinite(tangentU.z()) && sycl::isfinite(tangentV.x()) &&
                    sycl::isfinite(tangentV.y()) && sycl::isfinite(tangentV.z());
                if (!finiteFrame || tangentULengthSquared <= 1.0e-12f ||
                    tangentVLengthSquared <= 1.0e-12f) {
                    continue;
                }
                tangentU = tangentU / sycl::sqrt(tangentULengthSquared);
                tangentV = tangentV / sycl::sqrt(tangentVLengthSquared);
                float3 surfelNormal = cross(tangentU, tangentV);
                const float surfelNormalLengthSquared = dot(surfelNormal, surfelNormal);
                if (!sycl::isfinite(surfelNormalLengthSquared) ||
                    surfelNormalLengthSquared <= 1.0e-12f) {
                    continue;
                }
                surfelNormal = surfelNormal / sycl::sqrt(surfelNormalLengthSquared);

                float tensorUu = 0.0f;
                float tensorUv = 0.0f;
                float tensorVv = 0.0f;

                const float3 projectedDirectionX = rightPositionDifference -
                    surfelNormal * dot(surfelNormal, rightPositionDifference);
                const float projectedDirectionXLengthSquared =
                    dot(projectedDirectionX, projectedDirectionX);
                if (directionalViolationX > 0.0f &&
                    sycl::isfinite(projectedDirectionXLengthSquared) &&
                    projectedDirectionXLengthSquared > 1.0e-12f) {
                    const float3 directionX = projectedDirectionX /
                        sycl::sqrt(projectedDirectionXLengthSquared);
                    float axisU = dot(directionX, tangentU);
                    float axisV = dot(directionX, tangentV);
                    const float axisLengthSquared = axisU * axisU + axisV * axisV;
                    if (sycl::isfinite(axisLengthSquared) && axisLengthSquared > 1.0e-12f) {
                        const float inverseAxisLength = sycl::rsqrt(axisLengthSquared);
                        axisU *= inverseAxisLength;
                        axisV *= inverseAxisLength;
                        tensorUu += directionalViolationX * axisU * axisU;
                        tensorUv += directionalViolationX * axisU * axisV;
                        tensorVv += directionalViolationX * axisV * axisV;
                    }
                }

                const float3 projectedDirectionY = downPositionDifference -
                    surfelNormal * dot(surfelNormal, downPositionDifference);
                const float projectedDirectionYLengthSquared =
                    dot(projectedDirectionY, projectedDirectionY);
                if (directionalViolationY > 0.0f &&
                    sycl::isfinite(projectedDirectionYLengthSquared) &&
                    projectedDirectionYLengthSquared > 1.0e-12f) {
                    const float3 directionY = projectedDirectionY /
                        sycl::sqrt(projectedDirectionYLengthSquared);
                    float axisU = dot(directionY, tangentU);
                    float axisV = dot(directionY, tangentV);
                    const float axisLengthSquared = axisU * axisU + axisV * axisV;
                    if (sycl::isfinite(axisLengthSquared) && axisLengthSquared > 1.0e-12f) {
                        const float inverseAxisLength = sycl::rsqrt(axisLengthSquared);
                        axisU *= inverseAxisLength;
                        axisV *= inverseAxisLength;
                        tensorUu += directionalViolationY * axisU * axisU;
                        tensorUv += directionalViolationY * axisU * axisV;
                        tensorVv += directionalViolationY * axisV * axisV;
                    }
                }

                // K += v_x a_x a_x^T + v_y a_y a_y^T. The outer products
                // make this an axis statistic, so opposite directions agree.
                if (tensorUu != 0.0f || tensorUv != 0.0f || tensorVv != 0.0f) {
                    sycl::atomic_ref<float,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>(
                        curvatureDensificationStats.directionTensorUu[primitiveIndex]
                    ).fetch_add(tensorUu);
                    sycl::atomic_ref<float,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>(
                        curvatureDensificationStats.directionTensorUv[primitiveIndex]
                    ).fetch_add(tensorUv);
                    sycl::atomic_ref<float,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>(
                        curvatureDensificationStats.directionTensorVv[primitiveIndex]
                    ).fetch_add(tensorVv);
                }
            }

            const float selectedSlabScaleLoss = accumulatedScaleLoss /
                static_cast<float>(selectedLayer.hitCount);
            sensor.curvatureScaleBuffer[pixelIndex] = selectedSlabScaleLoss;
            sensor.curvatureScaleActiveSlabCountBuffer[pixelIndex] = 1u;
    });
    scaleCurvatureEvent.wait();
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
    queue.fill(sensor.intraSlabDepthBuffer, 0.0f, pixelCount).wait();
    queue.fill(sensor.intraSlabDepthAdjointBuffer, 0.0f, pixelCount).wait();
    queue.fill(sensor.intraSlabDepthActiveSlabCountBuffer, 0u, pixelCount).wait();
    queue.fill(sensor.curvatureScaleBuffer, 0.0f, pixelCount).wait();
    queue.fill(sensor.curvatureScaleAdjointBuffer, 0.0f, pixelCount).wait();
    queue.fill(sensor.curvatureScaleActiveSlabCountBuffer, 0u, pixelCount).wait();
    // -------------------------------------------------------------------------
    // Pass 1:
    //   - RGB gather
    //   - median depth
    //   - median world position
    //   - visible normal at median surface
    // -------------------------------------------------------------------------
    const uint64_t renderSeed = pkg.random.seed;
    sycl::event kernelEvent5 = queue.parallel_for<class CameraGatherKernel>(sycl::range<1>(pixelCount), [=](sycl::id<1> tid) {
        const std::uint32_t pixelIndex = tid[0];
        const std::uint32_t pixelX = pixelIndex % imageWidth;
        const std::uint32_t pixelY = pixelIndex / imageWidth;
        const uint64_t directionSeed = rng::makeSeed(renderSeed, pixelIndex, cameraIndex, rng::kStreamGather, 0u);
        rng::Xorshift128 rng(directionSeed);
        float3 accumulatedRadianceRGB(0.0f, 0.0f, 0.0f);
        // Center-of-pixel sample in your jitter convention
        Ray primaryRay = makePrimaryRayFromPixelJitteredFov(sensor.camera, static_cast<float>(pixelX), static_cast<float>(pixelY), 0.0f, 0.0f);
        const float cameraCosine = dot(sensor.camera.forward, primaryRay.direction);
        float transmittance = 1.0f;
        // Depth distortion accumulation
        float previousDepthDistortionWeights[kMaxSplatEventsPerRay];
        float previousDepthDistortionNdcDepths[kMaxSplatEventsPerRay];
        uint32_t previousDepthDistortionHitCount = 0u;
        float distortion = 0.0f;
        // Median-depth tracking
        float accumulatedCompositeWeight = 0.0f;
        bool medianFound = false;
        float medianDepth = 0.0f;
        float3 medianWorldPosition(0.0f, 0.0f, 0.0f);
        float3 medianNormalW(0.0f, 0.0f, 0.0f);
        float accumulatedMeanDepthWeight = 0.0f;
        float accumulatedMeanDepth = 0.0f;
        const uint32_t maxSplatEventsPerRay = rendererDebugMaxSplatEventsPerRay(settings);
        const uint32_t pointHitBatchSize = rendererDebugPointHitBatchSize(settings);
        const MinimumProjectedFootprintFilter minimumFootprintFilter =
            minimumProjectedFootprintFilterFromSettings(
                settings,
                sensor.camera,
                float2{static_cast<float>(pixelX), static_cast<float>(pixelY)});
        uint32_t directPointInstanceIndex = kInvalidIndex;
        const bool canUsePointHitBatches =
            pointHitBatchSize > 1u &&
            tryGetSinglePointCloudInstance(scene, directPointInstanceIndex);

        auto accumulatePointHit = [&](const LocalSurfelLayerHit &pointHit) {
            const Point &surfel = scene.points[pointHit.primitiveIndex];
            float3 normalW = normalize(cross(surfel.tanU, surfel.tanV));
            const bool hitBackside = dot(normalW, -primaryRay.direction) < 0.0f;
            if (hitBackside) { normalW = -normalW; }
            const float alphaEff = surfel.opacity * pointHit.alphaGeom;
            const float3 indirectIrradiance = gatherDiffuseIrradianceAtPoint(pointHit.hitPositionW, normalW, photonMap);
            const float3 indirectRadiance = indirectIrradiance * (surfel.alpha_r * surfel.albedo * M_1_PIf) * alphaEff;
            const float surfelArea = M_PIf * surfel.scale.x() * surfel.scale.y();
            float3 emittedRadiance{0.0f};
            if (surfelArea > 1.0e-12f && sycl::isfinite(surfelArea)) {
                emittedRadiance = surfel.albedo * (surfel.flux / (M_PIf * surfelArea)) * alphaEff;
            }
            if (surfel.isEmissive() && hitBackside) {
                // emittedRadiance = float3(0.0f, 0.0f, 0.0f);
            }
            /*
            const float3 directRadiance =
                    estimateDirectAreaLightAtDiffuseSurface(scene, pointHit.hitPositionW, normalW,
                                                            surfel.alpha_r * surfel.albedo, settings,
                                                            rng) * alphaEff;
            */
            const float3 directRadiance = estimateDirectPointSampledPointLights(scene, settings, pointHit.hitPositionW, normalW, surfel.alpha_r * surfel.albedo, RayEpsilon) * alphaEff;
            const float3 outgoingRadiance = emittedRadiance + indirectRadiance + directRadiance;
            accumulatedRadianceRGB += transmittance * outgoingRadiance;
            // Median depth using compositing weights w_i = T_i * alpha_i
            const float wi = transmittance * alphaEff;
            const float zi = dot(pointHit.hitPositionW - sensor.camera.pos, sensor.camera.forward);
            accumulatedMeanDepthWeight += wi;
            accumulatedMeanDepth += wi * zi;
            if (!medianFound && (accumulatedCompositeWeight + wi) >= 0.5f) {
                medianFound = true;
                medianDepth = zi;
                medianWorldPosition = pointHit.hitPositionW;
                medianNormalW = normalW;
            }
            accumulatedCompositeWeight += wi;
            const float ndcDepth = depthDistortionNdc01(zi);
            for (uint32_t previousIndex = 0u; previousIndex < previousDepthDistortionHitCount; ++previousIndex) {
                const float depthDifference = ndcDepth - previousDepthDistortionNdcDepths[previousIndex];
                distortion += previousDepthDistortionWeights[previousIndex] * wi *
                              depthDifference * depthDifference;
            }
            if (previousDepthDistortionHitCount < kMaxSplatEventsPerRay) {
                previousDepthDistortionWeights[previousDepthDistortionHitCount] = wi;
                previousDepthDistortionNdcDepths[previousDepthDistortionHitCount] = ndcDepth;
                ++previousDepthDistortionHitCount;
            }
            transmittance *= (1.0f - alphaEff);
        };

        if (canUsePointHitBatches) {
            for (uint32_t traversalIndex = 0u; traversalIndex < maxSplatEventsPerRay;) {
                LocalSurfelLayerHit pointHits[kMaxPointHitBatch];
                uint32_t pointInstanceIndex = kInvalidIndex;
                const uint32_t remainingTraversalBudget = maxSplatEventsPerRay - traversalIndex;
                const uint32_t batchCapacity =
                    pointHitBatchSize < remainingTraversalBudget ? pointHitBatchSize : remainingTraversalBudget;
                const uint32_t hitCount = collectScenePointHitsDirect(
                    primaryRay,
                    scene,
                    RayEpsilon,
                    std::numeric_limits<float>::infinity(),
                    pointHits,
                    batchCapacity,
                    pointInstanceIndex,
                    minimumFootprintFilter);

                if (hitCount == 0u) { break; }

                float furthestConsumedT = 0.0f;
                for (uint32_t batchIndex = 0u; batchIndex < hitCount; ++batchIndex) {
                    const LocalSurfelLayerHit &pointHit = pointHits[batchIndex];
                    furthestConsumedT = sycl::fmax(furthestConsumedT, pointHit.tWorld);
                    accumulatePointHit(pointHit);
                    ++traversalIndex;
                }

                if (furthestConsumedT <= 0.0f) { break; }

                primaryRay.origin += primaryRay.direction * (furthestConsumedT + RayEpsilon);
                if (hitCount < batchCapacity) { break; }
            }
        } else {
            for (uint32_t traversalIndex = 0u; traversalIndex < maxSplatEventsPerRay; ++traversalIndex) {
                WorldHit worldHit{};
                intersectScene(primaryRay, &worldHit, scene, SurfelIntersectMode::FirstHit);
                if (!worldHit.hit) { break; }
                buildIntersectionNormal(scene, worldHit);
                const auto &instance = scene.instances[worldHit.instanceIndex];
                // -------------------------------------------------------------
                // Visible point-cloud layer
                // -------------------------------------------------------------
                if (instance.geometryType == GeometryType::PointCloud) {
                    LocalSurfelLayerHit pointHit{};
                    pointHit.tWorld = worldHit.t;
                    pointHit.primitiveIndex = worldHit.primitiveIndex;
                    pointHit.alphaGeom = worldHit.alphaGeom;
                    pointHit.hitPositionW = worldHit.hitPositionW;
                    pointHit.uv = phiInverse(worldHit.hitPositionW, scene.points[worldHit.primitiveIndex]);
                    pointHit.objectAlphaGeom = worldHit.alphaGeom;
                    pointHit.lowPassAlphaGeom = 0.0f;
                    pointHit.lowPassDeltaPixels = float2{0.0f, 0.0f};
                    pointHit.lowPassSigmaPixels = 0.0f;
                    pointHit.alphaProfileBranch = kSurfelAlphaProfileObject;
                    pointHit.usesSurfelCenterHitPosition = 0u;
                    accumulatePointHit(pointHit);
                    primaryRay.origin = worldHit.hitPositionW + primaryRay.direction * RayEpsilon;
                    continue;
                }
                // -------------------------------------------------------------
                // Terminal mesh hit
                // -------------------------------------------------------------
                if (instance.geometryType == GeometryType::Mesh) {
                    const GPUMaterial &material = scene.materials[instance.materialIndex];
                    const bool isBackfaceHit = dot(primaryRay.direction, worldHit.geometricNormalW) > 0.0f;
                    const float3 normalW = isBackfaceHit ? -worldHit.geometricNormalW : worldHit.geometricNormalW;
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
                    } else {
                        const float3 indirectIrradiance = gatherDiffuseIrradianceAtPoint(worldHit.hitPositionW, normalW, photonMap);
                        const float3 indirectRadiance = (material.baseColor * M_1_PIf) * indirectIrradiance;
                        // const float3 directRadiance =
                        //         estimateDirectAreaLightAtDiffuseSurface(
                        //             scene, worldHit.hitPositionW, normalW, material.baseColor, settings, rng);
                        const float3 directRadiance = estimateDirectPointSampledPointLights(scene, settings, worldHit.hitPositionW, normalW, material.baseColor, RayEpsilon);
                        const float3 outgoingRadiance = indirectRadiance + directRadiance;
                        accumulatedRadianceRGB += transmittance * outgoingRadiance;
                    }
                    transmittance = 0.0f;
                    break;
                }
            }
        }
        const std::uint32_t framebufferIndex = pixelY * imageWidth + pixelX;
        // accumulatedRadianceRGB *= cameraCosine;
        const float alpha = sycl::clamp(accumulatedCompositeWeight, 0.0f, 1.0f);
        const float4 currentValue(accumulatedRadianceRGB.x(), accumulatedRadianceRGB.y(), accumulatedRadianceRGB.z(), alpha);
        sensor.framebuffer[framebufferIndex] += currentValue;
        sensor.depthDistortionBuffer[pixelIndex] = distortion;
        if (accumulatedMeanDepthWeight > 1.0e-6f) {
            sensor.meanDepthBuffer[pixelIndex] = accumulatedMeanDepth / accumulatedMeanDepthWeight;
        } else {
            sensor.meanDepthBuffer[pixelIndex] = 0.0f;
        }
        if (medianFound) {
            sensor.medianDepthBuffer[pixelIndex] = medianDepth;
            sensor.medianWorldPositionBuffer[pixelIndex] = float4{medianWorldPosition.x(), medianWorldPosition.y(), medianWorldPosition.z(), 1.0f};
            sensor.visibleNormalBuffer[pixelIndex] = float4{medianNormalW.x(), medianNormalW.y(), medianNormalW.z(), 1.0f};
        } else {
            sensor.medianDepthBuffer[pixelIndex] = 0.0f;
            sensor.medianWorldPositionBuffer[pixelIndex] = float4{0.0f};
            sensor.visibleNormalBuffer[pixelIndex] = float4{0.0f};
        }
    });
    kernelEvent5.wait();
    // -------------------------------------------------------------------------
    // Pass 2:
    //   Normal from 2DGS-style pseudo surface depth map
    // -------------------------------------------------------------------------
    if (settings.normalConsistencyWeight) {
        sycl::event kernelEvent6 = queue.parallel_for<class SurfaceDepthNormalKernel>(sycl::range<1>(pixelCount), [=](sycl::id<1> tid) {
            const std::uint32_t pixelIndex = tid[0];
            const std::uint32_t x = pixelIndex % imageWidth;
            const std::uint32_t y = pixelIndex / imageWidth;
            if (x == 0u || y == 0u || x + 1u >= imageWidth || y + 1u >= imageHeight) {
                sensor.normalFromDepthBuffer[pixelIndex] = float4{0.0f, 0.0f, 0.0f, 0.0f};
                return;
            }
            const uint32_t idxL = y * imageWidth + (x - 1u);
            const uint32_t idxR = y * imageWidth + (x + 1u);
            const uint32_t idxU = (y - 1u) * imageWidth + x;
            const uint32_t idxD = (y + 1u) * imageWidth + x;
            const bool useMeanDepth = settings.normalFromDepthUseMeanDepth;
            const float zC = useMeanDepth ? sensor.meanDepthBuffer[pixelIndex] : sensor.medianDepthBuffer[pixelIndex];
            const float zL = useMeanDepth ? sensor.meanDepthBuffer[idxL] : sensor.medianDepthBuffer[idxL];
            const float zR = useMeanDepth ? sensor.meanDepthBuffer[idxR] : sensor.medianDepthBuffer[idxR];
            const float zU = useMeanDepth ? sensor.meanDepthBuffer[idxU] : sensor.medianDepthBuffer[idxU];
            const float zD = useMeanDepth ? sensor.meanDepthBuffer[idxD] : sensor.medianDepthBuffer[idxD];
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
                sensor.normalFromDepthBuffer[pixelIndex] = float4{0.0f, 0.0f, 0.0f, 0.0f};
                return;
            }
            // Match 2DGS cross-product order exactly:
            //
            //   normal = normalize(cross(tangentY, tangentX))
            //
            // Do not flip toward the camera if you want exact 2DGS behavior.
            const float3 normalW = normalize(cross(tangentY, tangentX));
            sensor.normalFromDepthBuffer[pixelIndex] = float4{normalW.x(), normalW.y(), normalW.z(), 1.0f};
        });
        kernelEvent6.wait();
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
        queue.fill(sensor.intraSlabDepthBuffer, 0.0f, pixelCount);
        queue.fill(sensor.intraSlabDepthAdjointBuffer, 0.0f, pixelCount);
        queue.fill(sensor.intraSlabDepthActiveSlabCountBuffer, 0u, pixelCount);
        queue.fill(sensor.curvatureScaleBuffer, 0.0f, pixelCount);
        queue.fill(sensor.curvatureScaleAdjointBuffer, 0.0f, pixelCount);
        queue.fill(sensor.curvatureScaleActiveSlabCountBuffer, 0u, pixelCount);
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
void computePhotonCellIdsAndPermutation(sycl::queue &queue, DeviceSurfacePhotonMapGrid grid, std::uint32_t photonCount) {
    sycl::event kernelEvent7 = queue.parallel_for(sycl::range<1>(photonCount), [=](sycl::id<1> idx) {
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
    kernelEvent7.wait();
}
void clearCellArrays(sycl::queue &queue, DeviceSurfacePhotonMapGrid grid) {
    static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;
    const std::uint32_t cellCount = grid.totalCellCount;
    sycl::event kernelEvent8 = queue.parallel_for(sycl::range<1>(cellCount), [=](sycl::id<1> idx) {
        const std::uint32_t c = static_cast<std::uint32_t>(idx[0]);
        grid.cellCount[c] = 0u;
        grid.cellWriteOffset[c] = 0u;
        grid.cellStart[c] = kInvalidIndex;
        grid.cellEnd[c] = kInvalidIndex;
    });
    kernelEvent8.wait();
}
void countPhotonsPerCell(sycl::queue &queue, DeviceSurfacePhotonMapGrid grid, std::uint32_t photonCount) {
    static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;
    sycl::event kernelEvent9 = queue.parallel_for(sycl::range<1>(photonCount), [=](sycl::id<1> idx) {
        const std::uint32_t i = static_cast<std::uint32_t>(idx[0]);
        const std::uint32_t cellId = grid.photonCellId[i];
        if (cellId == kInvalidIndex) return;
        auto atomicCount = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(grid.cellCount[cellId]);
        atomicCount.fetch_add(1u);
    });
    kernelEvent9.wait();
}
void scatterPhotonsIntoCells(sycl::queue &queue, DeviceSurfacePhotonMapGrid grid, std::uint32_t photonCount) {
    static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;
    queue.parallel_for(sycl::range<1>(photonCount), [=](sycl::id<1> idx) {
        const std::uint32_t i = static_cast<std::uint32_t>(idx[0]);
        const std::uint32_t cellId = grid.photonCellId[i];
        if (cellId == kInvalidIndex) return;
        const std::uint32_t start = grid.cellStart[cellId];
        // start should be valid if count > 0
        if (start == kInvalidIndex) return;
        auto atomicOffset = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(grid.cellWriteOffset[cellId]);
        const std::uint32_t localOffset = atomicOffset.fetch_add(1u);
        const uint32_t end = grid.cellEnd[cellId];
        const uint32_t writeIndex = start + localOffset;
        if (writeIndex < end) { grid.sortedPhotonIndex[writeIndex] = i; }
    });
}
static constexpr std::uint32_t kScanBlockSize = 1024;
void exclusiveScanCellCountsToCellStart(sycl::queue &queue, DeviceSurfacePhotonMapGrid grid) {
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
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(blockCount * blockSize), sycl::range<1>(blockSize)), [=](sycl::nd_item<1> item) {
                const std::uint32_t localIndex = static_cast<std::uint32_t>(item.get_local_id(0));
                const std::uint32_t blockIndex = static_cast<std::uint32_t>(item.get_group(0));
                const std::uint32_t globalIndex = blockIndex * blockSize + localIndex;
                // Load into local memory (out-of-range -> 0)
                std::uint32_t value = 0u;
                if (globalIndex < totalCellCount) value = cellCount[globalIndex];
                localData[localIndex] = value;
                item.barrier(sycl::access::fence_space::local_space);
                // Blelloch upsweep
                for (std::uint32_t offset = 1u; offset < blockSize; offset <<= 1u) {
                    const std::uint32_t index = (localIndex + 1u) * offset * 2u - 1u;
                    if (index < blockSize) localData[index] += localData[index - offset];
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
                if (globalIndex < totalCellCount) cellStart[globalIndex] = localData[localIndex];
            });
        })
        .wait();
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
    sycl::event kernelEvent10 = queue.parallel_for(sycl::range<1>(totalCellCount), [=](sycl::id<1> idx) {
        const std::uint32_t globalIndex = static_cast<std::uint32_t>(idx[0]);
        const std::uint32_t blockIndex = globalIndex / blockSize;
        cellStart[globalIndex] += blockPrefix[blockIndex];
    });
    kernelEvent10.wait();
}
void finalizeCellRanges(sycl::queue &queue, DeviceSurfacePhotonMapGrid grid) {
    static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;
    const std::uint32_t totalCellCount = grid.totalCellCount;
    sycl::event kernelEvent11 = queue.parallel_for(sycl::range<1>(totalCellCount), [=](sycl::id<1> idx) {
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
    kernelEvent11.wait();
}
void buildPhotonCellRangesAndOrdering(sycl::queue &queue, DeviceSurfacePhotonMapGrid grid, std::uint32_t photonCount) {
    clearCellArrays(queue, grid);                                 // counts/start/end/offset=0/invalid
    computePhotonCellIdsAndPermutation(queue, grid, photonCount); // keys (optional now)
    countPhotonsPerCell(queue, grid, photonCount);                // histogram
    exclusiveScanCellCountsToCellStart(queue, grid);              // cellStart from cellCount (implement)
    finalizeCellRanges(queue, grid);                              // cellEnd = start + count, invalid if count==0, offset=0
    scatterPhotonsIntoCells(queue, grid, photonCount);            // sortedPhotonIndex
}
} // namespace Pale
