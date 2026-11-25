//
// Created by magnus on 9/12/25.
//
#include "PrimalKernels.h"
#include "IntersectionKernels.h"
#include "KernelHelpers.h"
#include "SharedHeightForward.h"
#include "Core/ScopedTimer.h"
#include <cmath>

namespace Pale {
    static void waitForForwardKernelTiming(sycl::event &event) {
        // The in-order queue already orders device consumers. Only host timing
        // needs an intermediate barrier; renderForward retains its final wait.
        if (ScopedTimerDetail::isProfilingEnabled() ||
            ScopedTimerDetail::isLogLevelEnabled(spdlog::level::debug)) {
            event.wait();
        }
    }

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
        sycl::event kernelEvent1 = queue.parallel_for<struct RayGenEmitterKernelTag>(
            sycl::range<1>(settings.photonsPerLaunch), [=](sycl::id<1> globalId) {
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
                ray.pathThroughput = initialThroughput;
                ray.bounceIndex = 0;
                ray.traversalIndex = 0;
                ray.lightIndex = ls.lightIndex;
                ray.pathId = pathId;
                auto counter = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(*countPrimary);
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
        sycl::event kernelEvent2 = queue.parallel_for<class launchIntersectKernel>(
            sycl::range<1>(activeRayCount), [=](sycl::id<1> globalId) {
                const uint32_t rayIndex = globalId[0];
                RayState currentRayState = intermediates.primaryRays[rayIndex];
                // Guard against pathological transparent stacks or self-intersection loops.
                constexpr uint32_t maxInlineNullTraversals = 32;
                for (uint32_t inlineTraversalIndex = 0; inlineTraversalIndex < maxInlineNullTraversals; ++
                     inlineTraversalIndex) {
                    const uint64_t stepSeed = rng::makeSeed(renderSeed, currentRayState.pathId,
                                                            currentRayState.traversalIndex, rng::kStreamTraversal,
                                                            107u);
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
                            appendContributionAtomic(intermediates.countContributions, intermediates.hitContribution,
                                                     intermediates.maxHitContributionCount, contribution);
                        }
                        const GPUMaterial material = scene.materials[instance.materialIndex];
                        if (settings.integratorKind == IntegratorKind::photonMapping && isIndirect && !material.
                            isEmissive()) {
                            depositPhotonSurface(worldHit, currentRayState.ray.direction, worldHit.geometricNormalW,
                                                 currentRayState.pathThroughput, intermediates.map);
                        }
                        float3 sampledOutgoingDirectionW = currentRayState.ray.direction;
                        float sampledPdf = 0.0f;
                        sampleCosineHemisphere(stepRng, worldHit.geometricNormalW, sampledOutgoingDirectionW,
                                               sampledPdf);
                        const float3 throughputMultiplier = material.baseColor;
                        RayState nextRayState{};
                        nextRayState.ray.origin = worldHit.hitPositionW + (worldHit.geometricNormalW * 1e-6f);
                        nextRayState.ray.direction = sampledOutgoingDirectionW;
                        nextRayState.bounceIndex = currentRayState.bounceIndex + 1; // real bounce
                        nextRayState.traversalIndex = currentRayState.traversalIndex + 1;
                        nextRayState.pixelIndex = currentRayState.pixelIndex;
                        nextRayState.pathThroughput = currentRayState.pathThroughput * throughputMultiplier;
                        nextRayState.pathId = currentRayState.pathId;
                        nextRayState.lightIndex = currentRayState.lightIndex;
                        if (!applyRussianRoulette(stepRng, nextRayState.bounceIndex, nextRayState.pathThroughput,
                                                  settings.russianRouletteStart)) { return; }
                        auto extensionCounter = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device, sycl::access::address_space::global_space>(
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
                        const Point &surfel = scene.points[worldHit.primitiveIndex];
                        const float effectiveOpacity = sycl::fmin(
                            1.0f, sycl::fmax(0.0f, worldHit.alphaGeom * surfel.opacity));
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
                            advanceRayPast(currentRayState.ray, worldHit.t);
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
                            appendContributionAtomic(intermediates.countContributions, intermediates.hitContribution,
                                                     intermediates.maxHitContributionCount, contribution);
                        }
                        RayState nextRayState{};
                        nextRayState.ray.origin = worldHit.hitPositionW + orientedNormal * RayEpsilon;
                        nextRayState.ray.direction = sampledOutgoingDirectionW;
                        nextRayState.bounceIndex = currentRayState.bounceIndex + 1;
                        nextRayState.traversalIndex = currentRayState.traversalIndex + 1;
                        nextRayState.pixelIndex = currentRayState.pixelIndex;
                        nextRayState.pathThroughput = nextPathThroughput;
                        nextRayState.pathId = currentRayState.pathId;
                        nextRayState.lightIndex = currentRayState.lightIndex;
                        if (!applyRussianRoulette(stepRng, nextRayState.bounceIndex, nextRayState.pathThroughput,
                                                  settings.russianRouletteStart)) { return; }
                        auto extensionCounter = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device, sycl::access::address_space::global_space>(
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
        kernelEvent2.wait();
    }

    namespace {
        constexpr float kAlphaEpsilon = 1.0e-8f;

        struct CameraGatherInputs {
            const GPUSceneBuffers &scene;
            const PathTracerSettings &settings;
            const SensorGPU &sensor;
            const DeviceSurfacePhotonMapGrid &photonMap;
            const PrimalActivityStats &primalActivityStats;

            float localLayerDepthEpsilon = rendererDebugLocalLayerDepthEpsilon(settings);
            uint32_t maxSplatEventsPerRay = rendererDebugMaxSplatEventsPerRay(settings);
            uint32_t maxLocalSurfelHits = rendererDebugMaxLocalSurfelHits(settings);
            uint32_t pointHitBatchSize = rendererDebugPointHitBatchSize(settings);
            uint32_t pointHitBatchLookaheadCapacity = rendererDebugPointHitBatchLookaheadCapacity(settings);
            float localLayerNormalCosineThreshold = rendererDebugLocalLayerNormalCosineThreshold(settings);
        };

        struct CameraRadianceAccumulation {
            float3 radiance{0.0f};
            float transmittance = 1.0f;
            float weight = 0.0f;
        };

        struct CameraSurfaceDiagnostics {
            float transmittance = 1.0f;
            float weight = 0.0f;
            float weightedDepth = 0.0f;
            float3 weightedNormal{0.0f};
            float intraSlabDepthLoss = 0.0f;
            float intraSlabRayDepthLoss = 0.0f;
            uint32_t intraSlabDepthActiveSlabCount = 0u;
            float depthDistortionWeights[kMaxSplatEventsPerRay];
            float depthDistortionCoordinates[kMaxSplatEventsPerRay];
            uint32_t depthDistortionHitCount = 0u;
            uint32_t hitCount = 0u;
            float distortion = 0.0f;
            bool medianFound = false;
            float medianDepth = 0.0f;
            uint32_t medianPrimitiveIndex = kInvalidIndex;
            float3 medianWorldPosition{0.0f};
            bool keepTracing = false;
        };

        struct CameraGatherProfile {
            uint64_t pointHitQueries = 0u;
            uint64_t pointHitCandidates = 0u;
            uint64_t localLayers = 0u;
            uint64_t localLayerHits = 0u;
            uint64_t objectProfileHits = 0u;
            uint64_t regularizerHits = 0u;
            uint64_t photonGatherCalls = 0u;
            uint64_t directLightCalls = 0u;
            uint64_t directLightLightVisits = 0u;
            uint64_t depthPairIterations = 0u;
            uint64_t meshHits = 0u;
            bool stoppedByNoHit = false;
            bool stoppedByOpacity = false;
        };

        struct CameraGatherPixel {
            Ray originalRay;
            CameraRadianceAccumulation rendering;
            CameraSurfaceDiagnostics surface;
            CameraGatherProfile profile;
        };

        // Surface diagnostics composite individual surfel hits in front-to-back order.
        template<bool ProfileCounters>
        inline bool accumulateRegularizerHit(
            const CameraGatherInputs &inputs,
            const LocalSurfelLayerHit &regularizerHit,
            CameraGatherPixel &pixel) {
            const auto &scene = inputs.scene;
            const auto &settings = inputs.settings;
            const auto &camera = inputs.sensor.camera;
            const auto &originalRay = pixel.originalRay;
            auto &surface = pixel.surface;
            auto &profile = pixel.profile;

            if (regularizerHit.primitiveIndex == kInvalidIndex) {
                return true;
            }
            const Point &surfel = scene.points[regularizerHit.primitiveIndex];
            const float alphaGeom = regularizerHit.alphaGeom;
            const float alpha = surfel.opacity * alphaGeom;
            if (alphaGeom <= kAlphaEpsilon) {
                return true;
            }
            const float depth = dot(regularizerHit.hitPositionW - camera.pos, camera.forward);
            if (depth <= 0.0f) {
                return true;
            }
            if (alpha <= kAlphaEpsilon) {
                return true;
            }

            const float compositeWeight = surface.transmittance * alpha;
            surface.weight += compositeWeight;
            surface.weightedDepth += compositeWeight * depth;

            if (!surface.medianFound) {
                // Retain the last valid candidate for the optional transparent-ray
                // fallback and the existing hit-budget fallback.
                surface.medianDepth = depth;
                surface.medianWorldPosition = regularizerHit.hitPositionW;
                surface.medianPrimitiveIndex = regularizerHit.primitiveIndex;
                surface.medianFound = surface.weight >= settings.medianDepthThreshold;
            }

            if (depthDistortionDepthAccepted(depth, settings.depthDistortionWorldSpace)) {
                const float depthCoordinate = depthDistortionCoordinate(depth, settings.depthDistortionWorldSpace);
                if constexpr (ProfileCounters) {
                    profile.regularizerHits += 1u;
                    profile.depthPairIterations += surface.depthDistortionHitCount;
                }
                for (uint32_t previousIndex = 0u; previousIndex < surface.depthDistortionHitCount; ++previousIndex) {
                    const float depthDifference =
                            depthCoordinate - surface.depthDistortionCoordinates[previousIndex];
                    surface.distortion += surface.depthDistortionWeights[previousIndex] * compositeWeight *
                    (settings.depthDistortionWorldSpace && !settings.depthDistortionSquared
                         ? sycl::fabs(depthDifference)
                         : depthDifference * depthDifference);
                }
                if (surface.depthDistortionHitCount < kMaxSplatEventsPerRay) {
                    surface.depthDistortionWeights[surface.depthDistortionHitCount] = compositeWeight;
                    surface.depthDistortionCoordinates[surface.depthDistortionHitCount] = depthCoordinate;
                    ++surface.depthDistortionHitCount;
                }
            }

            float3 orientedNormalW = normalize(cross(surfel.tanU, surfel.tanV));
            if (dot(orientedNormalW, -originalRay.direction) < 0.0f) { orientedNormalW = -orientedNormalW; }
            surface.weightedNormal += compositeWeight * orientedNormalW;

            surface.transmittance *= 1.0f - alpha;
            return surface.transmittance > kAlphaEpsilon;
        }

        inline void recordCameraSlabActivity(
            const PrimalActivityStats &primalActivityStats,
            const PointCloudLocalLayer &localLayer,
            size_t pointCount,
            float transmittance) {
            // A camera ray deterministically reached this physical slab before
            // opacity termination. Mark every member of the exact slab used by
            // rendering. Score its actual symmetric compositing weight as well;
            // neither statistic depends on adjoint sampling or the loss value.
            if (primalActivityStats.cameraSurfaceHitCount != nullptr &&
                primalActivityStats.numPoints == pointCount) {
                for (uint32_t localHitIndex = 0u;
                     localHitIndex < localLayer.hitCount;
                     ++localHitIndex) {
                    const uint32_t primitiveIndex =
                            localLayer.hits[localHitIndex].primitiveIndex;
                    if (primitiveIndex < primalActivityStats.numPoints) {
                        incrementPrimalActivityCounter(
                            primalActivityStats.cameraSurfaceHitCount[primitiveIndex]);
                        if (primalActivityStats.cameraSurfaceScore != nullptr) {
                            accumulatePrimalActivityScore(
                                primalActivityStats.cameraSurfaceScore[primitiveIndex],
                                transmittance * localLayer.weight[localHitIndex]);
                        }
                    }
                }
            }
        }

        inline void accumulateSlabDepthDiagnostics(
            const CameraGatherInputs &inputs,
            const PointCloudLocalLayer &localLayer,
            const PointCloudLocalLayerConsensus &slabConsensus,
            CameraGatherPixel &pixel) {
            const auto &scene = inputs.scene;
            const auto &sensor = inputs.sensor;
            const auto &originalRay = pixel.originalRay;
            const float localLayerDepthEpsilon = inputs.localLayerDepthEpsilon;
            const float3 anchorPositionW = localLayer.hits[0].hitPositionW;
            auto &surface = pixel.surface;

            const float inverseMemberCount = 1.0f / static_cast<float>(localLayer.hitCount);
            if (sensor.intraSlabRayDepthBuffer != nullptr) {
                // Relative world-space ray depths avoid cancellation far from the camera.
                // Every member has equal influence; no opacity or normal-angle weights.
                float meanDepthOffset = 0.0f;
                for (uint32_t i = 0u; i < localLayer.hitCount; ++i) {
                    meanDepthOffset += dot(
                        localLayer.hits[i].hitPositionW - anchorPositionW, originalRay.direction);
                }
                meanDepthOffset *= inverseMemberCount;
                float rayDepthLoss = 0.0f;
                for (uint32_t i = 0u; i < localLayer.hitCount; ++i) {
                    const float depthOffset = dot(
                        localLayer.hits[i].hitPositionW - anchorPositionW, originalRay.direction);
                    const float residual = (depthOffset - meanDepthOffset) / localLayerDepthEpsilon;
                    rayDepthLoss += residual * residual;
                }
                surface.intraSlabRayDepthLoss += rayDepthLoss * inverseMemberCount;
            }
            // Full, unblended point-to-plane consensus constraint:
            //
            //   x_Q = o + d * B_Q/A_Q
            //   delta_i = n_i . (sg(x_Q) - p_i)
            //   L_Q = 1/|Q| sum_i (delta_i/h)^2.
            //
            // A_Q and B_Q are reconstructed every forward pass, but x_Q is a
            // detached target in the corresponding adjoint.
            if (slabConsensus.valid != 0u) {
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
                            dot(memberNormalW, slabConsensus.pointW - memberSurfel.position);
                    slabLoss += signedPlaneDistance * signedPlaneDistance *
                            inverseSlabThicknessSquared;
                }
                surface.intraSlabDepthLoss += slabLoss * inverseMemberCount;
                ++surface.intraSlabDepthActiveSlabCount;
            }
        }

        template<bool SharedLighting, bool ProfileCounters>
        inline void accumulateSlabMemberRadiance(
            const CameraGatherInputs &inputs,
            const PointCloudLocalLayer &localLayer,
            const Ray &layerRay,
            CameraGatherPixel &pixel) {
            const auto &scene = inputs.scene;
            const auto &settings = inputs.settings;
            const auto &photonMap = inputs.photonMap;
            const auto &primalActivityStats = inputs.primalActivityStats;
            auto &rendering = pixel.rendering;
            auto &profile = pixel.profile;

            for (uint32_t localHitIndex = 0u; localHitIndex < localLayer.hitCount; ++localHitIndex) {
                const float layerWeight = localLayer.weight[localHitIndex];
                if (layerWeight <= 0.0f) continue;
                const LocalSurfelLayerHit &localHit = localLayer.hits[localHitIndex];
                const Point &surfel = scene.points[localHit.primitiveIndex];
                float3 normalW = normalize(cross(surfel.tanU, surfel.tanV));
                if (dot(normalW, -layerRay.direction) < 0.0f) normalW = -normalW;
                const float compositeWeight = rendering.transmittance * layerWeight;
                if constexpr (ProfileCounters) {
                    profile.photonGatherCalls += 1u;
                }
                const float3 indirectIrradiance = gatherDiffuseIrradianceAtPoint(
                    localHit.hitPositionW, normalW, photonMap);
                const float3 indirectRadiance = indirectIrradiance * (surfel.alpha_r * surfel.albedo * M_1_PIf);
                const float surfelArea = M_PIf * surfel.scale.x() * surfel.scale.y();
                float3 emittedRadiance{0.0f};
                if (surfelArea > 1.0e-12f && sycl::isfinite(surfelArea)) {
                    emittedRadiance = surfel.albedo * (surfel.flux / (M_PIf * surfelArea));
                }
                float3 directRadiance{0.0f};
                if constexpr (!SharedLighting) {
                    if constexpr (ProfileCounters) {
                        profile.directLightCalls += 1u;
                        profile.directLightLightVisits += scene.lightCount;
                    }
                    directRadiance = estimateDirectSurfelPointLights(
                        scene,
                        settings,
                        localHit.hitPositionW,
                        normalW,
                        surfel, -layerRay.direction,
                        localLayer.directLightEpsilon[localHitIndex],
                        primalActivityStats.shadowOccluderHitCount,
                        primalActivityStats.numPoints,
                        primalActivityStats.shadowOccluderScore,
                        compositeWeight);
                }
                rendering.radiance += compositeWeight * (emittedRadiance + indirectRadiance + directRadiance);
            }
        }

        template<bool ProfileCounters>
        inline void accumulateSharedSlabDirectLighting(
            const CameraGatherInputs &inputs,
            const PointCloudLocalLayer &localLayer,
            const Ray &layerRay,
            const float3 &sharedSlabPositionW,
            CameraGatherPixel &pixel) {
            const auto &scene = inputs.scene;
            const auto &settings = inputs.settings;
            const auto &primalActivityStats = inputs.primalActivityStats;
            auto &rendering = pixel.rendering;
            auto &profile = pixel.profile;

            if constexpr (ProfileCounters) {
                profile.directLightCalls += 1u;
                profile.directLightLightVisits += scene.lightCount;
            }

            const LocalSurfelLayerHit &anchorHit = localLayer.hits[0];
            float3 anchorNormalW = localLayer.referenceNormalW;
            if (dot(anchorNormalW, anchorNormalW) <= 1.0e-16f) {
                const Point &anchorSurfel = scene.points[anchorHit.primitiveIndex];
                anchorNormalW = normalize(cross(anchorSurfel.tanU, anchorSurfel.tanV));
                if (dot(anchorNormalW, -layerRay.direction) < 0.0f) anchorNormalW = -anchorNormalW;
            }
            const float sharedDirectLightEpsilon = localLayer.directLightEpsilon[0];
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
                float surfaceCosines[kMaxLocalSurfelHits]{};
                bool hasLightFacingMember = false;
                float shadowReceiverWeight = 0.0f;
                for (uint32_t localHitIndex = 0u; localHitIndex < localLayer.hitCount; ++localHitIndex) {
                    if (localLayer.weight[localHitIndex] <= 0.0f) continue;
                    const Point &surfel = scene.points[localLayer.hits[localHitIndex].primitiveIndex];
                    float3 normalW = normalize(cross(surfel.tanU, surfel.tanV));
                    if (dot(normalW, -layerRay.direction) < 0.0f) normalW = -normalW;
                    const float surfaceCosine = sycl::fmax(0.0f, dot(normalW, lightDirection));
                    surfaceCosines[localHitIndex] = surfaceCosine;
                    hasLightFacingMember |= surfaceCosine > 0.0f;
                    if (surfaceCosine > 0.0f) {
                        shadowReceiverWeight += rendering.transmittance * localLayer.weight[localHitIndex];
                    }
                }
                // Members can have different normals; reject only if none can receive this light.
                if (!hasLightFacingMember) {
                    continue;
                }
                const float shadowTransmission =
                        traceShadowTransmissionToPoint(
                            scene,
                            settings,
                            sharedSlabPositionW,
                            anchorNormalW,
                            lightPositionW,
                            sharedDirectLightEpsilon,
                            primalActivityStats.shadowOccluderHitCount,
                            primalActivityStats.numPoints,
                            primalActivityStats.shadowOccluderScore,
                            shadowReceiverWeight);
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
                    const float surfaceCosine = surfaceCosines[localHitIndex];
                    if (surfaceCosine <= 0.0f) {
                        continue;
                    }
                    float3 brdf = surfel.alpha_r * surfel.albedo * M_1_PIf;
                    if (settings.surfelBrdfKind == SurfelBrdfKind::Ggx) {
                        float3 normalW = normalize(cross(surfel.tanU, surfel.tanV));
                        if (dot(normalW, -layerRay.direction) < 0.0f) normalW = -normalW;
                        brdf = evaluateSurfelBrdf(
                            settings.stableGgx, settings.surfelBrdfKind, surfel, normalW, lightDirection, -layerRay.direction);
                    }
                    const float compositeWeight = rendering.transmittance * layerWeight;
                    rendering.radiance +=
                            compositeWeight * brdf * sharedIncident * surfaceCosine;
                }
            }
        }

        template<bool SharedLighting, bool ProfileCounters>
        inline void renderPointLocalLayer(
            const CameraGatherInputs &inputs,
            const PointCloudLocalLayer &localLayer,
            const Ray &layerRay,
            CameraGatherPixel &pixel) {
            if constexpr (ProfileCounters) {
                pixel.profile.localLayers += 1u;
                pixel.profile.localLayerHits += localLayer.hitCount;
                pixel.profile.objectProfileHits += localLayer.hitCount;
            }
            if (localLayer.hitCount == 0u) {
                return;
            }

            auto &rendering = pixel.rendering;
            recordCameraSlabActivity(
                inputs.primalActivityStats, localLayer, inputs.scene.pointCount, rendering.transmittance);

            PointCloudLocalLayerConsensus slabConsensus{};
            const bool computeSlabDepthDiagnostics =
                    inputs.settings.computeSurfaceDiagnostics && localLayer.hitCount > 1u;
            if (SharedLighting || computeSlabDepthDiagnostics) {
                slabConsensus = computePointCloudLocalLayerConsensus(localLayer, pixel.originalRay, inputs.scene);
            }
            if (computeSlabDepthDiagnostics) {
                accumulateSlabDepthDiagnostics(inputs, localLayer, slabConsensus, pixel);
            }

            accumulateSlabMemberRadiance<SharedLighting, ProfileCounters>(inputs, localLayer, layerRay, pixel);
            if constexpr (SharedLighting) {
                const float3 sharedSlabPositionW =
                        slabConsensus.valid != 0u ? slabConsensus.pointW : localLayer.hits[0].hitPositionW;
                accumulateSharedSlabDirectLighting<ProfileCounters>(
                    inputs, localLayer, layerRay, sharedSlabPositionW, pixel);
            }

            if (SharedLighting || localLayer.opacity > kAlphaEpsilon) {
                rendering.weight += rendering.transmittance * localLayer.opacity;
            }
            rendering.transmittance *= localLayer.transmission;
        }

        template<bool SharedLighting, bool ProfileCounters>
        inline void traceBatchedCameraSlabs(const CameraGatherInputs &inputs, CameraGatherPixel &pixel) {
            const auto &scene = inputs.scene;
            const auto &settings = inputs.settings;
            const uint32_t maxSplatEventsPerRay = inputs.maxSplatEventsPerRay;
            const uint32_t pointHitBatchSize = inputs.pointHitBatchSize;
            const uint32_t pointHitBatchLookaheadCapacity = inputs.pointHitBatchLookaheadCapacity;
            auto &rendering = pixel.rendering;
            auto &surface = pixel.surface;
            auto &profile = pixel.profile;
            Ray renderingRay = pixel.originalRay;

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
                    pointInstanceIndex);
                if constexpr (ProfileCounters) {
                    profile.pointHitQueries += 1u;
                    profile.pointHitCandidates += hitCount;
                }

                if (hitCount == 0u) {
                    if constexpr (ProfileCounters) {
                        profile.stoppedByNoHit = true;
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
                        inputs.localLayerDepthEpsilon,
                        inputs.maxLocalSurfelHits,
                        inputs.localLayerNormalCosineThreshold,
                        settings.rendererDebugLocalLayerDepthMode);

                    renderPointLocalLayer<SharedLighting, ProfileCounters>(inputs, localLayer, renderingRay, pixel);
                    ++traversalIndex;
                    furthestConsumedT = sycl::fmax(furthestConsumedT, localLayer.furthestT);

                    while (hitCursor < hitCount && pointHits[hitCursor].tWorld <= localLayer.furthestT + RayEpsilon) {
                        ++hitCursor;
                    }
                    if (hitCursor == oldHitCursor) {
                        ++hitCursor;
                    }
                    if (rendering.transmittance <= kCameraTransmittanceCutoff) {
                        if constexpr (ProfileCounters) {
                            profile.stoppedByOpacity = true;
                        }
                        break;
                    }
                }

                if (surface.keepTracing) {
                    for (uint32_t batchIndex = 0u;
                         batchIndex < hitCount && surface.hitCount < maxSplatEventsPerRay;
                         ++batchIndex) {
                        const LocalSurfelLayerHit &pointHit = pointHits[batchIndex];
                        if (pointHit.tWorld > furthestConsumedT + RayEpsilon) {
                            break;
                        }

                        surface.keepTracing = accumulateRegularizerHit<ProfileCounters>(inputs, pointHit, pixel);
                        ++surface.hitCount;
                        if (!surface.keepTracing) {
                            break;
                        }
                    }
                }

                const bool consumedAllFetchedHits = hitCursor >= hitCount;
                if (furthestConsumedT <= 0.0f) {
                    break;
                }

                advanceRayPast(renderingRay, furthestConsumedT, RayEpsilon);
                if (rendering.transmittance <= kCameraTransmittanceCutoff) {
                    break;
                }
                if (consumedAllFetchedHits && hitCount < pointHitBatchLookaheadCapacity) {
                    if constexpr (ProfileCounters) {
                        profile.stoppedByNoHit = rendering.transmittance > kCameraTransmittanceCutoff;
                    }
                    break;
                }
            }
        }

        template<bool ProfileCounters>
        inline void renderCameraMeshHit(
            const CameraGatherInputs &inputs,
            const WorldHit &worldHit,
            const InstanceRecord &instance,
            const Ray &renderingRay,
            CameraGatherPixel &pixel) {
            const auto &scene = inputs.scene;
            const auto &settings = inputs.settings;
            const auto &photonMap = inputs.photonMap;
            auto &rendering = pixel.rendering;
            auto &profile = pixel.profile;

            if constexpr (ProfileCounters) {
                profile.meshHits += 1u;
            }
            const GPUMaterial &material = scene.materials[instance.materialIndex];
            const bool isBackfaceHit = dot(renderingRay.direction, worldHit.geometricNormalW) > 0.0f;
            const float3 normalW = isBackfaceHit ? -worldHit.geometricNormalW : worldHit.geometricNormalW;
            rendering.weight += rendering.transmittance;
            if (material.isEmissive()) {
                const float3 emittedRadiance = material.power * material.baseColor;
                rendering.radiance += rendering.transmittance * min(emittedRadiance, 1.0f);
            } else {
                if constexpr (ProfileCounters) {
                    profile.photonGatherCalls += 1u;
                    profile.directLightCalls += 1u;
                    profile.directLightLightVisits += scene.lightCount;
                }
                const float3 indirectIrradiance = gatherDiffuseIrradianceAtPoint(
                    worldHit.hitPositionW, normalW, photonMap);
                const float3 indirectRadiance = (material.baseColor * M_1_PIf) * indirectIrradiance;
                const float3 directRadiance = estimateDirectPointSampledPointLights(
                    scene, settings, worldHit.hitPositionW, normalW, material.baseColor,
                    inputs.localLayerDepthEpsilon);
                rendering.radiance += rendering.transmittance * (indirectRadiance + directRadiance);
            }
            rendering.transmittance = 0.0f;
            if constexpr (ProfileCounters) {
                profile.stoppedByOpacity = true;
            }
        }

        template<bool SharedLighting, bool ProfileCounters>
        inline void traceCameraScene(const CameraGatherInputs &inputs, CameraGatherPixel &pixel) {
            const auto &scene = inputs.scene;
            const auto &settings = inputs.settings;
            const uint32_t maxSplatEventsPerRay = inputs.maxSplatEventsPerRay;
            auto &rendering = pixel.rendering;
            auto &surface = pixel.surface;
            auto &profile = pixel.profile;
            Ray renderingRay = pixel.originalRay;

            for (uint32_t traversalIndex = 0u; traversalIndex < maxSplatEventsPerRay; ++traversalIndex) {
                WorldHit worldHit{};
                if constexpr (ProfileCounters) {
                    profile.pointHitQueries += 1u;
                }
                intersectScene(renderingRay, &worldHit, scene, SurfelIntersectMode::FirstHit);
                if (!worldHit.hit) {
                    if constexpr (ProfileCounters) {
                        profile.stoppedByNoHit = true;
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
                        inputs.localLayerDepthEpsilon,
                        inputs.maxLocalSurfelHits,
                        inputs.localLayerNormalCosineThreshold,
                        settings.rendererDebugLocalLayerDepthMode);
                    if constexpr (ProfileCounters) {
                        profile.pointHitCandidates += localLayer.hitCount;
                    }
                    if (surface.keepTracing) {
                        for (uint32_t localHitIndex = 0u;
                             localHitIndex < localLayer.hitCount && surface.hitCount < maxSplatEventsPerRay;
                             ++localHitIndex) {
                            surface.keepTracing = accumulateRegularizerHit<ProfileCounters>(
                                inputs, localLayer.hits[localHitIndex], pixel);
                            ++surface.hitCount;
                            if (!surface.keepTracing) {
                                break;
                            }
                        }
                    }
                    renderPointLocalLayer<SharedLighting, ProfileCounters>(inputs, localLayer, renderingRay, pixel);
                    advanceRayPast(renderingRay, localLayer.furthestT, RayEpsilon);
                    // Match batched traversal: later slabs cannot contribute
                    // after opacity termination, including through their shadow
                    // connections. Do not mark them as live for pruning.
                    if (rendering.transmittance <= kCameraTransmittanceCutoff) {
                        if constexpr (ProfileCounters) {
                            profile.stoppedByOpacity = true;
                        }
                        break;
                    }
                    continue;
                }
                if (instance.geometryType == GeometryType::Mesh) {
                    renderCameraMeshHit<ProfileCounters>(inputs, worldHit, instance, renderingRay, pixel);
                    break;
                }
            }
        }

        inline void storeCameraGatherOutputs(
            const CameraGatherInputs &inputs, uint32_t pixelIndex, const CameraGatherPixel &pixel) {
            const auto &sensor = inputs.sensor;
            const auto &settings = inputs.settings;
            const auto &rendering = pixel.rendering;
            const auto &surface = pixel.surface;

            sensor.framebuffer[pixelIndex] = float4{
                rendering.radiance.x(), rendering.radiance.y(), rendering.radiance.z(),
                sycl::clamp(rendering.weight, 0.0f, 1.0f)
            };

            // Every forward output is overwritten by its producing kernel. Reset
            // only the adjoint inputs here, avoiding 15 separate blocking clears.
            if (sensor.depthDistortionAdjointBuffer) sensor.depthDistortionAdjointBuffer[pixelIndex] = 0.0f;
            if (sensor.intraSlabDepthAdjointBuffer) sensor.intraSlabDepthAdjointBuffer[pixelIndex] = 0.0f;
            if (sensor.visiblePrimitiveIndexBuffer != nullptr) {
                sensor.visiblePrimitiveIndexBuffer[pixelIndex] = UINT32_MAX;
            }
            sensor.depthDistortionBuffer[pixelIndex] = surface.distortion;
            sensor.intraSlabDepthBuffer[pixelIndex] = surface.intraSlabDepthLoss;
            if (sensor.intraSlabRayDepthBuffer != nullptr) {
                sensor.intraSlabRayDepthBuffer[pixelIndex] = surface.intraSlabRayDepthLoss;
            }
            sensor.intraSlabDepthActiveSlabCountBuffer[pixelIndex] =
                    surface.intraSlabDepthActiveSlabCount;
            if (surface.weight > 1.0e-6f) {
                sensor.meanDepthBuffer[pixelIndex] = surface.weightedDepth / surface.weight;
            } else {
                sensor.meanDepthBuffer[pixelIndex] = 0.0f;
            }
            // A truncated stream still supplies a surface depth for diagnostics
            // and normal consistency. Optionally keep the last accepted hit on
            // uncapped transparent rays too. Empty rays still have no depth.
            const bool keepLastDepth =
                    settings.medianDepthRetainLastHit || surface.hitCount >= inputs.maxSplatEventsPerRay;
            const bool surfaceDepthValid = surface.medianFound || (keepLastDepth && surface.medianDepth > 0.0f);
            if (sensor.medianPrimitiveIndexBuffer) {
                sensor.medianPrimitiveIndexBuffer[pixelIndex] =
                        surfaceDepthValid ? surface.medianPrimitiveIndex : kInvalidIndex;
            }
            if (surfaceDepthValid) {
                sensor.medianDepthBuffer[pixelIndex] = surface.medianDepth;
                sensor.medianWorldPositionBuffer[pixelIndex] = float4{
                    surface.medianWorldPosition.x(), surface.medianWorldPosition.y(),
                    surface.medianWorldPosition.z(), 1.0f
                };
            } else {
                sensor.medianDepthBuffer[pixelIndex] = 0.0f;
                sensor.medianWorldPositionBuffer[pixelIndex] = float4{0.0f};
            }
            // Normal RGBA stores accumulated opacity in W (zero also marks invalid pixels).
            if (surfaceDepthValid) {
                sensor.visibleNormalBuffer[pixelIndex] = float4{
                    surface.weightedNormal.x(), surface.weightedNormal.y(),
                    surface.weightedNormal.z(), surface.weight
                };
            } else {
                sensor.visibleNormalBuffer[pixelIndex] = float4{0.0f};
            }
        }

        inline void flushCameraGatherProfile(
            RenderProfilingCounters *counters, const CameraGatherPixel &pixel) {
            const auto &profile = pixel.profile;
            const bool reachedSplatLimit =
                    !profile.stoppedByNoHit && !profile.stoppedByOpacity &&
                    pixel.rendering.transmittance > kCameraTransmittanceCutoff;
            addRenderProfileCounter(&counters->forwardGatherPixels, 1u);
            addRenderProfileCounter(&counters->forwardGatherPointHitQueries, profile.pointHitQueries);
            addRenderProfileCounter(&counters->forwardGatherPointHitCandidates, profile.pointHitCandidates);
            addRenderProfileCounter(&counters->forwardGatherLocalLayers, profile.localLayers);
            addRenderProfileCounter(&counters->forwardGatherLocalLayerHits, profile.localLayerHits);
            addRenderProfileCounter(&counters->forwardGatherObjectProfileHits, profile.objectProfileHits);
            addRenderProfileCounter(&counters->forwardGatherRegularizerHits, profile.regularizerHits);
            addRenderProfileCounter(&counters->forwardGatherPhotonGatherCalls, profile.photonGatherCalls);
            addRenderProfileCounter(&counters->forwardGatherDirectLightCalls, profile.directLightCalls);
            addRenderProfileCounter(&counters->forwardGatherDirectLightLightVisits, profile.directLightLightVisits);
            addRenderProfileCounter(&counters->forwardGatherDepthPairIterations, profile.depthPairIterations);
            addRenderProfileCounter(&counters->forwardGatherMeshHits, profile.meshHits);
            addRenderProfileCounter(&counters->forwardGatherNoHitTerminations, profile.stoppedByNoHit);
            addRenderProfileCounter(&counters->forwardGatherOpacityTerminations, profile.stoppedByOpacity);
            addRenderProfileCounter(&counters->forwardGatherMaxSplatTerminations, reachedSplatLimit);
        }

        template<bool SharedLighting, bool ProfileCounters>
        inline void renderCameraGatherPixel(
            const GPUSceneBuffers &sceneCapture,
            const PathTracerSettings &settings,
            const SensorGPU &sensor,
            const DeviceSurfacePhotonMapGrid &photonMap,
            const PrimalActivityStats &primalActivityStats,
            uint32_t pixelIndex) {
            GPUSceneBuffers scene = sceneCapture;
            if constexpr (!ProfileCounters) {
                scene.profileCounters = nullptr;
            }
            const CameraGatherInputs inputs{scene, settings, sensor, photonMap, primalActivityStats};

            const uint32_t pixelX = pixelIndex % sensor.camera.width;
            const uint32_t pixelY = pixelIndex / sensor.camera.width;
            CameraGatherPixel pixel;
            pixel.originalRay = makePrimaryRayFromPixelJitteredFov(
                sensor.camera, static_cast<float>(pixelX), static_cast<float>(pixelY), 0.0f, 0.0f);
            pixel.surface.keepTracing = settings.computeSurfaceDiagnostics;

            uint32_t pointInstanceIndex = kInvalidIndex;
            if (inputs.pointHitBatchSize > 1u && tryGetSinglePointCloudInstance(scene, pointInstanceIndex)) {
                traceBatchedCameraSlabs<SharedLighting, ProfileCounters>(inputs, pixel);
            } else {
                traceCameraScene<SharedLighting, ProfileCounters>(inputs, pixel);
            }
            storeCameraGatherOutputs(inputs, pixelIndex, pixel);

            if constexpr (ProfileCounters) {
                flushCameraGatherProfile(scene.profileCounters, pixel);
            }
        }
    } // namespace

    // Host-selected variants remove unused lighting branches and profiling state
    // from each work item without changing samples, hit limits, or transport.
    template<bool SharedLighting, bool ProfileCounters>
    struct CameraRgbGatherKernel {
        GPUSceneBuffers scene;
        PathTracerSettings settings;
        SensorGPU sensor;
        DeviceSurfacePhotonMapGrid photonMap;
        PrimalActivityStats primalActivityStats;

        void operator()(sycl::id<1> tid) const {
            renderCameraGatherPixel<SharedLighting, ProfileCounters>(
                scene, settings, sensor, photonMap, primalActivityStats, static_cast<uint32_t>(tid[0]));
        }
    };

    template<bool SharedLighting, bool ProfileCounters>
    static void launchCameraRgbGatherKernel(RenderPackage &pkg, uint32_t cameraIndex) {
        const auto &sensor = pkg.sensors[cameraIndex];
        const uint32_t pixelCount = sensor.camera.width * sensor.camera.height;

        ScopedTimer gatherTimer("Forward camera: RGB gather", spdlog::level::debug);
        using Kernel = CameraRgbGatherKernel<SharedLighting, ProfileCounters>;
        sycl::event event = pkg.queue.parallel_for<Kernel>(
            sycl::range<1>(pixelCount), Kernel{
                pkg.scene, pkg.settings, sensor, pkg.intermediates.map, pkg.primalActivityStats
            });
        waitForForwardKernelTiming(event);
        gatherTimer.stop();
    }

    static void launchCameraRgbGather(RenderPackage &pkg, uint32_t cameraIndex) {
        const bool sharedLighting = pkg.settings.rendererDebugShareLocalLayerDirectLighting;
        const bool profile = pkg.scene.profileCounters != nullptr;
        if (sharedLighting && profile) {
            launchCameraRgbGatherKernel<true, true>(pkg, cameraIndex);
        } else if (sharedLighting) {
            launchCameraRgbGatherKernel<true, false>(pkg, cameraIndex);
        } else if (profile) {
            launchCameraRgbGatherKernel<false, true>(pkg, cameraIndex);
        } else {
            launchCameraRgbGatherKernel<false, false>(pkg, cameraIndex);
        }
    }

    static void computeNormalFromDepth(
        const SensorGPU &sensor, bool useMeanDepth, uint32_t pixelIndex) {
        const uint32_t imageWidth = sensor.camera.width;
        const uint32_t imageHeight = sensor.camera.height;
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
    }

    struct SurfaceDepthNormalKernel {
        SensorGPU sensor;
        bool useMeanDepth;

        void operator()(sycl::id<1> tid) const {
            computeNormalFromDepth(sensor, useMeanDepth, static_cast<uint32_t>(tid[0]));
        }
    };

    static sycl::event submitCameraNormalsFromDepth(
        sycl::queue &queue, const SensorGPU &sensor, bool useMeanDepth) {
        const uint32_t pixelCount = sensor.camera.width * sensor.camera.height;
        return queue.parallel_for<SurfaceDepthNormalKernel>(
            sycl::range<1>(pixelCount), SurfaceDepthNormalKernel{sensor, useMeanDepth});
    }

    static void findVisibleSlabPrimitive(
        const GPUSceneBuffers &scene,
        const PathTracerSettings &settings,
        const SensorGPU &sensor,
        float slabThickness,
        uint32_t pixelIndex) {
        const uint32_t imageWidth = sensor.camera.width;
        const uint32_t pixelX = pixelIndex % imageWidth;
        const uint32_t pixelY = pixelIndex / imageWidth;

        if (slabThickness <= 0.0f) { return; }
        const float centerDepth = settings.normalFromDepthUseMeanDepth
                                      ? sensor.meanDepthBuffer[pixelIndex]
                                      : sensor.medianDepthBuffer[pixelIndex];
        if (!sycl::isfinite(centerDepth) || centerDepth <= 0.0f) { return; }

        // Locate the local slab corresponding to the visible pseudo surface.
        // Retrace the ray and select the slab closest to the pseudo-surface.
        Ray visiblePrimitiveRay = makePrimaryRayFromPixelJitteredFov(
            sensor.camera, static_cast<float>(pixelX), static_cast<float>(pixelY), 0.0f, 0.0f);
        const uint32_t maxSplatEventsPerRay = rendererDebugMaxSplatEventsPerRay(settings);
        const uint32_t maxLocalSurfelHits = rendererDebugMaxLocalSurfelHits(settings);
        const float localLayerNormalCosineThreshold = rendererDebugLocalLayerNormalCosineThreshold(settings);

        float closestSurfaceDepthDifference = std::numeric_limits<float>::infinity();
        PointCloudLocalLayer selectedLayer{};
        bool foundPointSlab = false;

        for (uint32_t traversalIndex = 0u; traversalIndex < maxSplatEventsPerRay; ++traversalIndex) {
            WorldHit worldHit{};
            intersectScene(visiblePrimitiveRay, &worldHit, scene, SurfelIntersectMode::FirstHit);
            if (!worldHit.hit) { break; }

            buildIntersectionNormal(scene, worldHit);
            const InstanceRecord &instance = scene.instances[worldHit.instanceIndex];
            if (instance.geometryType != GeometryType::PointCloud) { break; }

            const PointCloudLocalLayer localLayer = collectPointCloudLocalLayer(
                visiblePrimitiveRay, worldHit, instance, scene, slabThickness, maxLocalSurfelHits,
                localLayerNormalCosineThreshold, settings.rendererDebugLocalLayerDepthMode);
            if (localLayer.hitCount == 0u) { break; }

            const LocalSurfelLayerHit &anchorHit = localLayer.hits[0];
            const float anchorSurfaceDepth = dot(anchorHit.hitPositionW - sensor.camera.pos, sensor.camera.forward);
            const float surfaceDepthDifference = sycl::fabs(anchorSurfaceDepth - centerDepth);

            if (surfaceDepthDifference < closestSurfaceDepthDifference) {
                selectedLayer = localLayer;
                closestSurfaceDepthDifference = surfaceDepthDifference;
                foundPointSlab = true;
            }

            advanceRayPast(visiblePrimitiveRay, localLayer.furthestT, RayEpsilon);
        }

        if (!foundPointSlab) {
            return;
        }

        if (sensor.visiblePrimitiveIndexBuffer != nullptr) {
            // Use the strongest renderer contributor as the displayed
            // primitive identity within the selected visible slab.
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
            sensor.visiblePrimitiveIndexBuffer[pixelIndex] =
                    selectedLayer.hits[dominantHitIndex].primitiveIndex;
        }
    }

    struct VisiblePrimitiveKernel2 {
        GPUSceneBuffers scene;
        PathTracerSettings settings;
        SensorGPU sensor;
        float slabThickness;

        void operator()(sycl::id<1> tid) const {
            findVisibleSlabPrimitive(scene, settings, sensor, slabThickness, static_cast<uint32_t>(tid[0]));
        }
    };

    static void launchVisibleSlabSearch(RenderPackage &pkg, uint32_t cameraIndex) {
        const auto &sensor = pkg.sensors[cameraIndex];
        const float slabThickness = rendererDebugLocalLayerDepthEpsilon(pkg.settings);
        const uint32_t pixelCount = sensor.camera.width * sensor.camera.height;

        ScopedTimer timer("Forward camera: visible slab search", spdlog::level::debug);
        sycl::event event = pkg.queue.parallel_for<VisiblePrimitiveKernel2>(
            sycl::range<1>(pixelCount), VisiblePrimitiveKernel2{pkg.scene, pkg.settings, sensor, slabThickness});
        waitForForwardKernelTiming(event);
        timer.stop();
    }

    void launchCameraGatherKernel2(RenderPackage &pkg, uint32_t cameraIndex, uint32_t /*gatherPass*/) {
        const auto &settings = pkg.settings;
        if (settings.sharedHeightEnabled) {
            launchSharedHeightForward(pkg, cameraIndex);
            return;
        }

        launchCameraRgbGather(pkg, cameraIndex);

        if (settings.computeDepthNormalDiagnostics || settings.normalConsistencyWeight != 0.0f) {
            ScopedTimer timer("Forward camera: normals from depth", spdlog::level::debug);
            sycl::event event = submitCameraNormalsFromDepth(
                pkg.queue, pkg.sensors[cameraIndex], settings.normalFromDepthUseMeanDepth);
            waitForForwardKernelTiming(event);
            timer.stop();
        }
        if (settings.computeVisiblePrimitiveDiagnostics) {
            launchVisibleSlabSearch(pkg, cameraIndex);
        }
    }

    namespace {
        struct LegacyCameraGatherInputs {
            const GPUSceneBuffers &scene;
            const PathTracerSettings &settings;
            const DeviceSurfacePhotonMapGrid &photonMap;
            const SensorGPU &sensor;
        };

        struct LegacyCameraGatherAccumulation {
            float3 radiance{0.0f};
            float transmittance = 1.0f;
            float compositeWeight = 0.0f;
            uint32_t pointHitCount = 0u;

            float weightedDepth = 0.0f;
            float depthWeight = 0.0f;
            bool medianFound = false;
            float medianDepth = 0.0f;
            float3 medianWorldPosition{0.0f};
            float3 medianNormal{0.0f};

            float depthDistortionWeights[kMaxSplatEventsPerRay];
            float depthDistortionCoordinates[kMaxSplatEventsPerRay];
            uint32_t depthDistortionHitCount = 0u;
            float distortion = 0.0f;
        };

        inline void accumulatePointHit(
            const LegacyCameraGatherInputs &inputs,
            const Ray &primaryRay,
            const LocalSurfelLayerHit &pointHit,
            LegacyCameraGatherAccumulation &accumulated) {
            const auto &scene = inputs.scene;
            const auto &settings = inputs.settings;
            const auto &photonMap = inputs.photonMap;
            const auto &sensor = inputs.sensor;

            ++accumulated.pointHitCount;
            const Point &surfel = scene.points[pointHit.primitiveIndex];
            float3 normalW = normalize(cross(surfel.tanU, surfel.tanV));
            if (dot(normalW, -primaryRay.direction) < 0.0f) {
                normalW = -normalW;
            }
            const float alpha = surfel.opacity * pointHit.alphaGeom;
            const float3 indirectIrradiance =
                    gatherDiffuseIrradianceAtPoint(pointHit.hitPositionW, normalW, photonMap);

            const float3 indirectRadiance =
                    indirectIrradiance * (surfel.alpha_r * surfel.albedo * M_1_PIf) * alpha;

            const float surfelArea = M_PIf * surfel.scale.x() * surfel.scale.y();
            float3 emittedRadiance{0.0f};
            if (surfelArea > 1.0e-12f && sycl::isfinite(surfelArea)) {
                emittedRadiance = surfel.albedo * (surfel.flux / (M_PIf * surfelArea)) * alpha;
            }
            const float3 directRadiance = estimateDirectSurfelPointLights(
                                              scene, settings, pointHit.hitPositionW, normalW,
                                              surfel, -primaryRay.direction,
                                              RayEpsilon) * alpha;

            const float3 outgoingRadiance = emittedRadiance + indirectRadiance + directRadiance;
            accumulated.radiance += accumulated.transmittance * outgoingRadiance;
            const float compositeWeight = accumulated.transmittance * alpha;
            accumulated.transmittance *= 1.0f - alpha;
            accumulated.compositeWeight += compositeWeight;
            if (!settings.computeSurfaceDiagnostics) {
                return;
            }
            const float depth = dot(pointHit.hitPositionW - sensor.camera.pos, sensor.camera.forward);
            accumulated.depthWeight += compositeWeight;
            accumulated.weightedDepth += compositeWeight * depth;
            if (!accumulated.medianFound && compositeWeight > 0.0f && depth > 0.0f) {
                accumulated.medianDepth = depth;
                accumulated.medianWorldPosition = pointHit.hitPositionW;
                accumulated.medianNormal = normalW;
                accumulated.medianFound = accumulated.compositeWeight >= settings.medianDepthThreshold;
            }
            if (depthDistortionDepthAccepted(depth, settings.depthDistortionWorldSpace)) {
                const float depthCoordinate = depthDistortionCoordinate(depth, settings.depthDistortionWorldSpace);
                for (uint32_t previousIndex = 0u;
                     previousIndex < accumulated.depthDistortionHitCount;
                     ++previousIndex) {
                    const float depthDifference =
                            depthCoordinate - accumulated.depthDistortionCoordinates[previousIndex];
                    accumulated.distortion += accumulated.depthDistortionWeights[previousIndex] * compositeWeight *
                    (settings.depthDistortionWorldSpace && !settings.depthDistortionSquared
                         ? sycl::fabs(depthDifference)
                         : depthDifference * depthDifference);
                }
                if (accumulated.depthDistortionHitCount < kMaxSplatEventsPerRay) {
                    accumulated.depthDistortionWeights[accumulated.depthDistortionHitCount] = compositeWeight;
                    accumulated.depthDistortionCoordinates[accumulated.depthDistortionHitCount] = depthCoordinate;
                    ++accumulated.depthDistortionHitCount;
                }
            }
        }

        inline void renderLegacyCameraGatherPixel(
            const LegacyCameraGatherInputs &inputs, uint32_t pixelIndex) {
            const auto &scene = inputs.scene;
            const auto &settings = inputs.settings;
            const auto &photonMap = inputs.photonMap;
            const auto &sensor = inputs.sensor;
            const uint32_t pixelX = pixelIndex % sensor.camera.width;
            const uint32_t pixelY = pixelIndex / sensor.camera.width;
            Ray primaryRay = makePrimaryRayFromPixelJitteredFov(
                sensor.camera, static_cast<float>(pixelX), static_cast<float>(pixelY), 0.0f, 0.0f);

            LegacyCameraGatherAccumulation accumulated;
            const uint32_t maxSplatEventsPerRay = rendererDebugMaxSplatEventsPerRay(settings);
            const uint32_t pointHitBatchSize = rendererDebugPointHitBatchSize(settings);
            uint32_t directPointInstanceIndex = kInvalidIndex;
            const bool canUsePointHitBatches =
                    pointHitBatchSize > 1u && tryGetSinglePointCloudInstance(scene, directPointInstanceIndex);

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
                        pointInstanceIndex);

                    if (hitCount == 0u) { break; }

                    float furthestConsumedT = 0.0f;
                    for (uint32_t batchIndex = 0u; batchIndex < hitCount; ++batchIndex) {
                        const LocalSurfelLayerHit &pointHit = pointHits[batchIndex];
                        furthestConsumedT = sycl::fmax(furthestConsumedT, pointHit.tWorld);
                        accumulatePointHit(inputs, primaryRay, pointHit, accumulated);
                        ++traversalIndex;
                    }

                    if (furthestConsumedT <= 0.0f) { break; }

                    advanceRayPast(primaryRay, furthestConsumedT, RayEpsilon);
                    if (hitCount < batchCapacity) { break; }
                }
            } else {
                for (uint32_t traversalIndex = 0u; traversalIndex < maxSplatEventsPerRay; ++traversalIndex) {
                    WorldHit worldHit{};
                    intersectScene(primaryRay, &worldHit, scene, SurfelIntersectMode::FirstHit);
                    if (!worldHit.hit) { break; }
                    buildIntersectionNormal(scene, worldHit);
                    const auto &instance = scene.instances[worldHit.instanceIndex];
                    // Visible point-cloud layer
                    if (instance.geometryType == GeometryType::PointCloud) {
                        LocalSurfelLayerHit pointHit{};
                        pointHit.tWorld = worldHit.t;
                        pointHit.primitiveIndex = worldHit.primitiveIndex;
                        pointHit.alphaGeom = worldHit.alphaGeom;
                        pointHit.hitPositionW = worldHit.hitPositionW;
                        pointHit.uv = phiInverse(worldHit.hitPositionW, scene.points[worldHit.primitiveIndex]);
                        accumulatePointHit(inputs, primaryRay, pointHit, accumulated);
                        advanceRayPast(primaryRay, worldHit.t);
                        continue;
                    }
                    // Terminal mesh hit
                    if (instance.geometryType == GeometryType::Mesh) {
                        const GPUMaterial &material = scene.materials[instance.materialIndex];
                        const bool isBackfaceHit = dot(primaryRay.direction, worldHit.geometricNormalW) > 0.0f;
                        const float3 normalW = isBackfaceHit ? -worldHit.geometricNormalW : worldHit.geometricNormalW;
                        const float compositeWeight = accumulated.transmittance;
                        accumulated.compositeWeight += compositeWeight;
                        // Treat terminal mesh as opaque for median-depth purposes.
                        if (settings.computeSurfaceDiagnostics) {
                            const float depth = dot(worldHit.hitPositionW - sensor.camera.pos, sensor.camera.forward);
                            accumulated.depthWeight += compositeWeight;
                            accumulated.weightedDepth += compositeWeight * depth;
                            if (!accumulated.medianFound && accumulated.compositeWeight >= settings.
                                medianDepthThreshold) {
                                accumulated.medianFound = true;
                                accumulated.medianDepth = depth;
                                accumulated.medianWorldPosition = worldHit.hitPositionW;
                                accumulated.medianNormal = normalW;
                            }
                        }
                        if (material.isEmissive()) {
                            const float3 emittedRadiance = material.power * material.baseColor;
                            accumulated.radiance += accumulated.transmittance * min(emittedRadiance, 1.0f);
                        } else {
                            const float3 indirectIrradiance =
                                    gatherDiffuseIrradianceAtPoint(worldHit.hitPositionW, normalW, photonMap);
                            const float3 indirectRadiance = (material.baseColor * M_1_PIf) * indirectIrradiance;
                            const float3 directRadiance = estimateDirectPointSampledPointLights(
                                scene, settings, worldHit.hitPositionW, normalW, material.baseColor, RayEpsilon);
                            const float3 outgoingRadiance = indirectRadiance + directRadiance;
                            accumulated.radiance += accumulated.transmittance * outgoingRadiance;
                        }
                        accumulated.transmittance = 0.0f;
                        break;
                    }
                }
            }
            const float alpha = sycl::clamp(accumulated.compositeWeight, 0.0f, 1.0f);
            const float4 currentValue{
                accumulated.radiance.x(), accumulated.radiance.y(), accumulated.radiance.z(), alpha
            };
            sensor.framebuffer[pixelIndex] += currentValue;
            sensor.depthDistortionBuffer[pixelIndex] = accumulated.distortion;
            if (accumulated.depthWeight > 1.0e-6f) {
                sensor.meanDepthBuffer[pixelIndex] = accumulated.weightedDepth / accumulated.depthWeight;
            } else {
                sensor.meanDepthBuffer[pixelIndex] = 0.0f;
            }
            const bool keepLastDepth =
                    settings.medianDepthRetainLastHit || accumulated.pointHitCount >= maxSplatEventsPerRay;
            const bool surfaceDepthValid = accumulated.medianFound || (keepLastDepth && accumulated.medianDepth > 0.0f);
            if (surfaceDepthValid) {
                sensor.medianDepthBuffer[pixelIndex] = accumulated.medianDepth;
                sensor.medianWorldPositionBuffer[pixelIndex] = float4{
                    accumulated.medianWorldPosition.x(), accumulated.medianWorldPosition.y(),
                    accumulated.medianWorldPosition.z(), 1.0f
                };
                sensor.visibleNormalBuffer[pixelIndex] = float4{
                    accumulated.medianNormal.x(), accumulated.medianNormal.y(),
                    accumulated.medianNormal.z(), accumulated.depthWeight
                };
            } else {
                sensor.medianDepthBuffer[pixelIndex] = 0.0f;
                sensor.medianWorldPositionBuffer[pixelIndex] = float4{0.0f};
                sensor.visibleNormalBuffer[pixelIndex] = float4{0.0f};
            }
        }

        void clearLegacyCameraGatherOutputs(sycl::queue &queue, const SensorGPU &sensor, uint32_t pixelCount) {
            queue.fill(sensor.framebuffer, float4{0.0f, 0.0f, 0.0f, 0.0f}, pixelCount);
            queue.fill(sensor.medianDepthBuffer, 0.0f, pixelCount);
            queue.fill(sensor.meanDepthBuffer, 0.0f, pixelCount);
            queue.fill(sensor.medianWorldPositionBuffer, float4{0.0f, 0.0f, 0.0f, 0.0f}, pixelCount);
            queue.fill(sensor.visibleNormalBuffer, float4{0.0f, 0.0f, 0.0f, 0.0f}, pixelCount);
            queue.fill(sensor.normalFromDepthBuffer, float4{0.0f, 0.0f, 0.0f, 0.0f}, pixelCount);
            queue.fill(sensor.depthDistortionBuffer, 0.0f, pixelCount);
            if (sensor.depthDistortionAdjointBuffer) queue.fill(sensor.depthDistortionAdjointBuffer, 0.0f, pixelCount);
            queue.fill(sensor.intraSlabDepthBuffer, 0.0f, pixelCount);
            if (sensor.intraSlabRayDepthBuffer != nullptr) {
                queue.fill(sensor.intraSlabRayDepthBuffer, 0.0f, pixelCount);
            }
            if (sensor.intraSlabDepthAdjointBuffer) queue.fill(sensor.intraSlabDepthAdjointBuffer, 0.0f, pixelCount);
            queue.fill(sensor.intraSlabDepthActiveSlabCountBuffer, 0u, pixelCount);
            queue.wait();
        }
    } // namespace

    struct CameraGatherKernel {
        GPUSceneBuffers scene;
        PathTracerSettings settings;
        DeviceSurfacePhotonMapGrid photonMap;
        SensorGPU sensor;

        void operator()(sycl::id<1> tid) const {
            renderLegacyCameraGatherPixel({scene, settings, photonMap, sensor}, static_cast<uint32_t>(tid[0]));
        }
    };

    void launchCameraGatherKernel(RenderPackage &pkg, uint32_t cameraIndex, uint32_t /*gatherPassIdx*/) {
        auto &queue = pkg.queue;
        const auto &settings = pkg.settings;
        const auto &sensor = pkg.sensors[cameraIndex];
        const uint32_t pixelCount = sensor.camera.width * sensor.camera.height;

        clearLegacyCameraGatherOutputs(queue, sensor, pixelCount);
        queue.parallel_for<CameraGatherKernel>(sycl::range<1>(pixelCount), CameraGatherKernel{
                                                   .scene = pkg.scene, .settings = settings,
                                                   .photonMap = pkg.intermediates.map, .sensor = sensor
                                               }).wait();

        if (settings.normalConsistencyWeight) {
            submitCameraNormalsFromDepth(queue, sensor, settings.normalFromDepthUseMeanDepth).wait();
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
            queue.fill(sensor.intraSlabDepthBuffer, 0.0f, pixelCount);
            if (sensor.intraSlabDepthAdjointBuffer) queue.fill(sensor.intraSlabDepthAdjointBuffer, 0.0f, pixelCount);
            queue.fill(sensor.intraSlabDepthActiveSlabCountBuffer, 0u, pixelCount);
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
                    float accumulatedCompositeWeight = 0.0f;
                    bool medianFound = false;
                    float medianDepth = 0.0f;
                    float3 medianWorldPosition(0.0f, 0.0f, 0.0f);
                    float3 medianNormalW(0.0f, 0.0f, 0.0f);
                    float accumulatedMeanDepthWeight = 0.0f;
                    float accumulatedMeanDepth = 0.0f;
                    uint32_t processedHitCount = 0u;
                    for (uint32_t traversalIndex = 0u; traversalIndex < kMaxSplatEventsPerRay; ++traversalIndex) {
                        PointSampledSceneHit hit{};
                        if (!intersectScenePointSampledGeometry(primaryRay, scene, settings, hit)) {
                            break;
                        }
                        ++processedHitCount;
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
                        if (!medianFound && compositeWeight > 0.0f && depth > 0.0f) {
                            medianDepth = depth;
                            medianWorldPosition = hit.hitPositionW;
                            medianNormalW = normalW;
                            medianFound = accumulatedCompositeWeight + compositeWeight >= settings.medianDepthThreshold;
                        }
                        accumulatedCompositeWeight += compositeWeight;
                        if (depthDistortionDepthAccepted(depth, settings.depthDistortionWorldSpace)) {
                            const float normalizedDepth = depthDistortionCoordinate(depth, settings.depthDistortionWorldSpace);
                            // Traversal visits intersections in increasing camera-forward depth.
                            const float pairSum = settings.depthDistortionWorldSpace && !settings.depthDistortionSquared
                                ? normalizedDepth * prefixWeight - prefixWeightDepth
                                : normalizedDepth * normalizedDepth * prefixWeight +
                                  prefixWeightDepthSquared - 2.0f * normalizedDepth * prefixWeightDepth;
                            distortion += compositeWeight * pairSum;
                            prefixWeight += compositeWeight;
                            prefixWeightDepth += compositeWeight * normalizedDepth;
                            prefixWeightDepthSquared += compositeWeight * normalizedDepth * normalizedDepth;
                        }
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
                    if (accumulatedMeanDepthWeight > 1.0e-6f) {
                        sensor.meanDepthBuffer[pixelIndex] = accumulatedMeanDepth / accumulatedMeanDepthWeight;
                    }
                    else {
                        sensor.meanDepthBuffer[pixelIndex] = 0.0f;
                    }
                    const bool surfaceDepthValid = medianFound ||
                            ((settings.medianDepthRetainLastHit || processedHitCount >= kMaxSplatEventsPerRay) && medianDepth > 0.0f);
                    if (surfaceDepthValid) {
                        sensor.medianDepthBuffer[pixelIndex] = medianDepth;
                        sensor.medianWorldPositionBuffer[pixelIndex] = float4{
                            medianWorldPosition.x(), medianWorldPosition.y(), medianWorldPosition.z(), 1.0f
                        };
                        sensor.visibleNormalBuffer[pixelIndex] = float4{
                            medianNormalW.x(), medianNormalW.y(), medianNormalW.z(), accumulatedMeanDepthWeight
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
    void computePhotonCellIdsAndPermutation(sycl::queue &queue, DeviceSurfacePhotonMapGrid grid,
                                            std::uint32_t photonCount) {
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
            auto atomicCount = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                sycl::access::address_space::global_space>(grid.cellCount[cellId]);
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
            auto atomicOffset = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                sycl::access::address_space::global_space>(grid.cellWriteOffset[cellId]);
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
                    cgh.parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(blockCount * blockSize), sycl::range<1>(blockSize)),
                        [=](sycl::nd_item<1> item) {
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

    void buildPhotonCellRangesAndOrdering(sycl::queue &queue, DeviceSurfacePhotonMapGrid grid,
                                          std::uint32_t photonCount) {
        clearCellArrays(queue, grid); // counts/start/end/offset=0/invalid
        computePhotonCellIdsAndPermutation(queue, grid, photonCount); // keys (optional now)
        countPhotonsPerCell(queue, grid, photonCount); // histogram
        exclusiveScanCellCountsToCellStart(queue, grid); // cellStart from cellCount (implement)
        finalizeCellRanges(queue, grid); // cellEnd = start + count, invalid if count==0, offset=0
        scatterPhotonsIntoCells(queue, grid, photonCount); // sortedPhotonIndex
    }
} // namespace Pale
