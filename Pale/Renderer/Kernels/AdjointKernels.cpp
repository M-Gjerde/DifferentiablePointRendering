//
// Created by magnus on 9/8/25.
//

#include "Renderer/Kernels/AdjointKernels.h"

#include <cmath>

#include "AdjointGradientKernels.h"
#include "IntersectionKernels.h"
#include "Core/ScopedTimer.h"
#include "Renderer/Kernels/KernelHelpers.h"

import Pale.Log;


namespace Pale {
    void launchRayGenAdjointKernel(RenderPackage &pkg, int spp, uint32_t cameraIndex) {
        auto &queue = pkg.queue;
        auto &settings = pkg.settings;
        auto &intermediates = pkg.intermediates;
        auto &sensor = pkg.sensors[cameraIndex];

        const uint32_t imageWidth = sensor.camera.width;
        const uint32_t imageHeight = sensor.camera.height;
        uint32_t raysPerSet = imageWidth * imageHeight;


        queue.submit([&](sycl::handler &commandGroupHandler) {
            const uint64_t renderSeed = settings.random.seed;

            commandGroupHandler.parallel_for<struct RayGenAdjointKernelTag>(
                sycl::range<1>(raysPerSet),
                [=](sycl::id<1> globalId) {
                    const auto globalRayIndex = static_cast<uint32_t>(globalId[0]);
                    // Map to pixel
                    const uint32_t pixelLinearIndexWithinImage = globalRayIndex; // 0..raysPerSet-1
                    uint32_t pixelX = pixelLinearIndexWithinImage % imageWidth;
                    uint32_t pixelY = pixelLinearIndexWithinImage / imageWidth;

                    const uint32_t pixelIndex = pixelLinearIndexWithinImage;
                    // RNG for this pixelhttps://www.chess.com/home
                    const uint64_t seed =
                            rng::makeSeed(renderSeed, globalRayIndex, spp, rng::kStreamRayGen, 0u);
                    rng::Xorshift128 rng128(seed);

                    // Adjoint source weight
                    const float4 dLoss_dI = sensor.framebuffer[pixelIndex];
                    float3 dLoss_dI3 = float3{dLoss_dI.x(), dLoss_dI.y(), dLoss_dI.z()}; // (I - T)
                    float3 initialAdjointWeight = dLoss_dI3;

                    // Base slot for this pixel’s N samples
                    const uint32_t baseOutputSlot = pixelIndex;
                    // --- Sample 0: forced Transmit (background path) ---
                    const float jitterX = rng128.nextFloat() - 0.5f;
                    const float jitterY = rng128.nextFloat() - 0.5f;

                    //if (isWatchedPixel(pixelX, pixelY)) {
                    //    int debug = 1;
                    //} else {
                    //    return;
                    //}


                    Ray primaryRay = makePrimaryRayFromPixelJitteredFov(
                        sensor.camera,
                        static_cast<float>(pixelX),
                        static_cast<float>(pixelY),
                        jitterX, jitterY
                    );

                    //primaryRay.direction = normalize(float3{-0.001, 0.982122211, 0.277827293});    // a
                    //primaryRay.direction = normalize(float3{-0.01, 1.0, 0.04}); // b
                    //primaryRay.origin = float3{0.0, -4.0, 1.0};


                    RayState rayState{};
                    rayState.ray = primaryRay;
                    rayState.pathThroughput = initialAdjointWeight;
                    rayState.bounceIndex = 0;
                    rayState.pixelIndex = pixelIndex;
                    rayState.pathId = pixelLinearIndexWithinImage; // 0 .. (W*H-1)

                    intermediates.primaryRays[baseOutputSlot] = rayState;

                    if (rayState.pathId < intermediates.maxPendingAdjointStateCount) {
                        PendingCameraSegment pendingCameraSegment{};
                        pendingCameraSegment.valid = true;
                        pendingCameraSegment.pathId = rayState.pathId;
                        pendingCameraSegment.pixelIndex = rayState.pixelIndex;
                        pendingCameraSegment.cameraPathThroughput = initialAdjointWeight;
                        pendingCameraSegment.cameraOriginWorld = primaryRay.origin;
                        pendingCameraSegment.cameraDirectionWorld = primaryRay.direction;
                        clearCachedSegmentTransmittance(pendingCameraSegment.segmentOccludersToFirstScatter);

                        intermediates.pendingCameraSegments[rayState.pathId] = pendingCameraSegment;

                        clearPendingAdjointStageX(intermediates.pendingStageX[rayState.pathId]);
                        clearPendingAdjointStageXY(intermediates.pendingStageXY[rayState.pathId]);
                    }
                });
        }).wait();
    }

    void launchAdjointIntersectKernel(RenderPackage &pkg, uint32_t spp, uint32_t activeRayCount) {
        auto &queue = pkg.queue;
        auto &settings = pkg.settings;
        auto &intermediates = pkg.intermediates;
        auto &scene = pkg.scene;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            const uint64_t renderSeed = settings.random.seed;

            commandGroupHandler.parallel_for<class launchIntersectKernel>(
                sycl::range<1>(activeRayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    RayState rayState = intermediates.primaryRays[rayIndex];

                    const uint64_t seed =
                            rng::makeSeed(
                                renderSeed,
                                rayState.pathId,
                                spp,
                                rng::kStreamTraversal,
                                rayState.bounceIndex);

                    rng::Xorshift128 rng(seed);

                    WorldHit worldHit{};
                    intersectScene(rayState.ray, &worldHit, scene, rng, SurfelIntersectMode::FirstHit);
                    if (!worldHit.hit) {
                        return;
                    }

                    buildIntersectionNormal(scene, worldHit);

                    const InstanceRecord &instance = scene.instances[worldHit.instanceIndex];
                    const GeometryType currentGeometryType = instance.geometryType;
                    const bool isPointCloudHit = currentGeometryType == GeometryType::PointCloud;

                    float3 orientedNormal{0.0f, 0.0f, 0.0f};
                    float uniformDirectionPdf = 1.0f / (2.0f * M_PIf);
                    float cosineDirectionPdf = 0.0f;
                    float3 sampledOutgoingDirectionWorld{0.0f, 0.0f, 0.0f};

                    switch (currentGeometryType) {
                        case GeometryType::Mesh: {
                            orientedNormal = worldHit.geometricNormalW;
                            if (dot(rayState.ray.direction, orientedNormal) > 0.0f) {
                                orientedNormal = -orientedNormal;
                            }

                            sampleCosineHemisphere(
                                rng,
                                worldHit.geometricNormalW,
                                sampledOutgoingDirectionWorld,
                                cosineDirectionPdf);
                            break;
                        }

                        case GeometryType::PointCloud: {
                            const Point &surfel = scene.points[worldHit.primitiveIndex];
                            const float3 canonicalNormal = normalize(cross(
                                surfel.scale.x() * surfel.tanU,
                                surfel.scale.y() * surfel.tanV));

                            const float signedCosineIncident = dot(canonicalNormal, -rayState.ray.direction);
                            const int sideSign = signNonZero(signedCosineIncident);
                            orientedNormal = static_cast<float>(sideSign) * canonicalNormal;

                            sampleUniformHemisphereAroundNormal(
                                rng,
                                orientedNormal,
                                sampledOutgoingDirectionWorld,
                                uniformDirectionPdf);
                            break;
                        }

                        default:
                            return;
                    }

                    const uint32_t pathId = rayState.pathId;
                    const bool canUsePendingAdjointState =
                            pathId < intermediates.maxPendingAdjointStateCount;

                    PendingCameraSegment previousPendingCameraSegment{};
                    PendingAdjointStageX previousPendingStageX{};
                    PendingAdjointStageXY previousPendingStageXY{};

                    if (canUsePendingAdjointState) {
                        previousPendingCameraSegment = intermediates.pendingCameraSegments[pathId];
                        previousPendingStageX = intermediates.pendingStageX[pathId];
                        previousPendingStageXY = intermediates.pendingStageXY[pathId];
                    }

                    PointCloudSurfaceRecord currentSurfaceRecord{};
                    if (isPointCloudHit) {
                        currentSurfaceRecord = makePointCloudSurfaceRecord(worldHit, rayState, scene);
                    }

                    RayState nextState{};
                    bool shouldEnqueueNextState = false;

                    SampledPointEventType sampledPointEventType = SampledPointEventType::None;

                    bool currentHitSeedsReflectPrefix = false;
                    float3 reflectPrefixThroughput{0.0f, 0.0f, 0.0f};

                    if (instance.geometryType == GeometryType::Mesh) {
                        const GPUMaterial material = scene.materials[instance.materialIndex];
                        const float3 throughputMultiplier = material.baseColor;

                        nextState.ray.origin = worldHit.hitPositionW + (worldHit.geometricNormalW * 1e-6f);
                        nextState.ray.direction = sampledOutgoingDirectionWorld;
                        nextState.ray.normal = worldHit.geometricNormalW;
                        nextState.bounceIndex = rayState.bounceIndex + 1u;
                        nextState.pixelIndex = rayState.pixelIndex;
                        nextState.pathId = rayState.pathId;
                        nextState.pathThroughput = rayState.pathThroughput * throughputMultiplier;

                        if (applyRussianRoulette(
                            rng,
                            nextState.bounceIndex,
                            nextState.pathThroughput,
                            settings.russianRouletteStart)) {
                            shouldEnqueueNextState = true;
                        }
                    } else {
                        const Point &surfel = scene.points[worldHit.primitiveIndex];
                        const float randomNumber = rng.nextFloat();

                        if (randomNumber < settings.sampling.qNull) {
                            sampledPointEventType = SampledPointEventType::Null;

                            const float attenuation = 1.0f - worldHit.alphaGeom * surfel.opacity;
                            const float weight = attenuation / settings.sampling.qNull;

                            nextState.ray.origin = worldHit.hitPositionW + (rayState.ray.direction * 1e-5f);
                            nextState.ray.direction = rayState.ray.direction;
                            nextState.ray.normal = worldHit.geometricNormalW;
                            nextState.bounceIndex = rayState.bounceIndex + 1u;
                            nextState.pixelIndex = rayState.pixelIndex;
                            nextState.pathId = rayState.pathId;
                            nextState.pathThroughput = rayState.pathThroughput * weight;

                            if (applyRussianRoulette(
                                rng,
                                nextState.bounceIndex,
                                nextState.pathThroughput,
                                settings.russianRouletteStart)) {
                                shouldEnqueueNextState = true;
                            }
                            reflectPrefixThroughput =
                                    rayState.pathThroughput / settings.sampling.qTransmit;
                        } else if (randomNumber < settings.sampling.qNull + settings.sampling.qReflect) {
                            sampledPointEventType = SampledPointEventType::Reflect;

                            const float3 surfelBrdf = surfel.alpha_r * surfel.albedo * M_1_PIf;
                            const float cosineTheta = sycl::fmax(
                                0.0f,
                                dot(sampledOutgoingDirectionWorld, orientedNormal));
                            const float alpha = worldHit.alphaGeom * surfel.opacity;

                            const float3 throughputMultiplier =
                                    ((alpha / settings.sampling.qReflect) * (surfelBrdf * cosineTheta)) /
                                    uniformDirectionPdf;

                            nextState.ray.origin = worldHit.hitPositionW + (orientedNormal * 1e-5f);
                            nextState.ray.direction = sampledOutgoingDirectionWorld;
                            nextState.ray.normal = orientedNormal;
                            nextState.bounceIndex = rayState.bounceIndex + 1u;
                            nextState.pixelIndex = rayState.pixelIndex;
                            nextState.pathId = rayState.pathId;
                            nextState.pathThroughput = rayState.pathThroughput * throughputMultiplier;

                            if (applyRussianRoulette(
                                rng,
                                nextState.bounceIndex,
                                nextState.pathThroughput,
                                settings.russianRouletteStart)) {
                                shouldEnqueueNextState = true;
                            }

                            currentHitSeedsReflectPrefix = true;
                            reflectPrefixThroughput =
                                    rayState.pathThroughput / settings.sampling.qReflect;
                        } else if (
                            randomNumber <
                            settings.sampling.qNull +
                            settings.sampling.qReflect +
                            settings.sampling.qTransmit) {
                            sampledPointEventType = SampledPointEventType::Transmit;

                            const float alpha = worldHit.alphaGeom * surfel.opacity;
                            const float weight =
                                    (alpha * surfel.alpha_t) / settings.sampling.qTransmit;
                            const float3 throughput = rayState.pathThroughput * weight;

                            const float3 canonicalNormal = normalize(cross(
                                surfel.scale.x() * surfel.tanU,
                                surfel.scale.y() * surfel.tanV));
                            const float signedCosineIncident =
                                    dot(canonicalNormal, -rayState.ray.direction);
                            const int sideSign = signNonZero(signedCosineIncident);
                            const float3 transmitOrientedNormal =
                                    static_cast<float>(sideSign) * canonicalNormal;

                            float3 transmitDirection = rayState.ray.direction;
                            float transmitPdf = 0.0f;
                            sampleUniformHemisphereAroundNormal(
                                rng,
                                transmitOrientedNormal,
                                transmitDirection,
                                transmitPdf);

                            nextState.ray.origin =
                                    worldHit.hitPositionW + (-transmitOrientedNormal * 1e-5f);
                            nextState.ray.direction = transmitDirection;
                            nextState.ray.normal = -transmitOrientedNormal;
                            nextState.bounceIndex = rayState.bounceIndex + 1u;
                            nextState.pixelIndex = rayState.pixelIndex;
                            nextState.pathId = rayState.pathId;
                            nextState.pathThroughput = throughput * surfel.albedo;

                            if (applyRussianRoulette(
                                rng,
                                nextState.bounceIndex,
                                nextState.pathThroughput,
                                settings.russianRouletteStart)) {
                                shouldEnqueueNextState = true;
                            }
                        }
                    }

                    if (canUsePendingAdjointState) {
                        PendingCameraSegment nextPendingCameraSegment{};
                        PendingAdjointStageX nextPendingStageX{};
                        PendingAdjointStageXY nextPendingStageXY{};

                        if (isPointCloudHit) {
                            switch (sampledPointEventType) {
                                case SampledPointEventType::Null: {
                                    if (previousPendingCameraSegment.valid) {
                                        nextPendingCameraSegment = previousPendingCameraSegment;

                                        const float distanceFromCamera =
                                                length(worldHit.hitPositionW -
                                                       previousPendingCameraSegment.cameraOriginWorld);

                                        appendOccluderToCachedSegment(
                                            nextPendingCameraSegment.segmentOccludersToFirstScatter,
                                            currentSurfaceRecord.primitiveIndex,
                                            currentSurfaceRecord.uv,
                                            currentSurfaceRecord.alphaGeom,
                                            distanceFromCamera);
                                    }

                                    if (previousPendingStageX.valid) {
                                        nextPendingStageX = previousPendingStageX;

                                        const float distanceFromSegmentStart =
                                                length(worldHit.hitPositionW -
                                                       previousPendingStageX.segmentStartPositionWorld);

                                        appendOccluderToCachedSegment(
                                            nextPendingStageX.segmentOccludersToNextHit,
                                            currentSurfaceRecord.primitiveIndex,
                                            currentSurfaceRecord.uv,
                                            currentSurfaceRecord.alphaGeom,
                                            distanceFromSegmentStart);
                                    }

                                    if (previousPendingStageXY.valid) {
                                        nextPendingStageXY = previousPendingStageXY;

                                        const float distanceFromSegmentStart =
                                                length(worldHit.hitPositionW -
                                                       previousPendingStageXY.segmentStartPositionWorld);

                                        appendOccluderToCachedSegment(
                                            nextPendingStageXY.segmentOccludersToNextHit,
                                            currentSurfaceRecord.primitiveIndex,
                                            currentSurfaceRecord.uv,
                                            currentSurfaceRecord.alphaGeom,
                                            distanceFromSegmentStart);
                                    }
                                    break;
                                }

                                case SampledPointEventType::Reflect: {
                                    const bool isFirstReflectiveRootHit =
                                            previousPendingCameraSegment.valid &&
                                            sampledPointEventType == SampledPointEventType::Reflect;

                                    if (isFirstReflectiveRootHit) {
                                        AttachedGradientProjectionEvent attachedProjectionEvent{};
                                        attachedProjectionEvent.xSurface = currentSurfaceRecord;
                                        attachedProjectionEvent.xPathThroughput =
                                                rayState.pathThroughput / settings.sampling.qReflect;

                                        appendEventAtomic(
                                            intermediates.countProjectionEvents,
                                            intermediates.projectionEvents,
                                            intermediates.maxProjectionEventCount,
                                            attachedProjectionEvent);
                                    }

                                    const bool hasCameraToSurfaceOcclusion =
                                            previousPendingCameraSegment.valid &&
                                            (previousPendingCameraSegment.segmentOccludersToFirstScatter.occluderCount >
                                             0u ||
                                             previousPendingCameraSegment.segmentOccludersToFirstScatter.overflowed);

                                    if (hasCameraToSurfaceOcclusion) {
                                        CameraToSurfaceScatterEvent cameraToSurfaceEvent{};
                                        cameraToSurfaceEvent.ySurface = currentSurfaceRecord;
                                        cameraToSurfaceEvent.cameraPathThroughput =
                                                previousPendingCameraSegment.cameraPathThroughput;
                                        cameraToSurfaceEvent.cameraOriginWorld =
                                                previousPendingCameraSegment.cameraOriginWorld;
                                        cameraToSurfaceEvent.cameraDirectionWorld =
                                                previousPendingCameraSegment.cameraDirectionWorld;
                                        cameraToSurfaceEvent.segmentCameraToY =
                                                previousPendingCameraSegment.segmentOccludersToFirstScatter;

                                        appendEventAtomic(
                                            intermediates.countCameraToSurfaceScatterEvents,
                                            intermediates.cameraToSurfaceScatterEvents,
                                            intermediates.maxCameraToSurfaceScatterEventCount,
                                            cameraToSurfaceEvent);
                                    }

                                    if (previousPendingStageX.valid) {
                                        AttachedGradientScatterEvent attachedScatterEvent{};
                                        attachedScatterEvent.xSurface = previousPendingStageX.xSurface;
                                        attachedScatterEvent.ySurface = currentSurfaceRecord;
                                        attachedScatterEvent.xPathThroughput =
                                                previousPendingStageX.xPathThroughput;
                                        attachedScatterEvent.applyIncomingRayHitJacobianToX =
                                                previousPendingStageX.applyIncomingRayHitJacobianToX;
                                        attachedScatterEvent.segmentXY =
                                                previousPendingStageX.segmentOccludersToNextHit;

                                        appendEventAtomic(
                                            intermediates.countProjectionScatterEvents,
                                            intermediates.projectionScatterEvents,
                                            intermediates.maxProjectionScatterEventCount,
                                            attachedScatterEvent);
                                    }

                                    if (previousPendingStageXY.valid) {
                                        DetachedThreePointGradientEvent detachedThreePointEvent{};
                                        detachedThreePointEvent.xSurface =
                                                previousPendingStageXY.xSurface;
                                        detachedThreePointEvent.ySurface =
                                                previousPendingStageXY.ySurface;
                                        detachedThreePointEvent.zSurface = currentSurfaceRecord;
                                        detachedThreePointEvent.xPathThroughput =
                                                previousPendingStageXY.xPathThroughput;
                                        detachedThreePointEvent.segmentYZ =
                                                previousPendingStageXY.segmentOccludersToNextHit;

                                        appendEventAtomic(
                                            intermediates.countReflectScatterEvents,
                                            intermediates.reflectScatterEvents,
                                            intermediates.maxReflectScatterEventCount,
                                            detachedThreePointEvent);
                                    }

                                    if (currentHitSeedsReflectPrefix) {
                                        nextPendingStageX.valid = true;
                                        nextPendingStageX.pathId = rayState.pathId;
                                        nextPendingStageX.pixelIndex = rayState.pixelIndex;
                                        nextPendingStageX.xSurface = currentSurfaceRecord;
                                        nextPendingStageX.xPathThroughput = reflectPrefixThroughput;
                                        nextPendingStageX.applyIncomingRayHitJacobianToX =
                                                (rayState.bounceIndex == 0u);
                                        nextPendingStageX.segmentStartPositionWorld =
                                                worldHit.hitPositionW;
                                        clearCachedSegmentTransmittance(
                                            nextPendingStageX.segmentOccludersToNextHit);

                                        if (previousPendingStageX.valid) {
                                            nextPendingStageXY.valid = true;
                                            nextPendingStageXY.pathId = rayState.pathId;
                                            nextPendingStageXY.pixelIndex = rayState.pixelIndex;
                                            nextPendingStageXY.xSurface =
                                                    previousPendingStageX.xSurface;
                                            nextPendingStageXY.ySurface = currentSurfaceRecord;
                                            nextPendingStageXY.xPathThroughput =
                                                    previousPendingStageX.xPathThroughput;
                                            nextPendingStageXY.applyIncomingRayHitJacobianToX =
                                                    previousPendingStageX.applyIncomingRayHitJacobianToX;
                                            nextPendingStageXY.segmentStartPositionWorld =
                                                    worldHit.hitPositionW;
                                            clearCachedSegmentTransmittance(
                                                nextPendingStageXY.segmentOccludersToNextHit);
                                        }
                                    }

                                    clearPendingCameraSegment(nextPendingCameraSegment);
                                    break;
                                }

                                case SampledPointEventType::Transmit: {
                                    clearPendingCameraSegment(nextPendingCameraSegment);
                                    break;
                                }

                                case SampledPointEventType::None:
                                default: {
                                    break;
                                }
                            }
                        }

                        if (nextPendingCameraSegment.valid) {
                            intermediates.pendingCameraSegments[pathId] = nextPendingCameraSegment;
                        } else {
                            clearPendingCameraSegment(intermediates.pendingCameraSegments[pathId]);
                        }

                        if (nextPendingStageX.valid) {
                            intermediates.pendingStageX[pathId] = nextPendingStageX;
                        } else {
                            clearPendingAdjointStageX(intermediates.pendingStageX[pathId]);
                        }

                        if (nextPendingStageXY.valid) {
                            intermediates.pendingStageXY[pathId] = nextPendingStageXY;
                        } else {
                            clearPendingAdjointStageXY(intermediates.pendingStageXY[pathId]);
                        }
                    }

                    if (shouldEnqueueNextState) {
                        auto extensionCounter = sycl::atomic_ref<
                            uint32_t,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                            *intermediates.countExtensionOut);

                        const uint32_t outIndex = extensionCounter.fetch_add(1u);
                        intermediates.extensionRaysA[outIndex] = nextState;
                    }
                });
        }).wait();
    }

    static void launchAttachedProjectionKernel(
        RenderPackage &pkg,
        uint32_t projectionEventCount,
        uint32_t baseOffset) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        const auto &photonMap = pkg.intermediates.map;
        AttachedGradientProjectionEvent *projectionEvents = pkg.intermediates.projectionEvents;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;

        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class launchProjectionContributionKernelTag>(
                sycl::range<1>(projectionEventCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex = globalId[0];
                    const uint32_t recordIndex = baseOffset + eventIndex;
                    const AttachedGradientProjectionEvent eventRecord = projectionEvents[eventIndex];

                    const Point &surfelX = scene.points[eventRecord.xSurface.primitiveIndex];
                    const ReconstructedSurfelState xState =
                            reconstructSurfelState(surfelX, eventRecord.xSurface);

                    const float3 irradiance = gatherDiffuseIrradianceAtPoint(
                        xState.position,
                        xState.orientedNormal,
                        photonMap);

                    const float3 outgoingRadiance =
                            surfelX.alpha_r * surfelX.albedo * M_1_PIf * irradiance;

                    const float gradAlphaEta = eventRecord.xSurface.alphaGeom;
                    const float3 pathWeight = eventRecord.xPathThroughput;

                    const float3 opacityGradientContribution =
                            gradAlphaEta * pathWeight * outgoingRadiance;

                    const float opacityGradientScalar =
                            sum(opacityGradientContribution) * invSpp;

                    const float u = eventRecord.xSurface.uv.x();
                    const float v = eventRecord.xSurface.uv.y();
                    const float radiusSquared = u * u + v * v;

                    const float scaleU = surfelX.scale.x();
                    const float scaleV = surfelX.scale.y();

                    const auto uvPositionJacobian = computeDuvDSurfelTranslationJacobian(
                        surfelX.tanU,
                        surfelX.tanV,
                        xState.orientedNormal,
                        eventRecord.xSurface.incomingDirection,
                        u,
                        v,
                        scaleU,
                        scaleV);

                    const float3 dUvDPosition =
                            u * uvPositionJacobian.du_d_surfel_translation +
                            v * uvPositionJacobian.dv_d_surfel_translation;

                    const float betaScale = 4.0f * sycl::exp(surfelX.beta);
                    const float factor =
                            (-2.0f * betaScale * eventRecord.xSurface.alphaGeom) / (1.0f - radiusSquared);

                    const float3 dAlphaGeomDPosition = factor * dUvDPosition;
                    const float3 dAlphaEffDPosition = surfelX.opacity * dAlphaGeomDPosition;

                    const float3 positionGradient =
                    (pathWeight[0] * dAlphaEffDPosition * outgoingRadiance[0] +
                     pathWeight[1] * dAlphaEffDPosition * outgoingRadiance[1] +
                     pathWeight[2] * dAlphaEffDPosition * outgoingRadiance[2]) * invSpp;


                    SurfelGradientRecord gradientRecord = {};
                    gradientRecord.primitiveIndex = eventRecord.xSurface.primitiveIndex;
                    gradientRecord.gradEta = opacityGradientScalar;
                    gradientRecord.gradPositionX = positionGradient.x();
                    gradientRecord.gradPositionY = positionGradient.y();
                    gradientRecord.gradPositionZ = positionGradient.z();

                    gradientRecords[recordIndex] = gradientRecord;
                });
        }).wait();
    }


    static void launchCameraToSurfaceScatterKernel(
        RenderPackage &pkg,
        uint32_t cameraToSurfaceScatterEventCount,
        uint32_t baseOffset) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        const auto &photonMap = pkg.intermediates.map;

        CameraToSurfaceScatterEvent *cameraToSurfaceScatterEvents =
                pkg.intermediates.cameraToSurfaceScatterEvents;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;

        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;
        const float qNull = settings.sampling.qNull;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class launchCameraToSurfaceScatterContributionKernelTag>(
                sycl::range<1>(cameraToSurfaceScatterEventCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex = globalId[0];
                    const uint32_t recordIndex = baseOffset + eventIndex;

                    const CameraToSurfaceScatterEvent eventRecord =
                            cameraToSurfaceScatterEvents[eventIndex];

                    const Point &surfelY = scene.points[eventRecord.ySurface.primitiveIndex];
                    const ReconstructedSurfelState yState =
                            reconstructSurfelState(surfelY, eventRecord.ySurface);

                    const float alphaY =
                            eventRecord.ySurface.alphaGeom * surfelY.opacity;

                    const float3 outgoingRadianceY =
                            evaluateOutgoingRadianceFromSurfel(
                                surfelY,
                                eventRecord.ySurface,
                                yState,
                                photonMap);

                    // ------------------------------------------------------------------
                    // We differentiate the weighted cached null-chain transmittance
                    //
                    //   weightedSegmentTransmittance =
                    //       Π_i ((1 - alpha_i) / qNull)
                    //
                    // with respect to the segment end position y.
                    //
                    // Then:
                    //
                    //   d/dy weightedSegmentTransmittance
                    //     = weightedSegmentTransmittance *
                    //       Σ_i ( - d alpha_i / dy / (1 - alpha_i) )
                    //
                    // where alpha_i = opacity_i * alphaGeom_i(u_i(y), v_i(y)).
                    // ------------------------------------------------------------------

                    const auto &cachedSegment = eventRecord.segmentCameraToY;

                    if (cachedSegment.overflowed) {
                        return;
                    }

                    if (qNull <= 0.0f) {
                        return;
                    }

                    const float3 cameraOriginWorld = eventRecord.cameraOriginWorld;
                    const float3 segmentEndPositionWorld = yState.position;
                    const float3 segmentDirectionWorld =
                            segmentEndPositionWorld - cameraOriginWorld;

                    const float segmentLengthSquared =
                            dot(segmentDirectionWorld, segmentDirectionWorld);
                    if (segmentLengthSquared <= 1e-20f) {
                        return;
                    }

                    const float segmentLength = sycl::sqrt(segmentLengthSquared);

                    float weightedSegmentTransmittance = 1.0f;
                    float3 gradientLogWeightedSegmentTransmittanceWrtEnd =
                            float3{0.0f, 0.0f, 0.0f};

                    for (uint32_t occluderIndex = 0u;
                         occluderIndex < cachedSegment.occluderCount;
                         ++occluderIndex) {
                        const auto &cachedOccluder = cachedSegment.occluders[occluderIndex];
                        const Point &occluderSurfel =
                                scene.points[cachedOccluder.primitiveIndex];

                        const float u = cachedOccluder.uv.x();
                        const float v = cachedOccluder.uv.y();

                        const float alphaGeomOccluder = cachedOccluder.alphaGeom;
                        const float alphaEffectiveOccluder =
                                alphaGeomOccluder * occluderSurfel.opacity;

                        const float oneMinusAlpha = 1.0f - alphaEffectiveOccluder;
                        if (oneMinusAlpha <= 1e-8f) {
                            return;
                        }

                        // Weighted null-chain factor for this cached occluder.
                        const float weightedNullFactor = oneMinusAlpha / qNull;
                        weightedSegmentTransmittance *= weightedNullFactor;

                        const float scaleU = occluderSurfel.scale.x();
                        const float scaleV = occluderSurfel.scale.y();
                        if (sycl::fabs(scaleU) <= 1e-12f || sycl::fabs(scaleV) <= 1e-12f) {
                            continue;
                        }

                        // Canonical plane normal. Its sign does not matter here because
                        // it appears both in numerator and denominator.
                        const float3 occluderPlaneNormal = normalize(cross(
                            scaleU * occluderSurfel.tanU,
                            scaleV * occluderSurfel.tanV));

                        const float normalDotSegmentDirection =
                                dot(occluderPlaneNormal, segmentDirectionWorld);
                        if (sycl::fabs(normalDotSegmentDirection) <= 1e-12f) {
                            continue;
                        }

                        // Cached segment stores the hit distance from the segment start.
                        // This gives the current lambda on the camera->y segment:
                        //
                        //   s_i(y) = cameraOrigin + lambda_i * (y - cameraOrigin)
                        //
                        const float lambda =
                                cachedOccluder.distanceFromSegmentStart / segmentLength;

                        // du_i / dy and dv_i / dy for a fixed startpoint and moving endpoint.
                        //
                        // ds_i/dy = lambda * [ I - d n^T / (n·d) ]
                        //
                        // u_i = ((s_i - s_p^i) · t_u^i) / s_u^i
                        // v_i = ((s_i - s_p^i) · t_v^i) / s_v^i
                        //
                        // Therefore:
                        //
                        // du_i/dy = lambda * [ t_u/s_u - n * (d·t_u)/(s_u * (n·d)) ]
                        // dv_i/dy = lambda * [ t_v/s_v - n * (d·t_v)/(s_v * (n·d)) ]
                        //
                        const float dDotTanU =
                                dot(segmentDirectionWorld, occluderSurfel.tanU);
                        const float dDotTanV =
                                dot(segmentDirectionWorld, occluderSurfel.tanV);

                        const float3 duWrtEndPosition =
                                lambda *
                                (occluderSurfel.tanU / scaleU -
                                 occluderPlaneNormal *
                                 (dDotTanU / (scaleU * normalDotSegmentDirection)));

                        const float3 dvWrtEndPosition =
                                lambda *
                                (occluderSurfel.tanV / scaleV -
                                 occluderPlaneNormal *
                                 (dDotTanV / (scaleV * normalDotSegmentDirection)));

                        // Beta-kernel derivative.
                        //
                        // alphaGeom(u,v) = (1 - r^2)^(b),   b = 4 exp(beta)
                        //
                        // d alphaGeom / du = -2 b u alphaGeom / (1 - r^2)
                        // d alphaGeom / dv = -2 b v alphaGeom / (1 - r^2)
                        //
                        const float radiusSquared = u * u + v * v;
                        const float oneMinusRadiusSquared = 1.0f - radiusSquared;
                        if (oneMinusRadiusSquared <= 1e-8f) {
                            continue;
                        }

                        const float betaScale = 4.0f * sycl::exp(occluderSurfel.beta);

                        const float dAlphaGeomDu =
                                (-2.0f * betaScale * u * alphaGeomOccluder) /
                                oneMinusRadiusSquared;
                        const float dAlphaGeomDv =
                                (-2.0f * betaScale * v * alphaGeomOccluder) /
                                oneMinusRadiusSquared;

                        const float3 dAlphaGeomWrtEndPosition =
                                dAlphaGeomDu * duWrtEndPosition +
                                dAlphaGeomDv * dvWrtEndPosition;

                        const float3 dAlphaEffectiveWrtEndPosition =
                                occluderSurfel.opacity * dAlphaGeomWrtEndPosition;

                        gradientLogWeightedSegmentTransmittanceWrtEnd -=
                                dAlphaEffectiveWrtEndPosition / oneMinusAlpha;
                    }

                    const float3 gradientWeightedSegmentTransmittanceWrtEndPosition =
                            weightedSegmentTransmittance *
                            gradientLogWeightedSegmentTransmittanceWrtEnd;

                    // This kernel only contributes the segment-transmittance derivative:
                    //
                    // d/dy [ weightedSegmentTransmittance(camera, y) *
                    //        alphaY * outgoingRadianceY ]
                    //
                    // keeping alphaY * outgoingRadianceY outside this derivative here.
                    const float3 transportWithoutSegmentDerivative =
                            alphaY * outgoingRadianceY;

                    const float3 pathWeight = eventRecord.cameraPathThroughput;

                    const float scalarTransportWeight =
                            pathWeight[0] * transportWithoutSegmentDerivative[0] +
                            pathWeight[1] * transportWithoutSegmentDerivative[1] +
                            pathWeight[2] * transportWithoutSegmentDerivative[2];

                    float3 gradientWrtHitPositionY =
                            scalarTransportWeight *
                            gradientWeightedSegmentTransmittanceWrtEndPosition;

                    // Chain rule through the primary camera-ray hitpoint:
                    //
                    //   dy / d s_p^y
                    //
                    const float3x3 hitPointJacobian = planeHitPointIntersectionJacobian(
                        eventRecord.cameraDirectionWorld,
                        yState.orientedNormal);

                    gradientWrtHitPositionY =
                            transpose(hitPointJacobian) * gradientWrtHitPositionY;

                    const float3 yContribution = gradientWrtHitPositionY * invSpp;

                    SurfelGradientRecord gradientRecord{};
                    gradientRecord.primitiveIndex = eventRecord.ySurface.primitiveIndex;
                    gradientRecord.gradPositionX = yContribution.x();
                    gradientRecord.gradPositionY = yContribution.y();
                    gradientRecord.gradPositionZ = yContribution.z();

                    gradientRecords[recordIndex] = gradientRecord;
                });
        }).wait();
    }

    static void launchAttachedScatterKernel(
        RenderPackage &pkg,
        uint32_t projectionScatterEventCount,
        uint32_t baseOffset) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        const auto &photonMap = pkg.intermediates.map;
        AttachedGradientScatterEvent *projectionScatterEvents =
                pkg.intermediates.projectionScatterEvents;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;

        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class launchProjectionScatterContributionKernelTag>(
                sycl::range<1>(projectionScatterEventCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex = globalId[0];
                    const uint32_t xRecordIndex = baseOffset + 2u * eventIndex + 0u;
                    const uint32_t yRecordIndex = baseOffset + 2u * eventIndex + 1u;

                    SurfelGradientRecord xRecordInit{};
                    xRecordInit.primitiveIndex = kInvalidIndex;
                    gradientRecords[xRecordIndex] = xRecordInit;

                    SurfelGradientRecord yRecordInit{};
                    yRecordInit.primitiveIndex = kInvalidIndex;
                    gradientRecords[yRecordIndex] = yRecordInit;

                    const AttachedGradientScatterEvent eventRecord =
                            projectionScatterEvents[eventIndex];

                    const uint32_t xPrimitiveIndex = eventRecord.xSurface.primitiveIndex;
                    const uint32_t yPrimitiveIndex = eventRecord.ySurface.primitiveIndex;

                    const Point &surfelX = scene.points[xPrimitiveIndex];
                    const Point &surfelY = scene.points[yPrimitiveIndex];

                    const ReconstructedSurfelState xState =
                            reconstructSurfelState(surfelX, eventRecord.xSurface);
                    const ReconstructedSurfelState yState =
                            reconstructSurfelState(surfelY, eventRecord.ySurface);

                    const float3 outgoingRadianceY =
                            evaluateOutgoingRadianceFromSurfel(
                                surfelY,
                                eventRecord.ySurface,
                                yState,
                                photonMap);


                    const float3 vectorXToY = yState.position - xState.position;
                    const float distanceSquared = dot(vectorXToY, vectorXToY);
                    if (distanceSquared <= 1e-12f) {
                        return;
                    }

                    const float distance = sycl::sqrt(distanceSquared);
                    const float3 directionXToY = vectorXToY / distance;
                    const float cosineAtY = dot(yState.orientedNormal, -directionXToY);
                    if (cosineAtY <= 1e-6f) {
                        return;
                    }

                    const float uniformHemispherePdf = 1.0f / (2.0f * M_PIf);
                    const float pAreaY = uniformHemispherePdf * cosineAtY / distanceSquared;
                    if (pAreaY <= 1e-20f) {
                        return;
                    }

                    const float alphaX = eventRecord.xSurface.alphaGeom * surfelX.opacity;
                    const float3 brdfX = surfelX.alpha_r * surfelX.albedo * M_1_PIf;

                    const float3 dGeometricTermDX = computeGeometricTermGradientWrtStartpoint(
                        xState.position,
                        yState.position,
                        xState.orientedNormal,
                        yState.orientedNormal);

                    float transmittance = 1.0f;
                    const auto &cachedSegment = eventRecord.segmentXY;
                    for (uint32_t occluderIndex = 0u;
                         occluderIndex < cachedSegment.occluderCount;
                         ++occluderIndex) {
                        const auto &cachedOccluder = cachedSegment.occluders[occluderIndex];
                        const Point &occluderSurfel =
                                scene.points[cachedOccluder.primitiveIndex];


                        const float u = cachedOccluder.uv.x();
                        const float v = cachedOccluder.uv.y();
                        float3 s_pi_attached = phiMapping(occluderSurfel, u, v);

                        const float alphaGeomOccluder = cachedOccluder.alphaGeom;
                        const float alphaEffectiveOccluder =
                                alphaGeomOccluder * occluderSurfel.opacity;

                        const float oneMinusAlpha = 1.0f - alphaEffectiveOccluder;
                        transmittance *= oneMinusAlpha;

                        if (oneMinusAlpha <= 1e-8f) {
                            return;
                        }
                        float3 n_i = cross(occluderSurfel.tanU, occluderSurfel.tanV);
                        // negative or positive normal, coulb be flipped but shouldn't matter.

                        float3 s_pi = occluderSurfel.position;
                        float3 y = phiMapping(surfelY, eventRecord.ySurface.uv.x(), eventRecord.ySurface.uv.y());
                        float3 x = phiMapping(surfelX, eventRecord.xSurface.uv.x(), eventRecord.xSurface.uv.y());
                        float lambda_i = dot(n_i, s_pi - y) / dot(n_i, x - y);

                        float3 s_i = y + lambda_i * (x - y);
                        int debug = 1;

                    }


                    const float3 transportWithoutGeometricTerm =
                            outgoingRadianceY * alphaX * brdfX * transmittance;

                    const float3 pathWeight = eventRecord.xPathThroughput;
                    const float scalarWeight =
                    (pathWeight[0] * transportWithoutGeometricTerm[0] +
                     pathWeight[1] * transportWithoutGeometricTerm[1] +
                     pathWeight[2] * transportWithoutGeometricTerm[2]) / (pAreaY);

                    float3 gradientWrtHitPositionX = scalarWeight * dGeometricTermDX; {
                        const float3x3 hitPointJacobian = planeHitPointIntersectionJacobian(
                            eventRecord.xSurface.incomingDirection,
                            xState.orientedNormal);
                        gradientWrtHitPositionX =
                                transpose(hitPointJacobian) * gradientWrtHitPositionX;
                    }

                    const float3 dGeometricTermDY = computeGeometricTermGradientWrtEndpoint(
                        xState.position,
                        yState.position,
                        xState.orientedNormal,
                        yState.orientedNormal);

                    const float3 gradientWrtHitPositionY =
                            scalarWeight * dGeometricTermDY;

                    const float3 xContribution = gradientWrtHitPositionX * invSpp / settings.sampling.qReflect;
                    const float3 yContribution = gradientWrtHitPositionY * invSpp / settings.sampling.qReflect;


                    SurfelGradientRecord xRecord{};
                    xRecord.primitiveIndex = xPrimitiveIndex;
                    xRecord.gradPositionX = xContribution.x();
                    xRecord.gradPositionY = xContribution.y();
                    xRecord.gradPositionZ = xContribution.z();
                    gradientRecords[xRecordIndex] = xRecord;

                    SurfelGradientRecord yRecord{};
                    yRecord.primitiveIndex = yPrimitiveIndex;
                    yRecord.gradPositionX = yContribution.x();
                    yRecord.gradPositionY = yContribution.y();
                    yRecord.gradPositionZ = yContribution.z();
                    gradientRecords[yRecordIndex] = yRecord;
                });
        }).wait();
    }

    static void launchDetachedGradientKernel(
        RenderPackage &pkg,
        uint32_t reflectScatterEventCount,
        uint32_t baseOffset) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        const auto &photonMap = pkg.intermediates.map;
        DetachedThreePointGradientEvent *reflectScatterEvents =
                pkg.intermediates.reflectScatterEvents;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;

        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;
        const float uniformHemispherePdf = 1.0f / (2.0f * M_PIf);

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class launchReflectScatterContributionKernelTag>(
                sycl::range<1>(reflectScatterEventCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex = globalId[0];
                    const uint32_t recordIndex = baseOffset + eventIndex;
                    SurfelGradientRecord gradientRecordInit{};
                    gradientRecordInit.primitiveIndex = kInvalidIndex;
                    gradientRecords[recordIndex] = gradientRecordInit;

                    const DetachedThreePointGradientEvent eventRecord =
                            reflectScatterEvents[eventIndex];

                    const Point &surfelX = scene.points[eventRecord.xSurface.primitiveIndex];
                    const Point &surfelY = scene.points[eventRecord.ySurface.primitiveIndex];
                    const Point &surfelZ = scene.points[eventRecord.zSurface.primitiveIndex];

                    const ReconstructedSurfelState xState =
                            reconstructSurfelState(surfelX, eventRecord.xSurface);
                    const ReconstructedSurfelState yState =
                            reconstructSurfelState(surfelY, eventRecord.ySurface);
                    const ReconstructedSurfelState zState =
                            reconstructSurfelState(surfelZ, eventRecord.zSurface);

                    const float3 outgoingRadianceZ =
                            evaluateOutgoingRadianceFromSurfel(
                                surfelZ,
                                eventRecord.zSurface,
                                zState,
                                photonMap);

                    const float3 vectorYToZ = zState.position - yState.position;
                    const float distanceSquaredYZ = dot(vectorYToZ, vectorYToZ);
                    if (distanceSquaredYZ <= 1e-12f) {
                        return;
                    }

                    const float distanceYZ = sycl::sqrt(distanceSquaredYZ);
                    const float3 directionYToZ = vectorYToZ / distanceYZ;
                    const float cosineAtZ = dot(zState.orientedNormal, -directionYToZ);
                    if (cosineAtZ <= 1e-6f) {
                        return;
                    }

                    const float pAreaZ = uniformHemispherePdf * cosineAtZ / distanceSquaredYZ;
                    if (pAreaZ <= 1e-20f) {
                        return;
                    }

                    const float3 vectorXToY = yState.position - xState.position;
                    const float distanceSquaredXY = dot(vectorXToY, vectorXToY);
                    if (distanceSquaredXY <= 1e-12f) {
                        return;
                    }

                    const float distanceXY = sycl::sqrt(distanceSquaredXY);
                    const float3 directionXToY = vectorXToY / distanceXY;
                    const float cosineAtY = dot(yState.orientedNormal, -directionXToY);
                    if (cosineAtY <= 1e-6f) {
                        return;
                    }

                    const float pAreaY = uniformHemispherePdf * cosineAtY / distanceSquaredXY;
                    if (pAreaY <= 1e-20f) {
                        return;
                    }

                    const float geometricTermXY = computeGeometricTermValue(
                        xState.position,
                        yState.position,
                        xState.orientedNormal,
                        yState.orientedNormal);

                    const float alphaX = eventRecord.xSurface.alphaGeom * surfelX.opacity;
                    const float alphaY = eventRecord.ySurface.alphaGeom * surfelY.opacity;

                    const float3 brdfX = surfelX.alpha_r * surfelX.albedo * M_1_PIf;
                    const float3 brdfY = surfelY.alpha_r * surfelY.albedo * M_1_PIf;

                    const float3 upstreamTransportWithoutGeometricTerm =
                            outgoingRadianceZ * alphaY * brdfY;

                    const float3 transportXY =
                            alphaX * brdfX * geometricTermXY;

                    const float3 combinedTransport =
                            transportXY * upstreamTransportWithoutGeometricTerm;

                    const float3 dGeometricTermDY = computeGeometricTermGradientWrtStartpoint(
                        yState.position,
                        zState.position,
                        yState.orientedNormal,
                        zState.orientedNormal);

                    const float3 pathWeight = eventRecord.xPathThroughput / (2.0f * settings.sampling.qReflect);

                    const float scalarWeightWithoutAreaZ =
                            (pathWeight[0] * combinedTransport[0] +
                             pathWeight[1] * combinedTransport[1] +
                             pathWeight[2] * combinedTransport[2]) /
                            (pAreaY);

                    float3 gradientWrtYPosition =
                            (scalarWeightWithoutAreaZ / pAreaZ) * dGeometricTermDY * invSpp;


                    SurfelGradientRecord gradientRecord{};
                    gradientRecord.primitiveIndex = eventRecord.ySurface.primitiveIndex;
                    gradientRecord.gradPositionX = gradientWrtYPosition.x();
                    gradientRecord.gradPositionY = gradientWrtYPosition.y();
                    gradientRecord.gradPositionZ = gradientWrtYPosition.z();

                    gradientRecords[recordIndex] = gradientRecord;
                });
        }).wait();
    }

    template<typename KernelTag>
    static void reduceSurfelGradientRecords(
        RenderPackage &pkg,
        uint32_t gradientRecordCount) {
        auto &queue = pkg.queue;
        auto gradients = pkg.gradients;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<KernelTag>(
                sycl::range<1>(gradientRecordCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t recordIndex = globalId[0];
                    const SurfelGradientRecord gradientRecord = gradientRecords[recordIndex];

                    if (gradientRecord.primitiveIndex == kInvalidIndex) {
                        return;
                    }

                    atomicAddFloat(gradients.gradPosition[gradientRecord.primitiveIndex].x(),
                                   gradientRecord.gradPositionX);
                    atomicAddFloat(gradients.gradPosition[gradientRecord.primitiveIndex].y(),
                                   gradientRecord.gradPositionY);
                    atomicAddFloat(gradients.gradPosition[gradientRecord.primitiveIndex].z(),
                                   gradientRecord.gradPositionZ);
                });
        }).wait();
    }

    void adjointContributionKernels(
        RenderPackage &pkg,
        uint32_t projectionEventCount,
        uint32_t projectionScatterEventCount,
        uint32_t reflectScatterEventCount,
        uint32_t cameraToSurfaceScatterEventCount,
        uint32_t cameraIndex) {
        (void) cameraIndex;

        const GradientRecordRanges ranges = makeGradientRecordRanges(
            projectionEventCount,
            cameraToSurfaceScatterEventCount,
            projectionScatterEventCount,
            reflectScatterEventCount);

        if (ranges.totalCount > pkg.intermediates.maxGradientRecordCount) {
            throw std::runtime_error("gradient record scratch buffer too small");
        }


        Log::PA_DEBUG(
            "Event counts: projection={}, projectionTransmit={}, projectionScatter={}, reflectScatter={}",
            projectionEventCount,
            cameraToSurfaceScatterEventCount,
            projectionScatterEventCount,
            reflectScatterEventCount);


        if (projectionEventCount > 0) {
            ScopedTimer timer("launchAttachedProjectionKernel", spdlog::level::debug);
            launchAttachedProjectionKernel(
                pkg,
                projectionEventCount,
                ranges.projectionOffset);
        }

        if (cameraToSurfaceScatterEventCount > 0) {
            ScopedTimer timer("launchAttachedProjectionKernel", spdlog::level::debug);
            launchCameraToSurfaceScatterKernel(
                pkg,
                cameraToSurfaceScatterEventCount,
                ranges.projectionTransmitOffset);
        }

        if (projectionScatterEventCount > 0) {
            ScopedTimer timer("launchAttachedScatterKernel", spdlog::level::debug);
            launchAttachedScatterKernel(
                pkg,
                projectionScatterEventCount,
                ranges.projectionScatterOffset);
        }

        if (reflectScatterEventCount > 0) {
            ScopedTimer timer("launchDetachedGradientKernel", spdlog::level::debug);
            launchDetachedGradientKernel(
                pkg,
                reflectScatterEventCount,
                ranges.detachedOffset);
        }

        if (ranges.totalCount > 0) {
            ScopedTimer timer("reduceSurfelGradientRecords", spdlog::level::debug);
            reduceSurfelGradientRecords<class reduceSurfelGradientRecordsTag>(
                pkg,
                ranges.totalCount);
        }
    }
}
