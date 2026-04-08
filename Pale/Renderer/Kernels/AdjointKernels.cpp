//
// Created by magnus on 9/8/25.
//

#include "Renderer/Kernels/AdjointKernels.h"

#include <cmath>

#include "AdjointGradientKernels.h"
#include "IntersectionKernels.h"
#include "Core/ScopedTimer.h"
#include "Renderer/Kernels/KernelHelpers.h"
#include "spdlog/fmt/bundled/base.h"

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
                    rayState.traversalIndex = 0u;
                    rayState.transmission = 1.0f;
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

            commandGroupHandler.parallel_for<class launchAdjointIntersectKernelTag>(
                sycl::range<1>(activeRayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    RayState currentRayState = intermediates.primaryRays[rayIndex];

                    const uint32_t pathId = currentRayState.pathId;
                    const bool canUsePendingAdjointState =
                            pathId < intermediates.maxPendingAdjointStateCount;

                    PendingCameraSegment currentPendingCameraSegment{};
                    PendingAdjointStageX currentPendingStageX{};
                    PendingAdjointStageXY currentPendingStageXY{};

                    clearPendingCameraSegment(currentPendingCameraSegment);
                    clearPendingAdjointStageX(currentPendingStageX);
                    clearPendingAdjointStageXY(currentPendingStageXY);

                    if (canUsePendingAdjointState) {
                        currentPendingCameraSegment = intermediates.pendingCameraSegments[pathId];
                        currentPendingStageX = intermediates.pendingStageX[pathId];
                        currentPendingStageXY = intermediates.pendingStageXY[pathId];
                    }

                    RayState nextRayState{};
                    bool shouldEnqueueNextRayState = false;

                    constexpr uint32_t maxInlineNullTraversals = 256u;

                    for (uint32_t inlineTraversalIndex = 0u;
                         inlineTraversalIndex < maxInlineNullTraversals;
                         ++inlineTraversalIndex) {
                        (void) inlineTraversalIndex;

                        const uint64_t stepSeed = rng::makeSeed(
                            renderSeed,
                            currentRayState.pathId,
                            spp,
                            rng::kStreamTraversal,
                            currentRayState.traversalIndex);

                        rng::Xorshift128 rng(stepSeed);

                        WorldHit worldHit{};
                        intersectScene(
                            currentRayState.ray,
                            &worldHit,
                            scene,
                            rng,
                            SurfelIntersectMode::FirstHit);

                        if (!worldHit.hit) {
                            clearPendingCameraSegment(currentPendingCameraSegment);
                            clearPendingAdjointStageX(currentPendingStageX);
                            clearPendingAdjointStageXY(currentPendingStageXY);
                            break;
                        }

                        buildIntersectionNormal(scene, worldHit);

                        const InstanceRecord &instance = scene.instances[worldHit.instanceIndex];
                        const GeometryType currentGeometryType = instance.geometryType;

                        if (currentGeometryType == GeometryType::Mesh) {
                            float3 orientedNormal = worldHit.geometricNormalW;
                            if (dot(currentRayState.ray.direction, orientedNormal) > 0.0f) {
                                orientedNormal = -orientedNormal;
                            }

                            float3 sampledOutgoingDirectionWorld{0.0f, 0.0f, 0.0f};
                            float cosineHemispherePdf = 0.0f;
                            sampleCosineHemisphere(
                                rng,
                                orientedNormal,
                                sampledOutgoingDirectionWorld,
                                cosineHemispherePdf);

                            const GPUMaterial material = scene.materials[instance.materialIndex];
                            const float3 throughputMultiplier = material.baseColor;

                            nextRayState.ray.origin = worldHit.hitPositionW + (orientedNormal * 1e-6f);
                            nextRayState.ray.direction = sampledOutgoingDirectionWorld;
                            nextRayState.ray.normal = orientedNormal;
                            nextRayState.bounceIndex = currentRayState.bounceIndex + 1u;
                            nextRayState.pixelIndex = currentRayState.pixelIndex;
                            nextRayState.pathId = currentRayState.pathId;
                            nextRayState.pathThroughput =
                                    currentRayState.pathThroughput * throughputMultiplier;
                            nextRayState.traversalIndex = currentRayState.traversalIndex + 1u;
                            nextRayState.transmission = 1.0f;

                            clearPendingCameraSegment(currentPendingCameraSegment);
                            clearPendingAdjointStageX(currentPendingStageX);
                            clearPendingAdjointStageXY(currentPendingStageXY);

                            if (applyRussianRoulette(
                                rng,
                                nextRayState.bounceIndex,
                                nextRayState.pathThroughput,
                                settings.russianRouletteStart)) {
                                shouldEnqueueNextRayState = true;
                            }

                            break;
                        }

                        if (currentGeometryType == GeometryType::PointCloud) {
                            const Point &surfel = scene.points[worldHit.primitiveIndex];

                            const float qNull = settings.sampling.qNull;
                            const float qReflect = settings.sampling.qReflect;

                            const float3 canonicalNormal = normalize(cross(
                                surfel.scale.x() * surfel.tanU,
                                surfel.scale.y() * surfel.tanV));

                            const float signedCosineIncident =
                                    dot(canonicalNormal, -currentRayState.ray.direction);
                            const int sideSign = signNonZero(signedCosineIncident);
                            const float3 orientedNormal =
                                    static_cast<float>(sideSign) * canonicalNormal;

                            const float randomNumber = rng.nextFloat();
                            const bool sampledNull = randomNumber < qNull;

                            if (sampledNull) {
                                const float attenuation =
                                        1.0f - worldHit.alphaGeom * surfel.opacity;

                                currentRayState.ray.origin =
                                        worldHit.hitPositionW + (currentRayState.ray.direction * 1e-5f);
                                currentRayState.ray.normal = orientedNormal;
                                currentRayState.pathThroughput =
                                        currentRayState.pathThroughput / qNull;
                                currentRayState.traversalIndex =
                                        currentRayState.traversalIndex + 1u;
                                currentRayState.transmission =
                                        currentRayState.transmission * attenuation;

                                continue;
                            }

                            float3 sampledOutgoingDirectionWorld{0.0f, 0.0f, 0.0f};
                            float uniformHemispherePdf = 1.0f / (2.0f * M_PIf);
                            sampleUniformHemisphereAroundNormal(
                                rng,
                                orientedNormal,
                                sampledOutgoingDirectionWorld,
                                uniformHemispherePdf);

                            PointCloudSurfaceRecord currentSurfaceRecord =
                                    makePointCloudSurfaceRecord(worldHit, currentRayState, scene);

                            const float alpha = worldHit.alphaGeom * surfel.opacity;
                            const float3 surfelBrdf =
                                    surfel.alpha_r * surfel.albedo * M_1_PIf;
                            const float cosineTheta = sycl::fmax(
                                0.0f,
                                dot(sampledOutgoingDirectionWorld, orientedNormal));

                            const float3 throughputMultiplier =
                                    ((alpha / qReflect) * (surfelBrdf * cosineTheta)) /
                                    uniformHemispherePdf;

                            if (canUsePendingAdjointState) {
                                const PendingCameraSegment previousPendingCameraSegment =
                                        currentPendingCameraSegment;
                                const PendingAdjointStageX previousPendingStageX =
                                        currentPendingStageX;
                                const PendingAdjointStageXY previousPendingStageXY =
                                        currentPendingStageXY;

                                // Camera-attached one-point event
                                if (previousPendingCameraSegment.valid) {
                                    MeasurementGradientEvent measurementEvent{};
                                    measurementEvent.xSurface = currentSurfaceRecord;
                                    measurementEvent.transmission = currentRayState.transmission;
                                    measurementEvent.xPathThroughput =
                                            currentRayState.pathThroughput / qReflect;

                                    appendEventAtomic(
                                        intermediates.countMeasurementEvents,
                                        intermediates.measurementEvents,
                                        intermediates.maxMeasurementEventCount,
                                        measurementEvent);
                                }

                                // Generic transport one-point event for every real reflected surfel hit
                                {
                                    TransportOnePointGradientEvent onePointEvent{};
                                    onePointEvent.xSurface = currentSurfaceRecord;
                                    onePointEvent.transmission = currentRayState.transmission;
                                    onePointEvent.xPathThroughput =
                                            currentRayState.pathThroughput / qReflect;
                                    onePointEvent.useImplicitRayHitJacobian =
                                            previousPendingCameraSegment.valid;

                                    appendEventAtomic(
                                        intermediates.countOnePointEvents,
                                        intermediates.transportOnePointEvents,
                                        intermediates.maxOnePointEventCount,
                                        onePointEvent);
                                }

                                if (previousPendingStageX.valid) {
                                    TransportTwoPointsGradientEvent twoPointEvent{};
                                    twoPointEvent.xSurface = previousPendingStageX.xSurface;
                                    twoPointEvent.ySurface = currentSurfaceRecord;
                                    twoPointEvent.xPathThroughput =
                                            previousPendingStageX.xPathThroughput / qReflect;
                                    twoPointEvent.transmissionPreviousSegment =
                                            previousPendingStageX.previousSegmentTransmission;
                                    twoPointEvent.transmission =
                                            currentRayState.transmission;
                                    twoPointEvent.useImplicitRayHitJacobian =
                                            previousPendingStageX.useImplicitRayHitJacobian;

                                    appendEventAtomic(
                                        intermediates.countTwoPointEvents,
                                        intermediates.transportTwoPointEvents,
                                        intermediates.maxTwoPointEventCount,
                                        twoPointEvent);
                                }

                                if (previousPendingStageXY.valid) {
                                    TransportThreePointsGradientEvent threePointEvent{};
                                    threePointEvent.xSurface = previousPendingStageXY.xSurface;
                                    threePointEvent.ySurface = previousPendingStageXY.ySurface;
                                    threePointEvent.zSurface = currentSurfaceRecord;
                                    threePointEvent.transmission = currentRayState.transmission;
                                    threePointEvent.xPathThroughput =
                                            previousPendingStageXY.xPathThroughput / qReflect;

                                    appendEventAtomic(
                                        intermediates.countThreePointEvents,
                                        intermediates.transportThreePointEvents,
                                        intermediates.maxThreePointEventCount,
                                        threePointEvent);
                                }

                                clearPendingCameraSegment(currentPendingCameraSegment);

                                currentPendingStageX.valid = true;
                                currentPendingStageX.pathId = currentRayState.pathId;
                                currentPendingStageX.pixelIndex = currentRayState.pixelIndex;
                                currentPendingStageX.xSurface = currentSurfaceRecord;
                                currentPendingStageX.xPathThroughput =
                                        currentRayState.pathThroughput / qReflect;
                                currentPendingStageX.previousSegmentTransmission =
                                        currentRayState.transmission;
                                currentPendingStageX.useImplicitRayHitJacobian =
                                        previousPendingCameraSegment.valid;

                                if (previousPendingStageX.valid) {
                                    currentPendingStageXY.valid = true;
                                    currentPendingStageXY.pathId = currentRayState.pathId;
                                    currentPendingStageXY.pixelIndex = currentRayState.pixelIndex;
                                    currentPendingStageXY.xSurface =
                                            previousPendingStageX.xSurface;
                                    currentPendingStageXY.ySurface =
                                            currentSurfaceRecord;
                                    currentPendingStageXY.xPathThroughput =
                                            previousPendingStageX.xPathThroughput / qReflect;
                                } else {
                                    clearPendingAdjointStageXY(currentPendingStageXY);
                                }
                            }

                            nextRayState.ray.origin =
                                    worldHit.hitPositionW + (orientedNormal * 1e-5f);
                            nextRayState.ray.direction = sampledOutgoingDirectionWorld;
                            nextRayState.ray.normal = orientedNormal;
                            nextRayState.bounceIndex = currentRayState.bounceIndex + 1u;
                            nextRayState.pixelIndex = currentRayState.pixelIndex;
                            nextRayState.pathId = currentRayState.pathId;
                            nextRayState.pathThroughput =
                                    currentRayState.pathThroughput * throughputMultiplier;
                            nextRayState.traversalIndex =
                                    currentRayState.traversalIndex + 1u;
                            nextRayState.transmission = 1.0f;

                            if (applyRussianRoulette(
                                rng,
                                nextRayState.bounceIndex,
                                nextRayState.pathThroughput,
                                settings.russianRouletteStart)) {
                                shouldEnqueueNextRayState = true;
                            } else {
                                clearPendingCameraSegment(currentPendingCameraSegment);
                                clearPendingAdjointStageX(currentPendingStageX);
                                clearPendingAdjointStageXY(currentPendingStageXY);
                            }

                            break;
                        }

                        clearPendingCameraSegment(currentPendingCameraSegment);
                        clearPendingAdjointStageX(currentPendingStageX);
                        clearPendingAdjointStageXY(currentPendingStageXY);
                        break;
                    }

                    if (canUsePendingAdjointState) {
                        if (currentPendingCameraSegment.valid) {
                            intermediates.pendingCameraSegments[pathId] =
                                    currentPendingCameraSegment;
                        } else {
                            clearPendingCameraSegment(
                                intermediates.pendingCameraSegments[pathId]);
                        }

                        if (currentPendingStageX.valid) {
                            intermediates.pendingStageX[pathId] = currentPendingStageX;
                        } else {
                            clearPendingAdjointStageX(
                                intermediates.pendingStageX[pathId]);
                        }

                        if (currentPendingStageXY.valid) {
                            intermediates.pendingStageXY[pathId] = currentPendingStageXY;
                        } else {
                            clearPendingAdjointStageXY(
                                intermediates.pendingStageXY[pathId]);
                        }
                    }

                    if (shouldEnqueueNextRayState) {
                        auto extensionCounter = sycl::atomic_ref<
                            uint32_t,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                            *intermediates.countExtensionOut);

                        const uint32_t outIndex = extensionCounter.fetch_add(1u);
                        intermediates.extensionRaysA[outIndex] = nextRayState;
                    }
                });
        }).wait();
    }

    static void measurementGradientEvent(
        RenderPackage &pkg,
        uint32_t cameraIndex,
        uint32_t measurementEventCount,
        uint32_t baseOffset) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        const auto &photonMap = pkg.intermediates.map;
        auto &sensor = pkg.sensors[cameraIndex];
        MeasurementGradientEvent *measurementEvents = pkg.intermediates.measurementEvents;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;

        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;
        const float qNullInv = 1.0f / settings.sampling.qNull;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class measurementGradientEventTag>(
                sycl::range<1>(measurementEventCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex = globalId[0];
                    static constexpr uint32_t recordsPerEvent = 1u + kMaxSplatEventsPerRay;
                    const uint32_t eventRecordBase = baseOffset + recordsPerEvent * eventIndex;
                    const uint32_t recordIndex = eventRecordBase + 0u;

                    for (uint32_t recordOffset = 0u; recordOffset < recordsPerEvent; ++recordOffset) {
                        SurfelGradientRecord invalidRecord{};
                        invalidRecord.primitiveIndex = kInvalidIndex;
                        gradientRecords[eventRecordBase + recordOffset] = invalidRecord;
                    }

                    const MeasurementGradientEvent eventRecord =
                            measurementEvents[eventIndex];

                    const Point &surfelX =
                            scene.points[eventRecord.xSurface.primitiveIndex];
                    const ReconstructedSurfelState xState =
                            reconstructSurfelState(surfelX, eventRecord.xSurface);

                    const float3 outgoingRadianceX =
                            evaluateOutgoingRadianceWithLocalAlpha(
                                surfelX,
                                eventRecord.xSurface,
                                xState,
                                photonMap);

                    const float3 vectorCameraToX = xState.position - sensor.camera.pos;
                    const float distanceSquared = dot(vectorCameraToX, vectorCameraToX);
                    if (distanceSquared <= 1e-12f) {
                        return;
                    }

                    const float distance = sycl::sqrt(distanceSquared);
                    const float targetDistance = distance;
                    const float distanceEpsilon = 1e-4f;

                    float transmittance = 1.0f;
                    float3 accumulatedAlphaDerivativeOverOneMinusAlphaEndpoint = float3{0.0f};

                    struct OccluderDerivatives {
                        float3 derivative{0.0f};
                        uint32_t primitiveIndex = UINT32_MAX;
                    };

                    OccluderDerivatives occluderDerivatives[kMaxSplatEventsPerRay];
                    uint32_t storedOccluderCount = 0u;
                    float qNullInvTotal = 1.0f;

                    if (eventRecord.transmission != 1.0f) {
                        const float3 rayDirection = normalize(vectorCameraToX);
                        Ray ray = {sensor.camera.pos, rayDirection};
                        const float3 segmentOrigin = ray.origin;
                        ray.origin = sensor.camera.pos + rayDirection * distanceEpsilon;

                        while (true) {
                            WorldHit worldHit{};
                            intersectScene(
                                ray,
                                &worldHit,
                                scene,
                                rng::Xorshift128(0.0),
                                SurfelIntersectMode::FirstHit);

                            if (!worldHit.hit) {
                                break;
                            }

                            const float hitDistance =
                                    length(worldHit.hitPositionW - sensor.camera.pos);
                            if (hitDistance >= targetDistance - distanceEpsilon) {
                                break;
                            }

                            buildIntersectionNormal(scene, worldHit);
                            const auto &instance = scene.instances[worldHit.instanceIndex];
                            if (instance.geometryType != GeometryType::PointCloud) {
                                break;
                            }

                            const Point &occluderSurfel =
                                    scene.points[worldHit.primitiveIndex];

                            float3 occluderNormal =
                                    normalize(cross(occluderSurfel.tanU, occluderSurfel.tanV));

                            const bool hitBackside =
                                    dot(occluderNormal, -ray.direction) < 0.0f;
                            if (hitBackside) {
                                occluderNormal = -occluderNormal;
                            }

                            const float alphaEffective =
                                    occluderSurfel.opacity * worldHit.alphaGeom;
                            const float oneMinusAlpha = 1.0f - alphaEffective;
                            if (oneMinusAlpha <= 1e-8f) {
                                break;
                            }

                            transmittance *= oneMinusAlpha;
                            ray.origin = worldHit.hitPositionW + (ray.direction * 1e-4f);

                            const float2 uv = phiInverse(worldHit.hitPositionW, occluderSurfel);
                            const float uOcc = uv.x();
                            const float vOcc = uv.y();

                            const float3 dxy = segmentOrigin - xState.position;
                            const float alphaGeomOccluder = worldHit.alphaGeom;
                            const float denominator = dot(occluderNormal, dxy);
                            if (sycl::fabs(denominator) <= 1e-8f) {
                                continue;
                            }

                            const float lambdaOccluder =
                                    dot(occluderNormal, occluderSurfel.position - xState.position) /
                                    denominator;

                            const float scaleU = occluderSurfel.scale.x();
                            const float scaleV = occluderSurfel.scale.y();
                            if (scaleU <= 1e-12f || scaleV <= 1e-12f) {
                                continue;
                            }

                            const float3 tangentU = occluderSurfel.tanU;
                            const float3 tangentV = occluderSurfel.tanV;
                            const float3 localBasisU = tangentU / scaleU;
                            const float3 localBasisV = tangentV / scaleV;
                            const float inverseDenominator = 1.0f / denominator;

                            const float3 dUiDx =
                                    (1.0f - lambdaOccluder) * (
                                        localBasisU -
                                        occluderNormal *
                                        (dot(dxy, localBasisU) * inverseDenominator));

                            const float3 dViDx =
                                    (1.0f - lambdaOccluder) * (
                                        localBasisV -
                                        occluderNormal *
                                        (dot(dxy, localBasisV) * inverseDenominator));

                            const float3 dUiDspi =
                                    occluderNormal *
                                    (dot(dxy, tangentU) / scaleU) * inverseDenominator -
                                    localBasisU;

                            const float3 dViDspi =
                                    occluderNormal *
                                    (dot(dxy, tangentV) / scaleV) * inverseDenominator -
                                    localBasisV;

                            const float radiusSquaredOcc = uOcc * uOcc + vOcc * vOcc;
                            const float oneMinusRadiusSquaredOcc = 1.0f - radiusSquaredOcc;
                            if (oneMinusRadiusSquaredOcc <= 1e-8f) {
                                continue;
                            }

                            const float betaScaleOcc = 4.0f * sycl::exp(occluderSurfel.beta);
                            const float dAlphaGeomDu =
                                    -2.0f * betaScaleOcc * uOcc * alphaGeomOccluder /
                                    oneMinusRadiusSquaredOcc;
                            const float dAlphaGeomDv =
                                    -2.0f * betaScaleOcc * vOcc * alphaGeomOccluder /
                                    oneMinusRadiusSquaredOcc;

                            const float3 dAlphaEffectiveDx =
                                    occluderSurfel.opacity * (
                                        dAlphaGeomDu * dUiDx +
                                        dAlphaGeomDv * dViDx);

                            const float3 dAlphaEffectiveDspi =
                                    occluderSurfel.opacity * (
                                        dAlphaGeomDu * dUiDspi +
                                        dAlphaGeomDv * dViDspi);

                            accumulatedAlphaDerivativeOverOneMinusAlphaEndpoint +=
                                    dAlphaEffectiveDx * (1.0f / oneMinusAlpha);

                            if (storedOccluderCount < kMaxSplatEventsPerRay) {
                                occluderDerivatives[storedOccluderCount].derivative =
                                        dAlphaEffectiveDspi * (1.0f / oneMinusAlpha);
                                occluderDerivatives[storedOccluderCount].primitiveIndex =
                                        worldHit.primitiveIndex;
                                storedOccluderCount++;
                            }

                            qNullInvTotal *= qNullInv;
                        }
                    }

                    const float3 pathWeight = eventRecord.xPathThroughput;
                    const float scalarWeight = dot(pathWeight, outgoingRadianceX);

                    SurfelGradientRecord gradientRecord{};
                    gradientRecord.primitiveIndex = eventRecord.xSurface.primitiveIndex;

                    const float3 dTransmittanceDx =
                            -transmittance * accumulatedAlphaDerivativeOverOneMinusAlphaEndpoint;

                    float3 gradientWrtHitPositionX =
                            scalarWeight * dTransmittanceDx * invSpp;

                    const float3x3 hitPointJacobian = planeHitPointIntersectionJacobian(
                        eventRecord.xSurface.incomingDirection,
                        xState.orientedNormal);

                    gradientWrtHitPositionX =
                            transpose(hitPointJacobian) * gradientWrtHitPositionX;

                    gradientRecord.gradPositionX = gradientWrtHitPositionX.x();
                    gradientRecord.gradPositionY = gradientWrtHitPositionX.y();
                    gradientRecord.gradPositionZ = gradientWrtHitPositionX.z();
                    gradientRecords[recordIndex] = gradientRecord;

                    const float occluderScale =
                            -transmittance * scalarWeight * invSpp;

                    for (uint32_t occluderIndex = 0u;
                         occluderIndex < storedOccluderCount;
                         ++occluderIndex) {
                        const uint32_t occluderRecordIndex = eventRecordBase + 1u + occluderIndex;

                        const OccluderDerivatives &occluderDerivative =
                                occluderDerivatives[occluderIndex];

                        SurfelGradientRecord gradientRecordOccluder{};
                        gradientRecordOccluder.primitiveIndex =
                                occluderDerivative.primitiveIndex;

                        const float3 occluderContribution =
                                occluderScale * occluderDerivative.derivative;

                        gradientRecordOccluder.gradPositionX = occluderContribution.x();
                        gradientRecordOccluder.gradPositionY = occluderContribution.y();
                        gradientRecordOccluder.gradPositionZ = occluderContribution.z();

                        gradientRecords[occluderRecordIndex] = gradientRecordOccluder;
                    }
                });
        }).wait();
    }

    static void firstHitGradientEvent(
        RenderPackage &pkg,
        uint32_t onePointEventCount,
        uint32_t baseOffset) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        const auto &photonMap = pkg.intermediates.map;
        TransportOnePointGradientEvent *onePointEvents =
                pkg.intermediates.transportOnePointEvents;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;

        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class firstHitGradientEventTag>(
                sycl::range<1>(onePointEventCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex = globalId[0];
                    const uint32_t recordIndex = baseOffset + eventIndex;

                    const TransportOnePointGradientEvent eventRecord =
                            onePointEvents[eventIndex];

                    const Point &surfelX =
                            scene.points[eventRecord.xSurface.primitiveIndex];
                    const ReconstructedSurfelState xState =
                            reconstructSurfelState(surfelX, eventRecord.xSurface);

                    // L_surfel(X, omega_x->c), without alpha_X
                    const float3 outgoingRadianceX = evaluateSurfelRadianceWithoutLocalAlpha(
                        surfelX,
                        eventRecord.xSurface, xState,
                        photonMap);

                    const float u = eventRecord.xSurface.uv.x();
                    const float v = eventRecord.xSurface.uv.y();
                    const float radiusSquared = u * u + v * v;
                    const float oneMinusRadiusSquared = 1.0f - radiusSquared;
                    if (oneMinusRadiusSquared <= 1e-8f) {
                        return;
                    }

                    const float scaleU = surfelX.scale.x();
                    const float scaleV = surfelX.scale.y();
                    UVPositionJacobian uvPositionJacobian{};
                    uvPositionJacobian =
                            computeDuvDSurfelTranslationJacobianForImplicitRayHit(
                                surfelX.tanU,
                                surfelX.tanV,
                                xState.orientedNormal,
                                eventRecord.xSurface.incomingDirection,
                                scaleU,
                                scaleV);

                    const float3 dUvDPosition =
                            u * uvPositionJacobian.du_d_surfel_translation +
                            v * uvPositionJacobian.dv_d_surfel_translation;

                    const float betaScale = 4.0f * sycl::exp(surfelX.beta);
                    const float factor =
                            (-2.0f * betaScale * eventRecord.xSurface.alphaGeom) /
                            oneMinusRadiusSquared;

                    const float3 dAlphaGeomDPosition = factor * dUvDPosition;
                    const float3 dAlphaEffectiveDPosition =
                            surfelX.opacity * dAlphaGeomDPosition;

                    const float scalarWeight =
                            dot(eventRecord.xPathThroughput, outgoingRadianceX);

                    const float3 positionGradient =
                            eventRecord.transmission *
                            dAlphaEffectiveDPosition *
                            scalarWeight *
                            invSpp;

                    SurfelGradientRecord gradientRecord{};
                    gradientRecord.primitiveIndex = eventRecord.xSurface.primitiveIndex;
                    gradientRecord.gradPositionX = positionGradient.x();
                    gradientRecord.gradPositionY = positionGradient.y();
                    gradientRecord.gradPositionZ = positionGradient.z();
                    gradientRecords[recordIndex] = gradientRecord;
                });
        }).wait();
    }

    static void twoPointGradientEvent(
        RenderPackage &pkg,
        uint32_t twoPointEventCount,
        uint32_t baseOffset) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        const auto &photonMap = pkg.intermediates.map;
        TransportTwoPointsGradientEvent *twoPointEvents =
                pkg.intermediates.transportTwoPointEvents;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;

        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;
        const float qNullInv = 1.0f / settings.sampling.qNull;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class twoPointGradientEventTag>(
                sycl::range<1>(twoPointEventCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex = globalId[0];

                    static constexpr uint32_t recordsPerEvent = 2u + kMaxSplatEventsPerRay;
                    const uint32_t eventRecordBase = baseOffset + recordsPerEvent * eventIndex;
                    const uint32_t xRecordIndex = eventRecordBase + 0u;
                    const uint32_t yRecordIndex = eventRecordBase + 1u;

                    for (uint32_t recordOffset = 0u; recordOffset < recordsPerEvent; ++recordOffset) {
                        SurfelGradientRecord invalidRecord{};
                        invalidRecord.primitiveIndex = kInvalidIndex;
                        gradientRecords[eventRecordBase + recordOffset] = invalidRecord;
                    }

                    const TransportTwoPointsGradientEvent eventRecord =
                            twoPointEvents[eventIndex];

                    const uint32_t xPrimitiveIndex = eventRecord.xSurface.primitiveIndex;
                    const uint32_t yPrimitiveIndex = eventRecord.ySurface.primitiveIndex;

                    const Point &surfelX = scene.points[xPrimitiveIndex];
                    const Point &surfelY = scene.points[yPrimitiveIndex];

                    const ReconstructedSurfelState xState =
                            reconstructSurfelState(surfelX, eventRecord.xSurface);
                    const ReconstructedSurfelState yState =
                            reconstructSurfelState(surfelY, eventRecord.ySurface);

                    const float3 outgoingRadianceY =
                            evaluateOutgoingRadianceWithLocalAlpha(
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

                    const float geometricTermXY = computeGeometricTermValue(
                        xState.position,
                        yState.position,
                        xState.orientedNormal,
                        yState.orientedNormal);

                    const float3 dGeometricTermDx = computeGeometricTermGradientWrtStartpoint(
                        xState.position,
                        yState.position,
                        xState.orientedNormal,
                        yState.orientedNormal);

                    const float3 dGeometricTermDy = computeGeometricTermGradientWrtEndpoint(
                        xState.position,
                        yState.position,
                        xState.orientedNormal,
                        yState.orientedNormal);

                    const float3 pathWeight =
                            eventRecord.xPathThroughput * eventRecord.transmissionPreviousSegment;

                    const float3 transportWithoutTauAndGeometric =
                            outgoingRadianceY * alphaX * brdfX;

                    float scalarWeightWithoutTauAndGeometric =
                            dot(pathWeight, transportWithoutTauAndGeometric) / pAreaY;

                    struct OccluderDerivative {
                        float3 derivative{0.0f};
                        uint32_t primitiveIndex = kInvalidIndex;
                    };

                    float transmittance = 1.0f;
                    float3 accumulatedAlphaDerivativeOverOneMinusAlphaStartPoint = float3{0.0f};
                    float3 accumulatedAlphaDerivativeOverOneMinusAlphaEndPoint = float3{0.0f};

                    OccluderDerivative occluderDerivatives[kMaxSplatEventsPerRay];
                    uint32_t storedOccluderCount = 0u;

                    if (eventRecord.transmission != 1.0f) {
                        const float distanceEpsilon = 1e-4f;

                        const float3 rayDirection = normalize(vectorXToY);
                        Ray ray = {xState.position, rayDirection};
                        ray.origin = xState.position + rayDirection * distanceEpsilon;

                        const float targetDistance = length(xState.position - yState.position);
                        const float3 xPosition = xState.position;
                        const float3 yPosition = yState.position;
                        const float3 dxy = xPosition - yPosition;

                        while (true) {
                            WorldHit worldHit{};
                            intersectScene(
                                ray,
                                &worldHit,
                                scene,
                                rng::Xorshift128(0.0),
                                SurfelIntersectMode::FirstHit);

                            if (!worldHit.hit) {
                                break;
                            }

                            const float hitDistance = length(worldHit.hitPositionW - xState.position);
                            if (hitDistance >= targetDistance - distanceEpsilon) {
                                break;
                            }

                            buildIntersectionNormal(scene, worldHit);
                            auto &instance = scene.instances[worldHit.instanceIndex];
                            if (instance.geometryType != GeometryType::PointCloud) {
                                ray.origin = worldHit.hitPositionW + (ray.direction * 1e-4f);
                                continue;
                            }

                            const Point &occluderSurfel = scene.points[worldHit.primitiveIndex];
                            float3 occluderNormal =
                                    normalize(cross(occluderSurfel.tanU, occluderSurfel.tanV));
                            const bool hitBackside =
                                    dot(occluderNormal, -ray.direction) < 0.0f;
                            if (hitBackside) {
                                occluderNormal = -occluderNormal;
                            }

                            const float alphaEffective = occluderSurfel.opacity * worldHit.alphaGeom;
                            const float oneMinusAlpha = 1.0f - alphaEffective;
                            if (oneMinusAlpha <= 1e-8f) {
                                break;
                            }

                            transmittance *= oneMinusAlpha;
                            ray.origin = worldHit.hitPositionW + (ray.direction * 1e-4f);

                            const float2 uv = phiInverse(worldHit.hitPositionW, occluderSurfel);
                            const float u = uv.x();
                            const float v = uv.y();

                            const float scaleU = occluderSurfel.scale.x();
                            const float scaleV = occluderSurfel.scale.y();
                            if (scaleU <= 1e-12f || scaleV <= 1e-12f) {
                                continue;
                            }

                            const float3 tangentU = occluderSurfel.tanU;
                            const float3 tangentV = occluderSurfel.tanV;
                            const float3 localBasisU = tangentU / scaleU;
                            const float3 localBasisV = tangentV / scaleV;
                            const float alphaGeomOccluder = worldHit.alphaGeom;

                            const float denominator = dot(occluderNormal, dxy);
                            if (sycl::fabs(denominator) <= 1e-8f) {
                                continue;
                            }

                            const float lambdaOccluder =
                                    dot(occluderNormal, occluderSurfel.position - yPosition) / denominator;
                            const float inverseDenominator = 1.0f / denominator;

                            const float3 dUiDx =
                                    lambdaOccluder * (
                                        localBasisU -
                                        occluderNormal * (dot(dxy, localBasisU) * inverseDenominator));

                            const float3 dUiDy =
                                    (1.0f - lambdaOccluder) * (
                                        localBasisU -
                                        occluderNormal * (dot(dxy, localBasisU) * inverseDenominator));

                            const float3 dViDx =
                                    lambdaOccluder * (
                                        localBasisV -
                                        occluderNormal * (dot(dxy, localBasisV) * inverseDenominator));

                            const float3 dViDy =
                                    (1.0f - lambdaOccluder) * (
                                        localBasisV -
                                        occluderNormal * (dot(dxy, localBasisV) * inverseDenominator));

                            const float3 dUiDspi =
                                    occluderNormal *
                                    (dot(dxy, tangentU) / scaleU) * inverseDenominator -
                                    localBasisU;

                            const float3 dViDspi =
                                    occluderNormal *
                                    (dot(dxy, tangentV) / scaleV) * inverseDenominator -
                                    localBasisV;

                            const float radiusSquared = u * u + v * v;
                            const float oneMinusRadiusSquared = 1.0f - radiusSquared;
                            if (oneMinusRadiusSquared <= 1e-8f) {
                                continue;
                            }

                            const float betaScale = 4.0f * sycl::exp(occluderSurfel.beta);

                            const float dAlphaGeomDu =
                                    -2.0f * betaScale * u * alphaGeomOccluder / oneMinusRadiusSquared;
                            const float dAlphaGeomDv =
                                    -2.0f * betaScale * v * alphaGeomOccluder / oneMinusRadiusSquared;

                            const float3 dAlphaEffectiveDx =
                                    occluderSurfel.opacity * (
                                        dAlphaGeomDu * dUiDx +
                                        dAlphaGeomDv * dViDx);

                            const float3 dAlphaEffectiveDy =
                                    occluderSurfel.opacity * (
                                        dAlphaGeomDu * dUiDy +
                                        dAlphaGeomDv * dViDy);

                            const float3 dAlphaEffectiveDspi =
                                    occluderSurfel.opacity * (
                                        dAlphaGeomDu * dUiDspi +
                                        dAlphaGeomDv * dViDspi);

                            accumulatedAlphaDerivativeOverOneMinusAlphaStartPoint +=
                                    dAlphaEffectiveDx * (1.0f / oneMinusAlpha);

                            accumulatedAlphaDerivativeOverOneMinusAlphaEndPoint +=
                                    dAlphaEffectiveDy * (1.0f / oneMinusAlpha);

                            if (storedOccluderCount < kMaxSplatEventsPerRay) {
                                occluderDerivatives[storedOccluderCount].derivative =
                                        dAlphaEffectiveDspi * (1.0f / oneMinusAlpha);
                                occluderDerivatives[storedOccluderCount].primitiveIndex =
                                        worldHit.primitiveIndex;
                                storedOccluderCount++;
                            }

                            scalarWeightWithoutTauAndGeometric *= qNullInv;
                        }
                    }

                    const float3 dTransmittanceDx =
                            -transmittance * accumulatedAlphaDerivativeOverOneMinusAlphaStartPoint;

                    float3 gradientWrtHitPositionX =
                            scalarWeightWithoutTauAndGeometric *
                            (geometricTermXY * dTransmittanceDx + transmittance * dGeometricTermDx);

                    //if (eventRecord.useImplicitRayHitJacobian)
                        {
                        const float3x3 hitPointJacobianX = planeHitPointIntersectionJacobian(
                            eventRecord.xSurface.incomingDirection,
                            xState.orientedNormal);
                        gradientWrtHitPositionX = transpose(hitPointJacobianX) * gradientWrtHitPositionX;
                    }


                    const float3 dTransmittanceDy =
                            -transmittance * accumulatedAlphaDerivativeOverOneMinusAlphaEndPoint;

                    float3 gradientWrtHitPositionY =
                            scalarWeightWithoutTauAndGeometric *
                            (geometricTermXY * dTransmittanceDy + transmittance * dGeometricTermDy);

                    //const float3x3 hitPointJacobianY = planeHitPointIntersectionJacobian(
                    //    eventRecord.ySurface.incomingDirection,
                    //    yState.orientedNormal);
                    //gradientWrtHitPositionY = transpose(hitPointJacobianY) * gradientWrtHitPositionY;

                    const float3 xContribution = gradientWrtHitPositionX * invSpp;
                    const float3 yContribution = gradientWrtHitPositionY * invSpp;

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

                    const float occluderScale =
                            -transmittance *
                            geometricTermXY *
                            scalarWeightWithoutTauAndGeometric *
                            invSpp;

                    for (uint32_t occluderIndex = 0u;
                         occluderIndex < storedOccluderCount;
                         ++occluderIndex) {
                        const uint32_t occluderRecordIndex =
                                eventRecordBase + 2u + occluderIndex;

                        const OccluderDerivative &occluderDerivative =
                                occluderDerivatives[occluderIndex];
                        const float3 occluderContribution =
                                occluderScale * occluderDerivative.derivative;

                        SurfelGradientRecord occluderRecord{};
                        occluderRecord.primitiveIndex = occluderDerivative.primitiveIndex;
                        occluderRecord.gradPositionX = occluderContribution.x();
                        occluderRecord.gradPositionY = occluderContribution.y();
                        occluderRecord.gradPositionZ = occluderContribution.z();

                        gradientRecords[occluderRecordIndex] = occluderRecord;
                    }
                });
        }).wait();
    }

    static void threePointGradientEvent(
        RenderPackage &pkg,
        uint32_t threePointEventCount,
        uint32_t baseOffset) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        const auto &photonMap = pkg.intermediates.map;
        TransportThreePointsGradientEvent *threePointEvents =
                pkg.intermediates.transportThreePointEvents;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;

        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;
        const float uniformHemispherePdf = 1.0f / (2.0f * M_PIf);
        const float qNullInv = 1.0f / settings.sampling.qNull;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class threePointGradientEventTag>(
                sycl::range<1>(threePointEventCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex = globalId[0];

                    static constexpr uint32_t recordsPerEvent = 1u + kMaxSplatEventsPerRay;
                    const uint32_t eventRecordBase = baseOffset + recordsPerEvent * eventIndex;
                    const uint32_t yRecordIndex = eventRecordBase + 0u;

                    for (uint32_t recordOffset = 0u; recordOffset < recordsPerEvent; ++recordOffset) {
                        SurfelGradientRecord invalidRecord{};
                        invalidRecord.primitiveIndex = kInvalidIndex;
                        gradientRecords[eventRecordBase + recordOffset] = invalidRecord;
                    }

                    const TransportThreePointsGradientEvent eventRecord =
                            threePointEvents[eventIndex];

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
                            evaluateOutgoingRadianceWithLocalAlpha(
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

                    const float pAreaY = uniformHemispherePdf;
                    if (pAreaY <= 1e-20f) {
                        return;
                    }

                    const float geometricTermXY = computeGeometricTermValue(
                        xState.position,
                        yState.position,
                        xState.orientedNormal,
                        yState.orientedNormal);

                    const float geometricTermYZ = computeGeometricTermValue(
                        yState.position,
                        zState.position,
                        yState.orientedNormal,
                        zState.orientedNormal);

                    const float3 dGeometricTermDyOnYZ = computeGeometricTermGradientWrtStartpoint(
                        yState.position,
                        zState.position,
                        yState.orientedNormal,
                        zState.orientedNormal);

                    const float alphaX = eventRecord.xSurface.alphaGeom * surfelX.opacity;
                    const float alphaY = eventRecord.ySurface.alphaGeom * surfelY.opacity;

                    const float3 brdfX = surfelX.alpha_r * surfelX.albedo * M_1_PIf;
                    const float3 brdfY = surfelY.alpha_r * surfelY.albedo * M_1_PIf;

                    const float3 pathWeight = eventRecord.xPathThroughput;

                    // Prefix transport up to Y, excluding the YZ tau/G terms being differentiated here.
                    // NOTE:
                    // If your three-point event payload should also include XY transmittance,
                    // multiply it in here as well.

                    const float cosineX = fmax(0.0f, dot(xState.orientedNormal, directionXToY));


                    const float3 prefixTransport =
                            alphaX * brdfX * cosineX *
                            alphaY * brdfY *
                            outgoingRadianceZ;

                    float scalarWeightWithoutTauAndGeometricYZ =
                            dot(pathWeight, prefixTransport) / (pAreaY * pAreaZ);

                    struct OccluderDerivative {
                        float3 derivative{0.0f};
                        uint32_t primitiveIndex = kInvalidIndex;
                    };

                    float transmittanceYZ = 1.0f;
                    float3 accumulatedAlphaDerivativeOverOneMinusAlphaStartPointYZ = float3{0.0f};

                    OccluderDerivative occluderDerivatives[kMaxSplatEventsPerRay];
                    uint32_t storedOccluderCount = 0u;

                    if (eventRecord.transmission != 1.0f) {
                        const float distanceEpsilon = 1e-4f;

                        Ray ray = {yState.position, directionYToZ};
                        ray.origin = yState.position + directionYToZ * distanceEpsilon;

                        const float targetDistance = distanceYZ;
                        const float3 yPosition = yState.position;
                        const float3 zPosition = zState.position;
                        const float3 dyz = yPosition - zPosition;

                        while (true) {
                            WorldHit worldHit{};
                            intersectScene(
                                ray,
                                &worldHit,
                                scene,
                                rng::Xorshift128(0.0),
                                SurfelIntersectMode::FirstHit);

                            if (!worldHit.hit) {
                                break;
                            }

                            const float hitDistance = length(worldHit.hitPositionW - yState.position);
                            if (hitDistance >= targetDistance - distanceEpsilon) {
                                break;
                            }

                            buildIntersectionNormal(scene, worldHit);
                            auto &instance = scene.instances[worldHit.instanceIndex];
                            if (instance.geometryType != GeometryType::PointCloud) {
                                ray.origin = worldHit.hitPositionW + (ray.direction * 1e-4f);
                                continue;
                            }

                            const Point &occluderSurfel = scene.points[worldHit.primitiveIndex];
                            float3 occluderNormal =
                                    normalize(cross(occluderSurfel.tanU, occluderSurfel.tanV));
                            const bool hitBackside = dot(occluderNormal, -ray.direction) < 0.0f;
                            if (hitBackside) {
                                occluderNormal = -occluderNormal;
                            }

                            const float alphaEffective = occluderSurfel.opacity * worldHit.alphaGeom;
                            const float oneMinusAlpha = 1.0f - alphaEffective;
                            if (oneMinusAlpha <= 1e-8f) {
                                break;
                            }

                            transmittanceYZ *= oneMinusAlpha;
                            ray.origin = worldHit.hitPositionW + (ray.direction * 1e-4f);

                            const float2 uv = phiInverse(worldHit.hitPositionW, occluderSurfel);
                            const float u = uv.x();
                            const float v = uv.y();

                            const float scaleU = occluderSurfel.scale.x();
                            const float scaleV = occluderSurfel.scale.y();
                            if (scaleU <= 1e-12f || scaleV <= 1e-12f) {
                                continue;
                            }

                            const float3 tangentU = occluderSurfel.tanU;
                            const float3 tangentV = occluderSurfel.tanV;
                            const float3 localBasisU = tangentU / scaleU;
                            const float3 localBasisV = tangentV / scaleV;
                            const float alphaGeomOccluder = worldHit.alphaGeom;

                            const float denominator = dot(occluderNormal, dyz);
                            if (sycl::fabs(denominator) <= 1e-8f) {
                                continue;
                            }

                            const float lambdaOccluder =
                                    dot(occluderNormal, occluderSurfel.position - zPosition) / denominator;
                            const float inverseDenominator = 1.0f / denominator;

                            const float3 dUiDy =
                                    lambdaOccluder * (
                                        localBasisU -
                                        occluderNormal * (dot(dyz, localBasisU) * inverseDenominator));

                            const float3 dViDy =
                                    lambdaOccluder * (
                                        localBasisV -
                                        occluderNormal * (dot(dyz, localBasisV) * inverseDenominator));

                            const float3 dUiDspi =
                                    occluderNormal *
                                    (dot(dyz, tangentU) / scaleU) * inverseDenominator -
                                    localBasisU;

                            const float3 dViDspi =
                                    occluderNormal *
                                    (dot(dyz, tangentV) / scaleV) * inverseDenominator -
                                    localBasisV;

                            const float radiusSquared = u * u + v * v;
                            const float oneMinusRadiusSquared = 1.0f - radiusSquared;
                            if (oneMinusRadiusSquared <= 1e-8f) {
                                continue;
                            }

                            const float betaScale = 4.0f * sycl::exp(occluderSurfel.beta);

                            const float dAlphaGeomDu =
                                    -2.0f * betaScale * u * alphaGeomOccluder / oneMinusRadiusSquared;
                            const float dAlphaGeomDv =
                                    -2.0f * betaScale * v * alphaGeomOccluder / oneMinusRadiusSquared;

                            const float3 dAlphaEffectiveDy =
                                    occluderSurfel.opacity * (
                                        dAlphaGeomDu * dUiDy +
                                        dAlphaGeomDv * dViDy);

                            const float3 dAlphaEffectiveDspi =
                                    occluderSurfel.opacity * (
                                        dAlphaGeomDu * dUiDspi +
                                        dAlphaGeomDv * dViDspi);

                            accumulatedAlphaDerivativeOverOneMinusAlphaStartPointYZ +=
                                    dAlphaEffectiveDy * (1.0f / oneMinusAlpha);

                            if (storedOccluderCount < kMaxSplatEventsPerRay) {
                                occluderDerivatives[storedOccluderCount].derivative =
                                        dAlphaEffectiveDspi * (1.0f / oneMinusAlpha);
                                occluderDerivatives[storedOccluderCount].primitiveIndex =
                                        worldHit.primitiveIndex;
                                storedOccluderCount++;
                            }

                            scalarWeightWithoutTauAndGeometricYZ *= qNullInv;
                        }
                    }

                    const float3 dTransmittanceDyOnYZ =
                            -transmittanceYZ * accumulatedAlphaDerivativeOverOneMinusAlphaStartPointYZ;

                    float3 gradientWrtYPosition =
                            scalarWeightWithoutTauAndGeometricYZ *
                            (geometricTermYZ * dTransmittanceDyOnYZ +
                             transmittanceYZ * dGeometricTermDyOnYZ) * invSpp;

                    //const float3x3 hitPointJacobianY = planeHitPointIntersectionJacobian(
                    //    eventRecord.ySurface.incomingDirection,
                    //    yState.orientedNormal);
                    //gradientWrtYPosition = transpose(hitPointJacobianY) * gradientWrtYPosition;

                    SurfelGradientRecord yRecord{};
                    yRecord.primitiveIndex = eventRecord.ySurface.primitiveIndex;
                    yRecord.gradPositionX = gradientWrtYPosition.x();
                    yRecord.gradPositionY = gradientWrtYPosition.y();
                    yRecord.gradPositionZ = gradientWrtYPosition.z();
                    gradientRecords[yRecordIndex] = yRecord;

                    const float occluderScale =
                            -transmittanceYZ *
                            geometricTermYZ *
                            scalarWeightWithoutTauAndGeometricYZ *
                            invSpp;

                    for (uint32_t occluderIndex = 0u;
                         occluderIndex < storedOccluderCount;
                         ++occluderIndex) {
                        const uint32_t occluderRecordIndex =
                                eventRecordBase + 1u + occluderIndex;

                        const OccluderDerivative &occluderDerivative =
                                occluderDerivatives[occluderIndex];
                        const float3 occluderContribution =
                                occluderScale * occluderDerivative.derivative;

                        SurfelGradientRecord occluderRecord{};
                        occluderRecord.primitiveIndex = occluderDerivative.primitiveIndex;
                        occluderRecord.gradPositionX = occluderContribution.x();
                        occluderRecord.gradPositionY = occluderContribution.y();
                        occluderRecord.gradPositionZ = occluderContribution.z();

                        gradientRecords[occluderRecordIndex] = occluderRecord;
                    }
                });
        }).wait();
    }

    static void reduceSurfelGradientRecords(
        RenderPackage &pkg,
        uint32_t gradientRecordCount) {
        auto &queue = pkg.queue;
        auto gradients = pkg.gradients;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<struct reduceSurfelGradientRecords>(
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
        uint32_t measurementEventCount,
        uint32_t onePointEventCount,
        uint32_t twoPointEventCount,
        uint32_t threePointEventCount,
        uint32_t cameraIndex) {
        const GradientRecordRanges ranges = makeGradientRecordRanges(
            measurementEventCount,
            onePointEventCount,
            twoPointEventCount,
            threePointEventCount);

        if (ranges.totalCount > pkg.intermediates.maxGradientRecordCount) {
            throw std::runtime_error("gradient record scratch buffer too small");
        }

        Log::PA_DEBUG(
            "Event counts: measurement={}, onePoint={}, twoPoint={}, threePoint={}",
            measurementEventCount,
            onePointEventCount,
            twoPointEventCount,
            threePointEventCount);

        if (measurementEventCount > 0) {
            ScopedTimer timer("measurementGradientEvent", spdlog::level::debug);
            measurementGradientEvent(
                pkg,
                cameraIndex,
                measurementEventCount,
                ranges.measurementOffset);
        }

        if (onePointEventCount > 0) {
            ScopedTimer timer("firstHitGradientEvent", spdlog::level::debug);
            firstHitGradientEvent(
                pkg,
                onePointEventCount,
                ranges.onePointOffset);
        }

        if (twoPointEventCount > 0) {
            ScopedTimer timer("twoPointGradientEvent", spdlog::level::debug);
            twoPointGradientEvent(
                pkg,
                twoPointEventCount,
                ranges.twoPointOffset);
        }

        if (threePointEventCount > 0) {
            ScopedTimer timer("threePointGradientEvent", spdlog::level::debug);
            threePointGradientEvent(
                pkg,
                threePointEventCount,
                ranges.threePointOffset);
        }

        if (ranges.totalCount > 0) {
            ScopedTimer timer("reduceSurfelGradientRecords", spdlog::level::debug);
            reduceSurfelGradientRecords(
                pkg,
                ranges.totalCount);
        }
    }
}
