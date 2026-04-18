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
    void launchRayGenAdjointKernel(RenderPackage& pkg, int spp, uint32_t cameraIndex) {
        auto& queue = pkg.queue;
        auto& settings = pkg.settings;
        auto& intermediates = pkg.intermediates;
        auto& sensor = pkg.sensors[cameraIndex];

        const uint32_t imageWidth = sensor.camera.width;
        const uint32_t imageHeight = sensor.camera.height;
        uint32_t raysPerSet = imageWidth * imageHeight;


        queue.submit([&](sycl::handler& commandGroupHandler) {
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
                    }
                });
        }).wait();
    }

    void launchAdjointIntersectKernel(RenderPackage& pkg, uint32_t spp, uint32_t activeRayCount, uint32_t cameraIndex) {
        auto& queue = pkg.queue;
        auto& settings = pkg.settings;
        auto& intermediates = pkg.intermediates;
        auto& scene = pkg.scene;
        auto& sensor = pkg.sensors[cameraIndex];

        queue.submit([&](sycl::handler& commandGroupHandler) {
            const uint64_t renderSeed = settings.random.seed;

            commandGroupHandler.parallel_for<class launchAdjointIntersectKernelTag>(
                sycl::range<1>(activeRayCount),
                [=](sycl::id<1> global_id) {
                    const uint32_t rayIndex = global_id[0];
                    RayState currentRayState = intermediates.primaryRays[rayIndex];

                    const uint32_t pathId = currentRayState.pathId;
                    const bool hasPendingState =
                        pathId < intermediates.maxPendingAdjointStateCount;

                    PendingCameraSegment pendingCameraSegment{};
                    PendingAdjointStageX pendingAdjointStage{};

                    clearPendingCameraSegment(pendingCameraSegment);
                    clearPendingAdjointStageX(pendingAdjointStage);

                    if (hasPendingState) {
                        pendingCameraSegment = intermediates.pendingCameraSegments[pathId];
                        pendingAdjointStage = intermediates.pendingStageX[pathId];
                    }

                    RayState nextRayState{};
                    bool shouldEnqueueNextRayState = false;

                    constexpr uint32_t maxInlineNullTraversals = 256u;

                    for (uint32_t inlineTraversalIndex = 0u;
                         inlineTraversalIndex < maxInlineNullTraversals;
                         ++inlineTraversalIndex) {
                        (void)inlineTraversalIndex;

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
                            clearPendingCameraSegment(pendingCameraSegment);
                            clearPendingAdjointStageX(pendingAdjointStage);
                            break;
                        }

                        buildIntersectionNormal(scene, worldHit);

                        const InstanceRecord& instance = scene.instances[worldHit.instanceIndex];
                        const GeometryType geometryType = instance.geometryType;

                        // ---------------------------------------------------------------------
                        // Mesh path
                        // ---------------------------------------------------------------------
                        if (geometryType == GeometryType::Mesh) {
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
                                currentRayState.pathThroughput * throughputMultiplier * currentRayState.transmission;
                            nextRayState.traversalIndex = currentRayState.traversalIndex + 1u;
                            nextRayState.transmission = 1.0f;

                            clearPendingCameraSegment(pendingCameraSegment);
                            clearPendingAdjointStageX(pendingAdjointStage);

                            if (applyRussianRoulette(
                                rng,
                                nextRayState.bounceIndex,
                                nextRayState.pathThroughput,
                                settings.russianRouletteStart)) {
                                shouldEnqueueNextRayState = true;
                            }

                            break;
                        }

                        // ---------------------------------------------------------------------
                        // Point cloud path
                        // ---------------------------------------------------------------------
                        if (geometryType == GeometryType::PointCloud) {
                            const Point& surfel = scene.points[worldHit.primitiveIndex];

                            const float qNull = settings.sampling.qNull;
                            const float qReflect = settings.sampling.qReflect;

                            const float3 orientedNormal =
                                computePointCloudOrientedNormal(surfel, currentRayState.ray.direction);

                            const float randomNumber = rng.nextFloat();
                            const bool sampledNull = randomNumber < qNull;

                            // -----------------------------------------------------------------
                            // Null event
                            // -----------------------------------------------------------------
                            if (sampledNull) {
                                const float attenuation =
                                    1.0f - (worldHit.alphaGeom * surfel.opacity);

                                currentRayState.ray.origin =
                                    worldHit.hitPositionW + (currentRayState.ray.direction * 1e-5f);
                                currentRayState.ray.normal = orientedNormal;
                                currentRayState.pathThroughput *= (1.0f / qNull);
                                currentRayState.transmission *= attenuation;
                                currentRayState.traversalIndex = currentRayState.traversalIndex + 1u;

                                continue;
                            }

                            // -----------------------------------------------------------------
                            // Real surfel hit
                            // -----------------------------------------------------------------
                            float3 sampledOutgoingDirectionWorld{0.0f, 0.0f, 0.0f};
                            float uniformHemispherePdf = 1.0f / (2.0f * M_PIf);

                            sampleUniformHemisphereAroundNormal(
                                rng,
                                orientedNormal,
                                sampledOutgoingDirectionWorld,
                                uniformHemispherePdf);

                            // If possible then use light sample ray

                            const PointCloudSurfaceRecord currentSurface =
                                makePointCloudSurfaceRecord(worldHit, currentRayState, scene);

                            const float alpha = worldHit.alphaGeom * surfel.opacity;
                            const float3 surfelBsdf =
                                surfel.alpha_r * surfel.albedo * M_1_PIf;

                            const float cosineTheta = sycl::fmax(
                                0.0f,
                                dot(sampledOutgoingDirectionWorld, orientedNormal));

                            const float3 throughputMultiplier =
                                ((alpha / qReflect) * (surfelBsdf * cosineTheta)) /
                                uniformHemispherePdf;

                            float segmentGeometryFromStoredVertex = 1.0f;
                            float segmentAreaPdfFromStoredVertex = 1.0f;

                            if (hasPendingState) {
                                const PendingCameraSegment previousCameraSegment = pendingCameraSegment;
                                const PendingAdjointStageX previousAdjointStage = pendingAdjointStage;

                                const bool isCameraAttachedSecondHit =
                                    previousAdjointStage.valid &&
                                    previousAdjointStage.useImplicitRayHitJacobian &&
                                    currentRayState.bounceIndex == 1u;

                                // -------------------------------------------------------------
                                // Measurement event
                                // -------------------------------------------------------------
                                if (previousCameraSegment.valid) {
                                    const float cosine =
                                        dot(sensor.camera.forward, currentRayState.ray.direction);

                                    MeasurementGradientEvent measurementEvent{};
                                    measurementEvent.xSurface = currentSurface;
                                    measurementEvent.transmission = currentRayState.transmission;
                                    measurementEvent.xPathThroughput =
                                        currentRayState.pathThroughput / qReflect * cosine;

                                    appendEventAtomic(
                                        intermediates.countMeasurementEvents,
                                        intermediates.measurementEvents,
                                        intermediates.maxMeasurementEventCount,
                                        measurementEvent);
                                }

                                // -------------------------------------------------------------
                                // Camera-attached two-point event
                                // -------------------------------------------------------------
                                if (isCameraAttachedSecondHit) {
                                    MeasurementGradientEventXY measurementTwoPointEvent{};
                                    measurementTwoPointEvent.xSurface = previousAdjointStage.current.surface;
                                    measurementTwoPointEvent.ySurface = currentSurface;
                                    measurementTwoPointEvent.xPathThroughput =
                                        previousAdjointStage.current.pathThroughput / qReflect *
                                        previousAdjointStage.current.cosineFromPrevious;
                                    measurementTwoPointEvent.transmissionPreviousSegment =
                                        previousAdjointStage.current.transmissionFromPrevious;
                                    measurementTwoPointEvent.transmission =
                                        currentRayState.transmission;

                                    appendEventAtomic(
                                        intermediates.countMeasurementTwoPointEvents,
                                        intermediates.measurementTwoPointEvents,
                                        intermediates.maxMeasurementTwoPointEventCount,
                                        measurementTwoPointEvent);
                                }

                                // -------------------------------------------------------------
                                // Camera-attached bridge event
                                // -------------------------------------------------------------
                                if (previousAdjointStage.valid &&
                                    previousAdjointStage.useImplicitRayHitJacobian) {
                                    CameraAttachedBridgeGradientEvent attachedBridgeEvent{};
                                    attachedBridgeEvent.xSurface = previousAdjointStage.current.surface;
                                    attachedBridgeEvent.ySurface = currentSurface;
                                    attachedBridgeEvent.xPathThroughput =
                                        previousAdjointStage.current.pathThroughput *
                                        previousAdjointStage.current.cosineFromPrevious / qReflect;
                                    attachedBridgeEvent.transmissionPreviousSegment =
                                        previousAdjointStage.current.transmissionFromPrevious;
                                    attachedBridgeEvent.geometryPreviousSegment =
                                        previousAdjointStage.current.geometryFromPrevious;
                                    attachedBridgeEvent.transmission =
                                        currentRayState.transmission;

                                    appendEventAtomic(
                                        intermediates.countAttachedBridgeEvents,
                                        intermediates.cameraAttachedBridgeEvents,
                                        intermediates.maxCameraAttachedEvents,
                                        attachedBridgeEvent);
                                }

                                // -------------------------------------------------------------
                                // Recursive bridge event
                                // previous = X, current = Y, live hit = Z
                                // -------------------------------------------------------------
                                if (previousAdjointStage.valid &&
                                    !previousAdjointStage.useImplicitRayHitJacobian) {
                                    RecursiveBridgeGradientEvent recursiveBridgeEvent{};
                                    recursiveBridgeEvent.xSurface = previousAdjointStage.current.surface;
                                    recursiveBridgeEvent.ySurface = currentSurface;

                                    recursiveBridgeEvent.xPathThroughput =
                                        previousAdjointStage.previous.pathThroughput *
                                        previousAdjointStage.previous.bsdf *
                                        previousAdjointStage.current.transmissionFromPrevious *
                                        previousAdjointStage.current.geometryFromPrevious /
                                        (previousAdjointStage.current.areaPdfFromPrevious *
                                            qReflect * qReflect);

                                    recursiveBridgeEvent.transmissionPreviousSegment =
                                        previousAdjointStage.current.transmissionFromPrevious;
                                    recursiveBridgeEvent.geometryPreviousSegment =
                                        previousAdjointStage.current.geometryFromPrevious;
                                    recursiveBridgeEvent.transmission =
                                        currentRayState.transmission;

                                    appendEventAtomic(
                                        intermediates.countRecursiveBridgeEvents,
                                        intermediates.recursiveBridgeEvents,
                                        intermediates.maxRecursiveBridgeEvent,
                                        recursiveBridgeEvent);
                                }

                                clearPendingCameraSegment(pendingCameraSegment);

                                // -------------------------------------------------------------
                                // Build segment metadata for stored-vertex -> currentSurface
                                // -------------------------------------------------------------
                                if (previousAdjointStage.valid) {
                                    const Point& storedSurfel =
                                        scene.points[previousAdjointStage.current.surface.primitiveIndex];

                                    const ReconstructedSurfelState storedState =
                                        reconstructSurfelState(
                                            storedSurfel,
                                            previousAdjointStage.current.surface);

                                    const ReconstructedSurfelState liveState =
                                        reconstructSurfelState(
                                            surfel,
                                            currentSurface);

                                    segmentGeometryFromStoredVertex = computeGeometricTermValue(
                                        storedState.position,
                                        liveState.position,
                                        storedState.orientedNormal,
                                        liveState.orientedNormal);

                                    segmentAreaPdfFromStoredVertex =
                                        computeSegmentAreaPdfFromUniformHemisphere(
                                            storedState,
                                            liveState,
                                            uniformHemispherePdf);
                                }
                                else {
                                    segmentGeometryFromStoredVertex = 1.0f;
                                    segmentAreaPdfFromStoredVertex = 1.0f;
                                }

                                // -------------------------------------------------------------
                                // Push live surface into rolling adjoint history
                                // previous = old current, current = live hit
                                // -------------------------------------------------------------
                                const float cosineFromPrevious =
                                    previousCameraSegment.valid
                                        ? dot(sensor.camera.forward, currentRayState.ray.direction)
                                        : 0.0f;

                                const PendingAdjointVertex newCurrentVertex =
                                    makePendingAdjointVertex(
                                        currentSurface,
                                        currentRayState.bounceIndex,
                                        currentRayState.pathThroughput / qReflect,
                                        currentRayState.transmission,
                                        segmentGeometryFromStoredVertex,
                                        segmentAreaPdfFromStoredVertex,
                                        alpha * surfelBsdf,
                                        cosineFromPrevious);

                                pushPendingAdjointVertex(
                                    pendingAdjointStage,
                                    currentRayState.pathId,
                                    currentRayState.pixelIndex,
                                    previousCameraSegment.valid,
                                    newCurrentVertex);
                            }

                            // -----------------------------------------------------------------
                            // Spawn next ray
                            // -----------------------------------------------------------------
                            nextRayState.ray.origin =
                                worldHit.hitPositionW + (orientedNormal * 1e-5f);
                            nextRayState.ray.direction = sampledOutgoingDirectionWorld;
                            nextRayState.ray.normal = orientedNormal;
                            nextRayState.bounceIndex = currentRayState.bounceIndex + 1u;
                            nextRayState.pixelIndex = currentRayState.pixelIndex;
                            nextRayState.pathId = currentRayState.pathId;
                            nextRayState.pathThroughput =
                                currentRayState.pathThroughput *
                                throughputMultiplier *
                                currentRayState.transmission;
                            nextRayState.traversalIndex =
                                currentRayState.traversalIndex + 1u;
                            nextRayState.transmission = 1.0f;

                            if (applyRussianRoulette(
                                rng,
                                nextRayState.bounceIndex,
                                nextRayState.pathThroughput,
                                settings.russianRouletteStart)) {
                                shouldEnqueueNextRayState = true;
                            }
                            else {
                                clearPendingCameraSegment(pendingCameraSegment);
                                clearPendingAdjointStageX(pendingAdjointStage);
                            }

                            break;
                        }

                        clearPendingCameraSegment(pendingCameraSegment);
                        clearPendingAdjointStageX(pendingAdjointStage);
                        break;
                    }

                    if (hasPendingState) {
                        if (pendingCameraSegment.valid) {
                            intermediates.pendingCameraSegments[pathId] = pendingCameraSegment;
                        }
                        else {
                            clearPendingCameraSegment(
                                intermediates.pendingCameraSegments[pathId]);
                        }

                        if (pendingAdjointStage.valid) {
                            intermediates.pendingStageX[pathId] = pendingAdjointStage;
                        }
                        else {
                            clearPendingAdjointStageX(
                                intermediates.pendingStageX[pathId]);
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
                        if (outIndex < intermediates.maxRayQueueCapacity) {
                            intermediates.extensionRaysA[outIndex] = nextRayState;
                        }
                    }
                });
        }).wait();
    }

    static void measurementGradientEvent(
        RenderPackage& pkg,
        uint32_t cameraIndex,
        uint32_t measurementEventCount,
        uint32_t baseOffset) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& settings = pkg.settings;
        const auto& photonMap = pkg.intermediates.map;
        auto& sensor = pkg.sensors[cameraIndex];
        MeasurementGradientEvent* measurementEvents = pkg.intermediates.measurementEvents;
        SurfelGradientRecord* gradientRecords = pkg.intermediates.gradientRecords;

        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;

        queue.submit([&](sycl::handler& commandGroupHandler) {
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

                    const Point& surfelX =
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
                    struct OccluderDerivatives {
                        float3 derivative{0.0f};
                        uint32_t primitiveIndex = UINT32_MAX;
                    };
                    OccluderDerivatives occluderDerivatives[kMaxSplatEventsPerRay];
                    uint32_t storedOccluderCount = 0u;

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
                            const auto& instance = scene.instances[worldHit.instanceIndex];
                            if (instance.geometryType != GeometryType::PointCloud) {
                                break;
                            }

                            const Point& occluderSurfel =
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

                            const float3 dAlphaEffectiveDspi =
                                occluderSurfel.opacity * (
                                    dAlphaGeomDu * dUiDspi +
                                    dAlphaGeomDv * dViDspi);

                            if (storedOccluderCount < kMaxSplatEventsPerRay) {
                                occluderDerivatives[storedOccluderCount].derivative =
                                    dAlphaEffectiveDspi * (1.0f / oneMinusAlpha);
                                occluderDerivatives[storedOccluderCount].primitiveIndex =
                                    worldHit.primitiveIndex;
                                storedOccluderCount++;
                            }
                        }
                    }

                    const float3 pathWeight = eventRecord.xPathThroughput;
                    const float scalarWeight = dot(pathWeight, outgoingRadianceX);


                    // L_surfel(X, omega_x->c), without alpha_X
                    const float3 outgoingRadianceXNoAlpha = evaluateSurfelRadianceWithoutLocalAlpha(
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

                    const float scalarWeightNoAlpha =
                        dot(eventRecord.xPathThroughput, outgoingRadianceXNoAlpha);

                    const float3 positionGradient =
                        eventRecord.transmission *
                        dAlphaEffectiveDPosition *
                        scalarWeightNoAlpha *
                        invSpp;

                    SurfelGradientRecord gradientRecord{};
                    gradientRecord.primitiveIndex = eventRecord.xSurface.primitiveIndex;
                    gradientRecord.gradPositionX = positionGradient.x();
                    gradientRecord.gradPositionY = positionGradient.y();
                    gradientRecord.gradPositionZ = positionGradient.z();
                    gradientRecords[recordIndex] = gradientRecord;

                    const float occluderScale =
                        -transmittance * scalarWeight * invSpp;

                    for (uint32_t occluderIndex = 0u;
                         occluderIndex < storedOccluderCount;
                         ++occluderIndex) {
                        const uint32_t occluderRecordIndex = eventRecordBase + 1u + occluderIndex;

                        const OccluderDerivatives& occluderDerivative =
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

    static void measurementGradientEventXY(RenderPackage& pkg,
                                           uint32_t onePointEventCount,
                                           uint32_t baseOffset) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& settings = pkg.settings;
        const auto& photonMap = pkg.intermediates.map;
        MeasurementGradientEventXY* measurementXYEvent =
            pkg.intermediates.measurementTwoPointEvents;
        SurfelGradientRecord* gradientRecords = pkg.intermediates.gradientRecords;

        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;
        const float qNullInv = 1.0f / settings.sampling.qNull;

        queue.submit([&](sycl::handler& commandGroupHandler) {
            commandGroupHandler.parallel_for<class firstHitGradientEventTag>(
                sycl::range<1>(onePointEventCount),
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

                    const MeasurementGradientEventXY eventRecord =
                        measurementXYEvent[eventIndex];

                    const uint32_t xPrimitiveIndex = eventRecord.xSurface.primitiveIndex;
                    const uint32_t yPrimitiveIndex = eventRecord.ySurface.primitiveIndex;

                    const Point& surfelX = scene.points[xPrimitiveIndex];
                    const Point& surfelY = scene.points[yPrimitiveIndex];

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

                    const float3 dGeometricTermDx = computeGeometricTermGradientWrtStartpoint(
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
                            auto& instance = scene.instances[worldHit.instanceIndex];
                            if (instance.geometryType != GeometryType::PointCloud) {
                                ray.origin = worldHit.hitPositionW + (ray.direction * 1e-4f);
                                continue;
                            }

                            const Point& occluderSurfel = scene.points[worldHit.primitiveIndex];
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
                            const float uOcc = uv.x();
                            const float vOcc = uv.y();

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

                            const float3 dUiDy =
                                (1.0f - lambdaOccluder) * (
                                    localBasisU -
                                    occluderNormal * (dot(dxy, localBasisU) * inverseDenominator));

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

                            const float radiusSquared = uOcc * uOcc + vOcc * vOcc;
                            const float oneMinusRadiusSquared = 1.0f - radiusSquared;
                            if (oneMinusRadiusSquared <= 1e-8f) {
                                continue;
                            }

                            const float betaScale = 4.0f * sycl::exp(occluderSurfel.beta);

                            const float dAlphaGeomDu =
                                -2.0f * betaScale * uOcc * alphaGeomOccluder / oneMinusRadiusSquared;
                            const float dAlphaGeomDv =
                                -2.0f * betaScale * vOcc * alphaGeomOccluder / oneMinusRadiusSquared;

                            const float3 dAlphaEffectiveDy =
                                occluderSurfel.opacity * (
                                    dAlphaGeomDu * dUiDy +
                                    dAlphaGeomDv * dViDy);

                            const float3 dAlphaEffectiveDspi =
                                occluderSurfel.opacity * (
                                    dAlphaGeomDu * dUiDspi +
                                    dAlphaGeomDv * dViDspi);

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

                    float3 gradientWrtHitPositionX =
                        scalarWeightWithoutTauAndGeometric *
                        transmittance * dGeometricTermDx;

                    const float3x3 hitPointJacobianX = planeHitPointIntersectionJacobian(
                        eventRecord.xSurface.incomingDirection,
                        xState.orientedNormal);
                    gradientWrtHitPositionX = transpose(hitPointJacobianX) * gradientWrtHitPositionX;

                    const float3 xContribution = gradientWrtHitPositionX * invSpp;

                    SurfelGradientRecord xRecord{};
                    xRecord.primitiveIndex = xPrimitiveIndex;
                    xRecord.gradPositionX = xContribution.x();
                    xRecord.gradPositionY = xContribution.y();
                    xRecord.gradPositionZ = xContribution.z();
                    gradientRecords[recordIndex] = xRecord;

                    const float geometricTermXY = computeGeometricTermValue(
                        xState.position,
                        yState.position,
                        xState.orientedNormal,
                        yState.orientedNormal);

                    const float occluderScale =
                        -transmittance *
                        geometricTermXY *
                        scalarWeightWithoutTauAndGeometric *
                        invSpp;

                    for (uint32_t occluderIndex = 0u;
                         occluderIndex < storedOccluderCount;
                         ++occluderIndex) {
                        const uint32_t occluderRecordIndex =
                            eventRecordBase + 1u + occluderIndex;

                        const OccluderDerivative& occluderDerivative =
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

    static void cameraAttachedBridgeEvent(
        RenderPackage& pkg,
        uint32_t twoPointEventCount,
        uint32_t baseOffset) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& settings = pkg.settings;
        const auto& photonMap = pkg.intermediates.map;
        CameraAttachedBridgeGradientEvent* cameraAttachedEvents =
            pkg.intermediates.cameraAttachedBridgeEvents;
        SurfelGradientRecord* gradientRecords = pkg.intermediates.gradientRecords;

        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;
        const float qNullInv = 1.0f / settings.sampling.qNull;
        auto& sensor = pkg.sensors.front();

        // Camera-attached first bridge X -> Y:
        // only differentiate the endpoint Y on the XY segment.
        // The startpoint X is handled separately by the camera-side measurement derivative.

        queue.submit([&](sycl::handler& commandGroupHandler) {
            commandGroupHandler.parallel_for<class cameraAttachedBridgeEventTag>(
                sycl::range<1>(twoPointEventCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex = globalId[0];

                    static constexpr uint32_t recordsPerEvent = 1u;
                    const uint32_t eventRecordBase = baseOffset + recordsPerEvent * eventIndex;
                    const uint32_t yRecordIndex = eventRecordBase + 0u;
                    const CameraAttachedBridgeGradientEvent eventRecord =
                        cameraAttachedEvents[eventIndex];

                    const uint32_t xPrimitiveIndex = eventRecord.xSurface.primitiveIndex;
                    const uint32_t yPrimitiveIndex = eventRecord.ySurface.primitiveIndex;

                    const Point& surfelX = scene.points[xPrimitiveIndex];
                    const Point& surfelY = scene.points[yPrimitiveIndex];

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

                    const float u = surfelY.scale.x();
                    const float v = surfelY.scale.y();
                    const float Juv = u * v;
                    const float PuvY = Juv * pAreaY;

                    const float alphaX = eventRecord.xSurface.alphaGeom * surfelX.opacity;
                    const float3 brdfX = surfelX.alpha_r * surfelX.albedo * M_1_PIf;

                    const float geometricTermXY = computeGeometricTermValue(
                        xState.position,
                        yState.position,
                        xState.orientedNormal,
                        yState.orientedNormal);

                    const float3 dGeometricTermDy = computeGeometricTermGradientWrtEndpoint(
                        xState.position,
                        yState.position,
                        xState.orientedNormal,
                        yState.orientedNormal);

                    // xPathThroughput already contains the 1/qReflect factor for the reflective event at X.
                    const float3 pathWeight =
                        eventRecord.xPathThroughput;

                    const float3 transportWithoutTauAndGeometric =
                        outgoingRadianceY * alphaX * brdfX;

                    float scalarWeightWithoutTauAndGeometric =
                        dot(pathWeight, transportWithoutTauAndGeometric) * Juv / PuvY;

                    float transmittance = 1.0f;
                    float3 accumulatedAlphaDerivativeOverOneMinusAlphaEndPoint = float3{0.0f};
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
                            auto& instance = scene.instances[worldHit.instanceIndex];
                            if (instance.geometryType != GeometryType::PointCloud) {
                                ray.origin = worldHit.hitPositionW + (ray.direction * 1e-4f);
                                continue;
                            }

                            const Point& occluderSurfel = scene.points[worldHit.primitiveIndex];
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
                            const float uOcc = uv.x();
                            const float vOcc = uv.y();

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

                            const float3 dUiDy =
                                (1.0f - lambdaOccluder) * (
                                    localBasisU -
                                    occluderNormal * (dot(dxy, localBasisU) * inverseDenominator));

                            const float3 dViDy =
                                (1.0f - lambdaOccluder) * (
                                    localBasisV -
                                    occluderNormal * (dot(dxy, localBasisV) * inverseDenominator));


                            const float radiusSquared = uOcc * uOcc + vOcc * vOcc;
                            const float oneMinusRadiusSquared = 1.0f - radiusSquared;
                            if (oneMinusRadiusSquared <= 1e-8f) {
                                continue;
                            }

                            const float betaScale = 4.0f * sycl::exp(occluderSurfel.beta);

                            const float dAlphaGeomDu =
                                -2.0f * betaScale * uOcc * alphaGeomOccluder / oneMinusRadiusSquared;
                            const float dAlphaGeomDv =
                                -2.0f * betaScale * vOcc * alphaGeomOccluder / oneMinusRadiusSquared;

                            const float3 dAlphaEffectiveDy =
                                occluderSurfel.opacity * (
                                    dAlphaGeomDu * dUiDy +
                                    dAlphaGeomDv * dViDy);

                            accumulatedAlphaDerivativeOverOneMinusAlphaEndPoint +=
                                dAlphaEffectiveDy * (1.0f / oneMinusAlpha);

                            scalarWeightWithoutTauAndGeometric *= qNullInv;
                        }
                    }

                    const float3 dTransmittanceDy =
                        -transmittance * accumulatedAlphaDerivativeOverOneMinusAlphaEndPoint;

                    float3 gradientWrtHitPositionY =
                        scalarWeightWithoutTauAndGeometric *
                        (geometricTermXY * dTransmittanceDy + transmittance * dGeometricTermDy);

                    const float3 yContribution = gradientWrtHitPositionY * invSpp;

                    SurfelGradientRecord yRecord{};
                    yRecord.primitiveIndex = yPrimitiveIndex;
                    yRecord.gradPositionX = yContribution.x();
                    yRecord.gradPositionY = yContribution.y();
                    yRecord.gradPositionZ = yContribution.z();
                    gradientRecords[yRecordIndex] = yRecord;
                });
        }).wait();
    }

    static void recursiveBridgeEvent(
        RenderPackage& pkg,
        uint32_t recursiveBridgeEventCount,
        uint32_t baseOffset) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& settings = pkg.settings;
        const auto& photonMap = pkg.intermediates.map;
        RecursiveBridgeGradientEvent* recursiveBridgeEvents =
            pkg.intermediates.recursiveBridgeEvents;
        SurfelGradientRecord* gradientRecords = pkg.intermediates.gradientRecords;

        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;
        const float qNullInv = 1.0f / settings.sampling.qNull;

        queue.submit([&](sycl::handler& commandGroupHandler) {
            commandGroupHandler.parallel_for<class recursiveBridgeEventTag>(
                sycl::range<1>(recursiveBridgeEventCount),
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

                    const RecursiveBridgeGradientEvent eventRecord =
                        recursiveBridgeEvents[eventIndex];

                    const uint32_t xPrimitiveIndex = eventRecord.xSurface.primitiveIndex;
                    const uint32_t yPrimitiveIndex = eventRecord.ySurface.primitiveIndex;

                    const Point& surfelX = scene.points[xPrimitiveIndex];
                    const Point& surfelY = scene.points[yPrimitiveIndex];

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

                    const float u = surfelY.scale.x();
                    const float v = surfelY.scale.y();
                    const float Juv = u * v;
                    const float PuvY = Juv * pAreaY;

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

                    // Prefix adjoint weight at X times incoming-segment transmittance tau(prev, X).
                    const float3 pathWeight =
                        eventRecord.xPathThroughput;
                    // Local transport at X, excluding tau(X,Y) and G(X,Y).
                    const float3 transportWithoutTauAndGeometric =
                        outgoingRadianceY * alphaX * brdfX;
                    float scalarWeightWithoutTauAndGeometric =
                        dot(pathWeight, transportWithoutTauAndGeometric) * Juv / PuvY;
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
                            auto& instance = scene.instances[worldHit.instanceIndex];
                            if (instance.geometryType != GeometryType::PointCloud) {
                                ray.origin = worldHit.hitPositionW + (ray.direction * 1e-4f);
                                continue;
                            }

                            const Point& occluderSurfel = scene.points[worldHit.primitiveIndex];
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
                            const float uOcc = uv.x();
                            const float vOcc = uv.y();

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

                            const float radiusSquared = uOcc * uOcc + vOcc * vOcc;
                            const float oneMinusRadiusSquared = 1.0f - radiusSquared;
                            if (oneMinusRadiusSquared <= 1e-8f) {
                                continue;
                            }

                            const float betaScale = 4.0f * sycl::exp(occluderSurfel.beta);

                            const float dAlphaGeomDu =
                                -2.0f * betaScale * uOcc * alphaGeomOccluder / oneMinusRadiusSquared;
                            const float dAlphaGeomDv =
                                -2.0f * betaScale * vOcc * alphaGeomOccluder / oneMinusRadiusSquared;

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

                    const float3 dTransmittanceDy =
                        -transmittance * accumulatedAlphaDerivativeOverOneMinusAlphaEndPoint;

                    const float3 gradientWrtXPosition =
                        scalarWeightWithoutTauAndGeometric *
                        (transmittance * dGeometricTermDx);

                    const float3 gradientWrtYPosition =
                        scalarWeightWithoutTauAndGeometric *
                        (geometricTermXY * dTransmittanceDy + transmittance * dGeometricTermDy);

                    const float3 xContribution = gradientWrtXPosition * invSpp;
                    const float3 yContribution = gradientWrtYPosition * invSpp;

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

                        const OccluderDerivative& occluderDerivative =
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
        RenderPackage& pkg,
        uint32_t gradientRecordCount) {
        auto& queue = pkg.queue;
        auto gradients = pkg.gradients;
        SurfelGradientRecord* gradientRecords = pkg.intermediates.gradientRecords;

        queue.submit([&](sycl::handler& commandGroupHandler) {
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
        RenderPackage& pkg,
        uint32_t measurementEventCount,
        uint32_t measurementTwoPointEventCount,
        uint32_t cameraAttachedBridgeEventCount,
        uint32_t threePointEventCount,
        uint32_t cameraIndex) {
        const GradientRecordRanges ranges = makeGradientRecordRanges(
            measurementEventCount,
            measurementTwoPointEventCount,
            cameraAttachedBridgeEventCount,
            threePointEventCount);

        if (ranges.totalCount > pkg.intermediates.maxGradientRecordCount) {
            throw std::runtime_error("gradient record scratch buffer too small");
        }

        Log::PA_DEBUG(
            "Event counts: measurement={}, measurementTwoPoint={}, cameraAttachedBridge={}, recursiveBridge={}",
            measurementEventCount,
            measurementTwoPointEventCount,
            cameraAttachedBridgeEventCount,
            threePointEventCount);

        if (measurementEventCount > 0) {
            ScopedTimer timer("measurementGradientEvent", spdlog::level::debug);
            measurementGradientEvent(
                pkg,
                cameraIndex,
                measurementEventCount,
                ranges.measurementOffset);
        }

        if (measurementTwoPointEventCount > 0) {
            ScopedTimer timer("measurementGradientEventXY", spdlog::level::debug);
            measurementGradientEventXY(
                pkg,
                measurementTwoPointEventCount,
                ranges.measurementTwoPointOffset);
        }

        if (cameraAttachedBridgeEventCount > 0) {
            ScopedTimer timer("twoPointGradientEvent", spdlog::level::debug);
            cameraAttachedBridgeEvent(
                pkg,
                cameraAttachedBridgeEventCount,
                ranges.cameraAttachedBridgeOffset);
        }

        if (threePointEventCount > 0) {
            ScopedTimer timer("recursiveBridgeEvent", spdlog::level::debug);
            recursiveBridgeEvent(
                pkg,
                threePointEventCount,
                ranges.recursiveBridgeOffset);
        }

        if (ranges.totalCount > 0) {
            ScopedTimer timer("reduceSurfelGradientRecords", spdlog::level::debug);
            reduceSurfelGradientRecords(
                pkg,
                ranges.totalCount);
        }
    }
}
