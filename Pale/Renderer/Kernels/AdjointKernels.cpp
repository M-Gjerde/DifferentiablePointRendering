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
    SYCL_EXTERNAL inline float3 evaluateOutgoingRadianceWithLocalAlpha(
        const Point &surfel,
        const PointCloudSurfaceRecord &surfaceRecord,
        const ReconstructedSurfelState &reconstructedState,
        const DeviceSurfacePhotonMapGrid &photonMap,
        const GPUSceneBuffers &scene,
        uint32_t numShadowRays,
        rng::Xorshift128 &rng128) {
        const float alpha = surfaceRecord.alphaGeom * surfel.opacity;

        const float3 indirectIrradiance = gatherDiffuseIrradianceAtPoint(
            reconstructedState.position,
            reconstructedState.orientedNormal,
            photonMap);

        const float3 indirectRadiance =
                indirectIrradiance *
                (surfel.alpha_r * surfel.albedo * M_1_PIf) *
                alpha;

        const float3 directRadiance =
                estimateDirectLightAtDiffuseSurface(
                    scene,
                    reconstructedState.position,
                    reconstructedState.orientedNormal,
                    surfel.alpha_r * surfel.albedo,
                    numShadowRays,
                    rng128) * alpha;

        float3 emittedRadiance =
                surfel.albedo *
                (surfel.flux / (M_PIf * reconstructedState.areaWorld)) *
                alpha;

        if (surfel.flux > 0.0f && surfaceRecord.sideSign < 0) {
            emittedRadiance = float3{0.0f, 0.0f, 0.0f};
        }

        return emittedRadiance + directRadiance + indirectRadiance;
    }

    SYCL_EXTERNAL inline float3 evaluateOutgoingRadianceWithLocalAlphaNoEmitters(
        const Point &surfel,
        const PointCloudSurfaceRecord &surfaceRecord,
        const ReconstructedSurfelState &reconstructedState,
        const DeviceSurfacePhotonMapGrid &photonMap,
        const GPUSceneBuffers &scene,
        uint32_t numShadowRays,
        rng::Xorshift128 &rng128) {
        const float alpha = surfaceRecord.alphaGeom * surfel.opacity;

        const float3 indirectIrradiance = gatherDiffuseIrradianceAtPoint(
            reconstructedState.position,
            reconstructedState.orientedNormal,
            photonMap);

        const float3 indirectRadiance =
                indirectIrradiance *
                (surfel.alpha_r * surfel.albedo * M_1_PIf) *
                alpha;

        const float3 directRadiance =
                estimateDirectLightAtDiffuseSurface(
                    scene,
                    reconstructedState.position,
                    reconstructedState.orientedNormal,
                    surfel.alpha_r * surfel.albedo,
                    numShadowRays,
                    rng128) * alpha;


        return directRadiance + indirectRadiance;
    }

    SYCL_EXTERNAL inline float3 evaluateOutgoingRadianceWithoutLocalAlpha(
        const Point &surfel,
        const PointCloudSurfaceRecord &surfaceRecord,
        const ReconstructedSurfelState &reconstructedState,
        const DeviceSurfacePhotonMapGrid &photonMap,
        const GPUSceneBuffers &scene,
        uint32_t numShadowRays,
        rng::Xorshift128 &rng128) {
        const float3 indirectIrradiance = gatherDiffuseIrradianceAtPoint(
            reconstructedState.position,
            reconstructedState.orientedNormal,
            photonMap);

        const float3 indirectRadiance =
                indirectIrradiance *
                (surfel.alpha_r * surfel.albedo * M_1_PIf);

        const float3 directRadiance =
                estimateDirectLightAtDiffuseSurface(
                    scene,
                    reconstructedState.position,
                    reconstructedState.orientedNormal,
                    surfel.alpha_r * surfel.albedo,
                    numShadowRays,
                    rng128);

        float3 emittedRadiance =
                surfel.albedo *
                (surfel.flux / (M_PIf * reconstructedState.areaWorld));

        if (surfel.flux > 0.0f && surfaceRecord.sideSign < 0) {
            emittedRadiance = float3{0.0f, 0.0f, 0.0f};
        }

        return emittedRadiance + directRadiance + indirectRadiance;
    }

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
                    }
                });
        }).wait();
    }

    void launchAdjointIntersectKernel(RenderPackage &pkg, uint32_t spp, uint32_t activeRayCount, uint32_t cameraIndex) {
        auto &queue = pkg.queue;
        auto &settings = pkg.settings;
        auto &intermediates = pkg.intermediates;
        auto &scene = pkg.scene;
        auto &sensor = pkg.sensors[cameraIndex];

        queue.submit([&](sycl::handler &commandGroupHandler) {
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
                            SurfelIntersectMode::FirstHit);

                        if (!worldHit.hit) {
                            clearPendingCameraSegment(pendingCameraSegment);
                            clearPendingAdjointStageX(pendingAdjointStage);
                            break;
                        }

                        buildIntersectionNormal(scene, worldHit);

                        const InstanceRecord &instance = scene.instances[worldHit.instanceIndex];
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
                                    currentRayState.pathThroughput * throughputMultiplier * currentRayState.
                                    transmission;
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
                            const Point &surfel = scene.points[worldHit.primitiveIndex];

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
                                    MeasurementGradientEvent measurementEvent{};
                                    measurementEvent.xSurface = currentSurface;
                                    measurementEvent.transmission = currentRayState.transmission;
                                    measurementEvent.xPathThroughput =
                                            currentRayState.transmission * currentRayState.pathThroughput / qReflect;

                                    appendEventAtomic(
                                        intermediates.countMeasurementEvents,
                                        intermediates.measurementEvents,
                                        intermediates.maxMeasurementEventCount,
                                        measurementEvent);

                                    // Direct light samples:
                                    if (settings.enableAdjointDirectLight) {
                                        float invSampleCount =
                                                1.0f / static_cast<float>(settings.numAdjointPathShadowRays);
                                        for (uint32_t shadowRaySample = 0;
                                             shadowRaySample < settings.numAdjointPathShadowRays; shadowRaySample++) {
                                            const auto lightSample = sampleMeshAreaLight(scene, rng);
                                            if (lightSample.valid) {
                                                // Trace a ray to check if it is valid:
                                                const float3 lightVector =
                                                        lightSample.positionW - worldHit.hitPositionW;
                                                const float lightDistanceSquared = dot(lightVector, lightVector);
                                                if (lightDistanceSquared <= 1e-12f) {
                                                    continue;
                                                }

                                                const float lightDistance = sycl::sqrt(lightDistanceSquared);
                                                const float3 lightDirection = lightVector / lightDistance;

                                                constexpr float distanceEpsilon = 1e-4f;
                                                constexpr uint32_t maxShadowTraversals = 256u;
                                                const float targetDistance = lightDistance;
                                                Ray shadowRay{};
                                                shadowRay.origin =
                                                        worldHit.hitPositionW + worldHit.geometricNormalW *
                                                        distanceEpsilon;
                                                shadowRay.direction = lightDirection;
                                                shadowRay.normal = worldHit.geometricNormalW;
                                                bool blockedByOpaqueGeometry = false;
                                                float transmission = 1.0f;
                                                for (uint32_t traversalIndex = 0u;
                                                     traversalIndex < maxShadowTraversals;
                                                     ++traversalIndex) {
                                                    (void) traversalIndex;
                                                    WorldHit shadowHit{};
                                                    intersectScene(
                                                        shadowRay,
                                                        &shadowHit,
                                                        scene,
                                                        SurfelIntersectMode::FirstHit);
                                                    if (!shadowHit.hit) {
                                                        break;
                                                    }
                                                    const float3 hitVector =
                                                            shadowHit.hitPositionW - worldHit.hitPositionW;
                                                    const float hitDistance = sycl::sqrt(dot(hitVector, hitVector));
                                                    // Nothing before the sampled light point anymore.
                                                    if (hitDistance >= targetDistance - distanceEpsilon) {
                                                        break;
                                                    }
                                                    const InstanceRecord &hitInstance =
                                                            scene.instances[shadowHit.instanceIndex];
                                                    // Meshes are treated as hard blockers for direct-light visibility.
                                                    if (hitInstance.geometryType == GeometryType::Mesh) {
                                                        blockedByOpaqueGeometry = true;
                                                        break;
                                                    }
                                                    // Point-cloud surfels are semi-transparent attenuators.
                                                    // Do not reject the sample here; just continue marching.
                                                    if (hitInstance.geometryType == GeometryType::PointCloud) {
                                                        shadowRay.origin =
                                                                shadowHit.hitPositionW + shadowRay.direction *
                                                                distanceEpsilon;
                                                        transmission *= (
                                                            1.0f - shadowHit.alphaGeom * scene.points[shadowHit.
                                                                primitiveIndex].opacity);
                                                        continue;
                                                    }
                                                    // Any other geometry type: conservatively treat as blocker.
                                                    blockedByOpaqueGeometry = true;
                                                    break;
                                                }
                                                if (blockedByOpaqueGeometry) {
                                                    continue;
                                                }


                                                MeasurementGradientEventXY measurementTwoPointEvent{};
                                                measurementTwoPointEvent.xSurface = currentSurface;
                                                measurementTwoPointEvent.ySurface = lightSample.surface;
                                                measurementTwoPointEvent.ySurface.pathId = currentRayState.pathId;
                                                measurementTwoPointEvent.xPathThroughput =
                                                        currentRayState.transmission * currentRayState.pathThroughput /
                                                        (lightSample.pdfArea * qReflect) * invSampleCount;
                                                measurementTwoPointEvent.transmission = transmission;
                                                measurementTwoPointEvent.directLightRadiance =
                                                        lightSample.flux / (M_PIf * lightSample.totalAreaWorld);
                                                measurementTwoPointEvent.isDirectLightSample = true;

                                                appendEventAtomic(
                                                    intermediates.countMeasurementTwoPointEvents,
                                                    intermediates.measurementTwoPointEvents,
                                                    intermediates.maxMeasurementTwoPointEventCount,
                                                    measurementTwoPointEvent);
                                            }
                                        }
                                    }
                                }

                                // -------------------------------------------------------------
                                // Camera-attached two-point event
                                // -------------------------------------------------------------
                                if (previousAdjointStage.valid) {
                                    const Point &storedSurfel =
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

                                if (isCameraAttachedSecondHit) {
                                    MeasurementGradientEventXY measurementTwoPointEvent{};
                                    measurementTwoPointEvent.xSurface = previousAdjointStage.current.surface;
                                    measurementTwoPointEvent.ySurface = currentSurface;
                                    measurementTwoPointEvent.xPathThroughput =
                                            previousAdjointStage.current.transmissionFromPrevious *
                                            previousAdjointStage.current.pathThroughput /
                                            (segmentAreaPdfFromStoredVertex * qReflect);

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
                                             qReflect);

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
                                    const Point &storedSurfel =
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
                                } else {
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
                            } else {
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
                        } else {
                            clearPendingCameraSegment(
                                intermediates.pendingCameraSegments[pathId]);
                        }

                        if (pendingAdjointStage.valid) {
                            intermediates.pendingStageX[pathId] = pendingAdjointStage;
                        } else {
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

    void launchAdjointDirectLightKernel(
        RenderPackage &pkg,
        uint32_t spp,
        uint32_t activeQueryCount,
        uint32_t cameraIndex) {
        auto &queue = pkg.queue;
        auto &settings = pkg.settings;
        auto &intermediates = pkg.intermediates;
        auto &scene = pkg.scene;
        (void) cameraIndex;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            const uint64_t renderSeed = settings.random.seed;

            commandGroupHandler.parallel_for<class launchAdjointDirectLightKernelTag>(
                sycl::range<1>(activeQueryCount),
                [=](sycl::id<1> global_id) {
                    const uint32_t queryIndex = global_id[0];
                    const DirectLightQuery query = intermediates.directLightQueries[queryIndex];

                    if (query.lightPdfArea <= 1e-20f) {
                        return;
                    }

                    const Point &surfel =
                            scene.points[query.surface.primitiveIndex];

                    const ReconstructedSurfelState surfaceState =
                            reconstructSurfelState(surfel, query.surface);

                    const float3 lightVector =
                            query.lightPositionWorld - surfaceState.position;
                    const float lightDistanceSquared = dot(lightVector, lightVector);
                    if (lightDistanceSquared <= 1e-12f) {
                        return;
                    }

                    const float lightDistance = sycl::sqrt(lightDistanceSquared);
                    const float3 lightDirection = lightVector / lightDistance;

                    // Early reject if the geometric term would be zero.
                    const float geometricTerm = computeGeometricTermValue(
                        surfaceState.position,
                        query.lightPositionWorld,
                        surfaceState.orientedNormal,
                        query.lightNormalWorld);

                    if (geometricTerm <= 0.0f) {
                        return;
                    }

                    const uint64_t stepSeed = rng::makeSeed(
                        renderSeed,
                        query.pathId,
                        spp,
                        rng::kStreamTraversal,
                        queryIndex + 0x9e3779b9u);

                    rng::Xorshift128 rng(stepSeed);

                    // March toward the sampled light point.
                    // Point-cloud hits are NOT treated as hard blockers here:
                    // they contribute attenuation later in the gradient kernel.
                    // Mesh hits before the light ARE treated as blockers.
                    constexpr float distanceEpsilon = 1e-4f;
                    constexpr uint32_t maxShadowTraversals = 256u;
                    const float targetDistance = lightDistance;
                    Ray shadowRay{};
                    shadowRay.origin =
                            surfaceState.position + surfaceState.orientedNormal * distanceEpsilon;
                    shadowRay.direction = lightDirection;
                    shadowRay.normal = surfaceState.orientedNormal;
                    bool blockedByOpaqueGeometry = false;
                    for (uint32_t traversalIndex = 0u;
                         traversalIndex < maxShadowTraversals;
                         ++traversalIndex) {
                        (void) traversalIndex;
                        WorldHit shadowHit{};
                        intersectScene(
                            shadowRay,
                            &shadowHit,
                            scene,
                            SurfelIntersectMode::FirstHit);
                        if (!shadowHit.hit) {
                            break;
                        }
                        const float3 hitVector = shadowHit.hitPositionW - surfaceState.position;
                        const float hitDistance = sycl::sqrt(dot(hitVector, hitVector));
                        // Nothing before the sampled light point anymore.
                        if (hitDistance >= targetDistance - distanceEpsilon) {
                            break;
                        }
                        const InstanceRecord &hitInstance =
                                scene.instances[shadowHit.instanceIndex];
                        // Meshes are treated as hard blockers for direct-light visibility.
                        if (hitInstance.geometryType == GeometryType::Mesh) {
                            blockedByOpaqueGeometry = true;
                            break;
                        }
                        // Point-cloud surfels are semi-transparent attenuators.
                        // Do not reject the sample here; just continue marching.
                        if (hitInstance.geometryType == GeometryType::PointCloud) {
                            shadowRay.origin =
                                    shadowHit.hitPositionW + shadowRay.direction * distanceEpsilon;
                            continue;
                        }
                        // Any other geometry type: conservatively treat as blocker.
                        blockedByOpaqueGeometry = true;
                        break;
                    }
                    if (blockedByOpaqueGeometry) {
                        return;
                    }

                    DirectLightGradientEvent event{};
                    event.surface = query.surface;
                    event.lightPositionWorld = query.lightPositionWorld;
                    event.lightNormalWorld = query.lightNormalWorld;
                    event.lightRadiance = query.lightRadiance;
                    event.lightPdfArea = query.lightPdfArea;
                    event.useImplicitRayHitJacobian = query.bounceIndex == 0u;
                    // Prefix weight only. The direct-light kernel will rebuild the
                    // local light transport and differentiate tau(x,l) * G(x,l).
                    event.transmissionToSurface = query.transmissionToSurface;
                    event.visibility = 1.0f;
                    event.xPathThroughput =
                            query.adjointWeight * query.transmissionToSurface;
                    event.localBsdf = query.localBsdf;
                    appendEventAtomic(
                        intermediates.countDirectLightEvents,
                        intermediates.directLightEvents,
                        intermediates.maxDirectLightEventCount,
                        event);
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

                    const uint64_t directLightSeed = rng::makeSeed(
                        settings.random.seed,
                        eventRecord.xSurface.pathId,
                        cameraIndex,
                        rng::kStreamDirectLight,
                        eventIndex);

                    rng::Xorshift128 directLightRng(directLightSeed);

                    const float3 outgoingRadianceX =
                            evaluateOutgoingRadianceWithLocalAlpha(
                                surfelX,
                                eventRecord.xSurface,
                                xState,
                                photonMap,
                                scene,
                                settings.numAdjointShadowRays,
                                directLightRng);

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
                        float3 gradPosition{0.0f};
                        float gradScaleU = 0.0f;
                        float gradScaleV = 0.0f;
                        float gradEta = 0.0f;
                        float gradBeta = 0.0f;
                        float3 gradTangentU{0.0f};
                        float3 gradTangentV{0.0f};
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

                            const float alphaGeomOccluder = worldHit.alphaGeom;
                            const float alphaEffective =
                                    occluderSurfel.opacity * alphaGeomOccluder;
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

                            const float scaleU = occluderSurfel.scale.x();
                            const float scaleV = occluderSurfel.scale.y();
                            if (scaleU <= 1e-12f || scaleV <= 1e-12f) {
                                continue;
                            }

                            const float3 tangentU = occluderSurfel.tanU;
                            const float3 tangentV = occluderSurfel.tanV;
                            const float3 localBasisU = tangentU / scaleU;
                            const float3 localBasisV = tangentV / scaleV;

                            const float denominator = dot(occluderNormal, dxy);
                            if (sycl::fabs(denominator) <= 1e-8f) {
                                continue;
                            }

                            const float inverseDenominator = 1.0f / denominator;

                            // -------------------------------------------------------------
                            // Translation derivative (existing)
                            // -------------------------------------------------------------
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

                            // -------------------------------------------------------------
                            // Scale derivatives (fixed ray, fixed plane hit point)
                            // -------------------------------------------------------------
                            const float dAlphaEffectiveDScaleU =
                                    (2.0f * betaScaleOcc * uOcc * uOcc * alphaEffective) /
                                    (scaleU * oneMinusRadiusSquaredOcc);

                            const float dAlphaEffectiveDScaleV =
                                    (2.0f * betaScaleOcc * vOcc * vOcc * alphaEffective) /
                                    (scaleV * oneMinusRadiusSquaredOcc);

                            // -------------------------------------------------------------
                            // Eta / opacity derivative
                            // -------------------------------------------------------------
                            const float dAlphaEffectiveDEta =
                                    alphaGeomOccluder;

                            // -------------------------------------------------------------
                            // Beta derivative
                            // -------------------------------------------------------------
                            const float dAlphaEffectiveDBeta =
                                    betaScaleOcc *
                                    sycl::log(oneMinusRadiusSquaredOcc) *
                                    alphaEffective;

                            // -------------------------------------------------------------
                            // Rotation derivative for fixed ray line
                            // -------------------------------------------------------------
                            float3 gradTangentUOcc = float3(0.0f);
                            float3 gradTangentVOcc = float3(0.0f);

                            const float nDotD = dot(occluderNormal, rayDirection);
                            if (sycl::fabs(nDotD) > 1e-8f) {
                                const float3 hitMinusSp =
                                        worldHit.hitPositionW - occluderSurfel.position;
                                const float3 aOcc =
                                        occluderSurfel.position - segmentOrigin;
                                const float nDotA = dot(occluderNormal, aOcc);

                                const float invNDotD = 1.0f / nDotD;
                                const float invNDotDSquared = invNDotD * invNDotD;

                                const float3 qOcc =
                                        ((cross(occluderNormal, aOcc) * nDotD) -
                                         (nDotA * cross(occluderNormal, rayDirection))) *
                                        invNDotDSquared;

                                const float3 duDRotation =
                                        qOcc * (dot(rayDirection, tangentU) / scaleU) +
                                        (cross(tangentU, hitMinusSp) / scaleU);

                                const float3 dvDRotation =
                                        qOcc * (dot(rayDirection, tangentV) / scaleV) +
                                        (cross(tangentV, hitMinusSp) / scaleV);

                                const float3 dAlphaEffectiveDRotation =
                                        occluderSurfel.opacity * (
                                            dAlphaGeomDu * duDRotation +
                                            dAlphaGeomDv * dvDRotation);

                                const float3 rotationDerivativeZeta =
                                        dAlphaEffectiveDRotation * (1.0f / oneMinusAlpha);

                                gradTangentUOcc =
                                        cross(rotationDerivativeZeta, occluderSurfel.tanU);

                                gradTangentVOcc =
                                        cross(rotationDerivativeZeta, occluderSurfel.tanV);
                            }

                            if (storedOccluderCount < kMaxSplatEventsPerRay) {
                                const float invOneMinusAlpha = 1.0f / oneMinusAlpha;

                                occluderDerivatives[storedOccluderCount].gradPosition =
                                        dAlphaEffectiveDspi * invOneMinusAlpha;

                                occluderDerivatives[storedOccluderCount].gradScaleU =
                                        dAlphaEffectiveDScaleU * invOneMinusAlpha;

                                occluderDerivatives[storedOccluderCount].gradScaleV =
                                        dAlphaEffectiveDScaleV * invOneMinusAlpha;

                                occluderDerivatives[storedOccluderCount].gradEta =
                                        dAlphaEffectiveDEta * invOneMinusAlpha;

                                occluderDerivatives[storedOccluderCount].gradBeta =
                                        dAlphaEffectiveDBeta * invOneMinusAlpha;

                                occluderDerivatives[storedOccluderCount].gradTangentU =
                                        gradTangentUOcc;

                                occluderDerivatives[storedOccluderCount].gradTangentV =
                                        gradTangentVOcc;

                                occluderDerivatives[storedOccluderCount].primitiveIndex =
                                        worldHit.primitiveIndex;
                                storedOccluderCount++;
                            }
                        }
                    }

                    const float3 pathWeight = eventRecord.xPathThroughput;
                    const float scalarWeight = dot(pathWeight, outgoingRadianceX);


                    const float3 outgoingRadianceXNoAlpha = evaluateOutgoingRadianceWithoutLocalAlpha(
                        surfelX,
                        eventRecord.xSurface,
                        xState,
                        photonMap,
                        scene,
                        settings.numAdjointShadowRays,
                        directLightRng);

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

                    // Existing position derivative
                    const float3 dAlphaGeomDPosition = factor * dUvDPosition;
                    const float3 dAlphaEffectiveDPosition =
                            surfelX.opacity * dAlphaGeomDPosition;

                    const float scalarWeightNoAlpha =
                            dot(eventRecord.xPathThroughput, outgoingRadianceXNoAlpha);

                    const float3 positionGradient =
                            dAlphaEffectiveDPosition *
                            scalarWeightNoAlpha *
                            invSpp;

                    // Existing scale derivatives
                    float dAlphaGeomDScaleU = 0.0f;
                    float dAlphaGeomDScaleV = 0.0f;

                    if (scaleU > 1e-12f) {
                        dAlphaGeomDScaleU =
                                (2.0f * betaScale * u * u * eventRecord.xSurface.alphaGeom) /
                                (scaleU * oneMinusRadiusSquared);
                    }

                    if (scaleV > 1e-12f) {
                        dAlphaGeomDScaleV =
                                (2.0f * betaScale * v * v * eventRecord.xSurface.alphaGeom) /
                                (scaleV * oneMinusRadiusSquared);
                    }

                    const float dAlphaEffectiveDScaleU =
                            surfelX.opacity * dAlphaGeomDScaleU;
                    const float dAlphaEffectiveDScaleV =
                            surfelX.opacity * dAlphaGeomDScaleV;

                    const float scaleGradientU =
                            dAlphaEffectiveDScaleU *
                            scalarWeightNoAlpha *
                            invSpp;

                    const float scaleGradientV =
                            dAlphaEffectiveDScaleV *
                            scalarWeightNoAlpha *
                            invSpp;

                    // Receiver-opacity rotation term, mapped to tanU / tanV updates
                    float3 tanUGradient = float3(0.0f);
                    float3 tanVGradient = float3(0.0f);

                    const float dAlphaGeomDu = factor * u;
                    const float dAlphaGeomDv = factor * v;

                    const float3 rayDirection = normalize(vectorCameraToX);
                    const float3 normalX = xState.orientedNormal;

                    const float nDotD = dot(normalX, rayDirection);
                    if (sycl::fabs(nDotD) > 1e-8f && scaleU > 1e-12f && scaleV > 1e-12f) {
                        const float3 xMinusSp =
                                (u * scaleU) * surfelX.tanU +
                                (v * scaleV) * surfelX.tanV;

                        const float3 a =
                                (xState.position - sensor.camera.pos) - xMinusSp;

                        const float nDotA = dot(normalX, a);
                        const float invNDotD = 1.0f / nDotD;
                        const float invNDotDSquared = invNDotD * invNDotD;

                        const float3 q =
                                ((cross(normalX, a) * nDotD) -
                                 (nDotA * cross(normalX, rayDirection))) *
                                invNDotDSquared;

                        const float3 duDRotation =
                                q * (dot(rayDirection, surfelX.tanU) / scaleU) +
                                (cross(surfelX.tanU, xMinusSp) / scaleU);

                        const float3 dvDRotation =
                                q * (dot(rayDirection, surfelX.tanV) / scaleV) +
                                (cross(surfelX.tanV, xMinusSp) / scaleV);

                        const float3 dAlphaGeomDRotation =
                                dAlphaGeomDu * duDRotation +
                                dAlphaGeomDv * dvDRotation;

                        const float3 dAlphaEffectiveDRotation =
                                surfelX.opacity * dAlphaGeomDRotation;

                        const float3 rotationGradientZeta =
                                dAlphaEffectiveDRotation *
                                scalarWeightNoAlpha *
                                invSpp;

                        tanUGradient = cross(rotationGradientZeta, surfelX.tanU);
                        tanVGradient = cross(rotationGradientZeta, surfelX.tanV);
                    }

                    // Local opacity and beta gradients for surfel X in the camera->X term
                    const float alphaGeomX = eventRecord.xSurface.alphaGeom;

                    const float opacityGradient =
                            alphaGeomX *
                            scalarWeightNoAlpha *
                            invSpp;

                    const float dAlphaGeomDBeta =
                            betaScale *
                            sycl::log(oneMinusRadiusSquared) *
                            alphaGeomX;

                    const float betaGradient =
                            surfelX.opacity *
                            dAlphaGeomDBeta *
                            scalarWeightNoAlpha *
                            invSpp;

                    SurfelGradientRecord gradientRecord{};
                    gradientRecord.primitiveIndex = eventRecord.xSurface.primitiveIndex;
                    gradientRecord.gradPositionX = positionGradient.x();
                    gradientRecord.gradPositionY = positionGradient.y();
                    gradientRecord.gradPositionZ = positionGradient.z();
                    gradientRecord.gradScaleU = scaleGradientU;
                    gradientRecord.gradScaleV = scaleGradientV;

                    gradientRecord.gradTangentUX = tanUGradient.x();
                    gradientRecord.gradTangentUY = tanUGradient.y();
                    gradientRecord.gradTangentUZ = tanUGradient.z();

                    gradientRecord.gradTangentVX = tanVGradient.x();
                    gradientRecord.gradTangentVY = tanVGradient.y();
                    gradientRecord.gradTangentVZ = tanVGradient.z();

                    gradientRecord.gradEta = opacityGradient;
                    gradientRecord.gradBeta = betaGradient;

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

                        const float3 occluderPositionContribution =
                                occluderScale * occluderDerivative.gradPosition;
                        const float3 occluderTanUContribution =
                                occluderScale * occluderDerivative.gradTangentU;
                        const float3 occluderTanVContribution =
                                occluderScale * occluderDerivative.gradTangentV;

                        gradientRecordOccluder.gradPositionX = occluderPositionContribution.x();
                        gradientRecordOccluder.gradPositionY = occluderPositionContribution.y();
                        gradientRecordOccluder.gradPositionZ = occluderPositionContribution.z();

                        gradientRecordOccluder.gradScaleU =
                                occluderScale * occluderDerivative.gradScaleU;
                        gradientRecordOccluder.gradScaleV =
                                occluderScale * occluderDerivative.gradScaleV;

                        gradientRecordOccluder.gradEta =
                                occluderScale * occluderDerivative.gradEta;
                        gradientRecordOccluder.gradBeta =
                                occluderScale * occluderDerivative.gradBeta;

                        gradientRecordOccluder.gradTangentUX = occluderTanUContribution.x();
                        gradientRecordOccluder.gradTangentUY = occluderTanUContribution.y();
                        gradientRecordOccluder.gradTangentUZ = occluderTanUContribution.z();

                        gradientRecordOccluder.gradTangentVX = occluderTanVContribution.x();
                        gradientRecordOccluder.gradTangentVY = occluderTanVContribution.y();
                        gradientRecordOccluder.gradTangentVZ = occluderTanVContribution.z();

                        gradientRecords[occluderRecordIndex] = gradientRecordOccluder;
                    }
                });
        }).wait();
    }

    static void measurementGradientEventXY(RenderPackage &pkg,
                                           uint32_t onePointEventCount,
                                           uint32_t baseOffset) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        const auto &photonMap = pkg.intermediates.map;
        MeasurementGradientEventXY *measurementXYEvent =
                pkg.intermediates.measurementTwoPointEvents;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;

        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;
        const float qNullInv = 1.0f / settings.sampling.qNull;

        queue.submit([&](sycl::handler &commandGroupHandler) {
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

                    const Point &surfelX = scene.points[xPrimitiveIndex];
                    const Point &surfelY = scene.points[yPrimitiveIndex];

                    const ReconstructedSurfelState xState =
                            reconstructSurfelState(surfelX, eventRecord.xSurface);
                    const ReconstructedSurfelState yState =
                            reconstructSurfelState(surfelY, eventRecord.ySurface);

                    const uint64_t directLightSeed = rng::makeSeed(
                        settings.random.seed,
                        eventRecord.xSurface.pathId,
                        0xffefeefef,
                        rng::kStreamDirectLight,
                        eventIndex);

                    rng::Xorshift128 directLightRng(directLightSeed);

                    float3 outgoingRadianceY;

                    if (settings.enableAdjointDirectLight) {
                        if (eventRecord.isDirectLightSample) {
                            outgoingRadianceY = eventRecord.directLightRadiance;
                        } else {
                            outgoingRadianceY =
                                    evaluateOutgoingRadianceWithLocalAlphaNoEmitters(
                                        surfelY,
                                        eventRecord.ySurface,
                                        yState,
                                        photonMap,
                                        scene,
                                        settings.numAdjointShadowRays,
                                        directLightRng);
                        }
                    } else {
                        outgoingRadianceY =
                                evaluateOutgoingRadianceWithLocalAlpha(
                                    surfelY,
                                    eventRecord.ySurface,
                                    yState,
                                    photonMap,
                                    scene,
                                    settings.numAdjointShadowRays,
                                    directLightRng);
                    }

                    const float alphaX = eventRecord.xSurface.alphaGeom * surfelX.opacity;
                    const float brdfScaleX = surfelX.alpha_r * M_1_PIf;
                    const float3 brdfX = brdfScaleX * surfelX.albedo;

                    const float3 dGeometricTermDx = computeGeometricTermGradientWrtStartpoint(
                        xState.position,
                        yState.position,
                        xState.orientedNormal,
                        yState.orientedNormal);

                    const float3 pathWeight =
                            eventRecord.xPathThroughput;

                    const float3 transportWithoutTauAndGeometric =
                            outgoingRadianceY * alphaX * brdfX;

                    float scalarWeightWithoutTauAndGeometric =
                            dot(pathWeight, transportWithoutTauAndGeometric);

                    float3 albedoWeightWithoutTauAndGeometric =
                            (pathWeight * outgoingRadianceY) * (alphaX * brdfScaleX);

                    struct OccluderDerivative {
                        float3 gradPosition{0.0f};
                        float gradScaleU = 0.0f;
                        float gradScaleV = 0.0f;
                        float gradEta = 0.0f;
                        float gradBeta = 0.0f;
                        float3 gradTangentU{0.0f};
                        float3 gradTangentV{0.0f};
                        uint32_t primitiveIndex = kInvalidIndex;
                    };

                    float transmittance = 1.0f;
                    float3 accumulatedAlphaDerivativeOverOneMinusAlphaStartPoint = float3{0.0f};
                    float3 accumulatedAlphaDerivativeOverOneMinusAlphaEndPoint = float3{0.0f};
                    OccluderDerivative occluderDerivatives[kMaxSplatEventsPerRay];
                    uint32_t storedOccluderCount = 0u;

                    if (eventRecord.transmission != 1.0f) {
                        const float distanceEpsilon = 1e-4f;

                        float3 segmentDirection = yState.position - xState.position;
                        const float3 rayDirection = normalize(segmentDirection);
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

                            const float alphaGeomOccluder = worldHit.alphaGeom;
                            const float alphaEffective = occluderSurfel.opacity * alphaGeomOccluder;
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

                            const float3 dUiDx =
                                    lambdaOccluder * (
                                        localBasisU -
                                        occluderNormal * (dot(dxy, localBasisU) * inverseDenominator));

                            const float3 dViDx =
                                    lambdaOccluder * (
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

                            // Endpoint derivative wrt y (existing)
                            const float3 dAlphaEffectiveDy =
                                    occluderSurfel.opacity * (
                                        dAlphaGeomDu * dUiDy +
                                        dAlphaGeomDv * dViDy);

                            // Translation derivative wrt surfel center (existing)
                            const float3 dAlphaEffectiveDspi =
                                    occluderSurfel.opacity * (
                                        dAlphaGeomDu * dUiDspi +
                                        dAlphaGeomDv * dViDspi);

                            const float3 dAlphaEffectiveDx =
                                    occluderSurfel.opacity * (
                                        dAlphaGeomDu * dUiDx +
                                        dAlphaGeomDv * dViDx);

                            // Scale derivatives
                            const float dAlphaEffectiveDScaleU =
                                    (2.0f * betaScale * uOcc * uOcc * alphaEffective) /
                                    (scaleU * oneMinusRadiusSquared);

                            const float dAlphaEffectiveDScaleV =
                                    (2.0f * betaScale * vOcc * vOcc * alphaEffective) /
                                    (scaleV * oneMinusRadiusSquared);

                            // Opacity / eta derivative
                            const float dAlphaEffectiveDEta =
                                    alphaGeomOccluder;

                            // Beta derivative
                            const float dAlphaEffectiveDBeta =
                                    betaScale * sycl::log(oneMinusRadiusSquared) * alphaEffective;

                            // Rotation derivative with fixed segment endpoints x,y
                            float3 gradTangentUOcc = float3(0.0f);
                            float3 gradTangentVOcc = float3(0.0f);

                            const float nDotD = dot(occluderNormal, rayDirection);
                            if (sycl::fabs(nDotD) > 1e-8f) {
                                const float3 hitMinusSp =
                                        worldHit.hitPositionW - occluderSurfel.position;
                                const float3 aOcc =
                                        occluderSurfel.position - yPosition;
                                const float nDotA = dot(occluderNormal, aOcc);

                                const float invNDotD = 1.0f / nDotD;
                                const float invNDotDSquared = invNDotD * invNDotD;

                                const float3 qOcc =
                                        ((cross(occluderNormal, aOcc) * nDotD) -
                                         (nDotA * cross(occluderNormal, rayDirection))) *
                                        invNDotDSquared;

                                const float3 duDRotation =
                                        qOcc * (dot(rayDirection, tangentU) / scaleU) +
                                        (cross(tangentU, hitMinusSp) / scaleU);

                                const float3 dvDRotation =
                                        qOcc * (dot(rayDirection, tangentV) / scaleV) +
                                        (cross(tangentV, hitMinusSp) / scaleV);

                                const float3 dAlphaEffectiveDRotation =
                                        occluderSurfel.opacity * (
                                            dAlphaGeomDu * duDRotation +
                                            dAlphaGeomDv * dvDRotation);

                                const float invOneMinusAlpha = 1.0f / oneMinusAlpha;
                                const float3 rotationDerivativeZeta =
                                        dAlphaEffectiveDRotation * invOneMinusAlpha;

                                gradTangentUOcc =
                                        cross(rotationDerivativeZeta, occluderSurfel.tanU);

                                gradTangentVOcc =
                                        cross(rotationDerivativeZeta, occluderSurfel.tanV);
                            }

                            accumulatedAlphaDerivativeOverOneMinusAlphaStartPoint +=
                                    dAlphaEffectiveDx * (1.0f / oneMinusAlpha);

                            accumulatedAlphaDerivativeOverOneMinusAlphaEndPoint +=
                                    dAlphaEffectiveDy * (1.0f / oneMinusAlpha);

                            if (storedOccluderCount < kMaxSplatEventsPerRay) {
                                const float invOneMinusAlpha = 1.0f / oneMinusAlpha;

                                occluderDerivatives[storedOccluderCount].gradPosition =
                                        dAlphaEffectiveDspi * invOneMinusAlpha;
                                occluderDerivatives[storedOccluderCount].gradScaleU =
                                        dAlphaEffectiveDScaleU * invOneMinusAlpha;
                                occluderDerivatives[storedOccluderCount].gradScaleV =
                                        dAlphaEffectiveDScaleV * invOneMinusAlpha;
                                occluderDerivatives[storedOccluderCount].gradEta =
                                        dAlphaEffectiveDEta * invOneMinusAlpha;
                                occluderDerivatives[storedOccluderCount].gradBeta =
                                        dAlphaEffectiveDBeta * invOneMinusAlpha;
                                occluderDerivatives[storedOccluderCount].gradTangentU =
                                        gradTangentUOcc;
                                occluderDerivatives[storedOccluderCount].gradTangentV =
                                        gradTangentVOcc;
                                occluderDerivatives[storedOccluderCount].primitiveIndex =
                                        worldHit.primitiveIndex;
                                storedOccluderCount++;
                            }

                            scalarWeightWithoutTauAndGeometric *= qNullInv;
                            albedoWeightWithoutTauAndGeometric *= qNullInv;
                        }
                    }

                    const float geometricTermXY = computeGeometricTermValue(
                        xState.position,
                        yState.position,
                        xState.orientedNormal,
                        yState.orientedNormal);

                    const float3 gradientWrtWorldHitPositionX =
                            scalarWeightWithoutTauAndGeometric *
                            transmittance *
                            (dGeometricTermDx
                             - geometricTermXY * accumulatedAlphaDerivativeOverOneMinusAlphaStartPoint);

                    float3 gradientWrtHitPositionX = gradientWrtWorldHitPositionX;

                    const float3x3 hitPointJacobianX = planeHitPointIntersectionJacobian(
                        eventRecord.xSurface.incomingDirection,
                        xState.orientedNormal);

                    gradientWrtHitPositionX = transpose(hitPointJacobianX) * gradientWrtHitPositionX;
                    const float3 xContribution = gradientWrtHitPositionX * invSpp;

                    // tanU / tanV gradients for surfel X on the X->Y segment
                    float3 tanUContribution = float3(0.0f);
                    float3 tanVContribution = float3(0.0f);

                    const float3 primaryRayDirection = eventRecord.xSurface.incomingDirection;
                    const float nDotD = dot(xState.orientedNormal, primaryRayDirection);

                    const float3 rawCross = cross(surfelX.tanU, surfelX.tanV);
                    const float rawCrossLength = length(rawCross);

                    if (sycl::fabs(nDotD) > 1e-8f && rawCrossLength > 1e-8f) {
                        const float3 rawNormal = rawCross / rawCrossLength;

                        const float orientationSign =
                                dot(rawNormal, xState.orientedNormal) >= 0.0f ? 1.0f : -1.0f;

                        const float3 pMinusX = surfelX.position - xState.position;

                        const float3 gradientWrtOrientedNormalFromMovedHit =
                                pMinusX *
                                (dot(primaryRayDirection, gradientWrtWorldHitPositionX) / nDotD);

                        float3 segmentDirection = yState.position - xState.position;
                        float distanceSquared = dot(segmentDirection, segmentDirection);
                        float3 directionXToY = normalize(segmentDirection);
                        const float cosineAtY = dot(yState.orientedNormal, -directionXToY);

                        const float3 dGeometricTermDStartNormal =
                                directionXToY * (cosineAtY / distanceSquared);

                        const float3 gradientWrtOrientedNormalExplicit =
                                scalarWeightWithoutTauAndGeometric *
                                transmittance *
                                dGeometricTermDStartNormal;

                        const float3 gradientWrtOrientedNormalX =
                                gradientWrtOrientedNormalFromMovedHit +
                                gradientWrtOrientedNormalExplicit;

                        const float3 gradientWrtRawNormal =
                                orientationSign * gradientWrtOrientedNormalX;

                        const float3 gradientProjectedToRawNormalTangent =
                                gradientWrtRawNormal -
                                rawNormal * dot(rawNormal, gradientWrtRawNormal);

                        float3 gradientWrtCross =
                                gradientProjectedToRawNormalTangent / rawCrossLength;

                        tanUContribution =
                                cross(surfelX.tanV, gradientWrtCross) * invSpp;

                        tanVContribution =
                                cross(gradientWrtCross, surfelX.tanU) * invSpp;
                    }


                    const float3 albedoContribution =
                            transmittance *
                            geometricTermXY *
                            albedoWeightWithoutTauAndGeometric *
                            invSpp;

                    SurfelGradientRecord xRecord{};
                    xRecord.primitiveIndex = xPrimitiveIndex;
                    xRecord.gradPositionX = xContribution.x();
                    xRecord.gradPositionY = xContribution.y();
                    xRecord.gradPositionZ = xContribution.z();

                    xRecord.gradTangentUX = tanUContribution.x();
                    xRecord.gradTangentUY = tanUContribution.y();
                    xRecord.gradTangentUZ = tanUContribution.z();

                    xRecord.gradTangentVX = tanVContribution.x();
                    xRecord.gradTangentVY = tanVContribution.y();
                    xRecord.gradTangentVZ = tanVContribution.z();

                    xRecord.gradAlbedoR = albedoContribution.x();
                    xRecord.gradAlbedoG = albedoContribution.y();
                    xRecord.gradAlbedoB = albedoContribution.z();

                    gradientRecords[recordIndex] = xRecord;

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

                        const OccluderDerivative &occluderDerivative =
                                occluderDerivatives[occluderIndex];

                        SurfelGradientRecord occluderRecord{};
                        occluderRecord.primitiveIndex =
                                occluderDerivative.primitiveIndex;

                        const float3 occluderPositionContribution =
                                occluderScale * occluderDerivative.gradPosition;
                        const float3 occluderTanUContribution =
                                occluderScale * occluderDerivative.gradTangentU;
                        const float3 occluderTanVContribution =
                                occluderScale * occluderDerivative.gradTangentV;

                        occluderRecord.gradPositionX = occluderPositionContribution.x();
                        occluderRecord.gradPositionY = occluderPositionContribution.y();
                        occluderRecord.gradPositionZ = occluderPositionContribution.z();

                        occluderRecord.gradScaleU =
                                occluderScale * occluderDerivative.gradScaleU;
                        occluderRecord.gradScaleV =
                                occluderScale * occluderDerivative.gradScaleV;

                        occluderRecord.gradEta =
                                occluderScale * occluderDerivative.gradEta;
                        occluderRecord.gradBeta =
                                occluderScale * occluderDerivative.gradBeta;

                        occluderRecord.gradTangentUX = occluderTanUContribution.x();
                        occluderRecord.gradTangentUY = occluderTanUContribution.y();
                        occluderRecord.gradTangentUZ = occluderTanUContribution.z();

                        occluderRecord.gradTangentVX = occluderTanVContribution.x();
                        occluderRecord.gradTangentVY = occluderTanVContribution.y();
                        occluderRecord.gradTangentVZ = occluderTanVContribution.z();

                        gradientRecords[occluderRecordIndex] = occluderRecord;
                    }
                });
        }).wait();
    }

    static void cameraAttachedBridgeEvent(
        RenderPackage &pkg,
        uint32_t twoPointEventCount,
        uint32_t baseOffset) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        const auto &photonMap = pkg.intermediates.map;
        CameraAttachedBridgeGradientEvent *cameraAttachedEvents =
                pkg.intermediates.cameraAttachedBridgeEvents;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;

        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;
        const float qNullInv = 1.0f / settings.sampling.qNull;
        auto &sensor = pkg.sensors.front();
        (void) sensor;

        // Camera-attached first bridge X -> Y:
        // Differentiate endpoint Y on the XY segment.
        // Under the current split:
        //   - BSDF / albedo gradient belongs to X, not Y.
        //   - Y contributes through:
        //       * y-position dependence of tau(x,y) G(x,y)
        //       * y-normal dependence of G(x,y)
        //       * local alpha_y (eta_y, beta_y)
        //       * local chart motion under scale / rotation of Y
        //
        // We defer d/d psi_y of L_surfel(Y, ...) itself to later adjoint propagation.

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class cameraAttachedBridgeEventTag>(
                sycl::range<1>(twoPointEventCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex = globalId[0];

                    static constexpr uint32_t recordsPerEvent = 1u;
                    const uint32_t eventRecordBase = baseOffset + recordsPerEvent * eventIndex;
                    const uint32_t yRecordIndex = eventRecordBase + 0u;
                    const CameraAttachedBridgeGradientEvent eventRecord =
                            cameraAttachedEvents[eventIndex];

                    // Clear record slot
                    {
                        SurfelGradientRecord invalidRecord{};
                        invalidRecord.primitiveIndex = kInvalidIndex;
                        gradientRecords[yRecordIndex] = invalidRecord;
                    }

                    const uint32_t xPrimitiveIndex = eventRecord.xSurface.primitiveIndex;
                    const uint32_t yPrimitiveIndex = eventRecord.ySurface.primitiveIndex;

                    const Point &surfelX = scene.points[xPrimitiveIndex];
                    const Point &surfelY = scene.points[yPrimitiveIndex];

                    const ReconstructedSurfelState xState =
                            reconstructSurfelState(surfelX, eventRecord.xSurface);
                    const ReconstructedSurfelState yState =
                            reconstructSurfelState(surfelY, eventRecord.ySurface);

                    const uint64_t directLightSeed = rng::makeSeed(
                        settings.random.seed,
                        eventRecord.xSurface.pathId,
                        0xffefeefef,
                        rng::kStreamDirectLight,
                        eventIndex);

                    rng::Xorshift128 directLightRng(directLightSeed);

                    // Includes local alpha_Y
                    const float3 outgoingRadianceY =
                            evaluateOutgoingRadianceWithLocalAlpha(
                                surfelY,
                                eventRecord.ySurface,
                                yState,
                                photonMap,
                                scene,
                                settings.numAdjointShadowRays,
                                directLightRng);

                    // Excludes local alpha_Y
                    const float3 outgoingRadianceYNoAlpha =
                            evaluateOutgoingRadianceWithoutLocalAlpha(
                                surfelY,
                                eventRecord.ySurface,
                                yState,
                                photonMap,
                                scene,
                                settings.numAdjointShadowRays,
                                directLightRng);

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

                    const float cosineAtX = dot(xState.orientedNormal, directionXToY);

                    const float uniformHemispherePdf = 1.0f / (2.0f * M_PIf);
                    const float pAreaY = uniformHemispherePdf * cosineAtY / distanceSquared;
                    if (pAreaY <= 1e-20f) {
                        return;
                    }

                    // Local UV/sample information on Y
                    const float uY = eventRecord.ySurface.uv.x();
                    const float vY = eventRecord.ySurface.uv.y();
                    const float radiusSquaredY = uY * uY + vY * vY;
                    const float oneMinusRadiusSquaredY = 1.0f - radiusSquaredY;
                    if (oneMinusRadiusSquaredY <= 1e-8f) {
                        return;
                    }

                    const float scaleYU = surfelY.scale.x();
                    const float scaleYV = surfelY.scale.y();
                    if (scaleYU <= 1e-12f || scaleYV <= 1e-12f) {
                        return;
                    }

                    // Current implementation uses the same estimator structure as before:
                    // scalar weight carries Juv / PuvY, which cancels the local chart Jacobian.
                    const float Juv = scaleYU * scaleYV;
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
                    const float3 pathWeight = eventRecord.xPathThroughput;

                    const float3 transportWithoutTauAndGeometric =
                            outgoingRadianceY * alphaX * brdfX;

                    float scalarWeightWithoutTauAndGeometric =
                            dot(pathWeight, transportWithoutTauAndGeometric) * Juv / PuvY;

                    // ---------------------------------------------------------------------
                    // New: local eta_y / beta_y weights.
                    //
                    // outgoingRadianceY          = alphaY * L_surfel(Y, ...)
                    // outgoingRadianceYNoAlpha  =         L_surfel(Y, ...)
                    //
                    // alphaY = etaY * alphaGeomY
                    // d alphaY / d etaY  = alphaGeomY
                    // d alphaY / d betaY = etaY * d alphaGeomY / d betaY
                    //
                    // beta kernel:
                    // alphaGeomY = (1 - r^2)^b, b = 4 exp(beta)
                    // d alphaGeomY / d betaY = b * log(1 - r^2) * alphaGeomY
                    // ---------------------------------------------------------------------
                    const float alphaGeomY = eventRecord.ySurface.alphaGeom;
                    const float betaScaleY = 4.0f * sycl::exp(surfelY.beta);
                    const float dAlphaGeomYDBeta =
                            betaScaleY * sycl::log(oneMinusRadiusSquaredY) * alphaGeomY;

                    float scalarWeightWithoutTauAndGeometricEtaY =
                            dot(pathWeight,
                                outgoingRadianceYNoAlpha *
                                (alphaGeomY * alphaX * brdfX)) * Juv / PuvY;

                    float scalarWeightWithoutTauAndGeometricBetaY =
                            dot(pathWeight,
                                outgoingRadianceYNoAlpha *
                                (surfelY.opacity * dAlphaGeomYDBeta * alphaX * brdfX)) * Juv / PuvY;

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
                            const float uOcc = uv.x();
                            const float vOcc = uv.y();

                            const float occScaleU = occluderSurfel.scale.x();
                            const float occScaleV = occluderSurfel.scale.y();
                            if (occScaleU <= 1e-12f || occScaleV <= 1e-12f) {
                                continue;
                            }

                            const float3 tangentU = occluderSurfel.tanU;
                            const float3 tangentV = occluderSurfel.tanV;
                            const float3 localBasisU = tangentU / occScaleU;
                            const float3 localBasisV = tangentV / occScaleV;
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

                            // Keep the same null-event compensation for all local scalar weights
                            scalarWeightWithoutTauAndGeometric *= qNullInv;
                            scalarWeightWithoutTauAndGeometricEtaY *= qNullInv;
                            scalarWeightWithoutTauAndGeometricBetaY *= qNullInv;
                        }
                    }

                    const float3 dTransmittanceDy =
                            -transmittance * accumulatedAlphaDerivativeOverOneMinusAlphaEndPoint;

                    // World-position gradient wrt endpoint y
                    const float3 gradientWrtHitPositionY =
                            scalarWeightWithoutTauAndGeometric *
                            (geometricTermXY * dTransmittanceDy + transmittance * dGeometricTermDy);

                    const float3 yContribution = gradientWrtHitPositionY * invSpp;

                    // ---------------------------------------------------------------------
                    // Scale gradients on Y:
                    // y = s_p + s_u t_u u + s_v t_v v
                    // dy/ds_u = u * t_u, dy/ds_v = v * t_v
                    //
                    // In the current estimator structure Juv cancels against PuvY,
                    // so only moved-point terms are included here.
                    // ---------------------------------------------------------------------
                    const float scaleYContributionU =
                            dot(gradientWrtHitPositionY, uY * surfelY.tanU) * invSpp;

                    const float scaleYContributionV =
                            dot(gradientWrtHitPositionY, vY * surfelY.tanV) * invSpp;

                    // ---------------------------------------------------------------------
                    // Tangent gradients on Y:
                    //
                    // (A) moved-point contribution:
                    //     dy/dt_u = s_u * u * I
                    //     dy/dt_v = s_v * v * I
                    //
                    // (B) explicit normal dependence of G wrt n_y:
                    //     dG/dn_y = (-omega_xy) * cos(theta_x) / d^2
                    //
                    // Convert gradient wrt raw normal n = normalize(t_u x t_v)
                    // to Euclidean gradients wrt tanU / tanV.
                    // ---------------------------------------------------------------------
                    float3 tanUYContribution = float3(0.0f);
                    float3 tanVYContribution = float3(0.0f);

                    // (A) moved-point contribution
                    tanUYContribution += (scaleYU * uY) * gradientWrtHitPositionY * invSpp;
                    tanVYContribution += (scaleYV * vY) * gradientWrtHitPositionY * invSpp;

                    // (B) explicit end-normal contribution
                    const float3 rawCross = cross(surfelY.tanU, surfelY.tanV);
                    const float rawCrossLength = length(rawCross);
                    if (rawCrossLength > 1e-8f) {
                        const float3 rawNormal = rawCross / rawCrossLength;

                        const float orientationSign =
                                dot(rawNormal, yState.orientedNormal) >= 0.0f ? 1.0f : -1.0f;

                        // dG / dn_y
                        const float3 dGeometricTermDEndNormal =
                                (-directionXToY) * (cosineAtX / distanceSquared);

                        const float3 gradientWrtOrientedNormalY =
                                scalarWeightWithoutTauAndGeometric *
                                transmittance *
                                dGeometricTermDEndNormal;

                        const float3 gradientWrtRawNormal =
                                orientationSign * gradientWrtOrientedNormalY;

                        const float3 gradientProjectedToRawNormalTangent =
                                gradientWrtRawNormal -
                                rawNormal * dot(rawNormal, gradientWrtRawNormal);

                        const float3 gradientWrtCross =
                                gradientProjectedToRawNormalTangent / rawCrossLength;

                        tanUYContribution +=
                                cross(surfelY.tanV, gradientWrtCross) * invSpp;

                        tanVYContribution +=
                                cross(gradientWrtCross, surfelY.tanU) * invSpp;
                    }

                    // ---------------------------------------------------------------------
                    // Local eta_y / beta_y on the XY leg.
                    //
                    // Contribution structure:
                    //   alphaY * (alphaX * brdfX * tau * G * ...)
                    //
                    // so:
                    //   dL/d etaY  = alphaGeomY * (...)
                    //   dL/d betaY = opacityY * dAlphaGeomY/dBetaY * (...)
                    // ---------------------------------------------------------------------
                    const float etaYContribution =
                            transmittance *
                            geometricTermXY *
                            scalarWeightWithoutTauAndGeometricEtaY *
                            invSpp;

                    const float betaYContribution =
                            transmittance *
                            geometricTermXY *
                            scalarWeightWithoutTauAndGeometricBetaY *
                            invSpp;

                    SurfelGradientRecord yRecord{};
                    yRecord.primitiveIndex = yPrimitiveIndex;

                    yRecord.gradPositionX = yContribution.x();
                    yRecord.gradPositionY = yContribution.y();
                    yRecord.gradPositionZ = yContribution.z();

                    yRecord.gradScaleU = scaleYContributionU;
                    yRecord.gradScaleV = scaleYContributionV;

                    yRecord.gradTangentUX = tanUYContribution.x();
                    yRecord.gradTangentUY = tanUYContribution.y();
                    yRecord.gradTangentUZ = tanUYContribution.z();

                    yRecord.gradTangentVX = tanVYContribution.x();
                    yRecord.gradTangentVY = tanVYContribution.y();
                    yRecord.gradTangentVZ = tanVYContribution.z();

                    yRecord.gradEta = etaYContribution;
                    yRecord.gradBeta = betaYContribution;

                    gradientRecords[yRecordIndex] = yRecord;
                });
        }).wait();
    }

    static void recursiveBridgeEvent(
        RenderPackage &pkg,
        uint32_t recursiveBridgeEventCount,
        uint32_t baseOffset) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        const auto &photonMap = pkg.intermediates.map;
        RecursiveBridgeGradientEvent *recursiveBridgeEvents =
                pkg.intermediates.recursiveBridgeEvents;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;

        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;
        const float qNullInv = 1.0f / settings.sampling.qNull;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class recursiveBridgeEventTag>(
                sycl::range<1>(recursiveBridgeEventCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex = globalId[0];

                    static constexpr uint32_t recordsPerEvent = 2u + kMaxSplatEventsPerRay;
                    const uint32_t eventRecordBase = baseOffset + recordsPerEvent * eventIndex;
                    const uint32_t xRecordIndex = eventRecordBase + 0u; // code xSurface = math Y
                    const uint32_t yRecordIndex = eventRecordBase + 1u; // code ySurface = math Z

                    for (uint32_t recordOffset = 0u; recordOffset < recordsPerEvent; ++recordOffset) {
                        SurfelGradientRecord invalidRecord{};
                        invalidRecord.primitiveIndex = kInvalidIndex;
                        gradientRecords[eventRecordBase + recordOffset] = invalidRecord;
                    }

                    const RecursiveBridgeGradientEvent eventRecord =
                            recursiveBridgeEvents[eventIndex];

                    const uint32_t xPrimitiveIndex = eventRecord.xSurface.primitiveIndex;
                    const uint32_t yPrimitiveIndex = eventRecord.ySurface.primitiveIndex;

                    const Point &surfelX = scene.points[xPrimitiveIndex];
                    const Point &surfelY = scene.points[yPrimitiveIndex];

                    const ReconstructedSurfelState xState =
                            reconstructSurfelState(surfelX, eventRecord.xSurface);
                    const ReconstructedSurfelState yState =
                            reconstructSurfelState(surfelY, eventRecord.ySurface);

                    const uint64_t directLightSeed = rng::makeSeed(
                        settings.random.seed,
                        eventRecord.xSurface.pathId,
                        0xffefeefef,
                        rng::kStreamDirectLight,
                        eventIndex);

                    rng::Xorshift128 directLightRng(directLightSeed);

                    // Includes local alpha at code ySurface (= math Z)
                    const float3 outgoingRadianceY =
                            evaluateOutgoingRadianceWithLocalAlpha(
                                surfelY,
                                eventRecord.ySurface,
                                yState,
                                photonMap,
                                scene,
                                settings.numAdjointShadowRays,
                                directLightRng);

                    // Excludes local alpha at code ySurface (= math Z)
                    const float3 outgoingRadianceYNoAlpha =
                            evaluateOutgoingRadianceWithoutLocalAlpha(
                                surfelY,
                                eventRecord.ySurface,
                                yState,
                                photonMap, scene,
                                settings.numAdjointShadowRays,
                                directLightRng);

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

                    const float cosineAtX = dot(xState.orientedNormal, directionXToY);

                    const float uniformHemispherePdf = 1.0f / (2.0f * M_PIf);
                    const float pAreaY = uniformHemispherePdf * cosineAtY / distanceSquared;
                    if (pAreaY <= 1e-20f) {
                        return;
                    }

                    // code xSurface = math Y
                    const float uX = eventRecord.xSurface.uv.x();
                    const float vX = eventRecord.xSurface.uv.y();
                    const float alphaGeomX = eventRecord.xSurface.alphaGeom;

                    const float scaleXU = surfelX.scale.x();
                    const float scaleXV = surfelX.scale.y();
                    const float Juv = scaleXU * scaleXV;
                    const float PuvY = Juv * pAreaY;
                    if (PuvY <= 1e-20f) {
                        return;
                    }

                    const float radiusSquaredX = uX * uX + vX * vX;
                    const float oneMinusRadiusSquaredX = 1.0f - radiusSquaredX;

                    const float alphaX = alphaGeomX * surfelX.opacity;
                    const float brdfScaleX = surfelX.alpha_r * M_1_PIf;
                    const float3 brdfX = brdfScaleX * surfelX.albedo;

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

                    // Prefix adjoint weight at code xSurface (= math Y)
                    const float3 pathWeight = eventRecord.xPathThroughput;

                    // Local transport at code xSurface, excluding tau(X,Y) and G(X,Y)
                    const float3 transportWithoutTauAndGeometric =
                            outgoingRadianceY * alphaX * brdfX;

                    float scalarWeightWithoutTauAndGeometric =
                            dot(pathWeight, transportWithoutTauAndGeometric) * Juv / PuvY;

                    // ---------------------------------------------------------------------
                    // Local albedo weight for code xSurface (= math Y)
                    // ---------------------------------------------------------------------
                    float3 albedoWeightWithoutTauAndGeometricX = float3{0.0f, 0.0f, 0.0f};
                    albedoWeightWithoutTauAndGeometricX =
                            (pathWeight * outgoingRadianceY) * (alphaX * brdfScaleX * Juv / PuvY);

                    // ---------------------------------------------------------------------
                    // Local eta / beta weights for code ySurface (= math Z)
                    // ---------------------------------------------------------------------
                    const float uYLocal = eventRecord.ySurface.uv.x();
                    const float vYLocal = eventRecord.ySurface.uv.y();
                    const float alphaGeomYLocal = eventRecord.ySurface.alphaGeom;
                    const float radiusSquaredYLocal = uYLocal * uYLocal + vYLocal * vYLocal;
                    const float oneMinusRadiusSquaredYLocal = 1.0f - radiusSquaredYLocal;

                    const float scaleYU = surfelY.scale.x();
                    const float scaleYV = surfelY.scale.y();

                    float scalarWeightWithoutTauAndGeometricEtaY = 0.0f;
                    float scalarWeightWithoutTauAndGeometricBetaY = 0.0f;

                    scalarWeightWithoutTauAndGeometricEtaY =
                            dot(pathWeight,
                                outgoingRadianceYNoAlpha *
                                (alphaGeomYLocal * alphaX * brdfX)) * Juv / PuvY;

                    if (oneMinusRadiusSquaredYLocal > 1e-8f) {
                        const float betaScaleYLocal = 4.0f * sycl::exp(surfelY.beta);
                        const float dAlphaGeomYLocalDBeta =
                                betaScaleYLocal *
                                sycl::log(oneMinusRadiusSquaredYLocal) *
                                alphaGeomYLocal;

                        scalarWeightWithoutTauAndGeometricBetaY =
                                dot(pathWeight,
                                    outgoingRadianceYNoAlpha *
                                    (surfelY.opacity * dAlphaGeomYLocalDBeta * alphaX * brdfX)) *
                                Juv / PuvY;
                    }

                    struct OccluderDerivative {
                        float3 gradPosition{0.0f, 0.0f, 0.0f};
                        float gradScaleU = 0.0f;
                        float gradScaleV = 0.0f;
                        float gradEta = 0.0f;
                        float gradBeta = 0.0f;
                        float3 gradTangentU{0.0f, 0.0f, 0.0f};
                        float3 gradTangentV{0.0f, 0.0f, 0.0f};
                        uint32_t primitiveIndex = kInvalidIndex;
                    };

                    float transmittance = 1.0f;
                    float3 accumulatedAlphaDerivativeOverOneMinusAlphaStartPoint =
                            float3{0.0f, 0.0f, 0.0f};
                    float3 accumulatedAlphaDerivativeOverOneMinusAlphaEndPoint =
                            float3{0.0f, 0.0f, 0.0f};
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

                            const float alphaGeomOccluder = worldHit.alphaGeom;
                            const float alphaEffective = occluderSurfel.opacity * alphaGeomOccluder;
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

                            const float dAlphaEffectiveDScaleU =
                                    (2.0f * betaScale * uOcc * uOcc * alphaEffective) /
                                    (scaleU * oneMinusRadiusSquared);

                            const float dAlphaEffectiveDScaleV =
                                    (2.0f * betaScale * vOcc * vOcc * alphaEffective) /
                                    (scaleV * oneMinusRadiusSquared);

                            const float dAlphaEffectiveDEta =
                                    alphaGeomOccluder;

                            const float dAlphaEffectiveDBeta =
                                    betaScale * sycl::log(oneMinusRadiusSquared) * alphaEffective;

                            float3 gradTangentUOcc = float3(0.0f);
                            float3 gradTangentVOcc = float3(0.0f);

                            const float nDotD = dot(occluderNormal, rayDirection);
                            if (sycl::fabs(nDotD) > 1e-8f) {
                                const float3 hitMinusSp =
                                        worldHit.hitPositionW - occluderSurfel.position;
                                const float3 aOcc =
                                        occluderSurfel.position - yPosition;
                                const float nDotA = dot(occluderNormal, aOcc);

                                const float invNDotD = 1.0f / nDotD;
                                const float invNDotDSquared = invNDotD * invNDotD;

                                const float3 qOcc =
                                        ((cross(occluderNormal, aOcc) * nDotD) -
                                         (nDotA * cross(occluderNormal, rayDirection))) *
                                        invNDotDSquared;

                                const float3 duDRotation =
                                        qOcc * (dot(rayDirection, tangentU) / scaleU) +
                                        (cross(tangentU, hitMinusSp) / scaleU);

                                const float3 dvDRotation =
                                        qOcc * (dot(rayDirection, tangentV) / scaleV) +
                                        (cross(tangentV, hitMinusSp) / scaleV);

                                const float3 dAlphaEffectiveDRotation =
                                        occluderSurfel.opacity * (
                                            dAlphaGeomDu * duDRotation +
                                            dAlphaGeomDv * dvDRotation);

                                const float invOneMinusAlpha = 1.0f / oneMinusAlpha;
                                const float3 rotationDerivativeZeta =
                                        dAlphaEffectiveDRotation * invOneMinusAlpha;

                                gradTangentUOcc =
                                        cross(rotationDerivativeZeta, occluderSurfel.tanU);

                                gradTangentVOcc =
                                        cross(rotationDerivativeZeta, occluderSurfel.tanV);
                            }

                            accumulatedAlphaDerivativeOverOneMinusAlphaStartPoint +=
                                    dAlphaEffectiveDx * (1.0f / oneMinusAlpha);

                            accumulatedAlphaDerivativeOverOneMinusAlphaEndPoint +=
                                    dAlphaEffectiveDy * (1.0f / oneMinusAlpha);

                            if (storedOccluderCount < kMaxSplatEventsPerRay) {
                                const float invOneMinusAlpha = 1.0f / oneMinusAlpha;

                                occluderDerivatives[storedOccluderCount].gradPosition =
                                        dAlphaEffectiveDspi * invOneMinusAlpha;
                                occluderDerivatives[storedOccluderCount].gradScaleU =
                                        dAlphaEffectiveDScaleU * invOneMinusAlpha;
                                occluderDerivatives[storedOccluderCount].gradScaleV =
                                        dAlphaEffectiveDScaleV * invOneMinusAlpha;
                                occluderDerivatives[storedOccluderCount].gradEta =
                                        dAlphaEffectiveDEta * invOneMinusAlpha;
                                occluderDerivatives[storedOccluderCount].gradBeta =
                                        dAlphaEffectiveDBeta * invOneMinusAlpha;
                                occluderDerivatives[storedOccluderCount].gradTangentU =
                                        gradTangentUOcc;
                                occluderDerivatives[storedOccluderCount].gradTangentV =
                                        gradTangentVOcc;
                                occluderDerivatives[storedOccluderCount].primitiveIndex =
                                        worldHit.primitiveIndex;
                                storedOccluderCount++;
                            }

                            scalarWeightWithoutTauAndGeometric *= qNullInv;
                            albedoWeightWithoutTauAndGeometricX *= qNullInv;
                            scalarWeightWithoutTauAndGeometricEtaY *= qNullInv;
                            scalarWeightWithoutTauAndGeometricBetaY *= qNullInv;
                        }
                    }

                    const float3 dTransmittanceDx =
                            -transmittance * accumulatedAlphaDerivativeOverOneMinusAlphaStartPoint;

                    const float3 dTransmittanceDy =
                            -transmittance * accumulatedAlphaDerivativeOverOneMinusAlphaEndPoint;

                    const float3 gradientWrtXPosition =
                            scalarWeightWithoutTauAndGeometric *
                            (geometricTermXY * dTransmittanceDx + transmittance * dGeometricTermDx);

                    const float3 gradientWrtYPosition =
                            scalarWeightWithoutTauAndGeometric *
                            (geometricTermXY * dTransmittanceDy + transmittance * dGeometricTermDy);

                    const float3 xContribution = gradientWrtXPosition * invSpp;
                    const float3 yContribution = gradientWrtYPosition * invSpp;

                    // ---------------------------------------------------------------------
                    // code xSurface (= math Y): scale gradients
                    // ---------------------------------------------------------------------
                    float scaleXContributionU = 0.0f;
                    float scaleXContributionV = 0.0f;
                    if (scaleXU > 1e-12f && scaleXV > 1e-12f) {
                        scaleXContributionU =
                                dot(gradientWrtXPosition, uX * surfelX.tanU) * invSpp;
                        scaleXContributionV =
                                dot(gradientWrtXPosition, vX * surfelX.tanV) * invSpp;
                    }

                    // ---------------------------------------------------------------------
                    // code xSurface (= math Y): tangent gradients
                    // ---------------------------------------------------------------------
                    float3 tanUXContribution = float3{0.0f, 0.0f, 0.0f};
                    float3 tanVXContribution = float3{0.0f, 0.0f, 0.0f};

                    if (scaleXU > 1e-12f && scaleXV > 1e-12f) {
                        tanUXContribution += (scaleXU * uX) * gradientWrtXPosition * invSpp;
                        tanVXContribution += (scaleXV * vX) * gradientWrtXPosition * invSpp;
                    }

                    const float3 rawCrossX = cross(surfelX.tanU, surfelX.tanV);
                    const float rawCrossXLength = length(rawCrossX);
                    if (rawCrossXLength > 1e-8f) {
                        const float3 rawNormalX = rawCrossX / rawCrossXLength;

                        const float orientationSignX =
                                dot(rawNormalX, xState.orientedNormal) >= 0.0f ? 1.0f : -1.0f;

                        const float3 dGeometricTermDStartNormal =
                                directionXToY * (cosineAtY / distanceSquared);

                        const float3 gradientWrtOrientedNormalX =
                                scalarWeightWithoutTauAndGeometric *
                                transmittance *
                                dGeometricTermDStartNormal;

                        const float3 gradientWrtRawNormalX =
                                orientationSignX * gradientWrtOrientedNormalX;

                        const float3 gradientProjectedToRawNormalXTangent =
                                gradientWrtRawNormalX -
                                rawNormalX * dot(rawNormalX, gradientWrtRawNormalX);

                        const float3 gradientWrtCrossX =
                                gradientProjectedToRawNormalXTangent / rawCrossXLength;

                        tanUXContribution +=
                                cross(surfelX.tanV, gradientWrtCrossX) * invSpp;

                        tanVXContribution +=
                                cross(gradientWrtCrossX, surfelX.tanU) * invSpp;
                    }

                    // ---------------------------------------------------------------------
                    // code xSurface (= math Y): albedo gradients
                    // ---------------------------------------------------------------------
                    const float3 albedoXContribution =
                            transmittance *
                            geometricTermXY *
                            albedoWeightWithoutTauAndGeometricX *
                            invSpp;

                    // ---------------------------------------------------------------------
                    // code ySurface (= math Z): scale gradients
                    // ---------------------------------------------------------------------
                    float scaleYContributionU = 0.0f;
                    float scaleYContributionV = 0.0f;

                    if (scaleYU > 1e-12f && scaleYV > 1e-12f) {
                        scaleYContributionU =
                                dot(gradientWrtYPosition, uYLocal * surfelY.tanU) * invSpp;
                        scaleYContributionV =
                                dot(gradientWrtYPosition, vYLocal * surfelY.tanV) * invSpp;
                    }

                    // ---------------------------------------------------------------------
                    // code ySurface (= math Z): tangent gradients
                    // ---------------------------------------------------------------------
                    float3 tanUYContribution = float3{0.0f, 0.0f, 0.0f};
                    float3 tanVYContribution = float3{0.0f, 0.0f, 0.0f};

                    if (scaleYU > 1e-12f && scaleYV > 1e-12f) {
                        tanUYContribution += (scaleYU * uYLocal) * gradientWrtYPosition * invSpp;
                        tanVYContribution += (scaleYV * vYLocal) * gradientWrtYPosition * invSpp;
                    }

                    const float3 rawCrossY = cross(surfelY.tanU, surfelY.tanV);
                    const float rawCrossYLength = length(rawCrossY);

                    if (rawCrossYLength > 1e-8f) {
                        const float3 rawNormalY = rawCrossY / rawCrossYLength;

                        const float orientationSignY =
                                dot(rawNormalY, yState.orientedNormal) >= 0.0f ? 1.0f : -1.0f;

                        const float3 dGeometricTermDEndNormal =
                                (-directionXToY) * (cosineAtX / distanceSquared);

                        const float3 gradientWrtOrientedNormalY =
                                scalarWeightWithoutTauAndGeometric *
                                transmittance *
                                dGeometricTermDEndNormal;

                        const float3 gradientWrtRawNormalY =
                                orientationSignY * gradientWrtOrientedNormalY;

                        const float3 gradientProjectedToRawNormalYTangent =
                                gradientWrtRawNormalY -
                                rawNormalY * dot(rawNormalY, gradientWrtRawNormalY);

                        const float3 gradientWrtCrossY =
                                gradientProjectedToRawNormalYTangent / rawCrossYLength;

                        tanUYContribution +=
                                cross(surfelY.tanV, gradientWrtCrossY) * invSpp;

                        tanVYContribution +=
                                cross(gradientWrtCrossY, surfelY.tanU) * invSpp;
                    }

                    // ---------------------------------------------------------------------
                    // code ySurface (= math Z): eta / beta gradients
                    // No local albedo gradient here since BSDF is evaluated at code xSurface.
                    // ---------------------------------------------------------------------
                    const float etaYContribution =
                            transmittance *
                            geometricTermXY *
                            scalarWeightWithoutTauAndGeometricEtaY *
                            invSpp;

                    float betaYContribution = 0.0f;
                    if (oneMinusRadiusSquaredYLocal > 1e-8f) {
                        betaYContribution =
                                transmittance *
                                geometricTermXY *
                                scalarWeightWithoutTauAndGeometricBetaY *
                                invSpp;
                    }

                    SurfelGradientRecord xRecord{};
                    xRecord.primitiveIndex = xPrimitiveIndex;
                    xRecord.gradPositionX = xContribution.x();
                    xRecord.gradPositionY = xContribution.y();
                    xRecord.gradPositionZ = xContribution.z();

                    xRecord.gradScaleU = scaleXContributionU;
                    xRecord.gradScaleV = scaleXContributionV;

                    xRecord.gradTangentUX = tanUXContribution.x();
                    xRecord.gradTangentUY = tanUXContribution.y();
                    xRecord.gradTangentUZ = tanUXContribution.z();

                    xRecord.gradTangentVX = tanVXContribution.x();
                    xRecord.gradTangentVY = tanVXContribution.y();
                    xRecord.gradTangentVZ = tanVXContribution.z();

                    xRecord.gradAlbedoR = albedoXContribution.x();
                    xRecord.gradAlbedoG = albedoXContribution.y();
                    xRecord.gradAlbedoB = albedoXContribution.z();

                    gradientRecords[xRecordIndex] = xRecord;

                    SurfelGradientRecord yRecord{};
                    yRecord.primitiveIndex = yPrimitiveIndex;
                    yRecord.gradPositionX = yContribution.x();
                    yRecord.gradPositionY = yContribution.y();
                    yRecord.gradPositionZ = yContribution.z();

                    yRecord.gradScaleU = scaleYContributionU;
                    yRecord.gradScaleV = scaleYContributionV;

                    yRecord.gradTangentUX = tanUYContribution.x();
                    yRecord.gradTangentUY = tanUYContribution.y();
                    yRecord.gradTangentUZ = tanUYContribution.z();

                    yRecord.gradTangentVX = tanVYContribution.x();
                    yRecord.gradTangentVY = tanVYContribution.y();
                    yRecord.gradTangentVZ = tanVYContribution.z();

                    yRecord.gradEta = etaYContribution;
                    yRecord.gradBeta = betaYContribution;

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

                        const float3 occluderPositionContribution =
                                occluderScale * occluderDerivative.gradPosition;
                        const float3 occluderTanUContribution =
                                occluderScale * occluderDerivative.gradTangentU;
                        const float3 occluderTanVContribution =
                                occluderScale * occluderDerivative.gradTangentV;

                        SurfelGradientRecord occluderRecord{};
                        occluderRecord.primitiveIndex = occluderDerivative.primitiveIndex;
                        occluderRecord.gradPositionX = occluderPositionContribution.x();
                        occluderRecord.gradPositionY = occluderPositionContribution.y();
                        occluderRecord.gradPositionZ = occluderPositionContribution.z();

                        occluderRecord.gradScaleU =
                                occluderScale * occluderDerivative.gradScaleU;
                        occluderRecord.gradScaleV =
                                occluderScale * occluderDerivative.gradScaleV;

                        occluderRecord.gradEta =
                                occluderScale * occluderDerivative.gradEta;
                        occluderRecord.gradBeta =
                                occluderScale * occluderDerivative.gradBeta;

                        occluderRecord.gradTangentUX = occluderTanUContribution.x();
                        occluderRecord.gradTangentUY = occluderTanUContribution.y();
                        occluderRecord.gradTangentUZ = occluderTanUContribution.z();

                        occluderRecord.gradTangentVX = occluderTanVContribution.x();
                        occluderRecord.gradTangentVY = occluderTanVContribution.y();
                        occluderRecord.gradTangentVZ = occluderTanVContribution.z();

                        gradientRecords[occluderRecordIndex] = occluderRecord;
                    }
                });
        }).wait();
    }

    static void launchAdjointDirectLightContributionKernel(
        RenderPackage &pkg,
        uint32_t directLightEventCount,
        uint32_t baseOffset) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto &sensor = pkg.sensors.front();
        DirectLightGradientEvent *directLightEvents =
                pkg.intermediates.directLightEvents;
        SurfelGradientRecord *gradientRecords =
                pkg.intermediates.gradientRecords;

        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class directLightEventTag>(
                sycl::range<1>(directLightEventCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex = globalId[0];

                    static constexpr uint32_t recordsPerEvent = 1u + kMaxSplatEventsPerRay;
                    const uint32_t eventRecordBase = baseOffset + recordsPerEvent * eventIndex;
                    const uint32_t xRecordIndex = eventRecordBase + 0u;

                    for (uint32_t recordOffset = 0u; recordOffset < recordsPerEvent; ++recordOffset) {
                        SurfelGradientRecord invalidRecord{};
                        invalidRecord.primitiveIndex = kInvalidIndex;
                        gradientRecords[eventRecordBase + recordOffset] = invalidRecord;
                    }

                    const DirectLightGradientEvent eventRecord =
                            directLightEvents[eventIndex];

                    const uint32_t xPrimitiveIndex = eventRecord.surface.primitiveIndex;
                    const Point &surfelX = scene.points[xPrimitiveIndex];

                    const ReconstructedSurfelState xState =
                            reconstructSurfelState(surfelX, eventRecord.surface);

                    const float3 lightPosition = eventRecord.lightPositionWorld;
                    const float3 lightNormal = eventRecord.lightNormalWorld;
                    const float3 lightRadiance = eventRecord.lightRadiance;
                    const float lightPdfArea = eventRecord.lightPdfArea;

                    if (lightPdfArea <= 1e-20f) {
                        return;
                    }

                    const float3 vectorXToLight = lightPosition - xState.position;
                    const float distanceSquared = dot(vectorXToLight, vectorXToLight);
                    if (distanceSquared <= 1e-12f) {
                        return;
                    }

                    const float distance = sycl::sqrt(distanceSquared);
                    const float3 directionXToLight = vectorXToLight / distance;

                    const float cosineAtLight = dot(lightNormal, -directionXToLight);
                    if (cosineAtLight <= 1e-6f) {
                        return;
                    }

                    const float geometricTermXL = computeGeometricTermValue(
                        xState.position,
                        lightPosition,
                        xState.orientedNormal,
                        lightNormal);

                    if (geometricTermXL <= 1e-20f) {
                        return;
                    }

                    const float3 dGeometricTermDx = computeGeometricTermGradientWrtStartpoint(
                        xState.position,
                        lightPosition,
                        xState.orientedNormal,
                        lightNormal);

                    const float alphaGeomX = eventRecord.surface.alphaGeom;
                    const float alphaX = alphaGeomX * surfelX.opacity;
                    const float3 surfelBsdf =
                            surfelX.alpha_r * surfelX.albedo * M_1_PIf;

                    const float inverseLightPdfArea = 1.0f / lightPdfArea;

                    // Prefix transported adjoint weight up to X.
                    const float3 pathWeight = eventRecord.xPathThroughput;

                    // Excludes alpha_X, tau(X,L), and G(X,L).
                    const float3 transportWithoutAlphaTauAndGeometric =
                            lightRadiance * surfelBsdf * inverseLightPdfArea;

                    const float scalarWeightWithoutAlphaTauAndGeometric =
                            dot(pathWeight, transportWithoutAlphaTauAndGeometric);

                    // -----------------------------------------------------------------
                    // Local X weights
                    // -----------------------------------------------------------------
                    const float scaleXU = surfelX.scale.x();
                    const float scaleXV = surfelX.scale.y();
                    const float uX = eventRecord.surface.uv.x();
                    const float vX = eventRecord.surface.uv.y();
                    const float radiusSquaredX = uX * uX + vX * vX;
                    const float oneMinusRadiusSquaredX = 1.0f - radiusSquaredX;

                    const float3 albedoWeightWithoutAlphaTauAndGeometric =
                            (pathWeight * lightRadiance) *
                            (alphaX * surfelX.alpha_r * M_1_PIf * inverseLightPdfArea);

                    const float scalarWeightWithoutAlphaTauAndGeometricEtaX =
                            dot(pathWeight,
                                transportWithoutAlphaTauAndGeometric * alphaGeomX);

                    float scalarWeightWithoutAlphaTauAndGeometricBetaX = 0.0f;
                    float betaScaleX = 0.0f;
                    if (oneMinusRadiusSquaredX > 1e-8f) {
                        betaScaleX = 4.0f * sycl::exp(surfelX.beta);
                        const float dAlphaGeomXDBeta =
                                betaScaleX * sycl::log(oneMinusRadiusSquaredX) * alphaGeomX;

                        scalarWeightWithoutAlphaTauAndGeometricBetaX =
                                dot(pathWeight,
                                    transportWithoutAlphaTauAndGeometric *
                                    (surfelX.opacity * dAlphaGeomXDBeta));
                    }

                    struct OccluderDerivative {
                        float3 gradPosition{0.0f};
                        float gradScaleU = 0.0f;
                        float gradScaleV = 0.0f;
                        float gradEta = 0.0f;
                        float gradBeta = 0.0f;
                        float3 gradTangentU{0.0f};
                        float3 gradTangentV{0.0f};
                        uint32_t primitiveIndex = kInvalidIndex;
                    };

                    float transmittance = 1.0f;
                    float3 accumulatedAlphaDerivativeOverOneMinusAlphaStartPoint = float3{0.0f};
                    OccluderDerivative occluderDerivatives[kMaxSplatEventsPerRay];
                    uint32_t storedOccluderCount = 0u;

                    // -----------------------------------------------------------------
                    // Differentiate attenuation on the open shadow segment X -> light
                    // -----------------------------------------------------------------
                    {
                        const float distanceEpsilon = 1e-4f;

                        Ray shadowRay{};
                        shadowRay.origin =
                                xState.position + xState.orientedNormal * distanceEpsilon;
                        shadowRay.direction = directionXToLight;
                        shadowRay.normal = xState.orientedNormal;

                        const float targetDistance = distance;
                        const float3 xPosition = xState.position;
                        const float3 lightSamplePosition = lightPosition;
                        const float3 xMinusLight = xPosition - lightSamplePosition;

                        while (true) {
                            WorldHit worldHit{};
                            intersectScene(
                                shadowRay,
                                &worldHit,
                                scene,
                                SurfelIntersectMode::FirstHit);

                            if (!worldHit.hit) {
                                break;
                            }

                            const float hitDistance =
                                    length(worldHit.hitPositionW - xState.position);
                            if (hitDistance >= targetDistance - distanceEpsilon) {
                                break;
                            }

                            buildIntersectionNormal(scene, worldHit);
                            const auto &instance = scene.instances[worldHit.instanceIndex];

                            if (instance.geometryType != GeometryType::PointCloud) {
                                shadowRay.origin =
                                        worldHit.hitPositionW + shadowRay.direction * 1e-4f;
                                continue;
                            }

                            const Point &occluderSurfel =
                                    scene.points[worldHit.primitiveIndex];

                            float3 occluderNormal =
                                    normalize(cross(occluderSurfel.tanU, occluderSurfel.tanV));
                            const bool hitBackside =
                                    dot(occluderNormal, -shadowRay.direction) < 0.0f;
                            if (hitBackside) {
                                occluderNormal = -occluderNormal;
                            }

                            const float alphaGeomOccluder = worldHit.alphaGeom;
                            const float alphaEffective =
                                    occluderSurfel.opacity * alphaGeomOccluder;
                            const float oneMinusAlpha = 1.0f - alphaEffective;
                            if (oneMinusAlpha <= 1e-8f) {
                                break;
                            }

                            transmittance *= oneMinusAlpha;
                            shadowRay.origin =
                                    worldHit.hitPositionW + shadowRay.direction * 1e-4f;

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

                            const float denominator = dot(occluderNormal, xMinusLight);
                            if (sycl::fabs(denominator) <= 1e-8f) {
                                continue;
                            }

                            const float lambdaOccluder =
                                    dot(occluderNormal, occluderSurfel.position - lightSamplePosition) /
                                    denominator;
                            const float inverseDenominator = 1.0f / denominator;

                            const float3 dUiDx =
                                    lambdaOccluder * (
                                        localBasisU -
                                        occluderNormal *
                                        (dot(xMinusLight, localBasisU) * inverseDenominator));

                            const float3 dViDx =
                                    lambdaOccluder * (
                                        localBasisV -
                                        occluderNormal *
                                        (dot(xMinusLight, localBasisV) * inverseDenominator));

                            const float3 dUiDspi =
                                    occluderNormal *
                                    (dot(xMinusLight, tangentU) / scaleU) * inverseDenominator -
                                    localBasisU;

                            const float3 dViDspi =
                                    occluderNormal *
                                    (dot(xMinusLight, tangentV) / scaleV) * inverseDenominator -
                                    localBasisV;

                            const float radiusSquaredOcc = uOcc * uOcc + vOcc * vOcc;
                            const float oneMinusRadiusSquaredOcc = 1.0f - radiusSquaredOcc;
                            if (oneMinusRadiusSquaredOcc <= 1e-8f) {
                                continue;
                            }

                            const float betaScaleOcc =
                                    4.0f * sycl::exp(occluderSurfel.beta);

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

                            const float dAlphaEffectiveDScaleU =
                                    (2.0f * betaScaleOcc * uOcc * uOcc * alphaEffective) /
                                    (scaleU * oneMinusRadiusSquaredOcc);

                            const float dAlphaEffectiveDScaleV =
                                    (2.0f * betaScaleOcc * vOcc * vOcc * alphaEffective) /
                                    (scaleV * oneMinusRadiusSquaredOcc);

                            const float dAlphaEffectiveDEta =
                                    alphaGeomOccluder;

                            const float dAlphaEffectiveDBeta =
                                    betaScaleOcc *
                                    sycl::log(oneMinusRadiusSquaredOcc) *
                                    alphaEffective;

                            float3 gradTangentUOcc = float3(0.0f);
                            float3 gradTangentVOcc = float3(0.0f);

                            const float nDotD = dot(occluderNormal, directionXToLight);
                            if (sycl::fabs(nDotD) > 1e-8f) {
                                const float3 hitMinusSp =
                                        worldHit.hitPositionW - occluderSurfel.position;
                                const float3 aOcc =
                                        occluderSurfel.position - lightSamplePosition;
                                const float nDotA = dot(occluderNormal, aOcc);

                                const float invNDotD = 1.0f / nDotD;
                                const float invNDotDSquared = invNDotD * invNDotD;

                                const float3 qOcc =
                                        ((cross(occluderNormal, aOcc) * nDotD) -
                                         (nDotA * cross(occluderNormal, directionXToLight))) *
                                        invNDotDSquared;

                                const float3 duDRotation =
                                        qOcc * (dot(directionXToLight, tangentU) / scaleU) +
                                        (cross(tangentU, hitMinusSp) / scaleU);

                                const float3 dvDRotation =
                                        qOcc * (dot(directionXToLight, tangentV) / scaleV) +
                                        (cross(tangentV, hitMinusSp) / scaleV);

                                const float3 dAlphaEffectiveDRotation =
                                        occluderSurfel.opacity * (
                                            dAlphaGeomDu * duDRotation +
                                            dAlphaGeomDv * dvDRotation);

                                const float invOneMinusAlpha = 1.0f / oneMinusAlpha;
                                const float3 rotationDerivativeZeta =
                                        dAlphaEffectiveDRotation * invOneMinusAlpha;

                                gradTangentUOcc =
                                        cross(rotationDerivativeZeta, occluderSurfel.tanU);

                                gradTangentVOcc =
                                        cross(rotationDerivativeZeta, occluderSurfel.tanV);
                            }

                            accumulatedAlphaDerivativeOverOneMinusAlphaStartPoint +=
                                    dAlphaEffectiveDx * (1.0f / oneMinusAlpha);

                            if (storedOccluderCount < kMaxSplatEventsPerRay) {
                                const float invOneMinusAlpha = 1.0f / oneMinusAlpha;

                                occluderDerivatives[storedOccluderCount].gradPosition =
                                        dAlphaEffectiveDspi * invOneMinusAlpha;
                                occluderDerivatives[storedOccluderCount].gradScaleU =
                                        dAlphaEffectiveDScaleU * invOneMinusAlpha;
                                occluderDerivatives[storedOccluderCount].gradScaleV =
                                        dAlphaEffectiveDScaleV * invOneMinusAlpha;
                                occluderDerivatives[storedOccluderCount].gradEta =
                                        dAlphaEffectiveDEta * invOneMinusAlpha;
                                occluderDerivatives[storedOccluderCount].gradBeta =
                                        dAlphaEffectiveDBeta * invOneMinusAlpha;
                                occluderDerivatives[storedOccluderCount].gradTangentU =
                                        gradTangentUOcc;
                                occluderDerivatives[storedOccluderCount].gradTangentV =
                                        gradTangentVOcc;
                                occluderDerivatives[storedOccluderCount].primitiveIndex =
                                        worldHit.primitiveIndex;
                                storedOccluderCount++;
                            }
                        }
                    }

                    // -----------------------------------------------------------------
                    // Segment X -> light gradient wrt startpoint world position
                    // -----------------------------------------------------------------
                    const float3 dTransmittanceDx =
                            -transmittance *
                            accumulatedAlphaDerivativeOverOneMinusAlphaStartPoint;

                    const float3 gradientWrtWorldHitPositionX =
                            scalarWeightWithoutAlphaTauAndGeometric *
                            alphaX *
                            (geometricTermXL * dTransmittanceDx +
                             transmittance * dGeometricTermDx);

                    // -----------------------------------------------------------------
                    // Start-surface X local gradients
                    // -----------------------------------------------------------------
                    float3 positionContribution = float3(0.0f);
                    float scaleXContributionU = 0.0f;
                    float scaleXContributionV = 0.0f;
                    float3 tanUXContribution = float3(0.0f);
                    float3 tanVXContribution = float3(0.0f);

                    const float etaXContribution =
                            transmittance *
                            geometricTermXL *
                            scalarWeightWithoutAlphaTauAndGeometricEtaX *
                            invSpp;

                    const float betaXContribution =
                            transmittance *
                            geometricTermXL *
                            scalarWeightWithoutAlphaTauAndGeometricBetaX *
                            invSpp;

                    const float3 albedoXContribution =
                            transmittance *
                            geometricTermXL *
                            albedoWeightWithoutAlphaTauAndGeometric *
                            invSpp;

                    const float3 dGeometricTermDStartNormal =
                            directionXToLight * (cosineAtLight / distanceSquared);

                    if (eventRecord.useImplicitRayHitJacobian != 0u) {
                        // -------------------------------------------------------------
                        // Camera-attached X:
                        // 1) local alpha_X via implicit hit
                        // 2) segment X->light via implicit translation Jacobian
                        // 3) segment tangent contribution via start normal + moved-hit
                        // -------------------------------------------------------------
                        if (oneMinusRadiusSquaredX > 1e-8f &&
                            scaleXU > 1e-12f &&
                            scaleXV > 1e-12f) {
                            /*
                            UVPositionJacobian uvPositionJacobian{};
                            uvPositionJacobian =
                                    computeDuvDSurfelTranslationJacobianForImplicitRayHit(
                                        surfelX.tanU,
                                        surfelX.tanV,
                                        xState.orientedNormal,
                                        eventRecord.surface.incomingDirection,
                                        scaleXU,
                                        scaleXV);

                            const float3 dUvDPosition =
                                    uX * uvPositionJacobian.du_d_surfel_translation +
                                    vX * uvPositionJacobian.dv_d_surfel_translation;

                            const float factor =
                                    (-2.0f * betaScaleX * alphaGeomX) /
                                    oneMinusRadiusSquaredX;

                            const float3 dAlphaGeomDPosition = factor * dUvDPosition;
                            const float3 dAlphaEffectiveDPosition =
                                    surfelX.opacity * dAlphaGeomDPosition;

                            const float scalarWeightNoLocalAlpha =
                                    scalarWeightWithoutAlphaTauAndGeometric *
                                    transmittance *
                                    geometricTermXL;


                            positionContribution +=
                                    dAlphaEffectiveDPosition *
                                    scalarWeightNoLocalAlpha *
                                    invSpp;


                            const float dAlphaEffectiveDScaleU =
                                    surfelX.opacity *
                                    (2.0f * betaScaleX * uX * uX * alphaGeomX) /
                                    (scaleXU * oneMinusRadiusSquaredX);

                            const float dAlphaEffectiveDScaleV =
                                    surfelX.opacity *
                                    (2.0f * betaScaleX * vX * vX * alphaGeomX) /
                                    (scaleXV * oneMinusRadiusSquaredX);

                            scaleXContributionU +=
                                    dAlphaEffectiveDScaleU *
                                    scalarWeightNoLocalAlpha *
                                    invSpp;

                            scaleXContributionV +=
                                    dAlphaEffectiveDScaleV *
                                    scalarWeightNoLocalAlpha *
                                    invSpp;

                            const float dAlphaGeomDuX = factor * uX;
                            const float dAlphaGeomDvX = factor * vX;

                            const float3 vectorCameraToX = xState.position - sensor.camera.pos;
                            const float3 rayDirectionCamera = normalize(vectorCameraToX);
                            const float nDotCamera = dot(xState.orientedNormal, rayDirectionCamera);

                            if (sycl::fabs(nDotCamera) > 1e-8f) {
                                const float3 xMinusSp =
                                        (uX * scaleXU) * surfelX.tanU +
                                        (vX * scaleXV) * surfelX.tanV;

                                const float3 a =
                                        (xState.position - sensor.camera.pos) - xMinusSp;

                                const float nDotA = dot(xState.orientedNormal, a);
                                const float invNDotD = 1.0f / nDotCamera;
                                const float invNDotDSquared = invNDotD * invNDotD;

                                const float3 q =
                                        ((cross(xState.orientedNormal, a) * nDotCamera) -
                                         (nDotA * cross(xState.orientedNormal, rayDirectionCamera))) *
                                        invNDotDSquared;

                                const float3 duDRotation =
                                        q * (dot(rayDirectionCamera, surfelX.tanU) / scaleXU) +
                                        (cross(surfelX.tanU, xMinusSp) / scaleXU);

                                const float3 dvDRotation =
                                        q * (dot(rayDirectionCamera, surfelX.tanV) / scaleXV) +
                                        (cross(surfelX.tanV, xMinusSp) / scaleXV);

                                const float3 dAlphaGeomDRotation =
                                        dAlphaGeomDuX * duDRotation +
                                        dAlphaGeomDvX * dvDRotation;

                                const float3 dAlphaEffectiveDRotation =
                                        surfelX.opacity * dAlphaGeomDRotation;

                                const float3 rotationGradientZeta =
                                        dAlphaEffectiveDRotation *
                                        scalarWeightNoLocalAlpha *
                                        invSpp;

                                tanUXContribution +=
                                        cross(rotationGradientZeta, surfelX.tanU);
                                tanVXContribution +=
                                        cross(rotationGradientZeta, surfelX.tanV);
                            }
                            */
                        }

                        // Segment translation via implicit hit jacobian
                        {
                            const float3x3 hitPointJacobian =
                                    planeHitPointIntersectionJacobian(
                                        eventRecord.surface.incomingDirection,
                                        xState.orientedNormal);

                            positionContribution +=
                                    transpose(hitPointJacobian) *
                                    gradientWrtWorldHitPositionX *
                                    invSpp;
                        }

                        // Segment tangent contribution via start normal + moved-hit
                        {
                            const float3 primaryRayDirection =
                                    eventRecord.surface.incomingDirection;
                            const float nDotD = dot(xState.orientedNormal, primaryRayDirection);

                            const float3 rawCross = cross(surfelX.tanU, surfelX.tanV);
                            const float rawCrossLength = length(rawCross);

                            if (sycl::fabs(nDotD) > 1e-8f && rawCrossLength > 1e-8f) {
                                const float3 rawNormal = rawCross / rawCrossLength;
                                const float orientationSign =
                                        dot(rawNormal, xState.orientedNormal) >= 0.0f ? 1.0f : -1.0f;

                                const float3 pMinusX = surfelX.position - xState.position;

                                const float3 gradientWrtOrientedNormalFromMovedHit =
                                        pMinusX *
                                        (dot(primaryRayDirection, gradientWrtWorldHitPositionX) / nDotD);

                                const float3 gradientWrtOrientedNormalExplicit =
                                        scalarWeightWithoutAlphaTauAndGeometric *
                                        alphaX *
                                        transmittance *
                                        dGeometricTermDStartNormal;

                                const float3 gradientWrtOrientedNormalX =
                                        gradientWrtOrientedNormalFromMovedHit +
                                        gradientWrtOrientedNormalExplicit;

                                const float3 gradientWrtRawNormal =
                                        orientationSign * gradientWrtOrientedNormalX;

                                const float3 gradientProjectedToRawNormalTangent =
                                        gradientWrtRawNormal -
                                        rawNormal * dot(rawNormal, gradientWrtRawNormal);

                                const float3 gradientWrtCross =
                                        gradientProjectedToRawNormalTangent / rawCrossLength;

                                tanUXContribution +=
                                        cross(surfelX.tanV, gradientWrtCross) * invSpp;

                                tanVXContribution +=
                                        cross(gradientWrtCross, surfelX.tanU) * invSpp;
                            }
                        }
                    } else {
                        // -------------------------------------------------------------
                        // Material-point X:
                        // world point moves directly with (s_p, s_u, s_v, t_u, t_v)
                        // -------------------------------------------------------------
                        //positionContribution += gradientWrtWorldHitPositionX * invSpp;

                        if (scaleXU > 1e-12f && scaleXV > 1e-12f) {
                            scaleXContributionU +=
                                    dot(gradientWrtWorldHitPositionX, uX * surfelX.tanU) * invSpp;
                            scaleXContributionV +=
                                    dot(gradientWrtWorldHitPositionX, vX * surfelX.tanV) * invSpp;

                            tanUXContribution +=
                                    (scaleXU * uX) * gradientWrtWorldHitPositionX * invSpp;
                            tanVXContribution +=
                                    (scaleXV * vX) * gradientWrtWorldHitPositionX * invSpp;
                        }

                        const float3 rawCross = cross(surfelX.tanU, surfelX.tanV);
                        const float rawCrossLength = length(rawCross);
                        if (rawCrossLength > 1e-8f) {
                            const float3 rawNormal = rawCross / rawCrossLength;
                            const float orientationSign =
                                    dot(rawNormal, xState.orientedNormal) >= 0.0f ? 1.0f : -1.0f;

                            const float3 gradientWrtOrientedNormalX =
                                    scalarWeightWithoutAlphaTauAndGeometric *
                                    alphaX *
                                    transmittance *
                                    dGeometricTermDStartNormal;

                            const float3 gradientWrtRawNormal =
                                    orientationSign * gradientWrtOrientedNormalX;

                            const float3 gradientProjectedToRawNormalTangent =
                                    gradientWrtRawNormal -
                                    rawNormal * dot(rawNormal, gradientWrtRawNormal);

                            const float3 gradientWrtCross =
                                    gradientProjectedToRawNormalTangent / rawCrossLength;

                            tanUXContribution +=
                                    cross(surfelX.tanV, gradientWrtCross) * invSpp;

                            tanVXContribution +=
                                    cross(gradientWrtCross, surfelX.tanU) * invSpp;
                        }
                    }

                    SurfelGradientRecord xRecord{};
                    xRecord.primitiveIndex = xPrimitiveIndex;
                    xRecord.gradPositionX = positionContribution.x();
                    xRecord.gradPositionY = positionContribution.y();
                    xRecord.gradPositionZ = positionContribution.z();

                    xRecord.gradScaleU = scaleXContributionU;
                    xRecord.gradScaleV = scaleXContributionV;

                    xRecord.gradTangentUX = tanUXContribution.x();
                    xRecord.gradTangentUY = tanUXContribution.y();
                    xRecord.gradTangentUZ = tanUXContribution.z();

                    xRecord.gradTangentVX = tanVXContribution.x();
                    xRecord.gradTangentVY = tanVXContribution.y();
                    xRecord.gradTangentVZ = tanVXContribution.z();

                    //xRecord.gradEta = etaXContribution;
                    //xRecord.gradBeta = betaXContribution;

                    xRecord.gradAlbedoR = albedoXContribution.x();
                    xRecord.gradAlbedoG = albedoXContribution.y();
                    xRecord.gradAlbedoB = albedoXContribution.z();

                    gradientRecords[xRecordIndex] = xRecord;

                    // -----------------------------------------------------------------
                    // Occluder gradients on the open segment X -> light
                    // -----------------------------------------------------------------
                    const float scalarWeightWithoutTau =
                            scalarWeightWithoutAlphaTauAndGeometric *
                            alphaX *
                            geometricTermXL;

                    const float occluderScale =
                            -transmittance *
                            scalarWeightWithoutTau *
                            invSpp;

                    for (uint32_t occluderIndex = 0u;
                         occluderIndex < storedOccluderCount;
                         ++occluderIndex) {
                        const uint32_t occluderRecordIndex =
                                eventRecordBase + 1u + occluderIndex;

                        const OccluderDerivative &occluderDerivative =
                                occluderDerivatives[occluderIndex];

                        const float3 occluderPositionContribution =
                                occluderScale * occluderDerivative.gradPosition;
                        const float3 occluderTanUContribution =
                                occluderScale * occluderDerivative.gradTangentU;
                        const float3 occluderTanVContribution =
                                occluderScale * occluderDerivative.gradTangentV;

                        SurfelGradientRecord occluderRecord{};
                        occluderRecord.primitiveIndex =
                                occluderDerivative.primitiveIndex;

                        occluderRecord.gradPositionX =
                                occluderPositionContribution.x();
                        occluderRecord.gradPositionY =
                                occluderPositionContribution.y();
                        occluderRecord.gradPositionZ =
                                occluderPositionContribution.z();

                        occluderRecord.gradScaleU =
                                occluderScale * occluderDerivative.gradScaleU;
                        occluderRecord.gradScaleV =
                                occluderScale * occluderDerivative.gradScaleV;

                        occluderRecord.gradEta =
                                occluderScale * occluderDerivative.gradEta;
                        occluderRecord.gradBeta =
                                occluderScale * occluderDerivative.gradBeta;

                        occluderRecord.gradTangentUX =
                                occluderTanUContribution.x();
                        occluderRecord.gradTangentUY =
                                occluderTanUContribution.y();
                        occluderRecord.gradTangentUZ =
                                occluderTanUContribution.z();

                        occluderRecord.gradTangentVX =
                                occluderTanVContribution.x();
                        occluderRecord.gradTangentVY =
                                occluderTanVContribution.y();
                        occluderRecord.gradTangentVZ =
                                occluderTanVContribution.z();

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

                    const uint32_t primitiveIndex = gradientRecord.primitiveIndex;

                    atomicAddFloat(
                        gradients.gradPosition[primitiveIndex].x(),
                        gradientRecord.gradPositionX);
                    atomicAddFloat(
                        gradients.gradPosition[primitiveIndex].y(),
                        gradientRecord.gradPositionY);
                    atomicAddFloat(
                        gradients.gradPosition[primitiveIndex].z(),
                        gradientRecord.gradPositionZ);

                    atomicAddFloat(
                        gradients.gradScale[primitiveIndex].x(),
                        gradientRecord.gradScaleU);
                    atomicAddFloat(
                        gradients.gradScale[primitiveIndex].y(),
                        gradientRecord.gradScaleV);

                    atomicAddFloat(
                        gradients.gradTanU[primitiveIndex].x(),
                        gradientRecord.gradTangentUX);
                    atomicAddFloat(
                        gradients.gradTanU[primitiveIndex].y(),
                        gradientRecord.gradTangentUY);
                    atomicAddFloat(
                        gradients.gradTanU[primitiveIndex].z(),
                        gradientRecord.gradTangentUZ);

                    atomicAddFloat(
                        gradients.gradTanV[primitiveIndex].x(),
                        gradientRecord.gradTangentVX);
                    atomicAddFloat(
                        gradients.gradTanV[primitiveIndex].y(),
                        gradientRecord.gradTangentVY);
                    atomicAddFloat(
                        gradients.gradTanV[primitiveIndex].z(),
                        gradientRecord.gradTangentVZ);

                    atomicAddFloat(
                        gradients.gradOpacity[primitiveIndex],
                        gradientRecord.gradEta);
                    atomicAddFloat(
                        gradients.gradBeta[primitiveIndex],
                        gradientRecord.gradBeta);

                    atomicAddFloat(
                        gradients.gradAlbedo[primitiveIndex].x(),
                        gradientRecord.gradAlbedoR);
                    atomicAddFloat(
                        gradients.gradAlbedo[primitiveIndex].y(),
                        gradientRecord.gradAlbedoG);
                    atomicAddFloat(
                        gradients.gradAlbedo[primitiveIndex].z(),
                        gradientRecord.gradAlbedoB);
                });
        }).wait();
    }

    struct DistortionHit {
        uint32_t primitiveIndex = 0u;
        float3 hitPositionW{0.0f};
        float3 rayOrigin0{0.0f};
        float3 rayDir0{0.0f};

        float ai = 0.0f; // a_i = eta_i * alphaGeom_i
        float wi = 0.0f; // w_i = T_{i-1} * a_i
        float Tprev = 1.0f; // T_{i-1}
        float zi = 0.0f; // scalar forward depth
        float alphaGeom = 0.0f;
        float u = 0.0f;
        float v = 0.0f;
    };

    struct AlphaKernelEval {
        float value = 0.0f; // alpha_geom
        float dValue_dU = 0.0f; // d alpha_geom / du
        float dValue_dV = 0.0f; // d alpha_geom / dv
        float dValue_dBeta = 0.0f; // d alpha_geom / d beta
    };

    SYCL_EXTERNAL inline AlphaKernelEval evaluateAlphaKernelAndDerivatives(
        const Point &surfel,
        float u,
        float v) {
        AlphaKernelEval out{};

        const float r2 = u * u + v * v;
        if (r2 >= 1.0f) {
            return out;
        }

        const float oneMinusRadiusSquared = 1.0f - r2;
        const float sSafe = sycl::fmax(oneMinusRadiusSquared, 1e-8f);

        // b(beta) = 4 * exp(beta)
        const float betaScale = 4.0f * sycl::exp(surfel.beta);

        // alpha_geom = (1 - r^2)^b
        const float alphaGeom = sycl::pow(sSafe, betaScale);

        out.value = alphaGeom;

        // d alpha_geom / du = -(2 b u / (1-r^2)) * alpha_geom
        out.dValue_dU = -2.0f * betaScale * u * alphaGeom / sSafe;

        // d alpha_geom / dv = -(2 b v / (1-r^2)) * alpha_geom
        out.dValue_dV = -2.0f * betaScale * v * alphaGeom / sSafe;

        // d alpha_geom / d beta = b * log(1-r^2) * alpha_geom
        out.dValue_dBeta = betaScale * sycl::log(sSafe) * alphaGeom;

        return out;
    }

    void launchDepthDistortionBackwardKernel(RenderPackage &pkg, uint32_t cameraIndex) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;

        SensorGPU sensor = pkg.sensors[cameraIndex];
        auto &grads = pkg.gradients;

        const std::uint32_t imageWidth = sensor.camera.width;
        const std::uint32_t imageHeight = sensor.camera.height;
        const std::uint32_t pixelCount = imageWidth * imageHeight;

        queue.submit([&](sycl::handler &cgh) {
            const uint64_t renderSeed = pkg.random.seed;

            cgh.parallel_for<class DepthDistortionBackwardKernel>(
                sycl::range<1>(pixelCount),
                [=](sycl::id<1> tid) {
                    constexpr uint32_t kMaxHits = 32u;
                    constexpr float kDenomEps = 1e-8f;

                    const uint32_t pixelIndex = tid[0];
                    const uint32_t pixelX = pixelIndex % imageWidth;
                    const uint32_t pixelY = pixelIndex / imageWidth;

                    // This must be dL / d(distortion[pixel]), not the forward distortion itself.
                    const float pixelAdjoint =
                            sensor.depthDistortionAdjointBuffer[pixelIndex];

                    if (pixelAdjoint == 0.0f) {
                        return;
                    }

                    // ---------------------------------------------------------
                    // Recreate the same primary ray as in launchCameraGatherKernel
                    // ---------------------------------------------------------
                    const uint64_t directionSeed =
                            rng::makeSeed(renderSeed, pixelIndex, cameraIndex, rng::kStreamGather, 0u);
                    rng::Xorshift128 rng(directionSeed);

                    const float jitterX = rng.nextFloat() - 0.5f;
                    const float jitterY = rng.nextFloat() - 0.5f;

                    Ray primaryRay = makePrimaryRayFromPixelJitteredFov(
                        sensor.camera,
                        static_cast<float>(pixelX),
                        static_cast<float>(pixelY),
                        jitterX,
                        jitterY);

                    const float3 rayOrigin0 = primaryRay.origin;
                    const float3 rayDir0 = primaryRay.direction;

                    // ---------------------------------------------------------
                    // Forward retrace: collect ordered surfel hits
                    // ---------------------------------------------------------
                    DistortionHit hits[kMaxHits];
                    uint32_t hitCount = 0u;
                    float transmittance = 1.0f;

                    for (uint32_t traversalIndex = 0u;
                         traversalIndex < kMaxHits;
                         ++traversalIndex) {
                        WorldHit worldHit{};
                        intersectScene(primaryRay, &worldHit, scene, SurfelIntersectMode::FirstHit);

                        if (!worldHit.hit) {
                            break;
                        }

                        buildIntersectionNormal(scene, worldHit);
                        const auto &instance = scene.instances[worldHit.instanceIndex];

                        if (instance.geometryType == GeometryType::PointCloud) {
                            const Point &surfel = scene.points[worldHit.primitiveIndex];

                            const float2 uv = phiInverse(worldHit.hitPositionW, surfel);
                            const float u = uv.x();
                            const float v = uv.y();

                            // Must match forward kernel/support exactly
                            const AlphaKernelEval kernelEval =
                                    evaluateAlphaKernelAndDerivatives(surfel, u, v);

                            const float alphaGeom = worldHit.alphaGeom; // forward-consistent
                            const float ai = surfel.opacity * alphaGeom;
                            const float wi = transmittance * ai;
                            const float zi =
                                    dot(worldHit.hitPositionW - sensor.camera.pos, sensor.camera.forward);

                            DistortionHit rec{};
                            rec.primitiveIndex = worldHit.primitiveIndex;
                            rec.hitPositionW = worldHit.hitPositionW;
                            rec.rayOrigin0 = rayOrigin0;
                            rec.rayDir0 = rayDir0;
                            rec.ai = ai;
                            rec.wi = wi;
                            rec.Tprev = transmittance;
                            rec.zi = zi;
                            rec.alphaGeom = alphaGeom;
                            rec.u = u;
                            rec.v = v;
                            hits[hitCount++] = rec;

                            transmittance *= (1.0f - ai);
                            primaryRay.origin =
                                    worldHit.hitPositionW + primaryRay.direction * 1e-8f;
                            continue;
                        }

                        if (instance.geometryType == GeometryType::Mesh) {
                            break;
                        }
                    }

                    if (hitCount <= 1u) {
                        return;
                    }

                    // ---------------------------------------------------------
                    // Hit-level adjoints for
                    // d = sum_{i,j} w_i w_j |z_i - z_j|
                    // ---------------------------------------------------------
                    float barW[kMaxHits];
                    float barZ[kMaxHits];
                    float barA[kMaxHits];

                    for (uint32_t i = 0u; i < hitCount; ++i) {
                        barW[i] = 0.0f;
                        barZ[i] = 0.0f;
                        barA[i] = 0.0f;
                    }

                    for (uint32_t i = 0u; i < hitCount; ++i) {
                        for (uint32_t j = i + 1u; j < hitCount; ++j) {
                            const float zi = hits[i].zi;
                            const float zj = hits[j].zi;
                            const float wi = hits[i].wi;
                            const float wj = hits[j].wi;

                            const float diff = zi - zj;
                            const float absDiff = sycl::fabs(diff);
                            const float signDiff = (diff >= 0.0f) ? 1.0f : -1.0f;

                            // contribution = 2 * wi * wj * |zi - zj|
                            const float pairScale = 2.0f * pixelAdjoint;

                            barW[i] += pairScale * wj * absDiff;
                            barW[j] += pairScale * wi * absDiff;

                            barZ[i] += pairScale * wi * wj * signDiff;
                            barZ[j] -= pairScale * wi * wj * signDiff;
                        }
                    }

                    // ---------------------------------------------------------
                    // Reverse through compositing
                    // wi = Tprev_i * ai
                    // Tnext = Tprev * (1 - ai)
                    // ---------------------------------------------------------
                    float barTnext = 0.0f;

                    for (int i = int(hitCount) - 1; i >= 0; --i) {
                        const float ai = hits[i].ai;
                        const float Tprev = hits[i].Tprev;

                        float barAi = 0.0f;
                        float barTprev = 0.0f;

                        // wi = Tprev * ai
                        barAi += Tprev * barW[i];
                        barTprev += ai * barW[i];

                        // Tnext = Tprev * (1 - ai)
                        barAi += -Tprev * barTnext;
                        barTprev += (1.0f - ai) * barTnext;

                        barA[i] = barAi;
                        barTnext = barTprev;
                    }

                    // ---------------------------------------------------------
                    // Chain to surfel parameters
                    // ---------------------------------------------------------
                    for (uint32_t i = 0u; i < hitCount; ++i) {
                        const DistortionHit &hit = hits[i];
                        const Point &surfel = scene.points[hit.primitiveIndex];

                        const float3 p = surfel.position;
                        const float3 tu = surfel.tanU;
                        const float3 tv = surfel.tanV;
                        const float su = surfel.scale.x();
                        const float sv = surfel.scale.y();
                        const float eta = surfel.opacity;

                        const float3 x = hit.hitPositionW;
                        const float3 q = x - p;

                        const AlphaKernelEval kernelEval =
                                evaluateAlphaKernelAndDerivatives(surfel, hit.u, hit.v);

                        // a_i = eta * alphaGeom
                        const float barAlphaGeom = barA[i] * eta;
                        const float barEta = barA[i] * hit.alphaGeom;

                        float barU = barAlphaGeom * kernelEval.dValue_dU;
                        float barV = barAlphaGeom * kernelEval.dValue_dV;
                        const float barBeta = barAlphaGeom * kernelEval.dValue_dBeta;

                        // z_i = dot(x_i - camPos, camForward)
                        float3 barX = barZ[i] * sensor.camera.forward;

                        // u = dot(q, tu) / su
                        // v = dot(q, tv) / sv
                        float3 barQ(0.0f);

                        barQ += (barU / su) * tu;
                        barQ += (barV / sv) * tv;

                        float3 barTu = (barU / su) * q;
                        float3 barTv = (barV / sv) * q;

                        float barSu = -barU * hit.u / su;
                        float barSv = -barV * hit.v / sv;

                        // q = x - p
                        barX += barQ;
                        float3 barP = -barQ;

                        // x = rayOrigin0 + lambda * rayDir0
                        // lambda = n·(p - rayOrigin0) / (n·rayDir0)
                        const float3 nRaw = cross(tu, tv);
                        const float nRawLen = sycl::sqrt(dot(nRaw, nRaw));

                        if (nRawLen > kDenomEps) {
                            const float3 n = nRaw / nRawLen;
                            const float denom = dot(n, rayDir0);

                            if (sycl::fabs(denom) > kDenomEps) {
                                const float barLambda = dot(barX, rayDir0);

                                // d lambda / d p = n / (n·d)
                                barP += (barLambda / denom) * n;

                                // d lambda / d n = (p - x) / (n·d)
                                float3 barN = (barLambda / denom) * (p - x);

                                // n = normalize(nRaw)
                                const float3 barNRaw =
                                        (barN - n * dot(n, barN)) / nRawLen;

                                // nRaw = tu x tv
                                barTu += cross(tv, barNRaw);
                                barTv += cross(barNRaw, tu);
                            }
                        }

                        // -------------------------------------------------
                        // Atomic accumulation into global gradient buffers
                        // -------------------------------------------------
                        atomicAddFloat3(grads.gradPosition[hit.primitiveIndex], barP);
                        atomicAddFloat3(grads.gradTanU[hit.primitiveIndex], barTu);
                        atomicAddFloat3(grads.gradTanV[hit.primitiveIndex], barTv);
                        atomicAddFloat2(grads.gradScale[hit.primitiveIndex], float2(barSu, barSv));
                        atomicAddFloat(grads.gradOpacity[hit.primitiveIndex], barEta);
                        atomicAddFloat(grads.gradBeta[hit.primitiveIndex], barBeta);
                    }
                });
        });

        queue.wait();
    }

    static inline bool isZero3(const float3 &v) {
        return sycl::fabs(v.x()) < 1e-12f &&
               sycl::fabs(v.y()) < 1e-12f &&
               sycl::fabs(v.z()) < 1e-12f;
    }

    static inline float3 loadFloat4Rgb(const float4 &v) {
        return float3{v.x(), v.y(), v.z()};
    }

    void launchNormalFromDepthAdjointKernel(RenderPackage &pkg, uint32_t cameraIndex) {
        auto &queue = pkg.queue;
        auto &sensor = pkg.sensors[cameraIndex];

        const uint32_t width = sensor.width;
        const uint32_t height = sensor.height;
        const uint32_t pixelCount = width * height;

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class NormalFromDepthAdjointKernel>(
                sycl::range<1>(pixelCount),
                [=](sycl::id<1> tid) {
                    const uint32_t pixelIndex = tid[0];
                    const uint32_t x = pixelIndex % width;
                    const uint32_t y = pixelIndex / width;

                    if (x == 0u || y == 0u || x + 1u >= width || y + 1u >= height) {
                        return;
                    }

                    const float4 gN4 = sensor.normalFromDepthAdjointBuffer[pixelIndex];
                    const float3 gN = loadFloat4Rgb(gN4);
                    if (isZero3(gN)) {
                        return;
                    }

                    const uint32_t idxL = y * width + (x - 1u);
                    const uint32_t idxR = y * width + (x + 1u);
                    const uint32_t idxU = (y - 1u) * width + x;
                    const uint32_t idxD = (y + 1u) * width + x;

                    const float zL = sensor.medianDepthBuffer[idxL];
                    const float zR = sensor.medianDepthBuffer[idxR];
                    const float zU = sensor.medianDepthBuffer[idxU];
                    const float zD = sensor.medianDepthBuffer[idxD];

                    if (zL <= 0.0f || zR <= 0.0f || zU <= 0.0f || zD <= 0.0f) {
                        return;
                    }

                    const float3 pL = reconstructWorldPositionFromDepthCenter(sensor.camera, x - 1u, y, zL);
                    const float3 pR = reconstructWorldPositionFromDepthCenter(sensor.camera, x + 1u, y, zR);
                    const float3 pU = reconstructWorldPositionFromDepthCenter(sensor.camera, x, y - 1u, zU);
                    const float3 pD = reconstructWorldPositionFromDepthCenter(sensor.camera, x, y + 1u, zD);

                    const float3 dx = pR - pL;
                    const float3 dy = pD - pU;

                    const float3 m = cross(dx, dy);
                    const float mLen = length(m);
                    if (mLen <= 1e-12f) {
                        return;
                    }

                    float3 n = m / mLen;
                    float sign = 1.0f;
                    if (dot(n, -sensor.camera.forward) < 0.0f) {
                        sign = -1.0f;
                    }

                    const float3 projected =
                            gN - n * dot(n, gN);

                    const float3 gM =
                            sign * (projected / mLen);

                    const float3 gPR = cross(dy, gM);
                    const float3 gPL = -gPR;
                    const float3 gPD = cross(gM, dx);
                    const float3 gPU = -gPD;

                    auto rayDirForPixel = [&](uint32_t px, uint32_t py) -> float3 {
                        Ray ray = makePrimaryRayFromPixelJitteredFov(
                            sensor.camera,
                            static_cast<float>(px),
                            static_cast<float>(py),
                            0.0f,
                            0.0f);
                        return ray.direction;
                    };

                    const float3 dL = rayDirForPixel(x - 1u, y);
                    const float3 dR = rayDirForPixel(x + 1u, y);
                    const float3 dU = rayDirForPixel(x, y - 1u);
                    const float3 dD = rayDirForPixel(x, y + 1u);

                    const float denomL = dot(sensor.camera.forward, dL);
                    const float denomR = dot(sensor.camera.forward, dR);
                    const float denomU = dot(sensor.camera.forward, dU);
                    const float denomD = dot(sensor.camera.forward, dD);

                    if (sycl::fabs(denomL) <= 1e-8f ||
                        sycl::fabs(denomR) <= 1e-8f ||
                        sycl::fabs(denomU) <= 1e-8f ||
                        sycl::fabs(denomD) <= 1e-8f) {
                        return;
                    }

                    const float gZL = dot(gPL, dL / denomL);
                    const float gZR = dot(gPR, dR / denomR);
                    const float gZU = dot(gPU, dU / denomU);
                    const float gZD = dot(gPD, dD / denomD);

                    atomicAddFloat(sensor.medianDepthAdjointBuffer[idxL], gZL);
                    atomicAddFloat(sensor.medianDepthAdjointBuffer[idxR], gZR);
                    atomicAddFloat(sensor.medianDepthAdjointBuffer[idxU], gZU);
                    atomicAddFloat(sensor.medianDepthAdjointBuffer[idxD], gZD);
                });
        }).wait();
    }

    void launchNormalConsistencyBackwardKernel(RenderPackage &pkg, uint32_t cameraIndex) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &sensor = pkg.sensors[cameraIndex];
        auto &gradients = pkg.gradients;

        const uint32_t width = sensor.width;
        const uint32_t height = sensor.height;
        const uint32_t pixelCount = width * height;

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class NormalConsistencyBackwardKernel>(
                sycl::range<1>(pixelCount),
                [=](sycl::id<1> tid) {
                    const uint32_t pixelIndex = tid[0];
                    const uint32_t pixelX = pixelIndex % width;
                    const uint32_t pixelY = pixelIndex / width;

                    const float3 gVisibleNormal =
                            float3{
                                sensor.visibleNormalAdjointBuffer[pixelIndex].x(),
                                sensor.visibleNormalAdjointBuffer[pixelIndex].y(),
                                sensor.visibleNormalAdjointBuffer[pixelIndex].z()
                            };

                    const float gMedianDepth =
                            sensor.medianDepthAdjointBuffer[pixelIndex];

                    const bool useVisibleNormal =
                            sycl::fabs(gVisibleNormal.x()) > 1e-12f ||
                            sycl::fabs(gVisibleNormal.y()) > 1e-12f ||
                            sycl::fabs(gVisibleNormal.z()) > 1e-12f;

                    const bool useMedianDepth =
                            sycl::fabs(gMedianDepth) > 1e-12f;

                    if (!useVisibleNormal && !useMedianDepth) {
                        return;
                    }

                    Ray primaryRay = makePrimaryRayFromPixelJitteredFov(
                        sensor.camera,
                        static_cast<float>(pixelX),
                        static_cast<float>(pixelY),
                        0.0f,
                        0.0f);

                    float transmittance = 1.0f;
                    float accumulatedCompositeWeight = 0.0f;
                    static constexpr uint32_t maxTraversalRays = 32u;

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

                        if (instance.geometryType == GeometryType::PointCloud) {
                            const uint32_t primitiveIndex = worldHit.primitiveIndex;
                            const Point &surfel = scene.points[primitiveIndex];

                            float3 orientedNormal =
                                    normalize(cross(surfel.tanU, surfel.tanV));

                            const bool hitBackside =
                                    dot(orientedNormal, -primaryRay.direction) < 0.0f;
                            if (hitBackside) {
                                orientedNormal = -orientedNormal;
                            }

                            const float alphaEff = surfel.opacity * worldHit.alphaGeom;
                            const float wi = transmittance * alphaEff;

                            const bool isMedian =
                                    (accumulatedCompositeWeight + wi) >= 0.5f;

                            if (isMedian) {
                                float3 gradPosition = float3{0.0f, 0.0f, 0.0f};
                                float3 gradTanU = float3{0.0f, 0.0f, 0.0f};
                                float3 gradTanV = float3{0.0f, 0.0f, 0.0f};

                                // -------------------------------------------------
                                // 1) Visible normal adjoint -> tangent frame
                                // -------------------------------------------------
                                if (useVisibleNormal) {
                                    const float3 rawCross = cross(surfel.tanU, surfel.tanV);
                                    const float rawCrossLen = length(rawCross);
                                    if (rawCrossLen > 1e-8f) {
                                        const float3 rawNormal = rawCross / rawCrossLen;

                                        const float orientationSign =
                                                dot(rawNormal, orientedNormal) >= 0.0f ? 1.0f : -1.0f;

                                        const float3 gradRawNormal =
                                                orientationSign * gVisibleNormal;

                                        const float3 gradProjected =
                                                gradRawNormal -
                                                rawNormal * dot(rawNormal, gradRawNormal);

                                        const float3 gradCross =
                                                gradProjected / rawCrossLen;

                                        gradTanU += cross(surfel.tanV, gradCross);
                                        gradTanV += cross(gradCross, surfel.tanU);
                                    }
                                }

                                // -------------------------------------------------
                                // 2) Median depth adjoint -> hit point geometry
                                // -------------------------------------------------
                                if (useMedianDepth) {
                                    const float3 gradWrtHitPoint =
                                            gMedianDepth * sensor.camera.forward;

                                    const float3x3 hitPointJacobian =
                                            planeHitPointIntersectionJacobian(
                                                primaryRay.direction,
                                                orientedNormal);

                                    gradPosition +=
                                            transpose(hitPointJacobian) * gradWrtHitPoint;

                                    const float3 rawCross = cross(surfel.tanU, surfel.tanV);
                                    const float rawCrossLen = length(rawCross);

                                    const float nDotD = dot(orientedNormal, primaryRay.direction);

                                    if (rawCrossLen > 1e-8f && sycl::fabs(nDotD) > 1e-8f) {
                                        const float3 rawNormal = rawCross / rawCrossLen;
                                        const float orientationSign =
                                                dot(rawNormal, orientedNormal) >= 0.0f ? 1.0f : -1.0f;

                                        const float3 pMinusX =
                                                surfel.position - worldHit.hitPositionW;

                                        const float3 gradOrientedNormal =
                                                pMinusX *
                                                (dot(primaryRay.direction, gradWrtHitPoint) / nDotD);

                                        const float3 gradRawNormal =
                                                orientationSign * gradOrientedNormal;

                                        const float3 gradProjected =
                                                gradRawNormal -
                                                rawNormal * dot(rawNormal, gradRawNormal);

                                        const float3 gradCross =
                                                gradProjected / rawCrossLen;

                                        gradTanU += cross(surfel.tanV, gradCross);
                                        gradTanV += cross(gradCross, surfel.tanU);
                                    }
                                }

                                atomicAddFloat(gradients.gradPosition[primitiveIndex].x(), gradPosition.x());
                                atomicAddFloat(gradients.gradPosition[primitiveIndex].y(), gradPosition.y());
                                atomicAddFloat(gradients.gradPosition[primitiveIndex].z(), gradPosition.z());

                                atomicAddFloat(gradients.gradTanU[primitiveIndex].x(), gradTanU.x());
                                atomicAddFloat(gradients.gradTanU[primitiveIndex].y(), gradTanU.y());
                                atomicAddFloat(gradients.gradTanU[primitiveIndex].z(), gradTanU.z());

                                atomicAddFloat(gradients.gradTanV[primitiveIndex].x(), gradTanV.x());
                                atomicAddFloat(gradients.gradTanV[primitiveIndex].y(), gradTanV.y());
                                atomicAddFloat(gradients.gradTanV[primitiveIndex].z(), gradTanV.z());
                                return;
                            }

                            accumulatedCompositeWeight += wi;
                            transmittance *= (1.0f - alphaEff);
                            primaryRay.origin =
                                    worldHit.hitPositionW + primaryRay.direction * 1e-8f;
                            continue;
                        }

                        if (instance.geometryType == GeometryType::Mesh) {
                            const float wi = transmittance;
                            const bool isMedian =
                                    (accumulatedCompositeWeight + wi) >= 0.5f;

                            // Mesh is not optimized in this point-gradient path
                            (void) isMedian;
                            return;
                        }

                        return;
                    }
                });
        }).wait();
    }

    void adjointContributionKernels(
        RenderPackage &pkg,
        uint32_t measurementEventCount,
        uint32_t measurementTwoPointEventCount,
        uint32_t cameraAttachedBridgeEventCount,
        uint32_t recursiveBridgeEventCount,
        uint32_t directLightEventCount,
        uint32_t cameraIndex) {
        const GradientRecordRanges ranges = makeGradientRecordRanges(
            measurementEventCount,
            measurementTwoPointEventCount,
            cameraAttachedBridgeEventCount,
            recursiveBridgeEventCount,
            directLightEventCount);

        if (ranges.totalCount > pkg.intermediates.maxGradientRecordCount) {
            throw std::runtime_error("gradient record scratch buffer too small");
        }

        Log::PA_DEBUG(
            "Event counts: measurement={}, measurementTwoPoint={}, cameraAttachedBridge={}, recursiveBridge={}",
            measurementEventCount,
            measurementTwoPointEventCount,
            cameraAttachedBridgeEventCount,
            recursiveBridgeEventCount);

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

        if (recursiveBridgeEventCount > 0) {
            ScopedTimer timer("recursiveBridgeEvent", spdlog::level::debug);
            recursiveBridgeEvent(
                pkg,
                recursiveBridgeEventCount,
                ranges.recursiveBridgeOffset);
        }


        if (directLightEventCount > 0) {
            launchAdjointDirectLightContributionKernel(
                pkg,
                directLightEventCount,
                ranges.directLightOffset);
        }

        if (ranges.totalCount > 0) {
            ScopedTimer timer("reduceSurfelGradientRecords", spdlog::level::debug);
            reduceSurfelGradientRecords(
                pkg,
                ranges.totalCount);
        }
    }
}
