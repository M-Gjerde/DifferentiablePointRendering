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

                    constexpr uint32_t maxInlineNullTraversals = 32;

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

                            const PointCloudSurfaceRecord currentSurface =
                                    makePointCloudSurfaceRecord(worldHit, currentRayState, scene);


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
                            float segmentLocalPdfFromStoredVertex = 1.0f;

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
                                    measurementEvent.xPathThroughput = currentRayState.pathThroughput / qReflect;

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
                                            const uint64_t lightSampleSeed = rng::makeSeed(
                                                renderSeed,
                                                currentRayState.pathId,
                                                spp,
                                                rng::kStreamDirectLight,
                                                currentRayState.traversalIndex * 1315423911u + shadowRaySample);

                                            rng::Xorshift128 directLightSampleRng(lightSampleSeed);

                                            const auto lightSample =
                                                    sampleMeshAreaLight(scene, directLightSampleRng);
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
                                                constexpr uint32_t maxShadowTraversals = 16u;
                                                const float targetDistance = lightDistance;
                                                Ray shadowRay{};
                                                shadowRay.origin =
                                                        worldHit.hitPositionW + orientedNormal *
                                                        distanceEpsilon;
                                                shadowRay.direction = lightDirection;
                                                shadowRay.normal = orientedNormal;
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

                                                    // Self intersection:
                                                    if (shadowHit.primitiveIndex == worldHit.primitiveIndex) {
                                                        shadowRay.origin =
                                                                shadowHit.hitPositionW + shadowRay.direction *
                                                                distanceEpsilon;
                                                        continue;
                                                    }

                                                    // Point-cloud surfels are semi-transparent attenuators.
                                                    // Do not reject the sample here; just continue marching.
                                                    if (hitInstance.geometryType == GeometryType::PointCloud) {
                                                        shadowRay.origin =
                                                                shadowHit.hitPositionW + shadowRay.direction *
                                                                distanceEpsilon;
                                                        transmission *= (
                                                            (1.0f - shadowHit.alphaGeom * scene.points[shadowHit.
                                                                 primitiveIndex].opacity));
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
                                                        (lightSample.pdfArea * lightSample.pdfSelectLight * qReflect) *
                                                        invSampleCount;
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

                                    segmentLocalPdfFromStoredVertex = surfel.scale.x() * surfel.scale.y();
                                }

                                if (previousAdjointStage.valid && currentRayState.bounceIndex > 0) {
                                    MaterialVertexGradientEvent materialVertexEvent{};
                                    materialVertexEvent.surface = currentSurface;
                                    // Adjoint at p1 we have:
                                    materialVertexEvent.adjointWeightAtVertex =
                                            currentRayState.pathThroughput *
                                            currentRayState.transmission / (qReflect);
                                    materialVertexEvent.pathId = currentRayState.pathId;
                                    materialVertexEvent.bounceIndex = currentRayState.bounceIndex;
                                    appendEventAtomic(
                                        intermediates.countMaterialVertexEvents,
                                        intermediates.materialVertexEvents,
                                        intermediates.maxMaterialVertexEventCount,
                                        materialVertexEvent);
                                }

                                // -------------------------------------------------------------
                                // Direct-light material edge samples: current material vertex -> light
                                // -------------------------------------------------------------
                                if (previousAdjointStage.valid &&
                                    currentRayState.bounceIndex > 0) {
                                    // Geometric gradient bettwen XYZ where Z is a light source we have two edge events for Y
                                    // Firs tbetween X and Y, which is adjoint path weight p0.
                                    // Second is between YZ which is adjoint path weight p1.
                                    // Start with XY path we only have End point gradients w.r.t to Y:

                                    MaterialEdgeGradientEvent materialEdgeEventXY{};
                                    materialEdgeEventXY.startSurface = previousAdjointStage.current.surface;
                                    materialEdgeEventXY.endSurface = currentSurface;
                                    materialEdgeEventXY.sampledEdgeThroughput =
                                            previousAdjointStage.current.pathThroughput *
                                            previousAdjointStage.current.transmissionFromPrevious *
                                            previousAdjointStage.current.bsdfAlpha /
                                            (qReflect * segmentAreaPdfFromStoredVertex *
                                             segmentLocalPdfFromStoredVertex);


                                    materialEdgeEventXY.isDirectLightSample = false;
                                    materialEdgeEventXY.isEndPointOnly = true;
                                    materialEdgeEventXY.pathId = currentRayState.pathId;
                                    materialEdgeEventXY.startBounceIndex = currentRayState.bounceIndex;


                                    appendEventAtomic(
                                        intermediates.countMaterialEdgeEvents,
                                        intermediates.materialEdgeEvents,
                                        intermediates.maxMaterialEdgeEventCount,
                                        materialEdgeEventXY);


                                    // second event:
                                    // Then make a new event on the YZ leg wit a direct light sample but differentiate now with respect to start point.

                                    const ReconstructedSurfelState &startState = reconstructSurfelState(
                                        surfel, currentSurface);
                                    const float invSampleCount =
                                            1.0f / static_cast<float>(settings.numAdjointPathShadowRays);

                                    for (uint32_t shadowRaySample = 0u;
                                         shadowRaySample < settings.numAdjointPathShadowRays;
                                         ++shadowRaySample) {
                                        const uint64_t lightSampleSeed =
                                                rng::makeSeed(
                                                    renderSeed,
                                                    currentRayState.pathId,
                                                    spp,
                                                    rng::kStreamDirectLight,
                                                    currentRayState.traversalIndex * 1315423911u +
                                                    shadowRaySample +
                                                    0x7f4a7c15u);

                                        rng::Xorshift128 directLightSampleRng(lightSampleSeed);

                                        const auto lightSample =
                                                sampleMeshAreaLight(scene, directLightSampleRng);

                                        if (!lightSample.valid) {
                                            continue;
                                        }

                                        const uint32_t lightPrimitiveIndex =
                                                lightSample.surface.primitiveIndex;

                                        if (lightPrimitiveIndex == kInvalidIndex) {
                                            continue;
                                        }

                                        const Point &lightSurfel = scene.points[lightPrimitiveIndex];
                                        const ReconstructedSurfelState lightState = reconstructSurfelState(
                                            lightSurfel, lightSample.surface);
                                        const float3 lightVector = lightState.position - startState.position;
                                        const float lightDistanceSquared = dot(lightVector, lightVector);
                                        if (lightDistanceSquared <= 1.0e-12f) {
                                            continue;
                                        }
                                        const float lightDistance = sycl::sqrt(lightDistanceSquared);
                                        const float3 lightDirection = lightVector / lightDistance;
                                        const float cosineAtStart = sycl::fmax(
                                            0.0f, dot(lightDirection, startState.orientedNormal));
                                        if (cosineAtStart <= 1.0e-8f) {
                                            continue;
                                        }

                                        constexpr float distanceEpsilon = 1.0e-4f;
                                        constexpr uint32_t maxShadowTraversals = 16u;
                                        Ray shadowRay{};
                                        shadowRay.origin =
                                                startState.position + lightDirection * distanceEpsilon;
                                        shadowRay.direction = lightDirection;
                                        shadowRay.normal = startState.orientedNormal;
                                        bool blockedByOpaqueGeometry = false;
                                        float shadowTransmission = 1.0f;
                                        for (uint32_t traversalIndex = 0u;
                                             traversalIndex < maxShadowTraversals;
                                             ++traversalIndex) {
                                            WorldHit shadowHit{};
                                            intersectScene(
                                                shadowRay,
                                                &shadowHit,
                                                scene,
                                                SurfelIntersectMode::FirstHit);
                                            if (!shadowHit.hit) {
                                                break;
                                            }
                                            const float3 hitVector = shadowHit.hitPositionW - startState.position;
                                            const float hitDistance = sycl::sqrt(dot(hitVector, hitVector));
                                            if (hitDistance >= lightDistance - distanceEpsilon) {
                                                break;
                                            }

                                            buildIntersectionNormal(scene, shadowHit);
                                            const InstanceRecord &hitInstance = scene.instances[shadowHit.
                                                instanceIndex];
                                            if (hitInstance.geometryType == GeometryType::Mesh) {
                                                blockedByOpaqueGeometry = true;
                                                break;
                                            }
                                            // Self intersection
                                            if (shadowHit.primitiveIndex == currentSurface.primitiveIndex) {
                                                shadowRay.origin =
                                                        shadowHit.hitPositionW + shadowRay.direction * distanceEpsilon;
                                                continue;
                                            }
                                            if (shadowHit.primitiveIndex == lightPrimitiveIndex) {
                                                shadowRay.origin =
                                                        shadowHit.hitPositionW + shadowRay.direction * distanceEpsilon;
                                                continue;
                                            }
                                            if (hitInstance.geometryType == GeometryType::PointCloud) {
                                                const Point &shadowSurfel = scene.points[shadowHit.primitiveIndex];
                                                shadowTransmission *= 1.0f - shadowHit.alphaGeom * shadowSurfel.opacity;
                                                shadowRay.origin =
                                                        shadowHit.hitPositionW + shadowRay.direction * distanceEpsilon;
                                                continue;
                                            }
                                            blockedByOpaqueGeometry = true;
                                            break;
                                        }
                                        if (blockedByOpaqueGeometry) {
                                            continue;
                                        }
                                        const float3 directLightRadiance =
                                                lightSample.flux / (M_PIf * lightSample.totalAreaWorld);
                                        MaterialEdgeGradientEvent materialDirectLightEdgeEvent{};
                                        materialDirectLightEdgeEvent.startSurface = currentSurface;
                                        materialDirectLightEdgeEvent.endSurface = lightSample.surface;
                                        // This is a direct area-sampled edge Y -> Z_light.
                                        // sampledEdgeThroughput contains the adjoint path weight, local alpha_Y,
                                        // BSDF at Y, qReflect, light area-selection pdfs, and the direct-light sample count.
                                        // materialEdgeGradientEvent multiplies by G_YZ, tau_YZ, and directLightRadiance.
                                        materialDirectLightEdgeEvent.sampledEdgeThroughput =
                                                currentRayState.pathThroughput *
                                                currentRayState.transmission *
                                                (alpha * surfelBsdf) /
                                                (qReflect * lightSample.pdfArea * lightSample.pdfSelectLight) *
                                                invSampleCount;

                                        materialDirectLightEdgeEvent.segmentTransmittance = shadowTransmission;
                                        materialDirectLightEdgeEvent.directLightRadiance = directLightRadiance;
                                        materialDirectLightEdgeEvent.isDirectLightSample = true;
                                        materialDirectLightEdgeEvent.pathId = currentRayState.pathId;
                                        materialDirectLightEdgeEvent.startBounceIndex = currentRayState.bounceIndex;


                                        appendEventAtomic(
                                            intermediates.countMaterialEdgeEvents,
                                            intermediates.materialEdgeEvents,
                                            intermediates.maxMaterialEdgeEventCount,
                                            materialDirectLightEdgeEvent);
                                    }
                                }

                                /*
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

                                if (previousAdjointStage.valid && currentRayState.bounceIndex > 0u) {
                                    MaterialEdgeGradientEvent materialEdgeEvent{};
                                    materialEdgeEvent.startSurface =
                                            previousAdjointStage.current.surface;
                                    materialEdgeEvent.endSurface = currentSurface;

                                    materialEdgeEvent.sampledEdgeThroughput =
                                            currentRayState.pathThroughput *
                                            currentRayState.transmission /
                                            qReflect;
                                    materialEdgeEvent.segmentTransmittance = currentRayState.transmission;
                                    materialEdgeEvent.segmentGeometricTerm = segmentGeometryFromStoredVertex;
                                    materialEdgeEvent.segmentAreaPdf = segmentAreaPdfFromStoredVertex;
                                    materialEdgeEvent.pathId = currentRayState.pathId;
                                    materialEdgeEvent.startBounceIndex = previousAdjointStage.current.bounceIndex;

                                    appendEventAtomic(
                                        intermediates.countMaterialEdgeEvents,
                                        intermediates.materialEdgeEvents,
                                        intermediates.maxMaterialEdgeEventCount,
                                        materialEdgeEvent);
                                } */


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

        MeasurementGradientEvent *measurementEvents =
                pkg.intermediates.measurementEvents;

        SurfelGradientRecord *gradientRecords =
                pkg.intermediates.gradientRecords;

        const float invSpp =
                1.0f / settings.adjointSamplesPerPixel;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class measurementGradientEventTag>(
                sycl::range<1>(measurementEventCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex =
                            static_cast<uint32_t>(globalId[0]);

                    static constexpr uint32_t recordsPerEvent = 1u + kMaxSplatEventsPerRay;

                    const uint32_t eventRecordBase = baseOffset + recordsPerEvent * eventIndex;

                    const uint32_t recordIndex = eventRecordBase;

                    for (uint32_t recordOffset = 0u;
                         recordOffset < recordsPerEvent;
                         ++recordOffset) {
                        SurfelGradientRecord invalidRecord{};
                        invalidRecord.primitiveIndex = kInvalidIndex;
                        gradientRecords[eventRecordBase + recordOffset] =
                                invalidRecord;
                    }

                    const MeasurementGradientEvent eventRecord = measurementEvents[eventIndex];

                    const Point &surfelX = scene.points[eventRecord.xSurface.primitiveIndex];
                    const ReconstructedSurfelState xState = reconstructSurfelState(surfelX, eventRecord.xSurface);

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

                    constexpr float distanceEpsilon = 1e-4f;

                    struct OccluderDerivative {
                        float3 gradPosition{0.0f, 0.0f, 0.0f};
                        float gradScaleU = 0.0f;
                        float gradScaleV = 0.0f;
                        float gradEta = 0.0f;
                        float gradBeta = 0.0f;
                        float3 gradTangentU{0.0f, 0.0f, 0.0f};
                        float3 gradTangentV{0.0f, 0.0f, 0.0f};
                        float prefixTransmittance = 1.0f;
                        float oneMinusAlpha = 1.0f;
                        uint32_t primitiveIndex = kInvalidIndex;
                    };

                    OccluderDerivative occluderDerivatives[kMaxSplatEventsPerRay];
                    uint32_t storedOccluderCount = 0u;

                    float segmentTransmittance = 1.0f;

                    const float3 pathWeight =
                            eventRecord.xPathThroughput;

                    const float scalarWeightOcclusion =
                            dot(pathWeight, outgoingRadianceX); {
                        const float3 rayDirection =
                                normalize(vectorCameraToX);

                        Ray ray{};
                        ray.origin = sensor.camera.pos + rayDirection * distanceEpsilon;
                        ray.direction = rayDirection;

                        const float3 segmentOrigin = sensor.camera.pos;

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

                            const float hitDistance = length(worldHit.hitPositionW - sensor.camera.pos);

                            if (hitDistance >= targetDistance - distanceEpsilon) {
                                break;
                            }

                            buildIntersectionNormal(scene, worldHit);

                            const auto &instance =
                                    scene.instances[worldHit.instanceIndex];

                            if (instance.geometryType != GeometryType::PointCloud) {
                                segmentTransmittance = 0.0f;
                                break;
                            }

                            const Point &occluderSurfel = scene.points[worldHit.primitiveIndex];

                            float3 occluderNormal =
                                    normalize(cross(occluderSurfel.tanU, occluderSurfel.tanV));

                            const bool hitBackside = dot(occluderNormal, -ray.direction) < 0.0f;

                            if (hitBackside) {
                                occluderNormal = -occluderNormal;
                            }

                            const float alphaGeomOccluder = worldHit.alphaGeom;
                            const float alphaEffective = occluderSurfel.opacity * alphaGeomOccluder;
                            const float oneMinusAlpha = sycl::fmax(0.0f, 1.0f - alphaEffective);
                            const float prefixTransmittance = segmentTransmittance;
                            segmentTransmittance *= oneMinusAlpha;
                            ray.origin = worldHit.hitPositionW + ray.direction * distanceEpsilon;
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
                            const float3 dxy = segmentOrigin - xState.position;

                            const float denominator =
                                    dot(occluderNormal, dxy);

                            if (sycl::fabs(denominator) <= 1e-8f) {
                                continue;
                            }

                            const float inverseDenominator =
                                    1.0f / denominator;

                            const float3 dUiDspi =
                                    occluderNormal *
                                    (dot(dxy, tangentU) / scaleU) *
                                    inverseDenominator -
                                    localBasisU;

                            const float3 dViDspi =
                                    occluderNormal *
                                    (dot(dxy, tangentV) / scaleV) *
                                    inverseDenominator -
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
                                (
                                    cross(occluderNormal, aOcc) * nDotD -
                                    nDotA * cross(occluderNormal, rayDirection)
                                ) * invNDotDSquared;

                                const float3 duDRotation =
                                        qOcc *
                                        (dot(rayDirection, tangentU) / scaleU) +
                                        cross(tangentU, hitMinusSp) / scaleU;

                                const float3 dvDRotation =
                                        qOcc *
                                        (dot(rayDirection, tangentV) / scaleV) +
                                        cross(tangentV, hitMinusSp) / scaleV;

                                const float3 dAlphaEffectiveDRotation =
                                        occluderSurfel.opacity *
                                        (
                                            dAlphaGeomDu * duDRotation +
                                            dAlphaGeomDv * dvDRotation
                                        );

                                gradTangentUOcc =
                                        cross(
                                            dAlphaEffectiveDRotation,
                                            occluderSurfel.tanU);

                                gradTangentVOcc =
                                        cross(
                                            dAlphaEffectiveDRotation,
                                            occluderSurfel.tanV);
                            }

                            if (storedOccluderCount < kMaxSplatEventsPerRay) {
                                OccluderDerivative &occluderDerivative =
                                        occluderDerivatives[storedOccluderCount];

                                occluderDerivative.gradPosition =
                                        dAlphaEffectiveDspi;

                                occluderDerivative.gradScaleU =
                                        dAlphaEffectiveDScaleU;

                                occluderDerivative.gradScaleV =
                                        dAlphaEffectiveDScaleV;

                                occluderDerivative.gradEta =
                                        dAlphaEffectiveDEta;

                                occluderDerivative.gradBeta =
                                        dAlphaEffectiveDBeta;

                                occluderDerivative.gradTangentU =
                                        gradTangentUOcc;

                                occluderDerivative.gradTangentV =
                                        gradTangentVOcc;

                                occluderDerivative.prefixTransmittance =
                                        prefixTransmittance;

                                occluderDerivative.oneMinusAlpha =
                                        oneMinusAlpha;

                                occluderDerivative.primitiveIndex =
                                        worldHit.primitiveIndex;

                                storedOccluderCount++;
                            }
                        }
                    }

                    const float3 outgoingRadianceXNoAlpha =
                            evaluateOutgoingRadianceWithoutLocalAlpha(
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

                    const UVPositionJacobian uvPositionJacobian =
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
                            -2.0f *
                            betaScale *
                            eventRecord.xSurface.alphaGeom /
                            oneMinusRadiusSquared;

                    const float3 dAlphaGeomDPosition = factor * dUvDPosition;

                    const float3 dAlphaEffectiveDPosition =
                            surfelX.opacity * dAlphaGeomDPosition;

                    const float scalarWeightNoAlpha = segmentTransmittance * dot(
                                                          eventRecord.xPathThroughput, outgoingRadianceXNoAlpha);

                    const float3 positionGradient = dAlphaEffectiveDPosition * scalarWeightNoAlpha * invSpp;

                    float dAlphaGeomDScaleU = 0.0f;
                    float dAlphaGeomDScaleV = 0.0f;

                    if (scaleU > 1e-12f) {
                        dAlphaGeomDScaleU =
                                2.0f *
                                betaScale *
                                u *
                                u *
                                eventRecord.xSurface.alphaGeom /
                                (scaleU * oneMinusRadiusSquared);
                    }

                    if (scaleV > 1e-12f) {
                        dAlphaGeomDScaleV =
                                2.0f *
                                betaScale *
                                v *
                                v *
                                eventRecord.xSurface.alphaGeom /
                                (scaleV * oneMinusRadiusSquared);
                    }
                    const float dAlphaEffectiveDScaleU = surfelX.opacity * dAlphaGeomDScaleU;
                    const float dAlphaEffectiveDScaleV = surfelX.opacity * dAlphaGeomDScaleV;
                    const float scaleGradientU =
                            dAlphaEffectiveDScaleU *
                            scalarWeightNoAlpha *
                            invSpp;

                    const float scaleGradientV =
                            dAlphaEffectiveDScaleV *
                            scalarWeightNoAlpha *
                            invSpp;
                    float3 tanUGradient{0.0f, 0.0f, 0.0f};
                    float3 tanVGradient{0.0f, 0.0f, 0.0f};
                    const float dAlphaGeomDu = factor * u;
                    const float dAlphaGeomDv = factor * v;
                    const float3 rayDirection = normalize(vectorCameraToX);
                    const float3 normalX = xState.orientedNormal;
                    const float nDotD = dot(normalX, rayDirection);

                    if (sycl::fabs(nDotD) > 1e-8f &&
                        scaleU > 1e-12f &&
                        scaleV > 1e-12f) {
                        const float3 xMinusSp = u * scaleU * surfelX.tanU + v * scaleV * surfelX.tanV;
                        const float3 a = xState.position - sensor.camera.pos - xMinusSp;
                        const float nDotA = dot(normalX, a);
                        const float invNDotD = 1.0f / nDotD;
                        const float invNDotDSquared = invNDotD * invNDotD;

                        const float3 q =
                        (
                            cross(normalX, a) * nDotD -
                            nDotA * cross(normalX, rayDirection)
                        ) * invNDotDSquared;

                        const float3 duDRotation =
                                q * (dot(rayDirection, surfelX.tanU) / scaleU) +
                                cross(surfelX.tanU, xMinusSp) / scaleU;

                        const float3 dvDRotation =
                                q * (dot(rayDirection, surfelX.tanV) / scaleV) +
                                cross(surfelX.tanV, xMinusSp) / scaleV;

                        const float3 dAlphaGeomDRotation =
                                dAlphaGeomDu * duDRotation +
                                dAlphaGeomDv * dvDRotation;

                        const float3 dAlphaEffectiveDRotation =
                                surfelX.opacity * dAlphaGeomDRotation;

                        const float3 rotationGradientZeta =
                                dAlphaEffectiveDRotation *
                                scalarWeightNoAlpha *
                                invSpp;

                        tanUGradient =
                                cross(rotationGradientZeta, surfelX.tanU);

                        tanVGradient =
                                cross(rotationGradientZeta, surfelX.tanV);
                    }

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
                    gradientRecord.primitiveIndex =
                            eventRecord.xSurface.primitiveIndex;

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

                    gradientRecord.gradAlbedoR = 0.0f;
                    gradientRecord.gradAlbedoG = 0.0f;
                    gradientRecord.gradAlbedoB = 0.0f;

                    gradientRecords[recordIndex] =
                            gradientRecord;

                    float suffixTransmittance = 1.0f;

                    for (uint32_t reverseIndex = storedOccluderCount;
                         reverseIndex > 0u;
                         --reverseIndex) {
                        const uint32_t occluderIndex =
                                reverseIndex - 1u;

                        const uint32_t occluderRecordIndex =
                                eventRecordBase + 1u + occluderIndex;

                        const OccluderDerivative &occluderDerivative =
                                occluderDerivatives[occluderIndex];

                        const float visibilityDerivativeScale =
                                -occluderDerivative.prefixTransmittance *
                                suffixTransmittance *
                                scalarWeightOcclusion *
                                invSpp;

                        SurfelGradientRecord occluderRecord{};
                        occluderRecord.primitiveIndex =
                                occluderDerivative.primitiveIndex;

                        const float3 positionContribution =
                                visibilityDerivativeScale *
                                occluderDerivative.gradPosition;

                        const float3 tangentUContribution =
                                visibilityDerivativeScale *
                                occluderDerivative.gradTangentU;

                        const float3 tangentVContribution =
                                visibilityDerivativeScale *
                                occluderDerivative.gradTangentV;

                        occluderRecord.gradPositionX = positionContribution.x();
                        occluderRecord.gradPositionY = positionContribution.y();
                        occluderRecord.gradPositionZ = positionContribution.z();

                        occluderRecord.gradScaleU =
                                visibilityDerivativeScale *
                                occluderDerivative.gradScaleU;

                        occluderRecord.gradScaleV =
                                visibilityDerivativeScale *
                                occluderDerivative.gradScaleV;

                        occluderRecord.gradEta =
                                visibilityDerivativeScale *
                                occluderDerivative.gradEta;

                        occluderRecord.gradBeta =
                                visibilityDerivativeScale *
                                occluderDerivative.gradBeta;

                        occluderRecord.gradTangentUX = tangentUContribution.x();
                        occluderRecord.gradTangentUY = tangentUContribution.y();
                        occluderRecord.gradTangentUZ = tangentUContribution.z();

                        occluderRecord.gradTangentVX = tangentVContribution.x();
                        occluderRecord.gradTangentVY = tangentVContribution.y();
                        occluderRecord.gradTangentVZ = tangentVContribution.z();

                        occluderRecord.gradAlbedoR = 0.0f;
                        occluderRecord.gradAlbedoG = 0.0f;
                        occluderRecord.gradAlbedoB = 0.0f;

                        gradientRecords[occluderRecordIndex] =
                                occluderRecord;

                        suffixTransmittance *=
                                occluderDerivative.oneMinusAlpha;
                    }
                });
        }).wait();
    }

    static void measurementGradientEventXY(
        RenderPackage &pkg,
        uint32_t onePointEventCount,
        uint32_t baseOffset) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        const auto &photonMap = pkg.intermediates.map;

        MeasurementGradientEventXY *measurementXYEvent =
                pkg.intermediates.measurementTwoPointEvents;

        SurfelGradientRecord *gradientRecords =
                pkg.intermediates.gradientRecords;

        const float invSpp =
                1.0f / settings.adjointSamplesPerPixel;

        const float qNullInv =
                1.0f / settings.sampling.qNull;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class firstHitGradientEventTag>(
                sycl::range<1>(onePointEventCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex =
                            static_cast<uint32_t>(globalId[0]);

                    static constexpr uint32_t recordsPerEvent =
                            1u + kMaxSplatEventsPerRay;

                    const uint32_t eventRecordBase =
                            baseOffset + recordsPerEvent * eventIndex;

                    const uint32_t recordIndex =
                            eventRecordBase;

                    for (uint32_t recordOffset = 0u;
                         recordOffset < recordsPerEvent;
                         ++recordOffset) {
                        SurfelGradientRecord invalidRecord{};
                        invalidRecord.primitiveIndex = kInvalidIndex;
                        gradientRecords[eventRecordBase + recordOffset] =
                                invalidRecord;
                    }

                    const MeasurementGradientEventXY eventRecord =
                            measurementXYEvent[eventIndex];

                    const uint32_t xPrimitiveIndex =
                            eventRecord.xSurface.primitiveIndex;

                    const uint32_t yPrimitiveIndex =
                            eventRecord.ySurface.primitiveIndex;

                    const Point &surfelX =
                            scene.points[xPrimitiveIndex];

                    const Point &surfelY =
                            scene.points[yPrimitiveIndex];

                    const ReconstructedSurfelState xState =
                            reconstructSurfelState(surfelX, eventRecord.xSurface);

                    const ReconstructedSurfelState yState =
                            reconstructSurfelState(surfelY, eventRecord.ySurface);

                    const uint64_t directLightSeed = rng::makeSeed(
                        settings.random.seed,
                        eventRecord.xSurface.pathId,
                        0xffefeefefu,
                        rng::kStreamDirectLight,
                        eventIndex);

                    rng::Xorshift128 directLightRng(directLightSeed);

                    float3 outgoingRadianceY{0.0f, 0.0f, 0.0f};

                    if (settings.enableAdjointDirectLight) {
                        if (eventRecord.isDirectLightSample) {
                            outgoingRadianceY =
                                    eventRecord.directLightRadiance;
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

                    const float alphaX =
                            eventRecord.xSurface.alphaGeom * surfelX.opacity;

                    const float brdfScaleX =
                            surfelX.alpha_r * M_1_PIf;

                    const float3 brdfX =
                            brdfScaleX * surfelX.albedo;

                    const float3 dGeometricTermDx =
                            computeGeometricTermGradientWrtStartpoint(
                                xState.position,
                                yState.position,
                                xState.orientedNormal,
                                yState.orientedNormal);

                    const float3 pathWeight =
                            eventRecord.xPathThroughput;

                    const float3 transportWithoutTauAndGeometric =
                            outgoingRadianceY *
                            alphaX *
                            brdfX;

                    const float scalarWeightWithoutTauAndGeometricBase =
                            dot(pathWeight, transportWithoutTauAndGeometric);

                    const float3 albedoWeightWithoutTauAndGeometricBase =
                            pathWeight *
                            outgoingRadianceY *
                            (alphaX * brdfScaleX);

                    struct OccluderDerivative {
                        float3 gradPosition{0.0f, 0.0f, 0.0f};
                        float gradScaleU = 0.0f;
                        float gradScaleV = 0.0f;
                        float gradEta = 0.0f;
                        float gradBeta = 0.0f;
                        float3 gradTangentU{0.0f, 0.0f, 0.0f};
                        float3 gradTangentV{0.0f, 0.0f, 0.0f};
                        float3 gradAlphaWrtStartPoint{0.0f, 0.0f, 0.0f};
                        float3 gradAlphaWrtEndPoint{0.0f, 0.0f, 0.0f};
                        float prefixTransmittance = 1.0f;
                        float oneMinusAlpha = 1.0f;
                        uint32_t primitiveIndex = kInvalidIndex;
                    };

                    OccluderDerivative occluderDerivatives[kMaxSplatEventsPerRay];
                    uint32_t storedOccluderCount = 0u;

                    float segmentTransmittance = 1.0f;
                    float nullSamplingWeight = 1.0f; {
                        constexpr float distanceEpsilon =
                                1e-4f;

                        const float3 segmentDirection =
                                yState.position - xState.position;

                        const float targetDistance =
                                length(segmentDirection);

                        if (targetDistance <= 1e-12f) {
                            return;
                        }

                        const float3 rayDirection =
                                segmentDirection / targetDistance;

                        Ray ray{};
                        ray.origin =
                                xState.position + rayDirection * distanceEpsilon;
                        ray.direction =
                                rayDirection;

                        const float3 xPosition =
                                xState.position;

                        const float3 yPosition =
                                yState.position;

                        const float3 dxy =
                                xPosition - yPosition;

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
                                    length(worldHit.hitPositionW - xState.position);

                            if (hitDistance >= targetDistance - distanceEpsilon) {
                                break;
                            }

                            buildIntersectionNormal(scene, worldHit);

                            const auto &instance =
                                    scene.instances[worldHit.instanceIndex];

                            if (instance.geometryType != GeometryType::PointCloud) {
                                segmentTransmittance = 0.0f;
                                break;
                            }

                            const Point &occluderSurfel =
                                    scene.points[worldHit.primitiveIndex];

                            float3 occluderNormal =
                                    normalize(cross(
                                        occluderSurfel.tanU,
                                        occluderSurfel.tanV));

                            const bool hitBackside =
                                    dot(occluderNormal, -ray.direction) < 0.0f;

                            if (hitBackside) {
                                occluderNormal = -occluderNormal;
                            }

                            const float alphaGeomOccluder =
                                    worldHit.alphaGeom;

                            const float alphaEffective =
                                    occluderSurfel.opacity * alphaGeomOccluder;

                            const float oneMinusAlpha =
                                    sycl::fmax(0.0f, 1.0f - alphaEffective);

                            const float prefixTransmittance =
                                    segmentTransmittance;

                            segmentTransmittance *= oneMinusAlpha;

                            if (!eventRecord.isDirectLightSample) {
                                nullSamplingWeight *= qNullInv;
                            }

                            ray.origin =
                                    worldHit.hitPositionW + ray.direction * distanceEpsilon;

                            const float2 uv =
                                    phiInverse(worldHit.hitPositionW, occluderSurfel);

                            const float uOcc =
                                    uv.x();

                            const float vOcc =
                                    uv.y();

                            const float scaleU =
                                    occluderSurfel.scale.x();

                            const float scaleV =
                                    occluderSurfel.scale.y();

                            if (scaleU <= 1e-12f || scaleV <= 1e-12f) {
                                continue;
                            }

                            const float3 tangentU =
                                    occluderSurfel.tanU;

                            const float3 tangentV =
                                    occluderSurfel.tanV;

                            const float3 localBasisU =
                                    tangentU / scaleU;

                            const float3 localBasisV =
                                    tangentV / scaleV;

                            const float denominator =
                                    dot(occluderNormal, dxy);

                            if (sycl::fabs(denominator) <= 1e-8f) {
                                continue;
                            }

                            const float inverseDenominator =
                                    1.0f / denominator;

                            const float lambdaOccluder =
                                    dot(
                                        occluderNormal,
                                        occluderSurfel.position - yPosition) *
                                    inverseDenominator;

                            const float3 dUiDy =
                                    (1.0f - lambdaOccluder) *
                                    (
                                        localBasisU -
                                        occluderNormal *
                                        (dot(dxy, localBasisU) * inverseDenominator)
                                    );

                            const float3 dViDy =
                                    (1.0f - lambdaOccluder) *
                                    (
                                        localBasisV -
                                        occluderNormal *
                                        (dot(dxy, localBasisV) * inverseDenominator)
                                    );

                            const float3 dUiDx =
                                    lambdaOccluder *
                                    (
                                        localBasisU -
                                        occluderNormal *
                                        (dot(dxy, localBasisU) * inverseDenominator)
                                    );

                            const float3 dViDx =
                                    lambdaOccluder *
                                    (
                                        localBasisV -
                                        occluderNormal *
                                        (dot(dxy, localBasisV) * inverseDenominator)
                                    );

                            const float3 dUiDspi =
                                    occluderNormal *
                                    (dot(dxy, tangentU) / scaleU) *
                                    inverseDenominator -
                                    localBasisU;

                            const float3 dViDspi =
                                    occluderNormal *
                                    (dot(dxy, tangentV) / scaleV) *
                                    inverseDenominator -
                                    localBasisV;

                            const float radiusSquared =
                                    uOcc * uOcc + vOcc * vOcc;

                            const float oneMinusRadiusSquared =
                                    1.0f - radiusSquared;

                            if (oneMinusRadiusSquared <= 1e-8f) {
                                continue;
                            }

                            const float betaScale =
                                    4.0f * sycl::exp(occluderSurfel.beta);

                            const float dAlphaGeomDu =
                                    -2.0f *
                                    betaScale *
                                    uOcc *
                                    alphaGeomOccluder /
                                    oneMinusRadiusSquared;

                            const float dAlphaGeomDv =
                                    -2.0f *
                                    betaScale *
                                    vOcc *
                                    alphaGeomOccluder /
                                    oneMinusRadiusSquared;

                            const float3 dAlphaEffectiveDx =
                                    occluderSurfel.opacity *
                                    (
                                        dAlphaGeomDu * dUiDx +
                                        dAlphaGeomDv * dViDx
                                    );

                            const float3 dAlphaEffectiveDy =
                                    occluderSurfel.opacity *
                                    (
                                        dAlphaGeomDu * dUiDy +
                                        dAlphaGeomDv * dViDy
                                    );

                            const float3 dAlphaEffectiveDspi =
                                    occluderSurfel.opacity *
                                    (
                                        dAlphaGeomDu * dUiDspi +
                                        dAlphaGeomDv * dViDspi
                                    );

                            const float dAlphaEffectiveDScaleU =
                                    2.0f *
                                    betaScale *
                                    uOcc *
                                    uOcc *
                                    alphaEffective /
                                    (scaleU * oneMinusRadiusSquared);

                            const float dAlphaEffectiveDScaleV =
                                    2.0f *
                                    betaScale *
                                    vOcc *
                                    vOcc *
                                    alphaEffective /
                                    (scaleV * oneMinusRadiusSquared);

                            const float dAlphaEffectiveDEta =
                                    alphaGeomOccluder;

                            const float dAlphaEffectiveDBeta =
                                    betaScale *
                                    sycl::log(oneMinusRadiusSquared) *
                                    alphaEffective;

                            float3 gradTangentUOcc{0.0f, 0.0f, 0.0f};
                            float3 gradTangentVOcc{0.0f, 0.0f, 0.0f};

                            const float nDotD =
                                    dot(occluderNormal, rayDirection);

                            if (sycl::fabs(nDotD) > 1e-8f) {
                                const float3 hitMinusSp =
                                        worldHit.hitPositionW -
                                        occluderSurfel.position;

                                const float3 aOcc =
                                        occluderSurfel.position -
                                        yPosition;

                                const float nDotA =
                                        dot(occluderNormal, aOcc);

                                const float invNDotD =
                                        1.0f / nDotD;

                                const float invNDotDSquared =
                                        invNDotD * invNDotD;

                                const float3 qOcc =
                                (
                                    cross(occluderNormal, aOcc) * nDotD -
                                    nDotA * cross(occluderNormal, rayDirection)
                                ) * invNDotDSquared;

                                const float3 duDRotation =
                                        qOcc *
                                        (dot(rayDirection, tangentU) / scaleU) +
                                        cross(tangentU, hitMinusSp) / scaleU;

                                const float3 dvDRotation =
                                        qOcc *
                                        (dot(rayDirection, tangentV) / scaleV) +
                                        cross(tangentV, hitMinusSp) / scaleV;

                                const float3 dAlphaEffectiveDRotation =
                                        occluderSurfel.opacity *
                                        (
                                            dAlphaGeomDu * duDRotation +
                                            dAlphaGeomDv * dvDRotation
                                        );

                                gradTangentUOcc =
                                        cross(
                                            dAlphaEffectiveDRotation,
                                            occluderSurfel.tanU);

                                gradTangentVOcc =
                                        cross(
                                            dAlphaEffectiveDRotation,
                                            occluderSurfel.tanV);
                            }

                            if (storedOccluderCount < kMaxSplatEventsPerRay) {
                                OccluderDerivative &occluderDerivative =
                                        occluderDerivatives[storedOccluderCount];

                                occluderDerivative.gradPosition =
                                        dAlphaEffectiveDspi;

                                occluderDerivative.gradScaleU =
                                        dAlphaEffectiveDScaleU;

                                occluderDerivative.gradScaleV =
                                        dAlphaEffectiveDScaleV;

                                occluderDerivative.gradEta =
                                        dAlphaEffectiveDEta;

                                occluderDerivative.gradBeta =
                                        dAlphaEffectiveDBeta;

                                occluderDerivative.gradTangentU =
                                        gradTangentUOcc;

                                occluderDerivative.gradTangentV =
                                        gradTangentVOcc;

                                occluderDerivative.gradAlphaWrtStartPoint =
                                        dAlphaEffectiveDx;

                                occluderDerivative.gradAlphaWrtEndPoint =
                                        dAlphaEffectiveDy;

                                occluderDerivative.prefixTransmittance =
                                        prefixTransmittance;

                                occluderDerivative.oneMinusAlpha =
                                        oneMinusAlpha;

                                occluderDerivative.primitiveIndex =
                                        worldHit.primitiveIndex;

                                storedOccluderCount++;
                            }
                        }
                    }

                    float3 gradTauWrtStartPoint{0.0f, 0.0f, 0.0f};
                    float3 gradTauWrtEndPoint{0.0f, 0.0f, 0.0f};

                    float suffixTransmittanceForTauGradient =
                            1.0f;

                    for (uint32_t reverseIndex = storedOccluderCount;
                         reverseIndex > 0u;
                         --reverseIndex) {
                        const uint32_t occluderIndex =
                                reverseIndex - 1u;

                        const OccluderDerivative &occluderDerivative =
                                occluderDerivatives[occluderIndex];

                        const float tauDerivativeScale =
                                -occluderDerivative.prefixTransmittance *
                                suffixTransmittanceForTauGradient;

                        gradTauWrtStartPoint +=
                                tauDerivativeScale *
                                occluderDerivative.gradAlphaWrtStartPoint;

                        gradTauWrtEndPoint +=
                                tauDerivativeScale *
                                occluderDerivative.gradAlphaWrtEndPoint;

                        suffixTransmittanceForTauGradient *=
                                occluderDerivative.oneMinusAlpha;
                    }

                    const float geometricTermXY =
                            computeGeometricTermValue(
                                xState.position,
                                yState.position,
                                xState.orientedNormal,
                                yState.orientedNormal);

                    const float scalarWeightWithoutTauAndGeometric =
                            scalarWeightWithoutTauAndGeometricBase *
                            nullSamplingWeight;

                    const float3 albedoWeightWithoutTauAndGeometric =
                            albedoWeightWithoutTauAndGeometricBase *
                            nullSamplingWeight;

                    const float3 gradientWrtWorldHitPositionX =
                            scalarWeightWithoutTauAndGeometric *
                            (
                                segmentTransmittance * dGeometricTermDx +
                                geometricTermXY * gradTauWrtStartPoint
                            );

                    const float3x3 hitPointJacobianX =
                            planeHitPointIntersectionJacobian(
                                eventRecord.xSurface.incomingDirection,
                                xState.orientedNormal);

                    const float3 gradientWrtHitPositionX =
                            transpose(hitPointJacobianX) *
                            gradientWrtWorldHitPositionX;

                    const float3 xContribution =
                            gradientWrtHitPositionX *
                            invSpp;

                    float3 tanUContribution{0.0f, 0.0f, 0.0f};
                    float3 tanVContribution{0.0f, 0.0f, 0.0f};

                    const float3 primaryRayDirection =
                            eventRecord.xSurface.incomingDirection;

                    const float nDotD =
                            dot(xState.orientedNormal, primaryRayDirection);

                    const float3 rawCross =
                            cross(surfelX.tanU, surfelX.tanV);

                    const float rawCrossLength =
                            length(rawCross);

                    if (sycl::fabs(nDotD) > 1e-8f &&
                        rawCrossLength > 1e-8f) {
                        const float3 rawNormal =
                                rawCross / rawCrossLength;

                        const float orientationSign =
                                dot(rawNormal, xState.orientedNormal) >= 0.0f
                                    ? 1.0f
                                    : -1.0f;

                        const float3 pMinusX =
                                surfelX.position - xState.position;

                        const float3 gradientWrtOrientedNormalFromMovedHit =
                                pMinusX *
                                (
                                    dot(
                                        primaryRayDirection,
                                        gradientWrtWorldHitPositionX) /
                                    nDotD
                                );

                        const float3 segmentDirection =
                                yState.position - xState.position;

                        const float distanceSquared =
                                dot(segmentDirection, segmentDirection);

                        const float3 directionXToY =
                                normalize(segmentDirection);

                        const float cosineAtY =
                                dot(yState.orientedNormal, -directionXToY);

                        const float3 dGeometricTermDStartNormal =
                                directionXToY *
                                (cosineAtY / distanceSquared);

                        const float3 gradientWrtOrientedNormalExplicit =
                                scalarWeightWithoutTauAndGeometric *
                                segmentTransmittance *
                                dGeometricTermDStartNormal;

                        const float3 gradientWrtOrientedNormalX =
                                gradientWrtOrientedNormalFromMovedHit +
                                gradientWrtOrientedNormalExplicit;

                        const float3 gradientWrtRawNormal =
                                orientationSign *
                                gradientWrtOrientedNormalX;

                        const float3 gradientProjectedToRawNormalTangent =
                                gradientWrtRawNormal -
                                rawNormal * dot(rawNormal, gradientWrtRawNormal);

                        const float3 gradientWrtCross =
                                gradientProjectedToRawNormalTangent /
                                rawCrossLength;

                        tanUContribution =
                                cross(surfelX.tanV, gradientWrtCross) *
                                invSpp;

                        tanVContribution =
                                cross(gradientWrtCross, surfelX.tanU) *
                                invSpp;
                    }

                    const float3 albedoContribution =
                            segmentTransmittance *
                            geometricTermXY *
                            albedoWeightWithoutTauAndGeometric *
                            invSpp;

                    SurfelGradientRecord xRecord{};
                    xRecord.primitiveIndex =
                            xPrimitiveIndex;

                    xRecord.gradPositionX = xContribution.x();
                    xRecord.gradPositionY = xContribution.y();
                    xRecord.gradPositionZ = xContribution.z();

                    xRecord.gradTangentUX = tanUContribution.x();
                    xRecord.gradTangentUY = tanUContribution.y();
                    xRecord.gradTangentUZ = tanUContribution.z();

                    xRecord.gradTangentVX = tanVContribution.x();
                    xRecord.gradTangentVY = tanVContribution.y();
                    xRecord.gradTangentVZ = tanVContribution.z();

                    xRecord.gradScaleU = 0.0f;
                    xRecord.gradScaleV = 0.0f;

                    xRecord.gradEta = 0.0f;
                    xRecord.gradBeta = 0.0f;

                    xRecord.gradAlbedoR = albedoContribution.x();
                    xRecord.gradAlbedoG = albedoContribution.y();
                    xRecord.gradAlbedoB = albedoContribution.z();

                    gradientRecords[recordIndex] =
                            xRecord;

                    float suffixTransmittance = 1.0f;

                    for (uint32_t reverseIndex = storedOccluderCount;
                         reverseIndex > 0u;
                         --reverseIndex) {
                        const uint32_t occluderIndex =
                                reverseIndex - 1u;

                        const uint32_t occluderRecordIndex =
                                eventRecordBase + 1u + occluderIndex;

                        const OccluderDerivative &occluderDerivative =
                                occluderDerivatives[occluderIndex];

                        const float visibilityDerivativeScale =
                                -occluderDerivative.prefixTransmittance *
                                suffixTransmittance *
                                geometricTermXY *
                                scalarWeightWithoutTauAndGeometric *
                                invSpp;

                        SurfelGradientRecord occluderRecord{};
                        occluderRecord.primitiveIndex =
                                occluderDerivative.primitiveIndex;

                        const float3 positionContribution =
                                visibilityDerivativeScale *
                                occluderDerivative.gradPosition;

                        const float3 tangentUContribution =
                                visibilityDerivativeScale *
                                occluderDerivative.gradTangentU;

                        const float3 tangentVContribution =
                                visibilityDerivativeScale *
                                occluderDerivative.gradTangentV;

                        occluderRecord.gradPositionX = positionContribution.x();
                        occluderRecord.gradPositionY = positionContribution.y();
                        occluderRecord.gradPositionZ = positionContribution.z();

                        occluderRecord.gradScaleU =
                                visibilityDerivativeScale *
                                occluderDerivative.gradScaleU;

                        occluderRecord.gradScaleV =
                                visibilityDerivativeScale *
                                occluderDerivative.gradScaleV;

                        occluderRecord.gradEta =
                                visibilityDerivativeScale *
                                occluderDerivative.gradEta;

                        occluderRecord.gradBeta =
                                visibilityDerivativeScale *
                                occluderDerivative.gradBeta;

                        occluderRecord.gradTangentUX = tangentUContribution.x();
                        occluderRecord.gradTangentUY = tangentUContribution.y();
                        occluderRecord.gradTangentUZ = tangentUContribution.z();

                        occluderRecord.gradTangentVX = tangentVContribution.x();
                        occluderRecord.gradTangentVY = tangentVContribution.y();
                        occluderRecord.gradTangentVZ = tangentVContribution.z();

                        occluderRecord.gradAlbedoR = 0.0f;
                        occluderRecord.gradAlbedoG = 0.0f;
                        occluderRecord.gradAlbedoB = 0.0f;

                        gradientRecords[occluderRecordIndex] =
                                occluderRecord;

                        suffixTransmittance *=
                                occluderDerivative.oneMinusAlpha;
                    }
                });
        }).wait();
    }

    static void materialVertexGradientEvent(
        RenderPackage &pkg,
        uint32_t materialVertexEventCount,
        uint32_t baseOffset) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        const auto &photonMap = pkg.intermediates.map;

        MaterialVertexGradientEvent *materialVertexEvents =
                pkg.intermediates.materialVertexEvents;

        SurfelGradientRecord *gradientRecords =
                pkg.intermediates.gradientRecords;

        const float invSpp =
                1.0f / settings.adjointSamplesPerPixel;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class materialVertexGradientEventTag>(
                sycl::range<1>(materialVertexEventCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex =
                            static_cast<uint32_t>(globalId[0]);

                    const uint32_t recordIndex =
                            baseOffset + eventIndex;

                    SurfelGradientRecord gradientRecord{};
                    gradientRecord.primitiveIndex = kInvalidIndex;
                    gradientRecords[recordIndex] = gradientRecord;

                    const MaterialVertexGradientEvent eventRecord =
                            materialVertexEvents[eventIndex];

                    const uint32_t primitiveIndex =
                            eventRecord.surface.primitiveIndex;

                    if (primitiveIndex == kInvalidIndex) {
                        return;
                    }

                    const Point &surfel =
                            scene.points[primitiveIndex];

                    const ReconstructedSurfelState surfelState =
                            reconstructSurfelState(surfel, eventRecord.surface);

                    const uint64_t directLightSeed = rng::makeSeed(
                        settings.random.seed,
                        eventRecord.pathId,
                        0x46ac91fbu,
                        rng::kStreamDirectLight,
                        eventIndex);

                    rng::Xorshift128 directLightRng(directLightSeed);

                    // This is L_surfel, i.e. outgoing radiance before multiplying
                    // by local alpha = eta * alpha_geom.
                    const float3 outgoingRadianceWithoutLocalAlpha =
                            evaluateOutgoingRadianceWithoutLocalAlpha(
                                surfel,
                                eventRecord.surface,
                                surfelState,
                                photonMap,
                                scene,
                                settings.numAdjointShadowRays,
                                directLightRng);


                    const float u =
                            eventRecord.surface.uv.x();

                    const float v =
                            eventRecord.surface.uv.y();

                    const float radiusSquared =
                            u * u + v * v;

                    const float oneMinusRadiusSquared =
                            1.0f - radiusSquared;

                    if (oneMinusRadiusSquared <= 1.0e-8f) {
                        return;
                    }

                    const float alphaGeom =
                            eventRecord.surface.alphaGeom;

                    const float betaScale =
                            4.0f * sycl::exp(surfel.beta);

                    const float dAlphaGeomDBeta =
                            betaScale *
                            sycl::log(oneMinusRadiusSquared) *
                            alphaGeom;

                    const float dAlphaEffectiveDBeta =
                            surfel.opacity * dAlphaGeomDBeta;

                    const float scalarWeightNoAlpha =
                            dot(eventRecord.adjointWeightAtVertex, outgoingRadianceWithoutLocalAlpha);

                    const float opacityGradient =
                            alphaGeom *
                            scalarWeightNoAlpha *
                            invSpp;

                    const float betaGradient =
                            dot(
                                eventRecord.adjointWeightAtVertex,
                                outgoingRadianceWithoutLocalAlpha *
                                dAlphaEffectiveDBeta) *
                            invSpp;

                    gradientRecord.primitiveIndex = primitiveIndex;

                    gradientRecord.gradPositionX = 0.0f;
                    gradientRecord.gradPositionY = 0.0f;
                    gradientRecord.gradPositionZ = 0.0f;

                    gradientRecord.gradScaleU = 0.0f;
                    gradientRecord.gradScaleV = 0.0f;

                    gradientRecord.gradTangentUX = 0.0f;
                    gradientRecord.gradTangentUY = 0.0f;
                    gradientRecord.gradTangentUZ = 0.0f;

                    gradientRecord.gradTangentVX = 0.0f;
                    gradientRecord.gradTangentVY = 0.0f;
                    gradientRecord.gradTangentVZ = 0.0f;

                    gradientRecord.gradEta = opacityGradient;
                    gradientRecord.gradBeta = betaGradient;

                    gradientRecord.gradAlbedoR = 0.0f;
                    gradientRecord.gradAlbedoG = 0.0f;
                    gradientRecord.gradAlbedoB = 0.0f;

                    gradientRecords[recordIndex] = gradientRecord;
                });
        }).wait();
    }

    static void materialEdgeGradientEvent(RenderPackage &pkg, uint32_t materialEdgeEventCount, uint32_t baseOffset) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        const auto &photonMap = pkg.intermediates.map;
        MaterialEdgeGradientEvent *materialEdgeEvents = pkg.intermediates.materialEdgeEvents;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;
        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;
        static constexpr uint32_t recordsPerMaterialEdgeEvent = 2u + kMaxSplatEventsPerRay;
        const float qNullInv = 1.0f / settings.sampling.qNull;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class materialEdgeGradientEventTag>(
                sycl::range<1>(materialEdgeEventCount), [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex = static_cast<uint32_t>(globalId[0]);
                    const uint32_t eventRecordBase = baseOffset + recordsPerMaterialEdgeEvent * eventIndex;
                    for (uint32_t recordOffset = 0u; recordOffset < recordsPerMaterialEdgeEvent; ++recordOffset) {
                        SurfelGradientRecord invalidRecord{};
                        invalidRecord.primitiveIndex = kInvalidIndex;
                        gradientRecords[eventRecordBase + recordOffset] = invalidRecord;
                    }

                    const MaterialEdgeGradientEvent eventRecord = materialEdgeEvents[eventIndex];
                    const uint32_t startPrimitiveIndex = eventRecord.startSurface.primitiveIndex;
                    const uint32_t endPrimitiveIndex = eventRecord.endSurface.primitiveIndex;
                    if (startPrimitiveIndex == kInvalidIndex || endPrimitiveIndex == kInvalidIndex) {
                        return;
                    }

                    const Point &startSurfel = scene.points[startPrimitiveIndex];
                    const Point &endSurfel = scene.points[endPrimitiveIndex];
                    const ReconstructedSurfelState startState = reconstructSurfelState(
                        startSurfel, eventRecord.startSurface);
                    const ReconstructedSurfelState endState = reconstructSurfelState(endSurfel, eventRecord.endSurface);

                    rng::Xorshift128 directLightRng(rng::makeSeed(settings.random.seed, eventRecord.pathId, 0x8d24f31bu,
                                                                  rng::kStreamDirectLight, eventIndex));
                    const float3 endRadiance = eventRecord.isDirectLightSample
                                                   ? eventRecord.directLightRadiance
                                                   : evaluateOutgoingRadianceWithLocalAlphaNoEmitters(
                                                       endSurfel, eventRecord.endSurface, endState, photonMap, scene,
                                                       settings.numAdjointShadowRays, directLightRng);

                    const float geometricTerm = computeGeometricTermValue(
                        startState.position, endState.position, startState.orientedNormal, endState.orientedNormal);
                    if (geometricTerm <= 1.0e-12f) {
                        return;
                    }

                    const float3 dGeometricTermDStartPosition = computeGeometricTermGradientWrtStartpoint(
                        startState.position, endState.position, startState.orientedNormal, endState.orientedNormal);
                    const float3 dGeometricTermDEndPosition = computeGeometricTermGradientWrtEndpoint(
                        startState.position, endState.position, startState.orientedNormal, endState.orientedNormal);
                    const float scalarEdgeWeightBase = dot(eventRecord.sampledEdgeThroughput, endRadiance);

                    const float endU = eventRecord.endSurface.uv.x();
                    const float endV = eventRecord.endSurface.uv.y();
                    const float endScaleU = endSurfel.scale.x();
                    const float endScaleV = endSurfel.scale.y();
                    const bool endIsMaterialMeasure = !eventRecord.isDirectLightSample;
                    const float endJacobianUv = endIsMaterialMeasure ? endScaleU * endScaleV : 1.0f;
                    const float dEndJacobianUvDScaleU = endIsMaterialMeasure ? endScaleV : 0.0f;
                    const float dEndJacobianUvDScaleV = endIsMaterialMeasure ? endScaleU : 0.0f;

                    struct OccluderDerivative {
                        float3 gradPosition{0.0f, 0.0f, 0.0f};
                        float gradScaleU = 0.0f;
                        float gradScaleV = 0.0f;
                        float gradEta = 0.0f;
                        float gradBeta = 0.0f;
                        float3 gradTangentU{0.0f, 0.0f, 0.0f};
                        float3 gradTangentV{0.0f, 0.0f, 0.0f};
                        float3 gradAlphaWrtStartPoint{0.0f, 0.0f, 0.0f};
                        float3 gradAlphaWrtEndPoint{0.0f, 0.0f, 0.0f};
                        float prefixTransmittance = 1.0f;
                        float oneMinusAlpha = 1.0f;
                        uint32_t primitiveIndex = kInvalidIndex;
                    };

                    OccluderDerivative occluderDerivatives[kMaxSplatEventsPerRay];
                    uint32_t storedOccluderCount = 0u;
                    float3 gradTauWrtStartPoint{0.0f, 0.0f, 0.0f};
                    float3 gradTauWrtEndPoint{0.0f, 0.0f, 0.0f};
                    float edgeTransmittance = eventRecord.segmentTransmittance;
                    float nullSamplingWeight = 1.0f; {
                        constexpr float distanceEpsilon = 1.0e-4f;
                        const float3 startToEnd = endState.position - startState.position;
                        const float targetDistance = length(startToEnd);
                        if (targetDistance <= 1.0e-12f) {
                            return;
                        }

                        const float3 rayDirection = startToEnd / targetDistance;
                        Ray ray{};
                        ray.origin = startState.position + rayDirection * distanceEpsilon;
                        ray.direction = rayDirection;
                        ray.normal = startState.orientedNormal;

                        const float3 xPosition = startState.position;
                        const float3 yPosition = endState.position;
                        const float3 dxy = xPosition - yPosition;
                        float tracedTransmittance = 1.0f;

                        for (uint32_t traversalIndex = 0u; traversalIndex < kMaxSplatEventsPerRay; ++traversalIndex) {
                            WorldHit worldHit{};
                            intersectScene(ray, &worldHit, scene, SurfelIntersectMode::FirstHit);
                            if (!worldHit.hit) {
                                break;
                            }

                            const float hitDistance = length(worldHit.hitPositionW - startState.position);
                            if (hitDistance >= targetDistance - distanceEpsilon) {
                                break;
                            }

                            buildIntersectionNormal(scene, worldHit);
                            const InstanceRecord &instance = scene.instances[worldHit.instanceIndex];
                            if (instance.geometryType != GeometryType::PointCloud) {
                                tracedTransmittance = 0.0f;
                                break;
                            }

                            if (worldHit.primitiveIndex == startPrimitiveIndex || worldHit.primitiveIndex ==
                                endPrimitiveIndex) {
                                ray.origin = worldHit.hitPositionW + ray.direction * distanceEpsilon;
                                continue;
                            }

                            const Point &occluderSurfel = scene.points[worldHit.primitiveIndex];
                            float3 occluderNormal = normalize(cross(occluderSurfel.tanU, occluderSurfel.tanV));
                            if (dot(occluderNormal, -ray.direction) < 0.0f) {
                                occluderNormal = -occluderNormal;
                            }

                            const float alphaGeomOccluder = worldHit.alphaGeom;
                            const float alphaEffective = occluderSurfel.opacity * alphaGeomOccluder;
                            const float oneMinusAlpha = sycl::fmax(0.0f, 1.0f - alphaEffective);
                            const float prefixTransmittance = tracedTransmittance;
                            tracedTransmittance *= oneMinusAlpha;
                            ray.origin = worldHit.hitPositionW + ray.direction * distanceEpsilon;

                            if (!eventRecord.isDirectLightSample) {
                                nullSamplingWeight *= qNullInv;
                            }

                            const float2 uv = phiInverse(worldHit.hitPositionW, occluderSurfel);
                            const float uOcc = uv.x();
                            const float vOcc = uv.y();
                            const float occluderScaleU = occluderSurfel.scale.x();
                            const float occluderScaleV = occluderSurfel.scale.y();
                            if (occluderScaleU <= 1.0e-12f || occluderScaleV <= 1.0e-12f) {
                                continue;
                            }

                            const float3 tangentU = occluderSurfel.tanU;
                            const float3 tangentV = occluderSurfel.tanV;
                            const float3 localBasisU = tangentU / occluderScaleU;
                            const float3 localBasisV = tangentV / occluderScaleV;
                            const float denominator = dot(occluderNormal, dxy);
                            if (sycl::fabs(denominator) <= 1.0e-8f) {
                                continue;
                            }

                            const float inverseDenominator = 1.0f / denominator;
                            const float lambdaOccluder =
                                    dot(occluderNormal, occluderSurfel.position - yPosition) * inverseDenominator;
                            const float3 commonU =
                                    localBasisU - occluderNormal * (dot(dxy, localBasisU) * inverseDenominator);
                            const float3 commonV =
                                    localBasisV - occluderNormal * (dot(dxy, localBasisV) * inverseDenominator);
                            const float3 dUiDx = lambdaOccluder * commonU;
                            const float3 dViDx = lambdaOccluder * commonV;
                            const float3 dUiDy = (1.0f - lambdaOccluder) * commonU;
                            const float3 dViDy = (1.0f - lambdaOccluder) * commonV;
                            const float3 dUiDspi =
                                    occluderNormal * (dot(dxy, tangentU) / occluderScaleU) * inverseDenominator -
                                    localBasisU;
                            const float3 dViDspi =
                                    occluderNormal * (dot(dxy, tangentV) / occluderScaleV) * inverseDenominator -
                                    localBasisV;

                            const float radiusSquared = uOcc * uOcc + vOcc * vOcc;
                            const float oneMinusRadiusSquared = 1.0f - radiusSquared;
                            if (oneMinusRadiusSquared <= 1.0e-8f) {
                                continue;
                            }

                            const float betaScale = 4.0f * sycl::exp(occluderSurfel.beta);
                            const float dAlphaGeomDu =
                                    -2.0f * betaScale * uOcc * alphaGeomOccluder / oneMinusRadiusSquared;
                            const float dAlphaGeomDv =
                                    -2.0f * betaScale * vOcc * alphaGeomOccluder / oneMinusRadiusSquared;
                            const float3 dAlphaEffectiveDx =
                                    occluderSurfel.opacity * (dAlphaGeomDu * dUiDx + dAlphaGeomDv * dViDx);
                            const float3 dAlphaEffectiveDy =
                                    occluderSurfel.opacity * (dAlphaGeomDu * dUiDy + dAlphaGeomDv * dViDy);

                            const float3 dAlphaEffectiveDspi =
                                    occluderSurfel.opacity * (dAlphaGeomDu * dUiDspi + dAlphaGeomDv * dViDspi);
                            const float dAlphaEffectiveDScaleU =
                                    2.0f * betaScale * uOcc * uOcc * alphaEffective / (
                                        occluderScaleU * oneMinusRadiusSquared);
                            const float dAlphaEffectiveDScaleV =
                                    2.0f * betaScale * vOcc * vOcc * alphaEffective / (
                                        occluderScaleV * oneMinusRadiusSquared);
                            const float dAlphaEffectiveDEta = alphaGeomOccluder;
                            const float dAlphaEffectiveDBeta =
                                    betaScale * sycl::log(oneMinusRadiusSquared) * alphaEffective;

                            float3 gradTangentUOcc{0.0f, 0.0f, 0.0f};
                            float3 gradTangentVOcc{0.0f, 0.0f, 0.0f};
                            const float nDotD = dot(occluderNormal, rayDirection);
                            if (sycl::fabs(nDotD) > 1.0e-8f) {
                                const float3 hitMinusSp = worldHit.hitPositionW - occluderSurfel.position;
                                const float3 aOcc = occluderSurfel.position - yPosition;
                                const float nDotA = dot(occluderNormal, aOcc);
                                const float invNDotD = 1.0f / nDotD;
                                const float invNDotDSquared = invNDotD * invNDotD;
                                const float3 qOcc = (cross(occluderNormal, aOcc) * nDotD - nDotA * cross(
                                                         occluderNormal, rayDirection)) * invNDotDSquared;
                                const float3 duDRotation =
                                        qOcc * (dot(rayDirection, tangentU) / occluderScaleU) + cross(
                                            tangentU, hitMinusSp) / occluderScaleU;
                                const float3 dvDRotation =
                                        qOcc * (dot(rayDirection, tangentV) / occluderScaleV) + cross(
                                            tangentV, hitMinusSp) / occluderScaleV;
                                const float3 dAlphaEffectiveDRotation =
                                        occluderSurfel.opacity * (
                                            dAlphaGeomDu * duDRotation + dAlphaGeomDv * dvDRotation);
                                gradTangentUOcc = cross(dAlphaEffectiveDRotation, occluderSurfel.tanU);
                                gradTangentVOcc = cross(dAlphaEffectiveDRotation, occluderSurfel.tanV);
                            }

                            if (storedOccluderCount < kMaxSplatEventsPerRay) {
                                OccluderDerivative &occluderDerivative = occluderDerivatives[storedOccluderCount];
                                occluderDerivative.gradPosition = dAlphaEffectiveDspi;
                                occluderDerivative.gradScaleU = dAlphaEffectiveDScaleU;
                                occluderDerivative.gradScaleV = dAlphaEffectiveDScaleV;
                                occluderDerivative.gradEta = dAlphaEffectiveDEta;
                                occluderDerivative.gradBeta = dAlphaEffectiveDBeta;
                                occluderDerivative.gradTangentU = gradTangentUOcc;
                                occluderDerivative.gradTangentV = gradTangentVOcc;
                                occluderDerivative.gradAlphaWrtStartPoint = dAlphaEffectiveDx;
                                occluderDerivative.gradAlphaWrtEndPoint = dAlphaEffectiveDy;
                                occluderDerivative.prefixTransmittance = prefixTransmittance;
                                occluderDerivative.oneMinusAlpha = oneMinusAlpha;
                                occluderDerivative.primitiveIndex = worldHit.primitiveIndex;
                                storedOccluderCount++;
                            }
                        }

                        edgeTransmittance = tracedTransmittance;
                        float suffixTransmittanceForTauGradient = 1.0f;

                        for (uint32_t reverseIndex = storedOccluderCount; reverseIndex > 0u; --reverseIndex) {
                            const uint32_t occluderIndex = reverseIndex - 1u;
                            const OccluderDerivative &occluderDerivative = occluderDerivatives[occluderIndex];
                            const float tauDerivativeScale =
                                    -occluderDerivative.prefixTransmittance * suffixTransmittanceForTauGradient;
                            gradTauWrtStartPoint += tauDerivativeScale * occluderDerivative.gradAlphaWrtStartPoint;
                            gradTauWrtEndPoint += tauDerivativeScale * occluderDerivative.gradAlphaWrtEndPoint;
                            suffixTransmittanceForTauGradient *= occluderDerivative.oneMinusAlpha;
                        }
                    }

                    const float scalarEdgeWeight = scalarEdgeWeightBase * nullSamplingWeight;
                    const float scalarMaterialEdgeWeight = scalarEdgeWeight * endJacobianUv;

                    const float3 gradientWrtStartPositionBeforeSpp =
                            scalarMaterialEdgeWeight * (
                                edgeTransmittance * dGeometricTermDStartPosition + geometricTerm *
                                gradTauWrtStartPoint);
                    const float3 gradientWrtEndPositionBeforeSpp =
                            scalarMaterialEdgeWeight * (
                                edgeTransmittance * dGeometricTermDEndPosition + geometricTerm * gradTauWrtEndPoint);

                    float3 startTranslationGradient = gradientWrtStartPositionBeforeSpp * invSpp;
                    float3 endTranslationGradient = gradientWrtEndPositionBeforeSpp * invSpp;
                    float startScaleUGradient = 0.0f;
                    float startScaleVGradient = 0.0f;
                    float endScaleUGradient = 0.0f;
                    float endScaleVGradient = 0.0f;
                    float3 startTangentUGradient{0.0f, 0.0f, 0.0f};
                    float3 startTangentVGradient{0.0f, 0.0f, 0.0f};
                    float3 endTangentUGradient{0.0f, 0.0f, 0.0f};
                    float3 endTangentVGradient{0.0f, 0.0f, 0.0f};
                    float3 startAlbedoGradient{0.0f, 0.0f, 0.0f};
                    float3 endAlbedoGradient{0.0f, 0.0f, 0.0f};

                    const float3 startToEnd = endState.position - startState.position;
                    const float distanceSquared = dot(startToEnd, startToEnd);

                    if (!eventRecord.isEndPointOnly) {
                        const float startU = eventRecord.startSurface.uv.x();
                        const float startV = eventRecord.startSurface.uv.y();
                        const float startScaleU = startSurfel.scale.x();
                        const float startScaleV = startSurfel.scale.y();

                        startScaleUGradient = dot(gradientWrtStartPositionBeforeSpp, startU * startSurfel.tanU) *
                                              invSpp;
                        startScaleVGradient = dot(gradientWrtStartPositionBeforeSpp, startV * startSurfel.tanV) *
                                              invSpp;
                        startTangentUGradient += gradientWrtStartPositionBeforeSpp * (startU * startScaleU * invSpp);
                        startTangentVGradient += gradientWrtStartPositionBeforeSpp * (startV * startScaleV * invSpp);

                        if (distanceSquared > 1.0e-12f) {
                            const float3 directionStartToEnd = startToEnd / sycl::sqrt(distanceSquared);
                            const float cosineAtEnd = dot(endState.orientedNormal, -directionStartToEnd);
                            const float3 dGeometricTermDStartNormal =
                                    directionStartToEnd * (cosineAtEnd / distanceSquared);
                            const float3 gradientWrtStartOrientedNormal =
                                    scalarMaterialEdgeWeight * edgeTransmittance * dGeometricTermDStartNormal;
                            const float3 rawStartNormal = cross(startSurfel.tanU, startSurfel.tanV);
                            const float rawStartNormalLength = length(rawStartNormal);

                            if (rawStartNormalLength > 1.0e-8f) {
                                const float3 unitRawStartNormal = rawStartNormal / rawStartNormalLength;
                                const float orientationSign = dot(unitRawStartNormal, startState.orientedNormal) >= 0.0f
                                                                  ? 1.0f
                                                                  : -1.0f;
                                const float3 gradientWrtRawStartNormal =
                                        orientationSign * gradientWrtStartOrientedNormal;
                                const float3 projectedGradientWrtRawStartNormal =
                                        gradientWrtRawStartNormal - unitRawStartNormal * dot(
                                            unitRawStartNormal, gradientWrtRawStartNormal);
                                const float3 gradientWrtStartCross =
                                        projectedGradientWrtRawStartNormal / rawStartNormalLength;
                                startTangentUGradient += cross(startSurfel.tanV, gradientWrtStartCross) * invSpp;
                                startTangentVGradient += cross(gradientWrtStartCross, startSurfel.tanU) * invSpp;
                            }
                        }

                        const float throughputWithoutAlbedoR = sycl::fabs(startSurfel.albedo.x()) > 1.0e-8f
                                                                   ? eventRecord.sampledEdgeThroughput.x() / startSurfel
                                                                     .albedo.x()
                                                                   : 0.0f;
                        const float throughputWithoutAlbedoG = sycl::fabs(startSurfel.albedo.y()) > 1.0e-8f
                                                                   ? eventRecord.sampledEdgeThroughput.y() / startSurfel
                                                                     .albedo.y()
                                                                   : 0.0f;
                        const float throughputWithoutAlbedoB = sycl::fabs(startSurfel.albedo.z()) > 1.0e-8f
                                                                   ? eventRecord.sampledEdgeThroughput.z() / startSurfel
                                                                     .albedo.z()
                                                                   : 0.0f;
                        startAlbedoGradient = float3{
                                                  throughputWithoutAlbedoR * endRadiance.x(),
                                                  throughputWithoutAlbedoG * endRadiance.y(),
                                                  throughputWithoutAlbedoB * endRadiance.z()
                                              } * (edgeTransmittance * geometricTerm * endJacobianUv * invSpp);
                    } else {
                        startTranslationGradient = float3{0.0f, 0.0f, 0.0f};
                    }

                    if (!eventRecord.isDirectLightSample) {
                        endScaleUGradient = dot(gradientWrtEndPositionBeforeSpp, endU * endSurfel.tanU) * invSpp;
                        endScaleVGradient = dot(gradientWrtEndPositionBeforeSpp, endV * endSurfel.tanV) * invSpp;
                        endScaleUGradient += scalarEdgeWeight * edgeTransmittance * geometricTerm *
                                dEndJacobianUvDScaleU * invSpp;
                        endScaleVGradient += scalarEdgeWeight * edgeTransmittance * geometricTerm *
                                dEndJacobianUvDScaleV * invSpp;
                        endTangentUGradient += gradientWrtEndPositionBeforeSpp * (endU * endScaleU * invSpp);
                        endTangentVGradient += gradientWrtEndPositionBeforeSpp * (endV * endScaleV * invSpp);

                        if (distanceSquared > 1.0e-12f) {
                            const float3 directionStartToEnd = startToEnd / sycl::sqrt(distanceSquared);
                            const float cosineAtStart = dot(startState.orientedNormal, directionStartToEnd);
                            const float3 dGeometricTermDEndNormal =
                                    -directionStartToEnd * (cosineAtStart / distanceSquared);
                            const float3 gradientWrtEndOrientedNormal =
                                    scalarMaterialEdgeWeight * edgeTransmittance * dGeometricTermDEndNormal;
                            const float3 rawEndNormal = cross(endSurfel.tanU, endSurfel.tanV);
                            const float rawEndNormalLength = length(rawEndNormal);

                            if (rawEndNormalLength > 1.0e-8f) {
                                const float3 unitRawEndNormal = rawEndNormal / rawEndNormalLength;
                                const float orientationSign = dot(unitRawEndNormal, endState.orientedNormal) >= 0.0f
                                                                  ? 1.0f
                                                                  : -1.0f;
                                const float3 gradientWrtRawEndNormal = orientationSign * gradientWrtEndOrientedNormal;
                                const float3 projectedGradientWrtRawEndNormal =
                                        gradientWrtRawEndNormal - unitRawEndNormal * dot(
                                            unitRawEndNormal, gradientWrtRawEndNormal);
                                const float3 gradientWrtEndCross =
                                        projectedGradientWrtRawEndNormal / rawEndNormalLength;
                                endTangentUGradient += cross(endSurfel.tanV, gradientWrtEndCross) * invSpp;
                                endTangentVGradient += cross(gradientWrtEndCross, endSurfel.tanU) * invSpp;
                            }
                        }

                        const float endRadianceWithoutAlbedoR = sycl::fabs(endSurfel.albedo.x()) > 1.0e-8f
                                                                    ? endRadiance.x() / endSurfel.albedo.x()
                                                                    : 0.0f;
                        const float endRadianceWithoutAlbedoG = sycl::fabs(endSurfel.albedo.y()) > 1.0e-8f
                                                                    ? endRadiance.y() / endSurfel.albedo.y()
                                                                    : 0.0f;
                        const float endRadianceWithoutAlbedoB = sycl::fabs(endSurfel.albedo.z()) > 1.0e-8f
                                                                    ? endRadiance.z() / endSurfel.albedo.z()
                                                                    : 0.0f;
                        endAlbedoGradient = float3{
                                                eventRecord.sampledEdgeThroughput.x() * endRadianceWithoutAlbedoR,
                                                eventRecord.sampledEdgeThroughput.y() * endRadianceWithoutAlbedoG,
                                                eventRecord.sampledEdgeThroughput.z() * endRadianceWithoutAlbedoB
                                            } * (edgeTransmittance * geometricTerm * endJacobianUv * invSpp);
                    } else {
                        endTranslationGradient = float3{0.0f, 0.0f, 0.0f};
                    }

                    float suffixTransmittance = 1.0f;
                    for (uint32_t reverseIndex = storedOccluderCount; reverseIndex > 0u; --reverseIndex) {
                        const uint32_t occluderIndex = reverseIndex - 1u;
                        const uint32_t occluderRecordIndex = eventRecordBase + 2u + occluderIndex;
                        const OccluderDerivative &occluderDerivative = occluderDerivatives[occluderIndex];
                        const float visibilityDerivativeScale =
                                -occluderDerivative.prefixTransmittance * suffixTransmittance * geometricTerm *
                                scalarMaterialEdgeWeight * invSpp;

                        SurfelGradientRecord occluderRecord{};
                        occluderRecord.primitiveIndex = occluderDerivative.primitiveIndex;
                        const float3 positionContribution = visibilityDerivativeScale * occluderDerivative.gradPosition;
                        const float3 tangentUContribution = visibilityDerivativeScale * occluderDerivative.gradTangentU;
                        const float3 tangentVContribution = visibilityDerivativeScale * occluderDerivative.gradTangentV;
                        occluderRecord.gradPositionX = positionContribution.x();
                        occluderRecord.gradPositionY = positionContribution.y();
                        occluderRecord.gradPositionZ = positionContribution.z();
                        occluderRecord.gradScaleU = visibilityDerivativeScale * occluderDerivative.gradScaleU;
                        occluderRecord.gradScaleV = visibilityDerivativeScale * occluderDerivative.gradScaleV;
                        occluderRecord.gradTangentUX = tangentUContribution.x();
                        occluderRecord.gradTangentUY = tangentUContribution.y();
                        occluderRecord.gradTangentUZ = tangentUContribution.z();
                        occluderRecord.gradTangentVX = tangentVContribution.x();
                        occluderRecord.gradTangentVY = tangentVContribution.y();
                        occluderRecord.gradTangentVZ = tangentVContribution.z();
                        occluderRecord.gradEta = visibilityDerivativeScale * occluderDerivative.gradEta;
                        occluderRecord.gradBeta = visibilityDerivativeScale * occluderDerivative.gradBeta;
                        occluderRecord.gradAlbedoR = 0.0f;
                        occluderRecord.gradAlbedoG = 0.0f;
                        occluderRecord.gradAlbedoB = 0.0f;
                        gradientRecords[occluderRecordIndex] = occluderRecord;
                        suffixTransmittance *= occluderDerivative.oneMinusAlpha;
                    }

                    SurfelGradientRecord startRecord{};
                    startRecord.primitiveIndex = startPrimitiveIndex;
                    startRecord.gradPositionX = startTranslationGradient.x();
                    startRecord.gradPositionY = startTranslationGradient.y();
                    startRecord.gradPositionZ = startTranslationGradient.z();
                    startRecord.gradScaleU = startScaleUGradient;
                    startRecord.gradScaleV = startScaleVGradient;
                    startRecord.gradTangentUX = startTangentUGradient.x();
                    startRecord.gradTangentUY = startTangentUGradient.y();
                    startRecord.gradTangentUZ = startTangentUGradient.z();
                    startRecord.gradTangentVX = startTangentVGradient.x();
                    startRecord.gradTangentVY = startTangentVGradient.y();
                    startRecord.gradTangentVZ = startTangentVGradient.z();
                    startRecord.gradEta = 0.0f;
                    startRecord.gradBeta = 0.0f;
                    startRecord.gradAlbedoR = 0.0f;
                    startRecord.gradAlbedoG = 0.0f;
                    startRecord.gradAlbedoB = 0.0f;

                    SurfelGradientRecord endRecord{};
                    endRecord.primitiveIndex = endPrimitiveIndex;
                    endRecord.gradPositionX = endTranslationGradient.x();
                    endRecord.gradPositionY = endTranslationGradient.y();
                    endRecord.gradPositionZ = endTranslationGradient.z();
                    endRecord.gradScaleU = endScaleUGradient;
                    endRecord.gradScaleV = endScaleVGradient;
                    endRecord.gradTangentUX = endTangentUGradient.x();
                    endRecord.gradTangentUY = endTangentUGradient.y();
                    endRecord.gradTangentUZ = endTangentUGradient.z();
                    endRecord.gradTangentVX = endTangentVGradient.x();
                    endRecord.gradTangentVY = endTangentVGradient.y();
                    endRecord.gradTangentVZ = endTangentVGradient.z();
                    endRecord.gradEta = 0.0f;
                    endRecord.gradBeta = 0.0f;
                    endRecord.gradAlbedoR = endAlbedoGradient.x();
                    endRecord.gradAlbedoG = endAlbedoGradient.y();
                    endRecord.gradAlbedoB = endAlbedoGradient.z();

                    gradientRecords[eventRecordBase + 0u] = startRecord;
                    gradientRecords[eventRecordBase + 1u] = endRecord;
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
        uint32_t materialVertexEventCount,
        uint32_t materialEdgeEventCount,
        uint32_t cameraIndex) {
        const GradientRecordRanges ranges = makeGradientRecordRanges(
            measurementEventCount,
            measurementTwoPointEventCount,
            materialVertexEventCount,
            materialEdgeEventCount);

        if (measurementEventCount >= pkg.intermediates.maxMeasurementEventCount) {
            Log::PA_ERROR(
                "Overflow: measurementEventCount={} >= maxMeasurementEventCount={}",
                measurementEventCount,
                pkg.intermediates.maxMeasurementEventCount);
        }

        if (measurementTwoPointEventCount >= pkg.intermediates.maxMeasurementTwoPointEventCount) {
            Log::PA_ERROR(
                "Overflow: measurementTwoPointEventCount={} >= maxMeasurementTwoPointEventCount={}",
                measurementTwoPointEventCount,
                pkg.intermediates.maxMeasurementTwoPointEventCount);
        }

        if (materialVertexEventCount >= pkg.intermediates.maxMaterialVertexEventCount) {
            Log::PA_ERROR(
                "Overflow: materialVertexEventCount={} >= maxMaterialVertexEventCount={}",
                materialVertexEventCount,
                pkg.intermediates.maxMaterialVertexEventCount);
        }

        if (materialEdgeEventCount >= pkg.intermediates.maxMaterialEdgeEventCount) {
            Log::PA_ERROR(
                "Overflow: materialEdgeEventCount={} >= maxMaterialEdgeEventCount={}",
                materialEdgeEventCount,
                pkg.intermediates.maxMaterialEdgeEventCount);
        }

        if (ranges.totalCount >= pkg.intermediates.maxGradientRecordCount) {
            Log::PA_ERROR(
                "Overflow: gradientRecordCount={} >= maxGradientRecordCount={}",
                ranges.totalCount,
                pkg.intermediates.maxGradientRecordCount);
        }

        if (ranges.totalCount > pkg.intermediates.maxGradientRecordCount) {
            throw std::runtime_error("gradient record scratch buffer too small");
        }

        Log::PA_DEBUG(
            "Event counts: measurement={}, measurementTwoPoint={}, materialVertex={}, materialEdge={}, gradientRecords={}",
            measurementEventCount,
            measurementTwoPointEventCount,
            materialVertexEventCount,
            materialEdgeEventCount,
            ranges.totalCount);

        if (measurementEventCount > 0u) {
            ScopedTimer timer("measurementGradientEvent", spdlog::level::debug);
            measurementGradientEvent(
                pkg,
                cameraIndex,
                measurementEventCount,
                ranges.measurementOffset);
        }

        if (measurementTwoPointEventCount > 0u) {
            ScopedTimer timer("measurementGradientEventXY", spdlog::level::debug);
            measurementGradientEventXY(
                pkg,
                measurementTwoPointEventCount,
                ranges.measurementTwoPointOffset);
        }

        if (materialVertexEventCount > 0u) {
            ScopedTimer timer("materialVertexGradientEvent", spdlog::level::debug);
            materialVertexGradientEvent(
                pkg,
                materialVertexEventCount,
                ranges.materialVertexOffset);
        }

        if (materialEdgeEventCount > 0u) {
            ScopedTimer timer("materialEdgeGradientEvent", spdlog::level::debug);
            materialEdgeGradientEvent(
                pkg,
                materialEdgeEventCount,
                ranges.materialEdgeOffset);
        }

        if (ranges.totalCount > 0u) {
            ScopedTimer timer("reduceSurfelGradientRecords", spdlog::level::debug);
            reduceSurfelGradientRecords(
                pkg,
                ranges.totalCount);
        }
    }
}
