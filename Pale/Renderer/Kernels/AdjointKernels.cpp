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

        return directRadiance + indirectRadiance;
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

                    float cameraCosine = dot(sensor.camera.forward,
                                             primaryRay.direction);
                    //initialAdjointWeight *= cameraCosine;

                    RayState rayState{};
                    rayState.ray = primaryRay;
                    rayState.pathThroughput = initialAdjointWeight;
                    rayState.bounceIndex = 0;
                    rayState.pixelIndex = pixelIndex;
                    rayState.traversalIndex = 0u;
                    rayState.transmission = 1.0f;
                    rayState.pathId = pixelIndex; // 0 .. (W*H-1)

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
                            const float invQReflect = 1.0f / settings.sampling.qReflect;

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
                            float segmentUvJacobianAtEnd = 1.0f;
                            float segmentPreviousLocalPdfFromStoredVertex = 1.0f;

                            if (hasPendingState) {
                                const PendingCameraSegment previousCameraSegment = pendingCameraSegment;
                                const PendingAdjointStageX previousAdjointStage = pendingAdjointStage;


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

                                    segmentUvJacobianAtEnd = surfel.scale.x() * surfel.scale.y();

                                    segmentPreviousLocalPdfFromStoredVertex = storedSurfel.scale.x() * storedSurfel.
                                                                              scale.y();
                                }


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

                                const bool isCameraAttachedRandomContinuation =
                                        previousAdjointStage.valid &&
                                        currentRayState.bounceIndex == 1u;
                                if (isCameraAttachedRandomContinuation) {
                                    MeasurementGradientEventXY measurementTwoPointEvent{};
                                    measurementTwoPointEvent.xSurface = previousAdjointStage.current.surface;
                                    measurementTwoPointEvent.ySurface = currentSurface;
                                    measurementTwoPointEvent.xPathThroughput =
                                            previousAdjointStage.current.transmission *
                                            previousAdjointStage.current.pathThroughput /
                                            (segmentAreaPdfFromStoredVertex * qReflect);
                                    measurementTwoPointEvent.transmissionPreviousSegment =
                                            previousAdjointStage.current.transmission;
                                    measurementTwoPointEvent.transmission =
                                            currentRayState.transmission;
                                    appendEventAtomic(
                                        intermediates.countMeasurementTwoPointEvents,
                                        intermediates.measurementTwoPointEvents,
                                        intermediates.maxMeasurementTwoPointEventCount,
                                        measurementTwoPointEvent);
                                }


                                // First material point after camera attached point with direct light sample
                                if (previousAdjointStage.valid && currentRayState.bounceIndex >= 1) {
                                    // This event collects gradient with respet to surface Y at the first material vertex event
                                    // Bounce index == 1, then pair with p1
                                    // Bounce index == 2, then pair with p2
                                    MaterialVertexGradientEvent materialVertexEvent{};
                                    materialVertexEvent.surface = currentSurface;


                                    materialVertexEvent.adjointWeightAtVertex =
                                            currentRayState.pathThroughput *
                                            currentRayState.transmission / qReflect;

                                    materialVertexEvent.pathId = currentRayState.pathId;
                                    materialVertexEvent.bounceIndex = currentRayState.bounceIndex;
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
                                    currentRayState.bounceIndex >= 1) {
                                    // Geometric gradient bettwen XYZ where Z is a light source we have two edge events for Y
                                    // Firs tbetween X and Y, which is adjoint path weight p0.
                                    // Second is between YZ which is adjoint path weight p1.
                                    // Start with XY path we only have End point gradients w.r.t to Y:
                                    // Curent ray state is from X to Y using cosine hemisphere sampling
                                    // We need p1 (outgoing at X) towards Y with local coordinate sampling
                                    MaterialEdgeGradientEvent materialEdgeEventXY{};
                                    materialEdgeEventXY.startSurface = previousAdjointStage.current.surface;
                                    materialEdgeEventXY.endSurface = currentSurface;
                                    float invSegmentUvPdf =
                                            1.0f / (segmentAreaPdfFromStoredVertex *
                                                    segmentUvJacobianAtEnd);

                                    materialEdgeEventXY.sampledEdgeThroughput =
                                            previousAdjointStage.current.pathThroughput *
                                            previousAdjointStage.current.transmission *
                                            previousAdjointStage.current.bsdf *
                                            previousAdjointStage.current.alpha
                                            * invSegmentUvPdf / qReflect;
                                    materialEdgeEventXY.isDirectLightSample = false;
                                    materialEdgeEventXY.writeOcclusionGradients = currentRayState.bounceIndex > 1;
                                    // Start writing occlusion gradients only on x2->x3 transport, x1->x2 is handled by measurement terms
                                    materialEdgeEventXY.pathId = currentRayState.pathId;
                                    materialEdgeEventXY.startBounceIndex = previousAdjointStage.current.bounceIndex;

                                    appendEventAtomic(
                                        intermediates.countMaterialEndEdgeEvents,
                                        intermediates.materialEndEdgeEvents,
                                        intermediates.maxMaterialEndEdgeEventCount,
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
                                                rng::makeSeed(renderSeed, currentRayState.pathId, spp,
                                                              rng::kStreamDirectLight,
                                                              currentRayState.traversalIndex * 1315423911u +
                                                              shadowRaySample
                                                              + 0x7f4a7c15u);
                                        rng::Xorshift128 directLightSampleRng(lightSampleSeed);
                                        const auto lightSample = sampleMeshAreaLight(scene, directLightSampleRng);
                                        if (!lightSample.valid) {
                                            continue;
                                        }
                                        const uint32_t lightPrimitiveIndex = lightSample.surface.primitiveIndex;
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
                                        shadowRay.origin = startState.position + lightDirection * distanceEpsilon;
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
                                        // ---------------------------------------------------------------------
                                        // Prefix carried to x2 BEFORE local scattering at x2
                                        // This is the correct adjoint prefix for the BSDF derivative event.
                                        // DO NOT multiply by segmentGeometryFromStoredVertex / segmentAreaPdfFromStoredVertex again.
                                        // ---------------------------------------------------------------------
                                        const float3 betaIncX2 =
                                                currentRayState.pathThroughput * currentRayState.transmission;

                                        const float alphaX2 = worldHit.alphaGeom * surfel.opacity;
                                        const float invLightPdfArea =
                                                1.0f / (lightSample.pdfSelectLight * lightSample.pdfArea);
                                        const float invSamplePDF = invLightPdfArea * invQReflect * invSampleCount;

                                        materialDirectLightEdgeEvent.betaIncrement = betaIncX2;
                                        materialDirectLightEdgeEvent.alpha = alphaX2;
                                        materialDirectLightEdgeEvent.bsdf = surfelBsdf;
                                        materialDirectLightEdgeEvent.invSamplePDF = invSamplePDF;


                                        materialDirectLightEdgeEvent.segmentTransmittance = shadowTransmission;
                                        materialDirectLightEdgeEvent.directLightRadiance = directLightRadiance;
                                        materialDirectLightEdgeEvent.isDirectLightSample = true;
                                        materialDirectLightEdgeEvent.writeOcclusionGradients = true;
                                        materialDirectLightEdgeEvent.pathId = currentRayState.pathId;
                                        materialDirectLightEdgeEvent.startBounceIndex = currentRayState.bounceIndex;

                                        appendEventAtomic(
                                            intermediates.countMaterialStartEdgeEvents,
                                            intermediates.materialStartEdgeEvents,
                                            intermediates.maxMaterialStartEdgeEventCount,
                                            materialDirectLightEdgeEvent);
                                    }

                                    // third event:
                                    // Auxiliary recursive start-edge sample at the current vertex Y.
                                    // Sample a direction at Y, then perform the same front-to-back
                                    // null/reflect selection logic along that ray to obtain Z_aux.
                                    // Finally convert the directional PDF to area measure.
                                    {
                                        const uint64_t auxiliaryRecursiveSeed =
                                                rng::makeSeed(
                                                    renderSeed,
                                                    currentRayState.pathId,
                                                    spp,
                                                    rng::kStreamDirection,
                                                    currentRayState.traversalIndex * 2246822519u + 0x51ed270bu);

                                        rng::Xorshift128 auxiliaryRecursiveRng(auxiliaryRecursiveSeed);

                                        float3 auxiliaryDirectionWorld{0.0f, 0.0f, 0.0f};
                                        float auxiliaryDirectionPdf = 0.0f;

                                        sampleUniformHemisphereAroundNormal(
                                            auxiliaryRecursiveRng,
                                            startState.orientedNormal,
                                            auxiliaryDirectionWorld,
                                            auxiliaryDirectionPdf);

                                        if (auxiliaryDirectionPdf > 1.0e-12f) {
                                            constexpr float distanceEpsilon = 1.0e-4f;
                                            constexpr uint32_t maxAuxiliaryTraversals = 32u;

                                            Ray auxiliaryRay{};
                                            auxiliaryRay.origin =
                                                    startState.position + startState.orientedNormal * distanceEpsilon;
                                            auxiliaryRay.direction = auxiliaryDirectionWorld;
                                            auxiliaryRay.normal = startState.orientedNormal;

                                            float auxiliaryDiscreteSelectionPdf = 1.0f;
                                            bool foundAuxiliarySurface = false;
                                            PointCloudSurfaceRecord auxiliarySurface{};

                                            for (uint32_t auxiliaryTraversalIndex = 0u;
                                                 auxiliaryTraversalIndex < maxAuxiliaryTraversals;
                                                 ++auxiliaryTraversalIndex) {
                                                WorldHit auxiliaryHit{};
                                                intersectScene(
                                                    auxiliaryRay,
                                                    &auxiliaryHit,
                                                    scene,
                                                    SurfelIntersectMode::FirstHit);

                                                if (!auxiliaryHit.hit) {
                                                    break;
                                                }

                                                buildIntersectionNormal(scene, auxiliaryHit);
                                                const InstanceRecord &auxiliaryInstance =
                                                        scene.instances[auxiliaryHit.instanceIndex];

                                                // Opaque mesh terminates the auxiliary sample.
                                                if (auxiliaryInstance.geometryType == GeometryType::Mesh) {
                                                    break;
                                                }

                                                if (auxiliaryInstance.geometryType != GeometryType::PointCloud) {
                                                    break;
                                                }

                                                // Avoid immediate self-hit on the current start surface.
                                                if (auxiliaryHit.primitiveIndex == currentSurface.primitiveIndex) {
                                                    auxiliaryRay.origin =
                                                            auxiliaryHit.hitPositionW + auxiliaryDirectionWorld *
                                                            distanceEpsilon;
                                                    continue;
                                                }

                                                const Point &auxiliarySurfel = scene.points[auxiliaryHit.
                                                    primitiveIndex];

                                                // Mirror the same Bernoulli null/reflect selection used by the actual path walk.
                                                const float randomNumber = auxiliaryRecursiveRng.nextFloat();

                                                if (randomNumber < qNull) {
                                                    // Null event: continue front-to-back.
                                                    auxiliaryDiscreteSelectionPdf *= qNull;
                                                    auxiliaryRay.origin =
                                                            auxiliaryHit.hitPositionW + auxiliaryDirectionWorld *
                                                            distanceEpsilon;
                                                    auxiliaryRay.normal =
                                                            computePointCloudOrientedNormal(
                                                                auxiliarySurfel, auxiliaryDirectionWorld);
                                                    continue;
                                                }

                                                // If you ever allow qNull + qReflect < 1, add a termination branch here.
                                                // For now this mirrors the current actual traversal logic:
                                                auxiliaryDiscreteSelectionPdf *= qReflect;

                                                RayState auxiliaryRayState{};
                                                auxiliaryRayState.ray = auxiliaryRay;
                                                auxiliaryRayState.pathId = currentRayState.pathId;
                                                auxiliaryRayState.pixelIndex = currentRayState.pixelIndex;

                                                auxiliarySurface =
                                                        makePointCloudSurfaceRecord(
                                                            auxiliaryHit,
                                                            auxiliaryRayState,
                                                            scene);

                                                foundAuxiliarySurface = true;
                                                break;
                                            }

                                            if (foundAuxiliarySurface) {
                                                const Point &auxiliarySurfel =
                                                        scene.points[auxiliarySurface.primitiveIndex];

                                                const ReconstructedSurfelState auxiliaryState =
                                                        reconstructSurfelState(
                                                            auxiliarySurfel,
                                                            auxiliarySurface);

                                                const float3 yzVector =
                                                        auxiliaryState.position - startState.position;

                                                const float yzDistanceSquared = dot(yzVector, yzVector);
                                                if (yzDistanceSquared > 1.0e-12f) {
                                                    const float yzDistance = sycl::sqrt(yzDistanceSquared);
                                                    const float3 yzDirection = yzVector / yzDistance;

                                                    const float cosineAtEnd = sycl::fmax(
                                                        0.0f,
                                                        dot(auxiliaryState.orientedNormal, -yzDirection));

                                                    if (cosineAtEnd > 1.0e-8f) {
                                                        // Full auxiliary area PDF:
                                                        // q_A(z | y) = q_omega(omega | y) * P_sel(z | omega, y) * |n_z . (-omega)| / ||z-y||^2
                                                        const float auxiliaryAreaPdf =
                                                                auxiliaryDirectionPdf *
                                                                auxiliaryDiscreteSelectionPdf *
                                                                cosineAtEnd / yzDistanceSquared;

                                                        if (auxiliaryAreaPdf > 1.0e-12f) {
                                                            MaterialEdgeGradientEvent innerIntegralStartEdge{};
                                                            innerIntegralStartEdge.startSurface = currentSurface; // Y
                                                            innerIntegralStartEdge.endSurface = auxiliarySurface;
                                                            // Z_aux

                                                            const float3 betaIncrementAtStart =
                                                                    currentRayState.pathThroughput * currentRayState.
                                                                    transmission;

                                                            const float alphaAtStart =
                                                                    worldHit.alphaGeom * surfel.opacity;

                                                            // Local reflect selection at Y is separate from the auxiliary inner-integral sample PDF.
                                                            const float inverseAuxiliarySamplePdf =
                                                                    1.0f / (qReflect * auxiliaryAreaPdf);

                                                            innerIntegralStartEdge.betaIncrement =
                                                                    betaIncrementAtStart;
                                                            innerIntegralStartEdge.alpha = alphaAtStart;
                                                            innerIntegralStartEdge.bsdf = surfelBsdf;
                                                            innerIntegralStartEdge.invSamplePDF =
                                                                    inverseAuxiliarySamplePdf;

                                                            innerIntegralStartEdge.segmentTransmittance = 1.0f;
                                                            innerIntegralStartEdge.directLightRadiance = float3{
                                                                0.0f, 0.0f, 0.0f
                                                            };
                                                            innerIntegralStartEdge.isDirectLightSample = false;

                                                            // This auxiliary Y -> Z edge owns its own tau / occlusion derivative.
                                                            innerIntegralStartEdge.writeOcclusionGradients = true;

                                                            innerIntegralStartEdge.pathId = currentRayState.pathId;
                                                            innerIntegralStartEdge.startBounceIndex = currentRayState.
                                                                    bounceIndex;

                                                            appendEventAtomic(
                                                                intermediates.countMaterialStartEdgeEvents,
                                                                intermediates.materialStartEdgeEvents,
                                                                intermediates.maxMaterialStartEdgeEventCount,
                                                                innerIntegralStartEdge);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }


                                clearPendingCameraSegment(pendingCameraSegment);
                                // -------------------------------------------------------------
                                // Build segment metadata for stored-vertex -> currentSurface
                                // -------------------------------------------------------------
                                if (previousAdjointStage.valid) {
                                    const Point &storedSurfel = scene.points[previousAdjointStage.current.surface.
                                        primitiveIndex];
                                    const ReconstructedSurfelState storedState = reconstructSurfelState(
                                        storedSurfel, previousAdjointStage.current.surface);
                                    const ReconstructedSurfelState liveState = reconstructSurfelState(
                                        surfel, currentSurface);
                                    segmentGeometryFromStoredVertex = computeGeometricTermValue(
                                        storedState.position,
                                        liveState.position,
                                        storedState.orientedNormal,
                                        liveState.orientedNormal);
                                    segmentAreaPdfFromStoredVertex = computeSegmentAreaPdfFromUniformHemisphere(
                                        storedState, liveState, uniformHemispherePdf);
                                } else {
                                    segmentGeometryFromStoredVertex = 1.0f;
                                    segmentAreaPdfFromStoredVertex = 1.0f;
                                }
                                // -------------------------------------------------------------
                                // Push live surface into rolling adjoint history
                                // previous = old current, current = live hit
                                // -------------------------------------------------------------
                                const float cosineFromPrevious = previousCameraSegment.valid
                                                                     ? dot(sensor.camera.forward,
                                                                           currentRayState.ray.direction)
                                                                     : 1.0f;
                                const PendingAdjointVertex newCurrentVertex = makePendingAdjointVertex(
                                    currentSurface, currentRayState.bounceIndex,
                                    currentRayState.pathThroughput / qReflect, currentRayState.transmission,
                                    segmentGeometryFromStoredVertex, segmentAreaPdfFromStoredVertex, surfelBsdf, alpha,
                                    cosineFromPrevious);
                                pushPendingAdjointVertex(pendingAdjointStage, currentRayState.pathId,
                                                         currentRayState.pixelIndex, previousCameraSegment.valid,
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
        auto &debugImages = pkg.debugImages;
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


                        accumulateDebugGradientIfSelected(
                            debugImages,
                            settings.renderDebugGradientImages,
                            settings.surfelIndexForDebugImages,
                            eventRecord.xSurface.pathId,
                            occluderRecord);
                    }


                    accumulateDebugGradientIfSelected(
                        debugImages,
                        settings.renderDebugGradientImages,
                        settings.surfelIndexForDebugImages,
                        eventRecord.xSurface.pathId,
                        gradientRecord);
                });
        }).wait();
    }

    static void measurementGradientEventXY(RenderPackage &pkg, uint32_t onePointEventCount, uint32_t baseOffset) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto &debugImages = pkg.debugImages;
        const auto &photonMap = pkg.intermediates.map;
        MeasurementGradientEventXY *measurementXYEvent = pkg.intermediates.measurementTwoPointEvents;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;
        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;
        const float qNullInv = 1.0f / settings.sampling.qNull;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class firstHitGradientEventTag>(
                sycl::range<1>(onePointEventCount), [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex = static_cast<uint32_t>(globalId[0]);
                    static constexpr uint32_t recordsPerEvent = 1u + kMaxSplatEventsPerRay;
                    const uint32_t eventRecordBase = baseOffset + recordsPerEvent * eventIndex;
                    const uint32_t recordIndex = eventRecordBase;

                    for (uint32_t recordOffset = 0u; recordOffset < recordsPerEvent; ++recordOffset) {
                        SurfelGradientRecord invalidRecord{};
                        invalidRecord.primitiveIndex = kInvalidIndex;
                        gradientRecords[eventRecordBase + recordOffset] = invalidRecord;
                    }

                    const MeasurementGradientEventXY eventRecord = measurementXYEvent[eventIndex];
                    const uint32_t xPrimitiveIndex = eventRecord.xSurface.primitiveIndex;
                    const uint32_t yPrimitiveIndex = eventRecord.ySurface.primitiveIndex;
                    const Point &surfelX = scene.points[xPrimitiveIndex];
                    const Point &surfelY = scene.points[yPrimitiveIndex];
                    const ReconstructedSurfelState xState = reconstructSurfelState(surfelX, eventRecord.xSurface);
                    const ReconstructedSurfelState yState = reconstructSurfelState(surfelY, eventRecord.ySurface);
                    const uint64_t directLightSeed = rng::makeSeed(settings.random.seed, eventRecord.xSurface.pathId,
                                                                   0xffefeefefu, rng::kStreamDirectLight, eventIndex);
                    rng::Xorshift128 directLightRng(directLightSeed);

                    float3 outgoingRadianceY{0.0f, 0.0f, 0.0f};
                    if (settings.enableAdjointDirectLight) {
                        outgoingRadianceY = eventRecord.isDirectLightSample
                                                ? eventRecord.directLightRadiance
                                                : evaluateOutgoingRadianceWithLocalAlphaNoEmitters(
                                                    surfelY, eventRecord.ySurface, yState, photonMap, scene,
                                                    settings.numAdjointShadowRays, directLightRng);
                    } else {
                        outgoingRadianceY = evaluateOutgoingRadianceWithLocalAlpha(
                            surfelY, eventRecord.ySurface, yState, photonMap, scene, settings.numAdjointShadowRays,
                            directLightRng);
                    }

                    const float alphaX = eventRecord.xSurface.alphaGeom * surfelX.opacity;
                    const float brdfScaleX = surfelX.alpha_r * M_1_PIf;
                    const float3 brdfX = brdfScaleX * surfelX.albedo;
                    const float3 dGeometricTermDx = computeGeometricTermGradientWrtStartpoint(
                        xState.position, yState.position, xState.orientedNormal, yState.orientedNormal);
                    const float3 pathWeight = eventRecord.xPathThroughput;
                    const float3 transportWithoutTauAndGeometric = outgoingRadianceY * alphaX * brdfX;
                    const float scalarWeightWithoutTauAndGeometricBase = dot(
                        pathWeight, transportWithoutTauAndGeometric);
                    const float3 albedoWeightWithoutTauAndGeometricBase =
                            pathWeight * outgoingRadianceY * (alphaX * brdfScaleX);

                    struct OccluderDerivative {
                        float3 gradPosition{0.0f, 0.0f, 0.0f};
                        float gradScaleU = 0.0f, gradScaleV = 0.0f, gradEta = 0.0f, gradBeta = 0.0f;
                        float3 gradTangentU{0.0f, 0.0f, 0.0f}, gradTangentV{0.0f, 0.0f, 0.0f};
                        float3 gradAlphaWrtStartPoint{0.0f, 0.0f, 0.0f}, gradAlphaWrtEndPoint{0.0f, 0.0f, 0.0f};
                        float prefixTransmittance = 1.0f, oneMinusAlpha = 1.0f;
                        uint32_t primitiveIndex = kInvalidIndex;
                    };

                    OccluderDerivative occluderDerivatives[kMaxSplatEventsPerRay];
                    uint32_t storedOccluderCount = 0u;
                    float segmentTransmittance = 1.0f;
                    float nullSamplingWeight = 1.0f; {
                        constexpr float distanceEpsilon = 1e-4f;
                        const float3 segmentDirection = yState.position - xState.position;
                        const float targetDistance = length(segmentDirection);
                        if (targetDistance <= 1e-12f) {
                            return;
                        }

                        const float3 rayDirection = segmentDirection / targetDistance;
                        Ray ray{};
                        ray.origin = xState.position + rayDirection * distanceEpsilon;
                        ray.direction = rayDirection;

                        const float3 xPosition = xState.position;
                        const float3 yPosition = yState.position;
                        const float3 dxy = xPosition - yPosition;

                        while (true) {
                            WorldHit worldHit{};
                            intersectScene(ray, &worldHit, scene, SurfelIntersectMode::FirstHit);

                            if (!worldHit.hit) {
                                break;
                            }

                            const float hitDistance = length(worldHit.hitPositionW - xState.position);
                            if (hitDistance >= targetDistance - distanceEpsilon) {
                                break;
                            }

                            buildIntersectionNormal(scene, worldHit);

                            const auto &instance = scene.instances[worldHit.instanceIndex];
                            if (instance.geometryType != GeometryType::PointCloud) {
                                segmentTransmittance = 0.0f;
                                break;
                            }

                            const Point &occluderSurfel = scene.points[worldHit.primitiveIndex];
                            float3 occluderNormal = normalize(cross(occluderSurfel.tanU, occluderSurfel.tanV));
                            if (dot(occluderNormal, -ray.direction) < 0.0f) {
                                occluderNormal = -occluderNormal;
                            }

                            const float alphaGeomOccluder = worldHit.alphaGeom;
                            const float alphaEffective = occluderSurfel.opacity * alphaGeomOccluder;
                            const float oneMinusAlpha = sycl::fmax(0.0f, 1.0f - alphaEffective);
                            const float prefixTransmittance = segmentTransmittance;

                            segmentTransmittance *= oneMinusAlpha;
                            if (!eventRecord.isDirectLightSample) {
                                nullSamplingWeight *= qNullInv;
                            }

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
                            const float denominator = dot(occluderNormal, dxy);

                            if (sycl::fabs(denominator) <= 1e-8f) {
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
                                    occluderNormal * (dot(dxy, tangentU) / scaleU) * inverseDenominator - localBasisU;
                            const float3 dViDspi =
                                    occluderNormal * (dot(dxy, tangentV) / scaleV) * inverseDenominator - localBasisV;

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
                                    occluderSurfel.opacity * (dAlphaGeomDu * dUiDx + dAlphaGeomDv * dViDx);
                            const float3 dAlphaEffectiveDy =
                                    occluderSurfel.opacity * (dAlphaGeomDu * dUiDy + dAlphaGeomDv * dViDy);
                            const float3 dAlphaEffectiveDspi =
                                    occluderSurfel.opacity * (dAlphaGeomDu * dUiDspi + dAlphaGeomDv * dViDspi);
                            const float dAlphaEffectiveDScaleU =
                                    2.0f * betaScale * uOcc * uOcc * alphaEffective / (scaleU * oneMinusRadiusSquared);
                            const float dAlphaEffectiveDScaleV =
                                    2.0f * betaScale * vOcc * vOcc * alphaEffective / (scaleV * oneMinusRadiusSquared);
                            const float dAlphaEffectiveDEta = alphaGeomOccluder;
                            const float dAlphaEffectiveDBeta =
                                    betaScale * sycl::log(oneMinusRadiusSquared) * alphaEffective;

                            float3 gradTangentUOcc{0.0f, 0.0f, 0.0f};
                            float3 gradTangentVOcc{0.0f, 0.0f, 0.0f};

                            const float nDotD = dot(occluderNormal, rayDirection);
                            if (sycl::fabs(nDotD) > 1e-8f) {
                                const float3 hitMinusSp = worldHit.hitPositionW - occluderSurfel.position;
                                const float3 aOcc = occluderSurfel.position - yPosition;
                                const float nDotA = dot(occluderNormal, aOcc);
                                const float invNDotD = 1.0f / nDotD;
                                const float invNDotDSquared = invNDotD * invNDotD;
                                const float3 qOcc = (cross(occluderNormal, aOcc) * nDotD - nDotA * cross(
                                                         occluderNormal, rayDirection)) * invNDotDSquared;
                                const float3 duDRotation = qOcc * (dot(rayDirection, tangentU) / scaleU) + cross(
                                                               tangentU, hitMinusSp) / scaleU;
                                const float3 dvDRotation = qOcc * (dot(rayDirection, tangentV) / scaleV) + cross(
                                                               tangentV, hitMinusSp) / scaleV;
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
                    }

                    float3 gradTauWrtStartPoint{0.0f, 0.0f, 0.0f};
                    float3 gradTauWrtEndPoint{0.0f, 0.0f, 0.0f};
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

                    const float geometricTermXY = computeGeometricTermValue(
                        xState.position, yState.position, xState.orientedNormal, yState.orientedNormal);
                    const float scalarWeightWithoutTauAndGeometric =
                            scalarWeightWithoutTauAndGeometricBase * nullSamplingWeight;
                    const float3 albedoWeightWithoutTauAndGeometric =
                            albedoWeightWithoutTauAndGeometricBase * nullSamplingWeight;
                    const float3 gradientWrtWorldHitPositionX =
                            scalarWeightWithoutTauAndGeometric * (
                                segmentTransmittance * dGeometricTermDx + geometricTermXY * gradTauWrtStartPoint);
                    const float3x3 hitPointJacobianX = planeHitPointIntersectionJacobian(
                        eventRecord.xSurface.incomingDirection, xState.orientedNormal);
                    const float3 gradientWrtHitPositionX = transpose(hitPointJacobianX) * gradientWrtWorldHitPositionX;
                    const float3 xContribution = gradientWrtHitPositionX * invSpp;

                    float3 tanUContribution{0.0f, 0.0f, 0.0f};
                    float3 tanVContribution{0.0f, 0.0f, 0.0f};
                    const float3 primaryRayDirection = eventRecord.xSurface.incomingDirection;
                    const float nDotD = dot(xState.orientedNormal, primaryRayDirection);
                    const float3 rawCross = cross(surfelX.tanU, surfelX.tanV);
                    const float rawCrossLength = length(rawCross);

                    if (sycl::fabs(nDotD) > 1e-8f && rawCrossLength > 1e-8f) {
                        const float3 rawNormal = rawCross / rawCrossLength;
                        const float orientationSign = dot(rawNormal, xState.orientedNormal) >= 0.0f ? 1.0f : -1.0f;
                        const float3 pMinusX = surfelX.position - xState.position;
                        const float3 gradientWrtOrientedNormalFromMovedHit = pMinusX * (dot(primaryRayDirection,
                                                                                     gradientWrtWorldHitPositionX) /
                                                                                 nDotD);
                        const float3 segmentDirection = yState.position - xState.position;
                        const float distanceSquared = dot(segmentDirection, segmentDirection);
                        const float3 directionXToY = normalize(segmentDirection);
                        const float cosineAtY = dot(yState.orientedNormal, -directionXToY);
                        const float3 dGeometricTermDStartNormal = directionXToY * (cosineAtY / distanceSquared);
                        const float3 gradientWrtOrientedNormalExplicit =
                                scalarWeightWithoutTauAndGeometric * segmentTransmittance * dGeometricTermDStartNormal;
                        const float3 gradientWrtOrientedNormalX =
                                gradientWrtOrientedNormalFromMovedHit + gradientWrtOrientedNormalExplicit;
                        const float3 gradientWrtRawNormal = orientationSign * gradientWrtOrientedNormalX;
                        const float3 gradientProjectedToRawNormalTangent = gradientWrtRawNormal - rawNormal * dot(
                                                                               rawNormal, gradientWrtRawNormal);
                        const float3 gradientWrtCross = gradientProjectedToRawNormalTangent / rawCrossLength;

                        tanUContribution = cross(surfelX.tanV, gradientWrtCross) * invSpp;
                        tanVContribution = cross(gradientWrtCross, surfelX.tanU) * invSpp;
                    }

                    const float3 albedoContribution =
                            segmentTransmittance * geometricTermXY * albedoWeightWithoutTauAndGeometric * invSpp;

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
                    xRecord.gradScaleU = 0.0f;
                    xRecord.gradScaleV = 0.0f;
                    xRecord.gradEta = 0.0f;
                    xRecord.gradBeta = 0.0f;
                    xRecord.gradAlbedoR = albedoContribution.x();
                    xRecord.gradAlbedoG = albedoContribution.y();
                    xRecord.gradAlbedoB = albedoContribution.z();

                    gradientRecords[recordIndex] = xRecord;

                    float suffixTransmittance = 1.0f;

                    for (uint32_t reverseIndex = storedOccluderCount; reverseIndex > 0u; --reverseIndex) {
                        const uint32_t occluderIndex = reverseIndex - 1u;
                        const uint32_t occluderRecordIndex = eventRecordBase + 1u + occluderIndex;
                        const OccluderDerivative &occluderDerivative = occluderDerivatives[occluderIndex];
                        const float visibilityDerivativeScale =
                                -occluderDerivative.prefixTransmittance * suffixTransmittance * geometricTermXY *
                                scalarWeightWithoutTauAndGeometric * invSpp;
                        const float3 positionContribution = visibilityDerivativeScale * occluderDerivative.gradPosition;
                        const float3 tangentUContribution = visibilityDerivativeScale * occluderDerivative.gradTangentU;
                        const float3 tangentVContribution = visibilityDerivativeScale * occluderDerivative.gradTangentV;

                        SurfelGradientRecord occluderRecord{};
                        occluderRecord.primitiveIndex = occluderDerivative.primitiveIndex;
                        occluderRecord.gradPositionX = positionContribution.x();
                        occluderRecord.gradPositionY = positionContribution.y();
                        occluderRecord.gradPositionZ = positionContribution.z();
                        occluderRecord.gradScaleU = visibilityDerivativeScale * occluderDerivative.gradScaleU;
                        occluderRecord.gradScaleV = visibilityDerivativeScale * occluderDerivative.gradScaleV;
                        occluderRecord.gradEta = visibilityDerivativeScale * occluderDerivative.gradEta;
                        occluderRecord.gradBeta = visibilityDerivativeScale * occluderDerivative.gradBeta;
                        occluderRecord.gradTangentUX = tangentUContribution.x();
                        occluderRecord.gradTangentUY = tangentUContribution.y();
                        occluderRecord.gradTangentUZ = tangentUContribution.z();
                        occluderRecord.gradTangentVX = tangentVContribution.x();
                        occluderRecord.gradTangentVY = tangentVContribution.y();
                        occluderRecord.gradTangentVZ = tangentVContribution.z();
                        occluderRecord.gradAlbedoR = 0.0f;
                        occluderRecord.gradAlbedoG = 0.0f;
                        occluderRecord.gradAlbedoB = 0.0f;

                        gradientRecords[occluderRecordIndex] = occluderRecord;
                        suffixTransmittance *= occluderDerivative.oneMinusAlpha;

                        accumulateDebugGradientIfSelected(debugImages, settings.renderDebugGradientImages,
                                                          settings.surfelIndexForDebugImages,
                                                          eventRecord.xSurface.pathId, occluderRecord);
                    }

                    accumulateDebugGradientIfSelected(debugImages, settings.renderDebugGradientImages,
                                                      settings.surfelIndexForDebugImages, eventRecord.xSurface.pathId,
                                                      xRecord);
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
        auto &debugImages = pkg.debugImages;
        const auto &photonMap = pkg.intermediates.map;
        MaterialVertexGradientEvent *materialVertexEvents = pkg.intermediates.materialVertexEvents;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;
        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;
        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class materialVertexGradientEventTag>(
                sycl::range<1>(materialVertexEventCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex = static_cast<uint32_t>(globalId[0]);
                    const uint32_t recordIndex = baseOffset + eventIndex;
                    SurfelGradientRecord gradientRecord{};
                    gradientRecord.primitiveIndex = kInvalidIndex;
                    gradientRecords[recordIndex] = gradientRecord;
                    const MaterialVertexGradientEvent eventRecord = materialVertexEvents[eventIndex];
                    const uint32_t primitiveIndex = eventRecord.surface.primitiveIndex;
                    if (primitiveIndex == kInvalidIndex) {
                        return;
                    }
                    const Point &surfel = scene.points[primitiveIndex];
                    const ReconstructedSurfelState surfelState = reconstructSurfelState(surfel, eventRecord.surface);
                    const uint64_t directLightSeed = rng::makeSeed(settings.random.seed, eventRecord.pathId,
                                                                   0x46ac91fbu, rng::kStreamDirectLight, eventIndex);
                    rng::Xorshift128 directLightRng(directLightSeed);
                    // This is L_surfel, i.e. outgoing radiance before multiplying
                    // by local alpha = eta * alpha_geom.
                    const float3 outgoingRadianceWithoutLocalAlpha = evaluateOutgoingRadianceWithoutLocalAlpha(
                        surfel, eventRecord.surface, surfelState, photonMap, scene, settings.numAdjointShadowRays,
                        directLightRng);
                    const float u = eventRecord.surface.uv.x();
                    const float v = eventRecord.surface.uv.y();
                    const float radiusSquared = u * u + v * v;
                    const float oneMinusRadiusSquared = 1.0f - radiusSquared;
                    if (oneMinusRadiusSquared <= 1.0e-8f) {
                        return;
                    }
                    const float alphaGeom = eventRecord.surface.alphaGeom;
                    const float betaScale = 4.0f * sycl::exp(surfel.beta);
                    const float dAlphaGeomDBeta = betaScale * sycl::log(oneMinusRadiusSquared) * alphaGeom;
                    const float dAlphaEffectiveDBeta = surfel.opacity * dAlphaGeomDBeta;
                    const float scalarWeightNoAlpha = dot(eventRecord.adjointWeightAtVertex,
                                                          outgoingRadianceWithoutLocalAlpha);
                    const float opacityGradient = alphaGeom * scalarWeightNoAlpha * invSpp;
                    const float betaGradient =
                            dot(eventRecord.adjointWeightAtVertex,
                                outgoingRadianceWithoutLocalAlpha * dAlphaEffectiveDBeta) * invSpp;

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

                    accumulateDebugGradientIfSelected(
                        debugImages,
                        settings.renderDebugGradientImages,
                        settings.surfelIndexForDebugImages,
                        eventRecord.pathId,
                        gradientRecord);
                });
        }).wait();
    }

    // Differentiate endpoint surfel. For instance if ray is coming from X to Y, we differentiate Y as the inner integral
    // Adjoint weight should be prior to surface X (p0).
    // Differentiate transmittance between X Y too.
    static void materialEndEdgeGradientEvent(
        RenderPackage &pkg,
        uint32_t materialEndEdgeEventCount,
        uint32_t baseOffset) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto &debugImages = pkg.debugImages;

        const auto &photonMap = pkg.intermediates.map;
        MaterialEdgeGradientEvent *materialEndEdgeEvents = pkg.intermediates.materialEndEdgeEvents;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;
        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;
        const float qNullInv = 1.0f / settings.sampling.qNull;
        static constexpr uint32_t recordsPerMaterialEndEdgeEvent = 1u + kMaxSplatEventsPerRay;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class materialEndEdgeGradientEventTag>(
                sycl::range<1>(materialEndEdgeEventCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex =
                            static_cast<uint32_t>(globalId[0]);

                    const uint32_t eventRecordBase =
                            baseOffset +
                            recordsPerMaterialEndEdgeEvent * eventIndex;

                    for (uint32_t recordOffset = 0u;
                         recordOffset < recordsPerMaterialEndEdgeEvent;
                         ++recordOffset) {
                        SurfelGradientRecord invalidRecord{};
                        invalidRecord.primitiveIndex = kInvalidIndex;
                        gradientRecords[eventRecordBase + recordOffset] =
                                invalidRecord;
                    }

                    const MaterialEdgeGradientEvent eventRecord =
                            materialEndEdgeEvents[eventIndex];
                    if (eventRecord.isDirectLightSample) {
                        return;
                    }
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
                    const float3 startToEnd = endState.position - startState.position;
                    const float distanceSquared = dot(startToEnd, startToEnd);
                    if (distanceSquared <= 1.0e-12f) {
                        return;
                    }
                    rng::Xorshift128 directLightRng(
                        rng::makeSeed(
                            settings.random.seed,
                            eventRecord.pathId,
                            0x8d24f31bu,
                            rng::kStreamDirectLight,
                            eventIndex));

                    const float3 endRadiance =
                            evaluateOutgoingRadianceWithLocalAlphaNoEmitters(
                                endSurfel,
                                eventRecord.endSurface,
                                endState,
                                photonMap,
                                scene,
                                settings.numAdjointShadowRays,
                                directLightRng);

                    const float geometricTerm =
                            computeGeometricTermValue(
                                startState.position,
                                endState.position,
                                startState.orientedNormal,
                                endState.orientedNormal);

                    if (geometricTerm <= 1.0e-12f) {
                        return;
                    }

                    const MaterialEdgeVisibilityDerivativeResult visibilityResult =
                            traceMaterialEdgeVisibilityDerivatives(
                                scene,
                                startState.position,
                                endState.position,
                                startState.orientedNormal,
                                startPrimitiveIndex,
                                endPrimitiveIndex,
                                true,
                                qNullInv);

                    const float endU = eventRecord.endSurface.uv.x();
                    const float endV = eventRecord.endSurface.uv.y();
                    const float endScaleU = endSurfel.scale.x();
                    const float endScaleV = endSurfel.scale.y();
                    const float endSurfaceMeasureJacobian = endScaleU * endScaleV;
                    const float dEndSurfaceMeasureJacobianDScaleU = endScaleV;
                    const float dEndSurfaceMeasureJacobianDScaleV = endScaleU;
                    const float scalarEdgeWeightBase = dot(eventRecord.sampledEdgeThroughput, endRadiance);
                    const float scalarEdgeWeight = scalarEdgeWeightBase * visibilityResult.nullSamplingWeight;
                    const float scalarMaterialEdgeWeight = scalarEdgeWeight * endSurfaceMeasureJacobian;
                    const float3 dGeometricTermDEndPosition =
                            computeGeometricTermGradientWrtEndpoint(startState.position, endState.position,
                                                                    startState.orientedNormal, endState.orientedNormal);
                    const float3 gradientWrtEndPositionBeforeSpp =
                            scalarMaterialEdgeWeight *
                            (
                                visibilityResult.segmentTransmittance *
                                dGeometricTermDEndPosition +
                                geometricTerm *
                                visibilityResult.gradTauWrtEndPoint
                            );
                    float3 endTranslationGradient = gradientWrtEndPositionBeforeSpp * invSpp;
                    float endScaleUGradient =
                            dot(
                                gradientWrtEndPositionBeforeSpp,
                                endU * endSurfel.tanU) *
                            invSpp;
                    float endScaleVGradient =
                            dot(
                                gradientWrtEndPositionBeforeSpp,
                                endV * endSurfel.tanV) * invSpp;

                    endScaleUGradient +=
                            scalarEdgeWeight *
                            visibilityResult.segmentTransmittance *
                            geometricTerm *
                            dEndSurfaceMeasureJacobianDScaleU *
                            invSpp;
                    endScaleVGradient +=
                            scalarEdgeWeight *
                            visibilityResult.segmentTransmittance *
                            geometricTerm *
                            dEndSurfaceMeasureJacobianDScaleV *
                            invSpp;
                    float3 endTangentUGradient = gradientWrtEndPositionBeforeSpp * (endU * endScaleU * invSpp);

                    float3 endTangentVGradient = gradientWrtEndPositionBeforeSpp * (endV * endScaleV * invSpp);
                    const float3 directionStartToEnd = startToEnd / sycl::sqrt(distanceSquared);
                    const float cosineAtStart = dot(startState.orientedNormal, directionStartToEnd);
                    const float3 dGeometricTermDEndNormal = -directionStartToEnd * (cosineAtStart / distanceSquared);
                    const float3 gradientWrtEndOrientedNormal = scalarMaterialEdgeWeight * visibilityResult.
                                                                segmentTransmittance * dGeometricTermDEndNormal;

                    const float3 rawEndNormal =
                            cross(endSurfel.tanU, endSurfel.tanV);

                    const float rawEndNormalLength =
                            length(rawEndNormal);

                    if (rawEndNormalLength > 1.0e-8f) {
                        const float3 unitRawEndNormal =
                                rawEndNormal / rawEndNormalLength;

                        const float orientationSign =
                                dot(unitRawEndNormal, endState.orientedNormal) >= 0.0f
                                    ? 1.0f
                                    : -1.0f;

                        const float3 gradientWrtRawEndNormal =
                                orientationSign *
                                gradientWrtEndOrientedNormal;

                        const float3 projectedGradientWrtRawEndNormal =
                                gradientWrtRawEndNormal -
                                unitRawEndNormal *
                                dot(unitRawEndNormal, gradientWrtRawEndNormal);

                        const float3 gradientWrtEndCross =
                                projectedGradientWrtRawEndNormal /
                                rawEndNormalLength;

                        endTangentUGradient +=
                                cross(endSurfel.tanV, gradientWrtEndCross) *
                                invSpp;

                        endTangentVGradient +=
                                cross(gradientWrtEndCross, endSurfel.tanU) *
                                invSpp;
                    }

                    SurfelGradientRecord endRecord{};
                    endRecord.primitiveIndex =
                            endPrimitiveIndex;

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

                    // End-edge events treat L_o(end) as fixed.
                    // End albedo/alpha gradients come through deeper/start/vertex events.
                    endRecord.gradAlbedoR = 0.0f;
                    endRecord.gradAlbedoG = 0.0f;
                    endRecord.gradAlbedoB = 0.0f;

                    gradientRecords[eventRecordBase] =
                            endRecord;

                    if (eventRecord.writeOcclusionGradients) {
                        writeMaterialEdgeOccluderGradientRecords(
                            gradientRecords,
                            eventRecordBase + 1u,
                            visibilityResult,
                            geometricTerm,
                            scalarMaterialEdgeWeight,
                            invSpp,
                            debugImages,
                            settings.renderDebugGradientImages,
                            settings.surfelIndexForDebugImages,
                            eventRecord.pathId);
                    }


                    accumulateDebugGradientIfSelected(
                        debugImages,
                        settings.renderDebugGradientImages,
                        settings.surfelIndexForDebugImages,
                        eventRecord.pathId,
                        endRecord);
                });
        }).wait();
    }

    static void materialStartEdgeGradientEvent(RenderPackage &pkg, uint32_t materialStartEdgeEventCount,
                                               uint32_t baseOffset) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto &debugImages = pkg.debugImages;

        const auto &photonMap = pkg.intermediates.map;
        MaterialEdgeGradientEvent *materialStartEdgeEvents = pkg.intermediates.materialStartEdgeEvents;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;

        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;
        const float qNullInv = 1.0f / settings.sampling.qNull;
        static constexpr uint32_t recordsPerMaterialStartEdgeEvent = 1u + kMaxSplatEventsPerRay;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class materialStartEdgeGradientEventTag>(
                sycl::range<1>(materialStartEdgeEventCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex = static_cast<uint32_t>(globalId[0]);
                    const uint32_t eventRecordBase = baseOffset + recordsPerMaterialStartEdgeEvent * eventIndex;
                    for (uint32_t recordOffset = 0u; recordOffset < recordsPerMaterialStartEdgeEvent; ++recordOffset) {
                        SurfelGradientRecord invalidRecord{};
                        invalidRecord.primitiveIndex = kInvalidIndex;
                        gradientRecords[eventRecordBase + recordOffset] = invalidRecord;
                    }
                    const MaterialEdgeGradientEvent eventRecord = materialStartEdgeEvents[eventIndex];
                    const uint32_t startPrimitiveIndex = eventRecord.startSurface.primitiveIndex;
                    const uint32_t endPrimitiveIndex = eventRecord.endSurface.primitiveIndex;
                    if (startPrimitiveIndex == kInvalidIndex || endPrimitiveIndex == kInvalidIndex) return;
                    const Point &startSurfel = scene.points[startPrimitiveIndex];
                    const Point &endSurfel = scene.points[endPrimitiveIndex];
                    const ReconstructedSurfelState startState = reconstructSurfelState(
                        startSurfel, eventRecord.startSurface);
                    const ReconstructedSurfelState endState = reconstructSurfelState(endSurfel, eventRecord.endSurface);
                    const float3 startToEnd = endState.position - startState.position;
                    const float distanceSquared = dot(startToEnd, startToEnd);
                    if (distanceSquared <= 1.0e-12f) return;
                    rng::Xorshift128 directLightRng(rng::makeSeed(settings.random.seed, eventRecord.pathId, 0xa15c3e41u,
                                                                  rng::kStreamDirectLight, eventIndex));
                    const float3 endRadiance = eventRecord.isDirectLightSample
                                                   ? eventRecord.directLightRadiance
                                                   : evaluateOutgoingRadianceWithLocalAlphaNoEmitters(
                                                       endSurfel, eventRecord.endSurface, endState, photonMap, scene,
                                                       settings.numAdjointShadowRays, directLightRng);
                    const float geometricTerm = computeGeometricTermValue(
                        startState.position, endState.position, startState.orientedNormal, endState.orientedNormal);
                    if (geometricTerm <= 1.0e-12f) return;
                    const bool applyNullSamplingWeight = !eventRecord.isDirectLightSample;
                    const MaterialEdgeVisibilityDerivativeResult visibilityResult =
                            traceMaterialEdgeVisibilityDerivatives(
                                scene,
                                startState.position,
                                endState.position,
                                startState.orientedNormal,
                                startPrimitiveIndex,
                                endPrimitiveIndex,
                                applyNullSamplingWeight,
                                qNullInv);
                    const float endScaleU = endSurfel.scale.x();
                    const float endScaleV = endSurfel.scale.y();

                    const float3 edgePrefix =
                            eventRecord.betaIncrement *
                            eventRecord.alpha *
                            eventRecord.bsdf *
                            eventRecord.invSamplePDF;
                    const float scalarEdgeWeightBase =
                            dot(edgePrefix, endRadiance);

                    const float scalarEdgeWeight = scalarEdgeWeightBase * visibilityResult.nullSamplingWeight;

                    const float3 dGeometricTermDStartPosition = computeGeometricTermGradientWrtStartpoint(
                        startState.position,
                        endState.position,
                        startState.orientedNormal,
                        endState.orientedNormal);

                    const float3 gradientWrtStartPositionBeforeSpp = scalarEdgeWeight * (
                                                                         visibilityResult.segmentTransmittance *
                                                                         dGeometricTermDStartPosition +
                                                                         geometricTerm * visibilityResult.
                                                                         gradTauWrtStartPoint);

                    const float startU = eventRecord.startSurface.uv.x();
                    const float startV = eventRecord.startSurface.uv.y();
                    const float startScaleU = startSurfel.scale.x();
                    const float startScaleV = startSurfel.scale.y();
                    float3 startTranslationGradient = gradientWrtStartPositionBeforeSpp * invSpp;
                    float startScaleUGradient = dot(gradientWrtStartPositionBeforeSpp, startU * startSurfel.tanU) *
                                                invSpp;
                    float startScaleVGradient = dot(gradientWrtStartPositionBeforeSpp, startV * startSurfel.tanV) *
                                                invSpp;

                    const float u = eventRecord.startSurface.uv.x();
                    const float v = eventRecord.startSurface.uv.y();
                    const float r2 = u * u + v * v;
                    const float oneMinusR2 = 1.0f - r2;

                    if (oneMinusR2 > 1.0e-8f) {
                        const float alphaGeom = eventRecord.startSurface.alphaGeom;
                        const float eta = startSurfel.opacity;
                        const float betaScale = 4.0f * sycl::exp(startSurfel.beta);

                        // Weight with alpha removed
                        const float3 prefixNoAlpha =
                                eventRecord.betaIncrement *
                                eventRecord.bsdf *
                                eventRecord.invSamplePDF;

                        const float scalarLocalAlphaWeight =
                                dot(prefixNoAlpha, endRadiance) *
                                visibilityResult.nullSamplingWeight *
                                visibilityResult.segmentTransmittance *
                                geometricTerm *
                                invSpp;

                        const float dAlphaEffectiveDScaleU =
                                eta * (2.0f * betaScale * u * u * alphaGeom) /
                                (startScaleU * oneMinusR2);

                        const float dAlphaEffectiveDScaleV =
                                eta * (2.0f * betaScale * v * v * alphaGeom) /
                                (startScaleV * oneMinusR2);

                        startScaleUGradient += scalarLocalAlphaWeight * dAlphaEffectiveDScaleU;
                        startScaleVGradient += scalarLocalAlphaWeight * dAlphaEffectiveDScaleV;
                    }

                    float3 startTangentUGradient = gradientWrtStartPositionBeforeSpp * (startU * startScaleU * invSpp);
                    float3 startTangentVGradient = gradientWrtStartPositionBeforeSpp * (startV * startScaleV * invSpp);
                    const float3 directionStartToEnd = startToEnd / sycl::sqrt(distanceSquared);
                    const float cosineAtEnd = dot(endState.orientedNormal, -directionStartToEnd);
                    const float3 dGeometricTermDStartNormal = directionStartToEnd * (cosineAtEnd / distanceSquared);
                    const float3 gradientWrtStartOrientedNormal = scalarEdgeWeight * visibilityResult.
                                                                  segmentTransmittance * dGeometricTermDStartNormal;
                    const float3 rawStartNormal = cross(startSurfel.tanU, startSurfel.tanV);
                    const float rawStartNormalLength = length(rawStartNormal);
                    if (rawStartNormalLength > 1.0e-8f) {
                        const float3 unitRawStartNormal = rawStartNormal / rawStartNormalLength;
                        const float orientationSign =
                                dot(unitRawStartNormal, startState.orientedNormal) >= 0.0f ? 1.0f : -1.0f;
                        const float3 gradientWrtRawStartNormal = orientationSign * gradientWrtStartOrientedNormal;
                        const float3 projectedGradientWrtRawStartNormal = gradientWrtRawStartNormal - unitRawStartNormal
                                                                          * dot(unitRawStartNormal,
                                                                              gradientWrtRawStartNormal);
                        const float3 gradientWrtStartCross = projectedGradientWrtRawStartNormal / rawStartNormalLength;
                        startTangentUGradient += cross(startSurfel.tanV, gradientWrtStartCross) * invSpp;
                        startTangentVGradient += cross(gradientWrtStartCross, startSurfel.tanU) * invSpp;
                    }

                    const float transportScale =
                            visibilityResult.nullSamplingWeight *
                            visibilityResult.segmentTransmittance *
                            geometricTerm *
                            invSpp;

                    float3 albedoGradientThroughput = eventRecord.betaIncrement * (startSurfel.alpha_r * M_1_PIf) *
                                                      eventRecord.alpha * eventRecord.invSamplePDF;

                    const float3 startAlbedoGradient{
                        albedoGradientThroughput.x() * endRadiance.x() * transportScale,
                        albedoGradientThroughput.y() * endRadiance.y() * transportScale,
                        albedoGradientThroughput.z() * endRadiance.z() * transportScale
                    };
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

                    startRecord.gradAlbedoR = startAlbedoGradient.x();
                    startRecord.gradAlbedoG = startAlbedoGradient.y();
                    startRecord.gradAlbedoB = startAlbedoGradient.z();

                    gradientRecords[eventRecordBase] = startRecord;


                    if (eventRecord.writeOcclusionGradients) {
                        writeMaterialEdgeOccluderGradientRecords(
                            gradientRecords,
                            eventRecordBase + 1u,
                            visibilityResult,
                            geometricTerm,
                            scalarEdgeWeight,
                            invSpp,
                            debugImages,
                            settings.renderDebugGradientImages,
                            settings.surfelIndexForDebugImages,
                            eventRecord.pathId);
                    }


                    accumulateDebugGradientIfSelected(
                        debugImages,
                        settings.renderDebugGradientImages,
                        settings.surfelIndexForDebugImages,
                        eventRecord.pathId,
                        startRecord);
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
                    atomicAddFloat(gradients.gradPosition[primitiveIndex].x(), gradientRecord.gradPositionX);
                    atomicAddFloat(gradients.gradPosition[primitiveIndex].y(), gradientRecord.gradPositionY);
                    atomicAddFloat(gradients.gradPosition[primitiveIndex].z(), gradientRecord.gradPositionZ);
                    atomicAddFloat(gradients.gradScale[primitiveIndex].x(), gradientRecord.gradScaleU);
                    atomicAddFloat(gradients.gradScale[primitiveIndex].y(), gradientRecord.gradScaleV);
                    atomicAddFloat(gradients.gradTanU[primitiveIndex].x(), gradientRecord.gradTangentUX);
                    atomicAddFloat(gradients.gradTanU[primitiveIndex].y(), gradientRecord.gradTangentUY);
                    atomicAddFloat(gradients.gradTanU[primitiveIndex].z(), gradientRecord.gradTangentUZ);
                    atomicAddFloat(gradients.gradTanV[primitiveIndex].x(), gradientRecord.gradTangentVX);
                    atomicAddFloat(gradients.gradTanV[primitiveIndex].y(), gradientRecord.gradTangentVY);
                    atomicAddFloat(gradients.gradTanV[primitiveIndex].z(), gradientRecord.gradTangentVZ);
                    atomicAddFloat(gradients.gradOpacity[primitiveIndex], gradientRecord.gradEta);
                    atomicAddFloat(gradients.gradBeta[primitiveIndex], gradientRecord.gradBeta);
                    atomicAddFloat(gradients.gradAlbedo[primitiveIndex].x(), gradientRecord.gradAlbedoR);
                    atomicAddFloat(gradients.gradAlbedo[primitiveIndex].y(), gradientRecord.gradAlbedoG);
                    atomicAddFloat(gradients.gradAlbedo[primitiveIndex].z(), gradientRecord.gradAlbedoB);
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

        SensorGPU sensor = pkg.sensors[cameraIndex];
        auto &grads = pkg.gradients;

        const std::uint32_t imageWidth = sensor.camera.width;
        const std::uint32_t imageHeight = sensor.camera.height;
        const std::uint32_t pixelCount = imageWidth * imageHeight;

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class DepthDistortionBackwardKernel>(
                sycl::range<1>(pixelCount),
                [=](sycl::id<1> tid) {
                    constexpr uint32_t kMaxHits = 32u;
                    constexpr float kDenomEps = 1.0e-8f;

                    const uint32_t pixelIndex = tid[0];
                    const uint32_t pixelX = pixelIndex % imageWidth;
                    const uint32_t pixelY = pixelIndex / imageWidth;

                    const float pixelAdjoint = sensor.depthDistortionAdjointBuffer[pixelIndex];

                    if (pixelAdjoint == 0.0f) {
                        return;
                    }

                    Ray primaryRay = makePrimaryRayFromPixelJitteredFov(
                        sensor.camera,
                        static_cast<float>(pixelX),
                        static_cast<float>(pixelY),
                        0.0f,
                        0.0f);

                    const float3 rayOrigin0 = primaryRay.origin;
                    const float3 rayDir0 = primaryRay.direction;

                    DistortionHit hits[kMaxHits];
                    uint32_t hitCount = 0u;
                    float transmittance = 1.0f;

                    for (uint32_t traversalIndex = 0u; traversalIndex < kMaxHits; ++traversalIndex) {
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

                            const float alphaGeom = worldHit.alphaGeom;
                            const float ai = surfel.opacity * alphaGeom;
                            const float wi = transmittance * ai;
                            const float zi = dot(worldHit.hitPositionW - sensor.camera.pos, sensor.camera.forward);

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
                            primaryRay.origin = worldHit.hitPositionW + primaryRay.direction * 1.0e-8f;

                            continue;
                        }

                        if (instance.geometryType == GeometryType::Mesh) {
                            break;
                        }
                    }

                    if (hitCount <= 1u) {
                        return;
                    }

                    float barW[kMaxHits];
                    float barM[kMaxHits];
                    float barZ[kMaxHits];
                    float barA[kMaxHits];

                    for (uint32_t i = 0u; i < hitCount; ++i) {
                        barW[i] = 0.0f;
                        barM[i] = 0.0f;
                        barZ[i] = 0.0f;
                        barA[i] = 0.0f;
                    }

                    // L = sum_i sum_{j<i} w_i w_j (m_i - m_j)^2
                    for (uint32_t i = 0u; i < hitCount; ++i) {
                        for (uint32_t j = i + 1u; j < hitCount; ++j) {
                            const float mi = depthDistortionNdc01(hits[i].zi);
                            const float mj = depthDistortionNdc01(hits[j].zi);

                            const float wi = hits[i].wi;
                            const float wj = hits[j].wi;

                            const float depthDifference = mi - mj;
                            const float depthDifferenceSquared = depthDifference * depthDifference;

                            barW[i] += pixelAdjoint * wj * depthDifferenceSquared;
                            barW[j] += pixelAdjoint * wi * depthDifferenceSquared;

                            const float depthAdjointScale = 2.0f * pixelAdjoint * wi * wj;
                            barM[i] += depthAdjointScale * depthDifference;
                            barM[j] -= depthAdjointScale * depthDifference;
                        }
                    }

                    for (uint32_t i = 0u; i < hitCount; ++i) {
                        const float dMiDZi = depthDistortionDndc01Ddepth(hits[i].zi);
                        barZ[i] += barM[i] * dMiDZi;
                    }

                    // Reverse through front-to-back alpha compositing:
                    // wi = Tprev_i * ai
                    // Tnext_i = Tprev_i * (1 - ai)
                    float barTnext = 0.0f;

                    for (int i = int(hitCount) - 1; i >= 0; --i) {
                        const float ai = hits[i].ai;
                        const float Tprev = hits[i].Tprev;

                        float barAi = 0.0f;
                        float barTprev = 0.0f;

                        barAi += Tprev * barW[i];
                        barTprev += ai * barW[i];

                        barAi += -Tprev * barTnext;
                        barTprev += (1.0f - ai) * barTnext;

                        barA[i] = barAi;
                        barTnext = barTprev;
                    }

                    for (uint32_t i = 0u; i < hitCount; ++i) {
                        const DistortionHit &hit = hits[i];
                        const Point &surfel = scene.points[hit.primitiveIndex];

                        const float3 p = surfel.position;
                        const float3 tu = surfel.tanU;
                        const float3 tv = surfel.tanV;
                        const float su = surfel.scale.x();
                        const float sv = surfel.scale.y();
                        const float eta = surfel.opacity;

                        if (sycl::fabs(su) <= kDenomEps || sycl::fabs(sv) <= kDenomEps) {
                            continue;
                        }

                        const float3 x = hit.hitPositionW;
                        const float3 q = x - p;

                        const AlphaKernelEval kernelEval =
                                evaluateAlphaKernelAndDerivatives(surfel, hit.u, hit.v);

                        const float barAlphaGeom = barA[i] * eta;
                        const float barEta = barA[i] * hit.alphaGeom;

                        const float barU = barAlphaGeom * kernelEval.dValue_dU;
                        const float barV = barAlphaGeom * kernelEval.dValue_dV;
                        const float barBeta = barAlphaGeom * kernelEval.dValue_dBeta;

                        float3 barX = barZ[i] * sensor.camera.forward;

                        float3 barQ{0.0f, 0.0f, 0.0f};
                        barQ += (barU / su) * tu;
                        barQ += (barV / sv) * tv;

                        float3 barTu = (barU / su) * q;
                        float3 barTv = (barV / sv) * q;

                        const float barSu = -barU * hit.u / su;
                        const float barSv = -barV * hit.v / sv;

                        barX += barQ;
                        float3 barP = -barQ;

                        // x = rayOrigin0 + lambda * rayDir0
                        // lambda = dot(n, p - rayOrigin0) / dot(n, rayDir0)
                        const float3 nRaw = cross(tu, tv);
                        const float nRawLen = sycl::sqrt(dot(nRaw, nRaw));

                        if (nRawLen > kDenomEps) {
                            const float3 n = nRaw / nRawLen;
                            const float denom = dot(n, rayDir0);

                            if (sycl::fabs(denom) > kDenomEps) {
                                const float barLambda = dot(barX, rayDir0);

                                barP += (barLambda / denom) * n;

                                const float3 barN = (barLambda / denom) * (p - x);
                                const float3 barNRaw = (barN - n * dot(n, barN)) / nRawLen;

                                barTu += cross(tv, barNRaw);
                                barTv += cross(barNRaw, tu);
                            }
                        }

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
        uint32_t materialEndEdgeEventCount,
        uint32_t materialStartEdgeEventCount,
        uint32_t cameraIndex) {
        const GradientRecordRanges ranges = makeGradientRecordRanges(
            measurementEventCount,
            measurementTwoPointEventCount,
            materialVertexEventCount,
            materialEndEdgeEventCount,
            materialStartEdgeEventCount);

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

        if (materialEndEdgeEventCount >= pkg.intermediates.maxMaterialEndEdgeEventCount) {
            Log::PA_ERROR(
                "Overflow: materialEndEdgeEventCount={} >= maxMaterialEndEdgeEventCount={}",
                materialEndEdgeEventCount,
                pkg.intermediates.maxMaterialEndEdgeEventCount);
        }

        if (materialStartEdgeEventCount >= pkg.intermediates.maxMaterialStartEdgeEventCount) {
            Log::PA_ERROR(
                "Overflow: materialStartEdgeEventCount={} >= maxMaterialStartEdgeEventCount={}",
                materialStartEdgeEventCount,
                pkg.intermediates.maxMaterialStartEdgeEventCount);
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
            "Event counts: measurement={}, measurementTwoPoint={}, materialVertex={}, materialEndEdge={}, materialStartEdge={}, gradientRecords={}",
            measurementEventCount,
            measurementTwoPointEventCount,
            materialVertexEventCount,
            materialEndEdgeEventCount,
            materialStartEdgeEventCount,
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

        if (materialEndEdgeEventCount > 0u) {
            ScopedTimer timer("materialEndEdgeGradientEvent", spdlog::level::debug);
            materialEndEdgeGradientEvent(
                pkg,
                materialEndEdgeEventCount,
                ranges.materialEndEdgeOffset);
        }

        if (materialStartEdgeEventCount > 0u) {
            ScopedTimer timer("materialStartEdgeGradientEvent", spdlog::level::debug);
            materialStartEdgeGradientEvent(
                pkg,
                materialStartEdgeEventCount,
                ranges.materialStartEdgeOffset);
        }

        if (ranges.totalCount > 0u) {
            ScopedTimer timer("reduceSurfelGradientRecords", spdlog::level::debug);
            reduceSurfelGradientRecords(
                pkg,
                ranges.totalCount);
        }
    }
}
