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
        const PathTracerSettings &settings,
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

        /*
        const float3 directRadiance =
                estimateDirectAreaLightAtDiffuseSurface(
                    scene,
                    reconstructedState.position,
                    reconstructedState.orientedNormal,
                    surfel.alpha_r * surfel.albedo, settings, rng128) * alpha;
        */

        const float3 directRadiance = estimateDirectPointSampledPointLights(
                                          scene,
                                          settings,
                                          reconstructedState.position,
                                          reconstructedState.orientedNormal,
                                          surfel.alpha_r * surfel.albedo) * alpha;

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
        const PathTracerSettings &settings,
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

        /*
        const float3 directRadiance =
            estimateDirectAreaLightAtDiffuseSurface(
                scene,
                reconstructedState.position,
                reconstructedState.orientedNormal,
                surfel.alpha_r * surfel.albedo, settings, rng128) * alpha;
        */
        const float3 directRadiance = estimateDirectPointSampledPointLights(
                                          scene,
                                          settings,
                                          reconstructedState.position,
                                          reconstructedState.orientedNormal,
                                          surfel.alpha_r * surfel.albedo) * alpha;

        return directRadiance + indirectRadiance;
    }

    SYCL_EXTERNAL inline float3 evaluateOutgoingRadianceWithoutLocalAlpha(
        const Point &surfel,
        const PointCloudSurfaceRecord &surfaceRecord,
        const ReconstructedSurfelState &reconstructedState,
        const DeviceSurfacePhotonMapGrid &photonMap,
        const GPUSceneBuffers &scene,
        const PathTracerSettings &settings,
        rng::Xorshift128 &rng128) {
        const float3 indirectIrradiance = gatherDiffuseIrradianceAtPoint(
            reconstructedState.position,
            reconstructedState.orientedNormal,
            photonMap);

        const float3 indirectRadiance =
                indirectIrradiance *
                (surfel.alpha_r * surfel.albedo * M_1_PIf);

        /*
        const float3 directRadiance =
            estimateDirectAreaLightAtDiffuseSurface(
                scene,
                reconstructedState.position,
                reconstructedState.orientedNormal,
                surfel.alpha_r * surfel.albedo, settings, rng128);
        */
        const float3 directRadiance = estimateDirectPointSampledPointLights(
            scene,
            settings,
            reconstructedState.position,
            reconstructedState.orientedNormal,
            surfel.alpha_r * surfel.albedo);

        return directRadiance + indirectRadiance;
    }

    // Add these helpers in the same file above launchAdjointIntersectKernel.
    struct AdjointAuxiliaryEndpoint {
        bool found = false;
        PointCloudSurfaceRecord surface{};
        float discreteSelectionPdf = 1.0f;
    };

    SYCL_EXTERNAL inline bool traceAdjointShadowTransmission(
        const GPUSceneBuffers &scene,
        Ray shadowRay,
        const float3 &segmentOrigin,
        float targetDistance,
        uint32_t skipPrimitiveA,
        uint32_t skipPrimitiveB,
        float &transmissionOut) {
        transmissionOut = 1.0f;
        for (uint32_t traversalIndex = 0u; traversalIndex < kMaxSplatEventsPerRay; ++traversalIndex) {
            WorldHit shadowHit{};
            intersectScene(shadowRay, &shadowHit, scene, SurfelIntersectMode::FirstHit);
            if (!shadowHit.hit) {
                break;
            }
            const float3 hitVector = shadowHit.hitPositionW - segmentOrigin;
            const float hitDistance = sycl::sqrt(dot(hitVector, hitVector));
            if (hitDistance >= targetDistance - RayEpsilon) {
                break;
            }
            buildIntersectionNormal(scene, shadowHit);
            const InstanceRecord &hitInstance = scene.instances[shadowHit.instanceIndex];
            if (hitInstance.geometryType == GeometryType::Mesh) {
                return false;
            }
            if (shadowHit.primitiveIndex == skipPrimitiveA || shadowHit.primitiveIndex == skipPrimitiveB) {
                shadowRay.origin = shadowHit.hitPositionW + shadowRay.direction * RayEpsilon;
                continue;
            }
            if (hitInstance.geometryType == GeometryType::PointCloud) {
                const Point &shadowSurfel = scene.points[shadowHit.primitiveIndex];
                transmissionOut *= 1.0f - shadowHit.alphaGeom * shadowSurfel.opacity;
                shadowRay.origin = shadowHit.hitPositionW + shadowRay.direction * RayEpsilon;
                continue;
            }
            return false;
        }
        return true;
    }

    SYCL_EXTERNAL inline AdjointAuxiliaryEndpoint sampleAdjointAuxiliaryPointEndpoint(
        const GPUSceneBuffers &scene,
        Ray auxiliaryRay,
        rng::Xorshift128 &rng128,
        float qNull,
        float qReflect,
        uint32_t startPrimitiveIndex,
        uint32_t pathId,
        uint32_t pixelIndex,
        bool includeSelectionPdf) {
        AdjointAuxiliaryEndpoint endpoint{};
        for (uint32_t traversalIndex = 0u; traversalIndex < kMaxSplatEventsPerRay; ++traversalIndex) {
            WorldHit auxiliaryHit{};
            intersectScene(auxiliaryRay, &auxiliaryHit, scene, SurfelIntersectMode::FirstHit);
            if (!auxiliaryHit.hit) {
                break;
            }
            buildIntersectionNormal(scene, auxiliaryHit);
            const InstanceRecord &auxiliaryInstance = scene.instances[auxiliaryHit.instanceIndex];
            if (auxiliaryInstance.geometryType == GeometryType::Mesh || auxiliaryInstance.geometryType !=
                GeometryType::PointCloud) {
                break;
            }
            if (auxiliaryHit.primitiveIndex == startPrimitiveIndex) {
                auxiliaryRay.origin = auxiliaryHit.hitPositionW + auxiliaryRay.direction * RayEpsilon;
                continue;
            }
            const Point &auxiliarySurfel = scene.points[auxiliaryHit.primitiveIndex];
            if (rng128.nextFloat() < qNull) {
                if (includeSelectionPdf) {
                    endpoint.discreteSelectionPdf *= qNull;
                }
                auxiliaryRay.origin = auxiliaryHit.hitPositionW + auxiliaryRay.direction * RayEpsilon;
                auxiliaryRay.normal = computePointCloudOrientedNormal(auxiliarySurfel, auxiliaryRay.direction);
                continue;
            }
            if (includeSelectionPdf) {
                endpoint.discreteSelectionPdf *= qReflect;
            }
            RayState auxiliaryRayState{};
            auxiliaryRayState.ray = auxiliaryRay;
            auxiliaryRayState.pathId = pathId;
            auxiliaryRayState.pixelIndex = pixelIndex;
            //endpoint.surface = makePointCloudSurfaceRecord(auxiliaryHit, auxiliaryRayState, scene);
            endpoint.found = true;
            break;
        }
        return endpoint;
    }

    static inline void clearAdjointLocalPendingState(
        PendingCameraSegment &pendingCameraSegment,
        PendingAdjointStageX &pendingAdjointStage) {
        clearPendingCameraSegment(pendingCameraSegment);
        clearPendingAdjointStageX(pendingAdjointStage);
    }

    template<typename IntermediatesType>
    static inline void storeAdjointPendingState(
        const IntermediatesType &intermediates,
        uint32_t pathId,
        bool hasPendingState,
        const PendingCameraSegment &pendingCameraSegment,
        const PendingAdjointStageX &pendingAdjointStage) {
        if (!hasPendingState) {
            return;
        }
        if (pendingCameraSegment.valid) {
            intermediates.pendingCameraSegments[pathId] = pendingCameraSegment;
        } else {
            clearPendingCameraSegment(intermediates.pendingCameraSegments[pathId]);
        }
        if (pendingAdjointStage.valid) {
            intermediates.pendingStageX[pathId] = pendingAdjointStage;
        } else {
            clearPendingAdjointStageX(intermediates.pendingStageX[pathId]);
        }
    }

    template<typename IntermediatesType>
    static inline void enqueueAdjointNextRayState(
        const IntermediatesType &intermediates,
        bool shouldEnqueueNextRayState,
        const RayState &nextRayState) {
        if (!shouldEnqueueNextRayState) {
            return;
        }
        auto extensionCounter = sycl::atomic_ref<
            uint32_t,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>(*intermediates.countExtensionOut);
        const uint32_t outIndex = extensionCounter.fetch_add(1u);
        if (outIndex < intermediates.maxRayQueueCapacity) {
            intermediates.extensionRaysA[outIndex] = nextRayState;
        }
    }

    template<typename SettingsType, typename SceneType, typename IntermediatesType>
    static inline void appendAdjointMeasurementDirectPointLightSlabEvents(
        const SettingsType &settings,
        const SceneType &scene,
        const IntermediatesType &intermediates,
        const MeasurementGradientEvent &slabEvent,
        const RayState &currentRayState,
        float qReflect) {
        if (!settings.enableAdjointDirectLight ||
            slabEvent.surfelSlabCount == 0u) {
            return;
        }

        for (uint32_t lightIndex = 0u;
             lightIndex < scene.lightCount;
             ++lightIndex) {
            const GPULightRecord &light =
                    scene.lights[lightIndex];

            if (light.lightType != LightType::Surfel ||
                light.primitiveIndex == kInvalidIndex) {
                continue;
            }
            const Point &lightCarrier = scene.points[light.primitiveIndex];
            const float3 lightPositionW = lightCarrier.position;

            // -------------------------------------------------------------
            // An XY event is only needed if at least one constituent of the
            // target slab has an opacity-bearing surfel on its shadow segment.
            // -------------------------------------------------------------
            MeasurementGradientEventXY event{};
            event.surfelSlabCount =
                    slabEvent.surfelSlabCount;

            for (uint32_t i = 0u;
                 i < slabEvent.surfelSlabCount;
                 ++i) {
                event.xSurface[i] =
                        slabEvent.xSurface[i];

                event.layerWeights[i] =
                        slabEvent.layerWeights[i];

                event.directLightEps[i] =
                        slabEvent.directLightEps[i];
            }

            event.xPathThroughput =
                    currentRayState.transmission *
                    currentRayState.pathThroughput / qReflect;

            event.pointLightPositionW =
                    lightPositionW;

            event.pointLightRadiantIntensity =
                    light.flux *
                    light.color *
                    (1.0f / (4.0f * M_PIf));

            event.pointLightPrimitiveIndex =
                    light.primitiveIndex;

            appendEventAtomic(
                intermediates.countMeasurementTwoPointEvents,
                intermediates.measurementTwoPointEvents,
                intermediates.maxMeasurementTwoPointEventCount,
                event);
        }
    }

    template<typename IntermediatesType>
    static inline void appendAdjointMaterialVertexEvent(
        const IntermediatesType &intermediates,
        const PointCloudSurfaceRecord &currentSurface,
        const RayState &currentRayState,
        float qReflect) {
        MaterialVertexGradientEvent materialVertexEvent{};
        materialVertexEvent.surface = currentSurface;
        materialVertexEvent.adjointWeightAtVertex =
                currentRayState.pathThroughput * currentRayState.transmission / qReflect;
        materialVertexEvent.pathId = currentRayState.pathId;
        materialVertexEvent.bounceIndex = currentRayState.bounceIndex;
        appendEventAtomic(
            intermediates.countMaterialVertexEvents,
            intermediates.materialVertexEvents,
            intermediates.maxMaterialVertexEventCount,
            materialVertexEvent);
    }

    template<typename IntermediatesType>
    static inline void appendAdjointMaterialEdgeXYEvent(
        const IntermediatesType &intermediates,
        const PendingAdjointStageX &previousAdjointStage,
        const PointCloudSurfaceRecord &currentSurface,
        const RayState &currentRayState,
        float segmentAreaPdfFromStoredVertex,
        float segmentUvJacobianAtEnd,
        float qReflect) {
        MaterialEdgeGradientEvent materialEdgeEventXY{};
        materialEdgeEventXY.startSurface = previousAdjointStage.current.surface;
        materialEdgeEventXY.endSurface = currentSurface;
        const float invSegmentUvPdf =
                1.0f / (segmentAreaPdfFromStoredVertex * segmentUvJacobianAtEnd);
        materialEdgeEventXY.sampledEdgeThroughput =
                previousAdjointStage.current.pathThroughput *
                previousAdjointStage.current.transmission *
                previousAdjointStage.current.bsdf *
                previousAdjointStage.current.alpha *
                invSegmentUvPdf / qReflect;
        materialEdgeEventXY.isDirectLightSample = false;
        materialEdgeEventXY.writeOcclusionGradients = currentRayState.bounceIndex > 1;
        materialEdgeEventXY.pathId = currentRayState.pathId;
        materialEdgeEventXY.startBounceIndex = previousAdjointStage.current.bounceIndex;
        appendEventAtomic(
            intermediates.countMaterialEndEdgeEvents,
            intermediates.materialEndEdgeEvents,
            intermediates.maxMaterialEndEdgeEventCount,
            materialEdgeEventXY);
    }

    template<typename SettingsType, typename SceneType, typename IntermediatesType>
    static inline void appendAdjointMaterialDirectLightEdgeSamples(
        const SettingsType &settings,
        const SceneType &scene,
        const IntermediatesType &intermediates,
        const Point &surfel,
        const PointCloudSurfaceRecord &currentSurface,
        const WorldHit &worldHit,
        const RayState &currentRayState,
        const float3 &surfelBsdf,
        uint64_t renderSeed,
        uint32_t spp,
        float invQReflect) {
        const ReconstructedSurfelState startState = reconstructSurfelState(
            surfel, currentSurface);
        const uint32_t samplesPerLight = settings.numAdjointPathShadowRays;
        if (samplesPerLight == 0u) {
            return;
        }
        const float invSamplesPerLight = 1.0f / static_cast<float>(samplesPerLight);
        for (uint32_t lightIndex = 0u; lightIndex < scene.lightCount; ++lightIndex) {
            const GPULightRecord light = scene.lights[lightIndex];
            if (light.lightType != LightType::Surfel) {
                continue;
            }
            for (uint32_t shadowRaySample = 0u; shadowRaySample < samplesPerLight; ++
                 shadowRaySample) {
                const uint64_t lightSampleSeed = rng::makeSeed(
                    renderSeed, currentRayState.pathId, spp, rng::kStreamDirectLight,
                    currentRayState.traversalIndex * 1315423911u + lightIndex * 9781u +
                    shadowRaySample + 0x7f4a7c15u);
                rng::Xorshift128 directLightSampleRng(lightSampleSeed);
                const AreaLightSample lightSample = sampleMeshAreaLightByIndex(
                    scene, lightIndex, directLightSampleRng);
                if (!lightSample.valid || lightSample.pdfArea <= 1.0e-12f) {
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
                Ray shadowRay{};
                shadowRay.origin = startState.position + lightDirection * RayEpsilon;
                shadowRay.direction = lightDirection;
                shadowRay.normal = startState.orientedNormal;
                float shadowTransmission = 1.0f;
                if (!traceAdjointShadowTransmission(
                    scene, shadowRay, startState.position, lightDistance,
                    currentSurface.primitiveIndex, lightPrimitiveIndex, shadowTransmission)) {
                    continue;
                }
                MaterialEdgeGradientEvent materialDirectLightEdgeEvent{};
                materialDirectLightEdgeEvent.startSurface = currentSurface;
                materialDirectLightEdgeEvent.endSurface = lightSample.surface;
                materialDirectLightEdgeEvent.betaIncrement =
                        currentRayState.pathThroughput * currentRayState.transmission;
                materialDirectLightEdgeEvent.alpha = worldHit.alphaGeom * surfel.opacity;
                materialDirectLightEdgeEvent.bsdf = surfelBsdf;
                materialDirectLightEdgeEvent.invSamplePDF =
                        (1.0f / lightSample.pdfArea) * invQReflect * invSamplesPerLight;
                materialDirectLightEdgeEvent.segmentTransmittance = shadowTransmission;
                materialDirectLightEdgeEvent.directLightRadiance =
                        lightSample.flux / (M_PIf * lightSample.totalAreaWorld);
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
        }
    }

    template<typename SceneType, typename IntermediatesType>
    static inline void appendAdjointMaterialAuxiliaryStartEdgeSample(
        const SceneType &scene,
        const IntermediatesType &intermediates,
        const Point &surfel,
        const PointCloudSurfaceRecord &currentSurface,
        const WorldHit &worldHit,
        const RayState &currentRayState,
        const float3 &orientedNormal,
        const float3 &surfelBsdf,
        uint64_t renderSeed,
        uint32_t spp,
        float qNull,
        float qReflect) {
        const ReconstructedSurfelState startState = reconstructSurfelState(
            surfel, currentSurface);
        const uint64_t auxiliaryRecursiveSeed = rng::makeSeed(
            renderSeed, currentRayState.pathId, spp, rng::kStreamDirection,
            currentRayState.traversalIndex * 2246822519u + 0x51ed270bu);
        rng::Xorshift128 auxiliaryRecursiveRng(auxiliaryRecursiveSeed);
        float3 auxiliaryDirectionWorld{0.0f, 0.0f, 0.0f};
        float auxiliaryDirectionPdf = 1.0f / (2.0f * M_PIf);
        sampleUniformHemisphereAroundNormal(
            auxiliaryRecursiveRng, orientedNormal, auxiliaryDirectionWorld,
            auxiliaryDirectionPdf);
        if (auxiliaryDirectionPdf <= 1.0e-12f) {
            return;
        }
        Ray auxiliaryRay{};
        auxiliaryRay.origin = startState.position + startState.orientedNormal * RayEpsilon;
        auxiliaryRay.direction = auxiliaryDirectionWorld;
        auxiliaryRay.normal = startState.orientedNormal;
        AdjointAuxiliaryEndpoint endpoint = sampleAdjointAuxiliaryPointEndpoint(
            scene, auxiliaryRay, auxiliaryRecursiveRng, qNull, qReflect,
            currentSurface.primitiveIndex, currentRayState.pathId, currentRayState.pixelIndex,
            true);
        if (!endpoint.found) {
            return;
        }
        const Point &auxiliarySurfel = scene.points[endpoint.surface.primitiveIndex];
        const ReconstructedSurfelState auxiliaryState = reconstructSurfelState(
            auxiliarySurfel, endpoint.surface);
        const float3 yzVector = auxiliaryState.position - startState.position;
        const float yzDistanceSquared = dot(yzVector, yzVector);
        if (yzDistanceSquared <= 1.0e-12f) {
            return;
        }
        const float yzDistance = sycl::sqrt(yzDistanceSquared);
        const float3 yzDirection = yzVector / yzDistance;
        const float cosineAtEnd = sycl::fmax(
            0.0f, dot(auxiliaryState.orientedNormal, -yzDirection));
        if (cosineAtEnd <= 1.0e-8f) {
            return;
        }
        const float auxiliaryAreaPdf =
                auxiliaryDirectionPdf * qReflect * cosineAtEnd / yzDistanceSquared;
        if (auxiliaryAreaPdf <= 1.0e-12f) {
            return;
        }
        MaterialEdgeGradientEvent innerIntegralStartEdge{};
        innerIntegralStartEdge.startSurface = currentSurface;
        innerIntegralStartEdge.endSurface = endpoint.surface;
        innerIntegralStartEdge.betaIncrement =
                currentRayState.pathThroughput * currentRayState.transmission;
        innerIntegralStartEdge.alpha = worldHit.alphaGeom * surfel.opacity;
        innerIntegralStartEdge.bsdf = surfelBsdf;
        innerIntegralStartEdge.invSamplePDF = 1.0f / auxiliaryAreaPdf;
        innerIntegralStartEdge.segmentTransmittance = 1.0f;
        innerIntegralStartEdge.directLightRadiance = float3{0.0f, 0.0f, 0.0f};
        innerIntegralStartEdge.isDirectLightSample = false;
        innerIntegralStartEdge.writeOcclusionGradients = true;
        innerIntegralStartEdge.pathId = currentRayState.pathId;
        innerIntegralStartEdge.startBounceIndex = currentRayState.bounceIndex;
        appendEventAtomic(
            intermediates.countMaterialStartEdgeEvents,
            intermediates.materialStartEdgeEvents,
            intermediates.maxMaterialStartEdgeEventCount,
            innerIntegralStartEdge);
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

                    const uint64_t seed =rng::makeSeed(renderSeed, globalRayIndex, spp, rng::kStreamRayGen, spp * spp);
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
                    Ray primaryRay = makePrimaryRayFromPixelJitteredFov(
                        sensor.camera,
                        static_cast<float>(pixelX),
                        static_cast<float>(pixelY),
                        jitterX, jitterY
                    );
                    //if (isWatchedPixel(pixelX, pixelY)) {
                    //    int debug = 1;
                    //} else {fin
                    //    return;
                    //}
                    //primaryRay.direction = normalize(float3{-0.001, 0.982122211, 0.277827293});    // a
                    //primaryRay.direction = normalize(float3{-0.01, 1.0, 0.04}); // b
                    //primaryRay.origin = float3{0.0, -4.0, 1.0};
                    float cameraCosine = dot(sensor.camera.forward, primaryRay.direction);
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
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = static_cast<uint32_t>(globalId[0]);
                    RayState currentRayState = intermediates.primaryRays[rayIndex];
                    const uint32_t pathId = currentRayState.pathId;
                    const bool hasPendingState = pathId < intermediates.maxPendingAdjointStateCount;
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

                    for (uint32_t inlineTraversalIndex = 0u; inlineTraversalIndex < kMaxSplatEventsPerRay; ++
                         inlineTraversalIndex) {
                        (void) inlineTraversalIndex;
                        const uint64_t stepSeed = rng::makeSeed(
                            renderSeed, currentRayState.pathId, spp, rng::kStreamTraversal,
                            currentRayState.traversalIndex);
                        rng::Xorshift128 rng(stepSeed);
                        WorldHit worldHit{};
                        intersectScene(currentRayState.ray, &worldHit, scene, SurfelIntersectMode::FirstHit);
                        if (!worldHit.hit) {
                            clearAdjointLocalPendingState(pendingCameraSegment, pendingAdjointStage);
                            break;
                        }
                        buildIntersectionNormal(scene, worldHit);
                        const InstanceRecord &instance = scene.instances[worldHit.instanceIndex];

                        if (instance.geometryType == GeometryType::PointCloud) {
                            const float localLayerDepthEpsilon = rendererDebugLocalLayerDepthEpsilon(settings);
                            const uint32_t maxLocalSurfelHits =
                                    rendererDebugMaxLocalSurfelHits(settings);
                            const float localLayerNormalCosineThreshold =
                                    rendererDebugLocalLayerNormalCosineThreshold(settings);

                            const PointCloudLocalLayer localLayer = collectPointCloudLocalLayer(
                                currentRayState.ray,
                                worldHit,
                                instance,
                                scene,
                                localLayerDepthEpsilon,
                                maxLocalSurfelHits,
                                localLayerNormalCosineThreshold);

                            const float qNull = settings.sampling.qNull;
                            const float qReflect = settings.sampling.qReflect;
                            float3 sampledOutgoingDirectionWorld{0.0f};
                            float3 throughputMultiplier{0.0f};
                            float3 slabNormal{0.0f};
                            MeasurementGradientEvent measurementEvent{};
                            uint32_t slabRecordCount = 0;

                            if (rng.nextFloat() < qNull) {
                                const float attenuation = localLayer.transmission;
                                currentRayState.ray.origin =
                                        currentRayState.ray.origin +
                                        currentRayState.ray.direction *
                                        (localLayer.furthestT + RayEpsilon);
                                currentRayState.pathThroughput *= 1.0f / qNull;
                                currentRayState.transmission *= attenuation;
                                currentRayState.traversalIndex++;
                                continue;
                            }

                            for (uint32_t localHitIndex = 0u; localHitIndex < localLayer.hitCount; ++localHitIndex) {
                                const float layerWeight = localLayer.weight[localHitIndex];
                                const LocalSurfelLayerHit &localHit = localLayer.hits[localHitIndex];
                                const Point &surfel = scene.points[localHit.primitiveIndex];
                                const float3 orientedNormal = computePointCloudOrientedNormal(
                                    surfel, currentRayState.ray.direction);
                                // First accepted hit decides the slab normal for continuation.
                                if (localHitIndex == 0)
                                    slabNormal = orientedNormal;
                                const PointCloudSurfaceRecord currentSurface = makePointCloudSurfaceRecord(
                                    localHit, currentRayState, scene);

                                float uniformHemispherePdf = 1.0f / (2.0f * M_PIf);
                                sampleUniformHemisphereAroundNormal(rng, orientedNormal, sampledOutgoingDirectionWorld,
                                                                    uniformHemispherePdf);
                                const float alpha = localHit.alphaGeom * surfel.opacity;
                                const float3 surfelBsdf = surfel.alpha_r * surfel.albedo * M_1_PIf;
                                const float cosineTheta = sycl::fmax(
                                    0.0f, dot(sampledOutgoingDirectionWorld, orientedNormal));
                                throughputMultiplier =
                                        ((alpha / qReflect) * surfelBsdf * cosineTheta) / uniformHemispherePdf;

                                if (hasPendingState) {
                                    const PendingCameraSegment previousCameraSegment = pendingCameraSegment;
                                    const PendingAdjointStageX previousAdjointStage = pendingAdjointStage;
                                    float segmentGeometryFromStoredVertex = 1.0f;
                                    float segmentAreaPdfFromStoredVertex = 1.0f;
                                    float segmentUvJacobianAtEnd = surfel.scale.x() * surfel.scale.y();
                                    if (previousAdjointStage.valid) {
                                        const Point &storedSurfel = scene.points[previousAdjointStage.current.surface.
                                            primitiveIndex];
                                        const ReconstructedSurfelState storedState = reconstructSurfelState(
                                            storedSurfel, previousAdjointStage.current.surface);
                                        const ReconstructedSurfelState liveState = reconstructSurfelState(
                                            surfel, currentSurface);
                                        segmentGeometryFromStoredVertex = computeGeometricTermValue(
                                            storedState.position, liveState.position, storedState.orientedNormal,
                                            liveState.orientedNormal);
                                        segmentAreaPdfFromStoredVertex = computeSegmentAreaPdfFromUniformHemisphere(
                                            storedState, liveState, uniformHemispherePdf);
                                    }
                                    if (previousCameraSegment.valid) {
                                        measurementEvent.xSurface[slabRecordCount] = currentSurface;
                                        measurementEvent.layerWeights[slabRecordCount] = layerWeight;
                                        measurementEvent.directLightEps[slabRecordCount] = localLayer.directLightEpsilon
                                                [localHitIndex];
                                        ++slabRecordCount;

                                        /*

                                        if (settings.maxAdjointBounces > 1)
                                            appendAdjointMeasurementAuxiliarySample(
                                                scene, intermediates, surfel, currentSurface, currentRayState,
                                                orientedNormal,
                                                renderSeed, spp, qNull, qReflect);
                                        */
                                    }
                                }

                                // Indirect bounces
                                /*
                                if (previousAdjointStage.valid && currentRayState.bounceIndex >= 1u) {
                                    appendAdjointMaterialVertexEvent(
                                        intermediates, currentSurface, currentRayState, qReflect);
                                    appendAdjointMaterialEdgeXYEvent(
                                        intermediates, previousAdjointStage, currentSurface, currentRayState,
                                        segmentAreaPdfFromStoredVertex, segmentUvJacobianAtEnd, qReflect);
                                    appendAdjointMaterialDirectLightEdgeSamples(
                                        settings, scene, intermediates, surfel, currentSurface, worldHit,
                                        currentRayState,
                                        surfelBsdf, renderSeed, spp, invQReflect);
                                    appendAdjointMaterialAuxiliaryStartEdgeSample(
                                        scene, intermediates, surfel, currentSurface, worldHit, currentRayState,
                                        orientedNormal, surfelBsdf, renderSeed, spp, qNull, qReflect);
                                }

                                clearPendingCameraSegment(pendingCameraSegment);
                                const float cosineFromPrevious =
                                        previousCameraSegment.valid
                                            ? dot(sensor.camera.forward, currentRayState.ray.direction)
                                            : 1.0f;
                                const PendingAdjointVertex newCurrentVertex = makePendingAdjointVertex(
                                    currentSurface,
                                    currentRayState.bounceIndex,
                                    currentRayState.pathThroughput / qReflect,
                                    currentRayState.transmission,
                                    segmentGeometryFromStoredVertex,
                                    segmentAreaPdfFromStoredVertex,
                                    surfelBsdf,
                                    alpha,
                                    cosineFromPrevious);
                                pushPendingAdjointVertex(
                                    pendingAdjointStage,
                                    currentRayState.pathId,
                                    currentRayState.pixelIndex,
                                    previousCameraSegment.valid,
                                    newCurrentVertex);
                                */
                            }

                            if (slabRecordCount > 0u) {
                                measurementEvent.surfelSlabCount = slabRecordCount;
                                measurementEvent.transmission = currentRayState.transmission;
                                measurementEvent.xPathThroughput = currentRayState.pathThroughput / qReflect;

                                appendEventAtomic(
                                    intermediates.countMeasurementEvents,
                                    intermediates.measurementEvents,
                                    intermediates.maxMeasurementEventCount,
                                    measurementEvent);

                                appendAdjointMeasurementDirectPointLightSlabEvents(
                                    settings, scene, intermediates, measurementEvent, currentRayState, qReflect);
                            }

                            nextRayState.ray.origin = worldHit.hitPositionW + slabNormal * RayEpsilon;
                            nextRayState.ray.direction = sampledOutgoingDirectionWorld;
                            nextRayState.ray.normal = slabNormal;
                            nextRayState.bounceIndex = currentRayState.bounceIndex + 1u;
                            nextRayState.pixelIndex = currentRayState.pixelIndex;
                            nextRayState.pathId = currentRayState.pathId;
                            nextRayState.pathThroughput =
                                    currentRayState.pathThroughput * throughputMultiplier * currentRayState.
                                    transmission;
                            nextRayState.traversalIndex = currentRayState.traversalIndex + 1u;
                            nextRayState.transmission = 1.0f;
                            if (applyRussianRoulette(
                                rng, nextRayState.bounceIndex, nextRayState.pathThroughput,
                                settings.russianRouletteStart)) {
                                shouldEnqueueNextRayState = true;
                            } else {
                                clearAdjointLocalPendingState(
                                    pendingCameraSegment, pendingAdjointStage);
                            }
                            // Finished qNull events, go to next bounce.
                            break;
                        }
                        storeAdjointPendingState(
                            intermediates, pathId, hasPendingState, pendingCameraSegment, pendingAdjointStage);
                        enqueueAdjointNextRayState(
                            intermediates, shouldEnqueueNextRayState, nextRayState);
                    }
                });
        }).wait();
    }


    SYCL_EXTERNAL inline SurfelGradientRecord makeZeroSurfelGradientRecord(uint32_t primitiveIndex = kInvalidIndex) {
        SurfelGradientRecord record{};
        record.primitiveIndex = primitiveIndex;
        record.gradBeta = 0.0f;
        record.gradEta = 0.0f;
        record.gradAlbedoR = 0.0f;
        record.gradAlbedoG = 0.0f;
        record.gradAlbedoB = 0.0f;
        record.gradPositionX = 0.0f;
        record.gradPositionY = 0.0f;
        record.gradPositionZ = 0.0f;
        record.gradScaleU = 0.0f;
        record.gradScaleV = 0.0f;
        record.gradRotationX = 0.0f;
        record.gradRotationY = 0.0f;
        record.gradRotationZ = 0.0f;
        return record;
    }

    static void measurementGradientEvent(
        RenderPackage &pkg,
        uint32_t cameraIndex,
        uint32_t measurementEventCount) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto &sensor = pkg.sensors[cameraIndex];
        auto debugImage = pkg.debugImages[cameraIndex];
        MeasurementGradientEvent *measurementEvents = pkg.intermediates.measurementEvents;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;
        uint32_t *gradientRecordCounter = pkg.intermediates.countGradientRecords;
        const uint32_t gradientRecordCapacity = pkg.intermediates.maxGradientRecordCount;

        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;
        const uint32_t pointCount = pkg.gradients.numPoints;

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<>(sycl::range<1>(measurementEventCount), [=](sycl::id<1> globalId) {
                const uint32_t eventIndex = static_cast<uint32_t>(globalId[0]);


                const MeasurementGradientEvent eventRecord = measurementEvents[eventIndex];
                const uint32_t slabCount = eventRecord.surfelSlabCount;
                if (slabCount == 0u || slabCount > kMaxLocalSurfelHits) return;

                const float3 pathWeight = eventRecord.xPathThroughput;
                float alphaEff[kMaxLocalSurfelHits];
                float3 slabDirectRadiance[kMaxLocalSurfelHits], targetSlabRadiance{0.0f}, targetAnchorPosition{0.0f};
                float targetDistance = 1.0e30f;
                uint32_t anchorSurfaceIndex = 0u;
                bool foundTargetSurface = false;

                for (uint32_t i = 0u; i < slabCount; ++i) {
                    const PointCloudSurfaceRecord &surface = eventRecord.xSurface[i];
                    const uint32_t primitiveIndex = surface.primitiveIndex;
                    alphaEff[i] = 0.0f;
                    slabDirectRadiance[i] = float3{0.0f};
                    if (primitiveIndex == kInvalidIndex || primitiveIndex >= pointCount) continue;

                    const Point &surfel = scene.points[primitiveIndex];
                    const ReconstructedSurfelState state = reconstructSurfelState(surfel, surface);
                    alphaEff[i] = sycl::clamp(surfel.opacity * surface.alphaGeom, 0.0f, 1.0f);
                    const float3 incidentIrradiance = computeIncidentRadianceFromPointLights(
                        scene, settings, state.position, state.orientedNormal, eventRecord.directLightEps[i]);
                    slabDirectRadiance[i] = incidentIrradiance * (surfel.alpha_r * surfel.albedo * M_1_PIf);
                    targetSlabRadiance += eventRecord.layerWeights[i] * slabDirectRadiance[i];

                    const float3 cameraToSurface = state.position - sensor.camera.pos;
                    const float distance = sycl::sqrt(dot(cameraToSurface, cameraToSurface));
                    if (distance < targetDistance) {
                        targetDistance = distance;
                        targetAnchorPosition = state.position;
                        anchorSurfaceIndex = i;
                        foundTargetSurface = true;
                    }
                }
                if (!foundTargetSurface || targetDistance <= 1.0e-8f) return;

                OccluderDerivative occluderDerivatives[kMaxCameraOccluderRecords];
                uint32_t storedOccluderCount = 0u;
                const float localLayerDepthEpsilon = rendererDebugLocalLayerDepthEpsilon(settings);
                const uint32_t maxLocalSurfelHits = rendererDebugMaxLocalSurfelHits(settings);
                const uint32_t maxSplatEventsPerRay = rendererDebugMaxSplatEventsPerRay(settings);
                const float localLayerNormalCosineThreshold = rendererDebugLocalLayerNormalCosineThreshold(settings);
                const float3 cameraToTarget = targetAnchorPosition - sensor.camera.pos;
                const float3 rayDirection = normalize(cameraToTarget);
                Ray ray{};
                ray.origin = sensor.camera.pos + rayDirection * RayEpsilon;
                ray.direction = rayDirection;
                float segmentTransmittance = 1.0f;

                for (uint32_t traversalIndex = 0u; traversalIndex < maxSplatEventsPerRay; ++traversalIndex) {
                    WorldHit worldHit{};
                    intersectScene(ray, &worldHit, scene, SurfelIntersectMode::FirstHit);
                    if (!worldHit.hit) break;

                    const float3 cameraToHit = worldHit.hitPositionW - sensor.camera.pos;
                    const float hitDistance = sycl::sqrt(dot(cameraToHit, cameraToHit));
                    if (hitDistance >= targetDistance - RayEpsilon) break;

                    buildIntersectionNormal(scene, worldHit);
                    const InstanceRecord &instance = scene.instances[worldHit.instanceIndex];
                    if (instance.geometryType != GeometryType::PointCloud) break;

                    const PointCloudLocalLayer occludingLayer = collectPointCloudLocalLayer(
                        ray, worldHit, instance, scene, localLayerDepthEpsilon, maxLocalSurfelHits,
                        localLayerNormalCosineThreshold);
                    const Point &referenceSurfel = scene.points[worldHit.primitiveIndex];
                    float3 referenceNormal = normalize(cross(referenceSurfel.tanU, referenceSurfel.tanV));
                    if (dot(referenceNormal, -rayDirection) < 0.0f) referenceNormal = -referenceNormal;

                    float prefixWithinLayer = 1.0f;
                    for (uint32_t hitIndex = 0u; hitIndex < occludingLayer.hitCount; ++hitIndex) {
                        const LocalSurfelLayerHit &localHit = occludingLayer.hits[hitIndex];
                        if (localHit.primitiveIndex == kInvalidIndex || localHit.primitiveIndex >= pointCount) continue;

                        const Point &occluderSurfel = scene.points[localHit.primitiveIndex];
                        float3 occluderNormal = normalize(cross(occluderSurfel.tanU, occluderSurfel.tanV));
                        if (dot(occluderNormal, -rayDirection) < 0.0f) occluderNormal = -occluderNormal;
                        if (dot(referenceNormal, occluderNormal) < localLayerNormalCosineThreshold) continue;

                        const float alphaGeom = localHit.alphaGeom;
                        const float alphaEffective = sycl::clamp(occluderSurfel.opacity * alphaGeom, 0.0f, 1.0f);
                        const float oneMinusAlpha = sycl::fmax(0.0f, 1.0f - alphaEffective);
                        const float2 uv = phiInverse(localHit.hitPositionW, occluderSurfel);
                        const float u = uv.x(), v = uv.y();
                        const float oneMinusR2 = 1.0f - u * u - v * v;
                        float3 gradPosition{0.0f}, gradWorldRotation{0.0f};
                        float gradScaleU = 0.0f, gradScaleV = 0.0f, gradBeta = 0.0f;

                        if (oneMinusR2 > 1.0e-8f) {
                            const float scaleU = occluderSurfel.scale.x();
                            const float scaleV = occluderSurfel.scale.y();
                            const float betaScale = 4.0f * sycl::exp(occluderSurfel.beta);
                            gradBeta = betaScale * sycl::log(oneMinusR2) * alphaEffective;
                            const float dAlphaGeomDu = -2.0f * betaScale * u * alphaGeom / oneMinusR2;
                            const float dAlphaGeomDv = -2.0f * betaScale * v * alphaGeom / oneMinusR2;
                            const float nDotD = dot(occluderNormal, rayDirection);

                            if (sycl::fabs(nDotD) > 1.0e-8f && scaleU > 1.0e-12f && scaleV > 1.0e-12f) {
                                const float3 dUdP =
                                        occluderNormal * (dot(rayDirection, occluderSurfel.tanU) / (scaleU * nDotD)) -
                                        occluderSurfel.tanU / scaleU;
                                const float3 dVdP =
                                        occluderNormal * (dot(rayDirection, occluderSurfel.tanV) / (scaleV * nDotD)) -
                                        occluderSurfel.tanV / scaleV;
                                gradPosition = occluderSurfel.opacity * (dAlphaGeomDu * dUdP + dAlphaGeomDv * dVdP);
                                gradScaleU = 2.0f * betaScale * u * u * alphaEffective / (scaleU * oneMinusR2);
                                gradScaleV = 2.0f * betaScale * v * v * alphaEffective / (scaleV * oneMinusR2);

                                const float3 hitMinusP = localHit.hitPositionW - occluderSurfel.position;
                                const float3 a = occluderSurfel.position - sensor.camera.pos;
                                const float nDotA = dot(occluderNormal, a);
                                const float invNDotD = 1.0f / nDotD;
                                const float3 q = (cross(occluderNormal, a) * nDotD - nDotA * cross(
                                                      occluderNormal, rayDirection)) * (invNDotD * invNDotD);
                                const float3 dUdRotation =
                                        q * (dot(rayDirection, occluderSurfel.tanU) / scaleU) + cross(
                                            occluderSurfel.tanU, hitMinusP) / scaleU;
                                const float3 dVdRotation =
                                        q * (dot(rayDirection, occluderSurfel.tanV) / scaleV) + cross(
                                            occluderSurfel.tanV, hitMinusP) / scaleV;
                                gradWorldRotation =
                                        occluderSurfel.opacity * (
                                            dAlphaGeomDu * dUdRotation + dAlphaGeomDv * dVdRotation);
                            }
                        }

                        if (storedOccluderCount < kMaxCameraOccluderRecords) {
                            OccluderDerivative &record = occluderDerivatives[storedOccluderCount++];
                            record = OccluderDerivative{};
                            record.primitiveIndex = localHit.primitiveIndex;
                            record.gradPosition = gradPosition;
                            record.gradRotation = computeLocalRotationGradientFromWorldRotationGradient(
                                occluderSurfel.tanU, occluderSurfel.tanV, gradWorldRotation);
                            record.gradScaleU = gradScaleU;
                            record.gradScaleV = gradScaleV;
                            record.gradEta = alphaGeom;
                            record.gradBeta = gradBeta;
                            record.prefixTransmittance = segmentTransmittance * prefixWithinLayer;
                            record.oneMinusAlpha = oneMinusAlpha;
                        }
                        prefixWithinLayer *= oneMinusAlpha;
                    }
                    segmentTransmittance *= prefixWithinLayer;
                    ray.origin += ray.direction * (occludingLayer.furthestT + RayEpsilon);
                }

                // Target slab parameters: only blending/profile terms here.
                for (uint32_t parameterIndex = 0u; parameterIndex < slabCount; ++parameterIndex) {
                    const PointCloudSurfaceRecord &xSurface = eventRecord.xSurface[parameterIndex];
                    const uint32_t primitiveIndex = xSurface.primitiveIndex;
                    if (primitiveIndex == kInvalidIndex || primitiveIndex >= pointCount) continue;

                    const Point &surfelX = scene.points[primitiveIndex];
                    const ReconstructedSurfelState xState = reconstructSurfelState(surfelX, xSurface);
                    float3 dSlabRadianceDAlphaK{0.0f};
                    for (uint32_t contributionIndex = 0u; contributionIndex < slabCount; ++contributionIndex) {
                        const float dWeightDAlphaK = computeNormalizedSlabWeightDerivativeWrtAlpha(
                            alphaEff, slabCount, contributionIndex, parameterIndex);
                        dSlabRadianceDAlphaK += slabDirectRadiance[contributionIndex] * dWeightDAlphaK;
                    }

                    const float dLossDAlphaK = dot(pathWeight, dSlabRadianceDAlphaK) * invSpp;
                    const float3 dAlphaEffDPosition = computeAlphaEffectiveGradientWrtTranslation(
                        surfelX, xSurface, xState);
                    const float3 gradPosition = dLossDAlphaK * dAlphaEffDPosition;
                    const float2 dAlphaEffDScale = computeAlphaEffectiveGradientWrtScale(surfelX, xSurface);
                    const float gradScaleU = dLossDAlphaK * dAlphaEffDScale.x();
                    const float gradScaleV = dLossDAlphaK * dAlphaEffDScale.y();
                    const float3 dAlphaEffDWorldRotation = computeAlphaEffectiveGradientWrtWorldRotation(
                        surfelX, xSurface, xState, sensor.camera.pos);
                    const float3 gradWorldRotation = dLossDAlphaK * dAlphaEffDWorldRotation;
                    const float3 gradRotation = computeLocalRotationGradientFromWorldRotationGradient(
                        surfelX.tanU, surfelX.tanV, gradWorldRotation);
                    const float gradEta = dLossDAlphaK * xSurface.alphaGeom;
                    const float u = xSurface.uv.x();
                    const float v = xSurface.uv.y();
                    const float oneMinusR2 = 1.0f - u * u - v * v;

                    float gradBeta = 0.0f;
                    if (oneMinusR2 > 1.0e-8f) {
                        const float betaScale = 4.0f * sycl::exp(surfelX.beta);
                        const float dAlphaGeomDBeta = betaScale * sycl::log(oneMinusR2) * xSurface.alphaGeom;
                        gradBeta = dLossDAlphaK * surfelX.opacity * dAlphaGeomDBeta;
                    }

                    const float3 incidentIrradiance = computeIncidentRadianceFromPointLights(
                        scene, settings, xState.position, xState.orientedNormal,
                        eventRecord.directLightEps[parameterIndex]);
                    const float albedoScale = M_1_PIf * eventRecord.layerWeights[parameterIndex] * invSpp;

                    SurfelGradientRecord gradientRecord = makeZeroSurfelGradientRecord(primitiveIndex);
                    gradientRecord.gradPositionX = gradPosition.x();
                    gradientRecord.gradPositionY = gradPosition.y();
                    gradientRecord.gradPositionZ = gradPosition.z();
                    gradientRecord.gradScaleU = gradScaleU;
                    gradientRecord.gradScaleV = gradScaleV;
                    gradientRecord.gradRotationX = gradRotation.x();
                    gradientRecord.gradRotationY = gradRotation.y();
                    gradientRecord.gradRotationZ = gradRotation.z();
                    gradientRecord.gradEta = gradEta;
                    gradientRecord.gradBeta = gradBeta;
                    gradientRecord.gradAlbedoR = pathWeight.x() * incidentIrradiance.x() * albedoScale;
                    gradientRecord.gradAlbedoG = pathWeight.y() * incidentIrradiance.y() * albedoScale;
                    gradientRecord.gradAlbedoB = pathWeight.z() * incidentIrradiance.z() * albedoScale;
                    if (appendGradientRecordBounded(
                        gradientRecordCounter,
                        gradientRecords,
                        gradientRecordCapacity,
                        gradientRecord)) {
                        accumulateDebugGradientIfSelected(
                            debugImage,
                            settings.renderDebugGradientImages,
                            settings.surfelIndexForDebugImages,
                            xSurface.pathId,
                            gradientRecord);
                    }
                }

                // Camera -> target slab transmission.
                const float scalarWeightOcclusion = dot(pathWeight, targetSlabRadiance);
                float suffixTransmittance = 1.0f;
                for (uint32_t reverseIndex = storedOccluderCount; reverseIndex > 0u; --reverseIndex) {
                    const uint32_t occluderIndex = reverseIndex - 1u;
                    const OccluderDerivative &occluder = occluderDerivatives[occluderIndex];
                    const float scale = -occluder.prefixTransmittance * suffixTransmittance * scalarWeightOcclusion *
                                        invSpp;

                    SurfelGradientRecord gradientRecord = makeZeroSurfelGradientRecord(occluder.primitiveIndex);

                    const float3 position = scale * occluder.gradPosition;
                    const float3 rotation = scale * occluder.gradRotation;

                    gradientRecord.gradPositionX = position.x();
                    gradientRecord.gradPositionY = position.y();
                    gradientRecord.gradPositionZ = position.z();
                    gradientRecord.gradScaleU = scale * occluder.gradScaleU;
                    gradientRecord.gradScaleV = scale * occluder.gradScaleV;
                    gradientRecord.gradRotationX = rotation.x();
                    gradientRecord.gradRotationY = rotation.y();
                    gradientRecord.gradRotationZ = rotation.z();
                    gradientRecord.gradEta = scale * occluder.gradEta;
                    gradientRecord.gradBeta = scale * occluder.gradBeta;

                    if (appendGradientRecordBounded(
                        gradientRecordCounter,
                        gradientRecords,
                        gradientRecordCapacity,
                        gradientRecord)) {
                        accumulateDebugGradientIfSelected(
                            debugImage,
                            settings.renderDebugGradientImages,
                            settings.surfelIndexForDebugImages,
                            eventRecord.xSurface[anchorSurfaceIndex].pathId,
                            gradientRecord);
                    }

                    suffixTransmittance *= occluder.oneMinusAlpha;
                }
            });
        }).wait();
    }

    static void measurementGradientEventXY(
        RenderPackage &pkg,
        uint32_t eventCount,
        uint32_t cameraIndex) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto &sensor = pkg.sensors[cameraIndex];
        auto debugImage = pkg.debugImages[cameraIndex];

        MeasurementGradientEventXY *measurementEvents = pkg.intermediates.measurementTwoPointEvents;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;
        uint32_t *gradientRecordCounter = pkg.intermediates.countGradientRecords;
        const uint32_t gradientRecordCapacity = pkg.intermediates.maxGradientRecordCount;

        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;
        const uint32_t pointCount = pkg.gradients.numPoints;
        const uint32_t maxSplatEventsPerRay = rendererDebugMaxSplatEventsPerRay(settings);
        const float localLayerDepthEpsilon = rendererDebugLocalLayerDepthEpsilon(settings);
        const uint32_t maxLocalSurfelHits = rendererDebugMaxLocalSurfelHits(settings);
        const float localLayerNormalCosineThreshold = rendererDebugLocalLayerNormalCosineThreshold(settings);

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class firstHitGradientEventTag>(sycl::range<1>(eventCount), [=](sycl::id<1> globalId) {
                const uint32_t eventIndex = static_cast<uint32_t>(globalId[0]);


                const MeasurementGradientEventXY eventRecord = measurementEvents[eventIndex];
                const uint32_t slabCount = eventRecord.surfelSlabCount;
                if (slabCount == 0u || slabCount > kMaxLocalSurfelHits) return;

                const float3 lightPositionW = eventRecord.pointLightPositionW;
                const float3 pointLightIntensity = eventRecord.pointLightRadiantIntensity;
                const float3 pathWeight = eventRecord.xPathThroughput;

                for (uint32_t localIndex = 0u; localIndex < slabCount; ++localIndex) {
                    const PointCloudSurfaceRecord &xSurface = eventRecord.xSurface[localIndex];
                    const uint32_t xPrimitiveIndex = xSurface.primitiveIndex;
                    if (xPrimitiveIndex == kInvalidIndex || xPrimitiveIndex >= pointCount) continue;

                    const float layerWeight = eventRecord.layerWeights[localIndex];
                    if (layerWeight <= 0.0f) continue;

                    const Point &surfelX = scene.points[xPrimitiveIndex];
                    const ReconstructedSurfelState xState = reconstructSurfelState(surfelX, xSurface);
                    PointLightGeometry lightGeometry{};
                    if (!computePointLightGeometry(xState.position, xState.orientedNormal, lightPositionW,
                                                   lightGeometry))
                        continue;

                    const float3 brdfX = surfelX.alpha_r * surfelX.albedo * M_1_PIf;
                    const float3 transportWithoutTauAndGeometric = pointLightIntensity * layerWeight * brdfX;
                    const float scalarWeightWithoutTauAndGeometric = dot(pathWeight, transportWithoutTauAndGeometric);
                    const float3 segmentVector = lightPositionW - xState.position;
                    const float targetDistanceSquared = dot(segmentVector, segmentVector);
                    if (targetDistanceSquared <= 1.0e-12f) continue;

                    const float targetDistance = sycl::sqrt(targetDistanceSquared);
                    const float3 rayDirection = segmentVector / targetDistance;
                    Ray ray{};
                    ray.direction = rayDirection;
                    ray.normal = xState.orientedNormal;
                    ray.origin = xState.position + rayDirection * sycl::fmax(
                                     RayEpsilon, eventRecord.directLightEps[localIndex]);

                    OccluderDerivative occluderDerivatives[kMaxShadowOccluderRecords];
                    uint32_t storedOccluderCount = 0u;
                    float segmentTransmittance = 1.0f;

                    for (uint32_t traversalIndex = 0u; traversalIndex < maxSplatEventsPerRay; ++traversalIndex) {
                        WorldHit shadowHit{};
                        intersectScene(ray, &shadowHit, scene, SurfelIntersectMode::FirstHit);
                        if (!shadowHit.hit) break;

                        const float3 hitVector = shadowHit.hitPositionW - xState.position;
                        const float hitDistance = sycl::sqrt(dot(hitVector, hitVector));
                        if (hitDistance >= targetDistance - RayEpsilon) break;

                        buildIntersectionNormal(scene, shadowHit);
                        const InstanceRecord &instance = scene.instances[shadowHit.instanceIndex];
                        if (instance.geometryType != GeometryType::PointCloud) break;

                        const PointCloudLocalLayer occludingLayer = collectPointCloudLocalLayer(
                            ray, shadowHit, instance, scene, localLayerDepthEpsilon, maxLocalSurfelHits,
                            localLayerNormalCosineThreshold);
                        float prefixWithinLayer = 1.0f;

                        for (uint32_t hitIndex = 0u; hitIndex < occludingLayer.hitCount; ++hitIndex) {
                            const LocalSurfelLayerHit &localHit = occludingLayer.hits[hitIndex];
                            const uint32_t occluderPrimitiveIndex = localHit.primitiveIndex;
                            if (occluderPrimitiveIndex == kInvalidIndex || occluderPrimitiveIndex >= pointCount)
                                continue;
                            if (isPrimitiveInMeasurementSlab(eventRecord, occluderPrimitiveIndex) ||
                                occluderPrimitiveIndex == eventRecord.pointLightPrimitiveIndex)
                                continue;

                            const Point &occluderSurfel = scene.points[occluderPrimitiveIndex];
                            float3 occluderNormal = normalize(cross(occluderSurfel.tanU, occluderSurfel.tanV));
                            if (dot(occluderNormal, -rayDirection) < 0.0f) occluderNormal = -occluderNormal;

                            const float alphaGeom = localHit.alphaGeom;
                            const float alphaEffective = sycl::clamp(occluderSurfel.opacity * alphaGeom, 0.0f, 1.0f);
                            const float oneMinusAlpha = sycl::fmax(0.0f, 1.0f - alphaEffective);
                            const float2 uv = phiInverse(localHit.hitPositionW, occluderSurfel);
                            const float u = uv.x(), v = uv.y();
                            const float oneMinusR2 = 1.0f - u * u - v * v;
                            float3 gradPosition{0.0f}, gradWorldRotation{0.0f};
                            float gradScaleU = 0.0f, gradScaleV = 0.0f, gradBeta = 0.0f;

                            if (oneMinusR2 > 1.0e-8f) {
                                const float scaleU = occluderSurfel.scale.x();
                                const float scaleV = occluderSurfel.scale.y();
                                const float betaScale = 4.0f * sycl::exp(occluderSurfel.beta);
                                gradBeta = betaScale * sycl::log(oneMinusR2) * alphaEffective;
                                const float dAlphaGeomDu = -2.0f * betaScale * u * alphaGeom / oneMinusR2;
                                const float dAlphaGeomDv = -2.0f * betaScale * v * alphaGeom / oneMinusR2;
                                const float nDotD = dot(occluderNormal, rayDirection);

                                if (sycl::fabs(nDotD) > 1.0e-8f && scaleU > 1.0e-12f && scaleV > 1.0e-12f) {
                                    const float3 dUdP =
                                            occluderNormal * (dot(rayDirection, occluderSurfel.tanU) / (scaleU * nDotD))
                                            - occluderSurfel.tanU / scaleU;
                                    const float3 dVdP =
                                            occluderNormal * (dot(rayDirection, occluderSurfel.tanV) / (scaleV * nDotD))
                                            - occluderSurfel.tanV / scaleV;
                                    gradPosition = occluderSurfel.opacity * (dAlphaGeomDu * dUdP + dAlphaGeomDv * dVdP);
                                    gradScaleU = 2.0f * betaScale * u * u * alphaEffective / (scaleU * oneMinusR2);
                                    gradScaleV = 2.0f * betaScale * v * v * alphaEffective / (scaleV * oneMinusR2);

                                    const float3 hitMinusP = localHit.hitPositionW - occluderSurfel.position;
                                    const float3 a = occluderSurfel.position - xState.position;
                                    const float nDotA = dot(occluderNormal, a);
                                    const float invNDotD = 1.0f / nDotD;
                                    const float3 q = (cross(occluderNormal, a) * nDotD - nDotA * cross(
                                                          occluderNormal, rayDirection)) * (invNDotD * invNDotD);
                                    const float3 dUdRotation =
                                            q * (dot(rayDirection, occluderSurfel.tanU) / scaleU) + cross(
                                                occluderSurfel.tanU, hitMinusP) / scaleU;
                                    const float3 dVdRotation =
                                            q * (dot(rayDirection, occluderSurfel.tanV) / scaleV) + cross(
                                                occluderSurfel.tanV, hitMinusP) / scaleV;
                                    gradWorldRotation =
                                            occluderSurfel.opacity * (
                                                dAlphaGeomDu * dUdRotation + dAlphaGeomDv * dVdRotation);
                                }
                            }

                            if (storedOccluderCount < kMaxShadowOccluderRecords) {
                                OccluderDerivative &record = occluderDerivatives[storedOccluderCount++];
                                record = OccluderDerivative{};
                                record.primitiveIndex = occluderPrimitiveIndex;
                                record.gradPosition = gradPosition;
                                record.gradRotation = computeLocalRotationGradientFromWorldRotationGradient(
                                    occluderSurfel.tanU, occluderSurfel.tanV, gradWorldRotation);
                                record.gradScaleU = gradScaleU;
                                record.gradScaleV = gradScaleV;
                                record.gradEta = alphaGeom;
                                record.gradBeta = gradBeta;
                                record.gradAlphaWrtSegmentStart = computeAlphaEffectiveGradientWrtSegmentStart(
                                    occluderSurfel, localHit, xState.position, lightPositionW);
                                record.prefixTransmittance = segmentTransmittance * prefixWithinLayer;
                                record.oneMinusAlpha = oneMinusAlpha;
                            }
                            prefixWithinLayer *= oneMinusAlpha;
                        }
                        segmentTransmittance *= prefixWithinLayer;
                        ray.origin += ray.direction * (occludingLayer.furthestT + RayEpsilon);
                    }

                    float suffixTransmittance = 1.0f;
                    float3 gradTauWrtSegmentStart{0.0f};
                    for (uint32_t reverseIndex = storedOccluderCount; reverseIndex > 0u; --reverseIndex) {
                        const OccluderDerivative &occluder = occluderDerivatives[reverseIndex - 1u];
                        const float dTauDAlpha = -occluder.prefixTransmittance * suffixTransmittance;
                        gradTauWrtSegmentStart += dTauDAlpha * occluder.gradAlphaWrtSegmentStart;
                        const float visibilityScale =
                                dTauDAlpha * lightGeometry.geometricTerm * scalarWeightWithoutTauAndGeometric * invSpp;

                        SurfelGradientRecord gradientRecord = makeZeroSurfelGradientRecord(occluder.primitiveIndex);
                        const float3 position = visibilityScale * occluder.gradPosition, rotation =
                                visibilityScale * occluder.gradRotation;
                        gradientRecord.gradPositionX = position.x();
                        gradientRecord.gradPositionY = position.y();
                        gradientRecord.gradPositionZ = position.z();
                        gradientRecord.gradScaleU = visibilityScale * occluder.gradScaleU;
                        gradientRecord.gradScaleV = visibilityScale * occluder.gradScaleV;
                        gradientRecord.gradRotationX = rotation.x();
                        gradientRecord.gradRotationY = rotation.y();
                        gradientRecord.gradRotationZ = rotation.z();
                        gradientRecord.gradEta = visibilityScale * occluder.gradEta;
                        gradientRecord.gradBeta = visibilityScale * occluder.gradBeta;
                        if (appendGradientRecordBounded(
                            gradientRecordCounter,
                            gradientRecords,
                            gradientRecordCapacity,
                            gradientRecord)) {
                            accumulateDebugGradientIfSelected(
                                debugImage,
                                settings.renderDebugGradientImages,
                                settings.surfelIndexForDebugImages,
                                xSurface.pathId,
                                gradientRecord);
                        }
                        suffixTransmittance *= occluder.oneMinusAlpha;
                    }

                    const float3 gradEdgeWrtHitPosition = scalarWeightWithoutTauAndGeometric *
                                                          (segmentTransmittance * lightGeometry.
                                                           gradientWrtSurfacePosition + lightGeometry.geometricTerm *
                                                           gradTauWrtSegmentStart) * invSpp;
                    const float3x3 hitPositionJacobian = planeHitPointIntersectionJacobian(
                        xSurface.incomingDirection, xState.orientedNormal);


                    const float3 gradPosition = transpose(hitPositionJacobian) * gradEdgeWrtHitPosition;

                    const float3x3 hitRotationJacobian = planeHitPointRotationJacobian(
                        sensor.camera.pos, xSurface.incomingDirection, surfelX.position, xState.orientedNormal);
                    const float3 normalRotationContribution =
                            scalarWeightWithoutTauAndGeometric * segmentTransmittance *
                            cross(xState.orientedNormal, lightGeometry.gradientWrtSurfaceNormal) * invSpp;
                    const float3 gradWorldRotation =
                            transpose(hitRotationJacobian) * gradEdgeWrtHitPosition + normalRotationContribution;
                    const float3 gradRotation = computeLocalRotationGradientFromWorldRotationGradient(
                        surfelX.tanU, surfelX.tanV, gradWorldRotation);

                    SurfelGradientRecord targetRecord = makeZeroSurfelGradientRecord(xPrimitiveIndex);
                    targetRecord.gradPositionX = gradPosition.x();
                    targetRecord.gradPositionY = gradPosition.y();
                    targetRecord.gradPositionZ = gradPosition.z();
                    targetRecord.gradRotationX = gradRotation.x();
                    targetRecord.gradRotationY = gradRotation.y();
                    targetRecord.gradRotationZ = gradRotation.z();
                    if (appendGradientRecordBounded(
                        gradientRecordCounter,
                        gradientRecords,
                        gradientRecordCapacity,
                        targetRecord)) {
                        accumulateDebugGradientIfSelected(
                            debugImage,
                            settings.renderDebugGradientImages,
                            settings.surfelIndexForDebugImages,
                            xSurface.pathId,
                            targetRecord);
                    }
                }
            });
        }).wait();
    }

    static void resetGradientRecordCounter(RenderPackage &pkg) {
        pkg.queue.memset(pkg.intermediates.countGradientRecords, 0, sizeof(uint32_t)).wait();
    }

    static uint32_t readBoundedGradientRecordCount(RenderPackage &pkg, const char *producerName) {
        uint32_t attemptedGradientRecordCount = 0u;

        pkg.queue.memcpy(
            &attemptedGradientRecordCount,
            pkg.intermediates.countGradientRecords,
            sizeof(uint32_t)).wait();

        const uint32_t storedGradientRecordCount =
                std::min(attemptedGradientRecordCount, pkg.intermediates.maxGradientRecordCount);

        if (attemptedGradientRecordCount > pkg.intermediates.maxGradientRecordCount) {
            const uint32_t discardedGradientRecordCount =
                    attemptedGradientRecordCount - pkg.intermediates.maxGradientRecordCount;

            Log::PA_WARN(
                "{} discarded {} gradient records because the gradient-record scratch buffer was full "
                "(attempted: {}, capacity: {}). Gradients are incomplete for this pass.",
                producerName,
                discardedGradientRecordCount,
                attemptedGradientRecordCount,
                pkg.intermediates.maxGradientRecordCount);
        }

        return storedGradientRecordCount;
    }

    static void reduceSurfelGradientRecords(
        RenderPackage &pkg,
        uint32_t gradientRecordCount,
        uint32_t cameraSlot,
        uint32_t cameraSlotCount) {
        auto &queue = pkg.queue;
        auto gradients = pkg.gradients;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;

        constexpr float maxAbsGradientComponent = 1.0e6f;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<struct reduceSurfelGradientRecords>(
                sycl::range<1>(gradientRecordCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t recordIndex = static_cast<uint32_t>(globalId[0]);
                    const SurfelGradientRecord gradientRecord = gradientRecords[recordIndex];

                    if (gradientRecord.primitiveIndex == kInvalidIndex) {
                        return;
                    }

                    const auto isValidGradientComponent = [](float value) -> bool {
                        return sycl::isfinite(value) && !sycl::isnan(value) && sycl::fabs(value) <=
                               maxAbsGradientComponent;
                    };

                    const bool validGradientRecord =
                            isValidGradientComponent(gradientRecord.gradPositionX) &&
                            isValidGradientComponent(gradientRecord.gradPositionY) &&
                            isValidGradientComponent(gradientRecord.gradPositionZ) &&
                            isValidGradientComponent(gradientRecord.gradScaleU) &&
                            isValidGradientComponent(gradientRecord.gradScaleV) &&
                            isValidGradientComponent(gradientRecord.gradRotationX) &&
                            isValidGradientComponent(gradientRecord.gradRotationY) &&
                            isValidGradientComponent(gradientRecord.gradRotationZ) &&
                            isValidGradientComponent(gradientRecord.gradEta) &&
                            isValidGradientComponent(gradientRecord.gradBeta) &&
                            isValidGradientComponent(gradientRecord.gradAlbedoR) &&
                            isValidGradientComponent(gradientRecord.gradAlbedoG) &&
                            isValidGradientComponent(gradientRecord.gradAlbedoB);

                    if (!validGradientRecord) {
                        return;
                    }

                    const uint32_t primitiveIndex = gradientRecord.primitiveIndex;
                    if (primitiveIndex >= gradients.numPoints || cameraSlot >= cameraSlotCount) {
                        return;
                    }

                    atomicAddFloat(gradients.gradPosition[primitiveIndex].x(), gradientRecord.gradPositionX);
                    atomicAddFloat(gradients.gradPosition[primitiveIndex].y(), gradientRecord.gradPositionY);
                    atomicAddFloat(gradients.gradPosition[primitiveIndex].z(), gradientRecord.gradPositionZ);

                    const uint32_t primitiveCameraIndex = primitiveIndex * cameraSlotCount + cameraSlot;
                    atomicAddFloat(gradients.gradPositionPerPrimitivePerCamera[primitiveCameraIndex].x(),
                                   gradientRecord.gradPositionX);
                    atomicAddFloat(gradients.gradPositionPerPrimitivePerCamera[primitiveCameraIndex].y(),
                                   gradientRecord.gradPositionY);
                    atomicAddFloat(gradients.gradPositionPerPrimitivePerCamera[primitiveCameraIndex].z(),
                                   gradientRecord.gradPositionZ);
                    atomicAddUint32(gradients.gradPositionRecordCountPerPrimitivePerCamera[primitiveCameraIndex], 1u);

                    atomicAddFloat(gradients.gradScale[primitiveIndex].x(), gradientRecord.gradScaleU);
                    atomicAddFloat(gradients.gradScale[primitiveIndex].y(), gradientRecord.gradScaleV);

                    atomicAddFloat(gradients.gradRotation[primitiveIndex].x(), gradientRecord.gradRotationX);
                    atomicAddFloat(gradients.gradRotation[primitiveIndex].y(), gradientRecord.gradRotationY);
                    atomicAddFloat(gradients.gradRotation[primitiveIndex].z(), gradientRecord.gradRotationZ);

                    atomicAddFloat(gradients.gradOpacity[primitiveIndex], gradientRecord.gradEta);
                    atomicAddFloat(gradients.gradBeta[primitiveIndex], gradientRecord.gradBeta);

                    atomicAddFloat(gradients.gradAlbedo[primitiveIndex].x(), gradientRecord.gradAlbedoR);
                    atomicAddFloat(gradients.gradAlbedo[primitiveIndex].y(), gradientRecord.gradAlbedoG);
                    atomicAddFloat(gradients.gradAlbedo[primitiveIndex].z(), gradientRecord.gradAlbedoB);
                });
        }).wait();
    }

    void computePerPrimitiveTranslationGradientStats(RenderPackage &pkg) {
        auto &queue = pkg.queue;
        auto gradients = pkg.gradients;

        const uint32_t pointCount = static_cast<uint32_t>(gradients.numPoints);
        const uint32_t cameraSlotCount = static_cast<uint32_t>(gradients.cameraSlotCount);

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class ComputePerPrimitiveTranslationGradientStatsKernel>(
                sycl::range<1>(pointCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t primitiveIndex = static_cast<uint32_t>(globalId[0]);

                    float3 gradientSum{0.0f, 0.0f, 0.0f};
                    float gradientNormSum = 0.0f;
                    float gradientSquaredNormSum = 0.0f;
                    uint32_t activeCameraCount = 0u;

                    for (uint32_t cameraIndex = 0u; cameraIndex < cameraSlotCount; ++cameraIndex) {
                        const uint32_t primitiveCameraIndex = primitiveIndex * cameraSlotCount + cameraIndex;

                        if (gradients.gradPositionRecordCountPerPrimitivePerCamera[primitiveCameraIndex] == 0u) {
                            continue;
                        }

                        const float3 cameraGradient =
                                gradients.gradPositionPerPrimitivePerCamera[primitiveCameraIndex];

                        const float cameraGradientSquaredNorm =
                                cameraGradient.x() * cameraGradient.x() +
                                cameraGradient.y() * cameraGradient.y() +
                                cameraGradient.z() * cameraGradient.z();

                        const float cameraGradientNorm = sycl::sqrt(cameraGradientSquaredNorm);

                        gradientSum += cameraGradient;
                        gradientNormSum += cameraGradientNorm;
                        gradientSquaredNormSum += cameraGradientSquaredNorm;
                        activeCameraCount += 1u;
                    }

                    if (activeCameraCount == 0u) {
                        gradients.gradPositionMeanNorm[primitiveIndex] = 0.0f;
                        gradients.gradPositionStd[primitiveIndex] = 0.0f;
                        gradients.gradPositionCoherence[primitiveIndex] = 0.0f;
                        gradients.gradPositionDisagreement[primitiveIndex] = 0.0f;
                        gradients.gradPositionActiveCameraCount[primitiveIndex] = 0u;
                        return;
                    }

                    const float inverseActiveCameraCount = 1.0f / static_cast<float>(activeCameraCount);
                    const float3 meanGradient = gradientSum * inverseActiveCameraCount;

                    const float meanGradientSquaredNorm =
                            meanGradient.x() * meanGradient.x() +
                            meanGradient.y() * meanGradient.y() +
                            meanGradient.z() * meanGradient.z();

                    const float meanGradientNorm = sycl::sqrt(meanGradientSquaredNorm);
                    const float meanPerCameraGradientNorm = gradientNormSum * inverseActiveCameraCount;
                    const float expectedSquaredNorm = gradientSquaredNormSum * inverseActiveCameraCount;
                    const float variance = sycl::fmax(0.0f, expectedSquaredNorm - meanGradientSquaredNorm);
                    const float translationStd = sycl::sqrt(variance);

                    constexpr float epsilon = 1.0e-12f;
                    const float coherence = meanGradientNorm / (meanPerCameraGradientNorm + epsilon);
                    const float clampedCoherence = sycl::fmin(1.0f, sycl::fmax(0.0f, coherence));
                    const float disagreement = meanPerCameraGradientNorm * (1.0f - clampedCoherence);

                    gradients.gradPositionMeanNorm[primitiveIndex] = meanPerCameraGradientNorm;
                    gradients.gradPositionStd[primitiveIndex] = translationStd;
                    gradients.gradPositionCoherence[primitiveIndex] = clampedCoherence;
                    gradients.gradPositionDisagreement[primitiveIndex] = disagreement;
                    gradients.gradPositionActiveCameraCount[primitiveIndex] = activeCameraCount;
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

    struct SurfaceRegularizerConstituentGradient {
        float3 position{0.0f};
        float3 tangentU{0.0f};
        float3 tangentV{0.0f};
        float scaleU = 0.0f;
        float scaleV = 0.0f;
        float opacity = 0.0f;
        float beta = 0.0f;
    };

    struct SurfaceRegularizerSlabRecord {
        float transmission = 1.0f;
        float opacity = 0.0f;
        float Tprev = 1.0f;
        float compositeWeight = 0.0f;
        float meanDepth = 0.0f;
        float meanNdcDepth = 0.0f;
    };

    SYCL_EXTERNAL inline SurfaceRegularizerConstituentGradient differentiateSurfaceRegularizerConstituent(
        const Point &surfel,
        const LocalSurfelLayerHit &hit,
        const float3 &rayDir,
        const float3 &cameraForward,
        float barAlphaEffective,
        float barDepth,
        const float3 &barOrientedNormal) {
        constexpr float kDenomEps = 1.0e-8f;

        SurfaceRegularizerConstituentGradient grad{};

        const float3 p = surfel.position;
        const float3 tu = surfel.tanU;
        const float3 tv = surfel.tanV;
        const float su = surfel.scale.x();
        const float sv = surfel.scale.y();
        const float eta = surfel.opacity;
        const float3 x = hit.hitPositionW;

        if (sycl::fabs(su) <= kDenomEps || sycl::fabs(sv) <= kDenomEps) return grad;

        const float2 uv = phiInverse(x, surfel);
        const float u = uv.x();
        const float v = uv.y();

        const AlphaKernelEval kernelEval = evaluateAlphaKernelAndDerivatives(surfel, u, v);

        const float barAlphaGeom = barAlphaEffective * eta;
        grad.opacity = barAlphaEffective * hit.alphaGeom;
        grad.beta = barAlphaGeom * kernelEval.dValue_dBeta;

        const float barU = barAlphaGeom * kernelEval.dValue_dU;
        const float barV = barAlphaGeom * kernelEval.dValue_dV;

        const float3 q = x - p;

        float3 barX = barDepth * cameraForward;
        float3 barQ{0.0f};

        barQ += (barU / su) * tu;
        barQ += (barV / sv) * tv;

        grad.tangentU += (barU / su) * q;
        grad.tangentV += (barV / sv) * q;

        grad.scaleU += -barU * u / su;
        grad.scaleV += -barV * v / sv;

        barX += barQ;
        grad.position -= barQ;

        const float3 nRaw = cross(tu, tv);
        const float nRawLength = sycl::sqrt(dot(nRaw, nRaw));

        if (nRawLength > kDenomEps) {
            const float3 n = nRaw / nRawLength;

            float3 barN{0.0f};

            // ---------------------------------------------------------------------
            // x = ray/plane intersection: propagate derivatives of depth/UV through x.
            // ---------------------------------------------------------------------
            const float nDotD = dot(n, rayDir);

            if (sycl::fabs(nDotD) > kDenomEps) {
                const float barLambda = dot(barX, rayDir);

                grad.position += (barLambda / nDotD) * n;
                barN += (barLambda / nDotD) * (p - x);
            }

            // ---------------------------------------------------------------------
            // Direct derivative of the camera-oriented surfel normal.
            // Orientation-side changes are treated as discrete.
            // ---------------------------------------------------------------------
            const float orientationSign = dot(n, -rayDir) < 0.0f ? -1.0f : 1.0f;
            barN += orientationSign * barOrientedNormal;

            const float3 barNRaw = (barN - n * dot(n, barN)) / nRawLength;

            grad.tangentU += cross(tv, barNRaw);
            grad.tangentV += cross(barNRaw, tu);
        }

        return grad;
    }

    void launchSurfaceRegularizersBackwardKernel(RenderPackage &pkg, uint32_t cameraIndex) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;

        SensorGPU sensor = pkg.sensors[cameraIndex];

        DebugImages debugImage{};
        const bool writeDebugImages = settings.renderDebugGradientImages && pkg.debugImages != nullptr;
        if (writeDebugImages) debugImage = pkg.debugImages[cameraIndex];

        const uint32_t imageWidth = sensor.camera.width;
        const uint32_t imageHeight = sensor.camera.height;
        const uint32_t pixelCount = imageWidth * imageHeight;

        PointGradients depthGradients = pkg.depthDistortionGradients;
        PointGradients normalGradients = pkg.normalConsistencyGradients;

        const uint32_t pointCount = static_cast<uint32_t>(depthGradients.numPoints);

        const bool enableDepthDistortionRegularizer = settings.depthDistortionWeight != 0.0f;
        const bool enableNormalConsistencyRegularizer = settings.normalConsistencyWeight != 0.0f;

        const float localLayerDepthEpsilon = rendererDebugLocalLayerDepthEpsilon(settings);
        const uint32_t maxSlabEvents = rendererDebugMaxSplatEventsPerRay(settings);
        const uint32_t maxLocalSurfelHits = rendererDebugMaxLocalSurfelHits(settings);
        const float localLayerNormalCosineThreshold = rendererDebugLocalLayerNormalCosineThreshold(settings);

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class SurfaceRegularizersBackwardKernel>(sycl::range<1>(pixelCount), [=](sycl::id<1> tid) {
                constexpr float kDenomEps = 1.0e-8f;

                const uint32_t pixelIndex = static_cast<uint32_t>(tid[0]);
                const uint32_t pixelX = pixelIndex % imageWidth;
                const uint32_t pixelY = pixelIndex / imageWidth;

                const float depthDistortionAdjoint = sensor.depthDistortionAdjointBuffer[pixelIndex];

                const float3 visibleNormalAdjoint{
                    sensor.visibleNormalAdjointBuffer[pixelIndex].x(),
                    sensor.visibleNormalAdjointBuffer[pixelIndex].y(),
                    sensor.visibleNormalAdjointBuffer[pixelIndex].z()
                };

                const float medianDepthAdjoint = sensor.medianDepthAdjointBuffer[pixelIndex];

                const bool useDepthDistortion =
                        enableDepthDistortionRegularizer && sycl::fabs(depthDistortionAdjoint) > 1.0e-12f;

                const bool useVisibleNormal =
                        sycl::fabs(visibleNormalAdjoint.x()) > 1.0e-12f ||
                        sycl::fabs(visibleNormalAdjoint.y()) > 1.0e-12f ||
                        sycl::fabs(visibleNormalAdjoint.z()) > 1.0e-12f;

                const bool useMedianDepth = sycl::fabs(medianDepthAdjoint) > 1.0e-12f;

                const bool useNormalConsistency =
                        enableNormalConsistencyRegularizer && (useVisibleNormal || useMedianDepth);

                if (!useDepthDistortion && !useNormalConsistency) return;

                const Ray originalRay = makePrimaryRayFromPixelJitteredFov(
                    sensor.camera,
                    static_cast<float>(pixelX),
                    static_cast<float>(pixelY),
                    0.0f,
                    0.0f);

                const float3 rayDir0 = originalRay.direction;

                // =====================================================================
                // PASS 1: collect only slab summaries.
                // =====================================================================
                SurfaceRegularizerSlabRecord slabs[kMaxSplatEventsPerRay];
                uint32_t slabCount = 0u;

                float transmittance = 1.0f;
                float accumulatedCompositeWeight = 0.0f;
                uint32_t medianSlabIndex = kInvalidIndex;

                Ray primaryRay = originalRay;

                for (uint32_t traversalIndex = 0u;
                     traversalIndex < maxSlabEvents && slabCount < kMaxSplatEventsPerRay;
                     ++traversalIndex) {
                    WorldHit worldHit{};
                    intersectScene(primaryRay, &worldHit, scene, SurfelIntersectMode::FirstHit);

                    if (!worldHit.hit) break;

                    buildIntersectionNormal(scene, worldHit);

                    const InstanceRecord &instance = scene.instances[worldHit.instanceIndex];

                    if (instance.geometryType != GeometryType::PointCloud) break;

                    const PointCloudLocalLayer localLayer = collectPointCloudLocalLayer(
                        primaryRay,
                        worldHit,
                        instance,
                        scene,
                        localLayerDepthEpsilon,
                        maxLocalSurfelHits,
                        localLayerNormalCosineThreshold);

                    const float A = localLayer.opacity;

                    if (A > kDenomEps) {
                        float weightedDepth = 0.0f;
                        float weightedNdcDepth = 0.0f;

                        for (uint32_t i = 0u; i < localLayer.hitCount; ++i) {
                            const float w = localLayer.weight[i];
                            if (w <= 0.0f) continue;

                            const float z = dot(localLayer.hits[i].hitPositionW - sensor.camera.pos,
                                                sensor.camera.forward);
                            weightedDepth += w * z;
                            weightedNdcDepth += w * depthDistortionNdc01(z);
                        }

                        SurfaceRegularizerSlabRecord &record = slabs[slabCount];

                        record.transmission = localLayer.transmission;
                        record.opacity = A;
                        record.Tprev = transmittance;
                        record.compositeWeight = transmittance * A;
                        record.meanDepth = weightedDepth / A;
                        record.meanNdcDepth = weightedNdcDepth / A;

                        if (useNormalConsistency &&
                            medianSlabIndex == kInvalidIndex &&
                            accumulatedCompositeWeight + record.compositeWeight >= 0.5f) {
                            medianSlabIndex = slabCount;
                        }

                        accumulatedCompositeWeight += record.compositeWeight;
                        ++slabCount;
                    }

                    transmittance *= localLayer.transmission;
                    primaryRay.origin += primaryRay.direction * (localLayer.furthestT + RayEpsilon);
                }

                if (slabCount == 0u) return;

                // =====================================================================
                // Distortion reverse pass between slabs.
                // =====================================================================
                float barCompositeWeight[kMaxSplatEventsPerRay];
                float barMeanNdcDepth[kMaxSplatEventsPerRay];
                float barOpacity[kMaxSplatEventsPerRay];

                for (uint32_t k = 0u; k < slabCount; ++k) {
                    barCompositeWeight[k] = 0.0f;
                    barMeanNdcDepth[k] = 0.0f;
                    barOpacity[k] = 0.0f;
                }

                if (useDepthDistortion && slabCount > 1u) {
                    for (uint32_t k = 0u; k < slabCount; ++k) {
                        for (uint32_t l = k + 1u; l < slabCount; ++l) {
                            const float mk = slabs[k].meanNdcDepth;
                            const float ml = slabs[l].meanNdcDepth;
                            const float Ck = slabs[k].compositeWeight;
                            const float Cl = slabs[l].compositeWeight;

                            const float difference = mk - ml;
                            const float differenceSquared = difference * difference;

                            barCompositeWeight[k] += depthDistortionAdjoint * Cl * differenceSquared;
                            barCompositeWeight[l] += depthDistortionAdjoint * Ck * differenceSquared;

                            const float scale = 2.0f * depthDistortionAdjoint * Ck * Cl;

                            barMeanNdcDepth[k] += scale * difference;
                            barMeanNdcDepth[l] -= scale * difference;
                        }
                    }

                    // C_k = T_k A_k
                    // T_{k+1} = T_k tau_k = T_k (1-A_k)
                    float barTnext = 0.0f;

                    for (int32_t k = static_cast<int32_t>(slabCount) - 1; k >= 0; --k) {
                        const float T = slabs[k].Tprev;
                        const float A = slabs[k].opacity;
                        const float tau = slabs[k].transmission;

                        barOpacity[k] += T * barCompositeWeight[k] - T * barTnext;

                        const float barTprev =
                                A * barCompositeWeight[k] +
                                tau * barTnext;

                        barTnext = barTprev;
                    }
                }

                // =====================================================================
                // PASS 2: retrace slabs and push their adjoints to constituents.
                // =====================================================================
                primaryRay = originalRay;
                uint32_t slabIndex = 0u;

                for (uint32_t traversalIndex = 0u;
                     traversalIndex < maxSlabEvents && slabIndex < slabCount;
                     ++traversalIndex) {
                    WorldHit worldHit{};
                    intersectScene(primaryRay, &worldHit, scene, SurfelIntersectMode::FirstHit);

                    if (!worldHit.hit) break;

                    buildIntersectionNormal(scene, worldHit);

                    const InstanceRecord &instance = scene.instances[worldHit.instanceIndex];

                    if (instance.geometryType != GeometryType::PointCloud) break;

                    const PointCloudLocalLayer localLayer = collectPointCloudLocalLayer(
                        primaryRay,
                        worldHit,
                        instance,
                        scene,
                        localLayerDepthEpsilon,
                        maxLocalSurfelHits,
                        localLayerNormalCosineThreshold);

                    const float A = localLayer.opacity;

                    if (A <= kDenomEps) {
                        primaryRay.origin += primaryRay.direction * (localLayer.furthestT + RayEpsilon);
                        continue;
                    }

                    const SurfaceRegularizerSlabRecord &slab = slabs[slabIndex];
                    const float invA = 1.0f / A;

                    float z[kMaxLocalSurfelHits];
                    float m[kMaxLocalSurfelHits];
                    float3 orientedNormal[kMaxLocalSurfelHits];

                    float barWeightDepth[kMaxLocalSurfelHits];
                    float barDepthDepth[kMaxLocalSurfelHits];

                    float barWeightNormal[kMaxLocalSurfelHits];
                    float barDepthNormal[kMaxLocalSurfelHits];
                    float3 barOrientedNormal[kMaxLocalSurfelHits];

                    float3 normalWeightedSum{0.0f};

                    for (uint32_t i = 0u; i < localLayer.hitCount; ++i) {
                        const LocalSurfelLayerHit &hit = localLayer.hits[i];
                        const Point &surfel = scene.points[hit.primitiveIndex];

                        z[i] = dot(hit.hitPositionW - sensor.camera.pos, sensor.camera.forward);
                        m[i] = depthDistortionNdc01(z[i]);

                        orientedNormal[i] = normalize(cross(surfel.tanU, surfel.tanV));
                        if (dot(orientedNormal[i], -rayDir0) < 0.0f) orientedNormal[i] = -orientedNormal[i];

                        normalWeightedSum += localLayer.weight[i] * orientedNormal[i];

                        barWeightDepth[i] = 0.0f;
                        barDepthDepth[i] = 0.0f;

                        barWeightNormal[i] = 0.0f;
                        barDepthNormal[i] = 0.0f;
                        barOrientedNormal[i] = float3{0.0f};
                    }

                    // -------------------------------------------------------------
                    // Depth distortion:
                    //
                    // A = sum_i w_i
                    // mbar = sum_i w_i m_i / A
                    // -------------------------------------------------------------
                    if (useDepthDistortion) {
                        for (uint32_t i = 0u; i < localLayer.hitCount; ++i) {
                            const float w = localLayer.weight[i];

                            barWeightDepth[i] += barOpacity[slabIndex];

                            barWeightDepth[i] +=
                                    barMeanNdcDepth[slabIndex] *
                                    (m[i] - slab.meanNdcDepth) *
                                    invA;

                            const float barMi =
                                    barMeanNdcDepth[slabIndex] * w * invA;

                            barDepthDepth[i] +=
                                    barMi * depthDistortionDndc01Ddepth(z[i]);
                        }
                    }

                    // -------------------------------------------------------------
                    // Normal-consistency outputs are only functions of the selected
                    // median slab. Selection itself is discrete.
                    // -------------------------------------------------------------
                    if (useNormalConsistency && slabIndex == medianSlabIndex) {
                        if (useMedianDepth) {
                            for (uint32_t i = 0u; i < localLayer.hitCount; ++i) {
                                const float w = localLayer.weight[i];

                                barWeightNormal[i] +=
                                        medianDepthAdjoint *
                                        (z[i] - slab.meanDepth) *
                                        invA;

                                barDepthNormal[i] +=
                                        medianDepthAdjoint *
                                        w *
                                        invA;
                            }
                        }

                        if (useVisibleNormal) {
                            const float normalLengthSquared = dot(normalWeightedSum, normalWeightedSum);

                            if (normalLengthSquared > 1.0e-16f) {
                                const float normalLength = sycl::sqrt(normalLengthSquared);
                                const float3 visibleNormal = normalWeightedSum / normalLength;

                                const float3 barNormalSum =
                                        (visibleNormalAdjoint -
                                         visibleNormal * dot(visibleNormal, visibleNormalAdjoint)) /
                                        normalLength;

                                for (uint32_t i = 0u; i < localLayer.hitCount; ++i) {
                                    const float w = localLayer.weight[i];

                                    barWeightNormal[i] += dot(barNormalSum, orientedNormal[i]);
                                    barOrientedNormal[i] += w * barNormalSum;
                                }
                            }
                        }
                    }

                    // -------------------------------------------------------------
                    // w_i = w_i(alpha_1,...,alpha_N)
                    //
                    // Convert all local weight adjoints to constituent alpha adjoints.
                    // -------------------------------------------------------------
                    for (uint32_t parameterIndex = 0u; parameterIndex < localLayer.hitCount; ++parameterIndex) {
                        const LocalSurfelLayerHit &hit = localLayer.hits[parameterIndex];
                        const uint32_t primitiveIndex = hit.primitiveIndex;

                        if (primitiveIndex == kInvalidIndex || primitiveIndex >= pointCount) continue;

                        float barAlphaDepth = 0.0f;
                        float barAlphaNormal = 0.0f;

                        for (uint32_t contributionIndex = 0u; contributionIndex < localLayer.hitCount; ++
                             contributionIndex) {
                            const float dWeightDAlpha =
                                    computeNormalizedSlabWeightDerivativeWrtAlpha(
                                        localLayer.alphaEff,
                                        localLayer.hitCount,
                                        contributionIndex,
                                        parameterIndex);

                            barAlphaDepth += barWeightDepth[contributionIndex] * dWeightDAlpha;
                            barAlphaNormal += barWeightNormal[contributionIndex] * dWeightDAlpha;
                        }

                        const Point &surfel = scene.points[primitiveIndex];

                        const SurfaceRegularizerConstituentGradient depthGradient =
                                differentiateSurfaceRegularizerConstituent(
                                    surfel,
                                    hit,
                                    rayDir0,
                                    sensor.camera.forward,
                                    barAlphaDepth,
                                    barDepthDepth[parameterIndex],
                                    float3{0.0f});

                        const SurfaceRegularizerConstituentGradient normalGradient =
                                differentiateSurfaceRegularizerConstituent(
                                    surfel,
                                    hit,
                                    rayDir0,
                                    sensor.camera.forward,
                                    barAlphaNormal,
                                    barDepthNormal[parameterIndex],
                                    barOrientedNormal[parameterIndex]);

                        if (useDepthDistortion) {
                            atomicAddFloat3(depthGradients.gradPosition[primitiveIndex], depthGradient.position);

                            const float3 gradRotation =
                                    computeLocalRotationGradientFromTangentGradients(
                                        surfel.tanU,
                                        surfel.tanV,
                                        depthGradient.tangentU,
                                        depthGradient.tangentV);

                            atomicAddFloat3(depthGradients.gradRotation[primitiveIndex], gradRotation);
                            atomicAddFloat2(depthGradients.gradScale[primitiveIndex],
                                            float2{depthGradient.scaleU, depthGradient.scaleV});
                            atomicAddFloat(depthGradients.gradOpacity[primitiveIndex], depthGradient.opacity);
                            atomicAddFloat(depthGradients.gradBeta[primitiveIndex], depthGradient.beta);
                        }

                        if (useNormalConsistency) {
                            atomicAddFloat3(normalGradients.gradPosition[primitiveIndex], normalGradient.position);

                            const float3 gradRotation =
                                    computeLocalRotationGradientFromTangentGradients(
                                        surfel.tanU,
                                        surfel.tanV,
                                        normalGradient.tangentU,
                                        normalGradient.tangentV);

                            atomicAddFloat3(normalGradients.gradRotation[primitiveIndex], gradRotation);
                            atomicAddFloat2(normalGradients.gradScale[primitiveIndex], float2{
                                                normalGradient.scaleU, normalGradient.scaleV
                                            });
                            atomicAddFloat(normalGradients.gradOpacity[primitiveIndex], normalGradient.opacity);
                            atomicAddFloat(normalGradients.gradBeta[primitiveIndex], normalGradient.beta);
                        }

                        if (writeDebugImages) {
                            const float3 totalPosition = depthGradient.position + normalGradient.position;

                            const float3 totalRotation =
                                    computeLocalRotationGradientFromTangentGradients(
                                        surfel.tanU,
                                        surfel.tanV,
                                        depthGradient.tangentU + normalGradient.tangentU,
                                        depthGradient.tangentV + normalGradient.tangentV);

                            SurfelGradientRecord debugRecord{};
                            debugRecord.primitiveIndex = primitiveIndex;
                            debugRecord.gradPositionX = totalPosition.x();
                            debugRecord.gradPositionY = totalPosition.y();
                            debugRecord.gradPositionZ = totalPosition.z();
                            debugRecord.gradScaleU = depthGradient.scaleU + normalGradient.scaleU;
                            debugRecord.gradScaleV = depthGradient.scaleV + normalGradient.scaleV;
                            debugRecord.gradRotationX = totalRotation.x();
                            debugRecord.gradRotationY = totalRotation.y();
                            debugRecord.gradRotationZ = totalRotation.z();
                            debugRecord.gradEta = depthGradient.opacity + normalGradient.opacity;
                            debugRecord.gradBeta = depthGradient.beta + normalGradient.beta;

                            accumulateDebugGradientIfSelected(
                                debugImage,
                                settings.renderDebugGradientImages,
                                settings.surfelIndexForDebugImages,
                                pixelIndex,
                                debugRecord);
                        }
                    }

                    ++slabIndex;

                    primaryRay.origin += primaryRay.direction * (localLayer.furthestT + RayEpsilon);
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
        const uint32_t safeMeasurementEventCount =
                sycl::min(measurementEventCount, pkg.intermediates.maxMeasurementEventCount);

        const uint32_t safeMeasurementTwoPointEventCount =
                sycl::min(measurementTwoPointEventCount, pkg.intermediates.maxMeasurementTwoPointEventCount);

        const uint32_t safeMaterialVertexEventCount =
                sycl::min(materialVertexEventCount, pkg.intermediates.maxMaterialVertexEventCount);

        const uint32_t safeMaterialEndEdgeEventCount =
                sycl::min(materialEndEdgeEventCount, pkg.intermediates.maxMaterialEndEdgeEventCount);

        const uint32_t safeMaterialStartEdgeEventCount =
                sycl::min(materialStartEdgeEventCount, pkg.intermediates.maxMaterialStartEdgeEventCount);

        if (measurementEventCount > safeMeasurementEventCount) {
            Log::PA_ERROR("Overflow: measurementEventCount={} max={}", measurementEventCount,
                          pkg.intermediates.maxMeasurementEventCount);
        }

        if (measurementTwoPointEventCount > safeMeasurementTwoPointEventCount) {
            Log::PA_ERROR("Overflow: measurementTwoPointEventCount={} max={}", measurementTwoPointEventCount,
                          pkg.intermediates.maxMeasurementTwoPointEventCount);
        }

        if (materialVertexEventCount > safeMaterialVertexEventCount) {
            Log::PA_ERROR("Overflow: materialVertexEventCount={} max={}", materialVertexEventCount,
                          pkg.intermediates.maxMaterialVertexEventCount);
        }

        if (materialEndEdgeEventCount > safeMaterialEndEdgeEventCount) {
            Log::PA_ERROR("Overflow: materialEndEdgeEventCount={} max={}", materialEndEdgeEventCount,
                          pkg.intermediates.maxMaterialEndEdgeEventCount);
        }

        if (materialStartEdgeEventCount > safeMaterialStartEdgeEventCount) {
            Log::PA_ERROR("Overflow: materialStartEdgeEventCount={} max={}", materialStartEdgeEventCount,
                          pkg.intermediates.maxMaterialStartEdgeEventCount);
        }

        const uint32_t cameraSlotIndex = pkg.sensors[cameraIndex].cameraSlotIndex;
        const uint32_t cameraSlotCount = static_cast<uint32_t>(pkg.gradients.cameraSlotCount);
        // -------------------------------------------------------------------------
        // Camera -> slab events.
        // Scratch starts at zero and is reduced immediately afterwards.
        // -------------------------------------------------------------------------
        if (safeMeasurementEventCount > 0u) {
            resetGradientRecordCounter(pkg); {
                ScopedTimer timer("measurementGradientEvent", spdlog::level::debug);
                measurementGradientEvent(pkg, cameraIndex, safeMeasurementEventCount);
            }

            const uint32_t gradientRecordCount =
                    readBoundedGradientRecordCount(pkg, "measurementGradientEvent");

            if (gradientRecordCount > 0u) {
                ScopedTimer timer("reduceMeasurementGradientRecords", spdlog::level::debug);
                reduceSurfelGradientRecords(pkg, gradientRecordCount, cameraSlotIndex, cameraSlotCount);
            }
        }

        // -------------------------------------------------------------------------
        // Surface -> point-light events.
        // Reuse exactly the same scratch memory.
        // -------------------------------------------------------------------------
        if (safeMeasurementTwoPointEventCount > 0u) {
            resetGradientRecordCounter(pkg); {
                ScopedTimer timer("measurementGradientEventXY", spdlog::level::debug);
                measurementGradientEventXY(pkg, safeMeasurementTwoPointEventCount, cameraIndex);
            }

            const uint32_t gradientRecordCount =
                    readBoundedGradientRecordCount(pkg, "measurementGradientEventXY");

            if (gradientRecordCount > 0u) {
                ScopedTimer timer("reduceMeasurementXYGradientRecords", spdlog::level::debug);
                reduceSurfelGradientRecords(pkg, gradientRecordCount, cameraSlotIndex, cameraSlotCount);
            }
        }

        // Same pattern later:
        //
        // materialVertexGradientEvent(..., 0u);
        // reduceSurfelGradientRecords(...);
        //
        // materialEndEdgeGradientEvent(..., 0u);
        // reduceSurfelGradientRecords(...);
        //
        // materialStartEdgeGradientEvent(..., 0u);
        // reduceSurfelGradientRecords(...);
    }
}
