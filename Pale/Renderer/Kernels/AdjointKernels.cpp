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

    SYCL_EXTERNAL inline bool hasAdjointPointLightOccluder(
        const GPUSceneBuffers &scene,
        const PointCloudSurfaceRecord &surface,
        float directLightEpsilon,
        const float3 &lightPositionW,
        uint32_t lightPrimitiveIndex) {
        const uint32_t xPrimitiveIndex = surface.primitiveIndex;
        if (xPrimitiveIndex == kInvalidIndex) {
            return false;
        }

        const Point &surfelX = scene.points[xPrimitiveIndex];
        const ReconstructedSurfelState xState =
                reconstructSurfelState(surfelX, surface);

        const float3 segmentVector =
                lightPositionW - xState.position;

        const float distanceSquared =
                dot(segmentVector, segmentVector);

        if (distanceSquared <= 1.0e-12f) {
            return false;
        }

        const float targetDistance =
                sycl::sqrt(distanceSquared);

        const float3 rayDirection =
                segmentVector / targetDistance;

        Ray ray{};
        ray.direction = rayDirection;
        ray.normal = xState.orientedNormal;
        ray.origin =
                xState.position +
                rayDirection * sycl::fmax(RayEpsilon, directLightEpsilon);

        for (uint32_t traversalIndex = 0u;
             traversalIndex < kMaxSplatEventsPerRay;
             ++traversalIndex) {
            WorldHit shadowHit{};
            intersectScene(
                ray, &shadowHit, scene,
                SurfelIntersectMode::FirstHit);

            if (!shadowHit.hit) {
                return false;
            }

            const float3 hitVector =
                    shadowHit.hitPositionW - xState.position;

            const float hitDistance =
                    sycl::sqrt(dot(hitVector, hitVector));

            if (hitDistance >= targetDistance - RayEpsilon) {
                return false;
            }

            const InstanceRecord &instance =
                    scene.instances[shadowHit.instanceIndex];

            // An opaque non-point-cloud blocks the light completely, but there is
            // no surfel transmission derivative to generate here.
            if (instance.geometryType != GeometryType::PointCloud) {
                return false;
            }

            const uint32_t primitiveIndex =
                    shadowHit.primitiveIndex;

            if (primitiveIndex == xPrimitiveIndex ||
                primitiveIndex == lightPrimitiveIndex) {
                ray.origin =
                        shadowHit.hitPositionW +
                        rayDirection * RayEpsilon;

                continue;
            }

            return true;
        }

        return false;
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
            bool hasLightOccluder = false;
            for (uint32_t i = 0u; i < slabEvent.surfelSlabCount; ++i) {
                if (slabEvent.layerWeights[i] <= 0.0f) {
                    continue;
                }

                if (hasAdjointPointLightOccluder(
                    scene,
                    slabEvent.xSurface[i],
                    slabEvent.directLightEps[i],
                    lightPositionW,
                    light.primitiveIndex)) {
                    hasLightOccluder = true;
                    break;
                }
            }

            if (!hasLightOccluder) {
                continue;
            }

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

    SYCL_EXTERNAL inline float integrateSlabPolynomial(
        const float *alpha, uint32_t count, uint32_t excludeA, uint32_t excludeB, uint32_t leadingZetaPower) {
        float coefficients[kMaxLocalSurfelHits];
        for (uint32_t i = 0u; i < kMaxLocalSurfelHits; ++i) {
            coefficients[i] = 0.0f;
        }
        coefficients[0] = 1.0f;
        uint32_t degree = 0u;
        for (uint32_t j = 0u; j < count; ++j) {
            if (j == excludeA || j == excludeB) {
                continue;
            }
            const float alphaJ = alpha[j];
            for (int32_t d = static_cast<int32_t>(degree); d >= 0; --d) {
                coefficients[d + 1] -= alphaJ * coefficients[d];
            }
            ++degree;
        }
        float integral = 0.0f;
        for (uint32_t d = 0u; d <= degree; ++d) {
            integral += coefficients[d] / static_cast<float>(d + leadingZetaPower + 1u);
        }
        return integral;
    }

    SYCL_EXTERNAL inline float computeRawSlabWeight(const float *alpha, uint32_t count, uint32_t surfelIndex) {
        const float Ii = integrateSlabPolynomial(alpha, count, surfelIndex, kInvalidIndex, 0u);
        return alpha[surfelIndex] * Ii;
    }

    SYCL_EXTERNAL inline float computeRawSlabWeightDerivativeWrtAlpha(const float *alpha, uint32_t count,
                                                                      uint32_t contributionIndex,
                                                                      uint32_t parameterIndex) {
        // d w_k / d alpha_k = I_k
        if (contributionIndex == parameterIndex) {
            return integrateSlabPolynomial(alpha, count, contributionIndex, kInvalidIndex, 0u);
        }
        // d w_i / d alpha_k = -alpha_i J_ik
        const float Jik = integrateSlabPolynomial(alpha, count, contributionIndex, parameterIndex, 1u);
        return -alpha[contributionIndex] * Jik;
    }

    SYCL_EXTERNAL inline float computeNormalizedSlabWeightDerivativeWrtAlpha(
        const float *alpha, uint32_t count, uint32_t contributionIndex, uint32_t parameterIndex) {
        float rawWeights[kMaxLocalSurfelHits];
        float rawWeightSum = 0.0f;
        float layerTransmission = 1.0f;
        for (uint32_t i = 0u; i < count; ++i) {
            rawWeights[i] = computeRawSlabWeight(alpha, count, i);
            rawWeightSum += rawWeights[i];
            layerTransmission *= sycl::fmax(0.0f, 1.0f - alpha[i]);
        }

        if (rawWeightSum <= 1.0e-8f) {
            return 0.0f;
        }
        const float layerOpacity = 1.0f - layerTransmission;
        float dRawWeightSumDAlphaK = 0.0f;
        for (uint32_t i = 0u; i < count; ++i) {
            dRawWeightSumDAlphaK += computeRawSlabWeightDerivativeWrtAlpha(alpha, count, i, parameterIndex);
        }
        // d alpha_Q / d alpha_k
        //
        // alpha_Q = 1 - prod_j (1-alpha_j)
        //
        float dLayerOpacityDAlphaK = 1.0f;
        for (uint32_t j = 0u; j < count; ++j) {
            if (j == parameterIndex) {
                continue;
            }
            dLayerOpacityDAlphaK *= sycl::fmax(0.0f, 1.0f - alpha[j]);
        }
        const float dRawWiDAlphaK = computeRawSlabWeightDerivativeWrtAlpha(
            alpha, count, contributionIndex, parameterIndex);
        const float normalization = layerOpacity / rawWeightSum;
        const float dNormalizationDAlphaK = (dLayerOpacityDAlphaK * rawWeightSum - layerOpacity * dRawWeightSumDAlphaK)
                                            / (rawWeightSum * rawWeightSum);
        return normalization * dRawWiDAlphaK + rawWeights[contributionIndex] * dNormalizationDAlphaK;
    }

    static void measurementGradientEvent(
        RenderPackage &pkg, uint32_t cameraIndex, uint32_t measurementEventCount, uint32_t baseOffset) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto &sensor = pkg.sensors[cameraIndex];
        auto debugImage = pkg.debugImages[cameraIndex];
        MeasurementGradientEvent *measurementEvents = pkg.intermediates.measurementEvents;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;
        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;
        const uint32_t pointCount = pkg.gradients.numPoints;
        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class measurementGradientEventTag>(
                sycl::range<1>(measurementEventCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex = static_cast<uint32_t>(globalId[0]);
                    // Keep the existing record architecture:
                    //   [0 .. kMaxLocalSurfelHits)              : target slab gradients
                    //   [kMaxLocalSurfelHits .. +kMaxSplat...) : transmission/occluder gradients
                    static constexpr uint32_t recordsPerEvent = 1u + kMaxLocalSurfelHits + kMaxSplatEventsPerRay;
                    const uint32_t eventRecordBase = baseOffset + recordsPerEvent * eventIndex;
                    const uint32_t occluderRecordBase = eventRecordBase + kMaxLocalSurfelHits;
                    for (uint32_t recordOffset = 0u; recordOffset < recordsPerEvent; ++recordOffset) {
                        SurfelGradientRecord invalidRecord{};
                        invalidRecord.primitiveIndex = kInvalidIndex;
                        gradientRecords[eventRecordBase + recordOffset] = invalidRecord;
                    }
                    const MeasurementGradientEvent eventRecord = measurementEvents[eventIndex];
                    const uint32_t slabCount = eventRecord.surfelSlabCount;
                    if (slabCount == 0u || slabCount > kMaxLocalSurfelHits) {
                        return;
                    }
                    const float3 pathWeight = eventRecord.xPathThroughput;
                    // ---------------------------------------------------------------------
                    // Reconstruct current target slab.
                    //
                    // alphaEff[i]          = eta_i * alphaGeom_i
                    // slabDirectRadiance[i] = R_i, i.e. direct outgoing radiance before w_i
                    // targetSlabRadiance    = sum_i w_i R_i
                    // ---------------------------------------------------------------------
                    float alphaEff[kMaxLocalSurfelHits];
                    float3 slabDirectRadiance[kMaxLocalSurfelHits];
                    float3 targetSlabRadiance{0.0f, 0.0f, 0.0f};
                    float3 targetAnchorPosition{0.0f, 0.0f, 0.0f};
                    float targetDistance = 1.0e30f;
                    uint32_t anchorSurfaceIndex = 0u;
                    bool foundTargetSurface = false;
                    for (uint32_t i = 0u; i < slabCount; ++i) {
                        const PointCloudSurfaceRecord &surface = eventRecord.xSurface[i];
                        const uint32_t primitiveIndex = surface.primitiveIndex;
                        alphaEff[i] = 0.0f;
                        slabDirectRadiance[i] = float3{0.0f};
                        if (primitiveIndex == kInvalidIndex || primitiveIndex >= pointCount) {
                            continue;
                        }
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
                    if (!foundTargetSurface || targetDistance <= 1.0e-8f) {
                        return;
                    }
                    // ---------------------------------------------------------------------
                    // Collect all opacity-bearing surfels on the OPEN camera -> target slab
                    // segment.
                    //
                    // Transmission remains:
                    //
                    //     tau = prod_m (1 - alpha_m)
                    //
                    // Each constituent surfel of an occluding slab therefore gets one
                    // OccluderDerivative record. Slab grouping changes traversal only,
                    // not the product derivative.
                    // ---------------------------------------------------------------------
                    OccluderDerivative occluderDerivatives[kMaxSplatEventsPerRay];
                    uint32_t storedOccluderCount = 0u;
                    const float localLayerDepthEpsilon = rendererDebugLocalLayerDepthEpsilon(settings);
                    const uint32_t maxLocalSurfelHits = rendererDebugMaxLocalSurfelHits(settings);
                    const uint32_t maxSplatEventsPerRay = rendererDebugMaxSplatEventsPerRay(settings);
                    const float localLayerNormalCosineThreshold =
                            rendererDebugLocalLayerNormalCosineThreshold(settings);
                    const float3 cameraToTarget = targetAnchorPosition - sensor.camera.pos;
                    const float3 rayDirection = normalize(cameraToTarget);
                    Ray ray{};
                    ray.origin = sensor.camera.pos + rayDirection * RayEpsilon;
                    ray.direction = rayDirection;
                    float segmentTransmittance = 1.0f;
                    for (uint32_t traversalIndex = 0u; traversalIndex < maxSplatEventsPerRay; ++traversalIndex) {
                        WorldHit worldHit{};
                        intersectScene(ray, &worldHit, scene, SurfelIntersectMode::FirstHit);
                        if (!worldHit.hit) {
                            break;
                        }
                        const float hitDistance = sycl::sqrt(dot(worldHit.hitPositionW - sensor.camera.pos,
                                                                 worldHit.hitPositionW - sensor.camera.pos));
                        // Open segment: do not differentiate the target slab here.
                        if (hitDistance >= targetDistance - RayEpsilon) {
                            break;
                        }
                        buildIntersectionNormal(scene, worldHit);
                        const InstanceRecord &instance = scene.instances[worldHit.instanceIndex];
                        if (instance.geometryType != GeometryType::PointCloud) {
                            segmentTransmittance = 0.0f;
                            break;
                        }
                        const PointCloudLocalLayer occludingLayer = collectPointCloudLocalLayer(
                            ray, worldHit, instance, scene, localLayerDepthEpsilon, maxLocalSurfelHits,
                            localLayerNormalCosineThreshold);
                        const Point &referenceSurfel = scene.points[worldHit.primitiveIndex];
                        float3 referenceNormal = normalize(cross(referenceSurfel.tanU, referenceSurfel.tanV));
                        if (dot(referenceNormal, -rayDirection) < 0.0f) {
                            referenceNormal = -referenceNormal;
                        }
                        float prefixWithinLayer = 1.0f;
                        for (uint32_t hitIndex = 0u; hitIndex < occludingLayer.hitCount; ++hitIndex) {
                            const LocalSurfelLayerHit &localHit = occludingLayer.hits[hitIndex];
                            if (localHit.primitiveIndex == kInvalidIndex ||
                                localHit.primitiveIndex >= pointCount) {
                                continue;
                            }
                            const Point &occluderSurfel = scene.points[localHit.primitiveIndex];
                            float3 occluderNormal = normalize(cross(occluderSurfel.tanU, occluderSurfel.tanV));
                            if (dot(occluderNormal, -rayDirection) < 0.0f) {
                                occluderNormal = -occluderNormal;
                            }
                            // Match the same slab-membership/normal filtering used in forward.
                            if (dot(referenceNormal, occluderNormal) < localLayerNormalCosineThreshold) {
                                continue;
                            }
                            const float alphaGeomOccluder = localHit.alphaGeom;
                            const float alphaEffective =
                                    sycl::clamp(occluderSurfel.opacity * alphaGeomOccluder, 0.0f, 1.0f);
                            const float oneMinusAlpha = sycl::fmax(0.0f, 1.0f - alphaEffective);
                            // -------------------------------------------------------------
                            // Local opacity derivatives.
                            // -------------------------------------------------------------
                            const float dAlphaEffectiveDEta = alphaGeomOccluder;
                            const float2 uv = phiInverse(localHit.hitPositionW, occluderSurfel);
                            const float uOcc = uv.x();
                            const float vOcc = uv.y();
                            const float radiusSquaredOcc = uOcc * uOcc + vOcc * vOcc;
                            const float oneMinusRadiusSquaredOcc = 1.0f - radiusSquaredOcc;
                            float dAlphaEffectiveDBeta = 0.0f;
                            float3 dAlphaEffectiveDPosition{0.0f};
                            float3 localRotationGradientOcc{0.0f};
                            float dAlphaEffectiveDScaleU = 0.0f;
                            float dAlphaEffectiveDScaleV = 0.0f;
                            if (oneMinusRadiusSquaredOcc > 1.0e-8f) {
                                const float scaleU = occluderSurfel.scale.x();
                                const float scaleV = occluderSurfel.scale.y();
                                const float3 tangentU = occluderSurfel.tanU;
                                const float3 tangentV = occluderSurfel.tanV;
                                const float betaScaleOcc = 4.0f * sycl::exp(occluderSurfel.beta);
                                dAlphaEffectiveDBeta =
                                        betaScaleOcc * sycl::log(oneMinusRadiusSquaredOcc) * alphaEffective;
                                const float dAlphaGeomDu =
                                        -2.0f * betaScaleOcc * uOcc * alphaGeomOccluder / oneMinusRadiusSquaredOcc;
                                const float dAlphaGeomDv =
                                        -2.0f * betaScaleOcc * vOcc * alphaGeomOccluder / oneMinusRadiusSquaredOcc;
                                const float nDotD = dot(occluderNormal, rayDirection);
                                if (sycl::fabs(nDotD) > 1.0e-8f && scaleU > 1.0e-12f && scaleV > 1.0e-12f) {
                                    const float invNDotD = 1.0f / nDotD;
                                    const float3 dUiDspi =
                                            occluderNormal *
                                            (dot(rayDirection, tangentU) / (scaleU * nDotD)) -
                                            tangentU / scaleU;
                                    const float3 dViDspi =
                                            occluderNormal *
                                            (dot(rayDirection, tangentV) / (scaleV * nDotD)) -
                                            tangentV / scaleV;
                                    dAlphaEffectiveDPosition =
                                            occluderSurfel.opacity * (dAlphaGeomDu * dUiDspi + dAlphaGeomDv * dViDspi);
                                    dAlphaEffectiveDScaleU =
                                            2.0f * betaScaleOcc * uOcc * uOcc * alphaEffective /
                                            (scaleU * oneMinusRadiusSquaredOcc);
                                    dAlphaEffectiveDScaleV =
                                            2.0f * betaScaleOcc * vOcc * vOcc * alphaEffective /
                                            (scaleV * oneMinusRadiusSquaredOcc);
                                    const float3 hitMinusSp = localHit.hitPositionW - occluderSurfel.position;
                                    const float3 aOcc = occluderSurfel.position - sensor.camera.pos;
                                    const float nDotA = dot(occluderNormal, aOcc);
                                    const float invNDotDSquared = invNDotD * invNDotD;
                                    const float3 qOcc =
                                            (cross(occluderNormal, aOcc) * nDotD -
                                             nDotA * cross(occluderNormal, rayDirection)) *
                                            invNDotDSquared;
                                    const float3 duDRotation =
                                            qOcc * (dot(rayDirection, tangentU) / scaleU) + cross(tangentU, hitMinusSp)
                                            / scaleU;
                                    const float3 dvDRotation =
                                            qOcc * (dot(rayDirection, tangentV) / scaleV) + cross(tangentV, hitMinusSp)
                                            / scaleV;
                                    const float3 dAlphaEffectiveDRotation =
                                            occluderSurfel.opacity * (
                                                dAlphaGeomDu * duDRotation + dAlphaGeomDv * dvDRotation);
                                    localRotationGradientOcc =
                                            computeLocalRotationGradientFromWorldRotationGradient(
                                                occluderSurfel.tanU, occluderSurfel.tanV,
                                                dAlphaEffectiveDRotation);
                                }
                            }
                            if (storedOccluderCount < kMaxSplatEventsPerRay) {
                                OccluderDerivative &record = occluderDerivatives[storedOccluderCount];
                                record.primitiveIndex = localHit.primitiveIndex;
                                record.gradPosition = dAlphaEffectiveDPosition;
                                record.gradRotation = localRotationGradientOcc;
                                record.gradScaleU = dAlphaEffectiveDScaleU;
                                record.gradScaleV = dAlphaEffectiveDScaleV;
                                record.gradEta = dAlphaEffectiveDEta;
                                record.gradBeta = dAlphaEffectiveDBeta;
                                // Product of all transmission factors before this surfel.
                                record.prefixTransmittance = segmentTransmittance * prefixWithinLayer;
                                record.oneMinusAlpha = oneMinusAlpha;
                                ++storedOccluderCount;
                            }
                            prefixWithinLayer *= oneMinusAlpha;
                        }
                        // This should equal prefixWithinLayer for the active constituents,
                        // but use the actual forward slab transmission.
                        segmentTransmittance *= occludingLayer.transmission;
                        ray.origin += ray.direction * (occludingLayer.furthestT + RayEpsilon);
                    }

                    // ---------------------------------------------------------------------
                    // Current target slab parameter gradients.
                    // ---------------------------------------------------------------------
                    for (uint32_t parameterIndex = 0u; parameterIndex < slabCount; ++parameterIndex) {
                        const uint32_t recordIndex = eventRecordBase + parameterIndex;
                        const PointCloudSurfaceRecord &xSurface = eventRecord.xSurface[parameterIndex];
                        const uint32_t primitiveIndex = xSurface.primitiveIndex;
                        if (primitiveIndex == kInvalidIndex || primitiveIndex >= pointCount) {
                            continue;
                        }
                        const Point &surfelX = scene.points[primitiveIndex];
                        // -------------------------------------------------------------
                        // d L_Q / d alpha_k
                        //
                        // Includes:
                        //   - surfel k's own weight derivative
                        //   - all other slab weights' derivative wrt alpha_k
                        // -------------------------------------------------------------
                        float3 dSlabRadianceDAlphaK{0.0f, 0.0f, 0.0f};
                        for (uint32_t contributionIndex = 0u; contributionIndex < slabCount; ++contributionIndex) {
                            const float dWeightDAlphaK = computeNormalizedSlabWeightDerivativeWrtAlpha(
                                alphaEff, slabCount, contributionIndex, parameterIndex);
                            dSlabRadianceDAlphaK += slabDirectRadiance[contributionIndex] * dWeightDAlphaK;
                        }
                        const float dLossDAlphaK = dot(pathWeight, dSlabRadianceDAlphaK) * invSpp;
                        // -------------------------------------------------------------
                        // eta:
                        //
                        // alpha_eff = eta * alpha_geom
                        // d alpha_eff / d eta = alpha_geom
                        // -------------------------------------------------------------
                        const float gradEta = dLossDAlphaK * xSurface.alphaGeom;
                        // -------------------------------------------------------------
                        // beta:
                        //
                        // alpha_geom = (1-r^2)^(4 exp(beta))
                        // -------------------------------------------------------------
                        const float u = xSurface.uv.x();
                        const float v = xSurface.uv.y();
                        const float radiusSquared = u * u + v * v;
                        const float oneMinusRadiusSquared = 1.0f - radiusSquared;
                        float gradBeta = 0.0f;
                        if (oneMinusRadiusSquared > 1.0e-8f) {
                            const float betaScale = 4.0f * sycl::exp(surfelX.beta);
                            const float dAlphaGeomDBeta =
                                    betaScale * sycl::log(oneMinusRadiusSquared) * xSurface.alphaGeom;
                            const float dAlphaEffDBeta = surfelX.opacity * dAlphaGeomDBeta;
                            gradBeta = dLossDAlphaK * dAlphaEffDBeta;
                        }
                        // -------------------------------------------------------------
                        // Verified Lambertian albedo gradients.
                        // -------------------------------------------------------------
                        const ReconstructedSurfelState xState = reconstructSurfelState(surfelX, xSurface);
                        const float3 incidentIrradiance =
                                computeIncidentRadianceFromPointLights(
                                    scene, settings, xState.position, xState.orientedNormal,
                                    eventRecord.directLightEps[parameterIndex]);
                        const float albedoScale =
                                M_1_PIf * eventRecord.layerWeights[parameterIndex] * invSpp;
                        const float gradAlbedoR = pathWeight.x() * incidentIrradiance.x() * albedoScale;
                        const float gradAlbedoG = pathWeight.y() * incidentIrradiance.y() * albedoScale;
                        const float gradAlbedoB = pathWeight.z() * incidentIrradiance.z() * albedoScale;
                        SurfelGradientRecord gradientRecord{};
                        gradientRecord.primitiveIndex = primitiveIndex;
                        gradientRecord.gradPositionX = 0.0f;
                        gradientRecord.gradPositionY = 0.0f;
                        gradientRecord.gradPositionZ = 0.0f;
                        gradientRecord.gradScaleU = 0.0f;
                        gradientRecord.gradScaleV = 0.0f;
                        gradientRecord.gradRotationX = 0.0f;
                        gradientRecord.gradRotationY = 0.0f;
                        gradientRecord.gradRotationZ = 0.0f;
                        gradientRecord.gradEta = gradEta;
                        gradientRecord.gradBeta = gradBeta;
                        gradientRecord.gradAlbedoR = gradAlbedoR;
                        gradientRecord.gradAlbedoG = gradAlbedoG;
                        gradientRecord.gradAlbedoB = gradAlbedoB;
                        gradientRecords[recordIndex] = gradientRecord;
                        accumulateDebugGradientIfSelected(
                            debugImage, settings.renderDebugGradientImages,
                            settings.surfelIndexForDebugImages, xSurface.pathId,
                            gradientRecord);
                    }

                    // ---------------------------------------------------------------------
                    // Camera-segment transmission gradients.
                    //
                    // targetSlabRadiance = sum_i w_i R_i
                    //
                    // For occluder k:
                    //
                    // d tau / d alpha_k
                    //   = - prefix_k * suffix_k
                    //
                    // and therefore
                    //
                    // dJ/dpsi_k =
                    //   p0 dot targetSlabRadiance
                    //   * d tau / d alpha_k
                    //   * d alpha_k / d psi_k.
                    // ---------------------------------------------------------------------
                    const float scalarWeightOcclusion = dot(pathWeight, targetSlabRadiance);
                    float suffixTransmittance = 1.0f;
                    for (uint32_t reverseIndex = storedOccluderCount; reverseIndex > 0u; --reverseIndex) {
                        const uint32_t occluderIndex = reverseIndex - 1u;
                        const uint32_t occluderRecordIndex = occluderRecordBase + occluderIndex;
                        const OccluderDerivative &occluder = occluderDerivatives[occluderIndex];
                        const float visibilityDerivativeScale =
                                -occluder.prefixTransmittance * suffixTransmittance * scalarWeightOcclusion * invSpp;
                        SurfelGradientRecord gradientRecord{};
                        gradientRecord.primitiveIndex = occluder.primitiveIndex;
                        const float3 positionContribution = visibilityDerivativeScale * occluder.gradPosition;
                        const float3 rotationContribution = visibilityDerivativeScale * occluder.gradRotation;
                        gradientRecord.gradPositionX = positionContribution.x();
                        gradientRecord.gradPositionY = positionContribution.y();
                        gradientRecord.gradPositionZ = positionContribution.z();
                        gradientRecord.gradScaleU = visibilityDerivativeScale * occluder.gradScaleU;
                        gradientRecord.gradScaleV = visibilityDerivativeScale * occluder.gradScaleV;
                        gradientRecord.gradRotationX = rotationContribution.x();
                        gradientRecord.gradRotationY = rotationContribution.y();
                        gradientRecord.gradRotationZ = rotationContribution.z();
                        gradientRecord.gradEta = visibilityDerivativeScale * occluder.gradEta;
                        gradientRecord.gradBeta = visibilityDerivativeScale * occluder.gradBeta;
                        gradientRecord.gradAlbedoR = 0.0f;
                        gradientRecord.gradAlbedoG = 0.0f;
                        gradientRecord.gradAlbedoB = 0.0f;
                        gradientRecords[occluderRecordIndex] = gradientRecord;
                        accumulateDebugGradientIfSelected(
                            debugImage, settings.renderDebugGradientImages,
                            settings.surfelIndexForDebugImages,
                            eventRecord.xSurface[anchorSurfaceIndex].pathId,
                            gradientRecord);
                        suffixTransmittance *= occluder.oneMinusAlpha;
                    }
                });
        }).wait();
    }

    struct PointLightGeometry {
        float geometricTerm = 0.0f;
        float3 gradientWrtSurfacePosition{0.0f, 0.0f, 0.0f};
        float3 gradientWrtSurfaceNormal{0.0f, 0.0f, 0.0f};
    };

    SYCL_EXTERNAL inline bool computePointLightGeometry(
        const float3 &surfacePositionW,
        const float3 &surfaceNormalW,
        const float3 &lightPositionW,
        PointLightGeometry &result) {
        const float3 vectorToLight = lightPositionW - surfacePositionW;
        const float distanceSquared = dot(vectorToLight, vectorToLight);
        if (distanceSquared <= 1.0e-12f) {
            return false;
        }
        const float inverseDistance = 1.0f / sycl::sqrt(distanceSquared);
        const float3 lightDirection = vectorToLight * inverseDistance;
        const float cosineAtSurface = dot(surfaceNormalW, lightDirection);
        // Same piecewise convention as max(0, n dot omega).
        if (cosineAtSurface <= 0.0f) {
            return false;
        }
        result.geometricTerm = cosineAtSurface / distanceSquared;
        // d/dX [ (n dot omega) / ||L - X||^2 ]
        result.gradientWrtSurfacePosition =
                (-surfaceNormalW + 3.0f * cosineAtSurface * lightDirection) *
                (inverseDistance / distanceSquared);
        // d/dn [ (n dot omega) / ||L - X||^2 ]
        result.gradientWrtSurfaceNormal =
                lightDirection / distanceSquared;

        return true;
    }

    SYCL_EXTERNAL inline bool isPrimitiveInMeasurementSlab(
        const MeasurementGradientEventXY &eventRecord,
        uint32_t primitiveIndex) {
        for (uint32_t i = 0u; i < eventRecord.surfelSlabCount; ++i) {
            if (eventRecord.xSurface[i].primitiveIndex == primitiveIndex) {
                return true;
            }
        }

        return false;
    }

    static void measurementGradientEventXY(RenderPackage &pkg, uint32_t eventCount, uint32_t baseOffset,
                                           uint32_t cameraIndex) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto debugImage = pkg.debugImages[cameraIndex];

        MeasurementGradientEventXY *measurementEvents = pkg.intermediates.measurementTwoPointEvents;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;

        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;
        const uint32_t pointCount = pkg.gradients.numPoints;
        const uint32_t maxSplatEventsPerRay = rendererDebugMaxSplatEventsPerRay(settings);

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class firstHitGradientEventTag>(
                sycl::range<1>(eventCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex = static_cast<uint32_t>(globalId[0]);
                    // One slab event can contain one shadow path per slab constituent.
                    static constexpr uint32_t recordsPerEvent = kMaxLocalSurfelHits * kMaxSplatEventsPerRay;
                    const uint32_t eventRecordBase = baseOffset + recordsPerEvent * eventIndex;
                    for (uint32_t recordOffset = 0u; recordOffset < recordsPerEvent; ++recordOffset) {
                        SurfelGradientRecord invalidRecord{};
                        invalidRecord.primitiveIndex = kInvalidIndex;
                        gradientRecords[eventRecordBase + recordOffset] = invalidRecord;
                    }
                    const MeasurementGradientEventXY eventRecord = measurementEvents[eventIndex];
                    const uint32_t slabCount = eventRecord.surfelSlabCount;
                    if (slabCount == 0u || slabCount > kMaxLocalSurfelHits) {
                        return;
                    }
                    const float3 lightPositionW = eventRecord.pointLightPositionW;
                    const float3 pointLightIntensity = eventRecord.pointLightRadiantIntensity;
                    const float3 pathWeight = eventRecord.xPathThroughput;
                    uint32_t outputRecordCount = 0u;
                    // ---------------------------------------------------------------------
                    // One event represents:
                    //
                    //   sum_i w_i I_l f_i G_il tau_il
                    //
                    // We still evaluate each constituent edge because x_i, n_i and tau_il
                    // may differ between surfels.
                    // ---------------------------------------------------------------------
                    for (uint32_t localIndex = 0u; localIndex < slabCount; ++localIndex) {
                        const PointCloudSurfaceRecord &xSurface = eventRecord.xSurface[localIndex];
                        const uint32_t xPrimitiveIndex = xSurface.primitiveIndex;
                        if (xPrimitiveIndex == kInvalidIndex || xPrimitiveIndex >= pointCount) {
                            continue;
                        }
                        const float layerWeight = eventRecord.layerWeights[localIndex];
                        if (layerWeight <= 0.0f) {
                            continue;
                        }
                        const Point &surfelX = scene.points[xPrimitiveIndex];
                        const ReconstructedSurfelState xState = reconstructSurfelState(surfelX, xSurface);
                        PointLightGeometry lightGeometry{};
                        if (!computePointLightGeometry(xState.position, xState.orientedNormal, lightPositionW,
                                                       lightGeometry)) {
                            continue;
                        }
                        const float3 brdfX = surfelX.alpha_r * surfelX.albedo * M_1_PIf;
                        const float3 transportWithoutTauAndGeometric = pointLightIntensity * layerWeight * brdfX;
                        const float scalarWeightWithoutTauAndGeometric = dot(
                            pathWeight, transportWithoutTauAndGeometric);
                        const float3 segmentVector = lightPositionW - xState.position;
                        const float targetDistanceSquared = dot(segmentVector, segmentVector);
                        if (targetDistanceSquared <= 1.0e-12f) {
                            continue;
                        }
                        const float targetDistance = sycl::sqrt(targetDistanceSquared);
                        const float3 rayDirection = segmentVector / targetDistance;
                        Ray ray{};
                        ray.direction = rayDirection;
                        ray.normal = xState.orientedNormal;
                        const float shadowRayEpsilon = sycl::fmax(
                            RayEpsilon, eventRecord.directLightEps[localIndex]);
                        ray.origin = xState.position + rayDirection * shadowRayEpsilon;
                        OccluderDerivative occluderDerivatives[kMaxSplatEventsPerRay];
                        uint32_t storedOccluderCount = 0u;
                        float segmentTransmittance = 1.0f;
                        // -------------------------------------------------------------
                        // Match the forward shadow estimator: individual FirstHit
                        // transmission factors.
                        // -------------------------------------------------------------
                        static constexpr uint32_t kMaxShadowOccluderRecords =
                                kMaxSplatEventsPerRay * kMaxLocalSurfelHits;

                        for (uint32_t traversalIndex = 0u; traversalIndex < maxSplatEventsPerRay; ++traversalIndex) {
                            WorldHit shadowHit{};
                            intersectScene(ray, &shadowHit, scene, SurfelIntersectMode::FirstHit);

                            if (!shadowHit.hit) {
                                break;
                            }

                            const float3 hitVector = shadowHit.hitPositionW - xState.position;
                            const float hitDistance = sycl::sqrt(dot(hitVector, hitVector));

                            if (hitDistance >= targetDistance - RayEpsilon) {
                                break;
                            }

                            buildIntersectionNormal(scene, shadowHit);

                            const InstanceRecord &instance = scene.instances[shadowHit.instanceIndex];
                            if (instance.geometryType != GeometryType::PointCloud) {
                                break;
                            }

                            const float localLayerDepthEpsilon = rendererDebugLocalLayerDepthEpsilon(settings);
                            const uint32_t maxLocalSurfelHits = rendererDebugMaxLocalSurfelHits(settings);
                            const float localLayerNormalCosineThreshold =
                                    rendererDebugLocalLayerNormalCosineThreshold(settings);
                            // FirstHit gives us the anchor of the occluding slab. Gather all nearby constituent surfels so coincident surfels are not missed.
                            const PointCloudLocalLayer occludingLayer = collectPointCloudLocalLayer(
                                ray, shadowHit, instance, scene, localLayerDepthEpsilon, maxLocalSurfelHits,
                                localLayerNormalCosineThreshold);

                            float prefixWithinLayer = 1.0f;

                            for (uint32_t hitIndex = 0u; hitIndex < occludingLayer.hitCount; ++hitIndex) {
                                const LocalSurfelLayerHit &localHit = occludingLayer.hits[hitIndex];
                                const uint32_t occluderPrimitiveIndex = localHit.primitiveIndex;

                                if (occluderPrimitiveIndex == kInvalidIndex || occluderPrimitiveIndex >= pointCount) {
                                    continue;
                                }

                                // The target slab belongs to the current interaction and must not contribute to its own outgoing shadow transmission.
                                if (isPrimitiveInMeasurementSlab(eventRecord, occluderPrimitiveIndex) ||
                                    occluderPrimitiveIndex == eventRecord.pointLightPrimitiveIndex) {
                                    continue;
                                }

                                const Point &occluderSurfel = scene.points[occluderPrimitiveIndex];
                                const float alphaGeom = localHit.alphaGeom;
                                const float alphaEffective =
                                        sycl::clamp(occluderSurfel.opacity * alphaGeom, 0.0f, 1.0f);
                                const float oneMinusAlpha = sycl::fmax(0.0f, 1.0f - alphaEffective);

                                if (storedOccluderCount < kMaxShadowOccluderRecords) {
                                    OccluderDerivative &record = occluderDerivatives[storedOccluderCount];
                                    record = OccluderDerivative{};
                                    record.primitiveIndex = occluderPrimitiveIndex;
                                    record.gradEta = alphaGeom;
                                    record.prefixTransmittance = segmentTransmittance * prefixWithinLayer;
                                    record.oneMinusAlpha = oneMinusAlpha;
                                    ++storedOccluderCount;
                                }

                                prefixWithinLayer *= oneMinusAlpha;
                            }

                            // Use the product of the constituents actually accepted above rather than occludingLayer.transmission,
                            // since target-slab members and the light carrier may have been excluded.
                            segmentTransmittance *= prefixWithinLayer;

                            // Advance past the complete occluding slab, including coincident surfels.
                            ray.origin += ray.direction * (occludingLayer.furthestT + RayEpsilon);
                        }
                        // -------------------------------------------------------------
                        // d tau_il / d eta_k for this constituent edge.
                        // -------------------------------------------------------------
                        float suffixTransmittance = 1.0f;
                        for (uint32_t reverseIndex = storedOccluderCount; reverseIndex > 0u; --reverseIndex) {
                            const uint32_t occluderIndex = reverseIndex - 1u;
                            const OccluderDerivative &occluder = occluderDerivatives[occluderIndex];
                            const float visibilityDerivativeScale =
                                    -occluder.prefixTransmittance * suffixTransmittance * lightGeometry.geometricTerm *
                                    scalarWeightWithoutTauAndGeometric * invSpp;
                            if (outputRecordCount < recordsPerEvent) {
                                SurfelGradientRecord gradientRecord{};
                                gradientRecord.primitiveIndex = occluder.primitiveIndex;
                                gradientRecord.gradPositionX = 0.0f;
                                gradientRecord.gradPositionY = 0.0f;
                                gradientRecord.gradPositionZ = 0.0f;
                                gradientRecord.gradScaleU = 0.0f;
                                gradientRecord.gradScaleV = 0.0f;
                                gradientRecord.gradRotationX = 0.0f;
                                gradientRecord.gradRotationY = 0.0f;
                                gradientRecord.gradRotationZ = 0.0f;
                                gradientRecord.gradEta = visibilityDerivativeScale * occluder.gradEta;
                                gradientRecord.gradBeta = 0.0f;
                                gradientRecord.gradAlbedoR = 0.0f;
                                gradientRecord.gradAlbedoG = 0.0f;
                                gradientRecord.gradAlbedoB = 0.0f;
                                //gradientRecord.gradEta = 0.0f;
                                gradientRecords[eventRecordBase + outputRecordCount] = gradientRecord;
                                accumulateDebugGradientIfSelected(debugImage, settings.renderDebugGradientImages,
                                                                  settings.surfelIndexForDebugImages, xSurface.pathId,
                                                                  gradientRecord);
                                ++outputRecordCount;
                            }
                            suffixTransmittance *= occluder.oneMinusAlpha;
                        }
                    }
                });
        }).wait();
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

    void launchSurfaceRegularizersBackwardKernel(RenderPackage &pkg, uint32_t cameraIndex) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        SensorGPU sensor = pkg.sensors[cameraIndex];
        DebugImages debugImage{};
        const bool writeDebugImages =
                settings.renderDebugGradientImages && pkg.debugImages != nullptr;

        if (writeDebugImages) {
            debugImage = pkg.debugImages[cameraIndex];
        }

        const uint32_t imageWidth = sensor.camera.width;
        const uint32_t imageHeight = sensor.camera.height;
        const uint32_t pixelCount = imageWidth * imageHeight;

        PointGradients depthGradients = pkg.depthDistortionGradients;
        PointGradients normalGradients = pkg.normalConsistencyGradients;
        PointGradients visibilityGradients = pkg.visibilityOpacityGradients;

        const uint32_t pointCount = static_cast<uint32_t>(depthGradients.numPoints);
        const float depthDistortionLossWeight = settings.depthDistortionWeight;
        const float normalConsistencyLossWeight = settings.normalConsistencyWeight;
        const float visibilityOpacityLossWeight = settings.visibilityWeightedOpacityRegularizerWeight;

        const bool enableDepthDistortionRegularizer = depthDistortionLossWeight != 0.0f;
        const bool enableNormalConsistencyRegularizer = normalConsistencyLossWeight != 0.0f;
        const bool enableVisibilityOpacityRegularizer = visibilityOpacityLossWeight != 0.0f;

        const float visibilityOpacityLossNormalization = 1.0f / static_cast<float>(pixelCount);

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class SurfaceRegularizersBackwardKernel>(
                sycl::range<1>(pixelCount),
                [=](sycl::id<1> tid) {
                    constexpr uint32_t kMaxHits = kMaxSplatEventsPerRay;
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
                            enableDepthDistortionRegularizer &&
                            sycl::fabs(depthDistortionAdjoint) > 1.0e-12f;

                    const bool useVisibleNormal =
                            sycl::fabs(visibleNormalAdjoint.x()) > 1.0e-12f ||
                            sycl::fabs(visibleNormalAdjoint.y()) > 1.0e-12f ||
                            sycl::fabs(visibleNormalAdjoint.z()) > 1.0e-12f;
                    const bool useMedianDepth = sycl::fabs(medianDepthAdjoint) > 1.0e-12f;
                    const bool useNormalConsistency =
                            enableNormalConsistencyRegularizer &&
                            (useVisibleNormal || useMedianDepth);

                    const bool useVisibilityOpacity = enableVisibilityOpacityRegularizer;

                    if (!useDepthDistortion && !useNormalConsistency && !useVisibilityOpacity) {
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
                        const InstanceRecord &instance = scene.instances[worldHit.instanceIndex];

                        if (instance.geometryType == GeometryType::Mesh) {
                            break;
                        }

                        if (instance.geometryType != GeometryType::PointCloud) {
                            break;
                        }

                        const uint32_t primitiveIndex = worldHit.primitiveIndex;
                        if (primitiveIndex == kInvalidIndex || primitiveIndex >= pointCount) {
                            break;
                        }

                        const Point &surfel = scene.points[primitiveIndex];
                        const float2 uv = phiInverse(worldHit.hitPositionW, surfel);
                        const float alphaGeom = worldHit.alphaGeom;
                        const float alphaEff = surfel.opacity * alphaGeom;
                        const float wi = transmittance * alphaEff;
                        const float zi = dot(worldHit.hitPositionW - sensor.camera.pos, sensor.camera.forward);

                        DistortionHit hit{};
                        hit.primitiveIndex = primitiveIndex;
                        hit.hitPositionW = worldHit.hitPositionW;
                        hit.rayOrigin0 = rayOrigin0;
                        hit.rayDir0 = rayDir0;
                        hit.ai = alphaEff;
                        hit.wi = wi;
                        hit.Tprev = transmittance;
                        hit.zi = zi;
                        hit.alphaGeom = alphaGeom;
                        hit.u = uv.x();
                        hit.v = uv.y();
                        hits[hitCount++] = hit;

                        transmittance *= sycl::fmax(0.0f, 1.0f - alphaEff);
                        primaryRay.origin = worldHit.hitPositionW + primaryRay.direction * RayEpsilon;
                    }

                    if (hitCount == 0u) {
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

                    constexpr bool detachDepthDistortionWeights = false;

                    if (useDepthDistortion && hitCount > 1u) {
                        for (uint32_t i = 0u; i < hitCount; ++i) {
                            for (uint32_t j = i + 1u; j < hitCount; ++j) {
                                const float mi = depthDistortionNdc01(hits[i].zi);
                                const float mj = depthDistortionNdc01(hits[j].zi);
                                const float wi = hits[i].wi;
                                const float wj = hits[j].wi;
                                const float depthDifference = mi - mj;
                                const float depthDifferenceSquared = depthDifference * depthDifference;

                                if constexpr (!detachDepthDistortionWeights) {
                                    barW[i] += depthDistortionAdjoint * wj * depthDifferenceSquared;
                                    barW[j] += depthDistortionAdjoint * wi * depthDifferenceSquared;
                                }

                                const float depthAdjointScale = 2.0f * depthDistortionAdjoint * wi * wj;
                                barM[i] += depthAdjointScale * depthDifference;
                                barM[j] -= depthAdjointScale * depthDifference;
                            }
                        }

                        for (uint32_t i = 0u; i < hitCount; ++i) {
                            barZ[i] += barM[i] * depthDistortionDndc01Ddepth(hits[i].zi);
                        }

                        if constexpr (!detachDepthDistortionWeights) {
                            float barTnext = 0.0f;
                            for (int i = static_cast<int>(hitCount) - 1; i >= 0; --i) {
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
                        }
                    }

                    uint32_t medianHitIndex = kInvalidIndex;
                    if (useNormalConsistency) {
                        float accumulatedCompositeWeight = 0.0f;
                        for (uint32_t i = 0u; i < hitCount; ++i) {
                            if ((accumulatedCompositeWeight + hits[i].wi) >= 0.5f) {
                                medianHitIndex = i;
                                break;
                            }
                            accumulatedCompositeWeight += hits[i].wi;
                        }
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

                        float3 depthGradPosition{0.0f, 0.0f, 0.0f};
                        float3 depthGradTanU{0.0f, 0.0f, 0.0f};
                        float3 depthGradTanV{0.0f, 0.0f, 0.0f};
                        float depthGradScaleU = 0.0f;
                        float depthGradScaleV = 0.0f;
                        float depthGradOpacity = 0.0f;
                        float depthGradBeta = 0.0f;

                        float3 normalGradPosition{0.0f, 0.0f, 0.0f};
                        float3 normalGradTanU{0.0f, 0.0f, 0.0f};
                        float3 normalGradTanV{0.0f, 0.0f, 0.0f};
                        float normalGradScaleU = 0.0f;
                        float normalGradScaleV = 0.0f;
                        float normalGradOpacity = 0.0f;
                        float normalGradBeta = 0.0f;

                        float visibilityGradOpacity = 0.0f;

                        if (useDepthDistortion && hitCount > 1u) {
                            const float3 x = hit.hitPositionW;
                            const float3 q = x - p;

                            const AlphaKernelEval kernelEval = evaluateAlphaKernelAndDerivatives(surfel, hit.u, hit.v);

                            constexpr bool detachOpacityInDepthDistortion = false;
                            const float barAlphaGeom = barA[i] * eta;
                            // Opacity still affects forward compositing weights, but the depth
                            // distortion loss does not update opacity.
                            const float barEta = detachOpacityInDepthDistortion
                                                     ? 0.0f
                                                     : barA[i] * hit.alphaGeom;

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

                            depthGradPosition += barP;
                            depthGradTanU += barTu;
                            depthGradTanV += barTv;
                            depthGradScaleU += barSu;
                            depthGradScaleV += barSv;
                            depthGradOpacity += barEta;
                            depthGradBeta += barBeta;
                        }

                        if (useVisibilityOpacity) {
                            const float visibilityOpacityAdjoint =
                                    visibilityOpacityLossWeight * visibilityOpacityLossNormalization;
                            visibilityGradOpacity += visibilityOpacityAdjoint * 2.0f * hit.wi * (eta - 1.0f);
                        }

                        if (useNormalConsistency && i == medianHitIndex) {
                            float3 orientedNormal = normalize(cross(tu, tv));
                            const bool hitBackside = dot(orientedNormal, -rayDir0) < 0.0f;
                            if (hitBackside) {
                                orientedNormal = -orientedNormal;
                            }

                            if (useVisibleNormal) {
                                const float3 rawCross = cross(tu, tv);
                                const float rawCrossLen = length(rawCross);
                                if (rawCrossLen > kDenomEps) {
                                    const float3 rawNormal = rawCross / rawCrossLen;
                                    const float orientationSign = dot(rawNormal, orientedNormal) >= 0.0f ? 1.0f : -1.0f;
                                    const float3 gradRawNormal = orientationSign * visibleNormalAdjoint;
                                    const float3 gradProjected = gradRawNormal - rawNormal * dot(
                                                                     rawNormal, gradRawNormal);
                                    const float3 gradCross = gradProjected / rawCrossLen;

                                    normalGradTanU += cross(tv, gradCross);
                                    normalGradTanV += cross(gradCross, tu);
                                }
                            }

                            if (useMedianDepth) {
                                const float3 gradWrtHitPoint = medianDepthAdjoint * sensor.camera.forward;
                                const float3x3 hitPointJacobian =
                                        planeHitPointIntersectionJacobian(rayDir0, orientedNormal);

                                normalGradPosition += transpose(hitPointJacobian) * gradWrtHitPoint;

                                const float3 rawCross = cross(tu, tv);
                                const float rawCrossLen = length(rawCross);
                                const float nDotD = dot(orientedNormal, rayDir0);

                                if (rawCrossLen > kDenomEps && sycl::fabs(nDotD) > kDenomEps) {
                                    const float3 rawNormal = rawCross / rawCrossLen;
                                    const float orientationSign = dot(rawNormal, orientedNormal) >= 0.0f ? 1.0f : -1.0f;
                                    const float3 pMinusX = p - hit.hitPositionW;
                                    const float3 gradOrientedNormal =
                                            pMinusX * (dot(rayDir0, gradWrtHitPoint) / nDotD);
                                    const float3 gradRawNormal = orientationSign * gradOrientedNormal;
                                    const float3 gradProjected = gradRawNormal - rawNormal * dot(
                                                                     rawNormal, gradRawNormal);
                                    const float3 gradCross = gradProjected / rawCrossLen;

                                    normalGradTanU += cross(tv, gradCross);
                                    normalGradTanV += cross(gradCross, tu);
                                }
                            }
                        }

                        if (useDepthDistortion) {
                            atomicAddFloat3(depthGradients.gradPosition[hit.primitiveIndex], depthGradPosition);
                            const float3 depthGradRotation = computeLocalRotationGradientFromTangentGradients(
                                tu, tv, depthGradTanU, depthGradTanV);
                            atomicAddFloat3(depthGradients.gradRotation[hit.primitiveIndex], depthGradRotation);
                            atomicAddFloat2(depthGradients.gradScale[hit.primitiveIndex],
                                            float2{depthGradScaleU, depthGradScaleV});
                            atomicAddFloat(depthGradients.gradOpacity[hit.primitiveIndex], depthGradOpacity);
                            atomicAddFloat(depthGradients.gradBeta[hit.primitiveIndex], depthGradBeta);
                        }

                        if (useNormalConsistency) {
                            atomicAddFloat3(normalGradients.gradPosition[hit.primitiveIndex], normalGradPosition);
                            const float3 normalGradRotation =
                                    computeLocalRotationGradientFromTangentGradients(
                                        tu, tv, normalGradTanU, normalGradTanV);
                            atomicAddFloat3(normalGradients.gradRotation[hit.primitiveIndex], normalGradRotation);
                            atomicAddFloat2(normalGradients.gradScale[hit.primitiveIndex],
                                            float2{normalGradScaleU, normalGradScaleV});
                            atomicAddFloat(normalGradients.gradOpacity[hit.primitiveIndex], normalGradOpacity);
                            atomicAddFloat(normalGradients.gradBeta[hit.primitiveIndex], normalGradBeta);
                        }

                        if (useVisibilityOpacity) {
                            atomicAddFloat(visibilityGradients.gradOpacity[hit.primitiveIndex], visibilityGradOpacity);
                        }

                        if (writeDebugImages) {
                            const float3 totalGradPosition = depthGradPosition + normalGradPosition;
                            const float3 totalGradRotation = computeLocalRotationGradientFromTangentGradients(
                                tu, tv, depthGradTanU + normalGradTanU, depthGradTanV + normalGradTanV);
                            const float totalGradScaleU = depthGradScaleU + normalGradScaleU;
                            const float totalGradScaleV = depthGradScaleV + normalGradScaleV;
                            const float totalGradOpacity = depthGradOpacity + normalGradOpacity + visibilityGradOpacity;
                            const float totalGradBeta = depthGradBeta + normalGradBeta;

                            SurfelGradientRecord debugRecord{};
                            debugRecord.primitiveIndex = hit.primitiveIndex;
                            debugRecord.gradPositionX = totalGradPosition.x();
                            debugRecord.gradPositionY = totalGradPosition.y();
                            debugRecord.gradPositionZ = totalGradPosition.z();
                            debugRecord.gradScaleU = totalGradScaleU;
                            debugRecord.gradScaleV = totalGradScaleV;
                            debugRecord.gradRotationX = totalGradRotation.x();
                            debugRecord.gradRotationY = totalGradRotation.y();
                            debugRecord.gradRotationZ = totalGradRotation.z();
                            debugRecord.gradEta = totalGradOpacity;
                            debugRecord.gradBeta = totalGradBeta;
                            debugRecord.gradAlbedoR = 0.0f;
                            debugRecord.gradAlbedoG = 0.0f;
                            debugRecord.gradAlbedoB = 0.0f;

                            accumulateDebugGradientIfSelected(
                                debugImage,
                                settings.renderDebugGradientImages,
                                settings.surfelIndexForDebugImages,
                                pixelIndex,
                                debugRecord);
                        }
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
        const GradientRecordRanges ranges = makeGradientRecordRanges(
            safeMeasurementEventCount,
            safeMeasurementTwoPointEventCount,
            safeMaterialVertexEventCount,
            safeMaterialEndEdgeEventCount,
            safeMaterialStartEdgeEventCount);
        if (ranges.totalCount > pkg.intermediates.maxGradientRecordCount) {
            throw std::runtime_error("gradient record scratch buffer too small");
        }
        if (safeMeasurementEventCount > 0u) {
            ScopedTimer timer("measurementGradientEvent", spdlog::level::debug);
            measurementGradientEvent(pkg, cameraIndex, safeMeasurementEventCount, ranges.measurementOffset);
        }
        if (safeMeasurementTwoPointEventCount > 0u) {
            ScopedTimer timer("measurementGradientEventXY", spdlog::level::debug);
            Log::PA_DEBUG("measurementTwoPointEventCount={} max={}", measurementTwoPointEventCount,
                          pkg.intermediates.maxMeasurementTwoPointEventCount);
            measurementGradientEventXY(pkg, safeMeasurementTwoPointEventCount, ranges.measurementTwoPointOffset,
                                       cameraIndex);
        }
        //if (safeMaterialVertexEventCount > 0u) {
        //    ScopedTimer timer("materialVertexGradientEvent", spdlog::level::debug);
        //    materialVertexGradientEvent(pkg, safeMaterialVertexEventCount, ranges.materialVertexOffset, cameraIndex);
        //}
        //if (safeMaterialEndEdgeEventCount > 0u) {
        //    ScopedTimer timer("materialEndEdgeGradientEvent", spdlog::level::debug);
        //    materialEndEdgeGradientEvent(pkg, safeMaterialEndEdgeEventCount, ranges.materialEndEdgeOffset, cameraIndex);
        //}
        //if (safeMaterialStartEdgeEventCount > 0u) {
        //    ScopedTimer timer("materialStartEdgeGradientEvent", spdlog::level::debug);
        //    materialStartEdgeGradientEvent(pkg, safeMaterialStartEdgeEventCount, ranges.materialStartEdgeOffset,
        //                                   cameraIndex);
        //}
        if (ranges.totalCount > 0u) {
            ScopedTimer timer("reduceSurfelGradientRecords", spdlog::level::debug);
            const uint32_t cameraSlotIndex = pkg.sensors[cameraIndex].cameraSlotIndex;
            const uint32_t cameraSlotCount = static_cast<uint32_t>(pkg.gradients.cameraSlotCount);
            reduceSurfelGradientRecords(pkg, ranges.totalCount, cameraSlotIndex, cameraSlotCount);
        }
    }
}
