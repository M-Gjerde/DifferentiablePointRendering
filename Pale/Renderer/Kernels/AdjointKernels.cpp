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

                    RayState nextState{};
                    bool shouldEnqueueNextState = false;

                    constexpr uint32_t maxInlineNullTraversals = 256u;

                    for (uint32_t inlineTraversalIndex = 0u;
                         inlineTraversalIndex < maxInlineNullTraversals;
                         ++inlineTraversalIndex) {
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

                            nextState.ray.origin = worldHit.hitPositionW + (orientedNormal * 1e-6f);
                            nextState.ray.direction = sampledOutgoingDirectionWorld;
                            nextState.ray.normal = orientedNormal;
                            nextState.bounceIndex = currentRayState.bounceIndex + 1u;
                            nextState.pixelIndex = currentRayState.pixelIndex;
                            nextState.pathId = currentRayState.pathId;
                            nextState.pathThroughput = currentRayState.pathThroughput * throughputMultiplier;
                            nextState.traversalIndex = currentRayState.traversalIndex + 1u;
                            nextState.transmission = 1.0f;

                            clearPendingCameraSegment(currentPendingCameraSegment);
                            clearPendingAdjointStageX(currentPendingStageX);
                            clearPendingAdjointStageXY(currentPendingStageXY);

                            if (applyRussianRoulette(
                                rng,
                                nextState.bounceIndex,
                                nextState.pathThroughput,
                                settings.russianRouletteStart)) {
                                shouldEnqueueNextState = true;
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
                            const float3 surfelBrdf = surfel.alpha_r * surfel.albedo * M_1_PIf;
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

                                if (previousPendingCameraSegment.valid) {
                                    AttachedGradientProjectionEvent attachedProjectionEvent{};
                                    attachedProjectionEvent.xSurface = currentSurfaceRecord;
                                    attachedProjectionEvent.transmission = currentRayState.transmission;
                                    attachedProjectionEvent.xPathThroughput =
                                            currentRayState.pathThroughput / qReflect;

                                    appendEventAtomic(
                                        intermediates.countProjectionEvents,
                                        intermediates.projectionEvents,
                                        intermediates.maxProjectionEventCount,
                                        attachedProjectionEvent);
                                }

                                if (previousPendingStageX.valid) {
                                    AttachedGradientScatterEvent attachedScatterEvent{};
                                    attachedScatterEvent.xSurface = previousPendingStageX.xSurface;
                                    attachedScatterEvent.ySurface = currentSurfaceRecord;
                                    attachedScatterEvent.xPathThroughput =
                                            previousPendingStageX.xPathThroughput / qReflect;
                                    attachedScatterEvent.transmissionPreviousSegment =
                                            previousPendingStageX.previousSegmentTransmission;
                                    attachedScatterEvent.transmission =
                                            currentRayState.transmission;

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
                                    detachedThreePointEvent.zSurface =
                                            currentSurfaceRecord;
                                    detachedThreePointEvent.transmission =
                                            currentRayState.transmission;
                                    detachedThreePointEvent.xPathThroughput =
                                            previousPendingStageXY.xPathThroughput / qReflect;;

                                    appendEventAtomic(
                                        intermediates.countReflectScatterEvents,
                                        intermediates.reflectScatterEvents,
                                        intermediates.maxReflectScatterEventCount,
                                        detachedThreePointEvent);
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

                            nextState.ray.origin = worldHit.hitPositionW + (orientedNormal * 1e-5f);
                            nextState.ray.direction = sampledOutgoingDirectionWorld;
                            nextState.ray.normal = orientedNormal;
                            nextState.bounceIndex = currentRayState.bounceIndex + 1u;
                            nextState.pixelIndex = currentRayState.pixelIndex;
                            nextState.pathId = currentRayState.pathId;
                            nextState.pathThroughput =
                                    currentRayState.pathThroughput * throughputMultiplier;
                            nextState.traversalIndex = currentRayState.traversalIndex + 1u;
                            nextState.transmission = 1.0f;

                            if (applyRussianRoulette(
                                rng,
                                nextState.bounceIndex,
                                nextState.pathThroughput,
                                settings.russianRouletteStart)) {
                                shouldEnqueueNextState = true;
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
        uint32_t cameraIndex,
        uint32_t projectionEventCount,
        uint32_t baseOffset) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        const auto &photonMap = pkg.intermediates.map;
        auto &sensor = pkg.sensors[cameraIndex];
        AttachedGradientProjectionEvent *projectionEvents = pkg.intermediates.projectionEvents;
        SurfelGradientRecord *gradientRecords = pkg.intermediates.gradientRecords;

        const float invSpp = 1.0f / settings.adjointSamplesPerPixel;
        const float qNullInv = 1.0f / settings.sampling.qNull;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class launchProjectionContributionKernelTag>(
                sycl::range<1>(projectionEventCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t eventIndex = globalId[0];
                    const uint32_t recordIndex = baseOffset + kMaxSplatEventsPerRay * eventIndex + 0u;

                    const AttachedGradientProjectionEvent eventRecord = projectionEvents[eventIndex];

                    const Point &surfelX = scene.points[eventRecord.xSurface.primitiveIndex];
                    const ReconstructedSurfelState xState =
                            reconstructSurfelState(surfelX, eventRecord.xSurface);

                    const float3 irradianceAtX = gatherDiffuseIrradianceAtPoint(
                        xState.position,
                        xState.orientedNormal,
                        photonMap);

                    // L_surfel(X, omega_x->c), without alpha_X
                    const float3 outgoingRadianceX =
                            surfelX.alpha_r * surfelX.albedo * M_1_PIf * irradianceAtX;

                    const float3 vectorCToX = xState.position - sensor.camera.pos;
                    const float distanceSquared = dot(vectorCToX, vectorCToX);
                    if (distanceSquared <= 1e-12f) {
                        return;
                    }

                    const float distance = sycl::sqrt(distanceSquared);
                    const float3 directionCToX = vectorCToX / distance;

                    const float cosineAtX = dot(xState.orientedNormal, -directionCToX);
                    if (cosineAtX <= 1e-6f) {
                        return;
                    }

                    float cosThetaCamera = dot(sensor.camera.forward, directionCToX);
                    if (cosThetaCamera <= 1e-6f) {
                        return;
                    }

                    const float width = float(sensor.width);
                    const float height = float(sensor.height);
                    const float fovYRad = glm::radians(sensor.camera.fovy);
                    const float tanHalfFovY = sycl::tan(0.5f * fovYRad);
                    const float tanHalfFovX = tanHalfFovY * (width / height);

                    // Film plane at z = 1
                    const float filmWidth = 2.0f * tanHalfFovX;
                    const float filmHeight = 2.0f * tanHalfFovY;
                    const float pixelArea = (filmWidth / width) * (filmHeight / height);
                    const float invPixelArea = 1.0f / pixelArea;

                    // p_omega(omega) for uniform sampling inside one pixel on the film plane
                    const float invPixelSolidAngle =
                            invPixelArea / (cosThetaCamera * cosThetaCamera * cosThetaCamera);

                    // Induced area PDF at X
                    const float pAreaX = invPixelSolidAngle * cosineAtX / distanceSquared;
                    if (pAreaX <= 1e-20f) {
                        return;
                    }
                    const float invPAreaX = 1.0f / pAreaX;
                    // dG(c, X) / dX
                    const float3 dGeometricTermDx = computeGeometricTermGradientWrtEndpoint(
                        sensor.camera.pos,
                        xState.position,
                        sensor.camera.forward,
                        xState.orientedNormal);

                    // dG(c, X) / dX
                    const float geometricTermCamera = computeGeometricTermValue(
                        sensor.camera.pos,
                        xState.position,
                        sensor.camera.forward,
                        xState.orientedNormal);

                    const float targetDistance = distance;
                    const float distanceEpsilon = 1e-4f;

                    float transmittance = 1.0f;
                    float3 accumulatedAlphaDerivativeOverOneMinusAlphaEndpoint = float3{0.0f};

                    struct OccluderDerivatives {
                        float3 derivative{0.0f};
                        uint32_t primitiveIndex = UINT32_MAX;
                    };

                    OccluderDerivatives occluderDerivatives[kMaxSplatEventsPerRay];
                    uint32_t storedOccluderCount = 0;

                    float qNullInvTotal = 1.0f;
                    if (eventRecord.transmission != 1.0f) {
                        const float3 dir = normalize(vectorCToX);
                        Ray ray = {sensor.camera.pos, dir};
                        const float3 origin = ray.origin;
                        ray.origin = sensor.camera.pos + dir * distanceEpsilon;

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

                            const float hitDistance = length(worldHit.hitPositionW - sensor.camera.pos);
                            if (hitDistance >= targetDistance - distanceEpsilon) {
                                break;
                            }

                            buildIntersectionNormal(scene, worldHit);
                            auto &instance = scene.instances[worldHit.instanceIndex];
                            if (instance.geometryType != GeometryType::PointCloud) {
                                break;
                            }

                            const Point &occluderSurfel = scene.points[worldHit.primitiveIndex];
                            float3 occluderNormal =
                                    normalize(cross(occluderSurfel.tanU, occluderSurfel.tanV));

                            bool hitBackside = dot(occluderNormal, -ray.direction) < 0.0f;
                            if (hitBackside) {
                                occluderNormal = -occluderNormal;
                            }

                            const float alphaEff = occluderSurfel.opacity * worldHit.alphaGeom;
                            const float oneMinusAlpha = 1.0f - alphaEff;
                            if (oneMinusAlpha <= 1e-8f) {
                                break;
                            }

                            transmittance *= oneMinusAlpha;
                            ray.origin = worldHit.hitPositionW + (ray.direction * 1e-4f);

                            const float2 uv = phiInverse(worldHit.hitPositionW, occluderSurfel);
                            const float uOcc = uv.x();
                            const float vOcc = uv.y();

                            const float3 dxy = origin - xState.position;
                            const float alphaGeomOccluder = worldHit.alphaGeom;
                            const float denominator = dot(occluderNormal, dxy);
                            if (sycl::fabs(denominator) <= 1e-8f) {
                                continue;
                            }

                            const float lambdaOccluder =
                                    dot(occluderNormal, occluderSurfel.position - xState.position) / denominator;

                            const float su_i = occluderSurfel.scale.x();
                            const float sv_i = occluderSurfel.scale.y();
                            if (su_i <= 1e-12f || sv_i <= 1e-12f) {
                                continue;
                            }

                            const float3 tanU_i = occluderSurfel.tanU;
                            const float3 tanV_i = occluderSurfel.tanV;
                            const float3 localBasisU = tanU_i / su_i;
                            const float3 localBasisV = tanV_i / sv_i;
                            const float inverseDenominator = 1.0f / denominator;

                            const float3 dUiDx =
                                    (1.0f - lambdaOccluder) * (
                                        localBasisU -
                                        occluderNormal * (dot(dxy, localBasisU) * inverseDenominator));

                            const float3 dViDx =
                                    (1.0f - lambdaOccluder) * (
                                        localBasisV -
                                        occluderNormal * (dot(dxy, localBasisV) * inverseDenominator));

                            const float3 dUiDspi =
                                    occluderNormal * (dot(dxy, tanU_i)) / su_i * inverseDenominator - localBasisU;

                            const float3 dViDspi =
                                    occluderNormal * (dot(dxy, tanV_i)) / sv_i * inverseDenominator - localBasisV;

                            const float radiusSquaredOcc = uOcc * uOcc + vOcc * vOcc;
                            const float oneMinusRadiusSquaredOcc = 1.0f - radiusSquaredOcc;
                            if (oneMinusRadiusSquaredOcc <= 1e-8f) {
                                continue;
                            }

                            const float betaScaleOcc = 4.0f * sycl::exp(occluderSurfel.beta);
                            const float dAlphaGeomDu =
                                    -2.0f * betaScaleOcc * uOcc * alphaGeomOccluder / oneMinusRadiusSquaredOcc;
                            const float dAlphaGeomDv =
                                    -2.0f * betaScaleOcc * vOcc * alphaGeomOccluder / oneMinusRadiusSquaredOcc;

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

                    const float u = eventRecord.xSurface.uv.x();
                    const float v = eventRecord.xSurface.uv.y();
                    const float radiusSquared = u * u + v * v;
                    const float oneMinusRadiusSquared = 1.0f - radiusSquared;
                    if (oneMinusRadiusSquared <= 1e-8f) {
                        return;
                    }

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
                            (-2.0f * betaScale * eventRecord.xSurface.alphaGeom) / oneMinusRadiusSquared;

                    const float3 dAlphaGeomDPosition = factor * dUvDPosition;
                    const float3 dAlphaEffDPosition = surfelX.opacity * dAlphaGeomDPosition;

                    const float scalarWeight = dot(pathWeight, outgoingRadianceX);
                    const float alphaX = eventRecord.xSurface.alphaGeom * surfelX.opacity;

                    // ---------------------------------------------------------------------
                    // Keep term 2 local alpha derivative:
                    //   (tau(c, X) * G(c, X) / p_A(X)) * d alpha_X * L_surfel(X, omega_x->c)
                    // ---------------------------------------------------------------------
                    const float3 positionGradient =
                            (transmittance) *
                            dAlphaEffDPosition * scalarWeight * invSpp;

                    SurfelGradientRecord gradientRecord = {};
                    gradientRecord.primitiveIndex = eventRecord.xSurface.primitiveIndex;
                    gradientRecord.gradPositionX = positionGradient.x();
                    gradientRecord.gradPositionY = positionGradient.y();
                    gradientRecord.gradPositionZ = positionGradient.z();

                    // ---------------------------------------------------------------------
                    // Term 1 endpoint contribution:
                    //   alpha_X * L_surfel(X, omega_x->c) * (G * d tau + tau * dG) / p_A(X)
                    // ---------------------------------------------------------------------
                    const float3 dTransmittanceDx =
                            -transmittance * accumulatedAlphaDerivativeOverOneMinusAlphaEndpoint;

                    float3 gradientWrtHitPositionX =
                            alphaX * scalarWeight *
                            (geometricTermCamera * dTransmittanceDx) * invSpp;

                    const float3x3 hitPointJacobian = planeHitPointIntersectionJacobian(
                        eventRecord.xSurface.incomingDirection,
                        xState.orientedNormal);

                    gradientWrtHitPositionX = transpose(hitPointJacobian) * gradientWrtHitPositionX;

                    gradientRecord.gradPositionX += gradientWrtHitPositionX.x();
                    gradientRecord.gradPositionY += gradientWrtHitPositionX.y();
                    gradientRecord.gradPositionZ += gradientWrtHitPositionX.z();
                    gradientRecords[recordIndex] = gradientRecord;

                    // ---------------------------------------------------------------------
                    // Term 1 intermediate occluder contributions on the camera segment:
                    //   alpha_X * L_surfel(X, omega_x->c) * G(c, X) * d tau / p_A(X)
                    // ---------------------------------------------------------------------

                    const float occluderScale =
                            -transmittance *
                            alphaX * scalarWeight * invSpp;

                    for (uint32_t occluderIndex = 0; occluderIndex < storedOccluderCount; occluderIndex++) {
                        const uint32_t occluderRecordIndex =
                                baseOffset + kMaxSplatEventsPerRay * eventIndex + 1u + occluderIndex;

                        const OccluderDerivatives &occluderDerivative =
                                occluderDerivatives[occluderIndex];

                        SurfelGradientRecord gradientRecordOccluders = {};
                        gradientRecordOccluders.primitiveIndex = occluderDerivative.primitiveIndex;

                        const float3 occluderContribution =
                                occluderScale * occluderDerivative.derivative;

                        gradientRecordOccluders.gradPositionX = occluderContribution.x();
                        gradientRecordOccluders.gradPositionY = occluderContribution.y();
                        gradientRecordOccluders.gradPositionZ = occluderContribution.z();

                        gradientRecords[occluderRecordIndex] = gradientRecordOccluders;
                    }
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
        const float qNullInv = 1.0f / settings.sampling.qNull;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<class launchProjectionScatterContributionKernelTag>(
                sycl::range<1>(projectionScatterEventCount),
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

                    const float3 pathWeight =
                            eventRecord.xPathThroughput * eventRecord.transmissionPreviousSegment;


                    const float3 outgoingTransportWithoutTauAndGeometric =
                            outgoingRadianceY * alphaX * brdfX;

                    float scalarWeightWithoutTauAndGeometric =
                            dot(pathWeight, outgoingTransportWithoutTauAndGeometric) / pAreaY;

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

                        float3 vectorXToY = yState.position - xState.position;
                        float3 rayDirection = normalize(vectorXToY);
                        Ray ray = {xState.position, rayDirection};
                        ray.origin = xState.position + rayDirection * distanceEpsilon;

                        const float targetDistance = length(xState.position - yState.position);
                        const float3 xPosition = xState.position;
                        const float3 yPosition = yState.position;
                        const float3 dxy = xPosition - yPosition;

                        while (true) {
                            WorldHit worldHit{};
                            intersectScene(ray, &worldHit, scene, rng::Xorshift128(0.0),
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
                            float3 occluderNormal = normalize(cross(occluderSurfel.tanU, occluderSurfel.tanV));
                            const bool hitBackside = dot(occluderNormal, -ray.direction) < 0.0f;
                            if (hitBackside) {
                                occluderNormal = -occluderNormal;
                            }

                            const float alphaEff = occluderSurfel.opacity * worldHit.alphaGeom;
                            const float oneMinusAlpha = 1.0f - alphaEff;
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

                            const float3 tanU_i = occluderSurfel.tanU;
                            const float3 tanV_i = occluderSurfel.tanV;
                            const float3 localBasisU = tanU_i / scaleU;
                            const float3 localBasisV = tanV_i / scaleV;
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
                                        localBasisU
                                        - occluderNormal * (dot(dxy, localBasisU) * inverseDenominator));

                            const float3 dUiDy =
                                    (1.0f - lambdaOccluder) * (
                                        localBasisU
                                        - occluderNormal * (dot(dxy, localBasisU) * inverseDenominator));

                            const float3 dViDx =
                                    lambdaOccluder * (
                                        localBasisV
                                        - occluderNormal * (dot(dxy, localBasisV) * inverseDenominator));

                            const float3 dViDy =
                                    (1.0f - lambdaOccluder) * (
                                        localBasisV
                                        - occluderNormal * (dot(dxy, localBasisV) * inverseDenominator));

                            const float3 dUiDspi =
                                    occluderNormal * (dot(dxy, tanU_i) / scaleU) * inverseDenominator
                                    - localBasisU;

                            const float3 dViDspi =
                                    occluderNormal * (dot(dxy, tanV_i) / scaleV) * inverseDenominator
                                    - localBasisV;

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


                    const float3x3 hitPointJacobian = planeHitPointIntersectionJacobian(
                        eventRecord.xSurface.incomingDirection,
                        xState.orientedNormal);
                    gradientWrtHitPositionX = transpose(hitPointJacobian) * gradientWrtHitPositionX;

                    const float3 dGeometricTermDy = computeGeometricTermGradientWrtEndpoint(
                        xState.position,
                        yState.position,
                        xState.orientedNormal,
                        yState.orientedNormal);

                    const float3 dTransmittanceDy =
                            -transmittance * accumulatedAlphaDerivativeOverOneMinusAlphaEndPoint;

                    float3 gradientWrtHitPositionY =
                            scalarWeightWithoutTauAndGeometric *
                            (geometricTermXY * dTransmittanceDy + transmittance * dGeometricTermDy);


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


                    float occluderScale =
                            -transmittance * geometricTermXY * scalarWeightWithoutTauAndGeometric * invSpp;

                    for (uint32_t occluderIndex = 0; occluderIndex < storedOccluderCount; ++occluderIndex) {
                        const uint32_t occluderRecordIndex = eventRecordBase + 2u + occluderIndex;

                        const OccluderDerivative &occluderDerivative = occluderDerivatives[occluderIndex];
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
        const float qNullInv = 1.0f / settings.sampling.qNull;
        const float qReflectInv = 1.0f / settings.sampling.qReflect;

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

                    float transmittance = 1.0f;

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

                    const float3 pathWeight = eventRecord.xPathThroughput;

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
            "Event counts: projection={}, projectionScatter={}, reflectScatter={}",
            projectionEventCount,
            projectionScatterEventCount,
            reflectScatterEventCount);


        if (projectionEventCount > 0) {
            ScopedTimer timer("launchAttachedProjectionKernel", spdlog::level::debug);
            launchAttachedProjectionKernel(
                pkg,
                cameraIndex,
                projectionEventCount,
                ranges.projectionOffset);
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
