//
// Created by magnus on 9/8/25.
//

#include "Renderer/Kernels/AdjointKernels.h"

#include <cmath>

#include "AdjointGradientKernels.h"
#include "IntersectionKernels.h"
#include "Renderer/Kernels/KernelHelpers.h"


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
                });
        }).wait();
    }


    void launchAdjointIntersectKernel(RenderPackage &pkg, uint32_t spp, uint32_t activeRayCount, uint32_t bounceIndex) {
        auto &queue = pkg.queue;
        auto &settings = pkg.settings;
        auto &intermediates = pkg.intermediates;
        auto &scene = pkg.scene;

        queue.submit([&](sycl::handler &cgh) {
            uint64_t renderSeed = settings.random.seed;
            cgh.parallel_for<class launchIntersectKernel>(
                sycl::range<1>(activeRayCount),
                // ReSharper disable once CppDFAUnusedValue
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    RayState rayState = intermediates.primaryRays[rayIndex];

                    const uint64_t seed =
                            rng::makeSeed(renderSeed,
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

                    // -----------------------------------------------------------------
                    // 1) Complete pending (if any) using THIS hit as the endpoint.
                    // -----------------------------------------------------------------
                    float3 orientedNormal{0.0f};
                    float uniformDirectionPDF = 1.0f / (2.0f * M_PI);
                    float cosineDirectionPDF{0.0f};
                    float3 sampledOutgoingDirectionW{0.0f};
                    const InstanceRecord &endpointInstance = scene.instances[worldHit.instanceIndex];
                    if (endpointInstance.geometryType == GeometryType::Mesh) {
                        orientedNormal = worldHit.geometricNormalW;
                        if (dot(rayState.ray.direction, orientedNormal) > 0.0f) {
                            orientedNormal = -orientedNormal;
                        }

                        // Cosine sampling
                        sampleCosineHemisphere(rng, worldHit.geometricNormalW, sampledOutgoingDirectionW,
                                               cosineDirectionPDF);
                        orientedNormal = worldHit.geometricNormalW;
                    } else if (endpointInstance.geometryType == GeometryType::PointCloud) {
                        // Hemisphere sampling
                        const auto &surfel = scene.points[worldHit.primitiveIndex];
                        // Find which side we hit the surfel:
                        const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));
                        const float signedCosineIncident = dot(canonicalNormalW, -rayState.ray.direction);
                        const int sideSign = signNonZero(signedCosineIncident);
                        // If positive we hit the front side if negative we hit the backside
                        orientedNormal = static_cast<float>(sideSign) * canonicalNormalW;
                        // If we hit instance was a mesh do ordinary BRDF stuff.
                        sampleUniformHemisphereAroundNormal(rng, orientedNormal, sampledOutgoingDirectionW,
                                                            uniformDirectionPDF);
                    }


                    const uint32_t pathId = rayState.pathId;
                    if (pathId < intermediates.maxPendingAdjointStateCount) {
                        const GeometryType currentGeometryType = endpointInstance.geometryType;
                        if (bounceIndex == 0) {
                            float3 throughput = rayState.pathThroughput;
                            CompletedGradientEvent completedProjection = makeCompletedGradientEventX(
                                worldHit,
                                rayState,
                                throughput,
                                orientedNormal,
                                currentGeometryType);
                            completedProjection.kind = PendingAdjointKind::Projection;
                            appendCompletedGradientEventAtomic(
                                intermediates.countCompletedGradientEvents,
                                intermediates.completedGradientEvents,
                                intermediates.maxCompletedGradientEventCount,
                                completedProjection);
                        }
                        auto &pendingStageX = intermediates.pendingStageX[pathId];
                        auto &pendingStageXY = intermediates.pendingStageXY[pathId];

                        if (pendingStageXY.valid) {
                            CompletedGradientEvent completedReflect = makeCompletedGradientEventXYZ(
                                pendingStageXY,
                                worldHit,
                                rayState,
                                orientedNormal,
                                currentGeometryType);
                            completedReflect.kind = PendingAdjointKind::ReflectScatter;
                            appendCompletedGradientEventAtomic(
                                intermediates.countCompletedGradientEvents,
                                intermediates.completedGradientEvents,
                                intermediates.maxCompletedGradientEventCount,
                                completedReflect);
                            clearPendingAdjointStageXY(pendingStageXY);
                        }

                        if (pendingStageX.valid) {
                            CompletedGradientEvent completedProjection = makeCompletedGradientEventXY(
                                pendingStageX,
                                worldHit,
                                rayState,
                                pendingStageX.xPathThroughput,
                                orientedNormal,
                                currentGeometryType);
                            completedProjection.kind = PendingAdjointKind::ProjectionScatter;
                            appendCompletedGradientEventAtomic(
                                intermediates.countCompletedGradientEvents,
                                intermediates.completedGradientEvents,
                                intermediates.maxCompletedGradientEventCount,
                                completedProjection);

                            pendingStageXY = makePendingStageXY(
                                pendingStageX,
                                worldHit,
                                rayState,
                                orientedNormal,
                                currentGeometryType);

                            clearPendingAdjointStageX(pendingStageX);
                        }
                    }

                    // Hitting mesh events
                    const auto &instance = scene.instances[worldHit.instanceIndex];
                    if (instance.geometryType == GeometryType::Mesh) {
                        // determine if we should make contributions from this position:
                        // Generate next ray
                        const GPUMaterial material = scene.materials[instance.materialIndex];
                        // If we hit instance was a mesh do ordinary BRDF stuff.
                        const float3 lambertBrdf = material.baseColor;
                        float3 throughputMultiplier = lambertBrdf;
                        // alpha_r on meshes is just always 1.0 (different brdf)
                        RayState nextState{};
                        // Spawn next ray
                        nextState.ray.origin = worldHit.hitPositionW + (worldHit.geometricNormalW * 1e-6f);
                        nextState.ray.direction = sampledOutgoingDirectionW;
                        nextState.ray.normal = worldHit.geometricNormalW;
                        nextState.bounceIndex = rayState.bounceIndex + 1;
                        nextState.pixelIndex = rayState.pixelIndex;
                        nextState.pathId = rayState.pathId;
                        nextState.pathThroughput = rayState.pathThroughput * throughputMultiplier;
                        if (!applyRussianRoulette(rng, nextState.bounceIndex, nextState.pathThroughput,
                                                  settings.russianRouletteStart))
                            return;

                        // Scatter calculations
                        auto extensionCounter = sycl::atomic_ref<uint32_t,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                            *intermediates.countExtensionOut);
                        const uint32_t outIndex = extensionCounter.fetch_add(1);
                        intermediates.extensionRaysA[outIndex] = nextState;
                    } else {
                        // Random event
                        // qAbsorb = 1 - (qNull + qReflect + qTransmit)

                        const float u = rng.nextFloat();
                        if (u < settings.sampling.qNull) {
                            const Point &surfel = scene.points[worldHit.primitiveIndex];
                            float attenuation = 1.0f - worldHit.alphaGeom * surfel.opacity;
                            float weight = attenuation / settings.sampling.qNull;
                            // Spawn next ray
                            RayState nextState{};
                            nextState.ray.origin = worldHit.hitPositionW + (rayState.ray.direction * 1e-5f);
                            nextState.ray.direction = rayState.ray.direction;
                            nextState.ray.normal = worldHit.geometricNormalW;
                            nextState.bounceIndex = rayState.bounceIndex + 1;
                            nextState.pixelIndex = rayState.pixelIndex;
                            nextState.pathThroughput = rayState.pathThroughput * weight;
                            nextState.pathId = rayState.pathId;
                            if (!applyRussianRoulette(rng, nextState.bounceIndex, nextState.pathThroughput,
                                                      settings.russianRouletteStart))
                                return;
                            // Populate pending (now that the path survives)
                            /*
                            if (rayState.pathId < intermediates.maxPendingAdjointStateCount) {
                                PendingAdjointState pending{};
                                pending.kind = PendingAdjointKind::NullTransmittance;
                                pending.primitiveIndex = worldHit.primitiveIndex;
                                pending.alphaGeom = worldHit.alphaGeom;
                                pending.hitNormal = orientedNormal;
                                pending.hitPosition = worldHit.hitPositionW;
                                pending.pathThroughput = rayState.pathThroughput / settings.sampling.qNull;
                                pending.pixelIndex = rayState.pixelIndex;
                                intermediates.pendingAdjointStates[rayState.pathId] = pending;
                            }
                            */
                            auto extensionCounter = sycl::atomic_ref<uint32_t,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>(
                                *intermediates.countExtensionOut);
                            const uint32_t outIndex = extensionCounter.fetch_add(1);
                            intermediates.extensionRaysA[outIndex] = nextState;
                        } else if (u < settings.sampling.qNull + settings.sampling.qReflect) {
                            // Generate next ray
                            const auto &surfel = scene.points[worldHit.primitiveIndex];
                            const float3 f_s = surfel.alpha_r * surfel.albedo * M_1_PIf; // ρ/π
                            const float cosTheta = sycl::fmax(0.0f, dot(sampledOutgoingDirectionW, orientedNormal));
                            float alpha = worldHit.alphaGeom * surfel.opacity;
                            const float3 &throughputMultiplier = ((alpha / settings.sampling.qReflect) * (f_s *
                                                                      cosTheta)) / uniformDirectionPDF;
                            RayState nextState{};
                            // Spawn next ray
                            nextState.ray.origin = worldHit.hitPositionW + (orientedNormal * 1e-5f);
                            nextState.ray.direction = sampledOutgoingDirectionW;
                            nextState.ray.normal = orientedNormal;
                            nextState.bounceIndex = rayState.bounceIndex + 1;
                            nextState.pixelIndex = rayState.pixelIndex;
                            nextState.pathId = rayState.pathId;
                            nextState.pathThroughput =
                                    rayState.pathThroughput * throughputMultiplier;
                            if (!applyRussianRoulette(rng, nextState.bounceIndex, nextState.pathThroughput,
                                                      settings.russianRouletteStart))
                                return;

                            // Projection kernels

                            /*
                            if (bounceIndex == 0 && rayState.pathId < intermediates.maxPendingAdjointStateCount) {
                                CompletedGradientEvent completed{};
                                completed.kind = PendingAdjointKind::ProjectionScatter;
                                completed.xPrimitiveIndex = worldHit.primitiveIndex;
                                completed.xAlphaGeom = worldHit.alphaGeom;
                                completed.xPosition = worldHit.hitPositionW;
                                completed.xIncomingRay = rayState.ray;
                                completed.xNormal = orientedNormal; // important
                                completed.xCosine = dot(-rayState.ray.direction, orientedNormal); // important
                                completed.xPathThroughput = rayState.pathThroughput / settings.sampling.qReflect;
                                completed.pixelIndex = rayState.pixelIndex;
                                appendCompletedGradientEventAtomic(
                                    intermediates.countCompletedGradientEvents,
                                    intermediates.completedGradientEvents,
                                    intermediates.maxCompletedGradientEventCount,
                                    completed);
                            }
                            */


                            if (rayState.pathId < intermediates.maxPendingAdjointStateCount) {
                                auto &pendingStageX = intermediates.pendingStageX[rayState.pathId];
                                auto &pendingStageXY = intermediates.pendingStageXY[rayState.pathId];

                                if (!pendingStageX.valid && !pendingStageXY.valid) {
                                    const float3 throughput = rayState.pathThroughput / settings.sampling.qReflect;
                                    pendingStageX = makePendingStageX(
                                        rayState.pathId,
                                        rayState.pixelIndex,
                                        worldHit,
                                        rayState,
                                        throughput,
                                        orientedNormal,
                                        instance.geometryType);
                                }
                            }
                            auto extensionCounter = sycl::atomic_ref<uint32_t,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>(
                                *intermediates.countExtensionOut);
                            const uint32_t outIndex = extensionCounter.fetch_add(1);
                            intermediates.extensionRaysA[outIndex] = nextState;
                        } else if (u < settings.sampling.qNull + settings.sampling.qReflect + settings.sampling.
                                   qTransmit) {
                            const auto &surfel = scene.points[worldHit.primitiveIndex];
                            float alpha = worldHit.alphaGeom * surfel.opacity;
                            float weight = (alpha * surfel.alpha_t) / settings.sampling.qTransmit;
                            float3 throughput = rayState.pathThroughput * weight;
                            // Find which side we hit the surfel:
                            const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));
                            const float signedCosineIncident = dot(canonicalNormalW, -rayState.ray.direction);
                            const int sideSign = signNonZero(signedCosineIncident);
                            // If positive we hit the front side if negative we hit the backside
                            float3 orientedNormal = static_cast<float>(sideSign) * canonicalNormalW;

                            float3 sampledOutgoingDirectionW = rayState.ray.direction;
                            const float3 &lambertBrdf = surfel.albedo;
                            // If we hit instance was a mesh do ordinary BRDF stuff.
                            float sampledPdf = 0.0f;
                            sampleUniformHemisphereAroundNormal(rng, orientedNormal, sampledOutgoingDirectionW,
                                                                sampledPdf);
                            RayState nextState{};
                            // Spawn next ray
                            nextState.ray.origin = worldHit.hitPositionW + (-orientedNormal * 1e-5f);
                            nextState.ray.direction = sampledOutgoingDirectionW;
                            nextState.ray.normal = -orientedNormal; // optional, but keep consistent
                            nextState.bounceIndex = rayState.bounceIndex + 1;
                            nextState.pixelIndex = rayState.pixelIndex;
                            nextState.pathId = rayState.pathId;
                            nextState.pathThroughput = throughput * lambertBrdf;

                            if (!applyRussianRoulette(rng, nextState.bounceIndex, nextState.pathThroughput,
                                                      settings.russianRouletteStart))
                                return;

                            auto extensionCounter = sycl::atomic_ref<uint32_t,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>(
                                *intermediates.countExtensionOut);
                            const uint32_t outIndex = extensionCounter.fetch_add(1);
                            intermediates.extensionRaysA[outIndex] = nextState;
                        } else {
                            // Absorb: do not enqueue nextState, do not enqueue contribution.
                            return;
                        }
                    }
                });
        });
        queue.wait(); // DEBUG: ensure the thread blocks here
    }


    void adjointContributionKernels(RenderPackage &pkg, uint32_t contributionTransmittanceCount,
                                    uint32_t cameraIndex) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        // Host-side (before launching kernel)
        SensorGPU &sensor = pkg.sensors[cameraIndex];
        auto *contributionRecords = pkg.intermediates.completedGradientEvents;

        auto &gradients = pkg.gradients;
        const auto &photonMap = pkg.intermediates.map;
        auto *raysIn = pkg.intermediates.primaryRays;

        float invSpp = 1.0f / settings.adjointSamplesPerPixel;

        DebugImages &debugImage = pkg.debugImages[cameraIndex];

        queue.submit([&](sycl::handler &cgh) {
                uint64_t baseSeed = settings.random.seed;
                cgh.parallel_for<class launchContributionKernel>(
                    sycl::range<1>(contributionTransmittanceCount),
                    [=](sycl::id<1> globalId) {
                        const uint32_t contributionIndex = globalId[0];

                        CompletedGradientEvent &contribution = contributionRecords[contributionIndex];
                        const auto &event = settings.sampling;

                        if (contribution.valid) {
                            if (contribution.kind == PendingAdjointKind::Projection) {
                                const Point &surfel = scene.points[contribution.xPrimitiveIndex];
                                const float3 E = gatherDiffuseIrradianceAtPoint(
                                    contribution.xPosition,
                                    contribution.xNormal,
                                    photonMap);
                                // Evaluate surfel outgoing radiance (direct/indirect via photon map)
                                const float3 f_r = surfel.alpha_r * surfel.albedo * M_1_PIf; // Lambert BRDF
                                const float3 Lo = f_r * E;
                                // opacity alpha = alphaGeom * eta  => dLo/deta = alphaGeom * Lo
                                const float grad_alpha_eta = contribution.xAlphaGeom;
                                const float3 p_e = contribution.xPathThroughput;

                                // p should be the adjoint weight carried from the camera (residual etc.)
                                float3 grad_cost_eta = grad_alpha_eta * p_e * Lo;
                                const float grad_cost_eta_sum = sum(grad_cost_eta) * invSpp;
                                atomicAddFloat(gradients.gradOpacity[contribution.xPrimitiveIndex], grad_cost_eta_sum);

                                float2 uv = phiInverse(contribution.xPosition, surfel);
                                float u = uv.x();
                                float v = uv.y();
                                float r2 = u * u + v * v;
                                float su = surfel.scale.x();
                                float sv = surfel.scale.y();
                                auto DuvDPositionJacobian = computeDuvDSurfelTranslationJacobian(
                                    surfel.tanU,
                                    surfel.tanV,
                                    contribution.xNormal,
                                    contribution.xIncomingRay.direction,
                                    u, v,
                                    su, sv);
                                float3 DuvDPosition =
                                (u * DuvDPositionJacobian.du_d_surfel_translation + v * DuvDPositionJacobian.
                                 dv_d_surfel_translation);
                                float beta = 4.0f * sycl::exp(surfel.beta);
                                float factor = (-2.0f * beta * contribution.xAlphaGeom) / (1.0f - r2);
                                float3 dAlpha_dPos = factor * DuvDPosition;
                                float3 dAlphaEff_dPos = surfel.opacity * dAlpha_dPos;
                                float3 gradPosition_R = p_e[0] * dAlphaEff_dPos * Lo[0];
                                float3 gradPosition_G = p_e[1] * dAlphaEff_dPos * Lo[1];
                                float3 gradPosition_B = p_e[2] * dAlphaEff_dPos * Lo[2];
                                const float3 grad_cost_sp_sum =
                                        (gradPosition_R + gradPosition_G + gradPosition_B) * invSpp;

                                atomicAddFloat3(gradients.gradPosition[contribution.xPrimitiveIndex], grad_cost_sp_sum);
                            }


                            if (contribution.kind == PendingAdjointKind::ProjectionScatter) {
                                float3 Le = {0.0f, 0.0f, 0.0f};
                                float3 Lr = {0.0f, 0.0f, 0.0f};
                                const Point &surfelY = scene.points[contribution.yPrimitiveIndex]; {
                                    const float3 E = gatherDiffuseIrradianceAtPoint(
                                        contribution.yPosition,
                                        contribution.yNormal,
                                        photonMap);
                                    float alpha = contribution.yAlphaGeom * surfelY.opacity;
                                    // Evaluate surfel outgoing radiance (direct/indirect via photon map)
                                    float3 tangentUWorld = surfelY.scale.x() * surfelY.tanU;
                                    float3 tangentVWorld = surfelY.scale.y() * surfelY.tanV;
                                    float3 canonicalNormal = cross(tangentUWorld, tangentVWorld);
                                    float totalAreaWorld = M_PIf * length(canonicalNormal);
                                    Lr = E * surfelY.alpha_r * surfelY.albedo * M_1_PIf * alpha;
                                    Le = surfelY.albedo * (surfelY.power / (M_PIf * totalAreaWorld)) * alpha;

                                    bool hitBackside =
                                            dot(canonicalNormal, -contribution.yIncomingRay.direction) < 0.0f;
                                    if (surfelY.power > 0.0f && hitBackside)
                                        Le = 0.0f;
                                }

                                const float3 Lo_y = Le + Lr;
                                const Point &surfelX = scene.points[contribution.xPrimitiveIndex];
                                // Detached sample on X: fixed local coordinates from the camera hit
                                const float2 uvDetached = phiInverse(contribution.xPosition, surfelX);
                                // Conceptually detach UV even though numerically xPosition and x is the same value.
                                const float3 x = phiMapping(surfelX, uvDetached.x(), uvDetached.y());
                                const float3 y = contribution.yPosition;
                                const float3 nx = contribution.xNormal; // or recompute from surfelX chart
                                const float3 ny = contribution.yNormal;

                                const float3 vectorXToY = y - x;
                                const float distanceSquared = dot(vectorXToY, vectorXToY);
                                if (distanceSquared <= 1e-12f) {
                                    return;
                                }
                                const float distance = sycl::sqrt(distanceSquared);
                                const float3 directionXToY = vectorXToY / distance;
                                const float cosineAtY = dot(ny, -directionXToY);
                                if (cosineAtY <= 1e-6f) {
                                    return;
                                }
                                const float uniformHemispherePdf = 1.0f / (2.0f * M_PIf);
                                const float pA_Y = uniformHemispherePdf * cosineAtY / distanceSquared;
                                const float J_A_Y = surfelY.scale.x() * surfelY.scale.y();
                                const float pU = pA_Y * J_A_Y;
                                if (pU <= 1e-20f) {
                                    return;
                                }
                                const float alphaX = contribution.xAlphaGeom * surfelX.opacity;
                                const float3 brdfX = surfelX.alpha_r * surfelX.albedo * M_1_PIf;
                                // Detached derivative wrt world-space start point x
                                const float3 dG_dX = computeGeometricTermGradientWrtStartpoint(x, y, nx, ny);
                                float visibility = 1.0f;
                                float transmittance = 1.0f;
                                const float3 transportWithoutG =
                                        Lo_y * alphaX * brdfX * visibility * transmittance;

                                const float3 p_e = contribution.xPathThroughput;
                                const float scalarWeight =
                                        // First form the gradient with respect to the observed world-space hit point x.
                                        (p_e[0] * transportWithoutG[0] +
                                         p_e[1] * transportWithoutG[1] +
                                         p_e[2] * transportWithoutG[2]) / pA_Y;
                                const float3 gradientWrtHitPosition = scalarWeight * dG_dX;
                                // Then map that to the surfel center translation s_p using the camera-ray / plane Jacobian.
                                float3x3 intersectionJacobian = planeHitPointIntersectionJacobian(
                                    contribution.xIncomingRay.direction, contribution.xNormal);

                                float3 gradientWrtSurfelTranslation =
                                        transpose(intersectionJacobian) * gradientWrtHitPosition;

                                atomicAddFloat3(gradients.gradPosition[contribution.xPrimitiveIndex],
                                                gradientWrtSurfelTranslation * invSpp);


                                // Detached Y-gradient:
                                {
                                    const float3 dG_dY = computeGeometricTermGradientWrtEndpoint(x, y, nx, ny);
                                    const float3 gradientY = scalarWeight * dG_dY;
                                    atomicAddFloat3(gradients.gradPosition[contribution.yPrimitiveIndex],
                                                    gradientY * invSpp);
                                }
                            }
                        }


                        if (contribution.kind == PendingAdjointKind::ReflectScatter) {
                            float3 Lo_Z = {0.0f, 0.0f, 0.0f};

                            const Point &surfelZ = scene.points[contribution.zPrimitiveIndex];
                            const Point &surfelY = scene.points[contribution.yPrimitiveIndex];
                            const Point &surfelX = scene.points[contribution.xPrimitiveIndex]; {
                                float3 Le = {0.0f, 0.0f, 0.0f};
                                float3 Lr = {0.0f, 0.0f, 0.0f};
                                const float3 E = gatherDiffuseIrradianceAtPoint(
                                    contribution.zPosition,
                                    contribution.zNormal,
                                    photonMap);

                                const float alphaZ = contribution.zAlphaGeom * surfelZ.opacity;
                                float3 tangentUWorld = surfelZ.scale.x() * surfelZ.tanU;
                                float3 tangentVWorld = surfelZ.scale.y() * surfelZ.tanV;
                                float3 canonicalNormal = cross(tangentUWorld, tangentVWorld);
                                float totalAreaWorld = M_PIf * length(canonicalNormal);
                                Lr = E * surfelZ.alpha_r * surfelZ.albedo * M_1_PIf * alphaZ;
                                Le = surfelZ.albedo * (surfelZ.power / (M_PIf * totalAreaWorld)) * alphaZ;
                                //Optional backside emission test if desired.
                                bool hitBackside = dot(canonicalNormal, -contribution.zIncomingRay.direction) < 0.0f;
                                if (surfelZ.power > 0.0f && hitBackside)
                                    Le = 0.0f;

                                Lo_Z = Le + Lr;
                            }
                            // Detached sample on Y
                            const float2 uvDetachedY = phiInverse(contribution.yPosition, surfelY);

                            const float3 x = contribution.xPosition;
                            const float3 y = contribution.yPosition;
                            const float3 z = contribution.zPosition;
                            const float3 nx = contribution.xNormal;
                            const float3 ny = contribution.yNormal;
                            const float3 nz = contribution.zNormal;

                            const float3 vectorYToZ = z - y;
                            const float distanceSquaredYZ = dot(vectorYToZ, vectorYToZ);
                            if (distanceSquaredYZ <= 1e-12f) {
                                return;
                            }

                            const float distanceXZ = sycl::sqrt(distanceSquaredYZ);
                            const float3 directionYToZ = vectorYToZ / distanceXZ;

                            const float cosineAtZ = dot(nz, -directionYToZ);
                            if (cosineAtZ <= 1e-6f) {
                                return;
                            }

                            const float3 vectorXToY = y - x;
                            const float distanceSquaredXY = dot(vectorXToY, vectorXToY);
                            if (distanceSquaredXY <= 1e-12f) {
                                return;
                            }
                            const float distanceXY = sycl::sqrt(distanceSquaredXY);
                            const float3 directionXToY = vectorXToY / distanceXY;
                            const float cosineAtY = dot(ny, -directionXToY);
                            if (cosineAtY <= 1e-6f) {
                                return;
                            }
                            const float uniformHemispherePdf = 1.0f / (2.0f * M_PIf);
                            const float pA_Y = uniformHemispherePdf * cosineAtY / distanceSquaredXY;
                            const float alphaX = contribution.xAlphaGeom * surfelX.opacity;
                            const float3 brdfX = surfelX.alpha_r * surfelX.albedo * M_1_PIf;
                            // Detached derivative wrt world-space start point x
                            const float Gxy = computeGeometricTermValue(x, y, nx, ny);


                            // Change of measure for Z sampled from a hemisphere at Y
                            const float pA_Z = uniformHemispherePdf * cosineAtZ / distanceSquaredYZ;
                            if (pA_Z <= 1e-20f) {
                                return;
                            }

                            const float alphaY = contribution.yAlphaGeom * surfelY.opacity;
                            const float3 brdfY = surfelY.alpha_r * surfelY.albedo * M_1_PIf;


                            const float visibility = 1.0f;
                            const float transmittance = 1.0f;

                            const float3 dG_dY = computeGeometricTermGradientWrtStartpoint(y, z, ny, nz);

                            const float3 upstreamTransportWithoutG =
                                    Lo_Z * alphaY * brdfY * visibility * transmittance;

                            const float3 transportXY =
                                    alphaX * brdfX * visibility * transmittance * Gxy;

                            const float3 combinedTransport =
                                    transportXY * upstreamTransportWithoutG;

                            const float3 p_e = contribution.xPathThroughput;

                            const float scalarWeightWithoutPAZ =
                            (p_e[0] * combinedTransport[0] +
                             p_e[1] * combinedTransport[1] +
                             p_e[2] * combinedTransport[2]) / (pA_Y * event.qReflect);

                            const float3 gradientWrtYPosition =
                                    (scalarWeightWithoutPAZ / (pA_Z)) * dG_dY;

                            atomicAddFloat3(
                                gradients.gradPosition[contribution.yPrimitiveIndex],
                                gradientWrtYPosition * invSpp);
                        }
                    }
                );
            }
        ).wait();
    }
}
