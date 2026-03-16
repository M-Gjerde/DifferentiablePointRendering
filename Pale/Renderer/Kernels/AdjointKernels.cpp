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
                            //if (currentGeometryType == GeometryType::PointCloud) {
                            //    //pendingStageX.xPathThroughput = pendingStageX.xPathThroughput * settings.sampling.qReflect;
                            //}
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


                            // If Y is a mesh
                            if (contribution.kind == PendingAdjointKind::ProjectionScatter) {
                                float3 Le = {0.0f, 0.0f, 0.0f};
                                float3 Lr = {0.0f, 0.0f, 0.0f};

                                if (contribution.yGeometryType == GeometryType::Mesh) {
                                    const auto &instance = scene.instances[contribution.yInstanceIndex];
                                    auto &material = scene.materials[instance.materialIndex];
                                    Lr = gatherDiffuseIrradianceAtPoint(
                                             contribution.yPosition,
                                             contribution.yNormal,
                                             photonMap) * material.baseColor * M_1_PIf;

                                    if (material.isEmissive()) {
                                        GPULightRecord emitter = scene.lights[0];
                                        Le = material.baseColor * (material.power / (M_PIf * emitter.totalAreaWorld));
                                    }
                                } else {
                                    const float3 E = gatherDiffuseIrradianceAtPoint(
                                        contribution.yPosition,
                                        contribution.yNormal,
                                        photonMap);
                                    const auto &pt = scene.points[contribution.yPrimitiveIndex];
                                    // Evaluate surfel outgoing radiance (direct/indirect via photon map)
                                    Lr = E * pt.alpha_r * pt.albedo * M_1_PIf * contribution.yAlphaGeom * pt.opacity;
                                    // Lambert BRDF
                                }

                                const float3 Lo = Le + Lr;
                                const float3 x = contribution.xPosition;
                                const float3 y = contribution.yPosition;
                                const float3 nx = contribution.xNormal;
                                const float3 ny = contribution.yNormal;
                                const float3 vectorXToY = y - x;
                                const float distanceSquared = dot(vectorXToY, vectorXToY);
                                if (distanceSquared <= 1e-12f) {
                                    return;
                                }
                                const float distance = sycl::sqrt(distanceSquared);
                                const float3 directionXToY = vectorXToY / distance;
                                const float cosineAtY = dot(ny, -directionXToY);
                                if (cosineAtY <= 1e-6f) return;
                                float uniformHemispherePDF = 1.0f / (2.0f * M_PIf);
                                const float pA_Y =
                                        uniformHemispherePDF * cosineAtY / distanceSquared;

                                const Point &surfelX = scene.points[contribution.xPrimitiveIndex];
                                const float alphaX = contribution.xAlphaGeom * surfelX.opacity;
                                const float3 brdfX = surfelX.alpha_r * surfelX.albedo * M_1_PIf;
                                const float3 dG_dPos = computeGeometricTermGradientWrtStartpoint(x, y, nx, ny);
                                float visibility = 1.0f;
                                float transmittance = 1.0f;
                                // This is the transport factor multiplying G(x, y).
                                const float3 transportWithoutG =
                                        Lo * alphaX * brdfX * visibility * transmittance;
                                // Adjoint-weighted scalar multiplier for the local derivative insertion.
                                const float3 &p_e = (contribution.xPathThroughput);
                                const float scalarWeight =
                                        // First form the gradient with respect to the observed world-space hit point x.
                                        (p_e[0] * transportWithoutG[0] +
                                         p_e[1] * transportWithoutG[1] +
                                         p_e[2] * transportWithoutG[2]) / pA_Y;
                                const float3 gradientWrtHitPosition = scalarWeight * dG_dPos;
                                // Then map that to the surfel center translation s_p using the camera-ray / plane Jacobian.
                                float3x3 intersectionJacobian = planeHitPointIntersectionJacobian(
                                    contribution.xIncomingRay.direction, contribution.xNormal);

                                float3 gradientWrtSurfelTranslation =
                                        transpose(intersectionJacobian) * gradientWrtHitPosition;

                                atomicAddFloat3(gradients.gradPosition[contribution.xPrimitiveIndex],
                                                gradientWrtSurfelTranslation * invSpp);
                            }
                        }


                        if (contribution.kind == PendingAdjointKind::ReflectScatter) {
                            if (contribution.valid) {
                                if (contribution.yGeometryType != GeometryType::PointCloud) {
                                    return;
                                }

                                auto evaluateOutgoingRadiance = [&](GeometryType geometryType,
                                                                    uint32_t instanceIndex,
                                                                    uint32_t primitiveIndex,
                                                                    const float3 &position,
                                                                    const float3 &normal,
                                                                    float alphaGeom) -> float3 {
                                    float3 emittedRadiance = float3{0.0f};
                                    float3 reflectedRadiance = float3{0.0f};

                                    if (geometryType == GeometryType::Mesh) {
                                        const auto &instance = scene.instances[instanceIndex];
                                        const auto &material = scene.materials[instance.materialIndex];

                                        const float3 irradiance =
                                                gatherDiffuseIrradianceAtPoint(position, normal, photonMap);

                                        reflectedRadiance = irradiance * material.baseColor * M_1_PIf;

                                        if (material.isEmissive()) {
                                            const GPULightRecord emitter = scene.lights[0];
                                            emittedRadiance =
                                                    material.baseColor * (
                                                        material.power / (M_PIf * emitter.totalAreaWorld));
                                        }
                                    } else {
                                        const auto &surfel = scene.points[primitiveIndex];
                                        const float3 irradiance =
                                                gatherDiffuseIrradianceAtPoint(position, normal, photonMap);

                                        reflectedRadiance =
                                                irradiance *
                                                surfel.alpha_r *
                                                surfel.albedo *
                                                M_1_PIf *
                                                alphaGeom *
                                                surfel.opacity;
                                    }

                                    return emittedRadiance + reflectedRadiance;
                                };

                                auto evaluateLambertianBrdf = [&](GeometryType geometryType,
                                                                  uint32_t instanceIndex,
                                                                  uint32_t primitiveIndex) -> float3 {
                                    if (geometryType == GeometryType::Mesh) {
                                        const auto &instance = scene.instances[instanceIndex];
                                        const auto &material = scene.materials[instance.materialIndex];
                                        return material.baseColor * M_1_PIf;
                                    }

                                    const auto &surfel = scene.points[primitiveIndex];
                                    return surfel.alpha_r * surfel.albedo * M_1_PIf;
                                };

                                auto computeAreaPdfFromUniformHemisphereSample = [&](const float3 &startPosition,
                                    const float3 &endPosition,
                                    const float3 &endNormal) -> float {
                                    const float3 vectorToEnd = endPosition - startPosition;
                                    const float distanceSquared = dot(vectorToEnd, vectorToEnd);
                                    if (distanceSquared <= 1e-12f) {
                                        return 0.0f;
                                    }

                                    const float distance = sycl::sqrt(distanceSquared);
                                    const float3 directionToEnd = vectorToEnd / distance;
                                    const float cosineAtEnd = dot(endNormal, -directionToEnd);
                                    if (cosineAtEnd <= 1e-6f) {
                                        return 0.0f;
                                    }

                                    const float uniformHemispherePdf = 1.0f / (2.0f * M_PIf);
                                    return uniformHemispherePdf * cosineAtEnd / distanceSquared;
                                };

                                const float3 outgoingRadianceAtZ = evaluateOutgoingRadiance(
                                    contribution.zGeometryType,
                                    contribution.zInstanceIndex,
                                    contribution.zPrimitiveIndex,
                                    contribution.zPosition,
                                    contribution.zNormal,
                                    contribution.zAlphaGeom);

                                const float3 outgoingRadianceAtY = evaluateOutgoingRadiance(
                                    contribution.yGeometryType,
                                    contribution.yInstanceIndex,
                                    contribution.yPrimitiveIndex,
                                    contribution.yPosition,
                                    contribution.yNormal,
                                    contribution.yAlphaGeom);

                                const float3 xPosition = contribution.xPosition;
                                const float3 yPosition = contribution.yPosition;
                                const float3 zPosition = contribution.zPosition;

                                const float3 xNormal = contribution.xNormal;
                                const float3 yNormal = contribution.yNormal;
                                const float3 zNormal = contribution.zNormal;

                                const float areaPdfXY =
                                        computeAreaPdfFromUniformHemisphereSample(xPosition, yPosition, yNormal);
                                if (areaPdfXY <= 0.0f) {
                                    return;
                                }

                                const float areaPdfYZ =
                                        computeAreaPdfFromUniformHemisphereSample(yPosition, zPosition, zNormal);
                                if (areaPdfYZ <= 0.0f) {
                                    return;
                                }
                                const float inverseAreaPdfXY = 1.0f / areaPdfXY;
                                const float inverseAreaPdfYZ = 1.0f / areaPdfYZ;
                                const float visibilityXY = 1.0f;
                                const float visibilityYZ = 1.0f;
                                const float transmittanceXY = 1.0f;
                                const float transmittanceYZ = 1.0f;
                                const float geometricTermXY =
                                        computeGeometricTermValue(xPosition, yPosition, xNormal, yNormal);
                                const float geometricTermYZ =
                                        computeGeometricTermValue(yPosition, zPosition, yNormal, zNormal);
                                const float3 gradientOfGeometricTermYZWrtY =
                                        computeGeometricTermGradientWrtStartpoint(
                                            yPosition, zPosition, yNormal, zNormal);
                                const float3 gradientOfGeometricTermXYWrtY =
                                        computeGeometricTermGradientWrtEndpoint(xPosition, yPosition, xNormal, yNormal);

                                const float3 brdfAtX = evaluateLambertianBrdf(
                                    contribution.xGeometryType,
                                    contribution.xInstanceIndex,
                                    contribution.xPrimitiveIndex);

                                const float3 brdfAtY = evaluateLambertianBrdf(
                                    contribution.yGeometryType,
                                    contribution.yInstanceIndex,
                                    contribution.yPrimitiveIndex);

                                const Point &surfelX = scene.points[contribution.xPrimitiveIndex];
                                const Point &surfelY = scene.points[contribution.yPrimitiveIndex];

                                const float alphaX = contribution.xAlphaGeom * surfelX.opacity;
                                const float alphaY = contribution.yAlphaGeom * surfelY.opacity;

                                // d alpha_eff(y) / d position_y
                                const float2 uv = phiInverse(contribution.yPosition, surfelY);
                                const float u = uv.x();
                                const float v = uv.y();
                                const float radiusSquared = u * u + v * v;

                                const float scaleU = surfelY.scale.x();
                                const float scaleV = surfelY.scale.y();

                                const auto uvJacobian = computeDuvDSurfelTranslationJacobian(
                                    surfelY.tanU,
                                    surfelY.tanV,
                                    contribution.yNormal,
                                    contribution.yIncomingRay.direction,
                                    u,
                                    v,
                                    scaleU,
                                    scaleV);

                                const float3 radiusSquaredGradientWrtPosition =
                                        u * uvJacobian.du_d_surfel_translation +
                                        v * uvJacobian.dv_d_surfel_translation;

                                const float beta = 4.0f * sycl::exp(surfelY.beta);
                                const float alphaGeomDerivativeFactor =
                                        (-2.0f * beta * contribution.yAlphaGeom) / (1.0f - radiusSquared);

                                const float3 alphaGeomGradientWrtPosition =
                                        alphaGeomDerivativeFactor * radiusSquaredGradientWrtPosition;

                                const float3 alphaEffectiveGradientWrtPosition =
                                        surfelY.opacity * alphaGeomGradientWrtPosition;

                                float3 gradientWrtYPosition = float3{0.0f};

                                const float3 adjointWeightAtX = contribution.xPathThroughput * alphaX / event.qReflect;

                                for (int colorChannel = 0; colorChannel < 3; ++colorChannel) {
                                    // Outer kernel K(x <- y), excluding alpha(x), with area-form MC weight.
                                    const float outerTransportWeight =
                                            brdfAtX[colorChannel] *
                                            visibilityXY *
                                            transmittanceXY *
                                            geometricTermXY *
                                            inverseAreaPdfXY;

                                    // Inner integral at y, excluding alpha(y), estimated in area form.
                                    const float innerTransportWithoutAlphaY =
                                            outgoingRadianceAtZ[colorChannel] *
                                            brdfAtY[colorChannel] *
                                            visibilityYZ *
                                            transmittanceYZ *
                                            geometricTermYZ *
                                            inverseAreaPdfYZ;

                                    // Term A:
                                    // d alpha(y) / d position_y * [inner integral without alpha(y)] * outer kernel
                                    const float alphaGradientScalarWeight =
                                            adjointWeightAtX[colorChannel] *
                                            innerTransportWithoutAlphaY *
                                            outerTransportWeight;

                                    // Term B:
                                    // alpha(y) * d G(y,z) / d position_y * remaining scalar factors
                                    const float innerGeometricScalarWeight =
                                            adjointWeightAtX[colorChannel] *
                                            outgoingRadianceAtZ[colorChannel] *
                                            brdfAtY[colorChannel] *
                                            visibilityYZ *
                                            transmittanceYZ *
                                            alphaY *
                                            brdfAtX[colorChannel] *
                                            visibilityXY *
                                            transmittanceXY *
                                            geometricTermXY *
                                            inverseAreaPdfYZ *
                                            inverseAreaPdfXY;

                                    // Term C:
                                    // L_o(y) * d G(x,y) / d position_y * outer remaining factors
                                    const float outerGeometricScalarWeight =
                                            adjointWeightAtX[colorChannel] *
                                            outgoingRadianceAtY[colorChannel] *
                                            brdfAtX[colorChannel] *
                                            visibilityXY *
                                            transmittanceXY *
                                            inverseAreaPdfXY;

                                    gradientWrtYPosition +=
                                            alphaEffectiveGradientWrtPosition * alphaGradientScalarWeight +
                                            gradientOfGeometricTermYZWrtY * innerGeometricScalarWeight +
                                            gradientOfGeometricTermXYWrtY * outerGeometricScalarWeight;
                                }

                                atomicAddFloat3(
                                    gradients.gradPosition[contribution.yPrimitiveIndex],
                                    gradientWrtYPosition * invSpp);
                            }
                        }
                    }
                );
            }
        ).wait();
    }
}
