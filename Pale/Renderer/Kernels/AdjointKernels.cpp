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
                    uint32_t index = flippedYLinearIndex(pixelLinearIndexWithinImage, sensor.width, sensor.height);

                    const uint32_t pixelIndex = pixelLinearIndexWithinImage;
                    // RNG for this pixelhttps://www.chess.com/home
                    const uint64_t seed =
                            rng::makeSeed(renderSeed, globalRayIndex, 0u, rng::kStreamRayGen, 0u);
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
                        0, 0
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


    void launchAdjointIntersectKernel(RenderPackage &pkg, uint32_t activeRayCount, uint32_t bounceIndex) {
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
                            rng::makeSeed(renderSeed, rayState.pathId, rayState.bounceIndex,
                                          rng::kStreamTraversal, 107u);
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
                    float sampledOutgoingDirectionPDF{0.0f};
                    float3 sampledOutgoingDirectionW{0.0f};
                    const InstanceRecord &endpointInstance = scene.instances[worldHit.instanceIndex];
                    if (endpointInstance.geometryType == GeometryType::Mesh) {
                        // Cosine sampling
                        sampleCosineHemisphere(rng, worldHit.geometricNormalW, sampledOutgoingDirectionW,
                                               sampledOutgoingDirectionPDF);
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
                                                            sampledOutgoingDirectionPDF);
                    }

                    if (rayState.pathId < intermediates.maxPendingAdjointStateCount) {
                        PendingAdjointState pending = intermediates.pendingAdjointStates[rayState.pathId];
                        if (pending.kind == PendingAdjointKind::ReflectScatter) {
                            // Obtain normal:
                            CompletedGradientEvent completed{};
                            completed.pathId = pending.pathId;
                            completed.kind = pending.kind;
                            completed.primitiveIndex = pending.primitiveIndex;
                            completed.instanceIndex = pending.instanceIndex;
                            completed.alphaGeom = pending.alphaGeom;
                            completed.hitPosition = pending.hitPosition;
                            completed.hitNormal = pending.hitNormal;
                            completed.pathThroughput = pending.pathThroughput;
                            completed.pixelIndex = pending.pixelIndex;
                            completed.cosineHitPoint = pending.cosine;
                            completed.geometryType = pending.geometryType;
                            completed.ray = rayState.ray;

                            completed.endPointAlphaGeom = worldHit.alphaGeom;
                            completed.endpointInstanceIndex = worldHit.instanceIndex;
                            completed.endpointPrimitiveIndex = worldHit.primitiveIndex;
                            completed.endpointPosition = worldHit.hitPositionW;
                            completed.endpointNormal = orientedNormal;
                            completed.endPointPDF = sampledOutgoingDirectionPDF;
                            completed.endpointGeometryType = endpointInstance.geometryType;
                            // check endpoint cosine
                            const float cosThetaOut = sycl::fmax(
                                0.0f, dot(sampledOutgoingDirectionW, orientedNormal));
                            completed.endpointCosine = cosThetaOut;
                            completed.endpointLightIndex = endpointInstance.geometryIndex;

                            appendCompletedGradientEventAtomic(
                                intermediates.countCompletedGradientEvents,
                                intermediates.completedGradientEvents,
                                intermediates.maxCompletedGradientEventCount,
                                completed);
                            // Clear pending in global memory
                            clearPendingAdjointState(intermediates.pendingAdjointStates[rayState.pathId]);
                        }
                        if (pending.kind == PendingAdjointKind::NullTransmittance) {
                            // Obtain normal:
                            CompletedGradientEvent completed{};
                            completed.pathId = pending.pathId;
                            completed.kind = pending.kind;
                            completed.primitiveIndex = pending.primitiveIndex;
                            completed.instanceIndex = pending.instanceIndex;
                            completed.alphaGeom = pending.alphaGeom;
                            completed.hitPosition = pending.hitPosition;
                            completed.pathThroughput = pending.pathThroughput;
                            completed.pixelIndex = pending.pixelIndex;
                            completed.cosineHitPoint = dot(orientedNormal, -rayState.ray.direction);
                            completed.ray = rayState.ray;
                            completed.hitNormal = pending.hitNormal;

                            completed.endPointAlphaGeom = worldHit.alphaGeom;
                            completed.endpointInstanceIndex = worldHit.instanceIndex;
                            completed.endpointPrimitiveIndex = worldHit.primitiveIndex;
                            completed.endpointPosition = worldHit.hitPositionW;
                            completed.endpointNormal = orientedNormal;
                            completed.endPointPDF = sampledOutgoingDirectionPDF;
                            completed.endpointGeometryType = endpointInstance.geometryType;
                            // check endpoint cosine
                            const float cosThetaOut = sycl::fmax(
                                0.0f, dot(sampledOutgoingDirectionW, orientedNormal));
                            completed.endpointCosine = cosThetaOut;
                            completed.endpointLightIndex = endpointInstance.geometryIndex;

                            appendCompletedGradientEventAtomic(
                                intermediates.countCompletedGradientEvents,
                                intermediates.completedGradientEvents,
                                intermediates.maxCompletedGradientEventCount,
                                completed);
                            // Clear pending in global memory
                            clearPendingAdjointState(intermediates.pendingAdjointStates[rayState.pathId]);
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


                        // Populate pending (now that the path survives)
                        if (rayState.pathId < intermediates.maxPendingAdjointStateCount) {
                            PendingAdjointState pending{};
                            pending.kind = PendingAdjointKind::ReflectScatter;
                            pending.instanceIndex = worldHit.instanceIndex;
                            pending.hitPosition = worldHit.hitPositionW;
                            pending.pathThroughput = rayState.pathThroughput * throughputMultiplier;
                            pending.pixelIndex = rayState.pixelIndex;
                            pending.pathId = rayState.pathId;
                            pending.geometryType = instance.geometryType;
                            pending.cosine = dot(-rayState.ray.direction, orientedNormal);
                            intermediates.pendingAdjointStates[rayState.pathId] = pending;
                        }

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
                                                                      cosTheta)) / sampledOutgoingDirectionPDF;
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
                            if (bounceIndex == 0 && rayState.pathId < intermediates.maxPendingAdjointStateCount) {
                                CompletedGradientEvent completed{};
                                completed.kind = PendingAdjointKind::ProjectionScatter;
                                completed.primitiveIndex = worldHit.primitiveIndex;
                                completed.alphaGeom = worldHit.alphaGeom;
                                completed.hitPosition = worldHit.hitPositionW;
                                completed.ray = rayState.ray;
                                completed.hitNormal = orientedNormal; // important
                                completed.cosineHitPoint = dot(-rayState.ray.direction, orientedNormal); // important
                                completed.pathThroughput = rayState.pathThroughput / settings.sampling.qReflect;
                                completed.pixelIndex = rayState.pixelIndex;
                                completed.hasEndpoint = false;
                                appendCompletedGradientEventAtomic(
                                    intermediates.countCompletedGradientEvents,
                                    intermediates.completedGradientEvents,
                                    intermediates.maxCompletedGradientEventCount,
                                    completed);
                            }

                            if (rayState.pathId < intermediates.maxPendingAdjointStateCount) {
                                PendingAdjointState pending{};
                                pending.kind = PendingAdjointKind::ReflectScatter;
                                pending.primitiveIndex = worldHit.primitiveIndex;
                                pending.hitPosition = worldHit.hitPositionW;
                                pending.hitNormal = orientedNormal;
                                pending.pathThroughput = rayState.pathThroughput * throughputMultiplier;
                                pending.pixelIndex = rayState.pixelIndex;
                                pending.alphaGeom = worldHit.alphaGeom;
                                pending.pathId = rayState.pathId;
                                pending.cosine = dot(-rayState.ray.direction, orientedNormal);
                                pending.geometryType = instance.geometryType;

                                intermediates.pendingAdjointStates[rayState.pathId] = pending;
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


    void launchAdjointTransportKernel(RenderPackage &pkg, uint32_t contributionTransmittanceCount,
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
                uint64_t baseSeed = settings.random.seed * (static_cast<uint64_t>(cameraIndex) + 5ull);
                cgh.parallel_for<class launchContributionKernel>(
                    sycl::range<1>(contributionTransmittanceCount),
                    [=](sycl::id<1> globalId) {
                        const uint32_t contributionIndex = globalId[0];
                        CompletedGradientEvent &contribution = contributionRecords[contributionIndex];
                        if (contribution.kind == PendingAdjointKind::ProjectionScatter) {
                            const Point &surfel = scene.points[contribution.primitiveIndex];
                            const float3 E = gatherDiffuseIrradianceAtPoint(
                                contribution.hitPosition,
                                contribution.hitNormal,
                                photonMap);

                            // Evaluate surfel outgoing radiance (direct/indirect via photon map)
                            const float3 f_r = surfel.alpha_r * surfel.albedo * M_1_PIf; // Lambert BRDF
                            const float3 Lo = f_r * E;
                            // opacity alpha = alphaGeom * eta  => dLo/deta = alphaGeom * Lo
                            const float grad_alpha_eta = contribution.alphaGeom;
                            const float3 p_e = contribution.pathThroughput;

                            // p should be the adjoint weight carried from the camera (residual etc.)
                            float3 grad_cost_eta = grad_alpha_eta * p_e * Lo;
                            const float grad_cost_eta_sum = sum(grad_cost_eta) * invSpp;


                            float3 canonicalNormalWorld = contribution.hitNormal;
                            float2 uv = phiInverse(contribution.hitPosition, surfel);
                            float u = uv.x();
                            float v = uv.y();
                            float r2 = u * u + v * v;
                            float su = surfel.scale.x();
                            float sv = surfel.scale.y();
                            auto DuvDPositionJacobian = computeDuvDPositionJacobian(
                                surfel.tanU,
                                surfel.tanV,
                                canonicalNormalWorld,
                                contribution.ray.direction,
                                u, v,
                                su, sv);

                            float3 DuvDPosition =
                                    u * DuvDPositionJacobian.du_d_position + v * DuvDPositionJacobian.dv_d_position;
                            float beta = 4.0f * sycl::exp(surfel.beta);
                            float factor = (-2.0f * beta * contribution.alphaGeom) / (1.0f - r2);
                            float3 dAlpha_dPos = factor * DuvDPosition;
                            float3 dAlphaEff_dPos = surfel.opacity * dAlpha_dPos;

                            float3 gradPosition_R = p_e[0] * dAlphaEff_dPos * Lo[0];
                            float3 gradPosition_G = p_e[1] * dAlphaEff_dPos * Lo[1];
                            float3 gradPosition_B = p_e[2] * dAlphaEff_dPos * Lo[2];
                            const float3 grad_cost_sp_sum = (gradPosition_R + gradPosition_G + gradPosition_B) * invSpp;

                            float y_grad = grad_cost_sp_sum.y();

                            // only if you truly have spp samples
                            //atomicAddFloat(gradients.gradOpacity[contribution.primitiveIndex], grad_cost_eta_sum);
                            //atomicAddFloat3(gradients.gradPosition[contribution.primitiveIndex], grad_cost_sp_sum);
                            if (settings.renderDebugGradientImages) {
                                uint32_t pixelIndex = contribution.pixelIndex;
                                atomicAddFloat4ToImage(
                                    &debugImage.framebufferOpacity[pixelIndex],
                                    float4{grad_cost_eta_sum}
                                );
                            }
                        }


                        if (contribution.kind == PendingAdjointKind::NullTransmittance) {
                            float3 Lo;
                            if (contribution.endpointGeometryType == GeometryType::Mesh) {
                                const auto &instance = scene.instances[contribution.endpointInstanceIndex];
                                const GPUMaterial material = scene.materials[instance.materialIndex];
                                float3 Lr = gatherDiffuseIrradianceAtPoint(
                                                contribution.endpointPosition,
                                                contribution.endpointNormal,
                                                photonMap) * material.baseColor * M_1_PIf;
                                float3 Le = {0.0f, 0.0f, 0.0f};
                                if (material.isEmissive()) {
                                    GPULightRecord emitter = scene.lights[0];
                                    Le = material.baseColor * (material.power / (M_PIf * emitter.totalAreaWorld));
                                }
                                Lo = Le + Lr;
                            } else if (contribution.endpointGeometryType == GeometryType::PointCloud) {
                                const Point &surfel = scene.points[contribution.endpointPrimitiveIndex];
                                float3 Lr = gatherDiffuseIrradianceAtPoint(
                                                contribution.endpointPosition,
                                                contribution.endpointNormal,
                                                photonMap) * surfel.albedo * M_1_PIf;
                                Lo = Lr;
                            }

                            float grad_tau_eta = -contribution.alphaGeom;
                            float3 p = contribution.pathThroughput;
                            float3 grad_cost_eta = grad_tau_eta * p * Lo;

                            if (contribution.endpointGeometryType == GeometryType::Mesh) {
                                const Point &surfel = scene.points[contribution.primitiveIndex];
                                float2 uv = phiInverse(contribution.hitPosition, surfel);
                                float u = uv.x();
                                float v = uv.y();
                                float r2 = u * u + v * v;
                                float su = surfel.scale.x();
                                float sv = surfel.scale.y();
                                float3 DuvDPosition = computeDuvDPosition(
                                    surfel.tanU,
                                    surfel.tanV,
                                    contribution.hitNormal,
                                    contribution.ray.direction,
                                    u, v,
                                    su, sv);

                                float beta = 4.0f * sycl::exp(surfel.beta);
                                float factor = (-2.0f * beta * contribution.alphaGeom) / (1.0f - r2);
                                float3 dAlpha_dPos = factor * DuvDPosition;
                                float3 dAlphaEff_dPos = -surfel.opacity * dAlpha_dPos;

                                float3 gradPosition_R = p[0] * dAlphaEff_dPos * Lo[0];
                                float3 gradPosition_G = p[1] * dAlphaEff_dPos * Lo[1];
                                float3 gradPosition_B = p[2] * dAlphaEff_dPos * Lo[2];
                                const float3 grad_cost_sp_sum =
                                        (gradPosition_R + gradPosition_G + gradPosition_B) * invSpp;
                                //atomicAddFloat3(gradients.gradPosition[contribution.primitiveIndex], grad_cost_sp_sum);
                            }
                            float grad_cost_eta_sum = sum(grad_cost_eta) * invSpp;
                            //atomicAddFloat(gradients.gradOpacity[contribution.primitiveIndex], grad_cost_eta_sum);

                            if (settings.renderDebugGradientImages) {
                                uint32_t pixelIndex = contribution.pixelIndex;
                                atomicAddFloat4ToImage(
                                    &debugImage.framebufferOpacity[pixelIndex],
                                    float4{grad_cost_eta_sum}
                                );
                            }
                        }

                        if (contribution.kind == PendingAdjointKind::ReflectScatter) {
                            if (contribution.endpointGeometryType == GeometryType::Mesh &&
                                contribution.geometryType == GeometryType::PointCloud
                            ) {
                                const auto &instance = scene.instances[contribution.endpointInstanceIndex];
                                const GPUMaterial material = scene.materials[instance.materialIndex];
                                float3 Lr = gatherDiffuseIrradianceAtPoint(
                                                contribution.endpointPosition,
                                                contribution.endpointNormal,
                                                photonMap) * material.baseColor * M_1_PIf;
                                float3 Le = {0.0f, 0.0f, 0.0f};
                                if (material.isEmissive()) {
                                    GPULightRecord emitter = scene.lights[0];
                                    Le = material.baseColor * (material.power / (M_PIf * emitter.totalAreaWorld));
                                }
                                const float3 Lo = Le + Lr;

                                const float3& x = contribution.hitPosition;
                                const float3& y = contribution.endpointPosition;
                                const float3& nx = contribution.hitNormal;
                                const float3& ny = contribution.endpointNormal;

                                const float3 G_grad_sp = computeGeometricTermGradientWrtX(x, y, nx, ny);
                                const float3 p = contribution.pathThroughput / (contribution.endPointPDF * contribution.endpointCosine);

                                float3 gradPosition_R = p[0] * G_grad_sp * Lo[0];
                                float3 gradPosition_G = p[1] * G_grad_sp * Lo[1];
                                float3 gradPosition_B = p[2] * G_grad_sp * Lo[2];
                                const float3 grad_cost_sp_sum =
                                        (gradPosition_R + gradPosition_G + gradPosition_B) * invSpp;

                                atomicAddFloat3(gradients.gradPosition[contribution.endpointPrimitiveIndex],
                                                grad_cost_sp_sum);

                                if (settings.renderDebugGradientImages) {
                                    uint32_t pixelIndex = contribution.pixelIndex;
                                    atomicAddFloat4ToImage(
                                        &debugImage.framebufferPosX[pixelIndex],
                                        float4{grad_cost_sp_sum.x()}
                                    );
                                }
                            }
                        }

                        if (contribution.kind == PendingAdjointKind::ReflectScatter) {
                            if (contribution.endpointGeometryType == GeometryType::PointCloud &&
                                contribution.geometryType == GeometryType::PointCloud
                            ) {
                                const Point &surfel = scene.points[contribution.endpointPrimitiveIndex];
                                float3 surfelIrradiance = gatherDiffuseIrradianceAtPoint(
                                    contribution.endpointPosition,
                                    contribution.endpointNormal,
                                    photonMap);

                                const float3 Lo_surfel = surfelIrradiance * surfel.alpha_r * surfel.albedo * M_1_PIf;
                                const float3 p = contribution.pathThroughput;
                                float3 grad_cost_eta = contribution.endPointAlphaGeom * p * Lo_surfel;
                                float grad_cost_eta_sum = sum(grad_cost_eta) * invSpp;
                                float3 canonicalNormalWorld = contribution.endpointNormal;
                                float2 uv = phiInverse(contribution.endpointPosition, surfel);
                                float u = uv.x();
                                float v = uv.y();
                                float r2 = u * u + v * v;
                                float su = surfel.scale.x();
                                float sv = surfel.scale.y();
                                float3 DuvDPosition = computeDuvDPosition(
                                    surfel.tanU,
                                    surfel.tanV,
                                    canonicalNormalWorld,
                                    contribution.ray.direction,
                                    u, v,
                                    su, sv);

                                float beta = 4.0f * sycl::exp(surfel.beta);
                                float factor = (-2.0f * beta * contribution.endPointAlphaGeom) / (1.0f - r2);
                                float3 dAlpha_dPos = factor * DuvDPosition;
                                float3 dAlphaEff_dPos = surfel.opacity * dAlpha_dPos;

                                float3 gradPosition_R = p[0] * dAlphaEff_dPos * Lo_surfel[0];
                                float3 gradPosition_G = p[1] * dAlphaEff_dPos * Lo_surfel[1];
                                float3 gradPosition_B = p[2] * dAlphaEff_dPos * Lo_surfel[2];
                                const float3 grad_cost_sp_sum =
                                        (gradPosition_R + gradPosition_G + gradPosition_B) * invSpp;


                                // only if you truly have spp samples
                                atomicAddFloat(gradients.gradOpacity[contribution.endpointPrimitiveIndex],
                                               grad_cost_eta_sum);
                                atomicAddFloat3(gradients.gradPosition[contribution.endpointPrimitiveIndex],
                                                grad_cost_sp_sum);
                                if (settings.renderDebugGradientImages) {
                                    uint32_t pixelIndex = contribution.pixelIndex;
                                    atomicAddFloat4ToImage(
                                        &debugImage.framebufferOpacity[pixelIndex],
                                        float4{grad_cost_eta_sum}
                                    );
                                }
                            }
                            if (contribution.endpointGeometryType == GeometryType::PointCloud && contribution.
                                geometryType == GeometryType::Mesh) {
                                const Point &surfel = scene.points[contribution.endpointPrimitiveIndex];
                                float3 surfelIrradiance = gatherDiffuseIrradianceAtPoint(
                                    contribution.endpointPosition,
                                    contribution.endpointNormal,
                                    photonMap);

                                const float3 Lo_surfel = surfelIrradiance * surfel.alpha_r * surfel.albedo * M_1_PIf;
                                const float3 p = contribution.pathThroughput;
                                float3 grad_cost_eta = contribution.endPointAlphaGeom * p * Lo_surfel;
                                float grad_cost_eta_sum = sum(grad_cost_eta) * invSpp;
                                float3 canonicalNormalWorld = contribution.endpointNormal;
                                float2 uv = phiInverse(contribution.endpointPosition, surfel);
                                float u = uv.x();
                                float v = uv.y();
                                float r2 = u * u + v * v;
                                float su = surfel.scale.x();
                                float sv = surfel.scale.y();
                                float3 DuvDPosition = computeDuvDPosition(
                                    surfel.tanU,
                                    surfel.tanV,
                                    canonicalNormalWorld,
                                    contribution.ray.direction,
                                    u, v,
                                    su, sv);

                                float beta = 4.0f * sycl::exp(surfel.beta);
                                float factor = (-2.0f * beta * contribution.endPointAlphaGeom) / (1.0f - r2);
                                float3 dAlpha_dPos = factor * DuvDPosition;
                                float3 dAlphaEff_dPos = surfel.opacity * dAlpha_dPos;

                                float3 gradPosition_R = p[0] * dAlphaEff_dPos * Lo_surfel[0];
                                float3 gradPosition_G = p[1] * dAlphaEff_dPos * Lo_surfel[1];
                                float3 gradPosition_B = p[2] * dAlphaEff_dPos * Lo_surfel[2];
                                const float3 grad_cost_sp_sum =
                                        (gradPosition_R + gradPosition_G + gradPosition_B) * invSpp;


                                // only if you truly have spp samples
                                atomicAddFloat(gradients.gradOpacity[contribution.endpointPrimitiveIndex],
                                               grad_cost_eta_sum);
                                atomicAddFloat3(gradients.gradPosition[contribution.endpointPrimitiveIndex],
                                                grad_cost_sp_sum);
                                if (settings.renderDebugGradientImages) {
                                    uint32_t pixelIndex = contribution.pixelIndex;
                                    atomicAddFloat4ToImage(
                                        &debugImage.framebufferOpacity[pixelIndex],
                                        float4{grad_cost_eta_sum}
                                    );
                                }
                            }
                        }
                    });
            }
        ).wait();
    }
}
