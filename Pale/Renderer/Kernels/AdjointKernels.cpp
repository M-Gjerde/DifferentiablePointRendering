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
            const uint64_t randomNumber = settings.random.number;

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
                    const uint64_t perPixelSeed = rng::makePerItemSeed1D(randomNumber, pixelLinearIndexWithinImage);
                    rng::Xorshift128 pixelRng(perPixelSeed);

                    // Adjoint source weight
                    const float4 dLoss_dI = sensor.framebuffer[pixelIndex];
                    float3 dLoss_dI3 = float3{dLoss_dI.x(), dLoss_dI.y(), dLoss_dI.z()}; // (I - T)
                    float3 initialAdjointWeight = dLoss_dI3;

                    // Base slot for this pixel’s N samples
                    const uint32_t baseOutputSlot = pixelIndex;
                    // --- Sample 0: forced Transmit (background path) ---
                    const float jitterX = pixelRng.nextFloat() - 0.5f;
                    const float jitterY = pixelRng.nextFloat() - 0.5f;

                    Ray primaryRay = makePrimaryRayFromPixelJitteredFov(
                        sensor.camera,
                        static_cast<float>(pixelX),
                        static_cast<float>(pixelY),
                        jitterX, jitterY
                    );

                    //primaryRay.direction = normalize(float3{-0.001, 0.982122211, 0.277827293});    // a
                    //primaryRay.direction = normalize(float3{-0.01, 1.0, 0.04}); // b
                    //primaryRay.origin = float3{0.0, -4.0, 1.0};

                    //if (isWatchedPixel(pixelX, pixelY)) {
                    //    int debug = 1;
                    //} else {
                    //    primaryRay.direction = normalize(float3{-0.00, -1.0, 0.00}); // b
                    //    primaryRay.origin = float3{0.0, -400.0, 1.0};
                    //}


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
            uint64_t randomNumber = settings.random.number;
            cgh.parallel_for<class launchIntersectKernel>(
                sycl::range<1>(activeRayCount),
                // ReSharper disable once CppDFAUnusedValue
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(randomNumber, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);
                    WorldHit worldHit{};
                    RayState rayState = intermediates.primaryRays[rayIndex];
                    intersectScene(rayState.ray, &worldHit, scene, rng128, SurfelIntersectMode::FirstHit);
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
                    sampleCosineHemisphere(rng128, worldHit.geometricNormalW, sampledOutgoingDirectionW,
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
                        sampleUniformHemisphereAroundNormal(rng128, orientedNormal, sampledOutgoingDirectionW,
                                                            sampledOutgoingDirectionPDF);
                    }

                    if (rayState.pathId < intermediates.maxPendingAdjointStateCount) {
                        PendingAdjointState pending = intermediates.pendingAdjointStates[rayState.pathId];
                        if (pending.kind == PendingAdjointKind::ReflectScatter && endpointInstance.geometryType == GeometryType::PointCloud) {
                            // Obtain normal:
                            CompletedGradientEvent completed{};
                            completed.pathId = pending.pathId;
                            completed.kind = pending.kind;
                            completed.primitiveIndex = pending.primitiveIndex;
                            completed.instanceIndex = pending.instanceIndex;
                            completed.alphaGeom = pending.alphaGeom;
                            completed.hitPositionSurfel = pending.hitPosition;
                            completed.pathThroughput = pending.pathThroughput * settings.sampling.qReflect; // Match the expected value only qReflect paths will actually scatter, some might transmit or attenuate
                            completed.pixelIndex = pending.pixelIndex;
                            completed.cosineSurfel = dot(orientedNormal, -rayState.ray.direction);

                            completed.endPointAlphaGeom = worldHit.alphaGeom;
                            completed.endpointInstanceIndex = worldHit.instanceIndex;
                            completed.endpointPrimitiveIndex = worldHit.primitiveIndex;
                            completed.endpointPosition = worldHit.hitPositionW;
                            completed.endpointNormal = orientedNormal;
                            completed.endPointPDF = sampledOutgoingDirectionPDF;
                            completed.endpointType = endpointInstance.geometryType;
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
                            completed.hitPositionSurfel = pending.hitPosition;
                            completed.pathThroughput = pending.pathThroughput * settings.sampling.qNull; // Match the expected value only qReflect paths will actually scatter, some might transmit or attenuate
                            completed.pixelIndex = pending.pixelIndex;
                            completed.cosineSurfel = dot(orientedNormal, -rayState.ray.direction);
                            completed.endPointAlphaGeom = worldHit.alphaGeom;
                            completed.endpointInstanceIndex = worldHit.instanceIndex;
                            completed.endpointPrimitiveIndex = worldHit.primitiveIndex;
                            completed.endpointPosition = worldHit.hitPositionW;
                            completed.endpointNormal = orientedNormal;
                            completed.endPointPDF = sampledOutgoingDirectionPDF;
                            completed.endpointType = endpointInstance.geometryType;
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
                        if (!applyRussianRoulette(rng128, nextState.bounceIndex, nextState.pathThroughput,
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
                        const float u = rng128.nextFloat();
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
                            if (!applyRussianRoulette(rng128, nextState.bounceIndex, nextState.pathThroughput,
                                                      settings.russianRouletteStart))
                                return;
                            // Populate pending (now that the path survives)
                            if (rayState.pathId < intermediates.maxPendingAdjointStateCount) {
                                PendingAdjointState pending{};
                                pending.kind = PendingAdjointKind::NullTransmittance;
                                pending.primitiveIndex = worldHit.primitiveIndex;
                                pending.alphaGeom = worldHit.alphaGeom;
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
                            const float3 &throughputMultiplier = ((alpha / settings.sampling.qReflect) * (f_s * cosTheta)) / sampledOutgoingDirectionPDF;
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
                            if (!applyRussianRoulette(rng128, nextState.bounceIndex, nextState.pathThroughput,
                                                      settings.russianRouletteStart))
                                return;

                            // Projection kernels
                            if (bounceIndex == 0 && rayState.pathId < intermediates.maxPendingAdjointStateCount) {
                                CompletedGradientEvent completed{};
                                completed.kind = PendingAdjointKind::ProjectionScatter;
                                completed.primitiveIndex = worldHit.primitiveIndex;
                                completed.alphaGeom = worldHit.alphaGeom;
                                completed.hitPositionSurfel = worldHit.hitPositionW;
                                completed.hitNormalSurfel = orientedNormal; // important
                                completed.cosineSurfel = dot(-rayState.ray.direction, orientedNormal); // important
                                completed.pathThroughput = rayState.pathThroughput / settings.sampling.qReflect;
                                completed.pixelIndex = rayState.pixelIndex;
                                completed.hasEndpoint = false;
                                appendCompletedGradientEventAtomic(
                                    intermediates.countCompletedGradientEvents,
                                    intermediates.completedGradientEvents,
                                    intermediates.maxCompletedGradientEventCount,
                                    completed);
                            }

                            auto extensionCounter = sycl::atomic_ref<uint32_t,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>(
                                *intermediates.countExtensionOut);
                            const uint32_t outIndex = extensionCounter.fetch_add(1);
                            intermediates.extensionRaysA[outIndex] = nextState;


                        } else if (u < settings.sampling.qNull + settings.sampling.qReflect + settings.sampling.qTransmit) {
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
                            sampleUniformHemisphereAroundNormal(rng128, orientedNormal, sampledOutgoingDirectionW,
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

                            if (!applyRussianRoulette(rng128, nextState.bounceIndex, nextState.pathThroughput,
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
                uint64_t baseSeed = settings.random.number * (static_cast<uint64_t>(cameraIndex) + 5ull);
                cgh.parallel_for<class launchContributionKernel>(
                    sycl::range<1>(contributionTransmittanceCount),
                    [=](sycl::id<1> globalId) {
                        const uint32_t contributionIndex = globalId[0];
                        CompletedGradientEvent &contribution = contributionRecords[contributionIndex];
                        if (contribution.kind == PendingAdjointKind::ProjectionScatter) {
                            const Point &surfel = scene.points[contribution.primitiveIndex];
                            const float3 E = gatherDiffuseIrradianceAtPoint(
                                contribution.hitPositionSurfel,
                                contribution.hitNormalSurfel,
                                photonMap
                            );

                            // Evaluate surfel outgoing radiance (direct/indirect via photon map)
                            const float3 f_r = surfel.alpha_r * surfel.albedo * M_1_PIf; // Lambert BRDF
                            const float3 Lr = f_r * E;
                            // If you also include emissive term at that surfel, add it here (Le)
                            const float3 Lo = Lr; // + Le if applicable
                            // opacity alpha = alphaGeom * eta  => dLo/deta = alphaGeom * Lo
                            const float grad_alpha_eta = contribution.alphaGeom;
                            // p should be the adjoint weight carried from the camera (residual etc.)
                            // DO NOT multiply p by f_r again.
                            const float3 grad_rgb = grad_alpha_eta * contribution.pathThroughput * Lo;
                            const float grad_cost_eta_sum = sum(grad_rgb) * invSpp; // only if you truly have spp samples
                            atomicAddFloat(gradients.gradOpacity[contribution.primitiveIndex], grad_cost_eta_sum);
                            if (settings.renderDebugGradientImages) {
                                uint32_t pixelIndex = contribution.pixelIndex;
                                atomicAddFloat4ToImage(
                                    &debugImage.framebufferOpacity[pixelIndex],
                                    float4{grad_cost_eta_sum}
                                );
                            }
                        }


                        if (contribution.kind == PendingAdjointKind::NullTransmittance) {
                            const Point &surfel = scene.points[contribution.primitiveIndex];
                            const auto &instance = scene.instances[contribution.endpointInstanceIndex];
                            const GPUMaterial material = scene.materials[instance.materialIndex];
                            float3 Lr = gatherDiffuseIrradianceAtPoint(
                                            contribution.endpointPosition,
                                            contribution.endpointNormal,
                                            photonMap
                                        ) * material.baseColor * M_1_PIf;

                            float3 Le = {0.0f, 0.0f, 0.0f};
                            if (material.isEmissive()) {
                                float cosine = contribution.endpointCosine;
                                GPULightRecord emitter = scene.lights[0];
                                const float3 flux = material.baseColor * material.power;
                                const float invArea = 1.0f / emitter.totalAreaWorld;
                                Le = flux * (invArea * M_1_PIf);
                                Le = material.baseColor * (material.power / (M_PIf * emitter.totalAreaWorld)) * cosine;
                            }
                            const float3 Lo = Le + Lr;
                            float grad_tau_eta = -contribution.alphaGeom;
                            float3 p = contribution.pathThroughput;
                            float3 grad_cost_eta = grad_tau_eta * p * Lo;

                            float grad_cost_eta_sum = sum(grad_cost_eta) * invSpp;
                            atomicAddFloat(gradients.gradOpacity[contribution.primitiveIndex], grad_cost_eta_sum);
                            if (settings.renderDebugGradientImages) {
                                uint32_t pixelIndex = contribution.pixelIndex;
                                atomicAddFloat4ToImage(
                                    &debugImage.framebufferOpacity[pixelIndex],
                                    float4{grad_cost_eta_sum}
                                );
                            }
                        }

                        if (contribution.kind == PendingAdjointKind::ReflectScatter) {
                            const Point &surfel = scene.points[contribution.endpointPrimitiveIndex];
                            const auto &instance = scene.instances[contribution.instanceIndex];
                            const GPUMaterial material = scene.materials[instance.materialIndex];

                            float3 Lo = gatherDiffuseIrradianceAtPoint(
                                            contribution.endpointPosition,
                                            contribution.endpointNormal,
                                            photonMap
                                        ) * material.baseColor * M_1_PIf;

                            const float cosTheta =contribution.endpointCosine;
                            const float3 p = contribution.pathThroughput / contribution.endPointPDF;
                            const float3 grad_cost_eta = p * Lo * contribution.endPointAlphaGeom;

                            float grad_cost_eta_sum = sum(grad_cost_eta) * invSpp;
                            atomicAddFloat(gradients.gradOpacity[contribution.endpointPrimitiveIndex], grad_cost_eta_sum);
                            if (settings.renderDebugGradientImages) {
                                uint32_t pixelIndex = contribution.pixelIndex;
                                atomicAddFloat4ToImage(
                                    &debugImage.framebufferOpacity[pixelIndex],
                                    float4{grad_cost_eta_sum}
                                );
                            }
                        }
                    });
            }
        ).wait();
    }
}
