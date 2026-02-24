//
// Created by magnus on 9/8/25.
//

#include "Renderer/Kernels/AdjointKernels.h"

#include <cmath>

#include "AdjointGradientKernels.h"
#include "IntersectionKernels.h"
#include "Renderer/Kernels/KernelHelpers.h"


namespace Pale {
    void launchRayGenAdjointKernel(RenderPackage& pkg, int spp, uint32_t cameraIndex) {
        auto& queue = pkg.queue;
        auto& settings = pkg.settings;
        auto& intermediates = pkg.intermediates;
        auto& sensor = pkg.sensors[cameraIndex];

        const uint32_t imageWidth = sensor.camera.width;
        const uint32_t imageHeight = sensor.camera.height;
        uint32_t raysPerSet = imageWidth * imageHeight;
        float raysTotal = settings.adjointSamplesPerPixel * raysPerSet;


        queue.submit([&](sycl::handler& commandGroupHandler) {
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
                    const float4 residualRgba = sensor.framebuffer[pixelIndex];
                    float3 residual = float3{residualRgba.x(), residualRgba.y(), residualRgba.z()}; // (I - T)
                    float invPixelCount = 1.0f / float(raysPerSet); // W*H
                    float3 initialAdjointWeight = residual * invPixelCount;

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

                    if (isWatchedPixel(pixelX, pixelY)) {
                        int debug = 1;
                    }
                    else {
                        //primaryRay.direction = normalize(float3{-0.01, -1.0, 0.04}); // b
                        //primaryRay.origin = float3{0.0, -4.0, 1.0};
                    }


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


    void launchAdjointIntersectKernel(RenderPackage& pkg, uint32_t activeRayCount, uint32_t bounceIndex) {
        auto& queue = pkg.queue;
        auto& settings = pkg.settings;
        auto& intermediates = pkg.intermediates;
        auto& scene = pkg.scene;

        queue.submit([&](sycl::handler& cgh) {
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
                    if (rayState.pathId < intermediates.maxPendingAdjointStateCount) {
                        PendingAdjointState pending = intermediates.pendingAdjointStates[rayState.pathId];
                        /*
                        if (pending.kind != PendingAdjointKind::None) {
                            CompletedGradientEvent completed{};
                            completed.pathId = pending.pathId;
                            completed.kind = pending.kind;
                            completed.primitiveIndex = pending.primitiveIndex;
                            completed.instanceIndex = pending.instanceIndex;
                            completed.alphaGeom = pending.alphaGeom;
                            completed.hitPositionSurfel = pending.hitPosition;
                            completed.pathThroughput = pending.pathThroughput;
                            completed.pixelIndex = pending.pixelIndex;
                            completed.endPointAlphaGeom = worldHit.alphaGeom;
                            completed.endpointInstanceIndex = worldHit.instanceIndex;
                            completed.endpointPrimitiveIndex = worldHit.primitiveIndex;
                            completed.endpointPosition = worldHit.hitPositionW;
                            completed.endpointNormal = worldHit.geometricNormalW;
                            const InstanceRecord& endpointInstance = scene.instances[worldHit.instanceIndex];
                            completed.endpointType = endpointInstance.geometryType;
                            // check endpoint cosine
                            float3 worldHitNormal = worldHit.geometricNormalW;
                            if (endpointInstance.geometryType == GeometryType::PointCloud) {
                                const float3& canonicalNormalW = worldHitNormal;
                                const float signedCosineIncident = dot(canonicalNormalW, -rayState.ray.direction);
                                const int sideSign = signNonZero(signedCosineIncident);
                                worldHitNormal = static_cast<float>(sideSign) * worldHitNormal;
                            }
                            completed.endpointCosine = computeEndpointCosine(rayState.ray, worldHitNormal);
                            completed.endpointLightIndex = endpointInstance.geometryIndex;

                            appendCompletedGradientEventAtomic(
                                intermediates.countCompletedGradientEvents,
                                intermediates.completedGradientEvents,
                                intermediates.maxCompletedGradientEventCount,
                                completed);
                            // Clear pending in global memory
                            clearPendingAdjointState(intermediates.pendingAdjointStates[rayState.pathId]);
                        }
                        */
                    }
                    // Hitting mesh events
                    const auto& instance = scene.instances[worldHit.instanceIndex];
                    if (instance.geometryType == GeometryType::Mesh) {
                        // determine if we should make contributions from this position:
                        // Generate next ray
                        float3 sampledOutgoingDirectionW = rayState.ray.direction;
                        const GPUMaterial material = scene.materials[instance.materialIndex];
                        // If we hit instance was a mesh do ordinary BRDF stuff.
                        float sampledPdf = 0.0f;
                        sampleCosineHemisphere(rng128, worldHit.geometricNormalW, sampledOutgoingDirectionW,
                                               sampledPdf);
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
                    }
                    else {
                        // Random event
                        const float qNull = 0.5f;
                        const float qReflect = 0.5f;
                        const float qTransmit = 0.0f;
                        // qAbsorb = 1 - (qNull + qReflect + qTransmit)
                        const float u = rng128.nextFloat();
                        uint32_t eventType = 0; // 0=null, 1=reflect, 2=transmit, 3=absorb
                        if (u < qNull) {
                            // Update path throughput
                            eventType = 0;
                            const Point& surfel = scene.points[worldHit.primitiveIndex];
                            float attenuation = 1.0f - worldHit.alphaGeom * surfel.opacity;
                            float weight = attenuation / qNull;
                            RayState nextState{};
                            // Spawn next ray
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
                                pending.pathThroughput = rayState.pathThroughput / qNull;
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
                        }
                        else if (u < qNull + qReflect) {
                            // Generate next ray
                            const auto& surfel = scene.points[worldHit.primitiveIndex];
                            // Find which side we hit the surfel:
                            const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));
                            const float signedCosineIncident = dot(canonicalNormalW, -rayState.ray.direction);
                            const int sideSign = signNonZero(signedCosineIncident);
                            // If positive we hit the front side if negative we hit the backside
                            float3 orientedNormal = static_cast<float>(sideSign) * canonicalNormalW;
                            //Generate next ry
                            float3 sampledOutgoingDirectionW = rayState.ray.direction;
                            // If we hit instance was a mesh do ordinary BRDF stuff.
                            float sampledPdf = 0.0f;
                            sampleUniformHemisphereAroundNormal(rng128, orientedNormal, sampledOutgoingDirectionW,
                                                                sampledPdf);
                            const float3 f_s = surfel.alpha_r * surfel.albedo * M_1_PIf; // ρ/π
                            const float cosTheta = sycl::fmax(0.0f, dot(sampledOutgoingDirectionW, orientedNormal));
                            float alpha = worldHit.alphaGeom * surfel.opacity;
                            const float3& throughputMultiplier = ((alpha / qReflect) * (f_s * cosTheta)) / sampledPdf;

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
                                completed.pathThroughput = rayState.pathThroughput / qReflect;
                                completed.pixelIndex = rayState.pixelIndex;
                                completed.hasEndpoint = false;

                                appendCompletedGradientEventAtomic(
                                    intermediates.countCompletedGradientEvents,
                                    intermediates.completedGradientEvents,
                                    intermediates.maxCompletedGradientEventCount,
                                    completed);
                            }


                            if (rayState.pathId < intermediates.maxPendingAdjointStateCount) {
                                PendingAdjointState pending = intermediates.pendingAdjointStates[rayState.pathId];
                                if (pending.kind == PendingAdjointKind::ReflectScatter) {
                                    CompletedGradientEvent completed{};
                                    completed.pathId = pending.pathId;
                                    completed.kind = pending.kind;
                                    completed.primitiveIndex = pending.primitiveIndex;
                                    completed.instanceIndex = pending.instanceIndex;
                                    completed.alphaGeom = pending.alphaGeom;
                                    completed.hitPositionSurfel = pending.hitPosition;
                                    completed.pathThroughput = pending.pathThroughput;
                                    completed.pixelIndex = pending.pixelIndex;
                                    completed.cosineSurfel = dot(orientedNormal, -rayState.ray.direction);

                                    completed.endPointAlphaGeom = worldHit.alphaGeom;
                                    completed.endpointInstanceIndex = worldHit.instanceIndex;
                                    completed.endpointPrimitiveIndex = worldHit.primitiveIndex;
                                    completed.endpointPosition = worldHit.hitPositionW;
                                    completed.endpointNormal = orientedNormal;
                                    completed.endPointPDF = sampledPdf;
                                    const InstanceRecord& endpointInstance = scene.instances[worldHit.instanceIndex];
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

                            auto extensionCounter = sycl::atomic_ref<uint32_t,
                                                                     sycl::memory_order::relaxed,
                                                                     sycl::memory_scope::device,
                                                                     sycl::access::address_space::global_space>(
                                *intermediates.countExtensionOut);
                            const uint32_t outIndex = extensionCounter.fetch_add(1);
                            intermediates.extensionRaysA[outIndex] = nextState;
                        }
                        else if (u < qNull + qReflect + qTransmit) {
                            const auto& surfel = scene.points[worldHit.primitiveIndex];
                            float alpha = worldHit.alphaGeom * surfel.opacity;
                            float weight = (alpha * surfel.alpha_t) / qTransmit;
                            float3 throughput = rayState.pathThroughput * weight;
                            // Find which side we hit the surfel:
                            const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));
                            const float signedCosineIncident = dot(canonicalNormalW, -rayState.ray.direction);
                            const int sideSign = signNonZero(signedCosineIncident);
                            // If positive we hit the front side if negative we hit the backside
                            float3 orientedNormal = static_cast<float>(sideSign) * canonicalNormalW;

                            float3 sampledOutgoingDirectionW = rayState.ray.direction;
                            const float3& lambertBrdf = surfel.albedo;

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
                        }
                        else {
                            // Absorb: do not enqueue nextState, do not enqueue contribution.
                            return;
                        }
                    }
                });
        });
        queue.wait(); // DEBUG: ensure the thread blocks here
    }


    // -----------------------------------------------------------------------------
    // launchAdjointProjectionKernel
    // -----------------------------------------------------------------------------
    // Performs the adjoint *projection* pass for the camera.
    // This is the **camera → scene adjoint leg**, i.e., the reverse-mode analogue
    // of the forward sensor integration.
    //
    // Conceptually:
    //   • In the forward pass, the radiance emitter travels scene → camera.
    //   • In the adjoint pass, the adjoint source travels camera → scene.
    //
    void launchAdjointProjectionKernel(RenderPackage& pkg, uint32_t contributionCount, uint32_t cameraIndex) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& settings = pkg.settings;
        // Host-side (before launching kernel)
        SensorGPU& sensor = pkg.sensors[cameraIndex];
        auto* contributionRecords = pkg.intermediates.hitContribution;

        auto& gradients = pkg.gradients;
        const auto& photonMap = pkg.intermediates.map;
        auto* raysIn = pkg.intermediates.primaryRays;

        float invSpp = 1.0f / settings.adjointSamplesPerPixel;

        DebugImages& debugImage = pkg.debugImages[cameraIndex];

        queue.submit([&](sycl::handler& cgh) {
            uint64_t baseSeed = settings.random.number * (static_cast<uint64_t>(cameraIndex) + 5ull);

            cgh.parallel_for<class launchContributionKernel>(
                sycl::range<1>(contributionCount),
                // ReSharper disable once CppDFAUnusedValue
                [=](sycl::id<1> globalId) {
                    const uint32_t contributionIndex = globalId[0];
                    const HitInfoContribution& contribution = contributionRecords[contributionIndex];
                    const Point& surfel = scene.points[contribution.primitiveIndex];
                    const RayState& rayState = raysIn[contribution.rayIndex];

                    float cosine = dot(-rayState.ray.direction, contribution.geometricNormalW);

                    if (cosine <= 0.0f)
                        return;

                    const float3 rho = surfel.albedo;

                    const float3 E = gatherDiffuseIrradianceAtPoint(
                        contribution.hitPositionW,
                        contribution.geometricNormalW,
                        photonMap
                    );
                    const float3 f_r = surfel.alpha_r * rho * M_1_PIf * M_1_PIf;
                    const float3 Lo = f_r * E;

                    float3 p = contribution.pathThroughput * f_r;

                    float grad_alpha_eta = contribution.alphaGeom;

                    float3 grad_cost_eta = grad_alpha_eta * p * E;
                    float grad_cost_eta_sum = sum(grad_cost_eta) * invSpp;
                    //atomicAddFloat(gradients.gradOpacity[contribution.primitiveIndex], grad_cost_eta_sum);


                    const float2 uv = phiInverse(contribution.hitPositionW, surfel);
                    const float u = uv.x();
                    const float v = uv.y();
                    const float r2 = u * u + v * v;

                    const float oneMinusR2 = sycl::max(1e-6f, 1.0f - r2);
                    if (oneMinusR2 <= 0.0f) return; // or clamp; see note below
                    const float betaValue = 4.0f * sycl::exp(surfel.beta); // beta(b)
                    const float dAlphaGeomDb = contribution.alphaGeom * sycl::log(oneMinusR2) * betaValue;
                    const float grad_alpha_beta = dAlphaGeomDb * surfel.opacity; // because alpha = alphaGeom * eta


                    float3 grad_cost_beta = grad_alpha_beta * p * E;
                    float grad_cost_beta_sum = sum(grad_cost_beta) * invSpp;

                    atomicAddFloat(gradients.gradBeta[contribution.primitiveIndex], grad_cost_beta_sum);

                    //if (settings.renderDebugGradientImages) {
                    //    uint32_t pixelIndex = contribution.pixelIndex;
                    //    atomicAddFloat4ToImage(
                    //        &debugImage.framebufferOpacity[pixelIndex],
                    //        float4{grad_cost_eta_sum}
                    //    );
                    //    atomicAddFloat4ToImage(
                    //        &debugImage.framebufferBeta[pixelIndex],
                    //        float4{grad_cost_beta_sum}
                    //    );
                    //}
                });
        }).wait();
    }


    // Generate new ray


    // -----------------------------------------------------------------------------
    // launchAdjointTransportKernel
    // -----------------------------------------------------------------------------
    // Performs the adjoint *transport* pass, which traces importance through
    // scene interactions after projection has injected adjoint signals.
    // This is the **scene → scene adjoint leg**, corresponding to the reverse-
    // mode counterpart of the forward radiative transport (bounce handling,
    // visibility, occlusion, and transmittance propagation).
    //
    // Conceptually:
    //   • Handles adjoint propagation along the light-transport graph.
    //   • Applies the full volumetric / surfel transmittance model, matching
    //     the forward leg’s compositional structure.
    //   • Derivatives from both the scatter leg and the shadow/light leg are
    //     included, following the unified treatment where transmittance τ(Pi)

    void launchAdjointTransportKernel(RenderPackage& pkg, uint32_t contributionTransmittanceCount,
                                      uint32_t cameraIndex) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& settings = pkg.settings;
        // Host-side (before launching kernel)
        SensorGPU& sensor = pkg.sensors[cameraIndex];
        auto* contributionRecords = pkg.intermediates.completedGradientEvents;

        auto& gradients = pkg.gradients;
        const auto& photonMap = pkg.intermediates.map;
        auto* raysIn = pkg.intermediates.primaryRays;

        float invSpp = 1.0f / settings.adjointSamplesPerPixel;

        DebugImages& debugImage = pkg.debugImages[cameraIndex];

        queue.submit([&](sycl::handler& cgh) {
                uint64_t baseSeed = settings.random.number * (static_cast<uint64_t>(cameraIndex) + 5ull);
                cgh.parallel_for<class launchContributionKernel>(
                    sycl::range<1>(contributionTransmittanceCount),
                    [=](sycl::id<1> globalId) {
                        const uint32_t contributionIndex = globalId[0];
                        CompletedGradientEvent& contribution = contributionRecords[contributionIndex];
                        if (contribution.kind == PendingAdjointKind::ProjectionScatter) {
                            const Point& surfel = scene.points[contribution.primitiveIndex];

                            //float cosine = contribution.endpointCosine;
                            //if (cosine <= 0.0f)
                            //    return;

                            const float3 rho = surfel.albedo;
                            const float3 E = gatherDiffuseIrradianceAtPoint(
                                contribution.hitPositionSurfel,
                                contribution.hitNormalSurfel,
                                photonMap
                            );
                            const float3 f_r = surfel.alpha_r * rho * M_1_PIf * M_1_PIf;
                            const float3 Lo = f_r * E;
                            float3 p = contribution.pathThroughput * f_r;
                            float grad_alpha_eta = contribution.alphaGeom;
                            float3 grad_cost_eta = grad_alpha_eta * p * E;
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


                        if (contribution.kind == PendingAdjointKind::NullTransmittance) {
                            const Point& surfel = scene.points[contribution.primitiveIndex];
                            const auto& instance = scene.instances[contribution.endpointInstanceIndex];
                            const GPUMaterial material = scene.materials[instance.materialIndex];
                            float3 Lr = gatherDiffuseIrradianceAtPoint(
                                contribution.endpointPosition,
                                contribution.endpointNormal,
                                photonMap
                            ) * material.baseColor * M_1_PIf;
                            float3 Le = {0.0f, 0.0f, 0.0f};
                            if (material.isEmissive()) {
                                GPULightRecord emitter = scene.lights[0];
                                Le = material.baseColor * (material.power / (M_PIf * emitter.totalAreaWorld));
                            }
                            const float3 Lo = Le + Lr;
                            float grad_tau_eta = -contribution.alphaGeom;
                            float cosine = contribution.endpointCosine;
                            float3 p = contribution.pathThroughput * cosine;
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
                            const Point& surfel = scene.points[contribution.endpointPrimitiveIndex];
                            const auto& instance = scene.instances[contribution.instanceIndex];
                            const GPUMaterial material = scene.materials[instance.materialIndex];

                            float3 Lo = gatherDiffuseIrradianceAtPoint(
                                contribution.endpointPosition,
                                contribution.endpointNormal,
                                photonMap
                            ) * material.baseColor * M_1_PIf;

                            const float3 f_s = surfel.alpha_r * surfel.albedo * M_1_PIf * M_1_PIf;
                            const float cosTheta = max(0.0f, contribution.endpointCosine);
                            float qReflect = 0.5f;
                            const float3 p = contribution.pathThroughput
                                * (contribution.endPointAlphaGeom / qReflect)
                                * f_s * cosTheta / contribution.endPointPDF;

                            const float3 grad_cost_eta = p * Lo;

                            float grad_cost_eta_sum = sum(grad_cost_eta) * invSpp;
                            atomicAddFloat(gradients.gradOpacity[contribution.endpointPrimitiveIndex],
                                           grad_cost_eta_sum);
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

    void generateNextAdjointRays(RenderPackage& pkg, uint32_t activeRayCount) {
        auto& queue = pkg.queue;
        auto& sensor = pkg.sensors;
        auto& settings = pkg.settings;
        auto& scene = pkg.scene;
        auto& photonMap = pkg.intermediates.map;

        auto* hitRecords = pkg.intermediates.hitRecords;
        auto* raysIn = pkg.intermediates.primaryRays;
        auto* raysOut = pkg.intermediates.extensionRaysA;
        auto* countExtensionOut = pkg.intermediates.countExtensionOut;

        const uint32_t perPassRayCount = activeRayCount; // total number of rays (With n-samples per ray)
        const uint32_t perPixelRayCount = activeRayCount; // Number of rays per pixel
        const uint32_t photonsPerLaunch = settings.photonsPerLaunch;

        /*
        queue.submit([&](sycl::handler &cgh) {
            const uint64_t baseSeed = settings.randomSeed;
            cgh.parallel_for<class GenerateNextAdjointRays>(
                sycl::range<1>(perPixelRayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t seed = rng::makePerItemSeed1D(baseSeed, rayIndex);
                    rng::Xorshift128 rng128(seed);
                    const RayState rayState = raysIn[rayIndex];
                    const WorldHit worldHit = hitRecords[rayIndex];
                    RayState nextState{};
                    if (!worldHit.hit) {
                        return;
                    } // dead ray

                    const InstanceRecord &instance = scene.instances[worldHit.instanceIndex];
                    const float3 canonicalNormalW = worldHit.geometricNormalW;
                    const int travelSideSign = signNonZero(dot(canonicalNormalW, -rayState.ray.direction));
                    const float3 enteredSideNormalW =
                            (travelSideSign >= 0) ? canonicalNormalW : (-canonicalNormalW);
                    float3 sampledOutgoingDirectionW{};
                    float3 throughputMultiplier = float3(1.0f);

                    // If we hit instance was a mesh do ordinary BRDF stuff.
                    if (instance.geometryType == GeometryType::Mesh) {
                        float sampledPdf = 0.0f;
                        const GPUMaterial material = scene.materials[instance.materialIndex];
                        sampleCosineHemisphere(rng128, enteredSideNormalW, sampledOutgoingDirectionW, sampledPdf);
                        sampledPdf = sycl::fmax(sampledPdf, 1e-6f);
                        const float cosTheta = sycl::fmax(0.0f, dot(sampledOutgoingDirectionW, enteredSideNormalW));
                        const float3 lambertBrdf = material.baseColor * M_1_PIf;
                        throughputMultiplier = lambertBrdf * (cosTheta / sampledPdf) * worldHit.transmissivity;
                    }
                    // If our hit instance was a point cloud it means we hit a surfel
                    // Now we do either BRDF or BTDF
                    bool reflectedRay = false;
                    if (instance.geometryType == GeometryType::PointCloud) {
                        // PointCloud
                        GPUMaterial material{};
                        auto &surfel = scene.points[worldHit.primitiveIndex];
                        float alpha = worldHit.splatEvents[worldHit.splatEventCount - 1].alpha;
                        material.baseColor = surfel.albedo;
                        const float3 lambertBrdf = material.baseColor * M_1_PIf;

                        const float interactionAlpha = worldHit.splatEvents[worldHit.splatEventCount - 1].alpha;
                        // α
                        float probReflect = 1.0f;
                        const bool reflectedRay = (rng128.nextFloat() < probReflect);
                        // event probabilities
                        // a 50/50 if we reflect or transmit
                        if (reflectedRay) {
                            float sampledPdf = 0.0f;
                            // Diffuse reflect on entered side
                            sampleCosineHemisphere(rng128, enteredSideNormalW, sampledOutgoingDirectionW,
                                                   sampledPdf);
                            sampledPdf = sycl::fmax(sampledPdf, 1e-6f);
                            const float cosTheta = sycl::fmax(
                                0.0f, dot(sampledOutgoingDirectionW, enteredSideNormalW));
                            throughputMultiplier =
                                    throughputMultiplier * lambertBrdf * (cosTheta / sampledPdf) * surfel.opacity *
                                    alpha;
                        } else {
                            float sampledPdf = 0.0f;
                            // Diffuse transmit: cosine hemisphere on the opposite side
                            const float3 oppositeSideNormalW = -enteredSideNormalW;
                            sampleCosineHemisphere(rng128, oppositeSideNormalW, sampledOutgoingDirectionW,
                                                   sampledPdf);
                            sampledPdf = sycl::fmax(sampledPdf, 1e-6f);
                            const float cosTheta =
                                    sycl::fmax(0.0f, dot(sampledOutgoingDirectionW, oppositeSideNormalW));
                            throughputMultiplier =
                                    throughputMultiplier * lambertBrdf * (cosTheta / sampledPdf) * surfel.opacity *
                                    alpha;;
                        }
                    }


                    // Offset origin robustly
                    constexpr float kEps = 1e-5f;
                    nextState.ray.origin = worldHit.hitPositionW + enteredSideNormalW * kEps;
                    nextState.ray.direction = sampledOutgoingDirectionW;
                    nextState.ray.normal = worldHit.geometricNormalW;
                    nextState.bounceIndex = rayState.bounceIndex + 1;
                    nextState.pixelIndex = rayState.pixelIndex;
                    nextState.pathThroughput = rayState.pathThroughput * throughputMultiplier;


                    // compacted enqueue
                    sycl::atomic_ref<uint32_t,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                            activeCounter(*countExtensionOut);

                    uint32_t outputSlot = activeCounter.fetch_add(1); // write only once to compacted slot
                    raysOut[outputSlot] = nextState;
                });
        });

        queue.wait();
        */
    }
}
