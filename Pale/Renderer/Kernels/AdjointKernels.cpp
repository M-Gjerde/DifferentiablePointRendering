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

        //raysPerSet = 1;
        float raysTotal = settings.adjointSamplesPerPixel * raysPerSet;

        const uint32_t perPassRayCount = raysPerSet;
        queue.memcpy(pkg.intermediates.countPrimary, &perPassRayCount, sizeof(uint32_t)).wait();

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
                    const float4 residualRgba = sensor.framebuffer[pixelIndex];
                    float3 residual = float3{residualRgba.x(), residualRgba.y(), residualRgba.z()}; // (I - T)
                    float invPixelCount = 1.0f / float(raysTotal);
                    float3 initialAdjointWeight = residual * invPixelCount;

                    // Or unit weights:
                    //initialAdjointWeight = float3(1.0f, 1.0f, 1.0f);

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

                    RayState rayState{};

                    rayState.ray = primaryRay;
                    rayState.pathThroughput = initialAdjointWeight;
                    rayState.bounceIndex = 0;
                    rayState.pixelIndex = pixelIndex;

                    if (pixelX == 200 && pixelY == (sensor.height - 1 - 325)) {
                        int debug = 1;
                    }

                    intermediates.primaryRays[baseOutputSlot] = rayState;
                });
        }).wait();
    }


    void launchAdjointIntersectKernel(RenderPackage &pkg, uint32_t activeCount) {
        auto &queue = pkg.queue;
        auto &settings = pkg.settings;
        auto &intermediates = pkg.intermediates;
        auto &scene = pkg.scene;

        struct ChosenScatterEvent {
            float scalarWeight = 1.0f;
            bool hasValue = false;
        };

        queue.submit([&](sycl::handler &commandGroupHandler) {
            commandGroupHandler.parallel_for<struct RayGenAdjointKernelTag>(
                sycl::range<1>(activeCount),
                [=](sycl::id<1> globalId) {
                    ChosenScatterEvent chosen;
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(settings.random.number, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);


                    RayState &rayState = intermediates.primaryRays[rayIndex];

                    WorldHit worldHit{};
                    intersectScene(rayState.ray, &worldHit, scene, rng128, SurfelIntersectMode::Uniform);
                    if (!worldHit.hit) {
                        intermediates.hitRecords[rayIndex] = worldHit;
                        return;
                    }
                    buildIntersectionNormal(scene, worldHit);

                    intermediates.hitRecords[rayIndex] = worldHit;
                });
        }).wait();
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
    void launchAdjointProjectionKernel(RenderPackage &pkg, uint32_t activeRayCount, uint32_t cameraIndex) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto &intermediates = pkg.intermediates;
        auto &gradients = pkg.gradients;
        const auto &photonMap = pkg.intermediates.map;
        auto *raysIn = pkg.intermediates.primaryRays;

        auto &sensor = pkg.sensors[cameraIndex];
        DebugImages &debugImage = pkg.debugImages[cameraIndex];

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<struct AdjointShadeKernelTag>(
                sycl::range<1>(activeRayCount),
                // ReSharper disable once CppDFAUnusedValue
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(settings.random.number, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);
                    RayState &rayState = intermediates.primaryRays[rayIndex];
                    WorldHit &worldHit = intermediates.hitRecords[rayIndex];

                    if (!worldHit.hit || !worldHit.hitSurfel) {
                        return;
                    }
                    // Transmission Gradients

                    // Check how many surfels we have along our ray

                    // Retrace the ray to get the correct attenuation?

                    float3 p = rayState.pathThroughput * worldHit.transmissivity * worldHit.invChosenSurfelPdf;

                    const Point &surfel = scene.points[worldHit.primitiveIndex];
                    const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));
                    const int travelSideSign = signNonZero(dot(canonicalNormalW, -rayState.ray.direction));
                    const float3 frontNormalW = canonicalNormalW * float(travelSideSign);
                    const float3 rho = surfel.albedo;
                    const float3 E = gatherDiffuseIrradianceAtPoint(
                        worldHit.hitPositionW,
                        frontNormalW,
                        photonMap,
                        travelSideSign,
                        true
                    );

                    float3 irradiance = E;
                    //float3 irradiance = float3{1.0f};
                    // If 1 we blend with background color

                    float grad_alpha_eta = worldHit.alpha;

                    float3 grad_cost_eta = grad_alpha_eta * p * irradiance;
                    float grad_cost_eta_sum = sum(grad_cost_eta);

                    atomicAddFloat(gradients.gradOpacity[worldHit.primitiveIndex], grad_cost_eta_sum);

                    if (settings.renderDebugGradientImages) {
                        uint32_t pixelIndex = rayState.pixelIndex;
                        atomicAddFloat4ToImage(
                        &debugImage.framebufferOpacity[pixelIndex],
                        float4{grad_cost_eta_sum}
                    );
                    }
                    // If 2, we can get transmission gradients for first surfel.
                });
        }).wait();

        queue.wait();
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

    void launchAdjointTransportKernel(RenderPackage &pkg, uint32_t activeRayCount, uint32_t cameraIndex) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto &gradients = pkg.gradients;
        auto &photonMap = pkg.intermediates.map;
        auto *raysIn = pkg.intermediates.primaryRays;
        auto *hitRecords = pkg.intermediates.hitRecords;
        auto &debugImage = pkg.debugImages[cameraIndex];

        auto &sensor = pkg.sensors[cameraIndex];

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<struct AdjointShadeKernelTag>(
                sycl::range<1>(activeRayCount),
                // ReSharper disable once CppDFAUnusedValue
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(settings.random.number, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);
                    const RayState &rayState = raysIn[rayIndex];
                    const WorldHit &worldHit = hitRecords[rayIndex];
                    // Shoot one transmit ray. The amount intersected here will tell us how many scatter rays we will transmit.
                    if (!worldHit.hit) {
                        return;
                    }
                    uint32_t numShadowRays = 1;
                    shadowRay(scene, rayState, worldHit, gradients, debugImage, photonMap, rng128,
                              settings.renderDebugGradientImages, numShadowRays);
                });
        });
        queue.wait();
    }

    void generateNextAdjointRays(RenderPackage &pkg, uint32_t activeRayCount) {
        auto &queue = pkg.queue;
        auto &sensor = pkg.sensors;
        auto &settings = pkg.settings;
        auto &scene = pkg.scene;
        auto &photonMap = pkg.intermediates.map;

        auto *hitRecords = pkg.intermediates.hitRecords;
        auto *raysIn = pkg.intermediates.primaryRays;
        auto *raysOut = pkg.intermediates.extensionRaysA;
        auto *countExtensionOut = pkg.intermediates.countExtensionOut;

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
        */
        queue.wait();
    }
}
