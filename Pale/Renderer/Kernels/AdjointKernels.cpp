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
        auto &sensor = pkg.sensor[cameraIndex];

        const uint32_t imageWidth = sensor.camera.width;
        const uint32_t imageHeight = sensor.camera.height;
        uint32_t raysPerSet = imageWidth * imageHeight;

        //raysPerSet = 1;
        float raysTotal = settings.adjointSamplesPerPixel * raysPerSet;

        const uint32_t perPassRayCount = raysPerSet;
        queue.memcpy(pkg.intermediates.countPrimary, &perPassRayCount, sizeof(uint32_t)).wait();

        queue.submit([&](sycl::handler &commandGroupHandler) {
            const uint64_t baseSeed = settings.randomSeed * static_cast<uint64_t>(spp);

            commandGroupHandler.parallel_for<struct RayGenAdjointKernelTag>(
                sycl::range<1>(raysPerSet),
                [=](sycl::id<1> globalId) {
                    const auto globalRayIndex = static_cast<uint32_t>(globalId[0]);
                    // Map to pixel
                    const uint32_t pixelLinearIndexWithinImage = globalRayIndex; // 0..raysPerSet-1
                    uint32_t pixelX = pixelLinearIndexWithinImage % imageWidth;
                    uint32_t pixelY = pixelLinearIndexWithinImage / imageWidth;


                    uint32_t index = flippedYLinearIndex(pixelLinearIndexWithinImage, sensor.width, sensor.height);

                    const uint32_t pixelIndex = index;
                    // RNG for this pixel
                    const uint64_t perPixelSeed = rng::makePerItemSeed1D(baseSeed, pixelLinearIndexWithinImage);
                    rng::Xorshift128 pixelRng(perPixelSeed);

                    // Adjoint source weight
                    const float4 residualRgba = sensor.framebuffer[pixelIndex];
                    float3 initialAdjointWeight = float3{
                                                      residualRgba.x(), residualRgba.y(), residualRgba.z()
                                                  } / raysTotal;
                    // Or unit weights:
                    //initialAdjointWeight = float3(1.0f, 1.0f, 1.0f);

                    // Base slot for this pixel’s N samples
                    const uint32_t baseOutputSlot = pixelIndex;

                    // --- Sample 0: forced Transmit (background path) ---
                    const float jitterX = pixelRng.nextFloat() - 0.5f;
                    const float jitterY = pixelRng.nextFloat() - 0.5f;


                    Ray primaryRay = makePrimaryRayFromPixelJittered(
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
                    rayState.pixelX = pixelX;
                    rayState.pixelY = pixelY;

                    intermediates.primaryRays[baseOutputSlot] = rayState;
                });
        }).wait();
    }

    struct DebugPixel {
        uint32_t pixelY;
        uint32_t pixelX;
    };

    DebugPixel kDebugPixels[] = {
        {300, 500},
        {301, 299},
        // {700, 250},
        // {700, 330},
        // {700, 400},
        // {700, 530},
    };

    bool isWatchedPixel(uint32_t pixelX, uint32_t pixelY) {
        bool isMatch = false;

        for (uint32_t i = 0; i < 2; ++i) {
            const DebugPixel &debugPixel = kDebugPixels[i];
            if (pixelY == debugPixel.pixelY && pixelX == debugPixel.pixelX) {
                isMatch = true;
            }
        }
        return isMatch;
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
    // This kernel traces adjoint rays originating at the camera, propagating
    // adjoint through the surfels using the same
    // **volumetric compositing model** as in the camera gather stage.
    // The compositing uses the formulation where
    // BRDF-like and BTDF-like contributions are handled as a single operator
    // rather than explicitly separating surface/refraction events.
    //
    void launchAdjointProjectionKernel(RenderPackage &pkg, uint32_t activeRayCount, uint32_t cameraIndex) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto &intermediates = pkg.intermediates;
        auto &gradients = pkg.gradients;
        auto &photonMap = pkg.intermediates.map;
        auto *raysIn = pkg.intermediates.primaryRays;

        auto &sensor = pkg.sensor[cameraIndex];
        auto &debugImage = pkg.debugImages[cameraIndex];

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<struct AdjointShadeKernelTag>(
                sycl::range<1>(activeRayCount),
                // ReSharper disable once CppDFAUnusedValue
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(settings.randomSeed, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);

                    RayState &rayState = intermediates.primaryRays[rayIndex];
                    uint32_t pixelX = rayState.pixelX;
                    uint32_t pixelY = sensor.height - 1 - rayState.pixelY;
                    bool isWatched = false;
                    if (pixelX == 410 && pixelY == 430) {
                        isWatched = true;
                        int debug = 1;
                    }
                    if (pixelX == 200 && pixelY == 200) {
                        isWatched = true;
                        int debug = 1;
                    }

                    // Shoot one transmit ray. The amount intersected here will tell us how many scatter rays we will transmit.
                    WorldHit whTransmit{};
                    intersectScene(rayState.ray, &whTransmit, scene, rng128, RayIntersectMode::Transmit);
                    if (!whTransmit.hit)
                        return;
                    buildIntersectionNormal(scene, whTransmit);

                    const InstanceRecord &meshInstance = scene.instances[whTransmit.instanceIndex];

                    //uint32_t debugIndex = 1;
                    uint32_t debugIndex = UINT32_MAX;
                    //debugIndex = 0;
                    uint32_t numShadowRays = 10;
                    // Transmission gradients with shadow rays
                    if (meshInstance.geometryType == GeometryType::Mesh) {
                        // Transmission
                        // Cost weighting: photon-map radiance for this segment
                        for (uint32_t i = 0; i < whTransmit.splatEventCount; ++i) {
                            // Forward-side values reused in adjoint:
                            float3 L_Mesh = estimateRadianceFromPhotonMap(whTransmit, scene, photonMap);
                            auto &splatEvent = whTransmit.splatEvents[i]; // one surfel
                            auto &surfel = scene.points[splatEvent.primitiveIndex];

                            float alphaGeom = splatEvent.alpha; // α(u,v,β)
                            float eta = surfel.opacity;
                            float alphaEff = eta * alphaGeom; // α_eff

                            // Shade surfel from photon map (front/back)
                            const bool useOneSidedScatter = true;
                            float3 surfelRadianceFront =
                                    estimateSurfelRadianceFromPhotonMap(
                                        splatEvent,
                                        rayState.ray.direction,
                                        scene,
                                        photonMap,
                                        useOneSidedScatter);

                            float3 surfelRadianceBack =
                                    estimateSurfelRadianceFromPhotonMap(
                                        splatEvent,
                                        -rayState.ray.direction,
                                        scene,
                                        photonMap,
                                        useOneSidedScatter);

                            float3 L_Surfel = (surfelRadianceFront * 0.5f + surfelRadianceBack * 0.5f);
                            // Geometry for u,v and Jacobian:

                            float3 canonicalNormalWorld = normalize(cross(surfel.tanU, surfel.tanV));
                            const float travelSideSign =
                                    signNonZero(dot(canonicalNormalWorld, -rayState.ray.direction));
                            canonicalNormalWorld = canonicalNormalWorld * travelSideSign;
                            float3 hitWorld = splatEvent.hitWorld;
                            float2 uv = phiInverse(hitWorld, surfel);
                            float u = uv.x();
                            float v = uv.y();
                            float r2 = u * u + v * v;
                            float su = surfel.scale.x();
                            float sv = surfel.scale.y();

                            float3 DuvDPosition = computeDuvDPosition(
                                surfel.tanU,
                                surfel.tanV,
                                canonicalNormalWorld,
                                rayState.ray.direction,
                                u, v,
                                su, sv);

                            // Beta kernel parameters:
                            float beta = 4.0f * sycl::exp(surfel.beta);
                            // d alpha / d position (beta kernel):
                            float factor = (-2.0f * beta * alphaGeom) / (1.0f - r2);
                            float3 dAlpha_dPos = factor * DuvDPosition;
                            // d alpha_eff / d position:
                            float3 dAlphaEff_dPos = eta * dAlpha_dPos;
                            // Color difference for volumetric compositing:
                            float3 colorDiff = (L_Surfel) - (L_Mesh); // L_s - L_m
                            // Pixel adjoint / path adjoint:
                            const float3 pathAdjoint = rayState.pathThroughput; // dJ/dC (RGB) * transport
                            // Scalar weight = ⟨adjoint, (L_s - L_m)⟩:
                            float weight = dot(pathAdjoint, colorDiff);
                            // Final position gradient vector:
                            float3 gradPosition = weight * dAlphaEff_dPos;
                            // Accumulate:
                            uint32_t primitiveIndex = splatEvent.primitiveIndex;
                            atomicAddFloat3(gradients.gradPosition[primitiveIndex], gradPosition);

                            const uint32_t pixelIndex = rayState.pixelIndex;

                            {
                                float3 parameterAxisX = float3{1.0f, 0.0f,0.0f};
                               float3 parameterAxisY = float3{0.0f, 1.0f,0.0f};
                               float3 parameterAxisZ = float3{0.0f, 0.0f,1.0f};
                               const float dCdpRX = dot(gradPosition, parameterAxisX);
                               const float4 posScalarRGBX{dCdpRX};
                               atomicAddFloat4ToImage(
                                   &debugImage.framebuffer_posX[pixelIndex],
                                   posScalarRGBX
                               );

                               const float dCdpRY = dot(gradPosition, parameterAxisY);
                               const float4 posScalarRGBY{dCdpRY};
                               atomicAddFloat4ToImage(
                                   &debugImage.framebuffer_posY[pixelIndex],
                                   posScalarRGBY
                               );

                               const float dCdpRZ = dot(gradPosition, parameterAxisZ);
                               const float4 posScalarRGBZ{dCdpRZ};
                               atomicAddFloat4ToImage(
                                   &debugImage.framebuffer_posZ[pixelIndex],
                                   posScalarRGBZ
                               );
                            }

                            if (isWatched) {
                                int debug = 1;
                            }
                        }
                    }

                    /*
                    // Shadow ray on mesh intersection
                    // apply bsdf, tau and cosine:



                    uint32_t numSurfelsOnRay = whTransmit.splatEventCount;
                    for (size_t scatterRay = 0; scatterRay < numSurfelsOnRay; ++scatterRay) {
                        const uint32_t scatterOnPrimitiveIndex = whTransmit.splatEvents[scatterRay].primitiveIndex;

                        RayState scatterRayState = rayState;
                        WorldHit whScatter{};
                        intersectScene(scatterRayState.ray, &whScatter, scene, rng128,
                                       RayIntersectMode::Scatter, scatterOnPrimitiveIndex);
                        buildIntersectionNormal(scene, whScatter);

                        if (!whScatter.hit) {
                            return;
                        }


                        accumulateBsdfGradientsAtScatterSurfel(
                            scatterRayState,
                            whScatter,
                            scene,
                            photonMap,
                            gradients,
                            debugImage,
                            settings.renderDebugGradientImages,
                            debugIndex,
                            isWatched
                        );



                        SplatEvent& splatEvent =  whTransmit.splatEvents[scatterRay];
                        auto& point = scene.points[splatEvent.primitiveIndex];
                        scatterRayState.pathThroughput *= splatEvent.alpha * scene.points[scatterOnPrimitiveIndex].opacity;

                        // apply bsdf, tau and cosine:
                        float cosine = fabs(dot(rayState.ray.direction, whScatter.geometricNormalW));
                        scatterRayState.pathThroughput *=  cosine * point.color * M_1_PIf;

                        //shadowRay(scene, rayState, whScatter,  gradients, debugImage, photonMap, rng128, settings.renderDebugGradientImages, numShadowRays, debugIndex);

                        /*
                        shadowRayAttachedOriginSelf(
                            scene,
                            scatterRayState,
                            whScatter,
                            splatEvent,
                            gradients,
                            debugImage,
                            photonMap,
                            rng128,
                            settings.renderDebugGradientImages,
                            numShadowRays,
                            scatterOnPrimitiveIndex,
                            isWatched
                        );



                    }
                    */
                }
            );
        });
        queue.wait();
    }

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

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<struct AdjointShadeKernelTag>(
                sycl::range<1>(activeRayCount),
                // ReSharper disable once CppDFAUnusedValue
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(settings.randomSeed, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);
                    const RayState &rayState = raysIn[rayIndex];
                    const WorldHit &worldHit = hitRecords[rayIndex];
                    // Shoot one transmit ray. The amount intersected here will tell us how many scatter rays we will transmit.
                    if (!worldHit.hit) {
                        return;
                    }

                    uint32_t numShadowRays = 1;
                    //meshShadowRayState.pathThroughput *= cosine; // * cosine * M_1_PIf; // TODO include bsdf here?
                    shadowRay(scene, rayState, worldHit,  gradients, debugImage, photonMap, rng128, settings.renderDebugGradientImages, numShadowRays);


                    /*
                    accumulateTransmittanceGradientsAlongRay(rayState, whTransmit, scene, photonMap,
                                     settings.renderDebugGradientImages, gradients,
                                     debugImage, debugIndex);
                    */

                });
        });
        queue.wait();
    }

    void generateNextAdjointRays(RenderPackage &pkg, uint32_t activeRayCount) {
        auto &queue = pkg.queue;
        auto &sensor = pkg.sensor;
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
                        material.baseColor = surfel.color;
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
    }
}
