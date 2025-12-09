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
        auto& sensor = pkg.sensor[cameraIndex];

        const uint32_t imageWidth = sensor.camera.width;
        const uint32_t imageHeight = sensor.camera.height;
        uint32_t raysPerSet = imageWidth * imageHeight;

        //raysPerSet = 1;
        float raysTotal = settings.adjointSamplesPerPixel * raysPerSet;

        const uint32_t perPassRayCount = raysPerSet;
        queue.memcpy(pkg.intermediates.countPrimary, &perPassRayCount, sizeof(uint32_t)).wait();

        queue.submit([&](sycl::handler& commandGroupHandler) {
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
                    // RNG for this pixelhttps://www.chess.com/home
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
                    rayState.pixelX = pixelX;
                    rayState.pixelY = pixelY;

                    if (pixelX == 200 && pixelY == (sensor.height - 1 - 325)) {
                        int debug = 1;
                    }

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
            const DebugPixel& debugPixel = kDebugPixels[i];
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
    void launchAdjointProjectionKernel(RenderPackage& pkg, uint32_t activeRayCount, uint32_t cameraIndex) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& settings = pkg.settings;
        auto& intermediates = pkg.intermediates;
        auto& gradients = pkg.gradients;
        const auto& photonMap = pkg.intermediates.map;
        auto* raysIn = pkg.intermediates.primaryRays;

        auto& sensor = pkg.sensor[cameraIndex];
        auto& debugImage = pkg.debugImages[cameraIndex];

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<struct AdjointShadeKernelTag>(
                sycl::range<1>(activeRayCount),
                // ReSharper disable once CppDFAUnusedValue
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(settings.randomSeed, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);

                    RayState& rayState = intermediates.primaryRays[rayIndex];
                    WorldHit& worldHit = intermediates.hitRecords[rayIndex];
                    uint32_t pixelX = rayState.pixelX;
                    uint32_t pixelY = sensor.height - 1 - rayState.pixelY;
                    bool isWatched = false;
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

                    const InstanceRecord& meshInstance = scene.instances[whTransmit.instanceIndex];

                    //uint32_t debugIndex = 1;
                    uint32_t debugIndex = UINT32_MAX;
                    //debugIndex = 0;
                    uint32_t numShadowRays = 10;
                    const Ray& ray = rayState.ray;
                    // Transmission gradients with shadow rays
                    if (meshInstance.geometryType == GeometryType::Mesh) {
                        // Transmission
                        // Cost weighting: photon-map radiance for this segment
                        const float3& L_Mesh = estimateRadianceFromPhotonMap(whTransmit, scene, photonMap);

                        for (uint32_t i = 0; i < whTransmit.splatEventCount; ++i) {
                            // Forward-side values reused in adjoint:
                            auto& splatEvent = whTransmit.splatEvents[i]; // one surfel
                            auto& surfel = scene.points[splatEvent.primitiveIndex];
                            float alphaGeom = splatEvent.alpha; // α(u,v,β)
                            float eta = surfel.opacity;
                            float alphaEff = eta * alphaGeom; // α_eff
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
                                ray.direction,
                                u, v,
                                su, sv);
                            // Beta kernel parameters:
                            float beta = 4.0f * sycl::exp(surfel.beta);
                            // d alpha / d position (beta kernel):
                            float factor = (-2.0f * beta * alphaGeom) / (1.0f - r2);
                            float3 dAlpha_dPos = factor * DuvDPosition;
                            // d alpha_eff / d position:
                            float3 dAlphaEff_dPos = eta * dAlpha_dPos;
                            // ROTATION
                            float3 dUdTu, dVdTu, dUdTv, dVdTv;
                            computeFullDuDvWrtTangents(
                                ray.origin,
                                ray.direction,
                                surfel.position,
                                hitWorld,
                                surfel.tanU,
                                surfel.tanV,
                                su, sv,
                                dUdTu, dVdTu, dUdTv, dVdTv
                            );
                            // dudv derivatives wrt tangents
                            float3 dUVDtU = (u * dUdTu + v * dVdTu);
                            float3 dUVDtV = (u * dUdTv + v * dVdTv);
                            float3 dAlpha_dTanU = factor * dUVDtU;
                            float3 dAlpha_dTanV = factor * dUVDtV;
                            // d alpha_eff / d position:
                            float3 dAlphaEff_dTanU = eta * dAlpha_dTanU;
                            float3 dAlphaEff_dTanV = eta * dAlpha_dTanV;
                            /// SCALE
                            const float3 dUdVdScale =
                                computeDuvDScale(u, v, su, sv);
                            float3 dAlpha_dScale = factor * dUdVdScale;
                            // d alpha_eff / d position:
                            float3 dAlphaEff_dScale = eta * dAlpha_dScale;
                            const float dAlphaEff_dScaleU = dAlphaEff_dScale.x();
                            const float dAlphaEff_dScaleV = dAlphaEff_dScale.y();
                            /// OPACITY
                            //alpha^(eff)(u, v)(&PartialD; eta)/(&PartialD; Pi)
                            float dAlphaEffOpacity = alphaGeom;
                            // Beta parameter:
                            const float dAlphaEffBeta =
                                alphaGeom * surfel.opacity * betaKernel(surfel.beta) *
                                sycl::log(1.0f - r2);
                            // Albedo
                            const float dAlphaEffAlbedo = alphaEff;
                            // Pixel adjoint / path adjoint:
                            const float3 pathAdjoint = rayState.pathThroughput; // dJ/dC (RGB) * transport
                            // Scalar weight = ⟨adjoint, (L_s - L_m)⟩:
                            float tauFront = splatEvent.tau;

                            float3 L_bg = float3{0.0f, 0.0f, 0.0f};

                            const uint32_t S = whTransmit.splatEventCount;
                            // Sum over all surfels behind m: j = m+1..S-1
                            for (uint32_t j = i + 1; j < S; ++j) {
                                auto& jEvent = whTransmit.splatEvents[j];
                                auto& jSurfel = scene.points[jEvent.primitiveIndex];
                                const float jAlphaEff = jEvent.alpha * jSurfel.opacity;
                                const float3 L_surfel_j_incident = computeLSurfel(scene, ray.direction, jEvent, photonMap);
                                // τ_j = Π_{k = i+1 .. j-1} (1 - α_k^eff)
                                float tau_j = 1.0f;
                                for (uint32_t k = i + 1; k < j; ++k) {
                                    auto& kEvent = whTransmit.splatEvents[k];
                                    auto& kSurfel = scene.points[kEvent.primitiveIndex];
                                    const float kAlphaEff = kEvent.alpha * kSurfel.opacity;
                                    tau_j *= (1.0f - kAlphaEff);
                                }
                                L_bg += L_surfel_j_incident * jAlphaEff * tau_j;
                            }
                            // 2) Background mesh term: τ_back = Π_{k = i+1 .. S-1} (1 - α_k^eff)
                            float tau_back = 1.0f;
                            for (uint32_t k = i + 1; k < S; ++k) {
                                auto& kEvent = whTransmit.splatEvents[k];
                                auto& kSurfel = scene.points[kEvent.primitiveIndex];
                                const float kAlphaEff = kEvent.alpha * kSurfel.opacity;
                                tau_back *= (1.0f - kAlphaEff);
                            }
                            L_bg += tau_back * L_Mesh; // Final position gradient vector:

                            const float3& L_surfel_incident = computeLSurfel(scene, ray.direction, splatEvent, photonMap);

                            const float3& L_o = L_surfel_incident * alphaEff * surfel.albedo * M_1_PIf;



                            float grad_luminance_opacity_R = tauFront * (L_o[0] - L_bg[0]);
                            float grad_luminance_opacity_G = tauFront * (L_o[1] - L_bg[1]);
                            float grad_luminance_opacity_B = tauFront * (L_o[2] - L_bg[2]);

                            float3 gradPosition_R = grad_luminance_opacity_R * dAlphaEff_dPos * pathAdjoint[0];
                            float3 gradPosition_G = grad_luminance_opacity_G * dAlphaEff_dPos * pathAdjoint[1];
                            float3 gradPosition_B = grad_luminance_opacity_B * dAlphaEff_dPos * pathAdjoint[2];

                            float3 gradTanU_R = grad_luminance_opacity_R * dAlphaEff_dTanU * pathAdjoint[0];
                            float3 gradTanU_G = grad_luminance_opacity_G * dAlphaEff_dTanU * pathAdjoint[1];
                            float3 gradTanU_B = grad_luminance_opacity_B * dAlphaEff_dTanU * pathAdjoint[2];

                            float3 gradTanV_R = grad_luminance_opacity_R * dAlphaEff_dTanV * pathAdjoint[0];
                            float3 gradTanV_G = grad_luminance_opacity_G * dAlphaEff_dTanV * pathAdjoint[1];
                            float3 gradTanV_B = grad_luminance_opacity_B * dAlphaEff_dTanV * pathAdjoint[2];

                            float gradScaleU_R = grad_luminance_opacity_R * dAlphaEff_dScaleU * pathAdjoint[0];
                            float gradScaleU_G = grad_luminance_opacity_G * dAlphaEff_dScaleU * pathAdjoint[1];
                            float gradScaleU_B = grad_luminance_opacity_B * dAlphaEff_dScaleU * pathAdjoint[2];

                            float gradScaleV_R = grad_luminance_opacity_R * dAlphaEff_dScaleV * pathAdjoint[0];
                            float gradScaleV_G = grad_luminance_opacity_G * dAlphaEff_dScaleV * pathAdjoint[1];
                            float gradScaleV_B = grad_luminance_opacity_B * dAlphaEff_dScaleV * pathAdjoint[2];

                            //float gradOpacity_R = tauFront * L_i[0] * dAlphaEffOpacity * pathAdjoint[0];
                            //float gradOpacity_G = tauFront * L_i[1] * dAlphaEffOpacity * pathAdjoint[1];
                            //float gradOpacity_B = tauFront * L_i[2] * dAlphaEffOpacity * pathAdjoint[2];
//
                            float gradBeta_R = grad_luminance_opacity_R * dAlphaEffBeta * pathAdjoint[0];
                            float gradBeta_G = grad_luminance_opacity_G * dAlphaEffBeta * pathAdjoint[1];
                            float gradBeta_B = grad_luminance_opacity_B * dAlphaEffBeta * pathAdjoint[2];

                            // dC/dρ = τ_front * α_eff * H
                            float3 gradAlbedo = tauFront * alphaEff * alphaEff * L_surfel_incident * M_1_PIf * pathAdjoint;

                            float3 gradOpacityRGB = alphaGeom * tauFront * (L_o - L_bg) * pathAdjoint;
                            float gradOpacity = gradOpacityRGB[0] + gradOpacityRGB[1] + gradOpacityRGB[2];
                            //if (splatEvent.primitiveIndex != 0)
                            //    continue;
                            uint32_t primitiveIndex = splatEvent.primitiveIndex;

                            float3 gradPosition = gradPosition_R + gradPosition_G + gradPosition_B;
                            atomicAddFloat3(gradients.gradPosition[primitiveIndex], gradPosition);

                            float3 gradTanU = gradTanU_R + gradTanU_G + gradTanU_B;
                            atomicAddFloat3(gradients.gradTanU[primitiveIndex], gradTanU);

                            float3 gradTanV = gradTanV_R + gradTanV_G + gradTanV_B;
                            atomicAddFloat3(gradients.gradTanV[primitiveIndex], gradTanV);

                            float2 gradScale = {
                                gradScaleU_R + gradScaleU_G + gradScaleU_B, gradScaleV_R + gradScaleV_G + gradScaleV_B
                            };
                            atomicAddFloat2(gradients.gradScale[primitiveIndex], gradScale);

                            atomicAddFloat(gradients.gradOpacity[primitiveIndex], gradOpacity);

                            float gradBeta = gradBeta_R + gradBeta_G + gradBeta_B;
                            atomicAddFloat(gradients.gradBeta[primitiveIndex], gradBeta);

                            atomicAddFloat3(gradients.gradAlbedo[primitiveIndex], gradAlbedo);


                            const uint32_t pixelIndex = rayState.pixelIndex;

                            if (settings.renderDebugGradientImages){
                                float3 parameterAxisX = float3{1.0f, 0.0f, 0.0f};
                                float3 parameterAxisY = float3{0.0f, 1.0f, 0.0f};
                                float3 parameterAxisZ = float3{0.0f, 0.0f, 1.0f};
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

                                float3 rotationAxis = float3{0.0f, 1.0f, 0.0f};
                                const float3 dBsdfWorld = float3{
                                    dot(gradTanU_R, cross(rotationAxis, surfel.tanU)) +
                                    dot(gradTanV_R, cross(rotationAxis, surfel.tanV)),

                                    dot(gradTanU_G, cross(rotationAxis, surfel.tanU)) +
                                    dot(gradTanV_G, cross(rotationAxis, surfel.tanV)),

                                    dot(gradTanU_B, cross(rotationAxis, surfel.tanU)) +
                                    dot(gradTanV_B, cross(rotationAxis, surfel.tanV))
                                };
                                const float4 rotScalarRGB{
                                    dBsdfWorld.x(), dBsdfWorld.y(), dBsdfWorld.z(), 0.0f
                                };

                                atomicAddFloat4ToImage(
                                    &debugImage.framebuffer_rot[pixelIndex],
                                    rotScalarRGB
                                );


                                atomicAddFloat4ToImage(
                                    &debugImage.framebuffer_scale[pixelIndex],
                                    float4{gradScale.x()}
                                );

                                atomicAddFloat4ToImage(
                                    &debugImage.framebuffer_opacity[pixelIndex],
                                    float4{gradOpacity}
                                );

                                atomicAddFloat4ToImage(
                                    &debugImage.framebuffer_beta[pixelIndex],
                                    float4{gradBeta}
                                );

                                atomicAddFloat4ToImage(
                                    &debugImage.framebuffer_albedo[pixelIndex],
                                    float4{gradAlbedo, 1.0f}
                                );
                            }
                        }
                    }
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

    void launchAdjointTransportKernel(RenderPackage& pkg, uint32_t activeRayCount, uint32_t cameraIndex) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& settings = pkg.settings;
        auto& gradients = pkg.gradients;
        auto& photonMap = pkg.intermediates.map;
        auto* raysIn = pkg.intermediates.primaryRays;
        auto* hitRecords = pkg.intermediates.hitRecords;
        auto& debugImage = pkg.debugImages[cameraIndex];

        auto& sensor = pkg.sensor[cameraIndex];

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<struct AdjointShadeKernelTag>(
                sycl::range<1>(activeRayCount),
                // ReSharper disable once CppDFAUnusedValue
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(settings.randomSeed, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);
                    const RayState& rayState = raysIn[rayIndex];
                    const WorldHit& worldHit = hitRecords[rayIndex];
                    // Shoot one transmit ray. The amount intersected here will tell us how many scatter rays we will transmit.
                    if (!worldHit.hit) {
                        return;
                    }

                    uint32_t pixelX = rayState.pixelX;
                    uint32_t pixelY = sensor.height - 1 - rayState.pixelY;


                    bool isWatched = false;
                    //if (pixelX == 200 && pixelY == 325) {
                    if (pixelX == 225 && pixelY == 220) {
                        isWatched = true;
                        int debug = 1;
                    }
                    uint32_t numShadowRays = 1;

                    shadowRay(scene, rayState, worldHit, gradients, debugImage, photonMap, rng128,
                              settings.renderDebugGradientImages, numShadowRays, UINT32_MAX, isWatched);
                });
        });
        queue.wait();
    }

    void generateNextAdjointRays(RenderPackage& pkg, uint32_t activeRayCount) {
        auto& queue = pkg.queue;
        auto& sensor = pkg.sensor;
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

        queue.submit([&](sycl::handler& cgh) {
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

                    const InstanceRecord& instance = scene.instances[worldHit.instanceIndex];
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
                        auto& surfel = scene.points[worldHit.primitiveIndex];
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
                        }
                        else {
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
