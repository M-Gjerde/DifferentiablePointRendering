//
// Created by magnus on 9/8/25.
//

#include "Renderer/Kernels/AdjointKernels.h"

#include <cmath>
#include <complex>

#include "AdjointGradientKernels.h"
#include "IntersectionKernels.h"
#include "glm/gtc/constants.hpp"
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


    void launchAdjointKernel(RenderPackage& pkg, uint32_t activeRayCount, uint32_t cameraIndex) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& settings = pkg.settings;
        auto& intermediates = pkg.intermediates;
        auto& gradients = pkg.gradients;
        auto& photonMap = pkg.intermediates.map;
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

                    const RayState& rayState = intermediates.primaryRays[rayIndex];
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
                    // Transmission gradients with shadow rays
                    if (meshInstance.geometryType == GeometryType::Mesh) {
                        // Transmission



                        accumulateTransmittanceGradientsAlongRay(rayState, whTransmit, scene, photonMap,
                                         settings.renderDebugGradientImages, gradients,
                                         debugImage, debugIndex);



                    }
                    // Shadow ray on mesh intersection
                    // apply bsdf, tau and cosine:
                    RayState meshShadowRayState = rayState;
                    float cosine = fabs(dot(rayState.ray.direction, whTransmit.geometricNormalW));
                    auto& material = scene.materials[meshInstance.materialIndex];
                    meshShadowRayState.pathThroughput *=  cosine * material.baseColor * M_1_PIf;
                    //shadowRay(scene, meshShadowRayState, whTransmit,  gradients, debugImage, photonMap, rng128, settings.renderDebugGradientImages, numShadowRays, debugIndex);

                    uint32_t pixelX = rayState.pixelX;
                    uint32_t pixelY = sensor.height - 1 - rayState.pixelY;
                    bool isWatched = false;
                    if (pixelX == 400 && pixelY == 510) {
                        isWatched = true;
                        int debug = 1;
                    }
                    if (pixelX == 400 && pixelY == 545) {
                        isWatched = true;
                        int debug = 1;
                    }

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
                            debugIndex
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
                        */


                    }

                }
            );
        });
        queue.wait();
    }


    void launchAdjointKernel2(RenderPackage& pkg, uint32_t activeRayCount, uint32_t cameraIndex) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& settings = pkg.settings;
        auto& intermediates = pkg.intermediates;
        auto& gradients = pkg.gradients;
        auto& photonMap = pkg.intermediates.map;
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
                    const RayState& rayState = intermediates.primaryRays[rayIndex];

                    uint32_t recordBounceIndex = 0;

                    // Shoot one transmit ray. The amount intersected here will tell us how many scatter rays we will transmit.
                    WorldHit whTransmit{};
                    intersectScene(rayState.ray, &whTransmit, scene, rng128, RayIntersectMode::Transmit);

                    if (!whTransmit.hit)
                        return;

                    uint32_t pixelX = rayIndex % sensor.camera.width;
                    uint32_t pixelY = (rayIndex / sensor.camera.width);

                    const bool isWatched = isWatchedPixel(pixelX, pixelY);

                    const InstanceRecord& meshInstance = scene.instances[whTransmit.instanceIndex];
                    const float3 parameterAxis = {1.0f, 0.0f, 0.00f};

                    // Transmission gradients with shadow rays
                    if (meshInstance.geometryType == GeometryType::Mesh) {
                        // Transmission
                        if (whTransmit.splatEventCount > 0) {
                            const Ray& ray = rayState.ray;
                            const float3 x = ray.origin;
                            const float3 y = whTransmit.hitPositionW;

                            // Cost weighting: keep RGB if your loss is RGB; else reduce at end
                            const float3 backgroundRadianceRGB = estimateRadianceFromPhotonMap(
                                whTransmit, scene, photonMap);

                            // Collect all alpha_i and d(alpha_i)/dPi for this segment
                            struct LocalTerm {
                                float alpha{};
                                float betaKernel{};
                                float eta{};
                                float r2{};
                                float3 dAlphaDPos; // dα/d(center)
                                float3 dAlphaDtU; // dα/d(tanU)
                                float3 dAlphaDtV; // dα/d(tanV)
                                float dAlphaDsu{}; // dα/ds_u
                                float dAlphaDsv{}; // dα/ds_v
                                float dalphaDbeta{}; // dα/ds_v
                                uint32_t primitiveIndex{};
                            };

                            LocalTerm localTerms[kMaxSplatEvents];
                            int validCount = 0;
                            float tau = 1.0f; // full product over all (1 - alpha_i)

                            for (size_t eventIdx = 0; eventIdx < whTransmit.splatEventCount; ++eventIdx) {
                                if (validCount >= kMaxSplatEvents)
                                    break;

                                const auto& splatEvent = whTransmit.splatEvents[eventIdx];
                                const Point& surfel = scene.points[splatEvent.primitiveIndex];

                                const float3 canonicalNormalW =
                                    normalize(cross(surfel.tanU, surfel.tanV));

                                const float2 uv = phiInverse(splatEvent.hitWorld, surfel);
                                const float u = uv.x();
                                const float v = uv.y();
                                const float r2 = u * u + v * v;
                                const float alpha = splatEvent.alpha;
                                const float su = surfel.scale.x();
                                const float sv = surfel.scale.y();

                                // ---- dα/d(position) ----
                                const float3 DuvPosition =
                                    computeDuvDPosition(
                                        surfel.tanU,
                                        surfel.tanV,
                                        canonicalNormalW,
                                        ray.direction,
                                        u, v,
                                        su, sv
                                    );

                                float3 dUdTu, dVdTu, dUdTv, dVdTv;
                                computeFullDuDvWrtTangents(
                                    ray.origin,
                                    ray.direction,
                                    surfel.position,
                                    splatEvent.hitWorld,
                                    surfel.tanU,
                                    surfel.tanV,
                                    su, sv,
                                    dUdTu, dVdTu, dUdTv, dVdTv
                                );

                                // Full dudv derivatives
                                float3 dFsDtU = (u * dUdTu + v * dVdTu);
                                float3 dFsDtV = (u * dUdTv + v * dVdTv);

                                float3 dFsDtUGaussian = dFsDtU * surfel.opacity * (-alpha);
                                float3 dFsDtVGaussian = dFsDtV * surfel.opacity * (-alpha);

                                float betaKernelFactor = computeSmoothedBetaFactorBSDF(
                                    surfel.beta, r2, alpha, surfel.opacity);

                                float3 dFsDtUBeta = betaKernelFactor * dFsDtU;
                                float3 dFsDtVBeta = betaKernelFactor * dFsDtV;


                                // ---- dα/d(scale) (s_u, s_v) ----
                                const float3 dUdVdScale =
                                    computeDuvDScale(
                                        u, v,
                                        su, sv
                                    );

                                // Example: BSDF gradients for each parameter group
                                //const float3 dFsDPosition = DalphaDuvPositionGaussian(DuvPosition, alpha, surfel.opacity);
                                const float3 dFsDPosition = betaKernelFactor * DuvPosition;

                                // dα/ds_u = (∂α/∂u)(∂u/∂s_u) = (-u α) * (du/dsu)
                                // Gaussian Like scale
                                //const float dFsDsu = u * dUdVdScale.x();
                                //const float dFsDsv = v * dUdVdScale.y();

                                //float2 DFsDsusv = computeDAlphaDScaleGaussian(dUdVdScale, u, v, alpha, surfel.opacity);
                                float3 DFsDsusv = betaKernelFactor * dUdVdScale;

                                // Beta parameter:
                                float DalphaDbeta = alpha * betaKernel(surfel.beta) * sycl::log(1.0f - r2) * surfel.
                                    opacity;


                                const float oneMinusAlpha = 1.0f - alpha * surfel.opacity;
                                if (oneMinusAlpha <= 1e-6f)
                                    continue;


                                // Store local term
                                auto& lt = localTerms[validCount++];
                                lt.alpha = alpha;
                                lt.eta = surfel.opacity;
                                lt.r2 = r2;
                                lt.betaKernel = betaKernel(surfel.beta);
                                lt.dAlphaDPos = dFsDPosition;
                                lt.dAlphaDtU = dFsDtUBeta;
                                lt.dAlphaDtV = dFsDtVBeta;
                                lt.dAlphaDsu = DFsDsusv.x();
                                lt.dAlphaDsv = DFsDsusv.y();
                                lt.dalphaDbeta = DalphaDbeta;
                                lt.primitiveIndex = splatEvent.primitiveIndex;

                                // Update τ
                                tau *= oneMinusAlpha;
                            }

                            if (validCount != 0) {
                                float3 pAdjoint = rayState.pathThroughput;

                                for (int i = 0; i < validCount; ++i) {
                                    const uint32_t primIdx = localTerms[i].primitiveIndex;

                                    const float alpha = localTerms[i].alpha;
                                    const float eta = localTerms[i].eta;
                                    const float r2 = localTerms[i].r2;
                                    const float betaKernel = localTerms[i].betaKernel;

                                    // Precompute τ_i = 1 - η_i α_i
                                    const float tau_i = 1.0f - eta * alpha;
                                    if (tau_i <= 1e-6f)
                                        continue;
                                    const float invTau_i = 1.0f / tau_i;

                                    // localTerms store d(ηα)/d(·) already:
                                    const float3 dEtaAlphaDPos = localTerms[i].dAlphaDPos;
                                    const float3 dEtaAlphaDtU = localTerms[i].dAlphaDtU;
                                    const float3 dEtaAlphaDtV = localTerms[i].dAlphaDtV;
                                    const float dEtaAlphaDsu = localTerms[i].dAlphaDsu;
                                    const float dEtaAlphaDsv = localTerms[i].dAlphaDsv;
                                    const float dEtaAlphaDbeta = localTerms[i].dalphaDbeta;

                                    // From derivation:
                                    // dτ/dΠ|_i = - τ * ( d(ηα)/dΠ ) / (1 - ηα)
                                    const float minusTauInvTau_i = -tau * invTau_i;

                                    const float3 dTauDPos = minusTauInvTau_i * dEtaAlphaDPos;
                                    const float3 dTauDtU = minusTauInvTau_i * dEtaAlphaDtU;
                                    const float3 dTauDtV = minusTauInvTau_i * dEtaAlphaDtV;
                                    const float dTauDsu = minusTauInvTau_i * dEtaAlphaDsu;
                                    const float dTauDsv = minusTauInvTau_i * dEtaAlphaDsv;
                                    const float dTauDbeta = minusTauInvTau_i * dEtaAlphaDbeta;

                                    // Opacity derivative stays:
                                    // dτ/dη_i = τ * [ -α_i / (1 - η_i α_i) ]
                                    const float dTauDeta = -tau * alpha * invTau_i;

                                    float R = pAdjoint[0] * backgroundRadianceRGB[0];
                                    float G = pAdjoint[1] * backgroundRadianceRGB[1];
                                    float B = pAdjoint[2] * backgroundRadianceRGB[2];

                                    // Cost gradients: ∂C/∂(·) = p * L_bg * dτ/d(·)
                                    const float3 gradCPosR = R * dTauDPos;
                                    const float3 gradCPosG = G * dTauDPos;
                                    const float3 gradCPosB = B * dTauDPos;

                                    const float3 gradCTanUR = R * dTauDtU;
                                    const float3 gradCTanUG = G * dTauDtU;
                                    const float3 gradCTanUB = B * dTauDtU;

                                    const float3 gradCTanVR = R * dTauDtV;
                                    const float3 gradCTanVG = G * dTauDtV;
                                    const float3 gradCTanVB = B * dTauDtV;

                                    const float gradCScaleUR = R * dTauDsu;
                                    const float gradCScaleUG = G * dTauDsu;
                                    const float gradCScaleUB = B * dTauDsu;

                                    const float gradCScaleVR = R * dTauDsv;
                                    const float gradCScaleVG = G * dTauDsv;
                                    const float gradCScaleVB = B * dTauDsv;

                                    const float gradCOpacityR = R * dTauDeta;
                                    const float gradCOpacityG = G * dTauDeta;
                                    const float gradCOpacityB = B * dTauDeta;

                                    // NEW: beta cost gradients
                                    const float gradCBetaR = R * dTauDbeta;
                                    const float gradCBetaG = G * dTauDbeta;
                                    const float gradCBetaB = B * dTauDbeta;

                                    uint32_t primitiveIndex = primIdx;

                                    // Accumulate parameter-space gradients
                                    atomicAddFloat3(
                                        gradients.gradPosition[primitiveIndex],
                                        gradCPosR + gradCPosG + gradCPosB
                                    );
                                    atomicAddFloat3(
                                        gradients.gradTanU[primitiveIndex],
                                        gradCTanUR + gradCTanUG + gradCTanUB
                                    );
                                    atomicAddFloat3(
                                        gradients.gradTanV[primitiveIndex],
                                        gradCTanVR + gradCTanVG + gradCTanVB
                                    );
                                    atomicAddFloat(
                                        gradients.gradScale[primitiveIndex].x(),
                                        gradCScaleUR + gradCScaleUG + gradCScaleUB
                                    );
                                    atomicAddFloat(
                                        gradients.gradScale[primitiveIndex].y(),
                                        gradCScaleVR + gradCScaleVG + gradCScaleVB
                                    );
                                    atomicAddFloat(
                                        gradients.gradOpacity[primitiveIndex],
                                        gradCOpacityR + gradCOpacityG + gradCOpacityB
                                    );
                                    atomicAddFloat(
                                        gradients.gradBeta[primitiveIndex],
                                        gradCBetaR + gradCBetaG + gradCBetaB
                                    );
                                    // Mapping to world coordinates:
                                    const auto& surfel = scene.points[primitiveIndex];
                                    float3 dBsdf_world = float3{
                                        dot(gradCTanUR, cross(float3{0.0f, 1.0f, 0.0f}, surfel.tanU)) +
                                        dot(gradCTanVR, cross(float3{0.0f, 1.0f, 0.0f}, surfel.tanV)),

                                        dot(gradCTanUG, cross(float3{0.0f, 1.0f, 0.0f}, surfel.tanU)) +
                                        dot(gradCTanVG, cross(float3{0.0f, 1.0f, 0.0f}, surfel.tanV)),

                                        dot(gradCTanUB, cross(float3{0.0f, 1.0f, 0.0f}, surfel.tanU)) +
                                        dot(gradCTanVB, cross(float3{0.0f, 1.0f, 0.0f}, surfel.tanV))
                                    };

                                    if (settings.renderDebugGradientImages && rayState.bounceIndex >=
                                        recordBounceIndex) {
                                        // --- Position debug (projection on parameterAxis) ------------------
                                        const float dCdp_R = dot(gradCPosR, parameterAxis);
                                        const float dCdp_G = dot(gradCPosG, parameterAxis);
                                        const float dCdp_B = dot(gradCPosB, parameterAxis);
                                        const float4 posScalarRGB{dCdp_R, dCdp_G, dCdp_B, 0.0f};

                                        // --- Rotation debug (use dBsdf_world directly) ---------------------
                                        const float4 rotScalarRGB{
                                            dBsdf_world.x(), dBsdf_world.y(), dBsdf_world.z(), 0.0f
                                        };

                                        // --- Scale debug (here: U-only contributions per color channel) ----
                                        const float gradScaleUR = gradCScaleUR;
                                        const float gradScaleUG = gradCScaleUG;
                                        const float gradScaleUB = gradCScaleUB;
                                        const float4 scaleScalarRGB{gradScaleUR, gradScaleUG, gradScaleUB, 0.0f};

                                        // --- Opacity debug --------------------------------------------------
                                        const float4 opacityScalarRGB{
                                            gradCOpacityR, gradCOpacityG, gradCOpacityB, 0.0f
                                        };
                                        // --- Opacity debug --------------------------------------------------
                                        const float4 betaScalarRGB{
                                            gradCBetaR, gradCBetaG, gradCBetaB, 0.0f
                                        };

                                        // Write to per-parameter debug framebuffers
                                        atomicAddFloat4ToImage(
                                            &debugImage.framebuffer_pos[rayState.pixelIndex],
                                            posScalarRGB
                                        );
                                        atomicAddFloat4ToImage(
                                            &debugImage.framebuffer_rot[rayState.pixelIndex],
                                            rotScalarRGB
                                        );
                                        atomicAddFloat4ToImage(
                                            &debugImage.framebuffer_scale[rayState.pixelIndex],
                                            scaleScalarRGB
                                        );
                                        atomicAddFloat4ToImage(
                                            &debugImage.framebuffer_opacity[rayState.pixelIndex],
                                            opacityScalarRGB
                                        );
                                        atomicAddFloat4ToImage(
                                            &debugImage.framebuffer_beta[rayState.pixelIndex],
                                            betaScalarRGB
                                        );
                                    }


                                    /*
                                                                        if (isWatched) {
                                                                            printf(
                                                                                "Index:%u Surfel: %u  Grad:(%f, %f, %f)  L_Background:%f alpha:%f  Adj:%f\n",
                                                                                rayIndex,
                                                                                primIdx,
                                                                                grad_C_pos.x(),grad_C_pos.x(),grad_C_pos.x(),
                                                                                L_bg, alpha, p);
                                                                            int debug = 0;
                                                                        }
                                                                        */
                                }
                            }
                        }
                    }


                    uint32_t numSurfelsOnRay = whTransmit.splatEventCount;

                    for (size_t scatterRay = 0; scatterRay < numSurfelsOnRay; ++scatterRay) {
                        uint32_t scatterOnPrimitiveIndex = whTransmit.splatEvents[scatterRay].primitiveIndex;

                        WorldHit whScatter{};
                        intersectScene(rayState.ray, &whScatter, scene, rng128, RayIntersectMode::Scatter,
                                       scatterOnPrimitiveIndex);

                        if (!whScatter.hit)
                            return;

                        float tau = 1.0f; // full product over all (1 - alpha_i)
                        if (whScatter.splatEventCount > 0) {
                            for (size_t eventIdx = 0; eventIdx < whScatter.splatEventCount - 1; ++eventIdx) {
                                const auto& splatEvent = whScatter.splatEvents[eventIdx];
                                const Point& surfel = scene.points[splatEvent.primitiveIndex];
                                const float oneMinusAlpha = 1.0f - splatEvent.alpha * surfel.opacity;
                                if (oneMinusAlpha <= 1e-6f)
                                    continue;
                                // Update τ
                                tau *= oneMinusAlpha;
                            }
                        }

                        const InstanceRecord& surfelInstance = scene.instances[whScatter.instanceIndex];
                        if (surfelInstance.geometryType == GeometryType::PointCloud) {
                            // 1) Build intervening splat list on (t_x, t_y), excluding terminal scatter surfel
                            // Do this however through a shadow ray
                            const Ray& ray = rayState.ray;
                            // 3) Terminal scatter at z: dα/dc with correct sign
                            const uint32_t terminalPrim = whScatter.primitiveIndex;
                            const SplatEvent& terminal = whScatter.splatEvents[
                                whScatter.splatEventCount - 1];
                            {
                                const auto surfel = scene.points[terminal.primitiveIndex];
                                const float3 canonicalNormalWorld = normalize(
                                    cross(surfel.tanU, surfel.tanV));
                                const float3 rayDirection = ray.direction;
                                const float3 hitWorld = terminal.hitWorld;

                                const float2 uv = phiInverse(hitWorld, surfel);
                                const float u = uv.x();
                                const float v = uv.y();
                                const float r2 = u * u + v * v;
                                const float alpha = terminal.alpha;
                                const float su = surfel.scale.x();
                                const float sv = surfel.scale.y();

                                // ---- dα/d(position) ----
                                const float3 DuvPosition =
                                    computeDuvDPosition(
                                        surfel.tanU,
                                        surfel.tanV,
                                        canonicalNormalWorld,
                                        rayDirection,
                                        u, v,
                                        su, sv
                                    );

                                float3 dUdTu, dVdTu, dUdTv, dVdTv;
                                computeFullDuDvWrtTangents(
                                    ray.origin,
                                    ray.direction,
                                    surfel.position,
                                    terminal.hitWorld,
                                    surfel.tanU,
                                    surfel.tanV,
                                    su, sv,
                                    dUdTu, dVdTu, dUdTv, dVdTv
                                );

                                // Full dudv derivatives
                                float3 dFsDtU = (u * dUdTu + v * dVdTu);
                                float3 dFsDtV = (u * dUdTv + v * dVdTv);

                                float3 dFsDtUGaussian = dFsDtU * surfel.opacity * (-alpha);
                                float3 dFsDtVGaussian = dFsDtV * surfel.opacity * (-alpha);

                                float betaKernelFactor = computeSmoothedBetaFactorBSDF(
                                    surfel.beta, r2, alpha, surfel.opacity);


                                float3 dFsDtUBeta = betaKernelFactor * dFsDtU;
                                float3 dFsDtVBeta = betaKernelFactor * dFsDtV;


                                // ---- dα/d(scale) (s_u, s_v) ----
                                const float3 dUdVdScale =
                                    computeDuvDScale(
                                        u, v,
                                        su, sv
                                    );

                                // Example: BSDF gradients for each parameter group
                                //const float3 dFsDPosition = DalphaDuvPositionGaussian(DuvPosition, alpha, surfel.opacity);
                                const float3 dFsDPosition = betaKernelFactor * DuvPosition;

                                // dα/ds_u = (∂α/∂u)(∂u/∂s_u) = (-u α) * (du/dsu)
                                // Gaussian Like scale
                                //const float dFsDsu = u * dUdVdScale.x();
                                //const float dFsDsv = v * dUdVdScale.y();

                                //float2 DFsDsusv = computeDAlphaDScaleGaussian(dUdVdScale, u, v, alpha, surfel.opacity);
                                float3 DFsDsusv = betaKernelFactor * dUdVdScale;

                                // Beta parameter:
                                float DalphaDbeta = alpha * betaKernel(surfel.beta) * sycl::log(1.0f - r2) * surfel.
                                    opacity;


                                const float3 surfelRadianceRGB = estimateSurfelRadianceFromPhotonMap(
                                    terminal, ray.direction, scene, photonMap,
                                    false, true, true);

                                // 5) Final gradient assembly
                                float3 pathAdjoint = rayState.pathThroughput;


                                float3 brdfGradPosition = dFsDPosition;
                                float3 brdfGradTanU = dFsDtUBeta;
                                float3 brdfGradTanV = dFsDtVBeta;
                                float brdfGradScaleU = DFsDsusv.x();
                                float brdfGradScaleV = DFsDsusv.y();

                                float brdfGradAlbedo = alpha * surfel.opacity;
                                float3 brdfGradOpacity = alpha * surfel.color;
                                float brdfGradBeta = DalphaDbeta;

                                // Per-channel adjoint * radiance factor
                                float R = pathAdjoint[0] * tau * surfelRadianceRGB[0];
                                float G = pathAdjoint[1] * tau * surfelRadianceRGB[1];
                                float B = pathAdjoint[2] * tau * surfelRadianceRGB[2];

                                // Cost gradients: ∂C/∂(·) = p * L_bg * dτ/d(·)
                                const float3 gradCPosR = R * brdfGradPosition;
                                const float3 gradCPosG = G * brdfGradPosition;
                                const float3 gradCPosB = B * brdfGradPosition;

                                const float3 gradCTanUR = R * brdfGradTanU;
                                const float3 gradCTanUG = G * brdfGradTanU;
                                const float3 gradCTanUB = B * brdfGradTanU;

                                const float3 gradCTanVR = R * brdfGradTanV;
                                const float3 gradCTanVG = G * brdfGradTanV;
                                const float3 gradCTanVB = B * brdfGradTanV;

                                const float gradCScaleUR = R * brdfGradScaleU;
                                const float gradCScaleUG = G * brdfGradScaleU;
                                const float gradCScaleUB = B * brdfGradScaleU;

                                const float gradCScaleVR = R * brdfGradScaleV;
                                const float gradCScaleVG = G * brdfGradScaleV;
                                const float gradCScaleVB = B * brdfGradScaleV;

                                const float gradCAlbedoR = R * brdfGradAlbedo;
                                const float gradCAlbedoG = G * brdfGradAlbedo;
                                const float gradCAlbedoB = B * brdfGradAlbedo;

                                const float gradCOpacityR = R * brdfGradOpacity[0];
                                const float gradCOpacityG = G * brdfGradOpacity[1];
                                const float gradCOpacityB = B * brdfGradOpacity[2];

                                const float gradCBetaR = R * brdfGradBeta;
                                const float gradCBetaG = G * brdfGradBeta;
                                const float gradCBetaB = B * brdfGradBeta;

                                uint32_t primitiveIndex = terminalPrim;

                                // Accumulate parameter-space gradients
                                atomicAddFloat3(
                                    gradients.gradPosition[primitiveIndex],
                                    gradCPosR + gradCPosG + gradCPosB
                                );
                                atomicAddFloat3(
                                    gradients.gradTanU[primitiveIndex],
                                    gradCTanUR + gradCTanUG + gradCTanUB
                                );
                                atomicAddFloat3(
                                    gradients.gradTanV[primitiveIndex],
                                    gradCTanVR + gradCTanVG + gradCTanVB
                                );
                                atomicAddFloat(
                                    gradients.gradScale[primitiveIndex].x(),
                                    gradCScaleUR + gradCScaleUG + gradCScaleUB
                                );
                                atomicAddFloat(
                                    gradients.gradScale[primitiveIndex].y(),
                                    gradCScaleVR + gradCScaleVG + gradCScaleVB
                                );
                                atomicAddFloat(
                                    gradients.gradOpacity[primitiveIndex],
                                    gradCOpacityR + gradCOpacityG + gradCOpacityB
                                );
                                atomicAddFloat(
                                    gradients.gradBeta[primitiveIndex],
                                    gradCBetaR + gradCBetaG + gradCBetaB
                                );

                                float3 gradColorValue{gradCAlbedoR, gradCAlbedoG, gradCAlbedoB};
                                atomicAddFloat3(gradients.gradColor[primitiveIndex], gradColorValue);

                                // Mapping BSDF gradient induced by tangents to world coordinates
                                float3 dBsdf_world = float3{
                                    dot(gradCTanUR, cross(float3{0.0f, 1.0f, 0.0f}, surfel.tanU)) +
                                    dot(gradCTanVR, cross(float3{0.0f, 1.0f, 0.0f}, surfel.tanV)),

                                    dot(gradCTanUG, cross(float3{0.0f, 1.0f, 0.0f}, surfel.tanU)) +
                                    dot(gradCTanVG, cross(float3{0.0f, 1.0f, 0.0f}, surfel.tanV)),

                                    dot(gradCTanUB, cross(float3{0.0f, 1.0f, 0.0f}, surfel.tanU)) +
                                    dot(gradCTanVB, cross(float3{0.0f, 1.0f, 0.0f}, surfel.tanV))
                                };

                                if (settings.renderDebugGradientImages && rayState.bounceIndex >= recordBounceIndex) {
                                    // --- Position debug (projection on parameterAxis) ------------------
                                    const float dCdp_R = dot(gradCPosR, parameterAxis);
                                    const float dCdp_G = dot(gradCPosG, parameterAxis);
                                    const float dCdp_B = dot(gradCPosB, parameterAxis);
                                    const float4 posScalarRGB{dCdp_R, dCdp_G, dCdp_B, 0.0f};

                                    // --- Rotation debug (use dBsdf_world directly) ---------------------
                                    const float4 rotScalarRGB{dBsdf_world.x(), dBsdf_world.y(), dBsdf_world.z(), 0.0f};

                                    // --- Scale debug (sum U/V contributions per color channel) ---------
                                    const float gradScaleUR = gradCScaleUR;
                                    const float gradScaleUG = gradCScaleUG;
                                    const float gradScaleUB = gradCScaleUB;
                                    const float4 scaleScalarRGB{gradScaleUR, gradScaleUG, gradScaleUB, 0.0f};

                                    // --- Opacity debug --------------------------------------------------
                                    const float4 opacityScalarRGB{
                                        gradCOpacityR, gradCOpacityG, gradCOpacityB, 0.0f
                                    };

                                    // --- Albedo debug ---------------------------------------------------
                                    const float4 albedoScalarRGB{
                                        gradCAlbedoR, gradCAlbedoG, gradCAlbedoB, 0.0f
                                    };
                                    // --- Beta debug ---------------------------------------------------
                                    const float4 betaScalarRGB{
                                        gradCBetaR, gradCBetaG, gradCBetaB, 0.0f
                                    };

                                    // Write to per-parameter debug framebuffers
                                    atomicAddFloat4ToImage(
                                        &debugImage.framebuffer_pos[rayState.pixelIndex],
                                        posScalarRGB
                                    );
                                    atomicAddFloat4ToImage(
                                        &debugImage.framebuffer_rot[rayState.pixelIndex],
                                        rotScalarRGB
                                    );
                                    atomicAddFloat4ToImage(
                                        &debugImage.framebuffer_scale[rayState.pixelIndex],
                                        scaleScalarRGB
                                    );
                                    atomicAddFloat4ToImage(
                                        &debugImage.framebuffer_opacity[rayState.pixelIndex],
                                        opacityScalarRGB
                                    );
                                    atomicAddFloat4ToImage(
                                        &debugImage.framebuffer_albedo[rayState.pixelIndex],
                                        albedoScalarRGB
                                    );
                                    atomicAddFloat4ToImage(
                                        &debugImage.framebuffer_beta[rayState.pixelIndex],
                                        betaScalarRGB
                                    );
                                }

                                /*
                                if (isWatched) {
                                    printf(
                                            "Index:%u Surfel: %u Grad:(%f, %f, %f)  L_Surfel:%f alpha:%f  Adj:%f\n",
                                            rayIndex,
                                            primitiveIndex,
                                            grad_C_pos.x(),grad_C_pos.x(),grad_C_pos.x(),
                                        L_surfel, alpha, p);
                                    int debug = 0;
                                }
                                */
                            }
                        }
                    }
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
                        const float3 lambertBrdf = material.baseColor * M_1_PIf;
                        auto& surfel = scene.points[worldHit.primitiveIndex];
                        float alpha = worldHit.splatEvents[worldHit.splatEventCount - 1].alpha;
                        material.baseColor =  surfel.color;

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
                                throughputMultiplier * lambertBrdf * (cosTheta / sampledPdf) * surfel.opacity * alpha;
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
                                throughputMultiplier * lambertBrdf * (cosTheta / sampledPdf) * surfel.opacity * alpha;;
                        }
                    }


                    // Offset origin robustly
                    constexpr float kEps = 1e-7f;
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
