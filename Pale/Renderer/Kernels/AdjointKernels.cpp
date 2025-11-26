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
    void launchRayGenAdjointKernel(RenderPackage &pkg, int spp) {
        auto &queue = pkg.queue;
        auto &sensor = pkg.sensor;
        auto &settings = pkg.settings;
        auto &intermediates = pkg.intermediates;

        const uint32_t imageWidth = sensor.camera.width;
        const uint32_t imageHeight = sensor.camera.height;

        uint32_t raysPerSet = imageWidth * imageHeight;

        //raysPerSet = 1;
        float raysTotal = settings.adjointSamplesPerPixel;

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

                    intermediates.primaryRays[baseOutputSlot] = rayState;
                });
        }).wait();
    }

    struct DebugPixel {
        uint32_t pixelY;
        uint32_t pixelX;
    };

    static constexpr DebugPixel kDebugPixels[] = {
        {265, 220},
        {290, 220},
        // {700, 250},
        // {700, 330},
        // {700, 400},
        // {700, 530},
    };

    SYCL_EXTERNAL inline bool isWatchedPixel(uint32_t pixelX, uint32_t pixelY) {
        bool isMatch = false;
        constexpr uint32_t count = sizeof(kDebugPixels) / sizeof(DebugPixel);

        for (uint32_t i = 0; i < count; ++i) {
            const DebugPixel &debugPixel = kDebugPixels[i];
            if (pixelY == debugPixel.pixelY && pixelX == debugPixel.pixelX) {
                isMatch = true;
            }
        }
        return isMatch;
    }


    void launchAdjointKernel(RenderPackage &pkg, uint32_t activeRayCount) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &sensor = pkg.sensor;
        auto &settings = pkg.settings;
        auto &intermediates = pkg.intermediates;
        auto &gradients = pkg.gradients;
        auto &photonMap = pkg.intermediates.map;
        auto *raysIn = pkg.intermediates.primaryRays;

        queue.submit([&](sycl::handler &cgh) {
                    cgh.parallel_for<struct AdjointShadeKernelTag>(
                        sycl::range<1>(activeRayCount),
                        // ReSharper disable once CppDFAUnusedValue
                        [=](sycl::id<1> globalId) {
                            const uint32_t rayIndex = globalId[0];
                            const uint64_t perItemSeed = rng::makePerItemSeed1D(settings.randomSeed, rayIndex);
                            rng::Xorshift128 rng128(perItemSeed);

                            const RayState &rayState = intermediates.primaryRays[rayIndex];
                            //const WorldHit &worldHit = intermediates.hitRecords[rayIndex];
                            uint32_t recordBounceIndex = 0;
                            // Shoot one transmit ray. The amount intersected here will tell us how many scatter rays we will transmit.
                            WorldHit whTransmit{};
                            intersectScene(rayState.ray, &whTransmit, scene, rng128, RayIntersectMode::Transmit);

                            if (!whTransmit.hit)
                                return;

                            uint32_t pixelX = rayIndex % sensor.camera.width;
                            uint32_t pixelY = (rayIndex / sensor.camera.width);

                            const bool isWatched = isWatchedPixel(pixelX, pixelY);

                            const InstanceRecord &meshInstance = scene.instances[whTransmit.instanceIndex];
                            const float3 parameterAxis = {1.0f, 0.0f, 0.00f};

                            // Transmission gradients with shadow rays
                            if (meshInstance.geometryType == GeometryType::Mesh) {
                                // Transmission
                                if (whTransmit.splatEventCount > 0) {
                                    const Ray &ray = rayState.ray;
                                    const float3 x = ray.origin;
                                    const float3 y = whTransmit.hitPositionW;

                                    // Cost weighting: keep RGB if your loss is RGB; else reduce at end
                                    const float3 backgroundRadianceRGB = estimateRadianceFromPhotonMap(
                                        whTransmit, scene, photonMap);

                                    // Collect all alpha_i and d(alpha_i)/dPi for this segment
                                    struct LocalTerm {
                                        float alpha{};
                                        float eta{};
                                        float3 dAlphaDPos; // dα/d(center)
                                        float3 dAlphaDtU; // dα/d(tanU)
                                        float3 dAlphaDtV; // dα/d(tanV)
                                        float dAlphaDsu{}; // dα/ds_u
                                        float dAlphaDsv{}; // dα/ds_v
                                        uint32_t primitiveIndex{};
                                    };

                                    LocalTerm localTerms[kMaxSplatEvents];
                                    int validCount = 0;
                                    float tau = 1.0f; // full product over all (1 - alpha_i)

                                    for (size_t eventIdx = 0; eventIdx < whTransmit.splatEventCount; ++eventIdx) {
                                        if (validCount >= kMaxSplatEvents)
                                            break;

                                        const auto &splatEvent = whTransmit.splatEvents[eventIdx];
                                        const Point &surfel = scene.points[splatEvent.primitiveIndex];

                                        const float3 canonicalNormalW =
                                                normalize(cross(surfel.tanU, surfel.tanV));

                                        const float denom = dot(canonicalNormalW, ray.direction);
                                        if (sycl::fabs(denom) <= 1e-4f)
                                            continue; // skip grazing

                                        const float2 uv = phiInverse(splatEvent.hitWorld, surfel);
                                        const float u = uv.x();
                                        const float v = uv.y();
                                        const float alpha = splatEvent.alpha;

                                        const float su = surfel.scale.x();
                                        const float sv = surfel.scale.y();

                                        // ---- dα/d(position) ----
                                        const float3 DuvDpos =
                                                computeDuvDPosition(
                                                    surfel.tanU,
                                                    surfel.tanV,
                                                    canonicalNormalW,
                                                    ray.direction,
                                                    u, v,
                                                    su, sv
                                                );

                                        // ---- dα/d(tangents) via your existing helper ----
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

                                        float3 dAlphaDtU = (u * dUdTu + v * dVdTu);
                                        float3 dAlphaDtV = (u * dUdTv + v * dVdTv);

                                        // ---- dα/d(scale) (kernel-only; no area Jacobian in τ) ----
                                        const float3 dUdVdScale =
                                                computeDuvDScale(u, v, su, sv);
                                        // du/dsu = -u/su
                                        // dv/dsv = -v/sv
                                        const float duDsu = dUdVdScale.x();
                                        const float dvDsv = dUdVdScale.y();

                                        // dα/ds_u = (∂α/∂u)(∂u/∂s_u) = (-u α) * duDsu
                                        const float dAlphaDsu = u * duDsu;
                                        // dα/ds_v = (∂α/∂v)(∂v/∂s_v) = (-v α) * dvDsv
                                        const float dAlphaDsv = v * dvDsv;

                                        const float oneMinusAlpha = 1.0f - alpha * surfel.opacity;
                                        if (oneMinusAlpha <= 1e-6f)
                                            continue;


                                        // Store local term
                                        auto &lt = localTerms[validCount++];
                                        lt.alpha = alpha;
                                        lt.eta = surfel.opacity;
                                        lt.dAlphaDPos = DuvDpos;
                                        lt.dAlphaDtU = dAlphaDtU;
                                        lt.dAlphaDtV = dAlphaDtV;
                                        lt.dAlphaDsu = dAlphaDsu;
                                        lt.dAlphaDsv = dAlphaDsv;
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
                                            const float weight = tau * (alpha * eta) / (1.0f - alpha * eta);
                                            // NOTE: no extra alpha, and minus sign

                                            const float3 dTauDPos = weight * localTerms[i].dAlphaDPos;
                                            const float3 dTauDtU = weight * localTerms[i].dAlphaDtU;
                                            const float3 dTauDtV = weight * localTerms[i].dAlphaDtV;
                                            const float dTauDsu = weight * localTerms[i].dAlphaDsu;
                                            const float dTauDsv = weight * localTerms[i].dAlphaDsv;
                                            const float dTauDeta = -tau * alpha / (1 - eta * alpha);
                                            // dτ/ds_u, dτ/ds_v

                                            float R = pAdjoint[0] * backgroundRadianceRGB[0];
                                            float G = pAdjoint[1] * backgroundRadianceRGB[1];
                                            float B = pAdjoint[2] * backgroundRadianceRGB[2];

                                            // Cost gradients: ∂C/∂(·) = p * L_bg * dτ/d(·)
                                            const float3 grad_C_pos_R = R * dTauDPos;
                                            const float3 grad_C_pos_G = G * dTauDPos;
                                            const float3 grad_C_pos_B = B * dTauDPos;
                                            const float3 grad_C_tanU_R = R * dTauDtU;
                                            const float3 grad_C_tanU_G = G * dTauDtU;
                                            const float3 grad_C_tanU_B = B * dTauDtU;
                                            const float3 grad_C_tanV_R = R * dTauDtV;
                                            const float3 grad_C_tanV_G = G * dTauDtV;
                                            const float3 grad_C_tanV_B = B * dTauDtV;
                                            const float grad_C_scaleU_R = R * dTauDsu;
                                            const float grad_C_scaleU_G = G * dTauDsu;
                                            const float grad_C_scaleU_B = B * dTauDsu;
                                            const float grad_C_scaleV_R = R * dTauDsv;
                                            const float grad_C_scaleV_G = G * dTauDsv;
                                            const float grad_C_scaleV_B = B * dTauDsv;

                                            const float grad_C_opacity_R = R * dTauDeta;
                                            const float grad_C_opacity_G = G * dTauDeta;
                                            const float grad_C_opacity_B = B * dTauDeta;

                                            // Accumulate
                                            atomicAddFloat3(gradients.gradPosition[primIdx],
                                                            grad_C_pos_R + grad_C_pos_G + grad_C_pos_B);
                                            atomicAddFloat3(gradients.gradTanU[primIdx],
                                                            grad_C_tanU_R + grad_C_tanU_G + grad_C_tanU_B);
                                            atomicAddFloat3(gradients.gradTanV[primIdx],
                                                            grad_C_tanV_R + grad_C_tanV_G + grad_C_tanV_B);
                                            atomicAddFloat(gradients.gradScale[primIdx].x(),
                                                           grad_C_scaleU_R + grad_C_scaleU_G + grad_C_scaleU_B);
                                            atomicAddFloat(gradients.gradScale[primIdx].y(),
                                                           grad_C_scaleV_R + grad_C_scaleV_G + grad_C_scaleV_B);

                                            atomicAddFloat(gradients.gradOpacity[primIdx],
                                                           grad_C_opacity_R + grad_C_opacity_G + grad_C_opacity_B);

                                            // Mapping to world coordinates:
                                            const auto &surfel = scene.points[primIdx];
                                            float3 dBsdf_world = float3{
                                                dot(
                                                    grad_C_tanU_R, cross(float3{0, 1, 0}, surfel.tanU)) + dot(
                                                    grad_C_tanV_R, cross(float3{0, 1, 0}, surfel.tanV)),
                                                dot(
                                                    grad_C_tanU_G, cross(float3{0, 1, 0}, surfel.tanU)) + dot(
                                                    grad_C_tanV_G, cross(float3{0, 1, 0}, surfel.tanV)),
                                                dot(
                                                    grad_C_tanU_B, cross(float3{0, 1, 0}, surfel.tanU)) + dot(
                                                    grad_C_tanV_B, cross(float3{0, 1, 0}, surfel.tanV))
                                            };

                                            if (rayState.bounceIndex >= recordBounceIndex) {
                                                const float dVdp_scalar_R = dot(grad_C_pos_R, parameterAxis);
                                                const float dVdp_scalar_G = dot(grad_C_pos_G, parameterAxis);
                                                const float dVdp_scalar_B = dot(grad_C_pos_B, parameterAxis);
                                                float4 &gradImageDst = gradients.framebuffer[rayState.pixelIndex];
                                                float4 gradVectorRGB = {
                                                    grad_C_opacity_R, grad_C_opacity_G, grad_C_opacity_B, 0.0f
                                                };
                                                //float4 gradVectorRGB = {dVdp_scalar_R, dVdp_scalar_G, dVdp_scalar_B, 0.0f};
                                                //float4 gradVectorRGB = {dBsdf_world[0], dBsdf_world[1], dBsdf_world[2], 0.0f};
                                                atomicAddFloat4ToImage(&gradImageDst, gradVectorRGB);
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
                                        const auto &splatEvent = whScatter.splatEvents[eventIdx];
                                        const Point &surfel = scene.points[splatEvent.primitiveIndex];
                                        const float oneMinusAlpha = 1.0f - splatEvent.alpha * surfel.opacity;
                                        if (oneMinusAlpha <= 1e-6f)
                                            continue;
                                        // Update τ
                                        tau *= oneMinusAlpha;
                                    }
                                }


                                    const InstanceRecord &surfelInstance = scene.instances[whScatter.instanceIndex];
                                    if (surfelInstance.geometryType == GeometryType::PointCloud) {
                                        // 1) Build intervening splat list on (t_x, t_y), excluding terminal scatter surfel
                                        // Do this however through a shadow ray
                                        const Ray &ray = rayState.ray;
                                        // 3) Terminal scatter at z: dα/dc with correct sign
                                        const uint32_t terminalPrim = whScatter.primitiveIndex;
                                        const SplatEvent &terminal = whScatter.splatEvents[
                                            whScatter.splatEventCount - 1]; {
                                            const auto surfel = scene.points[terminal.primitiveIndex];
                                            const float3 canonicalNormalWorld = normalize(
                                                cross(surfel.tanU, surfel.tanV));
                                            const float3 rayDirection = ray.direction;
                                            const float3 hitWorld = terminal.hitWorld;

                                            const float2 uv = phiInverse(hitWorld, surfel);
                                            const float u = uv.x();
                                            const float v = uv.y();
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


                                            // ---- dα/d(scale) (s_u, s_v) ----
                                            const float3 dUdVdScale =
                                                    computeDuvDScale(
                                                        u, v,
                                                        su, sv
                                                    );

                                            // Example: BSDF gradients for each parameter group
                                            const float3 dFsDPosition = DuvPosition;
                                            // dα/ds_u = (∂α/∂u)(∂u/∂s_u) = (-u α) * (du/dsu)
                                            const float dFsDsu = u * dUdVdScale.x();
                                            // dα/ds_v = (∂α/∂v)(∂v/∂s_v) = (-v α) * (dv/dsv)
                                            const float dFsDsv = v * dUdVdScale.y();

                                            const float3 surfelRadianceRGB = estimateSurfelRadianceFromPhotonMap(
                                                terminal, ray.direction, scene, photonMap,
                                                false, true, true);

                                            // 5) Final gradient assembly
                                            float3 pAdjoint = rayState.pathThroughput;
                                            float gradsu = dFsDsu;
                                            float gradsv = dFsDsv;

                                            float3 brdfGrad_Position = dFsDPosition * surfel.opacity * (-alpha);
                                            float3 brdfGrad_TanU = dFsDtU * surfel.opacity * (-alpha);
                                            float3 brdfGrad_TanV = dFsDtV * surfel.opacity * (-alpha);
                                            float brdfGrad_scaleU = gradsu * surfel.opacity * (-alpha);
                                            float brdfGrad_scaleV = gradsv * surfel.opacity * (-alpha);

                                            float brdfGrad_albedo = alpha * surfel.opacity;
                                            float3 brdfGrad_opacity = alpha * surfel.color;


                                            float R = pAdjoint[0] * tau * surfelRadianceRGB[0];
                                            float G = pAdjoint[1] * tau * surfelRadianceRGB[1];
                                            float B = pAdjoint[2] * tau * surfelRadianceRGB[2];

                                            // Cost gradients: ∂C/∂(·) = p * L_bg * dτ/d(·)
                                            const float3 grad_C_pos_R = R * brdfGrad_Position;
                                            const float3 grad_C_pos_G = G * brdfGrad_Position;
                                            const float3 grad_C_pos_B = B * brdfGrad_Position;
                                            const float3 grad_C_tanU_R = R * brdfGrad_TanU;
                                            const float3 grad_C_tanU_G = G * brdfGrad_TanU;
                                            const float3 grad_C_tanU_B = B * brdfGrad_TanU;
                                            const float3 grad_C_tanV_R = R * brdfGrad_TanV;
                                            const float3 grad_C_tanV_G = G * brdfGrad_TanV;
                                            const float3 grad_C_tanV_B = B * brdfGrad_TanV;
                                            const float grad_C_scaleU_R = R * brdfGrad_scaleU;
                                            const float grad_C_scaleU_G = G * brdfGrad_scaleU;
                                            const float grad_C_scaleU_B = B * brdfGrad_scaleU;
                                            const float grad_C_scaleV_R = R * brdfGrad_scaleV;
                                            const float grad_C_scaleV_G = G * brdfGrad_scaleV;
                                            const float grad_C_scaleV_B = B * brdfGrad_scaleV;

                                            const float grad_C_albedo_R = R * brdfGrad_albedo;
                                            const float grad_C_albedo_G = G * brdfGrad_albedo;
                                            const float grad_C_albedo_B = B * brdfGrad_albedo;

                                            const float grad_C_opactiy_R = R * brdfGrad_opacity[0];
                                            const float grad_C_opactiy_G = G * brdfGrad_opacity[1];
                                            const float grad_C_opactiy_B = B * brdfGrad_opacity[2];


                                            uint32_t primIdx = terminalPrim;
                                            // Accumulate
                                            atomicAddFloat3(gradients.gradPosition[primIdx],
                                                            grad_C_pos_R + grad_C_pos_G + grad_C_pos_B);
                                            atomicAddFloat3(gradients.gradTanU[primIdx],
                                                            grad_C_tanU_R + grad_C_tanU_G + grad_C_tanU_B);
                                            atomicAddFloat3(gradients.gradTanV[primIdx],
                                                            grad_C_tanV_R + grad_C_tanV_G + grad_C_tanV_B);
                                            atomicAddFloat(gradients.gradScale[primIdx].x(),
                                                           grad_C_scaleU_R + grad_C_scaleU_G + grad_C_scaleU_B);
                                            atomicAddFloat(gradients.gradScale[primIdx].y(),
                                                           grad_C_scaleV_R + grad_C_scaleV_G + grad_C_scaleV_B);

                                            atomicAddFloat(gradients.gradOpacity[primIdx],
                                                           grad_C_opactiy_R + grad_C_opactiy_G + grad_C_opactiy_B);

                                            float3 gradColorValue{grad_C_albedo_R, grad_C_albedo_G, grad_C_albedo_B};
                                            atomicAddFloat3(gradients.gradColor[primIdx], gradColorValue);

                                            // Mapping to world coordinates:
                                            float3 dBsdf_world = float3{
                                                dot(
                                                    grad_C_tanU_R, cross(float3{0, 1, 0}, surfel.tanU)) + dot(
                                                    grad_C_tanV_R, cross(float3{0, 1, 0}, surfel.tanV)),
                                                dot(
                                                    grad_C_tanU_G, cross(float3{0, 1, 0}, surfel.tanU)) + dot(
                                                    grad_C_tanV_G, cross(float3{0, 1, 0}, surfel.tanV)),
                                                dot(
                                                    grad_C_tanU_B, cross(float3{0, 1, 0}, surfel.tanU)) + dot(
                                                    grad_C_tanV_B, cross(float3{0, 1, 0}, surfel.tanV))
                                            };

                                            if (rayState.bounceIndex >= recordBounceIndex) {
                                                const float dVdp_scalar_R = dot(grad_C_pos_R, parameterAxis);
                                                const float dVdp_scalar_G = dot(grad_C_pos_G, parameterAxis);
                                                const float dVdp_scalar_B = dot(grad_C_pos_B, parameterAxis);
                                                float4 &gradImageDst = gradients.framebuffer[rayState.pixelIndex];
                                                float4 gradVectorRGB = {
                                                    grad_C_opactiy_R, grad_C_opactiy_G, grad_C_opactiy_B, 0.0f
                                                };
                                                //float4 gradVectorRGB = {dVdp_scalar_R, dVdp_scalar_G, dVdp_scalar_B, 0.0f};
                                                atomicAddFloat4ToImage(&gradImageDst, gradVectorRGB);
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
                            }
                            )
                            ;
                        });
                    queue.wait();
                }


        void launchAdjointKernel2(RenderPackage &pkg, uint32_t activeRayCount) {
            auto &queue = pkg.queue;
            auto &scene = pkg.scene;
            auto &sensor = pkg.sensor;
            auto &settings = pkg.settings;
            auto &intermediates = pkg.intermediates;
            auto &gradients = pkg.gradients;
            auto &photonMap = pkg.intermediates.map;
            auto *raysIn = pkg.intermediates.primaryRays;

            queue.submit([&](sycl::handler &cgh) {
                cgh.parallel_for<struct AdjointShadeKernelTag>(
                    sycl::range<1>(activeRayCount),
                    // ReSharper disable once CppDFAUnusedValue
                    [=](sycl::id<1> globalId) {
                        const uint32_t rayIndex = globalId[0];
                        const uint64_t perItemSeed = rng::makePerItemSeed1D(settings.randomSeed, rayIndex);
                        rng::Xorshift128 rng128(perItemSeed);
                        const RayState &rayState = intermediates.primaryRays[rayIndex];

                        uint32_t recordBounceIndex = 0;

                        // Shoot one transmit ray. The amount intersected here will tell us how many scatter rays we will transmit.
                        WorldHit whTransmit{};
                        intersectScene(rayState.ray, &whTransmit, scene, rng128, RayIntersectMode::Transmit);

                        if (!whTransmit.hit)
                            return;

                        uint32_t pixelX = rayIndex % sensor.camera.width;
                        uint32_t pixelY = (rayIndex / sensor.camera.width);

                        const bool isWatched = isWatchedPixel(pixelX, pixelY);

                        const InstanceRecord &meshInstance = scene.instances[whTransmit.instanceIndex];
                        const float3 parameterAxis = {1.0f, 0.0f, 0.00f};

                        // Transmission gradients with shadow rays
                        if (meshInstance.geometryType == GeometryType::Mesh) {
                            // Transmission
                            if (whTransmit.splatEventCount > 0) {
                                const Ray &ray = rayState.ray;
                                const float3 x = ray.origin;
                                const float3 y = whTransmit.hitPositionW;

                                // Cost weighting: keep RGB if your loss is RGB; else reduce at end
                                const float3 backgroundRadianceRGB = estimateRadianceFromPhotonMap(
                                    whTransmit, scene, photonMap);

                                // Collect all alpha_i and d(alpha_i)/dPi for this segment
                                struct LocalTerm {
                                    float alpha{};
                                    float eta{};
                                    float3 dAlphaDPos; // dα/d(center)
                                    float3 dAlphaDtU; // dα/d(tanU)
                                    float3 dAlphaDtV; // dα/d(tanV)
                                    float dAlphaDsu{}; // dα/ds_u
                                    float dAlphaDsv{}; // dα/ds_v
                                    uint32_t primitiveIndex{};
                                };

                                LocalTerm localTerms[kMaxSplatEvents];
                                int validCount = 0;
                                float tau = 1.0f; // full product over all (1 - alpha_i)

                                for (size_t eventIdx = 0; eventIdx < whTransmit.splatEventCount; ++eventIdx) {
                                    if (validCount >= kMaxSplatEvents)
                                        break;

                                    const auto &splatEvent = whTransmit.splatEvents[eventIdx];
                                    const Point &surfel = scene.points[splatEvent.primitiveIndex];

                                    const float3 canonicalNormalW =
                                            normalize(cross(surfel.tanU, surfel.tanV));

                                    const float denom = dot(canonicalNormalW, ray.direction);
                                    if (sycl::fabs(denom) <= 1e-4f)
                                        continue; // skip grazing

                                    const float2 uv = phiInverse(splatEvent.hitWorld, surfel);
                                    const float u = uv.x();
                                    const float v = uv.y();
                                    const float alpha = splatEvent.alpha;

                                    const float su = surfel.scale.x();
                                    const float sv = surfel.scale.y();

                                    // ---- dα/d(position) ----
                                    const float3 DuvDpos =
                                            computeDuvDPosition(
                                                surfel.tanU,
                                                surfel.tanV,
                                                canonicalNormalW,
                                                ray.direction,
                                                u, v,
                                                su, sv
                                            );

                                    // ---- dα/d(tangents) via your existing helper ----
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

                                    float3 dAlphaDtU = (u * dUdTu + v * dVdTu);
                                    float3 dAlphaDtV = (u * dUdTv + v * dVdTv);

                                    // ---- dα/d(scale) (kernel-only; no area Jacobian in τ) ----
                                    const float3 dUdVdScale =
                                            computeDuvDScale(u, v, su, sv);
                                    // du/dsu = -u/su
                                    // dv/dsv = -v/sv
                                    const float duDsu = dUdVdScale.x();
                                    const float dvDsv = dUdVdScale.y();

                                    // dα/ds_u = (∂α/∂u)(∂u/∂s_u) = (-u α) * duDsu
                                    const float dAlphaDsu = u * duDsu;
                                    // dα/ds_v = (∂α/∂v)(∂v/∂s_v) = (-v α) * dvDsv
                                    const float dAlphaDsv = v * dvDsv;

                                    const float oneMinusAlpha = 1.0f - alpha * surfel.opacity;
                                    if (oneMinusAlpha <= 1e-6f)
                                        continue;


                                    // Store local term
                                    auto &lt = localTerms[validCount++];
                                    lt.alpha = alpha;
                                    lt.eta = surfel.opacity;
                                    lt.dAlphaDPos = DuvDpos;
                                    lt.dAlphaDtU = dAlphaDtU;
                                    lt.dAlphaDtV = dAlphaDtV;
                                    lt.dAlphaDsu = dAlphaDsu;
                                    lt.dAlphaDsv = dAlphaDsv;
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
                                        const float weight = tau * (alpha * eta) / (1.0f - alpha * eta);
                                        // NOTE: no extra alpha, and minus sign

                                        const float3 dTauDPos = weight * localTerms[i].dAlphaDPos;
                                        const float3 dTauDtU = weight * localTerms[i].dAlphaDtU;
                                        const float3 dTauDtV = weight * localTerms[i].dAlphaDtV;
                                        const float dTauDsu = weight * localTerms[i].dAlphaDsu;
                                        const float dTauDsv = weight * localTerms[i].dAlphaDsv;
                                        const float dTauDeta = -tau * alpha / (1 - eta * alpha);
                                        // dτ/ds_u, dτ/ds_v

                                        float R = pAdjoint[0] * backgroundRadianceRGB[0];
                                        float G = pAdjoint[1] * backgroundRadianceRGB[1];
                                        float B = pAdjoint[2] * backgroundRadianceRGB[2];

                                        // Cost gradients: ∂C/∂(·) = p * L_bg * dτ/d(·)
                                        const float3 grad_C_pos_R = R * dTauDPos;
                                        const float3 grad_C_pos_G = G * dTauDPos;
                                        const float3 grad_C_pos_B = B * dTauDPos;
                                        const float3 grad_C_tanU_R = R * dTauDtU;
                                        const float3 grad_C_tanU_G = G * dTauDtU;
                                        const float3 grad_C_tanU_B = B * dTauDtU;
                                        const float3 grad_C_tanV_R = R * dTauDtV;
                                        const float3 grad_C_tanV_G = G * dTauDtV;
                                        const float3 grad_C_tanV_B = B * dTauDtV;
                                        const float grad_C_scaleU_R = R * dTauDsu;
                                        const float grad_C_scaleU_G = G * dTauDsu;
                                        const float grad_C_scaleU_B = B * dTauDsu;
                                        const float grad_C_scaleV_R = R * dTauDsv;
                                        const float grad_C_scaleV_G = G * dTauDsv;
                                        const float grad_C_scaleV_B = B * dTauDsv;

                                        const float grad_C_opacity_R = R * dTauDeta;
                                        const float grad_C_opacity_G = G * dTauDeta;
                                        const float grad_C_opacity_B = B * dTauDeta;

                                        // Accumulate
                                        atomicAddFloat3(gradients.gradPosition[primIdx],
                                                        grad_C_pos_R + grad_C_pos_G + grad_C_pos_B);
                                        atomicAddFloat3(gradients.gradTanU[primIdx],
                                                        grad_C_tanU_R + grad_C_tanU_G + grad_C_tanU_B);
                                        atomicAddFloat3(gradients.gradTanV[primIdx],
                                                        grad_C_tanV_R + grad_C_tanV_G + grad_C_tanV_B);
                                        atomicAddFloat(gradients.gradScale[primIdx].x(),
                                                       grad_C_scaleU_R + grad_C_scaleU_G + grad_C_scaleU_B);
                                        atomicAddFloat(gradients.gradScale[primIdx].y(),
                                                       grad_C_scaleV_R + grad_C_scaleV_G + grad_C_scaleV_B);

                                        atomicAddFloat(gradients.gradOpacity[primIdx],
                                                       grad_C_opacity_R + grad_C_opacity_G + grad_C_opacity_B);
                                        // Mapping to world coordinates:
                                        const auto &surfel = scene.points[primIdx];
                                        float3 dBsdf_world = float3{
                                            dot(
                                                grad_C_tanU_R, cross(float3{0, 1, 0}, surfel.tanU)) + dot(
                                                grad_C_tanV_R, cross(float3{0, 1, 0}, surfel.tanV)),
                                            dot(
                                                grad_C_tanU_G, cross(float3{0, 1, 0}, surfel.tanU)) + dot(
                                                grad_C_tanV_G, cross(float3{0, 1, 0}, surfel.tanV)),
                                            dot(
                                                grad_C_tanU_B, cross(float3{0, 1, 0}, surfel.tanU)) + dot(
                                                grad_C_tanV_B, cross(float3{0, 1, 0}, surfel.tanV))
                                        };

                                        if (rayState.bounceIndex >= recordBounceIndex) {
                                            const float dVdp_scalar_R = dot(grad_C_pos_R, parameterAxis);
                                            const float dVdp_scalar_G = dot(grad_C_pos_G, parameterAxis);
                                            const float dVdp_scalar_B = dot(grad_C_pos_B, parameterAxis);
                                            float4 &gradImageDst = gradients.framebuffer[rayState.pixelIndex];
                                            float4 gradVectorRGB = {
                                                grad_C_opacity_R, grad_C_opacity_G, grad_C_opacity_B, 0.0f
                                            };
                                            //float4 gradVectorRGB = {dBsdf_world[0], dBsdf_world[1], dBsdf_world[2], 0.0f};
                                            //float4 gradVectorRGB = {dVdp_scalar_R, dVdp_scalar_G, dVdp_scalar_B, 0.0f};

                                            atomicAddFloat4ToImage(&gradImageDst, gradVectorRGB);
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
                        } {
                            AreaLightSample ls = sampleMeshAreaLightReuse(scene, rng128);
                            // Direction to the sampled emitter point
                            const float3 toLightVector = ls.positionW - rayState.ray.origin;
                            const float distanceToLight = length(toLightVector);
                            if (distanceToLight > 1e-6f) {
                                const float3 lightDirection = toLightVector / distanceToLight;
                                // Cosines
                                const float3 shadingNormalW = rayState.ray.normal;
                                const float cosThetaSurface = sycl::max(0.0f, dot(shadingNormalW, lightDirection));
                                const float cosThetaLight = sycl::max(0.0f, dot(ls.normalW, -lightDirection));


                                if (cosThetaSurface != 0.0f && cosThetaLight != 0.0f) {
                                    const float r2 = distanceToLight * distanceToLight;
                                    const float geometryTerm = (cosThetaSurface * cosThetaLight) / r2;
                                    // PDFs from the sampler
                                    const float pdfArea = ls.pdfArea; // area-domain, world area
                                    const float pdfLight = ls.pdfSelectLight; // 1 / lightCount
                                    // Unbiased NEE estimator (area sampling):
                                    const float invPdf = 1.0f / (pdfLight * pdfArea);
                                    const float3 neeContribution =
                                            rayState.pathThroughput * geometryTerm * invPdf;

                                    Ray shadowRay{rayState.ray.origin, lightDirection};
                                    RayState shadowRayState = rayState;
                                    shadowRayState.ray = shadowRay;
                                    shadowRayState.pathThroughput = neeContribution;

                                    // BRDF
                                    WorldHit shadowWorldHit{};
                                    intersectScene(shadowRayState.ray, &shadowWorldHit, scene, rng128,
                                                   RayIntersectMode::Transmit);

                                    if (shadowWorldHit.hit) {
                                        const Ray &ray = shadowRay;
                                        const float3 x = ray.origin;
                                        const float3 y = shadowWorldHit.hitPositionW;

                                        // Cost weighting: keep RGB if your loss is RGB; else reduce at end
                                        const float3 backgroundRadianceRGB = estimateRadianceFromPhotonMap(
                                            shadowWorldHit, scene, photonMap);

                                        // Collect all alpha_i and d(alpha_i)/dPi for this segment
                                        struct LocalTerm {
                                            float alpha{};
                                            float eta{};
                                            float3 dAlphaDPos; // dα/d(center)
                                            float3 dAlphaDtU; // dα/d(tanU)
                                            float3 dAlphaDtV; // dα/d(tanV)
                                            float dAlphaDsu{}; // dα/ds_u
                                            float dAlphaDsv{}; // dα/ds_v
                                            uint32_t primitiveIndex{};
                                        };

                                        LocalTerm localTerms[kMaxSplatEvents];
                                        int validCount = 0;
                                        float tau = 1.0f; // full product over all (1 - alpha_i)

                                        for (size_t eventIdx = 0; eventIdx < shadowWorldHit.splatEventCount; ++
                                             eventIdx) {
                                            if (validCount >= kMaxSplatEvents)
                                                break;

                                            const auto &splatEvent = shadowWorldHit.splatEvents[eventIdx];
                                            const Point &surfel = scene.points[splatEvent.primitiveIndex];

                                            const float3 canonicalNormalW =
                                                    normalize(cross(surfel.tanU, surfel.tanV));

                                            const float denom = dot(canonicalNormalW, ray.direction);
                                            if (sycl::fabs(denom) <= 1e-4f)
                                                continue; // skip grazing

                                            const float2 uv = phiInverse(splatEvent.hitWorld, surfel);
                                            const float u = uv.x();
                                            const float v = uv.y();
                                            const float alpha = splatEvent.alpha;

                                            const float su = surfel.scale.x();
                                            const float sv = surfel.scale.y();

                                            // ---- dα/d(position) ----
                                            const float3 DuvDpos =
                                                    computeDuvDPosition(
                                                        surfel.tanU,
                                                        surfel.tanV,
                                                        canonicalNormalW,
                                                        ray.direction,
                                                        u, v,
                                                        su, sv
                                                    );

                                            // ---- dα/d(tangents) via your existing helper ----
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

                                            float3 dAlphaDtU = (u * dUdTu + v * dVdTu);
                                            float3 dAlphaDtV = (u * dUdTv + v * dVdTv);

                                            // ---- dα/d(scale) (kernel-only; no area Jacobian in τ) ----
                                            const float3 dUdVdScale =
                                                    computeDuvDScale(u, v, su, sv);
                                            // du/dsu = -u/su
                                            // dv/dsv = -v/sv
                                            const float duDsu = dUdVdScale.x();
                                            const float dvDsv = dUdVdScale.y();

                                            // dα/ds_u = (∂α/∂u)(∂u/∂s_u) = (-u α) * duDsu
                                            const float dAlphaDsu = u * duDsu;
                                            // dα/ds_v = (∂α/∂v)(∂v/∂s_v) = (-v α) * dvDsv
                                            const float dAlphaDsv = v * dvDsv;

                                            const float oneMinusAlpha = 1.0f - alpha * surfel.opacity;
                                            if (oneMinusAlpha <= 1e-6f)
                                                continue;


                                            // Store local term
                                            auto &lt = localTerms[validCount++];
                                            lt.alpha = alpha;
                                            lt.eta = surfel.opacity;
                                            lt.dAlphaDPos = DuvDpos;
                                            lt.dAlphaDtU = dAlphaDtU;
                                            lt.dAlphaDtV = dAlphaDtV;
                                            lt.dAlphaDsu = dAlphaDsu;
                                            lt.dAlphaDsv = dAlphaDsv;
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
                                                const float weight = tau * (alpha * eta) / (1.0f - alpha * eta);
                                                // NOTE: no extra alpha, and minus sign

                                                const float3 dTauDPos = weight * localTerms[i].dAlphaDPos;
                                                const float3 dTauDtU = weight * localTerms[i].dAlphaDtU;
                                                const float3 dTauDtV = weight * localTerms[i].dAlphaDtV;
                                                const float dTauDsu = weight * localTerms[i].dAlphaDsu;
                                                const float dTauDsv = weight * localTerms[i].dAlphaDsv;
                                                const float dTauDeta = -tau * alpha / (1 - eta * alpha);
                                                // dτ/ds_u, dτ/ds_v

                                                float R = pAdjoint[0] * backgroundRadianceRGB[0];
                                                float G = pAdjoint[1] * backgroundRadianceRGB[1];
                                                float B = pAdjoint[2] * backgroundRadianceRGB[2];

                                                // Cost gradients: ∂C/∂(·) = p * L_bg * dτ/d(·)
                                                const float3 grad_C_pos_R = R * dTauDPos;
                                                const float3 grad_C_pos_G = G * dTauDPos;
                                                const float3 grad_C_pos_B = B * dTauDPos;
                                                const float3 grad_C_tanU_R = R * dTauDtU;
                                                const float3 grad_C_tanU_G = G * dTauDtU;
                                                const float3 grad_C_tanU_B = B * dTauDtU;
                                                const float3 grad_C_tanV_R = R * dTauDtV;
                                                const float3 grad_C_tanV_G = G * dTauDtV;
                                                const float3 grad_C_tanV_B = B * dTauDtV;
                                                const float grad_C_scaleU_R = R * dTauDsu;
                                                const float grad_C_scaleU_G = G * dTauDsu;
                                                const float grad_C_scaleU_B = B * dTauDsu;
                                                const float grad_C_scaleV_R = R * dTauDsv;
                                                const float grad_C_scaleV_G = G * dTauDsv;
                                                const float grad_C_scaleV_B = B * dTauDsv;

                                                const float grad_C_opacity_R = R * dTauDeta;
                                                const float grad_C_opacity_G = G * dTauDeta;
                                                const float grad_C_opacity_B = B * dTauDeta;

                                                // Accumulate
                                                atomicAddFloat3(gradients.gradPosition[primIdx],
                                                                grad_C_pos_R + grad_C_pos_G + grad_C_pos_B);
                                                atomicAddFloat3(gradients.gradTanU[primIdx],
                                                                grad_C_tanU_R + grad_C_tanU_G + grad_C_tanU_B);
                                                atomicAddFloat3(gradients.gradTanV[primIdx],
                                                                grad_C_tanV_R + grad_C_tanV_G + grad_C_tanV_B);
                                                atomicAddFloat(gradients.gradScale[primIdx].x(),
                                                               grad_C_scaleU_R + grad_C_scaleU_G + grad_C_scaleU_B);
                                                atomicAddFloat(gradients.gradScale[primIdx].y(),
                                                               grad_C_scaleV_R + grad_C_scaleV_G + grad_C_scaleV_B);

                                                atomicAddFloat(gradients.gradOpacity[primIdx],
                                                               grad_C_opacity_R + grad_C_opacity_G + grad_C_opacity_B);
                                                // Mapping to world coordinates:
                                                const auto &surfel = scene.points[primIdx];
                                                float3 dBsdf_world = float3{
                                                    dot(
                                                        grad_C_tanU_R, cross(float3{0, 1, 0}, surfel.tanU)) + dot(
                                                        grad_C_tanV_R, cross(float3{0, 1, 0}, surfel.tanV)),
                                                    dot(
                                                        grad_C_tanU_G, cross(float3{0, 1, 0}, surfel.tanU)) + dot(
                                                        grad_C_tanV_G, cross(float3{0, 1, 0}, surfel.tanV)),
                                                    dot(
                                                        grad_C_tanU_B, cross(float3{0, 1, 0}, surfel.tanU)) + dot(
                                                        grad_C_tanV_B, cross(float3{0, 1, 0}, surfel.tanV))
                                                };

                                                if (rayState.bounceIndex >= recordBounceIndex) {
                                                    const float dVdp_scalar_R = dot(grad_C_pos_R, parameterAxis);
                                                    const float dVdp_scalar_G = dot(grad_C_pos_G, parameterAxis);
                                                    const float dVdp_scalar_B = dot(grad_C_pos_B, parameterAxis);
                                                    float4 &gradImageDst = gradients.framebuffer[rayState.pixelIndex];
                                                    float4 gradVectorRGB = {
                                                        grad_C_opacity_R, grad_C_opacity_G, grad_C_opacity_B, 0.0f
                                                    };
                                                    //float4 gradVectorRGB = {dVdp_scalar_R, dVdp_scalar_G, dVdp_scalar_B, 0.0f};
                                                    //float4 gradVectorRGB = {dBsdf_world[0], dBsdf_world[1], dBsdf_world[2], 0.0f};
                                                    atomicAddFloat4ToImage(&gradImageDst, gradVectorRGB);
                                                }
                                            }
                                        }
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
        const auto &splatEvent = whScatter.splatEvents[eventIdx];
        const Point &surfel = scene.points[splatEvent.primitiveIndex];
        const float oneMinusAlpha = 1.0f - splatEvent.alpha * surfel.opacity;
        if (oneMinusAlpha <= 1e-6f)
            continue;
        // Update τ
        tau *= oneMinusAlpha;
    }
}

                            const InstanceRecord &surfelInstance = scene.instances[whScatter.instanceIndex];
                            if (surfelInstance.geometryType == GeometryType::PointCloud) {
                                // 1) Build intervening splat list on (t_x, t_y), excluding terminal scatter surfel
                                // Do this however through a shadow ray
                                const Ray &ray = rayState.ray;
                                // 3) Terminal scatter at z: dα/dc with correct sign
                                const uint32_t terminalPrim = whScatter.primitiveIndex;
                                const SplatEvent &terminal = whScatter.splatEvents[whScatter.splatEventCount - 1]; {
                                    const auto surfel = scene.points[terminal.primitiveIndex];
                                    const float3 canonicalNormalWorld = normalize(cross(surfel.tanU, surfel.tanV));
                                    const float3 rayDirection = ray.direction;
                                    const float3 hitWorld = terminal.hitWorld;

                                    const float2 uv = phiInverse(hitWorld, surfel);
                                    const float u = uv.x();
                                    const float v = uv.y();
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


                                    // ---- dα/d(scale) (s_u, s_v) ----
                                    const float3 dUdVdScale =
                                            computeDuvDScale(
                                                u, v,
                                                su, sv
                                            );

                                    // Example: BSDF gradients for each parameter group
                                    const float3 dFsDPosition = DuvPosition;
                                    // dα/ds_u = (∂α/∂u)(∂u/∂s_u) = (-u α) * (du/dsu)
                                    const float dFsDsu = u * dUdVdScale.x();
                                    // dα/ds_v = (∂α/∂v)(∂v/∂s_v) = (-v α) * (dv/dsv)
                                    const float dFsDsv = v * dUdVdScale.y();

                                    const float3 surfelRadianceRGB = estimateSurfelRadianceFromPhotonMap(
                                        terminal, ray.direction, scene, photonMap,
                                        false, true, true);

                                    // 5) Final gradient assembly
                                    float3 pAdjoint = rayState.pathThroughput;
                                    float gradsu = dFsDsu;
                                    float gradsv = dFsDsv;

                                    float3 brdfGrad_Position = dFsDPosition * surfel.opacity * (-alpha);
                                    float3 brdfGrad_TanU = dFsDtU * surfel.opacity * (-alpha);
                                    float3 brdfGrad_TanV = dFsDtV * surfel.opacity * (-alpha);
                                    float brdfGrad_scaleU = gradsu * surfel.opacity * (-alpha);
                                    float brdfGrad_scaleV = gradsv * surfel.opacity * (-alpha);

                                    float brdfGrad_albedo = alpha * surfel.opacity;
                                    float3 brdfGrad_opacity = alpha * surfel.color;


                                    float R = pAdjoint[0] * tau * surfelRadianceRGB[0];
                                    float G = pAdjoint[1] * tau * surfelRadianceRGB[1];
                                    float B = pAdjoint[2] * tau * surfelRadianceRGB[2];

                                    // Cost gradients: ∂C/∂(·) = p * L_bg * dτ/d(·)
                                    const float3 grad_C_pos_R = R * brdfGrad_Position;
                                    const float3 grad_C_pos_G = G * brdfGrad_Position;
                                    const float3 grad_C_pos_B = B * brdfGrad_Position;
                                    const float3 grad_C_tanU_R = R * brdfGrad_TanU;
                                    const float3 grad_C_tanU_G = G * brdfGrad_TanU;
                                    const float3 grad_C_tanU_B = B * brdfGrad_TanU;
                                    const float3 grad_C_tanV_R = R * brdfGrad_TanV;
                                    const float3 grad_C_tanV_G = G * brdfGrad_TanV;
                                    const float3 grad_C_tanV_B = B * brdfGrad_TanV;
                                    const float grad_C_scaleU_R = R * brdfGrad_scaleU;
                                    const float grad_C_scaleU_G = G * brdfGrad_scaleU;
                                    const float grad_C_scaleU_B = B * brdfGrad_scaleU;
                                    const float grad_C_scaleV_R = R * brdfGrad_scaleV;
                                    const float grad_C_scaleV_G = G * brdfGrad_scaleV;
                                    const float grad_C_scaleV_B = B * brdfGrad_scaleV;

                                    const float grad_C_albedo_R = R * brdfGrad_albedo;
                                    const float grad_C_albedo_G = G * brdfGrad_albedo;
                                    const float grad_C_albedo_B = B * brdfGrad_albedo;

                                    const float grad_C_opactiy_R = R * brdfGrad_opacity[0];
                                    const float grad_C_opactiy_G = G * brdfGrad_opacity[1];
                                    const float grad_C_opactiy_B = B * brdfGrad_opacity[2];


                                    uint32_t primIdx = terminalPrim;
                                    // Accumulate
                                    atomicAddFloat3(gradients.gradPosition[primIdx],
                                                    grad_C_pos_R + grad_C_pos_G + grad_C_pos_B);
                                    atomicAddFloat3(gradients.gradTanU[primIdx],
                                                    grad_C_tanU_R + grad_C_tanU_G + grad_C_tanU_B);
                                    atomicAddFloat3(gradients.gradTanV[primIdx],
                                                    grad_C_tanV_R + grad_C_tanV_G + grad_C_tanV_B);
                                    atomicAddFloat(gradients.gradScale[primIdx].x(),
                                                   grad_C_scaleU_R + grad_C_scaleU_G + grad_C_scaleU_B);
                                    atomicAddFloat(gradients.gradScale[primIdx].y(),
                                                   grad_C_scaleV_R + grad_C_scaleV_G + grad_C_scaleV_B);

                                    atomicAddFloat(gradients.gradOpacity[primIdx],
                                                   grad_C_opactiy_R + grad_C_opactiy_G + grad_C_opactiy_B);

                                    float3 gradColorValue{grad_C_albedo_R, grad_C_albedo_G, grad_C_albedo_B};
                                    atomicAddFloat3(gradients.gradColor[primIdx], gradColorValue);

                                    // Mapping to world coordinates:
                                    float3 dBsdf_world = float3{
                                        dot(
                                            grad_C_tanU_R, cross(float3{0, 1, 0}, surfel.tanU)) + dot(
                                            grad_C_tanV_R, cross(float3{0, 1, 0}, surfel.tanV)),
                                        dot(
                                            grad_C_tanU_G, cross(float3{0, 1, 0}, surfel.tanU)) + dot(
                                            grad_C_tanV_G, cross(float3{0, 1, 0}, surfel.tanV)),
                                        dot(
                                            grad_C_tanU_B, cross(float3{0, 1, 0}, surfel.tanU)) + dot(
                                            grad_C_tanV_B, cross(float3{0, 1, 0}, surfel.tanV))
                                    };

                                    if (rayState.bounceIndex >= recordBounceIndex) {
                                        const float dVdp_scalar_R = dot(grad_C_pos_R, parameterAxis);
                                        const float dVdp_scalar_G = dot(grad_C_pos_G, parameterAxis);
                                        const float dVdp_scalar_B = dot(grad_C_pos_B, parameterAxis);
                                        float4 &gradImageDst = gradients.framebuffer[rayState.pixelIndex];
                                        float4 gradVectorRGB = {
                                            grad_C_opactiy_R, grad_C_opactiy_G, grad_C_opactiy_B, 0.0f
                                        };
                                        //float4 gradVectorRGB = {dVdp_scalar_R, dVdp_scalar_G, dVdp_scalar_B, 0.0f};
                                        //float4 gradVectorRGB = {dBsdf_world[0], dBsdf_world[1], dBsdf_world[2], 0.0f};
                                        atomicAddFloat4ToImage(&gradImageDst, gradVectorRGB);
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
                            material.baseColor = scene.points[worldHit.primitiveIndex].color;
                            const float3 lambertBrdf = material.baseColor * M_1_PIf;

                            const float interactionAlpha = worldHit.splatEvents[worldHit.splatEventCount - 1].alpha;
                            // α
                            const float reflectWeight = 0.5f * interactionAlpha; // ρ_r = α/2
                            const float transmitWeight = 0.5f * interactionAlpha; // ρ_t = α/2

                            // event probabilities
                            const float probReflect = reflectWeight / interactionAlpha;
                            // a 50/50 if we reflect or transmit
                            reflectedRay = (rng128.nextFloat() < probReflect);
                            if (reflectedRay) {
                                float sampledPdf = 0.0f;
                                // Diffuse reflect on entered side
                                sampleCosineHemisphere(rng128, enteredSideNormalW, sampledOutgoingDirectionW,
                                                       sampledPdf);
                                sampledPdf = sycl::fmax(sampledPdf, 1e-6f);
                                const float cosTheta = sycl::fmax(
                                    0.0f, dot(sampledOutgoingDirectionW, enteredSideNormalW));
                                throughputMultiplier =
                                        throughputMultiplier * lambertBrdf * (cosTheta / sampledPdf);
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
                                        throughputMultiplier * lambertBrdf * (cosTheta / sampledPdf);
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
