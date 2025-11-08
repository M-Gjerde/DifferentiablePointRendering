//
// Created by magnus on 9/8/25.
//

#include "Renderer/Kernels/AdjointKernels.h"

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

                    //pixelX = 301;
                    //pixelY = imageHeight - 1 - 932;


                    const uint32_t pixelIndex = pixelLinearIndexWithinImage;
                    // RNG for this pixel
                    const uint64_t perPixelSeed = rng::makePerItemSeed1D(baseSeed, pixelLinearIndexWithinImage);
                    rng::Xorshift128 pixelRng(perPixelSeed);

                    // Adjoint source weight
                    //const float4 residualRgba = adjoint.framebuffer[pixelIndex];
                    //float3 initialAdjointWeight = {residualRgba.x(), residualRgba.y(), residualRgba.z()};
                    // Or unit weights:
                    float3 initialAdjointWeight = float3(1.0f, 1.0f, 1.0f);

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


    void launchAdjointKernel(RenderPackage &pkg, uint32_t activeRayCount) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &sensor = pkg.sensor;
        auto &settings = pkg.settings;
        auto &photonMap = pkg.intermediates.map;
        auto *hitRecords = pkg.intermediates.hitRecords;
        auto *raysIn = pkg.intermediates.primaryRays;

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<struct AdjointShadeKernelTag>(
                sycl::range<1>(activeRayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(settings.randomSeed, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);
                    constexpr float epsilon = 1e-6f;

                    WorldHit worldHit = hitRecords[rayIndex];
                    RayState rayState = raysIn[rayIndex];

                    if (!worldHit.hit)
                        return;
                    //printf("adjoint item %u, hit t: %f SplatEvents: %u \n", rayIndex, worldHit.t, worldHit.splatEventCount);

                    auto &instance = scene.instances[worldHit.instanceIndex];
                    float3 d_grad_pos(0.0f);
                    float3 gradientCostWrtSurfelCenter(0.0f);
                    Ray &ray = rayState.ray;
                    // SHADOW RAY

                    // Transmission gradients with shadow rays
                    if (instance.geometryType == GeometryType::Mesh) {
                        d_grad_pos = transmissionGradients(scene, worldHit, rayState, instance, photonMap, rng128, false);
                        float gradMag = length(d_grad_pos);
                        int debug = 1;
                    }

                    if (instance.geometryType == GeometryType::PointCloud) {
                        struct LocalTerm {
                            float alpha{};
                            float3 dAlphaDc;
                        };
                        // 1) Build intervening splat list on (t_x, t_y), excluding terminal scatter surfel
                        // Do this however through a shadow ray
                        float3 tauGrad{0.0f};
                        float tauTotal = 1.0f;
                        {
                            BoundedVector<LocalTerm, kMaxSplatEvents> intervening;
                            intervening.clear();
                            for (int ei = 0; ei < worldHit.splatEventCount - 1; ++ei) {
                                const auto &evt = worldHit.splatEvents[ei];
                                const auto surfel = scene.points[evt.primitiveIndex];
                                const float3 dAlphaDc = gradTransmissionPosition(
                                    rayState.ray, evt, surfel, (worldHit.hitPositionW - ray.origin));
                                intervening.pushBack({evt.alpha, dAlphaDc});
                            }

                            // 2) τ and ∂τ/∂c from intervening set
                            float logTau = 0.0f;
                            constexpr float epsilon = 1e-6f;
                            for (int i = 0; i < intervening.size(); ++i) {
                                const float oneMinusAlpha = sycl::fmax(1.0f - intervening[i].alpha, epsilon);
                                logTau += sycl::log(oneMinusAlpha);
                            }
                            tauTotal = sycl::exp(logTau);

                            for (int i = 0; i < intervening.size(); ++i) {
                                const float oneMinusAlpha = sycl::fmax(1.0f - intervening[i].alpha, epsilon);
                                tauGrad = tauGrad + (-intervening[i].dAlphaDc) / oneMinusAlpha;
                            }
                            tauGrad = tauGrad * tauTotal;
                        }

                        // 3) Terminal scatter at y: dα/dc with correct sign
                        const auto &terminal = worldHit.splatEvents[worldHit.splatEventCount - 1];
                        if (worldHit.primitiveIndex == terminal.primitiveIndex) {
                            const auto surfel = scene.points[terminal.primitiveIndex];

                            // normals and frame
                            const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));
                            const int travelSideSign =
                                    (dot(canonicalNormalW, -rayState.ray.direction) >= 0.0f) ? 1 : -1;
                            const float3 surfelNormal = (travelSideSign == 1) ? canonicalNormalW : (-canonicalNormalW);

                            // BSDF gradient (Scatter)
                            const float denom = dot(surfelNormal, ray.direction);
                            if (sycl::fabs(denom) <= 1e-6f)
                                return;
                            const float2 uv = phiInverse(terminal.hitWorld, surfel);
                            const float alpha = terminal.alpha;
                            const float tuDotD = dot(surfel.tanU, ray.direction);
                            const float tvDotD = dot(surfel.tanV, ray.direction);
                            const float su = surfel.scale.x();
                            const float sv = surfel.scale.y();
                            const float3 duDc = ((tuDotD / denom) * surfelNormal - surfel.tanU) / su;
                            const float3 dvDc = ((tvDotD / denom) * surfelNormal - surfel.tanV) / sv;
                            // Correct sign and sum
                            const float3 dAlphaDc = -alpha * (uv.x() * duDc + uv.y() * dvDc);
                            float color = luminanceGrayscale(surfel.color);

                            const float fBsdf = (M_1_PIf) * alpha * color;
                            const float3 dFBsdfDc = (M_1_PIf) * dAlphaDc * color;

                            // COSINE derivative
                            const float3 d = worldHit.hitPositionW - ray.origin;
                            const float p = length(d);
                            if (p <= 1e-6f) return;
                            const float3 psi = d / p;
                            float cosine = fmax(0.0f, dot(surfelNormal, -psi));

                            // dG/dc for pure translation of surfel (dy/dc = I, dn/dc = 0)
                            const float3x3 I = identity3x3();
                            const float3x3 P = I - outerProduct(psi, psi);
                            float3 cosineGradPos = (ray.normal * P) / p ;

                            // Transmission gradient from direction change

                            // use in gradient assembly
                            float3 brdfGrad = dFBsdfDc * tauTotal * cosine;
                            float3 cosineGrad = cosineGradPos * tauTotal * fBsdf;
                            float3 transmissionGrad = tauGrad * cosine * fBsdf;

                            // 5) Final gradient assembly
                            const float3 surfelRadianceRGB = estimateSurfelRadianceFromPhotonMap(
                                worldHit.splatEvents[0], ray.direction, scene, photonMap, false);
                            const float L_surfel = luminanceGrayscale(surfelRadianceRGB);
                            const float3 pAdjoint = rayState.pathThroughput;

                            d_grad_pos = pAdjoint * L_surfel * (cosineGrad + brdfGrad + transmissionGrad);

                            //d_grad_pos = pAdjoint * L_surfel * brdfGrad;
                        }
                    }


                    if (rayState.bounceIndex >= 0) {
                        const float3 parameterAxis = {1.0f, 0.0f, 0.00f};
                        const float dVdp_scalar = dot(d_grad_pos, parameterAxis);

                        float4 &gradImageDst = sensor.framebuffer[rayState.pixelIndex];
                        atomicAddFloatToImage(&gradImageDst, dVdp_scalar);
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

                    const InstanceRecord instance = scene.instances[worldHit.instanceIndex];
                    const float3 canonicalNormalW = worldHit.geometricNormalW;
                    const int travelSideSign = signNonZero(dot(canonicalNormalW, -rayState.ray.direction));
                    const float3 enteredSideNormalW = (travelSideSign >= 0) ? canonicalNormalW : (-canonicalNormalW);
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
                    if (instance.geometryType == GeometryType::PointCloud) {
                        // PointCloud
                        GPUMaterial material{};
                        material.baseColor = scene.points[worldHit.primitiveIndex].color;
                        const float3 lambertBrdf = material.baseColor * M_1_PIf;

                        const float interactionAlpha = worldHit.splatEvents[0].alpha; // α
                        const float reflectWeight = 0.0f * interactionAlpha; // ρ_r = α/2
                        const float transmitWeight = 0.5f * interactionAlpha; // ρ_t = α/2

                        // event probabilities
                        const float probReflect = reflectWeight / interactionAlpha; // a 50/50 if we reflect or transmit
                        const bool chooseReflect = (rng128.nextFloat() < probReflect);
                        if (chooseReflect) {
                            float sampledPdf = 0.0f;
                            // Diffuse reflect on entered side
                            sampleCosineHemisphere(rng128, enteredSideNormalW, sampledOutgoingDirectionW, sampledPdf);
                            sampledPdf = sycl::fmax(sampledPdf, 1e-6f);
                            const float cosTheta = sycl::fmax(0.0f, dot(sampledOutgoingDirectionW, enteredSideNormalW));
                            throughputMultiplier =
                                    material.baseColor * throughputMultiplier * lambertBrdf * (cosTheta / sampledPdf);
                        } else {
                            float sampledPdf = 0.0f;
                            // Diffuse transmit: cosine hemisphere on the opposite side
                            const float3 oppositeSideNormalW = -enteredSideNormalW;
                            sampleCosineHemisphere(rng128, oppositeSideNormalW, sampledOutgoingDirectionW, sampledPdf);
                            sampledPdf = sycl::fmax(sampledPdf, 1e-6f);
                            const float cosTheta =
                                    sycl::fmax(0.0f, dot(sampledOutgoingDirectionW, oppositeSideNormalW));
                            throughputMultiplier =
                                    material.baseColor * throughputMultiplier * lambertBrdf * (cosTheta / sampledPdf);
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

                    //if (nextState.bounceIndex >= settings.russianRouletteStart) {
                    //    // Luminance-based continuation probability in [pMin, 1]
                    //    const float3 throughputRgb = nextState.pathThroughput;
                    //    const float luminance = 0.2126f * throughputRgb.x() + 0.7152f * throughputRgb.y() + 0.0722f *
                    //                            throughputRgb.z();
                    //    const float pMin = 0.10f; // safety floor to avoid zero-probability bias
                    //    const float continuationProbability = sycl::clamp(luminance, pMin, 1.0f);
                    //
                    //    if (rng128.nextFloat() >= continuationProbability) {
                    //        raysOut[rayIndex] = nextState;
                    //        hitRecords[rayIndex] = WorldHit{};
                    //        return; // terminate path, do not enqueue
                    //    }
                    //    nextState.pathThroughput = nextState.pathThroughput / continuationProbability; // unbiased
                    //}

                    // compacted enqueue
                    sycl::atomic_ref<uint32_t,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                            activeCounter(*countExtensionOut);

                    uint32_t outputSlot = activeCounter.fetch_add(1);

                    // write only once to compacted slot
                    raysOut[outputSlot] = nextState;
                });
        });
        queue.wait();
    }
}
