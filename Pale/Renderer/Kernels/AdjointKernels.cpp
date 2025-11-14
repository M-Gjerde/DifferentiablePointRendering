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
        float raysTotal = raysPerSet * settings.adjointSamplesPerPixel;

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

    static constexpr uint32_t kDebugTargetY = 622;
    static constexpr uint32_t kDebugTargetXs[] = {565, 585, 600, 622};

    //static constexpr uint32_t kDebugTargetY = 700;
    //static constexpr uint32_t kDebugTargetXs[] = {250, 330, 400, 530};

    SYCL_EXTERNAL inline bool isWatchedPixel(uint32_t pixelX, uint32_t pixelY) {
        if (pixelY != kDebugTargetY) return false;
        bool matchX = false;
        // unrolled small loop
        for (uint32_t i = 0; i < (uint32_t) (sizeof(kDebugTargetXs) / sizeof(uint32_t)); ++i) {
            matchX = matchX || (pixelX == kDebugTargetXs[i]);
        }
        return matchX;
    }


    void launchAdjointKernel(RenderPackage &pkg, uint32_t activeRayCount) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &sensor = pkg.sensor;
        auto &settings = pkg.settings;
        auto &intermediates = pkg.intermediates;
        auto &photonMap = pkg.intermediates.map;
        auto *raysIn = pkg.intermediates.primaryRays;

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<struct AdjointShadeKernelTag>(
                sycl::range<1>(activeRayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(settings.randomSeed, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);
                    constexpr float epsilon = 1e-6f;
                    RayState rayState = intermediates.primaryRays[rayIndex];

                    uint32_t pixelX = rayIndex % sensor.camera.width;
                    uint32_t pixelY = sensor.camera.height - 1 - (rayIndex / sensor.camera.width);

                    const bool isWatched = isWatchedPixel(pixelX, pixelY);
                    uint32_t recordBounceIndex = 0;

                    uint32_t targetSurfel = 0;
                    bool onlySingleSurfel = false;

                    const float3 parameterAxis = {1.0f, 0.0f, 0.00f};
                    float3 shadowRayGrad(0.0f);
                    WorldHit whTransmit{};
                    intersectScene(rayState.ray, &whTransmit, scene, rng128, RayIntersectMode::Random);
                    //intersectScene(rayState.ray, &worldHit, scene, rng128, RayIntersectMode::Scatter);
                    //

                    if (!whTransmit.hit)
                        return;

                    const InstanceRecord &meshInstance = scene.instances[whTransmit.instanceIndex];

                    /*
                    const Triangle &triangle = scene.triangles[whTransmit.primitiveIndex];
                    const Transform &objectWorldTransform = scene.transforms[meshInstance.transformIndex];
                    const Vertex &vertex0 = scene.vertices[triangle.v0];
                    const Vertex &vertex1 = scene.vertices[triangle.v1];
                    const Vertex &vertex2 = scene.vertices[triangle.v2];
                    // Canonical geometric normal (no face-forwarding)
                    const float3 worldP0 = toWorldPoint(vertex0.pos, objectWorldTransform);
                    const float3 worldP1 = toWorldPoint(vertex1.pos, objectWorldTransform);
                    const float3 worldP2 = toWorldPoint(vertex2.pos, objectWorldTransform);
                    const float3 canonicalNormalW = normalize(cross(worldP1 - worldP0, worldP2 - worldP0));
                    whTransmit.geometricNormalW = canonicalNormalW;
                    */


                    // Transmission gradients with shadow rays
                    float3 d_grad_pos(0.0f);

                    if (meshInstance.geometryType == GeometryType::Mesh) {
                        // Transmission
                        if (whTransmit.splatEventCount > 0) {
                            const Ray &ray = rayState.ray;
                            const float3 x = ray.origin;
                            const float3 y = whTransmit.hitPositionW;

                            // Cost weighting: keep RGB if your loss is RGB; else reduce at end
                            const float3 backgroundRadianceRGB = estimateRadianceFromPhotonMap(
                                whTransmit, scene, photonMap);
                            const float L_bg = luminance(backgroundRadianceRGB);

                            // Collect all alpha_i and d(alpha_i)/dPi for this segment
                            struct LocalTerm {
                                float alpha{};
                                float3 dAlphaDc;
                            };

                            LocalTerm localTerms[kMaxSplatEvents];
                            int validCount = 0;

                            float tau = 1.0f; // full product over all (1 - alpha_i)

                            for (size_t eventIdx = 0; eventIdx < whTransmit.splatEventCount; ++eventIdx) {
                                if (validCount >= kMaxSplatEvents)
                                    break; // safety, or assert

                                const auto &splatEvent = whTransmit.splatEvents[eventIdx];

                                const Point &surfel = scene.points[splatEvent.primitiveIndex];

                                const float3 canonicalNormalW =
                                        normalize(cross(surfel.tanU, surfel.tanV));

                                const float denom = dot(canonicalNormalW, ray.direction);
                                if (sycl::fabs(denom) <= 1e-4f) {
                                    // Grazing / parallel; skip this surfel for the transmission gradient
                                    continue;
                                }

                                const float2 uv = phiInverse(splatEvent.hitWorld, surfel);
                                const float alpha = splatEvent.alpha;

                                // Optional: guard against extreme alpha for numerical stability
                                const float clampedAlpha = sycl::clamp(alpha, 0.0f, 1.0f - 1e-6f);

                                const float tuDotD = dot(surfel.tanU, ray.direction);
                                const float tvDotD = dot(surfel.tanV, ray.direction);
                                const float su = surfel.scale.x();
                                const float sv = surfel.scale.y();

                                // Reuse the coordinate gradients from your BRDF derivation
                                const float3 duDc =
                                        ((tuDotD / denom) * canonicalNormalW - surfel.tanU) / su;
                                const float3 dvDc =
                                        ((tvDotD / denom) * canonicalNormalW - surfel.tanV) / sv;

                                // d(alpha_i)/dPi = - alpha_i (u_i du/dPi + v_i dv/dPi)
                                const float3 dAlphaDc =
                                        -clampedAlpha * (uv.x() * duDc + uv.y() * dvDc);

                                const float oneMinusAlpha = 1.0f - clampedAlpha;
                                if (oneMinusAlpha <= 1e-6f) {
                                    // Fully opaque; transmission beyond this is negligible
                                    // We still record the term (optional), but tau will go ~0.
                                    continue;
                                }

                                // Store local term
                                localTerms[validCount].alpha = clampedAlpha;
                                localTerms[validCount].dAlphaDc = dAlphaDc;
                                ++validCount;

                                // Update the global transmission product
                                tau *= oneMinusAlpha;
                            }

                            if (validCount != 0) {
                                // Compute d(tau)/dPi using:
                                // d tau / dPi = sum_i [ - (tau / (1 - alpha_i)) * d alpha_i / dPi ]
                                float3 dTauDc(0.0f);

                                for (int i = 0; i < validCount; ++i) {
                                    const float alpha_i = localTerms[i].alpha;
                                    const float3 dAlphaDc_i = localTerms[i].dAlphaDc;
                                    const float oneMinusAlpha_i = 1.0f - alpha_i;

                                    const float weight = -tau / oneMinusAlpha_i;
                                    dTauDc += weight * dAlphaDc_i;
                                }

                                float3 pAdjoint = rayState.pathThroughput;
                                float p = luminanceGrayscale(pAdjoint);

                                // Accumulate into the running gradient (do not overwrite)
                                d_grad_pos += p * (L_bg) * dTauDc;
                            }
                        }
                    }

                    // Stop at a specific pixel

                    /*
                    if (isWatched) {
                        printf("Index:%u  Grad:(%f,%f,%f)  L:%f  tau:%f  Adj:%f  dTau:(%f,%f,%f)\n",
                               rayIndex,
                               d_grad_pos.x(), d_grad_pos.y(), d_grad_pos.z(),
                               L_bg, tau, pAdjoint,
                               dTauDc.x(), dTauDc.y(), dTauDc.z());
                    }
                    */


                    //shadowRayGrad = shadowRays(scene, whTransmit, rayState, photonMap, rng128) * 0.025;
                    //d_grad_pos += shadowRayGrad;
                    //if (rayState.bounceIndex >= recordBounceIndex) {
                    //    const float dVdp_scalar = dot(d_grad_pos, parameterAxis);
                    //    float4 &gradImageDst = sensor.framebuffer[rayState.pixelIndex];
                    //    atomicAddFloatToImage(&gradImageDst, dVdp_scalar);
                    //}

                    /*
                    WorldHit whScatter{};
                    intersectScene(rayState.ray, &whScatter, scene, rng128, RayIntersectMode::Scatter);

                    if (!whScatter.hit)
                        return;
                    */

                    const InstanceRecord &surfelInstance = scene.instances[whTransmit.instanceIndex];

                    if (surfelInstance.geometryType == GeometryType::PointCloud) {
                        struct LocalTerm {
                            float alpha{};
                            float3 dAlphaDc;
                        };
                        // 1) Build intervening splat list on (t_x, t_y), excluding terminal scatter surfel
                        // Do this however through a shadow ray
                        float3 tauGrad{0.0f};
                        float tauTotal = 1.0f;

                        BoundedVector<LocalTerm, kMaxSplatEvents> intervening;
                        intervening.clear();
                        for (int ei = 0; ei < whTransmit.splatEventCount - 1; ++ei) {
                            const auto &evt = whTransmit.splatEvents[ei];
                            const auto surfel = scene.points[evt.primitiveIndex];
                            const float3 dAlphaDc = gradTransmissionPosition(
                                rayState.ray, evt, surfel, (whTransmit.hitPositionW - rayState.ray.origin));
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
                        // 1) Build intervening splat list on (t_x, t_y), excluding terminal scatter surfel
                        // Do this however through a shadow ray
                        const Ray &ray = rayState.ray;

                        // 3) Terminal scatter at z: dα/dc with correct sign
                        const auto &terminal = whTransmit.splatEvents[whTransmit.splatEventCount - 1];
                        if (whTransmit.primitiveIndex == terminal.primitiveIndex) {
                            const auto surfel = scene.points[terminal.primitiveIndex];
                            const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));


                            // BSDF gradient (Scatter)
                            const float denom = dot(canonicalNormalW, ray.direction);
                            if (sycl::fabs(denom) <= 1e-4f)
                                return;
                            const float2 uv = phiInverse(terminal.hitWorld, surfel);
                            const float alpha = terminal.alpha;
                            const float tuDotD = dot(surfel.tanU, ray.direction);
                            const float tvDotD = dot(surfel.tanV, ray.direction);
                            const float su = surfel.scale.x();
                            const float sv = surfel.scale.y();
                            const float3 duDc = ((tuDotD / denom) * canonicalNormalW - surfel.tanU) / su;
                            const float3 dvDc = ((tvDotD / denom) * canonicalNormalW - surfel.tanV) / sv;
                            // Correct sign and sum
                            const float3 dAlphaDc = -alpha * (uv.x() * duDc + uv.y() * dvDc);

                            const float3 albedoRgb = surfel.color;
                            float colorLum = luminanceGrayscale(albedoRgb);

                            const float3 dFs_dC = (M_1_PIf) * (dAlphaDc * colorLum);
                            float3 f_s_rgb = (M_1_PIf) * (alpha * albedoRgb);

                            /*
                            // Cosine and its gradient
                            float cosineSigned = 0.0f;
                            float3 dCosineDc = {0, 0, 0};
                            cosineAndGradientWrtPosition(ray.origin, terminal.hitWorld, canonicalNormalW,
                                                         cosineSigned, dCosineDc);
                            const int travelSideSign = signNonZero(dot(canonicalNormalW, -ray.direction));
                            const float cosTheta = cosineSigned * float(travelSideSign);
                            */

                            const float3 surfelRadianceRGB = estimateSurfelRadianceFromPhotonMap(
                                terminal, ray.direction, scene, photonMap,
                                false, true, true);
                            float L_surfel = luminanceGrayscale(surfelRadianceRGB);
                            // 5) Final gradient assembly
                            float3 pAdjoint = rayState.pathThroughput;
                            float p = luminanceGrayscale(pAdjoint);
                            // Transmission gradient from direction change
                            // use in gradient assembly

                            float brdfLuminance = luminanceGrayscale(f_s_rgb);

                            float3 brdfGrad = dFs_dC * tauTotal;
                            float3 transmissionGrad = tauGrad * brdfLuminance;

                            float3 grad = p * L_surfel * (brdfGrad + transmissionGrad) * 1 / alpha;
                            d_grad_pos += grad;
                            /*
                            if (isWatched) {
                                printf(
                                    "Index:%u  Grad:(%f,%f,%f)  L:%f cos:%f  alpha:%f  Adj:%f  dFs:(%f,%f,%f)\n",
                                    rayIndex,
                                    d_grad_pos.x(), d_grad_pos.y(), d_grad_pos.z(),
                                    L_surfel, cosTheta, terminal.alpha, p,
                                    brdfGrad.x(), brdfGrad.y(), brdfGrad.z());
                                int debug = 0;
                            }
                            */
                        }
                    }

                    if (rayState.bounceIndex >= recordBounceIndex) {
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
                    bool reflectedRay = false;
                    if (instance.geometryType == GeometryType::PointCloud) {
                        // PointCloud
                        GPUMaterial material{};
                        material.baseColor = scene.points[worldHit.primitiveIndex].color;
                        const float3 lambertBrdf = material.baseColor * M_1_PIf;

                        const float interactionAlpha = worldHit.splatEvents[worldHit.splatEventCount - 1].alpha; // α
                        const float reflectWeight = 0.5f * interactionAlpha; // ρ_r = α/2
                        const float transmitWeight = 0.5f * interactionAlpha; // ρ_t = α/2

                        // event probabilities
                        const float probReflect = reflectWeight / interactionAlpha; // a 50/50 if we reflect or transmit
                        reflectedRay = (rng128.nextFloat() < probReflect);
                        if (reflectedRay) {
                            float sampledPdf = 0.0f;
                            // Diffuse reflect on entered side
                            sampleCosineHemisphere(rng128, enteredSideNormalW, sampledOutgoingDirectionW, sampledPdf);
                            sampledPdf = sycl::fmax(sampledPdf, 1e-6f);
                            const float cosTheta = sycl::fmax(0.0f, dot(sampledOutgoingDirectionW, enteredSideNormalW));
                            throughputMultiplier =
                                    throughputMultiplier * lambertBrdf * (cosTheta / sampledPdf);
                        } else {
                            float sampledPdf = 0.0f;
                            // Diffuse transmit: cosine hemisphere on the opposite side
                            const float3 oppositeSideNormalW = -enteredSideNormalW;
                            sampleCosineHemisphere(rng128, oppositeSideNormalW, sampledOutgoingDirectionW, sampledPdf);
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

                    uint32_t outputSlot = activeCounter.fetch_add(1);

                    // write only once to compacted slot
                    raysOut[outputSlot] = nextState;
                });
        });
        queue.wait();
    }
}
