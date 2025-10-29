//
// Created by magnus on 9/8/25.
//

#include "Renderer/Kernels/AdjointKernels.h"

#include "IntersectionKernels.h"
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

                    WorldHit worldHit = hitRecords[rayIndex];
                    RayState rayState = raysIn[rayIndex];

                    if (!worldHit.hit)
                        return;
                    //printf("adjoint item %u, hit t: %f SplatEvents: %u \n", rayIndex, worldHit.t, worldHit.splatEventCount);

                    auto &instance = scene.instances[worldHit.instanceIndex];
                    float3 d_cost_d_pos(0.0f);
                    float3 gradientCostWrtSurfelCenter(0.0f);
                    Ray &ray = rayState.ray;
                    // SHADOW RAY


                    {
                        auto light = scene.lights[0];
                        auto lightTransform = scene.transforms[light.transformIndex];
                        float3 lightPosition = float3{
                            lightTransform.objectToWorld.row[0].w(), lightTransform.objectToWorld.row[1].w(),
                            lightTransform.objectToWorld.row[2].w()
                        };
                        const float3 toLight = lightPosition - worldHit.hitPositionW;
                        const float distanceToLight = length(toLight);
                        if (distanceToLight <= 1e-6f)
                            return;
                        const float3 lightDirection = toLight / distanceToLight;
                        float throughputMultiplier = 1.0f / (distanceToLight * distanceToLight);
                        Ray shadowRay{worldHit.hitPositionW, lightDirection};
                        RayState shadowRayState = rayState;
                        shadowRayState.ray = shadowRay;
                        shadowRayState.pathThroughput = rayState.pathThroughput * throughputMultiplier;

                        // BRDF
                        WorldHit shadowWorldHit{};
                        intersectScene(shadowRayState.ray, &shadowWorldHit, scene, rng128, RayIntersectMode::Transmit);
                        if (shadowWorldHit.hit) {
                            auto &instance = scene.instances[shadowWorldHit.instanceIndex];
                            Ray &ray = shadowRayState.ray;
                            if (shadowWorldHit.splatEventCount > 0 && instance.geometryType == GeometryType::Mesh) {
                                const float3 x = ray.origin;
                                const float3 y = shadowWorldHit.hitPositionW;
                                const float3 segmentVector = y - x;

                                const float3 backgroundRadianceRGB = estimateRadianceFromPhotonMap(
                                    shadowWorldHit, scene, photonMap);
                                const float luminanceMesh = luminanceGrayscale(backgroundRadianceRGB);

                                // Collect alpha_i and d(alpha_i)/dc for all valid splat intersections on the segment
                                struct LocalTerm {
                                    float alpha{};
                                    float3 dAlphaDc;
                                };
                                constexpr float epsilon = 1e-6f;
                                constexpr int maxSplatEvents = 16; // adjust if needed
                                LocalTerm localTerms[maxSplatEvents];
                                int validCount = 0;

                                for (int ei = 0; ei < shadowWorldHit.splatEventCount && validCount < maxSplatEvents; ++
                                     ei) {
                                    const auto splatEvent = shadowWorldHit.splatEvents[ei];
                                    const auto surfel = scene.points[splatEvent.primitiveIndex];

                                    const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));
                                    const int travelSideSign = (dot(canonicalNormalW, -ray.direction) >= 0.0f) ? 1 : -1;
                                    const float3 surfelNormal = (travelSideSign >= 0)
                                                                    ? canonicalNormalW
                                                                    : (-canonicalNormalW);

                                    const float denom = dot(surfelNormal, segmentVector);
                                    if (fabs(denom) <= epsilon) continue;

                                    const float tParam = dot(surfelNormal, (surfel.position - x)) / denom;
                                    if (tParam <= 0.0f || tParam >= 1.0f) continue;

                                    const float3 hitPoint = x + tParam * segmentVector;
                                    const float2 uv = phiInverse(hitPoint, surfel); // your mapping to local coords

                                    float alpha = splatEvent.alpha;

                                    const float tuDotD = dot(surfel.tanU, segmentVector);
                                    const float tvDotD = dot(surfel.tanV, segmentVector);

                                    const float su = surfel.scale.x();
                                    const float sv = surfel.scale.y();

                                    // d u / d c and d v / d c
                                    const float3 duDc = ((tuDotD / denom) * surfelNormal - surfel.tanU) / sycl::fmax(
                                                            su, epsilon);
                                    const float3 dvDc = ((tvDotD / denom) * surfelNormal - surfel.tanV) / sycl::fmax(
                                                            sv, epsilon);

                                    // helpers
                                    auto smoothAbs  = [](float x){ const float e=1e-12f; return sycl::sqrt(x*x + e); };
                                    auto smoothSign = [](float x){ const float e=1e-12f; return x / sycl::sqrt(x*x + e); };
                                    auto logit      = [](float x){ return sycl::log(x / (1.0f - x)); };

                                    const float uu = uv.x();
                                    const float vv = uv.y();

                                    // --- shape → p mapping so p(0)=2
                                    const float pMax = 32.0f;
                                    const float kShape = 1.0f;
                                    const float targetAtZero = 2.0f;
                                    const float qShape = (targetAtZero - 1.0f) / (pMax - 1.0f);
                                    const float bias = logit(qShape);
                                    const float sShape = 1.0f / (1.0f + sycl::exp(-(kShape * surfel.shape + bias)));
                                    const float p = 1.0f + (pMax - 1.0f) * sShape;

                                    // --- L^p radius r
                                    const float a = smoothAbs(uu);
                                    const float b = smoothAbs(vv);
                                    const float ap = sycl::pow(a, p);
                                    const float bp = sycl::pow(b, p);
                                    const float F  = ap + bp;
                                    const float eps = 1e-12f;
                                    const float r  = sycl::pow(sycl::fmax(F, eps), 1.0f / sycl::fmax(p, 1.0f + eps));
                                    if (r >= 1.0f) continue; // outside support

                                    // --- alpha = (1 - r)^beta
                                    const float g = sycl::fmax(1.0f - r, eps);
                                    const float beta = 4.0f * sycl::exp(surfel.beta);

                                    // --- ∂r/∂u, ∂r/∂v  (includes shape via p)
                                    const float rPow = sycl::pow(r, 1.0f - p);          // r^{1-p}
                                    const float aPow = (a > 0.0f) ? sycl::pow(a, p-1.0f) : 0.0f;
                                    const float bPow = (b > 0.0f) ? sycl::pow(b, p-1.0f) : 0.0f;
                                    const float du = smoothSign(uu);                    // d|u|/du
                                    const float dv = smoothSign(vv);                    // d|v|/dv
                                    const float drdu = aPow * rPow * du;
                                    const float drdv = bPow * rPow * dv;

                                    // --- dα/dc = -(α β / g) * ( ∂r/∂u * du/dc + ∂r/∂v * dv/dc )
                                    const float common = -alpha * beta / g;
                                    const float3 dAlphaDc = common * (drdu * duDc + drdv * dvDc);

                                    localTerms[validCount++] = LocalTerm{alpha, dAlphaDc};
                                }
                                if (validCount != 0) {
                                    // τ = Π (1-α_i) with stable log-space accumulation
                                    float logTau = 0.0f;
                                    for (int i = 0; i < validCount; ++i) {
                                        const float oneMinusAlpha = sycl::fmax(1.0f - localTerms[i].alpha, epsilon);
                                        logTau += sycl::log(oneMinusAlpha);
                                    }
                                    const float tauTotal = sycl::exp(logTau);
                                    // Σ_i [ - dα_i/dc / (1-α_i) ]
                                    float3 sumTerm(0.0f);
                                    for (int i = 0; i < validCount; ++i) {
                                        const float oneMinusAlpha = sycl::fmax(1.0f - localTerms[i].alpha, epsilon);
                                        sumTerm = sumTerm + (-localTerms[i].dAlphaDc) / oneMinusAlpha;
                                    }
                                    const float3 pAdjoint = shadowRayState.pathThroughput;
                                    // already includes fs, V, G, etc., for this segment
                                    gradientCostWrtSurfelCenter = pAdjoint * (tauTotal * luminanceMesh) * sumTerm;
                                    // Accumulate to your running gradient
                                    d_cost_d_pos = d_cost_d_pos + gradientCostWrtSurfelCenter;
                                }
                            }
                        }
                    }


                    if (worldHit.splatEventCount > 0 && instance.geometryType == GeometryType::Mesh && false) {
                        const float3 x = ray.origin;
                        const float3 y = worldHit.hitPositionW;
                        const float3 segmentVector = y - x;

                        const float r2Background = dot(segmentVector, segmentVector);
                        if (r2Background <= 0.0f) return;

                        const float3 backgroundRadianceRGB = estimateRadianceFromPhotonMap(worldHit, scene, photonMap);
                        const float luminanceMesh = luminanceGrayscale(backgroundRadianceRGB);

                        // Collect alpha_i and d(alpha_i)/dc for all valid splat intersections on the segment
                        struct LocalTerm {
                            float alpha;
                            float3 dAlphaDc;
                        };
                        constexpr float epsilon = 1e-6f;
                        constexpr int maxSplatEvents = 10; // adjust if needed
                        LocalTerm localTerms[maxSplatEvents];
                        int validCount = 0;

                        for (int ei = 0; ei < worldHit.splatEventCount && validCount < maxSplatEvents; ++ei) {
                            const auto splatEvent = worldHit.splatEvents[ei];
                            const auto surfel = scene.points[splatEvent.primitiveIndex];

                            const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));
                            const int travelSideSign = (dot(canonicalNormalW, -ray.direction) >= 0.0f) ? 1 : -1;
                            const float3 surfelNormal = (travelSideSign >= 0) ? canonicalNormalW : (-canonicalNormalW);

                            const float denom = dot(surfelNormal, segmentVector);
                            if (fabs(denom) <= epsilon) continue;

                            const float tParam = dot(surfelNormal, (surfel.position - x)) / denom;
                            if (tParam <= 0.0f || tParam >= 1.0f) continue;

                            const float3 hitPoint = x + tParam * segmentVector;
                            const float2 uv = phiInverse(hitPoint, surfel); // your mapping to local coords

                            // alpha_i: use event alpha, or recompute from Gaussian if preferred
                            float alpha = splatEvent.alpha;
                            // Optional exact Gaussian: alpha = sycl::clamp(exp(-0.5f * (uv.x()*uv.x() + uv.y()*uv.y())), 0.0f, 1.0f);

                            const float tuDotD = dot(surfel.tanU, segmentVector);
                            const float tvDotD = dot(surfel.tanV, segmentVector);

                            const float su = surfel.scale.x();
                            const float sv = surfel.scale.y();

                            // d u / d c and d v / d c
                            const float3 duDc = ((tuDotD / denom) * surfelNormal - surfel.tanU) / sycl::fmax(
                                                    su, epsilon);
                            const float3 dvDc = ((tvDotD / denom) * surfelNormal - surfel.tanV) / sycl::fmax(
                                                    sv, epsilon);

                            // d alpha / d c  for alpha = exp(-0.5*(u^2+v^2))  ==>  dα = -α*(u du + v dv)
                            const float3 dAlphaDc = -alpha * (uv.x() * duDc + uv.y() * dvDc);

                            localTerms[validCount++] = LocalTerm{alpha, dAlphaDc};
                        }
                        if (validCount == 0) return;
                        // τ = Π (1-α_i) with stable log-space accumulation
                        float logTau = 0.0f;
                        for (int i = 0; i < validCount; ++i) {
                            const float oneMinusAlpha = sycl::fmax(1.0f - localTerms[i].alpha, epsilon);
                            logTau += sycl::log(oneMinusAlpha);
                        }
                        const float tauTotal = sycl::exp(logTau);
                        // Σ_i [ - dα_i/dc / (1-α_i) ]
                        float3 sumTerm(0.0f);
                        for (int i = 0; i < validCount; ++i) {
                            const float oneMinusAlpha = sycl::fmax(1.0f - localTerms[i].alpha, epsilon);
                            sumTerm = sumTerm + (-localTerms[i].dAlphaDc) / oneMinusAlpha;
                        }
                        const float3 pAdjoint = rayState.pathThroughput;
                        // already includes fs, V, G, etc., for this segment
                        gradientCostWrtSurfelCenter = pAdjoint * (tauTotal * luminanceMesh) * sumTerm;
                        // Accumulate to your running gradient
                        d_cost_d_pos = d_cost_d_pos + gradientCostWrtSurfelCenter;
                    }


                    if (worldHit.splatEventCount > 0 && instance.geometryType == GeometryType::PointCloud && false) {
                        for (auto &splatEvent: worldHit.splatEvents) {
                            if (splatEvent.primitiveIndex == UINT32_MAX)
                                continue;
                            float3 x = ray.origin;
                            float3 y = splatEvent.hitWorld;
                            const auto surfel = scene.points[splatEvent.primitiveIndex];
                            const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));
                            const int travelSideSign =
                                    (dot(canonicalNormalW, -rayState.ray.direction) >= 0.0f) ? 1 : -1;
                            const float3 surfelNormal = (travelSideSign >= 0) ? canonicalNormalW : (-canonicalNormalW);
                            float denom = dot(surfelNormal, ray.direction);
                            if (abs(denom) <= 1e-6)
                                return;
                            float2 uv = phiInverse(y, surfel);
                            float alpha = splatEvent.alpha;
                            float tu_d = dot(surfel.tanU, ray.direction);
                            float tv_d = dot(surfel.tanV, ray.direction);
                            float su = surfel.scale.x();
                            float sv = surfel.scale.y();
                            float3 du_dc = ((tu_d / denom) * surfelNormal - surfel.tanU) / su;
                            float3 dv_dc = ((tv_d / denom) * surfelNormal - surfel.tanV) / sv;
                            float3 d_alpha_dc = -alpha * (uv[0] * du_dc + uv[1] * dv_dc);
                            float3 S_a_L = estimateSurfelRadianceFromPhotonMap(
                                worldHit.splatEvents[0], ray.direction, scene, photonMap, true);
                            float L_surfel = luminanceGrayscale(S_a_L);
                            float3 p_adjoint = rayState.pathThroughput;
                            d_cost_d_pos = p_adjoint * d_alpha_dc * L_surfel;
                        }
                    }


                    if (rayState.bounceIndex >= 0) {
                        const float3 parameterAxis = {1.0f, 0.0f, 0.00f};
                        const float dVdp_scalar = dot(d_cost_d_pos, parameterAxis);

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
