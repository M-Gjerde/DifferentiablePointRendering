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
        auto &adjoint = pkg.adjoint;
        auto &intermediates = pkg.intermediates;

        const uint32_t imageWidth = sensor.camera.width;
        const uint32_t imageHeight = sensor.camera.height;
        const uint32_t samplesPerRay = settings.samplesPerRay;

        const uint32_t raysPerSet = imageWidth * imageHeight;
        const uint32_t totalRayCount = raysPerSet * samplesPerRay;

        queue.memcpy(pkg.intermediates.countPrimary, &totalRayCount, sizeof(uint32_t)).wait();

        queue.submit([&](sycl::handler &commandGroupHandler) {
            const uint64_t baseSeed = settings.randomSeed * static_cast<uint64_t>(spp);

            commandGroupHandler.parallel_for<struct RayGenAdjointKernelTag>(
                sycl::range<1>(raysPerSet),
                [=](sycl::id<1> globalId) {
                    const uint32_t globalRayIndex = static_cast<uint32_t>(globalId[0]);

                    // Map to pixel
                    const uint32_t pixelLinearIndexWithinImage = globalRayIndex; // 0..raysPerSet-1
                    const uint32_t pixelX = pixelLinearIndexWithinImage % imageWidth;
                    const uint32_t pixelY = pixelLinearIndexWithinImage / imageWidth;

                    // RNG for this pixel
                    const uint64_t perPixelSeed = rng::makePerItemSeed1D(baseSeed, pixelLinearIndexWithinImage);
                    rng::Xorshift128 pixelRng(perPixelSeed);

                    // Adjoint source weight
                    const uint32_t pixelIndex = pixelLinearIndexWithinImage;
                    const float4 residualRgba = adjoint.framebuffer[pixelIndex];
                    float3 initialAdjointWeight = {residualRgba.x(), residualRgba.y(), residualRgba.z()};
                    // Or unit weights:
                    initialAdjointWeight = float3(1.0f, 1.0f, 1.0f);

                    // Base slot for this pixel’s N samples
                    const uint32_t baseOutputSlot = pixelIndex * samplesPerRay;

                    // --- Sample 0: forced Transmit (background path) ---
                    {
                        const float jitterX = 0.0f; // pixelRng.nextFloat() - 0.5f;
                        const float jitterY = 0.0f; // pixelRng.nextFloat() - 0.5f;

                        Ray primaryRay = makePrimaryRayFromPixelJittered(
                            sensor.camera,
                            static_cast<float>(pixelX),
                            static_cast<float>(pixelY),
                            jitterX, jitterY
                        );

                        RayState rayState{};
                        rayState.ray = primaryRay;
                        rayState.pathThroughput = initialAdjointWeight;
                        rayState.bounceIndex = 0;
                        rayState.pixelIndex = pixelIndex;
                        rayState.intersectMode = RayIntersectMode::Transmit; // ensure background evaluation

                        intermediates.primaryRays[baseOutputSlot] = rayState;
                    }

                    // --- Samples 1..N-1: Scatter (stochastic acceptance) ---
                    for (uint32_t sampleIndex = 1; sampleIndex < samplesPerRay; ++sampleIndex) {
                        // Optional per-sample jitter for AA or stratification
                        const float jitterX = 0.0f; // pixelRng.nextFloat() - 0.5f;
                        const float jitterY = 0.0f; // pixelRng.nextFloat() - 0.5f;

                        Ray primaryRay = makePrimaryRayFromPixelJittered(
                            sensor.camera,
                            static_cast<float>(pixelX),
                            static_cast<float>(pixelY),
                            jitterX, jitterY
                        );

                        RayState rayState{};
                        rayState.ray = primaryRay;
                        rayState.pathThroughput = initialAdjointWeight;
                        rayState.bounceIndex = 0;
                        rayState.pixelIndex = pixelIndex;
                        rayState.intersectMode = RayIntersectMode::Scatter; // stochastic binary-opacity path

                        const uint32_t outputSlot = baseOutputSlot + sampleIndex;
                        intermediates.primaryRays[outputSlot] = rayState;
                    }
                });
        }).wait();
    }


    void launchAdjointKernel(RenderPackage &pkg, uint32_t activeRayCount) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &sensor = pkg.sensor;
        auto &settings = pkg.settings;
        auto &adjoint = pkg.adjoint;
        auto &photonMap = pkg.intermediates.map;

        auto *hitRecords = pkg.intermediates.hitRecords;
        auto *raysIn = pkg.intermediates.primaryRays;
        // activeRayCount == total rays in the current buffer (must be even)

        const uint32_t samplesPerRay = settings.samplesPerRay;
        const uint32_t totalRayCount = activeRayCount; // total number of rays (With n-samples per ray)
        const uint32_t perPixelRayCount = activeRayCount / samplesPerRay; // Number of rays per pixel
        const uint32_t photonsPerLaunch = settings.photonsPerLaunch;

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<struct AdjointShadeKernelTag>(
                sycl::range<1>(perPixelRayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t pixelLinearIndex = globalId[0];

                    constexpr float kEps = 1e-4f;

                    float3 accumulatedGradientPkRGB = float3{0.0f, 0.0f, 0.0f};
                    float3 accumulatedRadianceRGB = float3{0.0f, 0.0f, 0.0f};

                    const uint32_t baseSampleSlot = pixelLinearIndex * samplesPerRay;

                    const RayState transmitRay = raysIn[baseSampleSlot];
                    const WorldHit transmitHit = hitRecords[baseSampleSlot];

                    if (!transmitHit.hit) {
                        return;
                    }
                    const float3 L_bg = estimateRadianceFromPhotonMap(transmitHit, scene, photonMap,
                                                                      photonsPerLaunch);

                    const float3 sampleRadiance = L_bg * transmitRay.pathThroughput; // diagnostics
                    accumulatedRadianceRGB = sampleRadiance;

                    float3 dcost_dpk;
                    float gradientMagnitude = 0.0f;

                    for (uint32_t sampleIndex = 1; sampleIndex < samplesPerRay; ++sampleIndex) {
                        const uint32_t raySlot = baseSampleSlot + sampleIndex;

                        const RayState perSampleRayState = raysIn[raySlot];
                        const WorldHit perSampleHit = hitRecords[raySlot];

                        if (scene.instances[perSampleHit.instanceIndex].geometryType != GeometryType::PointCloud)
                            continue;
                        // ----------------- Scatter sample: pathwise visibility gradient at accepted hit -----------------
                        const auto surfel = scene.points[perSampleHit.primitiveIndex];

                        const Ray ray = perSampleRayState.ray;
                        const float3 segmentDirection = perSampleHit.hitPositionW - ray.origin;

                        float3 tangentU = normalize(surfel.tanU);
                        float3 tangentV = normalize(surfel.tanV);
                        float3 surfelNormal = normalize(cross(tangentU, tangentV));

                        const float denominator = dot(surfelNormal, segmentDirection);
                        if (sycl::fabs(denominator) < kEps) {
                            continue;
                        }

                        const float3 pk = surfel.position;
                        const float3 x = ray.origin;
                        const float t = dot(surfelNormal, (pk - x)) / denominator;
                        if (t < 0.0f || t > 1.0f) {
                            continue;
                        }

                        const float3 surfelIntersectionPoint = x + t * segmentDirection;
                        const float3 offsetR = surfelIntersectionPoint - pk;

                        const float su = sycl::fmax(surfel.scale.x(), 1e-8f);
                        const float sv = sycl::fmax(surfel.scale.y(), 1e-8f);
                        const float invSu2 = 1.0f / (su * su);
                        const float invSv2 = 1.0f / (sv * sv);

                        const float u = dot(tangentU, offsetR);
                        const float v = dot(tangentV, offsetR);

                        const float quadraticForm = u * u * invSu2 + v * v * invSv2;
                        const float gaussianG = sycl::exp(-0.5f * quadraticForm); // G

                        // dp_dpk
                        const float invDenominator = 1.0f / denominator;
                        float dp_dpk[3][3] = {
                            {
                                segmentDirection.x() * surfelNormal.x() * invDenominator,
                                segmentDirection.x() * surfelNormal.y() * invDenominator,
                                segmentDirection.x() * surfelNormal.z() * invDenominator
                            },
                            {
                                segmentDirection.y() * surfelNormal.x() * invDenominator,
                                segmentDirection.y() * surfelNormal.y() * invDenominator,
                                segmentDirection.y() * surfelNormal.z() * invDenominator
                            },
                            {
                                segmentDirection.z() * surfelNormal.x() * invDenominator,
                                segmentDirection.z() * surfelNormal.y() * invDenominator,
                                segmentDirection.z() * surfelNormal.z() * invDenominator
                            }
                        };
                        // dr_dpk = dp_dpk - I
                        float dr_dpk[3][3] = {
                            {dp_dpk[0][0] - 1.0f, dp_dpk[0][1], dp_dpk[0][2]},
                            {dp_dpk[1][0], dp_dpk[1][1] - 1.0f, dp_dpk[1][2]},
                            {dp_dpk[2][0], dp_dpk[2][1], dp_dpk[2][2] - 1.0f}
                        };
                        // Rows of B * dr_dpk
                        float dplocRowU[3] = {
                            tangentU.x() * dr_dpk[0][0] + tangentU.y() * dr_dpk[1][0] + tangentU.z() * dr_dpk[2][0],
                            tangentU.x() * dr_dpk[0][1] + tangentU.y() * dr_dpk[1][1] + tangentU.z() * dr_dpk[2][1],
                            tangentU.x() * dr_dpk[0][2] + tangentU.y() * dr_dpk[1][2] + tangentU.z() * dr_dpk[2][2]
                        };
                        float dplocRowV[3] = {
                            tangentV.x() * dr_dpk[0][0] + tangentV.y() * dr_dpk[1][0] + tangentV.z() * dr_dpk[2][0],
                            tangentV.x() * dr_dpk[0][1] + tangentV.y() * dr_dpk[1][1] + tangentV.z() * dr_dpk[2][1],
                            tangentV.x() * dr_dpk[0][2] + tangentV.y() * dr_dpk[1][2] + tangentV.z() * dr_dpk[2][2]
                        };

                        const float gLocU = u * invSu2;
                        const float gLocV = v * invSv2;

                        // ∂V/∂pk for V = 1 - G
                        const float3 gradientVisibilityWrtPk = float3{
                            gaussianG * (gLocU * dplocRowU[0] + gLocV * dplocRowV[0]),
                            gaussianG * (gLocU * dplocRowU[1] + gLocV * dplocRowV[1]),
                            gaussianG * (gLocU * dplocRowU[2] + gLocV * dplocRowV[2])
                        };

                        // Foreground vs background radiances at this configuration
                        const float3 fgAlbedo = surfel.color;
                        const float3 L_fg = estimateRadianceFromPhotonMap(perSampleHit, scene, photonMap,
                                                                          photonsPerLaunch) * fgAlbedo;

                        float3 deltaL = (L_bg - L_fg);

                        float3 transportGradient = gradientVisibilityWrtPk;

                        float3 dL_dpk_r = transportGradient * deltaL.x();
                        float3 dL_dpk_g = transportGradient * deltaL.y();
                        float3 dL_dpk_b = transportGradient * deltaL.z();
                        // Adjoint Scalar q
                        float3 q = perSampleRayState.pathThroughput;
                        float3 dcost_dpk_r = dL_dpk_r * q.x();
                        float3 dcost_dpk_g = dL_dpk_g * q.y();
                        float3 dcost_dpk_b = dL_dpk_b * q.z();

                        dcost_dpk = dcost_dpk + (transportGradient * deltaL * q);

                        gradientMagnitude += (length(dcost_dpk_r) + length(dcost_dpk_g) + length(dcost_dpk_b)) / 3.0f;
                    }


                    /*
                    // Optional radiance accumulation for diagnostics
                    accumulatedRadianceRGB = accumulatedRadianceRGB + (L_fg * pathThroughput);

                    // Average across this pixel’s samples
                    const float invSamples = 1.0f / static_cast<float>(samplesPerRay);
                    accumulatedGradientPkRGB = accumulatedGradientPkRGB * invSamples;
                    accumulatedRadianceRGB = accumulatedRadianceRGB * invSamples;
                    */

                    // Write back (example atomic adds)
                    float3 &gradientPkOut = adjoint.gradient_pk[0];
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                            atomGX(gradientPkOut.x()),
                            atomGY(gradientPkOut.y()),
                            atomGZ(gradientPkOut.z());
                    atomGX.fetch_add(dcost_dpk.x());
                    atomGY.fetch_add(dcost_dpk.y());
                    atomGZ.fetch_add(dcost_dpk.z());

                    if (transmitRay.bounceIndex >= 0) {
                        const float3 parameterAxis = {1.0f, 0.0f, 0.0f};
                        float dVdp_scalar = dot(dcost_dpk, parameterAxis);

                        // write into the pixel that launched this adjoint path
                        float4 &gradImageDst = sensor.framebuffer[transmitRay.pixelIndex]; // make this buffer
                        const auto xGradImage = sycl::atomic_ref<float, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                            gradImageDst.x());
                        const auto yGradImage = sycl::atomic_ref<float, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                            gradImageDst.y());
                        const auto zGradImage = sycl::atomic_ref<float, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                            gradImageDst.z());


                        xGradImage.fetch_add(dVdp_scalar);
                        yGradImage.fetch_add(dVdp_scalar);
                        zGradImage.fetch_add(dVdp_scalar);
                    }
                });
        });
        queue.wait();
    }

    void generateNextAdjointRays(RenderPackage &pkg, uint32_t activeRayCount) {
        auto queue = pkg.queue;
        auto sensor = pkg.sensor;
        auto settings = pkg.settings;

        auto *hitRecords = pkg.intermediates.hitRecords;
        auto *raysIn = pkg.intermediates.primaryRays;
        auto *raysOut = pkg.intermediates.extensionRaysA;
        auto *countExtensionOut = pkg.intermediates.countExtensionOut;

        const uint32_t samplesPerRay = settings.samplesPerRay;
        const uint32_t totalRayCount = activeRayCount; // total number of rays (With n-samples per ray)
        const uint32_t perPixelRayCount = activeRayCount / samplesPerRay; // Number of rays per pixel
        const uint32_t photonsPerLaunch = settings.photonsPerLaunch;

        queue.memcpy(countExtensionOut, &activeRayCount, sizeof(uint32_t)).wait();

        queue.submit([&](sycl::handler &cgh) {
            const uint64_t baseSeed = settings.randomSeed;
            cgh.parallel_for<class GenerateNextAdjointRays>(
                sycl::range<1>(perPixelRayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t pixelIndex = globalId[0];
                    constexpr float kEps = 5e-4f;

                    const uint32_t baseOutputSlot = pixelIndex * samplesPerRay;

                    auto rng = rng::Xorshift128(rng::makePerItemSeed1D(baseSeed, baseOutputSlot));

                    const RayState rayState = raysIn[baseOutputSlot];
                    const WorldHit hit = hitRecords[baseOutputSlot];
                    if (!hit.hit) {
                        RayState dummyRayState{};
                        dummyRayState.ray.origin = 0.0f;
                        dummyRayState.ray.direction = 0.0f;
                        dummyRayState.pixelIndex = 0;
                        raysOut[baseOutputSlot] = dummyRayState;
                        return;
                    }
                    float3 newDirection;
                    float cosinePDF;
                    sampleCosineHemisphere(rng,
                                           hit.geometricNormalW,
                                           newDirection, cosinePDF);

                    RayState nextTransmitState{};
                    nextTransmitState.ray.origin = hit.hitPositionW +
                                                   hit.geometricNormalW * kEps;
                    nextTransmitState.ray.direction = newDirection;
                    nextTransmitState.bounceIndex = rayState.bounceIndex + 1;
                    nextTransmitState.pixelIndex = rayState.pixelIndex;
                    nextTransmitState.pathThroughput = rayState.pathThroughput;
                    nextTransmitState.intersectMode = rayState.intersectMode;
                    raysOut[baseOutputSlot] = nextTransmitState;


                    // --- Samples 1..N-1: Scatter (stochastic acceptance) ---
                    for (uint32_t sampleIndex = 1; sampleIndex < samplesPerRay; ++sampleIndex) {
                        const uint32_t outputSlot = baseOutputSlot + sampleIndex;

                        RayState nextScatterState{};
                        nextScatterState.ray.origin = hit.hitPositionW +
                                                       hit.geometricNormalW * kEps;
                        nextScatterState.ray.direction = newDirection;
                        nextScatterState.bounceIndex = rayState.bounceIndex + 1;
                        nextScatterState.pixelIndex = rayState.pixelIndex;
                        nextScatterState.pathThroughput = rayState.pathThroughput;
                        nextScatterState.intersectMode = raysIn[outputSlot].intersectMode;

                        raysOut[outputSlot] = nextScatterState;
                    }
                });
        });
        queue.wait();
    }
}
