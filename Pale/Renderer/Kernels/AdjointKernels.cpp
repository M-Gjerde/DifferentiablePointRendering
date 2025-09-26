//
// Created by magnus on 9/8/25.
//

#include "Renderer/Kernels/AdjointKernels.h"

#include "IntersectionKernels.h"
#include "Renderer/Kernels/KernelHelpers.h"


namespace Pale {
    void launchRayGenAdjointKernel(RenderPackage& pkg, int spp) {
        auto& queue = pkg.queue;
        auto& sensor = pkg.sensor;
        auto& settings = pkg.settings;
        auto& adjoint = pkg.adjoint;
        auto& intermediates = pkg.intermediates;

        const uint32_t imageWidth = sensor.camera.width;
        const uint32_t imageHeight = sensor.camera.height;
        const uint32_t raysPerSet = imageWidth * imageHeight;
        const uint32_t rayCount = raysPerSet * 2u;

        queue.memcpy(pkg.intermediates.countPrimary, &rayCount, sizeof(uint32_t)).wait();

        queue.submit([&](sycl::handler& commandGroupHandler) {
            const uint64_t baseSeed = settings.randomSeed * spp;

            commandGroupHandler.parallel_for<struct RayGenAdjointKernelTag>(
                sycl::range<1>(rayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t globalRayIndex = static_cast<uint32_t>(globalId[0]);

                    // Map to pixel within a single image
                    const uint32_t pixelLinearIndexWithinImage = globalRayIndex % raysPerSet;

                    const uint32_t pixelX = pixelLinearIndexWithinImage % imageWidth;
                    const uint32_t pixelY = pixelLinearIndexWithinImage / imageWidth;

                    // Decide set type
                    const bool isScatterSet = (globalRayIndex >= raysPerSet);

                    // RNG per ray
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, globalRayIndex);
                    rng::Xorshift128 rng128(perItemSeed);
                    const float jitterX = rng128.nextFloat() - 0.5f;
                    const float jitterY = rng128.nextFloat() - 0.5f;

                    // Primary ray
                    Ray cameraRay = makePrimaryRayFromPixelJittered(
                        sensor.camera,
                        static_cast<float>(pixelX),
                        static_cast<float>(pixelY),
                        jitterX, jitterY
                    );

                    //cameraRay.origin = float3(0, 1, 4);
                    //cameraRay.direction = normalize(float3(0.00, -0.12, -1.0));

                    // Adjoint source
                    const uint32_t pixelIndex = pixelLinearIndexWithinImage;
                    const float4 residualRgba = adjoint.framebuffer[pixelIndex];

                    float3 initialAdjointWeight = {
                        residualRgba.x(),
                        residualRgba.y(),
                        residualRgba.z()
                    };

                    // If you want unit weights instead, keep this line:
                    initialAdjointWeight = float3(1.0f);

                    RayState rayState{};
                    rayState.ray = cameraRay;
                    rayState.pathThroughput = initialAdjointWeight;
                    rayState.bounceIndex = 0;
                    rayState.pixelIndex = pixelIndex;
                    rayState.intersectMode = isScatterSet
                                                 ? RayIntersectMode::Scatter
                                                 : RayIntersectMode::Transmit;

                    /*
                    auto outputCounter = sycl::atomic_ref<uint32_t,
                                                          sycl::memory_order::relaxed,
                                                          sycl::memory_scope::device,
                                                          sycl::access::address_space::global_space>(
                        *intermediates.countPrimary);
                    */

                    const uint32_t outputSlot = pixelIndex + (isScatterSet ? raysPerSet : 0u);
                    intermediates.primaryRays[outputSlot] = rayState;
                });
        });
    }

    void launchAdjointKernel(RenderPackage& pkg, uint32_t activeRayCount) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& sensor = pkg.sensor;
        auto& settings = pkg.settings;
        auto& adjoint = pkg.adjoint;
        auto& photonMap = pkg.intermediates.map;

        auto* hitRecords = pkg.intermediates.hitRecords;
        auto* raysIn = pkg.intermediates.primaryRays;
        // activeRayCount == total rays in the current buffer (must be even)
        const uint32_t pairCount = activeRayCount / 2u;

        queue.submit([&](sycl::handler& cgh) {
            const uint64_t baseSeed = settings.randomSeed;
            cgh.parallel_for<struct AdjointShadeKernelTag>(
                sycl::range<1>(pairCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);
                    constexpr float kEps = 1e-4f;


                    const uint32_t scatterPairIndex = rayIndex + pairCount;

                    const RayState transmitRayState = raysIn[rayIndex];
                    const RayState scatterRayState = raysIn[scatterPairIndex];

                    const WorldHit transmitWorldHit = hitRecords[rayIndex];
                    const WorldHit scatterWorldHit = hitRecords[scatterPairIndex];

                    if (!transmitWorldHit.hit || !scatterWorldHit.hit) {
                        return;
                    }
                    GPUMaterial material;
                    float visibility = transmitWorldHit.transmissivity;

                    const Ray transmitRay = transmitRayState.ray;
                    auto surfel = scene.points[0];
                    const float3 segmentDirection = transmitWorldHit.hitPositionW - transmitRay.origin;
                    float3 tangentU = normalize(surfel.tanU);
                    float3 tangentV = normalize(surfel.tanV);
                    float3 surfelNormal = normalize(cross(tangentU, tangentV)); // no flip


                    float3 transportGradient(0.0f);
                    // Red type path
                    const float denominator = dot(surfelNormal, segmentDirection);
                    if (sycl::fabs(denominator) < kEps) return;

                    const float3 pk = surfel.position;
                    const float3 x = transmitRay.origin;
                    const float t = dot(surfelNormal, (pk - x)) / denominator;
                    if (t < 0.0f || t > 1.0f) return;

                    const float3 surfelIntersectionPoint = x + t * segmentDirection;
                    const float3 offsetR = surfelIntersectionPoint - pk;

                    // Local coordinates
                    const float su = sycl::fmax(surfel.scale.x(), 1e-8f);
                    const float sv = sycl::fmax(surfel.scale.y(), 1e-8f);
                    const float invSu2 = 1.0f / (su * su);
                    const float invSv2 = 1.0f / (sv * sv);

                    const float u = dot(tangentU, offsetR);
                    const float v = dot(tangentV, offsetR);

                    // Gaussian
                    const float quadraticForm = u * u * invSu2 + v * v * invSv2;
                    const float gaussianG = sycl::exp(-0.5f * quadraticForm);

                    // Jacobians
                    // dp_dpk = (d n^T)/(n^T d)
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
                    // Rows of B * dr_dpk, with B = [tU^T; tV^T]
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
                    // gLoc = [u/su^2, v/sv^2]
                    const float gLocU = u * invSu2;
                    const float gLocV = v * invSv2;
                    // ∂G/∂pk = -G * ( [gLocU,gLocV] * ∂ploc/∂pk )
                    // For V = 1 - G: ∂V/∂pk = +G * ( [gLocU,gLocV] * ∂ploc/∂pk )
                    float3 gradVisibilityWrtPk = float3{
                        gaussianG * (gLocU * dplocRowU[0] + gLocV * dplocRowV[0]),
                        gaussianG * (gLocU * dplocRowU[1] + gLocV * dplocRowV[1]),
                        gaussianG * (gLocU * dplocRowU[2] + gLocV * dplocRowV[2])
                    };
                    transportGradient = gradVisibilityWrtPk;


                    // Radiance L
                    auto bgInstance = scene.instances[transmitWorldHit.instanceIndex];
                    auto bgBRDF = scene.materials[bgInstance.materialIndex].baseColor;
                    float3 L_bg = estimateRadianceFromPhotonMap(transmitWorldHit, scene, photonMap,
                                                                settings.photonsPerLaunch) * bgBRDF;

                    auto fgBRDF = scene.points[scatterWorldHit.primitiveIndex].color;
                    float3 L_fg = estimateRadianceFromPhotonMap(scatterWorldHit, scene, photonMap,
                                                                settings.photonsPerLaunch) * fgBRDF;

                    // Material BRDF
                    //const float3 transportTimesBRDF = transportGradient * material.baseColor;
                    float3 deltaL = (L_bg - L_fg);
                    //float3 deltaL = (bgBRDF - fgBRDF);
                    float3 dL_dpk = transportGradient * deltaL;

                    const float3 gradTransportAndRadiance = dL_dpk;

                    // Adjoint Scalar q
                    float3 q = transmitRayState.pathThroughput;
                    float3 dcost_dpk = q * gradTransportAndRadiance;

                    float3& dst = adjoint.gradient_pk[0];
                    float3& gradCounter = adjoint.gradient_pk[1];
                    const sycl::atomic_ref<float,
                                           sycl::memory_order::relaxed,
                                           sycl::memory_scope::device,
                                           sycl::access::address_space::global_space>
                        xGrad(dst.x());
                    const sycl::atomic_ref<float,
                                           sycl::memory_order::relaxed,
                                           sycl::memory_scope::device,
                                           sycl::access::address_space::global_space>
                        yGrad(dst.y());
                    const sycl::atomic_ref<float,
                                           sycl::memory_order::relaxed,
                                           sycl::memory_scope::device,
                                           sycl::access::address_space::global_space>
                        zGrad(dst.z());

                    const sycl::atomic_ref<float,
                                           sycl::memory_order::relaxed,
                                           sycl::memory_scope::device,
                                           sycl::access::address_space::global_space>
                        gradCount(gradCounter.x());

                    gradCount.fetch_add(1.0f);

                    xGrad.fetch_add(dcost_dpk.x());
                    yGrad.fetch_add(dcost_dpk.y());
                    zGrad.fetch_add(dcost_dpk.z());


                    if (transmitRayState.bounceIndex >= 0) {
                        const float3 parameterAxis = {1.0f, 1.0f, 1.0f};
                        const float dVdp_scalar = dot(dcost_dpk, parameterAxis);

                        // write into the pixel that launched this adjoint path
                        float4& gradImageDst = sensor.framebuffer[transmitRayState.pixelIndex]; // make this buffer
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

    void generateNextAdjointRays(RenderPackage& pkg, uint32_t activeRayCount) {
        auto queue = pkg.queue;
        auto sensor = pkg.sensor;
        auto settings = pkg.settings;

        auto* hitRecords = pkg.intermediates.hitRecords;
        auto* raysIn = pkg.intermediates.primaryRays;
        auto* raysOut = pkg.intermediates.extensionRaysA;
        auto* countExtensionOut = pkg.intermediates.countExtensionOut;

        const uint32_t pairCount = activeRayCount / 2u;

        // You will output 2 rays per input pair
        const uint32_t nextActiveRayCount = 2u * pairCount;
        queue.memcpy(countExtensionOut, &nextActiveRayCount, sizeof(uint32_t)).wait();

        queue.submit([&](sycl::handler& cgh) {
            const uint64_t baseSeed = settings.randomSeed;
            cgh.parallel_for<class GenerateNextAdjointRays>(
                sycl::range<1>(pairCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    constexpr float kEps = 5e-4f;
                    auto rng = rng::Xorshift128(rng::makePerItemSeed1D(baseSeed, rayIndex));

                    const RayState transmitRayState = raysIn[rayIndex];
                    const WorldHit transmitWorldHit = hitRecords[rayIndex];
                    if (!transmitWorldHit.hit) {
                        RayState dummyRayState{};
                        dummyRayState.ray.origin = 0.0f;
                        dummyRayState.ray.direction = 0.0f;
                        dummyRayState.pixelIndex = 0;
                        const uint32_t outputBase = rayIndex;
                        raysOut[outputBase] = dummyRayState;
                        raysOut[outputBase + pairCount] = dummyRayState;
                    }
                    float3 newDirection;
                    float cosinePDF;
                    sampleCosineHemisphere(rng,
                                           transmitWorldHit.geometricNormalW,
                                           newDirection, cosinePDF);

                    RayState nextTransmitState{};
                    nextTransmitState.ray.origin = transmitWorldHit.hitPositionW +
                        transmitWorldHit.geometricNormalW * kEps;
                    nextTransmitState.ray.direction = newDirection;
                    nextTransmitState.ray.normal = transmitWorldHit.geometricNormalW;
                    nextTransmitState.bounceIndex = transmitRayState.bounceIndex + 1;
                    nextTransmitState.pixelIndex = transmitRayState.pixelIndex;
                    nextTransmitState.intersectMode = RayIntersectMode::Transmit;
                    nextTransmitState.pathThroughput = transmitRayState.pathThroughput;

                    RayState nextScatterState = nextTransmitState;
                    nextScatterState.intersectMode = RayIntersectMode::Scatter;

                    // pack: first half Transmit, second half Scatter
                    const uint32_t outputBase = rayIndex;
                    raysOut[outputBase] = nextTransmitState;
                    raysOut[outputBase + pairCount] = nextScatterState;
                });
        });
        queue.wait();
    }
}
