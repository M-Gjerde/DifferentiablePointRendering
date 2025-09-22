//
// Created by magnus on 9/8/25.
//

#include "Renderer/Kernels/AdjointKernels.h"

#include "IntersectionKernels.h"
#include "Renderer/Kernels/KernelHelpers.h"


namespace Pale {
    void launchRayGenAdjointKernel(RenderPackage& pkg) {
        auto& queue = pkg.queue;
        auto& sensor = pkg.sensor;
        auto& settings = pkg.settings;
        auto& adjoint = pkg.adjoint;

        auto& intermediates = pkg.intermediates;

        const uint32_t imageWidth = sensor.camera.width;
        const uint32_t imageHeight = sensor.camera.height;
        uint32_t rayCount = imageWidth * imageHeight;
        //rayCount = 4;
        queue.submit([&](sycl::handler& commandGroupHandler) {
            const uint64_t baseSeed = settings.randomSeed;
            const float invWidth = 1.f / static_cast<float>(imageWidth);
            const float invHeight = 1.f / static_cast<float>(imageHeight);

            commandGroupHandler.parallel_for<struct RayGenAdjointKernelTag>(
                sycl::range<1>(rayCount),
                [=](sycl::id<1> globalId) {
                    const auto pixelLinearIndex = static_cast<uint32_t>(globalId[0]);

                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);

                    uint32_t px = pixelLinearIndex % imageWidth;
                    uint32_t py = pixelLinearIndex / imageWidth;

                    // RNG per pixel
                    const CameraGPU adjointCamera = sensor.camera;
                    const float jx = rng128.nextFloat() - 0.5f;
                    const float jy = rng128.nextFloat() - 0.5f;

                    Ray cameraRay = makePrimaryRayFromPixelJittered(sensor.camera, static_cast<float>(px), static_cast<float>(py), jx, jy);

                    //px = sensor.width / 2;
                    //py = sensor.height / 3;
                    //cameraRay = makePrimaryRayFromPixel(sensor.camera, (px), (py));
                    // Initial adjoint throughput = residual (linear RGB). Alpha ignored.
                    //cameraRay.origin = float3(0, 1, 4);
                    //cameraRay.direction = normalize(float3(0.05, -0.18, -1.0));

                    const uint32_t pixelIndex = pixelLinearIndex;
                    const float4 residualRgba = adjoint.framebuffer[pixelIndex]; // linear RGB adjoint source

                    float3 initialAdjointWeight = {
                        residualRgba.x(),
                        residualRgba.y(),
                        residualRgba.z()
                    };

                    initialAdjointWeight = float3(1.0f);
                    //if (residualRgba.x() == 0.0f) return;
                    //if (residualRgba.x() == 0.f && residualRgba.y() == 0.f && residualRgba.z() == 0.f) return;

                    RayState rayState{};
                    rayState.ray = cameraRay;
                    rayState.pathThroughput = initialAdjointWeight;
                    rayState.bounceIndex = 0;
                    rayState.pixelIndex = pixelIndex;

                    auto outputCounter = sycl::atomic_ref<uint32_t,
                                                          sycl::memory_order::relaxed,
                                                          sycl::memory_scope::device,
                                                          sycl::access::address_space::global_space>(
                        *intermediates.countPrimary);


                    const uint32_t outputSlot = outputCounter.fetch_add(1u);
                    intermediates.primaryRays[outputSlot] = rayState;

                });
        });
        queue.wait();
    }


    void launchAdjointKernel(RenderPackage& pkg, uint32_t activeRayCount) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& sensor = pkg.sensor;
        auto& settings = pkg.settings;
        auto& adjoint = pkg.adjoint;

        auto* hitRecords = pkg.intermediates.hitRecords;
        auto* raysIn = pkg.intermediates.primaryRays;
        auto& photonMap = pkg.intermediates.map; // DeviceSurfacePhotonMapGrid

        uint32_t imageWidth = sensor.width;

        queue.submit([&](sycl::handler& cgh) {
            uint64_t baseSeed = settings.randomSeed;
            cgh.parallel_for<struct AjointShadeKernelTag>(
                sycl::range<1>(activeRayCount),
                // ReSharper disable once CppDFAUnusedValue
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);
                    constexpr float kEps = 1e-4f;
                    // Choose any generator you like:
                    WorldHit worldHit = hitRecords[rayIndex];
                    const RayState rayState = raysIn[rayIndex];
                    const Ray ray = rayState.ray;

                    if (!worldHit.hit) {
                        return;
                    }

                    auto& instance = scene.instances[worldHit.instanceIndex];
                    GPUMaterial material;
                    switch (instance.geometryType) {
                    case GeometryType::Mesh:
                        material = scene.materials[instance.materialIndex];
                        break;
                    case GeometryType::PointCloud:
                        material.baseColor = scene.points[worldHit.primitiveIndex].color;
                        break;
                    }

                    float visibility = worldHit.transmissivity;

                    auto surfel = scene.points[0];
                    const float3 segmentDirection = worldHit.hitPositionW - ray.origin;
                    float3 tangentU = normalize(surfel.tanU);
                    float3 tangentV = normalize(surfel.tanV);
                    float3 surfelNormal = normalize(cross(tangentU, tangentV)); // no flip


                    float3 transportGradient(0.0f);
                    // Red type path
                    if (instance.geometryType == GeometryType::Mesh) {
                        const float denominator = dot(surfelNormal, segmentDirection);
                        if (sycl::fabs(denominator) < kEps) return;

                        const float3 pk = surfel.position;
                        const float3 x = ray.origin;
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
                    }

                    if (instance.geometryType == GeometryType::PointCloud) {
                        float r = length(ray.origin - worldHit.hitPositionW);
                        float r2 = r * r;

                        if (dot(surfelNormal, -ray.direction) < 0)
                            surfelNormal = -surfelNormal;

                        float dot1 = dot(ray.normal, ray.direction);
                        float dot2 = dot(surfelNormal, -ray.direction);

                        float geometric = dot1 * dot2 / r2;

                        // Blue type path
                        const float3 d = worldHit.hitPositionW - ray.origin;
                        // Parallell check:
                        float nd = dot(surfelNormal, d);
                        if (abs(nd) < kEps) {
                            return;
                        }
                        // Hit position intersection parameter
                        float t = dot(surfelNormal, surfel.position - ray.origin) / nd;

                        float C = dot(ray.normal, ray.direction) * dot(surfelNormal, -ray.direction);
                        float dnorm2 = dot(d, d);
                        if (t == 0.0 || dnorm2 == 0.0f)
                            return;

                        float scale = -(2.0f * C) / (std::pow(t, 3) * dnorm2);
                        float3 gradGeometricWrtPk = (scale / nd) * surfelNormal;


                        //transportGradient = gradGeometricWrtPk;
                    }

                    // Material BRDF
                    const float3 transportTimesBRDF = transportGradient * material.baseColor;

                    // Radiance L
                    float3 radianceRGB = estimateRadianceFromPhotonMap(worldHit, scene, photonMap,
                                                                       settings.photonsPerLaunch);
                    const float3 gradTransportAndRadiance = transportTimesBRDF * radianceRGB;

                    // Adjoint Scalar q
                    float3 q = rayState.pathThroughput;
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


                    if (rayState.bounceIndex >= 0) {
                        const float3 parameterAxis = {0.0f, 1.0f, 0.00f};
                        const float dVdp_scalar = dot(dcost_dpk, parameterAxis);

                        // write into the pixel that launched this adjoint path
                        float4& gradImageDst = sensor.framebuffer[rayState.pixelIndex]; // make this buffer
                        const auto xGradImage = sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                           sycl::memory_scope::device,
                                                           sycl::access::address_space::global_space>(gradImageDst.x());
                        const auto yGradImage = sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                           sycl::memory_scope::device,
                                                           sycl::access::address_space::global_space>(gradImageDst.y());
                        const auto zGradImage = sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                           sycl::memory_scope::device,
                                                           sycl::access::address_space::global_space>(gradImageDst.z());


                        xGradImage.fetch_add(dVdp_scalar);
                        yGradImage.fetch_add(dVdp_scalar);
                        zGradImage.fetch_add(dVdp_scalar);
                    }
                });
        });
        queue.wait();
    }
}
