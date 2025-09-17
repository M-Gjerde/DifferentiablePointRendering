//
// Created by magnus on 9/8/25.
//

#include "Renderer/Kernels/AdjointKernels.h"

#include "IntersectionKernels.h"
#include "Renderer/Kernels/KernelHelpers.h"


namespace Pale {
    void launchRayGenAdjointKernel(RenderPackage &pkg) {
        auto &queue = pkg.queue;
        auto &sensor = pkg.sensor;
        auto &settings = pkg.settings;
        auto &adjoint = pkg.adjoint;

        auto &intermediates = pkg.intermediates;

        const uint32_t imageWidth = sensor.camera.width;
        const uint32_t imageHeight = sensor.camera.height;
        const uint32_t rayCount = imageWidth * imageHeight;

        queue.submit([&](sycl::handler &commandGroupHandler) {
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

                    const uint32_t px = pixelLinearIndex % imageWidth;
                    const uint32_t py = sensor.height - 1 - pixelLinearIndex / imageWidth;

                    // RNG per pixel
                    const CameraGPU adjointCamera = sensor.camera;
                    const float jx = rng128.nextFloat() - 0.5f;
                    const float jy = rng128.nextFloat() - 0.5f;

                    Ray cameraRay = makePrimaryRayFromPixelJittered(sensor.camera, static_cast<float>(px),
                                                                    static_cast<float>(py), jx, jy);

                    // Initial adjoint throughput = residual (linear RGB). Alpha ignored.
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
                        sycl::access::address_space::global_space>(*intermediates.countPrimary);

                    const uint32_t outputSlot = outputCounter.fetch_add(1u);
                    intermediates.primaryRays[outputSlot] = rayState;
                });
        });
        queue.wait();
    }


    void launchAdjointKernel(RenderPackage &pkg, uint32_t activeRayCount) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &sensor = pkg.sensor;
        auto &settings = pkg.settings;
        auto &adjoint = pkg.adjoint;

        auto *hitRecords = pkg.intermediates.hitRecords;
        auto *raysIn = pkg.intermediates.primaryRays;
        auto &photonMap = pkg.intermediates.map; // DeviceSurfacePhotonMapGrid

        uint32_t imageWidth = sensor.width;

        queue.submit([&](sycl::handler &cgh) {
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

                    auto &instance = scene.instances[worldHit.instanceIndex];
                    GPUMaterial material;
                    switch (instance.geometryType) {
                        case GeometryType::Mesh:
                            material = scene.materials[instance.materialIndex]  ;
                            break;
                        case GeometryType::PointCloud:
                            material.baseColor = scene.points[worldHit.primitiveIndex].color;
                            break;
                    }

                    float visibility = worldHit.transmissivity;

                    // Blue type path
                    float3 gradGeometricWrtPk = float3{0.0f};
                    const float3 delta = ray.origin - worldHit.hitPositionW;
                    const float  distanceR = length(delta);
                    const float3 unitDirection = delta / distanceR;           // h = (y-x)/r

                    const float3 normalNx = worldHit.geometricNormalW;                       // N_x
                    const float3 normalNy = ray.normal;        // N_y

                    const float a = dot(normalNx, unitDirection); // a = n_x · u
                    const float b = dot(normalNy, unitDirection); // b = n_y · u

                    // Geometric term: G(x,y) = -(a*b)/r^2
                    const float invR2 = 1.0f / (distanceR * distanceR);
                    float geometric = -(a * b) * invR2;

                    // Red type path
                    float3 gradVisibilityWrtPk(0.0f);
                    if (instance.geometryType == GeometryType::Mesh) {
                        if (worldHit.transmissivity >= 0.9999)
                            return;
                        // Calculate intersection derivative:
                        // Visibility gradient
                        auto &surfel = scene.points[0];

                        float3 segmentDirection = worldHit.hitPositionW - ray.origin;

                        float3 pointNormal = normalize(cross(surfel.tanU, surfel.tanV));
                        if (dot(pointNormal, rayState.ray.direction) < 0.0f)
                            pointNormal = -pointNormal;

                        const float denom = dot(pointNormal, segmentDirection);
                        if (sycl::fabs(denom) < kEps) return;


                        // Intersection parameter and clamp to the segment if needed.
                        const float t = dot(pointNormal, (surfel.position - ray.origin)) / denom;
                        if (t < 0.f || t > 1.0) return;

                        const float3 surfelIntersectionPoint = ray.origin + t * segmentDirection;
                        const float3 offsetR = surfelIntersectionPoint - surfel.position;


                        // Local coords
                        float u = dot(surfel.tanU, offsetR);
                        float v = dot(surfel.tanV, offsetR);

                        // scale terms
                        float su = surfel.scale.x();
                        float sv = surfel.scale.y();
                        float invSu2 = 1.0f / (su * su);
                        float invSv2 = 1.0f / (sv * sv);

                        // Quadratic form and Gaussian
                        float quadraticForm = u * u * invSu2 + v * v * invSv2;
                        float gaussianG = sycl::exp(-0.5f * quadraticForm);

                        // Jacobians
                        // dp_dpk = (d n^T) / (n^T d)  (3x3)
                        float dp_dpk[3][3] = {
                            {
                                segmentDirection.x() * pointNormal.x() / denom,
                                segmentDirection.x() * pointNormal.y() / denom,
                                segmentDirection.x() * pointNormal.z() / denom
                            },
                            {
                                segmentDirection.y() * pointNormal.x() / denom,
                                segmentDirection.y() * pointNormal.y() / denom,
                                segmentDirection.y() * pointNormal.z() / denom
                            },
                            {
                                segmentDirection.z() * pointNormal.x() / denom,
                                segmentDirection.z() * pointNormal.y() / denom,
                                segmentDirection.z() * pointNormal.z() / denom
                            }
                        };

                        // dr_dpk = dp_dpk - I  (3x3)
                        float dr_dpk[3][3] = {
                            {dp_dpk[0][0] - 1.0f, dp_dpk[0][1], dp_dpk[0][2]},
                            {dp_dpk[1][0], dp_dpk[1][1] - 1.0f, dp_dpk[1][2]},
                            {dp_dpk[2][0], dp_dpk[2][1], dp_dpk[2][2] - 1.0f}
                        };

                        // dploc_dpk = B * dr_dpk, where B rows are (tangentU^T) and (tangentV^T)
                        // Row for u:
                        float dplocRowU[3] = {
                            surfel.tanU.x() * dr_dpk[0][0] + surfel.tanU.y() * dr_dpk[1][0] + surfel.tanU.z() * dr_dpk[
                                2][
                                0],
                            surfel.tanU.x() * dr_dpk[0][1] + surfel.tanU.y() * dr_dpk[1][1] + surfel.tanU.z() * dr_dpk[
                                2][
                                1],
                            surfel.tanU.x() * dr_dpk[0][2] + surfel.tanU.y() * dr_dpk[1][2] + surfel.tanU.z() * dr_dpk[
                                2][2]
                        };
                        // Row for v:
                        float dplocRowV[3] = {
                            surfel.tanV.x() * dr_dpk[0][0] + surfel.tanV.y() * dr_dpk[1][0] + surfel.tanV.z() * dr_dpk[
                                2][
                                0],
                            surfel.tanV.x() * dr_dpk[0][1] + surfel.tanV.y() * dr_dpk[1][1] + surfel.tanV.z() * dr_dpk[
                                2][
                                1],
                            surfel.tanV.x() * dr_dpk[0][2] + surfel.tanV.y() * dr_dpk[1][2] + surfel.tanV.z() * dr_dpk[
                                2][2]
                        };

                        // g_loc = SigmaInv * ploc = [u/su^2, v/sv^2]
                        float gLocU = u * invSu2;
                        float gLocV = v * invSv2;

                        // grad = G * (g_loc @ dploc_dpk)  -> 3-vector
                        gradVisibilityWrtPk = float3{
                            gaussianG * (gLocU * dplocRowU[0] + gLocV * dplocRowV[0]),
                            gaussianG * (gLocU * dplocRowU[1] + gLocV * dplocRowV[1]),
                            gaussianG * (gLocU * dplocRowU[2] + gLocV * dplocRowV[2])
                        };

                        gradVisibilityWrtPk = gradVisibilityWrtPk * geometric;
                    }

                    if (instance.geometryType == GeometryType::PointCloud) {
                        // (I - h h^T)[ a Ny + b Nx ]
                        const float3 directionalTerm = (-b) * normalNx + a * normalNy;
                        const float  proj = dot(unitDirection, directionalTerm);
                        const float3 projectedDirectionalTerm = directionalTerm - proj * unitDirection;

                        // ∇_y G = (1/r^3)[ (I - h h^T)(a Ny + b Nx) + 2ab h ]
                        const float invR3 = invR2 / distanceR;
                        gradGeometricWrtPk = invR3 * (projectedDirectionalTerm + (2.0f * a * b) * unitDirection);
                        gradGeometricWrtPk = gradGeometricWrtPk;
                    }

                    // Combine product-rule terms
                    const float3 gradientVisibilityTimesGeometric = gradVisibilityWrtPk; // (∂V/∂p_k) * G
                    const float3 gradientGeometricTimesVisibility = gradGeometricWrtPk; // (∂G/∂p_k) * V
                    const float3 transportGradient = gradVisibilityWrtPk;

                    // Material BRDF
                    const float3 transportTimesBRDF = transportGradient * material.baseColor;

                    float3 radianceRGB = estimateRadianceFromPhotonMap(worldHit, scene, photonMap,
                                                                       settings.photonsPerLaunch);
                    const float3 gradTransportAndRadiance = transportTimesBRDF * radianceRGB;

                    // Multiply by the adjoint scalar
                    float3 q = rayState.pathThroughput;
                    float3 dcost_dpk = q * gradTransportAndRadiance;

                    float3 &dst = adjoint.gradient_pk[0];
                    float3 &gradCounter = adjoint.gradient_pk[1];
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
                        const float3 parameterAxis = {0.0f, 0.1f, 0.0f};
                        const float dVdp_scalar = dot(dcost_dpk, parameterAxis);

                        // write into the pixel that launched this adjoint path
                        float4 &gradImageDst = sensor.framebuffer[rayState.pixelIndex]; // make this buffer
                        auto xGradImage = sycl::atomic_ref<float, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(gradImageDst.x()).fetch_add(dVdp_scalar);
                        auto yGradImage = sycl::atomic_ref<float, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(gradImageDst.y()).fetch_add(dVdp_scalar);
                        auto zGradImage = sycl::atomic_ref<float, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(gradImageDst.z()).fetch_add(dVdp_scalar);
                    }
                });
        });
        queue.wait();
    }
}
