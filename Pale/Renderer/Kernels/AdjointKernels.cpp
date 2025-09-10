//
// Created by magnus on 9/8/25.
//

#include "Renderer/Kernels/AdjointKernels.h"
#include "Renderer/Kernels/KernelHelpers.h"


namespace Pale {
    void launchRayGenAdjointKernel(sycl::queue queue,
                                   PathTracerSettings settings,
                                   SensorGPU cameraSensor,
                                   AdjointGPU adjointSensor,
                                   GPUSceneBuffers scene,
                                   RenderIntermediatesGPU renderIntermediates) {
        const uint32_t imageWidth = cameraSensor.camera.width;
        const uint32_t imageHeight = cameraSensor.camera.height;
        const uint32_t rayCount = imageWidth * imageHeight;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            const uint64_t baseSeed = settings.randomSeed;
            const float invWidth = 1.f / imageWidth;
            const float invHeight = 1.f / imageHeight;

            commandGroupHandler.parallel_for<struct RayGenAdjointKernelTag>(
                sycl::range<1>(rayCount),
                [=](sycl::id<1> globalId) {
                    const auto pixelLinearIndex = static_cast<uint32_t>(globalId[0]);

                    const uint32_t pixelX = pixelLinearIndex % imageWidth;
                    const uint32_t pixelY = pixelLinearIndex / imageWidth;

                    // RNG per pixel
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, globalId[0]);
                    rng::Xorshift128 rng128(perItemSeed);
                    const float jitterX = rng128.nextFloat();
                    const float jitterY = rng128.nextFloat();

                    const CameraGPU adjointCamera = cameraSensor.camera;

                    // Subpixel sample in [0,1]^2
                    float sampleX = (static_cast<float>(pixelX) + jitterX) * invWidth;
                    float sampleY = (static_cast<float>(pixelY) + jitterY) * invHeight;

                    // NDC → clip (OpenGL-style NDC in [-1,1])
                    const float ndcX = 2.f * sampleX - 1.f;
                    const float ndcY = 1.f - 2.f * sampleY;

                    // Ray in view space: unproject near and far, then get direction
                    const float3 pNearView = transformPoint(adjointCamera.invProj, {ndcX, ndcY, -1.f}, 1.f);
                    const float3 pFarView = transformPoint(adjointCamera.invProj, {ndcX, ndcY, 1.f}, 1.f);
                    const float3 dirView = normalize(pFarView - pNearView);

                    // To world
                    const float3 rayOriginWorld = transformPoint(adjointCamera.invView, {0.f, 0.f, 0.f}, 1.f);
                    // camera position
                    const float3 rayDirectionWorld = transformDirection(adjointCamera.invView, dirView);

                    // Initial adjoint throughput = residual (linear RGB). Alpha ignored.
                    const uint32_t pixelIndex = pixelLinearIndex;
                    const float4 residualRgba = adjointSensor.framebuffer[pixelIndex]; // linear RGB adjoint source

                    float3 initialAdjointWeight = {
                        residualRgba.x(), // R
                        residualRgba.y(), // G
                        residualRgba.z() // B
                    };

                    const float weightLen2 =
                            initialAdjointWeight.x() * initialAdjointWeight.x() +
                            initialAdjointWeight.y() * initialAdjointWeight.y() +
                            initialAdjointWeight.z() * initialAdjointWeight.z();

                    //if (residualRgba.x() == 0.0f) return;

                    RayState rayState{};
                    rayState.ray.origin = rayOriginWorld;
                    rayState.ray.direction = rayDirectionWorld;
                    rayState.pathThroughput = initialAdjointWeight;
                    rayState.bounceIndex = 0;

                    auto outputCounter = sycl::atomic_ref<uint32_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(*renderIntermediates.countPrimary);

                    const uint32_t outputSlot = outputCounter.fetch_add(1u);
                    renderIntermediates.primaryRays[outputSlot] = rayState;
                });
        });
        queue.wait();
    }

    void launchAdjointShadeKernel(sycl::queue &queue,
                                  GPUSceneBuffers scene,
                                  SensorGPU cameraSensor,
                                  AdjointGPU adjointSensor,
                                  const WorldHit *hitRecords,
                                  const RayState *raysIn,
                                  uint32_t rayCount,
                                  RayState *raysOut,
                                  RenderIntermediatesGPU renderIntermediates,
                                  const PathTracerSettings &settings

    ) {
        queue.submit([&](sycl::handler &cgh) {
            uint64_t baseSeed = settings.randomSeed;
            cgh.parallel_for<struct AjointShadeKernelTag>(
                sycl::range<1>(rayCount),
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
                    if (worldHit.t == FLT_MAX) {
                        return;
                    }
                    auto &instance = scene.instances[worldHit.instanceIndex];
                    GPUMaterial material;
                    switch (instance.geometryType) {
                        case GeometryType::Mesh:
                            material = scene.materials[instance.materialIndex];
                            break;
                        case GeometryType::PointCloud: {
                            auto val = scene.points[worldHit.primitiveIndex];
                            material.baseColor = val.color;
                            return;
                        }
                        break;
                    }

                    auto &transform = scene.transforms[instance.transformIndex];
                    // Calculate intersection derivative:
                    // Visibility gradient
                    auto &surfel = scene.points[0];
                    float3 pointNormal = cross(surfel.tanU, surfel.tanV);

                    float3 segmentDirection = worldHit.hitPositionW - ray.origin;
                    const float denom = dot(pointNormal, segmentDirection);
                    if (sycl::fabs(denom) < kEps) return;

                    // Intersection parameter and clamp to the segment if needed.
                    const float t = dot(pointNormal, (surfel.position - ray.origin)) / denom;
                    if (t < 0.f || t > 1.f) return;

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
                        surfel.tanU.x() * dr_dpk[0][0] + surfel.tanU.y() * dr_dpk[1][0] + surfel.tanU.z() * dr_dpk[2][
                            0],
                        surfel.tanU.x() * dr_dpk[0][1] + surfel.tanU.y() * dr_dpk[1][1] + surfel.tanU.z() * dr_dpk[2][
                            1],
                        surfel.tanU.x() * dr_dpk[0][2] + surfel.tanU.y() * dr_dpk[1][2] + surfel.tanU.z() * dr_dpk[2][2]
                    };
                    // Row for v:
                    float dplocRowV[3] = {
                        surfel.tanV.x() * dr_dpk[0][0] + surfel.tanV.y() * dr_dpk[1][0] + surfel.tanV.z() * dr_dpk[2][
                            0],
                        surfel.tanV.x() * dr_dpk[0][1] + surfel.tanV.y() * dr_dpk[1][1] + surfel.tanV.z() * dr_dpk[2][
                            1],
                        surfel.tanV.x() * dr_dpk[0][2] + surfel.tanV.y() * dr_dpk[1][2] + surfel.tanV.z() * dr_dpk[2][2]
                    };

                    // g_loc = SigmaInv * ploc = [u/su^2, v/sv^2]
                    float gLocU = u * invSu2;
                    float gLocV = v * invSv2;

                    // grad = G * (g_loc @ dploc_dpk)  -> 3-vector
                    float3 gradVisibilityWrtPk = {
                        -gaussianG * (gLocU * dplocRowU[0] + gLocV * dplocRowV[0]),
                        -gaussianG * (gLocU * dplocRowU[1] + gLocV * dplocRowV[1]),
                        -gaussianG * (gLocU * dplocRowU[2] + gLocV * dplocRowV[2])
                    };

                    float3 &dst = adjointSensor.gradient_pk[0];
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

                    xGrad.fetch_add(gradVisibilityWrtPk.x());
                    yGrad.fetch_add(gradVisibilityWrtPk.y());
                    zGrad.fetch_add(gradVisibilityWrtPk.z());
                    /*
                    auto &camera = cameraSensor.camera;
                    float4 clip = camera.proj * (camera.view * float4(worldHit.hitPositionW, 1.f));
                    if (clip.w() > 0.0f) {
                        float2 ndc = {clip.x() / clip.w(), clip.y() / clip.w()};
                        if (ndc.x() >= -1.f && ndc.x() <= 1.f && ndc.y() >= -1.f && ndc.y() <= 1.f) {
                            uint32_t px = sycl::clamp(
                                static_cast<uint32_t>((ndc.x() * 0.5f + 0.5f) * camera.width),
                                0u, camera.width - 1);
                            uint32_t py = sycl::clamp(
                                static_cast<uint32_t>((ndc.y() * 0.5f + 0.5f) * camera.height),
                                0u, camera.height - 1);

                            // FLIP Y
                            const uint32_t idx = (camera.height - 1u - py) * camera.width + px;

                            float4 &dst = cameraSensor.framebuffer[idx];
                            const sycl::atomic_ref<float,
                                        sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                    r(dst.x());
                            const sycl::atomic_ref<float,
                                        sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                    g(dst.y());
                            const sycl::atomic_ref<float,
                                        sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                    b(dst.z());
                            const sycl::atomic_ref<float,
                                        sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                    a(dst.w());
                            float3 color = material.baseColor;
                            color = float3(rayState.pathThroughput);
                            r.fetch_add(color.x());
                            g.fetch_add(color.y());
                            b.fetch_add(color.z());
                            a.store(1.0f);
                        }
                    }
                    */

                    float cosinePDF = 0.0f;
                    float3 newDirection;
                    sampleCosineHemisphere(rng128, worldHit.geometricNormalW, newDirection, cosinePDF);

                    float3 shadingNormal = worldHit.geometricNormalW;
                    float cosTheta = sycl::fmax(0.f, dot(newDirection, shadingNormal));
                    float pdfCosine = cosinePDF; // cosθ/π
                    float minPdf = 1e-6f;
                    pdfCosine = sycl::fmax(pdfCosine, minPdf);

                    // Sample next
                    RayState next{};
                    next.ray.origin = worldHit.hitPositionW + worldHit.geometricNormalW * kEps;
                    next.ray.direction = newDirection;
                    next.pathThroughput = rayState.pathThroughput * (cosTheta / pdfCosine);
                    next.bounceIndex = rayState.bounceIndex + 1;

                    auto counter = sycl::atomic_ref<uint32_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(
                        *renderIntermediates.countExtensionOut);
                    uint32_t slot = counter.fetch_add(1);

                    raysOut[slot] = next;
                });
        });
        queue.wait();
    }
}
