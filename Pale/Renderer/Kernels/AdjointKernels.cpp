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
            const float invWidth = 1.f / static_cast<float>(imageWidth);
            const float invHeight = 1.f / static_cast<float>(imageHeight);

            commandGroupHandler.parallel_for<struct RayGenAdjointKernelTag>(
                sycl::range<1>(rayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t pixelLinearIndex = static_cast<uint32_t>(globalId[0]);

                    const uint32_t pixelX = pixelLinearIndex % imageWidth;
                    const uint32_t pixelY = pixelLinearIndex / imageWidth;

                    // RNG per pixel
                    const CameraGPU adjointCamera = cameraSensor.camera;

                    // Subpixel sample in [0,1]^2
                    float sampleX = static_cast<float>(pixelX) * invWidth;
                    float sampleY = static_cast<float>(pixelY) * invHeight;

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
                        residualRgba.x(),
                        residualRgba.y(),
                        residualRgba.z()
                    };

                    const float weightLen2 =
                            initialAdjointWeight.x() * initialAdjointWeight.x() +
                            initialAdjointWeight.y() * initialAdjointWeight.y() +
                            initialAdjointWeight.z() * initialAdjointWeight.z();

                    //if (residualRgba.x() == 0.0f) return;
                    if (residualRgba.x() == 0.f && residualRgba.y() == 0.f && residualRgba.z() == 0.f) return;

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
                    renderIntermediates.adjoint[outputSlot].pixelID = pixelIndex;
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
                        }
                        break;
                    }

                    if (!worldHit.visitedSplatField) {
                        return;
                    }
                    auto &camera = cameraSensor.camera;

                    // Calculate intersection derivative:
                    // Visibility gradient
                    auto &surfel = scene.points[0];
                    float3 pointNormal = normalize(cross(surfel.tanU, surfel.tanV));

                    float3 segmentDirection = worldHit.hitPositionW - ray.origin;
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
                        gaussianG * (gLocU * dplocRowU[0] + gLocV * dplocRowV[0]),
                        gaussianG * (gLocU * dplocRowU[1] + gLocV * dplocRowV[1]),
                        gaussianG * (gLocU * dplocRowU[2] + gLocV * dplocRowV[2])
                    };

                    // Multiply by Light contribution
                    gradVisibilityWrtPk = gradVisibilityWrtPk * rayState.pathThroughput;

                    // Multiply by Geometric Term:
                    if (rayState.bounceIndex == 0) {
                        float3 toPinhole = camera.pos - worldHit.hitPositionW;
                        float distanceToPinhole = length(toPinhole);
                        float surfaceCos = sycl::fabs(dot(worldHit.geometricNormalW, -ray.direction));
                        float cameraCos = sycl::fabs(dot(camera.forward, ray.direction));
                        float G_cam = (surfaceCos * cameraCos) / (distanceToPinhole * distanceToPinhole);
                        gradVisibilityWrtPk = gradVisibilityWrtPk * G_cam;
                    }


                    float cosinePDF = 0.0f;
                    float3 newDirection;
                    sampleCosineHemisphere(rng128, worldHit.geometricNormalW, newDirection, cosinePDF);

                    float3 shadingNormal = worldHit.geometricNormalW;
                    float cosTheta = sycl::fmax(0.f, dot(newDirection, shadingNormal));
                    float pdfCosine = cosinePDF; // cosθ/π
                    float minPdf = 1e-6f;
                    pdfCosine = sycl::fmax(pdfCosine, minPdf);


                    // Lambertian BRDF
                    float3 albedo = material.baseColor; // in [0,1]
                    float3 brdf = albedo * (1.0f / M_PIf);

                    float3 &dst = adjointSensor.gradient_pk[0];
                    float3 &gradCounter = adjointSensor.gradient_pk[1];
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

                    if (rayState.bounceIndex > 0) {
                        xGrad.fetch_add(gradVisibilityWrtPk.x());
                        yGrad.fetch_add(gradVisibilityWrtPk.y());
                        zGrad.fetch_add(gradVisibilityWrtPk.z());

                        gradCount.fetch_add(1.0f);

                        const size_t pixelIndex =  renderIntermediates.adjoint[rayIndex].pixelID;
                        float4 &gradImage = cameraSensor.framebuffer[pixelIndex];
                        // atomic adds here


                        const sycl::atomic_ref<float,
                                    sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                gradImageX(gradImage.x());
                        const sycl::atomic_ref<float,
                                    sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                gradImageY(gradImage.y());
                        const sycl::atomic_ref<float,
                                    sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                gradImageZ(gradImage.z());


                        float3 local_dKernel_dpk = gradVisibilityWrtPk;
                        float3 w = rayState.pathThroughput * brdf;
                        // include adjoint BSDF ratio consistent with forward
                        float3 contrib = gradVisibilityWrtPk * 1000;

                        float grad = contrib.y();
                        gradImageX.fetch_add(grad);
                        gradImageY.fetch_add(grad);
                        gradImageZ.fetch_add(grad);
                    }

                    // Sample next
                    RayState next{};
                    next.ray.origin = worldHit.hitPositionW + worldHit.geometricNormalW * kEps;
                    next.ray.direction = newDirection;
                    next.pathThroughput = rayState.pathThroughput; //* brdf * (cosTheta / pdfCosine);
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
