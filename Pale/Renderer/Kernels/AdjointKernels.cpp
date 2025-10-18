//
// Created by magnus on 9/8/25.
//

#include "Renderer/Kernels/AdjointKernels.h"

#include "IntersectionKernels.h"
#include "Renderer/Kernels/KernelHelpers.h"
#include "Renderer/Kernels/AdjointIntersectionKernels.h"


namespace Pale {
    void launchRayGenAdjointKernel(RenderPackage &pkg, int spp) {
        auto &queue = pkg.queue;
        auto &sensor = pkg.sensor;
        auto &settings = pkg.settings;
        auto &adjoint = pkg.adjoint;
        auto &intermediates = pkg.intermediates;

        const uint32_t imageWidth = sensor.camera.width;
        const uint32_t imageHeight = sensor.camera.height;

        uint32_t raysPerSet = imageWidth * imageHeight;

        //raysPerSet = 5;

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

                    //primaryRay.direction = normalize(float3{-0.01, 1.0, 0.04});
                    //primaryRay.origin = float3{0.0, -4.0, 1.0};
                    primaryRay.normal = normalize(float3{0.0, 1.0, 0.0});

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
        auto &adjoint = pkg.adjoint;
        auto &photonMap = pkg.intermediates.map;

        auto *hitRecords = pkg.intermediates.hitRecords;
        auto *raysIn = pkg.intermediates.primaryRays;
        // activeRayCount == total rays in the current buffer (must be even)

        const uint32_t perPassRayCount = activeRayCount; // total number of rays (With n-samples per ray)
        const uint32_t perPixelRayCount = activeRayCount; // Number of rays per pixel
        const uint32_t photonsPerLaunch = settings.photonsPerLaunch;

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<struct AdjointShadeKernelTag>(
                sycl::range<1>(perPixelRayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(settings.randomSeed, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);

                    WorldHit worldHit = hitRecords[rayIndex];
                    RayState rayState = raysIn[rayIndex];

                    //printf("adjoint item %u, hit t: %f SplatEvents: %u \n", rayIndex, worldHit.t, worldHit.splatEventCount);

                    // rayIndex 2 is mesh
                    // rayIndex 3 is surfel a
                    // rayIndex 4 is surfel b

                    // Mesh Scatters Ray index 2
                    // 1. Implement transmission gradients for all pixels terminating at meshes (1 path length)
                    // - Visualize with vertical movements of the splats.
                    // 2. Implement inscattering from surfels with emission geometric term. This will give me radiance gradients.
                    // - Visualize with different colors of the splats. Some darker some brighter than the mesh background.

                    // Surfel Scatters A, Ray index 3
                    // 1. Implement interior contributions, i.e. Geometric for surfels

                    // Surfel scatters B, ray index 4:
                    // 2. Add in-scattering to surfel events along side interior contribution. Use the attached points to propagate change to transmissivity.

                    // Mesh background with in-scattering:
                    auto &instance = scene.instances[worldHit.instanceIndex];

                    Ray &ray = rayState.ray;
                    float3 grad_L(0.0f);
                    if (worldHit.splatEventCount == 1 && instance.geometryType == GeometryType::Mesh) {
                        float3 d = worldHit.hitPositionW - ray.origin;
                        float r2 = dot(d, d);
                        if (r2 <= 0.0)
                            return;
                        float r = sqrtf(r2);
                        float3 w = d / r;

                        // Background Radiance and Geometric:
                        float cos_bg = dot(worldHit.geometricNormalW, -w);
                        float cos_fg = dot(ray.normal, w);
                        float G_bg = (cos_fg * cos_bg) / r2;

                        float3 L_mesh = estimateRadianceFromPhotonMap(
                            worldHit, scene, photonMap, settings.photonsPerLaunch);

                        // alpha surfel
                        auto surfel = scene.points[worldHit.splatEvents[0].primitiveIndex];
                        float3 n = cross(surfel.tanU, surfel.tanV);
                        float denom = dot(n, d);
                        if (abs(denom) <= 1e-6)
                            return;
                        float t_param = dot(n, surfel.position - ray.origin) / denom;

                        float3 p_hit = ray.origin + t_param * d;
                        float2 uv = phiInverse(p_hit, surfel.position, surfel.tanU, surfel.tanV, surfel.scale.x(),
                                               surfel.scale.y());
                        float u = uv[0];
                        float v = uv[1];
                        float alpha = expf(-0.5f * (u * u + v * v));

                        float3 grad_u = ((dot(surfel.tanU, d) / denom) * n - surfel.tanU) / surfel.scale.x();
                        float3 grad_v = ((dot(surfel.tanV, d) / denom) * n - surfel.tanV) / surfel.scale.y();
                        float3 grad_alpha = -alpha * (u * grad_u + v * grad_v);
                        float3 dT_dc = -grad_alpha;

                        // Volume geometric term
                        // Receiver-only geometry and its gradient wrt hit point z = p_hit
                        float3 z = p_hit;
                        float3 rvec = z - ray.origin;
                        float r2e = dot(rvec, rvec);
                        if (r2e <= 0.0f) return;
                        float re = sqrtf(r2e);

                        // w points into the camera (same as Python w_x)
                        float3 directionToHit = rvec / re;
                        float cosineReceiver = fmaxf(0.0f, dot(ray.normal, directionToHit));
                        float Gv = cosineReceiver / r2e;

                        float3 gradGv_z = float3(0.0f);
                        if (cosineReceiver > 0.0f) {
                            const float wDotN = dot(directionToHit, ray.normal);
                            // (I - w w^T) n
                            const float3 projectedNormal = ray.normal - directionToHit * wDotN;

                            const float inverseR3 = 1.0f / (re * re * re);

                            // d cos / dz = ((I - w w^T) n) / r
                            const float3 gradCosine = projectedNormal / re;

                            // d (1/r^2) / dz = -2 w / r^3
                            const float3 dInvR2_dz = -2.0f * directionToHit * inverseR3;

                            gradGv_z = gradCosine / r2e + cosineReceiver * dInvR2_dz;
                        }
                        // Map ∇_z Gv to surfel-position parameter p_i using dz/dp_i = d n^T / (n·d)
                        float dotD_grad = dot(d, gradGv_z);
                        float3 gradGv_p = (dotD_grad / denom) * n;

                        float3 S_a_L = estimateSurfelRadianceFromPhotonMap(worldHit.splatEvents[0], scene, photonMap,
                                                                           settings.photonsPerLaunch);
                        float L_surfel = luminance(S_a_L);
                        float L_bg = luminance(L_mesh);
                        //Use Rec.709 (sRGB) luminance.
                        float3 grad_L_bg = dT_dc * (L_bg * G_bg);
                        float3 grad_L_surf = grad_alpha * (L_surfel * Gv) + (alpha * L_surfel) * gradGv_p;
                        grad_L = grad_L_bg + grad_L_surf;
                    }

                    if (worldHit.splatEventCount == 1 && instance.geometryType == GeometryType::PointCloud) {
                        float3 &x = ray.origin;
                        float3 &y = worldHit.hitPositionW;
                        float3 rvec = y - x;
                        float r2 = dot(rvec, rvec);
                        if (r2 <= 0.0f) return;
                        float r = sqrtf(r2);

                        float3 u = (x- y) / r; // Into the camera
                        float3 surfel_normal = worldHit.geometricNormalW;
                        float a = dot(surfel_normal, u);
                        float b = dot(ray.normal, -u);

                        if (a <= 0.0f || b <= 0.0f) {
                            return;
                        }

                        float3 grad_G = (-b * surfel_normal + a * ray.normal + 4.0f * a *b * u) / (r * r * r);

                        float3 S_a_L = estimateSurfelRadianceFromPhotonMap(worldHit.splatEvents[0], scene, photonMap,
                                                   settings.photonsPerLaunch);
                        float L_surfel = luminance(S_a_L);

                        grad_L = L_surfel * grad_G;

                    }

                    if (!worldHit.hit) {
                        return;
                    }

                    if (rayState.bounceIndex >= 0) {
                        const float3 parameterAxis = {0.0f, 0.0f, 1.00f};
                        const float dVdp_scalar = dot(grad_L, parameterAxis);

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

        queue.memcpy(countExtensionOut, &activeRayCount, sizeof(uint32_t)).wait();

        queue.submit([&](sycl::handler &cgh) {
            const uint64_t baseSeed = settings.randomSeed;
            cgh.parallel_for<class GenerateNextAdjointRays>(
                sycl::range<1>(perPixelRayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t seed = rng::makePerItemSeed1D(baseSeed, rayIndex);
                    rng::Xorshift128 rng128(seed);

                    const RayState inState = raysIn[rayIndex];
                    const WorldHit hit = hitRecords[rayIndex];

                    RayState outState{};
                    if (!hit.hit) {
                        raysOut[rayIndex] = outState;
                        return;
                    } // dead ray

                    // Choose scattering model based on what we hit
                    float3 newDirection;
                    float pdf = 0.0f;
                    bool isSurfel = scene.instances[hit.instanceIndex].geometryType == GeometryType::PointCloud;
                    GPUMaterial material;
                    if (isSurfel) {
                        // volumetric or your chosen surfel phase sampling
                        sampleUniformSphere(rng128, newDirection, pdf); // example
                        material.baseColor = scene.points[hit.primitiveIndex].color;
                    } else {
                        // surface: cosine-weighted about the geometric normal
                        sampleCosineHemisphere(rng128, hit.geometricNormalW, newDirection, pdf);
                        // Material fetch: use the material bound to this hit
                        const uint32_t materialIndex = scene.instances[hit.instanceIndex].materialIndex;
                        material = scene.materials[materialIndex];
                    }


                    const float cosTheta = sycl::fmax(0.0f, dot(hit.geometricNormalW, newDirection));

                    float3 bsdfFactor;
                    if (isSurfel) {
                        // phase f_p and pdf_p; no cosine term
                        const float fp = 1.0f / (4.0f * M_PIf); // isotropic example
                        bsdfFactor = inState.pathThroughput * (fp / sycl::fmax(pdf, 1e-8f));
                    } else {
                        // Lambertian: simplify weight to albedo (since (albedo/π)*(cosθ/pdf) = albedo)
                        bsdfFactor = inState.pathThroughput * material.baseColor;
                        // If you keep explicit terms, use:
                        bsdfFactor = inState.pathThroughput * (material.baseColor / M_PIf) * (
                                         cosTheta / sycl::fmax(pdf, 1e-8f));
                    }

                    // Offset origin robustly
                    constexpr float kEps = 5e-4f;

                    outState.ray.origin = hit.hitPositionW + hit.geometricNormalW * kEps;
                    outState.ray.direction = newDirection;
                    outState.bounceIndex = inState.bounceIndex + 1;
                    outState.pixelIndex = inState.pixelIndex;
                    outState.pathThroughput = bsdfFactor;

                    raysOut[rayIndex] = outState;
                });
        });
        queue.wait();
    }
}
