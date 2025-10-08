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
        const uint32_t perPassRayCount = raysPerSet;

        //raysPerSet = 1;

        queue.memcpy(pkg.intermediates.countPrimary, &perPassRayCount, sizeof(uint32_t)).wait();

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

                    //primaryRay.direction = normalize(float3{0.0, 1.0, -0.18});
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

                    WorldHit worldHit{};
                    RayState rayState = raysIn[rayIndex];
                    Ray localRay;
                    intersectSceneAdjoint(rayState.ray, &worldHit, scene, &localRay, rng128);
                    if (!worldHit.hit) {
                        hitRecords[rayIndex] = worldHit;
                        return;
                    }

                    auto &instance = scene.instances[worldHit.instanceIndex];
                    switch (instance.geometryType) {
                        case GeometryType::Mesh: {
                            const Triangle &triangle = scene.triangles[worldHit.primitiveIndex];
                            const Transform &objectWorldTransform = scene.transforms[instance.transformIndex];

                            const Vertex &vertex0 = scene.vertices[triangle.v0];
                            const Vertex &vertex1 = scene.vertices[triangle.v1];
                            const Vertex &vertex2 = scene.vertices[triangle.v2];

                            // Geometric normal in world space
                            const float3 worldP0 = toWorldPoint(vertex0.pos, objectWorldTransform);
                            const float3 worldP1 = toWorldPoint(vertex1.pos, objectWorldTransform);
                            const float3 worldP2 = toWorldPoint(vertex2.pos, objectWorldTransform);
                            float3 geometricNormalW = normalize(cross(worldP1 - worldP0, worldP2 - worldP0));
                            worldHit.geometricNormalW = geometricNormalW;
                        }
                        break;
                        case GeometryType::PointCloud: {
                            auto &surfel = scene.points[worldHit.primitiveIndex];
                            // SURFACE: set geometric normal and record hit
                            float3 nW = normalize(cross(surfel.tanU, surfel.tanV));
                            if (dot(nW, -rayState.ray.direction) < 0.0f)
                                nW = -nW;
                            worldHit.geometricNormalW = nW;
                        }
                        break;
                    }
                    hitRecords[rayIndex] = worldHit;


                    // At this point I have a ray that intersects either a mesh or a surfel.
                    // I should attempt to propagate visibility gradients with the adjoint scalar field
                    auto &record = scene.instances[worldHit.instanceIndex];
                    if (record.geometryType == GeometryType::PointCloud) {
                        return;
                    }
                    // For each point-cloud BLAS range on this TLAS leaf (or iterate all if simple)
                    float3 dTdpkAccum = float3{0, 0, 0};
                    float transmittanceProduct = 1.0f;

                    for (uint32_t r = 0; r < scene.blasNodeCount; ++r) {
                        VisibilityGradResult rres =
                                accumulateVisibilityAndGradientPointCloud(localRay, worldHit.t, r,
                                                                          0, scene);
                        transmittanceProduct *= rres.transmittance;
                        dTdpkAccum = dTdpkAccum + rres.dTdPk; // already includes its local T
                    }

                    if (transmittanceProduct != 1.0f) {
                        int debug = 1;
                    }

                    // Weight for pure-visibility term in the adjoint inner product:
                    // w = L(y→x) * f_r * G for this segment.
                    // If you only need T’s derivative now, set w = 1 and plug the full weight later.

                    float3 q = rayState.pathThroughput;
                    float3 dcost_dpk_r = dTdpkAccum * q.x();
                    float3 dcost_dpk_g = dTdpkAccum * q.y();
                    float3 dcost_dpk_b = dTdpkAccum * q.z();

                    //float gradMag = (length(dcost_dpk_r) + length(dcost_dpk_g) + length(dcost_dpk_b)) / 3.0f;
                    const float w = (q.x() + q.y() + q.z()) / 3.0f; // q = ∂c/∂L per channel

                    const float3 L_bg = estimateRadianceFromPhotonMap(worldHit, scene, photonMap, photonsPerLaunch) * scene.materials[worldHit.instanceIndex].baseColor;

                    float3 dLseg_dpk = dTdpkAccum * L_bg;

                    const float3 parameterAxis = {1.0f, 0.0f, 0.0f};
                    float scalar = dot(dLseg_dpk, parameterAxis);

                    if (rayState.bounceIndex >= 0) {
                        float4 &gradImageDst = sensor.framebuffer[rayState.pixelIndex];
                        atomicAddFloatToImage(&gradImageDst, scalar);
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
