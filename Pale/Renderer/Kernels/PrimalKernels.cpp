//
// Created by magnus on 9/12/25.
//

#include "PrimalKernels.h"

#include "KernelHelpers.h"
#include "PathTracerKernels.h"

namespace Pale {
    struct LaunchIntersectKernel {
        LaunchIntersectKernel(GPUSceneBuffers scene, const RayState *ray, WorldHit *hit,
                              uint32_t seed) : m_scene(scene), m_rays(ray),
                                               m_hitRecords(hit), m_baseSeed(seed) {
        }

        void operator()(sycl::id<1> globalId) const {
            const uint32_t rayIndex = globalId[0];
            const uint64_t perItemSeed = rng::makePerItemSeed1D(m_baseSeed, rayIndex);
            rng::Xorshift128 rng128(perItemSeed);

            WorldHit worldHit{};
            RayState rayState = m_rays[rayIndex];
            intersectScene(rayState.ray, &worldHit, m_scene, rng128);
            if (worldHit.t == FLT_MAX) {
                m_hitRecords[rayIndex] = worldHit;
                return;
            }

            auto &instance = m_scene.instances[worldHit.instanceIndex];
            switch (instance.geometryType) {
                case GeometryType::Mesh: {
                    const Triangle &tri = m_scene.triangles[worldHit.primitiveIndex];
                    auto &transform = m_scene.transforms[instance.transformIndex];

                    // Build geometric normal in world space and normalize
                    float3 p0W = toWorldPoint(m_scene.vertices[tri.v0].pos, transform);
                    float3 p1W = toWorldPoint(m_scene.vertices[tri.v1].pos, transform);
                    float3 p2W = toWorldPoint(m_scene.vertices[tri.v2].pos, transform);
                    float3 geometricNormalW = normalize(cross(p1W - p0W, p2W - p0W));
                    worldHit.geometricNormalW = geometricNormalW;
                    std::string name = instance.name;
                    m_hitRecords[rayIndex] = worldHit;
                }
                break;
                case GeometryType::PointCloud: {
                    auto &surfel = m_scene.points[worldHit.primitiveIndex];
                    const float3 normalObject = normalize(cross(surfel.tanU, surfel.tanV));
                    worldHit.geometricNormalW = normalObject;
                    m_hitRecords[rayIndex] = worldHit;
                }
                break;
            }
        }

    private:
        GPUSceneBuffers m_scene{};
        const RayState *m_rays{};
        WorldHit *m_hitRecords{};
        uint32_t m_baseSeed = 0;
    };

    void launchIntersectKernel(RenderPackage &pkg, uint32_t activeRayCount) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto *hitRecords = pkg.intermediates.hitRecords;
        auto *raysIn = pkg.intermediates.primaryRays;

        queue.submit([&](sycl::handler &cgh) {
            LaunchIntersectKernel kernel(scene, raysIn, hitRecords, settings.randomSeed);
            cgh.parallel_for<struct IntersectKernelTag>(
                sycl::range<1>(activeRayCount), kernel);
        });
        queue.wait(); // DEBUG: ensure the thread blocks here
    }

    void launchVolumeKernel(RenderPackage &pkg, uint32_t activeRayCount) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto *hitRecords = pkg.intermediates.hitRecords;
        auto *raysIn = pkg.intermediates.primaryRays;

        queue.submit([&](sycl::handler &cgh) {
            uint64_t baseSeed = settings.randomSeed;
            cgh.parallel_for<class ShadeKernelTag>(
                sycl::range<1>(activeRayCount),
                // ReSharper disable once CppDFAUnusedValue
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);
                    constexpr float kEps = 1e-4f;
                    // Choose any generator you like:
                    WorldHit worldHit = hitRecords[rayIndex];
                    RayState &rayState = raysIn[rayIndex];
                    if (worldHit.t == FLT_MAX) {
                        return;
                    }
                    if (!worldHit.visitedSplatField)
                        return;
                    auto &instance = scene.instances[worldHit.instanceIndex];
                    auto geometryType = instance.geometryType;
                    // Use the segment transmittance for THIS segment, not a per-hit alpha.
                    float T = sycl::clamp(worldHit.transmissivity, 0.0f, 1.0f);
                    // For a visible tint, use albedo directly (drop /π). Scale if needed.
                    float3 splatAlbedo = scene.points[0].color;
                    // Deterministic mix: I*T + f*(1-T)
                    float3 tintFactor = T * float3{1.0f} + (1.0f - T) * (splatAlbedo);
                    rayState.pathThroughput = rayState.pathThroughput * tintFactor;
                });
        });
    }


    static WorldHit traceVisibility(const Ray &rayIn, float tMax, const GPUSceneBuffers &scene,
                                    rng::Xorshift128 &rng128) {
        WorldHit worldHit{};
        intersectScene(rayIn, &worldHit, scene, rng128);
        return worldHit; // opaque geometry
    }

    void launchContributionKernel(RenderPackage &pkg, uint32_t activeRayCount) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &sensor = pkg.sensor;
        auto &settings = pkg.settings;
        auto *hitRecords = pkg.intermediates.hitRecords;
        auto *raysIn = pkg.intermediates.primaryRays;
        auto *raysOut = pkg.intermediates.extensionRaysA;
        auto *countExtensionOut = pkg.intermediates.countExtensionOut;
        queue.submit([&](sycl::handler &cgh) {
            uint64_t baseSeed = settings.randomSeed;
            cgh.parallel_for<class ShadeKernelTag>(
                sycl::range<1>(activeRayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);
                    constexpr float kEps = 1e-4f;
                    // Choose any generator you like:
                    const WorldHit worldHit = hitRecords[rayIndex];
                    const RayState rayState = raysIn[rayIndex];
                    if (worldHit.t == FLT_MAX) {
                        return;
                    }
                    float3 throughput = rayState.pathThroughput;
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
                    // Construct Ray towards camera
                    auto &camera = sensor.camera;
                    float3 toPinhole = camera.pos - worldHit.hitPositionW;
                    float distanceToPinhole = length(toPinhole);
                    float3 directionToPinhole = toPinhole / distanceToPinhole;
                    // distance to camera:
                    Ray contribRay{
                        .origin = worldHit.hitPositionW + directionToPinhole * kEps,
                        .direction = directionToPinhole
                    };
                    // Shoot contribution ray towards camera
                    // If we have non-zero transmittance
                    float tMax = sycl::fmax(0.f, distanceToPinhole - kEps);
                    auto transmittance = traceVisibility(contribRay, tMax, scene, rng128);
                    if (transmittance.transmissivity > 0.0f) {
                        // perspective projection
                        float4 clip = camera.proj * (camera.view * float4(worldHit.hitPositionW, 1.f));
                        if (clip.w() > 0.0f) {
                            float2 ndc = {clip.x() / clip.w(), clip.y() / clip.w()};
                            if (ndc.x() >= -1.f && ndc.x() <= 1.f && ndc.y() >= -1.f && ndc.y() <= 1.f) {
                                /* 2)  raster coords (clamp to avoid the right/top fenceposts) */
                                uint32_t px = sycl::clamp(
                                    static_cast<uint32_t>((ndc.x() * 0.5f + 0.5f) * camera.width),
                                    0u, camera.width - 1);
                                uint32_t py = sycl::clamp(
                                    static_cast<uint32_t>((ndc.y() * 0.5f + 0.5f) * camera.height),
                                    0u, camera.height - 1);
                                // FLIP Y
                                const uint32_t idx = (camera.height - 1u - py) * camera.width + px;
                                float4 &dst = sensor.framebuffer[idx];
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

                                // BRDF to camera direction
                                float3 brdf = material.baseColor / M_PIf;
                                // Attenuation (Geometry term)
                                float surfaceCos = sycl::fabs(dot(worldHit.geometricNormalW, directionToPinhole));
                                float cameraCos = sycl::fabs(dot(camera.forward, -directionToPinhole));
                                float G_cam = (surfaceCos * cameraCos) / (distanceToPinhole * distanceToPinhole);

                                float3 color = throughput * brdf * G_cam;

                                r.fetch_add(color.x());
                                g.fetch_add(color.y());
                                b.fetch_add(color.z());
                                a.store(1.0f);
                            };
                        }
                    }
                });
        });
        queue.wait();
    }

    void generateNextRays(RenderPackage &pkg, uint32_t activeRayCount) {
        auto queue = pkg.queue;
        auto scene = pkg.scene;
        auto sensor = pkg.sensor;
        auto settings = pkg.settings;

        auto *hitRecords = pkg.intermediates.hitRecords;
        auto *raysIn = pkg.intermediates.primaryRays;
        auto *raysOut = pkg.intermediates.extensionRaysA;
        auto *countExtensionOut = pkg.intermediates.countExtensionOut;


        queue.submit([&](sycl::handler &cgh) {
            uint64_t baseSeed = settings.randomSeed;
            cgh.parallel_for<class ShadeKernelTag>(
                sycl::range<1>(activeRayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);
                    constexpr float kEps = 1e-4f;

                    const WorldHit worldHit = hitRecords[rayIndex];
                    const RayState rayState = raysIn[rayIndex];
                    if (worldHit.t == FLT_MAX) {
                        return;
                    }

                    float3 throughput = rayState.pathThroughput;

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
                    float cosinePDF = 0.0f;
                    float3 newDirection;
                    // Lambertian BRDF
                    float3 brdf;

                    if (instance.geometryType == GeometryType::PointCloud) {
                        sampleUniformSphere(rng128, newDirection, cosinePDF);
                    } else {
                        sampleCosineHemisphere(rng128, worldHit.geometricNormalW, newDirection, cosinePDF);
                    }

                    brdf = material.baseColor * (1.0f / M_PIf);

                    float3 shadingNormal = worldHit.geometricNormalW;
                    float cosTheta = sycl::fmax(0.f, dot(newDirection, shadingNormal));
                    float minPdf = 1e-6f;
                    cosinePDF = sycl::fmax(cosinePDF, minPdf);

                    // Sample next
                    RayState next{};
                    next.ray.origin = worldHit.hitPositionW + worldHit.geometricNormalW * kEps;
                    next.ray.direction = newDirection;
                    next.pathThroughput = throughput * brdf;
                    next.bounceIndex = rayState.bounceIndex + 1;

                    auto counter = sycl::atomic_ref<uint32_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(
                        *countExtensionOut);
                    uint32_t slot = counter.fetch_add(1);

                    raysOut[slot] = next;
                });
        });
        queue.wait();
    }
} // Pale
