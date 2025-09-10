// SyclWarmup.cpp (no imports of your modules)
#include <sycl/sycl.hpp>
#include "Renderer/Kernels/SyclBridge.h"
#include "Renderer/Kernels/AdjointKernels.h"

#include "Renderer/GPUDataStructures.h"

#include "Renderer/Kernels/PathTracerKernels.h"
#include "Renderer/Kernels/KernelHelpers.h"

namespace Pale {
    // ---- Tags ---------------------------------------------------------------
    struct WarmupKernelTag {
    };

    struct RayGenEmitterKernelTag {
    };

    struct IntersectKernelTag {
    };

    struct ShadeKernelTag {
    };

    float traceVisibility(const Ray &rayIn, float tMax, const GPUSceneBuffers &scene, rng::Xorshift128 &rng128) {
        WorldHit worldHit{};
        if (!intersectScene(rayIn, &worldHit, scene, rng128)) return 1.0f;

        return worldHit.opacityAtHit; // opaque geometry
    }

    inline void launchDirectShadeKernel(sycl::queue &queue,
                                        GPUSceneBuffers scene,
                                        SensorGPU sensor,
                                        const RayState *raysIn,
                                        uint32_t rayCount,
                                        const PathTracerSettings &settings

    ) {
        queue.submit([&](sycl::handler &cgh) {
            uint64_t baseSeed = settings.randomSeed;
            cgh.parallel_for<ShadeKernelTag>(
                sycl::range<1>(rayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);

                    constexpr float kEps = 1e-4f;

                    // Choose any generator you like:
                    const RayState rayState = raysIn[rayIndex];
                    float3 throughput = rayState.pathThroughput;

                    // Construct Ray towards camera
                    auto &camera = sensor.camera;
                    float3 toPinhole = camera.pos - rayState.ray.origin;
                    float distanceToPinhole = length(toPinhole);
                    float3 directionToPinhole = toPinhole / distanceToPinhole;
                    // distance to camera:

                    Ray contribRay{
                        .origin = rayState.ray.origin + directionToPinhole * kEps,
                        .direction = directionToPinhole
                    };
                    // Shoot contribution ray towards camera
                    // If we have non-zero transmittance
                    float tMax = sycl::fmax(0.f, distanceToPinhole - kEps);
                    if (traceVisibility(contribRay, tMax, scene, rng128) > 0.0f) {
                        // perspective projection
                        float4 clip = camera.proj * (camera.view * float4(rayState.ray.origin, 1.f));

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

                                // Attenuation (Geometry term)
                                float surfaceCos = sycl::fabs(dot(float3{0, -1, 0}, directionToPinhole));
                                float cameraCos = sycl::fabs(dot(camera.forward, -directionToPinhole));
                                float G_cam = (surfaceCos * cameraCos) / (distanceToPinhole * distanceToPinhole);
                                float3 color = throughput * G_cam;

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


    // ---- Warmup -------------------------------------------------------------
    void warmupKernelSubmit(void *queuePtr, std::size_t totalWorkItems) {
        auto &queue = *static_cast<sycl::queue *>(queuePtr);
        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<WarmupKernelTag>(
                sycl::range<1>(totalWorkItems),
                [](sycl::id<1>) {
                });
        }).wait();
    }


    // ---- Helpers ------------------------------------------------------------
    inline void resetDeviceCounter(sycl::queue &queue, uint32_t *counterPtr) {
        queue.memset(counterPtr, 0, sizeof(uint32_t));
    }

    // ---- Kernels ------------------------------------------------------------
    void launchRayGenEmitterKernel(sycl::queue queue,
                                   PathTracerSettings settings,
                                   GPUSceneBuffers scene,
                                   RenderIntermediatesGPU renderIntermediates
    ) {
        const uint32_t photonCount = settings.photonsPerLaunch;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            uint64_t baseSeed = settings.randomSeed;

            commandGroupHandler.parallel_for<RayGenEmitterKernelTag>(
                sycl::range<1>(photonCount),
                [=](sycl::id<1> globalId) {
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, globalId);
                    // Choose any generator you like:
                    rng::Xorshift128 rng128(perItemSeed);

                    if (scene.lightCount == 0) return;

                    const float uL = rng128.nextFloat();
                    uint32_t lightIndex = sycl::min((uint32_t) (uL * scene.lightCount), scene.lightCount - 1);
                    const GPULightRecord light = scene.lights[lightIndex];
                    const float pdfSelectLight = 1.0f / (float) scene.lightCount;

                    // 2) pick a triangle uniformly from this light
                    if (light.triangleCount == 0) return;
                    const float uT = rng128.nextFloat();
                    const uint32_t triangleRelativeIndex = sycl::min((uint32_t) (uT * light.triangleCount),
                                                                     light.triangleCount - 1);
                    const GPUEmissiveTriangle emissiveTri =
                            scene.emissiveTriangles[light.triangleOffset + triangleRelativeIndex];

                    const Triangle tri = scene.triangles[emissiveTri.globalTriangleIndex];
                    const Vertex v0 = scene.vertices[tri.v0];
                    const Vertex v1 = scene.vertices[tri.v1];
                    const Vertex v2 = scene.vertices[tri.v2];

                    // triangle area and geometric normal in object space
                    const float3 p0 = v0.pos, p1 = v1.pos, p2 = v2.pos;
                    const float3 e0 = p1 - p0, e1 = p2 - p0;
                    const float3 nObjU = float3{
                        e0.y() * e1.z() - e0.z() * e1.y(),
                        e0.z() * e1.x() - e0.x() * e1.z(),
                        e0.x() * e1.y() - e0.y() * e1.x()
                    };
                    const float triArea = 0.5f * sycl::sqrt(
                                              nObjU.x() * nObjU.x() + nObjU.y() * nObjU.y() + nObjU.z() * nObjU.z());
                    if (triArea <= 0.f) return;

                    const float3 nObj = nObjU * (1.0f / (2.0f * triArea));

                    // 2b) sample uniform point on triangle (barycentric)
                    const float u1 = rng128.nextFloat();
                    const float u2 = rng128.nextFloat();
                    const float su1 = sycl::sqrt(u1);
                    const float b1 = 1.f - su1;
                    const float b2 = u2 * su1;
                    const float b0 = 1.f - b1 - b2;
                    const float3 xObj = p0 * b0 + p1 * b1 + p2 * b2;

                    // transform to world
                    const Transform xf = scene.transforms[light.transformIndex];
                    float3 sampledWorldPoint = toWorldPoint(xObj, xf);
                    float3 lightNormal{0.0f, -1.0f, 0.0f};

                    // 2c) cosine-hemisphere direction about n
                    float cosTheta = 0;
                    float3 sampledDirection;
                    sampleCosineHemisphere(rng128, lightNormal, sampledDirection, cosTheta);


                    // 3) PDFs
                    const float pdfTriangle = 1.0f / (float) light.triangleCount;
                    const float pdfPointGivenTriangle = 1.0f / triArea; // area domain
                    const float pdfArea = pdfTriangle * pdfPointGivenTriangle; // P_A(x)
                    const float pdfDir = cosTheta > 0.f ? (cosTheta / 3.1415926535f) : 0.f; // cosine hemisphere
                    const float pdfTotal = pdfSelectLight * pdfArea * pdfDir;
                    if (pdfTotal <= 0.f || cosTheta <= 0.f) return;

                    // 4) initial throughput
                    const sycl::float3 Le = light.emissionRgb; // radiance scale
                    const float invPdf = 1.0f / pdfTotal;
                    sycl::float3 initialThroughput = Le * (cosTheta * invPdf);

                    // write ray
                    RayState ray{};
                    ray.ray.origin = sampledWorldPoint;
                    ray.ray.direction = sampledDirection;
                    ray.pathThroughput = initialThroughput;
                    ray.bounceIndex = 0u;

                    auto counter = sycl::atomic_ref<uint32_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(
                        *renderIntermediates.countPrimary);
                    const uint32_t slot = counter.fetch_add(1);
                    renderIntermediates.primaryRays[slot] = ray;
                });
        });
        queue.wait();
    }


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

    void launchIntersectKernel(sycl::queue &queue,
                               GPUSceneBuffers scene,
                               const RayState *raysIn,
                               uint32_t rayCount,
                               WorldHit *hitRecords,
                               PathTracerSettings settings) {
        queue.submit([&](sycl::handler &cgh) {
            LaunchIntersectKernel kernel(scene, raysIn, hitRecords, settings.randomSeed);
            cgh.parallel_for<IntersectKernelTag>(
                sycl::range<1>(rayCount), kernel);
        });
        queue.wait(); // DEBUG: ensure the thread blocks here
    }


    inline void launchShadeKernel(sycl::queue &queue,
                                  GPUSceneBuffers scene,
                                  SensorGPU sensor,
                                  const WorldHit *hitRecords,
                                  const RayState *raysIn,
                                  uint32_t rayCount,
                                  RayState *raysOut,
                                  RenderIntermediatesGPU renderIntermediates,
                                  const PathTracerSettings &settings

    ) {
        queue.submit([&](sycl::handler &cgh) {
            uint64_t baseSeed = settings.randomSeed;
            cgh.parallel_for<ShadeKernelTag>(
                sycl::range<1>(rayCount),
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
                    if (traceVisibility(contribRay, tMax, scene, rng128) > 0.0f) {
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

                    // Sample next
                    RayState next{};
                    next.ray.origin = worldHit.hitPositionW + worldHit.geometricNormalW * kEps;
                    next.ray.direction = newDirection;
                    next.pathThroughput = throughput * brdf * (cosTheta / pdfCosine);
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


    // ---- Orchestrator -------------------------------------------------------
    void submitKernel(RenderPackage &pkg) {
        pkg.queue.fill(pkg.sensor.framebuffer, sycl::float4{0, 0, 0, 0}, pkg.sensor.height * pkg.sensor.width).wait();
        // Ray generation mode
        pkg.queue.fill(pkg.intermediates.countPrimary, 0u, 1).wait();

        switch (pkg.settings.rayGenMode) {
            case RayGenMode::Emitter:
                launchRayGenEmitterKernel(pkg.queue, pkg.settings, pkg.scene, pkg.intermediates);
                break;
            case RayGenMode::Adjoint:
                launchRayGenAdjointKernel(pkg.queue, pkg.settings, pkg.sensor, pkg.adjoint, pkg.scene, pkg.intermediates);
                break;
            default:
                ;
        }
        uint32_t activeCount = 0;
        pkg.queue.memcpy(&activeCount, pkg.intermediates.countPrimary, sizeof(uint32_t)).wait();

        if (pkg.settings.rayGenMode == RayGenMode::Emitter) {
            // Launch direct light kernel
            launchDirectShadeKernel(pkg.queue, pkg.scene, pkg.sensor,
                                    pkg.intermediates.primaryRays, activeCount, pkg.settings);
            for (uint32_t bounce = 0; bounce < pkg.settings.maxBounces && activeCount > 0; ++bounce) {
                pkg.queue.fill(pkg.intermediates.countExtensionOut, static_cast<uint32_t>(0), 1);
                pkg.queue.fill(pkg.intermediates.hitRecords, WorldHit(), activeCount);
                pkg.queue.wait();
                launchIntersectKernel(pkg.queue, pkg.scene, pkg.intermediates.primaryRays, activeCount,
                                      pkg.intermediates.hitRecords, pkg.settings);
                launchShadeKernel(pkg.queue, pkg.scene, pkg.sensor, pkg.intermediates.hitRecords,
                                  pkg.intermediates.primaryRays, activeCount,
                                  pkg.intermediates.extensionRaysA, pkg.intermediates, pkg.settings);
                uint32_t nextCount = 0;
                pkg.queue.memcpy(&nextCount, pkg.intermediates.countExtensionOut, sizeof(uint32_t)).wait();
                pkg.queue.memcpy(pkg.intermediates.primaryRays, pkg.intermediates.extensionRaysA,
                                 nextCount * sizeof(RayState));
                pkg.queue.wait();
                activeCount = nextCount;
                pkg.queue.wait();
            }
        } else if (pkg.settings.rayGenMode == RayGenMode::Adjoint) {

            //for (uint32_t bounce = 0; bounce < 1  && activeCount > 0; ++bounce) {
            for (uint32_t bounce = 0; bounce < pkg.settings.maxBounces  && activeCount > 0; ++bounce) {
                pkg.queue.fill(pkg.intermediates.countExtensionOut, static_cast<uint32_t>(0), 1);
                pkg.queue.fill(pkg.intermediates.hitRecords, WorldHit(), activeCount);
                pkg.queue.wait();
                launchIntersectKernel(pkg.queue, pkg.scene, pkg.intermediates.primaryRays, activeCount,
                                      pkg.intermediates.hitRecords, pkg.settings);

                launchAdjointShadeKernel(pkg.queue, pkg.scene, pkg.sensor, pkg.adjoint, pkg.intermediates.hitRecords,
                                  pkg.intermediates.primaryRays, activeCount,
                                  pkg.intermediates.extensionRaysA, pkg.intermediates, pkg.settings);

                uint32_t nextCount = 0;
                pkg.queue.memcpy(&nextCount, pkg.intermediates.countExtensionOut, sizeof(uint32_t)).wait();
                pkg.queue.memcpy(pkg.intermediates.primaryRays, pkg.intermediates.extensionRaysA,
                                 nextCount * sizeof(RayState));
                pkg.queue.wait();
                activeCount = nextCount;
                pkg.queue.wait();
            }
            // Launch intersect kernel

            // Launch shade kernel//
        }
    }
}
