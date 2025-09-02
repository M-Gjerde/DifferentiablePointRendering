// SyclWarmup.cpp (no imports of your modules)
#include <sycl/sycl.hpp>
#include "Renderer/Kernels/SyclBridge.h"

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

    // ---- Warmup -------------------------------------------------------------
    void warmupKernelSubmit(void* queuePtr, std::size_t totalWorkItems) {
        auto& queue = *static_cast<sycl::queue*>(queuePtr);
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<WarmupKernelTag>(
                sycl::range<1>(totalWorkItems),
                [](sycl::id<1>) {
                });
        }).wait();
    }


    // ---- Helpers ------------------------------------------------------------
    inline void resetDeviceCounter(sycl::queue& queue, uint32_t* counterPtr) {
        queue.memset(counterPtr, 0, sizeof(uint32_t));
    }

    // ---- Kernels ------------------------------------------------------------
    void launchRayGenEmitterKernel(sycl::queue queue,
                                          PathTracerSettings settings,
                                          GPUSceneBuffers scene,
                                          SensorGPU sensor,
                                          RenderIntermediatesGPU renderIntermediates
                                          ) {
        const uint32_t photonCount = settings.photonsPerLaunch;

        queue.submit([&](sycl::handler& commandGroupHandler) {
            uint64_t baseSeed = settings.randomSeed;

            commandGroupHandler.parallel_for<RayGenEmitterKernelTag>(
                sycl::range<1>(photonCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t photonIndex = globalId[0];

                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, globalId);

                    // Choose any generator you like:
                   rng::Xorshift128 rng128(perItemSeed);

                    if (scene.lightCount == 0) return;

                    const float uL = rng128.nextFloat();
                    uint32_t lightIndex = sycl::min((uint32_t)(uL * scene.lightCount), scene.lightCount - 1);
                    const GPULightRecord light = scene.lights[lightIndex];
                    const float pdfSelectLight = 1.0f / (float)scene.lightCount;

                    // 2) pick a triangle uniformly from this light
                    if (light.triangleCount == 0) return;
                    const float uT = rng128.nextFloat();
                    const uint32_t triangleRelativeIndex = sycl::min((uint32_t)(uT * light.triangleCount), light.triangleCount - 1);
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
                        e0.y()*e1.z() - e0.z()*e1.y(),
                        e0.z()*e1.x() - e0.x()*e1.z(),
                        e0.x()*e1.y() - e0.y()*e1.x()
                    };
                    const float triArea = 0.5f * sycl::sqrt(nObjU.x()*nObjU.x() + nObjU.y()*nObjU.y() + nObjU.z()*nObjU.z());
                    if (triArea <= 0.f) return;
                    const float3 nObj = nObjU * (1.0f / (2.0f * triArea));

                    // 2b) sample uniform point on triangle (barycentric)
                    const float u1 = rng128.nextFloat();
                    const float u2 = rng128.nextFloat();
                    const float su1 = sycl::sqrt(u1);
                    const float b1 = 1.f - su1;
                    const float b2 = u2 * su1;
                    const float b0 = 1.f - b1 - b2;
                    const float3 xObj = p0*b0 + p1*b1 + p2*b2;

                    // transform to world
                    const Transform xf = scene.transforms[light.transformIndex];
                    const float4 x4 = xf.objectToWorld * float4{xObj.x(), xObj.y(), xObj.z(), 1.f};
                    float3 x = float3{x4.x(), x4.y(), x4.z()};
                    // normal with inverse-transpose; worldToObject^T on a direction
                    const float4 n4 = xf.objectToWorld * float4{nObj.x(), nObj.y(), nObj.z(), 0.f};
                    const float nL = sycl::sqrt(n4.x()*n4.x() + n4.y()*n4.y() + n4.z()*n4.z());
                    float3 n = (nL > 0.f) ? float3{n4.x()/nL, n4.y()/nL, n4.z()/nL} : float3{0,0,1};


                    // 2c) cosine-hemisphere direction about n
                    const float uD1 = rng128.nextFloat();
                    const float uD2 = rng128.nextFloat();
                    const float r   = sycl::sqrt(uD1);
                    const float phi = 6.28318530718f * uD2;
                    float3 lLocal{ r * sycl::cos(phi), r * sycl::sin(phi), sycl::sqrt(sycl::fmax(0.f, 1.f - uD1)) };
                    // build ONB(n)
                    float3 t, b;
                    if (sycl::fabs(n.z()) < 0.999f)
                        t = normalize(cross(float3{0,0,1}, n));
                    else
                        t = normalize(cross(float3{0,1,0}, n));

                    b = cross(n, t);

                    float3 vec = t*lLocal.x() + (b*lLocal.y()) + (n*lLocal.z());
                    float3 wo = normalize(vec);

                    // 3) PDFs
                    const float pdfTriangle = 1.0f / (float)light.triangleCount;
                    const float pdfPointGivenTriangle = 1.0f / triArea;          // area domain
                    const float pdfArea = pdfTriangle * pdfPointGivenTriangle;   // P_A(x)
                    const float cosTheta = sycl::fmax(0.f, wo.x()*n.x() + wo.y()*n.y() + wo.z()*n.z());
                    const float pdfDir = cosTheta > 0.f ? (cosTheta / 3.1415926535f) : 0.f; // cosine hemisphere
                    const float pdfTotal = pdfSelectLight * pdfArea * pdfDir;
                    if (pdfTotal <= 0.f || cosTheta <= 0.f) return;

                    // 4) initial throughput
                    const sycl::float3 Le = light.emissionRgb; // radiance scale
                    const float invPdf = 1.0f / pdfTotal;
                    sycl::float3 initialThroughput = Le * (cosTheta * invPdf);

                    // write ray
                    RayState ray{};
                    ray.ray.origin      = x;
                    ray.ray.direction   = wo;
                    ray.pathThroughput = initialThroughput;
                    ray.pixelIndex     = 0u;
                    ray.bounceIndex    = 0u;

                    auto counter = sycl::atomic_ref<uint32_t,
                                                    sycl::memory_order::relaxed,
                                                    sycl::memory_scope::device,
                                                    sycl::access::address_space::global_space>(
                        *renderIntermediates.countPrimary);
                    const uint32_t slot = counter.fetch_add(1);
                    renderIntermediates.primaryRays[slot] = ray;
                });
        });
    }

    SYCL_EXTERNAL [[clang::noinline]]
    void dbgHook_Intersect(uint32_t) {}

    struct LaunchIntersectKernel {

        LaunchIntersectKernel(GPUSceneBuffers scene, const RayState* ray, WorldHit* hit) : m_scene(scene), m_rays(ray), m_hitRecords(hit) {}
        void operator()(sycl::id<1> globalId) const {
            const uint32_t rayIndex = globalId[0];
            dbgHook_Intersect(rayIndex);            // set breakpoint here
            WorldHit hit{};
            RayState rayState = m_rays[rayIndex];
            intersectScene(rayState.ray, &hit, m_scene);
            m_hitRecords[rayIndex] = hit;
        }

    private:


        GPUSceneBuffers m_scene{};
        const RayState* m_rays{};
        WorldHit* m_hitRecords{};

    };

    void launchIntersectKernel(sycl::queue& queue,
                                      GPUSceneBuffers scene,
                                      const RayState* raysIn,
                                      uint32_t rayCount,
                                      WorldHit* hitRecords) {
        queue.submit([&](sycl::handler& cgh) {
            LaunchIntersectKernel kernel(scene, raysIn, hitRecords);
            cgh.parallel_for<IntersectKernelTag>(
                            sycl::range<1>(rayCount), kernel);
        });
        queue.wait_and_throw(); // DEBUG: ensure the thread blocks here

    }
    /*
    inline void launchShadeKernel(sycl::queue& queue,
                                  GPUSceneBuffers scene,
                                  SensorGPU sensor,
                                  const HitRecord* hitRecords,
                                  const RayState* raysIn,
                                  uint32_t rayCount,
                                  RayState* raysOut) {
        resetDeviceCounter(queue, sensor.countExtensionOut);

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<ShadeKernelTag>(
                sycl::range<1>(rayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t i = globalId[0];

                    const HitRecord hit = hitRecords[i];
                    const RayState  rs  = raysIn[i];

                    if (!hit.didHit) {
                        // TODO: accumulate environment
                        return;
                    }

                    // TODO: accumulate direct lighting

                    RayState next{};
                    next.rayOrigin      = hit.hitPosition;
                    next.rayDirection   = sycl::float3{0.f, 1.f, 0.f}; // placeholder
                    next.pathThroughput = rs.pathThroughput;
                    next.pixelIndex     = rs.pixelIndex;
                    next.bounceIndex    = rs.bounceIndex + 1;

                    auto counter = sycl::atomic_ref<uint32_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::ext_intel_global_device_space>(*sensor.countExtensionOut);
                    uint32_t slot = counter.fetch_add(1);
                    raysOut[slot] = next;
                });
        });
    }
    */

    // ---- Orchestrator -------------------------------------------------------
    void submitKernel(RenderPackage& pkg) {

        //pkg.queue.fill(pkg.sensor.framebuffer, sycl::float4{0, 0, 0, 0}, pkg.sensor.height * pkg.sensor.width).wait();
        // Ray generation mode
        switch (pkg.settings.rayGenMode) {
        case RayGenMode::Emitter:
            launchRayGenEmitterKernel(pkg.queue, pkg.settings, pkg.scene, pkg.sensor, pkg.intermediates);
            pkg.queue.wait();
            break;
        case RayGenMode::Adjoint:

            break;
        default:
            ;
        }


        uint32_t activeCount = 0;
        pkg.queue.memcpy(&activeCount, pkg.intermediates.countPrimary, sizeof(uint32_t)).wait();

        std::vector<RayState> rays(activeCount);

        pkg.queue.memcpy(rays.data(), pkg.intermediates.primaryRays, activeCount * sizeof(RayState)).wait();

        RayState* raysIn  = pkg.intermediates.primaryRays;
        RayState* raysOut = pkg.intermediates.extensionRaysA;


        for (uint32_t bounce = 0; bounce < pkg.settings.maxBounces && activeCount > 0; ++bounce) {
            launchIntersectKernel(pkg.queue, pkg.scene, raysIn, activeCount, pkg.intermediates.hitRecords);

            std::vector<WorldHit> worldHits(activeCount);
            pkg.queue.memcpy(worldHits.data(), pkg.intermediates.hitRecords, activeCount * sizeof(WorldHit)).wait();

            int debug = 1;

            /*
            launchIntersectKernel(queue, scene, raysIn, activeCount, sensor.hitRecords);
            launchShadeKernel(queue, scene, sensor, sensor.hitRecords, raysIn, activeCount, raysOut);

            uint32_t nextCount = 0;
            queue.memcpy(&nextCount, sensor.countExtensionOut, sizeof(uint32_t)).wait();

            std::swap(raysIn, raysOut);
            raysOut = (raysOut == sensor.extensionRaysA) ? sensor.extensionRaysB : sensor.extensionRaysA;
            activeCount = nextCount;
            */
        }

    }
}
