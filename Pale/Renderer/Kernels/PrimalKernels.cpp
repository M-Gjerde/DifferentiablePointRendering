//
// Created by magnus on 9/12/25.
//

#include "PrimalKernels.h"

#include <cmath>

#include "KernelHelpers.h"
#include "IntersectionKernels.h"

namespace Pale {
    void launchRayGenEmitterKernel(RenderPackage &pkg) {
        auto queue = pkg.queue;
        auto scene = pkg.scene;
        auto sensor = pkg.sensor;
        auto settings = pkg.settings;

        auto *hitRecords = pkg.intermediates.hitRecords;
        auto *raysIn = pkg.intermediates.primaryRays;
        auto *raysOut = pkg.intermediates.extensionRaysA;
        auto *countPrimary = pkg.intermediates.countPrimary;


        const uint32_t photonCount = settings.photonsPerLaunch;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            uint64_t baseSeed = settings.randomSeed;

            commandGroupHandler.parallel_for<struct RayGenEmitterKernelTag>(
                sycl::range<1>(photonCount),
                [=](sycl::id<1> globalId) {
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, globalId[0]);
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
                    sycl::float3 initialThroughput = Le * (cosTheta * invPdf) / photonCount;

                    // write ray
                    RayState ray{};
                    ray.ray.origin = sampledWorldPoint;
                    ray.ray.direction = sampledDirection;
                    ray.ray.normal = lightNormal;
                    ray.pathThroughput = initialThroughput;
                    ray.bounceIndex = 0u;

                    auto counter = sycl::atomic_ref<uint32_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(
                        *countPrimary);
                    const uint32_t slot = counter.fetch_add(1);
                    raysIn[slot] = ray;
                });
        });
        queue.wait();
    }

    struct LaunchIntersectKernel {
        LaunchIntersectKernel(GPUSceneBuffers scene, RenderIntermediatesGPU intermediates,
                              PathTracerSettings settings) : m_scene(scene),
                                                             m_intermediates(intermediates), m_settings(settings) {
        }

        void operator()(sycl::id<1> globalId) const {
            const uint32_t rayIndex = globalId[0];
            const uint64_t perItemSeed = rng::makePerItemSeed1D(m_settings.randomSeed, rayIndex);
            rng::Xorshift128 rng128(perItemSeed);

            WorldHit worldHit{};
            RayState rayState = m_intermediates.primaryRays[rayIndex];
            intersectScene(rayState.ray, &worldHit, m_scene, rng128);
            if (!worldHit.hit) {
                m_intermediates.hitRecords[rayIndex] = worldHit;
                return;
            }

            auto &instance = m_scene.instances[worldHit.instanceIndex];
            switch (instance.geometryType) {
                case GeometryType::Mesh: {
                    const Triangle &triangle = m_scene.triangles[worldHit.primitiveIndex];
                    const Transform &objectWorldTransform = m_scene.transforms[instance.transformIndex];

                    const Vertex &vertex0 = m_scene.vertices[triangle.v0];
                    const Vertex &vertex1 = m_scene.vertices[triangle.v1];
                    const Vertex &vertex2 = m_scene.vertices[triangle.v2];

                    // Geometric normal in world space
                    const float3 worldP0 = toWorldPoint(vertex0.pos, objectWorldTransform);
                    const float3 worldP1 = toWorldPoint(vertex1.pos, objectWorldTransform);
                    const float3 worldP2 = toWorldPoint(vertex2.pos, objectWorldTransform);
                    float3 geometricNormalW = normalize(cross(worldP1 - worldP0, worldP2 - worldP0));
                    worldHit.geometricNormalW = geometricNormalW;

                    //std::string name = instance.name;
                    m_intermediates.hitRecords[rayIndex] = worldHit;

                    // APpend to photon map
                    if (m_settings.rayGenMode == RayGenMode::Emitter) {
                        const sycl::atomic_ref<uint32_t,
                                    sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                counter(*m_intermediates.map.photonCountDevicePtr);
                        const uint32_t slot = counter.fetch_add(1u);
                        if (slot >= m_intermediates.map.photonCapacity) return; // drop safely

                        DevicePhotonSurface entry;
                        entry.position = worldHit.hitPositionW;
                        entry.power = rayState.pathThroughput;
                        entry.incidentDir = -rayState.ray.direction;
                        entry.cosineIncident = std::fmax(0.f, dot(worldHit.geometricNormalW, entry.incidentDir));
                        m_intermediates.map.photons[slot] = entry;
                    }
                }
                break;
                case GeometryType::PointCloud: {
                    auto &surfel = m_scene.points[worldHit.primitiveIndex];
                    // SURFACE: set geometric normal and record hit
                    float3 nW = normalize(cross(surfel.tanU, surfel.tanV));
                    if (dot(nW, -rayState.ray.direction) < 0.0f)
                        nW = -nW;
                    worldHit.geometricNormalW = nW;



                    m_intermediates.hitRecords[rayIndex] = worldHit;

                    if (m_settings.rayGenMode == RayGenMode::Emitter) {
                        // APpend to photon map
                        const sycl::atomic_ref<uint32_t,
                                    sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                counter(*m_intermediates.map.photonCountDevicePtr);
                        const uint32_t slot = counter.fetch_add(1u);
                        if (slot >= m_intermediates.map.photonCapacity) return; // drop safely

                        if (slot < m_intermediates.map.photonCapacity) {
                            DevicePhotonSurface entry;
                            entry.position = worldHit.hitPositionW;
                            entry.power = rayState.pathThroughput;
                            entry.incidentDir = -rayState.ray.direction;
                            entry.cosineIncident = sycl::fmax(0.f, dot(nW, entry.incidentDir));
                            m_intermediates.map.photons[slot] = entry;
                        }
                    }
                }
                break;
            }
        }

    private:
        GPUSceneBuffers m_scene{};
        RenderIntermediatesGPU m_intermediates{};
        PathTracerSettings m_settings{};
    };

    void launchIntersectKernel(RenderPackage &pkg, uint32_t activeRayCount) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto &intermediates = pkg.intermediates;

        queue.submit([&](sycl::handler &cgh) {
            LaunchIntersectKernel kernel(scene, intermediates, settings);
            cgh.parallel_for<struct IntersectKernelTag>(
                sycl::range<1>(activeRayCount), kernel);
        });
        queue.wait(); // DEBUG: ensure the thread blocks here
    }


    void clearGridHeads(sycl::queue &q, DeviceSurfacePhotonMapGrid &g) {
        q.fill(g.cellHeadIndexArray, kInvalidIndex, g.totalCellCount).wait();
    }

    void buildPhotonGridLinkedLists(sycl::queue &q, DeviceSurfacePhotonMapGrid g, uint32_t photonCount) {
        q.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(photonCount), [=](sycl::id<1> id) {
                uint32_t i = id[0];
                DevicePhotonSurface ph = g.photons[i];
                sycl::int3 c = worldToCell(ph.position, g);
                uint32_t cell = linearCellIndex(c, g.gridResolution);

                auto headRef = sycl::atomic_ref<uint32_t,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(g.cellHeadIndexArray[cell]);

                uint32_t oldHead = headRef.exchange(i);
                g.photonNextIndexArray[i] = oldHead;
            });
        }).wait();
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
                    constexpr float kEps = 5e-4f;

                    const WorldHit worldHit = hitRecords[rayIndex];
                    const RayState rayState = raysIn[rayIndex];
                    if (!worldHit.hit) {
                        return;
                    }


                    auto &instance = scene.instances[worldHit.instanceIndex];
                    GPUMaterial material;
                    switch (instance.geometryType) {
                        case GeometryType::Mesh:
                            material = scene.materials[instance.materialIndex];
                            break;
                        case GeometryType::PointCloud:
                            material.baseColor = scene.points[worldHit.primitiveIndex].color;
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
                    RayState nextState{};
                    nextState.ray.origin = worldHit.hitPositionW + worldHit.geometricNormalW * kEps;
                    nextState.ray.direction = newDirection;
                    nextState.ray.normal = worldHit.geometricNormalW;
                    nextState.bounceIndex = rayState.bounceIndex + 1;
                    nextState.pixelIndex = rayState.pixelIndex;


                    // Throughput update
                    float3 updatedThroughput = rayState.pathThroughput * (
                                                   brdf * cosTheta / sycl::fmax(cosinePDF, 1e-6f));

                    /*
                    // Russian roulette
                    if (nextState.bounceIndex >= settings.russianRouletteStart) {
                        // Use luminance to avoid color bias; clamp to keep variance bounded
                        const float survivalProbabilityRaw =
                                0.2126f * updatedThroughput.x() +
                                0.7152f * updatedThroughput.y() +
                                0.0722f * updatedThroughput.z();
                        const float survivalProbability =
                                sycl::clamp(survivalProbabilityRaw, 0.05f, 0.95f);

                        if (rng128.nextFloat() > survivalProbability) {
                            return; // kill path; do not enqueue
                        }
                        updatedThroughput = updatedThroughput * (1.0f / survivalProbability);
                    }
                    */
                    nextState.pathThroughput = updatedThroughput;

                    // Enqueue survived ray
                    auto extensionCounter = sycl::atomic_ref<uint32_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(*countExtensionOut);
                    const uint32_t outIndex = extensionCounter.fetch_add(1);
                    raysOut[outIndex] = nextState;
                });
        });
        queue.wait();
    }


    void launchDirectContributionKernel(RenderPackage &pkg, uint32_t activeRayCount

    ) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &sensor = pkg.sensor;
        auto &settings = pkg.settings;
        auto *raysIn = pkg.intermediates.primaryRays;

        queue.submit([&](sycl::handler &cgh) {
            uint64_t baseSeed = settings.randomSeed;
            cgh.parallel_for<struct launchDirectShadeKernel>(
                sycl::range<1>(activeRayCount),
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
                    auto transmittance = traceVisibility(contribRay, tMax, scene, rng128);
                    if (!transmittance.hit) {
                        // perspective projection
                        float4 clip = camera.proj * (camera.view * float4(rayState.ray.origin, 1.f));

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
                                float3 color = throughput * G_cam ;

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

    void launchContributionKernel(RenderPackage &pkg, uint32_t activeRayCount) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &sensor = pkg.sensor;
        auto &settings = pkg.settings;
        auto *hitRecords = pkg.intermediates.hitRecords;
        auto *raysIn = pkg.intermediates.primaryRays;

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
                    if (!worldHit.hit) {
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


    // ---- Kernel: Camera gather (one thread per pixel) --------------------------
    void launchCameraGatherKernel(RenderPackage &pkg) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &sensor = pkg.sensor;
        auto &photonMap = pkg.intermediates.map; // DeviceSurfacePhotonMapGrid

        const std::uint32_t imageWidth = sensor.camera.width;
        const std::uint32_t imageHeight = sensor.camera.height;
        const std::uint32_t pixelCount = imageWidth * imageHeight;

        // Clear framebuffer before calling this, outside.
        const float gatherRadius = photonMap.gatherRadiusWorld;
        //const float normalization = 1.0f / (static_cast<float>(M_PI) * gatherRadius * gatherRadius);
        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class CameraGatherKernel>(
                sycl::range<1>(pixelCount),
                [=](sycl::id<1> tid) {
                    const std::uint32_t pixelIndex = tid[0];
                    const std::uint32_t px = pixelIndex % imageWidth;
                    const std::uint32_t py = pixelIndex / imageWidth;

                    // 1) Trace camera ray to first diffuse mesh hit
                    rng::Xorshift128 rng128(rng::makePerItemSeed1D(pkg.settings.randomSeed, pixelIndex));
                    int samplesPerPixel = 32;
                    float3 radianceRGB(0.0f);
                    for (uint32_t sampleIndex = 0; sampleIndex < samplesPerPixel; ++sampleIndex) {
                        // subpixel jitter
                        const float jx = rng128.nextFloat() - 0.5f;
                        const float jy = rng128.nextFloat() - 0.5f;

                        Ray primary = makePrimaryRayFromPixelJittered(sensor.camera, float(px), float(py), jx, jy);
                        WorldHit worldHit{};
                        intersectScene(primary, &worldHit, scene, rng128);
                        if (!worldHit.hit) continue; // miss → black (or add env if desired)

                        radianceRGB = radianceRGB + estimateRadianceFromPhotonMap(
                                          worldHit, scene, photonMap, pkg.settings.photonsPerLaunch);
                    }

                    radianceRGB = radianceRGB * (1.0f / static_cast<float>(samplesPerPixel));
                    // Atomic accumulate
                    const std::uint32_t fbIndex = py * imageWidth + px; // flip Y like your code
                    float4 &dst = sensor.framebuffer[fbIndex];

                    const sycl::atomic_ref<float, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space> red(dst.x());
                    const sycl::atomic_ref<float, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space> green(dst.y());
                    const sycl::atomic_ref<float, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space> blue(dst.z());
                    const sycl::atomic_ref<float, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space> alpha(dst.w());

                    red.fetch_add(radianceRGB.x());
                    green.fetch_add(radianceRGB.y());
                    blue.fetch_add(radianceRGB.z());
                    alpha.store(1.0f);
                });
        }).wait();
    }
} // Pale
