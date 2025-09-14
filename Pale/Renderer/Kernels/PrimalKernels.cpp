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
                        *countPrimary);
                    const uint32_t slot = counter.fetch_add(1);
                    raysIn[slot] = ray;
                });
        });
        queue.wait();
    }

    struct LaunchIntersectKernel {
        LaunchIntersectKernel(GPUSceneBuffers scene, RenderIntermediatesGPU intermediates,
                              uint32_t seed) : m_scene(scene),
                                               m_intermediates(intermediates), m_baseSeed(seed) {
        }

        void operator()(sycl::id<1> globalId) const {
            const uint32_t rayIndex = globalId[0];
            const uint64_t perItemSeed = rng::makePerItemSeed1D(m_baseSeed, rayIndex);
            rng::Xorshift128 rng128(perItemSeed);

            WorldHit worldHit{};
            RayState rayState = m_intermediates.primaryRays[rayIndex];
            intersectScene(rayState.ray, &worldHit, m_scene, rng128);
            if (worldHit.t == FLT_MAX) {
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
                break;
                case GeometryType::PointCloud: {
                    auto &surfel = m_scene.points[worldHit.primitiveIndex];
                    const float3 normalObject = normalize(cross(surfel.tanU, surfel.tanV));
                    worldHit.geometricNormalW = normalObject;
                    m_intermediates.hitRecords[rayIndex] = worldHit;
                }
                break;
            }
        }

    private:
        GPUSceneBuffers m_scene{};
        RenderIntermediatesGPU m_intermediates{};
        uint32_t m_baseSeed = 0;
    };

    void launchIntersectKernel(RenderPackage &pkg, uint32_t activeRayCount) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto &intermediates = pkg.intermediates;

        queue.submit([&](sycl::handler &cgh) {
            LaunchIntersectKernel kernel(scene, intermediates, settings.randomSeed);
            cgh.parallel_for<struct IntersectKernelTag>(
                sycl::range<1>(activeRayCount), kernel);
        });
        queue.wait(); // DEBUG: ensure the thread blocks here
    }

    //
    static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;

    inline sycl::int3 worldToCell(const float3 &worldPosition,
                                  const DeviceSurfacePhotonMapGrid &grid) {
        const float3 safeCellSize = max(grid.cellSizeWorld, float3{1e-6f, 1e-6f, 1e-6f});
        const float3 relative = worldPosition - grid.gridOriginWorld;
        const float3 r = float3{
            relative.x() / safeCellSize.x(),
            relative.y() / safeCellSize.y(),
            relative.z() / safeCellSize.z()
        };

        return sycl::int3{
            static_cast<int>(sycl::floor(r.x())),
            static_cast<int>(sycl::floor(r.y())),
            static_cast<int>(sycl::floor(r.z()))
        };
    }

    inline std::uint32_t linearCellIndex(const sycl::int3 &cell,
                                         const sycl::int3 &resolution) {
        const int ix = sycl::clamp(static_cast<int>(cell.x()), 0, static_cast<int>(resolution.x()) - 1);
        const int iy = sycl::clamp(static_cast<int>(cell.y()), 0, static_cast<int>(resolution.y()) - 1);
        const int iz = sycl::clamp(static_cast<int>(cell.z()), 0, static_cast<int>(resolution.z()) - 1);

        const auto nx = static_cast<std::uint64_t>(resolution.x());
        const auto ny = static_cast<std::uint64_t>(resolution.y());
        const auto lix = static_cast<std::uint64_t>(ix);
        const auto liy = static_cast<std::uint64_t>(iy);
        const auto liz = static_cast<std::uint64_t>(iz);

        const std::uint64_t linear =
                liz * nx * ny + liy * nx + lix; // fits if totalCellCount < 2^32

        return static_cast<std::uint32_t>(linear);
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


    /*
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
*/

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
                    next.pathThroughput = throughput * (brdf * cosTheta / sycl::fmax(cosinePDF, 1e-6f)); // also ✅
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


    // Generate a primary ray from pixel (uses inverse view-projection on the camera)
    inline Ray makePrimaryRayFromPixel(const CameraGPU &camera,
                                       std::uint32_t pixelX,
                                       std::uint32_t pixelY,
                                       bool flipY = true) {
        // 1) Pixel center → NDC in [-1,1]^2
        const float sx = (static_cast<float>(pixelX) + 0.5f) / static_cast<float>(camera.width);
        float sy = (static_cast<float>(pixelY) + 0.5f) / static_cast<float>(camera.height);
        if (flipY) sy = 1.0f - sy;

        const float ndcX = 2.0f * sx - 1.0f;
        const float ndcY = 2.0f * sy - 1.0f;

        // 2) Unproject a far point on the view ray
        const float4 clipFar = float4{ndcX, ndcY, 1.0f, 1.0f}; // OpenGL-style clip
        const float4 worldFarH = camera.invView * (camera.invProj * clipFar);
        const float invW = 1.0f / worldFarH.w();
        const float3 worldFar = float3{
            worldFarH.x() * invW,
            worldFarH.y() * invW,
            worldFarH.z() * invW
        };

        // 3) Ray origin = camera position; dir = normalized (far - origin)
        const float4 camPosH = camera.invView * float4{0.0f, 0.0f, 0.0f, 1.0f};
        const float3 rayOrigin = float3{camPosH.x(), camPosH.y(), camPosH.z()};

        float3 rayDirection = worldFar - rayOrigin;
        rayDirection = normalize(rayDirection);

        Ray ray{};
        ray.origin = rayOrigin;
        ray.direction = rayDirection;
        return ray;
    }

    inline bool intersectRayWithViewPlane(const float3& rayOrigin,
                                      const float3& rayDir,
                                      const float3& planePoint,
                                      const float3& planeNormal,
                                      float3& outHit)
    {
        const float denom = dot(rayDir, planeNormal);
        if (sycl::fabs(denom) < 1e-6f) return false;
        const float t = dot(planePoint - rayOrigin, planeNormal) / denom;
        if (t <= 0.f) return false;
        outHit = rayOrigin + rayDir * t;
        return true;
    }

    // world-space pixel width at distance `depthFromCamera` along camera.forward
    inline float pixelWorldWidthAtDepth(const CameraGPU& cam,
                                        std::uint32_t pixelX,
                                        std::uint32_t pixelY,
                                        float depthFromCamera,
                                        bool flipY = true)
    {
        // center and one-pixel-right rays
        const Ray rCenter = makePrimaryRayFromPixel(cam, pixelX,   pixelY, flipY);
        const Ray rRight  = makePrimaryRayFromPixel(cam, pixelX+1, pixelY, flipY);

        const float3 planePoint  = cam.pos + cam.forward * depthFromCamera;
        const float3 planeNormal = cam.forward;

        float3 pCenter, pRight;
        if (!intersectRayWithViewPlane(rCenter.origin, rCenter.direction, planePoint, planeNormal, pCenter)) return 0.f;
        if (!intersectRayWithViewPlane(rRight.origin,  rRight.direction,  planePoint, planeNormal, pRight )) return 0.f;

        return length(pRight - pCenter);
    }


    // ---- Kernel: Camera gather (one thread per pixel) --------------------------
    void launchCameraGatherKernel(RenderPackage &pkg) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &sensor = pkg.sensor;
        auto &map = pkg.intermediates.map; // DeviceSurfacePhotonMapGrid

        const std::uint32_t imageWidth = sensor.camera.width;
        const std::uint32_t imageHeight = sensor.camera.height;
        const std::uint32_t pixelCount = imageWidth * imageHeight;

        // Clear framebuffer before calling this, outside.
        const float gatherRadius = map.gatherRadiusWorld;
        //const float normalization = 1.0f / (static_cast<float>(M_PI) * gatherRadius * gatherRadius);
        const float normalization = 1.0f / (static_cast<float>(M_PI) * gatherRadius * gatherRadius);

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class CameraGatherKernel>(
                sycl::range<1>(pixelCount),
                [=](sycl::id<1> tid) {
                    const std::uint32_t pixelIndex = tid[0];
                    const std::uint32_t px = pixelIndex % imageWidth;
                    const std::uint32_t py = pixelIndex / imageWidth;

                    // 1) Trace camera ray to first diffuse mesh hit
                    rng::Xorshift128 rng128(rng::makePerItemSeed1D(pkg.settings.randomSeed, pixelIndex));
                    Ray primary = makePrimaryRayFromPixel(sensor.camera, px, py);

                    WorldHit worldHit{};
                    intersectScene(primary, &worldHit, scene, rng128);
                    if (worldHit.t == FLT_MAX) return; // miss → black (or add env if desired)

                    const InstanceRecord &instance = scene.instances[worldHit.instanceIndex];
                    if (instance.geometryType != GeometryType::Mesh) return; // diffuse-only pass

                    const GPUMaterial material = scene.materials[instance.materialIndex];
                    const float3 diffuseAlbedoRGB = material.baseColor;

                    const bool isEmissiveHit =
                        (material.emissive.x() > 0.f) || (material.emissive.y() > 0.f) || (material.emissive.z() > 0.f);
                    float3 radianceDirect = float3{0,0,0};
                    if (isEmissiveHit) {
                        radianceDirect = radianceDirect + material.emissive; // Le toward camera
                    }

                    const float3 surfacePosition = worldHit.hitPositionW;

                    // 2) Gather photons from 27 neighbor cells
                    // --- per-hit local radius ---
                    const float depthFromCamera = dot(surfacePosition - sensor.camera.pos, sensor.camera.forward);

                    const float baseRadius = map.gatherRadiusWorld;
                    const float pixelWidth = sycl::fmax(1e-8f,
                        pixelWorldWidthAtDepth(sensor.camera, px, py, depthFromCamera, /*flipY=*/true));

                    const float betaScale = 2.0f;                 // tune 1–3
                    const float localRadius = sycl::clamp(betaScale * pixelWidth,
                                                          1e-3f * baseRadius,
                                                          4.0f  * baseRadius);

                    const float localRadiusSquared = localRadius * localRadius;

                    const float kappa = 1.3f;
                    const float invConeNorm = 1.0f / ((1.0f - 2.0f/(3.0f*kappa)) * float(M_PI) * localRadiusSquared);

                    // If you emit M photons this pass, invNumEmittedPhotons = 1/M.
                    // If you accumulate across P passes, use 1/(M*P) or track a running scale.
                    const float invNumEmittedPhotons = 1.0f / float(pkg.settings.photonsPerLaunch);

                    // center cell and accumulation
                    const sycl::int3 centerCell = worldToCell(surfacePosition, map);
                    float3 weightedSumPhotonPowerRGB = float3{0.f, 0.f, 0.f};

                    for (int dz = -1; dz <= 1; ++dz)
                        for (int dy = -1; dy <= 1; ++dy)
                            for (int dx = -1; dx <= 1; ++dx) {
                                const sycl::int3 neighborCell = sycl::int3{
                                    centerCell.x() + dx, centerCell.y() + dy, centerCell.z() + dz
                                };
                                const std::uint32_t cellIndex = linearCellIndex(neighborCell, map.gridResolution);

                                // walk linked list
                                for (std::uint32_t idx = map.cellHeadIndexArray[cellIndex];
                                     idx != kInvalidIndex;
                                     idx = map.photonNextIndexArray[idx]) {
                                    const DevicePhotonSurface photon = map.photons[idx];
                                    const float3 delta = photon.position - surfacePosition;

                                    const float distanceSquared = dot(delta, delta);
                                    if (distanceSquared > localRadiusSquared) continue;

                                    const float distance = sycl::sqrt(distanceSquared);
                                    const float weight = sycl::fmax(0.f, 1.f - distance / (kappa * localRadius));

                                    // If photon.power already includes cosine at store time, drop * photon.cosineIncident here.
                                    weightedSumPhotonPowerRGB =
                                            weightedSumPhotonPowerRGB + weight * (photon.power * photon.cosineIncident);
                                }
                            }
                    // irradiance and radiance
                    const float3 irradianceRGB = weightedSumPhotonPowerRGB * (invConeNorm * invNumEmittedPhotons);
                    const float3 radianceRGB = radianceDirect + irradianceRGB * (diffuseAlbedoRGB * (1.0f / static_cast<float>(M_PI)));

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
