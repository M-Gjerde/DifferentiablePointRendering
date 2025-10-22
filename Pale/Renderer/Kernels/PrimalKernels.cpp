//
// Created by magnus on 9/12/25.
//

#include "PrimalKernels.h"

#include <cmath>

#include "KernelHelpers.h"
#include "IntersectionKernels.h"

namespace Pale {
    void launchRayGenEmitterKernel(RenderPackage& pkg) {
        auto queue = pkg.queue;
        auto scene = pkg.scene;
        auto sensor = pkg.sensor;
        auto settings = pkg.settings;

        auto* hitRecords = pkg.intermediates.hitRecords;
        auto* raysIn = pkg.intermediates.primaryRays;
        auto* raysOut = pkg.intermediates.extensionRaysA;
        auto* countPrimary = pkg.intermediates.countPrimary;


        const uint32_t photonCount = settings.photonsPerLaunch;
        const uint32_t forwardPasses = settings.numForwardPasses;
        const uint32_t totalPhotons = photonCount * forwardPasses;

        queue.submit([&](sycl::handler& commandGroupHandler) {
            uint64_t baseSeed = settings.randomSeed;

            commandGroupHandler.parallel_for<struct RayGenEmitterKernelTag>(
                sycl::range<1>(photonCount),
                [=](sycl::id<1> globalId) {
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, globalId[0]);
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
                    const uint32_t triangleRelativeIndex = sycl::min((uint32_t)(uT * light.triangleCount),
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

                    // transform to world per-vertex, then interpolate
                    const Transform xform = scene.transforms[light.transformIndex];
                    const float3 worldP0 = toWorldPoint(p0, xform);
                    const float3 worldP1 = toWorldPoint(p1, xform);
                    const float3 worldP2 = toWorldPoint(p2, xform);
                    const float3 sampledWorldPoint = worldP0 * b0 + worldP1 * b1 + worldP2 * b2;

                    // world-space geometric normal and area
                    const float3 worldEdge0 = worldP1 - worldP0;
                    const float3 worldEdge1 = worldP2 - worldP0;
                    float3 unnormalizedWorldNormal = float3{
                        worldEdge0.y() * worldEdge1.z() - worldEdge0.z() * worldEdge1.y(),
                        worldEdge0.z() * worldEdge1.x() - worldEdge0.x() * worldEdge1.z(),
                        worldEdge0.x() * worldEdge1.y() - worldEdge0.y() * worldEdge1.x()
                    };
                    const float worldArea = 0.5f * sycl::sqrt(dot(unnormalizedWorldNormal, unnormalizedWorldNormal));
                    if (worldArea <= 0.f) return;

                    const float3 lightNormal = normalize(unnormalizedWorldNormal);
                    // 2c) cosine-hemisphere direction about n
                    float cosTheta = 0;
                    float3 sampledDirection;
                    sampleCosineHemisphere(rng128, lightNormal, sampledDirection, cosTheta);

                    // 3) PDFs
                    const float pdfTriangle = 1.0f / (float)light.triangleCount;
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
                    ray.pathThroughput = initialThroughput;
                    ray.bounceIndex = 0u;

                    // Write the same sample to memory n times.
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

            auto& instance = m_scene.instances[worldHit.instanceIndex];

            switch (instance.geometryType) {
            case GeometryType::Mesh:
                {
                    const Triangle& triangle = m_scene.triangles[worldHit.primitiveIndex];
                    const Transform& objectWorldTransform = m_scene.transforms[instance.transformIndex];

                    const Vertex& vertex0 = m_scene.vertices[triangle.v0];
                    const Vertex& vertex1 = m_scene.vertices[triangle.v1];
                    const Vertex& vertex2 = m_scene.vertices[triangle.v2];

                    // Canonical geometric normal (no face-forwarding)
                    const float3 worldP0 = toWorldPoint(vertex0.pos, objectWorldTransform);
                    const float3 worldP1 = toWorldPoint(vertex1.pos, objectWorldTransform);
                    const float3 worldP2 = toWorldPoint(vertex2.pos, objectWorldTransform);
                    const float3 canonicalNormalW = normalize(cross(worldP1 - worldP0, worldP2 - worldP0));
                    worldHit.geometricNormalW = canonicalNormalW;

                    m_intermediates.hitRecords[rayIndex] = worldHit;

                    if (m_settings.rayGenMode == RayGenMode::Emitter) {
                        auto& devicePtr = *m_intermediates.map.photonCountDevicePtr;
                        sycl::atomic_ref<uint32_t,
                                         sycl::memory_order::acq_rel,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            photonCounter(devicePtr);

                        const uint32_t reservedSlot = photonCounter.fetch_add(1u);
                        if (reservedSlot >= m_intermediates.map.photonCapacity) return;

                        DevicePhotonSurface photonEntry{};
                        photonEntry.position = worldHit.hitPositionW;
                        photonEntry.power = rayState.pathThroughput;
                        photonEntry.incidentDir = -rayState.ray.direction;

                        const float signedCosineIncident = dot(canonicalNormalW, photonEntry.incidentDir);
                        photonEntry.cosineIncident = sycl::fabs(signedCosineIncident);
                        photonEntry.sideSign = signNonZero(signedCosineIncident);

                        m_intermediates.map.photons[reservedSlot] = photonEntry;
                    }
                }
                break;

            case GeometryType::PointCloud:
                {
                    const auto& surfel = m_scene.points[worldHit.primitiveIndex];

                    // Canonical surfel normal from tangents (no face-forwarding)
                    const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));
                    worldHit.geometricNormalW = canonicalNormalW;
                    m_intermediates.hitRecords[rayIndex] = worldHit;

                    if (m_settings.rayGenMode == RayGenMode::Emitter) {
                        auto& devicePtr = *m_intermediates.map.photonCountDevicePtr;
                        sycl::atomic_ref<uint32_t,
                                         sycl::memory_order::acq_rel,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            photonCounter(devicePtr);

                        const uint32_t reservedSlot = photonCounter.fetch_add(1u);
                        if (reservedSlot >= m_intermediates.map.photonCapacity) return;

                        DevicePhotonSurface photonEntry{};
                        photonEntry.position = worldHit.hitPositionW;
                        photonEntry.power = rayState.pathThroughput;
                        photonEntry.incidentDir = -rayState.ray.direction;

                        const float signedCosineIncident = dot(canonicalNormalW, photonEntry.incidentDir);
                        photonEntry.cosineIncident = sycl::fabs(signedCosineIncident);
                        photonEntry.sideSign = 1.0;

                        m_intermediates.map.photons[reservedSlot] = photonEntry;
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

    void launchIntersectKernel(RenderPackage& pkg, uint32_t activeRayCount) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& settings = pkg.settings;
        auto& intermediates = pkg.intermediates;

        queue.submit([&](sycl::handler& cgh) {
            LaunchIntersectKernel kernel(scene, intermediates, settings);
            cgh.parallel_for<struct IntersectKernelTag>(
                sycl::range<1>(activeRayCount), kernel);
        });
        queue.wait(); // DEBUG: ensure the thread blocks here
    }


    void generateNextRays(RenderPackage& pkg, uint32_t activeRayCount) {
        auto queue = pkg.queue;
        auto scene = pkg.scene;
        auto settings = pkg.settings;

        auto* hitRecords = pkg.intermediates.hitRecords;
        auto* raysIn = pkg.intermediates.primaryRays;
        auto* raysOut = pkg.intermediates.extensionRaysA;
        auto* countExtensionOut = pkg.intermediates.countExtensionOut;

        queue.submit([&](sycl::handler& cgh) {
            uint64_t baseSeed = settings.randomSeed;
            cgh.parallel_for<class ShadeKernelTag>(
                sycl::range<1>(activeRayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);

                    constexpr float surfaceEpsilon = 5e-4f;

                    const WorldHit worldHit = hitRecords[rayIndex];
                    const RayState rayState = raysIn[rayIndex];
                    if (!worldHit.hit) return;

                    const InstanceRecord instance = scene.instances[worldHit.instanceIndex];

                    // --- Canonical normal at the hit (no face-forwarding stored in hit) ---
                    const float3 canonicalNormalW = worldHit.geometricNormalW;
                    // --- Orient to the side we entered first (two-sided, side-correct sampling) ---
                    // travelSideSign = sign(dot(n, -wo))
                    const int travelSideSign = signNonZero(dot(canonicalNormalW, -rayState.ray.direction));
                    const float3 shadingNormal = (travelSideSign >= 0) ? canonicalNormalW : (-canonicalNormalW);
                    // --- Cosine-hemisphere sampling over the *entered* side ---
                    float cosinePdf = 0.0f;
                    float3 sampledOutgoingDirection;
                    // --- Material fetch (two-sided diffuse) ---
                    GPUMaterial material{};
                    if (instance.geometryType == GeometryType::Mesh) {
                        material = scene.materials[instance.materialIndex];
                        sampleCosineHemisphere(rng128, shadingNormal, sampledOutgoingDirection, cosinePdf);
                    }
                    else {
                        // PointCloud
                        material.baseColor = scene.points[worldHit.primitiveIndex].color;
                        sampleUniformSphere(rng128, sampledOutgoingDirection, cosinePdf);
                        //sampleCosineHemisphere(rng128, shadingNormal, sampledOutgoingDirection, cosinePdf);

                    }


                    cosinePdf = sycl::fmax(cosinePdf, 1e-6f);

                    // --- Diffuse BRDF and geometry term for MIS-free cosine sampling ---
                    const float3 lambertBrdf = material.baseColor * (1.0f / M_PIf);
                    const float cosTheta = sycl::fmax(0.f, dot(sampledOutgoingDirection, shadingNormal));

                    // --- Spawn next ray with offset along *oriented* normal ---
                    RayState nextState{};
                    nextState.ray.origin = worldHit.hitPositionW + shadingNormal * surfaceEpsilon;
                    nextState.ray.direction = sampledOutgoingDirection;
                    nextState.bounceIndex = rayState.bounceIndex + 1;
                    nextState.pixelIndex = rayState.pixelIndex;

                    // --- Throughput update (cosine/cosinePdf cancel ideally; keep robust form) ---
                    const float3 updatedThroughput =
                        rayState.pathThroughput * (lambertBrdf * (cosTheta / cosinePdf));
                    nextState.pathThroughput = updatedThroughput;

                    // --- Enqueue ---
                    auto extensionCounter = sycl::atomic_ref<uint32_t,
                                                             sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>(
                        *countExtensionOut);
                    const uint32_t outIndex = extensionCounter.fetch_add(1);
                    raysOut[outIndex] = nextState;
                });
        });
        queue.wait();
    }


    // ---- Kernel: Camera gather (one thread per pixel) --------------------------
    void launchCameraGatherKernel(RenderPackage& pkg, int totalSamplesPerPixel) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& sensor = pkg.photonMapSensor;
        auto& settings = pkg.settings;
        auto& photonMap = pkg.intermediates.map; // DeviceSurfacePhotonMapGrid

        const std::uint32_t imageWidth = sensor.camera.width;
        const std::uint32_t imageHeight = sensor.camera.height;
        const std::uint32_t pixelCount = imageWidth * imageHeight;

        // Clear framebuffer before calling this, outside.
        const float gatherRadius = photonMap.gatherRadiusWorld;
        queue.submit([&](sycl::handler& cgh) {
            uint64_t baseSeed = pkg.settings.randomSeed;
            float samplesPerPixel = static_cast<float>(totalSamplesPerPixel);
            cgh.parallel_for<class CameraGatherKernel>(
                sycl::range<1>(pixelCount),
                [=](sycl::id<1> tid) {
                    const std::uint32_t pixelIndex = tid[0];
                    const std::uint32_t px = pixelIndex % imageWidth;
                    const std::uint32_t py = pixelIndex / imageWidth;

                    // 1) Trace camera ray to first diffuse mesh hit
                    rng::Xorshift128 rng128(rng::makePerItemSeed1D(baseSeed, pixelIndex));
                    float3 radianceRGB(0.0f);
                    // subpixel jitter
                    const float jx = rng128.nextFloat() - 0.5f;
                    const float jy = rng128.nextFloat() - 0.5f;

                    Ray ray = makePrimaryRayFromPixelJittered(sensor.camera, static_cast<float>(px),
                                                              static_cast<float>(py), jx, jy);

                    //ray.direction = normalize(float3{-0.187575308, 0.962122211, -0.277827293}); // b
                    //ray.origin = float3{0.0, -4.0, 1.0};

                    WorldHit worldHit{};
                    intersectScene(ray, &worldHit, scene, rng128, RayIntersectMode::Transmit); // Force transmit

                    //if (pixelIndex  > 2) {
                    //return;
                    //}
                    //ray.origin = worldHit.hitPositionW;
                    //ray.direction = normalize(float3{0.35, 0.25, 1});
                    //ray.normal = normalize(float3{0.0, 0.0, 1});
                    //intersectScene(ray, &worldHit, scene, rng128, RayIntersectMode::Transmit); // Force transmit


                    float3 Lsheet(0.0f);
                    float tau = 1.0f;

                    for (int i = 0; i < worldHit.splatEventCount; ++i) {
                        const auto& splatEvent = worldHit.splatEvents[i];
                        if (worldHit.hit && splatEvent.t >= worldHit.t)
                            break;

                        float3 L = estimateSurfelRadianceFromPhotonMap(splatEvent, ray.direction, scene, photonMap);

                        float distanceToCamera = length(splatEvent.hitWorld - ray.origin);
                        float cameraCos = sycl::fmax(0.f, dot(ray.normal, ray.direction)); // optional
                        float volumeGeometric = (cameraCos) / (distanceToCamera * distanceToCamera);

                        const float3 diffuseAlbedo = scene.points[splatEvent.primitiveIndex].color;
                        const float3 lambertBrdf = diffuseAlbedo * (1.0f / M_PIf);

                        Lsheet = Lsheet + L * splatEvent.alpha * tau * volumeGeometric * lambertBrdf;

                        tau *= (1.0f - splatEvent.alpha);

                    }

                    radianceRGB = radianceRGB + Lsheet;

                    if (!worldHit.hit) {
                        radianceRGB = radianceRGB + Lsheet;
                        return;
                    }
                    const InstanceRecord& instance = scene.instances[worldHit.instanceIndex];

                    if (instance.geometryType == GeometryType::Mesh) {
                        const Triangle& triangle = scene.triangles[worldHit.primitiveIndex];
                        const Transform& objectWorldTransform = scene.transforms[instance.transformIndex];
                        const Vertex& vertex0 = scene.vertices[triangle.v0];
                        const Vertex& vertex1 = scene.vertices[triangle.v1];
                        const Vertex& vertex2 = scene.vertices[triangle.v2];
                        // Canonical geometric normal (no face-forwarding)
                        const float3 worldP0 = toWorldPoint(vertex0.pos, objectWorldTransform);
                        const float3 worldP1 = toWorldPoint(vertex1.pos, objectWorldTransform);
                        const float3 worldP2 = toWorldPoint(vertex2.pos, objectWorldTransform);
                        worldHit.geometricNormalW = normalize(cross(worldP1 - worldP0, worldP2 - worldP0));
                        float distanceToCamera = length(worldHit.hitPositionW - ray.origin);
                        float surfaceCos = sycl::fmax(0.f, dot(worldHit.geometricNormalW, -ray.direction));
                        float cameraCos = sycl::fmax(0.f, dot(ray.normal, ray.direction)); // optional
                        float geometricToCamera = (surfaceCos * cameraCos) / (distanceToCamera * distanceToCamera);
                        // If the mesh itself is emissive, add its emitted radiance (already radiance units)
                        const GPUMaterial& material = scene.materials[instance.materialIndex];
                        const float3 lambertBrdf = material.baseColor * (1.0f / M_PIf);

                        if (material.isEmissive()) {
                            // Optionally evaluate Le(x, wo) from your material; shown as baseColor for brevity
                            const float3 emittedLe = material.emissive * worldHit.transmissivity * geometricToCamera;
                            const float3 reflectedL = estimateRadianceFromPhotonMap(worldHit, scene, photonMap) *
                                worldHit.transmissivity * lambertBrdf * geometricToCamera;
                            radianceRGB = radianceRGB + (emittedLe + reflectedL);
                        }
                        else {
                            const float3 L = estimateRadianceFromPhotonMap(worldHit, scene, photonMap) ;
                            radianceRGB = radianceRGB + (L *
                                worldHit.transmissivity * lambertBrdf * geometricToCamera);
                        }
                    }

                    // Atomic accumulate
                    const std::uint32_t fbIndex = py * imageWidth + px; // flip Y like your code
                    float4 previous = sensor.framebuffer[fbIndex];
                    float4 current = float4(radianceRGB.x(), radianceRGB.y(), radianceRGB.z(), 1.0f) / samplesPerPixel;
                    sensor.framebuffer[fbIndex] = previous + current; // or write to a staging buffer
                });
        }).wait();
    }


    void launchDirectContributionKernel(RenderPackage& pkg, uint32_t activeRayCount

    ) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& sensor = pkg.sensor;
        auto& settings = pkg.settings;
        auto* raysIn = pkg.intermediates.primaryRays;

        queue.submit([&](sycl::handler& cgh) {
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
                    auto& camera = sensor.camera;
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
                                const uint32_t idx = py * camera.width + px;

                                float4& dst = sensor.framebuffer[idx];
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

    void launchContributionKernel(RenderPackage& pkg, uint32_t activeRayCount) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& sensor = pkg.sensor;
        auto& settings = pkg.settings;
        auto* hitRecords = pkg.intermediates.hitRecords;
        auto* raysIn = pkg.intermediates.primaryRays;

        queue.submit([&](sycl::handler& cgh) {
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
                    float3 pathThroughput = rayState.pathThroughput;
                    auto& instance = scene.instances[worldHit.instanceIndex];
                    GPUMaterial material;
                    switch (instance.geometryType) {
                    case GeometryType::Mesh:
                        material = scene.materials[instance.materialIndex];
                        break;
                    case GeometryType::PointCloud:
                        {
                            auto val = scene.points[worldHit.primitiveIndex];
                            material.baseColor = val.color;
                        }
                        break;
                    }
                    // Construct Ray towards camera
                    auto& camera = sensor.camera;
                    float3 toPinhole = camera.pos - worldHit.hitPositionW;
                    float distanceToPinhole = length(toPinhole);
                    float3 directionToPinhole = toPinhole / distanceToPinhole;
                    // distance to camera:
                    Ray contribRay{
                        .origin = worldHit.hitPositionW + directionToPinhole * kEps,
                        .direction = directionToPinhole
                    };
                    // Shoot contribution ray towards camera
                    // Visibility to camera
                    const float tMax = sycl::fmax(0.f, distanceToPinhole - kEps);
                    WorldHit vis = traceVisibility(contribRay, tMax, scene, rng128);

                    // Otherwise attenuate by transmittance through splats
                    const float transmittanceToCamera = sycl::clamp(vis.transmissivity, 0.0f, 1.0f);
                    if (transmittanceToCamera <= 0.0f) return;

                    // Shading
                    float3 brdf = material.baseColor * (1.0f / M_PIf);
                    float surfaceCos = sycl::fmax(0.f, dot(worldHit.geometricNormalW, directionToPinhole));
                    float cameraCos = sycl::fmax(0.f, dot(camera.forward, -directionToPinhole)); // optional
                    float geometricToCamera = (surfaceCos * cameraCos) / (distanceToPinhole * distanceToPinhole);

                    float3 color = pathThroughput * brdf * geometricToCamera * transmittanceToCamera * 100000;
                    // TODO just for debug visualization but this approach is broken in its current form.

                    // Accumulate
                    float4 clip = camera.proj * (camera.view * float4(worldHit.hitPositionW, 1.f));
                    if (clip.w() <= 0.0f) return;
                    float2 ndc = {clip.x() / clip.w(), clip.y() / clip.w()};
                    if (ndc.x() < -1.f || ndc.x() > 1.f || ndc.y() < -1.f || ndc.y() > 1.f) return;

                    const uint32_t px = sycl::clamp(static_cast<uint32_t>((ndc.x() * 0.5f + 0.5f) * camera.width), 0u,
                                                    camera.width - 1);
                    const uint32_t py = sycl::clamp(static_cast<uint32_t>((ndc.y() * 0.5f + 0.5f) * camera.height), 0u,
                                                    camera.height - 1);
                    const uint32_t idx = py * camera.width + px;

                    float4& dst = sensor.framebuffer[idx];
                    const sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                           sycl::access::address_space::global_space>
                        r(dst.x()), g(dst.y()), b(dst.z()), a(dst.w());
                    r.fetch_add(color.x());
                    g.fetch_add(color.y());
                    b.fetch_add(color.z());
                    a.store(1.0f);
                });
        });
        queue.wait();
    }


    void clearGridHeads(sycl::queue& q, DeviceSurfacePhotonMapGrid& g) {
        q.fill(g.cellHeadIndexArray, kInvalidIndex, g.totalCellCount).wait();
    }

    void buildPhotonGridLinkedLists(sycl::queue& q, DeviceSurfacePhotonMapGrid g, uint32_t photonCount) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(photonCount), [=](sycl::id<1> id) {
                uint32_t i = id[0];
                DevicePhotonSurface ph = g.photons[i];
                if (ph.power == float3{0.0f})
                    return;
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
} // Pale
