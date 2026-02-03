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
        const uint32_t forwardPasses = settings.numForwardPasses;
        const float totalPhotons = photonCount * forwardPasses;

        queue.submit([&](sycl::handler &commandGroupHandler) {
            uint64_t baseSeed = settings.randomSeed;
            float invPhotonCount = 1.f / totalPhotons; // 1/N

            commandGroupHandler.parallel_for<struct RayGenEmitterKernelTag>(
                sycl::range<1>(photonCount),
                [=](sycl::id<1> globalId) {
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, globalId[0]);
                    // Choose any generator you like:
                    rng::Xorshift128 rng128(perItemSeed);

                    if (scene.lightCount == 0) return;

                    AreaLightSample ls = sampleMeshAreaLight(scene, rng128);
                    if (!ls.valid) return;

                    // Storing radiance as watt simplifies this line:
                    const float3 initialThroughput = ls.power * scene.lightCount * invPhotonCount;

                    RayState ray{};
                    ray.ray.origin = ls.positionW;
                    ray.ray.direction = ls.direction;
                    ray.ray.normal = ls.normalW;
                    ray.pathThroughput = initialThroughput;
                    ray.bounceIndex = 0;
                    ray.lightIndex = ls.lightIndex;

                    auto counter = sycl::atomic_ref<uint32_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(*countPrimary);
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
            intersectScene(rayState.ray, &worldHit, m_scene, rng128, RayIntersectMode::Random);
            if (!worldHit.hit) {
                m_intermediates.hitRecords[rayIndex] = worldHit;
                return;
            }
            buildIntersectionNormal(m_scene, worldHit);
            m_intermediates.hitRecords[rayIndex] = worldHit;
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


    void generateNextRays(RenderPackage &pkg, uint32_t activeRayCount) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;

        auto &intermediates = pkg.intermediates;
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
                    const WorldHit worldHit = hitRecords[rayIndex];
                    const RayState rayState = raysIn[rayIndex];
                    if (!worldHit.hit) return;

                    const InstanceRecord instance = scene.instances[worldHit.instanceIndex];
                    auto &geometryType = instance.geometryType;
                    float3 throughputMultiplier{0.0f};
                    float3 sampledOutgoingDirectionW = rayState.ray.direction;

                    if (geometryType == GeometryType::Mesh) {
                        const GPUMaterial material = scene.materials[instance.materialIndex];
                        // If we hit instance was a mesh do ordinary BRDF stuff.
                        float sampledPdf = 0.0f;
                        sampleCosineHemisphere(rng128, worldHit.geometricNormalW, sampledOutgoingDirectionW,
                                               sampledPdf);
                        const float3 lambertBrdf = material.baseColor;
                        throughputMultiplier = lambertBrdf;
                    }

                    if (geometryType == GeometryType::PointCloud) {
                        const Point point = scene.points[worldHit.primitiveIndex];
                        // Reuse same albedo for scatter/Transmission
                        float3 c = point.albedo;
                        float alpha_r = point.alpha_r;
                        float alpha_t = point.alpha_t;

                        float3 rho_r = c * alpha_r; // diffuse reflectance
                        float3 rho_t = c * alpha_t; // diffuse transmission

                        const float segmentTransmittance = worldHit.transmissivity;
                        // Choose scalar for lobe probability
                        const float alphaSum = alpha_r + alpha_t;
                        if (alphaSum <= 0.0f) {
                            throughputMultiplier = float3{0.0f};
                            return;
                        }

                        const float pReflect = alpha_r / alphaSum;
                        const float uLobe = rng128.nextFloat();
                        const bool chooseReflection = (uLobe < pReflect);
                        const float3 lobeNormalW =
                                chooseReflection
                                    ? worldHit.geometricNormalW
                                    : (-worldHit.geometricNormalW);

                        float sampledPdf = 0.0f;
                        sampleCosineHemisphere(
                            rng128,
                            lobeNormalW,
                            sampledOutgoingDirectionW,
                            sampledPdf
                        );
                        // Mixture BSDF update simplifies analytically
                        throughputMultiplier = (rho_r + rho_t) * segmentTransmittance;
                    }

                    if (settings.rayGenMode == RayGenMode::Emitter) {
                        auto &devicePtr = *intermediates.map.photonCountDevicePtr;
                        sycl::atomic_ref<uint32_t,
                                    sycl::memory_order::acq_rel,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                photonCounter(devicePtr);

                        const uint32_t slot = photonCounter.fetch_add(1u);
                        if (slot < intermediates.map.photonCapacity) {
                            DevicePhotonSurface photonEntry{};
                            photonEntry.position = worldHit.hitPositionW;
                            float3 baseNormalW = normalize(worldHit.geometricNormalW);
                            uint32_t primitiveIndexForDeposit = worldHit.instanceIndex;
                            const float signedCosineIncident = dot(baseNormalW, -rayState.ray.direction);
                            const int sideSign = signNonZero(signedCosineIncident);
                            //const float3 orientedNormalW = (sideSign >= 0) ? baseNormalW : (-baseNormalW);
                            photonEntry.power = rayState.pathThroughput;
                            //photonEntry.normal = orientedNormalW;
                            //photonEntry.sideSign = sideSign;
                            //photonEntry.geometryType = instance.geometryType;
                            photonEntry.isValid = 1u;
                            intermediates.map.photons[slot] = photonEntry;
                        }
                    }

                    // --- Spawn next ray with offset along *oriented* normal ---
                    RayState nextState{};
                    // Spawn next ray
                    nextState.ray.origin = worldHit.hitPositionW + (worldHit.geometricNormalW * 1e-6f);
                    nextState.ray.direction = sampledOutgoingDirectionW;
                    nextState.ray.normal = worldHit.geometricNormalW;
                    nextState.bounceIndex = rayState.bounceIndex + 1;
                    nextState.pixelIndex = rayState.pixelIndex;
                    nextState.pathThroughput = rayState.pathThroughput * throughputMultiplier;
                    // --- Russian roulette termination (after computing nextState) ---
                    //if (nextState.bounceIndex >= settings.russianRouletteStart) {
                    //    // Luminance-based continuation probability in [pMin, 1]
                    //    const float3 throughputRgb = nextState.pathThroughput;
                    //    const float luminance = luminanceGrayscale(throughputRgb);
                    //    const float pMin = 0.20f; // safety floor to avoid zero-probability bias
                    //    const float continuationProbability = sycl::clamp(luminance, pMin, 1.0f);
                    //
                    //    if (rng128.nextFloat() >= continuationProbability) {
                    //        return; // terminate path, do not enqueue
                    //    }
                    //    nextState.pathThroughput = nextState.pathThroughput / continuationProbability; // unbiased
                    //}
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

    void launchContributionKernel(RenderPackage &pkg, uint32_t activeRayCount, uint32_t cameraIndex) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto &photonMap = pkg.intermediates.map; // DeviceSurfacePhotonMapGrid
        uint64_t baseSeed = pkg.settings.randomSeed * (cameraIndex + 5);
        // Host-side (before launching kernel)
        SensorGPU sensor = pkg.sensor[cameraIndex];
        auto &intermediates = pkg.intermediates;
        auto *hitRecords = pkg.intermediates.hitRecords;
        auto *raysIn = pkg.intermediates.primaryRays;
        auto *raysOut = pkg.intermediates.extensionRaysA;
        auto *countExtensionOut = pkg.intermediates.countExtensionOut;
        queue.submit([&](sycl::handler &cgh) {
            uint64_t baseSeed = settings.randomSeed * (static_cast<uint64_t>(cameraIndex) + 5ull);

            cgh.parallel_for<class ShadeKernelTag>(
                sycl::range<1>(activeRayCount),
                // ReSharper disable once CppDFAUnusedValue
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);

                    const WorldHit worldHit = hitRecords[rayIndex];
                    const RayState rayState = raysIn[rayIndex];
                    if (!worldHit.hit)
                        return;

                    const InstanceRecord &instance = scene.instances[worldHit.instanceIndex];

                    const float3 &surfacePointWorld = worldHit.hitPositionW;
                    const float3 &surfaceNormalWorld = worldHit.geometricNormalW; // ensure normalized
                    const float3 &incomingDirectionWorld = -rayState.ray.direction; // direction arriving at surface

                    // Project to pixel and get omega_c (surface -> camera) and distance
                    uint32_t pixelIndex = 0u;
                    float3 omegaSurfaceToCamera;
                    float distanceToCamera = 0.0f;

                    bool debug = false;
                    if (!projectToPixelFromPinhole(sensor, surfacePointWorld, pixelIndex, omegaSurfaceToCamera,
                                                distanceToCamera, debug))
                        return;


                    // Backface / cosine term at surface
                    const float signedCosineToCamera = dot(surfaceNormalWorld, omegaSurfaceToCamera);
                    const int travelSideSign = signNonZero(signedCosineToCamera);

                    const float cosineAbsToCamera = sycl::fabs(signedCosineToCamera);

                    if (debug)
                        int i = 1;

                    if (cosineAbsToCamera <= 0.0f)
                        return;

                    //float cosThetaCamera = dot(sensor.camera.forward, -omegaSurfaceToCamera);
                    //if (cosThetaCamera <= 0.0f)
                    //    return;

                    // Visibility: shadow ray from surface to camera
                    const float3 contributionRayOrigin = surfacePointWorld + travelSideSign * surfaceNormalWorld * 1e-6f;
                    const float3 contributionDirection = omegaSurfaceToCamera;
                    const float shadowRayMaxT = distanceToCamera - 1e-4f;
                    Ray ray{contributionRayOrigin, contributionDirection};
                    WorldHit visibilityCheck = traceVisibility(ray, shadowRayMaxT, scene, rng128);
                    if (visibilityCheck.hit && visibilityCheck.t <= shadowRayMaxT) {
                        return;
                    }
                    // Lambertian BSDF: f = albedo / pi

                    const float tauDiffuse = 0.0f; // diffuse transmission

                    float3 rho{0.0f};
                    switch (instance.geometryType) {
                        case GeometryType::Mesh:
                            rho = scene.materials[instance.materialIndex].baseColor;
                            break;
                        case GeometryType::PointCloud:
                            rho = scene.points[worldHit.primitiveIndex].albedo;
                            break;
                        default:
                            break;
                    }

                    const float3 bsdfValue = rho * M_1_PIf;
                    // Geometry term from pinhole importance (1/r^2 and cosine at surface)
                    const float inverseDistanceSquared = 1.0f / (distanceToCamera * distanceToCamera);

                    const float width = float(sensor.width);
                    const float height = float(sensor.height);

                    const float fovYRad = glm::radians(sensor.camera.fovy);
                    const float tanHalfFovY = sycl::tan(0.5f * fovYRad);
                    const float tanHalfFovX = tanHalfFovY * (width / height);

                    // film plane at z=1 has size: 2*tanHalfFovX by 2*tanHalfFovY
                    const float filmWidth = 2.0f * tanHalfFovX;
                    const float filmHeight = 2.0f * tanHalfFovY;

                    const float pixelArea = (filmWidth / width) * (filmHeight / height);
                    const float invPixelArea = 1.0f / pixelArea;
                    // Contribution (delta sensor, pixel binning)
                    const float3 contribution =
                            rayState.pathThroughput *
                            (bsdfValue + float3{tauDiffuse}) *
                            (cosineAbsToCamera * inverseDistanceSquared) * invPixelArea;

                    // Atomic accumulate to framebuffer
                    atomicAddFloat3ToImage(&sensor.framebuffer[pixelIndex], contribution);
                }
            );
        });
    }

    // Check if samples from light sources connect directly to camera
    void launchContributionEmitterVisibleKernel(RenderPackage &pkg, uint32_t activeRayCount, uint32_t cameraIndex) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto &photonMap = pkg.intermediates.map; // DeviceSurfacePhotonMapGrid
        uint64_t baseSeed = pkg.settings.randomSeed * (cameraIndex + 5);


        // Host-side (before launching kernel)
        SensorGPU sensor = pkg.sensor[cameraIndex];

        auto &intermediates = pkg.intermediates;
        auto *hitRecords = pkg.intermediates.hitRecords;
        auto *raysIn = pkg.intermediates.primaryRays;
        auto *raysOut = pkg.intermediates.extensionRaysA;
        auto *countExtensionOut = pkg.intermediates.countExtensionOut;

        queue.submit([&](sycl::handler &cgh) {
            uint64_t baseSeed = settings.randomSeed * (static_cast<uint64_t>(cameraIndex) + 5ull);

            cgh.parallel_for<class ShadeKernelTag>(
                sycl::range<1>(activeRayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);

                    const RayState rayState = raysIn[rayIndex];

                    const float3 surfacePointWorld = rayState.ray.origin;
                    const float3 surfaceNormalWorld = rayState.ray.normal; // ensure normalized

                    // Project to pixel and get omega_c (surface -> camera) and distance
                    uint32_t pixelIndex = 0u;
                    float3 omegaSurfaceToCamera;
                    float distanceToCamera = 0.0f;
                    bool debugPixelBreakpoint = false;

                    if (!projectToPixelFromPinhole(sensor, surfacePointWorld, pixelIndex, omegaSurfaceToCamera,
                                                distanceToCamera, debugPixelBreakpoint))
                        return;

                    // Backface / cosine term at surface
                    const float cosineSurfaceToCamera = sycl::fmax(0.0f, dot(surfaceNormalWorld, omegaSurfaceToCamera));
                    if (cosineSurfaceToCamera <= 0.0f)
                        return;

                    // Visibility: shadow ray from surface to camera
                    const float3 contributionRayOrigin = surfacePointWorld + surfaceNormalWorld * 1e-4f;
                    const float3 contributionDirection = omegaSurfaceToCamera;
                    const float shadowRayMaxT = distanceToCamera - 2e-4f;
                    Ray ray{contributionRayOrigin, contributionDirection};
                    WorldHit visibilityCheck = traceVisibility(ray, shadowRayMaxT, scene, rng128);
                    if (visibilityCheck.t <= shadowRayMaxT)
                        return;

                    // Geometry term from pinhole importance (1/r^2 and cosine at surface)
                    const float inverseDistanceSquared = 1.0f / (distanceToCamera * distanceToCamera);

                    const float width = float(sensor.width);
                    const float height = float(sensor.height);

                    const float fovYRad = glm::radians(sensor.camera.fovy);
                    const float tanHalfFovY = sycl::tan(0.5f * fovYRad);
                    const float tanHalfFovX = tanHalfFovY * (width / height);

                    // film plane at z=1 has size: 2*tanHalfFovX by 2*tanHalfFovY
                    const float filmWidth = 2.0f * tanHalfFovX;
                    const float filmHeight = 2.0f * tanHalfFovY;

                    const float pixelArea = (filmWidth / width) * (filmHeight / height);
                    const float invPixelArea = 1.0f / pixelArea;
                    // Contribution (delta sensor, pixel binning)
                    const float3 contribution =
                            rayState.pathThroughput *
                            (cosineSurfaceToCamera * inverseDistanceSquared) * invPixelArea;

                    // Atomic accumulate to framebuffer
                    atomicAddFloat3ToImage(&sensor.framebuffer[pixelIndex], contribution);
                }
            );
        });
    }

    // ---- Kernel: Camera gather (one thread per pixel) --------------------------
    void launchCameraGatherKernel(RenderPackage &pkg, uint32_t cameraIndex) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto &photonMap = pkg.intermediates.map; // DeviceSurfacePhotonMapGrid


        // Host-side (before launching kernel)
        SensorGPU sensor = pkg.sensor[cameraIndex];

        // ReSharper disable once CppDFAUnreachableCode
        const std::uint32_t imageWidth = sensor.camera.width;
        const std::uint32_t imageHeight = sensor.camera.height;
        const std::uint32_t pixelCount = imageWidth * imageHeight;
        pkg.queue.fill(sensor.framebuffer, float4{0}, pixelCount).wait();

        for (int spp = 0; spp < settings.numGatherPasses; ++spp) {
            // Clear framebuffer before calling this, outside.
            queue.submit([&](sycl::handler &cgh) {
                float totalSamplesPerPixel = settings.numGatherPasses;
                uint64_t baseSeed = pkg.settings.randomSeed * random();

                cgh.parallel_for<class CameraGatherKernel>(
                    sycl::range<1>(pixelCount),
                    [=](sycl::id<1> tid) {
                        const std::uint32_t pixelIndex = tid[0];
                        const std::uint32_t pixelX = pixelIndex % imageWidth;
                        const std::uint32_t pixelY = pixelIndex / imageWidth;

                        rng::Xorshift128 randomNumberGenerator(
                            rng::makePerItemSeed1D(baseSeed, pixelIndex));

                        float3 accumulatedRadianceRGB(0.0f);

                        // Subpixel jitter
                        const float jitterX = randomNumberGenerator.nextFloat() - 0.5f;
                        const float jitterY = randomNumberGenerator.nextFloat() - 0.5f;

                        Ray primaryRay = makePrimaryRayFromPixelJitteredFov(
                            sensor.camera,
                            static_cast<float>(pixelX),
                            static_cast<float>(pixelY),
                            jitterX,
                            jitterY);

                        // -----------------------------------------------------------------
                        // 1) Transmit ray: collect all splat events + terminal mesh hit
                        // -----------------------------------------------------------------s
                        // Trace up to N layers (no sorting needed if each query returns closest hit > tMin)
                        const int maxLayers = 64;
                        float transmittanceProduct = 1.0f;
                        for (int layer = 0; layer < maxLayers; ++layer) {
                            WorldHit worldHit{};

                            intersectScene(primaryRay, &worldHit, scene, randomNumberGenerator,
                                           RayIntersectMode::Scatter);
                            if (!worldHit.hit) {
                                // No more surfels/meshes: add background/environment with remaining throughput
                                break;
                            }

                            buildIntersectionNormal(scene, worldHit);
                            auto &instance = scene.instances[worldHit.instanceIndex];

                            //if (pixelX == 200 && pixelY == 325) {

                            // -----------------------------------------------------------------
                            // 2) For each surfel along the transmit segment, fire a scatter ray
                            //    and shade that surfel from the photon map.
                            // -----------------------------------------------------------------
                            if (instance.geometryType == GeometryType::PointCloud) {
                                const Point &surfel = scene.points[worldHit.primitiveIndex];
                                const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));
                                const int travelSideSign = signNonZero(dot(canonicalNormalW, -primaryRay.direction));
                                const float3 frontNormalW = canonicalNormalW * float(travelSideSign);
                                const float3 rho = surfel.albedo;
                                const float3 E = gatherDiffuseIrradianceAtPoint(
                                    worldHit.hitPositionW,
                                    frontNormalW,
                                    photonMap,
                                    travelSideSign,
                                    true
                                );

                                float3 surfelShadedRadiance = E * (rho * M_1_PIf);

                                float alphaEff = surfel.opacity * worldHit.alpha;

                                accumulatedRadianceRGB += transmittanceProduct * surfelShadedRadiance;
                                transmittanceProduct *= (1.0f - alphaEff);

                                // Early out if we're nearly opqaue
                                if (transmittanceProduct < 0.001f) {
                                    break;
                                }

                                primaryRay.origin = worldHit.hitPositionW + (primaryRay.direction * 1e-6f);
                                continue;
                            }


                            // -----------------------------------------------------------------
                            // 3) Shade terminal mesh (if any) with remaining transmittance
                            // -----------------------------------------------------------------
                            if (instance.geometryType == GeometryType::Mesh) {
                                const GPUMaterial &material =
                                        scene.materials[instance.materialIndex];

                                if (material.isEmissive()) {
                                    const float distanceToCamera =
                                            length(worldHit.hitPositionW - primaryRay.origin);
                                    const float surfaceCosine =
                                            sycl::fmax(0.f, dot(worldHit.geometricNormalW,
                                                                -primaryRay.direction));
                                    const float cameraCosine =
                                            sycl::fmax(0.f, dot(sensor.camera.forward,
                                                                primaryRay.direction));

                                    const float geometricToCamera =
                                            (surfaceCosine * cameraCosine) /
                                            (distanceToCamera * distanceToCamera + 1e-8f);


                                    const float3 emittedRadiance = material.power * material.baseColor; // L_e

                                    accumulatedRadianceRGB += transmittanceProduct * emittedRadiance;
                                } else {
                                    const float3 rho = material.baseColor;

                                    const float3 E = gatherDiffuseIrradianceAtPoint(
                                        worldHit.hitPositionW, worldHit.geometricNormalW, photonMap);
                                    const float3 Lo = (rho * M_1_PIf) * E;

                                    accumulatedRadianceRGB += transmittanceProduct * Lo;
                                }
                                transmittanceProduct = 0.0f;
                                break;
                            }
                        }
                        // -----------------------------------------------------------------
                        // 4) Atomic accumulate into framebuffer
                        // -----------------------------------------------------------------
                        const std::uint32_t framebufferIndex =
                                pixelY * imageWidth + pixelX;

                        float4 previousValue = sensor.framebuffer[framebufferIndex];

                        float4 currentValue =
                                float4(accumulatedRadianceRGB.x(),
                                       accumulatedRadianceRGB.y(),
                                       accumulatedRadianceRGB.z(),
                                       1.0f) / totalSamplesPerPixel;

                        sensor.framebuffer[framebufferIndex] = currentValue + previousValue;
                    });
            });
            queue.wait();
        }
    }


    void computePhotonCellIdsAndPermutation(
        sycl::queue &queue,
        DeviceSurfacePhotonMapGrid grid,
        std::uint32_t photonCount) {
        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range<1>(photonCount), [=](sycl::id<1> idx) {
                const std::uint32_t photonIndex = static_cast<std::uint32_t>(idx[0]);
                const DevicePhotonSurface photon = grid.photons[photonIndex];

                grid.photonIndex[photonIndex] = photonIndex;

                if (photon.isValid == 0u) {
                    grid.photonCellId[photonIndex] = kInvalidIndex;
                    return;
                }

                const sycl::int3 cell = worldToCell(photon.position, grid);
                const std::uint32_t cellId = linearCellIndex(cell, grid.gridResolution);
                grid.photonCellId[photonIndex] = cellId;
            });
        }).wait();
    }

    void clearCellArrays(sycl::queue &queue, DeviceSurfacePhotonMapGrid grid) {
        static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;

        const std::uint32_t cellCount = grid.totalCellCount;
        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range<1>(cellCount), [=](sycl::id<1> idx) {
                const std::uint32_t c = static_cast<std::uint32_t>(idx[0]);
                grid.cellCount[c] = 0u;
                grid.cellWriteOffset[c] = 0u;
                grid.cellStart[c] = kInvalidIndex;
                grid.cellEnd[c] = kInvalidIndex;
            });
        }).wait();
    }


    void countPhotonsPerCell(
        sycl::queue &queue,
        DeviceSurfacePhotonMapGrid grid,
        std::uint32_t photonCount) {
        static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range<1>(photonCount), [=](sycl::id<1> idx) {
                const std::uint32_t i = static_cast<std::uint32_t>(idx[0]);
                const std::uint32_t cellId = grid.photonCellId[i];
                if (cellId == kInvalidIndex) return;

                auto atomicCount = sycl::atomic_ref<std::uint32_t,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(grid.cellCount[cellId]);

                atomicCount.fetch_add(1u);
            });
        }).wait();
    }


    void scatterPhotonsIntoCells(
        sycl::queue &queue,
        DeviceSurfacePhotonMapGrid grid,
        std::uint32_t photonCount) {
        static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range<1>(photonCount), [=](sycl::id<1> idx) {
                const std::uint32_t i = static_cast<std::uint32_t>(idx[0]);
                const std::uint32_t cellId = grid.photonCellId[i];
                if (cellId == kInvalidIndex) return;

                const std::uint32_t start = grid.cellStart[cellId];
                // start should be valid if count > 0
                if (start == kInvalidIndex) return;

                auto atomicOffset = sycl::atomic_ref<std::uint32_t,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(grid.cellWriteOffset[cellId]);

                const std::uint32_t localOffset = atomicOffset.fetch_add(1u);
                const uint32_t end = grid.cellEnd[cellId];
                const uint32_t writeIndex = start + localOffset;
                if (writeIndex < end) {
                    grid.sortedPhotonIndex[writeIndex] = i;
                }
            });
        });
    }

    static constexpr std::uint32_t kScanBlockSize = 1024;

    void exclusiveScanCellCountsToCellStart(
        sycl::queue &queue,
        DeviceSurfacePhotonMapGrid grid) {
        const std::uint32_t totalCellCount = grid.totalCellCount;
        const std::uint32_t blockSize = kScanBlockSize;
        const std::uint32_t blockCount = (totalCellCount + blockSize - 1u) / blockSize;

        std::uint32_t *cellCount = grid.cellCount;
        std::uint32_t *cellStart = grid.cellStart;
        std::uint32_t *blockSums = grid.blockSums;
        std::uint32_t *blockPrefix = grid.blockPrefix;

        // Pass 1: per-block exclusive scan into cellStart + write block sums
        queue.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<std::uint32_t, 1> localData(sycl::range<1>(blockSize), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(blockCount * blockSize), sycl::range<1>(blockSize)),
                [=](sycl::nd_item<1> item) {
                    const std::uint32_t localIndex = static_cast<std::uint32_t>(item.get_local_id(0));
                    const std::uint32_t blockIndex = static_cast<std::uint32_t>(item.get_group(0));
                    const std::uint32_t globalIndex = blockIndex * blockSize + localIndex;

                    // Load into local memory (out-of-range -> 0)
                    std::uint32_t value = 0u;
                    if (globalIndex < totalCellCount)
                        value = cellCount[globalIndex];

                    localData[localIndex] = value;
                    item.barrier(sycl::access::fence_space::local_space);

                    // Blelloch upsweep
                    for (std::uint32_t offset = 1u; offset < blockSize; offset <<= 1u) {
                        const std::uint32_t index = (localIndex + 1u) * offset * 2u - 1u;
                        if (index < blockSize)
                            localData[index] += localData[index - offset];

                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    // Write total sum for this block, then set root to 0 for exclusive scan
                    if (localIndex == blockSize - 1u) {
                        blockSums[blockIndex] = localData[localIndex];
                        localData[localIndex] = 0u;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    // Blelloch downsweep
                    for (std::uint32_t offset = blockSize >> 1u; offset > 0u; offset >>= 1u) {
                        const std::uint32_t index = (localIndex + 1u) * offset * 2u - 1u;
                        if (index < blockSize) {
                            const std::uint32_t left = localData[index - offset];
                            const std::uint32_t right = localData[index];
                            localData[index - offset] = right;
                            localData[index] = right + left;
                        }

                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    // Store per-element exclusive prefix into cellStart
                    if (globalIndex < totalCellCount)
                        cellStart[globalIndex] = localData[localIndex];
                });
        }).wait();

        // Pass 2: exclusive scan of block sums on CPU (blockCount is small)
        std::vector<std::uint32_t> blockSumsHost(blockCount);
        queue.memcpy(blockSumsHost.data(), blockSums, sizeof(std::uint32_t) * blockCount).wait();

        std::vector<std::uint32_t> blockPrefixHost(blockCount);
        std::uint32_t runningSum = 0u;
        for (std::uint32_t b = 0; b < blockCount; ++b) {
            blockPrefixHost[b] = runningSum;
            runningSum += blockSumsHost[b];
        }

        queue.memcpy(blockPrefix, blockPrefixHost.data(), sizeof(std::uint32_t) * blockCount).wait();

        // Pass 3: add block prefix to each element’s local prefix
        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range<1>(totalCellCount), [=](sycl::id<1> idx) {
                const std::uint32_t globalIndex = static_cast<std::uint32_t>(idx[0]);
                const std::uint32_t blockIndex = globalIndex / blockSize;
                cellStart[globalIndex] += blockPrefix[blockIndex];
            });
        }).wait();
    }

    void finalizeCellRanges(
        sycl::queue &queue,
        DeviceSurfacePhotonMapGrid grid) {
        static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;

        const std::uint32_t totalCellCount = grid.totalCellCount;

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range<1>(totalCellCount), [=](sycl::id<1> idx) {
                const std::uint32_t c = static_cast<std::uint32_t>(idx[0]);
                const std::uint32_t count = grid.cellCount[c];

                grid.cellWriteOffset[c] = 0u;

                if (count == 0u) {
                    grid.cellStart[c] = kInvalidIndex;
                    grid.cellEnd[c] = kInvalidIndex;
                } else {
                    const std::uint32_t start = grid.cellStart[c];
                    grid.cellEnd[c] = start + count;
                }
            });
        }).wait();
    }


    void buildPhotonCellRangesAndOrdering(
        sycl::queue &queue,
        DeviceSurfacePhotonMapGrid grid,
        std::uint32_t photonCount) {
        clearCellArrays(queue, grid); // counts/start/end/offset=0/invalid
        computePhotonCellIdsAndPermutation(queue, grid, photonCount); // keys (optional now)
        countPhotonsPerCell(queue, grid, photonCount); // histogram

        exclusiveScanCellCountsToCellStart(queue, grid); // cellStart from cellCount (implement)
        finalizeCellRanges(queue, grid); // cellEnd = start + count, invalid if count==0, offset=0
        scatterPhotonsIntoCells(queue, grid, photonCount); // sortedPhotonIndex
    }
} // Pale
