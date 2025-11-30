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
        const float totalPhotons = photonCount * forwardPasses;

        queue.submit([&](sycl::handler& commandGroupHandler) {
            uint64_t baseSeed = settings.randomSeed;

            commandGroupHandler.parallel_for<struct RayGenEmitterKernelTag>(
                sycl::range<1>(photonCount),
                [=](sycl::id<1> globalId) {
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, globalId[0]);
                    // Choose any generator you like:
                    rng::Xorshift128 rng128(perItemSeed);

                    if (scene.lightCount == 0) return;

                    AreaLightSample ls = sampleMeshAreaLightReuse(scene, rng128);
                    if (!ls.valid) return;

                    // Cosine-hemisphere about emitter normal
                    float cosTheta = 0.0f;
                    float3 sampledDirection;
                    sampleCosineHemisphere(rng128, ls.normalW, sampledDirection, cosTheta);
                    if (cosTheta <= 0.0f) return;

                    // PDFs
                    const float pdfDir = cosTheta * (1.0f / 3.1415926535f); // cosine hemisphere
                    const float pdfPos = ls.pdfArea; // area-domain, world area
                    const float pdfLight = ls.pdfSelectLight; // light selection
                    const float pdfTotal = pdfLight * pdfPos * pdfDir;
                    if (pdfTotal <= 0.0f) return;

                    // Initial throughput (power-conserving)
                    const float3 Le = ls.emittedRadianceRGB; // radiance
                    const float invPdf = 1.0f / pdfTotal;
                    const float3 initialThroughput = Le * (cosTheta * invPdf) / totalPhotons;

                    // Write ray
                    RayState ray{};
                    ray.ray.origin = ls.positionW;
                    ray.ray.direction = sampledDirection;
                    ray.pathThroughput = initialThroughput;
                    ray.bounceIndex = 0u;

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
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& settings = pkg.settings;

        auto& intermediates = pkg.intermediates;
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
                    const WorldHit worldHit = hitRecords[rayIndex];
                    const RayState rayState = raysIn[rayIndex];
                    if (!worldHit.hit) return;

                    const InstanceRecord instance = scene.instances[worldHit.instanceIndex];
                    const float3 canonicalNormalW = worldHit.geometricNormalW;
                    const int travelSideSign = signNonZero(dot(canonicalNormalW, -rayState.ray.direction));
                    const float3 enteredSideNormalW = (travelSideSign >= 0) ? canonicalNormalW : (-canonicalNormalW);
                    float3 sampledOutgoingDirectionW{};
                    float3 throughputMultiplier = float3(1.0f);

                    // If we hit instance was a mesh do ordinary BRDF stuff.
                    if (instance.geometryType == GeometryType::Mesh) {
                        float sampledPdf = 0.0f;
                        const GPUMaterial material = scene.materials[instance.materialIndex];
                        sampleCosineHemisphere(rng128, enteredSideNormalW, sampledOutgoingDirectionW, sampledPdf);
                        sampledPdf = sycl::fmax(sampledPdf, 1e-6f);
                        const float cosTheta = sycl::fmax(0.0f, dot(sampledOutgoingDirectionW, enteredSideNormalW));
                        const float3 lambertBrdf = material.baseColor * M_1_PIf;
                        throughputMultiplier = lambertBrdf * (cosTheta / sampledPdf) * worldHit.transmissivity;
                    }
                    // If our hit instance was a point cloud it means we hit a surfel
                    // Now we do either BRDF or BTDF
                    if (instance.geometryType == GeometryType::PointCloud) {
                        const float alpha = worldHit.splatEvents[0].alpha; // opacity at (u,v)
                        if (rng128.nextFloat() >= alpha) {
                            // NULL TRANSMIT: keep direction, no weight change
                            sampledOutgoingDirectionW = rayState.ray.direction;
                            // throughputMultiplier stays 1
                        }
                        else {
                            // SURFACE EVENT
                            const float3 enteredSideNormalW = (signNonZero(
                                                                  dot(worldHit.geometricNormalW,
                                                                      -rayState.ray.direction)) >= 0)
                                                                  ? worldHit.geometricNormalW
                                                                  : (-worldHit.geometricNormalW);

                            // 50/50 reflect vs transmit for a symmetric diffuse sheet
                            float probReflect = 1.0f;
                            const bool chooseReflect = (rng128.nextFloat() < probReflect);

                            float sampledPdf = 0.0f;
                            float cosTheta = 0.0f;
                            float3 eventNormalW = chooseReflect ? enteredSideNormalW : -enteredSideNormalW;

                            sampleCosineHemisphere(rng128, eventNormalW, sampledOutgoingDirectionW, sampledPdf);
                            sampledPdf = sycl::fmax(sampledPdf, 1e-6f);
                            cosTheta = sycl::fmax(0.0f, dot(sampledOutgoingDirectionW, eventNormalW));

                            // Single albedo factor through BRDF
                            const float3 baseColor = scene.points[worldHit.primitiveIndex].color;
                            const float3 lambertBrdf = baseColor * M_1_PIf;
                            throughputMultiplier = lambertBrdf * (cosTheta / sampledPdf);
                        }
                    }

                    if (settings.rayGenMode == RayGenMode::Emitter) {
                        auto& devicePtr = *intermediates.map.photonCountDevicePtr;
                        sycl::atomic_ref<uint32_t,
                                         sycl::memory_order::acq_rel,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            photonCounter(devicePtr);


                        // Also deposit splatEvents
                        for (const auto& event : worldHit.splatEvents) {
                            if (event.t >= worldHit.t)
                                continue;

                            const uint32_t slot = photonCounter.fetch_add(1u);
                            if (slot < intermediates.map.photonCapacity) {
                                DevicePhotonSurface photonEntry{};
                                photonEntry.position = event.hitWorld;
                                photonEntry.power = rayState.pathThroughput;
                                const auto& surfel = scene.points[event.primitiveIndex];
                                const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));

                                const float signedCosineIncident = dot(canonicalNormalW, -rayState.ray.direction);
                                photonEntry.cosineIncident = sycl::fabs(signedCosineIncident);
                                photonEntry.sideSign = signNonZero(signedCosineIncident);
                                photonEntry.geometryType = instance.geometryType;
                                photonEntry.primitiveIndex = event.primitiveIndex;
                                intermediates.map.photons[slot] = photonEntry;
                            }
                        }
                        const uint32_t slot = photonCounter.fetch_add(1u);
                        if (slot < intermediates.map.photonCapacity) {
                            DevicePhotonSurface photonEntry{};
                            photonEntry.position = worldHit.hitPositionW;
                            photonEntry.power = rayState.pathThroughput;

                            const float signedCosineIncident = dot(worldHit.geometricNormalW, -rayState.ray.direction);
                            photonEntry.cosineIncident = sycl::fabs(signedCosineIncident);
                            photonEntry.sideSign = signNonZero(signedCosineIncident);
                            photonEntry.geometryType = instance.geometryType;
                            photonEntry.primitiveIndex = instance.geometryType == GeometryType::Mesh
                                                             ? worldHit.instanceIndex
                                                             : worldHit.primitiveIndex;
                            intermediates.map.photons[slot] = photonEntry;
                        }
                    }

                    // --- Spawn next ray with offset along *oriented* normal ---
                    RayState nextState{};
                    // Spawn next ray
                    nextState.ray.origin = worldHit.hitPositionW + enteredSideNormalW * 1e-7f;
                    nextState.ray.direction = sampledOutgoingDirectionW;
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

    // ---- Kernel: Camera gather (one thread per pixel) --------------------------
    void launchCameraGatherKernel(RenderPackage& pkg, int totalSamplesPerPixel) {
        auto& queue = pkg.queue;
        auto& scene = pkg.scene;
        auto& settings = pkg.settings;
        auto& photonMap = pkg.intermediates.map; // DeviceSurfacePhotonMapGrid


        queue.wait();
        for (size_t cameraIndex = 0; cameraIndex < pkg.numSensors; ++cameraIndex) {
            // Host-side (before launching kernel)
            SensorGPU sensor = pkg.sensor[cameraIndex];

            const std::uint32_t imageWidth = sensor.camera.width;
            const std::uint32_t imageHeight = sensor.camera.height;
            const std::uint32_t pixelCount = imageWidth * imageHeight;
            pkg.queue.fill(sensor.framebuffer, float4{0}, pixelCount).wait();

            // Clear framebuffer before calling this, outside.
            queue.submit([&](sycl::handler& cgh) {
                uint64_t baseSeed = pkg.settings.randomSeed;
                auto samplesPerPixel = static_cast<float>(totalSamplesPerPixel);

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

                        Ray primaryRay = makePrimaryRayFromPixelJittered(
                            sensor.camera,
                            static_cast<float>(pixelX),
                            static_cast<float>(pixelY),
                            jitterX,
                            jitterY);


                        // -----------------------------------------------------------------
                        // 1) Transmit ray: collect all splat events + terminal mesh hit
                        // -----------------------------------------------------------------
                        WorldHit transmitWorldHit{};
                        intersectScene(primaryRay,
                                       &transmitWorldHit,
                                       scene,
                                       randomNumberGenerator,
                                       RayIntersectMode::Transmit);

                        if (!transmitWorldHit.hit) {
                            // No geometry / environment handled elsewhere
                            return;
                        }

                        // Compute geometric normal for terminal hit
                        const InstanceRecord& terminalInstance =
                            scene.instances[transmitWorldHit.instanceIndex];

                        switch (terminalInstance.geometryType) {
                        case GeometryType::Mesh:
                            {
                                const Triangle& triangle =
                                    scene.triangles[transmitWorldHit.primitiveIndex];
                                const Transform& objectToWorldTransform =
                                    scene.transforms[terminalInstance.transformIndex];

                                const Vertex& vertex0 = scene.vertices[triangle.v0];
                                const Vertex& vertex1 = scene.vertices[triangle.v1];
                                const Vertex& vertex2 = scene.vertices[triangle.v2];

                                const float3 worldP0 =
                                    toWorldPoint(vertex0.pos, objectToWorldTransform);
                                const float3 worldP1 =
                                    toWorldPoint(vertex1.pos, objectToWorldTransform);
                                const float3 worldP2 =
                                    toWorldPoint(vertex2.pos, objectToWorldTransform);

                                transmitWorldHit.geometricNormalW =
                                    normalize(cross(worldP1 - worldP0, worldP2 - worldP0));
                                break;
                            }
                        case GeometryType::PointCloud:
                            {
                                const Point& surfel =
                                    scene.points[transmitWorldHit.primitiveIndex];
                                const float3 canonicalNormalWorld =
                                    normalize(cross(surfel.tanU, surfel.tanV));
                                transmitWorldHit.geometricNormalW = canonicalNormalWorld;
                                break;
                            }
                        }

                        // -----------------------------------------------------------------
                        // 2) For each surfel along the transmit segment, fire a scatter ray
                        //    and shade that surfel from the photon map.
                        // -----------------------------------------------------------------
                        float transmittanceTau = 1.0f;
                        const std::uint32_t numberOfSurfelsOnRay =
                            static_cast<std::uint32_t>(transmitWorldHit.splatEventCount);

                        for (std::uint32_t surfelEventIndex = 0;
                             surfelEventIndex < numberOfSurfelsOnRay;
                             ++surfelEventIndex) {
                            const SplatEvent& transmitSplatEvent =
                                transmitWorldHit.splatEvents[surfelEventIndex];

                            const std::uint32_t scatterOnPrimitiveIndex =
                                transmitSplatEvent.primitiveIndex;

                            // Fire a scatter ray that treats this surfel as the scatter event
                            WorldHit scatterWorldHit{};
                            intersectScene(primaryRay,
                                           &scatterWorldHit,
                                           scene,
                                           randomNumberGenerator,
                                           RayIntersectMode::Scatter,
                                           scatterOnPrimitiveIndex);

                            if (!scatterWorldHit.hit) {
                                // Should not normally happen; skip this event
                                continue;
                            }

                            // The scatter call should have a terminal splat event for the surfel
                            if (scatterWorldHit.splatEventCount == 0) {
                                continue;
                            }

                            const std::uint32_t terminalEventIndex =
                                static_cast<std::uint32_t>(scatterWorldHit.splatEventCount - 1);
                            const SplatEvent& terminalSplatEvent =
                                scatterWorldHit.splatEvents[terminalEventIndex];

                            // Optional safety: ensure we are scattering on the same primitive
                            if (terminalSplatEvent.primitiveIndex != scatterOnPrimitiveIndex) {
                                // Inconsistent record; skip to be safe
                                continue;
                            }

                            // Shade surfel from photon map (front/back)
                            const bool useOneSidedScatter = true;
                            float3 surfelRadianceFront =
                                estimateSurfelRadianceFromPhotonMap(
                                    terminalSplatEvent,
                                    primaryRay.direction,
                                    scene,
                                    photonMap,
                                    false);

                            float3 surfelRadianceBack =
                                estimateSurfelRadianceFromPhotonMap(
                                    terminalSplatEvent,
                                    -primaryRay.direction,
                                    scene,
                                    photonMap,
                                    useOneSidedScatter);

                            float3 surfelShadedRadiance = surfelRadianceFront;


                            const Point& surfel = scene.points[terminalSplatEvent.primitiveIndex];

                            const float alphaGeom = terminalSplatEvent.alpha; // α(u,v)
                            const float eta = surfel.opacity; // eta
                            const float surfelOpacity = eta * alphaGeom; // α_eff = eta α

                            const float oneMinusTotalOpacity = 1.0f - surfelOpacity;

                            accumulatedRadianceRGB +=
                                transmittanceTau * surfelShadedRadiance * surfelOpacity;

                            transmittanceTau *= oneMinusTotalOpacity;
                            if (transmittanceTau <= 1e-4f) {
                                // Almost fully opaque, no need to continue
                                break;
                            }
                        }

                        // -----------------------------------------------------------------
                        // 3) Shade terminal mesh (if any) with remaining transmittance
                        // -----------------------------------------------------------------
                        if (transmittanceTau > 1e-4f &&
                            terminalInstance.geometryType == GeometryType::Mesh) {
                            const GPUMaterial& material =
                                scene.materials[terminalInstance.materialIndex];

                            if (material.isEmissive()) {
                                const float distanceToCamera =
                                    length(transmitWorldHit.hitPositionW - primaryRay.origin);
                                const float surfaceCosine =
                                    sycl::fmax(0.f, dot(transmitWorldHit.geometricNormalW,
                                                        -primaryRay.direction));
                                const float cameraCosine =
                                    sycl::fmax(0.f, dot(sensor.camera.forward,
                                                        primaryRay.direction));

                                const float geometricToCamera =
                                    (surfaceCosine * cameraCosine) /
                                    (distanceToCamera * distanceToCamera + 1e-8f);

                                const float3 emittedRadiance =
                                    material.emissive * transmittanceTau * geometricToCamera;

                                const float3 reflectedRadiance =
                                    estimateRadianceFromPhotonMap(
                                        transmitWorldHit, scene, photonMap) * transmittanceTau;

                                accumulatedRadianceRGB += (emittedRadiance + reflectedRadiance);
                            }
                            else {
                                const float3 meshRadiance =
                                    estimateRadianceFromPhotonMap(
                                        transmitWorldHit, scene, photonMap) * transmittanceTau;
                                accumulatedRadianceRGB += meshRadiance;
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
                                   1.0f) / samplesPerPixel;

                        sensor.framebuffer[framebufferIndex] = previousValue + currentValue;
                    });
            }).wait();
        }
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
