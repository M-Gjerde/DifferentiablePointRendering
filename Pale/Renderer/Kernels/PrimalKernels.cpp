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

            commandGroupHandler.parallel_for<struct RayGenEmitterKernelTag>(
                sycl::range<1>(photonCount),
                [=](sycl::id<1> globalId) {
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(baseSeed, globalId[0]);
                    // Choose any generator you like:
                    rng::Xorshift128 rng128(perItemSeed);

                    if (scene.lightCount == 0) return;

                    AreaLightSample ls = sampleMeshAreaLightReuse(scene, rng128);
                    if (!ls.valid) return;

                    float pdfDir = 0.0f;
                    float3 sampledDirectionW;
                    sampleCosineHemisphere(rng128, ls.normalW, sampledDirectionW, pdfDir);
                    if (pdfDir <= 0.0f) return;

                    const float cosTheta = sycl::fmax(0.0f, dot(sampledDirectionW, ls.normalW));
                    if (cosTheta <= 0.0f) return;

                    const float pdfPos = ls.pdfArea; // 1/A
                    const float pdfLight = ls.pdfSelectLight; // 1/lightCount
                    const float pdfTotal = pdfLight * pdfPos * pdfDir;

                    // Correct: include cosTheta from dPhi = Le cos dA dω
                    const float3 initialThroughput =
                            ls.Le * (cosTheta / pdfTotal) / totalPhotons;

                    RayState ray{};
                    ray.ray.origin = ls.positionW + ls.normalW * 1e-5f;
                    ray.ray.direction = sampledDirectionW;
                    ray.pathThroughput = initialThroughput;
                    ray.bounceIndex = 0;

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
                    const float3 canonicalNormalW = worldHit.geometricNormalW;
                    const int travelSideSign = signNonZero(dot(canonicalNormalW, -rayState.ray.direction));
                    const float3 enteredSideNormalW = (travelSideSign >= 0) ? canonicalNormalW : (-canonicalNormalW);
                    float3 sampledOutgoingDirectionW = rayState.ray.direction;
                    float3 throughputMultiplier = float3(1.0f);

                    bool chooseReflect = false;
                    // If we hit instance was a mesh do ordinary BRDF stuff.
                    if (instance.geometryType == GeometryType::Mesh) {
                        float sampledPdf = 0.0f;
                        const GPUMaterial material = scene.materials[instance.materialIndex];
                        sampleCosineHemisphere(rng128, enteredSideNormalW, sampledOutgoingDirectionW, sampledPdf);
                        sampledPdf = sycl::fmax(sampledPdf, 1e-6f);
                        const float cosTheta = sycl::fmax(0.0f, dot(sampledOutgoingDirectionW, enteredSideNormalW));
                        const float3 lambertBrdf = material.baseColor * M_1_PIf;
                        throughputMultiplier = lambertBrdf * (cosTheta / sampledPdf) * worldHit.transmissivity;
                        chooseReflect = true;
                    }
                    // If our hit instance was a point cloud it means we hit a surfel
                    // Now we do either BRDF or BTDF
                    if (instance.geometryType == GeometryType::PointCloud) {
                        // SURFACE EVENT
                        // 50/50 reflect vs transmit for a symmetric diffuse sheet
                        float probReflect = 0.5f;
                        chooseReflect = (rng128.nextFloat() < probReflect);

                        float sampledPdf = 0.0f;
                        float cosTheta = 0.0f;
                        float3 eventNormalW = chooseReflect ? enteredSideNormalW : -enteredSideNormalW;

                        sampleCosineHemisphere(rng128, eventNormalW, sampledOutgoingDirectionW, sampledPdf);
                        sampledPdf = sycl::fmax(sampledPdf, 1e-6f);
                        cosTheta = sycl::fmax(0.0f, dot(sampledOutgoingDirectionW, eventNormalW));

                        auto &surfel = scene.points[worldHit.primitiveIndex];
                        float alpha = worldHit.splatEvents[worldHit.splatEventCount - 1].alpha;
                        // Single albedo factor through BRDF
                        const float3 baseColor = surfel.albedo;
                        // Mixture pdf:
                        float branchProb = chooseReflect ? probReflect : (1.0f - probReflect);
                        float pdfMixture = branchProb * sampledPdf;
                        pdfMixture = sycl::fmax(pdfMixture, 1e-6f);

                        const float3 lambertBrdf = baseColor * M_1_PIf; // ρ/π
                        throughputMultiplier = lambertBrdf * (cosTheta / sampledPdf);
                    }

                    if (settings.rayGenMode == RayGenMode::Emitter) {
                        auto &devicePtr = *intermediates.map.photonCountDevicePtr;
                        sycl::atomic_ref<uint32_t,
                                    sycl::memory_order::acq_rel,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                photonCounter(devicePtr);

                        // Also deposit splatEvents

                        for (const auto &event: worldHit.splatEvents) {
                            if (event.t >= worldHit.t)
                                continue;

                            const uint32_t slot = photonCounter.fetch_add(1u);
                            if (slot < intermediates.map.photonCapacity) {
                                DevicePhotonSurface photonEntry{};
                                photonEntry.position = event.hitWorld;
                                const auto &surfel = scene.points[event.primitiveIndex];
                                const float alphaGeom = event.alpha;
                                const float eta = surfel.opacity;
                                const float alphaEff = alphaGeom * eta;

                                const float depositWeight = event.tau;
                                photonEntry.power = rayState.pathThroughput * depositWeight;

                                const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));
                                const float signedCosineIncident = dot(canonicalNormalW, -rayState.ray.direction);
                                const int sideSign = signNonZero(signedCosineIncident);
                                const float3 orientedNormalW = (sideSign >= 0) ? canonicalNormalW : (-canonicalNormalW);

                                photonEntry.cosineIncident = sycl::fabs(signedCosineIncident);
                                photonEntry.sideSign = signNonZero(signedCosineIncident);
                                photonEntry.geometryType = GeometryType::PointCloud;
                                photonEntry.primitiveIndex = event.primitiveIndex;
                                photonEntry.isValid = 1u;
                                photonEntry.normalW = orientedNormalW;

                                intermediates.map.photons[slot] = photonEntry;
                            }
                        }
                        const uint32_t slot = photonCounter.fetch_add(1u);

                        if (slot < intermediates.map.photonCapacity) {
                            DevicePhotonSurface photonEntry{};
                            photonEntry.position = worldHit.hitPositionW;


                            float3 baseNormalW;
                            uint32_t primitiveIndexForDeposit = 0;
                            float depositWeight = 1.0f;

                            if (instance.geometryType == GeometryType::Mesh) {
                                baseNormalW = normalize(worldHit.geometricNormalW);
                                primitiveIndexForDeposit = worldHit.instanceIndex;
                                depositWeight = worldHit.transmissivity;
                            } else {
                                const Point &surfel = scene.points[worldHit.primitiveIndex];
                                baseNormalW = normalize(cross(surfel.tanU, surfel.tanV));
                                primitiveIndexForDeposit = worldHit.primitiveIndex;
                                const float alphaGeomHit = worldHit.splatEvents[worldHit.splatEventCount - 1].alpha;
                                const float etaHit = scene.points[worldHit.primitiveIndex].opacity;
                                const float alphaEffHit = alphaGeomHit * etaHit;
                                depositWeight = worldHit.transmissivity;
                            }

                            const float signedCosineIncident = dot(baseNormalW, -rayState.ray.direction);
                            const int sideSign = signNonZero(signedCosineIncident);
                            const float3 orientedNormalW = (sideSign >= 0) ? baseNormalW : (-baseNormalW);

                            photonEntry.power = rayState.pathThroughput * depositWeight;

                            photonEntry.normalW = orientedNormalW;
                            photonEntry.sideSign = sideSign;
                            photonEntry.cosineIncident = sycl::fabs(signedCosineIncident);
                            photonEntry.geometryType = instance.geometryType;
                            photonEntry.primitiveIndex = primitiveIndexForDeposit;

                            photonEntry.isValid = 1u;
                            intermediates.map.photons[slot] = photonEntry;
                        }
                    }

                    // --- Spawn next ray with offset along *oriented* normal ---
                    RayState nextState{};
                    // Spawn next ray
                    nextState.ray.origin = worldHit.hitPositionW + (enteredSideNormalW) * 1e-5f;
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

    // ---- Kernel: Camera gather (one thread per pixel) --------------------------
    void launchCameraGatherKernel(RenderPackage &pkg, int totalSamplesPerPixel, uint32_t cameraIndex) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &settings = pkg.settings;
        auto &photonMap = pkg.intermediates.map; // DeviceSurfacePhotonMapGrid
        uint64_t baseSeed = pkg.settings.randomSeed * (cameraIndex + 5);


        // Host-side (before launching kernel)
        SensorGPU sensor = pkg.sensor[cameraIndex];

        // ReSharper disable once CppDFAUnreachableCode
        const std::uint32_t imageWidth = sensor.camera.width;
        const std::uint32_t imageHeight = sensor.camera.height;
        const std::uint32_t pixelCount = imageWidth * imageHeight;
        pkg.queue.fill(sensor.framebuffer, float4{0}, pixelCount).wait();

        // Clear framebuffer before calling this, outside.
        queue.submit([&](sycl::handler &cgh) {
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
                    int pixelYFlipped = imageHeight - 1 - pixelY;
                    bool isWatched = false;

                    if (pixelX == 240 && pixelYFlipped == 250) {
                        isWatched = true;
                        int debug = 1;
                    }
                    //if (pixelX == 200 && pixelYFlipped == 325) {
                    if (pixelX == 260 && pixelYFlipped == 250) {
                        isWatched = true;
                        int debug = 1;
                    }

                    if (!isWatched) {
                        //return;
                    }

                    // -----------------------------------------------------------------
                    // 1) Transmit ray: collect all splat events + terminal mesh hit
                    // -----------------------------------------------------------------
                    WorldHit transmitWorldHit{};
                    intersectScene(primaryRay,
                                   &transmitWorldHit,
                                   scene,
                                   randomNumberGenerator,
                                   RayIntersectMode::Transmit);

                    if (!transmitWorldHit.hit && !transmitWorldHit.splatEventCount) {
                        // No geometry / environment handled elsewhere
                        return;
                    }

                    buildIntersectionNormal(scene, transmitWorldHit);

                    //if (pixelX == 200 && pixelY == 325) {

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
                        const SplatEvent &terminalSplatEvent =
                                transmitWorldHit.splatEvents[surfelEventIndex];
                        const Point &surfel = scene.points[terminalSplatEvent.primitiveIndex];


                        const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));
                        const int travelSideSign = signNonZero(dot(canonicalNormalW, -primaryRay.direction));

                        const float3 frontNormalW = canonicalNormalW * float(travelSideSign);
                        const float3 backNormalW = -frontNormalW;

                        const float3 rho = surfel.albedo;

                        const float3 E = gatherDiffuseIrradianceAtPointNormalFiltered(
                            terminalSplatEvent.hitWorld,
                            frontNormalW,
                            photonMap,
                            travelSideSign,
                            true
                        );

                        float3 surfelShadedRadiance = E * (rho * M_1_PIf);


                        const float alphaGeom = terminalSplatEvent.alpha; // α(u,v)
                        const float eta = surfel.opacity; // eta
                        const float surfelOpacity = eta * alphaGeom; // α_eff = eta α

                        const float oneMinusTotalOpacity = 1.0f - surfelOpacity;

                        accumulatedRadianceRGB += transmittanceTau * surfelOpacity * surfelShadedRadiance;


                        transmittanceTau *= oneMinusTotalOpacity;


                        if (isWatched)
                            int debug = 1;

                        if (transmittanceTau <= 1e-4f) {
                            // Almost fully opaque, no need to continue
                            break;
                        }
                    }

                    // -----------------------------------------------------------------
                    // 3) Shade terminal mesh (if any) with remaining transmittance
                    // -----------------------------------------------------------------
                    if (transmitWorldHit.instanceIndex != UINT32_MAX) {
                        auto &terminalInstance = scene.instances[transmitWorldHit.instanceIndex];

                        const GPUMaterial &material =
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
                                    material.power * material.baseColor * transmittanceTau * geometricToCamera;

                            accumulatedRadianceRGB += emittedRadiance;
                        } else {
                        }

                        const float3 rho = material.baseColor;

                        const float3 E = gatherDiffuseIrradianceAtPointNormalFiltered(
                            transmitWorldHit.hitPositionW, transmitWorldHit.geometricNormalW, photonMap);
                        const float3 Lo = (rho * M_1_PIf) * E;

                        accumulatedRadianceRGB += transmittanceTau * Lo;
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
                                   1.0f);

                    sensor.framebuffer[framebufferIndex] = previousValue + currentValue;
                });
        });
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
