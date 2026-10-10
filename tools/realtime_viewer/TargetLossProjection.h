#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "Renderer/Kernels/IntersectionKernels.h"

namespace viewer {
    struct LossInstanceRange {
        std::size_t offset = 0;
        uint32_t firstPoint = 0;
    };
    inline bool crowdBlocks(float meanMembers, float threshold) {
        return threshold > 0.0f && std::isfinite(meanMembers) && meanMembers >= threshold;
    }

    inline std::vector<float> rgbHalfSquaredError(
            const std::vector<float>& rendered, const std::vector<float>& target) {
        if (rendered.size() != target.size() || rendered.size() % 4 != 0)
            throw std::runtime_error("Target and render RGBA dimensions do not match");
        std::vector<float> loss(rendered.size() / 4);
        for (std::size_t p = 0; p < loss.size(); ++p) {
            double squared = 0;
            for (std::size_t c = 0; c < 3; ++c) {
                const double difference = double(rendered[4*p+c]) - target[4*p+c];
                squared += difference * difference;
            }
            loss[p] = static_cast<float>(squared / 6.0);
            if (!std::isfinite(loss[p])) throw std::runtime_error("Non-finite target/render RGB loss");
        }
        return loss;
    }

    // Same geometric slab and opacity rules as camera rendering. Scores are
    // indexed by placed instance, then local point, so instanced geometry does
    // not accidentally share a diagnostic color.
    template<class Add>
    inline void forEachLossContributor(Pale::Ray ray, const Pale::GPUSceneBuffers& scene,
            const Pale::PathTracerSettings& settings, const LossInstanceRange* ranges, Add add) {
        float transmission = 1.0f;
        const auto applyLayer = [&](const Pale::PointCloudLocalLayer& layer, uint32_t instanceIndex) {
            const auto& range = ranges[instanceIndex];
            for (uint32_t member = 0; member < layer.hitCount; ++member) {
                const auto primitive = layer.hits[member].primitiveIndex;
                const float weight = transmission * layer.weight[member];
                if (weight > 0 && !scene.points[primitive].isEmissive())
                    add(range.offset + primitive - range.firstPoint, weight);
            }
            transmission *= layer.transmission;
        };
        uint32_t singleInstance = UINT32_MAX;
        const auto batchSize = Pale::rendererDebugPointHitBatchSize(settings);
        if (batchSize > 1u && Pale::tryGetSinglePointCloudInstance(scene, singleInstance)) {
            const auto capacity = Pale::rendererDebugPointHitBatchLookaheadCapacity(settings);
            for (uint32_t event = 0; event < Pale::rendererDebugMaxSplatEventsPerRay(settings);) {
                Pale::LocalSurfelLayerHit hits[Pale::kMaxPointHitBatchWithLookahead];
                uint32_t instanceIndex = UINT32_MAX;
                const auto hitCount = Pale::collectScenePointHitsDirect(ray, scene, Pale::RayEpsilon,
                    std::numeric_limits<float>::infinity(), hits, capacity, instanceIndex);
                if (!hitCount) break;
                const auto coreCount = sycl::min(hitCount, batchSize);
                uint32_t cursor = 0;
                float consumed = 0;
                while (cursor < coreCount && event < Pale::rendererDebugMaxSplatEventsPerRay(settings)) {
                    const auto previous = cursor;
                    const auto layer = Pale::buildPointCloudLocalLayerFromHits(ray, hits[cursor],
                        hits + cursor, hitCount - cursor, scene,
                        Pale::rendererDebugLocalLayerDepthEpsilon(settings),
                        Pale::rendererDebugMaxLocalSurfelHits(settings),
                        Pale::rendererDebugLocalLayerNormalCosineThreshold(settings),
                        settings.rendererDebugLocalLayerDepthMode);
                    applyLayer(layer, instanceIndex);
                    ++event;
                    consumed = sycl::fmax(consumed, layer.furthestT);
                    while (cursor < hitCount && hits[cursor].tWorld <= layer.furthestT + Pale::RayEpsilon) ++cursor;
                    if (cursor == previous) ++cursor;
                    if (transmission <= 1.0e-8f) break;
                }
                if (consumed <= 0) break;
                ray.origin += ray.direction * (consumed + Pale::RayEpsilon);
                if (transmission <= 1.0e-8f || (cursor >= hitCount && hitCount < capacity)) break;
            }
            return;
        }
        for (uint32_t event = 0; event < Pale::rendererDebugMaxSplatEventsPerRay(settings); ++event) {
            Pale::WorldHit hit{};
            Pale::intersectScene(ray, &hit, scene, Pale::SurfelIntersectMode::FirstHit);
            if (!hit.hit) break;
            const auto& instance = scene.instances[hit.instanceIndex];
            if (instance.geometryType != Pale::GeometryType::PointCloud) break;
            Pale::buildIntersectionNormal(scene, hit);
            const auto layer = Pale::collectPointCloudLocalLayer(ray, hit, instance, scene,
                Pale::rendererDebugLocalLayerDepthEpsilon(settings),
                Pale::rendererDebugMaxLocalSurfelHits(settings),
                Pale::rendererDebugLocalLayerNormalCosineThreshold(settings),
                settings.rendererDebugLocalLayerDepthMode);
            if (layer.hitCount == 0) break;
            applyLayer(layer, hit.instanceIndex);
            if (transmission <= 1.0e-8f) break;
            ray.origin += ray.direction * (layer.furthestT + Pale::RayEpsilon);
        }
    }

    struct ProjectedTargetLoss {
        std::string cameraName;
        std::vector<float> mean;
        double imageMean = 0;
        double unassignedMean = 0;
        std::size_t unassignedPixels = 0;
        std::size_t pixelCount = 0;
    };

    // Float atomics avoid retaining a pixel-by-surfel matrix. The two arrays
    // also permit a weighted merge across cameras without weighting large
    // images more heavily than training's mean-per-image objective.
    struct TargetLossAccumulation {
        std::vector<Pale::float2> sums;
        std::vector<float> coverage;
    };

    inline TargetLossAccumulation projectTargetLoss(sycl::queue queue, const Pale::CameraGPU& camera,
            Pale::GPUSceneBuffers scene, Pale::PathTracerSettings settings,
            const std::vector<LossInstanceRange>& ranges, std::size_t surfelCount,
            const std::vector<float>& loss) {
        const auto pixels = static_cast<std::size_t>(camera.width) * camera.height;
        if (loss.size() != pixels || ranges.empty() || surfelCount == 0)
            throw std::runtime_error("Invalid target projection dimensions or empty scene");
        auto makeDevice = [&]<class T>(std::size_t count) {
            auto deleter = [queue](T* p) { if (p) sycl::free(p, queue); };
            auto* p = sycl::malloc_device<T>(count, queue);
            if (!p) throw std::runtime_error("Failed to allocate target loss projection");
            return std::unique_ptr<T, decltype(deleter)>(p, deleter);
        };
        auto deviceLoss = makeDevice.template operator()<float>(pixels);
        auto deviceCoverage = makeDevice.template operator()<float>(pixels);
        auto deviceOffsets = makeDevice.template operator()<LossInstanceRange>(ranges.size());
        auto deviceSums = makeDevice.template operator()<float>(2 * surfelCount);
        float* source = deviceLoss.get();
        float* coverage = deviceCoverage.get();
        float* sums = deviceSums.get();
        auto* instanceOffsets = deviceOffsets.get();
        TargetLossAccumulation result;
        result.sums.resize(surfelCount);
        result.coverage.resize(pixels);
        scene.profileCounters = nullptr;
        try {
            queue.memcpy(source, loss.data(), pixels * sizeof(float));
            queue.memcpy(instanceOffsets, ranges.data(), ranges.size() * sizeof(LossInstanceRange));
            queue.memset(sums, 0, surfelCount * 2 * sizeof(float));
            queue.wait_and_throw();
            queue.parallel_for(sycl::range<1>(pixels), [=](sycl::id<1> id) {
                const auto p = static_cast<uint32_t>(id[0]);
                const auto ray = Pale::makePrimaryRayFromPixelJitteredFov(camera,
                    float(p % camera.width), float(p / camera.width), 0.0f, 0.0f);
                float assigned = 0;
                forEachLossContributor(ray, scene, settings, instanceOffsets,
                    [&](std::size_t index, float weight) {
                        if (index >= surfelCount) return;
                        using Atomic = sycl::atomic_ref<float, sycl::memory_order::relaxed,
                            sycl::memory_scope::device, sycl::access::address_space::global_space>;
                        Atomic(sums[2*index]).fetch_add(weight * source[p]);
                        Atomic(sums[2*index+1]).fetch_add(weight);
                        assigned += weight;
                    });
                coverage[p] = assigned;
            }).wait_and_throw();
            queue.memcpy(result.sums.data(), sums, surfelCount * sizeof(Pale::float2));
            queue.memcpy(result.coverage.data(), coverage, pixels * sizeof(float));
            queue.wait_and_throw();
        } catch (...) {
            queue.wait(); // Keep buffers alive until submitted work has stopped.
            throw;
        }
        return result;
    }
}
