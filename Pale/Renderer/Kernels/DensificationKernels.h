#pragma once

#include "Renderer/RenderPackage.h"

#include <new>
#include <stdexcept>

namespace Pale {

struct DensificationAccumulator {
    float *positionSum = nullptr;
    float *positionCount = nullptr;
    float *radianceSum = nullptr;
    float3 *directionSum = nullptr;
    std::size_t numPoints = 0;
};

inline void freeDensificationAccumulator(sycl::queue queue, DensificationAccumulator &state) {
    // USM allocations must outlive all kernels and readbacks using them.
    queue.wait();
    sycl::free(state.positionSum, queue);
    sycl::free(state.positionCount, queue);
    sycl::free(state.radianceSum, queue);
    sycl::free(state.directionSum, queue);
    state = {};
}

inline void resetDensificationAccumulator(sycl::queue queue, const DensificationAccumulator &state) {
    if (state.numPoints == 0) return;
    queue.memset(state.positionSum, 0, state.numPoints * sizeof(float));
    queue.memset(state.positionCount, 0, state.numPoints * sizeof(float));
    queue.memset(state.radianceSum, 0, state.numPoints * sizeof(float));
    queue.memset(state.directionSum, 0, state.numPoints * sizeof(float3));
    queue.wait_and_throw();
}

inline void ensureDensificationAccumulator(
    sycl::queue queue, DensificationAccumulator &state, std::size_t numPoints) {
    if (state.numPoints == numPoints) return;
    // Allocate separately so a failed resize leaves the previous state intact.
    DensificationAccumulator replacement{};
    replacement.numPoints = numPoints;
    try {
        if (numPoints != 0) {
            replacement.positionSum = sycl::malloc_device<float>(numPoints, queue);
            replacement.positionCount = sycl::malloc_device<float>(numPoints, queue);
            replacement.radianceSum = sycl::malloc_device<float>(numPoints, queue);
            replacement.directionSum = sycl::malloc_device<float3>(numPoints, queue);
            if (!replacement.positionSum || !replacement.positionCount ||
                !replacement.radianceSum || !replacement.directionSum) {
                throw std::bad_alloc();
            }
            resetDensificationAccumulator(queue, replacement);
        }
    } catch (...) {
        freeDensificationAccumulator(queue, replacement);
        throw;
    }
    freeDensificationAccumulator(queue, state);
    state = replacement;
}

// Like the renderer, this kernel uses an in-order queue. Per-camera gradients
// must be ready before submission; subsequent readbacks/optimizer steps follow it.
// Mirrors update_densification_statistics in python/training_helpers.py.
inline void launchAccumulateDensificationStatistics(
    sycl::queue queue, DensificationAccumulator state, const Point *points,
    PointGradients gradients, const bool *trainableMask,
    bool downweightNormal, bool relativeDensification) {
    if (state.numPoints == 0) return;
    if (state.numPoints != gradients.numPoints || !points ||
        !state.positionSum || !state.positionCount || !state.radianceSum || !state.directionSum) {
        throw std::invalid_argument("Invalid densification accumulator or point buffers");
    }
    if (gradients.cameraSlotCount == 0) return;
    if (!gradients.cloneSignalPerPrimitivePerCamera ||
        !gradients.cloneSignalRecordCountPerPrimitivePerCamera ||
        (relativeDensification && !gradients.cloneRadianceRmsSumPerPrimitivePerCamera)) {
        throw std::invalid_argument("Missing per-camera densification gradient statistics");
    }

    queue.parallel_for<class AccumulateDensificationStatisticsKernel>(
        sycl::range<1>(state.numPoints), [=](sycl::id<1> id) {
            const std::size_t pointIndex = id[0];
            if (points[pointIndex].isEmissive() ||
                (trainableMask && !trainableMask[pointIndex])) return;

            const float3 u = points[pointIndex].tanU /
                sycl::fmax(Pale::length(points[pointIndex].tanU), 1.0e-8f);
            const float3 v = points[pointIndex].tanV /
                sycl::fmax(Pale::length(points[pointIndex].tanV), 1.0e-8f);
            const float3 normal = Pale::cross(u, v);
            const float3 w = normal / sycl::fmax(Pale::length(normal), 1.0e-8f);
            float position = 0.0f;
            float radiance = 0.0f;
            float3 direction{0.0f};
            std::size_t visibleCameraCount = 0;

            for (std::size_t camera = 0; camera < gradients.cameraSlotCount; ++camera) {
                const std::size_t index = pointIndex * gradients.cameraSlotCount + camera;
                const auto recordCount = gradients.cloneSignalRecordCountPerPrimitivePerCamera[index];
                if (recordCount == 0) continue;
                ++visibleCameraCount;
                const float3 signal = gradients.cloneSignalPerPrimitivePerCamera[index];
                const float dotU = Pale::dot(signal, u);
                const float dotV = Pale::dot(signal, v);
                float tangentNorm = sycl::sqrt(dotU * dotU + dotV * dotV);
                if (!sycl::isfinite(tangentNorm)) tangentNorm = 0.0f;
                float weight = 1.0f;
                if (downweightNormal) {
                    const float dotW = Pale::dot(signal, w);
                    const float norm = sycl::sqrt(tangentNorm * tangentNorm + dotW * dotW);
                    weight = tangentNorm / sycl::fmax(norm, 1.0e-12f);
                    if (!sycl::isfinite(weight)) weight = 0.0f;
                }
                position += tangentNorm * weight;
                // Do not normalize by record count: each camera stores a sum
                // of gradient contributions. Only radiance is a per-record mean.
                direction += signal * weight;
                if (relativeDensification) {
                    const float meanRadiance =
                        gradients.cloneRadianceRmsSumPerPrimitivePerCamera[index] /
                        static_cast<float>(recordCount);
                    if (sycl::isfinite(meanRadiance)) radiance += meanRadiance;
                }
            }
            if (visibleCameraCount == 0) return;
            const float inverseCount = 1.0f / static_cast<float>(visibleCameraCount);
            position *= inverseCount;
            radiance *= inverseCount;
            direction *= inverseCount;
            if (sycl::isfinite(position) && position > 0.0f) {
                state.positionSum[pointIndex] += position;
                state.positionCount[pointIndex] += 1.0f;
                if (relativeDensification && sycl::isfinite(radiance)) {
                    state.radianceSum[pointIndex] += radiance;
                }
            }
            // Python sanitizes each direction component before checking its norm.
            if (!sycl::isfinite(direction.x())) direction.x() = 0.0f;
            if (!sycl::isfinite(direction.y())) direction.y() = 0.0f;
            if (!sycl::isfinite(direction.z())) direction.z() = 0.0f;
            const float directionNorm = Pale::length(direction);
            if (sycl::isfinite(directionNorm) && directionNorm > 0.0f) {
                state.directionSum[pointIndex] += direction;
            }
        });
}

} // namespace Pale
