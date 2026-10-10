#pragma once

#include "Renderer/RenderPackage.h"
#include <stdexcept>

namespace Pale {

// Window reductions live in world coordinates. The score uses the frame that
// produced the RGB gradient; splitting projects directionSum onto the current
// frame again after the optimizer update.
struct DensificationAccumulator {
    size_t pointCount = 0;
    float *positionSum = nullptr;
    float *positionCount = nullptr;
    float3 *directionSum = nullptr;
    float *radianceSum = nullptr;
};

inline void freeDensificationAccumulator(sycl::queue &queue, DensificationAccumulator &acc) {
    if (acc.positionSum || acc.positionCount || acc.directionSum || acc.radianceSum) {
        queue.wait_and_throw();
        sycl::free(acc.positionSum, queue);
        sycl::free(acc.positionCount, queue);
        sycl::free(acc.directionSum, queue);
        sycl::free(acc.radianceSum, queue);
    }
    acc = {};
}

inline void resetDensificationAccumulator(sycl::queue &queue, const DensificationAccumulator &acc) {
    if (!acc.pointCount) return;
    queue.fill(acc.positionSum, 0.0f, acc.pointCount);
    queue.fill(acc.positionCount, 0.0f, acc.pointCount);
    queue.fill(acc.directionSum, float3{0.0f}, acc.pointCount);
    queue.fill(acc.radianceSum, 0.0f, acc.pointCount);
}

inline void ensureDensificationAccumulator(sycl::queue &queue, DensificationAccumulator &acc,
                                            size_t pointCount) {
    if (acc.pointCount == pointCount && (pointCount == 0 ||
        (acc.positionSum && acc.positionCount && acc.directionSum && acc.radianceSum))) return;
    freeDensificationAccumulator(queue, acc);
    if (!pointCount) return;
    acc.pointCount = pointCount;
    acc.positionSum = sycl::malloc_device<float>(pointCount, queue);
    acc.positionCount = sycl::malloc_device<float>(pointCount, queue);
    acc.directionSum = sycl::malloc_device<float3>(pointCount, queue);
    acc.radianceSum = sycl::malloc_device<float>(pointCount, queue);
    if (!acc.positionSum || !acc.positionCount || !acc.directionSum || !acc.radianceSum) {
        freeDensificationAccumulator(queue, acc);
        throw std::runtime_error("Could not allocate densification window statistics");
    }
    resetDensificationAccumulator(queue, acc);
}

inline sycl::event launchAccumulateDensificationStatistics(
    sycl::queue &queue, const DensificationAccumulator &acc, const Point *points,
    const PointGradients &gradients, const uint8_t *trainableMask,
    bool downweightNormal, bool relativeError) {
    if (!points || acc.pointCount != gradients.numPoints || !acc.positionSum ||
        !gradients.cloneSignalPerPrimitivePerCamera ||
        !gradients.cloneSignalRecordCountPerPrimitivePerCamera ||
        (relativeError && !gradients.cloneRadianceRmsSumPerPrimitivePerCamera)) {
        throw std::runtime_error("Incompatible densification statistics buffers");
    }
    return queue.parallel_for<class AccumulateDensificationWindowKernel>(
        sycl::range<1>(acc.pointCount), [=](sycl::id<1> id) {
            const size_t i = id[0];
            if (trainableMask ? trainableMask[i] == 0u : points[i].isEmissive()) return;
            auto dot3 = [](const float3 &a, const float3 &b) {
                return a.x()*b.x() + a.y()*b.y() + a.z()*b.z();
            };
            auto finite = [](float v) { return sycl::isfinite(v) ? v : 0.0f; };
            const float3 u = points[i].tanU;
            const float3 v = points[i].tanV;
            const float3 n{u.y()*v.z()-u.z()*v.y(), u.z()*v.x()-u.x()*v.z(),
                           u.x()*v.y()-u.y()*v.x()};
            const float3 tu = u / sycl::fmax(sycl::sqrt(dot3(u,u)), 1.0e-8f);
            const float3 tv = v / sycl::fmax(sycl::sqrt(dot3(v,v)), 1.0e-8f);
            const float3 normal = n / sycl::fmax(sycl::sqrt(dot3(n,n)), 1.0e-8f);
            float score = 0.0f, radiance = 0.0f;
            float3 direction{0.0f};
            uint32_t visibleCameras = 0u;
            for (size_t c = 0; c < gradients.cameraSlotCount; ++c) {
                const size_t slot = i * gradients.cameraSlotCount + c;
                const uint32_t count = gradients.cloneSignalRecordCountPerPrimitivePerCamera[slot];
                if (!count) continue;
                ++visibleCameras;
                const float3 g = gradients.cloneSignalPerPrimitivePerCamera[slot];
                const float gu = dot3(g,tu), gv = dot3(g,tv), gn = dot3(g,normal);
                const float tangentNorm = finite(sycl::sqrt(gu*gu + gv*gv));
                const float totalNorm = sycl::sqrt(tangentNorm*tangentNorm + gn*gn);
                const float weight = downweightNormal
                    ? finite(tangentNorm / sycl::fmax(totalNorm, 1.0e-12f)) : 1.0f;
                score += tangentNorm * weight;
                direction += g * weight;
                if (relativeError) radiance += finite(
                    gradients.cloneRadianceRmsSumPerPrimitivePerCamera[slot] / static_cast<float>(count));
            }
            if (!visibleCameras) return;
            const float inverseCount = 1.0f / static_cast<float>(visibleCameras);
            score = finite(score * inverseCount);
            direction = float3{finite(direction.x()*inverseCount), finite(direction.y()*inverseCount),
                               finite(direction.z()*inverseCount)};
            // Preserve the established positive-score observation convention.
            if (score > 0.0f) {
                acc.positionSum[i] += score;
                acc.positionCount[i] += 1.0f;
                if (relativeError) acc.radianceSum[i] += finite(radiance * inverseCount);
            }
            const float directionNormSquared = dot3(direction, direction);
            if (sycl::isfinite(directionNormSquared) && directionNormSquared > 0.0f)
                acc.directionSum[i] += direction;
        });
}

} // namespace Pale
