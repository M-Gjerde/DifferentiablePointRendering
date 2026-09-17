#pragma once

#include "SharedHeightSurface.h"
#include "SharedSlabSurface.h"
#include "IntersectionKernels.h"
#include "Renderer/RenderPackage.h"
#include <memory>

namespace Pale {

struct SharedHeightForwardData {
    SharedHeightSurface surface;
    SharedSlabSurface slabs;
    Transform transform;
};

// Test renderer: a single point-cloud instance, including its emissive surfels.
// The small test scene is deliberately traversed exhaustively; neither capped
// hit lists nor the constituent planes' BVH define the reconstructed support.
inline void launchSharedHeightForward(RenderPackage &pkg, uint32_t cameraIndex) {
    const auto scene = pkg.scene;
    const auto settings = pkg.settings;
    const auto sensor = pkg.sensors[cameraIndex];
    auto *prepared = sycl::malloc_shared<SharedHeightForwardData>(1, pkg.queue);
    if (!prepared) throw std::bad_alloc();
    const auto releasePrepared = [&queue = pkg.queue](SharedHeightForwardData *p) {
        sycl::free(p, queue);
    };
    // Keep the USM allocation alive through both kernels, including exception paths.
    const std::unique_ptr<SharedHeightForwardData, decltype(releasePrepared)> allocation(prepared, releasePrepared);
    pkg.queue.single_task<class PrepareSharedHeightForward>([=]() {
        *prepared = SharedHeightForwardData{};
        uint32_t instanceIndex = kInvalidIndex;
        if (!tryGetSinglePointCloudInstance(scene, instanceIndex)) return;
        prepared->transform = scene.transforms[scene.instances[instanceIndex].transformIndex];
        auto fillMember = [&](SharedHeightMember &member, int index) {
            if (index < 0 || uint32_t(index) >= scene.pointCount) return false;
            const auto &p = scene.points[index];
            if (p.isEmissive()) return false;
            const auto &tf = prepared->transform.objectToWorld;
            member.center = transformPoint(tf, p.position);
            const auto a = tf * float4{p.tanU * p.scale.x(), 0.0f};
            const auto b = tf * float4{p.tanV * p.scale.y(), 0.0f};
            member.axisA = float3{a.x(), a.y(), a.z()};
            member.axisB = float3{b.x(), b.y(), b.z()};
            member.opacity = sycl::clamp(p.opacity, 0.0f, 1.0f);
            member.albedo = p.alpha_r * p.albedo;
            member.index = uint32_t(index);
            return true;
        };
        if (settings.sharedHeightTwoSlabs) {
            const int starts[2]{settings.sharedHeightSlabStartA, settings.sharedHeightSlabStartB};
            const int counts[2]{settings.sharedHeightSlabCountA, settings.sharedHeightSlabCountB};
            for (int k = 0; k < 2; ++k) {
                auto &chart = prepared->slabs.charts[k];
                if (counts[k] < 1 || counts[k] > SharedHeightSurface::capacity || starts[k] < 0 ||
                    uint64_t(starts[k]) + uint64_t(counts[k]) > scene.pointCount) return;
                chart.height.memberCount = counts[k];
                chart.halfWidth = settings.sharedHeightSlabHalfWidth;
                chart.halfDepth = settings.sharedHeightSlabHalfDepth;
                chart.lateralShift = k == 0 ? settings.sharedHeightSlabLateralShiftA : settings.sharedHeightSlabLateralShiftB;
                chart.depthShift = k == 0 ? settings.sharedHeightSlabDepthShiftA : settings.sharedHeightSlabDepthShiftB;
                for (int i = 0; i < counts[k]; ++i)
                    if (!fillMember(chart.height.members[i], starts[k] + i)) return;
            }
            prepared->slabs.coverage = sycl::fmax(0.0f, settings.sharedHeightSlabCoverage);
            prepareSharedSlabSurface(prepared->slabs);
            return;
        }
        int indices[2]{settings.sharedHeightMemberA, settings.sharedHeightMemberB};
        for (int m = 0; m < 2; ++m) {
            if (indices[m] < 0) {
                for (uint32_t i = 0; i < scene.pointCount; ++i) {
                    if (int(i) != indices[1 - m] && !scene.points[i].isEmissive()) {
                        indices[m] = int(i); break;
                    }
                }
            }
            if (indices[m] < 0 || uint32_t(indices[m]) >= scene.pointCount) return;
            if (!fillMember(prepared->surface.members[m], indices[m])) return;
        }
        if (indices[0] == indices[1]) return;
        prepareSharedHeightSurface(prepared->surface);
    }).wait_and_throw();
    const bool valid = settings.sharedHeightTwoSlabs ? prepared->slabs.valid : prepared->surface.valid;
    const float sceneExtent = settings.sharedHeightTwoSlabs ? length(prepared->slabs.upper - prepared->slabs.lower) :
        length(prepared->surface.upper - prepared->surface.lower);
    const uint32_t count = sensor.camera.width * sensor.camera.height;
    const auto renderPixel = [=](sycl::id<1> id) {
        // Capture only the pointer. Capturing this multi-kilobyte record by value
        // exceeds the parameter-space limit on older CUDA/PTX targets.
        const auto &data = *prepared;
        const uint32_t pixel = id[0];
        const Ray ray = makePrimaryRayFromPixelJitteredFov(sensor.camera,
            float(pixel % sensor.camera.width), float(pixel / sensor.camera.width), 0.0f, 0.0f);
        Ray objectRay{};
        if (valid) objectRay.origin = transformPoint(data.transform.worldToObject, ray.origin);
        const auto d = data.transform.worldToObject * float4{ray.direction, 0.0f};
        objectRay.direction = float3{d.x(), d.y(), d.z()}; // Preserve world ray parameter.
        float3 rgb{0.0f};
        float transmission = 1.0f, weightedDepth = 0.0f, weightSum = 0.0f;
        float medianDepth = 0.0f;
        float4 medianPosition{0.0f}, medianNormal{0.0f};
        float tMin = RayEpsilon;
        for (uint32_t event = 0; valid &&
             event < rendererDebugMaxSplatEventsPerRay(settings) && transmission > 1.0e-6f; ++event) {
            float closest = INFINITY;
            SharedHeightEvaluation evaluation;
            SharedSlabEvaluation slabEvaluation;
            bool found = settings.sharedHeightTwoSlabs ?
                intersectSharedSlabs(data.slabs, ray, tMin, closest, closest, slabEvaluation) :
                intersectSharedHeight(data.surface, ray, tMin, closest, closest, evaluation);
            float3 normal{0.0f}, albedo{0.0f}, emission{0.0f};
            float alpha = 0.0f;
            if (found) {
                normal = settings.sharedHeightTwoSlabs ? normalize(slabEvaluation.gradient) : normalize(data.surface.normal - evaluation.gradient);
                albedo = settings.sharedHeightTwoSlabs ? slabEvaluation.albedo : evaluation.albedo;
                alpha = settings.sharedHeightTwoSlabs ? slabEvaluation.opacity : evaluation.opacity;
                if (settings.sharedHeightTwoSlabs && settings.sharedHeightShading == 3)
                    albedo = float3{1.0f - slabEvaluation.weightB, 0.15f, slabEvaluation.weightB};
            }
            for (uint32_t i = 0; i < scene.pointCount; ++i) {
                bool member = false;
                if (settings.sharedHeightTwoSlabs) {
                    for (int k = 0; k < 2; ++k)
                        for (int j = 0; j < data.slabs.charts[k].height.memberCount; ++j)
                            member |= i == data.slabs.charts[k].height.members[j].index;
                } else member = i == data.surface.members[0].index || i == data.surface.members[1].index;
                if (member) continue;
                const auto &p = scene.points[i];
                float t, profile;
                if (!intersectSurfel(objectRay, p, tMin, closest, t, 1.0e-8f)) continue;
                const auto uv = phiInverse(objectRay.origin + t * objectRay.direction, p);
                if (!opacityBeta(uv, p, &profile) || p.opacity * profile <= 0.0f) continue;
                closest = t; found = true;
                alpha = sycl::clamp(p.opacity * profile, 0.0f, 1.0f);
                const auto a = data.transform.objectToWorld * float4{p.tanU, 0.0f};
                const auto b = data.transform.objectToWorld * float4{p.tanV, 0.0f};
                normal = normalize(cross(float3{a.x(), a.y(), a.z()}, float3{b.x(), b.y(), b.z()}));
                albedo = p.alpha_r * p.albedo;
                emission = p.isEmissive() ? min(p.flux * p.albedo, float3{1.0f}) : float3{0.0f};
            }
            if (!found) break;
            const float3 position = ray.origin + closest * ray.direction;
            if (dot(normal, ray.direction) > 0.0f) normal = -normal;
            float3 outgoing = emission;
            if (settings.sharedHeightShading == 1 || settings.sharedHeightShading == 3) outgoing = albedo;
            else if (settings.sharedHeightShading == 2) outgoing = 0.5f * (normal + float3{1.0f});
            else {
                // Geometry-only forward experiment: no old-plane photon lookup
                // or shadow query, which would reintroduce the removed surfaces.
                for (uint32_t lightIndex = 0; lightIndex < scene.lightCount; ++lightIndex) {
                    const auto &light = scene.lights[lightIndex];
                    if (light.lightType != LightType::Surfel) continue;
                    const auto lightPosition = transformPoint(data.transform.objectToWorld,
                        scene.points[light.primitiveIndex].position);
                    const auto toLight = lightPosition - position;
                    const float distance2 = dot(toLight, toLight);
                    if (distance2 <= 1.0e-12f) continue;
                    const float cosine = sycl::fmax(0.0f, dot(normal, toLight / sycl::sqrt(distance2)));
                    outgoing += albedo * light.flux * light.color *
                        (cosine / (4.0f * M_PIf * M_PIf * distance2));
                }
            }
            const float weight = transmission * alpha;
            rgb += weight * outgoing;
            const float depth = dot(position - sensor.camera.pos, sensor.camera.forward);
            weightedDepth += weight * depth;
            if (medianDepth == 0.0f && weightSum + weight >= 0.5f) {
                medianDepth = depth;
                medianPosition = float4{position, 1.0f};
                medianNormal = float4{normal, 1.0f};
            }
            weightSum += weight;
            transmission *= 1.0f - alpha;
            tMin = closest + sycl::fmax(RayEpsilon, 2.0e-6f * sceneExtent);
        }
        sensor.framebuffer[pixel] = float4{rgb, sycl::clamp(weightSum, 0.0f, 1.0f)};
        sensor.meanDepthBuffer[pixel] = weightSum > 0.0f ? weightedDepth / weightSum : 0.0f;
        sensor.medianDepthBuffer[pixel] = medianDepth;
        sensor.medianWorldPositionBuffer[pixel] = medianPosition;
        sensor.visibleNormalBuffer[pixel] = medianNormal;
        sensor.normalFromDepthBuffer[pixel] = float4{0.0f};
        sensor.depthDistortionBuffer[pixel] = 0.0f;
        if (sensor.depthDistortionAdjointBuffer) sensor.depthDistortionAdjointBuffer[pixel] = 0.0f;
        sensor.intraSlabDepthBuffer[pixel] = 0.0f;
        if (sensor.intraSlabRayDepthBuffer != nullptr) {
            sensor.intraSlabRayDepthBuffer[pixel] = 0.0f;
        }
        if (sensor.intraSlabDepthAdjointBuffer) sensor.intraSlabDepthAdjointBuffer[pixel] = 0.0f;
        sensor.intraSlabDepthActiveSlabCountBuffer[pixel] = 0u;
        sensor.curvatureScaleBuffer[pixel] = 0.0f;
        if (sensor.surfaceCurvatureBuffer != nullptr) {
            sensor.surfaceCurvatureBuffer[pixel] = std::numeric_limits<float>::quiet_NaN();
        }
        if (sensor.curvatureScaleAdjointBuffer) sensor.curvatureScaleAdjointBuffer[pixel] = 0.0f;
        sensor.curvatureScaleActiveSlabCountBuffer[pixel] = 0u;
        if (sensor.curvaturePrimitiveIndexBuffer) sensor.curvaturePrimitiveIndexBuffer[pixel] = kInvalidIndex;
    };
    // Leave room for AdaptiveCpp's launch wrapper under the 4352-byte PTX limit.
    static_assert(sizeof(renderPixel) <= 2048, "Shared-height pixel kernel captures must stay small; use USM pointers for scene data.");
    pkg.queue.parallel_for<class SharedHeightForwardCamera>(sycl::range<1>(count), renderPixel).wait_and_throw();
}

} // namespace Pale
