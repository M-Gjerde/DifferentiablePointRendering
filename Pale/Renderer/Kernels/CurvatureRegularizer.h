#pragma once

#include "IntersectionKernels.h"

namespace Pale {
    struct CurvatureTensor {
        float uu = 0.0f, uv = 0.0f, vv = 0.0f;
    };

    // Fit a symmetric normal derivative B from B * displacement = deltaNormal.
    // Normalize each observation by its baseline, so sampling distance alone
    // does not determine its influence. No pixel normals or depth derivatives
    // enter this fit.
    struct CurvatureFit {
        float cuu = 0.0f, cuv = 0.0f, cvv = 0.0f;
        float ruu = 0.0f, ruv = 0.0f, rvu = 0.0f, rvv = 0.0f;

        void add(float u, float v, float nu, float nv) {
            const float inverseDistanceSquared = 1.0f / (u * u + v * v);
            cuu += u * u * inverseDistanceSquared;
            cuv += u * v * inverseDistanceSquared;
            cvv += v * v * inverseDistanceSquared;
            ruu += nu * u * inverseDistanceSquared;
            ruv += nu * v * inverseDistanceSquared;
            rvu += nv * u * inverseDistanceSquared;
            rvv += nv * v * inverseDistanceSquared;
        }

        CurvatureTensor solve() const {
            if (cuu + cvv <= 0.0f) { return {}; }
            // In the eigenbasis of C, the symmetric least-squares solution is
            // B_ij = (R_ij + R_ji) / (C_ii + C_jj). Use a truncated inverse for
            // unobserved directions (one neighbor cannot determine all of B).
            const float angle = 0.5f * sycl::atan2(2.0f * cuv, cuu - cvv);
            const float c = sycl::cos(angle), s = sycl::sin(angle);
            const float c1 = c * c * cuu + 2.0f * c * s * cuv + s * s * cvv;
            const float c2 = sycl::fmax(0.0f, s * s * cuu - 2.0f * c * s * cuv + c * c * cvv);
            const float r11 = c * c * ruu + c * s * (ruv + rvu) + s * s * rvv;
            const float r22 = s * s * ruu - c * s * (ruv + rvu) + c * c * rvv;
            const float r12sum = (c * c - s * s) * (ruv + rvu) + 2.0f * c * s * (rvv - ruu);
            const float b11 = r11 / c1;
            const float b22 = c2 > 1.0e-4f * c1 ? r22 / c2 : 0.0f;
            const float b12 = r12sum / (c1 + c2);
            return {c * c * b11 - 2.0f * c * s * b12 + s * s * b22,
                    c * s * (b11 - b22) + (c * c - s * s) * b12,
                    s * s * b11 + 2.0f * c * s * b12 + c * c * b22};
        }
    };

    struct CurvatureFootprint {
        float residual = 0.0f;
        float2 lossScaleGradient{0.0f};
        CurvatureTensor splitTensor{};
    };

    inline CurvatureFootprint evaluateCurvatureFootprint(
        const CurvatureTensor &b, float scaleU, float scaleV, float thickness) {
        CurvatureFootprint result{};
        if (!(thickness > 0.0f) || !(scaleU > 0.0f) || !(scaleV > 0.0f)) { return result; }
        // The ellipse is D times the unit disk. Its maximum absolute quadratic
        // departure is rho(D B D)/2, including rotated ellipses and saddles.
        const float a = b.uu * scaleU * scaleU;
        const float off = b.uv * scaleU * scaleV;
        const float d = b.vv * scaleV * scaleV;
        const float trace = a + d;
        const float gap = sycl::hypot(a - d, 2.0f * off);
        const float radius = 0.5f * (sycl::fabs(trace) + gap);
        const float denominator = 2.0f * CurvatureScaleRegularizerGamma * thickness;
        if (!sycl::isfinite(radius)) { return result; }
        result.residual = sycl::fmax(0.0f, radius / denominator - 1.0f);
        if (result.residual == 0.0f) { return result; }

        // A balanced subgradient at tied absolute eigenvalues avoids choosing
        // an arbitrary axis for a sphere or a symmetric saddle.
        const float sign = trace > 0.0f ? 1.0f : (trace < 0.0f ? -1.0f : 0.0f);
        const float q = gap > 0.0f ? (a - d) / gap : 0.0f;
        const float t = gap > 0.0f ? 2.0f * off / gap : 0.0f;
        const float da = 0.5f * (sign + q), dd = 0.5f * (sign - q);
        const float common = 2.0f * result.residual / denominator;
        result.lossScaleGradient = float2{
            common * (2.0f * b.uu * scaleU * da + b.uv * scaleV * t),
            common * (2.0f * b.vv * scaleV * dd + b.uv * scaleU * t)};

        // Map the worst unit-disk eigenaxis through D into the surfel plane.
        // Ties produce a sum of both axes, instead of an unstable direction.
        const float puu = 0.5f * (1.0f + sign * q) * scaleU * scaleU;
        const float puv = 0.5f * sign * t * scaleU * scaleV;
        const float pvv = 0.5f * (1.0f - sign * q) * scaleV * scaleV;
        const float weight = result.residual / (puu + pvv);
        result.splitTensor = {weight * puu, weight * puv, weight * pvv};
        return result;
    }

    inline bool estimateSurfelCurvature(
        const Point &surfel, const PointCloudLocalLayer &layer,
        const Transform &transform, const GPUSceneBuffers &scene,
        float thickness, float normalThreshold, CurvatureTensor &footprintTensor,
        CurvatureTensor *worldTensor = nullptr) {
        // transformDirection normalizes its result; retain transformed lengths
        // here because they are part of the physical ellipse footprint.
        const float3x3 linear = linearPart(transform.objectToWorld);
        const float3 axisU = linear * surfel.tanU;
        const float3 axisV = linear * surfel.tanV;
        const float3 normalRaw = cross(axisU, axisV);
        const float normalSquared = dot(normalRaw, normalRaw);
        const float uSquared = dot(axisU, axisU);
        if (!sycl::isfinite(normalSquared) || normalSquared <= 1.0e-12f ||
            !sycl::isfinite(uSquared) || uSquared <= 1.0e-12f) { return false; }
        const float3 normal = normalRaw / sycl::sqrt(normalSquared);
        const float3 tangentU = axisU / sycl::sqrt(uSquared);
        const float3 tangentV = cross(normal, tangentU);
        const float3 center = transformPoint(transform.objectToWorld, surfel.position);
        CurvatureFit fit{};
        for (uint32_t i = 0u; i < layer.hitCount; ++i) {
            const Point &neighbor = scene.points[layer.hits[i].primitiveIndex];
            const float3 neighborNormalRaw = cross(linear * neighbor.tanU, linear * neighbor.tanV);
            const float neighborNormalSquared = dot(neighborNormalRaw, neighborNormalRaw);
            if (!sycl::isfinite(neighborNormalSquared) || neighborNormalSquared <= 1.0e-12f) { continue; }
            float3 neighborNormal = neighborNormalRaw / sycl::sqrt(neighborNormalSquared);
            if (dot(normal, neighborNormal) < 0.0f) { neighborNormal = -neighborNormal; }
            // Even if slab normal filtering is disabled, do not fit across a
            // sharp crease. The estimator is a local smooth-surface model.
            if (dot(normal, neighborNormal) < sycl::fmax(0.70710678f, normalThreshold)) { continue; }
            const float3 displacement = transformPoint(transform.objectToWorld, neighbor.position) - center;
            const float u = dot(displacement, tangentU), v = dot(displacement, tangentV);
            const float distanceSquared = u * u + v * v;
            if (!sycl::isfinite(distanceSquared) ||
                distanceSquared <= CurvatureRegularizerDistanceEpsilon * CurvatureRegularizerDistanceEpsilon) { continue; }
            // A smooth pair's chord is tangent to its average normal to first
            // order. Reject displaced sheets, even if their ray hits share a slab.
            const float chordOffset = dot(displacement, 0.5f * (normal + neighborNormal));
            if (!sycl::isfinite(chordOffset) || sycl::fabs(chordOffset) > thickness) { continue; }
            const float3 normalDifference = neighborNormal - normal;
            fit.add(u, v, dot(normalDifference, tangentU), dot(normalDifference, tangentV));
        }
        if (fit.cuu + fit.cvv <= 0.0f) { return false; }
        const CurvatureTensor b = fit.solve();
        if (worldTensor != nullptr) { *worldTensor = b; }
        // Pull the world-space form back through the transformed surfel axes.
        // This also accounts for nonuniform instance scale and shear.
        const float u = sycl::sqrt(uSquared);
        const float vu = dot(axisV, tangentU), vv = dot(axisV, tangentV);
        footprintTensor = {b.uu * u * u, u * (b.uu * vu + b.uv * vv),
                           b.uu * vu * vu + 2.0f * b.uv * vu * vv + b.vv * vv * vv};
        return sycl::isfinite(footprintTensor.uu) && sycl::isfinite(footprintTensor.uv) &&
               sycl::isfinite(footprintTensor.vv);
    }
}
