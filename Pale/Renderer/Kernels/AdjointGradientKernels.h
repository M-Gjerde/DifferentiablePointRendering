#pragma once
#include "IntersectionKernels.h"
#include "KernelHelpers.h"
#include "Renderer/GPUDataTypes.h"


namespace Pale {
    using AtomicFloat = sycl::atomic_ref<
        float,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>;

    inline void atomicAddFloat(float &destination, float valueToAdd) {
        AtomicFloat(destination).fetch_add(valueToAdd);
    }

    inline void atomicAddFloat3(float3 &destination, const float3 &valueToAdd) {
        atomicAddFloat(destination.x(), valueToAdd.x());
        atomicAddFloat(destination.y(), valueToAdd.y());
        atomicAddFloat(destination.z(), valueToAdd.z());
    }


    inline float3 gradTransmissionPosition(const Ray &ray, const SplatEvent &splatEvent, const Point &surfel,
                                           const float3 &segmentVector) {
        constexpr float epsilon = 1e-6f;

        const float3 tangentU = surfel.tanU;
        const float3 tangentV = surfel.tanV;
        const float3 normalW = normalize(cross(tangentU, tangentV)); // no flipping

        const float denom = dot(normalW, segmentVector);
        if (fabs(denom) <= epsilon) return float3(0.0f);

        const float2 uv = phiInverse(splatEvent.hitWorld, surfel); // your mapping to local coords

        const float alpha = splatEvent.alpha;

        const float tuDotD = dot(surfel.tanU, segmentVector);
        const float tvDotD = dot(surfel.tanV, segmentVector);

        const float su = surfel.scale.x();
        const float sv = surfel.scale.y();

        // d u / d c and d v / d c
        const float3 duDc = ((tuDotD / denom) * normalW - surfel.tanU) / su;
        const float3 dvDc = ((tvDotD / denom) * normalW - surfel.tanV) / sv;
        // d alpha / d c  for alpha = exp(-0.5*(u^2+v^2))  ==>  dα = -α*(u du + v dv)
        const float3 dAlphaDc = -alpha * (uv.x() * duDc + uv.y() * dvDc);

        return dAlphaDc;
    }

    inline float3 gradTransmissionProduct(const Ray &ray, const SplatEvent &splatEvent, const Point &surfel,
                                          const float3 &segmentVector) {
        constexpr float epsilon = 1e-6f;

        const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));
        const int travelSideSign = (dot(canonicalNormalW, -ray.direction) >= 0.0f) ? 1 : -1;
        const float3 surfelNormal = (travelSideSign >= 0)
                                        ? canonicalNormalW
                                        : (-canonicalNormalW);

        const float denom = dot(surfelNormal, segmentVector);
        if (fabs(denom) <= epsilon) return float3(0.0f);

        const float2 uv = phiInverse(splatEvent.hitWorld, surfel); // your mapping to local coords

        const float alpha = splatEvent.alpha;

        const float tuDotD = dot(surfel.tanU, segmentVector);
        const float tvDotD = dot(surfel.tanV, segmentVector);

        const float su = surfel.scale.x();
        const float sv = surfel.scale.y();

        // d u / d c and d v / d c
        const float3 duDc = ((tuDotD / denom) * surfelNormal - surfel.tanU) / sycl::fmax(
                                su, epsilon);
        const float3 dvDc = ((tvDotD / denom) * surfelNormal - surfel.tanV) / sycl::fmax(
                                sv, epsilon);
        // d alpha / d c  for alpha = exp(-0.5*(u^2+v^2))  ==>  dα = -α*(u du + v dv)
        const float3 dAlphaDc = -alpha * (uv.x() * duDc + uv.y() * dvDc);

        return dAlphaDc;
    }


    inline float3 transmissionBackgroundGradient(const GPUSceneBuffers &scene, const WorldHit &worldHit,
                                                 const RayState &rayState, const InstanceRecord &instance,
                                                 const DeviceSurfacePhotonMapGrid &photonMap) {
        float3 d_cost_d_pos(0.0f);

        // Transmission
        if (worldHit.splatEventCount > 0) {
            const float3 x = rayState.ray.origin;
            const float3 y = worldHit.hitPositionW;
            const float3 segmentVector = y - x;

            // Cost weighting: keep RGB if your loss is RGB; else reduce at end
            const float3 backgroundRadianceRGB = estimateRadianceFromPhotonMap(worldHit, scene, photonMap);
            const float L = luminance(backgroundRadianceRGB);

            // Collect alpha_i and d(alpha_i)/dc for all valid splat intersections on the segment
            struct LocalTerm {
                float alpha;
                float3 dAlphaDc;
            };
            constexpr float epsilon = 1e-6f;
            LocalTerm localTerms[kMaxSplatEvents];
            int validCount = 0;

            for (int ei = 0; ei < worldHit.splatEventCount && validCount < kMaxSplatEvents; ++ei) {
                const auto splatEvent = worldHit.splatEvents[ei];
                const auto surfel = scene.points[splatEvent.primitiveIndex];
                // d alpha / d c  for alpha = exp(-0.5*(u^2+v^2))  ==>  dα = -α*(u du + v dv)
                const float3 dAlphaDc =
                        gradTransmissionPosition(rayState.ray, splatEvent, surfel, segmentVector);
                localTerms[validCount++] = LocalTerm{splatEvent.alpha, dAlphaDc};
            }
            if (validCount != 0) {
                // τ = Π (1-α_i) with stable log-space accumulation
                float logTau = 0.f;
                for (int i = 0; i < validCount; ++i) {
                    const float oneMinusAlpha = sycl::fmax(1.f - localTerms[i].alpha, 1e-6f);
                    logTau += sycl::log(oneMinusAlpha);
                }
                const float tau = sycl::exp(logTau);

                // Σ_i [ - dα_i/dc / (1-α_i) ]
                float3 sumTerm(0.f);
                for (int i = 0; i < validCount; ++i) {
                    const float oneMinusAlpha = sycl::fmax(1.f - localTerms[i].alpha, 1e-6f);
                    sumTerm += localTerms[i].dAlphaDc / oneMinusAlpha;
                }
                float cosine = dot(worldHit.geometricNormalW, -rayState.ray.direction);

                const float pAdjoint = luminanceGrayscale(rayState.pathThroughput) * tau;
                // already includes fs, V, G, etc., for this segment
                // Accumulate to your running gradient
                const float3 tauGrad = -tau * sumTerm;
                d_cost_d_pos = pAdjoint * tauGrad * L;
            }
        }
        return d_cost_d_pos;
    }

    inline void cosineAndGradientWrtPosition(const float3 &rayOrigin,
                                             const float3 &hitPositionWorld,
                                             const float3 &surfelNormal,
                                             float &cosineSigned,
                                             float3 &gradCosineWrtPosition) {
        const float3 displacementVector = hitPositionWorld - rayOrigin; // d
        const float distanceMagnitude = length(displacementVector); // p
        if (distanceMagnitude <= 1e-6f) {
            cosineSigned = 0.0f;
            gradCosineWrtPosition = float3{0.0f};
            return;
        }

        const float3 unitPsi = displacementVector / distanceMagnitude; // ψ
        const float3 incidentDirection = -unitPsi; // ω_i

        cosineSigned = dot(surfelNormal, incidentDirection);

        const float3x3 projector = identity3x3() - outerProduct(unitPsi, unitPsi);
        gradCosineWrtPosition = -(projector * surfelNormal) / distanceMagnitude;
    }

    // Assuming float3, float2, dot(), cross(), etc. are defined as in your codebase.

    // ----------------- Position gradient (translation of surfel center) -----------------
    inline float3 computeDuvDPosition(
        const float3 &tangentUWorld,
        const float3 &tangentVWorld,
        const float3 &canonicalNormalWorld,
        const float3 &rayDirection,
        float u, float v,
        float su, float sv) {
        const float denom = dot(canonicalNormalWorld, rayDirection);
        if (sycl::fabs(denom) <= 1e-4f) {
            return float3{0.0f, 0.0f, 0.0f};
        }

        const float tuDotD = dot(tangentUWorld, rayDirection);
        const float tvDotD = dot(tangentVWorld, rayDirection);

        // du/dp_k and dv/dp_k (3x1 each), from your analytic expression
        const float3 duDPk = ((tuDotD / denom) * canonicalNormalWorld - tangentUWorld) / su;
        const float3 dvDPk = ((tvDotD / denom) * canonicalNormalWorld - tangentVWorld) / sv;

        // dα/dc_pos = -α (u du/dc + v dv/dc)
        const float3 dAlphaDPosition = u * duDPk + v * dvDPk;
        return dAlphaDPosition;
    }


    inline float3 computeGradRayParameterWrtTU(
        const float3 &rayOriginWorld, // x
        const float3 &rayDirectionWorld, // d
        const float3 &surfelCenterWorld, // p_k
        const float3 &tangentUWorld, // t_u
        const float3 &tangentVWorld) {
        // t_v
        float3 normalWorld = cross(tangentUWorld, tangentVWorld);
        const float3 centerMinusOrigin = surfelCenterWorld - rayOriginWorld;
        const float nd = dot(normalWorld, rayDirectionWorld);
        const float np = dot(normalWorld, centerMinusOrigin);
        const float epsilon = 1e-6f;
        if (sycl::fabs(nd) < epsilon) {
            return float3{0.0f, 0.0f, 0.0f};
        }
        const float3 crossTvWithPkMinusX = cross(tangentVWorld, centerMinusOrigin);
        const float3 crossTvWithD = cross(tangentVWorld, rayDirectionWorld);
        const float3 firstTerm = crossTvWithPkMinusX / nd;
        const float scale = np / (nd * nd);
        const float3 secondTerm = scale * crossTvWithD;
        return firstTerm - secondTerm; // ∇_{t_u} r_t
    }

    inline float3 computeGradRayParameterWrtTV(
        const float3 &rayOriginWorld, // x
        const float3 &rayDirectionWorld, // d
        const float3 &surfelCenterWorld, // p_k
        const float3 &tangentUWorld, // t_u
        const float3 &tangentVWorld) {
        // t_v
        const float3 centerMinusOrigin = surfelCenterWorld - rayOriginWorld;
        float3 normalWorld = cross(tangentUWorld, tangentVWorld);
        const float nd = dot(normalWorld, rayDirectionWorld);
        const float np = dot(normalWorld, centerMinusOrigin);
        const float epsilon = 1e-6f;
        if (sycl::fabs(nd) < epsilon) {
            return float3{0.0f, 0.0f, 0.0f};
        }
        const float3 crossTuWithPkMinusX = cross(tangentUWorld, centerMinusOrigin);
        const float3 crossTuWithD = cross(tangentUWorld, rayDirectionWorld);
        const float3 firstTerm = crossTuWithPkMinusX / nd;
        const float scale = np / (nd * nd);
        const float3 secondTerm = scale * crossTuWithD;
        return -firstTerm + secondTerm; // ∇_{t_v} r_t
    }

    inline void computeFullDuDvWrtTangents(
        const float3 &rayOriginWorld,
        const float3 &rayDirectionWorld,
        const float3 &surfelCenterWorld,
        const float3 &hitWorld,
        const float3 &tangentUWorld,
        const float3 &tangentVWorld,
        float su, float sv,
        // outputs
        float3 &dUdTu, float3 &dVdTu,
        float3 &dUdTv, float3 &dVdTv) {
        const float3 offsetFromCenter = hitWorld - surfelCenterWorld; // z - p_k

        const float3 gradRt_tu = computeGradRayParameterWrtTU(
            rayOriginWorld, rayDirectionWorld,
            surfelCenterWorld, tangentUWorld, tangentVWorld
        );
        const float3 gradRt_tv = computeGradRayParameterWrtTV(
            rayOriginWorld, rayDirectionWorld,
            surfelCenterWorld, tangentUWorld, tangentVWorld
        );

        // TODO enforcing a front/back-symmetric derivative with this -fabs trick gives FD agreement.
        // Might not be a problem but is noted in case issues with rotation appear.
        const float tuDotD = (dot(tangentUWorld, rayDirectionWorld));
        const float tvDotD = (dot(tangentVWorld, rayDirectionWorld));

        // Π = t_u
        dUdTu = (offsetFromCenter + tuDotD * gradRt_tu) / su;
        dVdTu = (tvDotD * gradRt_tu) / sv;

        // Π = t_v
        dVdTv = (offsetFromCenter + tvDotD * gradRt_tv) / sv;
        dUdTv = (tuDotD * gradRt_tv) / su;
    }


    // ----------------- Scale gradient (s_u, s_v) -----------------
    // Here we treat the plane geometry as fixed, scales only affect the local map Φ(u,v).
    // u = (t_u · (z - p_k)) / s_u  ⇒ ∂u/∂s_u = -u / s_u,  similarly for v, s_v.
    inline float3 computeDuvDScale(
        float u, float v,
        float su, float sv) {
        const float dAlphaDSu = -(u * u) / su;
        const float dAlphaDSv = -(v * v) / sv;
        // If you later add anisotropic / z-scale, you can extend this.
        return float3{dAlphaDSu, dAlphaDSv, 0.0f};
    }

    inline float betaKernel(float beta_param) {

        return 4.0f * sycl::exp(beta_param);
    }

    inline float computeSmoothedBetaFactor(float beta_param, float r2, float alpha, float opacity) {
        float beta       = 4.0f * sycl::exp(beta_param);
        float denom      = 1.0f - r2;
        const float eps  = 1e-3f;        // still keep a small epsilon
        denom           = sycl::fmax(denom, eps);
        float betaKernelFactor = -2.0f * beta * alpha * opacity / denom;

        return betaKernelFactor;
    }

    float3 DalphaDuvPositionGaussian(float3 DuvPosition, float alpha, float opacity) {
        return -alpha * opacity * DuvPosition;
    }

    /*
    float3 DalphaDuvScaleGaussian(float3 DuvDScale, float alpha) {
        return -alpha * DuvPosition;
    }
    */

    float3 computeDAlphaDPositionBeta(
        const float3 &dUvWeightedDPosition, // u*du/dpos + v*dv/dpos per component
        float beta,
        float rSquared,
        float alpha,
        float opacity
    ) {
        const float oneMinusRSquared = 1.0f - rSquared;
        if (oneMinusRSquared <= 0.0f) {
            return float3(0.0f); // or clamp
        }
        const float factor = -2.0f * beta * alpha * opacity / oneMinusRSquared;
        return factor * dUvWeightedDPosition;
    }
    float2 computeDAlphaDScaleGaussian(
        const float3 &dUdVdScale, // u*du/dpos + v*dv/dpos per component
        float u,
        float v,
        float alpha,
        float opacity
    ) {
        return alpha * opacity * float2{dUdVdScale.x() * u, dUdVdScale.y() * v};
    }

    float2 computeDAlphaDScaleBeta(
        const float3 &dUdVdScale, // u*du/dpos + v*dv/dpos per component
        float u,
        float v,
        float alpha,
        float opacity,
        float beta,
        float r2
    ) {

        return (-2 * beta * alpha * opacity / (1 - r2))  * float2{dUdVdScale.x() * u, dUdVdScale.y() * v};
    }

    float computeDAlphaDb(
        float beta, // beta = 4*exp(b)
        float rSquared,
        float alpha
    ) {
        const float oneMinusRSquared = 1.0f - rSquared;
        if (oneMinusRSquared <= 0.0f) {
            return 0.0f;
        }
        return alpha * beta * std::log(oneMinusRSquared);
    }


    static void calculateProjectionGradients() {
    }
}
