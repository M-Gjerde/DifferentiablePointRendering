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

    inline void atomicAddFloat2(float2 &destination, const float2 &valueToAdd) {
        atomicAddFloat(destination.x(), valueToAdd.x());
        atomicAddFloat(destination.y(), valueToAdd.y());
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

        /*
        const float3 d = rayDirection;
        float3 grad_rt = canonicalNormalWorld / denom;
        float3x3 grad_z = outerProduct(d, grad_rt);

        const float3 duDPk = (1 / su) * tangentUWorld * (grad_z - identity3x3());
        const float3 dvDPk = (1 / sv) * tangentVWorld * (grad_z - identity3x3());
        */

        const float tuDotD = dot(tangentUWorld, rayDirection);
        const float tvDotD = dot(tangentVWorld, rayDirection);

        // du/dp_k and dv/dp_k (3x1 each), from your analytic expression
        const float3 duDPk = ((tuDotD / denom) * canonicalNormalWorld - tangentUWorld) / su;
        const float3 dvDPk = ((tvDotD / denom) * canonicalNormalWorld - tangentVWorld) / sv;


        // duv/dc_pos = (u du/dc + v dv/dc)
        const float3 dUVPosition = (u * duDPk + v * dvDPk);
        return dUVPosition;
    }
    // ----------------- Position gradient (translation of surfel center) -----------------
    inline float3 computeDuvDPositionFull(
        const float3 &tangentUWorld,
        const float3 &tangentVWorld,
        const float3 &canonicalNormalWorld,
        const float3 &y,
        const float3 &x,
        const float3 &pk,
        float u, float v,
        float su, float sv) {

        /*
        const float tuDotD = dot(tangentUWorld, rayDirection);
        const float tvDotD = dot(tangentVWorld, rayDirection);

        // du/dp_k and dv/dp_k (3x1 each), from your analytic expression
        const float3 duDPk = ((tuDotD / denom) * canonicalNormalWorld - tangentUWorld) / su;
        const float3 dvDPk = ((tvDotD / denom) * canonicalNormalWorld - tangentVWorld) / sv;
        */
        // Direction from camera (x) to surfel (y)
        float3 d = x - y;
        const float rayLen = length(d);
        d = d / rayLen;

        const float3x3 I = identity3x3();
        // d(x) derivative wrt origin position
        const float3x3 grad_d_pk =
                1.0f / rayLen * (I - outerProduct(d, d));

        // derivative of intersection parameter

        // rt(x) quotient-rule derivative
        const float num = dot(canonicalNormalWorld, (pk - x));
        const float denom = dot(canonicalNormalWorld, d);

        const float3 grad_num =
                canonicalNormalWorld;

        const float3 grad_denom =
                1.0f / rayLen * canonicalNormalWorld * (I - outerProduct(d, d));

        const float3 grad_rt =
                (grad_num * denom - num * grad_denom) / (denom * denom);

        // Intersection parameter rt to blocker plane
        const float rt =
                dot(canonicalNormalWorld, (pk - x)) /
                dot(canonicalNormalWorld, d);

        // z(x) = x + rt(x) d(x)
        const float3x3 term1 = outerProduct(d, grad_rt);
        const float3x3 term2 = rt * grad_d_pk;
        const float3x3 grad_z = term1 + term2;

        const float3 duDpk = 1 / su * tangentUWorld * (grad_z - I);
        const float3 dvDpk = 1 / sv * tangentVWorld * (grad_z - I);

        // duv/dc_pos = (u du/dc + v dv/dc)
        const float3 dUVPosition = u * duDpk + v * dvDpk;

        return dUVPosition;
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

    inline float computeSmoothedBetaFactorBSDF(float beta_param, float r2, float alpha, float opacity) {
        float beta = 4.0f * sycl::exp(beta_param);
        float denom = 1.0f - r2;
        const float eps = 1e-3f; // still keep a small epsilon
        denom = sycl::fmax(denom, eps);
        float betaKernelFactor = -2.0f * beta * alpha * opacity / denom;

        return betaKernelFactor;
    }

    inline float computeSmoothedBetaFactor(float beta_param, float r2, float alpha) {
        float beta = 4.0f * sycl::exp(beta_param);
        float denom = 1.0f - r2;
        const float eps = 1e-3f; // still keep a small epsilon
        denom = sycl::fmax(denom, eps);
        float betaKernelFactor = (beta * alpha * 2.0f )/ denom;

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
        return (-2 * beta * alpha * opacity / (1 - r2)) * float2{dUdVdScale.x() * u, dUdVdScale.y() * v};
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


    SYCL_EXTERNAL inline void accumulateTransmittanceGradientsAlongRay(
        const RayState &rayState,
        const WorldHit &worldHit,
        const GPUSceneBuffers &scene,
        const DeviceSurfacePhotonMapGrid &photonMap,
        bool writeDebugImages,
        const PointGradients &gradients,
        const DebugImages &debugImage,
        uint32_t debugIndex = UINT32_MAX
    ) {
        if (!worldHit.hit || worldHit.splatEventCount == 0) {
            return;
        }

        // Cost weighting: photon-map radiance for this segment
        float3 backgroundRadianceRGB =
                estimateRadianceFromPhotonMap(worldHit, scene, photonMap);


        struct LocalTerm {
            float alpha{};
            float betaKernel{};
            float eta{};
            float r2{};
            float3 dAlphaDPos;
            float3 dAlphaDtU;
            float3 dAlphaDtV;
            float dAlphaDsu{};
            float dAlphaDsv{};
            float dAlphaDbeta{};
            uint32_t primitiveIndex{};
        };

        LocalTerm localTerms[kMaxSplatEvents];
        int validCount = 0;
        float tau = 1.0f; // product over (1 - eta * alpha)

        // Build local terms and τ
        for (size_t eventIndex = 0; eventIndex < worldHit.splatEventCount; ++eventIndex) {
            if (validCount >= kMaxSplatEvents) {
                break;
            }
            const SplatEvent &splatEvent = worldHit.splatEvents[eventIndex];
            const Point &surfel = scene.points[splatEvent.primitiveIndex];

            if (splatEvent.primitiveIndex != debugIndex && debugIndex != UINT32_MAX)
                continue;

            //if (splatEvent.primitiveIndex != 0)
            //    continue;

            const float3 canonicalNormalWorld =
                    normalize(cross(surfel.tanU, surfel.tanV));

            const float2 uv = phiInverse(splatEvent.hitWorld, surfel);
            const float u = uv.x();
            const float v = uv.y();
            const float r2 = u * u + v * v;
            const float alpha = splatEvent.alpha;
            const float su = surfel.scale.x();
            const float sv = surfel.scale.y();

            const float3 dUvDPosition =
                    computeDuvDPosition(
                        surfel.tanU,
                        surfel.tanV,
                        canonicalNormalWorld,
                        rayState.ray.direction,
                        u, v,
                        su, sv
                    );

            float3 dUdTu, dVdTu, dUdTv, dVdTv;
            computeFullDuDvWrtTangents(
                rayState.ray.origin,
                rayState.ray.direction,
                surfel.position,
                splatEvent.hitWorld,
                surfel.tanU,
                surfel.tanV,
                su, sv,
                dUdTu, dVdTu, dUdTv, dVdTv
            );

            float3 dFsDtU = (u * dUdTu + v * dVdTu);
            float3 dFsDtV = (u * dUdTv + v * dVdTv);

            const float betaKernelFactor = -surfel.opacity * computeSmoothedBetaFactor(surfel.beta, r2, alpha);
            float3 dFsDtUBeta = betaKernelFactor * dFsDtU;
            float3 dFsDtVBeta = betaKernelFactor * dFsDtV;

            const float3 dUdVdScale =
                    computeDuvDScale(u, v, su, sv);

            const float3 dFsDPosition = betaKernelFactor * dUvDPosition;


            float3 dFsDsusv = betaKernelFactor * dUdVdScale;

            const float dAlphaDbeta =
                    alpha * betaKernel(surfel.beta) *
                    sycl::log(1.0f - r2) *
                    surfel.opacity;

            const float oneMinusAlpha = 1.0f - alpha * surfel.opacity;
            if (oneMinusAlpha <= 1e-6f) {
                continue;
            }

            auto &localTerm = localTerms[validCount++];
            localTerm.alpha = alpha;
            localTerm.eta = surfel.opacity;
            localTerm.r2 = r2;
            localTerm.betaKernel = betaKernel(surfel.beta);
            localTerm.dAlphaDPos = dFsDPosition;
            localTerm.dAlphaDtU = dFsDtUBeta;
            localTerm.dAlphaDtV = dFsDtVBeta;
            localTerm.dAlphaDsu = dFsDsusv.x();
            localTerm.dAlphaDsv = dFsDsusv.y();
            localTerm.dAlphaDbeta = dAlphaDbeta;
            localTerm.primitiveIndex = splatEvent.primitiveIndex;

            tau *= oneMinusAlpha;
        }

        if (validCount == 0) {
            return;
        }

        float3 adjointWeight = rayState.pathThroughput;
        for (int localIndex = 0; localIndex < validCount; ++localIndex) {
            const LocalTerm &localTerm = localTerms[localIndex];
            const uint32_t primitiveIndex = localTerm.primitiveIndex;
            if (primitiveIndex != debugIndex && debugIndex != UINT32_MAX)
                continue;

            const float alpha = localTerm.alpha;
            const float eta = localTerm.eta;

            const float tauLocal = 1.0f - eta * alpha;
            if (tauLocal <= 1e-6f) {
                continue;
            }
            const float inverseTauLocal = 1.0f / tauLocal;

            const float3 dEtaAlphaDPos = localTerm.dAlphaDPos;
            const float3 dEtaAlphaDtU = localTerm.dAlphaDtU;
            const float3 dEtaAlphaDtV = localTerm.dAlphaDtV;
            const float dEtaAlphaDsu = localTerm.dAlphaDsu;
            const float dEtaAlphaDsv = localTerm.dAlphaDsv;
            const float dEtaAlphaDbeta = localTerm.dAlphaDbeta;

            const float minusTauInverseTauLocal = -tau * eta * inverseTauLocal;

            const float3 dTauDPos = minusTauInverseTauLocal * dEtaAlphaDPos;
            const float3 dTauDtU = minusTauInverseTauLocal * dEtaAlphaDtU;
            const float3 dTauDtV = minusTauInverseTauLocal * dEtaAlphaDtV;
            const float dTauDsu = minusTauInverseTauLocal * dEtaAlphaDsu;
            const float dTauDsv = minusTauInverseTauLocal * dEtaAlphaDsv;
            const float dTauDbeta = minusTauInverseTauLocal * dEtaAlphaDbeta;

            const float dTauDeta = -tau * alpha * inverseTauLocal;

            const float R = adjointWeight[0] * backgroundRadianceRGB[0];
            const float G = adjointWeight[1] * backgroundRadianceRGB[1];
            const float B = adjointWeight[2] * backgroundRadianceRGB[2];

            const float3 gradCPosR = R * dTauDPos;
            const float3 gradCPosG = G * dTauDPos;
            const float3 gradCPosB = B * dTauDPos;

            const float3 gradCTanUR = R * dTauDtU;
            const float3 gradCTanUG = G * dTauDtU;
            const float3 gradCTanUB = B * dTauDtU;

            const float3 gradCTanVR = R * dTauDtV;
            const float3 gradCTanVG = G * dTauDtV;
            const float3 gradCTanVB = B * dTauDtV;

            const float gradCScaleUR = R * dTauDsu;
            const float gradCScaleUG = G * dTauDsu;
            const float gradCScaleUB = B * dTauDsu;

            const float gradCScaleVR = R * dTauDsv;
            const float gradCScaleVG = G * dTauDsv;
            const float gradCScaleVB = B * dTauDsv;

            const float gradCOpacityR = R * dTauDeta;
            const float gradCOpacityG = G * dTauDeta;
            const float gradCOpacityB = B * dTauDeta;

            const float gradCBetaR = R * dTauDbeta;
            const float gradCBetaG = G * dTauDbeta;
            const float gradCBetaB = B * dTauDbeta;

            float3 gradPosition = gradCPosR + gradCPosG + gradCPosB;
            atomicAddFloat3(
                gradients.gradPosition[primitiveIndex],
                gradPosition
            );
            atomicAddFloat3(
                gradients.gradTanU[primitiveIndex],
                gradCTanUR + gradCTanUG + gradCTanUB
            );
            atomicAddFloat3(
                gradients.gradTanV[primitiveIndex],
                gradCTanVR + gradCTanVG + gradCTanVB
            );
            atomicAddFloat(
                gradients.gradScale[primitiveIndex].x(),
                gradCScaleUR + gradCScaleUG + gradCScaleUB
            );
            atomicAddFloat(
                gradients.gradScale[primitiveIndex].y(),
                gradCScaleVR + gradCScaleVG + gradCScaleVB
            );
            atomicAddFloat(
                gradients.gradOpacity[primitiveIndex],
                gradCOpacityR + gradCOpacityG + gradCOpacityB
            );
            atomicAddFloat(
                gradients.gradBeta[primitiveIndex],
                gradCBetaR + gradCBetaG + gradCBetaB
            );

            if (writeDebugImages) {
                const Point &surfel = scene.points[primitiveIndex];
                float3 rotationAxis = float3{0.0f, 1.0f, 0.0f};

                const float3 dBsdfWorld = float3{
                    dot(gradCTanUR, cross(rotationAxis, surfel.tanU)) +
                    dot(gradCTanVR, cross(rotationAxis, surfel.tanV)),

                    dot(gradCTanUG, cross(rotationAxis, surfel.tanU)) +
                    dot(gradCTanVG, cross(rotationAxis, surfel.tanV)),

                    dot(gradCTanUB, cross(rotationAxis, surfel.tanU)) +
                    dot(gradCTanVB, cross(rotationAxis, surfel.tanV))
                };
                float3 parameterAxisX = float3{1.0f, 0.0f, 0.0f};
                float3 parameterAxisY = float3{0.0f, 1.0f, 0.0f};
                float3 parameterAxisZ = float3{0.0f, 0.0f, 1.0f};

                const float dCdpRX = dot(gradPosition, parameterAxisX);
                const float4 posScalarRGBX{dCdpRX};
                atomicAddFloat4ToImage(
                    &debugImage.framebuffer_posX[rayState.pixelIndex],
                    posScalarRGBX
                );

                const float dCdpRY = dot(gradPosition, parameterAxisY);
                const float4 posScalarRGBY{dCdpRY};
                atomicAddFloat4ToImage(
                    &debugImage.framebuffer_posY[rayState.pixelIndex],
                    posScalarRGBY
                );

                const float dCdpRZ = dot(gradPosition, parameterAxisZ);
                const float4 posScalarRGBZ{dCdpRZ};
                atomicAddFloat4ToImage(
                    &debugImage.framebuffer_posZ[rayState.pixelIndex],
                    posScalarRGBZ
                );

                const float4 rotScalarRGB{
                    dBsdfWorld.x(), dBsdfWorld.y(), dBsdfWorld.z(), 0.0f
                };

                const float4 scaleScalarRGB{
                    gradCScaleUR, gradCScaleUG, gradCScaleUB, 0.0f
                };

                const float4 opacityScalarRGB{
                    gradCOpacityR, gradCOpacityG, gradCOpacityB, 0.0f
                };

                const float4 betaScalarRGB{
                    gradCBetaR, gradCBetaG, gradCBetaB, 0.0f
                };

                atomicAddFloat4ToImage(
                    &debugImage.framebuffer_rot[rayState.pixelIndex],
                    rotScalarRGB
                );
                atomicAddFloat4ToImage(
                    &debugImage.framebuffer_scale[rayState.pixelIndex],
                    scaleScalarRGB
                );
                atomicAddFloat4ToImage(
                    &debugImage.framebuffer_opacity[rayState.pixelIndex],
                    opacityScalarRGB
                );
                atomicAddFloat4ToImage(
                    &debugImage.framebuffer_beta[rayState.pixelIndex],
                    betaScalarRGB
                );
            }
        }
    }


    SYCL_EXTERNAL inline void accumulateBsdfGradientsAtScatterSurfel(
        const RayState &rayState,
        const WorldHit &scatterHit,
        const GPUSceneBuffers &scene,
        const DeviceSurfacePhotonMapGrid &photonMap,
        const PointGradients &gradients,
        const DebugImages &debugImage,
        bool renderDebugGradientImage,
        uint32_t debugIndex = UINT32_MAX,
        bool isWatched = false
    ) {
        if (!scatterHit.hit || scatterHit.splatEventCount == 0) {
            return;
        }

        const InstanceRecord &scatterInstance = scene.instances[scatterHit.instanceIndex];
        if (scatterInstance.geometryType != GeometryType::PointCloud) {
            return;
        }

        // Terminal surfel of this scatter event
        const uint32_t terminalPrimitiveIndex = scatterHit.primitiveIndex;
        if (terminalPrimitiveIndex != debugIndex && debugIndex != UINT32_MAX)
            return;

        const SplatEvent &terminalSplatEvent =
                scatterHit.splatEvents[scatterHit.splatEventCount - 1];


        const Point &surfel = scene.points[terminalPrimitiveIndex];

        const float3 canonicalNormalWorld =
                normalize(cross(surfel.tanU, surfel.tanV));

        const Ray &ray = rayState.ray;
        const float3 rayDirection = ray.direction;
        const float3 hitWorld = terminalSplatEvent.hitWorld;

        const float2 uv = phiInverse(hitWorld, surfel);
        const float u = uv.x();
        const float v = uv.y();
        const float r2 = u * u + v * v;
        const float alpha = terminalSplatEvent.alpha;
        const float su = surfel.scale.x();
        const float sv = surfel.scale.y();

        // dα/d(position) via d(u,v)/d(position)
        const float3 dUvDPosition =
                computeDuvDPosition(
                    surfel.tanU,
                    surfel.tanV,
                    canonicalNormalWorld,
                    rayState.ray.direction,
                    u, v,
                    su, sv);

        float3 dUdTu, dVdTu, dUdTv, dVdTv;
        computeFullDuDvWrtTangents(
            ray.origin,
            ray.direction,
            surfel.position,
            hitWorld,
            surfel.tanU,
            surfel.tanV,
            su, sv,
            dUdTu, dVdTu, dUdTv, dVdTv
        );

        // dudv derivatives wrt tangents
        float3 dFsDtU = (u * dUdTu + v * dVdTu);
        float3 dFsDtV = (u * dUdTv + v * dVdTv);

        const float betaKernelFactor =
                -surfel.opacity * computeSmoothedBetaFactor(surfel.beta, r2, alpha);
        float3 bsdfGradTanU = betaKernelFactor * dFsDtU;
        float3 bsdfGradTanV = betaKernelFactor * dFsDtV;

        const float3 dUdVdScale =
                computeDuvDScale(u, v, su, sv);

        float beta = 4.0f * sycl::exp(surfel.beta);

        const float3 bsdfGradPosition = betaKernelFactor * dUvDPosition * 0.000000022;
        const float3 dFsDsusv = betaKernelFactor * dUdVdScale;

        const float bsdfGradScaleU = dFsDsusv.x();
        const float bsdfGradScaleV = dFsDsusv.y();

        if (isWatched)
            int debug = 1;
        const float bsdfGradAlbedo = alpha * surfel.opacity;
        const float3 bsdfGradOpacity = alpha * surfel.albedo;
        const float bsdfGradBeta =
                alpha * betaKernel(surfel.beta) *
                sycl::log(1.0f - r2) *
                surfel.opacity;

        // Surfel-side radiance
        const float3 surfelRadianceRGB =
                estimateSurfelRadianceFromPhotonMap(
                    terminalSplatEvent,
                    ray.direction,
                    scene,
                    photonMap,
                    false,
                    true,
                    true
                );

        const float3 pathAdjoint = rayState.pathThroughput;

        // Per-channel adjoint * τ * radiance
        const float R = pathAdjoint[0] * surfelRadianceRGB[0] + 1.0;
        const float G = pathAdjoint[1] * surfelRadianceRGB[1] + 1.0;
        const float B = pathAdjoint[2] * surfelRadianceRGB[2] + 1.0;

        // Position gradients
        const float3 gradCPosR = R * bsdfGradPosition;
        const float3 gradCPosG = G * bsdfGradPosition;
        const float3 gradCPosB = B * bsdfGradPosition;

        // Tangent gradients
        const float3 gradCTanUR = R * bsdfGradTanU;
        const float3 gradCTanUG = G * bsdfGradTanU;
        const float3 gradCTanUB = B * bsdfGradTanU;

        const float3 gradCTanVR = R * bsdfGradTanV;
        const float3 gradCTanVG = G * bsdfGradTanV;
        const float3 gradCTanVB = B * bsdfGradTanV;

        // Scale gradients
        const float gradCScaleUR = R * bsdfGradScaleU;
        const float gradCScaleUG = G * bsdfGradScaleU;
        const float gradCScaleUB = B * bsdfGradScaleU;

        const float gradCScaleVR = R * bsdfGradScaleV;
        const float gradCScaleVG = G * bsdfGradScaleV;
        const float gradCScaleVB = B * bsdfGradScaleV;

        // Albedo gradients
        const float gradCAlbedoR = R * bsdfGradAlbedo;
        const float gradCAlbedoG = G * bsdfGradAlbedo;
        const float gradCAlbedoB = B * bsdfGradAlbedo;

        // Opacity gradients (RGB)
        const float gradCOpacityR = R * bsdfGradOpacity[0];
        const float gradCOpacityG = G * bsdfGradOpacity[1];
        const float gradCOpacityB = B * bsdfGradOpacity[2];

        // Beta gradients
        const float gradCBetaR = R * bsdfGradBeta;
        const float gradCBetaG = G * bsdfGradBeta;
        const float gradCBetaB = B * bsdfGradBeta;

        // Accumulate into parameter gradients
        const uint32_t primitiveIndex = terminalPrimitiveIndex;

        atomicAddFloat3(
            gradients.gradPosition[primitiveIndex],
            gradCPosR + gradCPosG + gradCPosB
        );
        atomicAddFloat3(
            gradients.gradTanU[primitiveIndex],
            gradCTanUR + gradCTanUG + gradCTanUB
        );
        atomicAddFloat3(
            gradients.gradTanV[primitiveIndex],
            gradCTanVR + gradCTanVG + gradCTanVB
        );
        atomicAddFloat(
            gradients.gradScale[primitiveIndex].x(),
            gradCScaleUR + gradCScaleUG + gradCScaleUB
        );
        atomicAddFloat(
            gradients.gradScale[primitiveIndex].y(),
            gradCScaleVR + gradCScaleVG + gradCScaleVB
        );
        atomicAddFloat(
            gradients.gradOpacity[primitiveIndex],
            gradCOpacityR + gradCOpacityG + gradCOpacityB
        );
        atomicAddFloat(
            gradients.gradBeta[primitiveIndex],
            gradCBetaR + gradCBetaG + gradCBetaB
        );

        float3 gradColorValue{gradCAlbedoR, gradCAlbedoG, gradCAlbedoB};
        atomicAddFloat3(gradients.gradAlbedo[primitiveIndex], gradColorValue);

        // Debug images
        if (renderDebugGradientImage) {
            float3 rotationAxis = float3{0.0f, 1.0f, 0.0f};
            const float3 dBsdfWorld = float3{
                dot(gradCTanUR, cross(rotationAxis, surfel.tanU)) +
                dot(gradCTanVR, cross(rotationAxis, surfel.tanV)),

                dot(gradCTanUG, cross(rotationAxis, surfel.tanU)) +
                dot(gradCTanVG, cross(rotationAxis, surfel.tanV)),

                dot(gradCTanUB, cross(rotationAxis, surfel.tanU)) +
                dot(gradCTanVB, cross(rotationAxis, surfel.tanV))
            };
            float3 parameterAxis = float3{0.0f, 1.0f, 0.0f};

            const float dCdpR = dot(gradCPosR, parameterAxis);
            const float dCdpG = dot(gradCPosG, parameterAxis);
            const float dCdpB = dot(gradCPosB, parameterAxis);
            const float4 posScalarRGB{dCdpR, dCdpG, dCdpB, 0.0f};

            const float4 rotScalarRGB{
                dBsdfWorld.x(), dBsdfWorld.y(), dBsdfWorld.z(), 0.0f
            };

            const float4 scaleScalarRGB{
                gradCScaleUR, gradCScaleUG, gradCScaleUB, 0.0f
            };

            const float4 opacityScalarRGB{
                gradCOpacityR, gradCOpacityG, gradCOpacityB, 0.0f
            };

            const float4 albedoScalarRGB{
                gradCAlbedoR, gradCAlbedoG, gradCAlbedoB, 0.0f
            };

            const float4 betaScalarRGB{
                gradCBetaR, gradCBetaG, gradCBetaB, 0.0f
            };

            const uint32_t pixelIndex = rayState.pixelIndex;

            atomicAddFloat4ToImage(
                &debugImage.framebuffer_posX[pixelIndex],
                posScalarRGB
            );
            atomicAddFloat4ToImage(
                &debugImage.framebuffer_rot[pixelIndex],
                rotScalarRGB
            );
            atomicAddFloat4ToImage(
                &debugImage.framebuffer_scale[pixelIndex],
                scaleScalarRGB
            );
            atomicAddFloat4ToImage(
                &debugImage.framebuffer_opacity[pixelIndex],
                opacityScalarRGB
            );
            atomicAddFloat4ToImage(
                &debugImage.framebuffer_albedo[pixelIndex],
                albedoScalarRGB
            );
            atomicAddFloat4ToImage(
                &debugImage.framebuffer_beta[pixelIndex],
                betaScalarRGB
            );
        }
    }


    inline void shadowRay(const GPUSceneBuffers &scene, const RayState &rayState, const WorldHit &worldHit,
                          const PointGradients &gradients, const DebugImages &debugImage,
                          const DeviceSurfacePhotonMapGrid &photonMap, rng::Xorshift128 &rng,
                          bool renderDebugGradientImages,
                          uint32_t numShadowRays = 1,
                          uint32_t debugIndex = UINT32_MAX,
                          bool debugBreakFlag = false) {

        for (int i = 0; i < numShadowRays; ++i) {
            AreaLightSample ls = sampleMeshAreaLightReuse(scene, rng);
            // Direction to the sampled emitter point
            const float3 toLightVector = ls.positionW - rayState.ray.origin;
            const float distanceToLight = length(toLightVector);
            if (distanceToLight > 1e-6f) {
                const float3 lightDirection = toLightVector / distanceToLight;
                // Cosines
                const float3 shadingNormalW = rayState.ray.normal;
                const float cosThetaSurface = sycl::max(0.0f, dot(shadingNormalW, lightDirection));
                const float cosThetaLight = sycl::max(0.0f, dot(ls.normalW, -lightDirection));


                if (cosThetaSurface != 0.0f && cosThetaLight != 0.0f) {
                    const float r2 = distanceToLight * distanceToLight;
                    const float geometryTerm = (cosThetaSurface * cosThetaLight) / r2;
                    // PDFs from the sampler
                    const float pdfArea = ls.pdfArea; // area-domain, world area
                    const float pdfLight = ls.pdfSelectLight; // 1 / lightCount
                    // Unbiased NEE estimator (area sampling):
                    const float invPdf = 1.0f / (pdfLight * pdfArea);
                    float oneOverNumRays = 1.0f / static_cast<float>(numShadowRays);

                    RayState shadowRayState = rayState;
                    Ray shadowRay{
                        rayState.ray.origin + (rayState.ray.normal * 1e-4f), lightDirection
                    };
                    shadowRayState.ray = shadowRay;
                    shadowRayState.pathThroughput =
                            rayState.pathThroughput * geometryTerm * invPdf * oneOverNumRays;

                    // BRDF
                    WorldHit shadowWorldHit{};
                    intersectScene(shadowRayState.ray, &shadowWorldHit, scene, rng,
                                   RayIntersectMode::Transmit);
                    buildIntersectionNormal(scene, shadowWorldHit);

                    accumulateTransmittanceGradientsAlongRay(shadowRayState, shadowWorldHit, scene, photonMap,
                                                             renderDebugGradientImages, gradients,
                                                             debugImage, debugIndex);
                    if (debugBreakFlag)
                        int debug = 1;
                }
            }
        }
    }

    SYCL_EXTERNAL inline void accumulateTransmittanceGradientsAlongRayAttachedOriginSelf(
        const RayState &rayState,
        const WorldHit &worldHit,
        const GPUSceneBuffers &scene,
        const DeviceSurfacePhotonMapGrid &photonMap,
        bool writeDebugImages,
        const PointGradients &gradients,
        uint32_t originPrimitiveIndex, // surfel controlling ray origin (attached origin)
        const DebugImages &debugImage,
        uint32_t debugIndex = UINT32_MAX,
        bool isWatched = false
    ) {
        if (!worldHit.hit || worldHit.splatEventCount == 0) {
            return;
        }

        const float3 backgroundRadianceRGB =
                estimateRadianceFromPhotonMap(worldHit, scene, photonMap);

        struct LocalTerm {
            float alpha{};
            float betaKernel{};
            float eta{};
            float r2{};

            // Existing blocker-parameter derivatives (unused here but kept for completeness)
            float3 dAlphaDPos;
            float3 dAlphaDtU;
            float3 dAlphaDtV;
            float dAlphaDsu{};
            float dAlphaDsv{};
            float dAlphaDbeta{};

            // NEW: derivative wrt origin position (attached origin)
            float3 dAlphaDOrigin;

            uint32_t primitiveIndex{};
        };

        LocalTerm localTerms[kMaxSplatEvents];
        int validCount = 0;
        float tau = 1.0f;

        // -------------------------------------------------------------------------
        // Build local terms and τ (blockers along the ray)
        // -------------------------------------------------------------------------
        for (size_t eventIndex = 0; eventIndex < worldHit.splatEventCount; ++eventIndex) {
            if (validCount >= kMaxSplatEvents) {
                break;
            }

            const SplatEvent &splatEvent = worldHit.splatEvents[eventIndex];

            if (originPrimitiveIndex != debugIndex && debugIndex != UINT32_MAX) {
                continue;
            }

            // Blocker surfel (intersection surfel; no π-dependence)
            const Point &surfel = scene.points[splatEvent.primitiveIndex];
            const float2 uvInt = phiInverse(splatEvent.hitWorld, surfel);
            const float uInt = uvInt.x();
            const float vInt = uvInt.y();

            const float3 tuInt = surfel.tanU;
            const float3 tvInt = surfel.tanV;
            const float3 pkInt = surfel.position;
            const float suInt = surfel.scale.x();
            const float svInt = surfel.scale.y();

            const float3 intersectionNormal =
                    normalize(cross(surfel.tanU, surfel.tanV));

            // Origin surfel (the one carrying π)
            const Point &originSurfel = scene.points[originPrimitiveIndex];
            const float3 pkOrigin = originSurfel.position;
            const float3 tuOrigin = originSurfel.tanU;
            const float3 tvOrigin = originSurfel.tanV;
            const float suOrigin = originSurfel.scale.x();
            const float svOrigin = originSurfel.scale.y();

            const float3 x = rayState.ray.origin;

            // Direction from origin to fixed light point y
            float3 d = worldHit.hitPositionW - x;
            const float rayLen = length(d);
            if (rayLen <= 1e-6f) {
                continue;
            }
            d = d / rayLen;

            // Intersection parameter rt to blocker plane
            const float rt =
                    dot(intersectionNormal, (pkInt - x)) /
                    dot(intersectionNormal, d);

            const float3x3 I = identity3x3();

            // d(x) derivative wrt origin position
            const float3x3 grad_d_pk =
                    -1.0f / rayLen * (I - outerProduct(d, d));

            // rt(x) quotient-rule derivative
            const float num = dot(intersectionNormal, (pkInt - x));
            const float denom = dot(intersectionNormal, d);

            const float3 grad_num =
                    -intersectionNormal * I;
            const float3 grad_denom =
                    -1.0f / rayLen * intersectionNormal * (I - outerProduct(d, d));

            const float3 grad_rt =
                    (grad_num * denom - num * grad_denom) / (denom * denom);

            // z(x) = x + rt(x) d(x)
            const float3x3 term2 = outerProduct(d, grad_rt);
            const float3x3 term3 = rt * grad_d_pk;
            const float3x3 grad_z = I + term2 + term3;

            const float r2 = uInt * uInt + vInt * vInt;
            const float alpha = splatEvent.alpha;

            // ∂u/∂x and ∂v/∂x for blocker local coords
            const float3 gradU_pk = (1.0f / suInt) * (tuInt * grad_z);
            const float3 gradV_pk = (1.0f / svInt) * (tvInt * grad_z);

            // u ∂u/∂x + v ∂v/∂x
            const float3 dUVDPosition = uInt * gradU_pk + vInt * gradV_pk;

            // Smoothed β-kernel scaling (includes dα/du,dv structure)
            const float betaKernelFactor =
                    computeSmoothedBetaFactor(surfel.beta, r2, alpha);

            // d(η α)/d(origin position)
            const float3 dAlphaDPosition =
                    -surfel.opacity * betaKernelFactor * dUVDPosition;

            const float dAlphaDbeta =
                    alpha * betaKernel(surfel.beta) *
                    sycl::log(1.0f - r2) *
                    surfel.opacity;

            const float oneMinusAlpha = 1.0f - alpha * surfel.opacity;
            if (oneMinusAlpha <= 1e-6f) {
                continue;
            }

            auto &localTerm = localTerms[validCount++];
            localTerm.alpha = alpha;
            localTerm.eta = surfel.opacity;
            localTerm.r2 = r2;
            localTerm.dAlphaDbeta = dAlphaDbeta;
            localTerm.dAlphaDOrigin = dAlphaDPosition;
            localTerm.primitiveIndex = splatEvent.primitiveIndex;

            tau *= oneMinusAlpha;
        }

        if (validCount == 0) {
            return;
        }

        const float3 adjointWeight = rayState.pathThroughput;

        // -------------------------------------------------------------------------
        // Accumulate gradients to origin surfel parameters (position, rotation, scale)
        // -------------------------------------------------------------------------
        for (int localIndex = 0; localIndex < validCount; ++localIndex) {
            const LocalTerm &localTerm = localTerms[localIndex];
            const uint32_t primitiveIndex = localTerm.primitiveIndex;

            const float alpha = localTerm.alpha;
            const float eta = localTerm.eta;

            const float tauLocal = 1.0f - eta * alpha;
            if (tauLocal <= 1e-6f) {
                continue;
            }
            const float inverseTauLocal = 1.0f / tauLocal;

            // d(η α)/d(origin position) for this blocker
            const float3 dEtaAlphaDOrigin = localTerm.dAlphaDOrigin;

            const float R = adjointWeight[0] * backgroundRadianceRGB[0];
            const float G = adjointWeight[1] * backgroundRadianceRGB[1];
            const float B = adjointWeight[2] * backgroundRadianceRGB[2];

            if (originPrimitiveIndex == UINT32_MAX) {
                continue;
            }

            // dτ/d(origin) for this blocker contribution:
            // τ = ∏ (1 - η_i α_i),  ∂τ_i = -∂(η_i α_i)
            // ∂τ ≈ -τ * (1/τ_i) ∂(η_i α_i)  for this i
            const float3 dTauDOrigin =
                    -tau * inverseTauLocal * dEtaAlphaDOrigin;

            // Per-channel contribution to ∂C/∂x (origin position)
            const float3 gradCOriginR = R * dTauDOrigin;
            const float3 gradCOriginG = G * dTauDOrigin;
            const float3 gradCOriginB = B * dTauDOrigin;

            const float3 gradCOrigin =
                    gradCOriginR + gradCOriginG + gradCOriginB; // ∂C/∂x

            // ---------------------------------------------------------------------
            // 1) Position gradient
            // ---------------------------------------------------------------------
            atomicAddFloat3(
                gradients.gradPosition[originPrimitiveIndex],
                gradCOrigin
            );

            // ---------------------------------------------------------------------
            // 2) Rotation/scale gradients for the origin surfel
            // ---------------------------------------------------------------------
            const Point &originSurfel = scene.points[originPrimitiveIndex];
            const float3 tuOrigin = originSurfel.tanU;
            const float3 tvOrigin = originSurfel.tanV;
            const float suOrigin = originSurfel.scale.x();
            const float svOrigin = originSurfel.scale.y();

            const float2 uvOrigin = phiInverse(rayState.ray.origin, originSurfel);
            const float uOrigin = uvOrigin.x();
            const float vOrigin = uvOrigin.y();

            // ∂x/∂tu = su * u * I,   ∂x/∂tv = sv * v * I
            const float suUOrigin = suOrigin * uOrigin;
            const float svVOrigin = svOrigin * vOrigin;

            // Per-channel rotation gradients
            const float3 gradTanUOriginR = gradCOriginR * suUOrigin;
            const float3 gradTanUOriginG = gradCOriginG * suUOrigin;
            const float3 gradTanUOriginB = gradCOriginB * suUOrigin;

            const float3 gradTanVOriginR = gradCOriginR * svVOrigin;
            const float3 gradTanVOriginG = gradCOriginG * svVOrigin;
            const float3 gradTanVOriginB = gradCOriginB * svVOrigin;

            const float3 gradTanUOrigin =
                    gradTanUOriginR + gradTanUOriginG + gradTanUOriginB;
            const float3 gradTanVOrigin =
                    gradTanVOriginR + gradTanVOriginG + gradTanVOriginB;

            atomicAddFloat3(
                gradients.gradTanU[originPrimitiveIndex],
                gradTanUOrigin
            );
            atomicAddFloat3(
                gradients.gradTanV[originPrimitiveIndex],
                gradTanVOrigin
            );

            // ∂x/∂su = u * tu,   ∂x/∂sv = v * tv
            const float dCdsuR = dot(gradCOriginR, tuOrigin) * uOrigin;
            const float dCdsuG = dot(gradCOriginG, tuOrigin) * uOrigin;
            const float dCdsuB = dot(gradCOriginB, tuOrigin) * uOrigin;

            const float dCdsvR = dot(gradCOriginR, tvOrigin) * vOrigin;
            const float dCdsvG = dot(gradCOriginG, tvOrigin) * vOrigin;
            const float dCdsvB = dot(gradCOriginB, tvOrigin) * vOrigin;

            const float dCdsu = dCdsuR + dCdsuG + dCdsuB;
            const float dCdsv = dCdsvR + dCdsvG + dCdsvB;

            atomicAddFloat2(
                gradients.gradScale[originPrimitiveIndex],
                float2{dCdsu, dCdsv}
            );

            if (isWatched) {
                int debug = 1;
            }

            // ---------------------------------------------------------------------
            // Debug images (per-channel scalars)
            // ---------------------------------------------------------------------
            if (writeDebugImages) {
                // Position debug (project ∂C/∂x onto some axis, e.g. z-axis)
                const float3 parameterAxis = float3{0.0f, 0.0f, 1.0f};
                const float dCdpR = dot(gradCOriginR, parameterAxis);
                const float dCdpG = dot(gradCOriginG, parameterAxis);
                const float dCdpB = dot(gradCOriginB, parameterAxis);
                const float4 posScalarRGB{dCdpR, dCdpG, dCdpB, 0.0f};

                // Rotation debug:
                // dC/dθ_c = gradC_tu_c · (a × tu) + gradC_tv_c · (a × tv)
                const float3 rotationAxis = float3{0.0f, 1.0f, 0.0f};

                const float3 tuAxisCross = cross(rotationAxis, tuOrigin);
                const float3 tvAxisCross = cross(rotationAxis, tvOrigin);

                const float dRotR =
                        dot(gradTanUOriginR, tuAxisCross) +
                        dot(gradTanVOriginR, tvAxisCross);
                const float dRotG =
                        dot(gradTanUOriginG, tuAxisCross) +
                        dot(gradTanVOriginG, tvAxisCross);
                const float dRotB =
                        dot(gradTanUOriginB, tuAxisCross) +
                        dot(gradTanVOriginB, tvAxisCross);

                const float4 rotScalarRGB{dRotR, dRotG, dRotB, 0.0f};

                // Scale debug: pack su, sv for visualization
                const float4 scaleScalarRGB{dCdsu, dCdsv, 0.0f, 0.0f};

                // Opacity/beta: no attached-origin contribution here; zeros
                const float4 opacityScalarRGB{0.0f, 0.0f, 0.0f, 0.0f};
                const float4 betaScalarRGB{0.0f, 0.0f, 0.0f, 0.0f};

                atomicAddFloat4ToImage(
                    &debugImage.framebuffer_posX[rayState.pixelIndex],
                    posScalarRGB
                );
                atomicAddFloat4ToImage(
                    &debugImage.framebuffer_rot[rayState.pixelIndex],
                    rotScalarRGB
                );
                atomicAddFloat4ToImage(
                    &debugImage.framebuffer_scale[rayState.pixelIndex],
                    scaleScalarRGB
                );
                atomicAddFloat4ToImage(
                    &debugImage.framebuffer_opacity[rayState.pixelIndex],
                    opacityScalarRGB
                );
                atomicAddFloat4ToImage(
                    &debugImage.framebuffer_beta[rayState.pixelIndex],
                    betaScalarRGB
                );
            }
        }
    }


    inline void shadowRayAttachedOriginSelf(const GPUSceneBuffers &scene, const RayState &rayState,
                                            const WorldHit &worldHit, SplatEvent &splatEvent,
                                            const PointGradients &gradients, const DebugImages &debugImage,
                                            const DeviceSurfacePhotonMapGrid &photonMap, rng::Xorshift128 &rng,
                                            bool renderDebugGradientImages,
                                            uint32_t numShadowRays = 1,
                                            uint32_t debugIndex = UINT32_MAX,
                                            bool isWatched = false) {
        const InstanceRecord &instance = scene.instances[worldHit.instanceIndex];

        GPUMaterial material;
        switch (instance.geometryType) {
            case GeometryType::Mesh:
                material = scene.materials[instance.materialIndex];
                break;
            case GeometryType::PointCloud:
                material.baseColor = scene.points[worldHit.primitiveIndex].albedo;
                break;
            case GeometryType::InvalidType:
                break;
        }

        for (int i = 0; i < numShadowRays; ++i) {
            AreaLightSample ls = sampleMeshAreaLightReuse(scene, rng);
            // Direction to the sampled emitter point
            const float3 toLightVector = ls.positionW - worldHit.hitPositionW;
            const float distanceToLight = length(toLightVector);
            if (distanceToLight > 1e-6f) {
                const float3 lightDirection = toLightVector / distanceToLight;
                // Cosines
                const float3 shadingNormalW = worldHit.geometricNormalW;
                const float cosThetaSurface = sycl::max(0.0f, dot(shadingNormalW, lightDirection));
                const float cosThetaLight = sycl::max(0.0f, dot(ls.normalW, -lightDirection));


                if (cosThetaSurface != 0.0f && cosThetaLight != 0.0f) {
                    const float r2 = distanceToLight * distanceToLight;
                    const float geometryTerm = (cosThetaSurface * cosThetaLight) / r2;
                    // PDFs from the sampler
                    const float pdfArea = ls.pdfArea; // area-domain, world area
                    const float pdfLight = ls.pdfSelectLight; // 1 / lightCount
                    // Unbiased NEE estimator (area sampling):
                    const float invPdf = 1.0f / (pdfLight * pdfArea);
                    float oneOverNumRays = 1.0f / static_cast<float>(numShadowRays);

                    Ray shadowRay{
                        worldHit.hitPositionW + (worldHit.geometricNormalW * 1e-6f), lightDirection
                    };
                    RayState shadowRayState = rayState;
                    shadowRayState.ray = shadowRay;

                    // apply bsdf, tau and cosine:
                    float cosine = fabs(dot(rayState.ray.direction, worldHit.geometricNormalW));

                    shadowRayState.pathThroughput =
                            rayState.pathThroughput * geometryTerm * invPdf * cosine * material.baseColor * M_1_PIf *
                            oneOverNumRays;

                    // BRDF
                    WorldHit shadowWorldHit{};
                    intersectScene(shadowRayState.ray, &shadowWorldHit, scene, rng,
                                   RayIntersectMode::Transmit);

                    accumulateTransmittanceGradientsAlongRayAttachedOriginSelf(
                        shadowRayState, shadowWorldHit, scene, photonMap,
                        renderDebugGradientImages, gradients, splatEvent.primitiveIndex,
                        debugImage, debugIndex, isWatched);
                }
            }
        }
    }
}
