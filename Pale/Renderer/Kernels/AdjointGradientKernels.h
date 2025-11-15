#pragma once
#include "IntersectionKernels.h"
#include "KernelHelpers.h"
#include "Renderer/GPUDataTypes.h"


namespace Pale {
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

    inline float3 shadowRays(const GPUSceneBuffers &scene,
                             const RayState &rayState, const DeviceSurfacePhotonMapGrid &photonMap,
                             rng::Xorshift128 &rng128) {
        constexpr float epsilon = 1e-6f;
        float3 d_grad_pos(0.0f);


        AreaLightSample ls = sampleMeshAreaLightReuse(scene, rng128);
        // Direction to the sampled emitter point
        const float3 toLightVector = ls.positionW - rayState.ray.origin;
        const float distanceToLight = length(toLightVector);
        if (distanceToLight > 1e-6f) {
            const float3 lightDirection = toLightVector / distanceToLight;
            // Cosines
            const float3 shadingNormalW = rayState.ray.normal; // or your surfel normal
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
                const float3 neeContribution =
                        rayState.pathThroughput * geometryTerm * invPdf;

                Ray shadowRay{rayState.ray.origin, lightDirection};
                RayState shadowRayState = rayState;
                shadowRayState.ray = shadowRay;
                shadowRayState.pathThroughput = neeContribution;

                // BRDF
                WorldHit shadowWorldHit{};
                intersectScene(shadowRayState.ray, &shadowWorldHit, scene, rng128,
                               RayIntersectMode::Transmit);

                if (shadowWorldHit.hit) {
                    if (shadowWorldHit.splatEventCount > 0) {
                        const Ray &ray = shadowRayState.ray;
                        const float3 x = ray.origin;
                        const float3 y = shadowWorldHit.hitPositionW;

                        // Cost weighting: keep RGB if your loss is RGB; else reduce at end
                        const float3 backgroundRadianceRGB = estimateRadianceFromPhotonMap(
                            shadowWorldHit, scene, photonMap);
                        const float L_bg = luminance(backgroundRadianceRGB);

                        // Collect all alpha_i and d(alpha_i)/dPi for this segment
                        struct LocalTerm {
                            float alpha{};
                            float3 dAlphaDc;
                        };

                        LocalTerm localTerms[kMaxSplatEvents];
                        int validCount = 0;

                        float tau = 1.0f; // full product over all (1 - alpha_i)

                        for (size_t eventIdx = 0; eventIdx < shadowWorldHit.splatEventCount; ++eventIdx) {
                            if (validCount >= kMaxSplatEvents)
                                break; // safety, or assert

                            const auto &splatEvent = shadowWorldHit.splatEvents[eventIdx];

                            const Point &surfel = scene.points[splatEvent.primitiveIndex];

                            const float3 canonicalNormalW =
                                    normalize(cross(surfel.tanU, surfel.tanV));

                            const float denom = dot(canonicalNormalW, ray.direction);
                            if (sycl::fabs(denom) <= 1e-4f) {
                                // Grazing / parallel; skip this surfel for the transmission gradient
                                continue;
                            }

                            const float2 uv = phiInverse(splatEvent.hitWorld, surfel);
                            const float alpha = splatEvent.alpha;

                            // Optional: guard against extreme alpha for numerical stability
                            const float clampedAlpha = sycl::clamp(alpha, 0.0f, 1.0f - 1e-6f);

                            const float tuDotD = dot(surfel.tanU, ray.direction);
                            const float tvDotD = dot(surfel.tanV, ray.direction);
                            const float su = surfel.scale.x();
                            const float sv = surfel.scale.y();

                            // Reuse the coordinate gradients from your BRDF derivation
                            const float3 duDc =
                                    ((tuDotD / denom) * canonicalNormalW - surfel.tanU) / su;
                            const float3 dvDc =
                                    ((tvDotD / denom) * canonicalNormalW - surfel.tanV) / sv;

                            // d(alpha_i)/dPi = - alpha_i (u_i du/dPi + v_i dv/dPi)
                            const float3 dAlphaDc =
                                    -clampedAlpha * (uv.x() * duDc + uv.y() * dvDc);

                            const float oneMinusAlpha = 1.0f - clampedAlpha;
                            if (oneMinusAlpha <= 1e-6f) {
                                // Fully opaque; transmission beyond this is negligible
                                // We still record the term (optional), but tau will go ~0.
                                continue;
                            }

                            // Store local term
                            localTerms[validCount].alpha = clampedAlpha;
                            localTerms[validCount].dAlphaDc = dAlphaDc;
                            ++validCount;

                            // Update the global transmission product
                            tau *= oneMinusAlpha;
                        }

                        if (validCount != 0) {
                            // Compute d(tau)/dPi using:
                            // d tau / dPi = sum_i [ - (tau / (1 - alpha_i)) * d alpha_i / dPi ]
                            float3 dTauDc(0.0f);

                            for (int i = 0; i < validCount; ++i) {
                                const float alpha_i = localTerms[i].alpha;
                                const float3 dAlphaDc_i = localTerms[i].dAlphaDc;
                                const float oneMinusAlpha_i = 1.0f - alpha_i;

                                const float weight = -tau / oneMinusAlpha_i;
                                dTauDc += weight * dAlphaDc_i;
                            }

                            float3 pAdjoint = shadowRayState.pathThroughput;
                            float p = luminanceGrayscale(pAdjoint);

                            // Accumulate into the running gradient (do not overwrite)
                            d_grad_pos += p * (L_bg) * dTauDc;
                        }
                    }
                }
            }
        }
        return d_grad_pos;
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
}
