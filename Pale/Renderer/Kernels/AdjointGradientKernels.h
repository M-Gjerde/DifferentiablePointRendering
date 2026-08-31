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

    SYCL_EXTERNAL inline void atomicAddUint32(uint32_t &destination, uint32_t value) {
        auto atomicDestination = sycl::atomic_ref<
            uint32_t,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>(destination);
        atomicDestination.fetch_add(value);
    }

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

    SYCL_EXTERNAL inline void accumulateDebugGradientIfSelected(
        DebugImages debugImage,
        bool renderDebugGradientImages,
        uint32_t selectedPrimitiveIndex,
        uint32_t pathId,
        const SurfelGradientRecord &gradientRecord) {
        if (!renderDebugGradientImages) {
            return;
        }

        if (selectedPrimitiveIndex == UINT32_MAX ||
            gradientRecord.primitiveIndex == UINT32_MAX) {
            return;
        }

        if (gradientRecord.primitiveIndex != selectedPrimitiveIndex) {
            return;
        }

        constexpr float maxAbsGradientComponent = 1.0e3f;

        const auto isValidGradientComponent = [](float value) -> bool {
            return sycl::isfinite(value) &&
                   !sycl::isnan(value) &&
                   sycl::fabs(value) <= maxAbsGradientComponent;
        };

        const bool validGradientRecord =
                isValidGradientComponent(gradientRecord.gradPositionX) &&
                isValidGradientComponent(gradientRecord.gradPositionY) &&
                isValidGradientComponent(gradientRecord.gradPositionZ) &&
                isValidGradientComponent(gradientRecord.gradScaleU) &&
                isValidGradientComponent(gradientRecord.gradScaleV) &&
                isValidGradientComponent(gradientRecord.gradRotationX) &&
                isValidGradientComponent(gradientRecord.gradRotationY) &&
                isValidGradientComponent(gradientRecord.gradRotationZ) &&
                isValidGradientComponent(gradientRecord.gradEta) &&
                isValidGradientComponent(gradientRecord.gradBeta) &&
                isValidGradientComponent(gradientRecord.gradAlbedoR) &&
                isValidGradientComponent(gradientRecord.gradAlbedoG) &&
                isValidGradientComponent(gradientRecord.gradAlbedoB);

        if (!validGradientRecord) {
            return;
        }

        if (pathId >= debugImage.numPixels) {
            return;
        }

        const float rotationGradientMagnitude = sycl::sqrt(
            gradientRecord.gradRotationX * gradientRecord.gradRotationX +
            gradientRecord.gradRotationY * gradientRecord.gradRotationY +
            gradientRecord.gradRotationZ * gradientRecord.gradRotationZ);

        atomicAddFloat(debugImage.framebufferPosX[pathId], gradientRecord.gradPositionX);
        atomicAddFloat(debugImage.framebufferPosY[pathId], gradientRecord.gradPositionY);
        atomicAddFloat(debugImage.framebufferPosZ[pathId], gradientRecord.gradPositionZ);

        // Signed local SO(3) components.
        atomicAddFloat(debugImage.framebufferRotX[pathId], gradientRecord.gradRotationX);
        atomicAddFloat(debugImage.framebufferRotY[pathId], gradientRecord.gradRotationY);
        atomicAddFloat(debugImage.framebufferRotZ[pathId], gradientRecord.gradRotationZ);

        atomicAddFloat(debugImage.framebufferScaleU[pathId], gradientRecord.gradScaleU);
        atomicAddFloat(debugImage.framebufferScaleV[pathId], gradientRecord.gradScaleV);

        atomicAddFloat(debugImage.framebufferOpacity[pathId], gradientRecord.gradEta);
        atomicAddFloat(debugImage.framebufferAlbedo[pathId], gradientRecord.gradAlbedoR);
        atomicAddFloat(debugImage.framebufferBeta[pathId], gradientRecord.gradBeta);
    }


    SYCL_EXTERNAL inline float3x3 planeHitPointIntersectionJacobian(
        const float3 &rayDirection,
        const float3 &planeNormal) {
        float3x3 numerator = outerProduct(rayDirection, planeNormal);
        float denom = dot(rayDirection, planeNormal);
        return numerator / denom;
    }

    SYCL_EXTERNAL inline float integrateSlabPolynomial(
        const float *alpha, uint32_t count, uint32_t excludeA, uint32_t excludeB, uint32_t leadingZetaPower) {
        float coefficients[kMaxLocalSurfelHits];
        for (uint32_t i = 0u; i < kMaxLocalSurfelHits; ++i) {
            coefficients[i] = 0.0f;
        }
        coefficients[0] = 1.0f;
        uint32_t degree = 0u;
        for (uint32_t j = 0u; j < count; ++j) {
            if (j == excludeA || j == excludeB) {
                continue;
            }
            const float alphaJ = alpha[j];
            for (int32_t d = static_cast<int32_t>(degree); d >= 0; --d) {
                coefficients[d + 1] -= alphaJ * coefficients[d];
            }
            ++degree;
        }
        float integral = 0.0f;
        for (uint32_t d = 0u; d <= degree; ++d) {
            integral += coefficients[d] / static_cast<float>(d + leadingZetaPower + 1u);
        }
        return integral;
    }

    SYCL_EXTERNAL inline float computeRawSlabWeight(const float *alpha, uint32_t count, uint32_t surfelIndex) {
        const float Ii = integrateSlabPolynomial(alpha, count, surfelIndex, kInvalidIndex, 0u);
        return alpha[surfelIndex] * Ii;
    }

    SYCL_EXTERNAL inline float computeRawSlabWeightDerivativeWrtAlpha(const float *alpha, uint32_t count,
                                                                      uint32_t contributionIndex,
                                                                      uint32_t parameterIndex) {
        // d w_k / d alpha_k = I_k
        if (contributionIndex == parameterIndex) {
            return integrateSlabPolynomial(alpha, count, contributionIndex, kInvalidIndex, 0u);
        }
        // d w_i / d alpha_k = -alpha_i J_ik
        const float Jik = integrateSlabPolynomial(alpha, count, contributionIndex, parameterIndex, 1u);
        return -alpha[contributionIndex] * Jik;
    }

    SYCL_EXTERNAL inline float computeNormalizedSlabWeightDerivativeWrtAlpha(
        const float *alpha, uint32_t count, uint32_t contributionIndex, uint32_t parameterIndex) {
        float rawWeights[kMaxLocalSurfelHits];
        float rawWeightSum = 0.0f;
        float layerTransmission = 1.0f;
        for (uint32_t i = 0u; i < count; ++i) {
            rawWeights[i] = computeRawSlabWeight(alpha, count, i);
            rawWeightSum += rawWeights[i];
            layerTransmission *= sycl::fmax(0.0f, 1.0f - alpha[i]);
        }

        if (rawWeightSum <= 1.0e-8f) {
            return 0.0f;
        }
        const float layerOpacity = 1.0f - layerTransmission;
        float dRawWeightSumDAlphaK = 0.0f;
        for (uint32_t i = 0u; i < count; ++i) {
            dRawWeightSumDAlphaK += computeRawSlabWeightDerivativeWrtAlpha(alpha, count, i, parameterIndex);
        }
        // d alpha_Q / d alpha_k
        //
        // alpha_Q = 1 - prod_j (1-alpha_j)
        //
        float dLayerOpacityDAlphaK = 1.0f;
        for (uint32_t j = 0u; j < count; ++j) {
            if (j == parameterIndex) {
                continue;
            }
            dLayerOpacityDAlphaK *= sycl::fmax(0.0f, 1.0f - alpha[j]);
        }
        const float dRawWiDAlphaK = computeRawSlabWeightDerivativeWrtAlpha(
            alpha, count, contributionIndex, parameterIndex);
        const float normalization = layerOpacity / rawWeightSum;
        const float dNormalizationDAlphaK = (dLayerOpacityDAlphaK * rawWeightSum - layerOpacity * dRawWeightSumDAlphaK)
                                            / (rawWeightSum * rawWeightSum);
        return normalization * dRawWiDAlphaK + rawWeights[contributionIndex] * dNormalizationDAlphaK;
    }


    struct PointLightGeometry {
        float geometricTerm = 0.0f;
        float3 gradientWrtSurfacePosition{0.0f, 0.0f, 0.0f};
        float3 gradientWrtSurfaceNormal{0.0f, 0.0f, 0.0f};
    };

    SYCL_EXTERNAL inline float2 cameraProjectionFocalPixels(const CameraGPU &camera) {
        if (camera.hasPinholeIntrinsics != 0u && camera.fx > 0.0f && camera.fy > 0.0f) {
            return float2{sycl::fabs(camera.fx), sycl::fabs(camera.fy)};
        }

        const float width = sycl::fmax(static_cast<float>(camera.width), 1.0f);
        const float height = sycl::fmax(static_cast<float>(camera.height), 1.0f);
        const float fy = 0.5f * height / sycl::tan(0.5f * glm::radians(camera.fovy));
        const float fx = fy * (width / height);
        return float2{fx, fy};
    }

    SYCL_EXTERNAL inline float3 accumulateProjectedCenterGradientToWorld(
        const CameraGPU &camera,
        const float3 &positionW,
        const float2 &barCenterPixels) {
        const float4 viewPosition = camera.view * float4{positionW, 1.0f};
        const float zForward = -viewPosition.z();
        if (!sycl::isfinite(zForward) || zForward <= 1.0e-8f) {
            return float3{0.0f, 0.0f, 0.0f};
        }

        const float2 focal = cameraProjectionFocalPixels(camera);
        if (!(focal.x() > 0.0f) || !(focal.y() > 0.0f)) {
            return float3{0.0f, 0.0f, 0.0f};
        }

        const float inverseDepth = 1.0f / zForward;
        const float inverseDepthSquared = inverseDepth * inverseDepth;
        const float3 barCameraPosition{
            barCenterPixels.x() * focal.x() * inverseDepth,
            barCenterPixels.y() * (-focal.y() * inverseDepth),
            barCenterPixels.x() * focal.x() * viewPosition.x() * inverseDepthSquared +
            barCenterPixels.y() * (-focal.y() * viewPosition.y() * inverseDepthSquared)
        };

        return float3{
            camera.view.row[0].x() * barCameraPosition.x() +
            camera.view.row[1].x() * barCameraPosition.y() +
            camera.view.row[2].x() * barCameraPosition.z(),
            camera.view.row[0].y() * barCameraPosition.x() +
            camera.view.row[1].y() * barCameraPosition.y() +
            camera.view.row[2].y() * barCameraPosition.z(),
            camera.view.row[0].z() * barCameraPosition.x() +
            camera.view.row[1].z() * barCameraPosition.y() +
            camera.view.row[2].z() * barCameraPosition.z()
        };
    }

    SYCL_EXTERNAL inline float3 computeMinimumFootprintAlphaEffectiveGradientWrtTranslation(
        const Point &surfel,
        const PointCloudSurfaceRecord &surface,
        const CameraGPU &camera) {
        if (surface.alphaProfileBranch != kSurfelAlphaProfileLowPass ||
            surface.lowPassAlphaGeom <= 0.0f ||
            surface.lowPassSigmaPixels <= 0.0f) {
            return float3{0.0f, 0.0f, 0.0f};
        }

        const float sigmaSquared =
                sycl::fmax(surface.lowPassSigmaPixels * surface.lowPassSigmaPixels, 1.0e-8f);
        const float centerScale = surfel.opacity * surface.lowPassAlphaGeom / sigmaSquared;
        const float2 barCenterPixels = surface.lowPassDeltaPixels * centerScale;
        return accumulateProjectedCenterGradientToWorld(camera, surfel.position, barCenterPixels);
    }

    SYCL_EXTERNAL inline float3 computeMinimumFootprintAlphaEffectiveGradientWrtTranslation(
        const Point &surfel,
        const LocalSurfelLayerHit &hit,
        const CameraGPU &camera) {
        if (hit.alphaProfileBranch != kSurfelAlphaProfileLowPass ||
            hit.lowPassAlphaGeom <= 0.0f ||
            hit.lowPassSigmaPixels <= 0.0f) {
            return float3{0.0f, 0.0f, 0.0f};
        }

        const float sigmaSquared =
                sycl::fmax(hit.lowPassSigmaPixels * hit.lowPassSigmaPixels, 1.0e-8f);
        const float centerScale = surfel.opacity * hit.lowPassAlphaGeom / sigmaSquared;
        const float2 barCenterPixels = hit.lowPassDeltaPixels * centerScale;
        return accumulateProjectedCenterGradientToWorld(camera, surfel.position, barCenterPixels);
    }

    SYCL_EXTERNAL inline bool computePointLightGeometry(
        const float3 &surfacePositionW,
        const float3 &surfaceNormalW,
        const float3 &lightPositionW,
        PointLightGeometry &result) {
        const float3 vectorToLight = lightPositionW - surfacePositionW;
        const float distanceSquared = dot(vectorToLight, vectorToLight);
        if (distanceSquared <= 1.0e-12f) {
            return false;
        }
        const float inverseDistance = 1.0f / sycl::sqrt(distanceSquared);
        const float3 lightDirection = vectorToLight * inverseDistance;
        const float cosineAtSurface = dot(surfaceNormalW, lightDirection);
        // Same piecewise convention as max(0, n dot omega).
        if (cosineAtSurface <= 0.0f) {
            return false;
        }
        result.geometricTerm = cosineAtSurface / distanceSquared;
        // d/dX [ (n dot omega) / ||L - X||^2 ]
        result.gradientWrtSurfacePosition =
                (-surfaceNormalW + 3.0f * cosineAtSurface * lightDirection) *
                (inverseDistance / distanceSquared);
        // d/dn [ (n dot omega) / ||L - X||^2 ]
        result.gradientWrtSurfaceNormal =
                lightDirection / distanceSquared;

        return true;
    }

    SYCL_EXTERNAL inline bool isPrimitiveInMeasurementSlab(
        const MeasurementGradientEventXY &eventRecord,
        uint32_t primitiveIndex) {
        for (uint32_t i = 0u; i < eventRecord.surfelSlabCount; ++i) {
            if (eventRecord.xSurface[i].primitiveIndex == primitiveIndex) {
                return true;
            }
        }

        return false;
    }

    SYCL_EXTERNAL inline float3 computeAlphaEffectiveGradientWrtTranslation(
        const Point &surfel,
        const PointCloudSurfaceRecord &surface,
        const ReconstructedSurfelState &state,
        const CameraGPU &camera) {
        if (surface.alphaProfileBranch == kSurfelAlphaProfileLowPass) {
            return computeMinimumFootprintAlphaEffectiveGradientWrtTranslation(surfel, surface, camera);
        }

        const float u = surface.uv.x();
        const float v = surface.uv.y();
        const float r2 = u * u + v * v;
        const float oneMinusR2 = 1.0f - r2;

        if (oneMinusR2 <= 1.0e-8f || surfel.scale.x() <= 1.0e-12f || surfel.scale.y() <= 1.0e-12f) {
            return float3{0.0f};
        }

        const float3 rayDirection = surface.incomingDirection;
        const float3 normal = state.orientedNormal;

        if (sycl::fabs(dot(rayDirection, normal)) <= 1.0e-8f) {
            return float3{0.0f};
        }

        const float betaScale = 4.0f * sycl::exp(surfel.beta);
        const float dAlphaGeomDu = -2.0f * betaScale * u * surface.alphaGeom / oneMinusR2;
        const float dAlphaGeomDv = -2.0f * betaScale * v * surface.alphaGeom / oneMinusR2;

        const float3x3 hitJacobian = planeHitPointIntersectionJacobian(rayDirection, normal);

        // u = t_u . (x - p) / s_u, so du/dp = (dx/dp - I)^T t_u / s_u.
        const float3 duDPosition = transpose(hitJacobian) * (surfel.tanU / surfel.scale.x()) - surfel.tanU / surfel.
                                   scale.x();
        const float3 dvDPosition = transpose(hitJacobian) * (surfel.tanV / surfel.scale.y()) - surfel.tanV / surfel.
                                   scale.y();

        return surfel.opacity * (dAlphaGeomDu * duDPosition + dAlphaGeomDv * dvDPosition);
    }

    SYCL_EXTERNAL inline float2 computeAlphaEffectiveGradientWrtScale(const Point &surfel,
                                                                      const PointCloudSurfaceRecord &surface) {
        if (surface.alphaProfileBranch == kSurfelAlphaProfileLowPass) {
            return float2{0.0f, 0.0f};
        }

        const float u = surface.uv.x();
        const float v = surface.uv.y();
        const float oneMinusR2 = 1.0f - u * u - v * v;
        const float scaleU = surfel.scale.x();
        const float scaleV = surfel.scale.y();

        if (oneMinusR2 <= 1.0e-8f || scaleU <= 1.0e-12f || scaleV <= 1.0e-12f) {
            return float2{0.0f, 0.0f};
        }

        const float betaScale = 4.0f * sycl::exp(surfel.beta);
        const float alphaEffective = surfel.opacity * surface.alphaGeom;

        const float dAlphaDScaleU = 2.0f * betaScale * u * u * alphaEffective / (scaleU * oneMinusR2);
        const float dAlphaDScaleV = 2.0f * betaScale * v * v * alphaEffective / (scaleV * oneMinusR2);

        return float2{dAlphaDScaleU, dAlphaDScaleV};
    }

    SYCL_EXTERNAL inline float3x3 planeHitPointRotationJacobian(
        const float3 &rayOrigin, const float3 &rayDirection, const float3 &planePosition, const float3 &planeNormal) {
        const float nDotD = dot(planeNormal, rayDirection);
        if (sycl::fabs(nDotD) <= 1.0e-8f) {
            return float3x3{};
        }

        const float3 a = planePosition - rayOrigin;
        const float nDotA = dot(planeNormal, a);
        const float invNDotDSquared = 1.0f / (nDotD * nDotD);

        const float3 dtDRotation =
                (cross(planeNormal, a) * nDotD - nDotA * cross(planeNormal, rayDirection)) * invNDotDSquared;

        return outerProduct(rayDirection, dtDRotation);
    }

    SYCL_EXTERNAL inline float3 computeAlphaEffectiveGradientWrtWorldRotation(
        const Point &surfel, const PointCloudSurfaceRecord &surface, const ReconstructedSurfelState &state,
        const float3 &rayOrigin) {
        if (surface.alphaProfileBranch == kSurfelAlphaProfileLowPass) {
            return float3{0.0f};
        }

        const float u = surface.uv.x();
        const float v = surface.uv.y();
        const float oneMinusR2 = 1.0f - u * u - v * v;
        const float scaleU = surfel.scale.x();
        const float scaleV = surfel.scale.y();

        if (oneMinusR2 <= 1.0e-8f || scaleU <= 1.0e-12f || scaleV <= 1.0e-12f) {
            return float3{0.0f};
        }

        const float3 rayDirection = surface.incomingDirection;
        const float3 normal = state.orientedNormal;
        const float nDotD = dot(normal, rayDirection);

        if (sycl::fabs(nDotD) <= 1.0e-8f) {
            return float3{0.0f};
        }

        const float betaScale = 4.0f * sycl::exp(surfel.beta);
        const float dAlphaGeomDu = -2.0f * betaScale * u * surface.alphaGeom / oneMinusR2;
        const float dAlphaGeomDv = -2.0f * betaScale * v * surface.alphaGeom / oneMinusR2;

        const float3 xMinusP = state.position - surfel.position;
        const float3 a = surfel.position - rayOrigin;
        const float nDotA = dot(normal, a);
        const float invNDotD = 1.0f / nDotD;
        const float invNDotDSquared = invNDotD * invNDotD;

        const float3 q = (cross(normal, a) * nDotD - nDotA * cross(normal, rayDirection)) * invNDotDSquared;

        const float3 duDRotation = q * (dot(rayDirection, surfel.tanU) / scaleU) + cross(surfel.tanU, xMinusP) / scaleU;
        const float3 dvDRotation = q * (dot(rayDirection, surfel.tanV) / scaleV) + cross(surfel.tanV, xMinusP) / scaleV;

        return surfel.opacity * (dAlphaGeomDu * duDRotation + dAlphaGeomDv * dvDRotation);
    }
    

    SYCL_EXTERNAL inline float3 computeAlphaEffectiveGradientWrtSegmentStart(
        const Point &surfel, const LocalSurfelLayerHit &hit, const float3 &segmentStart, const float3 &segmentEnd) {
        float3 normal = normalize(cross(surfel.tanU, surfel.tanV));
        const float3 segment = segmentEnd - segmentStart;
        if (dot(normal, -segment) < 0.0f) normal = -normal;

        const float nDotSegment = dot(normal, segment);
        const float scaleU = surfel.scale.x();
        const float scaleV = surfel.scale.y();
        if (sycl::fabs(nDotSegment) <= 1.0e-8f || scaleU <= 1.0e-12f || scaleV <= 1.0e-12f) return float3{0.0f};

        const float2 uv = phiInverse(hit.hitPositionW, surfel);
        const float u = uv.x();
        const float v = uv.y();
        const float oneMinusR2 = 1.0f - u * u - v * v;
        if (oneMinusR2 <= 1.0e-8f) return float3{0.0f};

        const float lambda = dot(normal, surfel.position - segmentStart) / nDotSegment;
        const float oneMinusLambda = 1.0f - lambda;

        const float3 duDStart = oneMinusLambda * (surfel.tanU - normal * dot(segment, surfel.tanU) / nDotSegment) /
                                scaleU;
        const float3 dvDStart = oneMinusLambda * (surfel.tanV - normal * dot(segment, surfel.tanV) / nDotSegment) /
                                scaleV;

        const float betaScale = 4.0f * sycl::exp(surfel.beta);
        const float dAlphaGeomDu = -2.0f * betaScale * u * hit.alphaGeom / oneMinusR2;
        const float dAlphaGeomDv = -2.0f * betaScale * v * hit.alphaGeom / oneMinusR2;

        return surfel.opacity * (dAlphaGeomDu * duDStart + dAlphaGeomDv * dvDStart);
    }

    inline float3 computeGeometricTermGradientWrtStartpoint(
        const float3 &xPosition,
        const float3 &yPosition,
        const float3 &xNormal,
        const float3 &yNormal) {
        const float3 vectorFromXToY = yPosition - xPosition;
        const float squaredDistance = dot(vectorFromXToY, vectorFromXToY);

        if (squaredDistance <= 1e-12f) {
            return float3{0.0f};
        }

        const float distance = sycl::sqrt(squaredDistance);
        const float inverseDistance = 1.0f / distance;
        const float inverseDistanceCubed = inverseDistance * inverseDistance * inverseDistance;
        const float3 directionFromXToY = vectorFromXToY * inverseDistance;

        const float cosineAtX = dot(xNormal, directionFromXToY);
        const float cosineAtY = dot(yNormal, -directionFromXToY);


        auto directionOuter = outerProduct(directionFromXToY, directionFromXToY);
        float3x3 tmp1 = identity3x3() - directionOuter;
        float3 tmp2 = (-cosineAtY * (tmp1 * xNormal)) + (cosineAtX * (tmp1 * yNormal)) + (
                          2 * cosineAtX * cosineAtY * directionFromXToY);

        return inverseDistanceCubed * tmp2;
    }

    struct CachedSegmentEndpointDerivativeResult {
        bool valid = true;
        bool overflowed = false;
        float weightedTransmittance = 1.0f;
        float3 gradWeightedTransmittanceWrtEndPosition{0.0f, 0.0f, 0.0f};
    };

    inline float3 computeGeometricTermGradientWrtEndpoint(
        const float3 &xPosition,
        const float3 &yPosition,
        const float3 &xNormal,
        const float3 &yNormal) {
        return -computeGeometricTermGradientWrtStartpoint(
            xPosition, yPosition, xNormal, yNormal);
    }

    SYCL_EXTERNAL inline float3 computeLocalRotationGradientFromWorldRotationGradient(
        const float3 &tangentU,
        const float3 &tangentV,
        const float3 &worldRotationGradient) {
        const float3 tangentW = normalize(cross(tangentU, tangentV));

        return float3{
            dot(worldRotationGradient, tangentU),
            dot(worldRotationGradient, tangentV),
            dot(worldRotationGradient, tangentW)
        };
    }

    SYCL_EXTERNAL inline float3 computeLocalRotationGradientFromTangentGradients(
        const float3 &tangentU,
        const float3 &tangentV,
        const float3 &gradientTangentU,
        const float3 &gradientTangentV) {
        const float3 tangentW = normalize(cross(tangentU, tangentV));

        return float3{
            dot(gradientTangentV, tangentW),
            -dot(gradientTangentU, tangentW),
            dot(gradientTangentU, tangentV) - dot(gradientTangentV, tangentU)
        };
    }


    struct UVPositionJacobian {
        float3 du_d_surfel_translation;
        float3 dv_d_surfel_translation;
    };

    inline UVPositionJacobian computeDuvDSurfelTranslationJacobianForImplicitRayHit(
        const float3 &tangentUWorld,
        const float3 &tangentVWorld,
        const float3 &canonicalNormalWorld,
        const float3 &rayDirection,
        float scaleU,
        float scaleV) {
        const float denominator = dot(canonicalNormalWorld, rayDirection);
        if (sycl::fabs(denominator) <= 1e-4f) {
            return {};
        }

        const float tangentUDotRayDirection = dot(tangentUWorld, rayDirection);
        const float tangentVDotRayDirection = dot(tangentVWorld, rayDirection);

        const float3 duDTranslation =
                ((tangentUDotRayDirection / denominator) * canonicalNormalWorld - tangentUWorld) / scaleU;

        const float3 dvDTranslation =
                ((tangentVDotRayDirection / denominator) * canonicalNormalWorld - tangentVWorld) / scaleV;

        return {duDTranslation, dvDTranslation};
    }

    struct MaterialEdgeOccluderDerivative {
        float3 gradPosition{0.0f, 0.0f, 0.0f};

        float gradScaleU = 0.0f;
        float gradScaleV = 0.0f;
        float gradEta = 0.0f;
        float gradBeta = 0.0f;

        float3 gradRotation{0.0f, 0.0f, 0.0f};

        float3 gradAlphaWrtStartPoint{0.0f, 0.0f, 0.0f};
        float3 gradAlphaWrtEndPoint{0.0f, 0.0f, 0.0f};

        float prefixTransmittance = 1.0f;
        float oneMinusAlpha = 1.0f;

        uint32_t primitiveIndex = kInvalidIndex;
    };

    struct MaterialEdgeVisibilityDerivativeResult {
        MaterialEdgeOccluderDerivative occluderDerivatives[kMaxSplatEventsPerRay];

        uint32_t storedOccluderCount = 0u;

        float segmentTransmittance = 1.0f;
        float nullSamplingWeight = 1.0f;

        float3 gradTauWrtStartPoint{0.0f, 0.0f, 0.0f};
        float3 gradTauWrtEndPoint{0.0f, 0.0f, 0.0f};
    };


    SYCL_EXTERNAL inline MaterialEdgeVisibilityDerivativeResult traceMaterialEdgeVisibilityDerivatives(
        const GPUSceneBuffers &scene,
        const float3 &startPositionWorld,
        const float3 &endPositionWorld,
        const float3 &startNormalWorld,
        uint32_t startPrimitiveIndex,
        uint32_t endPrimitiveIndex,
        bool applyNullSamplingWeight,
        float qNullInv) {
        MaterialEdgeVisibilityDerivativeResult result{};
        const float3 startToEnd = endPositionWorld - startPositionWorld;
        const float targetDistance = length(startToEnd);
        if (targetDistance <= 1.0e-12f) {
            return result;
        }
        const float3 rayDirection = startToEnd / targetDistance;
        Ray ray{};
        ray.origin = startPositionWorld + rayDirection * RayEpsilon;
        ray.direction = rayDirection;
        ray.normal = startNormalWorld;
        const float3 dxy = startPositionWorld - endPositionWorld;
        float tracedTransmittance = 1.0f;
        for (uint32_t traversalIndex = 0u; traversalIndex < kMaxSplatEventsPerRay; ++traversalIndex) {
            WorldHit worldHit{};
            intersectScene(ray, &worldHit, scene, SurfelIntersectMode::FirstHit);
            if (!worldHit.hit) {
                break;
            }
            const float hitDistance = length(worldHit.hitPositionW - startPositionWorld);
            if (hitDistance >= targetDistance - RayEpsilon) {
                break;
            }
            buildIntersectionNormal(scene, worldHit);
            const InstanceRecord &instance = scene.instances[worldHit.instanceIndex];
            if (instance.geometryType != GeometryType::PointCloud) {
                tracedTransmittance = 0.0f;
                break;
            }
            if (worldHit.primitiveIndex == startPrimitiveIndex || worldHit.primitiveIndex == endPrimitiveIndex) {
                ray.origin = worldHit.hitPositionW + ray.direction * RayEpsilon;
                continue;
            }
            const Point &occluderSurfel = scene.points[worldHit.primitiveIndex];
            float3 occluderNormal = normalize(cross(occluderSurfel.tanU, occluderSurfel.tanV));
            if (dot(occluderNormal, -ray.direction) < 0.0f) {
                occluderNormal = -occluderNormal;
            }
            const float alphaGeomOccluder = worldHit.alphaGeom;
            const float alphaEffective = occluderSurfel.opacity * alphaGeomOccluder;
            const float oneMinusAlpha = sycl::fmax(0.0f, 1.0f - alphaEffective);
            const float prefixTransmittance = tracedTransmittance;
            tracedTransmittance *= oneMinusAlpha;
            ray.origin = worldHit.hitPositionW + ray.direction * RayEpsilon;
            if (applyNullSamplingWeight) {
                result.nullSamplingWeight *= qNullInv;
            }
            const float2 uv = phiInverse(worldHit.hitPositionW, occluderSurfel);
            const float uOcc = uv.x();
            const float vOcc = uv.y();
            const float occluderScaleU = occluderSurfel.scale.x();
            const float occluderScaleV = occluderSurfel.scale.y();
            if (occluderScaleU <= 1.0e-12f || occluderScaleV <= 1.0e-12f) {
                continue;
            }
            const float3 tangentU = occluderSurfel.tanU;
            const float3 tangentV = occluderSurfel.tanV;
            const float3 localBasisU = tangentU / occluderScaleU;
            const float3 localBasisV = tangentV / occluderScaleV;

            const float denominator = dot(occluderNormal, dxy);
            if (sycl::fabs(denominator) <= 1.0e-8f) {
                continue;
            }
            const float inverseDenominator = 1.0f / denominator;
            const float lambdaOccluder = dot(occluderNormal, occluderSurfel.position - endPositionWorld) *
                                         inverseDenominator;
            const float3 commonU = localBasisU - occluderNormal * (dot(dxy, localBasisU) * inverseDenominator);
            const float3 commonV = localBasisV - occluderNormal * (dot(dxy, localBasisV) * inverseDenominator);
            const float3 dUiDx = lambdaOccluder * commonU;
            const float3 dViDx = lambdaOccluder * commonV;
            const float3 dUiDy = (1.0f - lambdaOccluder) * commonU;
            const float3 dViDy = (1.0f - lambdaOccluder) * commonV;
            const float3 dUiDspi = occluderNormal * (dot(dxy, tangentU) / occluderScaleU) * inverseDenominator -
                                   localBasisU;
            const float3 dViDspi = occluderNormal * (dot(dxy, tangentV) / occluderScaleV) * inverseDenominator -
                                   localBasisV;
            const float radiusSquared = uOcc * uOcc + vOcc * vOcc;
            const float oneMinusRadiusSquared = 1.0f - radiusSquared;
            if (oneMinusRadiusSquared <= 1.0e-8f) {
                continue;
            }
            const float betaScale = 4.0f * sycl::exp(occluderSurfel.beta);
            const float dAlphaGeomDu = -2.0f * betaScale * uOcc * alphaGeomOccluder / oneMinusRadiusSquared;
            const float dAlphaGeomDv = -2.0f * betaScale * vOcc * alphaGeomOccluder / oneMinusRadiusSquared;
            const float3 dAlphaEffectiveDx = occluderSurfel.opacity * (dAlphaGeomDu * dUiDx + dAlphaGeomDv * dViDx);
            const float3 dAlphaEffectiveDy = occluderSurfel.opacity * (dAlphaGeomDu * dUiDy + dAlphaGeomDv * dViDy);
            const float3 dAlphaEffectiveDspi = occluderSurfel.opacity * (
                                                   dAlphaGeomDu * dUiDspi + dAlphaGeomDv * dViDspi);
            const float dAlphaEffectiveDScaleU =
                    2.0f * betaScale * uOcc * uOcc * alphaEffective / (occluderScaleU * oneMinusRadiusSquared);
            const float dAlphaEffectiveDScaleV =
                    2.0f * betaScale * vOcc * vOcc * alphaEffective / (occluderScaleV * oneMinusRadiusSquared);
            const float dAlphaEffectiveDEta = alphaGeomOccluder;
            const float dAlphaEffectiveDBeta = betaScale * sycl::log(oneMinusRadiusSquared) * alphaEffective;
            float3 localRotationGradientOcc{0.0f, 0.0f, 0.0f};

            const float nDotD = dot(occluderNormal, rayDirection);
            if (sycl::fabs(nDotD) > 1.0e-8f) {
                const float3 hitMinusSp = worldHit.hitPositionW - occluderSurfel.position;
                const float3 aOcc = occluderSurfel.position - endPositionWorld;
                const float nDotA = dot(occluderNormal, aOcc);
                const float invNDotD = 1.0f / nDotD;
                const float invNDotDSquared = invNDotD * invNDotD;

                const float3 qOcc =
                        (cross(occluderNormal, aOcc) * nDotD -
                         nDotA * cross(occluderNormal, rayDirection)) *
                        invNDotDSquared;

                const float3 duDRotation =
                        qOcc * (dot(rayDirection, tangentU) / occluderScaleU) +
                        cross(tangentU, hitMinusSp) / occluderScaleU;

                const float3 dvDRotation =
                        qOcc * (dot(rayDirection, tangentV) / occluderScaleV) +
                        cross(tangentV, hitMinusSp) / occluderScaleV;

                const float3 worldRotationGradient =
                        occluderSurfel.opacity *
                        (dAlphaGeomDu * duDRotation + dAlphaGeomDv * dvDRotation);

                localRotationGradientOcc =
                        computeLocalRotationGradientFromWorldRotationGradient(
                            occluderSurfel.tanU,
                            occluderSurfel.tanV,
                            worldRotationGradient);
            }

            if (result.storedOccluderCount < kMaxSplatEventsPerRay) {
                MaterialEdgeOccluderDerivative &occluderDerivative = result.occluderDerivatives[result.
                    storedOccluderCount];
                occluderDerivative.gradPosition = dAlphaEffectiveDspi;
                occluderDerivative.gradScaleU = dAlphaEffectiveDScaleU;
                occluderDerivative.gradScaleV = dAlphaEffectiveDScaleV;
                occluderDerivative.gradEta = dAlphaEffectiveDEta;
                occluderDerivative.gradBeta = dAlphaEffectiveDBeta;
                occluderDerivative.gradRotation = localRotationGradientOcc;
                occluderDerivative.gradAlphaWrtStartPoint = dAlphaEffectiveDx;
                occluderDerivative.gradAlphaWrtEndPoint = dAlphaEffectiveDy;
                occluderDerivative.prefixTransmittance = prefixTransmittance;
                occluderDerivative.oneMinusAlpha = oneMinusAlpha;
                occluderDerivative.primitiveIndex = worldHit.primitiveIndex;

                result.storedOccluderCount++;
            }
        }

        result.segmentTransmittance = tracedTransmittance;

        float suffixTransmittanceForTauGradient = 1.0f;
        for (uint32_t reverseIndex = result.storedOccluderCount; reverseIndex > 0u; --reverseIndex) {
            const uint32_t occluderIndex = reverseIndex - 1u;
            const MaterialEdgeOccluderDerivative &occluderDerivative = result.occluderDerivatives[occluderIndex];
            const float tauDerivativeScale = -occluderDerivative.prefixTransmittance *
                                             suffixTransmittanceForTauGradient;

            result.gradTauWrtStartPoint += tauDerivativeScale * occluderDerivative.gradAlphaWrtStartPoint;
            result.gradTauWrtEndPoint += tauDerivativeScale * occluderDerivative.gradAlphaWrtEndPoint;

            suffixTransmittanceForTauGradient *= occluderDerivative.oneMinusAlpha;
        }

        return result;
    }


    SYCL_EXTERNAL inline void writeMaterialEdgeOccluderGradientRecords(
        SurfelGradientRecord *gradientRecords,
        uint32_t firstOccluderRecordIndex,
        const MaterialEdgeVisibilityDerivativeResult &visibilityResult,
        float geometricTerm,
        float scalarMaterialEdgeWeight,
        float invSpp,
        const DebugImages &debugImage,
        bool renderDebugGradientImages,
        uint32_t selectedPrimitiveIndex,
        uint32_t pathId) {
        float suffixTransmittance = 1.0f;

        for (uint32_t reverseIndex = visibilityResult.storedOccluderCount; reverseIndex > 0u; --reverseIndex) {
            const uint32_t occluderIndex = reverseIndex - 1u;
            const uint32_t occluderRecordIndex = firstOccluderRecordIndex + occluderIndex;

            const MaterialEdgeOccluderDerivative &occluderDerivative =
                    visibilityResult.occluderDerivatives[occluderIndex];

            const float visibilityDerivativeScale =
                    -occluderDerivative.prefixTransmittance *
                    suffixTransmittance *
                    geometricTerm *
                    scalarMaterialEdgeWeight *
                    invSpp;

            SurfelGradientRecord occluderRecord{};
            occluderRecord.primitiveIndex = occluderDerivative.primitiveIndex;

            const float3 positionContribution =
                    visibilityDerivativeScale * occluderDerivative.gradPosition;

            const float3 rotationContribution =
                    visibilityDerivativeScale * occluderDerivative.gradRotation;

            occluderRecord.gradPositionX = positionContribution.x();
            occluderRecord.gradPositionY = positionContribution.y();
            occluderRecord.gradPositionZ = positionContribution.z();

            occluderRecord.gradScaleU = visibilityDerivativeScale * occluderDerivative.gradScaleU;
            occluderRecord.gradScaleV = visibilityDerivativeScale * occluderDerivative.gradScaleV;

            occluderRecord.gradRotationX = rotationContribution.x();
            occluderRecord.gradRotationY = rotationContribution.y();
            occluderRecord.gradRotationZ = rotationContribution.z();

            occluderRecord.gradEta = visibilityDerivativeScale * occluderDerivative.gradEta;
            occluderRecord.gradBeta = visibilityDerivativeScale * occluderDerivative.gradBeta;

            occluderRecord.gradAlbedoR = 0.0f;
            occluderRecord.gradAlbedoG = 0.0f;
            occluderRecord.gradAlbedoB = 0.0f;

            gradientRecords[occluderRecordIndex] = occluderRecord;

            suffixTransmittance *= occluderDerivative.oneMinusAlpha;

            accumulateDebugGradientIfSelected(
                debugImage,
                renderDebugGradientImages,
                selectedPrimitiveIndex,
                pathId,
                occluderRecord);
        }
    }
}
