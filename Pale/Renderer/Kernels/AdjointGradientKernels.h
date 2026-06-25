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

        if (            selectedPrimitiveIndex == UINT32_MAX ||
            gradientRecord.primitiveIndex == UINT32_MAX) {
            return;
        }

        if (gradientRecord.primitiveIndex != selectedPrimitiveIndex) {
            //return;
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
