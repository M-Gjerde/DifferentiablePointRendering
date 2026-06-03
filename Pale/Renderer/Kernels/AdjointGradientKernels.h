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

    SYCL_EXTERNAL inline void accumulateDebugGradientIfSelected(
        DebugImages debugImage,
        bool renderDebugGradientImages,
        uint32_t selectedPrimitiveIndex,
        uint32_t pathId,
        const SurfelGradientRecord &gradientRecord) {
        if (!renderDebugGradientImages) {
            return;
        }
        if (gradientRecord.primitiveIndex != selectedPrimitiveIndex || selectedPrimitiveIndex == UINT32_MAX || gradientRecord.primitiveIndex == UINT32_MAX) {
            return;
        }
        constexpr float maxAbsGradientComponent = 1.0e3f;

        const auto isValidGradientComponent = [](float value) -> bool {
            return sycl::isfinite(value) && !sycl::isnan(value) && sycl::fabs(value) <=
                   maxAbsGradientComponent;
        };

        const bool validGradientRecord =
                isValidGradientComponent(gradientRecord.gradPositionX) &&
                isValidGradientComponent(gradientRecord.gradPositionY) &&
                isValidGradientComponent(gradientRecord.gradPositionZ) &&
                isValidGradientComponent(gradientRecord.gradScaleU) &&
                isValidGradientComponent(gradientRecord.gradScaleV) &&
                isValidGradientComponent(gradientRecord.gradTangentUX) &&
                isValidGradientComponent(gradientRecord.gradTangentUY) &&
                isValidGradientComponent(gradientRecord.gradTangentUZ) &&
                isValidGradientComponent(gradientRecord.gradTangentVX) &&
                isValidGradientComponent(gradientRecord.gradTangentVY) &&
                isValidGradientComponent(gradientRecord.gradTangentVZ) &&
                isValidGradientComponent(gradientRecord.gradEta) &&
                isValidGradientComponent(gradientRecord.gradBeta) &&
                isValidGradientComponent(gradientRecord.gradAlbedoR) &&
                isValidGradientComponent(gradientRecord.gradAlbedoG) &&
                isValidGradientComponent(gradientRecord.gradAlbedoB);
        if (!validGradientRecord)
            return;
        if (pathId >= debugImage.numPixels)
            return;

        atomicAddFloat(debugImage.framebufferPosX[pathId], gradientRecord.gradPositionX);
        atomicAddFloat(debugImage.framebufferPosY[pathId], gradientRecord.gradPositionY);
        atomicAddFloat(debugImage.framebufferPosZ[pathId], gradientRecord.gradPositionZ);
        const float3 tangentUGradient{
            gradientRecord.gradTangentUX, gradientRecord.gradTangentUY, gradientRecord.gradTangentUZ
        };
        const float3 tangentVGradient{
            gradientRecord.gradTangentVX, gradientRecord.gradTangentVY, gradientRecord.gradTangentVZ
        };

        //atomicAddFloat(debugImage.framebufferRot[pathId], rotationGradientMagnitude);
        atomicAddFloat(debugImage.framebufferScale[pathId], gradientRecord.gradScaleU);
        atomicAddFloat(debugImage.framebufferOpacity[pathId], gradientRecord.gradEta);
        atomicAddFloat(debugImage.framebufferAlbedo[pathId],  gradientRecord.gradAlbedoR);
        atomicAddFloat(debugImage.framebufferBeta[pathId], gradientRecord.gradBeta);
    }

    SYCL_EXTERNAL inline void accumulateSurfelGradientAtomic(
        const PointGradients &gradients,
        uint32_t primitiveIndex,
        const float3 &gradPosition,
        const float2 &gradScale,
        const float3 &gradTanU,
        const float3 &gradTanV,
        float gradEta,
        float gradBeta,
        const float3 &gradAlbedo = float3{0.0f, 0.0f, 0.0f}) {
        atomicAddFloat(gradients.gradPosition[primitiveIndex].x(), gradPosition.x());
        atomicAddFloat(gradients.gradPosition[primitiveIndex].y(), gradPosition.y());
        atomicAddFloat(gradients.gradPosition[primitiveIndex].z(), gradPosition.z());

        atomicAddFloat(gradients.gradScale[primitiveIndex].x(), gradScale.x());
        atomicAddFloat(gradients.gradScale[primitiveIndex].y(), gradScale.y());

        atomicAddFloat(gradients.gradTanU[primitiveIndex].x(), gradTanU.x());
        atomicAddFloat(gradients.gradTanU[primitiveIndex].y(), gradTanU.y());
        atomicAddFloat(gradients.gradTanU[primitiveIndex].z(), gradTanU.z());

        atomicAddFloat(gradients.gradTanV[primitiveIndex].x(), gradTanV.x());
        atomicAddFloat(gradients.gradTanV[primitiveIndex].y(), gradTanV.y());
        atomicAddFloat(gradients.gradTanV[primitiveIndex].z(), gradTanV.z());

        atomicAddFloat(gradients.gradOpacity[primitiveIndex], gradEta);
        atomicAddFloat(gradients.gradBeta[primitiveIndex], gradBeta);

        atomicAddFloat(gradients.gradAlbedo[primitiveIndex].x(), gradAlbedo.x());
        atomicAddFloat(gradients.gradAlbedo[primitiveIndex].y(), gradAlbedo.y());
        atomicAddFloat(gradients.gradAlbedo[primitiveIndex].z(), gradAlbedo.z());
    }

    SYCL_EXTERNAL inline float3 mapHitPointGradientToSurfelTranslation(
        const float3 &gradientWrtHitPosition,
        const float3 &cameraRayDirection,
        const float3 &surfelNormal) {
        const float denominator = dot(surfelNormal, cameraRayDirection);

        if (sycl::fabs(denominator) <= 1e-6f) {
            return float3{0.0f};
        }

        const float numerator = dot(cameraRayDirection, gradientWrtHitPosition);

        // J^T * g_x = n * (w · g_x) / (n · w)
        return surfelNormal * (numerator / denominator);
    }

    SYCL_EXTERNAL inline float3x3 planeHitPointJacobianWrtOrigin(
        const float3 &rayDirection,
        const float3 &planeNormal) {
        float3x3 identity = identity3x3();

        float3x3 numerator = outerProduct(rayDirection, planeNormal);
        float denom = dot(rayDirection, planeNormal);
        return identity - numerator / denom;
    }

    SYCL_EXTERNAL inline float3x3 planeHitPointIntersectionJacobian(
        const float3 &rayDirection,
        const float3 &planeNormal) {
        float3x3 numerator = outerProduct(rayDirection, planeNormal);
        float denom = dot(rayDirection, planeNormal);
        return numerator / denom;
    }

    inline float3 computeGeometricTermGradientWrtEndpointFixedDirection(
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
        const float inverseDistanceCubed =
                inverseDistance * inverseDistance * inverseDistance;

        const float3 directionFromXToY =
                vectorFromXToY * inverseDistance;

        const float cosineAtX =
                dot(xNormal, directionFromXToY);

        const float cosineAtY =
                dot(yNormal, -directionFromXToY);

        return
                -2.0f *
                cosineAtX *
                cosineAtY *
                inverseDistanceCubed *
                directionFromXToY;
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

    float3 computeLogDGradientWrtX(
        const float3 &x,
        const float3 &y,
        const float3 &ny) {
        float3 d = y - x;
        float dist2 = dot(d, d);
        float dist = sycl::sqrt(dist2);
        float3 phi = d / dist;

        float cosY = dot(ny, -phi);
        if (cosY <= 1e-6f) return float3{0};

        float3 projectedNy = ny - phi * dot(ny, phi);

        return projectedNy / (cosY * dist)
               + (2.0f / dist) * phi;
    }

    inline float computeGeometryOverAreaPdfFromUniformHemisphereSample(
        const float3 &startPosition,
        const float3 &endPosition,
        const float3 &startNormal) {
        const float3 vectorToEnd = endPosition - startPosition;
        const float distanceSquared = dot(vectorToEnd, vectorToEnd);
        if (distanceSquared <= 1e-12f) {
            return 0.0f;
        }

        const float distance = sycl::sqrt(distanceSquared);
        const float3 directionToEnd = vectorToEnd / distance;

        const float cosineAtStart = dot(startNormal, directionToEnd);
        if (cosineAtStart <= 1e-6f) {
            return 0.0f;
        }

        const float uniformHemispherePdf = 1.0f / (2.0f * M_PIf);
        return cosineAtStart / uniformHemispherePdf;
    }

    inline float3 computeGeometryOverAreaPdfGradientWrtEndpointFromUniformHemisphereSample(
        const float3 &startPosition,
        const float3 &endPosition,
        const float3 &startNormal) {
        const float3 vectorToEnd = endPosition - startPosition;
        const float distanceSquared = dot(vectorToEnd, vectorToEnd);
        if (distanceSquared <= 1e-12f) {
            return float3{0.0f};
        }

        const float distance = sycl::sqrt(distanceSquared);
        const float3 directionToEnd = vectorToEnd / distance;

        const float cosineAtStart = dot(startNormal, directionToEnd);
        if (cosineAtStart <= 1e-6f) {
            return float3{0.0f};
        }

        const float uniformHemispherePdf = 1.0f / (2.0f * M_PIf);

        const float3 projectedStartNormal =
                startNormal - directionToEnd * dot(startNormal, directionToEnd);

        return projectedStartNormal * (1.0f / (uniformHemispherePdf * distance));
    }

    inline float3 computeGeometryOverAreaPdfGradientWrtStartpointFromUniformHemisphereSample(
        const float3 &startPosition,
        const float3 &endPosition,
        const float3 &startNormal) {
        const float3 vectorToEnd = endPosition - startPosition;
        const float distanceSquared = dot(vectorToEnd, vectorToEnd);
        if (distanceSquared <= 1e-12f) {
            return float3{0.0f};
        }

        const float distance = sycl::sqrt(distanceSquared);
        const float3 directionToEnd = vectorToEnd / distance;

        const float cosineAtStart = dot(startNormal, directionToEnd);
        if (cosineAtStart <= 1e-6f) {
            return float3{0.0f};
        }

        const float uniformHemispherePdf = 1.0f / (2.0f * M_PIf);

        const float3 projectedStartNormal =
                startNormal - directionToEnd * dot(startNormal, directionToEnd);

        return -projectedStartNormal * (1.0f / (uniformHemispherePdf * distance));
    }

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


        // duv/dc_pos = (u du/dc + v dv/dc)
        const float3 dUVPosition = (u * duDPk + v * dvDPk);
        return dUVPosition;
    }

    inline float computeGeometricAlphaDerivativeWrtScaleU(
        const Point &surfel,
        const float3 &worldPosition) {
        const float3 offset_from_center = worldPosition - surfel.position;

        const float s_u = surfel.scale.x();
        const float s_v = surfel.scale.y();

        if (s_u <= 1e-8f || s_v <= 1e-8f) {
            return 0.0f;
        }

        const float u = dot(offset_from_center, surfel.tanU) / s_u;
        const float v = dot(offset_from_center, surfel.tanV) / s_v;

        const float radius_squared = u * u + v * v;
        if (radius_squared >= 1.0f) {
            return 0.0f;
        }

        const float beta = 4.0f * sycl::exp(surfel.beta);
        const float base = sycl::fmax(1.0f - radius_squared, 1e-8f);

        return (2.0f * beta * u * u / s_u) * sycl::pow(base, beta - 1.0f);
    }

    inline float computeGeometricAlphaDerivativeWrtScaleV(
        const Point &surfel,
        const float3 &worldPosition) {
        const float3 offset_from_center = worldPosition - surfel.position;

        const float s_u = surfel.scale.x();
        const float s_v = surfel.scale.y();

        if (s_u <= 1e-8f || s_v <= 1e-8f) {
            return 0.0f;
        }

        const float u = dot(offset_from_center, surfel.tanU) / s_u;
        const float v = dot(offset_from_center, surfel.tanV) / s_v;

        const float radius_squared = u * u + v * v;
        if (radius_squared >= 1.0f) {
            return 0.0f;
        }

        const float beta = 4.0f * sycl::exp(surfel.beta);
        const float base = sycl::fmax(1.0f - radius_squared, 1e-8f);

        return (2.0f * beta * v * v / s_v) * sycl::pow(base, beta - 1.0f);
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

    inline UVPositionJacobian computeDuvDSurfelTranslationJacobianForExplicitSurfacePoint(
        const float3 &tangentUWorld,
        const float3 &tangentVWorld,
        float scaleU,
        float scaleV) {
        if (scaleU <= 1e-12f || scaleV <= 1e-12f) {
            return {};
        }

        return {
            -tangentUWorld / scaleU,
            -tangentVWorld / scaleV
        };
    }

    inline float3 computeAreaPdfGradientWrtX(
        const float3 &xPosition,
        const float3 &yPosition,
        const float3 &yNormal,
        float directionPdf) {
        const float3 vectorFromXToY = yPosition - xPosition;
        const float squaredDistance = dot(vectorFromXToY, vectorFromXToY);
        if (squaredDistance <= 1e-12f) {
            return float3{0.0f};
        }

        const float distance = sycl::sqrt(squaredDistance);
        const float inverseDistance = 1.0f / distance;
        const float inverseDistanceCubed =
                inverseDistance * inverseDistance * inverseDistance;

        const float3 directionXToY = vectorFromXToY * inverseDistance;
        const float cosineAtY = dot(yNormal, -directionXToY);

        if (cosineAtY <= 1e-6f) {
            return float3{0.0f};
        }

        // p_A(x -> y) = p_omega * cosineAtY / r^2
        //
        // d cosineAtY / d x = (yNormal + cosineAtY * directionXToY) / r
        // d (1 / r^2) / d x = 2 * directionXToY / r^3
        //
        // Therefore:
        // d p_A / d x = p_omega / r^3 * (yNormal + 3 * cosineAtY * directionXToY)
        return directionPdf * inverseDistanceCubed *
               (yNormal + 3.0f * cosineAtY * directionXToY);
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
        float3 d = y - x;
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
        const float3 dUVPosition = -(u * duDpk + v * dvDpk);

        return dUVPosition;
    }

    inline float3 computeDuvDPositionDetached(
        const float3 &tangentUWorld,
        const float3 &tangentVWorld,
        const float3 &canonicalNormalWorld,
        const float3 &y,
        const float3 &x,
        float u, float v,
        float su, float sv) {
        // Use a consistent direction convention. Here: camera -> hit
        float3 rayDirection = y - x;
        const float rayLen = length(rayDirection);
        rayDirection = rayDirection / rayLen;

        const float denom = dot(canonicalNormalWorld, rayDirection);

        if (fabs(denom) < 1e-8f) {
            return float3{0.0f, 0.0f, 0.0f};
        }

        const float tangentUDotRay = dot(tangentUWorld, rayDirection);
        const float tangentVDotRay = dot(tangentVWorld, rayDirection);

        const float3 duDpk =
                ((tangentUDotRay / denom) * canonicalNormalWorld - tangentUWorld) / su;

        const float3 dvDpk =
                ((tangentVDotRay / denom) * canonicalNormalWorld - tangentVWorld) / sv;

        return u * duDpk + v * dvDpk;
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
        float betaKernelFactor = (beta * alpha * 2.0f) / denom;

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


    SYCL_EXTERNAL inline bool computeRayPlaneIntersectionRtDerivatives(
        const float3 &rayOrigin,
        const float3 &rayDirection,
        const float3 &surfelPosition,
        const float3 &surfelTanU,
        const float3 &surfelTanV,
        float &outRt,
        float3 &outDRtDPosition, // vector: ∂rt/∂p (so δrt = dot(outDRtDPosition, δp))
        float3 &outDRtDTanU, // vector: ∂rt/∂tanU
        float3 &outDRtDTanV // vector: ∂rt/∂tanV
    ) {
        const float3 n = cross(surfelTanU, surfelTanV); // unnormalized plane normal
        const float denom = dot(n, rayDirection);
        // Avoid exploding gradients if nearly parallel
        if (sycl::fabs(denom) < 1e-8f) {
            outRt = 0.0f;
            outDRtDPosition = float3{0.0f, 0.0f, 0.0f};
            outDRtDTanU = float3{0.0f, 0.0f, 0.0f};
            outDRtDTanV = float3{0.0f, 0.0f, 0.0f};
            return false;
        }

        const float3 w = surfelPosition - rayOrigin;
        const float num = dot(n, w);

        outRt = num / denom;

        // ∂rt/∂p = n / denom
        outDRtDPosition = n / denom;

        // For tanU/tanV we use quotient rule:
        // rt = num/denom
        // d(rt) = (dnum*denom - num*ddenom) / denom^2
        const float invDenom = 1.0f / denom;
        const float invDenom2 = invDenom * invDenom;

        // dnum/dtanU = cross(tanV, w)
        // dden/dtanU = cross(tanV, rayDirection)
        const float3 dNumDTanU = cross(surfelTanV, w);
        const float3 dDenDTanU = cross(surfelTanV, rayDirection);

        outDRtDTanU = (dNumDTanU * denom - dDenDTanU * num) * invDenom2;

        // dnum/dtanV = cross(w, tanU)
        // dden/dtanV = cross(rayDirection, tanU)
        const float3 dNumDTanV = cross(w, surfelTanU);
        const float3 dDenDTanV = cross(rayDirection, surfelTanU);

        outDRtDTanV = (dNumDTanV * denom - dDenDTanV * num) * invDenom2;

        return true;
    }


    inline void shadowRay(const GPUSceneBuffers &scene, const RayState &rayState, const WorldHit &worldHit,
                          const PointGradients &gradients, const DebugImages &debugImage,
                          const DeviceSurfacePhotonMapGrid &photonMap, rng::Xorshift128 &rng,
                          bool renderDebugGradientImages,
                          uint32_t numShadowRays = 1,
                          uint32_t debugIndex = UINT32_MAX,
                          bool debugBreakFlag = false) {
        for (int i = 0; i < numShadowRays; ++i) {
            AreaLightSample ls = sampleMeshAreaLight(scene, rng);
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

                    /*
                    // BRDF
                    WorldHit shadowWorldHit{};
                    intersectScene(shadowRayState.ray, &shadowWorldHit, scene, rng,
                                   RayIntersectMode::DetachedMode);
                    buildIntersectionNormal(scene, shadowWorldHit);

                    accumulateTransmittanceGradientsAlongRay(shadowRayState, shadowWorldHit, scene, photonMap,
                                                             renderDebugGradientImages, gradients,
                                                             debugImage, debugIndex);
                    if (debugBreakFlag)
                        int debug = 1;
                    */
                }
            }
        }
    }

    struct MaterialEdgeOccluderDerivative {
        float3 gradPosition{0.0f, 0.0f, 0.0f};

        float gradScaleU = 0.0f;
        float gradScaleV = 0.0f;
        float gradEta = 0.0f;
        float gradBeta = 0.0f;

        float3 gradTangentU{0.0f, 0.0f, 0.0f};
        float3 gradTangentV{0.0f, 0.0f, 0.0f};

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
        constexpr float distanceEpsilon = 1.0e-5f;
        const float3 startToEnd = endPositionWorld - startPositionWorld;
        const float targetDistance = length(startToEnd);
        if (targetDistance <= 1.0e-12f) {
            return result;
        }
        const float3 rayDirection = startToEnd / targetDistance;
        Ray ray{};
        ray.origin = startPositionWorld + rayDirection * distanceEpsilon;
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
            if (hitDistance >= targetDistance - distanceEpsilon) {
                break;
            }
            buildIntersectionNormal(scene, worldHit);
            const InstanceRecord &instance = scene.instances[worldHit.instanceIndex];
            if (instance.geometryType != GeometryType::PointCloud) {
                tracedTransmittance = 0.0f;
                break;
            }
            if (worldHit.primitiveIndex == startPrimitiveIndex || worldHit.primitiveIndex == endPrimitiveIndex) {
                ray.origin = worldHit.hitPositionW + ray.direction * distanceEpsilon;
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
            ray.origin = worldHit.hitPositionW + ray.direction * distanceEpsilon;
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
            float3 gradTangentUOcc{0.0f, 0.0f, 0.0f};
            float3 gradTangentVOcc{0.0f, 0.0f, 0.0f};
            const float nDotD = dot(occluderNormal, rayDirection);
            if (sycl::fabs(nDotD) > 1.0e-8f) {
                const float3 hitMinusSp = worldHit.hitPositionW - occluderSurfel.position;
                const float3 aOcc = occluderSurfel.position - endPositionWorld;
                const float nDotA = dot(occluderNormal, aOcc);
                const float invNDotD = 1.0f / nDotD;
                const float invNDotDSquared = invNDotD * invNDotD;
                const float3 qOcc = (cross(occluderNormal, aOcc) * nDotD - nDotA * cross(occluderNormal, rayDirection))
                                    * invNDotDSquared;
                const float3 duDRotation = qOcc * (dot(rayDirection, tangentU) / occluderScaleU) + cross(
                                               tangentU, hitMinusSp) / occluderScaleU;
                const float3 dvDRotation = qOcc * (dot(rayDirection, tangentV) / occluderScaleV) + cross(
                                               tangentV, hitMinusSp) / occluderScaleV;
                const float3 dAlphaEffectiveDRotation =
                        occluderSurfel.opacity * (dAlphaGeomDu * duDRotation + dAlphaGeomDv * dvDRotation);
                gradTangentUOcc = cross(dAlphaEffectiveDRotation, occluderSurfel.tanU);
                gradTangentVOcc = cross(dAlphaEffectiveDRotation, occluderSurfel.tanV);
            }

            if (result.storedOccluderCount < kMaxSplatEventsPerRay) {
                MaterialEdgeOccluderDerivative &occluderDerivative = result.occluderDerivatives[result.
                    storedOccluderCount];
                occluderDerivative.gradPosition = dAlphaEffectiveDspi;
                occluderDerivative.gradScaleU = dAlphaEffectiveDScaleU;
                occluderDerivative.gradScaleV = dAlphaEffectiveDScaleV;
                occluderDerivative.gradEta = dAlphaEffectiveDEta;
                occluderDerivative.gradBeta = dAlphaEffectiveDBeta;
                occluderDerivative.gradTangentU = gradTangentUOcc;
                occluderDerivative.gradTangentV = gradTangentVOcc;
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
        const DebugImages& debugImage,
        bool renderDebugGradientImages,
        uint32_t selectedPrimitiveIndex,
        uint32_t pathId) {
        float suffixTransmittance = 1.0f;
        for (uint32_t reverseIndex = visibilityResult.storedOccluderCount;
             reverseIndex > 0u;
             --reverseIndex) {
            const uint32_t occluderIndex = reverseIndex - 1u;
            const uint32_t occluderRecordIndex = firstOccluderRecordIndex + occluderIndex;
            const MaterialEdgeOccluderDerivative &occluderDerivative = visibilityResult.occluderDerivatives[
                occluderIndex];
            const float visibilityDerivativeScale =
                    -occluderDerivative.prefixTransmittance * suffixTransmittance * geometricTerm *
                    scalarMaterialEdgeWeight * invSpp;

            SurfelGradientRecord occluderRecord{};
            occluderRecord.primitiveIndex = occluderDerivative.primitiveIndex;
            const float3 positionContribution = visibilityDerivativeScale * occluderDerivative.gradPosition;
            const float3 tangentUContribution = visibilityDerivativeScale * occluderDerivative.gradTangentU;
            const float3 tangentVContribution = visibilityDerivativeScale * occluderDerivative.gradTangentV;

            occluderRecord.gradPositionX = positionContribution.x();
            occluderRecord.gradPositionY = positionContribution.y();
            occluderRecord.gradPositionZ = positionContribution.z();
            occluderRecord.gradScaleU = visibilityDerivativeScale * occluderDerivative.gradScaleU;
            occluderRecord.gradScaleV = visibilityDerivativeScale * occluderDerivative.gradScaleV;
            occluderRecord.gradTangentUX = tangentUContribution.x();
            occluderRecord.gradTangentUY = tangentUContribution.y();
            occluderRecord.gradTangentUZ = tangentUContribution.z();
            occluderRecord.gradTangentVX = tangentVContribution.x();
            occluderRecord.gradTangentVY = tangentVContribution.y();
            occluderRecord.gradTangentVZ = tangentVContribution.z();
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
