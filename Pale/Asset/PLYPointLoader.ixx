// =====================================
// File: Pale.Assets.PLYPointLoader.ixx
// =====================================
module;

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/norm.hpp>
#include "tinyply.h"

export module Pale.Assets:PLYPointLoader;

import Pale.Assets.Core;
import :Point;
import Pale.Log;

namespace Pale::ply_detail {
    inline bool copyScalarsToFloatVector(const tinyply::PlyData &plyData, std::vector<float> &outFloats, std::size_t componentsPerItem) {
        const std::size_t elementCount = plyData.count;
        if (elementCount == 0 || componentsPerItem == 0) return false;
        outFloats.resize(elementCount * componentsPerItem);
        switch (plyData.t) {
            case tinyply::Type::FLOAT32: {
                const auto *src = reinterpret_cast<const float *>(plyData.buffer.get_const());
                std::memcpy(outFloats.data(), src, outFloats.size() * sizeof(float));
                return true;
            }
            case tinyply::Type::FLOAT64: {
                const auto *src = reinterpret_cast<const double *>(plyData.buffer.get_const());
                for (std::size_t i = 0; i < outFloats.size(); ++i) outFloats[i] = static_cast<float>(src[i]);
                return true;
            }
            default:
                Log::PA_ERROR("PLYPointLoader: unsupported scalar type {}. Expected float32/float64.", static_cast<int>(plyData.t));
                return false;
        }
    }

    inline std::unordered_set<std::string> collectVertexPropertyNames(const tinyply::PlyFile &plyFile) {
        std::unordered_set<std::string> names;
        for (const auto &element: plyFile.get_elements()) {
            if (element.name != "vertex") continue;
            for (const auto &prop: element.properties) names.insert(prop.name);
        }
        return names;
    }

    inline bool hasAll(const std::unordered_set<std::string> &properties, std::initializer_list<const char *> keys) {
        for (const char *key: keys) if (!properties.count(key)) return false;
        return true;
    }

    inline glm::quat normalizeQuaternionOrIdentity(glm::quat q) {
        const bool finite =
            std::isfinite(q.w) &&
            std::isfinite(q.x) &&
            std::isfinite(q.y) &&
            std::isfinite(q.z);

        const float lengthSquared =
            q.w * q.w +
            q.x * q.x +
            q.y * q.y +
            q.z * q.z;

        if (!finite || lengthSquared <= 1.0e-20f) {
            return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        }

        q *= 1.0f / std::sqrt(lengthSquared);

        if (q.w < 0.0f) {
            q.w = -q.w;
            q.x = -q.x;
            q.y = -q.y;
            q.z = -q.z;
        }

        return q;
    }
}

export namespace Pale {
    struct PLYPointLoader : IAssetLoader<PointAsset> {
        AssetPtr<PointAsset> load(const AssetHandle & /*id*/, const AssetMeta &meta) override {
            Log::PA_INFO("PLYPointLoader: loading '{}'", meta.path.string());
            std::ifstream inputFile(meta.path, std::ios::binary);
            if (!inputFile) {
                Log::PA_ERROR("PLYPointLoader: cannot open file '{}'", meta.path.string());
                return {};
            }
            tinyply::PlyFile plyFile;
            try { plyFile.parse_header(inputFile); } catch (const std::exception &e) {
                Log::PA_ERROR("PLYPointLoader: header parse failed: {}", e.what());
                return {};
            }

            const auto vertexProps = ply_detail::collectVertexPropertyNames(plyFile);
            const bool looksQuaternionSurfel = ply_detail::hasAll(vertexProps, {
                "x", "y", "z", "rot_w", "rot_x", "rot_y", "rot_z", "su", "sv", "albedo_r", "albedo_g", "albedo_b", "opacity", "beta", "shape", "power"
            });
            const bool hasDensificationOrigin =
                vertexProps.contains("densification_origin");
            const bool hasPrimitiveAge = vertexProps.contains("primitive_age");
            const bool hasPositionDensificationSignal =
                vertexProps.contains("densification_position_signal");
            const bool hasPositionDensificationSampleCount =
                vertexProps.contains("densification_position_sample_count");
            const bool hasPositionDensificationThreshold =
                vertexProps.contains("densification_position_threshold");
            const bool hasPositionDensificationRadianceRms =
                vertexProps.contains("densification_position_radiance_rms");
            const bool hasPositionDensificationBaseThreshold =
                vertexProps.contains("densification_position_base_threshold");
            const bool hasPositionDensificationMetadata =
                hasPositionDensificationSignal &&
                hasPositionDensificationSampleCount &&
                hasPositionDensificationThreshold;
            if ((hasPositionDensificationSignal ||
                 hasPositionDensificationSampleCount ||
                 hasPositionDensificationThreshold) &&
                !hasPositionDensificationMetadata) {
                Log::PA_ERROR(
                    "PLYPointLoader: incomplete position densification metadata");
                return {};
            }
            const bool hasPositionRadianceBiasMetadata =
                hasPositionDensificationRadianceRms &&
                hasPositionDensificationBaseThreshold;
            if ((hasPositionDensificationRadianceRms ||
                 hasPositionDensificationBaseThreshold) &&
                (!hasPositionDensificationMetadata || !hasPositionRadianceBiasMetadata)) {
                Log::PA_ERROR(
                    "PLYPointLoader: incomplete position radiance-bias metadata");
                return {};
            }
            if (!looksQuaternionSurfel) {
                Log::PA_ERROR("PLYPointLoader: unsupported vertex schema. Expected quaternion surfel format: x y z rot_w rot_x rot_y rot_z su sv albedo_r albedo_g albedo_b opacity beta shape power.");
                return {};
            }

            std::shared_ptr<tinyply::PlyData> posData = plyFile.request_properties_from_element("vertex", {"x", "y", "z"});
            std::shared_ptr<tinyply::PlyData> rotData = plyFile.request_properties_from_element("vertex", {"rot_w", "rot_x", "rot_y", "rot_z"});
            std::shared_ptr<tinyply::PlyData> scaleData = plyFile.request_properties_from_element("vertex", {"su", "sv"});
            std::shared_ptr<tinyply::PlyData> colorData = plyFile.request_properties_from_element("vertex", {"albedo_r", "albedo_g", "albedo_b"});
            std::shared_ptr<tinyply::PlyData> opacityData = plyFile.request_properties_from_element("vertex", {"opacity"});
            std::shared_ptr<tinyply::PlyData> betaData = plyFile.request_properties_from_element("vertex", {"beta"});
            std::shared_ptr<tinyply::PlyData> shapeData = plyFile.request_properties_from_element("vertex", {"shape"});
            std::shared_ptr<tinyply::PlyData> powerData = plyFile.request_properties_from_element("vertex", {"power"});
            std::shared_ptr<tinyply::PlyData> densificationOriginData;
            std::shared_ptr<tinyply::PlyData> primitiveAgeData;
            std::shared_ptr<tinyply::PlyData> positionDensificationSignalData;
            std::shared_ptr<tinyply::PlyData> positionDensificationSampleCountData;
            std::shared_ptr<tinyply::PlyData> positionDensificationThresholdData;
            std::shared_ptr<tinyply::PlyData> positionDensificationRadianceRmsData;
            std::shared_ptr<tinyply::PlyData> positionDensificationBaseThresholdData;
            if (hasDensificationOrigin) {
                densificationOriginData =
                    plyFile.request_properties_from_element("vertex", {"densification_origin"});
            }
            if (hasPrimitiveAge) {
                primitiveAgeData =
                    plyFile.request_properties_from_element("vertex", {"primitive_age"});
            }
            if (hasPositionDensificationMetadata) {
                positionDensificationSignalData = plyFile.request_properties_from_element(
                    "vertex", {"densification_position_signal"});
                positionDensificationSampleCountData = plyFile.request_properties_from_element(
                    "vertex", {"densification_position_sample_count"});
                positionDensificationThresholdData = plyFile.request_properties_from_element(
                    "vertex", {"densification_position_threshold"});
            }
            if (hasPositionRadianceBiasMetadata) {
                positionDensificationRadianceRmsData = plyFile.request_properties_from_element(
                    "vertex", {"densification_position_radiance_rms"});
                positionDensificationBaseThresholdData = plyFile.request_properties_from_element(
                    "vertex", {"densification_position_base_threshold"});
            }

            try { plyFile.read(inputFile); } catch (const std::exception &e) {
                Log::PA_ERROR("PLYPointLoader: read failed: {}", e.what());
                return {};
            }

            const std::size_t vertexCount = posData ? posData->count : 0;
            if (vertexCount == 0) {
                Log::PA_ERROR("PLYPointLoader: zero vertices");
                return {};
            }
            auto sameCount = [&](const char *name, const std::shared_ptr<tinyply::PlyData> &data) -> bool {
                if (!data || data->count != vertexCount) {
                    Log::PA_ERROR("PLYPointLoader: '{}' count mismatch. expected {}, got {}", name, vertexCount, data ? data->count : 0);
                    return false;
                }
                return true;
            };
            if (!(sameCount("rot_*", rotData) && sameCount("su,sv", scaleData) && sameCount("albedo_*", colorData) && sameCount("opacity", opacityData) && sameCount("beta", betaData) && sameCount("shape", shapeData) && sameCount("power", powerData))) return {};
            if (hasDensificationOrigin && !sameCount("densification_origin", densificationOriginData)) return {};
            if (hasPrimitiveAge && !sameCount("primitive_age", primitiveAgeData)) return {};
            if (hasPositionDensificationMetadata &&
                !(sameCount("densification_position_signal", positionDensificationSignalData) &&
                  sameCount("densification_position_sample_count", positionDensificationSampleCountData) &&
                  sameCount("densification_position_threshold", positionDensificationThresholdData))) return {};
            if (hasPositionRadianceBiasMetadata &&
                !(sameCount("densification_position_radiance_rms", positionDensificationRadianceRmsData) &&
                  sameCount("densification_position_base_threshold", positionDensificationBaseThresholdData))) return {};

            std::vector<float> posFloats, rotFloats, scaleFloats, colorFloats, opacityFloats,
                betaFloats, shapeFloats, powerFloats, densificationOriginFloats,
                primitiveAgeFloats, positionDensificationSignalFloats,
                positionDensificationSampleCountFloats, positionDensificationThresholdFloats,
                positionDensificationRadianceRmsFloats,
                positionDensificationBaseThresholdFloats;
            bool ok = true;
            ok &= ply_detail::copyScalarsToFloatVector(*posData, posFloats, 3);
            ok &= ply_detail::copyScalarsToFloatVector(*rotData, rotFloats, 4);
            ok &= ply_detail::copyScalarsToFloatVector(*scaleData, scaleFloats, 2);
            ok &= ply_detail::copyScalarsToFloatVector(*colorData, colorFloats, 3);
            ok &= ply_detail::copyScalarsToFloatVector(*opacityData, opacityFloats, 1);
            ok &= ply_detail::copyScalarsToFloatVector(*betaData, betaFloats, 1);
            ok &= ply_detail::copyScalarsToFloatVector(*shapeData, shapeFloats, 1);
            ok &= ply_detail::copyScalarsToFloatVector(*powerData, powerFloats, 1);
            if (hasDensificationOrigin) {
                ok &= ply_detail::copyScalarsToFloatVector(
                    *densificationOriginData, densificationOriginFloats, 1);
            }
            if (hasPrimitiveAge) {
                ok &= ply_detail::copyScalarsToFloatVector(
                    *primitiveAgeData, primitiveAgeFloats, 1);
            }
            if (hasPositionDensificationMetadata) {
                ok &= ply_detail::copyScalarsToFloatVector(
                    *positionDensificationSignalData, positionDensificationSignalFloats, 1);
                ok &= ply_detail::copyScalarsToFloatVector(
                    *positionDensificationSampleCountData,
                    positionDensificationSampleCountFloats,
                    1);
                ok &= ply_detail::copyScalarsToFloatVector(
                    *positionDensificationThresholdData,
                    positionDensificationThresholdFloats,
                    1);
            }
            if (hasPositionRadianceBiasMetadata) {
                ok &= ply_detail::copyScalarsToFloatVector(
                    *positionDensificationRadianceRmsData,
                    positionDensificationRadianceRmsFloats,
                    1);
                ok &= ply_detail::copyScalarsToFloatVector(
                    *positionDensificationBaseThresholdData,
                    positionDensificationBaseThresholdFloats,
                    1);
            }
            if (!ok) {
                Log::PA_ERROR("PLYPointLoader: failed to unpack quaternion surfel streams");
                return {};
            }

            auto pointAsset = std::make_shared<PointAsset>();
            pointAsset->points.emplace_back();
            PointGeometry &geometry = pointAsset->points.back();
            geometry.positions.resize(vertexCount);
            geometry.quat.resize(vertexCount);
            geometry.scales.resize(vertexCount);
            geometry.albedos.resize(vertexCount);
            geometry.opacities.resize(vertexCount);
            geometry.betas.resize(vertexCount);
            geometry.shapes.resize(vertexCount);
            geometry.powers.resize(vertexCount);
            geometry.densificationOrigins.resize(vertexCount, 0u);
            if (hasPrimitiveAge) {
                geometry.primitiveAges.resize(vertexCount, 0u);
            }
            if (hasPositionDensificationMetadata) {
                geometry.densificationPositionSignals.resize(vertexCount, 0.0f);
                geometry.densificationPositionSampleCounts.resize(vertexCount, 0u);
                geometry.densificationPositionThresholds.resize(vertexCount, 0.0f);
            }
            if (hasPositionRadianceBiasMetadata) {
                geometry.densificationPositionRadianceRms.resize(vertexCount, 0.0f);
                geometry.densificationPositionBaseThresholds.resize(vertexCount, 0.0f);
            }

            for (std::size_t i = 0; i < vertexCount; ++i) {
                const std::size_t i2 = i * 2, i3 = i * 3, i4 = i * 4;
                geometry.positions[i] = glm::vec3(posFloats[i3 + 0], posFloats[i3 + 1], posFloats[i3 + 2]);
                geometry.quat[i] = ply_detail::normalizeQuaternionOrIdentity(glm::quat(rotFloats[i4 + 0], rotFloats[i4 + 1], rotFloats[i4 + 2], rotFloats[i4 + 3]));
                geometry.scales[i] = glm::vec2(scaleFloats[i2 + 0], scaleFloats[i2 + 1]);
                geometry.albedos[i] = glm::clamp(glm::vec3(colorFloats[i3 + 0], colorFloats[i3 + 1], colorFloats[i3 + 2]), 0.0f, 1.0f);
                geometry.opacities[i] = opacityFloats[i];
                geometry.betas[i] = betaFloats[i];
                geometry.shapes[i] = shapeFloats[i];
                geometry.powers[i] = powerFloats[i];
                if (hasDensificationOrigin &&
                    std::isfinite(densificationOriginFloats[i])) {
                    geometry.densificationOrigins[i] = static_cast<std::uint8_t>(
                        std::clamp(std::lround(densificationOriginFloats[i]), 0l, 3l));
                }
                if (hasPrimitiveAge && std::isfinite(primitiveAgeFloats[i])) {
                    const double age = std::clamp(
                        static_cast<double>(primitiveAgeFloats[i]),
                        0.0,
                        static_cast<double>(std::numeric_limits<std::uint32_t>::max()));
                    geometry.primitiveAges[i] = static_cast<std::uint32_t>(std::llround(age));
                }
                if (hasPositionDensificationMetadata) {
                    if (std::isfinite(positionDensificationSignalFloats[i])) {
                        geometry.densificationPositionSignals[i] =
                            std::max(positionDensificationSignalFloats[i], 0.0f);
                    }
                    if (std::isfinite(positionDensificationSampleCountFloats[i])) {
                        const double sampleCount = std::clamp(
                            static_cast<double>(positionDensificationSampleCountFloats[i]),
                            0.0,
                            static_cast<double>(std::numeric_limits<std::uint32_t>::max()));
                        geometry.densificationPositionSampleCounts[i] =
                            static_cast<std::uint32_t>(std::llround(sampleCount));
                    }
                    if (std::isfinite(positionDensificationThresholdFloats[i]) &&
                        positionDensificationThresholdFloats[i] > 0.0f) {
                        geometry.densificationPositionThresholds[i] =
                            positionDensificationThresholdFloats[i];
                    }
                }
                if (hasPositionRadianceBiasMetadata) {
                    if (std::isfinite(positionDensificationRadianceRmsFloats[i])) {
                        geometry.densificationPositionRadianceRms[i] =
                            std::max(positionDensificationRadianceRmsFloats[i], 0.0f);
                    }
                    if (std::isfinite(positionDensificationBaseThresholdFloats[i]) &&
                        positionDensificationBaseThresholdFloats[i] > 0.0f) {
                        geometry.densificationPositionBaseThresholds[i] =
                            positionDensificationBaseThresholdFloats[i];
                    }
                }
            }

            Log::PA_INFO("PLYPointLoader: loaded {} quaternion surfels", vertexCount);
            return pointAsset;
        }
    };
}
