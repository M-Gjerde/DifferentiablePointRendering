// =====================================
// File: Pale.Assets.PLYPointLoader.ixx
// =====================================
module;

#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>
#include <cmath>
#define GLM_ENABLE_EXPERIMENTAL
#include <cstring>
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtx/orthonormalize.hpp>

#include "tinyply.h"
#include "glm/gtx/compatibility.hpp"

export module Pale.Assets:PLYPointLoader;

import Pale.Assets.Core;
import :Point;
import Pale.Log;

namespace Pale::ply_detail {

inline bool copyScalarsToFloatVector(const tinyply::PlyData &plyData,
                                     std::vector<float> &outFloats,
                                     std::size_t componentsPerItem)
{
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
            for (std::size_t i = 0; i < elementCount * componentsPerItem; ++i)
                outFloats[i] = static_cast<float>(src[i]);
            return true;
        }
        default:
            Log::PA_ERROR("PLYPointLoader: unsupported scalar type {}. Expected float32/float64.",
                          static_cast<int>(plyData.t));
            return false;
    }
}

inline void orthonormalizeTangents(glm::vec3 &tangentU, glm::vec3 &tangentV)
{
    if (glm::length2(tangentU) > 0.f) tangentU = glm::normalize(tangentU);
    tangentV -= glm::dot(tangentV, tangentU) * tangentU;
    const float len2 = glm::length2(tangentV);
    tangentV = (len2 > 0.f) ? tangentV / std::sqrt(len2) : glm::vec3(0.f, 1.f, 0.f);
}

inline glm::mat3 quaternionToMatrix3x3(float qw, float qx, float qy, float qz)
{
    glm::quat quaternionNormalized = glm::normalize(glm::quat(qw, qx, qy, qz));
    return glm::mat3(quaternionNormalized);
}

inline glm::vec3 safeNormalFromTangents(const glm::vec3 &tangentU, const glm::vec3 &tangentV)
{
    glm::vec3 n = glm::cross(tangentU, tangentV);
    const float len2 = glm::length2(n);
    return (len2 > 0.f) ? n / std::sqrt(len2) : glm::vec3(0.f, 0.f, 1.f);
}

    inline std::unordered_set<std::string>
collectVertexPropertyNames(const tinyply::PlyFile &plyFile)
{
    std::unordered_set<std::string> names;
    for (const auto &element : plyFile.get_elements()) {
        if (element.name != "vertex") continue;
        for (const auto &prop : element.properties) names.insert(prop.name);
    }
    return names;
}

    inline bool hasAll(const std::unordered_set<std::string> &s,
                       std::initializer_list<const char*> keys)
{
    for (auto *k : keys) if (!s.count(k)) return false;
    return true;
}


} // namespace Pale::ply_detail


export namespace Pale {

    // helpers (top of file, near ply_detail)
    inline float sigmoid(float x) {
        // guard to avoid overflow in exp
        x = std::clamp(x, -20.0f, 20.0f);
        return 1.0f / (1.0f + std::exp(-x));
    }
    inline float expSafe(float x) {
        // guard: exp(±20) ~ [2e-9, 4.85e8]
        x = std::clamp(x, -20.0f, 20.0f);
        return std::exp(x);
    }

struct PLYPointLoader : IAssetLoader<PointAsset>
{
    AssetPtr<PointAsset> load(const AssetHandle& /*id*/, const AssetMeta& meta) override
    {
        Log::PA_INFO("PLYPointLoader: loading '{}'", meta.path.string());

        std::ifstream inputFile(meta.path, std::ios::binary);
        if (!inputFile) {
            Log::PA_ERROR("PLYPointLoader: cannot open file '{}'", meta.path.string());
            return {};
        }

        tinyply::PlyFile plyFile;
        try { plyFile.parse_header(inputFile); }
        catch (const std::exception &e) {
            Log::PA_ERROR("PLYPointLoader: header parse failed: {}", e.what());
            return {};
        }

        // Probe header
        const auto vertexProps = ply_detail::collectVertexPropertyNames(plyFile);

        const bool looks2DGS = ply_detail::hasAll(vertexProps,
            {"x","y","z","tu_x","tu_y","tu_z","tv_x","tv_y","tv_z",
             "su","sv","albedo_r","albedo_g","albedo_b","opacity"});

        const bool looks3DGS = ply_detail::hasAll(vertexProps,
            {"x","y","z","f_dc_0","f_dc_1","f_dc_2","opacity",
             "scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"});

        if (!looks2DGS && !looks3DGS) {
            Log::PA_ERROR("PLYPointLoader: unsupported vertex schema");
            return {};
        }

        const bool hasNormals3D = vertexProps.count("nx") && vertexProps.count("ny") && vertexProps.count("nz");

        // Request properties exactly once
        std::shared_ptr<tinyply::PlyData> posData, tuData, tvData, scale2D, color2D, opacityData;
        std::shared_ptr<tinyply::PlyData> color3D, scale3D, rot3D, normal3D;

        if (looks2DGS) {
            posData     = plyFile.request_properties_from_element("vertex", {"x","y","z"});
            tuData      = plyFile.request_properties_from_element("vertex", {"tu_x","tu_y","tu_z"});
            tvData      = plyFile.request_properties_from_element("vertex", {"tv_x","tv_y","tv_z"});
            scale2D     = plyFile.request_properties_from_element("vertex", {"su","sv"});
            color2D     = plyFile.request_properties_from_element("vertex", {"albedo_r","albedo_g","albedo_b"});
            opacityData = plyFile.request_properties_from_element("vertex", {"opacity"});
        } else { // 3DGS
            posData     = plyFile.request_properties_from_element("vertex", {"x","y","z"});
            color3D     = plyFile.request_properties_from_element("vertex", {"f_dc_0","f_dc_1","f_dc_2"});
            opacityData = plyFile.request_properties_from_element("vertex", {"opacity"});
            scale3D     = plyFile.request_properties_from_element("vertex", {"scale_0","scale_1","scale_2"});
            rot3D       = plyFile.request_properties_from_element("vertex", {"rot_0","rot_1","rot_2","rot_3"});
            if (hasNormals3D) {
                normal3D = plyFile.request_properties_from_element("vertex", {"nx","ny","nz"});
            }
        }

        // Read once after all requests
        try { plyFile.read(inputFile); }
        catch (const std::exception &e) {
            Log::PA_ERROR("PLYPointLoader: read failed: {}", e.what());
            return {};
        }

        auto pointAsset = std::make_shared<PointAsset>();
        pointAsset->points.emplace_back();
        PointGeometry &geometry = pointAsset->points.back();

        const std::size_t vertexCount = posData ? posData->count : 0;
        if (vertexCount == 0) {
            Log::PA_ERROR("PLYPointLoader: zero vertices");
            return {};
        }

        auto sameCount = [&](const char *name, const std::shared_ptr<tinyply::PlyData> &d)->bool {
            if (!d || d->count != vertexCount) {
                Log::PA_ERROR("PLYPointLoader: '{}' count mismatch. expected {}, got {}",
                              name, vertexCount, d ? d->count : 0);
                return false;
            }
            return true;
        };

        if (looks2DGS) {
            if (!(sameCount("tu_*", tuData) && sameCount("tv_*", tvData) &&
                  sameCount("su,sv", scale2D) && sameCount("albedo_*", color2D) &&
                  sameCount("opacity", opacityData)))
                return {};

            std::vector<float> posFloats, tuFloats, tvFloats, scaleFloats, colorFloats, opacityFloats;
            bool ok = true;
            ok &= ply_detail::copyScalarsToFloatVector(*posData,     posFloats,     3);
            ok &= ply_detail::copyScalarsToFloatVector(*tuData,      tuFloats,      3);
            ok &= ply_detail::copyScalarsToFloatVector(*tvData,      tvFloats,      3);
            ok &= ply_detail::copyScalarsToFloatVector(*scale2D,     scaleFloats,   2);
            ok &= ply_detail::copyScalarsToFloatVector(*color2D,     colorFloats,   3);
            ok &= ply_detail::copyScalarsToFloatVector(*opacityData, opacityFloats, 1);
            if (!ok) {
                Log::PA_ERROR("PLYPointLoader: failed to unpack one or more 2DGS streams");
                return {};
            }

            geometry.positions.resize(vertexCount);
            geometry.tanU.resize(vertexCount);
            geometry.tanV.resize(vertexCount);
            geometry.scales.resize(vertexCount);
            geometry.colors.resize(vertexCount);
            geometry.opacities.resize(vertexCount);

            for (std::size_t i = 0; i < vertexCount; ++i) {
                const std::size_t i3 = i * 3, i2 = i * 2;

                glm::vec3 tangentU = glm::normalize(glm::vec3(tuFloats[i3 + 0], tuFloats[i3 + 1], tuFloats[i3 + 2]));
                glm::vec3 tangentV = glm::normalize(glm::vec3(tvFloats[i3 + 0], tvFloats[i3 + 1], tvFloats[i3 + 2]));
                //ply_detail::orthonormalizeTangents(tangentU, tangentV);

                geometry.positions[i] = glm::vec3(posFloats[i3 + 0], posFloats[i3 + 1], posFloats[i3 + 2]);
                geometry.tanU[i]      = tangentU;
                geometry.tanV[i]      = tangentV;
                geometry.scales[i]    = glm::vec2(scaleFloats[i2 + 0], scaleFloats[i2 + 1]);
                geometry.colors[i]    = glm::clamp(glm::vec3(colorFloats[i3 + 0], colorFloats[i3 + 1], colorFloats[i3 + 2]), 0.0f, 1.0f);
                geometry.opacities[i] = opacityFloats[i];
            }

            Log::PA_INFO("PLYPointLoader: loaded {} 2DGS splats", vertexCount);
            return pointAsset;
        }

        // -------- 3DGS → 2D surface-aligned representation --------
        {
            if (!(sameCount("f_dc_*", color3D) &&
                  sameCount("opacity", opacityData) &&
                  sameCount("scale_*",  scale3D) &&
                  sameCount("rot_*",    rot3D)))
                return {};

            std::vector<float> posFloats, colorFloats, opacityFloats, scaleFloats, rotFloats, normalFloats;
            bool ok = true;
            ok &= ply_detail::copyScalarsToFloatVector(*posData,     posFloats,     3);
            ok &= ply_detail::copyScalarsToFloatVector(*color3D,     colorFloats,   3);
            ok &= ply_detail::copyScalarsToFloatVector(*opacityData, opacityFloats, 1);
            ok &= ply_detail::copyScalarsToFloatVector(*scale3D,     scaleFloats,   3);
            ok &= ply_detail::copyScalarsToFloatVector(*rot3D,       rotFloats,     4);
            if (normal3D) ok &= ply_detail::copyScalarsToFloatVector(*normal3D, normalFloats, 3);
            if (!ok) {
                Log::PA_ERROR("PLYPointLoader: failed to unpack one or more 3DGS streams");
                return {};
            }

            geometry.positions.resize(vertexCount);
            geometry.tanU.resize(vertexCount);
            geometry.tanV.resize(vertexCount);
            geometry.scales.resize(vertexCount);
            geometry.colors.resize(vertexCount);
            geometry.opacities.resize(vertexCount);

            for (std::size_t i = 0; i < vertexCount; ++i) {
                const std::size_t i3 = i * 3;
                const std::size_t i4 = i * 4;

                geometry.positions[i] = glm::vec3(posFloats[i3 + 0], posFloats[i3 + 1], posFloats[i3 + 2]) / 10.0f;

                glm::vec3 rgb = glm::vec3(colorFloats[i3 + 0], colorFloats[i3 + 1], colorFloats[i3 + 2]);
                geometry.colors[i] = glm::clamp(rgb, 0.1f, 1.0f);

                geometry.opacities[i] = sigmoid(opacityFloats[i]) + 0.5f;

                const float qw = rotFloats[i4 + 0];
                const float qx = rotFloats[i4 + 1];
                const float qy = rotFloats[i4 + 2];
                const float qz = rotFloats[i4 + 3];
                glm::mat3 rotationMatrix = glm::mat3_cast(glm::normalize(glm::quat(qw, qx, qy, qz)));

                // take first two columns
                glm::vec3 tangentUCandidate(rotationMatrix[0][0], rotationMatrix[1][0], rotationMatrix[2][0]);
                glm::vec3 tangentVCandidate(rotationMatrix[0][1], rotationMatrix[1][1], rotationMatrix[2][1]);

                // optional normal
                const bool haveNormal = !normalFloats.empty();
                glm::vec3 providedNormal;
                if (haveNormal) {
                    providedNormal = glm::vec3(normalFloats[i3 + 0], normalFloats[i3 + 1], normalFloats[i3 + 2]);
                    if (!glm::all(glm::isfinite(providedNormal)))
                        providedNormal = glm::vec3(0,0,0);
                }


                ply_detail::orthonormalizeTangents(tangentUCandidate, tangentVCandidate);
                geometry.tanU[i] = tangentUCandidate;
                geometry.tanV[i] = tangentVCandidate;

                const float scaleU = expSafe(scaleFloats[i3 + 0]);
                const float scaleV = expSafe(scaleFloats[i3 + 1]);
                geometry.scales[i] = glm::vec2(scaleU, scaleV);
            }

            Log::PA_INFO("PLYPointLoader: loaded {} 3DGS splats (DC color, rot[0:1] → in-plane basis, scale[0:1] → radii)", vertexCount);
            return pointAsset;
        }
    }
};


} // namespace Pale
