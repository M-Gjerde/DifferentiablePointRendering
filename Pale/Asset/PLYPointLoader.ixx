// =====================================
// File: Pale.Assets.PLYPointLoader.ixx
// =====================================
module;

#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#define GLM_ENABLE_EXPERIMENTAL
#include <cstring>
#include <glm/gtx/norm.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtx/orthonormalize.hpp>

#include "tinyply.h"

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

} // namespace Pale::ply_detail

export namespace Pale {

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
        try {
            plyFile.parse_header(inputFile);
        } catch (const std::exception &e) {
            Log::PA_ERROR("PLYPointLoader: header parse failed: {}", e.what());
            return {};
        }

        // Required 2DGS properties
        std::shared_ptr<tinyply::PlyData> posData;
        std::shared_ptr<tinyply::PlyData> tuData, tvData;
        std::shared_ptr<tinyply::PlyData> scaleData;
        std::shared_ptr<tinyply::PlyData> colorData;
        std::shared_ptr<tinyply::PlyData> opacityData;

        try {
            posData    = plyFile.request_properties_from_element("vertex", {"x","y","z"});
            tuData     = plyFile.request_properties_from_element("vertex", {"tu_x","tu_y","tu_z"});
            tvData     = plyFile.request_properties_from_element("vertex", {"tv_x","tv_y","tv_z"});
            scaleData  = plyFile.request_properties_from_element("vertex", {"su","sv"});
            colorData  = plyFile.request_properties_from_element("vertex", {"albedo_r","albedo_g","albedo_b"});
            opacityData= plyFile.request_properties_from_element("vertex", {"opacity"});
        } catch (const std::exception &e) {
            Log::PA_ERROR("PLYPointLoader: required vertex properties missing: {}", e.what());
            return {};
        }

        try { plyFile.read(inputFile); }
        catch (const std::exception &e) {
            Log::PA_ERROR("PLYPointLoader: read failed: {}", e.what());
            return {};
        }

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
        if (!(sameCount("tu_*", tuData) && sameCount("tv_*", tvData) &&
              sameCount("su,sv", scaleData) && sameCount("albedo_*", colorData) &&
              sameCount("opacity", opacityData)))
            return {};

        // Unpack into float arrays
        std::vector<float> posFloats, tuFloats, tvFloats, scaleFloats, colorFloats, opacityFloats;
        bool ok = true;
        ok &= ply_detail::copyScalarsToFloatVector(*posData,    posFloats,    3);
        ok &= ply_detail::copyScalarsToFloatVector(*tuData,     tuFloats,     3);
        ok &= ply_detail::copyScalarsToFloatVector(*tvData,     tvFloats,     3);
        ok &= ply_detail::copyScalarsToFloatVector(*scaleData,  scaleFloats,  2);
        ok &= ply_detail::copyScalarsToFloatVector(*colorData,  colorFloats,  3);
        ok &= ply_detail::copyScalarsToFloatVector(*opacityData,opacityFloats,1);
        if (!ok) {
            Log::PA_ERROR("PLYPointLoader: failed to unpack one or more streams");
            return {};
        }

        // Build asset
        auto pointAsset = std::make_shared<PointAsset>();
        pointAsset->points.emplace_back();
        PointGeometry &geom = pointAsset->points.back();

        geom.positions.resize(vertexCount);
        geom.tanU.resize(vertexCount);
        geom.tanV.resize(vertexCount);
        geom.scales.resize(vertexCount);
        geom.colors.resize(vertexCount);
        geom.opacities.resize(vertexCount);

        for (std::size_t i = 0; i < vertexCount; ++i) {
            const std::size_t i3 = i * 3, i2 = i * 2;

            geom.positions[i] = glm::vec3(posFloats[i3 + 0], posFloats[i3 + 1], posFloats[i3 + 2]);
            geom.tanU[i]      = glm::vec3(tuFloats[i3 + 0],  tuFloats[i3 + 1],  tuFloats[i3 + 2]);
            geom.tanV[i]      = glm::vec3(tvFloats[i3 + 0],  tvFloats[i3 + 1],  tvFloats[i3 + 2]);
            geom.scales[i]     = glm::vec2(scaleFloats[i2 + 0], scaleFloats[i2 + 1]);
            geom.colors[i]    = glm::vec3(colorFloats[i3 + 0], colorFloats[i3 + 1], colorFloats[i3 + 2]);
            geom.opacities[i] = opacityFloats[i];
            ply_detail::orthonormalizeTangents(geom.tanU[i], geom.tanV[i]);
        }

        Log::PA_INFO("PLYPointLoader: loaded {} splats", vertexCount);
        return pointAsset;
    }
};

} // namespace Pale