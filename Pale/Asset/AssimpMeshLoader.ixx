// =====================================
// File: Pale.Assets.AssimpMeshLoader.ixx
// =====================================
module;

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

export module Pale.Assets:AssimpMeshLoader;

import :Core;
import :Mesh;

export namespace Pale {
    struct AssimpMeshLoader : IAssetLoader<Mesh> {
        AssetPtr<Mesh> load(const AssetHandle& /*id*/, const AssetMeta& meta) override {
            Assimp::Importer importer;
            const aiScene* scene = importer.ReadFile(meta.path.string(),
            aiProcess_Triangulate |
            aiProcess_GenNormals |
            aiProcess_ImproveCacheLocality |
            aiProcess_CalcTangentSpace |
            aiProcess_JoinIdenticalVertices |
            aiProcess_ValidateDataStructure);


            if (!scene || !scene->HasMeshes()) return {};


            auto mesh = std::make_shared<Mesh>();
            mesh->submeshes.reserve(scene->mNumMeshes);


            for (unsigned m = 0; m < scene->mNumMeshes; ++m) {
                const aiMesh* am = scene->mMeshes[m];
                Submesh sm{};
                sm.positions.reserve(am->mNumVertices);
                sm.normals.reserve(am->mNumVertices);
                if (am->HasTextureCoords(0)) sm.uvs.reserve(am->mNumVertices);


                for (unsigned v = 0; v < am->mNumVertices; ++v) {
                    auto P = am->mVertices[v];
                    sm.positions.push_back({P.x, P.y, P.z});
                    if (am->HasNormals()) {
                        auto N = am->mNormals[v];
                        sm.normals.push_back({N.x, N.y, N.z});
                    } else {
                        sm.normals.push_back({0.f, 0.f, 1.f});
                    }
                    if (am->HasTextureCoords(0)) {
                        auto T = am->mTextureCoords[0][v];
                        sm.uvs.push_back({T.x, T.y});
                    }
                }


                for (unsigned f = 0; f < am->mNumFaces; ++f) {
                    const aiFace& face = am->mFaces[f];
                    if (face.mNumIndices == 3) {
                        sm.indices.push_back(face.mIndices[0]);
                        sm.indices.push_back(face.mIndices[1]);
                        sm.indices.push_back(face.mIndices[2]);
                    }
                }
                sm.materialIndex = static_cast<int>(am->mMaterialIndex);
                mesh->submeshes.emplace_back(std::move(sm));
            }
            return mesh;
        }
    };
}