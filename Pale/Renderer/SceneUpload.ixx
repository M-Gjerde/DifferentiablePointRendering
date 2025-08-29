//
// Created by magnus on 8/29/25.
//
module;

#include <cstdint>
#include <memory>
#include <sycl/sycl.hpp>

export module Pale.Render.SceneUpload;

import Pale.Render.GPUDataStructures;
import Pale.Scene;
import Pale.Render.SceneBuild;

export namespace Pale {

    // device pointers & counts (triangles, vertices, instances, materials, lights, cameras, BVHs…)
    class SceneUpload {
    public:

        struct GPUSceneBuffers {
            BVHNode*    d_blasNodes{nullptr};
            BLASRange*  d_blasRanges{nullptr};
            TLASNode*   d_tlasNodes{nullptr};
            Triangle*   d_triangles{nullptr};
            Vertex*     d_vertices{nullptr};
            Transform*  d_transforms{nullptr};
            GPUMaterial*d_materials{nullptr};
            uint32_t    blasNodeCount{0}, tlasNodeCount{0}, triangleCount{0}, vertexCount{0};
        };

        static GPUSceneBuffers upload(const SceneBuild::BuildProducts& bp, sycl::queue queue) {
            GPUSceneBuffers g{};

            g.vertexCount   = (uint32_t)bp.vertices.size();
            g.triangleCount = (uint32_t)bp.triangles.size();
            g.blasNodeCount = (uint32_t)bp.bottomLevelNodes.size();
            g.tlasNodeCount = (uint32_t)bp.topLevelNodes.size();

            g.d_vertices   = (Vertex*)   sycl::malloc_device(g.vertexCount*sizeof(Vertex), queue);
            g.d_triangles  = (Triangle*) sycl::malloc_device(g.triangleCount*sizeof(Triangle), queue);
            g.d_blasNodes  = (BVHNode*)  sycl::malloc_device(g.blasNodeCount*sizeof(BVHNode), queue);
            g.d_blasRanges = (BLASRange*)sycl::malloc_device(bp.bottomLevelRanges.size()*sizeof(BLASRange), queue);
            g.d_tlasNodes  = (TLASNode*) sycl::malloc_device(g.tlasNodeCount*sizeof(TLASNode), queue);
            g.d_transforms = (Transform*)sycl::malloc_device(bp.transforms.size()*sizeof(Transform), queue);
            g.d_materials  = (GPUMaterial*)sycl::malloc_device(bp.materials.size()*sizeof(GPUMaterial), queue);

            queue.memcpy(g.d_vertices,   bp.vertices.data(),   g.vertexCount*sizeof(Vertex));
            queue.memcpy(g.d_triangles,  bp.triangles.data(),  g.triangleCount*sizeof(Triangle));
            queue.memcpy(g.d_blasNodes,  bp.bottomLevelNodes.data(), g.blasNodeCount*sizeof(BVHNode));
            queue.memcpy(g.d_blasRanges, bp.bottomLevelRanges.data(), bp.bottomLevelRanges.size()*sizeof(BLASRange));
            queue.memcpy(g.d_tlasNodes,  bp.topLevelNodes.data(), g.tlasNodeCount*sizeof(TLASNode));
            if (!bp.transforms.empty())
                queue.memcpy(g.d_transforms, bp.transforms.data(), bp.transforms.size()*sizeof(Transform));
            if (!bp.materials.empty())
                queue.memcpy(g.d_materials, bp.materials.data(), bp.materials.size()*sizeof(GPUMaterial));
            queue.wait();
            return g;
        };

    private:
    };
}