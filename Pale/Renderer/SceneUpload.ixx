//
// Created by magnus on 8/29/25.
//
module;

#include <cstdint>
#include <memory>
#include <sycl/sycl.hpp>

#include "Renderer/GPUDataStructures.h"

export module Pale.Render.SceneUpload;

import Pale.Scene;
import Pale.Render.SceneBuild;

export namespace Pale {
    // device pointers & counts (triangles, vertices, instances, materials, lights, cameras, BVHs…)
    class SceneUpload {
    public:
        static GPUSceneBuffers upload(const SceneBuild::BuildProducts &bp,
                                      sycl::queue queue) {
            GPUSceneBuffers g{};

            // geometry sizes
            g.vertexCount = static_cast<uint32_t>(bp.vertices.size());
            g.triangleCount = static_cast<uint32_t>(bp.triangles.size());
            g.blasNodeCount = static_cast<uint32_t>(bp.bottomLevelNodes.size());
            g.tlasNodeCount = static_cast<uint32_t>(bp.topLevelNodes.size());

            // light sizes
            g.lightCount = static_cast<uint32_t>(bp.lights.size());
            g.emissiveTriangleCount = static_cast<uint32_t>(bp.emissiveTriangles.size());

            // alloc
            g.d_vertices = (Vertex *) sycl::malloc_device(g.vertexCount * sizeof(Vertex), queue);
            g.d_triangles = (Triangle *) sycl::malloc_device(g.triangleCount * sizeof(Triangle), queue);
            g.d_blasNodes = (BVHNode *) sycl::malloc_device(g.blasNodeCount * sizeof(BVHNode), queue);
            g.d_blasRanges = (BLASRange *) sycl::malloc_device(bp.bottomLevelRanges.size() * sizeof(BLASRange), queue);
            g.d_tlasNodes = (TLASNode *) sycl::malloc_device(g.tlasNodeCount * sizeof(TLASNode), queue);
            g.d_transforms = (Transform *) sycl::malloc_device(bp.transforms.size() * sizeof(Transform), queue);
            g.d_materials = (GPUMaterial *) sycl::malloc_device(bp.materials.size() * sizeof(GPUMaterial), queue);

            if (g.lightCount) {
                g.d_lights = (GPULightRecord *) sycl::malloc_device(
                    g.lightCount * sizeof(GPULightRecord), queue);
            }
            if (g.emissiveTriangleCount) {
                g.d_emissiveTriangles = (GPUEmissiveTriangle *) sycl::malloc_device(
                    g.emissiveTriangleCount * sizeof(GPUEmissiveTriangle), queue);
            }

            // copy
            if (g.vertexCount)
                queue.memcpy(g.d_vertices, bp.vertices.data(),
                             g.vertexCount * sizeof(Vertex));
            if (g.triangleCount)
                queue.memcpy(g.d_triangles, bp.triangles.data(),
                             g.triangleCount * sizeof(Triangle));
            if (g.blasNodeCount)
                queue.memcpy(g.d_blasNodes, bp.bottomLevelNodes.data(),
                             g.blasNodeCount * sizeof(BVHNode));
            if (!bp.bottomLevelRanges.empty())
                queue.memcpy(g.d_blasRanges, bp.bottomLevelRanges.data(),
                             bp.bottomLevelRanges.size() * sizeof(BLASRange));
            if (g.tlasNodeCount)
                queue.memcpy(g.d_tlasNodes, bp.topLevelNodes.data(),
                             g.tlasNodeCount * sizeof(TLASNode));
            if (!bp.transforms.empty())
                queue.memcpy(g.d_transforms, bp.transforms.data(),
                             bp.transforms.size() * sizeof(Transform));
            if (!bp.materials.empty())
                queue.memcpy(g.d_materials, bp.materials.data(),
                             bp.materials.size() * sizeof(GPUMaterial));
            if (g.lightCount)
                queue.memcpy(g.d_lights, bp.lights.data(),
                             g.lightCount * sizeof(GPULightRecord));
            if (g.emissiveTriangleCount)
                queue.memcpy(g.d_emissiveTriangles, bp.emissiveTriangles.data(),
                             g.emissiveTriangleCount * sizeof(GPUEmissiveTriangle));

            queue.wait();
            return g;
        }

    private:
    };
}
