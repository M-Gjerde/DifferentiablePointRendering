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

    class SceneUpload {
    public:
        // ---------------------------------------------------------------------
        // 1) Allocate: device memory based on BuildProducts, no memcpy
        // ---------------------------------------------------------------------
        static GPUSceneBuffers allocate(const SceneBuild::BuildProducts &bp,
                                        sycl::queue queue) {
            GPUSceneBuffers g{};

            // geometry sizes
            g.vertexCount        = static_cast<uint32_t>(bp.vertices.size());
            g.pointCount         = static_cast<uint32_t>(bp.points.size());
            g.triangleCount      = static_cast<uint32_t>(bp.triangles.size());
            g.blasNodeCount      = static_cast<uint32_t>(bp.bottomLevelNodes.size());
            g.tlasNodeCount      = static_cast<uint32_t>(bp.topLevelNodes.size());

            // light sizes
            g.lightCount             = static_cast<uint32_t>(bp.lights.size());
            g.emissiveTriangleCount  = static_cast<uint32_t>(bp.emissiveTriangles.size());

            // allocate geometry
            if (g.vertexCount) {
                g.vertices = static_cast<Vertex *>(
                    sycl::malloc_device(g.vertexCount * sizeof(Vertex), queue));
            }
            if (g.triangleCount) {
                g.triangles = static_cast<Triangle *>(
                    sycl::malloc_device(g.triangleCount * sizeof(Triangle), queue));
            }
            if (g.pointCount) {
                g.points = static_cast<Point *>(
                    sycl::malloc_device(g.pointCount * sizeof(Point), queue));
            }
            if (g.blasNodeCount) {
                g.blasNodes = static_cast<BVHNode *>(
                    sycl::malloc_device(g.blasNodeCount * sizeof(BVHNode), queue));
            }
            if (!bp.bottomLevelRanges.empty()) {
                g.blasRanges = static_cast<BLASRange *>(
                    sycl::malloc_device(bp.bottomLevelRanges.size() * sizeof(BLASRange), queue));
            }
            if (g.tlasNodeCount) {
                g.tlasNodes = static_cast<TLASNode *>(
                    sycl::malloc_device(g.tlasNodeCount * sizeof(TLASNode), queue));
            }
            if (!bp.transforms.empty()) {
                g.transforms = static_cast<Transform *>(
                    sycl::malloc_device(bp.transforms.size() * sizeof(Transform), queue));
            }
            if (!bp.materials.empty()) {
                g.materials = static_cast<GPUMaterial *>(
                    sycl::malloc_device(bp.materials.size() * sizeof(GPUMaterial), queue));
            }
            if (!bp.instances.empty()) {
                g.instances = static_cast<InstanceRecord *>(
                    sycl::malloc_device(bp.instances.size() * sizeof(InstanceRecord), queue));
            }

            // allocate lights
            if (g.lightCount) {
                g.lights = static_cast<GPULightRecord *>(
                    sycl::malloc_device(g.lightCount * sizeof(GPULightRecord), queue));
            }
            if (g.emissiveTriangleCount) {
                g.emissiveTriangles = static_cast<GPUEmissiveTriangle *>(
                    sycl::malloc_device(g.emissiveTriangleCount * sizeof(GPUEmissiveTriangle), queue));
            }

            return g;
        }

        // ---------------------------------------------------------------------
        // 2) Upload: memcpy CPU BuildProducts into already-allocated GPU buffers
        // ---------------------------------------------------------------------
        static void upload(const SceneBuild::BuildProducts &bp,
                           GPUSceneBuffers &g,
                           sycl::queue queue) {
            // Optional: sanity checks – you can tighten/relax as you wish
            if (g.vertexCount != bp.vertices.size() ||
                g.pointCount  != bp.points.size()  ||
                g.triangleCount != bp.triangles.size() ||
                g.blasNodeCount != bp.bottomLevelNodes.size() ||
                g.tlasNodeCount != bp.topLevelNodes.size()) {
                throw std::runtime_error("SceneUpload::upload: GPU buffer sizes "
                                         "do not match BuildProducts sizes");
            }

            if (g.vertexCount && g.vertices && !bp.vertices.empty()) {
                queue.memcpy(g.vertices,
                             bp.vertices.data(),
                             g.vertexCount * sizeof(Vertex));
            }
            if (g.triangleCount && g.triangles && !bp.triangles.empty()) {
                queue.memcpy(g.triangles,
                             bp.triangles.data(),
                             g.triangleCount * sizeof(Triangle));
            }
            if (g.pointCount && g.points && !bp.points.empty()) {
                queue.memcpy(g.points,
                             bp.points.data(),
                             g.pointCount * sizeof(Point));
            }
            if (g.blasNodeCount && g.blasNodes && !bp.bottomLevelNodes.empty()) {
                queue.memcpy(g.blasNodes,
                             bp.bottomLevelNodes.data(),
                             g.blasNodeCount * sizeof(BVHNode));
            }
            if (g.blasRanges && !bp.bottomLevelRanges.empty()) {
                queue.memcpy(g.blasRanges,
                             bp.bottomLevelRanges.data(),
                             bp.bottomLevelRanges.size() * sizeof(BLASRange));
            }
            if (g.tlasNodeCount && g.tlasNodes && !bp.topLevelNodes.empty()) {
                queue.memcpy(g.tlasNodes,
                             bp.topLevelNodes.data(),
                             g.tlasNodeCount * sizeof(TLASNode));
            }
            if (g.transforms && !bp.transforms.empty()) {
                queue.memcpy(g.transforms,
                             bp.transforms.data(),
                             bp.transforms.size() * sizeof(Transform));
            }
            if (g.materials && !bp.materials.empty()) {
                queue.memcpy(g.materials,
                             bp.materials.data(),
                             bp.materials.size() * sizeof(GPUMaterial));
            }
            if (g.instances && !bp.instances.empty()) {
                queue.memcpy(g.instances,
                             bp.instances.data(),
                             bp.instances.size() * sizeof(InstanceRecord));
            }
            if (g.lightCount && g.lights && !bp.lights.empty()) {
                queue.memcpy(g.lights,
                             bp.lights.data(),
                             g.lightCount * sizeof(GPULightRecord));
            }
            if (g.emissiveTriangleCount && g.emissiveTriangles && !bp.emissiveTriangles.empty()) {
                queue.memcpy(g.emissiveTriangles,
                             bp.emissiveTriangles.data(),
                             g.emissiveTriangleCount * sizeof(GPUEmissiveTriangle));
            }

            queue.wait();
        }

        // ---------------------------------------------------------------------
        // 3) Convenience: old behavior (allocate + upload)
        // ---------------------------------------------------------------------
        static GPUSceneBuffers allocateAndUpload(const SceneBuild::BuildProducts &bp,
                                                 sycl::queue queue) {
            GPUSceneBuffers g = allocate(bp, queue);
            upload(bp, g, queue);
            return g;
        }
    };
}
