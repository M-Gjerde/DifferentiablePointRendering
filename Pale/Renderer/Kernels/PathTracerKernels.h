#pragma once

#include <sycl/sycl.hpp>
#include <Renderer/GPUDataStructures.h>

#include "KernelHelpers.h"

namespace Pale {
    // ── PathTracerMeshKernel.cpp ────────────────────────────────────────────────
    // Returns the closest hit inside one mesh’s BLAS (object space)
    SYCL_EXTERNAL bool intersectBLASMesh(const Ray &rayO,
                                         uint32_t geomIdx,
                                         LocalHit &out, const GPUSceneBuffers &scene) {
        /* 1.  Locate the sub‑tree that belongs to this mesh
               ────────────────────────────────────────────── */
        const BLASRange &br = scene.blasRanges[geomIdx];
        const BVHNode *nodes = scene.blasNodes + br.firstNode; // root = nodes[0]
        const Triangle *tris = scene.triangles;
        const Vertex *verts = scene.vertices;

        /* 2.  Standard iterative depth‑first traversal
               ────────────────────────────────────────────── */
        float bestT = std::numeric_limits<float>::infinity();
        bool hitAny = false;
        float3 invDir = safeInvDir(rayO.direction);

        SmallStack<512> stack;
        stack.push(0); // root

        while (!stack.empty()) {
            int nIdx = stack.pop();

            const BVHNode &N = nodes[nIdx];

            float tEntry;
            if (!slabIntersectAABB(rayO, N, invDir, bestT, tEntry))
                continue; // miss or farther than current best

            if (N.triCount == 0) // ── internal ─────────────────────
            {
                /* Push children – right first so left is processed next.
                   Children are stored immediately after the parent once
                   we patched indices in buildBLASForAllMeshes().          */
                // when you push children:

                if (!stack.push(N.leftFirst + 1)) return hitAny; // overflow → miss
                if (!stack.push(N.leftFirst)) return hitAny;
            } else // ── leaf ─────────────────────────
            {
                for (uint32_t i = 0; i < N.triCount; ++i) {
                    uint32_t triIdx = N.leftFirst + i; // *global* index

                    const Triangle &T = tris[triIdx]; // existing line
                    const float3 A = verts[T.v0].pos;
                    const float3 B = verts[T.v1].pos;
                    const float3 C = verts[T.v2].pos;

                    float t = FLT_MAX;
                    float u = 0;
                    float v = 0;

                    /* ---- new call --------------------------------------------------------- */
                    if (intersectTriangle(rayO, A, B, C,
                                          t, u, v, 1e-4f)
                        && t < bestT) {
                        bestT = t;
                        hitAny = true;
                        out.t = t;
                        out.primitiveIndex = triIdx; // global – good for shading
                    }
                }
            }
        }

        return hitAny;
    }

    // ── PathTracerMeshKernel.cpp ────────────────────────────────────────────────
    // Returns the closest hit inside one mesh’s BLAS (object space)
    SYCL_EXTERNAL bool intersectBLASPointCloud(const Ray &rayObject,
                                               uint32_t blasRangeIndex,
                                               LocalHit &out, const GPUSceneBuffers &scene, rng::Xorshift128& rng128) {
        const BLASRange &blasRange = scene.blasRanges[blasRangeIndex];
        const BVHNode *bvhNodes = scene.blasNodes + blasRange.firstNode;

        float bestAcceptedT = std::numeric_limits<float>::infinity();
        bool foundAccepted = false;
        const float3 inverseDirection = safeInvDir(rayObject.direction);
        constexpr float rayEpsilon = 1e-4f;
        out.hasVisibilityTest = false;                  // reset for this BLAS
        bool sawTransmissionOnThisBLAS = false;         // sticky flag

        SmallStack<512> traversalStack;
        traversalStack.push(0);

        // Optional: obtain frameIndex from your per-frame constants

        while (!traversalStack.empty()) {
            const int nodeIndex = traversalStack.pop();
            const BVHNode &node = bvhNodes[nodeIndex];

            float tEntry = 0.0f;
            if (!slabIntersectAABB(rayObject, node, inverseDirection, bestAcceptedT, tEntry))
                continue;

            if (node.triCount == 0) {
                if (!traversalStack.push(node.leftFirst + 1)) return foundAccepted;
                if (!traversalStack.push(node.leftFirst)) return foundAccepted;
                continue;
            }

            // Leaf node - Start intersecting Gaussians
            for (uint32_t local = 0; local < node.triCount; ++local) {
                const uint32_t pointIndex = node.leftFirst + local;
                const Point &surfel = scene.points[pointIndex];


                float tHit = 0.0f;
                float contributionAtHit = 0.0f;
                if (intersectSurfel(rayObject, surfel, rayEpsilon, 10.0f, tHit, contributionAtHit)) {

                    // Find intersection point // Still in local coords
                    const float3 hitPointLocal = rayObject.origin + tHit * rayObject.direction;

                    // Generate a random number [0, 1] dependent on the position
                    float xi = rng128.nextFloat();
                    const float alphaAtHit = contributionAtHit * surfel.opacity;
                    if (xi < alphaAtHit) {                 // accept with probability alphaAtHit
                        bestAcceptedT = tHit;
                        foundAccepted = true;
                        out.primitiveIndex = pointIndex;
                        out.t = tHit;
                        out.transmissivity = 1 - alphaAtHit;
                    } else {
                        sawTransmissionOnThisBLAS = true;
                        out.transmissivity = 1 - alphaAtHit;
                        out.primitiveIndex = pointIndex;
                        out.t = tHit;
                    }

                }
            }
        }
        out.hasVisibilityTest = sawTransmissionOnThisBLAS;
        return foundAccepted;
    }

    SYCL_EXTERNAL bool intersectScene(const Ray &rayWorld, WorldHit *worldHit, const GPUSceneBuffers &scene, rng::Xorshift128& rng128) {
        /* abort if scene is empty */
        const TLASNode *tlas = scene.tlasNodes;
        const InstanceRecord *instances = scene.instances;
        const Transform *xforms = scene.transforms;

        /* ------------------------------------------------------------------ */
        /* stack‑based depth‑first traversal                                   */
        /* ------------------------------------------------------------------ */
        bool foundAnyHit = false;
        float3 invDir = safeInvDir(rayWorld.direction);;

        // initialize output
        worldHit->t = FLT_MAX;
        worldHit->hasVisibilityTest = false;
        SmallStack<256> stack;
        stack.push(0); // root
        float bestTWorld = std::numeric_limits<float>::infinity();

        while (!stack.empty()) {
            int nIdx = stack.pop();
            const TLASNode &node = tlas[nIdx];


            float tEntry;
            if (!slabIntersectAABB(rayWorld, node, invDir, bestTWorld, tEntry))
                continue;

            if (node.count == 0) // internal
            {
                if (!stack.push(node.rightChild)) return false;
                if (!stack.push(node.leftChild)) return false;
            } // leaf – exactly one instance
            else {
                const uint32_t instanceIndex = node.leftChild;
                const InstanceRecord &instance = instances[instanceIndex];
                const Transform &transform = xforms[instance.transformIndex];
                Ray rayObject = toObjectSpace(rayWorld, transform);
                LocalHit localHit{};
                bool ok = false;
                if (instance.geometryType == GeometryType::Mesh) {
                    ok = intersectBLASMesh(rayObject, instance.blasRangeIndex, localHit, scene);
                } else {
                    // point cloud
                    ok = intersectBLASPointCloud(rayObject, instance.blasRangeIndex, localHit, scene, rng128);
                }

                if (ok) {
                    const float3 hitPointW = toWorldPoint(rayObject.origin + localHit.t * rayObject.direction,
                                                          transform);
                    const float tWorld = dot(hitPointW - rayWorld.origin, rayWorld.direction);
                    if (tWorld < 0.0f || tWorld >= bestTWorld) continue;
                    bestTWorld = tWorld;
                    foundAnyHit = true;
                    worldHit->t = tWorld;
                    worldHit->transmissivity = localHit.transmissivity;
                    worldHit->primitiveIndex = localHit.primitiveIndex;
                    worldHit->instanceIndex = instanceIndex;
                    worldHit->hitPositionW = hitPointW;
                }

                if (!ok && localHit.hasVisibilityTest) {
                                        const float3 hitPointW = toWorldPoint(rayObject.origin + localHit.t * rayObject.direction,
                                                          transform);
                    const float tWorld = dot(hitPointW - rayWorld.origin, rayWorld.direction);

                    worldHit->t = tWorld;
                    worldHit->transmissivity = localHit.transmissivity;
                    worldHit->hasVisibilityTest = localHit.hasVisibilityTest;

                }
            }
        }

        return foundAnyHit;
    }
}
