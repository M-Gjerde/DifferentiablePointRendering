//
// Created by magnus on 8/29/25.
//
module;
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cfloat>
#include <sycl/sycl.hpp>

#include <Renderer/GPUDataStructures.h>

export module Pale.Render.BVH;


export namespace Pale {
    struct AABB { float3 minP, maxP; };
    inline float3 aabbCentroid(const AABB& b){ return (b.minP + b.maxP) * 0.5f; }

      //------------------------------------------------------------------------------
    // Basic BVH interface
    //------------------------------------------------------------------------------
    class BasicBVH {
    public:
        /// Build over 'tris' & 'verts'.
        static void build(std::vector<Triangle> &inTris,
                          const std::vector<Vertex> &verts,
                          std::vector<BVHNode> &nodes,
                          std::vector<uint32_t> &triIndices,
                          uint32_t maxLeafSize = 16) {
            // 1) copy and centroid
            std::vector<Triangle> &tris = inTris;
            // 2) init root
            nodes.clear();
            nodes.reserve(tris.size() * 2);
            triIndices.resize(tris.size());
            for (uint32_t i = 0; i < triIndices.size(); ++i)
                triIndices[i] = i;
            nodes.emplace_back();
            nodes[0].leftFirst = 0;
            nodes[0].triCount = uint32_t(tris.size());
            updateBounds(tris, verts, triIndices, nodes[0]);
            // 3) recursive split
            subdivide(tris, verts, nodes, triIndices, 0, maxLeafSize);
        }

         /// Build BVH over boxes (AABBs). Produces node array and a permutation
    /// newOrder[i] = oldLocalIndex of the i-th primitive in BVH order.
    static void buildFromBoxes(const std::vector<AABB>& localAabbs,
                               const std::vector<float3>& localCentroids, // same size as localAabbs
                               std::vector<BVHNode>& outNodes,
                               std::vector<uint32_t>& outNewOrder,
                               uint32_t maxLeafSize)
    {
        const auto primitiveCount = static_cast<uint32_t>(localAabbs.size());
        outNodes.clear();
        outNodes.reserve(std::max<uint32_t>(1u, primitiveCount * 2));

        outNewOrder.resize(primitiveCount);
        for (uint32_t i = 0; i < primitiveCount; ++i) outNewOrder[i] = i;

        outNodes.emplace_back(); // root
        outNodes[0].leftFirst = 0;
        outNodes[0].triCount  = primitiveCount;
        updateBoundsFromBoxes(localAabbs, outNewOrder, outNodes[0]);
        subdivideBoxes(localAabbs, localCentroids, outNodes, outNewOrder, 0, maxLeafSize);
    }

private:
    // Fit node AABB to the primitives in [leftFirst, leftFirst+triCount)
    static void updateBoundsFromBoxes(const std::vector<AABB>& localAabbs,
                                      const std::vector<uint32_t>& newOrder,
                                      BVHNode& node)
    {
        float3 boxMin{ std::numeric_limits<float>::infinity() };
        float3 boxMax{ -std::numeric_limits<float>::infinity() };
        for (uint32_t i = 0; i < node.triCount; ++i) {
            const AABB& b = localAabbs[newOrder[node.leftFirst + i]];
            boxMin = min(boxMin, b.minP);
            boxMax = max(boxMax, b.maxP);
        }
        node.aabbMin = boxMin;
        node.aabbMax = boxMax;
    }

    // Recursive median split on centroids along longest axis
    static void subdivideBoxes(const std::vector<AABB>& localAabbs,
                               const std::vector<float3>& localCentroids,
                               std::vector<BVHNode>& nodes,
                               std::vector<uint32_t>& newOrder,
                               uint32_t nodeIndex,
                               uint32_t maxLeafSize)
    {
        BVHNode& node = nodes[nodeIndex];
        if (node.triCount <= maxLeafSize) return;

        const float3 nodeExtent = node.aabbMax - node.aabbMin;
        const int splitAxis =
            (nodeExtent.y() > nodeExtent.x() && nodeExtent.y() > nodeExtent.z()) ? 1 :
            (nodeExtent.z() > nodeExtent.x() && nodeExtent.z() > nodeExtent.y()) ? 2 : 0;

        auto rangeBegin = newOrder.begin() + node.leftFirst;
        auto rangeEnd   = rangeBegin + node.triCount;
        auto rangeMid   = rangeBegin + (node.triCount >> 1);

        std::nth_element(rangeBegin, rangeMid, rangeEnd,
            [&](uint32_t a, uint32_t b) {
                const float ca = axisValue(localCentroids[a], splitAxis);
                const float cb = axisValue(localCentroids[b], splitAxis);
                return ca < cb;
            });
        const uint32_t leftCount = static_cast<uint32_t>(rangeMid - rangeBegin);
        if (leftCount == 0 || leftCount == node.triCount) return; // degenerate; stop

        const uint32_t leftChild  = static_cast<uint32_t>(nodes.size());
        const uint32_t rightChild = leftChild + 1;
        nodes.emplace_back();
        nodes.emplace_back();

        nodes[leftChild].leftFirst  = node.leftFirst;
        nodes[leftChild].triCount   = leftCount;
        nodes[rightChild].leftFirst = node.leftFirst + leftCount;
        nodes[rightChild].triCount  = node.triCount - leftCount;

        node.leftFirst = leftChild; // internal
        node.triCount  = 0;

        updateBoundsFromBoxes(localAabbs, newOrder, nodes[leftChild]);
        updateBoundsFromBoxes(localAabbs, newOrder, nodes[rightChild]);

        subdivideBoxes(localAabbs, localCentroids, nodes, newOrder, leftChild,  maxLeafSize);
        subdivideBoxes(localAabbs, localCentroids, nodes, newOrder, rightChild, maxLeafSize);
    }

    private:

        static inline float axisValue(const float3& v, int axis) {
            return axis == 0 ? v.x() : (axis == 1 ? v.y() : v.z());
        }
        // (b) fit node.aabb to its triangles
        static void updateBounds(const std::vector<Triangle> &tris,
                                 const std::vector<Vertex> &verts,
                                 const std::vector<uint32_t> &triIndices,
                                 BVHNode &node) {
            float3 bmin{std::numeric_limits<float>::infinity()};
            float3 bmax{-std::numeric_limits<float>::infinity()};
            for (uint32_t i = 0; i < node.triCount; ++i) {
                const Triangle &T = tris[triIndices[node.leftFirst + i]];
                const float3 &p0 = verts[T.v0].pos;
                const float3 &p1 = verts[T.v1].pos;
                const float3 &p2 = verts[T.v2].pos;
                bmin = min(bmin, p0);
                bmin = min(bmin, p1);
                bmin = min(bmin, p2);
                bmax = max(bmax, p0);
                bmax = max(bmax, p1);
                bmax = max(bmax, p2);
            }
            node.aabbMin = bmin;
            node.aabbMax = bmax;
        }

        // (c) recursive subdivision
        static void subdivide(const std::vector<Triangle> &tris,
                              const std::vector<Vertex> &verts,
                              std::vector<BVHNode> &nodes,
                              std::vector<uint32_t> &triIndices,
                              uint32_t nodeIdx,
                              uint32_t maxLeafSize) {
            BVHNode &node = nodes[nodeIdx];
            if (node.triCount <= maxLeafSize)
                return;

            // 1) pick split axis = longest extent
            float3 extent = node.aabbMax - node.aabbMin;
            int axis = (extent.y() > extent.x() && extent.y() > extent.z())
                           ? 1
                           : (extent.z() > extent.x() && extent.z() > extent.y())
                                 ? 2
                                 : 0;

            // 2) find median by centroid along that axis
            auto start = triIndices.begin() + node.leftFirst;
            auto end = start + node.triCount;
            auto mid = start + (node.triCount >> 1);

            std::nth_element(start, mid, end,
                             [&](uint32_t a, uint32_t b) {
                                 return tris[a].centroid[axis] < tris[b].centroid[axis];
                             });

            uint32_t leftCount = uint32_t(mid - start);
            // 2b) if that was degenerate (all centroids equal), bail out
            if (leftCount == 0 || leftCount == node.triCount)
                return;

            // 3) carve out two new nodes
            uint32_t leftChild = uint32_t(nodes.size());
            uint32_t rightChild = leftChild + 1;
            nodes.emplace_back(); // for left
            nodes.emplace_back(); // for right

            // 4) assign their ranges
            nodes[leftChild].leftFirst = node.leftFirst;
            nodes[leftChild].triCount = leftCount;
            nodes[rightChild].leftFirst = node.leftFirst + leftCount;
            nodes[rightChild].triCount = node.triCount - leftCount;

            // 5) flip this into an internal node
            node.leftFirst = leftChild;
            node.triCount = 0;

            // 6) refit their AABBs
            updateBounds(tris, verts, triIndices, nodes[leftChild]);
            updateBounds(tris, verts, triIndices, nodes[rightChild]);

            // 7) recurse
            subdivide(tris, verts, nodes, triIndices, leftChild, maxLeafSize);
            subdivide(tris, verts, nodes, triIndices, rightChild, maxLeafSize);
        }
    };



}