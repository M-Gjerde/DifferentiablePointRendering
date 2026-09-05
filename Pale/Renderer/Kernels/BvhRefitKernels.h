#pragma once

#include "KernelHelpers.h"
#include <cfloat>

// Shared arithmetic for serial and level-scheduled refits. Neither path changes
// support, topology, primitive order, or the packed traversal representation.
namespace Pale::PointBvhRefitDetail {

struct NodeTask {
    std::uint32_t blasRangeIndex;
    std::uint32_t localNodeIndex;
};

inline float dot3(const Pale::float3 &a, const Pale::float3 &b) {
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}
inline Pale::float3 cross3(const Pale::float3 &a, const Pale::float3 &b) {
    return Pale::float3{
        a.y() * b.z() - a.z() * b.y(),
        a.z() * b.x() - a.x() * b.z(),
        a.x() * b.y() - a.y() * b.x()
    };
}
inline Pale::float3 normalizeRefitVector(const Pale::float3 &value,
                               const Pale::float3 &fallback) {
    const float lengthSquared = dot3(value, value);
    if (!sycl::isfinite(lengthSquared) || lengthSquared <= 1.0e-20f) {
        return fallback;
    }
    return value * sycl::rsqrt(lengthSquared);
}
inline Pale::float3 min3(const Pale::float3 &a, const Pale::float3 &b) {
    return Pale::float3{
        sycl::fmin(a.x(), b.x()),
        sycl::fmin(a.y(), b.y()),
        sycl::fmin(a.z(), b.z())
    };
}
inline Pale::float3 max3(const Pale::float3 &a, const Pale::float3 &b) {
    return Pale::float3{
        sycl::fmax(a.x(), b.x()),
        sycl::fmax(a.y(), b.y()),
        sycl::fmax(a.z(), b.z())
    };
}
inline void makeSurfelAabbBeta(const Pale::Point &surfel,
                               Pale::float3 &aabbMin,
                               Pale::float3 &aabbMax) {
    const Pale::float3 tangentU =
            normalizeRefitVector(surfel.tanU, Pale::float3{1.0f, 0.0f, 0.0f});
    const Pale::float3 tangentV =
            normalizeRefitVector(surfel.tanV, Pale::float3{0.0f, 1.0f, 0.0f});
    const Pale::float3 normalDirection =
            normalizeRefitVector(cross3(tangentU, tangentV), Pale::float3{0.0f, 0.0f, 1.0f});

    const float supportRadiusU = sycl::fmax(surfel.scale.x(), 0.0f);
    const float supportRadiusV = sycl::fmax(surfel.scale.y(), 0.0f);
    constexpr float normalThickness = 0.0001f;

    auto computeAxisExtent = [&](int axisIndex) -> float {
        const float tangentUComponent =
                axisIndex == 0
                    ? tangentU.x()
                    : (axisIndex == 1 ? tangentU.y() : tangentU.z());
        const float tangentVComponent =
                axisIndex == 0
                    ? tangentV.x()
                    : (axisIndex == 1 ? tangentV.y() : tangentV.z());
        const float normalComponent =
                sycl::fabs(axisIndex == 0
                               ? normalDirection.x()
                               : (axisIndex == 1 ? normalDirection.y() : normalDirection.z()));
        const float projectedInPlane =
                sycl::sqrt((supportRadiusU * tangentUComponent) *
                           (supportRadiusU * tangentUComponent) +
                           (supportRadiusV * tangentVComponent) *
                           (supportRadiusV * tangentVComponent));
        return projectedInPlane + normalThickness * normalComponent;
    };

    const Pale::float3 halfExtent{
        computeAxisExtent(0),
        computeAxisExtent(1),
        computeAxisExtent(2)
    };
    aabbMin = surfel.position - halfExtent;
    aabbMax = surfel.position + halfExtent;
}
inline void writePackedPointBvhChild(Pale::PackedPointBVHNode &packedNode,
                                   bool writeLeftChild,
                                   const Pale::BVHNode *nodes,
                                   std::uint32_t childNodeIndex) {
    const Pale::BVHNode &childNode = nodes[childNodeIndex];
    const std::uint32_t childIndex =
            childNode.triCount > 0u ? childNode.leftFirst : childNodeIndex;
    const std::uint32_t childCount =
            childNode.triCount > 0u ? childNode.triCount : 0u;

    if (writeLeftChild) {
        packedNode.leftAabbMin = childNode.aabbMin;
        packedNode.leftAabbMax = childNode.aabbMax;
        packedNode.leftIndex = childIndex;
        packedNode.leftCount = childCount;
    } else {
        packedNode.rightAabbMin = childNode.aabbMin;
        packedNode.rightAabbMax = childNode.aabbMax;
        packedNode.rightIndex = childIndex;
        packedNode.rightCount = childCount;
    }
}
inline void writePackedPointBvhNode(Pale::PackedPointBVHNode *packedNodes,
                                   const Pale::BVHNode *nodes,
                                   std::uint32_t localNodeIndex) {
    const Pale::BVHNode &node = nodes[localNodeIndex];
    Pale::PackedPointBVHNode packedNode{};
    if (node.triCount > 0u) {
        writePackedPointBvhChild(packedNode, true, nodes, localNodeIndex);
    } else {
        writePackedPointBvhChild(packedNode, true, nodes, node.leftFirst);
        writePackedPointBvhChild(packedNode, false, nodes, node.leftFirst + 1u);
    }
    packedNodes[localNodeIndex] = packedNode;
}
inline void updatePackedPointQbvhNode(Pale::PackedPointQBVHNode &qbvhNode,
                                    const Pale::BVHNode *nodes) {
    for (std::uint32_t slot = 0u; slot < 4u; ++slot) {
        const std::uint32_t sourceNodeIndex = qbvhNode.childSourceNodeIndex[slot];
        if (sourceNodeIndex == UINT32_MAX) {
            continue;
        }

        const Pale::BVHNode &sourceNode = nodes[sourceNodeIndex];
        qbvhNode.minX[slot] = sourceNode.aabbMin.x();
        qbvhNode.minY[slot] = sourceNode.aabbMin.y();
        qbvhNode.minZ[slot] = sourceNode.aabbMin.z();
        qbvhNode.maxX[slot] = sourceNode.aabbMax.x();
        qbvhNode.maxY[slot] = sourceNode.aabbMax.y();
        qbvhNode.maxZ[slot] = sourceNode.aabbMax.z();

        if (sourceNode.triCount > 0u) {
            qbvhNode.childIndex[slot] = sourceNode.leftFirst;
            qbvhNode.childCount[slot] = sourceNode.triCount;
        } else {
            qbvhNode.childCount[slot] = 0u;
        }
    }
}


template<bool Leaf>
inline void refitNode(const GPUSceneBuffers &scene, const NodeTask task) {
    const BLASRange range = scene.blasRanges[task.blasRangeIndex];
    BVHNode *nodes = scene.blasNodes + range.firstNode;
    BVHNode &node = nodes[task.localNodeIndex];
    if constexpr (Leaf) {
        Pale::float3 nodeMin{FLT_MAX, FLT_MAX, FLT_MAX};
        Pale::float3 nodeMax{-FLT_MAX, -FLT_MAX, -FLT_MAX};
        for (std::uint32_t primitiveOffset = 0u;
             primitiveOffset < node.triCount;
             ++primitiveOffset) {
            const std::uint32_t primitiveIndex =
                    scene.pointPermutation[node.leftFirst + primitiveOffset];
            scene.pointTraversalData[node.leftFirst + primitiveOffset] =
                    Pale::makeSurfelTraversalData(scene.points[primitiveIndex], primitiveIndex);
            Pale::float3 surfelMin{0.0f};
            Pale::float3 surfelMax{0.0f};
            makeSurfelAabbBeta(scene.points[primitiveIndex], surfelMin, surfelMax);
            nodeMin = min3(nodeMin, surfelMin);
            nodeMax = max3(nodeMax, surfelMax);
        }
        node.aabbMin = nodeMin;
        node.aabbMax = nodeMax;
    } else {
        const Pale::BVHNode &leftNode = nodes[node.leftFirst];
        const Pale::BVHNode &rightNode = nodes[node.leftFirst + 1u];
        node.aabbMin = min3(leftNode.aabbMin, rightNode.aabbMin);
        node.aabbMax = max3(leftNode.aabbMax, rightNode.aabbMax);
    }
    if (scene.pointPackedBvhNodes && scene.pointPackedBvhRanges &&
        task.blasRangeIndex < scene.pointPackedBvhRangeCount) {
        const BLASRange packed = scene.pointPackedBvhRanges[task.blasRangeIndex];
        if (packed.nodeCount >= range.nodeCount &&
            packed.firstNode + packed.nodeCount <= scene.pointPackedBvhNodeCount) {
            writePackedPointBvhNode(scene.pointPackedBvhNodes + packed.firstNode,
                                    nodes, task.localNodeIndex);
        }
    }
}

inline void refitQbvhNode(const GPUSceneBuffers &scene, const NodeTask task) {
    const BLASRange range = scene.blasRanges[task.blasRangeIndex];
    const BLASRange packed = scene.pointQbvhRanges[task.blasRangeIndex];
    updatePackedPointQbvhNode(scene.pointQbvhNodes[packed.firstNode + task.localNodeIndex],
                             scene.blasNodes + range.firstNode);
}

inline void refitTlas(const GPUSceneBuffers &scene, std::uint32_t instanceCount) {
    for (int tlasNodeIndex = static_cast<int>(scene.tlasNodeCount) - 1;
         tlasNodeIndex >= 0;
         --tlasNodeIndex) {
        Pale::TLASNode &node = scene.tlasNodes[tlasNodeIndex];
        if (node.count > 0u) {
            const std::uint32_t instanceIndex = node.leftChild;
            if (instanceIndex >= instanceCount) {
                continue;
            }
            const Pale::InstanceRecord &instance = scene.instances[instanceIndex];
            const Pale::BLASRange blasRange = scene.blasRanges[instance.blasRangeIndex];
            if (blasRange.nodeCount == 0u) {
                continue;
            }
            const Pale::BVHNode &rootNode = scene.blasNodes[blasRange.firstNode];
            const Pale::Transform &transform = scene.transforms[instance.transformIndex];

            Pale::float3 worldMin{FLT_MAX, FLT_MAX, FLT_MAX};
            Pale::float3 worldMax{-FLT_MAX, -FLT_MAX, -FLT_MAX};
            for (int cornerIndex = 0; cornerIndex < 8; ++cornerIndex) {
                const bool bx = (cornerIndex & 4) != 0;
                const bool by = (cornerIndex & 2) != 0;
                const bool bz = (cornerIndex & 1) != 0;
                const Pale::float3 pointObject{
                    bx ? rootNode.aabbMax.x() : rootNode.aabbMin.x(),
                    by ? rootNode.aabbMax.y() : rootNode.aabbMin.y(),
                    bz ? rootNode.aabbMax.z() : rootNode.aabbMin.z()
                };
                const Pale::float3 pointWorld = Pale::toWorldPoint(pointObject, transform);
                worldMin = min3(worldMin, pointWorld);
                worldMax = max3(worldMax, pointWorld);
            }
            node.aabbMin = worldMin;
            node.aabbMax = worldMax;
        } else {
            const Pale::TLASNode &leftNode = scene.tlasNodes[node.leftChild];
            const Pale::TLASNode &rightNode = scene.tlasNodes[node.rightChild];
            node.aabbMin = min3(leftNode.aabbMin, rightNode.aabbMin);
            node.aabbMax = max3(leftNode.aabbMax, rightNode.aabbMax);
        }
    }
}

} // namespace Pale::PointBvhRefitDetail
