#pragma once


namespace Pale {
    struct VisibilityGradResult {
        float transmittance;
        float3 dTdPk;
    };


    // -----------------------------------------------------------------------------
    // TLAS traversal with near-to-far ordering and multiplicative transmittance
    // -----------------------------------------------------------------------------
    SYCL_EXTERNAL static bool intersectSceneAdjoint(const Ray &rayWorld,
                                             WorldHit *worldHitOut,
                                             const GPUSceneBuffers &scene,
                                             Ray *rayLocal,
                                             rng::Xorshift128 &rng128) {
        const TLASNode *tlasNodes = scene.tlasNodes;
        const InstanceRecord *instanceRecords = scene.instances;
        const Transform *transforms = scene.transforms;

        bool foundAnySurfaceHit = false;
        const float3 inverseDirectionWorld = safeInvDir(rayWorld.direction);

        worldHitOut->t = FLT_MAX;

        SmallStack<256> traversalStack;
        traversalStack.push(0); // root

        float bestWorldTHit = std::numeric_limits<float>::infinity();
        float transmittanceProduct = 1.0f; // accumulate product over visited splat instances in front of the first hit

        while (!traversalStack.empty()) {
            const int nodeIndex = traversalStack.pop();
            const TLASNode &node = tlasNodes[nodeIndex];

            float nodeTEntry = 0.0f;
            if (!slabIntersectAABB(rayWorld, node, inverseDirectionWorld, bestWorldTHit, nodeTEntry))
                continue;

            if (node.count == 0) {
                // Internal TLAS node: near-to-far push
                const int leftIndex = node.leftChild;
                const int rightIndex = node.rightChild;

                float leftTEntry = std::numeric_limits<float>::infinity();
                float rightTEntry = std::numeric_limits<float>::infinity();

                const bool hitLeft = computeAabbEntry(rayWorld, tlasNodes[leftIndex], inverseDirectionWorld,
                                                      bestWorldTHit, leftTEntry);
                const bool hitRight = computeAabbEntry(rayWorld, tlasNodes[rightIndex], inverseDirectionWorld,
                                                       bestWorldTHit, rightTEntry);

                if (hitLeft && hitRight) pushNearFar(traversalStack, leftIndex, leftTEntry, rightIndex, rightTEntry);
                else if (hitLeft) traversalStack.push(leftIndex);
                else if (hitRight) traversalStack.push(rightIndex);
                continue;
            }

            // Leaf: exactly one instance
            const uint32_t instanceIndex = node.leftChild;
            const InstanceRecord &instance = instanceRecords[instanceIndex];
            const Transform &objectToWorld = transforms[instance.transformIndex];
            Ray rayObject = toObjectSpace(rayWorld, objectToWorld);
            LocalHit localHit{};
            bool acceptedHitInInstance = false;

            if (instance.geometryType == GeometryType::Mesh) {
                acceptedHitInInstance = intersectBLASMesh(rayObject, instance.blasRangeIndex, localHit, scene);
            } else {

                /*
                 *acceptedHitInInstance = intersectBLASPointCloud(rayObject, instance.blasRangeIndex, localHit, scene,
                                                                rng128);
                */
                *rayLocal = rayObject;

            }

            if (acceptedHitInInstance) {
                // Convert to world, test depth
                const float3 hitPointWorld = toWorldPoint(rayObject.origin + localHit.t * rayObject.direction,
                                                          objectToWorld);
                const float tWorld = dot(hitPointWorld - rayWorld.origin, rayWorld.direction);
                // assumes normalized direction
                if (tWorld < 0.0f || tWorld >= bestWorldTHit) continue;

                bestWorldTHit = tWorld;
                foundAnySurfaceHit = true;

                worldHitOut->hit = true;
                worldHitOut->t = tWorld;
                worldHitOut->hitPositionW = hitPointWorld;
                worldHitOut->instanceIndex = instanceIndex;
                worldHitOut->primitiveIndex = localHit.primitiveIndex;

                // Multiply transmissions seen before entering this instance with transmission before the accepted event inside it
                worldHitOut->transmissivity = transmittanceProduct * (instance.geometryType == GeometryType::PointCloud
                                                                          ? localHit.transmissivity
                                                                          : 1.0f);
                // Stop traversal because we found the nearest accepted hit
                continue;
            }

            // No accepted hit, but if this was a splat field we may have partial transmission through it
            if (!acceptedHitInInstance && instance.geometryType == GeometryType::PointCloud) {
                transmittanceProduct *= localHit.transmissivity;
            }
        }

        // If no surface hit at all, expose total transmission accumulated
        if (!foundAnySurfaceHit) {
            worldHitOut->hit = false;
            worldHitOut->transmissivity = transmittanceProduct;
        }

        return foundAnySurfaceHit;
    }


    SYCL_EXTERNAL static VisibilityGradResult
    accumulateVisibilityAndGradientPointCloud(const Ray &rayObject,
                                              float tMax, // segment end (e.g., next surface)
                                              uint32_t blasRangeIndex,
                                              uint32_t parameterSplatIndexK,
                                              const GPUSceneBuffers &scene) {
        const BLASRange &blasRange = scene.blasRanges[blasRangeIndex];
        const BVHNode *bvhNodes = scene.blasNodes + blasRange.firstNode;

        constexpr float rayEpsilon = 1e-4f;
        constexpr float sameDepthEpsilon = 1e-5f;

        float transmittanceRunning = 1.0f;
        float3 dLogTransmittance_dPk = float3{0, 0, 0};

        SmallStack<64> traversalStack;
        traversalStack.push(0);
        const float3 inverseDirection = safeInvDir(rayObject.direction);

        float pruningTHit = tMax; // prune anything beyond segment end

        float currentGroupDepth = -std::numeric_limits<float>::infinity();
        BoundedVector<float, 64> groupTHits;
        BoundedVector<float, 64> groupAlphas;
        BoundedVector<uint32_t, 64> groupIndices;
        auto clearGroup = [&]() {
            groupTHits.clear();
            groupAlphas.clear();
            groupIndices.clear();
        };

        auto flushGroup = [&]() {
            if (groupTHits.empty()) return;

            // composite alpha for this depth slice
            float productOneMinusAlpha = 1.0f;
            for (size_t i = 0; i < groupAlphas.size(); ++i) productOneMinusAlpha *= (1.0f - groupAlphas[i]);
            const float compositeAlpha = 1.0f - productOneMinusAlpha;

            // accumulate log-derivative only for k
            for (size_t i = 0; i < groupIndices.size(); ++i) {
                const uint32_t surfelIndex = groupIndices[i];
                const Point &surfel = scene.points[surfelIndex];

                const float3 unitTangentU = normalize(surfel.tanU);
                const float3 unitTangentV = normalize(surfel.tanV - unitTangentU * dot(unitTangentU, surfel.tanV));
                const float3 unitNormal = normalize(cross(unitTangentU, unitTangentV));

                float3 x = rayObject.origin;
                const float3 y = rayObject.origin + groupTHits[i] * rayObject.direction;
                float3 d= y - x;
                float D = dot(unitNormal,d);
                if (abs(D) <= 1e-6f)
                    continue;

                float t = dot(unitNormal, (surfel.position - x) / D);

                float3 pA = x + t * d;

                float u = dot(surfel.tanU, (pA - surfel.position)) / surfel.scale.x();
                float v = dot(surfel.tanV, (pA - surfel.position)) / surfel.scale.y();
                float alpha = exp(-0.5 * (u*u + v*v));
                const float alphaI = groupAlphas[i];

                float3 grad_u = (dot(surfel.tanU, d / D) * unitNormal - surfel.tanU) / surfel.scale.x();
                float3 grad_v = (dot(surfel.tanV, d / D) * unitNormal - surfel.tanV) / surfel.scale.y();

                float3 grad = alpha * (u * grad_u + v * grad_v);


                dLogTransmittance_dPk = dLogTransmittance_dPk + grad;
            }

            // update transmission
            transmittanceRunning *= (1.0f - compositeAlpha);
            clearGroup();
        };

        while (!traversalStack.empty()) {
            const int nodeIndex = traversalStack.pop();
            const BVHNode &node = bvhNodes[nodeIndex];

            float nodeTEntry = 0.0f;
            if (!slabIntersectAABB(rayObject, node, inverseDirection, pruningTHit, nodeTEntry))
                continue;

            if (node.triCount == 0) {
                const int leftIndex = node.leftFirst;
                const int rightIndex = node.leftFirst + 1;
                float leftT = INFINITY, rightT = INFINITY;
                const bool hitL = computeAabbEntry(rayObject, bvhNodes[leftIndex], inverseDirection, pruningTHit,
                                                   leftT);
                const bool hitR = computeAabbEntry(rayObject, bvhNodes[rightIndex], inverseDirection, pruningTHit,
                                                   rightT);
                if (hitL && hitR) pushNearFar(traversalStack, leftIndex, leftT, rightIndex, rightT);
                else if (hitL) traversalStack.push(leftIndex);
                else if (hitR) traversalStack.push(rightIndex);
                continue;
            }

            // leaf: gather and sort by t
            BoundedVector<float, 32> leafTHits;
            leafTHits.clear();
            BoundedVector<float, 32> leafAlphas;
            leafAlphas.clear();
            BoundedVector<uint32_t, 32> leafIndices;
            leafIndices.clear();

            for (uint32_t l = 0; l < node.triCount; ++l) {
                const uint32_t surfelIndex = node.leftFirst + l;
                const Point &surfel = scene.points[surfelIndex];

                //float tHit = 0.0f, opacityGeom = 0.0f;
                //if (!intersectSurfel(rayObject, surfel, rayEpsilon, pruningTHit, tHit, opacityGeom))
                //    continue;

                //const float alphaAtHit = sycl::clamp(opacityGeom * surfel.opacity, 0.0f, 1.0f);
//
                //leafTHits.pushBack(tHit);
                //leafAlphas.pushBack(alphaAtHit);
                //leafIndices.pushBack(surfelIndex);
            }

            BoundedVector<int, 32> order;
            order.clear();
            BoundedVector<float, 32> keys;
            keys.clear();
            for (int i = 0; i < leafTHits.size(); ++i) {
                order.pushBack(i);
                keys.pushBack(leafTHits[i]);
            }
            insertionSortByKey(keys.data(), order.data(), leafTHits.size());

            for (int k = 0; k < leafTHits.size(); ++k) {
                const int idx = order[k];
                const float tHit = leafTHits[idx];
                const float alpha = leafAlphas[idx];

                if (groupTHits.size() == 0 || sycl::fabs(tHit - currentGroupDepth) <= sameDepthEpsilon) {
                    if (groupTHits.size() == 0) currentGroupDepth = tHit;
                    groupTHits.pushBack(tHit);
                    groupAlphas.pushBack(alpha);
                    groupIndices.pushBack(leafIndices[idx]);
                } else {
                    flushGroup();
                    currentGroupDepth = tHit;
                    groupTHits.clear();
                    groupAlphas.clear();
                    groupIndices.clear();
                    groupTHits.pushBack(tHit);
                    groupAlphas.pushBack(alpha);
                    groupIndices.pushBack(leafIndices[idx]);
                }
            }
        }
        if (groupTHits.size() > 0) flushGroup();

        return VisibilityGradResult{
            transmittanceRunning,
            transmittanceRunning * dLogTransmittance_dPk
        };
    }
}
