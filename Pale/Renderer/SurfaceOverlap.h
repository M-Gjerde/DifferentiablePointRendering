#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <glm/glm.hpp>
#include "Renderer/LocalLayerMembership.h"

namespace Pale {
    struct SurfaceFootprint {
        glm::vec3 center{};
        // World-space ellipse axes, including radii and instance transforms.
        glm::vec3 u{}, v{};
        bool enabled = true;
        std::uint32_t instance = 0;

        SurfaceFootprint transformed(const glm::mat4& objectToWorld) const {
            auto result = *this;
            result.center = glm::vec3(objectToWorld * glm::vec4(center, 1.0f));
            // Preserve displacement lengths; these are not unit directions.
            result.u = glm::mat3(objectToWorld) * u;
            result.v = glm::mat3(objectToWorld) * v;
            return result;
        }
    };

    struct SurfaceOverlapSettings {
        float depthTolerance = 0.005f;
        float normalCosine = -1.0f;
        Pale::LocalLayerDepthMode depthMode = Pale::LocalLayerDepthMode::SymmetricRayDepth;
        unsigned maxSlabMembers = 8;
        float rayEpsilon = 1.0e-6f;
        static constexpr int sampleCount = 64;
        bool operator==(const SurfaceOverlapSettings&) const = default;
    };

    struct SurfaceOverlapView {
        glm::vec3 origin{};
        glm::mat4 worldToCamera{1};
        float fx = 1, fy = 1, cx = 0, cy = 0;
        unsigned width = 0, height = 0;
        bool operator==(const SurfaceOverlapView&) const = default;

        bool rayTo(const glm::vec3& position, glm::vec3& direction, float& distance) const {
            if (width && height) {
                const auto local = glm::vec3(worldToCamera * glm::vec4(position, 1));
                if (!(local.z < 0)) return false;
                const float x = cx + fx * local.x / -local.z;
                const float y = cy - fy * local.y / -local.z;
                if (!(x >= 0 && x < width && y >= 0 && y < height)) return false;
            }
            const auto delta = position - origin;
            distance = glm::length(delta);
            if (!(distance > 0) || !std::isfinite(distance)) return false;
            direction = delta / distance;
            return true;
        }
    };

    struct SurfaceOverlapScore {
        // Total potential members, INCLUDING the anchor surfel. Unobserved = NaN.
        float centerMembers = std::numeric_limits<float>::quiet_NaN();
        float meanMembers = std::numeric_limits<float>::quiet_NaN();
        float crowdedPercent = std::numeric_limits<float>::quiet_NaN();
        std::uint64_t observations = 0;
    };

    // Uncapped potential slab membership at surfel footprint samples, along
    // supplied camera rays. Shared renderer predicates define depth and normals.
    // No opacity filter, occlusion termination, hit-batch limit or member cap.
    class SurfaceOverlap {
        struct Footprint {
            glm::vec3 center{}, u{}, v{}, normal{}, dualU{}, dualV{}, lo{}, hi{};
            bool valid = false;
            std::uint32_t instance = 0;
        };
        struct Node {
            glm::vec3 lo{}, hi{};
            std::size_t begin{}, end{}, left{}, right{};
        };
        struct Query {
            glm::vec3 position{}, direction{}, lo{}, hi{};
            float anchorT = 0, rayHalfWidth = 0, referenceDotRay = 0;
        };
        std::vector<Footprint> footprints;
        std::vector<std::size_t> order;
        std::vector<Node> nodes;
        SurfaceOverlapSettings settings;

        std::size_t build(std::size_t begin, std::size_t end) {
            const auto index = nodes.size();
            const float infinity = std::numeric_limits<float>::infinity();
            Node node{glm::vec3(infinity), glm::vec3(-infinity), begin, end};
            for (auto i = begin; i < end; ++i) {
                node.lo = glm::min(node.lo, footprints[order[i]].lo);
                node.hi = glm::max(node.hi, footprints[order[i]].hi);
            }
            nodes.push_back(node);
            if (end - begin > 8) {
                const auto extent = node.hi - node.lo;
                const int axis = extent.x > extent.y
                    ? (extent.x > extent.z ? 0 : 2) : (extent.y > extent.z ? 1 : 2);
                const auto middle = begin + (end - begin) / 2;
                std::nth_element(order.begin() + begin, order.begin() + middle, order.begin() + end,
                    [&](auto a, auto b) { return footprints[a].center[axis] < footprints[b].center[axis]; });
                const auto left = build(begin, middle);
                const auto right = build(middle, end);
                nodes[index].left = left;
                nodes[index].right = right;
            }
            return index;
        }

        unsigned count(std::size_t nodeIndex, std::size_t self, const Query& query) const {
            const auto& node = nodes[nodeIndex];
            if (glm::any(glm::lessThan(query.hi, node.lo)) ||
                glm::any(glm::greaterThan(query.lo, node.hi))) return 0;
            if (node.left != 0) return count(node.left, self, query) + count(node.right, self, query);
            const auto& anchor = footprints[self];
            unsigned result = 0;
            for (auto i = node.begin; i < node.end; ++i) {
                const auto other = order[i];
                if (other == self) continue;
                const auto& neighbor = footprints[other];
                if (neighbor.instance != anchor.instance) continue;
                const float denominator = glm::dot(query.direction, neighbor.normal);
                if (std::abs(denominator) <= settings.rayEpsilon) continue;
                const float offset = glm::dot(neighbor.center - query.position, neighbor.normal) / denominator;
                const float candidateT = query.anchorT + offset;
                if (candidateT <= settings.rayEpsilon || !Pale::localLayerDepthRangeContains(
                    query.anchorT, candidateT, query.rayHalfWidth, settings.depthMode, settings.rayEpsilon)) continue;
                const float agreement = Pale::localLayerFacingNormalAgreement(
                    glm::dot(anchor.normal, neighbor.normal), query.referenceDotRay, denominator);
                const float normalDistance = std::abs(offset * query.referenceDotRay);
                if (!Pale::localLayerSurfaceMatches(agreement, normalDistance,
                    settings.normalCosine, settings.depthTolerance, settings.depthMode)) continue;
                const auto delta = query.position + offset * query.direction - neighbor.center;
                const float u = glm::dot(delta, neighbor.dualU);
                const float v = glm::dot(delta, neighbor.dualV);
                if (u * u + v * v < 1.0f) ++result;
            }
            return result;
        }

        bool queryAt(std::size_t self, const glm::vec3& position, const SurfaceOverlapView& view,
                     unsigned& members) const {
            Query query;
            query.position = position;
            if (!view.rayTo(position, query.direction, query.anchorT) || query.anchorT <= settings.rayEpsilon) return false;
            query.referenceDotRay = glm::dot(footprints[self].normal, query.direction);
            if (std::abs(query.referenceDotRay) <= settings.rayEpsilon) return false;
            query.rayHalfWidth = Pale::localLayerRayHalfWidth(
                query.referenceDotRay, settings.depthTolerance, settings.depthMode);
            const float minimumOffset = settings.depthMode == Pale::LocalLayerDepthMode::SymmetricRayDepth
                ? -query.rayHalfWidth : 0.0f;
            const auto start = position + (minimumOffset - settings.rayEpsilon) * query.direction;
            const auto end = position + query.rayHalfWidth * query.direction;
            query.lo = glm::min(start, end) - glm::vec3(settings.rayEpsilon);
            query.hi = glm::max(start, end) + glm::vec3(settings.rayEpsilon);
            members = 1u + count(0, self, query);
            return true;
        }

    public:
        SurfaceOverlap(const std::vector<SurfaceFootprint>& inputs, SurfaceOverlapSettings parameters)
            : footprints(inputs.size()), settings(parameters) {
            for (std::size_t i = 0; i < inputs.size(); ++i) {
                const auto& input = inputs[i];
                if (!input.enabled) continue;
                auto& footprint = footprints[i];
                footprint.center = input.center;
                footprint.u = input.u;
                footprint.v = input.v;
                footprint.instance = input.instance;
                const auto cross = glm::cross(footprint.u, footprint.v);
                const float determinant = glm::dot(cross, cross);
                if (!(determinant > 0.0f) || !std::isfinite(determinant)) continue;
                footprint.normal = cross / std::sqrt(determinant);
                footprint.dualU = glm::cross(footprint.v, cross) / determinant;
                footprint.dualV = glm::cross(cross, footprint.u) / determinant;
                const auto extent = glm::sqrt(footprint.u * footprint.u + footprint.v * footprint.v)
                    + glm::vec3(settings.rayEpsilon);
                footprint.lo = footprint.center - extent;
                footprint.hi = footprint.center + extent;
                footprint.valid = true;
                for (int axis = 0; axis < 3; ++axis) {
                    footprint.valid &= std::isfinite(footprint.lo[axis]) && std::isfinite(footprint.hi[axis]) &&
                        std::isfinite(footprint.dualU[axis]) && std::isfinite(footprint.dualV[axis]);
                }
                if (footprint.valid) order.push_back(i);
            }
            if (!order.empty()) build(0, order.size());
        }

        std::vector<SurfaceOverlapScore> evaluate(const std::vector<SurfaceOverlapView>& views) const {
            std::vector<SurfaceOverlapScore> result(footprints.size());
            constexpr float goldenAngle = 2.39996323f;
            glm::vec2 samples[SurfaceOverlapSettings::sampleCount];
            for (int s = 0; s < SurfaceOverlapSettings::sampleCount; ++s) {
                const float radius = std::sqrt((s + 0.5f) / SurfaceOverlapSettings::sampleCount);
                samples[s] = radius * glm::vec2(std::cos(s * goldenAngle), std::sin(s * goldenAngle));
            }
            #pragma omp parallel for schedule(dynamic, 64)
            for (std::size_t i = 0; i < footprints.size(); ++i) {
                const auto& footprint = footprints[i];
                if (!footprint.valid) continue;
                auto& score = result[i];
                std::uint64_t centerSum = 0, centerObservations = 0, sum = 0, crowded = 0;
                for (const auto& view : views) {
                    unsigned members;
                    if (queryAt(i, footprint.center, view, members)) {
                        centerSum += members;
                        ++centerObservations;
                    }
                    for (const auto& sample : samples) {
                        if (!queryAt(i, footprint.center + footprint.u * sample.x + footprint.v * sample.y,
                                     view, members)) continue;
                        sum += members;
                        crowded += members > settings.maxSlabMembers;
                        ++score.observations;
                    }
                }
                if (centerObservations) score.centerMembers = static_cast<float>(centerSum) / centerObservations;
                if (score.observations) {
                    score.meanMembers = static_cast<float>(sum) / score.observations;
                    score.crowdedPercent = 100.0f * crowded / score.observations;
                }
            }
            return result;
        }
    };
}
