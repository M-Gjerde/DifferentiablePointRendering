#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

#include "Renderer/RenderPackage.h"
#include "Renderer/GPUDataStructures.h"
#include "Renderer/GPUDataTypes.h"

namespace Pale::rng {
    // ---------- SplitMix64 for robust seeding ----------
    struct SplitMix64 {
        uint64_t currentState;

        explicit SplitMix64(uint64_t initialState) : currentState(initialState) {
        }

        inline uint64_t nextUint64() {
            uint64_t z = (currentState += 0x9E3779B97F4A7C15ull);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            return z ^ (z >> 31);
        }
    };

    // ---------- Xorshift32 (Marsaglia) ----------
    struct Xorshift32 {
        uint32_t currentState;

        explicit Xorshift32(uint32_t seed) : currentState(seed ? seed : 0xA341316Cu) {
        }

        inline uint32_t nextUint32() {
            uint32_t x = currentState;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            currentState = x;
            return x;
        }

        inline float nextFloat01() {
            // [0,1)
            // Use top 24 bits to avoid denorms and keep uniform mantissa fill
            return static_cast<float>(nextUint32() >> 8) * 0x1.0p-24f;
        }
    };

    // ---------- Xorshift64* (better quality than plain 64) ----------
    struct Xorshift64Star {
        uint64_t currentState;

        explicit Xorshift64Star(uint64_t seed) : currentState(seed ? seed : 0x9E3779B97F4A7C15ull) {
        }

        inline uint64_t nextUint64() {
            uint64_t x = currentState;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            currentState = x;
            return x * 0x2545F4914F6CDD1Dull;
        }

        inline double nextDouble01() {
            // [0,1)
            // Use top 53 bits for full double mantissa precision
            return (nextUint64() >> 11) * 0x1.0p-53;
        }
    };

    static constexpr uint32_t kStreamRayGen = 10u;
    static constexpr uint32_t kStreamTraversal = 1u;
    static constexpr uint32_t kStreamEvent = 2u;
    static constexpr uint32_t kStreamDirection = 3u;
    static constexpr uint32_t kStreamRoulette = 4u;
    static constexpr uint32_t kStreamDeposit = 5u;
    static constexpr uint32_t kStreamGather = 6u;
    static constexpr uint32_t kStreamDirectLight = 7u;

    // ---------- Xorshift128 (32-bit state, very fast) ----------
    struct Xorshift128 {
        uint32_t stateX, stateY, stateZ, stateW;

        Xorshift128(uint32_t sx, uint32_t sy, uint32_t sz, uint32_t sw)
            : stateX(sx ? sx : 0x1u),
              stateY(sy ? sy : 0x9E3779B9u),
              stateZ(sz ? sz : 0x7F4A7C15u),
              stateW(sw ? sw : 0x94D049BBu) {
        }

        explicit Xorshift128(uint64_t seed) {
            // Fill with SplitMix64 to avoid correlated seeds
            SplitMix64 sm(seed);
            stateX = static_cast<uint32_t>(sm.nextUint64());
            stateY = static_cast<uint32_t>(sm.nextUint64());
            stateZ = static_cast<uint32_t>(sm.nextUint64());
            stateW = static_cast<uint32_t>(sm.nextUint64());
            if (!(stateX | stateY | stateZ | stateW)) stateW = 0xA511E9B3u; // avoid all-zero
        }

        uint32_t nextUint() {
            uint32_t t = stateX ^ (stateX << 11);
            stateX = stateY;
            stateY = stateZ;
            stateZ = stateW;
            stateW = (stateW ^ (stateW >> 19)) ^ (t ^ (t >> 8));

            // output scrambling (one multiply)
            return stateW * 0x9E3779B1u;
        }

        float nextFloat() {
            // [0,1)
            return static_cast<float>(nextUint() >> 8) * 0x1.0p-24f;
        }
    };

    SYCL_EXTERNAL inline uint64_t mix64(uint64_t x) {
        // SplitMix64 finalizer (good 64-bit avalanche)
        x += 0x9E3779B97F4A7C15ull;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        return x ^ (x >> 31);
    }

    SYCL_EXTERNAL inline uint64_t hashCombine64(uint64_t state, uint64_t v) {
        // Similar spirit to boost hash combine, but with strong mixer
        return mix64(state ^ mix64(v));
    }

    SYCL_EXTERNAL inline uint64_t makeSeed(
        uint64_t renderSeed,
        uint64_t pathId,
        uint32_t bounceIndex,
        uint32_t streamTag,
        uint32_t dimension) {
        uint64_t s = mix64(renderSeed ^ 0xA0761D6478BD642Full);
        s = hashCombine64(s, pathId ^ 0xE7037ED1A0B428DBull);
        s = hashCombine64(s, uint64_t(bounceIndex) ^ 0x8EBC6AF09C88C6E3ull);
        s = hashCombine64(s, uint64_t(streamTag) ^ 0x589965CC75374CC3ull);
        s = hashCombine64(s, uint64_t(dimension) ^ 0x1D8E4E27C47D124Full);
        return s;
    }

    SYCL_EXTERNAL inline float rand01(
        uint64_t renderSeed,
        uint64_t pathId,
        uint32_t bounceIndex,
        uint32_t streamTag,
        uint32_t dimension) {
        const uint64_t s = rng::makeSeed(renderSeed, pathId, bounceIndex, streamTag, dimension);
        const uint32_t u = static_cast<uint32_t>(s >> 40); // top 24 bits
        return float(u) * 0x1.0p-24f;
    }
} // namespace pale::rng

namespace Pale {
    struct DebugPixel {
        uint32_t pixelX;
        uint32_t pixelY;
    };

    static DebugPixel kDebugPixels[] = {
        {300, 225},
    };

    static bool isWatchedPixel(uint32_t pixelX, uint32_t pixelY) {
        bool isMatch = false;

        for (uint32_t i = 0; i < 1; ++i) {
            const DebugPixel& debugPixel = kDebugPixels[i];
            if (pixelY == debugPixel.pixelY && pixelX == debugPixel.pixelX) {
                isMatch = true;
            }
        }
        return isMatch;
    }

    SYCL_EXTERNAL inline float3 safeInvDir(const float3& dir) {
        constexpr float EPS = 1e-6f; // treat anything smaller as “zero”
        constexpr float HUGE = 1e30f; // 2^100 ≃ 1.27e30 still fits in float
        float3 inv;
        inv.x() = (abs(dir.x()) < EPS) ? HUGE : 1.f / dir.x();
        inv.y() = (abs(dir.y()) < EPS) ? HUGE : 1.f / dir.y();
        inv.z() = (abs(dir.z()) < EPS) ? HUGE : 1.f / dir.z();
        return inv;
    }

    SYCL_EXTERNAL inline bool slabIntersectAABB(const Ray& ray,
                                                const TLASNode& node,
                                                const float3& invDir,
                                                float tMaxLimit,
                                                float& tEntry) {
        float3 t0 = (node.aabbMin - ray.origin) * invDir;
        float3 t1 = (node.aabbMax - ray.origin) * invDir;

        // Component-wise interval test to avoid relying on vector min/max
        float txmin = sycl::min(t0.x(), t1.x());
        float txmax = sycl::max(t0.x(), t1.x());
        float tymin = sycl::min(t0.y(), t1.y());
        float tymax = sycl::max(t0.y(), t1.y());
        float tzmin = sycl::min(t0.z(), t1.z());
        float tzmax = sycl::max(t0.z(), t1.z());

        float tmin = sycl::max(sycl::max(txmin, tymin), tzmin);
        float tmax = sycl::min(sycl::min(txmax, tymax), tzmax);

        /* 1.  Origin outside slabs AND entry after exit  ➜  miss          */
        if (tmin > tmax) return false;

        /* 2.  Whole box lies behind the ray                                  */
        if (tmax < 0.0f) return false;

        /* 3.  Already found a closer hit in the SAME SPACE                   */
        if (tmin > tMaxLimit) return false;

        tEntry = max(tmin, 0.0f); // clamp if origin is inside
        return true;
    }


    SYCL_EXTERNAL inline bool slabIntersectAABB(const Ray& ray,
                                                const BVHNode& node,
                                                const float3& invDir,
                                                float tMaxLimit,
                                                float& tEntry) {
        float3 t0 = (node.aabbMin - ray.origin) * invDir;
        float3 t1 = (node.aabbMax - ray.origin) * invDir;

        float txmin = sycl::min(t0.x(), t1.x());
        float txmax = sycl::max(t0.x(), t1.x());
        float tymin = sycl::min(t0.y(), t1.y());
        float tymax = sycl::max(t0.y(), t1.y());
        float tzmin = sycl::min(t0.z(), t1.z());
        float tzmax = sycl::max(t0.z(), t1.z());

        float tmin = sycl::max(sycl::max(txmin, tymin), tzmin);
        float tmax = sycl::min(sycl::min(txmax, tymax), tzmax);

        /* 1.  Origin outside slabs AND entry after exit  ➜  miss          */
        if (tmin > tmax) {
            return false;
        }
        /* 2.  Whole box lies behind the ray                                  */
        if (tmax < 0.0f) return false;

        /* 3.  Already found a closer hit in the SAME SPACE                   */

        if (tmin > tMaxLimit) return false;
        constexpr float kEps = 1e-6f;
        tEntry = max(tmin, kEps); // clamp if origin is inside
        return true;
    }

    //──────────────── world → object and back ────────────────────────────────
    SYCL_EXTERNAL inline Ray toObjectSpace(const Ray& rayW, const Transform& xf) {
        Ray r;
        /* 1.  Transform origin – w = 1                                      */
        float4 hO = xf.worldToObject * float4{rayW.origin, 1.f};
        r.origin = float3{hO.x(), hO.y(), hO.z()} / hO.w(); // <- perspective divide

        /* 2.  Transform direction – w = 0  (no translation component)       */
        float4 hD = xf.worldToObject * float4{rayW.direction, 0.f};
        r.direction = normalize(float3{hD.x(), hD.y(), hD.z()}); // w is already 0
        return r;
    }


    SYCL_EXTERNAL inline float3 transformPoint(const float4x4& tf, const float3& p, float w = 1.0f) {
        const float4 v = {p, w};
        float4 result = tf * v;
        float invW = 1.f / result.w();
        return float3{result.x() * invW, result.y() * invW, result.z() * invW};
    }

    SYCL_EXTERNAL inline float3 transformDirection(const float4x4& tf, const float3& dir) {
        const float4 v = {dir, 0.f};
        float4 r = tf * v;
        return normalize(float3{r.x(), r.y(), r.z()});
    }

    SYCL_EXTERNAL inline float3 toWorldPoint(const float3& pO, const Transform& xf) {
        float4 hp = xf.objectToWorld * float4{pO, 1.f};
        return float3{hp.x(), hp.y(), hp.z()} / hp.w();
    }

    SYCL_EXTERNAL inline bool intersectTriangle(const Ray& ray, const float3 v0, const float3 v1, const float3 v2,
                                                float& outT, float& outU,
                                                float& outV, float tMin) {
        const float3 e1 = v1 - v0;
        const float3 e2 = v2 - v0;

        const float3 h = cross(ray.direction, e2);
        const float a = dot(e1, h);

        // 1. Parallel?
        if (abs(a) < 1.0e-4f) return false;

        const float f = 1.0f / a;
        const float3 s = ray.origin - v0;
        const float u = f * dot(s, h);
        if (u < 0.0f || u > 1.0f) return false;

        const float3 q = cross(s, e1);
        const float v = f * dot(ray.direction, q);
        if (v < 0.0f || u + v > 1.0f) return false;

        const float t = f * dot(e2, q);
        if (t <= tMin) return false; // behind the ray or farther than a previous hit

        outT = t;
        outU = u;
        outV = v;

        return true;
    }

    inline float3 buildTangentFrisvad(const float3& unitNormal) {
        // Frisvad 2012: "Building an Orthonormal Basis, Revisited"
        // Handles all normals without branching issues.
        const float sign = std::copysign(1.0f, unitNormal.z());
        const float a = -1.0f / (sign + unitNormal.z());
        const float b = unitNormal.x() * unitNormal.y() * a;

        float3 tangent{
            1.0f + sign * unitNormal.x() * unitNormal.x() * a,
            sign * b,
            -sign * unitNormal.x()
        };
        return normalize(tangent);
    }


    inline void buildOrthonormalBasis(const float3& unitNormal, float3& tangent, float3& bitangent) {
        tangent = buildTangentFrisvad(unitNormal);
        bitangent = cross(unitNormal, tangent);
    }

    SYCL_EXTERNAL inline uint32_t sampleTriangleByCdf(
        const GPUEmissiveTriangle* emissive_triangles,
        uint32_t offset,
        uint32_t count,
        float u) {
        // binary search first cdf >= u
        uint32_t lo = 0, hi = count - 1;
        while (lo < hi) {
            uint32_t mid = (lo + hi) >> 1;
            float c = emissive_triangles[offset + mid].cdf;
            if (u <= c) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    }


    inline float3 phiMapping(const Point& surfel, float u, float v) {
        return surfel.position + surfel.scale.x() * surfel.tanU * u + surfel.scale.y() * surfel.tanV * v;
    }

    inline float3 phiMapping(const Point& surfel, float2 uv) {
        return surfel.position + surfel.scale.x() * surfel.tanU * uv[0] + surfel.scale.y() * surfel.tanV * uv[1];
    }

    SYCL_EXTERNAL inline void sampleCosineHemisphere(
        rng::Xorshift128& rng, const float3& n,
        float3& outDir, float& outPdf) {
        float u1 = rng.nextFloat();
        float u2 = rng.nextFloat();

        float z = sycl::sqrt(1.f - u1);
        float r = sycl::sqrt(1 - (z * z));

        float phi = 2.f * M_PIf * u2;
        float x = r * sycl::cos(phi);
        float y = r * sycl::sin(phi);

        // build an ONB around n
        float3 up = abs(n.z()) < .999f ? float3{0, 0, 1} : float3{1, 0, 0};
        float3 tang = normalize(cross(up, n));
        float3 bit = cross(n, tang);

        outDir = normalize(x * tang + y * bit + z * n);
        outPdf = max(0.f, dot(outDir, n)) / M_PIf; // cosθ/π
    }


    inline void buildIntersectionNormal(const GPUSceneBuffers& scene, WorldHit& worldHit) {
        if (!worldHit.hit)
            return;
        InstanceRecord& instance = scene.instances[worldHit.instanceIndex];
        if (instance.geometryType == GeometryType::Mesh) {
            const Triangle& triangle = scene.triangles[worldHit.primitiveIndex];
            const Transform& objectWorldTransform = scene.transforms[instance.transformIndex];
            const Vertex& vertex0 = scene.vertices[triangle.v0];
            const Vertex& vertex1 = scene.vertices[triangle.v1];
            const Vertex& vertex2 = scene.vertices[triangle.v2];
            // Canonical geometric normal (no face-forwarding)
            const float3 worldP0 = toWorldPoint(vertex0.pos, objectWorldTransform);
            const float3 worldP1 = toWorldPoint(vertex1.pos, objectWorldTransform);
            const float3 worldP2 = toWorldPoint(vertex2.pos, objectWorldTransform);
            const float3 canonicalNormalW = normalize(cross(worldP1 - worldP0, worldP2 - worldP0));
            worldHit.geometricNormalW = canonicalNormalW;
        }
        else if (instance.geometryType == GeometryType::PointCloud) {
            const auto& surfel = scene.points[worldHit.primitiveIndex];
            // Canonical surfel normal from tangents (no face-forwarding)
            const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));
            worldHit.geometricNormalW = canonicalNormalW;
        }
    }


    inline float2 phiInverse(const float3& hitWorld, const Point& surfel) {
        float3 r = hitWorld - surfel.position;
        float2 uv;
        uv[0] = dot(surfel.tanU, r) / surfel.scale.x();
        uv[1] = dot(surfel.tanV, r) / surfel.scale.y();
        return uv;
    }

    SYCL_EXTERNAL inline void buildOrthonormalBasisFromNormal(
        const float3& unitNormal,
        float3& outTangent,
        float3& outBitangent
    ) {
        // Robust ONB construction (Frisvad-style branchless variant)
        // Assumes unitNormal is normalized.
        const float sign = sycl::copysign(1.0f, unitNormal.z());
        const float a = -1.0f / (sign + unitNormal.z());
        const float b = unitNormal.x() * unitNormal.y() * a;

        outTangent = float3{
            1.0f + sign * unitNormal.x() * unitNormal.x() * a,
            sign * b,
            -sign * unitNormal.x()
        };

        outBitangent = float3{
            b,
            sign + unitNormal.y() * unitNormal.y() * a,
            -unitNormal.y()
        };
    }

    SYCL_EXTERNAL inline void sampleUniformHemisphereAroundNormal(
        rng::Xorshift128& randomNumberGenerator,
        const float3& normal,
        float3& outDirectionWorld,
        float& outPdf
    ) {
        // Ensure a valid unit normal
        const float3 unitNormal = normalize(normal);

        // Sample in local frame: +Z hemisphere
        float3 localDirection;
        {
            const float uniformRandomOne = randomNumberGenerator.nextFloat();
            const float uniformRandomTwo = randomNumberGenerator.nextFloat();

            const float zCoordinate = uniformRandomOne; // [0,1]
            const float azimuthAngle = 2.0f * M_PIf * uniformRandomTwo;

            const float radialComponent =
                sycl::sqrt(sycl::fmax(0.0f, 1.0f - zCoordinate * zCoordinate));

            const float xCoordinate = radialComponent * sycl::cos(azimuthAngle);
            const float yCoordinate = radialComponent * sycl::sin(azimuthAngle);

            localDirection = float3{xCoordinate, yCoordinate, zCoordinate};
        }

        // Build basis (tangent, bitangent, normal)
        float3 tangent;
        float3 bitangent;
        buildOrthonormalBasisFromNormal(unitNormal, tangent, bitangent);

        // Transform to world
        outDirectionWorld =
            tangent * localDirection.x() +
            bitangent * localDirection.y() +
            unitNormal * localDirection.z();

        // Numerical safety
        outDirectionWorld = normalize(outDirectionWorld);

        // Uniform hemisphere PDF (in solid angle)
        outPdf = 1.0f / (2.0f * M_PIf);

        // Optional: enforce hemisphere (should already be true)
        if (dot(outDirectionWorld, unitNormal) < 0.0f) outDirectionWorld = -outDirectionWorld;
    }

    SYCL_EXTERNAL static bool opacityGaussian(float u, float v, float* outOpacity, float kSigmas = 2.2f) {
        const float r2 = u * u + v * v;
        // Optional accel window. Prefer k=3..4. If you keep this, you lose tail mass.
        if (r2 > kSigmas * kSigmas)
            return false;

        *outOpacity = sycl::exp(-0.5f * r2);
        return true;
    }

    SYCL_EXTERNAL static bool opacityBeta(float u, float v, const Point& surfel, float* outOpacity) {
        const float r2 = u * u + v * v;
        // Optional accel window. Prefer k=3..4. If you keep this, you lose tail mass.
        if (r2 >= 1.0)
            return false;

        float base = 1 - r2;
        float exp = 4 * std::exp(surfel.beta);

        *outOpacity = std::pow(base, exp);
        return true;
    }


    SYCL_EXTERNAL static bool intersectSurfel(const Ray& rayObject,
                                              const Point& surfel,
                                              float tMin, float tMax,
                                              float& outTHit,
                                              float3& outHitLocal,
                                              float& outOpacity,
                                              const float& eps = 1e-6f) {
        // Should match the same kSigmas as in BVH construction
        // 1) Orthonormal in-plane frame (assumes your rotation already baked into tanU/tanV)
        const float3 unitTangentU = normalize(surfel.tanU);
        const float3 unitTangentV = normalize(surfel.tanV - unitTangentU * dot(unitTangentU, surfel.tanV));
        const float3 unitNormal = normalize(cross(unitTangentU, unitTangentV));

        // 2) Ray-plane hit
        const float nDotD = dot(unitNormal, rayObject.direction);
        if (sycl::fabs(nDotD) < eps)
            return false;

        const float tHit = dot(unitNormal, (surfel.position - rayObject.origin)) / nDotD;
        if (tHit <= tMin || tHit >= tMax)
            return false;

        outHitLocal = rayObject.origin + tHit * rayObject.direction;
        float2 uv = phiInverse(outHitLocal, surfel);

        //if (!opacityGaussian(uv[0], uv[1], &outOpacity))
        //    return false;

        if (!opacityBeta(uv[0], uv[1], surfel, &outOpacity))
            return false;
        outTHit = tHit;
        return true;
    }


    SYCL_EXTERNAL inline Ray makePrimaryRayFromPixelJitteredFov(
        const CameraGPU& cam,
        float px, float py,
        float jx, float jy) {
        const float width = static_cast<float>(cam.width);
        const float height = static_cast<float>(cam.height);

        const float u = (px + jx);
        const float v = (py + jy);

        // If your image origin is top-left, flip v:
        // const float v_flipped = height - v;
        const float v_flipped = height - v;

        const float ndcX = (2.0f * u / width - 1.0f);
        const float ndcY = (2.0f * v_flipped / height - 1.0f);

        const float f_y = 0.5f * height / sycl::tan(0.5f * glm::radians(cam.fovy));
        const float f_x = f_y * (width / height);

        // Camera looks down -Z (OpenGL-style view space)
        float3 dirCamera = normalize(float3{
            ndcX * (0.5f * width) / f_x,
            ndcY * (0.5f * height) / f_y,
            -1.0f
        });

        // Transform direction to world (use a direction transform, w=0)
        const float3 dirWorld = transformDirection(cam.invView, dirCamera);
        const float3 originWorld = transformPoint(cam.invView, float3{0, 0, 0});

        return Ray{originWorld, dirWorld, cam.forward};
    }

    SYCL_EXTERNAL inline bool projectToPixelFromPinhole(
        const SensorGPU& sensor,
        const float3& pointWorld,
        uint32_t& outPixelIndex,
        float3& outOmegaFromSurfaceToCamera,
        float& outDistance,
        int& pixelX,
        int& pixelY,
        bool& debug) {
        const float3 cameraPositionWorld = sensor.camera.pos;

        const float3 vectorFromSurfaceToCamera = cameraPositionWorld - pointWorld;
        const float distance = length(vectorFromSurfaceToCamera);
        if (distance <= 0.0f) return false;

        outOmegaFromSurfaceToCamera = vectorFromSurfaceToCamera / distance;
        outDistance = distance;

        // World -> camera
        const float3 pointCamera = transformPoint(sensor.camera.view, pointWorld);

        // In front (OpenGL-style: camera looks along -Z)
        if (pointCamera.z() >= 0.0f) return false;

        const float width = static_cast<float>(sensor.width);
        const float height = static_cast<float>(sensor.height);

        const float fyFallback = 0.5f * height / sycl::tan(0.5f * glm::radians(sensor.camera.fovy));
        const float fxFallback = fyFallback * (width / height);
        float fx = fxFallback;
        float fy = fyFallback;
        float cx = 0.5f * width;
        float cy = 0.5f * height;


        const float z = -pointCamera.z(); // positive depth
        const float u = fx * (pointCamera.x() / z) + cx;
        const float v = height - (fy * (pointCamera.y() / z) + cy);

        pixelX = static_cast<int>(sycl::floor(u));
        pixelY = static_cast<int>(sycl::floor(v));

        if (pixelX < 0 || pixelX >= static_cast<int>(sensor.width) ||
            pixelY < 0 || pixelY >= static_cast<int>(sensor.height))
            return false;

        if ((pixelX == 925 && pixelY == 500) || (pixelX == 895 && pixelY == 500))
            debug = true;

        outPixelIndex = static_cast<uint32_t>(pixelY) * sensor.width + static_cast<uint32_t>(pixelX);
        return true;
    }


    inline sycl::int3 worldToCell(const float3& positionWorld, const DeviceSurfacePhotonMapGrid& grid) {
        const float3 local = (positionWorld - grid.gridOriginWorld);
        const float3 cellFloat = float3{
            local.x() / grid.cellSizeWorld.x(),
            local.y() / grid.cellSizeWorld.y(),
            local.z() / grid.cellSizeWorld.z()
        };

        sycl::int3 cell = sycl::int3{
            static_cast<int>(sycl::floor(cellFloat.x())),
            static_cast<int>(sycl::floor(cellFloat.y())),
            static_cast<int>(sycl::floor(cellFloat.z()))
        };

        // Clamp
        cell.x() = sycl::clamp(cell.x(), 0, grid.gridResolution.x() - 1);
        cell.y() = sycl::clamp(cell.y(), 0, grid.gridResolution.y() - 1);
        cell.z() = sycl::clamp(cell.z(), 0, grid.gridResolution.z() - 1);
        return cell;
    }

    inline std::uint32_t linearCellIndex(const sycl::int3& cell, const sycl::int3& gridResolution) {
        // x fastest
        return static_cast<std::uint32_t>(
            (cell.z() * gridResolution.y() + cell.y()) * gridResolution.x() + cell.x()
        );
    }


    inline int signNonZero(float x) { return (x >= 0.0f) ? 1 : -1; }

    inline sycl::int3 worldToCellClamped(const float3& positionWorld, const DeviceSurfacePhotonMapGrid& grid) {
        const float3 local = positionWorld - grid.gridOriginWorld;
        const float3 cellFloat = local / grid.cellSizeWorld;

        sycl::int3 cell{
            static_cast<int>(sycl::floor(cellFloat.x())),
            static_cast<int>(sycl::floor(cellFloat.y())),
            static_cast<int>(sycl::floor(cellFloat.z()))
        };

        cell.x() = sycl::clamp(cell.x(), 0, grid.gridResolution.x() - 1);
        cell.y() = sycl::clamp(cell.y(), 0, grid.gridResolution.y() - 1);
        cell.z() = sycl::clamp(cell.z(), 0, grid.gridResolution.z() - 1);
        return cell;
    }


    inline void atomicAddFloat3ToImage(float4* dst, const float3& v) {
        for (int c = 0; c < 3; ++c) {
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                a(reinterpret_cast<float*>(dst)[c]);
            a.fetch_add(v[c]);
        }
        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            a(reinterpret_cast<float*>(dst)[3]);
        a.store(1.0f);
    }


    SYCL_EXTERNAL inline bool applyRussianRoulette(
        rng::Xorshift128& rng128,
        uint32_t bounceIndex,
        float3& pathThroughput,
        uint32_t rrStartBounce,
        float rrMinProbability = 0.00f,
        float rrMaxProbability = 0.99f,
        float maxCompensationFactor = 10.0f) // set <= 0 to disable capping
    {
        if (bounceIndex < rrStartBounce) {
            return true;
        }

        // Reject invalid throughput early (prevents NaN fireflies).
        const float tx = pathThroughput.x();
        const float ty = pathThroughput.y();
        const float tz = pathThroughput.z();

        const bool throughputIsFinite =
            sycl::isfinite(tx) && sycl::isfinite(ty) && sycl::isfinite(tz);

        if (!throughputIsFinite) {
            return false;
        }

        // Luminance-based continuation probability (more stable than max).
        // Clamp also prevents huge 1/p factors.
        const float luminance =
            0.2126f * tx + 0.7152f * ty + 0.0722f * tz;

        float continuationProbability = sycl::clamp(luminance, rrMinProbability, rrMaxProbability);

        if (rng128.nextFloat() > continuationProbability) {
            return false;
        }

        float compensationFactor = 1.0f / continuationProbability;

        // Optional biased clamp to prevent rare massive weights.
        if (maxCompensationFactor > 0.0f) {
            compensationFactor = sycl::fmin(compensationFactor, maxCompensationFactor);
        }

        pathThroughput *= compensationFactor;
        return true;
    }

    SYCL_EXTERNAL inline void appendContributionAtomic(
        uint32_t* globalContributionCounter,
        HitInfoContribution* globalContributionBuffer,
        uint32_t maxContributionCapacity,
        const HitInfoContribution& contributionValue) {
        sycl::atomic_ref<
            uint32_t,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space
        > contributionsCounter(*globalContributionCounter);

        const uint32_t insertionIndex = contributionsCounter.fetch_add(1);

        if (insertionIndex >= maxContributionCapacity) {
            // Counter may now exceed capacity; caller can reset each frame/pass if desired.
            return;
        }

        globalContributionBuffer[insertionIndex] = contributionValue;
    }

    SYCL_EXTERNAL inline PointCloudSurfaceRecord makePointCloudSurfaceRecord(
        const WorldHit& worldHit,
        const RayState& rayState,
        const GPUSceneBuffers& scene) {
        PointCloudSurfaceRecord surfaceRecord{};
        surfaceRecord.primitiveIndex = worldHit.primitiveIndex;
        surfaceRecord.alphaGeom = worldHit.alphaGeom;
        surfaceRecord.pathId = rayState.pathId;
        surfaceRecord.incomingDirection = rayState.ray.direction;

        const Point& surfel = scene.points[worldHit.primitiveIndex];
        surfaceRecord.uv = phiInverse(worldHit.hitPositionW, surfel);

        const float3 tangentUWorld = surfel.scale.x() * surfel.tanU;
        const float3 tangentVWorld = surfel.scale.y() * surfel.tanV;
        const float3 canonicalNormal = normalize(cross(tangentUWorld, tangentVWorld));

        const float signedCosineIncident = dot(canonicalNormal, -rayState.ray.direction);
        surfaceRecord.sideSign = signNonZero(signedCosineIncident);

        return surfaceRecord;
    }

    SYCL_EXTERNAL inline ReconstructedSurfelState reconstructSurfelState(
        const Point& surfel,
        const PointCloudSurfaceRecord& surfaceRecord) {
        ReconstructedSurfelState reconstructedState{};
        reconstructedState.position = phiMapping(surfel, surfaceRecord.uv.x(), surfaceRecord.uv.y());
        reconstructedState.tangentUWorld = surfel.scale.x() * surfel.tanU;
        reconstructedState.tangentVWorld = surfel.scale.y() * surfel.tanV;

        const float3 scaledCross = cross(reconstructedState.tangentUWorld, reconstructedState.tangentVWorld);
        reconstructedState.canonicalNormal = normalize(scaledCross);
        reconstructedState.orientedNormal = static_cast<float>(surfaceRecord.sideSign) * reconstructedState.
            canonicalNormal;
        reconstructedState.areaWorld = M_PIf * length(scaledCross);
        return reconstructedState;
    }

    template <typename EventType>
    SYCL_EXTERNAL inline void appendEventAtomic(
        uint32_t* countBuffer,
        EventType* eventBuffer,
        uint32_t maxEventCount,
        const EventType& eventRecord) {
        auto eventCounter = sycl::atomic_ref<uint32_t,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>(*countBuffer);

        const uint32_t eventIndex = eventCounter.fetch_add(1);
        if (eventIndex < maxEventCount) {
            eventBuffer[eventIndex] = eventRecord;
        }
    }

    inline float3 gatherDiffuseIrradianceAtPointKNN(
        const float3& queryPositionWorld,
        const float3& queryNormal,
        const DeviceSurfacePhotonMapGrid& grid) {
        constexpr uint32_t kNearestPhotonCount = 64;
        constexpr uint32_t maxRadiusExpansionSteps = 8u;
        constexpr float minimumPlaneThickness = 1e-5f;

        const float3 normalizedQueryNormal = normalize(queryNormal);

        const float minimumRadius = grid.minimumGatherRadiusWorld;
        const float minimumRadiusSquared = minimumRadius * minimumRadius;

        float currentSearchRadius = minimumRadius;

        float nearestTangentialDistanceSquared[kNearestPhotonCount];
        float3 nearestPhotonPower[kNearestPhotonCount];

        uint32_t nearestPhotonCount = 0u;
        float worstKeptDistanceSquared = 0.0f;
        uint32_t worstKeptIndex = 0u;

        auto reset_knn = [&]() {
            nearestPhotonCount = 0u;
            worstKeptDistanceSquared = 0.0f;
            worstKeptIndex = 0u;

            for (uint32_t photonIndex = 0u; photonIndex < kNearestPhotonCount; ++photonIndex) {
                nearestTangentialDistanceSquared[photonIndex] = 0.0f;
                nearestPhotonPower[photonIndex] = float3{0.0f};
            }
        };

        auto recompute_worst_kept = [&]() {
            if (nearestPhotonCount == 0u) {
                worstKeptDistanceSquared = 0.0f;
                worstKeptIndex = 0u;
                return;
            }

            uint32_t newWorstIndex = 0u;
            float newWorstDistanceSquared = nearestTangentialDistanceSquared[0];

            for (uint32_t photonIndex = 1u; photonIndex < nearestPhotonCount; ++photonIndex) {
                if (nearestTangentialDistanceSquared[photonIndex] > newWorstDistanceSquared) {
                    newWorstDistanceSquared = nearestTangentialDistanceSquared[photonIndex];
                    newWorstIndex = photonIndex;
                }
            }

            worstKeptDistanceSquared = newWorstDistanceSquared;
            worstKeptIndex = newWorstIndex;
        };

        auto try_insert_candidate = [&](float tangentialDistanceSquared, const float3& photonPower) {
            if (nearestPhotonCount < kNearestPhotonCount) {
                nearestTangentialDistanceSquared[nearestPhotonCount] = tangentialDistanceSquared;
                nearestPhotonPower[nearestPhotonCount] = photonPower;
                ++nearestPhotonCount;
                recompute_worst_kept();
                return;
            }

            if (tangentialDistanceSquared >= worstKeptDistanceSquared) {
                return;
            }

            nearestTangentialDistanceSquared[worstKeptIndex] = tangentialDistanceSquared;
            nearestPhotonPower[worstKeptIndex] = photonPower;
            recompute_worst_kept();
        };

        bool foundEnoughPhotons = false;

        for (uint32_t expansionStep = 0u;
             expansionStep < maxRadiusExpansionSteps;
             ++expansionStep) {
            reset_knn();

            const float currentSearchRadiusSquared =
                currentSearchRadius * currentSearchRadius;

            const float slabHalfThickness =
                sycl::fmax(minimumPlaneThickness, 0.25f * currentSearchRadius);

            const float supportExtent = currentSearchRadius + slabHalfThickness;
            const float3 supportOffset = float3{
                supportExtent, supportExtent, supportExtent
            };

            const sycl::int3 minCell =
                worldToCellClamped(queryPositionWorld - supportOffset, grid);
            const sycl::int3 maxCell =
                worldToCellClamped(queryPositionWorld + supportOffset, grid);

            for (int cellZ = minCell.z(); cellZ <= maxCell.z(); ++cellZ) {
                for (int cellY = minCell.y(); cellY <= maxCell.y(); ++cellY) {
                    for (int cellX = minCell.x(); cellX <= maxCell.x(); ++cellX) {
                        const uint32_t cellId =
                            linearCellIndex(sycl::int3{cellX, cellY, cellZ}, grid.gridResolution);

                        const uint32_t start = grid.cellStart[cellId];
                        if (start == kInvalidIndex) {
                            continue;
                        }

                        const uint32_t end = grid.cellEnd[cellId];
                        for (uint32_t sortedIndex = start; sortedIndex < end; ++sortedIndex) {
                            const uint32_t photonIndex = grid.sortedPhotonIndex[sortedIndex];
                            const DevicePhotonSurface photon = grid.photons[photonIndex];

                            const float sameHemisphere =
                                dot(normalizedQueryNormal, -photon.incomingDirection) > 0.0f ? 1.0f : 0.0f;
                            if (sameHemisphere == 0.0f) {
                                continue;
                            }

                            const float3 offsetWorld = photon.position - queryPositionWorld;
                            const float signedPlaneDistance =
                                dot(offsetWorld, normalizedQueryNormal);

                            if (sycl::fabs(signedPlaneDistance) > slabHalfThickness) {
                                continue;
                            }

                            const float3 tangentOffset =
                                offsetWorld - signedPlaneDistance * normalizedQueryNormal;
                            const float tangentialDistanceSquared =
                                dot(tangentOffset, tangentOffset);

                            if (tangentialDistanceSquared > currentSearchRadiusSquared) {
                                continue;
                            }

                            try_insert_candidate(tangentialDistanceSquared, photon.power);
                        }
                    }
                }
            }

            if (nearestPhotonCount >= kNearestPhotonCount) {
                foundEnoughPhotons = true;
                break;
            }

            currentSearchRadius *= 2.0f;
        }

        if (nearestPhotonCount == 0u) {
            return float3{0.0f};
        }

        float adaptiveRadiusSquared = minimumRadiusSquared;

        if (foundEnoughPhotons) {
            adaptiveRadiusSquared =
                sycl::fmax(minimumRadiusSquared, worstKeptDistanceSquared);
        }
        else {
            // Fallback: not enough photons found, so use the final search radius
            // rather than the farthest kept photon distance to avoid over-brightening.
            adaptiveRadiusSquared =
                sycl::fmax(minimumRadiusSquared, currentSearchRadius * currentSearchRadius);
        }

        float3 accumulatedFlux = float3{0.0f};

        for (uint32_t photonIndex = 0u; photonIndex < nearestPhotonCount; ++photonIndex) {
            accumulatedFlux += nearestPhotonPower[photonIndex];
        }

        const float gatherArea = M_PIf * adaptiveRadiusSquared;
        const float inverseGatherArea = 1.0f / sycl::fmax(gatherArea, 1e-12f);

        // Returns irradiance [W / m^2]
        return accumulatedFlux * inverseGatherArea;
    }

    inline float3 gatherDiffuseIrradianceAtPoint(
        const float3& queryPositionWorld,
        const float3& queryNormal,
        const DeviceSurfacePhotonMapGrid& grid) {
        const float3 normalizedQueryNormal = normalize(queryNormal);

        const float radius = grid.minimumGatherRadiusWorld;
        const float radiusSquared = radius * radius;

        const float slabHalfThickness = sycl::fmax(1e-5f, 0.25f * radius);

        // 2D Epanechnikov-style normalization over the tangent disk.
        const float kernelNormalization = 2.0f / (M_PIf * radiusSquared);

        const float supportExtent = radius + slabHalfThickness;
        const float3 supportOffset = float3{supportExtent, supportExtent, supportExtent};

        const sycl::int3 minCell =
            worldToCellClamped(queryPositionWorld - supportOffset, grid);
        const sycl::int3 maxCell =
            worldToCellClamped(queryPositionWorld + supportOffset, grid);

        float3 irradiance = float3{0.0f};

        for (int cellZ = minCell.z(); cellZ <= maxCell.z(); ++cellZ) {
            for (int cellY = minCell.y(); cellY <= maxCell.y(); ++cellY) {
                for (int cellX = minCell.x(); cellX <= maxCell.x(); ++cellX) {
                    const uint32_t cellId =
                        linearCellIndex(sycl::int3{cellX, cellY, cellZ}, grid.gridResolution);

                    const uint32_t start = grid.cellStart[cellId];
                    if (start == kInvalidIndex) {
                        continue;
                    }

                    const uint32_t end = grid.cellEnd[cellId];
                    for (uint32_t sortedIndex = start; sortedIndex < end; ++sortedIndex) {
                        const uint32_t photonIndex = grid.sortedPhotonIndex[sortedIndex];
                        const DevicePhotonSurface photon = grid.photons[photonIndex];

                        const float sameHemisphere =
                            dot(normalizedQueryNormal, -photon.incomingDirection) > 0.0f ? 1.0f : 0.0f;
                        if (sameHemisphere == 0.0f) {
                            continue;
                        }

                        const float3 offsetWorld = photon.position - queryPositionWorld;

                        // Signed deviation from the tangent plane along the query normal.
                        const float planeDistance =
                            dot(offsetWorld, normalizedQueryNormal);
                        const float absolutePlaneDistance = sycl::fabs(planeDistance);

                        if (absolutePlaneDistance > slabHalfThickness) {
                            continue;
                        }

                        // Tangential offset inside the local tangent plane.
                        const float3 tangentialOffset =
                            offsetWorld - planeDistance * normalizedQueryNormal;
                        const float tangentialDistanceSquared =
                            dot(tangentialOffset, tangentialOffset);

                        if (tangentialDistanceSquared > radiusSquared) {
                            continue;
                        }

                        // Radial falloff in the tangent plane.
                        const float radialWeight =
                            1.0f - tangentialDistanceSquared / radiusSquared;

                        // Penalize photons that deviate from the tangent plane.
                        // Quadratic falloff suppresses near-slab-edge photons more strongly.
                        const float normalizedPlaneDistance =
                            absolutePlaneDistance / slabHalfThickness;
                        const float tangentDeviationWeight =
                            1.0f - normalizedPlaneDistance * normalizedPlaneDistance;

                        const float kernelWeight =
                            kernelNormalization *
                            radialWeight *
                            tangentDeviationWeight *
                            sameHemisphere;

                        irradiance += photon.power * kernelWeight;
                    }
                }
            }
        }
        return irradiance;
    }

    inline float3 gatherDiffuseIrradianceAtPointStandard(
        const float3& queryPositionWorld,
        const float3& queryNormal,
        const DeviceSurfacePhotonMapGrid& grid) {
        const float radius = grid.minimumGatherRadiusWorld;
        const float radiusSquared = radius * radius;

        const float slabHalfThickness = sycl::fmax(1e-5f, 0.25f * radius);

        const float area = 2.0f / (M_PIf * radiusSquared);

        const float supportExtent = radius + slabHalfThickness;
        const float3 supportOffset = float3{supportExtent, supportExtent, supportExtent};

        const sycl::int3 minCell =
            worldToCellClamped(queryPositionWorld - supportOffset, grid);
        const sycl::int3 maxCell =
            worldToCellClamped(queryPositionWorld + supportOffset, grid);

        float3 irradiance = float3{0.0f};

        for (int cellZ = minCell.z(); cellZ <= maxCell.z(); ++cellZ) {
            for (int cellY = minCell.y(); cellY <= maxCell.y(); ++cellY) {
                for (int cellX = minCell.x(); cellX <= maxCell.x(); ++cellX) {
                    const uint32_t cellId =
                        linearCellIndex(sycl::int3{cellX, cellY, cellZ}, grid.gridResolution);

                    const uint32_t start = grid.cellStart[cellId];
                    if (start == kInvalidIndex) {
                        continue;
                    }

                    const uint32_t end = grid.cellEnd[cellId];
                    for (uint32_t sortedIndex = start; sortedIndex < end; ++sortedIndex) {
                        const uint32_t photonIndex = grid.sortedPhotonIndex[sortedIndex];
                        const DevicePhotonSurface photon = grid.photons[photonIndex];

                        const float sameHemisphere =
                            dot(queryNormal, -photon.incomingDirection) > 0.0f ? 1.0f : 0.0f;

                        // Square the plane weight to aggressively suppress near-slab-edge photons.
                        const float kernelWeight =
                            area * sameHemisphere;

                        irradiance += photon.power * kernelWeight;
                    }
                }
            }
        }
        float3 radiance = irradiance * M_1_PIf; // Irradiance E[W/M²] -> L[W/(M²sr¹)
        return radiance;
    }

    SYCL_EXTERNAL inline void depositPhotonSurface(
        const WorldHit& worldHit,
        const float3& incomingDirection,
        const float3& normal,
        const float3& flux,
        const DeviceSurfacePhotonMapGrid& photonMap) {
        // Atomic counter for photon slots
        auto photonCounter = sycl::atomic_ref<uint32_t,
                                              sycl::memory_order::relaxed,
                                              sycl::memory_scope::device,
                                              sycl::access::address_space::global_space>(
            *photonMap.photonCountDevicePtr);

        const uint32_t slot = photonCounter.fetch_add(1u);
        if (slot >= photonMap.photonCapacity) {
            return;
        }
        if (length(flux) <= 0.0)
            return;


        DevicePhotonSurface photonEntry{};
        photonEntry.position = worldHit.hitPositionW;

        // Geometric normal (unoriented by design)
        photonEntry.incomingDirection = incomingDirection;

        // Incoming direction (towards surface)
        //photonEntry.normal = normal;

        // Power carried by the photon
        photonEntry.power = flux;

        photonEntry.isValid = 1u;

        photonMap.photons[slot] = photonEntry;
    }


    SYCL_EXTERNAL inline void clearPendingAdjointVertex(PendingAdjointVertex& v) {
        v.surface = {};
        v.bounceIndex = UINT32_MAX;
        v.pathThroughput = float3{FLT_MAX, FLT_MAX, FLT_MAX};
        v.transmissionFromPrevious = FLT_MAX;
        v.geometryFromPrevious = FLT_MAX;
        v.areaPdfFromPrevious = FLT_MAX;
        v.bsdf = float3{FLT_MAX, FLT_MAX, FLT_MAX};
        v.cosineFromPrevious = FLT_MAX;
    }

    SYCL_EXTERNAL inline void clearPendingAdjointStageX(PendingAdjointStageX& s) {
        s.valid = false;
        s.pathId = 0u;
        s.pixelIndex = UINT32_MAX;
        s.useImplicitRayHitJacobian = false;
        s.hasPrevious = false;
        clearPendingAdjointVertex(s.previous);
        clearPendingAdjointVertex(s.current);
    }

    SYCL_EXTERNAL inline PendingAdjointVertex makePendingAdjointVertex(
        const PointCloudSurfaceRecord& surface,
        uint32_t bounceIndex,
        const float3& pathThroughput,
        float transmissionFromPrevious,
        float geometryFromPrevious,
        float areaPdfFromPrevious,
        const float3& bsdf,
        float cosineFromPrevious) {
        PendingAdjointVertex v{};
        v.surface = surface;
        v.bounceIndex = bounceIndex;
        v.pathThroughput = pathThroughput;
        v.transmissionFromPrevious = transmissionFromPrevious;
        v.geometryFromPrevious = geometryFromPrevious;
        v.areaPdfFromPrevious = areaPdfFromPrevious;
        v.bsdf = bsdf;
        v.cosineFromPrevious = cosineFromPrevious;
        return v;
    }

    SYCL_EXTERNAL inline void pushPendingAdjointVertex(
        PendingAdjointStageX& stage,
        uint32_t pathId,
        uint32_t pixelIndex,
        bool useImplicitRayHitJacobian,
        const PendingAdjointVertex& newCurrent) {
        if (stage.valid) {
            stage.hasPrevious = true;
            stage.previous = stage.current;
        }
        else {
            stage.hasPrevious = false;
            clearPendingAdjointVertex(stage.previous);
        }

        stage.valid = true;
        stage.pathId = pathId;
        stage.pixelIndex = pixelIndex;
        stage.useImplicitRayHitJacobian = useImplicitRayHitJacobian;
        stage.current = newCurrent;
    }

    inline float3 computePointCloudOrientedNormal(
        const Point& surfel,
        const float3& rayDirectionWorld) {
        const float3 canonicalNormal = normalize(cross(
            surfel.scale.x() * surfel.tanU,
            surfel.scale.y() * surfel.tanV));

        const float signedCosineIncident =
            dot(canonicalNormal, -rayDirectionWorld);
        const int sideSign = signNonZero(signedCosineIncident);

        return static_cast<float>(sideSign) * canonicalNormal;
    }

    inline float computeSegmentAreaPdfFromUniformHemisphere(
        const ReconstructedSurfelState& fromState,
        const ReconstructedSurfelState& toState,
        float hemispherePdf) {
        const float3 vectorFromTo = toState.position - fromState.position;
        const float distanceSquared = dot(vectorFromTo, vectorFromTo);
        const float distance = sycl::sqrt(distanceSquared);
        const float3 directionFromTo = vectorFromTo / distance;
        const float cosineAtTo = dot(toState.orientedNormal, -directionFromTo);

        return hemispherePdf * cosineAtTo / distanceSquared;
    }

    SYCL_EXTERNAL inline AreaLightSample sampleMeshAreaLight(
        const GPUSceneBuffers& scene,
        rng::Xorshift128& rng128) {
        AreaLightSample sample{};
        sample.valid = false;

        if (scene.lightCount == 0)
            return sample;

        // 1) Pick a light (keep uniform for now; later you can switch to flux-weighted)
        const float u_light = rng128.nextFloat();
        const uint32_t light_index =
            sycl::min(static_cast<uint32_t>(u_light * scene.lightCount), scene.lightCount - 1u);

        const GPULightRecord light = scene.lights[light_index];
        sample.pdfSelectLight = 1.0f / static_cast<float>(scene.lightCount);

        if (light.lightType == LightType::Mesh) {
            if (light.triangleCount == 0u || light.totalAreaWorld <= 0.0f)
                return sample;

            // 2) Pick a triangle proportional to WORLD area using the precomputed CDF
            const float u_tri = rng128.nextFloat();

            uint32_t tri_rel = 0u;
            {
                // Binary search first cdf >= u_tri (CDF is inclusive and last entry is exactly 1)
                uint32_t lo = 0u;
                uint32_t hi = light.triangleCount - 1u;

                while (lo < hi) {
                    const uint32_t mid = (lo + hi) >> 1u;
                    const float cdf_mid = scene.emissiveTriangles[light.triangleOffset + mid].cdf;
                    if (u_tri <= cdf_mid) {
                        hi = mid;
                    }
                    else {
                        lo = mid + 1u;
                    }
                }
                tri_rel = lo;
            }

            const GPUEmissiveTriangle emissive_triangle =
                scene.emissiveTriangles[light.triangleOffset + tri_rel];

            const Triangle tri = scene.triangles[emissive_triangle.globalTriangleIndex];
            const Vertex v0 = scene.vertices[tri.v0];
            const Vertex v1 = scene.vertices[tri.v1];
            const Vertex v2 = scene.vertices[tri.v2];

            // 3) Uniform barycentric sample on the triangle in OBJECT space
            const float u1 = rng128.nextFloat();
            const float u2 = rng128.nextFloat();
            const float sqrt_u1 = sycl::sqrt(u1);

            const float b0 = 1.0f - sqrt_u1;
            const float b1 = sqrt_u1 * (1.0f - u2);
            const float b2 = sqrt_u1 * u2;

            const float3 p0_obj = v0.pos;
            const float3 p1_obj = v1.pos;
            const float3 p2_obj = v2.pos;
            const float3 x_obj = p0_obj * b0 + p1_obj * b1 + p2_obj * b2;

            // 4) Transform to WORLD and compute WORLD normal using WORLD vertices
            const Transform transform = scene.transforms[light.transformIndex];

            const float3 p0_world = toWorldPoint(p0_obj, transform);
            const float3 p1_world = toWorldPoint(p1_obj, transform);
            const float3 p2_world = toWorldPoint(p2_obj, transform);

            const float3 e0_world = p1_world - p0_world;
            const float3 e1_world = p2_world - p0_world;

            float3 normalWorld = float3{
                e0_world.y() * e1_world.z() - e0_world.z() * e1_world.y(),
                e0_world.z() * e1_world.x() - e0_world.x() * e1_world.z(),
                e0_world.x() * e1_world.y() - e0_world.y() * e1_world.x()
            };

            const float normal_length = sycl::sqrt(dot(normalWorld, normalWorld));
            if (normal_length <= 0.0f)
                return sample;
            normalWorld = normalWorld / normal_length;
            // Emissive Direction
            float pdfDir = 0.0f;
            float3 sampledDirectionW;
            sampleCosineHemisphere(rng128, normalWorld, sampledDirectionW, pdfDir);
            // 5) Fill sample
            sample.positionW = toWorldPoint(x_obj, transform);
            sample.normalW = normalWorld;
            sample.direction = sampledDirectionW;
            // Set as Radiant Flux (WATT)
            sample.flux = light.flux * light.color;
            // Because we sampled proportional to triangle area, then uniformly on that triangle:
            // pdfArea is uniform over the whole emitter area.
            sample.pdfArea = 1.0f / light.totalAreaWorld;
            sample.totalAreaWorld = light.totalAreaWorld;
            sample.pdfDir = pdfDir;
            sample.valid = true;
            sample.lightIndex = light_index;
        }
        else if (light.lightType == LightType::Surfel) {
            const auto& surfel = scene.points[light.primitiveIndex];

            float xi1 = rng128.nextFloat();
            float xi2 = rng128.nextFloat();

            float radius = std::sqrt(xi1);
            float angle = 2.0f * M_PIf * xi2;

            float localU = radius * std::cos(angle);
            float localV = radius * std::sin(angle);

            float3 tangentUWorld = surfel.scale.x() * surfel.tanU;
            float3 tangentVWorld = surfel.scale.y() * surfel.tanV;

            float3 positionWorld =
                surfel.position +
                localU * tangentUWorld +
                localV * tangentVWorld;

            float totalAreaWorld = M_PIf * surfel.scale.x() * surfel.scale.y();
            float pdfArea = 1.0f / totalAreaWorld;

            const float3 normalWorld = normalize(cross(surfel.tanU, surfel.tanV));

            float pdfDir = 0.0f;
            float3 sampledDirectionW;
            sampleCosineHemisphere(rng128, normalWorld, sampledDirectionW, pdfDir);
            // 5) Fill sample
            sample.normalW = normalWorld;
            sample.direction = sampledDirectionW;
            sample.positionW = positionWorld;

            // Set as Radiant Flux (WATT)
            float alphaGeom = 1.0f;
            opacityBeta(localU, localV, surfel, &alphaGeom);
            sample.flux = light.flux * light.color * alphaGeom * surfel.opacity;
            // Because we sampled proportional to triangle area, then uniformly on that triangle:
            // pdfArea is uniform over the whole emitter area.
            sample.pdfArea = pdfArea;
            sample.totalAreaWorld = totalAreaWorld;
            sample.pdfDir = pdfDir;
            sample.valid = true;
            sample.lightIndex = light_index;
            sample.surface.primitiveIndex = light.primitiveIndex;
            sample.surface.alphaGeom = alphaGeom;
            sample.surface.uv = {localU, localV};
        }
        return sample;
    }

    SYCL_EXTERNAL inline GradientRecordRanges makeGradientRecordRanges(
        uint32_t measurementEventCount,
        uint32_t measurementTwoPointEventCount,
        uint32_t materialVertexEventCount) {
        GradientRecordRanges ranges{};

        static constexpr uint32_t measurementRecordsPerEvent =
            1u + kMaxSplatEventsPerRay;

        static constexpr uint32_t measurementTwoPointRecordsPerEvent =
            1u + kMaxSplatEventsPerRay;

        static constexpr uint32_t materialVertexRecordsPerEvent =
            1u;

        ranges.measurementOffset = 0u;
        ranges.measurementCount =
            measurementRecordsPerEvent * measurementEventCount;

        ranges.measurementTwoPointOffset =
            ranges.measurementOffset + ranges.measurementCount;

        ranges.measurementTwoPointCount =
            measurementTwoPointRecordsPerEvent * measurementTwoPointEventCount;

        ranges.materialVertexOffset =
            ranges.measurementTwoPointOffset + ranges.measurementTwoPointCount;

        ranges.materialVertexCount =
            materialVertexRecordsPerEvent * materialVertexEventCount;

        ranges.totalCount =
            ranges.materialVertexOffset + ranges.materialVertexCount;

        return ranges;
    }

    inline void clearPendingCameraSegment(PendingCameraSegment& pendingCameraSegment) {
        pendingCameraSegment.valid = false;
        pendingCameraSegment.pathId = 0u;
        pendingCameraSegment.pixelIndex = 0u;
        pendingCameraSegment.cameraPathThroughput = float3{0.0f, 0.0f, 0.0f};
        pendingCameraSegment.cameraOriginWorld = float3{0.0f, 0.0f, 0.0f};
        pendingCameraSegment.cameraDirectionWorld = float3{0.0f, 0.0f, 0.0f};
    }


    inline float computeGeometricTermValue(
        const float3& x_position,
        const float3& y_position,
        const float3& x_normal,
        const float3& y_normal) {
        const float3 vector_from_x_to_y = y_position - x_position;
        const float squared_distance = dot(vector_from_x_to_y, vector_from_x_to_y);

        if (squared_distance <= 1e-12f) {
            return 0.0f;
        }

        const float inverse_distance = sycl::rsqrt(squared_distance);
        const float3 direction_from_x_to_y = vector_from_x_to_y * inverse_distance;

        const float cosine_at_x = fmax(0.0f, dot(x_normal, direction_from_x_to_y));
        const float cosine_at_y = fmax(0.0f, dot(y_normal, -direction_from_x_to_y));

        return cosine_at_x * cosine_at_y * inverse_distance * inverse_distance;
    }

    static inline float3 reconstructWorldPositionFromDepthCenter(
        const CameraGPU& camera,
        const uint32_t pixelX,
        const uint32_t pixelY,
        const float depth) {
        Ray ray = makePrimaryRayFromPixelJitteredFov(
            camera,
            static_cast<float>(pixelX),
            static_cast<float>(pixelY),
            0.0f, // center pixel in your convention
            0.0f);

        const float denom = dot(camera.forward, ray.direction);
        if (sycl::fabs(denom) <= 1e-8f) {
            return camera.pos;
        }

        const float t = depth / denom;
        return ray.origin + ray.direction * t;
    }
}
