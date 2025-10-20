#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

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
            return stateW;
        }

        float nextFloat() {
            // [0,1)
            return static_cast<float>(nextUint() >> 8) * 0x1.0p-24f;
        }
    };

    // ---------- Helper: per-item deterministic seeding ----------
    inline uint64_t makePerItemSeed1D(uint64_t baseSeed, uint64_t globalId) {
        // Mix base seed with global id to minimize collisions
        uint64_t mixed = baseSeed ^ (0x9E3779B97F4A7C15ull + static_cast<uint64_t>(globalId));
        // One SplitMix64 step to decorrelate contiguous ids
        SplitMix64 sm(mixed);
        return sm.nextUint64();
    }
} // namespace pale::rng

namespace Pale {
    SYCL_EXTERNAL inline float3 safeInvDir(const float3 &dir) {
        constexpr float EPS = 1e-8f; // treat anything smaller as “zero”
        constexpr float HUGE = 1e30f; // 2^100 ≃ 1.27e30 still fits in float
        float3 inv;
        inv.x() = (abs(dir.x()) < EPS) ? HUGE : 1.f / dir.x();
        inv.y() = (abs(dir.y()) < EPS) ? HUGE : 1.f / dir.y();
        inv.z() = (abs(dir.z()) < EPS) ? HUGE : 1.f / dir.z();
        return inv;
    }

    SYCL_EXTERNAL inline bool slabIntersectAABB(const Ray &ray,
                                                const TLASNode &node,
                                                const float3 &invDir,
                                                float tMaxLimit,
                                                float &tEntry) {
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


    SYCL_EXTERNAL inline bool slabIntersectAABB(const Ray &ray,
                                                const BVHNode &node,
                                                const float3 &invDir,
                                                float tMaxLimit,
                                                float &tEntry) {
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
        constexpr float kEps = 1e-4f;
        tEntry = max(tmin, kEps); // clamp if origin is inside
        return true;
    }

    //──────────────── world → object and back ────────────────────────────────
    SYCL_EXTERNAL inline Ray toObjectSpace(const Ray &rayW, const Transform &xf) {
        Ray r;
        /* 1.  Transform origin – w = 1                                      */
        float4 hO = xf.worldToObject * float4{rayW.origin, 1.f};
        r.origin = float3{hO.x(), hO.y(), hO.z()} / hO.w(); // <- perspective divide

        /* 2.  Transform direction – w = 0  (no translation component)       */
        float4 hD = xf.worldToObject * float4{rayW.direction, 0.f};
        r.direction = normalize(float3{hD.x(), hD.y(), hD.z()}); // w is already 0
        return r;
    }


    SYCL_EXTERNAL inline float3 transformPoint(const float4x4 &tf, const float3 &p, float w = 1.0f) {
        const float4 v = {p, w};
        float4 result = tf * v;
        float invW = 1.f / result.w();
        return float3{result.x() * invW, result.y() * invW, result.z() * invW};
    }

    SYCL_EXTERNAL inline float3 transformDirection(const float4x4 &tf, const float3 &dir) {
        const float4 v = {dir, 0.f};
        float4 r = tf * v;
        return normalize(float3{r.x(), r.y(), r.z()});
    }

    SYCL_EXTERNAL inline float3 toWorldPoint(const float3 &pO, const Transform &xf) {
        float4 hp = xf.objectToWorld * float4{pO, 1.f};
        return float3{hp.x(), hp.y(), hp.z()} / hp.w();
    }

    SYCL_EXTERNAL inline bool intersectTriangle(const Ray &ray, const float3 v0, const float3 v1, const float3 v2,
                                                float &outT, float &outU,
                                                float &outV, float tMin) {
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

    inline float3 buildTangentFrisvad(const float3 &unitNormal) {
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


    inline void buildOrthonormalBasis(const float3 &unitNormal, float3 &tangent, float3 &bitangent) {
        tangent = buildTangentFrisvad(unitNormal);
        bitangent = cross(unitNormal, tangent);
    }

    SYCL_EXTERNAL inline float3 sampleCosineHemisphere(const float3 &unitNormal, rng::Xorshift128 &rng, float &pdf) {
        // 1) Draw two uniform variates
        const float uniformSample1 = rng.nextFloat(); // in [0,1)
        const float uniformSample2 = rng.nextFloat(); // in [0,1)

        // 2) Map to cosine-weighted polar coords on hemisphere (z = up)
        const float radial = sycl::sqrt(uniformSample1);
        const float azimuth = 6.28318530717958647692f * uniformSample2; // 2*pi
        const float localX = radial * sycl::cos(azimuth);
        const float localY = radial * sycl::sin(azimuth);
        const float localZ = sycl::sqrt(max(0.0f, 1.0f - uniformSample1)); // cosTheta

        // 3) Build ONB around the normal
        float3 tangent, bitangent;
        buildOrthonormalBasis(unitNormal, tangent, bitangent);

        // 4) Transform to world
        const float3 sampledDirectionW =
                tangent * localX + bitangent * localY + unitNormal * localZ;

        // 5) Normalize for safety and compute pdf = cosTheta / pi
        const float3 unitSampledDirectionW = normalize(sampledDirectionW);
        const float cosineTheta = max(0.0f, dot(unitSampledDirectionW, unitNormal));
        pdf = cosineTheta * (1.0f / 3.14159265358979323846f);

        return unitSampledDirectionW;
    }

    SYCL_EXTERNAL inline void sampleUniformSphere(
        rng::Xorshift128 &randomNumberGenerator,
        float3 &outDirection,
        float &outPdf
    ) {
        const float uniformRandomOne = randomNumberGenerator.nextFloat();
        const float uniformRandomTwo = randomNumberGenerator.nextFloat();

        const float zCoordinate = 1.0f - 2.0f * uniformRandomOne; // in [-1, 1]
        const float azimuthAngle = 2.0f * M_PIf * uniformRandomTwo;
        const float radialComponent = sycl::sqrt(sycl::fmax(0.0f, 1.0f - zCoordinate * zCoordinate));

        const float xCoordinate = radialComponent * sycl::cos(azimuthAngle);
        const float yCoordinate = radialComponent * sycl::sin(azimuthAngle);

        outDirection = normalize(float3{xCoordinate, yCoordinate, zCoordinate});
        outPdf = 1.0f / (4.0f * M_PIf); // uniform sphere pdf
    }


    SYCL_EXTERNAL inline void sampleCosineHemisphere(
        rng::Xorshift128 &rng, const float3 &n,
        float3 &outDir, float &outPdf) {
        float u1 = rng.nextFloat();
        float u2 = rng.nextFloat();

        float r = sycl::sqrt(u1);
        float phi = 2.f * M_PIf * u2;

        float x = r * sycl::cos(phi);
        float y = r * sycl::sin(phi);
        float z = sycl::sqrt(1.f - u1);

        // build an ONB around n
        float3 up = abs(n.z()) < .999f ? float3{0, 0, 1} : float3{1, 0, 0};
        float3 tang = normalize(cross(up, n));
        float3 bit = cross(n, tang);

        outDir = normalize(x * tang + y * bit + z * n);
        outPdf = max(0.f, dot(outDir, n)) / M_PIf; // cosθ/π
    }

    SYCL_EXTERNAL static bool intersectNearestSurfel(const Ray &rayObject,
                                                     const Point &surfel,
                                                     float tMin,
                                                     float tMax,
                                                     float &outTHit) {
        const float3 tangentU = normalize(surfel.tanU);
        const float3 tangentV = normalize(surfel.tanV);
        float3 normalObject = normalize(cross(tangentU, tangentV));

        const float nDotD = dot(rayObject.direction, normalObject);
        if (abs(nDotD) < 1e-6f) return false;

        const float t = dot(surfel.position - rayObject.origin, normalObject) / nDotD;
        if (t <= tMin || t >= tMax) return false;

        const float3 hitPoint = rayObject.origin + t * rayObject.direction;
        const float3 relative = hitPoint - surfel.position;

        const float alpha = dot(relative, tangentU);
        const float beta = dot(relative, tangentV);

        const float su = fmax(surfel.scale.x(), 1e-8f);
        const float sv = fmax(surfel.scale.y(), 1e-8f);
        const float uHat = alpha / su;
        const float vHat = beta / sv;
        float exponent = -((uHat * uHat) + (vHat * vHat)) / 2.0f;
        float G = std::exp(exponent);
        if (G < 0.75f) return false; // clip to ellipse

        outTHit = t;
        return true;
    }

    SYCL_EXTERNAL static bool intersectSurfel(const Ray &rayObject,
                                              const Point &surfel,
                                              float tMin, float tMax,
                                              float &outTHit,
                                              float3 &outHitLocal,
                                              float &outOpacity,
                                              float kSigmas = 2.1f) {
        // Should match the same kSigmas as in BVH construction
        // 1) Orthonormal in-plane frame (assumes your rotation already baked into tanU/tanV)
        const float3 unitTangentU = normalize(surfel.tanU);
        const float3 unitTangentV = normalize(surfel.tanV - unitTangentU * dot(unitTangentU, surfel.tanV));
        const float3 unitNormal = normalize(cross(unitTangentU, unitTangentV));

        // 2) Ray-plane hit
        const float nDotD = dot(unitNormal, rayObject.direction);
        if (sycl::fabs(nDotD) < 1e-6f)
            return false;

        const float tHit = dot(unitNormal, (surfel.position - rayObject.origin)) / nDotD;
        if (tHit <= tMin || tHit >= tMax)
            return false;

        outHitLocal = rayObject.origin + tHit * rayObject.direction;
        const float3 offsetInPlane = outHitLocal - surfel.position;

        // 3) Local coords
        const float uCoord = dot(unitTangentU, offsetInPlane);
        const float vCoord = dot(unitTangentV, offsetInPlane);

        const float scaleU = sycl::fmax(surfel.scale.x(), 1e-8f);
        const float scaleV = sycl::fmax(surfel.scale.y(), 1e-8f);
        const float uNorm = uCoord / scaleU;
        const float vNorm = vCoord / scaleV;
        const float r2 = uNorm * uNorm + vNorm * vNorm;

        // Optional accel window. Prefer k=3..4. If you keep this, you lose tail mass.
        if (r2 > kSigmas * kSigmas)
            return false;

        // 5) Area → length Jacobian
        const float alpha_i = sycl::exp(-0.5f * r2);
        outTHit = tHit;
        outOpacity = alpha_i;
        return true;
    }

    SYCL_EXTERNAL inline float2 computeSurfelUVAtTHit(const Ray &rayObject,
                                                      const Point &surfel,
                                                      float tHit) {
        const float3 unitTangentU = normalize(surfel.tanU);
        const float3 unitTangentV = normalize(surfel.tanV - unitTangentU * dot(unitTangentU, surfel.tanV));
        const float3 unitNormal = normalize(cross(unitTangentU, unitTangentV));

        // 2) Ray-plane hit
        const float3 hitPointWorld = rayObject.origin + tHit * rayObject.direction;
        const float3 offsetInPlane = hitPointWorld - surfel.position;

        // 3) Local coords
        const float uCoord = dot(unitTangentU, offsetInPlane);
        const float vCoord = dot(unitTangentV, offsetInPlane);

        return {uCoord, vCoord};
    }


    SYCL_EXTERNAL inline bool worldRayToPixel(const CameraGPU &camera,
                                              const float3 &rayOriginWorld,
                                              const float3 &rayDirectionWorld,
                                              uint32_t &outPixelX,
                                              uint32_t &outPixelY) {
        // 1) World -> view
        const float3 rayOriginView = transformPoint(camera.view, rayOriginWorld, 1.f);
        const float3 rayDirectionView = transformDirection(camera.view, rayDirectionWorld);

        // Ray must go toward the image plane (z decreases in OpenGL view)
        constexpr float kEps = 1e-6f;
        if (rayDirectionView.z() >= -kEps) return false;

        // 2) Intersect view-space ray with canonical plane z = -1
        const float zPlane = -1.f;
        const float t = (zPlane - rayOriginView.z()) / rayDirectionView.z();
        if (t <= 0.f) return false;

        const float3 pointOnPlaneView = rayOriginView + t * rayDirectionView;

        // 3) View -> clip -> NDC
        const float4 clip = camera.proj * float4(pointOnPlaneView, 1.f);
        if (clip.w() <= 0.f) return false;

        const float2 ndc = {clip.x() / clip.w(), clip.y() / clip.w()};
        if (ndc.x() < -1.f || ndc.x() > 1.f || ndc.y() < -1.f || ndc.y() > 1.f) return false;

        // 4) NDC -> raster (match your forward mapping and Y flip)
        const float u = ndc.x() * 0.5f + 0.5f;
        const float v = ndc.y() * 0.5f + 0.5f;

        const uint32_t px = sycl::clamp<uint32_t>(static_cast<uint32_t>(u * camera.width), 0u, camera.width - 1u);
        const uint32_t py = sycl::clamp<uint32_t>(static_cast<uint32_t>(v * camera.height), 0u, camera.height - 1u);

        outPixelX = px;
        outPixelY = camera.height - 1u - py; // flip to match your writes
        return true;
    }

    // Generate a primary ray from pixel (uses inverse view-projection on the camera)
    inline Ray makePrimaryRayFromPixel(const CameraGPU &camera,
                                       std::uint32_t pixelX,
                                       std::uint32_t pixelY) {
        // 1) Pixel center → NDC in [-1,1]^2
        const float sx = (static_cast<float>(pixelX) + 0.5f) / static_cast<float>(camera.width);
        float sy = (static_cast<float>(pixelY) + 0.5f) / static_cast<float>(camera.height);

        const float ndcX = 2.0f * sx - 1.0f;
        const float ndcY = 2.0f * sy - 1.0f;

        // 2) Unproject a far point on the view ray
        const float4 clipFar = float4{ndcX, ndcY, 1.0f, 1.0f}; // OpenGL-style clip
        const float4 worldFarH = camera.invView * (camera.invProj * clipFar);
        const float invW = 1.0f / worldFarH.w();
        const float3 worldFar = float3{
            worldFarH.x() * invW,
            worldFarH.y() * invW,
            worldFarH.z() * invW
        };

        // 3) Ray origin = camera position; dir = normalized (far - origin)
        const float4 camPosH = camera.invView * float4{0.0f, 0.0f, 0.0f, 1.0f};
        const float3 rayOrigin = float3{camPosH.x(), camPosH.y(), camPosH.z()};

        float3 rayDirection = worldFar - rayOrigin;
        rayDirection = normalize(rayDirection);

        Ray ray{};
        ray.origin = rayOrigin;
        ray.direction = rayDirection;
        return ray;
    }


    inline Ray makePrimaryRayFromPixelJittered(const CameraGPU &cam,
                                               float px, float py, float jx, float jy) {
        const float sx = (px + 0.5f + jx) / float(cam.width);
        float sy = (py + 0.5f + jy) / float(cam.height);
        const float ndcX = 2.f * sx - 1.f, ndcY = 2.f * sy - 1.f;
        const float4 clipFar = float4{ndcX, ndcY, 1.f, 1.f};
        const float4 worldFarH = cam.invView * (cam.invProj * clipFar);
        const float invW = 1.f / worldFarH.w();
        const float3 worldFar{worldFarH.x() * invW, worldFarH.y() * invW, worldFarH.z() * invW};
        const float4 camPosH = cam.invView * float4{0, 0, 0, 1};
        const float3 origin{camPosH.x(), camPosH.y(), camPosH.z()};
        return Ray{origin, normalize(worldFar - origin)};
    }

    // Photon map lookup helper.
    // Assumes: float3 has + and * operators, dot(), and component-wise ops.
    // Assumes: worldToCell(), linearCellIndex(), and kInvalidIndex are available.

    inline bool isInsideGrid(const sycl::int3 &cell, const sycl::int3 &resolution) {
        return (cell.x() >= 0 && cell.y() >= 0 && cell.z() >= 0 &&
                cell.x() < resolution.x() &&
                cell.y() < resolution.y() &&
                cell.z() < resolution.z());
    }

    //
    static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;

    inline sycl::int3 worldToCell(const float3 &worldPosition,
                                  const DeviceSurfacePhotonMapGrid &grid) {
        const float3 safeCellSize = max(grid.cellSizeWorld, float3{1e-6f, 1e-6f, 1e-6f});
        const float3 relative = worldPosition - grid.gridOriginWorld;
        const float3 r = float3{
            relative.x() / safeCellSize.x(),
            relative.y() / safeCellSize.y(),
            relative.z() / safeCellSize.z()
        };

        return sycl::int3{
            static_cast<int>(sycl::floor(r.x())),
            static_cast<int>(sycl::floor(r.y())),
            static_cast<int>(sycl::floor(r.z()))
        };
    }

    inline std::uint32_t linearCellIndex(const sycl::int3 &cell,
                                         const sycl::int3 &resolution) {
        const int ix = sycl::clamp(static_cast<int>(cell.x()), 0, static_cast<int>(resolution.x()) - 1);
        const int iy = sycl::clamp(static_cast<int>(cell.y()), 0, static_cast<int>(resolution.y()) - 1);
        const int iz = sycl::clamp(static_cast<int>(cell.z()), 0, static_cast<int>(resolution.z()) - 1);

        const auto nx = static_cast<std::uint64_t>(resolution.x());
        const auto ny = static_cast<std::uint64_t>(resolution.y());
        const auto lix = static_cast<std::uint64_t>(ix);
        const auto liy = static_cast<std::uint64_t>(iy);
        const auto liz = static_cast<std::uint64_t>(iz);

        const std::uint64_t linear =
                liz * nx * ny + liy * nx + lix; // fits if totalCellCount < 2^32

        return static_cast<std::uint32_t>(linear);
    }

    inline int signNonZero(float x) { return (x >= 0.0f) ? 1 : -1; }


    inline float3 estimateRadianceFromPhotonMap(
        const WorldHit &worldHit,
        const GPUSceneBuffers &scene,
        const DeviceSurfacePhotonMapGrid &photonMap,
        float perHitRadiusScale = 1.0f, // tune 1–3
        bool photonsIncludeCosineAtStore = false // set false if power excludes cos(θ)
    ) {
        // 1) Material and direct emissive term
        const InstanceRecord &instanceRecord = scene.instances[worldHit.instanceIndex];

        float3 diffuseAlbedoRGB{0.f, 0.f, 0.f};
        float3 radianceDirectRGB{0.f, 0.f, 0.f};

        if (instanceRecord.geometryType == GeometryType::Mesh) {
            const GPUMaterial material = scene.materials[instanceRecord.materialIndex];
            diffuseAlbedoRGB = material.baseColor;

            const bool isEmissiveHit = (material.emissive.x() > 0.f) ||
                                       (material.emissive.y() > 0.f) ||
                                       (material.emissive.z() > 0.f);
            if (isEmissiveHit) {
                // Le toward camera
                radianceDirectRGB = radianceDirectRGB + material.emissive;
            }
        } else {
            // Point-cloud splat surface
            const Point splat = scene.points[worldHit.primitiveIndex];
            diffuseAlbedoRGB = splat.color; // Lambertian splat
            // If you support emissive splats, add them here.
        }

        // 2) Photon gather
        const float3 surfacePositionW = worldHit.hitPositionW;

        const float baseRadius = photonMap.gatherRadiusWorld;
        const float localRadius = sycl::clamp(perHitRadiusScale,
                                              1e-3f * baseRadius,
                                              4.0f * baseRadius);
        const float localRadiusSquared = localRadius * localRadius;

        const float kappa = photonMap.kappa; // e.g. 2.0f
        // Cone-kernel normalization on a disk of radius r: ∫_A w = (1 - 2/(3kappa)) π r^2
        const float inverseConeNormalization =
                1.0f / ((1.0f - 2.0f / (3.0f * kappa)) * M_PIf * localRadiusSquared);


        const sycl::int3 centerCell = worldToCell(surfacePositionW, photonMap);

        float3 weightedSumPhotonPowerRGB{0.f, 0.f, 0.f};

        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const sycl::int3 neighborCell{
                        centerCell.x() + dx,
                        centerCell.y() + dy,
                        centerCell.z() + dz
                    };
                    if (!isInsideGrid(neighborCell, photonMap.gridResolution)) continue;

                    const std::uint32_t cellIndex =
                            linearCellIndex(neighborCell, photonMap.gridResolution);

                    for (std::uint32_t photonIndex = photonMap.cellHeadIndexArray[cellIndex];
                         photonIndex != kInvalidIndex;
                         photonIndex = photonMap.photonNextIndexArray[photonIndex]) {
                        const DevicePhotonSurface photon = photonMap.photons[photonIndex];

                        const float3 displacement = photon.position - surfacePositionW;
                        const float distanceSquared = dot(displacement, displacement);
                        if (distanceSquared > localRadiusSquared) continue;

                        const float distance = sycl::sqrt(distanceSquared);
                        const float kernelWeight = sycl::fmax(0.f, 1.f - distance / (kappa * localRadius));

                        // If photon.power already had cosine at store time, use as-is.
                        // Otherwise multiply by photon.cosineIncident here if available.
                        const float3 photonContributionRGB =
                                photonsIncludeCosineAtStore
                                    ? photon.power
                                    : (photon.power * photon.cosineIncident);

                        weightedSumPhotonPowerRGB = weightedSumPhotonPowerRGB + kernelWeight * photonContributionRGB;
                    }
                }
            }
        }

        const float3 irradianceRGB =
                weightedSumPhotonPowerRGB * inverseConeNormalization;

        const float3 lambertianBRDFRGB = diffuseAlbedoRGB * (1.0f / M_PIf);
        const float3 radianceFromIrradianceRGB = irradianceRGB * lambertianBRDFRGB;

        return radianceDirectRGB + radianceFromIrradianceRGB;
    }


    inline float3 estimateSurfelRadianceFromPhotonMap(
        const SplatEvent &event,
        const float3 &direction,
        const GPUSceneBuffers &scene,
        const DeviceSurfacePhotonMapGrid &photonMap
    ) {
        const float perHitRadiusScale = 1.0f;
        const bool photonsIncludeCosineAtStore = false;
        const float grazingEpsilon = 1e-6f;
        // Material (two-sided Lambert by construction; irradiance already includes cos)
        const Point surfelPoint = scene.points[event.primitiveIndex];
        const float3 diffuseAlbedoRgb = surfelPoint.color;

        // Canonical normal (no face-forwarding). Two-sided shading is fine.
        const float3 canonicalNormalW = normalize(cross(surfelPoint.tanU, surfelPoint.tanV));

        // Side we are *entering first* this segment: dot(n, -wo)
        const int travelSideSign = signNonZero(dot(canonicalNormalW, -direction));

        // Gather setup
        const float3 surfacePositionW = event.hitWorld;
        const float baseRadius = photonMap.gatherRadiusWorld;
        const float requestedRadius = sycl::fmax(1e-6f, perHitRadiusScale) * baseRadius;
        const float localRadius = sycl::clamp(requestedRadius, 1e-3f * baseRadius, 4.0f * baseRadius);
        const float localRadiusSq = localRadius * localRadius;

        const float kappa = photonMap.kappa; // e.g. 2.0
        const float inverseConeNormalization =
                1.0f / ((1.0f - 2.0f / (3.0f * kappa)) * M_PIf * localRadiusSq);

        const sycl::int3 centerCell = worldToCell(surfacePositionW, photonMap);

        float3 weightedSumPhotonPowerRgb{0.f, 0.f, 0.f};

        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const sycl::int3 neighborCell{centerCell.x() + dx, centerCell.y() + dy, centerCell.z() + dz};
                    if (!isInsideGrid(neighborCell, photonMap.gridResolution)) continue;

                    const std::uint32_t cellIndex = linearCellIndex(neighborCell, photonMap.gridResolution);

                    for (std::uint32_t photonIndex = photonMap.cellHeadIndexArray[cellIndex];
                         photonIndex != kInvalidIndex;
                         photonIndex = photonMap.photonNextIndexArray[photonIndex]) {
                        const DevicePhotonSurface photon = photonMap.photons[photonIndex];

                        // Hemisphere gate: accept only photons from the same side we enter first
                        const float nDotWi = dot(canonicalNormalW, photon.incidentDir);
                        if (sycl::fabs(nDotWi) < grazingEpsilon) continue; // ambiguous grazing
                        if (photon.sideSign != travelSideSign) continue;

                        // Distance + kernel
                        const float3 displacement = photon.position - surfacePositionW;
                        const float distSq = dot(displacement, displacement);
                        if (distSq > localRadiusSq) continue;

                        const float dist = sycl::sqrt(distSq);
                        const float kernelWeight = sycl::fmax(0.f, 1.f - dist / (kappa * localRadius));

                        const float3 photonContributionRgb =
                                photonsIncludeCosineAtStore
                                    ? photon.power
                                    : (photon.power * photon.cosineIncident);

                        weightedSumPhotonPowerRgb = weightedSumPhotonPowerRgb + kernelWeight * photonContributionRgb;
                    }
                }
            }
        }

        const float3 irradianceRgb = weightedSumPhotonPowerRgb * inverseConeNormalization;

        // Two-sided Lambert: no extra abs(cos) on outgoing; irradiance already integrated over cos.
        const float3 lambertBrdfRgb = diffuseAlbedoRgb * (1.0f / M_PIf);

        return irradianceRgb * lambertBrdfRgb;
    }


    inline void atomicAddFloat4ToImage(float4 *dst, const float3 &v) {
        for (int c = 0; c < 3; ++c) {
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                    a(reinterpret_cast<float *>(dst)[c]);
            a.fetch_add(v[c]);
        }
        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>
                a(reinterpret_cast<float *>(dst)[3]);
        a.store(1.0f);
    }

    inline void atomicAddFloatToImage(float4 *dst, const float &v) {
        for (int c = 0; c < 3; ++c) {
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                    a(reinterpret_cast<float *>(dst)[c]);
            a.fetch_add(v);
        }
        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>
                a(reinterpret_cast<float *>(dst)[3]);
        a.store(1.0f);
    }

    inline float2 phiInverse(float3 hitWorld, float3 surfelCenter, float3 tu, float3 tv, float su, float sv) {
        float3 r = hitWorld - surfelCenter;
        float2 uv;
        uv[0] = dot(tu, r) / su;
        uv[1] = dot(tv, r) / sv;
        return uv;
    }

    inline float2 phiInverse(const float3 &hitWorld, const Point &surfel) {
        float3 r = hitWorld - surfel.position;
        float2 uv;
        uv[0] = dot(surfel.tanU, r) / surfel.scale.x();
        uv[1] = dot(surfel.tanV, r) / surfel.scale.y();
        return uv;
    }

    inline float3 phiMapping(float3 surfelCenter, float3 tu, float3 tv, float su, float sv, float u, float v) {
        return surfelCenter + su * tu * u + sv * tv * v;
    }

    inline float computeLuminanceRec709(const float3 &inputRgbLinear) {
        const float redWeight = 0.2126f;
        const float greenWeight = 0.7152f;
        const float blueWeight = 0.0722f;
        return redWeight * inputRgbLinear[0]
               + greenWeight * inputRgbLinear[1]
               + blueWeight * inputRgbLinear[2];
    }

    inline float luminance(const float3 &rgb) {
        return computeLuminanceRec709(rgb);
    }

    inline float luminanceGrayscale(const float3 &inputRgbLinear) {
        const float redWeight = 0.33f;
        const float greenWeight = 0.33f;
        const float blueWeight = 0.33f;
        return redWeight * inputRgbLinear[0]
               + greenWeight * inputRgbLinear[1]
               + blueWeight * inputRgbLinear[2];
    }
}
