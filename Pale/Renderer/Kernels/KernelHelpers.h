#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include "Renderer/GPUDataTypes.h"

namespace Pale::rng {

// ---------- SplitMix64 for robust seeding ----------
struct SplitMix64 {
    uint64_t currentState;

    explicit SplitMix64(uint64_t initialState) : currentState(initialState) {}

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

    explicit Xorshift32(uint32_t seed) : currentState(seed ? seed : 0xA341316Cu) {}

    inline uint32_t nextUint32() {
        uint32_t x = currentState;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        currentState = x;
        return x;
    }

    inline float nextFloat01() {                  // [0,1)
        // Use top 24 bits to avoid denorms and keep uniform mantissa fill
        return static_cast<float>(nextUint32() >> 8) * 0x1.0p-24f;
    }
};

// ---------- Xorshift64* (better quality than plain 64) ----------
struct Xorshift64Star {
    uint64_t currentState;

    explicit Xorshift64Star(uint64_t seed) : currentState(seed ? seed : 0x9E3779B97F4A7C15ull) {}

    inline uint64_t nextUint64() {
        uint64_t x = currentState;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        currentState = x;
        return x * 0x2545F4914F6CDD1Dull;
    }

    inline double nextDouble01() {                // [0,1)
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
          stateW(sw ? sw : 0x94D049BBu) {}

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
        stateX = stateY; stateY = stateZ; stateZ = stateW;
        stateW = (stateW ^ (stateW >> 19)) ^ (t ^ (t >> 8));
        return stateW;
    }

    float nextFloat() {                  // [0,1)
        return static_cast<float>(nextUint() >> 8) * 0x1.0p-24f;
    }
};

// ---------- Helper: per-item deterministic seeding ----------
inline uint64_t makePerItemSeed1D(uint64_t baseSeed, sycl::id<1> globalId) {
    // Mix base seed with global id to minimize collisions
    uint64_t mixed = baseSeed ^ (0x9E3779B97F4A7C15ull + static_cast<uint64_t>(globalId[0]));
    // One SplitMix64 step to decorrelate contiguous ids
    SplitMix64 sm(mixed);
    return sm.nextUint64();
}



} // namespace pale::rng

namespace Pale {
    inline float3 safeInvDir(const float3 &dir) {
        constexpr float EPS = 1e-8f; // treat anything smaller as “zero”
        constexpr float HUGE = 1e30f; // 2^100 ≃ 1.27e30 still fits in float
        float3 inv;
        inv.x() = (sycl::fabs(dir.x()) < EPS) ? HUGE : 1.f / dir.x();
        inv.y() = (sycl::fabs(dir.y()) < EPS) ? HUGE : 1.f / dir.y();
        inv.z() = (sycl::fabs(dir.z()) < EPS) ? HUGE : 1.f / dir.z();
        return inv;
    }

        inline bool slabIntersectAABB(const Ray &ray,
                                  const TLASNode &node,
                                  const float3 &invDir,
                                  float tMaxLimit,
                                  float &tEntry) {
        float3 t0 = (node.aabbMin - ray.origin) * invDir;
        float3 t1 = (node.aabbMax - ray.origin) * invDir;

        float3 tmin3 = min(t0, t1);
        float3 tmax3 = max(t0, t1);

        float tmin = sycl::fmax(sycl::fmax(tmin3.x(), tmin3.y()), tmin3.z());
        float tmax = sycl::fmin(sycl::fmin(tmax3.x(), tmax3.y()), tmax3.z());

        /* 1.  Origin outside slabs AND entry after exit  ➜  miss          */
        if (tmin > tmax) return false;

        /* 2.  Whole box lies behind the ray                                  */
        if (tmax < 0.0f) return false;

        /* 3.  Already found a closer hit in the SAME SPACE                   */
        if (tmin > tMaxLimit) return false;

        tEntry = sycl::fmax(tmin, 0.0f); // clamp if origin is inside
        return true;
    }


    inline bool slabIntersectAABB(const Ray &ray,
                                  const BVHNode &node,
                                  const float3 &invDir,
                                  float tMaxLimit,
                                  float &tEntry) {
        float3 t0 = (node.aabbMin - ray.origin) * invDir;
        float3 t1 = (node.aabbMax - ray.origin) * invDir;

        float3 tmin3 = min(t0, t1);
        float3 tmax3 = max(t0, t1);

        float tmin = sycl::fmax(sycl::fmax(tmin3.x(), tmin3.y()), tmin3.z());
        float tmax = sycl::fmin(sycl::fmin(tmax3.x(), tmax3.y()), tmax3.z());

        /* 1.  Origin outside slabs AND entry after exit  ➜  miss          */
        if (tmin > tmax) {
            return false;
        }
        /* 2.  Whole box lies behind the ray                                  */
        if (tmax < 0.0f) return false;

        /* 3.  Already found a closer hit in the SAME SPACE                   */

        if (tmin > tMaxLimit) return false;
        constexpr float kEps = 1e-4f;
        tEntry = sycl::fmax(tmin, kEps); // clamp if origin is inside
        return true;
    }

    //──────────────── world → object and back ────────────────────────────────
    inline Ray toObjectSpace(const Ray &rayW, const Transform &xf) {
        Ray r;
        /* 1.  Transform origin – w = 1                                      */
        float4 hO = xf.worldToObject * float4{rayW.origin, 1.f};
        r.origin = float3{hO.x(), hO.y(), hO.z()} / hO.w(); // <- perspective divide

        /* 2.  Transform direction – w = 0  (no translation component)       */
        float4 hD = xf.worldToObject * float4{rayW.direction, 0.f};
        r.direction = normalize(float3{hD.x(), hD.y(), hD.z()}); // w is already 0
        return r;
    }

    inline float3 toWorldPoint(const float3 &pO, const Transform &xf) {
        float4 hp = xf.objectToWorld * float4{pO, 1.f};
        return float3{hp.x(), hp.y(), hp.z()} / hp.w();
    }

    SYCL_EXTERNAL inline bool intersectTriangle(const Ray &ray, const float3 v0, const float3 v1, const float3 v2, float &outT, float &outU,
                                                float &outV, float tMin)     {
        const float3 e1 = v1 - v0;
        const float3 e2 = v2 - v0;

        const float3 h  = cross(ray.direction, e2);
        const float  a  = dot(e1, h);

        // 1. Parallel?
        if (sycl::fabs(a) < 1.0e-4f) return false;

        const float  f  = 1.0f / a;
        const float3 s  = ray.origin - v0;
        const float  u  = f * dot(s, h);
        if (u < 0.0f || u > 1.0f) return false;

        const float3 q  = cross(s, e1);
        const float  v  = f * dot(ray.direction, q);
        if (v < 0.0f || u + v > 1.0f) return false;

        const float  t  = f * dot(e2, q);
        if (t <= tMin) return false;  // behind the ray or farther than a previous hit

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

    inline float3 sampleCosineHemisphere(const float3& unitNormal, rng::Xorshift128& rng, float& pdf) {
        // 1) Draw two uniform variates
        const float uniformSample1 = rng.nextFloat(); // in [0,1)
        const float uniformSample2 = rng.nextFloat(); // in [0,1)

        // 2) Map to cosine-weighted polar coords on hemisphere (z = up)
        const float radial = sycl::sqrt(uniformSample1);
        const float azimuth = 6.28318530717958647692f * uniformSample2; // 2*pi
        const float localX = radial * sycl::cos(azimuth);
        const float localY = radial * sycl::sin(azimuth);
        const float localZ = sycl::sqrt(sycl::fmax(0.0f, 1.0f - uniformSample1)); // cosTheta

        // 3) Build ONB around the normal
        float3 tangent, bitangent;
        buildOrthonormalBasis(unitNormal, tangent, bitangent);

        // 4) Transform to world
        const float3 sampledDirectionW =
            tangent * localX + bitangent * localY + unitNormal * localZ;

        // 5) Normalize for safety and compute pdf = cosTheta / pi
        const float3 unitSampledDirectionW = normalize(sampledDirectionW);
        const float cosineTheta = sycl::fmax(0.0f, dot(unitSampledDirectionW, unitNormal));
        pdf = cosineTheta * (1.0f / 3.14159265358979323846f);

        return unitSampledDirectionW;
    }

    inline void sampleCosineHemisphere(
        rng::Xorshift128& rng, const float3 &n,
        float3 &outDir, float &outPdf) {
        float u1 = rng.nextFloat();
        float u2 = rng.nextFloat();

        float r = sycl::sqrt(u1);
        float phi = 2.f * M_PIf * u2;

        float x = r * sycl::cos(phi);
        float y = r * sycl::sin(phi);
        float z = sycl::sqrt(1.f - u1);

        // build an ONB around n
        float3 up = fabs(n.z()) < .999f ? float3{0, 0, 1} : float3{1, 0, 0};
        float3 tang = normalize(cross(up, n));
        float3 bit = cross(n, tang);

        outDir = normalize(x * tang + y * bit + z * n);
        outPdf = max(0.f, dot(outDir, n)) / M_PIf; // cosθ/π
    }

}
