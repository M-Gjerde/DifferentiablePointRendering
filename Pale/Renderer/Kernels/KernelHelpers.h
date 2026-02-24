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
    struct DebugPixel {
        uint32_t pixelX;
        uint32_t pixelY;
    };

    static DebugPixel kDebugPixels[] = {
        {800, 500},
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
        sample.power = light.power * light.color;

        // Because we sampled proportional to triangle area, then uniformly on that triangle:
        // pdfArea is uniform over the whole emitter area.
        sample.pdfArea = 1.0f / light.totalAreaWorld;
        sample.pdfDir = pdfDir;

        sample.valid = true;
        sample.lightIndex = light_index;
        return sample;
    }


    SYCL_EXTERNAL inline float3 sampleCosineHemisphere(const float3& unitNormal, rng::Xorshift128& rng, float& pdf) {
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


    inline float2 phiInverse(float3 hitWorld, float3 surfelCenter, float3 tu, float3 tv, float su, float sv) {
        float3 r = hitWorld - surfelCenter;
        float2 uv;
        uv[0] = dot(tu, r) / su;
        uv[1] = dot(tv, r) / sv;
        return uv;
    }

    inline float2 phiInverse(const float3& hitWorld, const Point& surfel) {
        float3 r = hitWorld - surfel.position;
        float2 uv;
        uv[0] = dot(surfel.tanU, r) / surfel.scale.x();
        uv[1] = dot(surfel.tanV, r) / surfel.scale.y();
        return uv;
    }

    SYCL_EXTERNAL inline void sampleUniformSphere(
        rng::Xorshift128& randomNumberGenerator,
        float3& outDirection,
        float& outPdf
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
        // if (dot(outDirectionWorld, unitNormal) < 0.0f) outDirectionWorld = -outDirectionWorld;
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

    SYCL_EXTERNAL inline float2 computeSurfelUVAtTHit(const Ray& rayObject,
                                                      const Point& surfel,
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


    SYCL_EXTERNAL inline bool worldRayToPixel(const CameraGPU& camera,
                                              const float3& rayOriginWorld,
                                              const float3& rayDirectionWorld,
                                              uint32_t& outPixelX,
                                              uint32_t& outPixelY) {
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
    inline Ray makePrimaryRayFromPixel(const CameraGPU& camera,
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


    SYCL_EXTERNAL inline Ray makePrimaryRayFromPixelJitteredFov(
        const CameraGPU& cam,
        float px, float py,
        float jx, float jy) {
        const float width = static_cast<float>(cam.width);
        const float height = static_cast<float>(cam.height);

        const float u = (px + 0.5f + jx);
        const float v = (py + 0.5f + jy);

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


    SYCL_EXTERNAL inline bool projectPinholeToPixel(
        const SensorGPU& sensor,
        const float3& pointWorld,
        uint32_t& outPixelIndex,
        float3& outOmegaFromSurfaceToCamera,
        float& outDistance) {
        /*
        const float3 cameraPositionWorld = sensor.camera.pos;

        const float3 vectorFromSurfaceToCamera = cameraPositionWorld - pointWorld;
        const float distance = length(vectorFromSurfaceToCamera);
        if (distance <= 0.0f)
            return false;

        const float3 omegaSurfaceToCamera = vectorFromSurfaceToCamera / distance;

        // Transform point into camera space
        const float3 pointCamera = transformPoint(sensor.camera.invView, omegaSurfaceToCamera);

        if (pointCamera.z() <= 0.0f)
            return false;

        const float u = sensor.camera.fx * (pointCamera.x() / pointCamera.z()) + sensor.camera.cx;
        const float v = sensor.camera.fy * (pointCamera.y() / pointCamera.z()) + sensor.camera.cy;

        const int pixelX = static_cast<int>(sycl::floor(u));
        const int pixelY = static_cast<int>(sycl::floor(v));

        if (pixelX < 0 || pixelX >= static_cast<int>(sensor.width) ||
            pixelY < 0 || pixelY >= static_cast<int>(sensor.height))
            return false;

        outPixelIndex = static_cast<uint32_t>(pixelY) * sensor.width + static_cast<uint32_t>(pixelX);
        outOmegaFromSurfaceToCamera = omegaSurfaceToCamera;
        outDistance = distance;
        */
        return false;
    }


    inline bool isInsideGrid(const sycl::int3& cell, const sycl::int3& resolution) {
        return (cell.x() >= 0 && cell.y() >= 0 && cell.z() >= 0 &&
            cell.x() < resolution.x() &&
            cell.y() < resolution.y() &&
            cell.z() < resolution.z());
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


    inline float squaredDistanceToCellBounds(const float3& point,
                                             const sycl::int3& cell,
                                             const DeviceSurfacePhotonMapGrid& grid) {
        const float3 cellMin = grid.gridOriginWorld + float3{
            static_cast<float>(cell.x()) * grid.cellSizeWorld.x(),
            static_cast<float>(cell.y()) * grid.cellSizeWorld.y(),
            static_cast<float>(cell.z()) * grid.cellSizeWorld.z()
        };
        const float3 cellMax = cellMin + grid.cellSizeWorld;

        const float dx = sycl::fmax(cellMin.x() - point.x(), point.x() - cellMax.x());
        const float dy = sycl::fmax(cellMin.y() - point.y(), point.y() - cellMax.y());
        const float dz = sycl::fmax(cellMin.z() - point.z(), point.z() - cellMax.z());

        return dx * dx + dy * dy + dz * dz;
    }

    inline sycl::int3 clampCell(
        const sycl::int3& cell,
        const DeviceSurfacePhotonMapGrid& grid) {
        return sycl::int3{
            sycl::clamp(cell.x(), 0, int(grid.gridResolution.x()) - 1),
            sycl::clamp(cell.y(), 0, int(grid.gridResolution.y()) - 1),
            sycl::clamp(cell.z(), 0, int(grid.gridResolution.z()) - 1)
        };
    }


    inline int signNonZero(float x) { return (x >= 0.0f) ? 1 : -1; }

    inline float3 cellToWorldCenter(
        const sycl::int3& cellCoord,
        const DeviceSurfacePhotonMapGrid& grid) {
        // Convert integer cell index to center position in world space
        const float3 cellCoordFloat = float3{
            float(cellCoord.x()) + 0.5f,
            float(cellCoord.y()) + 0.5f,
            float(cellCoord.z()) + 0.5f
        };

        return grid.gridOriginWorld + cellCoordFloat * grid.cellSizeWorld;
    }


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


    inline float3 gatherDiffuseIrradianceAtPoint(
        const float3& queryPositionWorld,
        const float3& surfelNormalW,
        const DeviceSurfacePhotonMapGrid& grid) {
        const float r = grid.gatherRadiusWorld;
        const float r2 = r * r;
        const float invArea = 1.0f / (M_PIf * r2);

        const sycl::int3 minCell = worldToCellClamped(queryPositionWorld - float3{r, r, r}, grid);
        const sycl::int3 maxCell = worldToCellClamped(queryPositionWorld + float3{r, r, r}, grid);

        float3 irradiance = float3{0.0f};

        for (int cz = minCell.z(); cz <= maxCell.z(); ++cz)
            for (int cy = minCell.y(); cy <= maxCell.y(); ++cy)
                for (int cx = minCell.x(); cx <= maxCell.x(); ++cx) {
                    const uint32_t cellId = linearCellIndex(sycl::int3{cx, cy, cz}, grid.gridResolution);
                    const uint32_t start = grid.cellStart[cellId];
                    if (start == kInvalidIndex)
                        continue;

                    const uint32_t end = grid.cellEnd[cellId];
                    for (uint32_t j = start; j < end; ++j) {
                        const uint32_t photonIndex = grid.sortedPhotonIndex[j];
                        const DevicePhotonSurface ph = grid.photons[photonIndex];

                        const float3 d = ph.position - queryPositionWorld;

                        const float cosine = dot(surfelNormalW, ph.normal);
                        if (cosine < 0.3f) continue;

                        const float dist2 = dot(d, d);
                        if (dist2 > r2)
                            continue;

                        irradiance += (ph.power * invArea);
                    }
                }

        return irradiance;
    }

    template <int kNumNearest>
    struct KnnBuffer {
        float distanceSquared[kNumNearest];
        uint32_t photonIndex[kNumNearest];

        int count = 0;
        int worstIndex = 0;
        float worstDistanceSquared = FLT_MAX;

        inline void clear() {
            count = 0;
            worstIndex = 0;
            worstDistanceSquared = FLT_MAX;
        }

        inline void recomputeWorst() {
            if (count <= 0) {
                worstIndex = 0;
                worstDistanceSquared = FLT_MAX;
                return;
            }

            int currentWorstIndex = 0;
            float currentWorstDistanceSquared = distanceSquared[0];

            for (int i = 1; i < count; ++i) {
                if (distanceSquared[i] > currentWorstDistanceSquared) {
                    currentWorstDistanceSquared = distanceSquared[i];
                    currentWorstIndex = i;
                }
            }

            worstIndex = currentWorstIndex;
            worstDistanceSquared = currentWorstDistanceSquared;
        }

        inline void tryInsert(float newDistanceSquared, uint32_t newPhotonIndex) {
            // Reject NaNs early to avoid poisoning comparisons.
            if (!(newDistanceSquared >= 0.0f) || sycl::isnan(newDistanceSquared))
                return;

            if (count < kNumNearest) {
                distanceSquared[count] = newDistanceSquared;
                photonIndex[count] = newPhotonIndex;
                ++count;

                // Keep worstDistanceSquared valid as soon as we become full,
                // and also correct when we just inserted the first element.
                if (count == 1 || count == kNumNearest)
                    recomputeWorst();

                return;
            }

            // Buffer full: replace current worst if better
            if (newDistanceSquared >= worstDistanceSquared)
                return;

            distanceSquared[worstIndex] = newDistanceSquared;
            photonIndex[worstIndex] = newPhotonIndex;
            recomputeWorst();
        }

        inline bool isFull() const { return count == kNumNearest; }
        inline float currentRadiusSquared() const { return worstDistanceSquared; }
        // valid if count>0, exact if isFull()
    };

    // Epanechnikov (2D) kernel for disk radius r:
    // w(u) = max(0, 1 - u), where u = d^2 / r^2
    // Normalized density uses: (2 / (pi r^2)) * sum_i w_i * flux_i
    // (Normalization constant differs by dimension; for your invArea form this is the consistent 2D one.)

    inline float3 gatherDiffuseIrradianceAtPointKNN(
        const float3& queryPositionWorld,
        const float3& surfelNormalW,
        const DeviceSurfacePhotonMapGrid& grid,
        int travelSideSign = 0,
        bool readOneSidedRadiance = false) {
        static constexpr int kNumNearest = 2048;

        KnnBuffer<kNumNearest> knn;

        const float3 cellSizeWorld = grid.cellSizeWorld;
        const sycl::int3 queryCell = worldToCellClamped(queryPositionWorld, grid);

        const float3 queryCellCenterWorld = cellToWorldCenter(queryCell, grid);
        const float3 queryOffsetWorld = queryPositionWorld - queryCellCenterWorld;

        auto processCell = [&](const sycl::int3& cellCoord) {
            const uint32_t cellId = linearCellIndex(cellCoord, grid.gridResolution);
            const uint32_t start = grid.cellStart[cellId];
            if (start == kInvalidIndex)
                return;

            const uint32_t end = grid.cellEnd[cellId];
            for (uint32_t j = start; j < end; ++j) {
                const uint32_t photonArrayIndex = grid.sortedPhotonIndex[j];
                const DevicePhotonSurface photon = grid.photons[photonArrayIndex];

                // ---- CHANGE (1): use tangent-plane distance (2D) instead of 3D distance ----
                const float3 deltaWorld = photon.position - queryPositionWorld;
                const float planeOffset = dot(deltaWorld, surfelNormalW);
                const float3 tangentDeltaWorld = deltaWorld - planeOffset * surfelNormalW;
                const float distanceSquared = dot(tangentDeltaWorld, tangentDeltaWorld);

                knn.tryInsert(distanceSquared, photonArrayIndex);
            }
        };

        const int maxRing = sycl::max(
                int(grid.gridResolution.x()),
                sycl::max(int(grid.gridResolution.y()), int(grid.gridResolution.z())))
            / 5;

        for (int ring = 0; ring <= maxRing; ++ring) {
            const int minX = sycl::max(0, queryCell.x() - ring);
            const int maxX = sycl::min(int(grid.gridResolution.x() - 1), queryCell.x() + ring);
            const int minY = sycl::max(0, queryCell.y() - ring);
            const int maxY = sycl::min(int(grid.gridResolution.y() - 1), queryCell.y() + ring);
            const int minZ = sycl::max(0, queryCell.z() - ring);
            const int maxZ = sycl::min(int(grid.gridResolution.z() - 1), queryCell.z() + ring);

            for (int cz = minZ; cz <= maxZ; ++cz) {
                for (int cy = minY; cy <= maxY; ++cy) {
                    processCell(sycl::int3{minX, cy, cz});
                    if (maxX != minX)
                        processCell(sycl::int3{maxX, cy, cz});
                }
            }

            for (int cz = minZ; cz <= maxZ; ++cz) {
                for (int cx = minX + 1; cx <= maxX - 1; ++cx) {
                    processCell(sycl::int3{cx, minY, cz});
                    if (maxY != minY)
                        processCell(sycl::int3{cx, maxY, cz});
                }
            }

            for (int cy = minY + 1; cy <= maxY - 1; ++cy) {
                for (int cx = minX + 1; cx <= maxX - 1; ++cx) {
                    processCell(sycl::int3{cx, cy, minZ});
                    if (maxZ != minZ)
                        processCell(sycl::int3{cx, cy, maxZ});
                }
            }

            if (knn.isFull()) {
                const float3 visitedHalfExtentWorld = (float(ring) + 0.5f) * cellSizeWorld;

                float dx = visitedHalfExtentWorld.x() - sycl::fabs(queryOffsetWorld.x());
                float dy = visitedHalfExtentWorld.y() - sycl::fabs(queryOffsetWorld.y());
                float dz = visitedHalfExtentWorld.z() - sycl::fabs(queryOffsetWorld.z());

                dx = sycl::fmax(dx, 0.0f);
                dy = sycl::fmax(dy, 0.0f);
                dz = sycl::fmax(dz, 0.0f);

                const float minOutsideDistance = sycl::fmin(dx, sycl::fmin(dy, dz));
                const float minOutsideDistanceSquared = minOutsideDistance * minOutsideDistance;

                if (minOutsideDistanceSquared >= knn.currentRadiusSquared())
                    break;
            }
        }

        if (knn.count == 0)
            return float3{0.0f};

        knn.recomputeWorst();

        float radiusSquared = knn.currentRadiusSquared();
        if (!(radiusSquared > 0.0f) || sycl::isnan(radiusSquared) || !sycl::isfinite(radiusSquared))
            return float3{0.0f};

        const float invRadiusSquared = 1.0f / radiusSquared;
        const float normalization = 2.0f / (M_PIf * radiusSquared);

        float3 weightedFluxSum = float3{0.0f};

        // ---- OPTIONAL part of (1): soft thickness weight to reduce cross-surface photons ----
        // If you do not have this field, add one (or hardcode a constant) in your grid settings.
        // Choose thickness ~ 1–3x surfel spacing (world units).
        const float thicknessWorld = 0.001f; // add to DeviceSurfacePhotonMapGrid
        const float invThicknessSquared = (thicknessWorld > 0.0f) ? (1.0f / (thicknessWorld * thicknessWorld)) : 0.0f;

        for (int i = 0; i < knn.count; ++i) {
            const DevicePhotonSurface photon = grid.photons[knn.photonIndex[i]];

            float sideWeight = 1.0f;
            if (readOneSidedRadiance) {
                //sideWeight = (photon.sideSign == travelSideSign) ? 1.0f : 0.0f;
                //if (sideWeight == 0.0f)
                //    continue;
            }

            const float3 deltaWorld = photon.position - queryPositionWorld;

            // ---- CHANGE (1): tangent-plane distance used for kernel radius u ----
            const float planeOffset = dot(deltaWorld, surfelNormalW);
            const float3 tangentDeltaWorld = deltaWorld - planeOffset * surfelNormalW;
            const float distanceSquared = dot(tangentDeltaWorld, tangentDeltaWorld);

            const float u = sycl::fmin(distanceSquared * invRadiusSquared, 1.0f);
            const float epanechnikovWeight = sycl::fmax(0.0f, 1.0f - u);

            // Soft thickness (Gaussian) in the normal direction.
            float planeWeight = 1.0f;
            if (thicknessWorld > 0.0f)
                planeWeight = sycl::exp(-(planeOffset * planeOffset) * invThicknessSquared);

            const float combinedWeight = epanechnikovWeight * sideWeight * planeWeight;

            weightedFluxSum += photon.power * combinedWeight;
        }

        return weightedFluxSum * normalization;
    }

    /*
    inline float3 computeLSurfel(const Point &surfel, const float3 &direction, const SplatEvent &splatEvent,
                                 const DeviceSurfacePhotonMapGrid &photonMap) {
        const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));
        const int travelSideSign = signNonZero(dot(canonicalNormalW, -direction));

        const float3 frontNormalW = canonicalNormalW * float(travelSideSign);
        const float3 rho = surfel.albedo;

        const float3 E = gatherDiffuseIrradianceAtPoint(
            splatEvent.hitWorld,
            frontNormalW,
            photonMap,
            travelSideSign,
            true
        );

        float3 surfelShadedRadiance = E * (rho * M_1_PIf);
        return surfelShadedRadiance;

        return float3{0.0f};
    }


    inline float3 estimateRadianceFromPhotonMap(
        const WorldHit& worldHit,
        const GPUSceneBuffers& scene,
        const DeviceSurfacePhotonMapGrid& photonMap,
        bool includeBRDF = true,
        bool includeCosine = true, // set false if power excludes cos(θ
        float perHitRadiusScale = 1.0f // tune 1–3
    ) {

        // 1) Material and direct emissive term
        const InstanceRecord& instanceRecord = scene.instances[worldHit.instanceIndex];

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
        }
        else {
            // Point-cloud splat surface
            const Point splat = scene.points[worldHit.primitiveIndex];
            diffuseAlbedoRGB = splat.albedo; // Lambertian splat
            // If you support emissive splats, add them here.
        }

        // 2) Photon gather
        const float3 surfacePositionW = worldHit.hitPositionW;

        const float baseRadius = photonMap.gatherRadiusWorld;
        const float requestedRadius = sycl::fmax(1e-6f, perHitRadiusScale) * baseRadius;
        const float localRadius = sycl::clamp(requestedRadius, 1e-3f * baseRadius, 4.0f * baseRadius);
        const float localRadiusSquared = localRadius * localRadius;

        const float kappa = photonMap.kappa; // e.g. 2.0f
        // Cone-kernel normalization on a disk of radius r: ∫_A w = (1 - 2/(3kappa)) π r^2
        const float inverseConeNormalization =
            1.0f / ((1.0f - 2.0f / (3.0f * kappa)) * M_PIf * localRadiusSquared);


        const sycl::int3 centerCell = worldToCell(surfacePositionW, photonMap);

        float3 weightedSumPhotonPowerRGB{0.f, 0.f, 0.f};

        const float3 cellSize = photonMap.cellSizeWorld; // store this
        const int rx = sycl::min(int(sycl::ceil(localRadius / cellSize.x())), 1 << 10);
        const int ry = sycl::min(int(sycl::ceil(localRadius / cellSize.y())), 1 << 10);
        const int rz = sycl::min(int(sycl::ceil(localRadius / cellSize.z())), 1 << 10);
        for (int dz = -rz; dz <= rz; ++dz) {
            for (int dy = -ry; dy <= ry; ++dy) {
                for (int dx = -rx; dx <= rx; ++dx) {
                    const sycl::int3 neighborCell{
                        centerCell.x() + dx,
                        centerCell.y() + dy,
                        centerCell.z() + dz
                    };

                    if (!isInsideGrid(neighborCell, photonMap.gridResolution)) continue;
                    if (squaredDistanceToCellBounds(surfacePositionW, neighborCell, photonMap) > localRadiusSquared)
                        continue;

                    const std::uint32_t cellIndex =
                        linearCellIndex(neighborCell, photonMap.gridResolution);

                    for (std::uint32_t photonIndex = photonMap.cellHeadIndexArray[cellIndex];
                         photonIndex != kInvalidIndex;
                         photonIndex = photonMap.photonNextIndexArray[photonIndex]) {
                        const DevicePhotonSurface photon = photonMap.photons[photonIndex];
                        if (photon.primitiveIndex != worldHit.instanceIndex)
                            continue;

                        const float3 displacement = photon.position - surfacePositionW;
                        const float distanceSquared = dot(displacement, displacement);
                        if (distanceSquared > localRadiusSquared) continue;

                        const float distance = sycl::sqrt(distanceSquared);
                        const float kernelWeight = sycl::fmax(0.f, 1.f - distance / (kappa * localRadius));

                        // If photon.power already had cosine at store time, use as-is.
                        // Otherwise multiply by photon.cosineIncident here if available.
                        const float3 photonContributionRGB =
                            includeCosine
                                ? (photon.power * photon.cosineIncident)
                                : photon.power;

                        weightedSumPhotonPowerRGB = weightedSumPhotonPowerRGB + kernelWeight * photonContributionRGB;
                    }
                }
            }
        }

        const float3 irradianceRGB =
            weightedSumPhotonPowerRGB * inverseConeNormalization;

        float3 lambertBrdfRgb = includeBRDF ? diffuseAlbedoRGB * M_1_PIf : float3{1.0f};

        const float3 radianceFromIrradianceRGB = irradianceRGB * lambertBrdfRgb;

        return radianceDirectRGB + radianceFromIrradianceRGB;

        return float3{0.0f};
    }


    inline float3 estimateSurfelRadianceFromPhotonMap(
        const SplatEvent& event,
        const float3& direction,
        const GPUSceneBuffers& scene,
        const DeviceSurfacePhotonMapGrid& photonMap,
        bool readOneSidedRadiance = false,
        bool includeBrdf = true,
        bool includeCosine = true,
        bool filterPrimitiveIndex = true
    ) {

        const float perHitRadiusScale = 1.0f;
        // Material (two-sided Lambert by construction; irradiance already includes cos)
        const Point surfelPoint = scene.points[event.primitiveIndex];
        const float3 diffuseAlbedoRgb = surfelPoint.albedo;

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

        const float3 cellSize = photonMap.cellSizeWorld; // store this
        const int rx = sycl::min(int(sycl::ceil(localRadius / cellSize.x())), 1 << 10);
        const int ry = sycl::min(int(sycl::ceil(localRadius / cellSize.y())), 1 << 10);
        const int rz = sycl::min(int(sycl::ceil(localRadius / cellSize.z())), 1 << 10);
        for (int dz = -rz; dz <= rz; ++dz) {
            for (int dy = -ry; dy <= ry; ++dy) {
                for (int dx = -rx; dx <= rx; ++dx) {
                    const sycl::int3 neighborCell{centerCell.x() + dx, centerCell.y() + dy, centerCell.z() + dz};
                    if (!isInsideGrid(neighborCell, photonMap.gridResolution)) continue;
                    if (squaredDistanceToCellBounds(surfacePositionW, neighborCell, photonMap) > localRadiusSq)
                        continue;

                    const std::uint32_t cellIndex = linearCellIndex(neighborCell, photonMap.gridResolution);

                    for (std::uint32_t photonIndex = photonMap.cellHeadIndexArray[cellIndex];
                         photonIndex != kInvalidIndex;
                         photonIndex = photonMap.photonNextIndexArray[photonIndex]) {
                        const DevicePhotonSurface photon = photonMap.photons[photonIndex];
                        if (photon.primitiveIndex != event.primitiveIndex && filterPrimitiveIndex)
                            continue;
                        if (photon.sideSign != travelSideSign && readOneSidedRadiance) continue;
                        // Hemisphere gate: accept only photons from the same side we enter first
                        //const float nDotWi = dot(canonicalNormalW, photon.incidentDir);
                        //if (sycl::fabs(nDotWi) < grazingEpsilon) continue; // ambiguous grazing

                        // Distance + kernel
                        const float3 displacement = photon.position - surfacePositionW;
                        const float distSq = dot(displacement, displacement);
                        if (distSq > localRadiusSq) continue;

                        const float dist = sycl::sqrt(distSq);
                        const float kernelWeight = sycl::fmax(0.f, 1.f - dist / (kappa * localRadius));

                        const float3 photonContributionRgb =
                            includeCosine
                                ? (photon.power * photon.cosineIncident)
                                : photon.power;

                        weightedSumPhotonPowerRgb = weightedSumPhotonPowerRgb + kernelWeight * photonContributionRgb;
                    }
                }
            }
        }

        float3 irradianceRgb = weightedSumPhotonPowerRgb * inverseConeNormalization;

        // Two-sided Lambert: no extra abs(cos) on outgoing; irradiance already integrated over cos.
        if (includeBrdf)
            irradianceRgb *= diffuseAlbedoRgb * M_1_PIf;

        return irradianceRgb;

        return float3{0.0f};
    }

        */


    inline void atomicAddFloat4ToImage(float4* dst, const float4& v) {
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

    inline void atomicAddFloatToImage(float4* dst, const float& v) {
        for (int c = 0; c < 3; ++c) {
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                a(reinterpret_cast<float*>(dst)[c]);
            a.fetch_add(v);
        }
        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            a(reinterpret_cast<float*>(dst)[3]);
        a.store(1.0f);
    }

    inline float3 phiMapping(float3 surfelCenter, float3 tu, float3 tv, float su, float sv, float u, float v) {
        return surfelCenter + su * tu * u + sv * tv * v;
    }

    inline float3 phiMapping(const Point& surfel, float u, float v) {
        return surfel.position + surfel.scale.x() * surfel.tanU * u + surfel.scale.y() * surfel.tanV * v;
    }

    inline float computeLuminanceRec709(const float3& inputRgbLinear) {
        const float redWeight = 0.2126f;
        const float greenWeight = 0.7152f;
        const float blueWeight = 0.0722f;
        return redWeight * inputRgbLinear[0]
            + greenWeight * inputRgbLinear[1]
            + blueWeight * inputRgbLinear[2];
    }

    inline float luminance(const float3& rgb) {
        return computeLuminanceRec709(rgb);
    }

    inline float luminanceGrayscale(const float3& inputRgbLinear) {
        const float redWeight = 0.33f;
        const float greenWeight = 0.33f;
        const float blueWeight = 0.33f;
        return redWeight * inputRgbLinear[0]
            + greenWeight * inputRgbLinear[1]
            + blueWeight * inputRgbLinear[2];
    }

    inline void breakAtMeshInstance(const WorldHit& worldHit) {
        switch (worldHit.instanceIndex) {
        case 0:
            break;
        case 1:
            break;
        case 2:
            break;
        case 5:
            break;
        default:
            break;
        }
    }

    inline void breakAtPointCloudInstance(const WorldHit& worldHit) {
        switch (worldHit.primitiveIndex) {
        case 0:
            break;
        case 1:
            break;
        case 2:
            break;
        case 7:
            break;
        default:
            break;
        }
    }


    inline uint32_t flippedYLinearIndex(uint32_t linearIndex, uint32_t W, uint32_t H) {
        // Map to pixel (linear index; X/Y only needed if you want them)
        // ------------------------------------------------------------
        // 1. Recover pixel coordinates (unflipped)
        // ------------------------------------------------------------
        const uint32_t pixelX = linearIndex % W;
        const uint32_t pixelY = linearIndex / W;

        // ------------------------------------------------------------
        // 2. Compute flipped Y coordinate for LDR output
        // ------------------------------------------------------------
        const uint32_t flippedY = (H - 1u) - pixelY;
        const uint32_t flippedLinearIndex = flippedY * W + pixelX;
        return flippedLinearIndex;
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

    SYCL_EXTERNAL inline void appendTransmittanceContributionAtomic(
        uint32_t* globalContributionCounter,
        HitTransmittanceContribution* globalContributionBuffer,
        uint32_t maxContributionCapacity,
        const HitTransmittanceContribution& contributionValue) {
    }

    SYCL_EXTERNAL inline void depositPhotonSurface(
        const WorldHit& worldHit,
        const float3& normal,
        const float3& flux,
        const DeviceSurfacePhotonMapGrid& photonMap) {
        // Atomic counter for photon slots
        auto photonCounter = sycl::atomic_ref<
            uint32_t,
            sycl::memory_order::acq_rel,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>(
            *photonMap.photonCountDevicePtr);

        const uint32_t slot = photonCounter.fetch_add(1u);

        // Capacity guard
        if (slot >= photonMap.photonCapacity) {
            return;
        }

        DevicePhotonSurface photonEntry{};
        photonEntry.position = worldHit.hitPositionW;

        // Geometric normal (unoriented by design)
        photonEntry.normal = normal;

        // Incoming direction (towards surface)
        //photonEntry.incomingDirection = -rayState.ray.direction;

        // Power carried by the photon
        photonEntry.power = flux;

        photonEntry.isValid = 1u;

        photonMap.photons[slot] = photonEntry;
    }

    SYCL_EXTERNAL inline void appendCompletedGradientEventAtomic(
        uint32_t* globalCompletedCounter,
        CompletedGradientEvent* globalCompletedBuffer,
        uint32_t maxCompletedCapacity,
        const CompletedGradientEvent& eventValue) {
        sycl::atomic_ref<
            uint32_t,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space
        > counter(*globalCompletedCounter);

        const uint32_t insertionIndex = counter.fetch_add(1);

        if (insertionIndex >= maxCompletedCapacity) {
            return;
        }

        globalCompletedBuffer[insertionIndex] = eventValue;
    }

    SYCL_EXTERNAL inline void clearPendingAdjointState(PendingAdjointState& pending) {
        pending.kind = PendingAdjointKind::None;
        pending.primitiveIndex = UINT32_MAX;
        pending.alphaGeom = 0.0f;
        pending.hitPosition = float3{0.0f};
        pending.pathThroughput = float3{0.0f};
        pending.pixelIndex = 0;
    }

    SYCL_EXTERNAL inline float computeEndpointCosine(const Ray& incomingRay, const float3& endpointNormalW) {
        return dot(-incomingRay.direction, endpointNormalW);
    }
}
