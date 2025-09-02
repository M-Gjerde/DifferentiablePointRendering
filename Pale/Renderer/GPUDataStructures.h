#pragma once

#include <numbers>

#include "Renderer/GPUDataTypes.h"


namespace Pale {
    /*────────────────────────────────────────────────────────────────────────────*/
    /*  Helper macro – verify every struct is 16‑byte aligned & sized             */
    /*────────────────────────────────────────────────────────────────────────────*/
#define CHECK_16(T) static_assert(alignof(T)==16 && sizeof(T)%16==0,           \
                                 "" #T " must be 16‑byte aligned & sized")

    /*************************  Core Geometry *************************/
    struct alignas(16) Vertex {
        float3 pos; // 16 B
        float3 norm; // 16 B
    };

    CHECK_16(Vertex);

    struct alignas(16) Triangle {
        uint32_t v0{}, v1{}, v2{}; // 12 B
        uint32_t _pad0{}; // 16 B     ← keeps centroid 16‑aligned
        float3 centroid; // 16 B
        uint32_t _pad1{}; // 32 B ↦ sizeof()==32
    };

    CHECK_16(Triangle);

    enum PointType : uint32_t {
        Gaussian2DPoint,
        QuadricPoint
    };

    struct alignas(16) OrientedPoint {
        // QUadric version
        float c{};
        float threshold{};
        float beta{};
        float _pad0{}; // align next float2 to 16‑byte boundary
        uint32_t material{};
        uint32_t _pad1{}; // 32 B
        // 2DGS
        float opacity;
        float3 color{};
        float covX;
        float covY;
        PointType type = QuadricPoint;
        uint32_t _pad2{}; // 32 B
    };

    CHECK_16(OrientedPoint);

    struct alignas(16) BVHNode {
        float3 aabbMin; // 16
        float3 aabbMax; // 32
        uint32_t leftFirst{}; // 36
        uint32_t triCount{}; // 40
        bool isLeaf() const {
            return triCount > 0;
        }
    };

    CHECK_16(BVHNode);

    struct alignas(16) BLASRange {
        uint32_t firstNode{};
        uint32_t nodeCount{};
        uint32_t _pad0{};
        uint32_t _pad1{}; // 16
    };

    CHECK_16(BLASRange);

    struct alignas(16) TLASNode {
        float3 aabbMin; // 16
        float3 aabbMax; // 32
        uint32_t leftChild{}; // 36
        uint32_t count{}; // 40
        uint32_t rightChild{}; // 44
        uint32_t _pad0{}; // 48
    };

    CHECK_16(TLASNode);

    /*************************  Appearance ***************************/
    struct alignas(16) GPUMaterial {
        float3 baseColor{};
        float diffuse{};
        float specular{};
        float phongExp{}; // 16
        float3 emissive{};
    };

    CHECK_16(GPUMaterial);

    /*************************  Transform ****************************/
    struct alignas(16) Transform {
        float4x4 objectToWorld{}; //  64
        float4x4 worldToObject{}; // 128
    };

    CHECK_16(Transform);

    /*************************  Mesh‑area Lights *********************/
    struct alignas(16) MeshLight {
        static constexpr size_t MAX_TRIANGLES = 64;

        float3 v0[MAX_TRIANGLES];
        float3 edge1[MAX_TRIANGLES];
        float3 edge2[MAX_TRIANGLES];
        float3 normal[MAX_TRIANGLES];
        float cdf[MAX_TRIANGLES]{};

        uint32_t triangleCount{0};
        float totalArea{0.f};
        float flux{0.f};
        float radiance{0.f};

        Transform transform{};

        void addTriangle(const float3 &p0,
                         const float3 &e1,
                         const float3 &e2,
                         const float3 &n,
                         float area) {
            if (triangleCount >= MAX_TRIANGLES)
                return; // (host version throws; device just skips)

            v0[triangleCount] = p0;
            edge1[triangleCount] = e1;
            edge2[triangleCount] = e2;
            normal[triangleCount] = n;

            totalArea += area;
            cdf[triangleCount] = totalArea;
            ++triangleCount;
        }

        void finalize() {
            if (totalArea == 0.f) return;
            for (uint32_t i = 0; i < triangleCount; ++i)
                cdf[i] /= totalArea;

            radiance = flux / (std::numbers::pi_v<float> * totalArea);
        }
    };

    CHECK_16(MeshLight);

    /*************************  Scene graph **************************/
    enum class GeometryType : uint32_t { Mesh = 0, PointCloud = 1 };

    struct alignas(16) MeshRange {
        uint32_t firstTri{}, triCount{};
        uint32_t firstVert{}, vertCount{}; // 16
    };

    CHECK_16(MeshRange);

    struct alignas(16) CameraGPU {
        float4x4 view{}; //  64
        float4x4 proj{}; // 128
        float4x4 invView{}; //  64
        float4x4 invProj{}; // 128
        float3 pos{}; // 144
        float3 forward{}; // 160
        uint32_t width{}, height{}; // 168
        uint32_t firstPixel{}; // 172
    };

    CHECK_16(CameraGPU);


    struct GPULightRecord {
        uint32_t lightType;          // 0 = mesh area
        uint32_t geometryIndex;
        uint32_t transformIndex;
        uint32_t triangleOffset;     // into emissiveTriangles[]
        uint32_t triangleCount;
        float3 emissionRgb;    // radiance scale
        float  totalArea;       // sum of areas of this light’s tris
    };

    struct GPUEmissiveTriangle {
        uint32_t globalTriangleIndex; // index into your global triangle pool
    };

    struct InstanceRecord {
        GeometryType geometryType{GeometryType::Mesh};
        uint32_t geometryIndex{0}; // meshRanges index or pointRanges index
        uint32_t materialIndex{0}; // dense index into GPUMaterial array
        uint32_t transformIndex{0}; // index into transforms
        std::string name;
    };


    // UPLOAD CPU-GPU Structures

    struct GPUSceneBuffers {
        BVHNode *blasNodes{nullptr};
        BLASRange *blasRanges{nullptr};
        TLASNode *tlasNodes{nullptr};
        Triangle *triangles{nullptr};
        Vertex *vertices{nullptr};
        Transform *transforms{nullptr};
        GPUMaterial *materials{nullptr};
        InstanceRecord* instances{nullptr};
        uint32_t blasNodeCount{0}, tlasNodeCount{0}, triangleCount{0}, vertexCount{0};

        GPULightRecord*      lights{nullptr};
        GPUEmissiveTriangle* emissiveTriangles{nullptr};
        uint32_t             lightCount{0};
        uint32_t             emissiveTriangleCount{0};

    };

    struct SensorGPU {
        CameraGPU camera; // camera parameters
        sycl::float4 *framebuffer{nullptr}; // device pointer
        uint32_t width{}, height{};
    };

    // ---- PODs ---------------------------------------------------------------
    // ---- Config -------------------------------------------------------------
    enum class RayGenMode : uint32_t { Camera = 0, Emitter = 1, Hybrid = 2, Adjoint = 3 };

    /*************************  Ray & Hit *****************************/
    struct alignas(16) Ray {
        float3 origin; // 16
        float3 direction; // 32
    };

    struct RayState {
        Ray ray;
        float3 pathThroughput;
        uint32_t pixelIndex{};
        uint32_t bounceIndex{};
    };

    struct LocalHit {
        float t;                 // object-space t
        float u;                 // barycentric u
        float v;                 // barycentric v
        uint32_t primitiveIndex; // triangle or prim id within the BLAS geometry
        uint32_t geometryIndex;  // mesh/geometry id within scene
    };

    struct WorldHit {
        float t{};                 // world-space t
        float u{};
        float v{};
        uint32_t primitiveIndex{};
        uint32_t geometryIndex{};
        uint32_t instanceIndex{};
        float3 hitPositionW;
        float3 geometricNormalW; // optional: fill if you have it cheaply
    };

    struct PathTracerSettings {
        uint32_t photonsPerLaunch = 1e4;
        uint64_t randomSeed = 42;
        RayGenMode rayGenMode = RayGenMode::Emitter;
        uint32_t maxBounces = 12;
    };

    struct RenderIntermediatesGPU {
        RayState *primaryRays;
        RayState *extensionRaysA;
        RayState *extensionRaysB;
        WorldHit *hitRecords;
        uint32_t *countPrimary;
        uint32_t *countExtensionOut;
    };

    struct RenderPackage {
        sycl::queue queue;
        PathTracerSettings settings{};
        GPUSceneBuffers scene{};
        RenderIntermediatesGPU intermediates{};
        SensorGPU sensor{};
    };
}
