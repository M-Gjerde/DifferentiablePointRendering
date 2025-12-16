#pragma once

#include <numbers>

#include "entt/entity/entity.hpp"
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
        float3 centroid; // 16 B
    };

    CHECK_16(Triangle);

    enum PointType : uint32_t {
        Gaussian2DPoint,
        QuadricPoint
    };

    struct alignas(16) Point {
        float3 position{0.0f};
        float3 tanU{0.0f};
        float3 tanV{0.0f};
        float2 scale{0.0f};
        float3 albedo{0.0f};
        float opacity{0.0f};
        float beta{0.0f};
        float shape{0.0f};

        uint64_t pointId{0};
    };

    CHECK_16(Point);

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

        bool isEmissive() const {
            return emissive.x() > 0.0f || emissive.y() > 0.0f || emissive.z() > 0.0f;
        }
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
    constexpr uint32_t kInvalidMaterialIndex = 0xFFFFFFFFu;

    enum class GeometryType : uint32_t { Mesh = 0, PointCloud = 1, InvalidType = UINT32_MAX };

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
        float fovy = 60.0f;
        
        char name[16];
    };

    CHECK_16(CameraGPU);


    struct GPULightRecord {
        uint32_t lightType; // 0 = mesh area
        uint32_t geometryIndex;
        uint32_t transformIndex;
        uint32_t triangleOffset; // into emissiveTriangles[]
        uint32_t triangleCount;
        float3 emissionRgb; // radiance scale
        float totalArea; // sum of areas of this light’s tris
    };

    struct AreaLightSample {
        float3 positionW;
        float3 normalW;            // unit
        float3 emittedRadianceRGB; // GPULightRecord::emissionRgb
        float  pdfSelectLight;     // 1 / lightCount
        float  pdfArea;            // 1 / (triangleCount * triArea)
        bool   valid;
    };
    CHECK_16(AreaLightSample);

    struct GPUEmissiveTriangle {
        uint32_t globalTriangleIndex; // index into your global triangle pool
    };

    struct InstanceRecord {
        GeometryType geometryType{GeometryType::InvalidType};
        uint32_t geometryIndex{0}; // meshRanges index or pointRanges index
        uint32_t materialIndex{0}; // mesh only; point cloud = kInvalidMaterialIndex
        uint32_t transformIndex{0}; // index into transforms
        uint32_t blasRangeIndex; // index into bottomLevelRanges of mesh or pointcloud
        char name[16];
    };

    inline void copyName(char (&dst)[16], const std::string &src) {
        std::snprintf(dst, sizeof(dst), "%s", src.c_str()); // always null-terminated
    }

    static_assert(std::is_trivially_copyable_v<InstanceRecord>);
    static_assert(sycl::is_device_copyable<InstanceRecord>::value);

    // UPLOAD CPU-GPU Structures

    struct GPUSceneBuffers {
        BVHNode *blasNodes{nullptr};
        BLASRange *blasRanges{nullptr};
        TLASNode *tlasNodes{nullptr};
        Triangle *triangles{nullptr};
        Vertex *vertices{nullptr};
        Transform *transforms{nullptr};
        GPUMaterial *materials{nullptr};
        Point *points{nullptr};
        InstanceRecord *instances{nullptr};
        uint32_t blasNodeCount{0};
        uint32_t tlasNodeCount{0};
        uint32_t triangleCount{0};
        uint32_t vertexCount{0};
        uint32_t pointCount{0};

        GPULightRecord *lights{nullptr};
        GPUEmissiveTriangle *emissiveTriangles{nullptr};
        uint32_t lightCount{0};
        uint32_t emissiveTriangleCount{0};
    };

    static_assert(std::is_trivially_copyable_v<GPUSceneBuffers>);
    static_assert(sycl::is_device_copyable<GPUSceneBuffers>::value);


    // ---- PODs ---------------------------------------------------------------
    // ---- Config -------------------------------------------------------------
    enum class RayGenMode : uint32_t { Emitter = 1, Adjoint = 3 };

    enum class RayIntersectMode : uint32_t { Random = 0, Transmit = 1, Scatter = 2};

    /*************************  Ray & Hit *****************************/
    struct alignas(16) Ray {
        float3 origin{0.0f}; // 16
        float3 direction{0.0f}; // 32
        float3 normal{0.0f};
    };

    static_assert(std::is_trivially_copyable_v<Ray>);

    struct alignas(16) RayState {
        Ray ray{};
        float3 pathThroughput{0.0f};
        uint32_t bounceIndex{0};
        uint32_t pixelIndex = UINT32_MAX; // NEW: source pixel that launched this adjoint path
        uint32_t pixelX;
        uint32_t pixelY;
    };

    static_assert(std::is_trivially_copyable_v<RayState>);


    constexpr int kMaxSplatEvents = 88;

    struct SplatEvent {
        float t = FLT_MAX; // local space t
        float3 hitWorld = float3(0.0f); // World hit position
        float alpha = 1.0f;
        float tau = 1.0f;
        uint32_t primitiveIndex = UINT32_MAX; // primitiveIndex
    };

    struct alignas(16) LocalHit {
        float t = FLT_MAX; // world-space t
        float transmissivity = FLT_MAX;
        uint32_t primitiveIndex = UINT32_MAX; // triangle or prim id within the BLAS geometry
        uint32_t geometryIndex = UINT32_MAX; // mesh/geometry id within scene

        SplatEvent splatEvents[kMaxSplatEvents];
        int splatEventCount = 0;
    };

    static_assert(std::is_trivially_copyable_v<LocalHit>);


    struct alignas(16) WorldHit {
        bool hit = false;
        float t = FLT_MAX; // world-space t
        float transmissivity = 1.0f;
        // 0.0 = No transmission. 1.0 Full transmission (I.e. default until we interact with someething)
        uint32_t primitiveIndex = UINT32_MAX;
        uint32_t instanceIndex = UINT32_MAX;
        float3 hitPositionW = float3(0.0f);
        float3 geometricNormalW = float3(0.0f);; // optional: fill if you have it cheaply

        SplatEvent splatEvents[kMaxSplatEvents];
        int splatEventCount = 0;
    };

    static_assert(std::is_trivially_copyable_v<WorldHit>);

    struct alignas(16) PathTracerSettings {
        uint32_t photonsPerLaunch = 1e6;
        uint64_t randomSeed = 42; // should be more than maxBounces
        RayGenMode rayGenMode = RayGenMode::Emitter;
        uint32_t maxBounces = 6;
        uint32_t numForwardPasses = 6;
        uint32_t numGatherPasses = 6; // Which bounce to start RR
        uint32_t maxAdjointBounces = 6;
        uint32_t adjointSamplesPerPixel = 6;
        uint32_t russianRouletteStart = 3; // Which bounce to start RR

        bool renderDebugGradientImages = false;
    };

    static_assert(std::is_trivially_copyable_v<PathTracerSettings>);
    static_assert(sycl::is_device_copyable<PathTracerSettings>::value);

    // -------------------- Photon storage (device) --------------------------
    // Filled during the emitter pass by appending at an atomic counter.
    // One entry per stored photon (only diffuse hits).
    struct alignas(16) DevicePhotonSurface {
        // Positions in world space
        float3 position{0.0f};

        // Photon power (throughput × emission), RGB channels
        float3 power{0.0f};

        // |n · ω_i| at the hit (used to convert flux→irradiance)
        float cosineIncident = 0.0f;
        int    sideSign{};       // +1 or -1: hemisphere relative to canonical surfel normal
        GeometryType geometryType{GeometryType::InvalidType};       // +1 or -1: hemisphere relative to canonical surfel normal
        uint32_t primitiveIndex = UINT32_MAX;

    };

    static_assert(std::is_trivially_copyable_v<DevicePhotonSurface>);

    // ----------------- Full surface photon map handle (device) -------------------
    struct alignas(16) DeviceSurfacePhotonMapGrid {
        DevicePhotonSurface *photons; // [photonCapacity]
        uint32_t *photonCountDevicePtr; // atomic append counter on device
        uint32_t photonCapacity; // total allocated slots

        // Grid params
        float3 gridOriginWorld;
        float3 cellSizeWorld; // set = gatherRadiusWorld
        sycl::int3 gridResolution; // Nx,Ny,Nz
        uint32_t totalCellCount;

        // Per-cell lists
        uint32_t *cellHeadIndexArray; // [totalCellCount], init to kInvalidIndex
        uint32_t *photonNextIndexArray; // [photonCapacity]

        // For clarity
        float gatherRadiusWorld = 0.0f;
        float kappa = 2.0f;
    };

    static_assert(std::is_trivially_copyable_v<DeviceSurfacePhotonMapGrid>);




    struct alignas(16) RenderIntermediatesGPU {
        RayState *primaryRays;
        RayState *extensionRaysA;
        WorldHit *hitRecords;
        uint32_t *countPrimary;
        uint32_t *countExtensionOut;

        DeviceSurfacePhotonMapGrid map;
    };

    static_assert(std::is_trivially_copyable_v<RenderIntermediatesGPU>);
    static_assert(sycl::is_device_copyable<RenderIntermediatesGPU>::value);


}
