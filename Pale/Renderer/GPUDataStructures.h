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
        float alpha_t{0.0f};
        float alpha_r{0.0f};
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
        float power{};
        float diffuse{};
        float specular{};
        float phongExp{}; // 16

        bool isEmissive() const {
            return power > 0.f;
        }
    };

    CHECK_16(GPUMaterial);

    /*************************  Transform ****************************/
    struct alignas(16) Transform {
        float4x4 objectToWorld{}; //  64
        float4x4 worldToObject{}; // 128
    };

    CHECK_16(Transform);

    /*************************  Scene graph **************************/
    constexpr uint32_t kInvalidMaterialIndex = 0xFFFFFFFFu;
    static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;

    enum class GeometryType : uint32_t { Mesh = 0, PointCloud = 1, InvalidType = UINT32_MAX };

    struct alignas(16) MeshRange {
        uint32_t firstTri{}, triCount{};
        uint32_t firstVert{}, vertCount{}; // 16
    };

    CHECK_16(MeshRange);

    struct alignas(16) CameraGPU {
        float4x4 view{};
        float4x4 proj{};
        float4x4 invView{};
        float4x4 invProj{};

        float3 pos{};
        uint32_t width = 0;

        float3 forward{};
        uint32_t height = 0;

        // Legacy (keep for debug / fallback)
        float fovy = 60.0f; // degrees

        // New: pinhole intrinsics in pixels
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;

        // Flags
        uint32_t hasPinholeIntrinsics = 0; // 0/1
        uint32_t useForAdjointPass = 1; // 0/1

        char name[16]{};
    };

    CHECK_16(CameraGPU);


    struct GPULightRecord {
        uint32_t lightType; // 0 = mesh area
        uint32_t geometryIndex;
        uint32_t transformIndex;
        uint32_t triangleOffset; // into emissiveTriangles[]
        uint32_t triangleCount;
        float3 color; // lght color
        float power;
        float totalAreaWorld; // sum of worldArea of its triangles
    };

    struct AreaLightSample {
        float3 positionW;
        float3 normalW; // unit
        float3 direction;
        float3 power;
        float lightIndex;
        float pdfSelectLight; // 1 / lightCount
        float pdfDir;
        float pdfArea; // 1 / (triangleCount * triArea)
        bool valid;
    };

    CHECK_16(AreaLightSample);

    struct GPUEmissiveTriangle {
        uint32_t globalTriangleIndex;
        float worldArea; // triangle area after transform
        float cdf; // inclusive CDF in [0,1] within its light’s triangle range
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

    enum class SurfelIntersectMode : uint32_t { Bernoulli = 0, Transmit = 1, FirstHit = 2 , Uniform = 3 };

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
        uint32_t lightIndex = UINT32_MAX;
    };

    static_assert(std::is_trivially_copyable_v<RayState>);


    // Maximum expected per-ray surfel intersections.
    // Must be compile-time constant for stack arrays in SYCL device code.
    constexpr int kMaxSplatEventsPerRay = 16;


    struct SurfelEvent {
        float t = FLT_MAX; // local space t
        float alphaGeom = 1.0f;
        float transmissivity = 1.0f;
        uint32_t primitiveIndex = UINT32_MAX; // primitiveIndex
    };

    struct alignas(16) LocalHit {
        float3 worldHit{0.0f};
        float t = FLT_MAX; // world-space t
        float transmissivity = FLT_MAX;
        float alpha = 0.0f;
        uint32_t primitiveIndex = UINT32_MAX; // triangle or prim id within the BLAS geometry
        uint32_t geometryIndex = UINT32_MAX; // mesh/geometry id within scene

        uint32_t invChosenSurfelPdf = 0; // Used only for adjoint pass
    };

    static_assert(std::is_trivially_copyable_v<LocalHit>);

    struct alignas(16) WorldHit {
        bool hit = false;
        bool hitSurfel = false;
        GeometryType type = GeometryType::InvalidType;
        float t = FLT_MAX; // world-space t
        float transmissivity = 1.0f;         // 0.0 = No transmission. 1.0 Full transmission (I.e. default until we interact with someething)
        float alpha = 0.0f;
        uint32_t primitiveIndex = UINT32_MAX;
        uint32_t instanceIndex = UINT32_MAX;
        float3 hitPositionW = float3(0.0f);
        float3 geometricNormalW = float3(0.0f);
        uint32_t invChosenSurfelPdf = 0; //chosen surfel PDF for adjoint pass
    };

    static_assert(std::is_trivially_copyable_v<WorldHit>);

    enum class IntegratorKind : uint32_t {
        lightTracing,
        lightTracingCylinderRay,
        photonMapping
    };

    struct Random {
        uint64_t seed = 42; // should be more than maxBounces
        uint32_t number = 42; // should be more than maxBounces
    };

    struct alignas(16) PathTracerSettings {
        IntegratorKind integratorKind = IntegratorKind::photonMapping;
        uint32_t photonsPerLaunch = 1e6;
        Random random{}; // should be more than maxBounces
        RayGenMode rayGenMode = RayGenMode::Emitter;
        uint32_t maxBounces = 6;
        uint32_t numForwardPasses = 6;
        uint32_t numGatherPasses = 6; // Which bounce to start RR
        uint32_t maxAdjointBounces = 6;
        uint32_t adjointSamplesPerPixel = 6;
        uint32_t russianRouletteStart = 3; // Which bounce to start RR
        bool renderDebugGradientImages = false;
        float depthDistortionWeight = 0.0f;
        float normalConsistencyWeight = 0.0f;
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
        //float3 normal{0.0f};

        // |n · ω_i| at the hit (used to convert flux→irradiance)
        //int sideSign{}; // +1 or -1: hemisphere relative to canonical surfel normal
        //GeometryType geometryType{GeometryType::InvalidType};
        float3 incomingDirection{0.0f};

        std::uint32_t isValid = 0;
    };

    static_assert(std::is_trivially_copyable_v<DevicePhotonSurface>);

    // ----------------- Full surface photon map handle (device) -------------------
    struct DeviceSurfacePhotonMapGrid {
        float gatherRadiusWorld = 0.00f;
        float3 cellSizeWorld = float3{0};
        float3 gridOriginWorld = float3{0};
        sycl::int3 gridResolution = sycl::int3{0};
        std::uint32_t totalCellCount = 0;

        // Photon storage (written during emission)
        DevicePhotonSurface *photons = nullptr;
        std::uint32_t photonCapacity = 0;
        std::uint32_t *photonCountDevicePtr = nullptr;

        std::uint32_t allocatedCellCount = 0;
        std::uint32_t allocatedPhotonCapacity = 0;
        std::uint32_t allocatedBlockCount = 0;
        // Per-photon build buffers
        std::uint32_t *photonCellId = nullptr; // [photonCapacity]
        std::uint32_t *photonIndex = nullptr; // [photonCapacity] optional if you scatter into sortedPhotonIndex
        std::uint32_t *sortedPhotonIndex = nullptr; // [photonCapacity]

        // Per-cell build buffers
        std::uint32_t *cellStart = nullptr; // [totalCellCount]
        std::uint32_t *cellEnd = nullptr; // [totalCellCount]
        std::uint32_t *cellCount = nullptr; // [totalCellCount]
        std::uint32_t *cellWriteOffset = nullptr; // [totalCellCount]

        // Scan temporaries
        std::uint32_t *blockSums = nullptr; // [numBlocks]
        std::uint32_t *blockPrefix = nullptr; // [numBlocks] (optional; can reuse blockSums if you overwrite carefully)
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
