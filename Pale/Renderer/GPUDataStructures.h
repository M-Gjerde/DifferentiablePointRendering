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

    // GPU Struct
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
        float flux{0.0f};

        uint64_t pointId{0};

        bool isEmissive() const {
            return flux > 0.0f;
        };
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
    };

    CHECK_16(BLASRange);

    struct alignas(16) TLASNode {
        float3 aabbMin; // 16
        float3 aabbMax; // 32
        uint32_t leftChild{}; // 36
        uint32_t count{}; // 40
        uint32_t rightChild{}; // 44
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


    enum class LightType : uint32_t { Mesh = 0, Surfel = 1 };

    struct GPULightRecord {
        LightType lightType; // 0 = mesh area
        uint32_t geometryIndex;
        uint32_t transformIndex;
        uint32_t materialIndex;
        uint32_t triangleOffset; // into emissiveTriangles[]
        uint32_t triangleCount;
        float3 color; // lght color
        float flux;
        float totalAreaWorld; // sum of worldArea of its triangles

        // Surfel
        uint32_t primitiveIndex;
    };


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
        uint32_t *pointPermutation{nullptr};

        uint32_t blasNodeCount{0};
        uint32_t tlasNodeCount{0};
        uint32_t triangleCount{0};
        uint32_t vertexCount{0};
        uint32_t pointCount{0};
        uint32_t pointPermutationCount{0};

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

    enum class SurfelIntersectMode : uint32_t { Bernoulli = 0, Transmit = 1, FirstHit = 2, Uniform = 3 };

    enum class EventType : uint32_t { Null = 0, Reflect = 1, Transmit = 2, Absorb = 3, TransmittanceGradient = 4 };

    // Maximum expected per-ray surfel intersections.
    // Must be compile-time constant for stack arrays in SYCL device code.
    constexpr uint32_t kMaxSplatEventsPerRay = 24;
    constexpr float RayEpsilon = 1e-6f;
    constexpr float RayEpsilon2 = 1e-6f;
    constexpr uint32_t kInvalidMaterialIndex = 0xFFFFFFFFu;
    static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;
    constexpr uint32_t kMaxLocalSurfelHits = 8;
    constexpr float LocalLayerDepthEpsilon = 5.0e-2f;

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
        float transmission = 1.0f;
        uint32_t bounceIndex{0};
        uint32_t traversalIndex{0};
        float openSegmentProposalInverse = 1.0f;
        uint32_t pixelIndex = UINT32_MAX; // NEW: source pixel that launched this adjoint path
        uint32_t lightIndex = UINT32_MAX;
        uint32_t hasTrackedParameter = UINT32_MAX;
        uint32_t pathId;
    };

    static_assert(std::is_trivially_copyable_v<RayState>);

    struct LocalSurfelLayerHit {
        float tWorld;
        uint32_t primitiveIndex;
        float alphaGeom;
        float3 hitPositionW;
    };

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
        float transmissivity = 1.0f;
        // 0.0 = No transmission. 1.0 Full transmission (I.e. default until we interact with someething)
        float alphaGeom = 0.0f;
        uint32_t primitiveIndex = UINT32_MAX;
        uint32_t instanceIndex = UINT32_MAX;
        float3 hitPositionW = float3(0.0f);
        float3 geometricNormalW = float3(0.0f);
        uint32_t invChosenSurfelPdf = 0; //chosen surfel PDF for adjoint pass
    };

    struct alignas(16) HitInfoContribution {
        float3 pathThroughput = float3(0.0f);
        float3 hitPositionW = float3(0.0f);
        float3 geometricNormalW = float3(0.0f);
        uint32_t primitiveIndex = UINT32_MAX;
        uint32_t instanceIndex = UINT32_MAX;
        GeometryType type = GeometryType::InvalidType;
        EventType eventType = EventType::Absorb;
        uint32_t pixelIndex;
        uint32_t rayIndex;
        float alphaGeom;
    };

    enum class PendingAdjointKind : uint8_t {
        None = 0,
        // Light transport
        NullTransmittance,
        ReflectScatter,
        TransmitScatter,
        // Projection states
        ProjectionScatter,
        Projection,
    };

    enum class SampledPointEventType : uint32_t {
        None = 0u,
        Null = 1u,
        Reflect = 2u,
        Transmit = 3u
    };

    struct PendingCameraSegment {
        bool valid = false;
        uint32_t pathId = 0u;
        uint32_t pixelIndex = UINT32_MAX;
        float3 cameraPathThroughput{FLT_MAX, FLT_MAX, FLT_MAX};
        float3 cameraOriginWorld{FLT_MAX, FLT_MAX, FLT_MAX};
        float3 cameraDirectionWorld{FLT_MAX, FLT_MAX, FLT_MAX};
    };

    struct PointCloudSurfaceRecord {
        uint32_t primitiveIndex = UINT32_MAX;
        float2 uv = float2{FLT_MAX, FLT_MAX};
        float alphaGeom = FLT_MAX;
        int32_t sideSign = 1;
        float3 incomingDirection = float3{FLT_MAX, FLT_MAX, FLT_MAX};
        uint32_t pathId = UINT32_MAX;
    };


    struct AreaLightSample {
        float3 positionW;
        float3 normalW; // unit
        float3 direction;
        float3 flux;
        uint32_t lightIndex;
        float pdfSelectLight; // 1 / lightCount
        float pdfDir;
        float pdfLocalCoordsSample;
        float pdfArea; // 1 / (triangleCount * triArea)
        float totalAreaWorld;
        bool valid;

        PointCloudSurfaceRecord surface;
        float lightJacobian;
    };

    struct PointLightSample {
        float3 positionW;
        float3 normalW; // unit
        float3 direction;
        float3 flux;
        uint32_t lightIndex;
        float pdfSelectLight; // 1 / lightCount
        float pdfDir;
        bool valid;

        PointCloudSurfaceRecord surface;
        float lightJacobian;
    };

    CHECK_16(AreaLightSample);

    struct DirectLightQuery {
        PointCloudSurfaceRecord surface{};

        // Adjoint weight transported to the current surface before local direct-light evaluation.
        float3 adjointWeight{0.0f, 0.0f, 0.0f};

        // Open-segment transmission on the incoming path up to this surface.
        float transmissionToSurface = 1.0f;

        // Local scattering factor at the current surface.
        float3 localBsdf{0.0f, 0.0f, 0.0f};

        // For non-Lambertian extensions. For diffuse this is not strictly needed,
        // but it is useful to keep.
        float3 outgoingDirectionWorld{0.0f, 0.0f, 0.0f};

        // Sampled emitter point.
        float3 lightPositionWorld{0.0f, 0.0f, 0.0f};
        float3 lightNormalWorld{0.0f, 0.0f, 0.0f};
        float3 lightRadiance{0.0f, 0.0f, 0.0f};
        float lightPdfArea = 1.0f;

        uint32_t pixelIndex = 0u;
        uint32_t pathId = 0u;
        uint32_t bounceIndex = 0u;
    };

    struct DirectLightGradientEvent {
        PointCloudSurfaceRecord surface{};

        // Prefix weight up to the current surface x.
        float3 xPathThroughput{0.0f, 0.0f, 0.0f};
        float3 localBsdf{0.0f, 0.0f, 0.0f};

        // Surface-to-light sample.
        float3 lightPositionWorld{0.0f, 0.0f, 0.0f};
        float3 lightNormalWorld{0.0f, 0.0f, 0.0f};
        float3 lightRadiance{0.0f, 0.0f, 0.0f};
        float lightPdfArea = 1.0f;

        // Prefix transmission up to x, kept separate for readability if you want it.
        float transmissionToSurface = 1.0f;

        // Optional, currently binary and not differentiated.
        float visibility = 1.0f;
        bool useImplicitRayHitJacobian = false;
    };

    struct PendingAdjointVertex {
        PointCloudSurfaceRecord surface{};

        uint32_t bounceIndex = 0u;

        // Throughput stored before the reflect-sampling factor of the current vertex
        // is applied in the next state update. This preserves your current convention.
        float3 pathThroughput = float3{FLT_MAX, FLT_MAX, FLT_MAX};

        // Segment metadata for the segment arriving at this vertex from the previous one.
        float transmission = FLT_MAX;
        float geometryFromPrevious = FLT_MAX;
        float areaPdfFromPrevious = FLT_MAX;

        // Local scattering factor stored at this vertex.
        float3 bsdf = float3{FLT_MAX, FLT_MAX, FLT_MAX};
        float alpha = FLT_MAX;

        // Only used for the camera-attached path case.
        float cosineFromPrevious = FLT_MAX;
    };

    struct PendingAdjointStageX {
        bool valid = false;
        uint32_t pathId = 0u;
        uint32_t pixelIndex = UINT32_MAX;

        // Whether the CURRENT stored vertex was produced from the camera-attached
        // branch that uses the implicit ray-hit Jacobian convention.
        bool useImplicitRayHitJacobian = false;

        // Rolling two-vertex history:
        // previous = X, current = Y, live hit = Z
        bool hasPrevious = false;
        PendingAdjointVertex previous{};
        PendingAdjointVertex current{};
    };

    struct MeasurementGradientEvent {
        PointCloudSurfaceRecord xSurface;
        float transmission{};
        float3 xPathThroughput;
        bool useImplicitRayHitJacobian = false;
    };

    struct OccluderDerivative {
        float3 gradPosition{0.0f, 0.0f, 0.0f};
        float gradScaleU = 0.0f;
        float gradScaleV = 0.0f;
        float gradEta = 0.0f;
        float gradBeta = 0.0f;
        float3 gradRotation{0.0f, 0.0f, 0.0f};

        float3 gradAlphaWrtStartPoint{0.0f, 0.0f, 0.0f};
        float3 gradAlphaWrtEndPoint{0.0f, 0.0f, 0.0f};

        float prefixTransmittance = 1.0f;
        float oneMinusAlpha = 1.0f;
        uint32_t primitiveIndex = kInvalidIndex;
    };

    struct MeasurementGradientEventXY {
        PointCloudSurfaceRecord xSurface;
        PointCloudSurfaceRecord ySurface;
        float3 xPathThroughput = float3{FLT_MAX, FLT_MAX, FLT_MAX};
        float transmission = 1.0f;
        float transmissionPreviousSegment = FLT_MAX;
        float geometryPreviousSegment = FLT_MAX;
        float cosinePreviousSegment = FLT_MAX;
        bool useImplicitRayHitJacobian = false;
        bool isDirectLightSample = false;
        float3 directLightRadiance{FLT_MAX, FLT_MAX, FLT_MAX};

        // Used only when isDirectLightSample == true.
        float3 pointLightPositionW{FLT_MAX, FLT_MAX, FLT_MAX};
        float3 pointLightRadiantIntensity{FLT_MAX, FLT_MAX, FLT_MAX};

    };

    struct MaterialVertexGradientEvent {
        PointCloudSurfaceRecord surface;
        float3 adjointWeightAtVertex{0.0f, 0.0f, 0.0f};
        uint32_t pathId = kInvalidIndex;
        uint32_t bounceIndex = 0u;
    };


    struct MaterialEdgeGradientEvent {
        PointCloudSurfaceRecord startSurface{};
        PointCloudSurfaceRecord endSurface{};
        float3 sampledEdgeThroughput{FLT_MAX, FLT_MAX, FLT_MAX};
        float3 betaIncrement{FLT_MAX, FLT_MAX, FLT_MAX};
        float3 bsdf{FLT_MAX, FLT_MAX, FLT_MAX};
        float alpha = FLT_MAX;
        float invSamplePDF = FLT_MAX;
        float segmentTransmittance = 1.0f;
        float segmentGeometricTerm = 1.0f;
        float segmentAreaPdf = 1.0f;
        float3 directLightRadiance{0.0f, 0.0f, 0.0f};
        bool isDirectLightSample = false;
        bool writeOcclusionGradients = true; // True only for the first time as we handled camera rays spearately
        uint32_t pathId = kInvalidIndex;
        uint32_t startBounceIndex = 0u;
    };

    struct ReconstructedSurfelState {
        float3 position = float3{FLT_MAX, FLT_MAX, FLT_MAX};
        float3 canonicalNormal = float3{FLT_MAX, FLT_MAX, FLT_MAX};
        float3 orientedNormal = float3{FLT_MAX, FLT_MAX, FLT_MAX};
        float3 tangentUWorld = float3{FLT_MAX, FLT_MAX, FLT_MAX};
        float3 tangentVWorld = float3{FLT_MAX, FLT_MAX, FLT_MAX};
        float areaWorld = FLT_MAX;
    };

    struct SurfelGradientRecord {
        uint32_t primitiveIndex = UINT32_MAX;

        float gradBeta = FLT_MAX;
        float gradEta = FLT_MAX;

        float gradAlbedoR = FLT_MAX;
        float gradAlbedoG = FLT_MAX;
        float gradAlbedoB = FLT_MAX;

        float gradPositionX = FLT_MAX;
        float gradPositionY = FLT_MAX;
        float gradPositionZ = FLT_MAX;

        float gradScaleU = FLT_MAX;
        float gradScaleV = FLT_MAX;

        float gradRotationX = FLT_MAX;
        float gradRotationY = FLT_MAX;
        float gradRotationZ = FLT_MAX;
    };


    struct GradientRecordRanges {
        uint32_t measurementOffset = 0u;
        uint32_t measurementCount = 0u;

        uint32_t measurementTwoPointOffset = 0u;
        uint32_t measurementTwoPointCount = 0u;

        uint32_t materialVertexOffset = 0u;
        uint32_t materialVertexCount = 0u;

        uint32_t materialEndEdgeOffset = 0u;
        uint32_t materialEndEdgeCount = 0u;

        uint32_t materialStartEdgeOffset = 0u;
        uint32_t materialStartEdgeCount = 0u;

        uint32_t totalCount = 0u;
    };


    struct CompletedGradientEvent {
        bool valid = false;

        uint32_t pathId = UINT32_MAX;
        uint32_t pixelIndex = UINT32_MAX;
        PendingAdjointKind kind = PendingAdjointKind::ReflectScatter;

        // X = first surfel
        uint32_t xInstanceIndex = UINT32_MAX;
        uint32_t xPrimitiveIndex = UINT32_MAX;
        GeometryType xGeometryType = GeometryType::InvalidType;
        float xAlphaGeom = 0.0f;
        float xCosine = 0.0f;
        float3 xPosition{0.0f};
        float3 xNormal{0.0f};
        Ray xIncomingRay{};
        float3 xPathThroughput{0.0f};

        // Y = second surfel
        uint32_t yInstanceIndex = UINT32_MAX;
        uint32_t yPrimitiveIndex = UINT32_MAX;
        GeometryType yGeometryType = GeometryType::InvalidType;
        float yAlphaGeom = 0.0f;
        float yCosine = 0.0f;
        float3 yPosition{0.0f};
        float3 yNormal{0.0f};
        Ray yIncomingRay{};
        float3 yPathThroughput{0.0f};
        // Z = final mesh hit
        uint32_t zInstanceIndex = UINT32_MAX;
        uint32_t zPrimitiveIndex = UINT32_MAX;
        GeometryType zGeometryType = GeometryType::InvalidType;
        float zAlphaGeom = 0.0f;
        float zCosine = 0.0f;
        float3 zPosition{0.0f};
        float3 zNormal{0.0f};
        Ray zIncomingRay{};
        float3 zPathThroughput{0.0f};
    };


    struct alignas(16) HitTransmittanceContribution {
        float3 pathThroughput = float3(0.0f);
        float3 hitPositionSurfel = float3(0.0f);
        float3 hitPositionEnd = float3(0.0f);
        float3 geometricNormalEndW = float3(0.0f);
        uint32_t primitiveIndex = UINT32_MAX;
        uint32_t instanceIndex = UINT32_MAX;
        uint32_t pixelIndex;
        uint32_t rayIndex;
        float alphaGeom;
        bool valid = false;
        bool validEnd = false;
        float cosine = 0.0f;
    };

    static_assert(std::is_trivially_copyable_v<WorldHit>);

    enum class IntegratorKind : uint32_t {
        lightTracing = 0x0001,
        lightTracingCylinderRay = 0x0002,
        photonMapping = 0x0004
    };

    struct Random {
        uint64_t seed = 42; // should be more than maxBounces
        uint32_t number = 42; // should be more than maxBounces
    };

    struct AdjointSampleSettings {
        float qNull = 0.3f;
        float qReflect = 0.7f;
        float qTransmit = 0.0f;
        float qAbsorb = 1.0f - qNull - qReflect - qTransmit;
    };

    struct alignas(16) PathTracerSettings {
        IntegratorKind integratorKind = IntegratorKind::photonMapping;
        uint32_t photonsPerLaunch = 1e6;
        Random random{}; // should be more than maxBounces
        RayGenMode rayGenMode = RayGenMode::Emitter;
        uint32_t maxBounces = 6;
        uint32_t numForwardPasses = 6;
        uint32_t maxAdjointBounces = 6;
        uint32_t adjointSamplesPerPixel = 6;
        uint32_t russianRouletteStart = 12; // Which bounce to start RR
        uint32_t numShadowRays = 8;
        uint32_t numGatherPasses = 1;
        uint32_t numAdjointShadowRays = 8;
        bool renderDebugGradientImages = false;
        uint32_t surfelIndexForDebugImages = 1;
        float depthDistortionWeight = 0.0f;
        float normalConsistencyWeight = 0.0f;
        float visibilityWeightedOpacityRegularizerWeight = 0.0f;
        AdjointSampleSettings sampling;
        bool enableAdjointDirectLight = false;
        uint32_t numAdjointPathShadowRays = 1;

        // Cylinder ray:
        // EGWR 2000 point-sampled geometry debug renderer.
        float pointGeometrySupportRadius = 0.002f;
        float pointGeometryReconstructionLength = 0.04f;
        float pointGeometryRayOffsetMultiplier = 2.0f;
        float pointGeometryCoverageScale = 1.1f;
        uint32_t pointGeometryMinimumContributors = 1u;
        bool pointGeometryDebugShowAlbedo = false;
    };

    static_assert(std::is_trivially_copyable_v<PathTracerSettings>);
    static_assert(sycl::is_device_copyable<PathTracerSettings>::value);
    // -------------------- Photon storage (device) --------------------------
    // Filled during the emitter pass by appending at an atomic counter.
    // One entry per stored photon (only diffuse hits).
    struct DevicePhotonSurface {
        // Positions in world space
        float3 position{0.0f};
        // Photon power (throughput × emission), RGB channels
        float3 flux{0.0f};
        float3 incomingDirection{0.0f};
        //float3 normal{0.0f};
        // |n · ω_i| at the hit (used to convert flux→irradiance)
        //int sideSign{}; // +1 or -1: hemisphere relative to canonical surfel normal
        //GeometryType geometryType{GeometryType::InvalidType};
        //float3 incomingDirection{0.0f};
        std::uint32_t isValid = 0;
    };

    static_assert(std::is_trivially_copyable_v<DevicePhotonSurface>);

    // ----------------- Full surface photon map handle (device) -------------------
    struct DeviceSurfacePhotonMapGrid {
        float minimumGatherRadiusWorld = 0.00f;
        float maximumGatherRadiusWorld = 0.00f;
        float gatherPadWorld = 0.00f;
        float3 cellSizeWorld = float3{0};
        float3 gridOriginWorld = float3{0};
        sycl::int3 gridResolution = sycl::int3{0};
        std::uint32_t totalCellCount = 0;

        // Photon storage (written during emission)
        DevicePhotonSurface *photons = nullptr;
        std::uint32_t photonCapacity = 0;
        std::uint32_t *photonCountDevicePtr = nullptr;
        std::uint32_t *photonStreamCountDevicePtr = nullptr;
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

        HitInfoContribution *hitContribution;
        uint32_t maxHitContributionCount = 0;
        uint32_t *countContributions;

        PendingAdjointStageX *pendingStageX = nullptr;
        uint32_t maxPendingAdjointStateCount = 0;

        MeasurementGradientEvent *measurementEvents;
        MeasurementGradientEventXY *measurementTwoPointEvents = nullptr;

        MaterialVertexGradientEvent *materialVertexEvents = nullptr;
        MaterialEdgeGradientEvent *materialEndEdgeEvents = nullptr;
        MaterialEdgeGradientEvent *materialStartEdgeEvents = nullptr;

        SurfelGradientRecord *gradientRecords = nullptr;
        PendingCameraSegment *pendingCameraSegments = nullptr;

        uint32_t *countMeasurementEvents = nullptr;
        uint32_t *countMeasurementTwoPointEvents = nullptr;
        uint32_t *countMaterialVertexEvents = nullptr;
        uint32_t *countMaterialEndEdgeEvents = nullptr;
        uint32_t *countMaterialStartEdgeEvents = nullptr;

        // capacities
        uint32_t maxMeasurementEventCount = 0u;
        uint32_t maxMeasurementTwoPointEventCount = 0u;
        uint32_t maxMaterialVertexEventCount = 0u;
        uint32_t maxMaterialEndEdgeEventCount = 0u;
        uint32_t maxMaterialStartEdgeEventCount = 0u;
        uint32_t maxGradientRecordCount = 0;
        uint32_t maxRayQueueCapacity = 0;

        uint32_t *countPrimary;
        uint32_t *countExtensionOut;
        DeviceSurfacePhotonMapGrid map;
    };

    static_assert(std::is_trivially_copyable_v<RenderIntermediatesGPU>);
    static_assert(sycl::is_device_copyable<RenderIntermediatesGPU>::value);
}
