//
// Created by magnus on 9/8/25.
//

#include "Renderer/Kernels/AdjointKernels.h"

#include "IntersectionKernels.h"
#include "Renderer/Kernels/KernelHelpers.h"
#include "Renderer/Kernels/AdjointIntersectionKernels.h"


namespace Pale {
    void launchRayGenAdjointKernel(RenderPackage &pkg, int spp) {
        auto &queue = pkg.queue;
        auto &sensor = pkg.sensor;
        auto &settings = pkg.settings;
        auto &adjoint = pkg.adjoint;
        auto &intermediates = pkg.intermediates;

        const uint32_t imageWidth = sensor.camera.width;
        const uint32_t imageHeight = sensor.camera.height;

        uint32_t raysPerSet = imageWidth * imageHeight;

        //raysPerSet = 10;

        const uint32_t perPassRayCount = raysPerSet;
        queue.memcpy(pkg.intermediates.countPrimary, &perPassRayCount, sizeof(uint32_t)).wait();

        queue.submit([&](sycl::handler &commandGroupHandler) {
            const uint64_t baseSeed = settings.randomSeed * static_cast<uint64_t>(spp);

            commandGroupHandler.parallel_for<struct RayGenAdjointKernelTag>(
                sycl::range<1>(raysPerSet),
                [=](sycl::id<1> globalId) {
                    const auto globalRayIndex = static_cast<uint32_t>(globalId[0]);
                    // Map to pixel
                    const uint32_t pixelLinearIndexWithinImage = globalRayIndex; // 0..raysPerSet-1
                    uint32_t pixelX = pixelLinearIndexWithinImage % imageWidth;
                    uint32_t pixelY = pixelLinearIndexWithinImage / imageWidth;

                    //pixelX = 301;
                    //pixelY = imageHeight - 1 - 932;


                    const uint32_t pixelIndex = pixelLinearIndexWithinImage;
                    // RNG for this pixel
                    const uint64_t perPixelSeed = rng::makePerItemSeed1D(baseSeed, pixelLinearIndexWithinImage);
                    rng::Xorshift128 pixelRng(perPixelSeed);

                    // Adjoint source weight
                    const float4 residualRgba = adjoint.framebuffer[pixelIndex];
                    float3 initialAdjointWeight = {residualRgba.x(), residualRgba.y(), residualRgba.z()};
                    // Or unit weights:
                    initialAdjointWeight = float3(1.0f, 1.0f, 1.0f);

                    // Base slot for this pixel’s N samples
                    const uint32_t baseOutputSlot = pixelIndex;

                    // --- Sample 0: forced Transmit (background path) ---
                    const float jitterX = pixelRng.nextFloat() - 0.5f;
                    const float jitterY = pixelRng.nextFloat() - 0.5f;

                    Ray primaryRay = makePrimaryRayFromPixelJittered(
                        sensor.camera,
                        static_cast<float>(pixelX),
                        static_cast<float>(pixelY),
                        jitterX, jitterY
                    );

                    //primaryRay.direction = normalize(float3{-0.001, 0.982122211, 0.277827293});    // a
                    primaryRay.direction = normalize(float3{-0.127575308, 0.952122211, -0.277827293}); // b
                    primaryRay.origin = float3{0.0, -4.0, 1.0};

                    RayState rayState{};

                    rayState.ray = primaryRay;
                    rayState.pathThroughput = initialAdjointWeight;
                    rayState.bounceIndex = 0;
                    rayState.pixelIndex = pixelIndex;

                    intermediates.primaryRays[baseOutputSlot] = rayState;
                });
        }).wait();
    }


    void launchAdjointKernel(RenderPackage &pkg, uint32_t activeRayCount) {
        auto &queue = pkg.queue;
        auto &scene = pkg.scene;
        auto &sensor = pkg.sensor;
        auto &settings = pkg.settings;
        auto &adjoint = pkg.adjoint;
        auto &photonMap = pkg.intermediates.map;

        auto *hitRecords = pkg.intermediates.hitRecords;
        auto *raysIn = pkg.intermediates.primaryRays;
        // activeRayCount == total rays in the current buffer (must be even)

        const uint32_t perPassRayCount = activeRayCount; // total number of rays (With n-samples per ray)
        const uint32_t perPixelRayCount = activeRayCount; // Number of rays per pixel
        const uint32_t photonsPerLaunch = settings.photonsPerLaunch;

        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<struct AdjointShadeKernelTag>(
                sycl::range<1>(perPixelRayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t perItemSeed = rng::makePerItemSeed1D(settings.randomSeed, rayIndex);
                    rng::Xorshift128 rng128(perItemSeed);

                    WorldHit worldHit = hitRecords[rayIndex];
                    RayState rayState = raysIn[rayIndex];

                    if (!worldHit.hit)
                        return;
                    //printf("adjoint item %u, hit t: %f SplatEvents: %u \n", rayIndex, worldHit.t, worldHit.splatEventCount);

                    // rayIndex 2 is mesh
                    // rayIndex 3 is surfel a
                    // rayIndex 4 is surfel b

                    // Mesh Scatters Ray index 2
                    // 1. Implement transmission gradients for all pixels terminating at meshes (1 path length)
                    // - Visualize with vertical movements of the splats.
                    // 2. Implement inscattering from surfels with emission geometric term. This will give me radiance gradients.
                    // - Visualize with different colors of the splats. Some darker some brighter than the mesh background.

                    // Surfel Scatters A, Ray index 3
                    // 1. Implement interior contributions, i.e. Geometric for surfels

                    // Surfel scatters B, ray index 4:
                    // 2. Add in-scattering to surfel events along side interior contribution. Use the attached points to propagate change to transmissivity.
                    if (rayState.bounceIndex != 1)
                        return;

                    // Mesh background with in-scattering:
                    auto &instance = scene.instances[worldHit.instanceIndex];

                    Ray &ray = rayState.ray;
                    float3 grad_L(0.0f);

                    if (worldHit.splatEventCount == 1 && instance.geometryType == GeometryType::Mesh) {
                        const float emitterRadius = 0.01f; // set from mesh patch/texel size near y
                        const float pixelRadius = 0.01f; // set from sensor footprint at z
                        const float diffuseAlbedo = 1.0f; // use your albedo here (luminance)
                        const float invPi = 1.0f / M_PIf;

                        // x = ray.origin (sensor), y = mesh worldHit.hitPositionW
                        const float3 xPoint = ray.origin;
                        const float3 yPoint = worldHit.hitPositionW;

                        // Background geom x->y
                        const float3 segmentVectorXY = yPoint - xPoint;
                        const float r2Background = dot(segmentVectorXY, segmentVectorXY);
                        if (r2Background <= 0.0f) return;

                        const float rBackground = sycl::sqrt(r2Background);
                        const float3 unitDirXY = segmentVectorXY / rBackground;
                        const float cosineEmitBackground = dot(worldHit.geometricNormalW, -unitDirXY);
                        const float cosineReceiveBackground = dot(ray.normal, unitDirXY);
                        const float geometricBackground =
                                (cosineEmitBackground * cosineReceiveBackground) / r2Background;
                        if (geometricBackground <= 0.0f) return;

                        // Background radiance (mesh emitter)
                        const float3 backgroundRadianceRGB = estimateRadianceFromPhotonMap(worldHit, scene, photonMap);
                        const float backgroundRadianceY = luminanceGrayscale(backgroundRadianceRGB);

                        float3 S_b_L = estimateSurfelRadianceFromPhotonMap(
                        worldHit.splatEvents[0], ray.direction, scene, photonMap);
                        float L_surfel = luminance(S_b_L);

                        float3 S_b_L_rev = estimateSurfelRadianceFromPhotonMap(
                        worldHit.splatEvents[0], -ray.direction, scene, photonMap);
                        float L_surfel_rev = luminance(S_b_L_rev);

                        //printf("Background %f, Surfel: %f \n", backgroundRadianceY, L_surfel, L_surfel_rev);


                        // Surfel canonical normal and entered side
                        const auto surfel = scene.points[worldHit.splatEvents[0].primitiveIndex];
                        const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));
                        const int travelSideSign = (dot(canonicalNormalW, -rayState.ray.direction) >= 0.0f) ? 1 : -1;
                        const float3 surfelNormal = (travelSideSign >= 0) ? canonicalNormalW : (-canonicalNormalW);

                        // Intersect surfel plane on the x->y segment
                        const float nDotD = dot(surfelNormal, segmentVectorXY);
                        if (sycl::fabs(nDotD) <= 1e-8f) return;

                        const float tParam = dot(surfelNormal, surfel.position - xPoint) / nDotD;
                        if (!(tParam > 0.0f && tParam < 1.0f)) return;

                        const float3 hitPointZ = xPoint + tParam * segmentVectorXY;

                        // Side gate: only the side we enter first from x
                        if (dot(-surfelNormal, hitPointZ - xPoint) <= 0.0f) {
                            const float3 gradTransmittanceOnly = float3(0); // no α before this surfel ⇒ dT/dp = 0 here
                            grad_L = gradTransmittanceOnly * (backgroundRadianceY * geometricBackground);
                            return;
                        }

                        // Local uv and opacity α at z (chart-fixed)
                        const sycl::vec<float, 2> uv = phiInverse(hitPointZ, surfel.position,
                                                                  surfel.tanU, surfel.tanV,
                                                                  surfel.scale.x(), surfel.scale.y());
                        const float u = uv[0], v = uv[1];
                        const float alpha = sycl::exp(-0.5f * (u * u + v * v));

                        const float scaleU = surfel.scale.x();
                        const float scaleV = surfel.scale.y();
                        if (sycl::fabs(scaleU) <= 1e-12f || sycl::fabs(scaleV) <= 1e-12f) return;

                        // du/dp, dv/dp
                        const float dotTanU_d = dot(surfel.tanU, segmentVectorXY);
                        const float dotTanV_d = dot(surfel.tanV, segmentVectorXY);
                        const float3 gradU = ((dotTanU_d / nDotD) * surfelNormal - surfel.tanU) / scaleU;
                        const float3 gradV = ((dotTanV_d / nDotD) * surfelNormal - surfel.tanV) / scaleV;

                        // dα/dp and dT/dp; T_pre = 1-α here
                        const float3 gradAlpha = -alpha * (u * gradU + v * gradV);
                        const float3 dTransmittance_dCenter = -gradAlpha;

                        // -------- bounded surfel→sensor geometry G_xz and ∂/∂z (no helpers) --------
                        // G_xz = cos(n_x, w) / (|r|^2 + a_pix^2), r = z - x
                        const float3 rVecXZ = hitPointZ - xPoint;
                        const float r2XZ = dot(rVecXZ, rVecXZ);
                        if (r2XZ <= 0.0f) return;

                        const float rXZ = sycl::sqrt(r2XZ);
                        const float3 unitDirXZ = rVecXZ / rXZ;
                        const float cosineReceiveXZ = dot(ray.normal, unitDirXZ);
                        if (cosineReceiveXZ <= 0.0f) {
                            // only background gradient via T
                            grad_L = dTransmittance_dCenter * (backgroundRadianceY * geometricBackground);
                            return;
                        }
                        const float invDenomXZ = 1.0f / (r2XZ + pixelRadius * pixelRadius);
                        const float G_xz = cosineReceiveXZ * invDenomXZ;

                        // ∇_z G_xz
                        const float unitDirXZdotNx = dot(unitDirXZ, ray.normal);
                        const float3 gradCosineXZ = (ray.normal - unitDirXZ * unitDirXZdotNx) / rXZ; // (I - w w^T)n / r
                        const float3 gradInvXZ = -2.0f * rVecXZ * (invDenomXZ * invDenomXZ);
                        const float3 dGxz_dz = gradCosineXZ * invDenomXZ + cosineReceiveXZ * gradInvXZ;

                        // -------- bounded emitter→surfel geometry G_yz and ∂/∂z (no helpers) --------
                        // G_yz = (cos(n_emit,-w) * cos(n_surf,w)) / (|r|^2 + a_em^2), r = z - y
                        const float3 rVecYZ = hitPointZ - yPoint;
                        const float r2YZ = dot(rVecYZ, rVecYZ);
                        if (r2YZ <= 0.0f) return;

                        const float rYZ = sycl::sqrt(r2YZ);
                        const float3 unitDirYZ = rVecYZ / rYZ;
                        const float cosineEmitYZ = dot(worldHit.geometricNormalW, -unitDirYZ);
                        const float cosineRecvYZ = dot(surfelNormal, unitDirYZ);
                        if (cosineEmitYZ <= 0.0f || cosineRecvYZ <= 0.0f) {
                            // only background gradient via T
                            grad_L = dTransmittance_dCenter * (backgroundRadianceY * geometricBackground);
                        } else {
                            const float invDenomYZ = 1.0f / (r2YZ + emitterRadius * emitterRadius);
                            const float G_yz = (cosineEmitYZ * cosineRecvYZ) * invDenomYZ;

                            // ∇_z G_yz
                            const float unitDirYZdotNe = dot(unitDirYZ, worldHit.geometricNormalW);
                            const float3 dCosEmit_dz = -(worldHit.geometricNormalW - unitDirYZ * unitDirYZdotNe) / rYZ;
                            const float unitDirYZdotNs = dot(unitDirYZ, surfelNormal);
                            const float3 dCosRecv_dz = (surfelNormal - unitDirYZ * unitDirYZdotNs) / rYZ;
                            const float3 gradInvYZ = -2.0f * rVecYZ * (invDenomYZ * invDenomYZ);
                            const float3 dGyz_dz =
                                    (dCosEmit_dz * cosineRecvYZ + cosineEmitYZ * dCosRecv_dz) * invDenomYZ
                                    + (cosineEmitYZ * cosineRecvYZ) * gradInvYZ;

                            // Pull back ∂/∂z to ∂/∂p via J = d n^T / (n·d)
                            const float dDotdGxz = dot(segmentVectorXY, dGxz_dz);
                            const float dDotdGyz = dot(segmentVectorXY, dGyz_dz);
                            const float3 dGxz_dp = (dDotdGxz / nDotD) * surfelNormal;
                            const float3 dGyz_dp = (dDotdGyz / nDotD) * surfelNormal;

                            // Irradiance at surfel from mesh and its gradient
                            const float emitterRadianceY = backgroundRadianceY;
                            // use mesh radiance as emitter radiance (luminance)
                            const float Ez = emitterRadianceY * G_yz;
                            const float3 dEz_dp = emitterRadianceY * dGyz_dp;

                            // Background gradient via T (post-surface)
                            const float3 gradLbg = dTransmittance_dCenter * (backgroundRadianceY * geometricBackground);

                            // Surfel two-bounce term: L_surf = T_pre * α * (ρ/π) * Ez * G_xz, with T_pre = (1-α)
                            const float oneMinusAlpha = 1.0f - alpha;
                            const float3 gradLsurf =
                                    (1.0f - 2.0f * alpha) * gradAlpha * (diffuseAlbedo * invPi * Ez * G_xz) +
                                    oneMinusAlpha * alpha * diffuseAlbedo * invPi * (dEz_dp * G_xz + Ez * dGxz_dp);

                            grad_L = gradLbg + gradLsurf;

                            // Robustness clamp
                            const float gradMagnitude = length(grad_L);
                            if (gradMagnitude > 5.0f)
                                return;
                        }
                    }
                    if (worldHit.splatEventCount == 1 && instance.geometryType == GeometryType::PointCloud) {
                        float3 &x = ray.origin;
                        float3 &y = worldHit.hitPositionW;
                        float3 rvec = y - x;
                        float r2 = dot(rvec, rvec);
                        if (r2 <= 0.0f) return;
                        float r = sqrtf(r2);

                        const auto surfel = scene.points[worldHit.splatEvents[0].primitiveIndex];
                        const float3 canonicalNormalW = normalize(cross(surfel.tanU, surfel.tanV));
                        const int travelSideSign = signNonZero(dot(canonicalNormalW, -rayState.ray.direction));
                        const float3 surfelNormal = (travelSideSign >= 0) ? canonicalNormalW : (-canonicalNormalW);

                        float3 u = (x - y) / r; // Into the camera
                        float a = dot(surfelNormal, u);
                        float b = dot(ray.normal, -u);

                        if (a <= 0.0f || b <= 0.0f) {
                            return;
                        }

                        float3 grad_G = (-b * surfelNormal + a * ray.normal + 4.0f * a * b * u) / (r * r * r);

                        float3 S_a_L = estimateSurfelRadianceFromPhotonMap(
                            worldHit.splatEvents[0], ray.direction, scene, photonMap);
                        float L_surfel = luminance(S_a_L);

                        //grad_L = L_surfel * grad_G;

                        float grad_mag = length(grad_L);

                        if (grad_mag > 1.0f)
                            return;
                    }

                    /*
                    // Implement analytical gradient for surfel bg
                    if (worldHit.splatEventCount == 2 && instance.geometryType == GeometryType::PointCloud) {
                        float3 &x = ray.origin;
                        float3 &y = worldHit.hitPositionW;
                        float3 rvec = y - x;
                        float3 &n_y = worldHit.geometricNormalW;
                        float r2 = dot(rvec, rvec);
                        if (r2 <= 0.0f) return;

                        // Geometric terms
                        float r = sqrtf(r2);
                        float3 w = rvec / r;
                        float a = dot(n_y, -w);
                        float b = dot(ray.normal, w);

                        float G = (a * b) / r2;
                        float3 gradG_y = (-b * n_y + a * ray.normal + 4.0f * a * b * (-w)) / (r * r * r);

                        // Hit point Jacobian on inscatter surfel
                        auto &surfel = scene.points[worldHit.splatEvents[0].primitiveIndex];
                        float3 surfel_n = normalize(cross(surfel.tanU, surfel.tanV));
                        float denom = abs(dot(surfel_n, w));
                        float t_i = dot(surfel_n, (surfel.position - x)) / denom;

                        float3x3 I = identityMatrix<3>();

                        float3x3 tmp = (t_i / denom) * outerProduct(w, surfel_n);
                        float3x3 A_left = t_i * I - tmp;
                        float3x3 projector = I - outerProduct(w, w);
                        float3x3 A = (A_left * projector) / r;

                        // Gradient of radiance on B with respect to movement on A:
                        float alpha = worldHit.splatEvents[0].alpha;
                        float2 uv = phiInverse(worldHit.splatEvents[0].hitWorld, surfel);

                        float3 du_dx_row = surfel.tanU / surfel.scale.x();
                        float3 du_dy_row = surfel.tanV / surfel.scale.y();
                        float3 prefactor = (alpha / (1.0f - alpha)) * (uv[0] * du_dx_row + uv[1] * du_dy_row);


                        float tau_val = (1.0f - alpha);
                        float3 dlogtau_dy_row = prefactor * A;
                        float3 gradTau_pb = tau_val * dlogtau_dy_row;

                        // --- In-scatter term: L_ins = alphaA * S_a_L * Gv(zA) ---
                        // alphaA gradient wrt y (row before A_y): d alpha = -alpha (u du + v dv)
                        float3 alpha_row_before_A = -alpha * ((uv[0] * du_dx_row) + (uv[1] * du_dy_row));
                        float3 gradAlpha_pb = (alpha_row_before_A * A);

                        // Gv Grad on A:
                        float3 gradGv_p(0.0f);
                        {
                            float3 p = worldHit.splatEvents[0].hitWorld;
                           float3 r_vec_A = p -x;
                           float r2_A = dot(r_vec_A, r_vec_A);
                           float r_A = sqrtf(r2_A);
                           float3 u_dir = (x - p) / r_A;
                           float b = dot(-ray.normal, u_dir);
                           float Gv = b / r2_A;

                          gradGv_p  = (-ray.normal + 3.0 * b * u_dir) / (r * r * r);


                        }
                        float3 gradGv_pb = gradGv_p * A;
                        int debug = 1;
                    }
                    */

                    if (rayState.bounceIndex != 0) {
                        const float3 parameterAxis = {1.0f, 0.0f, 0.00f};
                        const float dVdp_scalar = dot(grad_L, parameterAxis);

                        float4 &gradImageDst = sensor.framebuffer[rayState.pixelIndex];
                        atomicAddFloatToImage(&gradImageDst, dVdp_scalar);
                    }
                });
        });
        queue.wait();
    }

    void generateNextAdjointRays(RenderPackage &pkg, uint32_t activeRayCount) {
        auto &queue = pkg.queue;
        auto &sensor = pkg.sensor;
        auto &settings = pkg.settings;
        auto &scene = pkg.scene;

        auto *hitRecords = pkg.intermediates.hitRecords;
        auto *raysIn = pkg.intermediates.primaryRays;
        auto *raysOut = pkg.intermediates.extensionRaysA;
        auto *countExtensionOut = pkg.intermediates.countExtensionOut;

        const uint32_t perPassRayCount = activeRayCount; // total number of rays (With n-samples per ray)
        const uint32_t perPixelRayCount = activeRayCount; // Number of rays per pixel
        const uint32_t photonsPerLaunch = settings.photonsPerLaunch;

        queue.memcpy(countExtensionOut, &activeRayCount, sizeof(uint32_t)).wait();

        queue.submit([&](sycl::handler &cgh) {
            const uint64_t baseSeed = settings.randomSeed;
            cgh.parallel_for<class GenerateNextAdjointRays>(
                sycl::range<1>(perPixelRayCount),
                [=](sycl::id<1> globalId) {
                    const uint32_t rayIndex = globalId[0];
                    const uint64_t seed = rng::makePerItemSeed1D(baseSeed, rayIndex);
                    rng::Xorshift128 rng128(seed);

                    const RayState inState = raysIn[rayIndex];
                    const WorldHit hit = hitRecords[rayIndex];

                    RayState outState{};
                    if (!hit.hit) {
                        raysOut[rayIndex] = outState;
                        hitRecords[rayIndex] = WorldHit{};
                        return;
                    } // dead ray

                    // --- Canonical normal at the hit (no face-forwarding stored in hit) ---
                    const float3 canonicalNormalW = hit.geometricNormalW;
                    // --- Orient to the side we entered first (two-sided, side-correct sampling) ---
                    // travelSideSign = sign(dot(n, -wo))
                    const int travelSideSign = signNonZero(dot(canonicalNormalW, -inState.ray.direction));
                    const float3 shadingNormal = (travelSideSign >= 0) ? canonicalNormalW : (-canonicalNormalW);

                    // Choose scattering model based on what we hit
                    float3 newDirection;
                    float pdf = 0.0f;
                    bool isSurfel = scene.instances[hit.instanceIndex].geometryType == GeometryType::PointCloud;
                    GPUMaterial material;
                    if (isSurfel) {
                        // volumetric or your chosen surfel phase sampling
                        sampleCosineHemisphere(rng128, shadingNormal, newDirection, pdf);
                        material.baseColor = scene.points[hit.primitiveIndex].color;
                    } else {
                        // surface: cosine-weighted about the geometric normal
                        sampleCosineHemisphere(rng128, shadingNormal, newDirection, pdf);
                        // Material fetch: use the material bound to this hit
                        const uint32_t materialIndex = scene.instances[hit.instanceIndex].materialIndex;
                        material = scene.materials[materialIndex];
                    }

                    //newDirection = normalize(float3{0.05, 0.04, -1}); // a
                    newDirection = normalize(float3{0.05, -0.04, 1}); // b

                    const float cosTheta = sycl::fmax(0.0f, dot(shadingNormal, newDirection));

                    float3 bsdfFactor;
                    if (isSurfel) {
                        // phase f_p and pdf_p; no cosine term
                        bsdfFactor = inState.pathThroughput * (material.baseColor / M_PIf) * (
                                         cosTheta / sycl::fmax(pdf, 1e-8f));
                    } else {
                        // If you keep explicit terms, use:
                        bsdfFactor = inState.pathThroughput * (material.baseColor / M_PIf) * (
                                         cosTheta / sycl::fmax(pdf, 1e-8f));
                    }

                    // Offset origin robustly
                    constexpr float kEps = 1e-5f;

                    outState.ray.origin = hit.hitPositionW + shadingNormal * kEps;
                    outState.ray.direction = newDirection;
                    outState.ray.normal = shadingNormal;
                    outState.bounceIndex = inState.bounceIndex + 1;
                    outState.pixelIndex = inState.pixelIndex;
                    outState.pathThroughput = bsdfFactor;

                    raysOut[rayIndex] = outState;
                });
        });
        queue.wait();
    }
}
