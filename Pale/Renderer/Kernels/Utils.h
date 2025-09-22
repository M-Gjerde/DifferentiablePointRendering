// PhotonMapExport.h
#pragma once
#include <vector>
#include <fstream>
#include <filesystem>
#include <cstdint>
#include <cmath>
#include <sycl/sycl.hpp>

#include "Renderer/GPUDataStructures.h"

static uint8_t linearToSRGB8(float linear)
{
    float clamped = sycl::clamp(linear, 0.0f, 1.0f);
    float srgb = (clamped <= 0.0031308f) ? (12.92f * clamped)
                                         : (1.055f * std::pow(clamped, 1.0f/2.4f) - 0.055f);
    int value = static_cast<int>(std::round(srgb * 255.0f));
    return static_cast<uint8_t>(sycl::clamp(value, 0, 255));
}

static void dumpPhotonMapToPLY(sycl::queue &queue,
                               const Pale::DevicePhotonSurface *devicePhotonArray,
                               std::uint32_t devicePhotonCount,
                               const std::filesystem::path &outputFilePath,
                               float exposureScale = 1.0f,
                               bool writeNormals = true)
{
    if (devicePhotonCount == 0) return;

    std::vector<Pale::DevicePhotonSurface> hostPhotons(devicePhotonCount);
    queue.memcpy(hostPhotons.data(), devicePhotonArray,
                 sizeof(Pale::DevicePhotonSurface) * devicePhotonCount).wait();

    std::ofstream outFile(outputFilePath, std::ios::binary);
    if (!outFile)
        return;

    // Header (binary_little_endian PLY)
    outFile << "ply\nformat binary_little_endian 1.0\n";
    outFile << "element vertex " << devicePhotonCount << "\n";
    outFile << "property float x\nproperty float y\nproperty float z\n";
    if (writeNormals) {
        outFile << "property float nx\nproperty float ny\nproperty float nz\n";
    }
    outFile << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    outFile << "end_header\n";

    for (const Pale::DevicePhotonSurface &ph : hostPhotons) {
        // Position
        float px = ph.position.x();
        float py = ph.position.y();
        float pz = ph.position.z();
        outFile.write(reinterpret_cast<const char*>(&px), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&py), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&pz), sizeof(float));

        // Normal from incident direction (pointing toward light)
        if (writeNormals) {
            // Use negative incidentDir as an outward shading normal guess
            float nx = -ph.incidentDir.x();
            float ny = -ph.incidentDir.y();
            float nz = -ph.incidentDir.z();
            outFile.write(reinterpret_cast<const char*>(&nx), sizeof(float));
            outFile.write(reinterpret_cast<const char*>(&ny), sizeof(float));
            outFile.write(reinterpret_cast<const char*>(&nz), sizeof(float));
        }

        // Color from power → simple exposure then sRGB
        float rLin = ph.power.x() * exposureScale;
        float gLin = ph.power.y() * exposureScale;
        float bLin = ph.power.z() * exposureScale;

        uint8_t r = linearToSRGB8(rLin);
        uint8_t g = linearToSRGB8(gLin);
        uint8_t b = linearToSRGB8(bLin);

        outFile.write(reinterpret_cast<const char*>(&r), sizeof(uint8_t));
        outFile.write(reinterpret_cast<const char*>(&g), sizeof(uint8_t));
        outFile.write(reinterpret_cast<const char*>(&b), sizeof(uint8_t));
    }

    outFile.flush();
    outFile.close();
}
