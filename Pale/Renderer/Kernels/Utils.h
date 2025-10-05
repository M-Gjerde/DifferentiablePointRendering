// PhotonMapExport.h
#pragma once
#include <vector>
#include <fstream>
#include <filesystem>
#include <cstdint>
#include <cmath>
#include <sycl/sycl.hpp>

#include "Renderer/GPUDataStructures.h"

static uint8_t linearToSRGB8(float linear) {
    float clamped = sycl::clamp(linear, 0.0f, 1.0f);
    float srgb = (clamped <= 0.0031308f)
                     ? (12.92f * clamped)
                     : (1.055f * std::pow(clamped, 1.0f / 2.4f) - 0.055f);
    int value = static_cast<int>(std::round(srgb * 255.0f));
    return static_cast<uint8_t>(sycl::clamp(value, 0, 255));
}

static void dumpPhotonMapToPLY(sycl::queue &queue,
                               const Pale::DevicePhotonSurface *devicePhotonArray,
                               std::uint32_t devicePhotonCount,
                               const std::filesystem::path &outputFilePath,
                               float exposureScale = 1.0f,
                               bool writeNormals = true) {
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

    for (const Pale::DevicePhotonSurface &ph: hostPhotons) {
        // Position
        float px = ph.position.x();
        float py = ph.position.y();
        float pz = ph.position.z();
        outFile.write(reinterpret_cast<const char *>(&px), sizeof(float));
        outFile.write(reinterpret_cast<const char *>(&py), sizeof(float));
        outFile.write(reinterpret_cast<const char *>(&pz), sizeof(float));

        // Normal from incident direction (pointing toward light)
        if (writeNormals) {
            // Use negative incidentDir as an outward shading normal guess
            float nx = -ph.incidentDir.x();
            float ny = -ph.incidentDir.y();
            float nz = -ph.incidentDir.z();
            outFile.write(reinterpret_cast<const char *>(&nx), sizeof(float));
            outFile.write(reinterpret_cast<const char *>(&ny), sizeof(float));
            outFile.write(reinterpret_cast<const char *>(&nz), sizeof(float));
        }

        // Color from power → simple exposure then sRGB
        float rLin = ph.power.x() * exposureScale;
        float gLin = ph.power.y() * exposureScale;
        float bLin = ph.power.z() * exposureScale;

        uint8_t r = linearToSRGB8(rLin);
        uint8_t g = linearToSRGB8(gLin);
        uint8_t b = linearToSRGB8(bLin);

        outFile.write(reinterpret_cast<const char *>(&r), sizeof(uint8_t));
        outFile.write(reinterpret_cast<const char *>(&g), sizeof(uint8_t));
        outFile.write(reinterpret_cast<const char *>(&b), sizeof(uint8_t));
    }

    outFile.flush();
    outFile.close();
}

namespace Pale {
    /*
        BoundedVector<T, MaxCount>
        --------------------------
        Fixed-capacity, POD-friendly container for device code.
        No dynamic allocation, no exceptions, no STL.
        Safe for SYCL/CUDA JIT: avoids operator new (_Znwm) and std::sort.

        Intended use: small scratch buffers inside kernels (e.g., leaf events).
    */
    template<typename T, int MaxCount>
    struct BoundedVector {
        static_assert(MaxCount > 0, "MaxCount must be positive");

        T storage[MaxCount];
        int elementCount = 0;

        SYCL_EXTERNAL inline void clear() { elementCount = 0; }
        SYCL_EXTERNAL inline int size() const { return elementCount; }
        SYCL_EXTERNAL inline int capacity() const { return MaxCount; }
        SYCL_EXTERNAL inline bool empty() const { return elementCount == 0; }
        SYCL_EXTERNAL inline T *data() { return storage; }
        SYCL_EXTERNAL inline const T *data() const { return storage; }

        // Push returns false if full. Caller decides whether to drop or handle overflow.
        SYCL_EXTERNAL inline bool pushBack(const T &value) {
            if (elementCount >= MaxCount) return false;
            storage[elementCount++] = value;
            return true;
        }

        // Read/write access. Use only with known-valid indices inside kernels.
        SYCL_EXTERNAL inline T &operator[](int index) { return storage[index]; }
        SYCL_EXTERNAL inline const T &operator[](int index) const { return storage[index]; }

        SYCL_EXTERNAL inline T &back() { return storage[elementCount - 1]; }
        SYCL_EXTERNAL inline const T &back() const { return storage[elementCount - 1]; }
    };

    /*
        insertionSortByKey(keys, indices, n)
        ------------------------------------
        Stable enough for small n (≤ a few hundred).
        Sorts keys ascending and carries an index permutation array alongside.

        Usage:
          - Fill `order[i] = i`, copy keys if you want to keep originals unchanged,
            then call insertionSortByKey(tempKeys.data(), order.data(), count).
          - Access items in sorted order via `order[k]`.
    */
    SYCL_EXTERNAL inline void insertionSortByKey(float *keyArray,
                                                 int *indexArray,
                                                 int elementCount) {
        for (int i = 1; i < elementCount; ++i) {
            float keyToInsert = keyArray[i];
            int indexToInsert = indexArray[i];
            int j = i - 1;
            // Move larger elements one position ahead
            for (; j >= 0 && keyArray[j] > keyToInsert; --j) {
                keyArray[j + 1] = keyArray[j];
                indexArray[j + 1] = indexArray[j];
            }
            keyArray[j + 1] = keyToInsert;
            indexArray[j + 1] = indexToInsert;
        }
    }
} // namespace Pale
