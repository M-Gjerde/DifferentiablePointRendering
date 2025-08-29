//
// Created by magnus-desktop on 8/28/25.
//
module;
#include <string>
#include <cstdint>
#include <fstream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <stb_image_write.h>


export module Pale.Utils.ImageIO;

export namespace Pale::Utils {

        // PNG saver: expects display-space (already tone-mapped + gamma-encoded) RGBA in [0,1]
    // If writeAlpha=false, the alpha channel is ignored and RGB is written.
    inline bool savePNG(
        const std::filesystem::path& filePath,
        const std::vector<float>& displaySpaceRGBA,
        std::uint32_t imageWidth,
        std::uint32_t imageHeight,
        bool writeAlpha = false)
    {

        if (!std::filesystem::exists(filePath.parent_path())) {
            std::filesystem::create_directories(filePath.parent_path());
        }
        if (imageWidth == 0 || imageHeight == 0) return false;
        const std::size_t expected = std::size_t(imageWidth) * imageHeight * 4u;
        if (displaySpaceRGBA.size() < expected) return false;

        const std::size_t pixelCount = std::size_t(imageWidth) * imageHeight;

        auto toUNorm8 = [](float x) -> std::uint8_t {
            const float clamped = std::clamp(x, 0.0f, 1.0f);
            return static_cast<std::uint8_t>(clamped * 255.0f + 0.5f);
        };

        if (writeAlpha) {
            std::vector<std::uint8_t> rgba8(pixelCount * 4u);
            for (std::size_t i = 0; i < pixelCount; ++i) {
                rgba8[i*4+0] = toUNorm8(displaySpaceRGBA[i*4+0]);
                rgba8[i*4+1] = toUNorm8(displaySpaceRGBA[i*4+1]);
                rgba8[i*4+2] = toUNorm8(displaySpaceRGBA[i*4+2]);
                rgba8[i*4+3] = toUNorm8(displaySpaceRGBA[i*4+3]);
            }
            return stbi_write_png(filePath.c_str(),
                                  static_cast<int>(imageWidth),
                                  static_cast<int>(imageHeight),
                                  /*components*/4,
                                  rgba8.data(),
                                  static_cast<int>(imageWidth * 4)) != 0;
        } else {
            std::vector<std::uint8_t> rgb8(pixelCount * 3u);
            for (std::size_t i = 0; i < pixelCount; ++i) {
                rgb8[i*3+0] = toUNorm8(displaySpaceRGBA[i*4+0]);
                rgb8[i*3+1] = toUNorm8(displaySpaceRGBA[i*4+1]);
                rgb8[i*3+2] = toUNorm8(displaySpaceRGBA[i*4+2]);
            }
            return stbi_write_png(filePath.c_str(),
                                  static_cast<int>(imageWidth),
                                  static_cast<int>(imageHeight),
                                  /*components*/3,
                                  rgb8.data(),
                                  static_cast<int>(imageWidth * 3)) != 0;
        }
    }

    // --- PFM (Portable Float Map) -------------------------------------------
    // Writes floating point image in PFM format. Data is written as 32-bit floats.
    // channels = 1 → grayscale "Pf", channels = 3 → RGB "PF".
    // PFM convention is bottom-to-top; set flipY=true if your buffer is top-to-bottom.
    inline bool savePFM(const std::string& filePath,
                               const float* floatData,
                               std::uint32_t imageWidth,
                               std::uint32_t imageHeight,
                               std::uint32_t channels = 3,
                               bool flipY = true)
    {
        if (!floatData || imageWidth == 0 || imageHeight == 0) return false;
        if (channels != 1 && channels != 3) return false;

        std::ofstream fileStream(filePath, std::ios::binary);
        if (!fileStream) return false;

        // Header
        // "PF" for RGB, "Pf" for grayscale
        fileStream << (channels == 3 ? "PF\n" : "Pf\n");
        fileStream << imageWidth << " " << imageHeight << "\n";

        // Scale: negative means little-endian (common on x86)
        // Typically -1.0 is used.
        fileStream << "-1.0\n";

        // Write rows. PFM expects the first row in the file to be the bottom row of the image.
        const std::size_t rowFloatCount = static_cast<std::size_t>(imageWidth) * channels;
        for (std::uint32_t row = 0; row < imageHeight; ++row) {
            const std::uint32_t sourceRow = flipY ? (imageHeight - 1 - row) : row;
            const float* rowPtr = floatData + static_cast<std::size_t>(sourceRow) * rowFloatCount;
            fileStream.write(reinterpret_cast<const char*>(rowPtr),
                             static_cast<std::streamsize>(rowFloatCount * sizeof(float)));
            if (!fileStream) return false;
        }
        return true;
    }

    // Convenience overload for std::vector<float> (linear HDR)
    inline bool savePFM(
        const std::filesystem::path& filePath,
        const std::vector<float>& linearFloatData,
        std::uint32_t imageWidth,
        std::uint32_t imageHeight,
        std::uint32_t channels = 3,
        bool flipY = true)
    {
        if (!std::filesystem::exists(filePath.parent_path())) {
            std::filesystem::create_directories(filePath.parent_path());
        }

        const std::size_t expected = std::size_t(imageWidth) * imageHeight * channels;
        if (linearFloatData.size() < expected) return false;
        return savePFM(filePath, linearFloatData.data(), imageWidth, imageHeight, channels, flipY);
    }

}
