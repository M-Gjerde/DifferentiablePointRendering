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
#include <cmath>
#include <stb_image_write.h>


export module Pale.Utils.ImageIO;

export namespace Pale::Utils {
    // exposureEV:   photographic EV; 0 keeps values, +1 doubles brightness
    // gammaEncode:  typical 2.2 for sRGB-like output
    // normalizeHDR: if true, first map global RGB min..max to [0,1]
    inline bool savePNGWithToneMap(
        const std::filesystem::path& filePath,
        const std::vector<float>& inputRGBA,
        std::uint32_t imageWidth,
        std::uint32_t imageHeight,
        float exposureEV = 0.0f,
        float gammaEncode = 2.2f,
        bool normalizeHDR = false,
        bool writeAlpha = false) {
        if (!std::filesystem::exists(filePath.parent_path())) {
            std::filesystem::create_directories(filePath.parent_path());
        }
        if (imageWidth == 0 || imageHeight == 0) return false;
        const std::size_t pixelCount = std::size_t(imageWidth) * imageHeight;
        const std::size_t expected = pixelCount * 4u;
        if (inputRGBA.size() < expected) return false;

        // Optional global min–max normalization over RGB
        float rgbMin = std::numeric_limits<float>::max();
        float rgbMax = std::numeric_limits<float>::lowest();
        if (normalizeHDR) {
            for (std::size_t i = 0; i < pixelCount; ++i) {
                const float r = inputRGBA[i * 4 + 0];
                const float g = inputRGBA[i * 4 + 1];
                const float b = inputRGBA[i * 4 + 2];
                if (std::isfinite(r)) {
                    rgbMin = std::min(rgbMin, r);
                    rgbMax = std::max(rgbMax, r);
                }
                if (std::isfinite(g)) {
                    rgbMin = std::min(rgbMin, g);
                    rgbMax = std::max(rgbMax, g);
                }
                if (std::isfinite(b)) {
                    rgbMin = std::min(rgbMin, b);
                    rgbMax = std::max(rgbMax, b);
                }
            }
            if (!(rgbMax > rgbMin)) {
                rgbMin = 0.0f;
                rgbMax = 1.0f;
            }
        }
        const float invRange = normalizeHDR ? 1.0f / (rgbMax - rgbMin) : 1.0f;

        // Exposure scale in linear space
        const float exposureScale = std::exp2(exposureEV);
        const float invGamma = 1.0f / std::max(1e-6f, gammaEncode);

        auto encodeChannel = [&](float linearValue) -> std::uint8_t {
            float v = linearValue;
            if (normalizeHDR) v = (v - rgbMin) * invRange;
            v = std::max(0.0f, v * exposureScale); // no negative light
            v = std::pow(v, invGamma); // gamma encode
            v = std::clamp(v, 0.0f, 1.0f);
            return static_cast<std::uint8_t>(v * 255.0f + 0.5f);
        };

        if (writeAlpha) {
            std::vector<std::uint8_t> rgba8(pixelCount * 4u);
            for (std::size_t i = 0; i < pixelCount; ++i) {
                rgba8[i * 4 + 0] = encodeChannel(inputRGBA[i * 4 + 0]);
                rgba8[i * 4 + 1] = encodeChannel(inputRGBA[i * 4 + 1]);
                rgba8[i * 4 + 2] = encodeChannel(inputRGBA[i * 4 + 2]);
                // Alpha is not tone-mapped. Clamp to [0,1].
                float alphaLinear = std::clamp(inputRGBA[i * 4 + 3], 0.0f, 1.0f);
                rgba8[i * 4 + 3] = static_cast<std::uint8_t>(alphaLinear * 255.0f + 0.5f);
            }
            return stbi_write_png(filePath.c_str(),
                                  static_cast<int>(imageWidth),
                                  static_cast<int>(imageHeight),
                                  4, rgba8.data(),
                                  static_cast<int>(imageWidth * 4)) != 0;
        }
        else {
            std::vector<std::uint8_t> rgb8(pixelCount * 3u);
            for (std::size_t i = 0; i < pixelCount; ++i) {
                rgb8[i * 3 + 0] = encodeChannel(inputRGBA[i * 4 + 0]);
                rgb8[i * 3 + 1] = encodeChannel(inputRGBA[i * 4 + 1]);
                rgb8[i * 3 + 2] = encodeChannel(inputRGBA[i * 4 + 2]);
            }
            return stbi_write_png(filePath.c_str(),
                                  static_cast<int>(imageWidth),
                                  static_cast<int>(imageHeight),
                                  3, rgb8.data(),
                                  static_cast<int>(imageWidth * 3)) != 0;
        }
    }

    // PNG saver: expects display-space (already tone-mapped + gamma-encoded) RGBA in [0,1]
    // If writeAlpha=false, the alpha channel is ignored and RGB is written.
    inline bool savePNG(
        const std::filesystem::path& filePath,
        const std::vector<float>& displaySpaceRGBA,
        std::uint32_t imageWidth,
        std::uint32_t imageHeight,
        bool writeAlpha = false) {
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
                rgba8[i * 4 + 0] = toUNorm8(displaySpaceRGBA[i * 4 + 0]);
                rgba8[i * 4 + 1] = toUNorm8(displaySpaceRGBA[i * 4 + 1]);
                rgba8[i * 4 + 2] = toUNorm8(displaySpaceRGBA[i * 4 + 2]);
                rgba8[i * 4 + 3] = toUNorm8(displaySpaceRGBA[i * 4 + 3]);
            }
            return stbi_write_png(filePath.c_str(),
                                  static_cast<int>(imageWidth),
                                  static_cast<int>(imageHeight),
                                  /*components*/4,
                                  rgba8.data(),
                                  static_cast<int>(imageWidth * 4)) != 0;
        }
        else {
            std::vector<std::uint8_t> rgb8(pixelCount * 3u);
            for (std::size_t i = 0; i < pixelCount; ++i) {
                rgb8[i * 3 + 0] = toUNorm8(displaySpaceRGBA[i * 4 + 0]);
                rgb8[i * 3 + 1] = toUNorm8(displaySpaceRGBA[i * 4 + 1]);
                rgb8[i * 3 + 2] = toUNorm8(displaySpaceRGBA[i * 4 + 2]);
            }
            return stbi_write_png(filePath.c_str(),
                                  static_cast<int>(imageWidth),
                                  static_cast<int>(imageHeight),
                                  /*components*/3,
                                  rgb8.data(),
                                  static_cast<int>(imageWidth * 3)) != 0;
        }
    }

    // --- Core writer: accepts only 1 or 3 channels (PFM spec) -------------------
    inline bool writePFM_RGBorGray(const std::filesystem::path& filePath,
                                   const float* floatData,
                                   std::uint32_t imageWidth,
                                   std::uint32_t imageHeight,
                                   std::uint32_t channels, // 1 or 3
                                   bool flipY = true) {
        if (!floatData || imageWidth == 0 || imageHeight == 0) return false;
        if (channels != 1 && channels != 3) return false;

        if (!std::filesystem::exists(filePath.parent_path())) {
            std::filesystem::create_directories(filePath.parent_path());
        }

        std::ofstream fileStream(filePath, std::ios::binary);
        if (!fileStream) return false;

        // Header: "PF" for RGB, "Pf" for grayscale
        fileStream << (channels == 3 ? "PF\n" : "Pf\n");
        fileStream << imageWidth << " " << imageHeight << "\n";
        fileStream << "-1.0\n"; // little-endian scale

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

    // --- Convenience: accepts vector with 1, 3, or 4 channels -------------------
    inline bool savePFM(const std::filesystem::path& filePath,
                        const std::vector<float>& linearFloatDataRGBAorRGB,
                        std::uint32_t imageWidth,
                        std::uint32_t imageHeight,
                        bool flipY = true,
                        bool writeAlphaAsPf = false,
                        std::filesystem::path alphaFilePath = {}) {
        if (imageWidth == 0 || imageHeight == 0) return false;

        const std::size_t pixelCount = static_cast<std::size_t>(imageWidth) * imageHeight;
        if (linearFloatDataRGBAorRGB.size() % pixelCount != 0) return false;

        const std::uint32_t channels =
            static_cast<std::uint32_t>(linearFloatDataRGBAorRGB.size() / pixelCount);

        if (channels == 1) {
            return writePFM_RGBorGray(filePath,
                                      linearFloatDataRGBAorRGB.data(),
                                      imageWidth, imageHeight, 1, flipY);
        }
        if (channels == 3) {
            return writePFM_RGBorGray(filePath,
                                      linearFloatDataRGBAorRGB.data(),
                                      imageWidth, imageHeight, 3, flipY);
        }
        if (channels == 4) {
            // 1) Write RGB by dropping alpha
            std::vector<float> rgbPixels;
            rgbPixels.resize(pixelCount * 3u);
            for (std::size_t i = 0, j = 0; i < pixelCount; ++i) {
                rgbPixels[j++] = linearFloatDataRGBAorRGB[4 * i + 0];
                rgbPixels[j++] = linearFloatDataRGBAorRGB[4 * i + 1];
                rgbPixels[j++] = linearFloatDataRGBAorRGB[4 * i + 2];
            }
            const bool okRGB = writePFM_RGBorGray(filePath,
                                                  rgbPixels.data(),
                                                  imageWidth, imageHeight, 3, flipY);
            if (!okRGB) return false;

            // 2) Optionally write alpha as Pf
            if (writeAlphaAsPf) {
                if (alphaFilePath.empty()) {
                    alphaFilePath = filePath;
                    alphaFilePath.replace_filename(
                        alphaFilePath.stem().string() + std::string(".alpha") + alphaFilePath.extension().string());
                }
                std::vector<float> alphaPixels;
                alphaPixels.resize(pixelCount);
                for (std::size_t i = 0; i < pixelCount; ++i) {
                    alphaPixels[i] = linearFloatDataRGBAorRGB[4 * i + 3];
                }
                const bool okA = writePFM_RGBorGray(alphaFilePath,
                                                    alphaPixels.data(),
                                                    imageWidth, imageHeight, 1, flipY);
                if (!okA) return false;
            }
            return true;
        }
        return false; // unsupported channel count
    }
}
