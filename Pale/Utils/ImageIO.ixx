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
#include <cstring>
#include <stb_image_write.h>

#include <OpenEXR/ImfRgbaFile.h>


export module Pale.Utils.ImageIO;


export namespace Pale::Utils {

    std::vector<float> computeL2ImageGradientRGBA(
    const std::vector<float>& renderedRgba,
    const std::vector<float>& targetRgba,
    std::uint32_t imageWidth,
    std::uint32_t imageHeight)
    {
        const std::size_t expectedSize =
            static_cast<std::size_t>(imageWidth) * static_cast<std::size_t>(imageHeight) * 4ull;

        if (renderedRgba.size() != expectedSize) {
            throw std::runtime_error("renderedRgba size does not match width * height * 4");
        }
        if (targetRgba.size() != expectedSize) {
            throw std::runtime_error("targetRgba size does not match width * height * 4");
        }

        std::vector<float> gradientRgba(expectedSize);

        for (std::size_t elementIndex = 0; elementIndex < expectedSize; ++elementIndex) {
            gradientRgba[elementIndex] = renderedRgba[elementIndex] - targetRgba[elementIndex];
        }

        return gradientRgba;
    }


void loadEXRAsRGBAFloat(
    const std::filesystem::path& filePath,
    std::vector<float>& rgbaOut,
    std::uint32_t& imageWidthOut,
    std::uint32_t& imageHeightOut)
{
    Imf::RgbaInputFile inputFile(filePath.string().c_str());

    const Imath::Box2i dataWindow = inputFile.dataWindow();
    const int dataMinX = dataWindow.min.x;
    const int dataMinY = dataWindow.min.y;
    const int dataMaxX = dataWindow.max.x;
    const int dataMaxY = dataWindow.max.y;

    if (dataMaxX < dataMinX || dataMaxY < dataMinY) {
        throw std::runtime_error("Invalid EXR dataWindow.");
    }

    const std::uint32_t imageWidth = static_cast<std::uint32_t>(dataMaxX - dataMinX + 1);
    const std::uint32_t imageHeight = static_cast<std::uint32_t>(dataMaxY - dataMinY + 1);

    const std::size_t pixelCount = static_cast<std::size_t>(imageWidth) * static_cast<std::size_t>(imageHeight);

    std::vector<Imf::Rgba> exrPixels;
    exrPixels.resize(pixelCount);

    // OpenEXR's dataWindow may not start at (0,0). The framebuffer base pointer must be adjusted.
    // This makes pixel (dataMinX, dataMinY) map to exrPixels[0].
    Imf::Rgba* frameBufferBasePointer =
        exrPixels.data() - static_cast<std::ptrdiff_t>(dataMinX) - static_cast<std::ptrdiff_t>(dataMinY) * static_cast<std::ptrdiff_t>(imageWidth);

    inputFile.setFrameBuffer(frameBufferBasePointer, 1, static_cast<int>(imageWidth));
    inputFile.readPixels(dataMinY, dataMaxY);

    rgbaOut.resize(pixelCount * 4ull);

    for (std::uint32_t y = 0; y < imageHeight; ++y) {
        for (std::uint32_t x = 0; x < imageWidth; ++x) {
            const std::size_t pixelIndex = static_cast<std::size_t>(y) * imageWidth + x;
            const std::size_t dstIndex = pixelIndex * 4ull;

            const Imf::Rgba& pixel = exrPixels[pixelIndex];

            // Imf::Rgba stores half by default; implicit conversion to float is fine.
            rgbaOut[dstIndex + 0] = static_cast<float>(pixel.r);
            rgbaOut[dstIndex + 1] = static_cast<float>(pixel.g);
            rgbaOut[dstIndex + 2] = static_cast<float>(pixel.b);
            rgbaOut[dstIndex + 3] = static_cast<float>(pixel.a);
        }
    }

    imageWidthOut = imageWidth;
    imageHeightOut = imageHeight;
}

    void saveRGBAFloatAsEXR(
        const std::filesystem::path& filePath,
        const std::vector<float>& rgbaRaw,
        uint32_t imageWidth,
        uint32_t imageHeight)
    {
        const size_t expectedSize = static_cast<size_t>(imageWidth) *
                                    static_cast<size_t>(imageHeight) * 4ull;

        if (rgbaRaw.size() != expectedSize) {
            throw std::runtime_error("rgbaRaw size does not match width * height * 4");
        }

        // OpenEXR expects RGBA pixels in scanline order
        std::vector<Imf::Rgba> exrPixels;
        exrPixels.resize(imageWidth * imageHeight);

        for (uint32_t y = 0; y < imageHeight; ++y) {
            for (uint32_t x = 0; x < imageWidth; ++x) {
                const size_t srcIndex = (static_cast<size_t>(y) * imageWidth + x) * 4ull;
                const size_t dstIndex = static_cast<size_t>(y) * imageWidth + x;

                Imf::Rgba& pixel = exrPixels[dstIndex];
                pixel.r = rgbaRaw[srcIndex + 0];
                pixel.g = rgbaRaw[srcIndex + 1];
                pixel.b = rgbaRaw[srcIndex + 2];
                pixel.a = rgbaRaw[srcIndex + 3];
            }
        }

        Imf::RgbaOutputFile outputFile(
            filePath.string().c_str(),
            imageWidth,
            imageHeight,
            Imf::WRITE_RGBA
        );

        outputFile.setFrameBuffer(exrPixels.data(), 1, imageWidth);
        outputFile.writePixels(imageHeight);
    }


    inline bool savePNG(
        const std::filesystem::path &filePath,
        const std::vector<uint8_t> &inputRGBA,
        std::uint32_t imageWidth,
        std::uint32_t imageHeight) {

        if (!std::filesystem::exists(filePath.parent_path())) {
            std::filesystem::create_directories(filePath.parent_path());
        }
        return  stbi_write_png(filePath.c_str(),
                       static_cast<int>(imageWidth),
                       static_cast<int>(imageHeight),
                       4, inputRGBA.data(), imageWidth * 4);
    }

    // exposureEV:   photographic EV; 0 keeps values, +1 doubles brightness
    // gammaEncode:  typical 2.2 for sRGB-like output
    // normalizeHDR: if true, first map global RGB min..max to [0,1]
    inline bool savePNGWithToneMap(
        const std::filesystem::path &filePath,
        const std::vector<float> &inputRGBA,
        std::uint32_t imageWidth,
        std::uint32_t imageHeight,
        float exposureEV = 0.0f,
        float gammaEncode = 2.2f,
        bool normalizeHDR = false,
        bool writeAlpha = false,
        bool flipY = false) {
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

            const int bytesPerRow = static_cast<int>(imageWidth * 4);
            const std::uint8_t *srcPtr = rgba8.data();

            std::vector<std::uint8_t> rowOrdered;
            if (flipY) {
                rowOrdered.resize(rgba8.size());
                for (std::uint32_t y = 0; y < imageHeight; ++y) {
                    const std::uint32_t srcY = imageHeight - 1 - y;
                    std::memcpy(rowOrdered.data() + y * bytesPerRow,
                                srcPtr + srcY * bytesPerRow,
                                bytesPerRow);
                }
                srcPtr = rowOrdered.data();
            }
            return stbi_write_png(filePath.c_str(),
                                  static_cast<int>(imageWidth),
                                  static_cast<int>(imageHeight),
                                  4, srcPtr,
                                  static_cast<int>(imageWidth * 4)) != 0;
        } else {
            std::vector<std::uint8_t> rgb8(pixelCount * 3u);
            for (std::size_t i = 0; i < pixelCount; ++i) {
                rgb8[i * 3 + 0] = encodeChannel(inputRGBA[i * 4 + 0]);
                rgb8[i * 3 + 1] = encodeChannel(inputRGBA[i * 4 + 1]);
                rgb8[i * 3 + 2] = encodeChannel(inputRGBA[i * 4 + 2]);
            }
            const int bytesPerRow = static_cast<int>(imageWidth * 3);
            const std::uint8_t *srcPtr = rgb8.data();

            std::vector<std::uint8_t> rowOrdered;
            if (flipY) {
                rowOrdered.resize(rgb8.size());
                for (std::uint32_t y = 0; y < imageHeight; ++y) {
                    const std::uint32_t srcY = imageHeight - 1 - y;
                    std::memcpy(rowOrdered.data() + y * bytesPerRow,
                                srcPtr + srcY * bytesPerRow,
                                bytesPerRow);
                }
                srcPtr = rowOrdered.data();
            }

            return stbi_write_png(filePath.c_str(),
                                  static_cast<int>(imageWidth),
                                  static_cast<int>(imageHeight),
                                  3, srcPtr, bytesPerRow) != 0;
        }
    }
    // exposureEV:   photographic EV; 0 keeps values, +1 doubles brightness
    // gammaEncode:  typical 2.2 for sRGB-like output
    // normalizeHDR: if true, first map global RGB min..max to [0,1]
    inline bool savePNGWith3Channel(
        const std::filesystem::path &filePath,
        const std::vector<float> &inputRGB,
        std::uint32_t imageWidth,
        std::uint32_t imageHeight) {
        if (!std::filesystem::exists(filePath.parent_path())) {
            std::filesystem::create_directories(filePath.parent_path());
        }
        if (imageWidth == 0 || imageHeight == 0) return false;
        const std::size_t pixelCount = std::size_t(imageWidth) * imageHeight;
        const std::size_t expected = pixelCount * 3u;
        if (inputRGB.size() < expected) return false;


        auto encodeChannel = [](float linearValue) -> std::uint8_t {
            // Clamp to [0,1] for safety
            float v = std::clamp(linearValue, 0.0f, 1.0f);
            float scaled = v * 255.0f + 0.5f; // round to nearest
            if (scaled < 0.0f)   scaled = 0.0f;
            if (scaled > 255.0f) scaled = 255.0f;
            return static_cast<std::uint8_t>(scaled);
        };

        std::vector<std::uint8_t> rgb8(pixelCount * 3u);
        for (std::size_t i = 0; i < pixelCount; ++i) {
            rgb8[i * 3 + 0] = encodeChannel(inputRGB[i * 3 + 0]);
            rgb8[i * 3 + 1] = encodeChannel(inputRGB[i * 3 + 1]);
            rgb8[i * 3 + 2] = encodeChannel(inputRGB[i * 3 + 2]);
        }
        const int bytesPerRow = static_cast<int>(imageWidth * 3);
        const std::uint8_t *srcPtr = rgb8.data();

        std::vector<std::uint8_t> rowOrdered;

        return stbi_write_png(filePath.c_str(),
                              static_cast<int>(imageWidth),
                              static_cast<int>(imageHeight),
                              3, srcPtr, bytesPerRow) != 0;
    }

    // PNG saver: expects display-space (already tone-mapped + gamma-encoded) RGBA in [0,1]
    // If writeAlpha=false, the alpha channel is ignored and RGB is written.
    inline bool savePNG(
        const std::filesystem::path &filePath,
        const std::vector<float> &displaySpaceRGBA,
        std::uint32_t imageWidth,
        std::uint32_t imageHeight,
        bool writeAlpha = false,
        bool flipY = false) {
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
            const int bytesPerRow = static_cast<int>(imageWidth * 4);
            const std::uint8_t *srcPtr = rgba8.data();

            std::vector<std::uint8_t> rowOrdered;
            if (flipY) {
                rowOrdered.resize(rgba8.size());
                for (std::uint32_t y = 0; y < imageHeight; ++y) {
                    const std::uint32_t srcY = imageHeight - 1 - y;
                    std::memcpy(rowOrdered.data() + y * bytesPerRow,
                                srcPtr + srcY * bytesPerRow,
                                bytesPerRow);
                }
                srcPtr = rowOrdered.data();
            }

            return stbi_write_png(filePath.c_str(),
                                  static_cast<int>(imageWidth),
                                  static_cast<int>(imageHeight),
                                  4, srcPtr, bytesPerRow) != 0;
        } else {
            std::vector<std::uint8_t> rgb8(pixelCount * 3u);
            for (std::size_t i = 0; i < pixelCount; ++i) {
                rgb8[i * 3 + 0] = toUNorm8(displaySpaceRGBA[i * 4 + 0]);
                rgb8[i * 3 + 1] = toUNorm8(displaySpaceRGBA[i * 4 + 1]);
                rgb8[i * 3 + 2] = toUNorm8(displaySpaceRGBA[i * 4 + 2]);
            }

            const int bytesPerRow = static_cast<int>(imageWidth * 3);
            const std::uint8_t *srcPtr = rgb8.data();

            std::vector<std::uint8_t> rowOrdered;
            if (flipY) {
                rowOrdered.resize(rgb8.size());
                for (std::uint32_t y = 0; y < imageHeight; ++y) {
                    const std::uint32_t srcY = imageHeight - 1 - y;
                    std::memcpy(rowOrdered.data() + y * bytesPerRow,
                                srcPtr + srcY * bytesPerRow,
                                bytesPerRow);
                }
                srcPtr = rowOrdered.data();
            }

            return stbi_write_png(filePath.c_str(),
                                  static_cast<int>(imageWidth),
                                  static_cast<int>(imageHeight),
                                  3, srcPtr, bytesPerRow) != 0;
        }
    }

    // --- Core writer: accepts only 1 or 3 channels (PFM spec) -------------------
    inline bool writePFM_RGBorGray(const std::filesystem::path &filePath,
                                   const float *floatData,
                                   std::uint32_t imageWidth,
                                   std::uint32_t imageHeight,
                                   std::uint32_t channels, // 1 or 3
                                   bool flipY = false) {
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
            const float *rowPtr = floatData + static_cast<std::size_t>(sourceRow) * rowFloatCount;
            fileStream.write(reinterpret_cast<const char *>(rowPtr),
                             static_cast<std::streamsize>(rowFloatCount * sizeof(float)));
            if (!fileStream) return false;
        }
        return true;
    }

    // --- Convenience: accepts vector with 1, 3, or 4 channels -------------------
    inline bool savePFM(const std::filesystem::path &filePath,
                        const std::vector<float> &linearFloatDataRGBAorRGB,
                        std::uint32_t imageWidth,
                        std::uint32_t imageHeight,
                        bool flipY = false,
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

    // Reads next non-empty, non-comment line.
    inline bool readNextHeaderLine(std::istream &in, std::string &outLine) {
        while (std::getline(in, outLine)) {
            // Trim
            auto notSpace = [](unsigned char c) { return !std::isspace(c); };
            outLine.erase(outLine.begin(), std::find_if(outLine.begin(), outLine.end(), notSpace));
            outLine.erase(std::find_if(outLine.rbegin(), outLine.rend(), notSpace).base(), outLine.end());
            if (outLine.empty()) continue;
            if (!outLine.empty() && outLine[0] == '#') continue;
            return true;
        }
        return false;
    }

    // Load 3-channel PFM. Returns true on success.
    // Output pixels are RGB float32, size = width*height*3.
    // If flipY=true, the first row in outPixels is the top image row.
    inline bool loadPFM_RGB(const std::filesystem::path &filePath,
                            std::vector<float> &outPixelsRGB,
                            std::uint32_t &outWidth,
                            std::uint32_t &outHeight,
                            bool flipY = false) {
        outPixelsRGB.clear();
        outWidth = 0;
        outHeight = 0;

        std::ifstream fileStream(filePath, std::ios::binary);
        if (!fileStream) return false;

        // 1) Magic
        std::string line;
        if (!readNextHeaderLine(fileStream, line)) return false;
        if (line != "PF") return false; // enforce 3-channel RGB

        // 2) Dimensions
        if (!readNextHeaderLine(fileStream, line)) return false; {
            std::istringstream dimStream(line);
            int w = 0, h = 0;
            if (!(dimStream >> w >> h)) return false;
            if (w <= 0 || h <= 0) return false;
            outWidth = static_cast<std::uint32_t>(w);
            outHeight = static_cast<std::uint32_t>(h);
        }

        // 3) Scale (sign encodes endianness per spec)
        if (!readNextHeaderLine(fileStream, line)) return false;
        float scale = 0.f; {
            std::istringstream scaleStream(line);
            if (!(scaleStream >> scale)) return false;
            if (scale == 0.f) return false;
        }
        const bool littleEndianStream = (scale < 0.f);
        const float multiplicativeScale = std::abs(scale); // intensity scale factor

        // Basic endianness check: typical files are little-endian with negative scale.
        // This loader expects native little-endian. Refuse big-endian streams.
        // If you need big-endian support, add byte-swapping here.
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        if (!littleEndianStream) return false;
#else
        if (littleEndianStream) return false; // not implemented
#endif

        // 4) Binary blob
        const std::size_t pixelCount = static_cast<std::size_t>(outWidth) * outHeight;
        const std::size_t floatCount = pixelCount * 3u;
        outPixelsRGB.resize(floatCount);

        // The next byte is the start of binary data; ensure stream at correct pos
        fileStream.read(reinterpret_cast<char *>(outPixelsRGB.data()),
                        static_cast<std::streamsize>(floatCount * sizeof(float)));
        if (!fileStream) return false;

        // 5) Apply scale if not 1.0. Most files use |scale|=1.0
        if (multiplicativeScale != 1.f) {
            for (float &v: outPixelsRGB) v *= multiplicativeScale;
        }

        // 6) Flip vertically if requested (PFM stores bottom-to-top by convention)
        if (flipY) {
            const std::size_t rowFloatCount = static_cast<std::size_t>(outWidth) * 3u;
            for (std::uint32_t yTop = 0, yBot = outHeight - 1; yTop < yBot; ++yTop, --yBot) {
                float *topRow = outPixelsRGB.data() + static_cast<std::size_t>(yTop) * rowFloatCount;
                float *botRow = outPixelsRGB.data() + static_cast<std::size_t>(yBot) * rowFloatCount;
                std::swap_ranges(topRow, topRow + rowFloatCount, botRow);
            }
        }

        return true;
    }


    inline void sampleBlueBlackRed(float normalizedSigned, float &outR, float &outG, float &outB) {
        // normalizedSigned in [-1, 1]; 0 -> black center
        outR = normalizedSigned > 0.0f ? normalizedSigned : 0.0f;
        outB = normalizedSigned < 0.0f ? -normalizedSigned : 0.0f;
        outG = 0.0f;
    }

    inline void sampleSeismic(float normalizedSigned, float &outR, float &outG, float &outB) {
        // normalizedSigned in [-1, 1]; 0 -> white center
        normalizedSigned = std::clamp(normalizedSigned, -1.0f, 1.0f);
        if (normalizedSigned <= 0.0f) {
            // blue -> white
            float t = normalizedSigned + 1.0f; // [-1,0] -> [0,1]
            outR = t; // 0 -> 1
            outG = t; // 0 -> 1
            outB = 1.0f; // stays blue
        } else {
            // white -> red
            float t = 1.0f - normalizedSigned; // [0,1] -> [1,0]
            outR = 1.0f; // stays red
            outG = t; // 1 -> 0
            outB = t; // 1 -> 0
        }
    }


    inline bool saveGradientSignPNG(
        const std::filesystem::path &filePath,
        const std::vector<float> &inputRGBA,
        std::uint32_t imageWidth,
        std::uint32_t imageHeight,
        float adjointSamplesPerPixel = 32.0f,
        float absQuantile = 0.99f,
        bool flipY = false,
        bool useSeismic = true
    ) {
        if (!std::filesystem::exists(filePath.parent_path())) {
            std::filesystem::create_directories(filePath.parent_path());
        }
        if (imageWidth == 0 || imageHeight == 0) return false;

        const std::size_t pixelCount = std::size_t(imageWidth) * imageHeight;
        const std::size_t expectedSize = pixelCount * 4u;
        if (inputRGBA.size() < expectedSize) return false;

        // Collect finite absolute scalar values for robust scaling
        std::vector<float> finiteAbsScalars;
        finiteAbsScalars.reserve(pixelCount);

        for (std::size_t i = 0; i < pixelCount; ++i) {
            const float r = inputRGBA[i * 4 + 0] / adjointSamplesPerPixel;
            // Divide by m_settings.adjointSamplesPerPixel
            const float g = inputRGBA[i * 4 + 1] / adjointSamplesPerPixel;
            // Divide by m_settings.adjointSamplesPerPixel
            const float b = inputRGBA[i * 4 + 2] / adjointSamplesPerPixel;
            // Divide by m_settings.adjointSamplesPerPixel

            float scalarValue = (r + g + b) / 3.0f;
            if (std::isfinite(scalarValue)) {
                finiteAbsScalars.push_back(std::fabs(scalarValue));
            }
        }

        // Determine symmetric scale
        float scaleAbs = 1.0f;
        if (!finiteAbsScalars.empty()) {
            absQuantile = std::clamp(absQuantile, 0.0f, 1.0f);
            if (absQuantile < 1.0f) {
                const std::size_t k = static_cast<std::size_t>(
                    std::floor(absQuantile * (finiteAbsScalars.size() - 1))
                );
                std::nth_element(finiteAbsScalars.begin(),
                                 finiteAbsScalars.begin() + k,
                                 finiteAbsScalars.end());
                scaleAbs = finiteAbsScalars[k];
            } else {
                scaleAbs = *std::max_element(finiteAbsScalars.begin(), finiteAbsScalars.end());
            }
            if (!(scaleAbs > 0.0f) || !std::isfinite(scaleAbs)) scaleAbs = 1.0f;
        }

        const float invScale = 1.0f / scaleAbs;

        auto toByte = [](float v) -> std::uint8_t {
            v = std::clamp(v, 0.0f, 1.0f);
            return static_cast<std::uint8_t>(v * 255.0f + 0.5f);
        };

        std::vector<std::uint8_t> rgb8(pixelCount * 3u);
        for (std::size_t i = 0; i < pixelCount; ++i) {
            const float rIn = inputRGBA[i * 4 + 0] / adjointSamplesPerPixel;
            const float gIn = inputRGBA[i * 4 + 1] / adjointSamplesPerPixel;
            const float bIn = inputRGBA[i * 4 + 2] / adjointSamplesPerPixel;

            float scalarValue = (rIn + gIn + bIn) / 3.0f;
            if (!std::isfinite(scalarValue)) scalarValue = 0.0f;

            float normalizedSigned = std::clamp(scalarValue * invScale, -1.0f, 1.0f);

            float colR, colG, colB;
            useSeismic
                ? sampleSeismic(normalizedSigned, colR, colG, colB)
                : sampleBlueBlackRed(normalizedSigned, colR, colG, colB);


            rgb8[i * 3 + 0] = toByte(colR);
            rgb8[i * 3 + 1] = toByte(colG);
            rgb8[i * 3 + 2] = toByte(colB);
        }


        const int bytesPerRow = static_cast<int>(imageWidth * 3);
        const std::uint8_t *srcPtr = rgb8.data();

        std::vector<std::uint8_t> rowOrdered;
        if (flipY) {
            rowOrdered.resize(rgb8.size());
            for (std::uint32_t y = 0; y < imageHeight; ++y) {
                const std::uint32_t srcY = imageHeight - 1 - y;
                std::memcpy(rowOrdered.data() + y * bytesPerRow,
                            srcPtr + srcY * bytesPerRow,
                            bytesPerRow);
            }
            srcPtr = rowOrdered.data();
        }

        return stbi_write_png(filePath.c_str(),
                              static_cast<int>(imageWidth),
                              static_cast<int>(imageHeight),
                              3, srcPtr, bytesPerRow) != 0;
    }

    inline bool saveGradientSingleChannelPNG(
    const std::filesystem::path &filePath,
    const std::vector<float> &inputRGBA,
    std::uint32_t imageWidth,
    std::uint32_t imageHeight,
    std::uint32_t channelIndex,          // 0=R, 1=G, 2=B, 3=A
    float adjointSamplesPerPixel = 32.0f,
    float absQuantile = 0.99f,
    bool flipY = false,
    bool useSeismic = true
) {
        if (channelIndex > 3) {
            return false;
        }

        if (imageWidth == 0 || imageHeight == 0) {
            return false;
        }

        const std::size_t pixelCount = std::size_t(imageWidth) * imageHeight;
        const std::size_t expectedSize = pixelCount * 4u;
        if (inputRGBA.size() < expectedSize) {
            return false;
        }

        // Build a temporary RGBA where R=G=B = chosen channel, A=0.
        std::vector<float> replicatedRGBA(expectedSize);
        for (std::size_t i = 0; i < pixelCount; ++i) {
            const float channelValue = inputRGBA[i * 4 + channelIndex];
            if (channelValue > 0)
                int debug = 1;
            replicatedRGBA[i * 4 + 0] = channelValue; // R
            replicatedRGBA[i * 4 + 1] = channelValue; // G
            replicatedRGBA[i * 4 + 2] = channelValue; // B
            replicatedRGBA[i * 4 + 3] = 0.0f;         // A (unused)
        }

        // Reuse your existing robust scaling + seismic colormap pipeline.
        return saveGradientSignPNG(
            filePath,
            replicatedRGBA,
            imageWidth,
            imageHeight,
            adjointSamplesPerPixel,
            absQuantile,
            flipY,
            useSeismic
        );
    }


    inline bool saveGradientSignRGB(
        const std::filesystem::path &filePath,
        const std::vector<float> &inputRGB, // 3 floats per pixel
        std::uint32_t imageWidth,
        std::uint32_t imageHeight,
        float absQuantile = 0.99f,
        bool flipY = false // robust scale from abs-quantile in (0,1]; 1.0 = max-abs
    ) {
        if (!std::filesystem::exists(filePath.parent_path())) {
            std::filesystem::create_directories(filePath.parent_path());
        }
        if (imageWidth == 0 || imageHeight == 0) return false;

        const std::size_t pixelCount = std::size_t(imageWidth) * imageHeight;
        const std::size_t expectedSize = pixelCount * 3u;
        if (inputRGB.size() < expectedSize) return false;

        // Collect finite absolute scalar values for robust scaling
        std::vector<float> finiteAbsScalars;
        finiteAbsScalars.reserve(pixelCount);

        for (std::size_t i = 0; i < pixelCount; ++i) {
            const float r = inputRGB[i * 3 + 0];
            const float g = inputRGB[i * 3 + 1];
            const float b = inputRGB[i * 3 + 2];

            float scalarValue = (r + g + b) / 3.0f;
            if (std::isfinite(scalarValue)) {
                finiteAbsScalars.push_back(std::fabs(scalarValue));
            }
        }

        // Determine symmetric scale
        float scaleAbs = 1.0f;
        if (!finiteAbsScalars.empty()) {
            absQuantile = std::clamp(absQuantile, 0.0f, 1.0f);
            if (absQuantile < 1.0f) {
                const std::size_t k = static_cast<std::size_t>(
                    std::floor(absQuantile * (finiteAbsScalars.size() - 1))
                );
                std::nth_element(finiteAbsScalars.begin(),
                                 finiteAbsScalars.begin() + k,
                                 finiteAbsScalars.end());
                scaleAbs = finiteAbsScalars[k];
            } else {
                scaleAbs = *std::max_element(finiteAbsScalars.begin(), finiteAbsScalars.end());
            }
            if (!(scaleAbs > 0.0f) || !std::isfinite(scaleAbs)) scaleAbs = 1.0f;
        }

        const float invScale = 1.0f / scaleAbs;

        auto toByte = [](float v) -> std::uint8_t {
            v = std::clamp(v, 0.0f, 1.0f);
            return static_cast<std::uint8_t>(v * 255.0f + 0.5f);
        };

        std::vector<std::uint8_t> rgb8(pixelCount * 3u);
        for (std::size_t i = 0; i < pixelCount; ++i) {
            const float r = inputRGB[i * 3 + 0];
            const float g = inputRGB[i * 3 + 1];
            const float b = inputRGB[i * 3 + 2];

            float scalarValue = (r + g + b) / 3.0f;
            if (!std::isfinite(scalarValue)) scalarValue = 0.0f;

            float normalized = scalarValue * invScale;
            normalized = std::clamp(normalized, -1.0f, 1.0f);

            const float redMapped = normalized > 0.0f ? normalized : 0.0f;
            const float blueMapped = normalized < 0.0f ? -normalized : 0.0f;
            const float greenMapped = 0.0f;

            rgb8[i * 3 + 0] = toByte(redMapped);
            rgb8[i * 3 + 1] = toByte(greenMapped);
            rgb8[i * 3 + 2] = toByte(blueMapped);
        }

        const int bytesPerRow = static_cast<int>(imageWidth * 3);
        const std::uint8_t *srcPtr = rgb8.data();

        std::vector<std::uint8_t> rowOrdered;
        if (flipY) {
            rowOrdered.resize(rgb8.size());
            for (std::uint32_t y = 0; y < imageHeight; ++y) {
                const std::uint32_t srcY = imageHeight - 1 - y;
                std::memcpy(rowOrdered.data() + y * bytesPerRow,
                            srcPtr + srcY * bytesPerRow,
                            bytesPerRow);
            }
            srcPtr = rowOrdered.data();
        }

        return stbi_write_png(filePath.c_str(),
                              static_cast<int>(imageWidth),
                              static_cast<int>(imageHeight),
                              3, srcPtr, bytesPerRow) != 0;

    }
}
