#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace viewer {
    // Blender 5.3 config.ocio: scene-linear Rec.709 -> FilmLight E-Gamut ->
    // lg2 allocation -> AgX_Base_sRGB.cube (tetrahedral) -> Rec.1886 EOTF.
    // Return display-linear Rec.709 so the viewer can composite its own grid.
    class BlenderAgx {
        using Rgb = std::array<float, 3>;
        int extent = 0;
        std::vector<Rgb> entries;

    public:
        explicit BlenderAgx(const std::filesystem::path& path) {
            std::ifstream input(path);
            if (!input) throw std::runtime_error("Cannot open Blender AgX LUT: " + path.string());
            std::string line;
            bool titleMatches = false;
            while (std::getline(input, line)) {
                std::istringstream row(line);
                std::string tag;
                if (!(row >> tag) || tag.front() == '#') continue;
                if (tag == "TITLE") {
                    titleMatches = line.find("AgX_Formation sRGB") != std::string::npos;
                } else if (tag == "LUT_3D_SIZE") {
                    if (!(row >> extent) || extent < 2 || extent > 128 || !entries.empty())
                        throw std::runtime_error("Invalid Blender AgX LUT dimensions");
                    entries.reserve(extent * extent * extent);
                } else if (tag == "DOMAIN_MIN" || tag == "DOMAIN_MAX") {
                    const float expected = tag == "DOMAIN_MIN" ? 0.f : 1.f;
                    Rgb v{};
                    if (!(row >> v[0] >> v[1] >> v[2]) || v != Rgb{expected, expected, expected})
                        throw std::runtime_error("Unsupported Blender AgX LUT domain");
                } else {
                    Rgb v{};
                    std::istringstream values(line);
                    if (!extent || !(values >> v[0] >> v[1] >> v[2]) ||
                        !std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2]) ||
                        entries.size() >= static_cast<std::size_t>(extent * extent * extent))
                        throw std::runtime_error("Malformed Blender AgX LUT entry");
                    entries.push_back(v);
                }
            }
            if (!titleMatches || extent < 2 || entries.size() != static_cast<std::size_t>(extent * extent * extent))
                throw std::runtime_error("Expected Blender's complete AgX_Base_sRGB.cube");
        }

        Rgb displayLinear(float r, float g, float b) const {
            const Rgb egamut{
                .55937110f*r + .30478334f*g + .13584556f*b,
                .07622071f*r + .78797180f*g + .13580747f*b,
                .06552671f*r + .16454676f*g + .76992650f*b
            };
            Rgb fraction{};
            const std::array<int, 3> stride{1, extent, extent * extent};
            int base = 0;
            for (int c = 0; c < 3; ++c) {
                const float logValue = std::log2(std::fmax(egamut[c], 0.00017578125f));
                const float coord = std::clamp((logValue + 12.47393f) / 25.f, 0.f, 1.f) * (extent - 1);
                const int index = std::min(static_cast<int>(coord), extent - 2);
                base += index * stride[c];
                fraction[c] = coord - index;
            }
            std::array<int, 3> order{0, 1, 2};
            std::sort(order.begin(), order.end(), [&](int i, int j) { return fraction[i] > fraction[j]; });
            const int second = base + stride[order[0]];
            const int third = second + stride[order[1]];
            const int fourth = third + stride[order[2]];
            Rgb output{};
            for (int c = 0; c < 3; ++c) {
                const float value = entries[base][c] +
                    fraction[order[0]] * (entries[second][c] - entries[base][c]) +
                    fraction[order[1]] * (entries[third][c] - entries[second][c]) +
                    fraction[order[2]] * (entries[fourth][c] - entries[third][c]);
                output[c] = std::pow(std::max(value, 0.f), 2.4f);
            }
            return output;
        }
    };
}
