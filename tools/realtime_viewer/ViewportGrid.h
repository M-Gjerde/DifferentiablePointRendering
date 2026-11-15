#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>

namespace viewer {
    [[nodiscard]] inline float chooseGridSpacing(float extent) {
        const float rawSpacing = std::max(extent / 80.0f, 0.001f);
        const float magnitude = std::pow(10.0f, std::floor(std::log10(rawSpacing)));
        const float normalized = rawSpacing / magnitude;
        if (normalized <= 1.0f) {
            return magnitude;
        }
        if (normalized <= 2.0f) {
            return 2.0f * magnitude;
        }
        if (normalized <= 5.0f) {
            return 5.0f * magnitude;
        }
        return 10.0f * magnitude;
    }

    struct GridLineStyle {
        glm::vec3 color{1.0f};
        float alpha = 1.0f;
        float thickness = 1.0f;
    };

    [[nodiscard]] inline uint8_t channelToByte(float value) {
        return static_cast<uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
    }

    inline void blendPixel(std::vector<uint8_t>& rgba, std::size_t pixelIndex, const GridLineStyle& style, float coverage) {
        const float alpha = std::clamp(style.alpha * coverage, 0.0f, 1.0f);
        if (alpha <= 0.0f) {
            return;
        }

        const std::size_t baseIndex = pixelIndex * 4u;
        const float invAlpha = 1.0f - alpha;
        rgba[baseIndex + 0u] = channelToByte(
            static_cast<float>(rgba[baseIndex + 0u]) / 255.0f * invAlpha + style.color.r * alpha);
        rgba[baseIndex + 1u] = channelToByte(
            static_cast<float>(rgba[baseIndex + 1u]) / 255.0f * invAlpha + style.color.g * alpha);
        rgba[baseIndex + 2u] = channelToByte(
            static_cast<float>(rgba[baseIndex + 2u]) / 255.0f * invAlpha + style.color.b * alpha);
        rgba[baseIndex + 3u] = 255u;
    }

    [[nodiscard]] inline float gridLineCoverage(float pixelDistance, float thickness) {
        // Integrate a one-pixel footprint across the line edge. The ramp keeps
        // subpixel motion smooth without changing the line's apparent width.
        return std::clamp(0.5f * thickness + 0.5f - pixelDistance, 0.0f, 1.0f);
    }

    [[nodiscard]] inline float gridSmoothFade(float value) {
        const float t=std::clamp(value,0.0f,1.0f);
        return t*t*(3.0f-2.0f*t);
    }

    [[nodiscard]] inline float blendGridCoordinate(std::vector<uint8_t>& rgba,std::size_t pixelIndex,
        float coordinate,float worldPerPixel,float spacing,float invSpacing,float depthFade,
        const GridLineStyle& axis,const GridLineStyle& major,const GridLineStyle& minor) {
        const float invPixel=1.0f/std::max(worldPerPixel,1.0e-6f);
        const float gridCoordinate=coordinate*invSpacing;
        const float minorDistance=std::abs(coordinate-std::round(gridCoordinate)*spacing)*invPixel;
        const float majorDistance=std::abs(coordinate-std::round(gridCoordinate*.1f)*(10.f*spacing))*invPixel;
        // Remove unresolved fine lines before they turn into a solid band or
        // moire near the horizon. Major lines have their own, coarser footprint.
        const float period=spacing*invPixel;
        const float minorCoverage=gridLineCoverage(minorDistance,minor.thickness)*gridSmoothFade((period-2.f)/3.f);
        const float majorCoverage=gridLineCoverage(majorDistance,major.thickness)*gridSmoothFade((10.f*period-2.f)/3.f);
        const float axisCoverage=gridLineCoverage(std::abs(coordinate)*invPixel,axis.thickness);
        blendPixel(rgba,pixelIndex,minor,depthFade*minorCoverage*(1.f-majorCoverage)*(1.f-axisCoverage));
        blendPixel(rgba,pixelIndex,major,depthFade*majorCoverage*(1.f-axisCoverage));
        return axisCoverage;
    }

    struct GridProjection {
        glm::mat4 worldFromCamera{1.0f};
        float fx=1, fy=1, cx=0, cy=0;
        bool operator==(const GridProjection&) const = default;
    };

    template<class OrbitCamera>
    [[nodiscard]] GridProjection orbitGridProjection(const OrbitCamera& orbit,uint32_t width,uint32_t height) {
        const float focal=0.5f*std::max(height,1u)/std::tan(0.5f*glm::radians(orbit.fovyDegrees));
        return {glm::inverse(orbit.viewMatrix()),focal,focal,0.5f*width,0.5f*height};
    }

    [[nodiscard]] inline glm::vec3 gridRayDirection(const GridProjection& projection,float x,float y) {
        // Identical pinhole convention and zero-jitter pixel coordinates to the
        // renderer. Plane intersections do not require normalizing this ray.
        return glm::mat3(projection.worldFromCamera)*glm::vec3{
            (x-projection.cx)/projection.fx,(projection.cy-y)/projection.fy,-1.0f};
    }

    template<class OrbitCamera, class SceneBounds, class BackgroundColor>
    void shadeViewportBackground(
        std::vector<uint8_t>& rgba,
        const OrbitCamera& orbit,
        const SceneBounds& bounds,
        uint32_t renderWidth,
        uint32_t renderHeight,
        bool showViewportGrid,
        const BackgroundColor& backgroundColor,
        const GridProjection& projection) {
        const float extent = std::max(
            10.0f,
            std::max(bounds.radius * 2.5f, orbit.distance * 2.5f));
        const float spacing = chooseGridSpacing(extent);
        const float invSpacing=1.f/spacing;
        const GridLineStyle minorLine{{0.29f, 0.29f, 0.29f}, 0.42f, 1.0f};
        const GridLineStyle majorLine{{0.38f, 0.38f, 0.38f}, 0.57f, 1.15f};
        const GridLineStyle xAxisLine{{0.75f, 0.29f, 0.29f}, 0.82f, 3.0f};
        const GridLineStyle yAxisLine{{0.32f, 0.59f, 0.32f}, 0.82f, 3.0f};
        const glm::vec3 cameraPosition=glm::vec3(projection.worldFromCamera[3]);
        const glm::vec3 cameraForward=glm::normalize(-glm::vec3(projection.worldFromCamera[2]));
        const glm::vec3 rayDx=glm::vec3(projection.worldFromCamera[0])/projection.fx;
        const glm::vec3 rayDy=-glm::vec3(projection.worldFromCamera[1])/projection.fy;
        const float focusDistance=std::max({orbit.distance,bounds.radius,1.0e-3f});
        const float fadeStart=focusDistance*1.25f;
        const float invFadeRange=1.0f/(focusDistance*4.75f);
        const std::size_t pixelCount =
            static_cast<std::size_t>(renderWidth) * static_cast<std::size_t>(renderHeight);
        const uint8_t backgroundR = channelToByte(backgroundColor.x);
        const uint8_t backgroundG = channelToByte(backgroundColor.y);
        const uint8_t backgroundB = channelToByte(backgroundColor.z);

        // Independent rows; keep the worker count modest alongside rendering/training.
        #pragma omp parallel for schedule(static) num_threads(4) if(pixelCount >= 262144)
        for (int y = 0; y < static_cast<int>(renderHeight); ++y) {
            for (uint32_t x = 0; x < renderWidth; ++x) {
                const std::size_t pixelIndex =
                    static_cast<std::size_t>(y) * static_cast<std::size_t>(renderWidth) + static_cast<std::size_t>(x);
                if (pixelIndex >= pixelCount) {
                    continue;
                }

                const std::size_t baseIndex = pixelIndex * 4u;
                rgba[baseIndex + 0u] = backgroundR;
                rgba[baseIndex + 1u] = backgroundG;
                rgba[baseIndex + 2u] = backgroundB;
                rgba[baseIndex + 3u] = 255u;
                if (!showViewportGrid) {
                    continue;
                }

                const glm::vec3 rayDirection=gridRayDirection(projection,static_cast<float>(x),static_cast<float>(y));
                const float rayLengthSquared = glm::dot(rayDirection, rayDirection);
                if (rayDirection.z * rayDirection.z <= 1.0e-12f * rayLengthSquared) {
                    continue;
                }

                const float invRayZ=1.f/rayDirection.z;
                const float t = -cameraPosition.z*invRayZ;
                if (t <= 0.0f || t * t * rayLengthSquared <= orbit.nearClip * orbit.nearClip) {
                    continue;
                }

                const float forwardDepth = t*glm::dot(rayDirection,cameraForward);
                if (forwardDepth <= orbit.nearClip) {
                    continue;
                }
                const float depthFade=1.f-gridSmoothFade((forwardDepth-fadeStart)*invFadeRange);
                if(depthFade<=0)continue;
                const glm::vec3 hit=cameraPosition+rayDirection*t;
                // Differentiate the ray/ground intersection itself. Using only
                // focal depth ignores foreshortening and makes axes subpixel at
                // grazing camera angles, even when their nominal width is >1px.
                const glm::vec3 hitDx=t*(rayDx-rayDirection*(rayDx.z*invRayZ));
                const glm::vec3 hitDy=t*(rayDy-rayDirection*(rayDy.z*invRayZ));
                const float pixelX=glm::length(glm::vec2(hitDx.x,hitDy.x));
                const float pixelY=glm::length(glm::vec2(hitDx.y,hitDy.y));
                const float yAxis=blendGridCoordinate(rgba,pixelIndex,hit.x,pixelX,spacing,invSpacing,depthFade,yAxisLine,majorLine,minorLine);
                const float xAxis=blendGridCoordinate(rgba,pixelIndex,hit.y,pixelY,spacing,invSpacing,depthFade,xAxisLine,majorLine,minorLine);
                // Draw both colored axes last so perpendicular gray grid lines
                // do not punctuate them with dim spots at every crossing.
                blendPixel(rgba,pixelIndex,yAxisLine,depthFade*yAxis);
                blendPixel(rgba,pixelIndex,xAxisLine,depthFade*xAxis);
            }
        }
    }


    template<class OrbitCamera, class SceneBounds, class BackgroundColor>
    void shadeViewportBackground(std::vector<uint8_t>& rgba,const OrbitCamera& orbit,const SceneBounds& bounds,
            uint32_t width,uint32_t height,bool grid,const BackgroundColor& color) {
        shadeViewportBackground(rgba,orbit,bounds,width,height,grid,color,orbitGridProjection(orbit,width,height));
    }

    // Repeated renders and display/exposure changes reuse the background. Camera,
    // dimensions, bounds and appearance are all part of the cache key.
    class ViewportBackgroundCache {
        struct Key {
            GridProjection projection;
            glm::vec3 color;
            float radius, distance, nearClip;
            uint32_t width, height;
            bool grid;
            bool operator==(const Key&) const = default;
        };
        std::optional<Key> key;
        std::vector<uint8_t> background;
    public:
        template<class OrbitCamera, class SceneBounds, class BackgroundColor>
        const std::vector<uint8_t>& get(const OrbitCamera& orbit, const SceneBounds& bounds,
                uint32_t width, uint32_t height, bool grid, const BackgroundColor& color) {
            return get(orbit,bounds,width,height,grid,color,orbitGridProjection(orbit,width,height));
        }
        template<class OrbitCamera, class SceneBounds, class BackgroundColor>
        const std::vector<uint8_t>& get(const OrbitCamera& orbit,const SceneBounds& bounds,
                uint32_t width,uint32_t height,bool grid,const BackgroundColor& color,const GridProjection& projection) {
            const Key next{projection, {color.x,color.y,color.z},bounds.radius,orbit.distance,orbit.nearClip,width,height,grid};
            if (!key || *key != next) {
                background.resize(static_cast<std::size_t>(width) * height * 4u);
                shadeViewportBackground(background, orbit, bounds, width, height, grid, color, projection);
                key = next;
            }
            return background;
        }
    };
}
