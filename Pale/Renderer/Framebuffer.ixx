//
// Created by magnus-desktop on 8/28/25.
//
module;
#include <sycl/sycl.hpp>;
#include <cstdint>;
#include <vector>;

export module Pale.Render.Framebuffer;

export namespace Pale {
    struct Framebuffer {
        sycl::float4* dev = nullptr;
        std::uint32_t width = 0, height = 0;
    };

    Framebuffer createFramebuffer(sycl::queue& q, std::uint32_t w, std::uint32_t h) {
        Framebuffer frameBuffer;
        frameBuffer.width = w; frameBuffer.height = h;
        frameBuffer.dev = static_cast<sycl::float4*>(sycl::malloc_device(sizeof(sycl::float4)*w*h, q));
        q.fill(frameBuffer.dev, sycl::float4{0,0,0,0}, w*h).wait();
        return frameBuffer;
    }

    void destroyFramebuffer(sycl::queue& q, Framebuffer& frameBuffer) {
        if (frameBuffer.dev) { sycl::free(frameBuffer.dev, q); frameBuffer.dev=nullptr; }
    }

    std::vector<float> downloadRGBA(sycl::queue& q, const Framebuffer& frameBuffer) {
        std::vector<float> rgba(std::size_t(frameBuffer.width)*frameBuffer.height*4u);
        q.memcpy(rgba.data(), frameBuffer.dev, rgba.size()*sizeof(float)).wait();
        return rgba;
    }
} // namespace Pale
