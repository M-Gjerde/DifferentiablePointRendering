#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <sycl/sycl.hpp>

namespace Pale {
    // Host-side observation of the first dispatch of each kernel in an adjoint
    // pass. The runtime may compile or load cached code before dispatch.
    void enableAdjointKernelProgress(bool enabled);
    [[nodiscard]] std::string adjointKernelProgressMessage();

    class AdjointKernelProgressScope {
    public:
        explicit AdjointKernelProgressScope(const sycl::queue& queue);
        ~AdjointKernelProgressScope();
        AdjointKernelProgressScope(const AdjointKernelProgressScope&) = delete;
        AdjointKernelProgressScope& operator=(const AdjointKernelProgressScope&) = delete;

    private:
        std::uint64_t pass_ = 0;
        int exceptions_ = 0;
    };

    class AdjointKernelDispatch {
    public:
        explicit AdjointKernelDispatch(std::string_view name);
        void submitted(const sycl::event& event) const;

    private:
        std::size_t index_ = 0;
        std::uint64_t pass_ = 0;
    };
}
