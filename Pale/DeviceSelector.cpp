//
// Created by magnus on 8/27/25.
//

module;
#include <sycl/sycl.hpp>

module Pale.DeviceSelector;

import Pale.Log;

namespace Pale {

    DeviceSelector::DeviceSelector() {
        sycl::device d;

        try {
            d = sycl::device(sycl::gpu_selector_v);
        } catch (const sycl::exception& e) {
            Log::PA_ERROR("Cannot select a GPU: {}", e.what());
            Log::PA_WARN("Falling back to CPU device");
            d = sycl::device(sycl::cpu_selector_v);
        }

        m_syclDevice = d;
        Log::PA_INFO("Using {}", d.get_info<sycl::info::device::name>());
    }

    sycl::queue DeviceSelector::getQueue() {
        return sycl::queue(m_syclDevice);
    }

} // namespace Pale