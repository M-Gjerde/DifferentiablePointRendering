//
// Created by magnus on 8/27/25.
//

module;
#include <sycl/sycl.hpp>

module Pale.DeviceSelector;

import Pale.Log;

namespace Pale {

    DeviceSelector::DeviceSelector() {
        try {
            m_device = sycl::device(sycl::cpu_selector_v);
        } catch (const sycl::exception& exception_object) {
            Log::PA_ERROR("Cannot select a GPU: {}", exception_object.what());
            Log::PA_WARN("Falling back to CPU device");
            m_device = sycl::device(sycl::cpu_selector_v);
        }

        m_context = sycl::context{m_device};

        m_queue = sycl::queue{
            m_device,
            &DeviceSelector::asyncHandler,
             sycl::property_list{sycl::property::queue::in_order{}},
        };


        Log::PA_INFO("Using {}", m_device.get_info<sycl::info::device::name>());

        m_queue.submit([&](sycl::handler& commandGroupHandler){
            commandGroupHandler.single_task<class WarmupKernel>([](){});
        }).wait();


        Log::PA_INFO("Warmup kernel succeeded: {}", m_device.get_info<sycl::info::device::name>());
    }

    sycl::queue DeviceSelector::getQueue() {
        return m_queue;
    }

    void DeviceSelector::asyncHandler(sycl::exception_list exceptions) {
        for (const auto& exception_ptr : exceptions) {
            try { std::rethrow_exception(exception_ptr); }
            catch (const sycl::exception& e) { Log::PA_ERROR("SYCL async error: {}", e.what()); }
        }
    }

} // namespace Pale