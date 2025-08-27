//
// Created by magnus on 8/27/25.
//

module;
#include <sycl/sycl.hpp>

module Vale.DeviceSelector;

namespace Vale {
    DeviceSelector::DeviceSelector() {
        // Find appropriate sycl devices
        sycl::device d;

        try {
            d = sycl::device(sycl::gpu_selector_v);
        } catch (sycl::exception const &e) {
            std::cout << "Cannot select a GPU\n" << e.what() << "\n";
            std::cout << "Using a CPU device\n";
            d = sycl::device(sycl::cpu_selector_v);
        }
        m_syclDevice = d;
        std::cout << "Using " << d.get_info<sycl::info::device::name>();
    }
}
