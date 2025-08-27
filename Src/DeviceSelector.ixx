//
// Created by magnus on 8/27/25.
//

module;

#include <sycl/sycl.hpp>


export module Vale.DeviceSelector;


export namespace Vale {

    class DeviceSelector {
    public:
        DeviceSelector();

    private:
        sycl::device m_syclDevice;
    };


}

