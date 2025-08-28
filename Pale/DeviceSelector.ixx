//
// Created by magnus on 8/27/25.
//

module;

#include <sycl/sycl.hpp>


export module Pale.DeviceSelector;


export namespace Pale {

    class DeviceSelector {
    public:
        DeviceSelector();

        sycl::queue getQueue();

    private:
        sycl::device m_syclDevice;
    };


}

