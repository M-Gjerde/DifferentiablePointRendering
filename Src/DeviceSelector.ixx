//
// Created by magnus on 8/27/25.
//

module;

#include <sycl/sycl.hpp>


export module MG.DeviceSelector;


export namespace MG {

    class DeviceSelector {
    public:
        DeviceSelector();

        sycl::queue getQueue();

    private:
        sycl::device m_syclDevice;
    };


}

