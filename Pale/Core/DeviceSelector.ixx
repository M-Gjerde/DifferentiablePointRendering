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
        static void asyncHandler(sycl::exception_list exceptions);

        sycl::device m_device;
        sycl::queue m_queue;
        sycl::context m_context;
    };


}

