#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <new>
#include <utility>

namespace Pale::detail {
    // Own allocations until a complete device object has been constructed.
    // The release callback keeps this independent of a particular SYCL device
    // and lets allocation-failure paths be exercised without a GPU.
    template<std::size_t Capacity, class Release>
    class AllocationBatch {
        std::array<void*, Capacity> pointers_{};
        std::size_t count_ = 0;
        Release release_;
    public:
        explicit AllocationBatch(Release release) : release_(std::move(release)) {}
        AllocationBatch(const AllocationBatch&) = delete;
        AllocationBatch& operator=(const AllocationBatch&) = delete;
        ~AllocationBatch() noexcept {
            while(count_) {
                try { release_(pointers_[--count_]); } catch(...) {}
            }
        }
        template<class T> T* add(T* pointer) {
            if(!pointer)throw std::bad_alloc();
            // All production callers have a fixed, explicit allocation count.
            if(count_ == Capacity) {
                release_(pointer);
                throw std::bad_alloc();
            }
            pointers_[count_++] = pointer;
            return pointer;
        }
        void commit() noexcept { count_ = 0; }
    };

    // Allocate before releasing the old storage. A failed allocation preserves
    // the old pointer and the caller's count, permitting a later retry.
    template<class T, class Allocate, class Release>
    void replaceDeviceAllocation(T*& current, std::size_t count, Allocate allocate, Release release) {
        if(count > std::numeric_limits<std::size_t>::max()/sizeof(T))throw std::bad_array_new_length();
        AllocationBatch<1,Release> pending(release);
        T* replacement = count ? pending.add(allocate(count)) : nullptr;
        if(current)release(current);
        current = replacement;
        pending.commit();
    }
}
