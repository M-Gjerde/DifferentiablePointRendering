#pragma once

#include <chrono>
#include <future>
#include <utility>

namespace viewer {
    // The caller keeps exclusive ownership of the UI/GL context. During work,
    // pump only platform events: do not edit state read by the worker. Completion
    // (including exceptions) is joined before the caller resumes using that state.
    template<class Work, class PumpEvents>
    void runWithEventPump(Work&& work, PumpEvents&& pumpEvents) {
        auto task = std::async(std::launch::async, std::forward<Work>(work));
        while (task.wait_for(std::chrono::milliseconds(8)) != std::future_status::ready) {
            pumpEvents();
        }
        task.get();
    }
}
