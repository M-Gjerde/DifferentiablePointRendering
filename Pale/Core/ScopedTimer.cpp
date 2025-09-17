#include "Core/ScopedTimer.h"

import Pale.Log;

namespace Pale::ScopedTimerDetail {

bool isLogLevelEnabled( int logLevel) noexcept {
    return Log::isLogLevelEnabled(logLevel);
}

void logTraceDuration(std::string_view scopeName, double durationMs, spdlog::level::level_enum logLevel) {
    switch (logLevel) {
        case spdlog::level::trace:
            Log::PA_TRACE("{} took {:.3f} ms", scopeName, durationMs);
            break;
        case spdlog::level::debug:
            Log::PA_DEBUG("{} took {:.3f} ms", scopeName, durationMs);
            break;
        case spdlog::level::info:
            Log::PA_INFO("{} took {:.3f} ms", scopeName, durationMs);
            break;
        case spdlog::level::warn:
            Log::PA_WARN("{} took {:.3f} ms", scopeName, durationMs);
            break;
        case spdlog::level::err:
            Log::PA_ERROR("{} took {:.3f} ms", scopeName, durationMs);
            break;
        case spdlog::level::critical:
            break;
        case spdlog::level::off:
            break;
        case spdlog::level::n_levels:
            break;
    }
}

} // namespace Pale::ScopedTimerDetail

