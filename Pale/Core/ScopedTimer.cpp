#include "Core/ScopedTimer.h"

import Pale.Log;

namespace Pale::ScopedTimerDetail {

bool isTraceLoggingEnabled() noexcept {
    return Log::isTraceEnabled();
}

void logTraceDuration(std::string_view scopeName, double durationMs) {
    Log::PA_TRACE("{} took {:.3f} ms", scopeName, durationMs);
}

} // namespace Pale::ScopedTimerDetail

