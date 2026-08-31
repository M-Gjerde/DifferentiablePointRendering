#include "Core/ScopedTimer.h"

#include <atomic>
#include <mutex>

import Pale.Log;

namespace Pale::ScopedTimerDetail {
namespace {
    std::atomic_bool g_profilingEnabled{false};
    std::mutex g_recordsMutex;
    std::vector<ScopedTimerRecord> g_records;
    uint64_t g_nextSequence = 0u;
}

bool isLogLevelEnabled( int logLevel) noexcept {
    return Log::isLogLevelEnabled(logLevel);
}

bool isProfilingEnabled() noexcept {
    return g_profilingEnabled.load(std::memory_order_relaxed);
}

void setProfilingEnabled(bool enabled) {
    g_profilingEnabled.store(enabled, std::memory_order_relaxed);
}

void clearProfilingRecords() {
    std::lock_guard lock(g_recordsMutex);
    g_records.clear();
    g_nextSequence = 0u;
}

std::vector<ScopedTimerRecord> snapshotProfilingRecords() {
    std::lock_guard lock(g_recordsMutex);
    return g_records;
}

void recordDuration(std::string_view scopeName, double durationMs, spdlog::level::level_enum logLevel) {
    std::lock_guard lock(g_recordsMutex);
    ScopedTimerRecord record{};
    record.name.assign(scopeName.data(), scopeName.size());
    record.durationMs = durationMs;
    record.sequence = g_nextSequence++;
    record.logLevel = logLevel;
    g_records.push_back(std::move(record));
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
