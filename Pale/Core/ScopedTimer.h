#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "spdlog/spdlog.h"

namespace Pale {
struct ScopedTimerRecord {
    std::string name{};
    double durationMs = 0.0;
    uint64_t sequence = 0u;
    spdlog::level::level_enum logLevel = spdlog::level::trace;
};

namespace ScopedTimerDetail {
    [[nodiscard]] bool isLogLevelEnabled( int logLevel) noexcept;
    [[nodiscard]] bool isProfilingEnabled() noexcept;
    void setProfilingEnabled(bool enabled);
    void clearProfilingRecords();
    [[nodiscard]] std::vector<ScopedTimerRecord> snapshotProfilingRecords();
    void recordDuration(std::string_view scopeName, double durationMs, spdlog::level::level_enum logLevel);
    void logTraceDuration(std::string_view scopeName, double durationMs, spdlog::level::level_enum logLevel);
}

class ScopedTimer {
public:
    using Clock = std::chrono::steady_clock;

    /*
    explicit ScopedTimer(std::string_view name) : m_enabled(ScopedTimerDetail::isTraceLoggingEnabled()) {
        if (m_enabled) {
            m_name.assign(name.data(), name.size());
            m_start = Clock::now();
        }
    }
    */
    explicit ScopedTimer(std::string name, spdlog::level::level_enum logLevel = spdlog::level::trace)
        : m_logEnabled(ScopedTimerDetail::isLogLevelEnabled(logLevel)),
          m_profileEnabled(ScopedTimerDetail::isProfilingEnabled()),
          m_enabled(m_logEnabled || m_profileEnabled),
          m_logLevel(logLevel) {
        if (m_enabled) {
            m_name = std::move(name);
            m_start = Clock::now();
        }

    }

    ScopedTimer(const ScopedTimer &) = delete;
    ScopedTimer &operator=(const ScopedTimer &) = delete;

    ScopedTimer(ScopedTimer &&) = delete;
    ScopedTimer &operator=(ScopedTimer &&) = delete;

    ~ScopedTimer() {
        stop();
    }

    // End a stage before the surrounding function returns. Calling stop again
    // (including from the destructor) must not record the same stage twice.
    void stop() {
        if (!m_enabled) {
            return;
        }
        m_enabled = false;

        const auto end = Clock::now();
        const double durationMs = std::chrono::duration<double, std::milli>(end - m_start).count();
        if (m_profileEnabled) {
            ScopedTimerDetail::recordDuration(m_name, durationMs, m_logLevel);
        }
        if (m_logEnabled) {
            ScopedTimerDetail::logTraceDuration(m_name, durationMs, m_logLevel);
        }
    }

private:
    bool m_logEnabled{false};
    bool m_profileEnabled{false};
    bool m_enabled{false};
    std::string m_name{};
    Clock::time_point m_start{};
    spdlog::level::level_enum m_logLevel{spdlog::level::trace};
};

} // namespace Pale
