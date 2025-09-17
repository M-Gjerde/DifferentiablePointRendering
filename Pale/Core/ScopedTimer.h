#pragma once

#include <chrono>
#include <string>
#include <string_view>
#include <utility>
#include "spdlog/spdlog.h"

namespace Pale {
namespace ScopedTimerDetail {
    [[nodiscard]] bool isLogLevelEnabled( int logLevel) noexcept;
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
    explicit ScopedTimer(std::string name, spdlog::level::level_enum logLevel = spdlog::level::trace) : m_enabled(ScopedTimerDetail::isLogLevelEnabled(logLevel)), m_logLevel(logLevel) {
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
        if (!m_enabled) {
            return;
        }

        const auto end = Clock::now();
        const double durationMs = std::chrono::duration<double, std::milli>(end - m_start).count();
        ScopedTimerDetail::logTraceDuration(m_name, durationMs, m_logLevel);
    }

private:
    bool m_enabled{false};
    std::string m_name{};
    Clock::time_point m_start{};
    spdlog::level::level_enum m_logLevel{spdlog::level::trace};
};

} // namespace Pale

