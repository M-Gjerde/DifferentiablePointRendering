module; // ── global fragment

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

export module Pale.Log; // ── named module


export namespace Pale {
    class Log {
    public:
        static void init() {
            // create colour console sinks for core and client
            s_coreLogger = spdlog::stdout_color_mt("PALE_CORE");
            s_coreLogger->set_pattern("[CORE] [%H:%M:%S.%e] [%^%l%$] %v");
            s_coreLogger->set_level(spdlog::level::trace);

            s_coreLogger->info("Logger initialized");
        }
        template<typename... Args>
        static void PA_TRACE(spdlog::format_string_t<Args...> fmt,
                                  Args&&... args) {
            getCoreLogger()->trace(fmt, std::forward<Args>(args)...);
        }
        static void PA_TRACE(std::string_view msg){ getCoreLogger()->trace(msg);}

        static bool isTraceEnabled() {
            return s_coreLogger && s_coreLogger->should_log(spdlog::level::trace);
        }

        template<typename... Args>
        static void PA_DEBUG(spdlog::format_string_t<Args...> fmt,
                                  Args&&... args) {
            getCoreLogger()->debug(fmt, std::forward<Args>(args)...);
        }
        static void PA_DEBUG(std::string_view msg){ getCoreLogger()->debug(msg);}

        template<typename... Args>
        static void PA_INFO(spdlog::format_string_t<Args...> fmt,
                                  Args&&... args) {
            getCoreLogger()->info(fmt, std::forward<Args>(args)...);
        }
        static void PA_INFO(std::string_view msg){ getCoreLogger()->info(msg);}

        template<typename... Args>
        static void PA_WARN(spdlog::format_string_t<Args...> fmt,
                                  Args&&... args) {
            getCoreLogger()->warn(fmt, std::forward<Args>(args)...);
        }
        static void PA_WARN(std::string_view msg){ getCoreLogger()->warn(msg);}

        // 1) Plain message, no placeholders
        static void PA_ERROR(std::string_view msg)
        {
            getCoreLogger()->error(msg);
        }

        // 2) Message that contains `{}` placeholders
        template<typename... Args>
        static void PA_ERROR(spdlog::format_string_t<Args...> fmt,
                                  Args&&... args)
        {
            getCoreLogger()->error(fmt, std::forward<Args>(args)...);
        }


        template<typename... Args>
    static void PA_ASSERT(bool condition,
                      spdlog::format_string_t<Args...> fmt,
                      Args&&... args)
        {
            if (!condition) {
                Log::getCoreLogger()->error(fmt, std::forward<Args>(args)...);
                // choose one of these behaviors:
                std::abort(); // terminate immediately
                // throw std::runtime_error(getCoreLogger()->formatter()->format(fmt, ...));
            }
        }

        static void PA_ASSERT(bool condition, std::string_view msg) {
            if (!condition) {
                Log::getCoreLogger()->error(msg);
                std::abort();
            }
        }

    private:
        static std::shared_ptr<spdlog::logger> &getCoreLogger() { return s_coreLogger; }

        inline static std::shared_ptr<spdlog::logger> s_coreLogger{};
    };


} // namespace Vale
