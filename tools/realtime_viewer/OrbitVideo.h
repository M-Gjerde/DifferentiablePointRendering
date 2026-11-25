#pragma once

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <glm/vec3.hpp>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include "TrainingSession.h"

namespace viewer {
    // Sample [0, 360) so playback can loop without repeating the first frame.
    inline glm::vec3 orbitVideoEyeDirection(glm::vec3 initialOffset, float elevation, int frame, int frameCount) {
        constexpr float radiansPerDegree = std::numbers::pi_v<float> / 180.0f;
        const float yaw = std::atan2(initialOffset.y, initialOffset.x) +
            2.0f * std::numbers::pi_v<float> * static_cast<float>(frame) / static_cast<float>(frameCount);
        const float pitch = elevation * radiansPerDegree;
        const float horizontal = std::cos(pitch);
        return {horizontal * std::cos(yaw), horizontal * std::sin(yaw), std::sin(pitch)};
    }

    class OrbitVideoEncoder {
        const std::atomic<bool>& cancel_;
        std::filesystem::path staging_, output_;
        std::size_t frameBytes_ = 0;
        int socket_ = -1;
        pid_t child_ = -1;

        void closeInput() noexcept {
            if (socket_ >= 0) ::close(socket_);
            socket_ = -1;
        }

        [[noreturn]] void encoderFailed() const {
            std::ifstream log(staging_ / "ffmpeg.log");
            std::string details(600, '\0');
            log.read(details.data(), static_cast<std::streamsize>(details.size()));
            details.resize(static_cast<std::size_t>(log.gcount()));
            throw std::runtime_error(details.empty() ? "FFmpeg could not encode the orbit video" : details);
        }

    public:
        explicit OrbitVideoEncoder(const std::atomic<bool>& cancel) : cancel_(cancel) {}
        OrbitVideoEncoder(const OrbitVideoEncoder&) = delete;
        OrbitVideoEncoder& operator=(const OrbitVideoEncoder&) = delete;
        ~OrbitVideoEncoder() {
            closeInput();
            if (child_ > 0) {
                ::kill(child_, SIGKILL);
                while (::waitpid(child_, nullptr, 0) < 0 && errno == EINTR) {}
            }
            if (!staging_.empty()) {
                std::error_code error;
                std::filesystem::remove_all(staging_, error);
            }
        }

        void start(const std::filesystem::path& directory, int width, int height, int fps) {
            frameBytes_ = static_cast<std::size_t>(width) * height * 4u;
            if (!std::filesystem::is_directory(directory))
                throw std::runtime_error("The reconstruction output folder is unavailable");
            staging_ = reserveDirectory(directory, ".orbit_");
            output_ = directory / (staging_.filename().string().substr(1) + ".mp4");
            const auto log = staging_ / "ffmpeg.log";
            const auto encoded = staging_ / "video.mp4";
            std::vector<std::string> arguments{
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                "-f", "rawvideo", "-pixel_format", "rgba", "-video_size",
                std::to_string(width) + "x" + std::to_string(height), "-framerate", std::to_string(fps),
                "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", encoded.string()
            };
            std::vector<char*> argv;
            for (auto& argument : arguments) argv.push_back(argument.data());
            argv.push_back(nullptr);

            int pair[2];
            if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, pair) < 0)
                throw std::system_error(errno, std::generic_category(), "Open FFmpeg input");
            socket_ = pair[0];
            posix_spawn_file_actions_t actions;
            int error = ::posix_spawn_file_actions_init(&actions);
            if (!error) {
                error = ::posix_spawn_file_actions_adddup2(&actions, pair[1], STDIN_FILENO);
                if (!error) error = ::posix_spawn_file_actions_addopen(&actions, STDERR_FILENO,
                    log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
                if (!error) error = ::posix_spawnp(&child_, argv.front(), &actions, nullptr, argv.data(), environ);
                ::posix_spawn_file_actions_destroy(&actions);
            }
            ::close(pair[1]);
            if (error) {
                child_ = -1;
                if (error == ENOENT) throw std::runtime_error("Install FFmpeg with libx264 and put ffmpeg on PATH");
                throw std::system_error(error, std::generic_category(), "Start FFmpeg");
            }
        }

        bool writeFrame(std::span<const uint8_t> pixels) const {
            if (pixels.size() != frameBytes_) throw std::invalid_argument("Orbit frame has the wrong pixel count");
            while (!pixels.empty() && !cancel_.load()) {
                const auto written = ::send(socket_, pixels.data(), pixels.size(), MSG_DONTWAIT | MSG_NOSIGNAL);
                if (written > 0) pixels = pixels.subspan(static_cast<std::size_t>(written));
                else if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                    pollfd writable{socket_, POLLOUT, 0};
                    if (::poll(&writable, 1, 100) < 0 && errno != EINTR) encoderFailed();
                } else if (written == 0 || errno != EINTR) {
                    encoderFailed();
                }
            }
            return !cancel_.load();
        }

        std::filesystem::path finish() {
            closeInput();
            while (!cancel_.load()) {
                int status;
                const pid_t result = ::waitpid(child_, &status, WNOHANG);
                if (result == child_) {
                    child_ = -1;
                    if (cancel_.load()) break;
                    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) encoderFailed();
                    // Publish only a complete MP4 and never replace an existing render.
                    std::filesystem::create_hard_link(staging_ / "video.mp4", output_);
                    return output_;
                }
                if (result < 0 && errno != EINTR) {
                    const int error = errno;
                    if (error == ECHILD) child_ = -1;
                    throw std::system_error(error, std::generic_category(), "Wait for FFmpeg");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            return {};
        }
    };
}
