#pragma once

#include <array>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace viewer {
    inline std::filesystem::path expandedPath(const std::string& text, const std::filesystem::path& base) {
        std::filesystem::path path(text);
        if (text == "~" || text.starts_with("~/")) {
            const char* home = std::getenv("HOME");
            if (!home) throw std::runtime_error("HOME is unavailable");
            path = std::filesystem::path(home) / (text.size() > 2 ? text.substr(2) : "");
        }
        return (path.is_absolute() ? path : base / path).lexically_normal();
    }

    inline std::string shellDisplay(const std::vector<std::string>& arguments) {
        std::string result;
        for (const auto& argument : arguments) {
            if (!result.empty()) result += ' ';
            result += '\'';
            for (char c : argument) result += c == '\'' ? "'\\''" : std::string(1, c);
            result += '\'';
        }
        return result; // Display/copy only. Execution never invokes a shell.
    }

    inline std::filesystem::path reserveDirectory(const std::filesystem::path& root, const char* prefix) {
        std::filesystem::create_directories(root);
        const auto stamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        for (unsigned i = 0; i < 100; ++i) {
            auto candidate = root / (std::string(prefix) + std::to_string(stamp) + "_" + std::to_string(i));
            if (std::filesystem::create_directory(candidate)) return candidate;
        }
        throw std::runtime_error("Could not reserve a fresh directory");
    }

    // Owns a private supervisor/process group, never adopts external processes.
    // The supervisor installs its parent-death protection before Python starts,
    // so a viewer crash cannot leave the optimizer or its workers using the GPU.
    class TrainingSession {
        pid_t pid_ = -1;
        int resultPipe_ = -1;
        bool shutdownRequested_ = false;
        static void terminateOwnedGroup(int) {
            // The supervisor becomes its own session/group leader first. This
            // signal handler uses only async-signal-safe system calls.
            (void)::kill(0, SIGKILL);
            ::_exit(128 + SIGKILL);
        }
        static void supervise(pid_t viewer, const char* cwd, char* const* argv,
                              int errors, int result, int log) {
            int failure = 0;
            if (::setsid() < 0) failure = errno;
            if (!failure) {
                struct sigaction action{};
                action.sa_handler = &terminateOwnedGroup;
                ::sigemptyset(&action.sa_mask);
                if (::sigaction(SIGUSR1, &action, nullptr) < 0 ||
                    ::prctl(PR_SET_PDEATHSIG, SIGUSR1) < 0) failure = errno;
                // A closed viewer result pipe must not terminate the supervisor
                // before it cleans its process group. Do not inherit a viewer
                // SIGCHLD handler into this small, synchronous supervisor.
                action.sa_handler = SIG_IGN;
                if (::sigaction(SIGPIPE, &action, nullptr) < 0) failure = errno;
                action.sa_handler = SIG_DFL;
                if (::sigaction(SIGCHLD, &action, nullptr) < 0) failure = errno;
                sigset_t unblock;
                ::sigemptyset(&unblock); ::sigaddset(&unblock, SIGUSR1);
                if (::sigprocmask(SIG_UNBLOCK, &unblock, nullptr) < 0) failure = errno;
                // PR_SET_PDEATHSIG is not retroactive if the parent died before
                // it was installed. No training descendants exist yet.
                if (::getppid() != viewer) terminateOwnedGroup(0);
            }
            if (failure) {
                (void)::write(errors, &failure, sizeof(failure));
                ::_exit(127);
            }
            const pid_t supervisor = ::getpid();
            const pid_t trainer = ::fork();
            if (trainer == 0) {
                ::close(result);
                // A supervisor failure also stops the direct optimizer. The
                // supervisor normally kills its entire group on viewer death.
                if (::prctl(PR_SET_PDEATHSIG, SIGKILL) < 0) failure = errno;
                if (::getppid() != supervisor) ::_exit(127);
                if (::chdir(cwd) < 0) failure = errno;
                const int input = ::open("/dev/null", O_RDONLY);
                if (input < 0 || ::dup2(input, STDIN_FILENO) < 0 ||
                    ::dup2(log, STDOUT_FILENO) < 0 || ::dup2(log, STDERR_FILENO) < 0) failure = errno;
                if (input >= 0) ::close(input);
                ::close(log);
                if (!failure) { ::execvp(argv[0], argv); failure = errno; }
                (void)::write(errors, &failure, sizeof(failure));
                ::_exit(127);
            }
            const int forkFailure = trainer < 0 ? errno : 0;
            ::close(log);
            if (trainer < 0) {
                failure = forkFailure;
                (void)::write(errors, &failure, sizeof(failure));
                ::_exit(127);
            }
            ::close(errors);
            int status = 0;
            pid_t waited;
            do { waited = ::waitpid(trainer, &status, 0); } while (waited < 0 && errno == EINTR);
            const int code = waited < 0 ? 127 : (WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status));
            // Relay the optimizer's real result before killing any remaining
            // group members. The supervisor is still the group leader here, so
            // its process-group ID cannot have been reused by an unrelated job.
            (void)::write(result, &code, sizeof(code));
            ::close(result);
            terminateOwnedGroup(0);
        }
    public:
        std::filesystem::path directory;
        std::filesystem::path runDirectory;
        std::vector<std::string> arguments;
        std::optional<int> exitCode;
        unsigned commandSequence = 0;

        TrainingSession() = default;
        TrainingSession(const TrainingSession&) = delete;
        TrainingSession& operator=(const TrainingSession&) = delete;
        ~TrainingSession() {
            shutdown();
            // shutdown() itself is nonblocking. Allow a small bounded reap at
            // destruction; a stuck kernel/driver must not hold up viewer exit.
            for (int i = 0; i < 20 && active(); ++i) {
                poll();
                if (active()) ::usleep(5000);
            }
            if (resultPipe_ >= 0) ::close(resultPipe_);
        }
        void shutdown() noexcept {
            if (pid_ > 0 && !exitCode && !shutdownRequested_) {
                shutdownRequested_ = true;
                // pid_ remains our unreaped supervisor/group leader. Killing
                // its private group also stops workers ignoring SIGTERM.
                (void)::kill(-pid_, SIGKILL);
            }
        }
        [[nodiscard]] bool active() const { return pid_ > 0 && !exitCode; }
        [[nodiscard]] bool launched() const { return pid_ > 0; }
        void poll() {
            if (!active()) return;
            int status = 0;
            const auto result = ::waitpid(pid_, &status, WNOHANG);
            if (result == pid_) {
                int code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
                if (resultPipe_ >= 0) {
                    int reported = code;
                    ssize_t received;
                    do { received = ::read(resultPipe_, &reported, sizeof(reported)); } while (received < 0 && errno == EINTR);
                    if (received == sizeof(reported)) code = reported;
                    ::close(resultPipe_); resultPipe_ = -1;
                }
                exitCode = code;
            }
        }
        void launch(const std::filesystem::path& outputRoot, const std::filesystem::path& workingDirectory,
                    std::vector<std::string> command) {
            poll();
            if (active()) throw std::runtime_error("A training process is already running");
            runDirectory = reserveDirectory(outputRoot, "run_");
            directory = reserveDirectory(outputRoot / ".viewer_sessions", "session_");
            command.push_back("-o"); command.push_back(runDirectory.string());
            command.push_back("--viewer-session"); command.push_back(directory.string());
            arguments = std::move(command);
            std::ofstream(directory / "launch_command.txt") << shellDisplay(arguments) << '\n';
            int errors[2];
            if (::pipe2(errors, O_CLOEXEC) != 0) throw std::runtime_error("Could not open launch pipe");
            int result[2];
            // poll() must never wait for a descriptor inherited by another
            // concurrently forked JIT/helper process. The supervisor writes one
            // atomic integer before exit, so a nonblocking result pipe suffices.
            if (::pipe2(result, O_CLOEXEC | O_NONBLOCK) != 0) {
                ::close(errors[0]); ::close(errors[1]);
                throw std::runtime_error("Could not open training result pipe");
            }
            const int log = ::open((directory / "training.log").c_str(), O_CREAT | O_WRONLY | O_APPEND, 0600);
            if (log < 0) { ::close(errors[0]); ::close(errors[1]); ::close(result[0]); ::close(result[1]); throw std::runtime_error("Cannot create training log"); }
            std::vector<char*> argv;
            for (auto& argument : arguments) argv.push_back(argument.data());
            argv.push_back(nullptr);
            const pid_t viewer = ::getpid();
            const pid_t child = ::fork();
            if (child == 0) {
                ::close(errors[0]); ::close(result[0]);
                supervise(viewer, workingDirectory.c_str(), argv.data(), errors[1], result[1], log);
                ::_exit(127);
            }
            ::close(errors[1]); ::close(log); ::close(result[1]);
            if (child < 0) {
                ::close(errors[0]); ::close(result[0]);
                throw std::runtime_error("Cannot start training process");
            }
            int failure = 0;
            ssize_t received;
            do { received = ::read(errors[0], &failure, sizeof(failure)); } while (received < 0 && errno == EINTR);
            ::close(errors[0]);
            if (received > 0) {
                int status;
                while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {}
                ::close(result[0]);
                throw std::runtime_error("Training launch failed: " + std::string(std::strerror(failure)));
            }
            if (resultPipe_ >= 0) ::close(resultPipe_);
            pid_ = child; resultPipe_ = result[0]; shutdownRequested_ = false;
            exitCode.reset(); commandSequence = 0;
        }
    };
}
