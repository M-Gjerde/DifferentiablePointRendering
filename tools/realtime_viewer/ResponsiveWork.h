#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <optional>
#include <string>
#include <stdexcept>
#include <thread>
#include <utility>

namespace viewer {
    // Retain the prior CPU presentation through both worker completion and GL
    // upload. A later metadata/export failure must not roll back an image that
    // has already been presented successfully.
    template<class State> class PreviewPublication {
        std::optional<State> prior_;
    public:
        void begin(State prior) {
            if(prior_)throw std::logic_error("Preview publication already pending");
            prior_.emplace(std::move(prior));
        }
        template<class Restore> void finish(bool presented,Restore&& restore) {
            if(prior_&&!presented)restore(std::move(*prior_));
            prior_.reset();
        }
        [[nodiscard]] bool pending() const {return prior_.has_value();}
    };

    // Requested generations and uploaded generations are separate. Navigation
    // can supersede a CPU-loaded scene before that scene reaches the GPU.
    [[nodiscard]] inline bool canPollTrainingScene(bool uploadPending, unsigned long long loadedGeneration) {
        return !uploadPending || loadedGeneration == 0;
    }

    class PreviewRecovery {
    public:
        using Clock = std::chrono::steady_clock;
        static constexpr unsigned automaticRetries = 3;
    private:
        unsigned failures_ = 0;
        Clock::time_point retryAt_{};
        std::string error_;
        bool failed_ = false;
    public:
        void fail(std::string error, Clock::time_point now = Clock::now()) {
            error_ = std::move(error); failed_ = true; ++failures_;
            const auto delay = std::chrono::milliseconds(250u << std::min(failures_ - 1, 3u));
            retryAt_ = now + delay;
        }
        [[nodiscard]] bool dispatchAllowed(Clock::time_point now = Clock::now()) const {
            return !failed_ || (failures_ <= automaticRetries && now >= retryAt_);
        }
        [[nodiscard]] bool retryDue(Clock::time_point now = Clock::now()) const {
            return failed_ && dispatchAllowed(now);
        }
        [[nodiscard]] bool failed() const { return failed_; }
        [[nodiscard]] bool paused() const { return failed_ && failures_ > automaticRetries; }
        [[nodiscard]] unsigned failures() const { return failures_; }
        [[nodiscard]] const std::string& error() const { return error_; }
        // Keep the explanation until dismissed, even if a fast retry succeeds.
        void succeeded() { failed_ = false; failures_ = 0; }
        void manualRetry() { failures_ = 0; retryAt_ = {}; }
        void dismiss() { if(!failed_)error_.clear(); }
    };

    class PreviewWatchdog {
    public:
        using Clock = std::chrono::steady_clock;
        static constexpr auto reportAfter = std::chrono::seconds(15);
    private:
        Clock::time_point began_{};
        bool active_ = false, reported_ = false;
    public:
        void begin(Clock::time_point now = Clock::now()) { began_=now;active_=true;reported_=false; }
        void finish() { active_=reported_=false; }
        [[nodiscard]] bool reportDue(Clock::time_point now = Clock::now()) {
            if(!active_||reported_||now-began_<reportAfter)return false;
            reported_=true;return true;
        }
        [[nodiscard]] bool longRunning() const { return active_&&reported_; }
        [[nodiscard]] long long elapsedMilliseconds(Clock::time_point now = Clock::now()) const {
            return std::chrono::duration_cast<std::chrono::milliseconds>(now-began_).count();
        }
    };

    // A frame-loop worker: start() never waits for GPU work and finish() is
    // called only after ready(), before handing the scene back to the UI.
    class PreviewWork {
        std::future<void> task_;
        std::thread worker_;
        std::mutex mutex_;
        std::condition_variable workAvailable_;
        std::packaged_task<void()> pending_;
        bool stopping_=false;
        void run() {
            for(;;) {
                std::packaged_task<void()> work;
                {
                    std::unique_lock lock(mutex_);
                    workAvailable_.wait(lock,[&] {return stopping_||pending_.valid();});
                    if(stopping_&&!pending_.valid())return;
                    work=std::move(pending_);
                }
                // User exceptions reach finish() through the future. An
                // unexpected invocation failure releases its promise too.
                try {work();}catch(...){}
            }
        }
    public:
        PreviewWork()=default;
        PreviewWork(const PreviewWork&)=delete;
        PreviewWork& operator=(const PreviewWork&)=delete;
        ~PreviewWork() {
            try {finish();}catch(...){}
            {std::lock_guard lock(mutex_);stopping_=true;}
            workAvailable_.notify_one();
            if(worker_.joinable())worker_.join();
        }
        [[nodiscard]] bool busy() const {return task_.valid();}
        [[nodiscard]] bool ready() const {return busy()&&task_.wait_for(std::chrono::milliseconds(0))==std::future_status::ready;}
        template<class Work> void start(Work&& work) {
            if(busy())throw std::logic_error("Preview work already in progress");
            std::packaged_task<void()> packaged(std::forward<Work>(work));
            auto result=packaged.get_future();
            {
                std::lock_guard lock(mutex_);
                if(!worker_.joinable())worker_=std::thread([this] {run();});
                pending_=std::move(packaged);task_=std::move(result);
            }
            workAvailable_.notify_one();
        }
        void finish() {if(task_.valid())task_.get();}
    };
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
