#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace viewer {
    struct PointCloudSnapshot {
        std::filesystem::path path;
        uint64_t iteration = 0u;
        std::filesystem::file_time_type writeTime = std::filesystem::file_time_type::min();
    };
    [[nodiscard]] inline std::size_t checkpointNavigationIndex(const std::vector<PointCloudSnapshot>& saved,uint64_t current,int direction) {
        if(saved.empty())return 0;
        if(direction==3)return saved.size()-1;
        if(direction==-1) {
            auto it=std::lower_bound(saved.begin(),saved.end(),current,[](const auto& item,uint64_t step){return item.iteration<step;});
            return it==saved.begin()?0:static_cast<std::size_t>(it-saved.begin()-1);
        }
        if(direction==1) {
            auto it=std::upper_bound(saved.begin(),saved.end(),current,[](uint64_t step,const auto& item){return step<item.iteration;});
            return it==saved.end()?saved.size()-1:static_cast<std::size_t>(it-saved.begin());
        }
        const uint64_t target=direction<0?(current?((current-1)/1000)*1000:0):(current/1000+1)*1000;
        // Missing boundary checkpoints must still move in the requested
        // direction, rather than repeatedly selecting the current snapshot.
        const auto begin=direction<0?saved.begin():std::upper_bound(saved.begin(),saved.end(),current,[](uint64_t step,const auto& item){return step<item.iteration;});
        const auto end=direction<0?std::lower_bound(saved.begin(),saved.end(),current,[](const auto& item,uint64_t step){return item.iteration<step;}):saved.end();
        if(begin==end)return direction<0?0:saved.size()-1;
        auto it=std::lower_bound(begin,end,target,[](const auto& item,uint64_t step){return item.iteration<step;});
        if(it==end)return static_cast<std::size_t>(end-saved.begin()-1);
        if(it==begin)return static_cast<std::size_t>(begin-saved.begin());
        const auto previous=it-1;
        const bool preferPrevious=target-previous->iteration<it->iteration-target ||
            (target-previous->iteration==it->iteration-target && direction<0);
        return static_cast<std::size_t>((preferPrevious?previous:it)-saved.begin());
    }

    [[nodiscard]] inline std::filesystem::file_time_type lastWriteTimeOrMin(const std::filesystem::path& path) {
        std::error_code error;
        const std::filesystem::file_time_type time = std::filesystem::last_write_time(path, error);
        return error ? std::filesystem::file_time_type::min() : time;
    }

    [[nodiscard]] inline std::optional<uint64_t> parseIterationPointCloudFilename(const std::filesystem::path& path) {
        const std::string filename = path.filename().string();
        const std::string prefix = "iter_";
        const std::string suffix = "_points.ply";
        if (filename.rfind(prefix, 0u) != 0u ||
            filename.size() <= prefix.size() + suffix.size() ||
            filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) != 0) {
            return std::nullopt;
        }

        uint64_t iteration = 0u;
        for (std::size_t index = prefix.size(); index < filename.size() - suffix.size(); ++index) {
            const char c = filename[index];
            if (c < '0' || c > '9') {
                return std::nullopt;
            }
            iteration = iteration * 10u + static_cast<uint64_t>(c - '0');
        }
        return iteration;
    }

    [[nodiscard]] inline std::vector<PointCloudSnapshot> listOptimizationPointCloudSnapshots(
        const std::filesystem::path& pointsDirectory) {
        std::vector<PointCloudSnapshot> snapshots;
        std::error_code error;
        if (!std::filesystem::is_directory(pointsDirectory, error)) {
            return snapshots;
        }

        for (const auto& pointEntry : std::filesystem::directory_iterator(pointsDirectory, error)) {
            if (error) {
                break;
            }
            if (!pointEntry.is_regular_file(error) || error) {
                error.clear();
                continue;
            }

            const std::optional<uint64_t> iteration = parseIterationPointCloudFilename(pointEntry.path());
            if (!iteration) {
                continue;
            }
            snapshots.push_back({
                .path = pointEntry.path(),
                .iteration = *iteration,
                .writeTime = lastWriteTimeOrMin(pointEntry.path()),
            });
        }

        std::sort(snapshots.begin(), snapshots.end(), [](const PointCloudSnapshot& lhs, const PointCloudSnapshot& rhs) {
            if (lhs.iteration != rhs.iteration) {
                return lhs.iteration < rhs.iteration;
            }
            if (lhs.writeTime != rhs.writeTime) {
                return lhs.writeTime < rhs.writeTime;
            }
            return lhs.path.filename().string() < rhs.path.filename().string();
        });
        return snapshots;
    }

    [[nodiscard]] inline std::optional<std::filesystem::path> findLatestOptimizationPointCloud(
        const std::filesystem::path& optimizationOutputDirectory) {
        struct RunCandidate {
            std::filesystem::path runDirectory;
            std::filesystem::path metricsPath;
            PointCloudSnapshot pointCloud;
            std::filesystem::file_time_type metricsWriteTime = std::filesystem::file_time_type::min();
        };

        std::error_code error;
        if (!std::filesystem::is_directory(optimizationOutputDirectory, error)) {
            return std::nullopt;
        }

        std::optional<RunCandidate> bestRun;
        std::filesystem::recursive_directory_iterator runEntry(
            optimizationOutputDirectory,
            std::filesystem::directory_options::skip_permission_denied,
            error);
        const std::filesystem::recursive_directory_iterator end;
        while (runEntry != end) {
            std::error_code entryError;
            const std::filesystem::directory_entry entry = *runEntry;
            const bool isMetricsFile =
                entry.path().filename() == "metrics.csv" &&
                entry.is_regular_file(entryError) &&
                !entryError;

            runEntry.increment(error);
            if (error) {
                error.clear();
            }
            if (!isMetricsFile) {
                continue;
            }

            const std::filesystem::path metricsPath = entry.path();
            const std::filesystem::path runDirectory = metricsPath.parent_path();
            const std::filesystem::path pointsDirectory = runDirectory / "points";
            std::error_code pointsError;
            if (!std::filesystem::is_directory(pointsDirectory, pointsError) || pointsError) {
                continue;
            }

            const std::vector<PointCloudSnapshot> snapshots = listOptimizationPointCloudSnapshots(pointsDirectory);
            if (snapshots.empty()) {
                continue;
            }
            const PointCloudSnapshot& bestPointCloud = snapshots.back();

            RunCandidate candidate{
                .runDirectory = runDirectory,
                .metricsPath = metricsPath,
                .pointCloud = bestPointCloud,
                .metricsWriteTime = lastWriteTimeOrMin(metricsPath),
            };
            if (!bestRun ||
                candidate.metricsWriteTime > bestRun->metricsWriteTime ||
                (candidate.metricsWriteTime == bestRun->metricsWriteTime &&
                 candidate.metricsPath.string() > bestRun->metricsPath.string())) {
                bestRun = candidate;
            }
        }

        if (!bestRun) {
            return std::nullopt;
        }
        return bestRun->pointCloud.path;
    }

    // Accept a points directory, one run, or a collection of training runs.
    [[nodiscard]] inline std::optional<std::filesystem::path> findOptimizationPointsDirectory(
        const std::filesystem::path& folder) {
        std::error_code error;
        const auto directory = std::filesystem::canonical(folder, error);
        if (error || !std::filesystem::is_directory(directory, error)) {
            return std::nullopt;
        }
        if (!listOptimizationPointCloudSnapshots(directory).empty()) {
            return directory;
        }
        if (!listOptimizationPointCloudSnapshots(directory / "points").empty()) {
            return directory / "points";
        }
        if (const auto latest = findLatestOptimizationPointCloud(directory)) {
            return latest->parent_path();
        }
        return std::nullopt;
    }

    struct OptimizationFolderAnchor {
        std::filesystem::path folder;
        std::optional<std::filesystem::path> pointsDirectory;

        void set(const std::filesystem::path& path) {
            folder = std::filesystem::absolute(path).lexically_normal();
            pointsDirectory.reset();
        }

        void clear() {
            folder.clear();
            pointsDirectory.reset();
        }

        [[nodiscard]] bool active() const { return !folder.empty(); }

        // Resolve a collection once, then stay in that run even if another starts.
        // An empty folder can be retried when its first checkpoint is written.
        [[nodiscard]] std::optional<std::filesystem::path> resolve() {
            if (!pointsDirectory && active()) {
                pointsDirectory = findOptimizationPointsDirectory(folder);
            }
            return pointsDirectory;
        }
    };
} // namespace viewer
