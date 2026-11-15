#pragma once
#include "OptimizationSnapshots.h"
#include "TrainingMetrics.h"
#include "TrainingLog.h"
#include "FileTimes.h"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <cmath>
#include <stdexcept>

namespace viewer {
    struct ReconstructionEntry {
        std::filesystem::path directory;
        std::string label;
        bool hasLoss=false,hasGeometry=false;
        FolderTimestamp created;
    };

    inline YAML::Node reconstructionJson(const std::filesystem::path& file) {
        try {return YAML::LoadFile(file.string());}catch(...) {return {};}
    }
    inline YAML::Node reconstructionSetting(const YAML::Node& config,const char* name) {
        if(config.IsMap()) {
            if(config[name])return config[name];
            if(config["optimization_config"].IsMap())return config["optimization_config"][name];
        }
        return {};
    }
    inline bool reconstructionFile(const std::filesystem::path& path) {
        std::error_code error;return std::filesystem::is_regular_file(path,error);
    }
    inline bool isReconstructionDirectory(const std::filesystem::path& path) {
        return reconstructionFile(path/"run_config.json")||reconstructionFile(path/"metrics.csv")||
            reconstructionFile(path/"points_final.ply");
    }
    inline std::filesystem::path reconstructionDirectory(std::filesystem::path path) {
        path=std::filesystem::absolute(path).lexically_normal();
        if(path.filename()=="points"&&isReconstructionDirectory(path.parent_path()))path=path.parent_path();
        return path;
    }
    inline ReconstructionEntry reconstructionEntry(const std::filesystem::path& directory) {
        ReconstructionEntry entry;entry.directory=reconstructionDirectory(directory);
        entry.created=folderTimestamp(entry.directory);
        const auto config=reconstructionJson(entry.directory/"run_config.json");
        const auto dataset=reconstructionSetting(config,"dataset_path");
        entry.label=dataset.IsScalar()?std::filesystem::path(dataset.as<std::string>()).filename().string():entry.directory.filename().string();
        if(entry.label.empty())entry.label=entry.directory.filename().string();
        entry.hasLoss=reconstructionFile(entry.directory/"metrics.csv");
        entry.hasGeometry=reconstructionFile(entry.directory/"geometry_metrics.csv");
        return entry;
    }
    inline std::vector<ReconstructionEntry> scanReconstructions(const std::filesystem::path& root) {
        std::vector<ReconstructionEntry> entries;
        std::error_code error;
        if(!std::filesystem::is_directory(root,error))return entries;
        if(isReconstructionDirectory(root))entries.push_back(reconstructionEntry(root));
        else {
            std::filesystem::recursive_directory_iterator it(root,std::filesystem::directory_options::skip_permission_denied,error),end;
            while(it!=end) {
                const auto path=it->path();
                if(it->is_directory(error)) {
                    const auto name=path.filename().string();
                    if(name.starts_with('.')||name=="points"||name=="renders"||name=="mesh_checkpoints"||name=="mesh_inputs")
                        it.disable_recursion_pending();
                    else if(isReconstructionDirectory(path)) {
                        entries.push_back(reconstructionEntry(path));it.disable_recursion_pending();
                    }
                }
                error.clear();it.increment(error);
                if(error)error.clear();
            }
        }
        std::sort(entries.begin(),entries.end(),[](const auto& a,const auto& b) {
            return a.created.value!=b.created.value?a.created.value>b.created.value:a.directory.string()>b.directory.string();
        });
        return entries;
    }

    struct SavedReconstruction {
        ReconstructionEntry entry;
        std::filesystem::path scene,sessionDirectory;
        YAML::Node config;
        std::vector<PointCloudSnapshot> checkpoints;
        TrainingMetrics metrics,geometryMetrics;
        std::string log,launchCommand;
        uint64_t iteration=0,targetIteration=0;
    };
    inline uint64_t reconstructionIteration(double value) {
        return std::isfinite(value)&&value>=0&&value<static_cast<double>(std::numeric_limits<uint64_t>::max())?
            static_cast<uint64_t>(value):0;
    }
    // Disk/CSV work runs off the UI thread. This is a fixed inspection snapshot:
    // no process is adopted and no commands or changes are written to the run.
    inline SavedReconstruction loadSavedReconstruction(const std::filesystem::path& folder) {
        SavedReconstruction saved;saved.entry=reconstructionEntry(folder);
        const auto& root=saved.entry.directory;
        if(!isReconstructionDirectory(root))throw std::runtime_error("Choose a reconstruction folder containing run_config.json or metrics.csv");
        saved.config=reconstructionJson(root/"run_config.json");
        if(reconstructionFile(root/"scene.xml"))saved.scene=root/"scene.xml";
        else if(const auto scene=reconstructionSetting(saved.config,"scene_xml");scene.IsScalar()) {
            const std::filesystem::path path=scene.as<std::string>();
            const auto candidate=path.is_absolute()?path:root/path;
            if(reconstructionFile(candidate))saved.scene=candidate;
        }
        if(saved.scene.empty())throw std::runtime_error("This reconstruction has no accessible scene.xml. Restore its scene file before opening it.");
        while(saved.metrics.update(root/"metrics.csv")) {}
        while(saved.geometryMetrics.update(root/"geometry_metrics.csv")) {}
        saved.iteration=reconstructionIteration(saved.metrics.latest("iteration"));
        saved.checkpoints=listOptimizationPointCloudSnapshots(root/"points");
        if(!saved.checkpoints.empty())saved.iteration=std::max(saved.iteration,saved.checkpoints.back().iteration);
        const auto iterations=reconstructionSetting(saved.config,"iterations");
        const auto offset=reconstructionSetting(saved.config,"resume_iteration_offset");
        try {saved.targetIteration=reconstructionIteration(iterations.as<double>(0)+offset.as<double>(0));}catch(...) {}
        const auto final=root/"points_final.ply";
        if(reconstructionFile(final)&&(saved.checkpoints.empty()||lastWriteTimeOrMin(final)>=saved.checkpoints.back().writeTime)) {
            // The CSV is authoritative for stopped runs; the target is only a
            // fallback for a final output whose metrics were not retained.
            const auto step=saved.iteration?saved.iteration:saved.targetIteration;
            const PointCloudSnapshot snapshot{final,step,lastWriteTimeOrMin(final)};
            if(!saved.checkpoints.empty()&&saved.checkpoints.back().iteration==step)saved.checkpoints.back()=snapshot;
            else saved.checkpoints.push_back(snapshot);
            saved.iteration=step;
        }
        if(saved.checkpoints.empty())throw std::runtime_error("No saved point cloud found. Expected points/iter_*_points.ply or points_final.ply.");
        saved.targetIteration=std::max(saved.targetIteration,saved.iteration);
        auto logPath=root/"training.log";
        auto commandPath=root/"launch_command.txt";
        // Viewer logs live beside outputs in .viewer_sessions. Match by the
        // recorded run path rather than guessing session timestamps or PIDs.
        for(auto parent=root.parent_path();!parent.empty();parent=parent.parent_path()) {
            const auto sessions=parent/".viewer_sessions";
            std::error_code error;
            for(std::filesystem::directory_iterator it(sessions,error),end;it!=end;it.increment(error)) {
                if(error)break;
                const auto status=reconstructionJson(it->path()/"status.json");
                if(!status.IsMap()||!status["run_dir"].IsScalar())continue;
                const auto original=std::filesystem::path(status["run_dir"].as<std::string>()).lexically_normal();
                if(original!=root)continue;
                saved.sessionDirectory=it->path();
                logPath=it->path()/"training.log";commandPath=it->path()/"launch_command.txt";
                break;
            }
            if(!saved.sessionDirectory.empty()||parent==parent.root_path())break;
        }
        TrainingLog log;
        {std::ifstream stream(logPath,std::ios::binary);
            if(stream) {
                stream.seekg(0,std::ios::end);const auto bytes=stream.tellg();
                stream.seekg(bytes>131072?bytes-std::streamoff(131072):std::streampos(0));
                log.append(std::string(std::istreambuf_iterator<char>(stream),{}));
            }
        }
        saved.log=log.text();
        {std::ifstream command(commandPath);std::getline(command,saved.launchCommand);}
        return saved;
    }
    inline SavedReconstruction loadLatestSavedReconstruction(const std::filesystem::path& root) {
        const auto runs=scanReconstructions(reconstructionDirectory(root));
        if(runs.empty())throw std::runtime_error("No saved reconstructions found in "+root.string());
        std::string error;
        for(const auto& run:runs) {
            try {return loadSavedReconstruction(run.directory);}
            catch(const std::runtime_error& failure) {if(error.empty())error=failure.what();}
        }
        throw std::runtime_error("No loadable reconstruction found in "+root.string()+". "+error);
    }
}
