#pragma once
#include "TrainingSession.h"
#include "OptimizationSnapshots.h"
#include "DatasetLibrary.h"
#include "WorkspaceUi.h"
#include "TrainingLog.h"
#include "TrainingMetrics.h"
#include <imgui.h>
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <iterator>

namespace viewer {
    template<class Number> YAML::Node jsonNumber(Number value) {
        if (!std::isfinite(static_cast<double>(value))) throw std::runtime_error("JSON numbers must be finite");
        YAML::Node node(value); node.SetTag("tag:yaml.org,2002:float"); return node;
    }
    inline void writeJsonString(std::ostream& stream, const std::string& text) {
        constexpr char hex[] = "0123456789abcdef";
        stream << '"';
        for (unsigned char c : text) {
            if (c == '"' || c == '\\') stream << '\\' << c;
            else if (c < 0x20) stream << "\\u00" << hex[c >> 4] << hex[c & 15];
            else stream << c;
        }
        stream << '"';
    }
    inline void writeJsonValue(std::ostream& stream, const YAML::Node& node) {
        if (!node || node.IsNull()) { stream << "null"; return; }
        if (node.IsScalar()) {
            const auto& value = node.Scalar();
            // YAML nodes erase the C++ scalar type. Mark generated numbers
            // explicitly; parsed JSON numbers carry the implicit '?' tag.
            if (node.Tag() == "tag:yaml.org,2002:float" || node.Tag() == "?") {
                if (value == "true" || value == "false") { stream << value; return; }
                char* end = nullptr;
                const double number = std::strtod(value.c_str(), &end);
                if (!value.empty() && end == value.c_str() + value.size() && std::isfinite(number)) {
                    stream << value; return;
                }
            }
            writeJsonString(stream, value); return;
        }
        bool first = true;
        stream << (node.IsMap() ? '{' : '[');
        if (node.IsMap()) for (const auto& entry : node) {
            if (!first) stream << ','; first = false;
            writeJsonString(stream, entry.first.as<std::string>()); stream << ':';
            writeJsonValue(stream, entry.second);
        }
        else for (const auto& entry : node) {
            if (!first) stream << ','; first = false; writeJsonValue(stream, entry);
        }
        stream << (node.IsMap() ? '}' : ']');
    }
    inline void writeJsonAtomic(const std::filesystem::path& path, const YAML::Node& node) {
        std::filesystem::create_directories(path.parent_path());
        const auto temporary = path.string() + ".tmp";
        { std::ofstream file(temporary); writeJsonValue(file, node); file.close();
          if (!file) throw std::runtime_error("Cannot write " + temporary); }
        std::filesystem::rename(temporary, path);
    }
    inline YAML::Node readJson(const std::filesystem::path& path) {
        try { return YAML::LoadFile(path.string()); } catch (...) { return {}; }
    }
    inline std::string tailText(const std::filesystem::path& path, std::size_t limit = 12000) {
        std::ifstream file(path, std::ios::binary);
        if (!file) return {};
        file.seekg(0, std::ios::end); const auto size = file.tellg();
        file.seekg(size > static_cast<std::streamoff>(limit) ? size - static_cast<std::streamoff>(limit) : std::streampos(0));
        return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    }
    inline constexpr std::array<const char*, 10> trainingParameterNames{
        "learning_rate_position", "learning_rate_rotation", "learning_rate_scale", "learning_rate_albedo",
        "learning_rate_opacity", "learning_rate_beta", "alpha_loss_weight", "depth_distort_weight",
        "normal_consistency_weight", "intra_slab_depth_weight"};
    inline constexpr std::array<const char*, 10> trainingParameterLabels{
        "Position LR", "Rotation LR", "Scale LR", "Albedo LR", "Opacity LR", "Beta LR",
        "Alpha weight", "Depth distortion weight", "Normal consistency weight", "Slab weight"};
    enum class TrainingView { Live, Saved, History, Snapshot };
    inline std::string trainingNumber(float value) {
        std::ostringstream stream; stream << std::setprecision(9) << value; return stream.str();
    }

    struct TrainingControls {
        TrainingSession process;
        std::array<char, 2048> dataset{}, output{}, groundTruth{}, python{}, workingDirectory{}, moduleDirectory{}, datasetRoot{};
        std::vector<DatasetEntry> datasets;
        bool evaluateGroundTruth=true;
        bool datasetChanged=false;
        std::optional<std::filesystem::path> initializedPointCloudRequest;
        int iterations = 30000, snapshotInterval = 100, samples = 1;
        float liveInterval = 1.0f;
        bool overrideParameters = false, useWorkingSnapshot = false;
        bool overrideRunSettings=false,stopRequested=false;
        bool rollingMetrics=false,logMetrics=false;
        int metricWindow=1000;
        int requestedMonitorTab=-1;
        TrainingMetrics metrics,geometryMetrics;
        std::filesystem::path historyPath;
        uint64_t historyIteration=0;
        // Navigation is consumed by the UI at a renderer handoff; it never
        // rewinds or pauses the optimizer. -2/+2 are 1000-iteration jumps.
        std::optional<int> checkpointNavigation;
        std::optional<std::size_t> checkpointSelection;
        std::array<float, 10> parameters{0.00016f, .005f, .005f, .003f, .003f, .0005f, .0005f, .005f, .001f, 0.f};
        TrainingView view = TrainingView::Snapshot;
        YAML::Node status, live;
        std::string message, log, selectableLog;
        ImGuiID logScrollInput=0,logScrollWindow=0;
        TrainingLog logReader;
        std::vector<float> losses;
        uint64_t plottedIteration = 0, viewedIteration = 0, loadedGeneration = 0;
        bool parametersInitialized = false;
        std::chrono::steady_clock::time_point lastPoll{};
        std::filesystem::path preferences;
        std::string detectedPython, detectedModuleDirectory;

        static void set(std::array<char, 2048>& buffer, const std::string& value) {
            std::snprintf(buffer.data(), buffer.size(), "%s", value.c_str());
        }
        explicit TrainingControls(const std::filesystem::path& repository) {
            set(output, (repository / "python/OptimizationOutput").string());
            set(workingDirectory, (repository / "python").string());
            set(datasetRoot, expandedPath("~/ValeDrive/phd/pbdr/datasets/pbdr_synthetic",repository).string());
            const char* environment = std::getenv("VIRTUAL_ENV");
            set(python,"python3");
            if(environment)set(python,(std::filesystem::path(environment)/"bin/python").string());
            else {
                // Prefer an existing project environment before the system Python.
                for(const auto& candidate:{repository/".venv/bin/python",
                    expandedPath("~/miniconda3/envs/pale/bin/python",repository),
                    expandedPath("~/anaconda3/envs/pale/bin/python",repository)})
                    if(std::filesystem::is_regular_file(candidate)) {set(python,candidate.string());break;}
            }
            if(std::filesystem::is_regular_file(repository/"cmake-build-pybind/pale.so"))
                set(moduleDirectory,(repository/"cmake-build-pybind").string());
            detectedPython=python.data(); detectedModuleDirectory=moduleDirectory.data();
            preferences = repository / "output/viewer_training.json";
            const auto saved = readJson(preferences);
            if (saved.IsMap()) {
                // Older builds persisted the generic launcher default, which can
                // resolve to a different Python in a desktop app than in a shell.
                const bool legacyPython = !saved["environment_preferences_version"] && saved["python"] &&
                    (saved["python"].as<std::string>() == "python3" || saved["python"].as<std::string>() == "python");
                for (const auto& [key, buffer] : std::array<std::pair<const char*, std::array<char,2048>*>,7>{
                    {{"dataset", &dataset}, {"output", &output}, {"ground_truth", &groundTruth},
                     {"python", &python}, {"working_directory", &workingDirectory}, {"module_directory", &moduleDirectory}, {"dataset_root",&datasetRoot}}})
                    if (saved[key]) {
                        if (legacyPython && (std::string(key)=="python" ||
                            (std::string(key)=="module_directory" && saved[key].as<std::string>().empty()))) continue;
                        set(*buffer, saved[key].as<std::string>());
                    }
            }
            refreshDatasets();
        }
        void selectDataset(const DatasetEntry& item) {
            const bool changed=std::filesystem::path(dataset.data())!=item.directory;
            datasetChanged=datasetChanged || changed;
            if(changed)useWorkingSnapshot=false;
            set(dataset,item.directory.string());
            set(groundTruth,item.groundTruth ? item.groundTruth->string() : "");
            evaluateGroundTruth=item.groundTruth.has_value();
        }
        void refreshDatasets() {
            datasets=scanDatasets(expandedPath(datasetRoot.data(),std::filesystem::current_path()));
            auto found=std::find_if(datasets.begin(),datasets.end(),[&](const auto& entry){return entry.directory==std::filesystem::path(dataset.data());});
            if(found!=datasets.end()) {
                const bool evaluation=evaluateGroundTruth;
                selectDataset(*found); evaluateGroundTruth=evaluation&&found->groundTruth.has_value();
            }
            else if(!datasets.empty()) {
                auto teapot=std::find_if(datasets.begin(),datasets.end(),[](const auto& entry){return entry.directory.filename()=="teapot_pbdr";});
                selectDataset(teapot!=datasets.end()?*teapot:datasets.front());
            }
        }
        [[nodiscard]] uint64_t integer(const char* name) const {
            try { return status[name].as<uint64_t>(); } catch (...) { return 0; }
        }
        [[nodiscard]] std::string state() const {
            if(stopRequested&&process.active())return "stopping";
            try { return status["state"].as<std::string>(); } catch (...) { return process.active() ? "starting" : "idle"; }
        }
        [[nodiscard]] bool pending() const { return process.active() && process.commandSequence > integer("ack"); }
        [[nodiscard]] std::vector<PointCloudSnapshot> checkpoints() const {
            auto result=listOptimizationPointCloudSnapshots(process.runDirectory/"points");
            if(auto latest=latestSavedSnapshot();latest && (result.empty()||latest->path!=result.back().path))result.push_back(*latest);
            return result;
        }
        [[nodiscard]] std::optional<PointCloudSnapshot> latestSavedSnapshot() const {
            const auto snapshots = listOptimizationPointCloudSnapshots(process.runDirectory / "points");
            std::optional<PointCloudSnapshot> latest;
            if (!snapshots.empty()) latest = snapshots.back();
            const auto final = process.runDirectory / "points_final.ply";
            const auto currentState = state();
            std::error_code error;
            if ((currentState == "completed" || currentState == "stopped") &&
                std::filesystem::is_regular_file(final, error) &&
                (!latest || integer("iteration") >= latest->iteration))
                latest = PointCloudSnapshot{final, integer("iteration"), lastWriteTimeOrMin(final)};
            return latest;
        }
        void poll() {
            process.poll();
            const auto now = std::chrono::steady_clock::now();
            if (!process.launched() || now - lastPoll < std::chrono::milliseconds(250)) return;
            lastPoll = now;
            if (auto next = readJson(process.directory / "status.json"); next.IsMap()) status = next;
            if (auto next = readJson(process.directory / "live.json"); next.IsMap()) live = next;
            if (!parametersInitialized && status["parameters"].IsMap()) {
                for (std::size_t i = 0; i < parameters.size(); ++i)
                    if (status["parameters"][trainingParameterNames[i]]) parameters[i] = status["parameters"][trainingParameterNames[i]].as<float>();
                parametersInitialized = true;
            }
            if (integer("iteration") != plottedIteration && status["loss"].IsScalar()) {
                try { losses.push_back(status["loss"].as<float>()); } catch (...) {}
                plottedIteration = integer("iteration");
            }
            if(logReader.update(process.directory / "training.log"))log=logReader.text();
            metrics.update(process.runDirectory/"metrics.csv");
            geometryMetrics.update(process.runDirectory/"geometry_metrics.csv",true);
            if (process.exitCode && *process.exitCode != 0)
                message = "Training exited with code " + std::to_string(*process.exitCode) + ". Open Run monitor > Log for details.";
        }
        void request(const char* command, bool tune = false) {
            if (!process.active() || pending()) return;
            YAML::Node request;
            request["sequence"] = jsonNumber(++process.commandSequence); request["command"] = command;
            if (tune) for (std::size_t i = 0; i < parameters.size(); ++i) {
                if (!std::isfinite(parameters[i]) || parameters[i] < 0) {
                    --process.commandSequence; throw std::runtime_error("Parameters must be finite and non-negative");
                }
                request["parameters"][trainingParameterNames[i]] = jsonNumber(parameters[i]);
            }
            try { writeJsonAtomic(process.directory / "command.json", request); }
            catch (...) { --process.commandSequence; throw; }
            if(std::string(command)=="stop")stopRequested=true;
        }
        void send(const char* command, bool tune = false) {
            try { request(command, tune); } catch (const std::exception& error) { message = error.what(); }
        }
        void launch(const std::filesystem::path& workspace = {}) {
            if(process.active())throw std::runtime_error("A reconstruction is already running");
            const auto cwd = expandedPath(workingDirectory.data(), std::filesystem::current_path());
            const auto input = expandedPath(dataset.data(), cwd);
            if (!std::filesystem::is_directory(input / "images")) throw std::runtime_error("Dataset needs an images/ directory");
            if (workspace.empty() && (!std::filesystem::is_regular_file(input / "scene.xml") ||
                                      !std::filesystem::is_regular_file(input / "points.ply")))
                throw std::runtime_error("Dataset needs scene.xml and points.ply");
            if ((overrideRunSettings&&(iterations < 1 || snapshotInterval < 1 || samples < 1)) || !std::isfinite(liveInterval) || liveInterval < .1f)
                throw std::runtime_error("Iterations, snapshot interval and samples must be positive; live interval must be >= 0.1 s");
            std::vector<std::string> args;
            if (moduleDirectory[0]) {
                std::string path = expandedPath(moduleDirectory.data(), cwd).string();
                if (const char* inherited = std::getenv("PYTHONPATH")) path += ":" + std::string(inherited);
                args = {"/usr/bin/env", "PYTHONPATH=" + path};
            }
            const std::string executable = python.data();
            args.push_back(executable.find('/') == std::string::npos ? executable : expandedPath(executable, cwd).string());
            args.insert(args.end(), {"-u", (cwd / "main.py").string(), "-s", input.string(),
                "--viewer-live-interval", std::to_string(liveInterval),
                "--no-metrics", "--no-image-preview"});
            if(overrideRunSettings)args.insert(args.end(),{"--iterations",std::to_string(iterations),"--save-ply-files-interval",std::to_string(snapshotInterval),"--adjoint-spp",std::to_string(samples)});
            if(!evaluateGroundTruth)args.push_back("--no-dataset-ground-truth");
            if (evaluateGroundTruth && groundTruth[0]) {
                const auto truth = expandedPath(groundTruth.data(), cwd);
                if (!std::filesystem::is_regular_file(truth)) throw std::runtime_error("Ground truth file does not exist");
                args.insert(args.end(), {"--gt", truth.string()});
            }
            if (!workspace.empty()) {
                const auto seed = reserveDirectory(expandedPath(output.data(), cwd) / ".viewer_seeds", "seed_");
                auto saved = readJson(workspace);
                if (!saved.IsMap()) throw std::runtime_error("Invalid snapshot workspace");
                std::filesystem::copy_file(saved["pointcloud"].as<std::string>(), seed / "points.ply");
                saved["pointcloud"] = (seed / "points.ply").string();
                writeJsonAtomic(seed / "workspace.json", saved);
                args.insert(args.end(), {"--viewer-workspace", (seed / "workspace.json").string()});
            }
            if (overrideParameters) for (std::size_t i = 0; i < parameters.size(); ++i) {
                if (!std::isfinite(parameters[i]) || parameters[i] < 0) throw std::runtime_error("Invalid hyperparameter");
                std::string flag = i < 6 ? std::string("--lr-") + std::array<const char*,6>{"pos","rot","scale","albedo","opacity","beta"}[i]
                    : std::string("--") + trainingParameterNames[i];
                for (char& c : flag) if (c == '_') c = '-';
                args.insert(args.end(), {flag, trainingNumber(parameters[i])});
            }
            process.launch(expandedPath(output.data(), cwd), cwd, args);
            status = YAML::Node(); live = YAML::Node(); losses.clear(); parametersInitialized = false;
            metrics.reset();geometryMetrics.reset();stopRequested=false;
            logReader.reset();log.clear();selectableLog.clear();
            checkpointNavigation.reset();checkpointSelection.reset();historyPath.clear();
            plottedIteration = viewedIteration = loadedGeneration = 0; view = TrainingView::Live;
            YAML::Node saved;
            saved["dataset"] = std::string(dataset.data()); saved["output"] = std::string(output.data());
            saved["ground_truth"] = std::string(groundTruth.data()); saved["python"] = std::string(python.data());
            saved["working_directory"] = std::string(workingDirectory.data()); saved["module_directory"] = std::string(moduleDirectory.data());
            saved["dataset_root"]=std::string(datasetRoot.data());
            saved["environment_preferences_version"]=jsonNumber(1);
            try { writeJsonAtomic(preferences, saved); } catch (...) {} // A launched job must not be reported as a failed launch.
            message = "Training launched; waiting for the first live scene";
        }
        // Returns true when the user requests a launch; ownership changes live in main.cpp.
        bool drawSetup(bool hasWorkspace, bool previewBusy=false) {
            bool start = false;
            ImGui::TextColored(ui::accent,"JOINT PBdR");
            ImGui::TextDisabled("Geometry  /  Materials  /  Emitters");
            ImGui::Spacing(); ImGui::SeparatorText("Dataset");
            const auto selected=std::find_if(datasets.begin(),datasets.end(),[&](const auto& entry){return entry.directory==std::filesystem::path(dataset.data());});
            const std::string label=selected==datasets.end() ? (dataset[0] ? std::filesystem::path(dataset.data()).filename().string() : "Choose dataset") : selected->label;
            ImGui::SetNextItemWidth(-38);
            if(ImGui::BeginCombo("##Dataset",label.c_str())) {
                for(const auto& entry:datasets) {
                    const bool active=entry.directory==std::filesystem::path(dataset.data());
                    if(ImGui::Selectable((entry.label+(entry.groundTruth?"  ·  GT":"")).c_str(),active))selectDataset(entry);
                    if(active)ImGui::SetItemDefaultFocus();
                }
                if(datasets.empty())ImGui::TextDisabled("No complete datasets found");
                ImGui::EndCombo();
            }
            if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",dataset.data());
            ImGui::SameLine();
            if(ui::iconButton("Refresh dataset library",ui::Icon::Refresh,false,ImVec2(28,ImGui::GetFrameHeight())))refreshDatasets();
            if(!dataset[0])ImGui::BeginDisabled();
            if(ImGui::Button("Load initialized point cloud",ImVec2(-1,0))) {
                const auto cwd=expandedPath(workingDirectory.data(),std::filesystem::current_path());
                initializedPointCloudRequest=expandedPath(dataset.data(),cwd);
            }
            if(!dataset[0])ImGui::EndDisabled();
            if(ImGui::IsItemHovered())ImGui::SetTooltip("Inspect the selected dataset's starting point cloud without starting reconstruction.");
            if(groundTruth[0])ImGui::Checkbox("Evaluate with dataset ground truth",&evaluateGroundTruth);
            else ImGui::TextDisabled("Ground truth optional · not included");
            ImGui::Spacing(); ImGui::SeparatorText("Reconstruction");
            ImGui::Checkbox("Override run settings from config.py",&overrideRunSettings);
            if(overrideRunSettings) {ImGui::SetNextItemWidth(-1);ImGui::InputInt("##iterations",&iterations);ImGui::TextDisabled("Target iterations");}
            else ImGui::TextWrapped("Next run reads config.py defaults for iterations, sampling, checkpoint cadence and optimizer settings.");
            if (hasWorkspace) { ImGui::Checkbox("Initialize from edited snapshot", &useWorkingSnapshot); }
            else useWorkingSnapshot = false;
            if (ImGui::TreeNode("Python environment")) {
                ImGui::SetNextItemWidth(-1);
                ui::pathInput("Python executable",python);
                ui::pathInput("Working directory",workingDirectory);
                ui::pathInput("pale module directory",moduleDirectory);
                if(ImGui::Button("Use detected environment")) {
                    set(python,detectedPython); set(moduleDirectory,detectedModuleDirectory);
                }
                ImGui::TextWrapped("Leave module directory empty to inherit PYTHONPATH. Select the same Python environment used by your training script.");
                ImGui::TreePop();
            }
            if(ImGui::TreeNode("Run options")) {
                if(overrideRunSettings) {ImGui::InputInt("Snapshot every", &snapshotInterval);ImGui::InputInt("Adjoint samples", &samples);}
                ImGui::DragFloat("Live interval (s)", &liveInterval, .1f, .1f, 60.f, "%.1f");
                ui::pathInput("Output root",output);
                ui::pathInput("Dataset library",datasetRoot);
                if(ImGui::Button("Rescan library"))refreshDatasets();
                if(ui::pathInput("Custom dataset folder",dataset)) {datasetChanged=true;useWorkingSnapshot=false;}
                if(ImGui::Button("Detect local GT"))set(groundTruth,datasetGroundTruth(dataset.data()).value_or(std::filesystem::path{}).string());
                ui::pathInput("GT override",groundTruth);
                ImGui::TreePop();
            }
            if (ImGui::TreeNode("Hyperparameters")) {
                ImGui::Checkbox("Override launch defaults", &overrideParameters);
                if(ImGui::BeginTable("HyperparameterValues",2,ImGuiTableFlags_SizingStretchProp)) {
                    ImGui::TableSetupColumn("Parameter",ImGuiTableColumnFlags_WidthStretch,.57f);
                    ImGui::TableSetupColumn("Value",ImGuiTableColumnFlags_WidthStretch,.43f);
                    for (std::size_t i = 0; i < parameters.size(); ++i) {
                        ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::AlignTextToFramePadding();
                        ImGui::TextUnformatted(trainingParameterLabels[i]); ImGui::TableNextColumn();
                        ImGui::PushID(static_cast<int>(i)); ImGui::SetNextItemWidth(-1);
                        if (ImGui::DragFloat("##value", &parameters[i], .00001f, 0, 1000, "%.7g")&&!process.active()) overrideParameters = true;
                        ImGui::PopID();
                    }
                    ImGui::EndTable();
                }
                if (process.active() && !stopRequested && parametersInitialized) {
                    const bool disabled = pending();
                    if (disabled) ImGui::BeginDisabled();
                    if (ImGui::Button("Apply to training at next iteration")) send("tune", true);
                    if (disabled) ImGui::EndDisabled();
                }
                ImGui::TextWrapped("Live changes are acknowledged at an iteration boundary and recorded in the run's parameter_changes.jsonl.");
                ImGui::TextDisabled("Base rates precede scene and schedule scaling.");
                ImGui::TextDisabled("Emitter power optimization is fixed by the current training backend.");
                ImGui::TreePop();
            }
            start=drawStartButton(ImVec2(-1,32),previewBusy&&useWorkingSnapshot);
            if (!message.empty()) ImGui::TextWrapped("%s", message.c_str());
            return start;
        }
        void drawProgress(bool& visible) {
            if (!visible) return;
            ImGui::Begin("Run monitor", &visible);
            const int requested=requestedMonitorTab;requestedMonitorTab=-1;
            if(ImGui::BeginTabBar("MonitorTabs")) {
                if(ImGui::BeginTabItem("Progress",nullptr,requested==0?ImGuiTabItemFlags_SetSelected:0)) {
                    if(ImGui::BeginTable("RunOverview",3,ImGuiTableFlags_SizingStretchSame|ImGuiTableFlags_BordersInnerV)) {
                        ImGui::TableNextColumn();
                        ImGui::TextDisabled("Reconstruction objective");
                        if(!metrics.iterations.empty())metrics.plot("Objective",{{"loss_total_mean","Total"}},rollingMetrics,metricWindow,logMetrics,44);
                        else if(!losses.empty()) {
                            ImGui::SameLine(); ImGui::TextColored(ui::green,"%.6g",losses.back());
                            const int count=rollingMetrics?std::min(static_cast<int>(losses.size()),metricWindow):static_cast<int>(losses.size());
                            ImGui::PlotLines("##loss",losses.data()+losses.size()-count,count,0,nullptr,FLT_MAX,FLT_MAX,ImVec2(-1,44));
                        } else ImGui::TextWrapped("Start a reconstruction to track its objective here.");
                        ImGui::TableNextColumn();
                        ImGui::Text("Iteration %llu / %llu",static_cast<unsigned long long>(integer("iteration")),static_cast<unsigned long long>(integer("target_iteration")));
                        const float progress=integer("target_iteration") ? static_cast<float>(integer("iteration"))/integer("target_iteration") : 0;
                        ImGui::ProgressBar(progress,ImVec2(-1,7),"");
                        ImGui::TextDisabled("Viewing %llu  ·  Saved %llu",static_cast<unsigned long long>(viewedIteration),static_cast<unsigned long long>(integer("saved_iteration")));
                        ImGui::TextDisabled("%s",view==TrainingView::Snapshot?"Editable snapshot · training independent":"Read-only training view");
                        ImGui::TableNextColumn();
                        ImGui::TextColored(process.active()?ui::green:ui::muted,"%s",state().c_str());
                        if(ImGui::IsItemHovered())ImGui::SetTooltip("%s\nClosing this viewer stops its reconstruction.",process.runDirectory.string().c_str());
                        const bool disabled=!process.active()||pending()||stopRequested||state()=="finalizing";
                        if(disabled)ImGui::BeginDisabled();
                        if(ImGui::Button(state()=="paused"?"Continue":"Pause"))send(state()=="paused"?"resume":"pause");
                        ImGui::SameLine(); if(ImGui::Button("Stop"))send("stop");
                        ImGui::SameLine(); if(ImGui::Button("Save checkpoint"))send("save");
                        if(disabled)ImGui::EndDisabled();
                        if(pending())ImGui::TextDisabled("Waiting for iteration boundary");
                        else if(process.exitCode)ImGui::TextDisabled("Process exited: %d",*process.exitCode);
                        else if(status["iteration_ms"].IsScalar()) {
                            try {
                                const float milliseconds=status["iteration_ms"].as<float>();
                                if(std::isfinite(milliseconds)&&milliseconds>0)
                                    ImGui::TextDisabled("Update %.1f ms  ·  %.2f iter/s",milliseconds,1000.f/milliseconds);
                            }catch(...){}
                        }
                        ImGui::EndTable();
                    }
                    for(const char* key:{"command_error","preview_error"})if(status[key].IsScalar()&&!status[key].as<std::string>().empty())ImGui::TextWrapped("%s",status[key].as<std::string>().c_str());
                    ImGui::Checkbox("Rolling window",&rollingMetrics);ImGui::SameLine();
                    if(ImGui::SmallButton("Metrics..."))requestedMonitorTab=1;
                    ImGui::SameLine();
                    if(std::isfinite(geometryMetrics.latest("cd")))ImGui::Text("CD %.5g  Accuracy %.5g  Completion %.5g",geometryMetrics.latest("cd"),geometryMetrics.latest("accuracy"),geometryMetrics.latest("completion"));
                    if(process.exitCode&&*process.exitCode!=0)ImGui::TextWrapped("%s",message.c_str());
                    ImGui::EndTabItem();
                }
                if(ImGui::BeginTabItem("Metrics",nullptr,requested==1?ImGuiTabItemFlags_SetSelected:0)) {drawMetrics();ImGui::EndTabItem();}
                if(ImGui::BeginTabItem("Checkpoints",nullptr,requested==2?ImGuiTabItemFlags_SetSelected:0)) {drawCheckpoints();ImGui::EndTabItem();}
                if(ImGui::BeginTabItem("Log",nullptr,requested==3?ImGuiTabItemFlags_SetSelected:0)) {
                    if(ImGui::Button("Copy log"))ImGui::SetClipboardText(log.c_str());
                    ImGui::SameLine(); if(ImGui::Button("Copy launch command"))ImGui::SetClipboardText(shellDisplay(process.arguments).c_str());
                    ImGui::SameLine(); ImGui::TextDisabled("%s",process.directory.filename().string().c_str());
                    if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",process.directory.string().c_str());
                    drawLogText(); ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
            ImGui::End();
        }
        void drawLogText() {
            // Focus alone must not stop progress updates. Only an actual text
            // selection holds the buffer steady for copying.
            const auto id=ImGui::GetID("##TrainingLog");
            auto* parent=ImGui::GetCurrentWindow();
            auto* child=logScrollInput==id?ImGui::FindWindowByID(logScrollWindow):nullptr;
            if(child&&(child->ParentWindow!=parent||child->ChildId!=id))child=nullptr;
            auto* state=ImGui::GetInputTextState(id);
            const bool selecting=ImGui::GetActiveID()==id&&state&&state->HasSelection();
            // A wheel gesture can set a pending scroll target before the child
            // begins. Respect that target as well as scrollbar position, so
            // appended output cannot pull someone away from older lines.
            constexpr float tail=1.0e9f;
            if(child) {
                const float requested=child->ScrollTarget.y;
                const float position=requested!=FLT_MAX&&requested!=tail?requested:child->Scroll.y;
                const bool follow=!selecting&&position>=child->ScrollMax.y-1.f;
                if(follow)ImGui::SetScrollY(child,tail);
                else if(child->ScrollTarget.y==tail)child->ScrollTarget.y=FLT_MAX;
            }
            if(!selecting)selectableLog=log;
            ImGui::InputTextMultiline("##TrainingLog",selectableLog.data(),selectableLog.size()+1,
                ImVec2(-1,std::max(ImGui::GetContentRegionAvail().y,ImGui::GetFrameHeight()*2)),
                ImGuiInputTextFlags_ReadOnly);
            if(!child)for(auto* window:parent->DC.ChildWindows)if(window->ChildId==id) {
                logScrollInput=id;logScrollWindow=window->ID;
                if(!selecting)ImGui::SetScrollY(window,tail);
                break;
            }
        }
        bool drawStartButton(ImVec2 size, bool previewBlocked=false) {
            const bool running=process.active(),disabled=running||previewBlocked;
            if(disabled)ImGui::BeginDisabled();
            ImGui::PushStyleColor(ImGuiCol_Button,ui::accent);
            ImGui::PushStyleColor(ImGuiCol_Text,ImVec4(.13f,.11f,.09f,1));
            const bool pressed=ImGui::Button("Start reconstruction",size);
            ImGui::PopStyleColor(2);
            const auto min=ImGui::GetItemRectMin(),max=ImGui::GetItemRectMax();
            if(disabled)ImGui::EndDisabled();
            if(running) {
                const float text=ImGui::CalcTextSize("Start reconstruction").x;
                const ImVec2 center(min.x+std::max(9.f,(max.x-min.x-text)*.5f-12.f),(min.y+max.y)*.5f);
                const float angle=static_cast<float>(ImGui::GetTime())*5.f;
                auto* draw=ImGui::GetWindowDrawList();
                draw->PathArcTo(center,5.f,angle,angle+4.8f,18);
                draw->PathStroke(ImGui::GetColorU32(ui::accent),0,2.f);
            }
            if(disabled&&ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("%s",running?"A reconstruction is running. Start becomes available after it exits.":"Wait for the current preview before starting from an edited snapshot.");
            return pressed&&!disabled;
        }
        void drawCheckpoints() {
            if(!process.launched()) {ImGui::TextWrapped("Start a reconstruction to browse its saved checkpoints.");return;}
            const auto saved=checkpoints();
            ImGui::Text("Viewing iteration %llu · training %llu",static_cast<unsigned long long>(viewedIteration),static_cast<unsigned long long>(integer("iteration")));
            if(ImGui::Button("Live (R)"))checkpointNavigation=0;
            ImGui::SameLine();if(ImGui::Button("-1000 (N)"))checkpointNavigation=-2;
            ImGui::SameLine();if(ImGui::Button("Previous"))checkpointNavigation=-1;
            ImGui::SameLine();if(ImGui::Button("Next"))checkpointNavigation=1;
            ImGui::SameLine();if(ImGui::Button("+1000 (M)"))checkpointNavigation=2;
            ImGui::SameLine();if(ImGui::Button("Latest saved"))checkpointNavigation=3;
            if(saved.empty())ImGui::TextDisabled("Waiting for the first persistent checkpoint");
            else {
                int index=static_cast<int>(checkpointNavigationIndex(saved,view==TrainingView::History?historyIteration:viewedIteration,-1));
                const auto exact=std::lower_bound(saved.begin(),saved.end(),view==TrainingView::History?historyIteration:viewedIteration,[](const auto& item,uint64_t step){return item.iteration<step;});
                if(exact!=saved.end())index=static_cast<int>(exact-saved.begin());
                ImGui::SetNextItemWidth(std::min(500.f,ImGui::GetContentRegionAvail().x));
                if(ImGui::SliderInt("Checkpoint",&index,0,static_cast<int>(saved.size()-1),std::to_string(saved[index].iteration).c_str()))checkpointSelection=static_cast<std::size_t>(index);
                ImGui::SetNextItemWidth(std::min(320.f,ImGui::GetContentRegionAvail().x));
                if(ImGui::BeginCombo("Saved iteration",std::to_string(viewedIteration).c_str())) {
                    for(std::size_t i=0;i<saved.size();++i)if(ImGui::Selectable((std::to_string(saved[i].iteration)+"###checkpoint"+std::to_string(i)).c_str(),view==TrainingView::History&&historyIteration==saved[i].iteration))checkpointSelection=i;
                    ImGui::EndCombo();
                }
                ImGui::TextDisabled("%zu saved checkpoints · browsing leaves optimization running",saved.size());
            }
        }
        void drawMetrics() {
            ImGui::Checkbox("Rolling window##metrics",&rollingMetrics);ImGui::SameLine();
            ImGui::SetNextItemWidth(100);ImGui::InputInt("Window (iterations)",&metricWindow);metricWindow=std::max(1,metricWindow);
            ImGui::SameLine();ImGui::Checkbox("Log Y scale",&logMetrics);
            const int count=ImGui::GetContentRegionAvail().x>900?3:2;
            if(ImGui::BeginTable("MetricPanels",count,ImGuiTableFlags_SizingStretchSame)) {
                ImGui::TableNextColumn();metrics.plot("Image loss / objective",{{"loss_rgb_mean","RGB"},{"loss_rgb_l2_mean","RGB L2"},{"loss_total_mean","Total"}},rollingMetrics,metricWindow,logMetrics);
                ImGui::TableNextColumn();metrics.plot("Weighted regularizers",{{"loss_normal_consistency_weighted_mean","Normal (left)"},{"loss_depth_distortion_weighted_mean","Depth (right)"}},rollingMetrics,metricWindow,logMetrics,120,true);
                ImGui::TableNextColumn();
                if(!geometryMetrics.iterations.empty()) {
                    geometryMetrics.plot("Ground truth distances",{{"cd","Chamfer"},{"accuracy","Accuracy"},{"completion","Completion"}},rollingMetrics,metricWindow,logMetrics);
                    ImGui::TextDisabled("Evaluated mesh at iteration %.0f · evaluation distance units",geometryMetrics.iterations.back());
                } else if(status.IsMap()&&status["ground_truth"].IsNull())ImGui::TextWrapped("GT evaluation is disabled for this run.");
                else ImGui::TextWrapped("GT Chamfer, accuracy and completion appear after mesh checkpoint evaluation. Configure mesh_extraction_interval in config.py to control evaluation cadence.");
                ImGui::EndTable();
            }
            if(ImGui::TreeNodeEx("Geometry / density / runtime",ImGuiTreeNodeFlags_DefaultOpen)) {
                if(ImGui::BeginTable("PopulationPanels",3,ImGuiTableFlags_SizingStretchSame)) {
                    ImGui::TableNextColumn();metrics.plot("Primitive population",{{"num_points","Surfels"},{"densification_clone_points_active","Clones"},{"densification_split_points_active","Splits"}},rollingMetrics,metricWindow,false,120,false,true);
                    ImGui::TableNextColumn();metrics.plot("Pruning events",{{"prune_scale_area_points","Area"},{"prune_opacity_points","Opacity"},{"prune_inactive_transport_points","Transport"}},rollingMetrics,metricWindow);
                    ImGui::TableNextColumn();metrics.plot("Iteration runtime (seconds)",{{"iteration_time_sec","Update"}},rollingMetrics,metricWindow);
                    ImGui::EndTable();
                }
                ImGui::TreePop();
            }
            if(ImGui::TreeNode("Other regularizers / raw terms / gradients")) {
                metrics.plot("Other weighted regularizers",{{"loss_alpha_weighted_mean","Alpha"},{"loss_intra_slab_depth_weighted_mean","Slab"}},rollingMetrics,metricWindow,logMetrics);
                metrics.plot("Raw consistency terms",{{"loss_normal_consistency_raw_mean","Normal"},{"loss_depth_distortion_raw_mean","Depth"},{"loss_intra_slab_depth_raw_mean","Slab"}},rollingMetrics,metricWindow,logMetrics);
                metrics.plot("Position gradient norms",{{"grad_position_renderer_norm","RGB"},{"grad_position_surface_regularizer_norm","Regularizer"},{"grad_position_total_norm","Total"}},rollingMetrics,metricWindow,logMetrics);
                metrics.plot("Opacity gradients",{{"grad_opacity_total_norm","Norm"},{"grad_opacity_total_max","Maximum"}},rollingMetrics,metricWindow,logMetrics);
                ImGui::TreePop();
            }
            if(ImGui::TreeNode("All recorded metrics (latest row)")) {
                if(ImGui::BeginTable("AllMetricValues",2,ImGuiTableFlags_BordersInnerH|ImGuiTableFlags_SizingStretchProp)) {
                    for(const auto& [name,value]:metrics.latestText) {ImGui::TableNextRow();ImGui::TableNextColumn();ImGui::TextUnformatted(name.c_str());ImGui::TableNextColumn();ImGui::TextUnformatted(value.c_str());}
                    for(const auto& [name,value]:geometryMetrics.latestText) {ImGui::TableNextRow();ImGui::TableNextColumn();ImGui::Text("GT %s",name.c_str());ImGui::TableNextColumn();ImGui::TextUnformatted(value.c_str());}
                    ImGui::EndTable();
                }
                ImGui::TreePop();
            }
        }
    };
}
