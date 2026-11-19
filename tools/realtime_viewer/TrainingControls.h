#pragma once
#include "TrainingSession.h"
#include "OptimizationSnapshots.h"
#include "DatasetLibrary.h"
#include "DatasetBrowser.h"
#include "WorkspaceUi.h"
#include "TrainingLog.h"
#include "TrainingMetrics.h"
#include "ReconstructionBrowser.h"
#include <imgui.h>
#include <yaml-cpp/yaml.h>
#include <charconv>
#include <cmath>
#include <cfloat>
#include <cstdio>
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
    enum class TrainingView { Live, Saved, History, Snapshot, Initializer };
    enum class TrainingLaunch { Primary, New };
    inline std::string trainingNumber(double value) {
        char buffer[32];
        const auto result = std::to_chars(std::begin(buffer), std::end(buffer), value);
        return {buffer, result.ptr};
    }
    inline std::optional<double> trainingParameterValue(const char* text) {
        char* end = nullptr;
        const double value = std::strtod(text, &end);
        if (end == text || *end || !std::isfinite(value) || value < 0.0) return std::nullopt;
        return value;
    }
    inline void trainingParameterRow(const char* label) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(label);
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
    }

    struct DensificationControls {
        enum class Preset { PythonDefaults, ObjectLevel, SceneLevel };
        inline static constexpr std::array labels{"Python defaults", "Object level", "Scene level"};
        inline static constexpr std::array keys{"python", "object", "scene"};

        struct Settings {
            int interval = 200;
            int start = 0;
            double splitFactor = 1.6;
            double maxNewFraction = 1.0;
            double gradientThreshold = 0.0004;
            double cloneScaleFraction = 0.0035;
            bool useCloning = true;

            bool operator==(const Settings&) const = default;
        };

        static Settings presetSettings(Preset value) {
            Settings result;
            if (value == Preset::SceneLevel) result.interval = 1000;
            return result;
        }

        Preset preset = Preset::PythonDefaults;
        Settings settings;
        std::array<Settings, 2> profiles{
            presetSettings(Preset::ObjectLevel), presetSettings(Preset::SceneLevel)};

        bool selectPreset(Preset value) {
            if (value == preset) return false;
            preset = value;
            settings = value == Preset::PythonDefaults ? Settings{} : profiles[profileIndex(value)];
            return true;
        }

        static std::size_t profileIndex(Preset value) {
            return static_cast<std::size_t>(value) - 1;
        }

        [[nodiscard]] static const char* validationError(const Settings& settings) {
            if (settings.interval < 0 || settings.start < 0)
                return "Densification interval and start must be non-negative";
            if (!std::isfinite(settings.splitFactor) || settings.splitFactor < 1.0)
                return "Split scale factor must be finite and at least 1";
            if (!std::isfinite(settings.maxNewFraction) || settings.maxNewFraction < 0.0 ||
                settings.maxNewFraction > 1.0)
                return "Max new fraction must be between 0 and 1";
            if (!std::isfinite(settings.gradientThreshold) || settings.gradientThreshold < 0.0)
                return "Position threshold must be finite and non-negative";
            if (!std::isfinite(settings.cloneScaleFraction) || settings.cloneScaleFraction <= 0.0)
                return "Clone scale fraction must be finite and positive";
            return nullptr;
        }

        bool rememberCurrentProfile() {
            if (preset == Preset::PythonDefaults || validationError(settings)) return false;
            auto& profile = profiles[profileIndex(preset)];
            if (profile == settings) return false;
            profile = settings;
            return true;
        }

        void appendArguments(std::vector<std::string>& args) const {
            if (preset == Preset::PythonDefaults) return;
            if (const auto* error = validationError(settings)) throw std::runtime_error(error);
            args.insert(args.end(), {
                "--densification-interval", std::to_string(settings.interval),
                "--densify-after", std::to_string(settings.start),
                "--densification-split-scale-factor", trainingNumber(settings.splitFactor),
                "--densification-max-new-fraction", trainingNumber(settings.maxNewFraction),
                "--densification-grad-abs-min", trainingNumber(settings.gradientThreshold),
                "--densification-grad-abs-min-final", trainingNumber(settings.gradientThreshold),
                "--densification-clone-scale-fraction", trainingNumber(settings.cloneScaleFraction),
                settings.useCloning ? "--densification-use-cloning" : "--no-densification-use-cloning"
            });
        }

        [[nodiscard]] static YAML::Node saveProfile(const Settings& settings) {
            YAML::Node saved;
            saved["interval"] = jsonNumber(settings.interval);
            saved["start"] = jsonNumber(settings.start);
            saved["split_factor"] = jsonNumber(settings.splitFactor);
            saved["max_new_fraction"] = jsonNumber(settings.maxNewFraction);
            saved["gradient_threshold"] = jsonNumber(settings.gradientThreshold);
            saved["clone_scale_fraction"] = jsonNumber(settings.cloneScaleFraction);
            saved["use_cloning"] = settings.useCloning;
            saved["use_cloning"].SetTag("?");
            return saved;
        }

        [[nodiscard]] YAML::Node preferences() const {
            YAML::Node saved;
            saved["preset"] = keys[static_cast<std::size_t>(preset)];
            for (std::size_t i = 1; i < keys.size(); ++i)
                saved["profiles"][keys[i]] = saveProfile(profiles[i - 1]);
            return saved;
        }

        [[nodiscard]] static Settings loadProfile(const YAML::Node& saved, Preset value,
                                                  bool legacy = false) {
            const Settings defaults = presetSettings(value);
            if (!saved || !saved.IsMap()) return defaults;
            Settings settings = defaults;
            settings.interval = saved["interval"].as<int>(settings.interval);
            settings.start = saved["start"].as<int>(settings.start);
            settings.splitFactor = saved["split_factor"].as<double>(settings.splitFactor);
            settings.maxNewFraction = saved["max_new_fraction"].as<double>(settings.maxNewFraction);
            settings.gradientThreshold = saved["gradient_threshold"].as<double>(settings.gradientThreshold);
            settings.cloneScaleFraction = saved["clone_scale_fraction"].as<double>(settings.cloneScaleFraction);
            settings.useCloning = saved["use_cloning"].as<bool>(settings.useCloning);
            // Older builds saved the full active profile, including the old Scene level defaults.
            if (legacy && value == Preset::SceneLevel) {
                if (settings.splitFactor == 1.1) settings.splitFactor = defaults.splitFactor;
                if (settings.maxNewFraction == 0.2) settings.maxNewFraction = defaults.maxNewFraction;
            }
            return validationError(settings) ? defaults : settings;
        }

        void restore(const YAML::Node& saved) {
            if (!saved || !saved.IsMap()) return;
            const auto key = saved["preset"].as<std::string>("");
            const auto storedProfiles = saved["profiles"];
            const bool hasProfiles = storedProfiles && storedProfiles.IsMap();
            preset = Preset::PythonDefaults;
            for (std::size_t i = 1; i < keys.size(); ++i) {
                const auto value = static_cast<Preset>(i);
                if (key == keys[i]) preset = value;
                const auto profile = hasProfiles ? storedProfiles[keys[i]] : (key == keys[i] ? saved : YAML::Node{});
                profiles[i - 1] = loadProfile(profile, value, !hasProfiles);
            }
            settings = preset == Preset::PythonDefaults ? Settings{} : profiles[profileIndex(preset)];
        }

        static void drawRecordedSettings(const YAML::Node& config, const char* title) {
            if (!config.IsMap() || !ImGui::TreeNode(title)) return;
            const std::array fields{
                std::pair{"Interval", "densification_interval"},
                std::pair{"Start iteration", "densify_after"},
                std::pair{"Split scale factor", "densification_split_scale_factor"},
                std::pair{"Max new fraction", "densification_max_new_fraction"},
                std::pair{"Position threshold", "densification_grad_abs_min"},
                std::pair{"Final threshold", "densification_grad_abs_min_final"},
                std::pair{"Clone small surfels", "densification_use_cloning"},
                std::pair{"Clone scale fraction", "densification_clone_scale_fraction"},
            };
            if (ImGui::BeginTable("RecordedDensification", 2, ImGuiTableFlags_SizingStretchProp)) {
                for (const auto& [label, key] : fields) {
                    trainingParameterRow(label);
                    const auto value = reconstructionSetting(config, key);
                    ImGui::TextUnformatted(value.IsScalar() ? value.Scalar().c_str() : "Unavailable");
                }
                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        bool draw(const YAML::Node& runConfig = {}, const char* recordedTitle = "Current run values") {
            if (!ImGui::TreeNodeEx("Densification", ImGuiTreeNodeFlags_Framed)) return false;
            bool selectionChanged = false;
            ImGui::TextUnformatted("Profile for new runs");
            const auto index = static_cast<std::size_t>(preset);
            std::string label = labels[index];
            if (settings != presetSettings(preset)) label += " (edited)";
            ImGui::SetNextItemWidth(-1);
            if (ImGui::BeginCombo("##DensificationPreset", label.c_str())) {
                for (std::size_t i = 0; i < labels.size(); ++i) {
                    if (ImGui::Selectable(labels[i], i == index))
                        selectionChanged = selectPreset(static_cast<Preset>(i));
                    if (i == index) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            if (preset == Preset::PythonDefaults)
                ImGui::TextWrapped("New runs use config.py. Select a profile to edit values.");
            else ImGui::TextWrapped("Click a value to edit. Valid edits save and apply when you start a new run.");
            if (preset != Preset::PythonDefaults &&
                ImGui::BeginTable("DensificationSettings", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Parameter", ImGuiTableColumnFlags_WidthStretch, .57f);
                ImGui::TableSetupColumn("Edit value", ImGuiTableColumnFlags_WidthStretch, .43f);
                ImGui::TableHeadersRow();
                trainingParameterRow("Interval");
                ImGui::InputInt("##interval", &settings.interval, 0);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Iterations between updates. Zero disables densification.");
                trainingParameterRow("Start iteration");
                ImGui::InputInt("##start", &settings.start, 0);
                trainingParameterRow("Split scale factor");
                ImGui::InputDouble("##split", &settings.splitFactor, 0, 0, "%.7g");
                trainingParameterRow("Max new fraction");
                ImGui::InputDouble("##max_new", &settings.maxNewFraction, 0, 0, "%.7g");
                trainingParameterRow("Position threshold");
                ImGui::InputDouble("##threshold", &settings.gradientThreshold, 0, 0, "%.7g");
                trainingParameterRow("Clone small surfels");
                ImGui::Checkbox("##cloning", &settings.useCloning);
                ImGui::BeginDisabled(!settings.useCloning);
                trainingParameterRow("Clone scale fraction");
                ImGui::InputDouble("##clone_fraction", &settings.cloneScaleFraction, 0, 0, "%.7g");
                ImGui::EndDisabled();
                ImGui::EndTable();
            }
            if (preset != Preset::PythonDefaults) {
                const auto defaults = presetSettings(preset);
                ImGui::BeginDisabled(settings == defaults);
                if (ImGui::Button("Reset profile")) settings = defaults;
                ImGui::EndDisabled();
                if (const auto* error = validationError(settings)) ImGui::TextWrapped("%s", error);
            }
            drawRecordedSettings(runConfig, recordedTitle);
            ImGui::TreePop();
            return rememberCurrentProfile() || selectionChanged;
        }
    };

    struct TrainingControls {
        TrainingSession process;
        std::array<char, 2048> dataset{}, output{}, groundTruth{}, python{}, workingDirectory{}, moduleDirectory{}, datasetRoot{};
        std::vector<DatasetEntry> datasets;
        DatasetBrowser datasetBrowser;
        bool evaluateGroundTruth=true;
        bool datasetChanged=false;
        std::optional<std::filesystem::path> initializedPointCloudRequest;
        bool loadingInitializer=false;
        int iterations = 30000, snapshotInterval = 100, samples = 1;
        float liveInterval = 1.0f;
        bool useWorkingSnapshot = false;
        bool overrideRunSettings=false,stopRequested=false;
        DensificationControls densification;
        bool rollingMetrics=false,logMetrics=false;
        int metricWindow=10'000;
        int requestedMonitorTab=-1;
        TrainingMetrics metrics,geometryMetrics,statusLossMetrics;
        ReconstructionBrowser reconstructionBrowser;
        std::optional<SavedReconstruction> openedReconstruction,preparedReconstruction;
        std::future<SavedReconstruction> reconstructionLoad;
        std::filesystem::path historyPath;
        uint64_t historyIteration=0;
        // Navigation is consumed by the UI at a renderer handoff; it never
        // rewinds or pauses the optimizer. -2/+2 are 1000-iteration jumps.
        std::optional<int> checkpointNavigation;
        std::optional<std::size_t> checkpointSelection;
        using ParameterValues = std::array<double, trainingParameterNames.size()>;
        std::array<std::array<char, 48>, trainingParameterNames.size()> launchParameters{};
        ParameterValues liveParameters{};
        std::optional<ParameterValues> appliedParameters;
        TrainingView view = TrainingView::Snapshot;
        YAML::Node status, live, runConfig;
        std::string message, log, selectableLog;
        ImGuiID logScrollInput=0,logScrollWindow=0;
        TrainingLog logReader;
        std::vector<float> losses;
        uint64_t plottedIteration = 0, viewedIteration = 0, loadedGeneration = 0;
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
                // Prefer the project's named Conda environment when it exists;
                // a leftover .venv may contain only the virtualenv bootstrap.
                const auto environmentName=repository.filename().string();
                for(const auto& candidate:{
                    expandedPath("~/miniconda3/envs/"+environmentName+"/bin/python",repository),
                    expandedPath("~/anaconda3/envs/"+environmentName+"/bin/python",repository),
                    repository/".venv/bin/python",
                    expandedPath("~/miniconda3/envs/pale/bin/python",repository),
                    expandedPath("~/anaconda3/envs/pale/bin/python",repository)})
                    if(std::filesystem::is_regular_file(candidate)) {set(python,candidate.string());break;}
            }
#ifdef PALE_TRAINING_MODULE_DIR
            set(moduleDirectory,PALE_TRAINING_MODULE_DIR);
#else
            if(std::filesystem::is_regular_file(repository/"cmake-build-pybind/pale.so"))
                set(moduleDirectory,(repository/"cmake-build-pybind").string());
#endif
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
                densification.restore(saved["densification"]);
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
            const auto input=expandedPath(dataset.data(),expandedPath(workingDirectory.data(),std::filesystem::current_path()));
            auto found=std::find_if(datasets.begin(),datasets.end(),[&](const auto& entry){return entry.directory==std::filesystem::path(dataset.data());});
            if(found!=datasets.end()) {
                const bool evaluation=evaluateGroundTruth;
                selectDataset(*found); evaluateGroundTruth=evaluation&&found->groundTruth.has_value();
            }
            else if(dataset[0]&&isDatasetDirectory(input)) {
                const bool evaluation=evaluateGroundTruth;
                selectDataset(datasetEntry(input));
                evaluateGroundTruth=evaluation&&groundTruth[0];
            }
            else if(!datasets.empty()) {
                auto teapot=std::find_if(datasets.begin(),datasets.end(),[](const auto& entry){return entry.directory.filename()=="teapot_pbdr";});
                selectDataset(teapot!=datasets.end()?*teapot:datasets.front());
            }
        }
        void chooseDatasetFolder() {
            try {
                const auto cwd=expandedPath(workingDirectory.data(),std::filesystem::current_path());
                datasetBrowser.open(expandedPath(dataset[0]?dataset.data():datasetRoot.data(),cwd));
            }catch(const std::exception& error) {message=error.what();}
        }
        void openDatasetLibrary() {
            try {
                if(!datasetRoot[0])throw std::runtime_error("Set the Dataset library folder in Run options first");
                const auto root=expandedPath(datasetRoot.data(),std::filesystem::current_path());
                std::filesystem::create_directories(root);
                const auto open=ImGui::GetPlatformIO().Platform_OpenInShellFn;
                if(!open||!open(ImGui::GetCurrentContext(),root.c_str()))
                    throw std::runtime_error("Could not open dataset folder: "+root.string());
                message="Dataset library: "+root.string();
            }catch(const std::exception& error) {message=error.what();}
        }
        [[nodiscard]] uint64_t integer(const char* name) const {
            try { return status[name].as<uint64_t>(); } catch (...) { return 0; }
        }
        [[nodiscard]] std::string state() const {
            if(stopRequested)return process.active()?"stopping":"stopped";
            try { return status["state"].as<std::string>(); } catch (...) { return process.active() ? "starting" : "idle"; }
        }
        [[nodiscard]] bool pending() const { return process.active() && !stopRequested && process.commandSequence > integer("ack"); }
        [[nodiscard]] bool hasRun() const {return openedReconstruction.has_value()||process.launched();}
        [[nodiscard]] bool initializingDataset() const {return initializedPointCloudRequest.has_value()||loadingInitializer;}
        [[nodiscard]] bool hasCurrentRun() const {return hasRun()&&view!=TrainingView::Initializer&&!initializingDataset();}
        [[nodiscard]] bool viewingEditableScene() const {return view==TrainingView::Snapshot||view==TrainingView::Initializer;}
        [[nodiscard]] bool openingSaved() const {return reconstructionLoad.valid()||preparedReconstruction.has_value();}
        [[nodiscard]] bool continuesSavedRun(TrainingLaunch mode=TrainingLaunch::Primary) const {
            return mode==TrainingLaunch::Primary&&hasCurrentRun()&&openedReconstruction&&!openedReconstruction->completed();
        }
        [[nodiscard]] bool usesWorkingSnapshot(TrainingLaunch mode) const {
            return useWorkingSnapshot&&!continuesSavedRun(mode);
        }
        [[nodiscard]] const char* continuationUnavailable() const {
            const auto& saved=*openedReconstruction;
            if(!saved.checkpointIterationKnown)return "The saved point cloud has no recorded iteration.";
            if(!saved.config.IsMap()||!reconstructionSetting(saved.config,"dataset_path").IsScalar())
                return "Continuing requires run_config.json with the original dataset path.";
            return nullptr;
        }
        [[nodiscard]] std::string progressState() const {
            const auto phase=state();
            if((process.active()&&phase!="completed")||!hasRun())return phase;
            const bool completed=openedReconstruction?openedReconstruction->completed():
                std::max(integer("iteration"),reconstructionIteration(metrics.latest("iteration")))>=reconstructionCompletionIteration;
            return completed?"Completed":phase=="failed"?"Failed":"Unfinished";
        }
        void markSavedRunCompleted() {
            if(!continuesSavedRun()||openingSaved()||process.active())return;
            const auto path=runDirectory()/reconstructionCompletionMarker;
            std::ofstream marker(path,std::ios::app);
            if(!marker) {message="Could not mark reconstruction as completed: "+path.string();return;}
            openedReconstruction->manuallyCompleted=true;
            message="Marked "+openedReconstruction->entry.label+" as completed";
        }
        [[nodiscard]] const std::filesystem::path& runDirectory() const {
            return openedReconstruction?openedReconstruction->entry.directory:process.runDirectory;
        }
        void openRunOutputFolder() {
            if(!hasCurrentRun()) {
                message="Start a reconstruction to create an output folder for this scene";
                return;
            }
            const auto& folder=runDirectory();
            std::error_code error;
            if(!std::filesystem::is_directory(folder,error)) {
                message="Reconstruction output folder is unavailable: "+folder.string();
                return;
            }
            const auto open=ImGui::GetPlatformIO().Platform_OpenInShellFn;
            if(!open||!open(ImGui::GetCurrentContext(),folder.c_str()))
                message="Could not open reconstruction output folder: "+folder.string();
        }
        [[nodiscard]] const std::filesystem::path& runIdentity() const {
            return openedReconstruction?openedReconstruction->entry.directory:process.directory;
        }
        void openReconstructionBrowser() {
            try {
                const auto cwd=expandedPath(workingDirectory.data(),std::filesystem::current_path());
                reconstructionBrowser.open(expandedPath(output.data(),cwd),hasRun()?runDirectory():std::filesystem::path{});
            }catch(const std::exception& error) {message=error.what();}
        }
        void requestOpenReconstruction(const std::filesystem::path& folder) {
            if(process.active())throw std::runtime_error("Stop the current reconstruction before opening a saved run");
            if(openingSaved())throw std::runtime_error("A saved reconstruction is already loading");
            reconstructionLoad=std::async(std::launch::async,[folder] {return loadSavedReconstruction(folder);});
            message="Loading saved reconstruction...";
        }
        void requestLatestReconstruction(const std::filesystem::path& folder={}) {
            // Repeated R presses coalesce; they never queue scene loads.
            if(openingSaved())return;
            if(openedReconstruction) {requestOpenReconstruction(runDirectory());return;}
            if(process.launched()) {checkpointNavigation=0;return;}
            const auto cwd=expandedPath(workingDirectory.data(),std::filesystem::current_path());
            const auto root=folder.empty()?expandedPath(output.data(),cwd):folder;
            reconstructionLoad=std::async(std::launch::async,[root] {return loadLatestSavedReconstruction(root);});
            message="Finding the latest reconstruction...";
        }
        void drawReconstructionBrowser() {
            if(const auto folder=reconstructionBrowser.draw(!process.active()&&!openingSaved())) {
                try {requestOpenReconstruction(*folder);}catch(const std::exception& error) {message=error.what();}
            }
        }
        // The UI calls this only at a renderer handoff, after preserving edits.
        void openPreparedReconstruction() {
            if(!preparedReconstruction||process.active())return;
            const bool sameRun=openedReconstruction&&runDirectory()==preparedReconstruction->entry.directory;
            openedReconstruction=std::move(preparedReconstruction);preparedReconstruction.reset();
            auto& saved=*openedReconstruction;
            metrics=std::move(saved.metrics);geometryMetrics=std::move(saved.geometryMetrics);
            status=YAML::Node(YAML::NodeType::Map);live=YAML::Node();
            status["state"]="saved";status["scene"]=saved.scene.string();status["run_dir"]=runDirectory().string();
            status["iteration"]=jsonNumber(saved.iteration);status["target_iteration"]=jsonNumber(saved.targetIteration);
            status["saved_iteration"]=jsonNumber(saved.checkpoints.back().iteration);
            status["ground_truth"]=reconstructionSetting(saved.config,"ground_truth");
            const float time=metrics.latest("iteration_time_sec");
            if(std::isfinite(time))status["iteration_ms"]=jsonNumber(static_cast<double>(time)*1000.);
            losses.clear();statusLossMetrics.reset();plottedIteration=0;loadedGeneration=0;stopRequested=false;
            checkpointNavigation.reset();checkpointSelection.reset();historyPath.clear();historyIteration=0;
            logReader.reset();log=std::move(saved.log);selectableLog.clear();logScrollInput=logScrollWindow=0;
            view=TrainingView::Saved;useWorkingSnapshot=false;if(!sameRun)requestedMonitorTab=1;
            message="Opened "+saved.entry.label+" · "+std::to_string(metrics.iterations.size())+" loss samples · "+
                std::to_string(geometryMetrics.iterations.size())+" geometry evaluations";
        }
        [[nodiscard]] std::vector<PointCloudSnapshot> checkpoints() const {
            if(openedReconstruction)return openedReconstruction->checkpoints;
            auto result=listOptimizationPointCloudSnapshots(process.runDirectory/"points");
            if(auto latest=latestSavedSnapshot();latest && (result.empty()||latest->path!=result.back().path))result.push_back(*latest);
            return result;
        }
        [[nodiscard]] std::optional<PointCloudSnapshot> latestSavedSnapshot() const {
            if(openedReconstruction)return openedReconstruction->checkpoints.empty()?std::nullopt:
                std::optional<PointCloudSnapshot>(openedReconstruction->checkpoints.back());
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
        void updateAppliedParameters(const YAML::Node& values) {
            if (!values || !values.IsMap()) return;
            ParameterValues next;
            for (std::size_t i = 0; i < next.size(); ++i) {
                const auto value = values[trainingParameterNames[i]];
                if (!value || !value.IsScalar()) return;
                const auto parsed = trainingParameterValue(value.Scalar().c_str());
                if (!parsed) return;
                next[i] = *parsed;
            }
            if (!appliedParameters || liveParameters == *appliedParameters) liveParameters = next;
            appliedParameters = next;
        }
        void poll() {
            try {
                if(const auto folder=datasetBrowser.poll()) {
                    selectDataset(datasetEntry(*folder));
                    message="Dataset selected for the next reconstruction";
                }
            }catch(const std::exception& error) {message=error.what();}
            process.poll();
            if(reconstructionLoad.valid()&&reconstructionLoad.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
                try {preparedReconstruction=reconstructionLoad.get();}
                catch(const std::exception& error) {message="Could not open reconstruction: "+std::string(error.what());}
            }
            if(openedReconstruction)return;
            const auto now = std::chrono::steady_clock::now();
            if (!process.launched() || now - lastPoll < std::chrono::milliseconds(250)) return;
            lastPoll = now;
            if (const auto next = readJson(process.directory / "status.json"); next.IsMap()) {
                status = next;
                updateAppliedParameters(next["parameters"]);
            }
            if (auto next = readJson(process.directory / "live.json"); next.IsMap()) live = next;
            if (!runConfig.IsMap()) runConfig = readJson(process.runDirectory / "run_config.json");
            if (integer("iteration") != plottedIteration && status["loss"].IsScalar()) {
                try {
                    const auto loss=status["loss"].as<float>();
                    if(std::isfinite(loss)) {
                        losses.push_back(loss);
                        statusLossMetrics.iterations.push_back(static_cast<float>(integer("iteration")));
                        statusLossMetrics.columns["loss_total_mean"].push_back(loss);
                    }
                } catch (...) {}
                plottedIteration = integer("iteration");
            }
            if(logReader.update(process.directory / "training.log"))log=logReader.text();
            metrics.update(process.runDirectory/"metrics.csv");
            geometryMetrics.update(process.runDirectory/"geometry_metrics.csv",true);
            if(stopRequested&&process.exitCode)message="Reconstruction stopped";
            else if (process.exitCode && *process.exitCode != 0)
                message = "Training exited with code " + std::to_string(*process.exitCode) + ". Open Run monitor > Log for details.";
        }
        void request(const char* command, bool tune = false) {
            if (!process.active() || stopRequested) return;
            if(std::string(command)=="stop") {
                // Kill the owned process group without waiting for a training
                // boundary, final PLY/mesh extraction, or textured glTF export.
                stopRequested=true;process.shutdown();message="Stopping reconstruction...";
                return;
            }
            if(pending())return;
            YAML::Node request;
            if (tune) {
                if (!appliedParameters || liveParameters == *appliedParameters) return;
                for (std::size_t i = 0; i < liveParameters.size(); ++i) {
                    const double value = liveParameters[i];
                    if (!std::isfinite(value) || value < 0)
                        throw std::runtime_error("Parameters must be finite and non-negative");
                    if (value != (*appliedParameters)[i])
                        request["parameters"][trainingParameterNames[i]] = jsonNumber(value);
                }
            }
            request["sequence"] = jsonNumber(++process.commandSequence); request["command"] = command;
            try { writeJsonAtomic(process.directory / "command.json", request); }
            catch (...) { --process.commandSequence; throw; }
        }
        void send(const char* command, bool tune = false) {
            try { request(command, tune); } catch (const std::exception& error) { message = error.what(); }
        }
        [[nodiscard]] std::vector<std::string> trainingCommand(const std::filesystem::path& cwd) const {
            if (!std::isfinite(liveInterval) || liveInterval < .1f)
                throw std::runtime_error("Live interval must be >= 0.1 s");
            std::vector<std::string> args;
            if (moduleDirectory[0]) {
                std::string path = expandedPath(moduleDirectory.data(), cwd).string();
                if (const char* inherited = std::getenv("PYTHONPATH")) path += ":" + std::string(inherited);
                args = {"/usr/bin/env", "PYTHONPATH=" + path};
            }
            const std::string executable = python.data();
            args.push_back(executable.find('/') == std::string::npos ? executable : expandedPath(executable, cwd).string());
            args.insert(args.end(), {"-u", (cwd / "main.py").string(),
                "--viewer-live-interval", std::to_string(liveInterval),
                "--no-metrics", "--no-image-preview"});
            return args;
        }
        void addNewRunArguments(std::vector<std::string>& args,const std::filesystem::path& cwd,
                                const std::filesystem::path& workspace) const {
            const auto input = expandedPath(dataset.data(), cwd);
            if (!std::filesystem::is_directory(input / "images")) throw std::runtime_error("Dataset needs an images/ directory");
            if (workspace.empty() && (!std::filesystem::is_regular_file(input / "scene.xml") ||
                                      !std::filesystem::is_regular_file(input / "points.ply")))
                throw std::runtime_error("Dataset needs scene.xml and points.ply");
            if (overrideRunSettings&&(iterations < 1 || snapshotInterval < 1 || samples < 1))
                throw std::runtime_error("Iterations, snapshot interval and samples must be positive");
            args.insert(args.end(), {"-s", input.string()});
            densification.appendArguments(args);
            for (std::size_t i = 0; i < launchParameters.size(); ++i) {
                const char* text = launchParameters[i].data();
                if (!*text) continue;
                if (!trainingParameterValue(text))
                    throw std::runtime_error(std::string(trainingParameterLabels[i]) + " must be finite and non-negative");
                std::string flag = i < 6 ? std::string("--lr-") + std::array<const char*,6>{"pos","rot","scale","albedo","opacity","beta"}[i]
                    : std::string("--") + trainingParameterNames[i];
                std::replace(flag.begin(), flag.end(), '_', '-');
                args.insert(args.end(), {flag, text});
            }
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
        }
        void saveDensificationPreferences() {
            try {
                auto saved = readJson(preferences);
                if (!saved.IsMap()) saved = YAML::Node(YAML::NodeType::Map);
                saved["densification"] = densification.preferences();
                writeJsonAtomic(preferences, saved);
            } catch (const std::exception& error) {
                message = "Could not save densification profiles. " + std::string(error.what());
            }
        }
        void launch(const std::filesystem::path& workspace = {},TrainingLaunch mode=TrainingLaunch::Primary) {
            if(process.active())throw std::runtime_error("A reconstruction is already running");
            if(openingSaved())throw std::runtime_error("Wait for the saved reconstruction to finish loading");
            if(initializingDataset())throw std::runtime_error("Wait for the initialized point cloud to finish loading");
            const auto cwd = expandedPath(workingDirectory.data(), std::filesystem::current_path());
            auto args=trainingCommand(cwd);
            const bool continuing=continuesSavedRun(mode);
            if(continuing) {
                const auto& saved=*openedReconstruction;
                if(reconstructionFile(runDirectory()/reconstructionCompletionMarker)) {
                    openedReconstruction->manuallyCompleted=true;
                    throw std::runtime_error("This reconstruction has been marked as completed");
                }
                if(const char* reason=continuationUnavailable())throw std::runtime_error(reason);
                const auto& checkpoint=saved.checkpoints.back();
                args.insert(args.end(),{"--checkpoint",runDirectory().string(),"--scene",saved.scene.string(),
                    "--ply",checkpoint.path.string(),"--resume-iteration-offset",std::to_string(checkpoint.iteration),
                    "--iterations",std::to_string(reconstructionCompletionIteration)});
            } else addNewRunArguments(args,cwd,workspace);
            process.launch(expandedPath(output.data(), cwd), cwd, args);
            openedReconstruction.reset();
            status = YAML::Node(); live = YAML::Node(); runConfig = YAML::Node();
            losses.clear(); appliedParameters.reset();
            metrics.reset();geometryMetrics.reset();statusLossMetrics.reset();stopRequested=false;
            logReader.reset();log.clear();selectableLog.clear();
            checkpointNavigation.reset();checkpointSelection.reset();historyPath.clear();
            plottedIteration = viewedIteration = loadedGeneration = 0; view = TrainingView::Live;
            YAML::Node saved;
            saved["dataset"] = std::string(dataset.data()); saved["output"] = std::string(output.data());
            saved["ground_truth"] = std::string(groundTruth.data()); saved["python"] = std::string(python.data());
            saved["working_directory"] = std::string(workingDirectory.data()); saved["module_directory"] = std::string(moduleDirectory.data());
            saved["dataset_root"]=std::string(datasetRoot.data());
            saved["environment_preferences_version"]=jsonNumber(1);
            try {
                densification.rememberCurrentProfile();
                saved["densification"] = densification.preferences();
                writeJsonAtomic(preferences, saved);
            } catch (...) {} // A launched job must not be reported as a failed launch.
            message = continuing?"Continuing from the saved checkpoint; waiting for the first live scene":
                "Training launched; waiting for the first live scene";
        }

        void drawHyperparameters() {
            if (!ImGui::TreeNodeEx("Hyperparameters", ImGuiTreeNodeFlags_Framed)) return;
            const bool running = process.active();
            if (running && !appliedParameters) {
                ImGui::TextWrapped("Waiting for the optimizer to report its values.");
                ImGui::TreePop();
                return;
            }
            ImGui::TextUnformatted(running ? "Current run" : "New reconstruction");
            ImGui::TextWrapped("%s", running ? "Click a value to edit, then apply the changes." :
                "Type a value to use it when you start a new run. Blank fields use config.py.");
            const bool awaiting = running && pending();
            ImGui::BeginDisabled(running && (awaiting || stopRequested));
            if (ImGui::BeginTable("HyperparameterValues", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Parameter", ImGuiTableColumnFlags_WidthStretch, .57f);
                ImGui::TableSetupColumn("Edit value", ImGuiTableColumnFlags_WidthStretch, .43f);
                ImGui::TableHeadersRow();
                for (std::size_t i = 0; i < launchParameters.size(); ++i) {
                    trainingParameterRow(trainingParameterLabels[i]);
                    ImGui::PushID(static_cast<int>(i));
                    if (running) {
                        ImGui::InputDouble("##live", &liveParameters[i], 0, 0, "%.9g");
                        if (liveParameters[i] != (*appliedParameters)[i])
                            ImGui::TextDisabled("In use %s", trainingNumber((*appliedParameters)[i]).c_str());
                    } else {
                        auto& text = launchParameters[i];
                        ImGui::InputTextWithHint("##launch", "config.py", text.data(), text.size(),
                            ImGuiInputTextFlags_CharsScientific);
                        if (text[0] && !trainingParameterValue(text.data()))
                            ImGui::TextWrapped("Enter a non-negative number.");
                    }
                    ImGui::PopID();
                }
                ImGui::EndTable();
            }
            ImGui::EndDisabled();
            if (running) {
                const bool changed = liveParameters != *appliedParameters;
                ImGui::BeginDisabled(!changed || awaiting || stopRequested);
                if (ImGui::Button("Apply at next iteration", ImVec2(-1, 0))) send("tune", true);
                ImGui::EndDisabled();
                ImGui::TextDisabled("%s", !changed ? "Shown values are in use." :
                    awaiting ? "Waiting for the optimizer." : "Edits waiting to be applied.");
                const auto error = status["command_error"];
                if (changed && !awaiting && error.IsScalar() && !error.Scalar().empty())
                    ImGui::TextWrapped("%s", error.Scalar().c_str());
            }
            ImGui::TextWrapped("Learning rates are bases for scaling and decay.");
            ImGui::TreePop();
        }
        // Returns true when the user requests a launch; ownership changes live in main.cpp.
        bool drawSetup(bool hasWorkspace, bool previewBusy=false) {
            bool start = false;
            ImGui::TextColored(ui::accent,"JOINT PBdR");
            ImGui::TextDisabled("Geometry  /  Materials  /  Emitters");
            ImGui::Spacing(); ImGui::SeparatorText("Dataset");
            const auto selected=std::find_if(datasets.begin(),datasets.end(),[&](const auto& entry){return entry.directory==std::filesystem::path(dataset.data());});
            const std::string label=selected==datasets.end() ? (dataset[0] ? std::filesystem::path(dataset.data()).filename().string() : "Choose dataset") : selected->label;
            const float datasetButtonWidth=28.f;
            ImGui::SetNextItemWidth(-2.f*(datasetButtonWidth+ImGui::GetStyle().ItemSpacing.x));
            if(ImGui::BeginCombo("##Dataset",label.c_str())) {
                for(const auto& entry:datasets) {
                    const bool active=entry.directory==std::filesystem::path(dataset.data());
                    if(ImGui::Selectable((entry.label+(entry.groundTruth?"  ·  GT":"")).c_str(),active))selectDataset(entry);
                    if(active)ImGui::SetItemDefaultFocus();
                }
                if(datasets.empty())ImGui::TextDisabled("No complete datasets found");
                ImGui::Separator();
                if(ImGui::Selectable("Choose dataset folder..."))chooseDatasetFolder();
                ImGui::EndCombo();
            }
            if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",dataset.data());
            ImGui::SameLine();
            if(ui::iconButton("Choose dataset folder",ui::Icon::Folder,false,ImVec2(datasetButtonWidth,ImGui::GetFrameHeight())))chooseDatasetFolder();
            if(ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)||ImGui::IsItemFocused())
                ImGui::SetTooltip("Choose a dataset from any folder, or paste its path.\nThe folder must contain images/, scene.xml and points.ply.");
            ImGui::SameLine();
            if(ui::iconButton("Refresh dataset library",ui::Icon::Refresh,false,ImVec2(datasetButtonWidth,ImGui::GetFrameHeight())))refreshDatasets();
            const bool canLoadInitializer=dataset[0]&&!initializingDataset();
            ImGui::BeginDisabled(!canLoadInitializer);
            if(ImGui::Button("Load initialized point cloud",ImVec2(-1,0))) {
                const auto cwd=expandedPath(workingDirectory.data(),std::filesystem::current_path());
                initializedPointCloudRequest=expandedPath(dataset.data(),cwd);
            }
            ImGui::EndDisabled();
            if(ImGui::IsItemHovered())ImGui::SetTooltip("Inspect the selected dataset's starting point cloud without starting reconstruction.");
            if(groundTruth[0])ImGui::Checkbox("Evaluate with dataset ground truth",&evaluateGroundTruth);
            else ImGui::TextDisabled("Ground truth optional · not included");
            ImGui::Spacing(); ImGui::SeparatorText("Reconstruction");
            ImGui::Checkbox("Override run settings from config.py",&overrideRunSettings);
            if(overrideRunSettings) {ImGui::SetNextItemWidth(-1);ImGui::InputInt("##iterations",&iterations);ImGui::TextDisabled("Target iterations");}
            else ImGui::TextWrapped("Iterations, sampling and checkpoint cadence use config.py.");
            if (hasWorkspace) { ImGui::Checkbox("Initialize from edited snapshot", &useWorkingSnapshot); }
            else useWorkingSnapshot = false;
            if (ImGui::TreeNode("Python environment")) {
                ImGui::SetNextItemWidth(-1);
                ui::pathInput("Python executable",python);
                ui::pathInput("Working directory",workingDirectory);
                ui::pathInput("Native module directory",moduleDirectory);
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
                ImGui::SameLine();if(ImGui::Button("Open library folder"))openDatasetLibrary();
                ui::pathInput("GT override",groundTruth);
                ImGui::TreePop();
            }
            drawHyperparameters();
            const auto& recordedConfig = openedReconstruction ? openedReconstruction->config : runConfig;
            const char* recordedTitle = openedReconstruction ? "Saved run values" :
                process.active() ? "Current run values" : "Last run values";
            if (densification.draw(recordedConfig, recordedTitle)) saveDensificationPreferences();
            if(continuesSavedRun())
                ImGui::TextWrapped("Continue %s to 30,000 iterations using its saved settings.",openedReconstruction->entry.label.c_str());
            start=drawRunButtons(ImVec2(-1,32),previewBusy&&useWorkingSnapshot);
            if(continuesSavedRun()) {
                if(openingSaved())ImGui::BeginDisabled();
                if(ImGui::Button("Mark as completed",ImVec2(-1,0)))markSavedRunCompleted();
                if(openingSaved())ImGui::EndDisabled();
            }
            if(process.active()||openingSaved())ImGui::BeginDisabled();
            if(ImGui::Button("Open previous reconstruction...",ImVec2(-1,0)))openReconstructionBrowser();
            if(process.active()||openingSaved())ImGui::EndDisabled();
            if(ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))ImGui::SetTooltip("%s",process.active()?
                "Stop the current reconstruction before opening a saved run.":"Restore a saved scene, loss history and GT metrics from an output folder.");
            const bool canOpenOutput=hasCurrentRun();
            ImGui::BeginDisabled(!canOpenOutput);
            if(ImGui::Button("Open output folder",ImVec2(-1,0)))openRunOutputFolder();
            ImGui::EndDisabled();
            if(!canOpenOutput&&ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("%s",initializingDataset()?"Loading the initialized point cloud.":"Start a reconstruction to create an output folder for this scene.");
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
                        if(!metrics.iterations.empty())metrics.plot("Objective",{{"loss_total_mean","Total"}},rollingMetrics,metricWindow,logMetrics,100);
                        else if(!losses.empty()) {
                            statusLossMetrics.plot("Objective",{{"loss_total_mean","Total"}},rollingMetrics,metricWindow,logMetrics,100);
                        } else ImGui::TextWrapped("Start a reconstruction to track its objective here.");
                        ImGui::TableNextColumn();
                        ImGui::Text("Iteration %llu / %llu",static_cast<unsigned long long>(integer("iteration")),static_cast<unsigned long long>(integer("target_iteration")));
                        const float progress=integer("target_iteration") ? static_cast<float>(integer("iteration"))/integer("target_iteration") : 0;
                        ImGui::ProgressBar(progress,ImVec2(-1,7),"");
                        ImGui::TextDisabled("Viewing %llu  ·  Saved %llu",static_cast<unsigned long long>(viewedIteration),static_cast<unsigned long long>(integer("saved_iteration")));
                        ImGui::TextDisabled("%s",view==TrainingView::Initializer?"Viewing initialized point cloud":
                            view==TrainingView::Snapshot?"Editable snapshot · training independent":openedReconstruction?"Read-only saved reconstruction":"Read-only training view");
                        ImGui::TableNextColumn();
                        ImGui::TextColored(process.active()?ui::green:ui::muted,"%s",progressState().c_str());
                        if(ImGui::IsItemHovered())ImGui::SetTooltip("%s\n%s",runDirectory().string().c_str(),openedReconstruction?"Saved output opened for inspection.":"Closing this viewer stops its reconstruction.");
                        const bool disabled=!process.active()||pending()||stopRequested||state()=="finalizing";
                        if(disabled)ImGui::BeginDisabled();
                        if(ImGui::Button(state()=="paused"?"Continue":"Pause"))send(state()=="paused"?"resume":"pause");
                        if(disabled)ImGui::EndDisabled();
                        if(process.active()) {ImGui::SameLine();drawStopButton(ImVec2(0,0));}
                        ImGui::SameLine();
                        if(disabled||stopRequested)ImGui::BeginDisabled();
                        if(ImGui::Button("Save checkpoint"))send("save");
                        if(disabled||stopRequested)ImGui::EndDisabled();
                        if(pending())ImGui::TextDisabled("Waiting for iteration boundary");
                        else if(!openedReconstruction&&process.exitCode) {
                            if(stopRequested)ImGui::TextDisabled("Stopped by user");
                            else ImGui::TextDisabled("Process exited: %d",*process.exitCode);
                        }
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
                    if(!openedReconstruction&&!stopRequested&&process.exitCode&&*process.exitCode!=0)ImGui::TextWrapped("%s",message.c_str());
                    ImGui::EndTabItem();
                }
                if(ImGui::BeginTabItem("Metrics",nullptr,requested==1?ImGuiTabItemFlags_SetSelected:0)) {drawMetrics();ImGui::EndTabItem();}
                if(ImGui::BeginTabItem("Checkpoints",nullptr,requested==2?ImGuiTabItemFlags_SetSelected:0)) {drawCheckpoints();ImGui::EndTabItem();}
                if(ImGui::BeginTabItem("Log",nullptr,requested==3?ImGuiTabItemFlags_SetSelected:0)) {
                    if(ImGui::Button("Copy log"))ImGui::SetClipboardText(log.c_str());
                    ImGui::SameLine();
                    const bool hasCommand=openedReconstruction?!openedReconstruction->launchCommand.empty():!process.arguments.empty();
                    if(!hasCommand)ImGui::BeginDisabled();
                    if(ImGui::Button("Copy launch command"))ImGui::SetClipboardText(openedReconstruction?openedReconstruction->launchCommand.c_str():shellDisplay(process.arguments).c_str());
                    if(!hasCommand)ImGui::EndDisabled();
                    ImGui::SameLine(); ImGui::TextDisabled("%s",runDirectory().filename().string().c_str());
                    if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",runDirectory().string().c_str());
                    if(openedReconstruction&&log.empty())ImGui::TextDisabled("No training log was retained for this reconstruction.");
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
        static constexpr float stopButtonWidth=60.f;
        bool drawRunButtons(ImVec2 size,bool previewBlocked=false) {
            const bool running=process.active();
            if(running&&size.x<0)size.x=std::max(1.f,ImGui::GetContentRegionAvail().x-stopButtonWidth-ImGui::GetStyle().ItemSpacing.x);
            const bool start=drawStartButton(size,previewBlocked);
            if(running) {ImGui::SameLine();drawStopButton(ImVec2(stopButtonWidth,size.y));}
            return start;
        }
        bool drawStopButton(ImVec2 size) {
            if(!process.active())return false;
            const bool disabled=stopRequested;
            if(disabled)ImGui::BeginDisabled();
            const bool pressed=ImGui::Button("Stop###StopReconstruction",size);
            if(disabled)ImGui::EndDisabled();
            if(ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("%s",stopRequested?"Stopping reconstruction...":
                    "Stop reconstruction immediately, including mesh extraction and glTF export. Keep the last displayed scene and existing checkpoints.");
            if(pressed&&!disabled)send("stop");
            return pressed&&!disabled;
        }
        bool drawStartButton(ImVec2 size, bool previewBlocked=false) {
            const bool running=process.active(),continuing=continuesSavedRun();
            const char* unavailable=continuing?continuationUnavailable():nullptr;
            const bool disabled=running||openingSaved()||initializingDataset()||unavailable||(!continuing&&previewBlocked);
            const auto phase=running?state():std::string{};
            const char* caption=!running?(continuing?"Continue reconstruction":"Start reconstruction"):
                phase=="stopping"?"Stopping...":
                phase=="finalizing"||phase=="completed"||phase=="stopped"?"Finalizing...":
                phase=="paused"?"Paused":"Reconstructing...";
            const std::string label=std::string(caption)+"###StartReconstruction";
            if(disabled)ImGui::BeginDisabled();
            ImGui::PushStyleColor(ImGuiCol_Button,ui::accent);
            ImGui::PushStyleColor(ImGuiCol_Text,ImVec4(.13f,.11f,.09f,1));
            const bool pressed=ImGui::Button(label.c_str(),size);
            ImGui::PopStyleColor(2);
            const auto min=ImGui::GetItemRectMin(),max=ImGui::GetItemRectMax();
            if(disabled)ImGui::EndDisabled();
            if(running) {
                const float text=ImGui::CalcTextSize(caption).x;
                const ImVec2 center(min.x+std::max(9.f,(max.x-min.x-text)*.5f-12.f),(min.y+max.y)*.5f);
                const float angle=static_cast<float>(ImGui::GetTime())*5.f;
                auto* draw=ImGui::GetWindowDrawList();
                draw->PathArcTo(center,5.f,angle,angle+4.8f,18);
                draw->PathStroke(ImGui::GetColorU32(ui::accent),0,2.f);
            }
            if(disabled&&ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("%s",running?"A reconstruction is running. You can launch another after it exits.":openingSaved()?"A saved reconstruction is loading.":
                    initializingDataset()?"Loading the initialized point cloud.":unavailable?unavailable:"Wait for the current preview before starting from an edited snapshot.");
            else if(continuing&&ImGui::IsItemHovered())
                ImGui::SetTooltip("Resume the saved checkpoint at iteration %llu using its recorded settings. Save the continuation in a new output folder.",
                    static_cast<unsigned long long>(openedReconstruction->checkpoints.back().iteration));
            return pressed&&!disabled;
        }
        void drawCheckpoints() {
            if(!hasRun()) {ImGui::TextWrapped("Start or open a reconstruction to browse its saved checkpoints.");return;}
            const auto saved=checkpoints();
            ImGui::Text("Viewing iteration %llu · training %llu",static_cast<unsigned long long>(viewedIteration),static_cast<unsigned long long>(integer("iteration")));
            if(ImGui::Button(process.active()&&!openedReconstruction?"Live (R)":"Latest (R)"))checkpointNavigation=0;
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
                ImGui::TextDisabled("%zu saved checkpoints · %s",saved.size(),openedReconstruction?"saved output inspection":"browsing leaves optimization running");
            }
        }
        void drawMetrics() {
            ImGui::Checkbox("Rolling window##metrics",&rollingMetrics);ImGui::SameLine();
            ImGui::SetNextItemWidth(100);ImGui::InputInt("Window (iterations)",&metricWindow);metricWindow=std::max(1,metricWindow);
            ImGui::SameLine();ImGui::Checkbox("Log Y scale",&logMetrics);
            const int count=ImGui::GetContentRegionAvail().x>900?3:2;
            if(ImGui::BeginTable("MetricPanels",count,ImGuiTableFlags_SizingStretchSame)) {
                ImGui::TableNextColumn();metrics.plot("Image loss / objective",{{"loss_rgb_mean","RGB"},{"loss_rgb_l2_mean","RGB L2"},{"loss_total_mean","Total"}},rollingMetrics,metricWindow,logMetrics);
                ImGui::TableNextColumn();
                metrics.plot("Weighted regularizers",{
                    {"loss_normal_consistency_weighted_mean","Normal (left)"},
                    {"loss_depth_distortion_weighted_mean","Depth (right)"},
                    {"loss_alpha_weighted_mean","Alpha (right)"}
                },rollingMetrics,metricWindow,logMetrics,120,true);
                ImGui::TableNextColumn();
                if(!geometryMetrics.iterations.empty()) {
                    geometryMetrics.plot("Ground truth distances",{{"cd","Chamfer"},{"accuracy","Accuracy"},{"completion","Completion"}},rollingMetrics,metricWindow,logMetrics);
                    ImGui::TextDisabled("Evaluated mesh at iteration %.0f · evaluation distance units",geometryMetrics.iterations.back());
                } else if(openedReconstruction)ImGui::TextWrapped("No Chamfer, accuracy or completion evaluations were saved for this reconstruction.");
                else if(status.IsMap()&&status["ground_truth"].IsNull())ImGui::TextWrapped("GT evaluation is disabled for this run.");
                else ImGui::TextWrapped("GT Chamfer, accuracy and completion appear after mesh checkpoint evaluation. Configure mesh_extraction_interval in config.py to control evaluation cadence.");
                ImGui::EndTable();
            }
            if(ImGui::TreeNodeEx("Geometry / density / runtime",ImGuiTreeNodeFlags_DefaultOpen)) {
                if(ImGui::BeginTable("PopulationPanels",3,ImGuiTableFlags_SizingStretchSame)) {
                    ImGui::TableNextColumn();metrics.plot("Primitive population",{{"num_points","Surfels"},{"densification_clone_points_active","Clones"},{"densification_split_points_active","Splits"}},rollingMetrics,metricWindow,false,120,false,true);
                    ImGui::TableNextColumn();metrics.plot("Cumulative pruning",{{"prune_scale_area_points","Area"},{"prune_opacity_points","Opacity"},{"prune_inactive_transport_points","Transport"}},rollingMetrics,metricWindow,false,120,false,true,true);
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
