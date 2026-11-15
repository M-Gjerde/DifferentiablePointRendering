#pragma once
#include "TrainingControls.h"

namespace viewer {
    // A separate, viewer-owned extractor process. It never replaces the
    // inspected point cloud and cannot queue another export while active.
    struct MeshExportControls {
        TrainingSession process;
        std::filesystem::path repository;
        std::array<char,2048> outputRoot{};
        bool texturedGltf=false,includeLights=true,includeCameras=false;
        int textureSize=1024,uvPartitions=30,uvThreads=8;
        int meshResolution=2048,components=0,minTriangles=50;
        float voxelSize=-1,sdfTrunc=-1,depthTrunc=-1,metersPerUnit=1;
        int depthKind=0;
        std::string message,log,selectableLog;
        TrainingLog logReader;
        bool cancelled=false,expectedGltf=false;
        std::chrono::steady_clock::time_point lastPoll{};

        explicit MeshExportControls(const std::filesystem::path& root):repository(root) {
            TrainingControls::set(outputRoot,(root/"output/mesh_exports").string());
        }
        static std::filesystem::path freezeWorkspace(const std::filesystem::path& workspace,
                                                     const std::filesystem::path& destination) {
            auto state=readJson(workspace);
            if(!state.IsMap()||!state["pointcloud"].IsScalar()||!state["source_scene"].IsScalar()||!state["asset_root"].IsScalar())
                throw std::runtime_error("Mesh export needs a complete scene snapshot");
            const auto points=std::filesystem::absolute(state["pointcloud"].as<std::string>());
            const auto scene=std::filesystem::absolute(state["source_scene"].as<std::string>());
            if(!std::filesystem::is_regular_file(points)||!std::filesystem::is_regular_file(scene))
                throw std::runtime_error("Mesh export snapshot source is unavailable");
            std::filesystem::create_directories(destination);
            // Viewer PLY files are immutable or atomically replaced. A private
            // hard link freezes that inode even after a live packet is removed.
            std::error_code error;
            std::filesystem::create_hard_link(points,destination/"points.ply",error);
            if(error)std::filesystem::copy_file(points,destination/"points.ply");
            std::filesystem::copy_file(scene,destination/"source_scene.xml");
            state["source_scene_base"]=scene.parent_path().string();
            state["source_scene"]=(destination/"source_scene.xml").string();
            state["pointcloud"]=(destination/"points.ply").string();
            if(!state["entities"].IsSequence())state["entities"]=YAML::Node(YAML::NodeType::Sequence);
            const auto frozen=destination/"workspace.json";
            writeJsonAtomic(frozen,state);
            return frozen;
        }
        [[nodiscard]] std::vector<std::string> extractionOptions() const {
            if(meshResolution<1||components<0||minTriangles<0)
                throw std::runtime_error("Mesh resolution must be positive; component limits must be nonnegative");
            for(const float value:{voxelSize,sdfTrunc,depthTrunc})if(!std::isfinite(value)||value==0)
                throw std::runtime_error("TSDF distances must be positive, or negative for automatic selection");
            std::vector<std::string> args{"--mesh-res",std::to_string(meshResolution),
                "--num-cluster",std::to_string(components),"--min-cluster-triangles",std::to_string(minTriangles),
                "--voxel-size",trainingNumber(voxelSize),"--sdf-trunc",trainingNumber(sdfTrunc),
                "--depth-trunc",trainingNumber(depthTrunc),"--depth-key",depthKind==0?"median_depth":"mean_depth",
                texturedGltf?"--export-gltf":"--no-export-gltf"};
            if(texturedGltf) {
                if(textureSize<16||textureSize>16384||uvPartitions<0||uvThreads<0||
                    !std::isfinite(metersPerUnit)||metersPerUnit<=0)
                    throw std::runtime_error("Texture size must be 16–16384; UV settings nonnegative; scene units positive");
                args.insert(args.end(),{"--texture-size",std::to_string(textureSize),
                    "--uv-partitions",std::to_string(uvPartitions),"--uv-threads",std::to_string(uvThreads),
                    "--meters-per-unit",trainingNumber(metersPerUnit),
                    includeLights?"--export-lights":"--no-export-lights",
                    includeCameras?"--export-cameras":"--no-export-cameras"});
            }
            return args;
        }
        void launch(const TrainingControls& environment,const std::filesystem::path& workspace,
                    const std::filesystem::path& runConfig={}) {
            if(process.active())throw std::runtime_error("A mesh export is already running");
            auto options=extractionOptions();
            const auto cwd=expandedPath(environment.workingDirectory.data(),repository);
            const auto output=expandedPath(outputRoot.data(),repository);
            const auto bridge=cwd/"viewer_mesh_export.py";
            if(!std::filesystem::is_regular_file(bridge))throw std::runtime_error("Python working directory needs viewer_mesh_export.py");
            const auto input=reserveDirectory(output/".inputs","snapshot_");
            const auto frozen=freezeWorkspace(workspace,input);
            std::vector<std::string> args;
            if(environment.moduleDirectory[0]) {
                std::string path=expandedPath(environment.moduleDirectory.data(),cwd).string();
                if(const char* inherited=std::getenv("PYTHONPATH"))path+=":"+std::string(inherited);
                args={"/usr/bin/env","PYTHONPATH="+path};
            }
            const std::string python=environment.python.data();
            args.push_back(python.find('/')==std::string::npos?python:expandedPath(python,cwd).string());
            args.insert(args.end(),{"-u",bridge.string(),"--workspace",frozen.string()});
            if(!runConfig.empty()&&std::filesystem::is_regular_file(runConfig)) {
                std::filesystem::copy_file(runConfig,input/"source_run_config.json");
                args.insert(args.end(),{"--run-config",(input/"source_run_config.json").string()});
            }
            args.insert(args.end(),options.begin(),options.end());
            process.launch(output,cwd,args);
            cancelled=false;expectedGltf=texturedGltf;logReader.reset();log.clear();selectableLog.clear();lastPoll={};
            message="Extracting the frozen point cloud; the viewport stays on its PLY.";
        }
        void shutdown() {process.shutdown();}
        void poll() {
            const auto now=std::chrono::steady_clock::now();
            if(now-lastPoll<std::chrono::milliseconds(200))return;
            lastPoll=now;process.poll();
            if(process.directory.empty())return;
            if(logReader.update(process.directory/"training.log"))log=logReader.text();
            if(process.exitCode) {
                if(cancelled)message="Mesh export cancelled.";
                else if(*process.exitCode!=0)message="Mesh export failed. See its log below.";
                else if(expectedGltf&&!std::filesystem::is_regular_file(process.runDirectory/"mesh/reconstruction.glb"))message="The requested glTF file was not produced. See the export log below.";
                else if(std::filesystem::is_regular_file(process.runDirectory/"mesh/fuse_post.ply"))message="Mesh exported. The viewport remains on the inspected PLY.";
                else message="Extraction finished without a post-processed mesh. See its log below.";
            }
        }
        bool draw(bool& visible,bool canExport,const std::string& sourceLabel={}) {
            if(!visible)return false;
            bool requested=false;
            ImGui::SetNextWindowSize(ImVec2(490,580),ImGuiCond_FirstUseEver);
            ImGui::Begin("Export mesh",&visible);
            ImGui::TextWrapped("Extract a TSDF mesh from the currently inspected point cloud.");
            if(!sourceLabel.empty())ImGui::TextDisabled("%s",sourceLabel.c_str());
            ImGui::Separator();
            ImGui::Checkbox("Textured glTF (.glb)",&texturedGltf);
            if(texturedGltf) {
                ImGui::InputInt("Texture size (pixels)",&textureSize);
                ImGui::InputInt("UV partitions (0 = auto)",&uvPartitions);
                ImGui::InputInt("UV threads (0 = auto)",&uvThreads);
                ImGui::Checkbox("Include lights",&includeLights);ImGui::SameLine();ImGui::Checkbox("Include cameras",&includeCameras);
                ImGui::DragFloat("Meters per scene unit",&metersPerUnit,.01f,.000001f,1000000.f,"%.6g");
                ImGui::TextWrapped("UV unwrapping and texture baking add substantial work. Leave this off for TSDF extraction only.");
            }
            if(ImGui::TreeNode("TSDF settings")) {
                ImGui::InputInt("Mesh resolution",&meshResolution);
                const char* depths[]={"Median depth","Mean depth"};ImGui::Combo("Depth source",&depthKind,depths,2);
                ImGui::DragFloat("Voxel size",&voxelSize,.001f,-1.f,10000.f,"%.6g");
                ImGui::DragFloat("SDF truncation",&sdfTrunc,.001f,-1.f,10000.f,"%.6g");
                ImGui::DragFloat("Depth cutoff",&depthTrunc,.01f,-1.f,1000000.f,"%.6g");
                ImGui::TextDisabled("Negative distances select automatic values.");
                ImGui::InputInt("Keep largest components",&components);
                ImGui::InputInt("Minimum component triangles",&minTriangles);
                ImGui::TreePop();
            }
            if(ImGui::TreeNode("Output location")) {ui::pathInput("Export root",outputRoot);ImGui::TreePop();}
            const bool disabled=!canExport||process.active();
            if(disabled)ImGui::BeginDisabled();
            if(ImGui::Button("Extract mesh",ImVec2(-1,32)))requested=true;
            if(disabled)ImGui::EndDisabled();
            if(process.active()) {
                const auto p=ImGui::GetCursorScreenPos();const float angle=static_cast<float>(ImGui::GetTime())*5;
                auto* draw=ImGui::GetWindowDrawList();draw->PathArcTo(ImVec2(p.x+8,p.y+8),5,angle,angle+4.8f,18);draw->PathStroke(ImGui::GetColorU32(ui::accent),2.f,ImDrawFlags_None);
                ImGui::Dummy(ImVec2(18,16));ImGui::SameLine();ImGui::TextUnformatted("Exporting mesh");
                ImGui::SameLine();if(ImGui::SmallButton("Cancel export")) {cancelled=true;process.shutdown();}
            }
            if(!message.empty())ImGui::TextWrapped("%s",message.c_str());
            if(!process.runDirectory.empty()) {
                ImGui::TextWrapped("%s",(process.runDirectory/"mesh").string().c_str());
                if(ImGui::SmallButton("Copy output folder"))ImGui::SetClipboardText((process.runDirectory/"mesh").string().c_str());
                if(!log.empty()) {ImGui::SameLine();if(ImGui::SmallButton("Copy export log"))ImGui::SetClipboardText(log.c_str());}
                for(const auto* file:{"fuse_post.ply","reconstruction.glb"})if(std::filesystem::is_regular_file(process.runDirectory/"mesh"/file)) {
                    ImGui::TextUnformatted(file);ImGui::SameLine();ImGui::PushID(file);
                    if(ImGui::SmallButton("Copy path"))ImGui::SetClipboardText((process.runDirectory/"mesh"/file).string().c_str());
                    ImGui::PopID();
                }
            }
            if(!log.empty()&&ImGui::CollapsingHeader("Export log",ImGuiTreeNodeFlags_DefaultOpen)) {
                const auto id=ImGui::GetID("##MeshExportLog");auto* input=ImGui::GetInputTextState(id);
                if(!(ImGui::GetActiveID()==id&&input&&input->HasSelection()))selectableLog=log;
                ImGui::InputTextMultiline("##MeshExportLog",selectableLog.data(),selectableLog.size()+1,
                    ImVec2(-1,std::max(80.f,ImGui::GetContentRegionAvail().y)),ImGuiInputTextFlags_ReadOnly);
            }
            ImGui::End();return requested&&!disabled;
        }
    };
}
