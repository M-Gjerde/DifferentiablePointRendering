#pragma once
#include "ReconstructionLibrary.h"
#include "TrainingSession.h"
#include <future>
#include <set>
#include <array>
#include <cctype>
#include <cfloat>

namespace viewer {
    class ReconstructionBrowser {
        bool requested_=false,visible_=false,runsChanged_=false,revealSelection_=false;
        std::filesystem::path root_,folder_,scanRoot_,selection_;
        std::array<char,2048> folderText_{};
        ImGuiTextFilter filter_;
        std::string datasetFilter_;
        std::vector<ReconstructionEntry> runs_;
        std::vector<ReconstructionFolder> folders_;
        std::future<std::vector<ReconstructionEntry>> scan_;
        std::string error_;
        static std::string displayName(std::string label) {
            if(label.ends_with("_pbdr"))label.resize(label.size()-5);
            std::replace(label.begin(),label.end(),'_',' ');
            if(!label.empty())label[0]=static_cast<char>(std::toupper(static_cast<unsigned char>(label[0])));
            return label;
        }
        static void drawCreated(const FolderTimestamp& created) {
            ImGui::TextUnformatted(folderTimestampLabel(created).c_str());
            if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",created.creation?
                "Folder creation time (local time).":"* Modification time: this filesystem does not provide folder creation time.");
        }
        bool matches(const ReconstructionEntry& run) const {
            const auto name=displayName(run.label);
            const auto search=name+" "+run.label+" "+run.directory.lexically_relative(root_).string();
            return (datasetFilter_.empty()||name==datasetFilter_)&&filter_.PassFilter(search.c_str());
        }
        void sortRuns(const ImGuiTableSortSpecs* specs) {
            if(!specs||(!specs->SpecsDirty&&!runsChanged_))return;
            std::sort(runs_.begin(),runs_.end(),[&](const auto& a,const auto& b) {
                for(int i=0;i<specs->SpecsCount;++i) {
                    const auto& spec=specs->Specs[i];int order=0;
                    switch(spec.ColumnUserID) {
                        case 0:order=displayName(a.label).compare(displayName(b.label));break;
                        case 1:order=a.created.value<b.created.value?-1:a.created.value>b.created.value?1:0;break;
                        case 2:order=a.directory.string().compare(b.directory.string());break;
                        case 3:order=int(a.hasLoss)-int(b.hasLoss);break;
                        case 4:order=int(a.hasGeometry)-int(b.hasGeometry);break;
                    }
                    if(order)return spec.SortDirection==ImGuiSortDirection_Descending?order>0:order<0;
                }
                return a.directory.string()<b.directory.string();
            });
            runsChanged_=false;
        }
        void refresh() {
            if(scan_.valid())return;
            const auto root=root_;
            scanRoot_=root;
            scan_=std::async(std::launch::async,[root] {return scanReconstructions(root);});
        }
        void navigate(const std::filesystem::path& path) {
            std::error_code error;
            if(!std::filesystem::is_directory(path,error)) {error_="Folder does not exist or is inaccessible";return;}
            folder_=std::filesystem::absolute(path).lexically_normal();
            std::snprintf(folderText_.data(),folderText_.size(),"%s",folder_.string().c_str());
            folders_=reconstructionFolders(folder_);error_.clear();
        }
    public:
        void open(const std::filesystem::path& root,const std::filesystem::path& selected={}) {
            if(root_!=root) {root_=root;runs_.clear();}
            selection_=selected;revealSelection_=true;
            auto folder=root;
            std::error_code error;
            while(!folder.empty()&&!std::filesystem::is_directory(folder,error)&&folder!=folder.parent_path())folder=folder.parent_path();
            navigate(folder);
            requested_=true;visible_=true;filter_.Clear();datasetFilter_.clear();refresh();
        }
        [[nodiscard]] const std::filesystem::path& selection() const {return selection_;}
        [[nodiscard]] bool scanning() const {return scan_.valid();}
        [[nodiscard]] const std::filesystem::path& currentFolder() const {return folder_;}
        std::optional<std::filesystem::path> draw(bool enabled) {
            if(scan_.valid()&&scan_.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
                try {
                    auto runs=scan_.get();
                    if(scanRoot_==root_) {runs_=std::move(runs);runsChanged_=true;revealSelection_=true;}else refresh();
                }catch(const std::exception& error) {error_=error.what();}
            }
            if(requested_) {ImGui::OpenPopup("Open reconstruction");requested_=false;}
            std::optional<std::filesystem::path> selected;
            ImGui::SetNextWindowSize(ImVec2(940,600),ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSizeConstraints(ImVec2(720,440),ImVec2(FLT_MAX,FLT_MAX));
            if(ImGui::BeginPopupModal("Open reconstruction",&visible_)) {
                ImGui::TextUnformatted("Open a saved point cloud, objective history and recorded geometry metrics.");
                if(!enabled)ImGui::TextWrapped("Stop the current reconstruction before opening a saved run.");
                if(ImGui::BeginTabBar("Sources")) {
                    if(ImGui::BeginTabItem("App outputs")) {
                        ImGui::TextWrapped("%s",root_.string().c_str());
                        const bool scanning=scan_.valid();
                        if(scanning)ImGui::BeginDisabled();
                        if(ImGui::SmallButton("Refresh outputs"))refresh();
                        if(scanning)ImGui::EndDisabled();
                        ImGui::SameLine();ImGui::SetNextItemWidth(std::max(160.f,ImGui::GetContentRegionAvail().x-205));
                        if(ImGui::InputTextWithHint("##Filter","Filter by dataset or folder",filter_.InputBuf,sizeof(filter_.InputBuf)))filter_.Build();
                        ImGui::SameLine();ImGui::SetNextItemWidth(-1);
                        if(ImGui::BeginCombo("##DatasetFilter",datasetFilter_.empty()?"All datasets":datasetFilter_.c_str())) {
                            if(ImGui::Selectable("All datasets",datasetFilter_.empty()))datasetFilter_.clear();
                            std::set<std::string> datasets;
                            for(const auto& run:runs_)datasets.insert(displayName(run.label));
                            for(const auto& dataset:datasets)
                                if(ImGui::Selectable(dataset.c_str(),dataset==datasetFilter_))datasetFilter_=dataset;
                            ImGui::EndCombo();
                        }
                        if(scan_.valid())ImGui::TextDisabled("Scanning saved reconstructions...");
                        else ImGui::TextDisabled("%zu saved runs · click headers to sort · Up/Down to select · Enter or double-click to open",runs_.size());
                        if(ImGui::BeginTable("SavedRuns",5,ImGuiTableFlags_Sortable|ImGuiTableFlags_Resizable|ImGuiTableFlags_RowBg|
                            ImGuiTableFlags_BordersInnerH|ImGuiTableFlags_ScrollY,ImVec2(0,-98))) {
                            ImGui::TableSetupColumn("Dataset",ImGuiTableColumnFlags_WidthStretch,.30f,0);
                            ImGui::TableSetupColumn("Created",ImGuiTableColumnFlags_WidthFixed|ImGuiTableColumnFlags_DefaultSort|
                                ImGuiTableColumnFlags_PreferSortDescending,155,1);
                            ImGui::TableSetupColumn("Run folder",ImGuiTableColumnFlags_WidthStretch,.70f,2);
                            ImGui::TableSetupColumn("Loss",ImGuiTableColumnFlags_WidthFixed,45,3);
                            ImGui::TableSetupColumn("GT",ImGuiTableColumnFlags_WidthFixed,40,4);
                            ImGui::TableSetupScrollFreeze(0,1);ImGui::TableHeadersRow();
                            auto* specs=ImGui::TableGetSortSpecs();sortRuns(specs);if(specs)specs->SpecsDirty=false;
                            std::vector<std::size_t> filtered;
                            for(std::size_t i=0;i<runs_.size();++i)if(matches(runs_[i]))filtered.push_back(i);
                            auto current=std::find_if(filtered.begin(),filtered.end(),[&](auto i) {return runs_[i].directory==selection_;});
                            if(current==filtered.end()&&!filtered.empty()) {selection_=runs_[filtered.front()].directory;current=filtered.begin();revealSelection_=true;}
                            if(filtered.empty()&&!scan_.valid())selection_.clear();
                            const bool keyboard=enabled&&!ImGui::GetIO().WantTextInput&&ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);
                            if(keyboard&&!filtered.empty()) {
                                const int step=int(ImGui::IsKeyPressed(ImGuiKey_DownArrow))-int(ImGui::IsKeyPressed(ImGuiKey_UpArrow));
                                if(step) {
                                    const auto index=std::clamp(static_cast<int>(current-filtered.begin())+step,0,static_cast<int>(filtered.size()-1));
                                    selection_=runs_[filtered[index]].directory;revealSelection_=true;
                                }
                                if(ImGui::IsKeyPressed(ImGuiKey_Enter,false)||ImGui::IsKeyPressed(ImGuiKey_KeypadEnter,false))selected=selection_;
                            }
                            for(const auto index:filtered) {
                                const auto& run=runs_[index];const bool active=run.directory==selection_;
                                ImGui::TableNextRow();ImGui::TableSetColumnIndex(0);ImGui::PushID(run.directory.string().c_str());
                                if(ImGui::Selectable(displayName(run.label).c_str(),active,
                                    ImGuiSelectableFlags_SpanAllColumns|ImGuiSelectableFlags_AllowDoubleClick|
                                    ImGuiSelectableFlags_NoAutoClosePopups|(!enabled?ImGuiSelectableFlags_Disabled:0))) {
                                    selection_=run.directory;
                                    if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))selected=selection_;
                                }
                                if(ImGui::IsItemHovered())ImGui::SetTooltip("%s\nDouble-click or press Enter to open",run.directory.string().c_str());
                                if(active&&revealSelection_)ImGui::SetScrollHereY();
                                ImGui::TableSetColumnIndex(1);drawCreated(run.created);
                                ImGui::TableSetColumnIndex(2);ImGui::TextUnformatted(run.directory.lexically_relative(root_).string().c_str());
                                ImGui::TableSetColumnIndex(3);ImGui::TextDisabled("%s",run.hasLoss?"Yes":"—");
                                ImGui::TableSetColumnIndex(4);ImGui::TextDisabled("%s",run.hasGeometry?"Yes":"—");
                                ImGui::PopID();
                            }
                            if(filtered.empty()) {ImGui::TableNextRow();ImGui::TableSetColumnIndex(0);ImGui::TextDisabled(scan_.valid()?"Scanning...":"No matching runs");}
                            revealSelection_=false;ImGui::EndTable();
                        }
                        ImGui::TextDisabled("Selected: %s",selection_.empty()?"None":selection_.lexically_relative(root_).string().c_str());
                        if(!enabled||selection_.empty())ImGui::BeginDisabled();
                        if(ImGui::Button("Open selected reconstruction",ImVec2(-1,30)))selected=selection_;
                        if(!enabled||selection_.empty())ImGui::EndDisabled();
                        ImGui::EndTabItem();
                    }
                    if(ImGui::BeginTabItem("Choose folder")) {
                        ImGui::SetNextItemWidth(-52);
                        const bool enter=ImGui::InputText("##Folder",folderText_.data(),folderText_.size(),ImGuiInputTextFlags_EnterReturnsTrue);
                        ImGui::SameLine();const bool go=ImGui::Button("Go");
                        if(enter||go) {
                            try {navigate(expandedPath(folderText_.data(),folder_.empty()?std::filesystem::current_path():folder_));}
                            catch(const std::exception& error) {error_=error.what();}
                        }
                        if(ImGui::Button("Up"))navigate(folder_.parent_path());
                        ImGui::SameLine();ImGui::TextUnformatted("Select the folder containing run_config.json and the saved outputs.");
                        const bool valid=enabled&&isReconstructionDirectory(folder_);
                        if(!valid)ImGui::BeginDisabled();
                        if(ImGui::Button("Open this reconstruction",ImVec2(-1,30)))selected=folder_;
                        if(!valid)ImGui::EndDisabled();
                        ImGui::TextDisabled("Folders · newest creation time first");
                        if(ImGui::BeginTable("RunFolders",2,ImGuiTableFlags_RowBg|ImGuiTableFlags_BordersInnerH|ImGuiTableFlags_ScrollY,ImVec2(0,-42))) {
                            ImGui::TableSetupColumn("Folder",ImGuiTableColumnFlags_WidthStretch);
                            ImGui::TableSetupColumn("Created",ImGuiTableColumnFlags_WidthFixed,175);
                            ImGui::TableSetupScrollFreeze(0,1);ImGui::TableHeadersRow();
                            std::optional<std::filesystem::path> next;
                            for(const auto& entry:folders_) {
                                ImGui::TableNextRow();ImGui::TableSetColumnIndex(0);
                                if(ImGui::Selectable(entry.path.filename().string().c_str(),false,ImGuiSelectableFlags_NoAutoClosePopups))next=entry.path;
                                ImGui::TableSetColumnIndex(1);drawCreated(entry.created);
                            }
                            ImGui::EndTable();if(next)navigate(*next);
                        }
                        ImGui::EndTabItem();
                    }
                    ImGui::EndTabBar();
                }
                if(!error_.empty())ImGui::TextWrapped("%s",error_.c_str());
                if(ImGui::Button("Cancel")||ImGui::IsKeyPressed(ImGuiKey_Escape,false)||selected) {visible_=false;ImGui::CloseCurrentPopup();}
                ImGui::EndPopup();
            }
            return selected;
        }
    };
}
