#pragma once
#include "WorkspaceUi.h"
#include <string>
#include <vector>
#include <optional>
#include <cctype>

namespace viewer {
    enum class SceneItemKind { Geometry, Mesh, Material, Emitter, SurfelEmitter, Camera };
    struct SceneTreeItem {
        SceneItemKind kind;
        int index=0;
        std::string name, detail;
        ui::Icon icon=ui::Icon::Cube;
        bool selected=false;
    };
    struct SceneTreeGroup { std::string name; std::vector<SceneTreeItem> items; std::size_t total=0; };
    struct SceneTree {
        std::array<char,160> search{};
        static std::string lower(std::string s) {
            for(char& c:s)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c))); return s;
        }
        std::optional<SceneTreeItem> draw(const std::string& sceneName,const std::vector<SceneTreeGroup>& groups,bool& visible) {
            if(!visible)return {};
            std::optional<SceneTreeItem> picked;
            ImGui::Begin(ui::sceneTreeTitle,&visible);
            ImGui::SetNextItemWidth(-1);
            ImGui::InputTextWithHint("##sceneSearch","Search scene...",search.data(),search.size());
            const auto query=lower(search.data());
            ImGui::SetNextItemOpen(true,ImGuiCond_Once);
            if(ImGui::TreeNodeEx(sceneName.c_str(),ImGuiTreeNodeFlags_SpanAvailWidth)) {
                for(const auto& group:groups) {
                    bool any=false;
                    for(const auto& item:group.items)if(query.empty()||lower(item.name).find(query)!=std::string::npos)any=true;
                    if(!any)continue;
                    ImGui::PushID(group.name.c_str());
                    ImGui::SetNextItemOpen(true,query.empty()?ImGuiCond_Once:ImGuiCond_Always);
                    if(ImGui::TreeNodeEx("##group",ImGuiTreeNodeFlags_SpanAvailWidth,"%s  ·  %zu",group.name.c_str(),group.total?group.total:group.items.size())) {
                        for(const auto& item:group.items) {
                            if(!query.empty()&&lower(item.name).find(query)==std::string::npos)continue;
                            ImGui::PushID(item.index);
                            const auto start=ImGui::GetCursorScreenPos();
                            if(ImGui::Selectable("##row",item.selected,ImGuiSelectableFlags_None,ImVec2(0,21)))picked=item;
                            auto* draw=ImGui::GetWindowDrawList();
                            ui::icon(draw,item.icon,ImVec2(start.x+2,start.y+2),ImGui::GetColorU32(item.selected?ui::accent:ui::muted));
                            draw->AddText(ImVec2(start.x+24,start.y+2),ImGui::GetColorU32(item.selected?ui::accent:ui::text),item.name.c_str());
                            if(ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("%s%s%s",item.name.c_str(),item.detail.empty()?"":"\n",item.detail.c_str());
                            ImGui::PopID();
                        }
                        ImGui::TreePop();
                    }
                    ImGui::PopID();
                }
                ImGui::TreePop();
            }
            ImGui::End(); return picked;
        }
    };
}
