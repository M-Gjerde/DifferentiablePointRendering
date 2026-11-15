#pragma once
#include <imgui.h>
#include <imgui_internal.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <tuple>

namespace viewer::ui {
    // Pale workspace tokens from the approved Figma / supplied SVG.
    inline constexpr ImVec4 canvas{.145f, .157f, .176f, 1};
    inline constexpr ImVec4 panel{.114f, .125f, .141f, 1};
    inline constexpr ImVec4 bar{.145f, .161f, .180f, 1};
    inline constexpr ImVec4 surface{.188f, .208f, .235f, 1};
    inline constexpr ImVec4 border{.235f, .259f, .294f, 1};
    inline constexpr ImVec4 text{.882f, .898f, .922f, 1};
    inline constexpr ImVec4 muted{.639f, .678f, .725f, 1};
    inline constexpr ImVec4 accent{.925f, .714f, .427f, 1};
    inline constexpr ImVec4 selection{.286f, .235f, .169f, 1};
    inline constexpr ImVec4 blue{.478f, .663f, .910f, 1};
    inline constexpr ImVec4 green{.569f, .737f, .553f, 1};
    enum class Icon { Select, Tune, View, Camera, Folder, Light, Cube, Material, Tools, Grid, Move, Rotate, Scale, Refresh, Play, Save, Maximize };
    enum class Inspector { Reconstruction, Selection, View, Camera, Files, Advanced };
    inline void applyStyle() {
        ImGui::StyleColorsDark();
        auto& s = ImGui::GetStyle();
        s.WindowPadding = ImVec2(12, 10); s.FramePadding = ImVec2(7, 4);
        s.ItemSpacing = ImVec2(7, 6); s.ItemInnerSpacing = ImVec2(5, 4);
        s.IndentSpacing = 12; s.ScrollbarSize = 10; s.GrabMinSize = 8;
        s.FrameRounding = 4; s.ChildRounding = 4; s.PopupRounding = 6;
        s.ScrollbarRounding = 6; s.GrabRounding = 3; s.TabRounding = 3;
        s.WindowBorderSize = 0; s.ChildBorderSize = 0; s.FrameBorderSize = 0;
        s.PopupBorderSize = 1; s.DockingSeparatorSize = 2;
        auto& c = s.Colors;
        c[ImGuiCol_WindowBg] = panel; c[ImGuiCol_ChildBg] = panel;
        c[ImGuiCol_PopupBg] = panel; c[ImGuiCol_Text] = text; c[ImGuiCol_TextDisabled] = muted;
        c[ImGuiCol_Border] = border; c[ImGuiCol_Separator] = border;
        c[ImGuiCol_FrameBg] = surface; c[ImGuiCol_FrameBgHovered] = ImVec4(.24f,.27f,.31f,1);
        c[ImGuiCol_FrameBgActive] = selection;
        c[ImGuiCol_Button] = surface; c[ImGuiCol_ButtonHovered] = ImVec4(.27f,.29f,.33f,1);
        c[ImGuiCol_ButtonActive] = selection;
        c[ImGuiCol_Header] = selection; c[ImGuiCol_HeaderHovered] = surface; c[ImGuiCol_HeaderActive] = selection;
        c[ImGuiCol_TitleBg] = bar; c[ImGuiCol_TitleBgActive] = bar; c[ImGuiCol_TitleBgCollapsed] = bar;
        c[ImGuiCol_MenuBarBg] = panel; c[ImGuiCol_Tab] = panel; c[ImGuiCol_TabHovered] = surface;
        c[ImGuiCol_TabSelected] = selection; c[ImGuiCol_TabSelectedOverline] = accent;
        c[ImGuiCol_TabDimmed] = panel; c[ImGuiCol_TabDimmedSelected] = surface;
        c[ImGuiCol_TabDimmedSelectedOverline] = border;
        c[ImGuiCol_CheckMark] = accent; c[ImGuiCol_SliderGrab] = accent; c[ImGuiCol_SliderGrabActive] = accent;
#if IMGUI_VERSION_NUM >= 19297
        c[ImGuiCol_CheckboxSelectedBg] = selection;
#endif
        c[ImGuiCol_TableHeaderBg] = bar;
        c[ImGuiCol_PlotLines] = green; c[ImGuiCol_PlotHistogram] = accent;
        c[ImGuiCol_SeparatorHovered] = accent; c[ImGuiCol_SeparatorActive] = accent;
        c[ImGuiCol_ResizeGrip] = panel; c[ImGuiCol_ResizeGripHovered] = border; c[ImGuiCol_ResizeGripActive] = accent;
        c[ImGuiCol_DockingPreview] = ImVec4(accent.x,accent.y,accent.z,.35f);
        c[ImGuiCol_TextSelectedBg] = selection; c[ImGuiCol_NavCursor] = accent;
    }
    inline void loadFont() {
        // Prefer the mockup family when installed; use a readable system family otherwise.
        for (const auto& path : {"/usr/share/fonts/truetype/ibm-plex/IBMPlexSans-Regular.ttf",
                                 "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
                                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"}) {
            if (std::filesystem::is_regular_file(path)) {
                ImGui::GetIO().Fonts->AddFontFromFileTTF(path, 13.f); return;
            }
        }
        ImGui::GetIO().Fonts->AddFontDefault();
    }
    template<class Draw> bool property(const char* label,Draw draw) {
        bool changed=false;
        ImGui::PushID(label);
        if(ImGui::BeginTable("##property",2,ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Label",ImGuiTableColumnFlags_WidthStretch,.30f);
            ImGui::TableSetupColumn("Value",ImGuiTableColumnFlags_WidthStretch,.70f);
            ImGui::TableNextColumn(); ImGui::AlignTextToFramePadding(); ImGui::TextDisabled("%s",label);
            ImGui::TableNextColumn(); ImGui::SetNextItemWidth(-1); changed=draw();
            ImGui::EndTable();
        }
        ImGui::PopID(); return changed;
    }
    inline bool float3(const char* label,float* value,float speed,float min,float max,const char* format) {
        return property(label,[&]{return ImGui::DragFloat3("##value",value,speed,min,max,format);});
    }
    inline bool scalar(const char* label,float* value,float speed,float min,float max,const char* format) {
        return property(label,[&]{return ImGui::DragFloat("##value",value,speed,min,max,format);});
    }
    inline bool color3(const char* label,float* value) {
        return property(label,[&]{return ImGui::ColorEdit3("##value",value,ImGuiColorEditFlags_NoInputs|ImGuiColorEditFlags_Float);});
    }
    template<std::size_t N> bool pathInput(const char* label,std::array<char,N>& buffer) {
        ImGui::TextDisabled("%s",label); ImGui::SetNextItemWidth(-1);
        ImGui::PushID(label); const bool changed=ImGui::InputText("##path",buffer.data(),buffer.size()); ImGui::PopID(); return changed;
    }
    inline void icon(ImDrawList* d, Icon kind, ImVec2 p, ImU32 color, float size = 16) {
        auto pt = [&](float x, float y) { return ImVec2(p.x+x*size, p.y+y*size); };
        auto line = [&](float x,float y,float a,float b) { d->AddLine(pt(x,y),pt(a,b),color,1.2f); };
        auto box = [&](float x,float y,float a,float b) { d->AddRect(pt(x,y),pt(a,b),color,1,0,1.2f); };
        switch (kind) {
            case Icon::Select: line(.25,.12,.25,.85); line(.25,.12,.8,.6); line(.25,.85,.45,.62); line(.45,.62,.8,.6); break;
            case Icon::Tune: for (int i=0;i<3;++i) { float y=.22f+i*.28f, x=i==1?.65f:.35f; line(.12,y,.88,y); box(x-.07f,y-.09f,x+.07f,y+.09f); } break;
            case Icon::View: for (float y : {.25f,.5f,.75f}) { line(.12,y,.5,y+.16); line(.5,y+.16,.88,y); if(y==.25f) {line(.12,y,.5,y-.16);line(.5,y-.16,.88,y);} } break;
            case Icon::Camera: box(.12,.28,.68,.78); line(.68,.42,.9,.25);line(.9,.25,.9,.8);line(.9,.8,.68,.64);break;
            case Icon::Folder: line(.1,.25,.4,.25); line(.4,.25,.5,.4); line(.5,.4,.9,.4);line(.9,.4,.9,.8);line(.9,.8,.1,.8);line(.1,.8,.1,.25);break;
            case Icon::Light: d->AddCircle(pt(.5,.5),size*.18f,color,12,1.2f); for(int i=0;i<8;++i){float a=i*3.141593f/4;line(.5f+.29f*std::cos(a),.5f+.29f*std::sin(a),.5f+.43f*std::cos(a),.5f+.43f*std::sin(a));} break;
            case Icon::Cube: line(.5,.08,.9,.3);line(.9,.3,.9,.72);line(.9,.72,.5,.94);line(.5,.94,.1,.72);line(.1,.72,.1,.3);line(.1,.3,.5,.08);line(.1,.3,.5,.52);line(.5,.52,.9,.3);line(.5,.52,.5,.94);break;
            case Icon::Material: d->AddCircle(pt(.5,.5),size*.36f,color,18,1.2f);line(.25,.75,.75,.25);break;
            case Icon::Grid: box(.12,.12,.88,.88);for(float v:{.37f,.63f}){line(v,.12,v,.88);line(.12,v,.88,v);}break;
            case Icon::Move: line(.5,.08,.5,.92);line(.08,.5,.92,.5);line(.5,.08,.35,.25);line(.5,.08,.65,.25);line(.92,.5,.75,.35);line(.92,.5,.75,.65);break;
            case Icon::Rotate: case Icon::Refresh: d->PathArcTo(pt(.5,.5),size*.33f,-1.5f,3.9f,18);d->PathStroke(color,0,1.2f);line(.5,.16,.75,.1);line(.5,.16,.64,.36);break;
            case Icon::Scale: box(.13,.57,.42,.86);line(.38,.63,.87,.14);line(.87,.14,.55,.14);line(.87,.14,.87,.46);break;
            case Icon::Play: line(.28,.15,.85,.5);line(.85,.5,.28,.85);line(.28,.85,.28,.15);break;
            case Icon::Save: box(.15,.12,.85,.88);box(.32,.12,.67,.42);box(.32,.62,.68,.88);break;
            case Icon::Maximize: line(.12,.4,.12,.12);line(.12,.12,.4,.12);line(.6,.12,.88,.12);line(.88,.12,.88,.4);line(.12,.6,.12,.88);line(.12,.88,.4,.88);line(.6,.88,.88,.88);line(.88,.88,.88,.6);break;
            case Icon::Tools: d->AddCircle(pt(.5,.5),size*.18f,color,10,1.2f);for(int i=0;i<6;++i){float a=i*3.141593f/3;line(.5f+.25f*std::cos(a),.5f+.25f*std::sin(a),.5f+.43f*std::cos(a),.5f+.43f*std::sin(a));}break;
        }
    }
    inline bool iconButton(const char* label, Icon kind, bool selected = false, ImVec2 size = ImVec2(30,30)) {
        ImGui::PushID(label);
        ImGui::PushStyleColor(ImGuiCol_Button, selected ? selection : panel);
        bool clicked = ImGui::Button("##icon", size);
        const auto min = ImGui::GetItemRectMin();
        icon(ImGui::GetWindowDrawList(),kind,ImVec2(min.x+(size.x-16)/2,min.y+(size.y-16)/2),ImGui::GetColorU32(selected?accent:muted));
        if(selected) ImGui::GetWindowDrawList()->AddRectFilled(min,ImVec2(min.x+2,min.y+size.y),ImGui::GetColorU32(accent),1);
        if(ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort|ImGuiHoveredFlags_AllowWhenDisabled)||ImGui::IsItemFocused()) ImGui::SetTooltip("%s",label);
        ImGui::PopStyleColor(); ImGui::PopID(); return clicked;
    }
    inline void inspectorRail(Inspector& selected, bool advanced) {
        for (const auto& [page, symbol, label] : std::array<std::tuple<Inspector,Icon,const char*>,6>{{
            {Inspector::Selection,Icon::Select,"Selection properties"},
            {Inspector::Reconstruction,Icon::Tune,"Reconstruction setup & hyperparameters"},
            {Inspector::View,Icon::View,"Viewport appearance & diagnostics"},
            {Inspector::Camera,Icon::Camera,"Viewport & calibrated cameras"},
            {Inspector::Files,Icon::Folder,"Scene files & saved snapshots"},
            {Inspector::Advanced,Icon::Tools,"Advanced renderer diagnostics"}}}) {
            if(page==Inspector::Advanced&&!advanced)continue;
            if(iconButton(label,symbol,selected==page)) selected=page;
        }
    }
    inline void defaultDockLayout(ImGuiID id, ImGuiViewport* viewport) {
        ImGui::DockBuilderRemoveNode(id); ImGui::DockBuilderAddNode(id,ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodePos(id,viewport->WorkPos); ImGui::DockBuilderSetNodeSize(id,viewport->WorkSize);
        ImGuiID right=0, main=0, tree=0, inspector=0, monitor=0;
        const float rightFraction=std::clamp(344.f/viewport->WorkSize.x,.20f,.30f);
        ImGui::DockBuilderSplitNode(id,ImGuiDir_Right,rightFraction,&right,&main);
        ImGui::DockBuilderSplitNode(right,ImGuiDir_Up,.30f,&tree,&inspector);
        ImGui::DockBuilderSplitNode(main,ImGuiDir_Down,.16f,&monitor,&main);
        ImGui::DockBuilderDockWindow("Scene Collection",tree); ImGui::DockBuilderDockWindow("Properties",inspector);
        ImGui::DockBuilderDockWindow("Run monitor",monitor); ImGui::DockBuilderDockWindow("Viewport",main);
        ImGui::DockBuilderGetNode(main)->LocalFlags |= ImGuiDockNodeFlags_HiddenTabBar;
        ImGui::DockBuilderFinish(id);
    }
}
