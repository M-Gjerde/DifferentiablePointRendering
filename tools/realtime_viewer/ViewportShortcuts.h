#pragma once
#include <imgui.h>
#include <imgui_internal.h>
#include <algorithm>
#include <cstddef>
#include <optional>

namespace viewer {
    // Consume actual press events in order, on both idle and rendering frames.
    // Looking at held keys or scanning 1..0 by priority loses fast sequences
    // such as 1, 2, 1. Relative shortcuts start from the latest queued choice.
    class ViewportModeShortcuts {
        ImGuiKeyChord modifiers_=ImGuiMod_None;
    public:
        std::optional<std::size_t> readFrame(std::size_t current,std::size_t count) {
            const auto& io=ImGui::GetIO();
            const bool typing=io.WantTextInput||(io.ConfigFlags&ImGuiConfigFlags_NoKeyboard)||ImGui::GetTopMostPopupModal();
            std::optional<std::size_t> requested;
            for(const auto& event:ImGui::GetCurrentContext()->InputEventsTrail) {
                if(event.Type==ImGuiInputEventType_Focus&&!event.AppFocused.Focused) {
                    modifiers_=ImGuiMod_None;
                    continue;
                }
                if(event.Type!=ImGuiInputEventType_Key)continue;
                const auto key=event.Key.Key;
                if(key==ImGuiMod_Ctrl||key==ImGuiMod_Alt||key==ImGuiMod_Super||key==ImGuiMod_Shift) {
                    if(event.Key.Down)modifiers_|=key;else modifiers_&=~key;
                    continue;
                }
                if(!event.Key.Down||typing||count==0||
                   (modifiers_&(ImGuiMod_Ctrl|ImGuiMod_Alt|ImGuiMod_Super)))continue;
                if(key>=ImGuiKey_0&&key<=ImGuiKey_9) {
                    const auto index=key==ImGuiKey_0?9u:static_cast<std::size_t>(key-ImGuiKey_1);
                    if(index<count)requested=current=index;
                } else if(key==ImGuiKey_KeypadAdd||
                          (key==ImGuiKey_Equal&&(modifiers_&ImGuiMod_Shift))) {
                    requested=current=(std::min(current,count-1)+1)%count;
                } else if(key==ImGuiKey_KeypadSubtract||key==ImGuiKey_Minus) {
                    requested=current=(std::min(current,count-1)+count-1)%count;
                }
            }
            // ImGui can clear held modifiers on focus loss without key events.
            modifiers_=io.KeyMods;
            return requested;
        }
    };
}
