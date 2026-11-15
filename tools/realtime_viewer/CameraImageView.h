#pragma once
#include <imgui.h>
#include <algorithm>
#include <cmath>

namespace viewer {
    struct CameraImageView {
        float zoom=1;
        ImVec2 pan{};
        struct Placement {ImVec2 min,size;};
        void reset() {zoom=1;pan=ImVec2(0,0);}
        [[nodiscard]] Placement placement(ImVec2 origin,ImVec2 available,unsigned width,unsigned height) const {
            const float scale=std::min(available.x/std::max(width,1u),available.y/std::max(height,1u))*zoom;
            const ImVec2 size(std::max(1.f,width*scale),std::max(1.f,height*scale));
            return {ImVec2(origin.x+(available.x-size.x)/2+pan.x,origin.y+(available.y-size.y)/2+pan.y),size};
        }
        void zoomAt(ImVec2 mouse,ImVec2 origin,ImVec2 available,float wheel) {
            const float next=std::clamp(zoom*std::exp(wheel*.12f),.1f,20.f),factor=next/zoom;
            const ImVec2 offset(mouse.x-origin.x-available.x*.5f-pan.x,mouse.y-origin.y-available.y*.5f-pan.y);
            pan.x+=offset.x*(1-factor);pan.y+=offset.y*(1-factor);zoom=next;
        }
        void drag(ImVec2 delta) {pan.x+=delta.x;pan.y+=delta.y;}
    };
}
