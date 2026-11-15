#pragma once
#include <imgui.h>
#include <algorithm>
#include <vector>

namespace viewer {
    // Retain the last displayed transform handles while their scene is replaced.
    // Replaying geometry accepts no input and preserves the original axis colors.
    class ViewportOverlayCache {
        struct Part { ImTextureRef texture; ImVec4 clip;std::vector<ImDrawVert> vertices; };
        std::vector<Part> parts_;
        ImVec2 min_{},max_{};
        bool bounded_=false;
    public:
        void capture(const ImDrawList& draw,int firstIndex,ImVec2 origin,ImVec2 size) {
            parts_.clear();
            bounded_=false;
            if(size.x<=0||size.y<=0)return;
            for(const auto& command:draw.CmdBuffer) {
                if(command.UserCallback)continue;
                const auto begin=std::max<unsigned>(firstIndex,command.IdxOffset);
                const auto end=command.IdxOffset+command.ElemCount;
                if(begin>=end)continue;
                Part part{command.TexRef,ImVec4((command.ClipRect.x-origin.x)/size.x,(command.ClipRect.y-origin.y)/size.y,
                    (command.ClipRect.z-origin.x)/size.x,(command.ClipRect.w-origin.y)/size.y),{}};part.vertices.reserve(end-begin);
                for(unsigned i=begin;i<end;++i) {
                    auto vertex=draw.VtxBuffer[command.VtxOffset+draw.IdxBuffer[i]];
                    vertex.pos=ImVec2((vertex.pos.x-origin.x)/size.x,(vertex.pos.y-origin.y)/size.y);
                    if(!bounded_) {min_=max_=vertex.pos;bounded_=true;}
                    else {min_.x=std::min(min_.x,vertex.pos.x);min_.y=std::min(min_.y,vertex.pos.y);max_.x=std::max(max_.x,vertex.pos.x);max_.y=std::max(max_.y,vertex.pos.y);}
                    part.vertices.push_back(vertex);
                }
                parts_.push_back(std::move(part));
            }
        }
        void draw(ImDrawList& draw,ImVec2 origin,ImVec2 size) const {
            for(const auto& part:parts_) {
                draw.PushTexture(part.texture);
                draw.PushClipRect(ImVec2(origin.x+part.clip.x*size.x,origin.y+part.clip.y*size.y),ImVec2(origin.x+part.clip.z*size.x,origin.y+part.clip.w*size.y),true);
                for(std::size_t first=0;first<part.vertices.size();) {
                    const int count=static_cast<int>(std::min<std::size_t>(60000,part.vertices.size()-first));
                    draw.PrimReserve(count,count);
                    const auto base=draw._VtxCurrentIdx;
                    for(int i=0;i<count;++i) {
                        const auto& vertex=part.vertices[first+i];
                        draw.PrimWriteIdx(static_cast<ImDrawIdx>(base+i));
                        draw.PrimWriteVtx(ImVec2(origin.x+vertex.pos.x*size.x,origin.y+vertex.pos.y*size.y),vertex.uv,vertex.col);
                    }
                    first+=count;
                }
                draw.PopClipRect();draw.PopTexture();
            }
        }
        [[nodiscard]] bool empty() const {return parts_.empty();}
        [[nodiscard]] bool contains(ImVec2 point,ImVec2 origin,ImVec2 size) const {
            return bounded_&&point.x>=origin.x+min_.x*size.x-4&&point.x<=origin.x+max_.x*size.x+4&&
                point.y>=origin.y+min_.y*size.y-4&&point.y<=origin.y+max_.y*size.y+4;
        }
    };
}
