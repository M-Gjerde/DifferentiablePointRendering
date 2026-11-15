#pragma once
#include <imgui.h>
#include <optional>
#include <string>

namespace viewer {
    // A click on a cached inspector is delivered to the real widgets after the
    // worker returns. Its press/release get two UI frames before another job.
    class InspectorInputHandoff {
        struct Input {int page;std::string key;ImVec2 mouse;float wheel;bool click;int phase=0;};
        std::optional<Input> input_;
        bool syntheticPress_=false;
        void releaseSyntheticPress(bool physicalMouseDown) {
            if(syntheticPress_&&!physicalMouseDown)ImGui::GetIO().AddMouseButtonEvent(0,false);
            // A genuine held button resumes ownership until its OS release.
            syntheticPress_=false;
        }
    public:
        void queue(int page,const std::string& key,ImVec2 mouse,float wheel,bool click) {
            if(input_&&input_->phase==0&&input_->page==page&&input_->key==key) {
                // Keep one coherent gesture: scrolling changes widget positions
                // and must not relocate an earlier queued click onto a new row.
                if(click) {input_->mouse=mouse;input_->wheel=0;input_->click=true;}
                else if(wheel!=0) {input_->mouse=mouse;input_->wheel+=wheel;input_->click=false;}
            } else input_=Input{page,key,mouse,click?0:wheel,click};
        }
        [[nodiscard]] bool pending() const {return input_.has_value();}
        void deliver(int page,const std::string& key,bool physicalMouseDown) {
            if(!input_) {releaseSyntheticPress(physicalMouseDown);return;}
            if(input_->page!=page||input_->key!=key) {
                releaseSyntheticPress(physicalMouseDown);input_.reset();return;
            }
            auto& io=ImGui::GetIO();io.ConfigInputTrickleEventQueue=false;
            if(input_->phase==0) {
                io.AddMousePosEvent(input_->mouse.x,input_->mouse.y);
                if(input_->wheel)io.AddMouseWheelEvent(0,input_->wheel);
                if(input_->click) {
                    // A physical press may have happened several cached frames
                    // ago. The real widget still needs a fresh press edge.
                    io.MouseDown[0]=false;io.MouseDownDuration[0]=-1;
                    io.AddMouseButtonEvent(0,true);
                    syntheticPress_=true;
                }
                input_->phase=1;
            } else if(input_->phase==1) {
                if(input_->click&&!physicalMouseDown) {
                    io.AddMousePosEvent(input_->mouse.x,input_->mouse.y);releaseSyntheticPress(false);
                }
                input_->phase=2;
            } else {releaseSyntheticPress(physicalMouseDown);input_.reset();}
        }
    };
}
