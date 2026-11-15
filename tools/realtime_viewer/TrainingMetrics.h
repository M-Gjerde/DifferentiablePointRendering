#pragma once
#include "WorkspaceUi.h"
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <optional>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace viewer {
    class TrainingMetrics {
        std::filesystem::path path_;
        std::uintmax_t offset_=0;
        std::filesystem::file_time_type stamp_{};
        std::string partial_;
        std::vector<std::string> header_;
        static std::vector<std::string> fields(const std::string& line) {
            std::vector<std::string> result;std::string value;bool quoted=false;
            for(std::size_t i=0;i<line.size();++i) {
                const char c=line[i];
                if(c=='"') {if(quoted&&i+1<line.size()&&line[i+1]=='"') {value+='"';++i;}else quoted=!quoted;}
                else if(c==','&&!quoted) {result.push_back(value);value.clear();}
                else if(c!='\r')value+=c;
            }
            result.push_back(value);return result;
        }
        static float number(const std::string& value) {
            char* end=nullptr;const float result=std::strtof(value.c_str(),&end);
            return end==value.c_str()||*end!='\0'?std::numeric_limits<float>::quiet_NaN():result;
        }
    public:
        std::vector<float> iterations;
        std::map<std::string,std::vector<float>> columns;
        std::map<std::string,std::string> latestText;
        void reset() {offset_=0;partial_.clear();header_.clear();iterations.clear();columns.clear();latestText.clear();}
        bool update(const std::filesystem::path& path,bool rewritten=false) {
            std::error_code error;const auto size=std::filesystem::file_size(path,error);if(error)return false;
            const auto stamp=std::filesystem::last_write_time(path,error);
            if(path!=path_||size<offset_||(rewritten&&stamp!=stamp_)) {reset();path_=path;}
            stamp_=stamp;if(size==offset_)return false;
            std::ifstream stream(path,std::ios::binary);if(!stream)return false;
            stream.seekg(static_cast<std::streamoff>(offset_));
            std::string chunk(static_cast<std::size_t>(std::min<std::uintmax_t>(size-offset_,512*1024)),'\0');
            stream.read(chunk.data(),chunk.size());chunk.resize(static_cast<std::size_t>(stream.gcount()));offset_+=chunk.size();partial_+=chunk;
            std::size_t begin=0,end=0;
            while((end=partial_.find('\n',begin))!=std::string::npos) {
                const auto row=fields(partial_.substr(begin,end-begin));begin=end+1;
                if(header_.empty()) {
                    header_=row;
                    // Keep the first occurrence of a named column. Duplicate
                    // headers must not append two values per iteration.
                    std::set<std::string> names;
                    for(auto& name:header_)if(name.empty()||!names.insert(name).second)name.clear();
                    continue;
                }
                if(row.size()!=header_.size())continue;
                auto it=std::find(header_.begin(),header_.end(),"iteration");if(it==header_.end())continue;
                const float step=number(row[it-header_.begin()]);if(!std::isfinite(step))continue;
                iterations.push_back(step);
                for(std::size_t i=0;i<row.size();++i)if(!header_[i].empty()) {columns[header_[i]].push_back(number(row[i]));latestText[header_[i]]=row[i];}
            }
            partial_.erase(0,begin);return true;
        }
        [[nodiscard]] float latest(const std::string& key) const {
            const auto it=columns.find(key);return it==columns.end()||it->second.empty()?std::numeric_limits<float>::quiet_NaN():it->second.back();
        }
        [[nodiscard]] double sample(const std::string& key,std::size_t index,bool cumulative=false) const {
            const auto it=columns.find(key);
            if(it==columns.end()||index>=it->second.size())return std::numeric_limits<double>::quiet_NaN();
            if(!cumulative)return it->second[index];
            double total=0;
            for(std::size_t i=0;i<=index;++i)if(std::isfinite(it->second[i])&&it->second[i]>0)total+=it->second[i];
            return total;
        }
        [[nodiscard]] std::optional<std::size_t> nearestIteration(double target,double first,double last) const {
            std::optional<std::size_t> nearest;double distance=std::numeric_limits<double>::infinity();
            for(std::size_t i=0;i<iterations.size();++i) {
                const double step=iterations[i];
                if(!std::isfinite(step)||step<first||step>last)continue;
                const double next=std::abs(step-target);
                if(next<=distance) {distance=next;nearest=i;}
            }
            return nearest;
        }
        void plot(const char* label,const std::vector<std::pair<const char*,const char*>>& series,
                  bool rolling,int window,bool logarithmic=false,float height=120,bool independentRightAxis=false,bool integerCounts=false,bool cumulative=false) const {
            ImGui::PushID(label);ImGui::TextUnformatted(label);
            static constexpr ImVec4 colors[]={ui::blue,ui::accent,ui::green,{.82f,.5f,.76f,1},{.85f,.5f,.47f,1}};
            const float legendRight=ImGui::GetCursorScreenPos().x+ImGui::GetContentRegionAvail().x;
            for(std::size_t i=0;i<series.size();++i) {
                char text[160];const double value=cumulative&&!iterations.empty()?
                    sample(series[i].first,iterations.size()-1,true):latest(series[i].first);
                if(!std::isfinite(value))std::snprintf(text,sizeof(text),"%s —",series[i].second);
                else std::snprintf(text,sizeof(text),integerCounts?"%s %.0f":"%s %.4g",series[i].second,value);
                if(i&&ImGui::GetItemRectMax().x+ImGui::GetStyle().ItemSpacing.x+ImGui::CalcTextSize(text).x<legendRight)ImGui::SameLine();
                ImGui::TextColored(colors[i%5],"%s",text);
            }
            const auto origin=ImGui::GetCursorScreenPos();const ImVec2 size(std::max(60.f,ImGui::GetContentRegionAvail().x),height);
            ImGui::InvisibleButton("##plot",size);const bool hovered=ImGui::IsItemHovered();auto* draw=ImGui::GetWindowDrawList();
            draw->AddRectFilled(origin,ImVec2(origin.x+size.x,origin.y+size.y),ImGui::GetColorU32(ui::canvas),4);
            if(iterations.empty()) {draw->AddText(ImVec2(origin.x+8,origin.y+8),ImGui::GetColorU32(ui::muted),"Waiting for metrics.csv");ImGui::PopID();return;}
            double first=std::numeric_limits<double>::infinity(),last=-first;
            for(float step:iterations)if(std::isfinite(step)) {first=std::min(first,static_cast<double>(step));last=std::max(last,static_cast<double>(step));}
            if(!std::isfinite(first)) {ImGui::PopID();return;}
            if(rolling)first=std::max(first,last-std::max(1,window));
            // Adding one is ineffective at large iteration magnitudes, even in
            // double precision. Reserve a representable nonzero plotting span.
            if(last<=first)last=first+std::max(1.,std::abs(first)*1e-6);
            std::array<double,2> low{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()},high{-low[0],-low[1]};
            const auto scale=[&](double value) {return logarithmic?(value>0?std::log10(value):std::numeric_limits<double>::quiet_NaN()):value;};
            for(std::size_t c=0;c<series.size();++c)if(const auto it=columns.find(series[c].first);it!=columns.end()) {
                const int axis=independentRightAxis&&c==1?1:0;
                const auto count=std::min(iterations.size(),it->second.size());
                double total=0;
                for(std::size_t i=0;i<count;++i) {
                    const double raw=it->second[i];
                    if(cumulative&&std::isfinite(raw)&&raw>0)total+=raw;
                    if(!std::isfinite(iterations[i])||iterations[i]<first)continue;
                    const double y=scale(cumulative?total:raw);if(std::isfinite(y)) {low[axis]=std::min(low[axis],y);high[axis]=std::max(high[axis],y);}
                }
            }
            if(!std::isfinite(low[0])&&(!independentRightAxis||!std::isfinite(low[1]))) {ImGui::PopID();return;}
            for(int axis=0;axis<(independentRightAxis?2:1);++axis) {
                if(!std::isfinite(low[axis])) {low[axis]=0;high[axis]=1;}
                if(high[axis]<=low[axis])high[axis]=low[axis]+std::max(std::numeric_limits<double>::min(),low[axis]==0?1.:std::abs(low[axis])*.1);
            }
            const float margin=independentRightAxis?std::min(48.f,size.x*.22f):5.f;
            const float left=origin.x+margin,right=origin.x+size.x-margin,top=origin.y+7,bottom=origin.y+size.y-19;
            for(int i=1;i<4;++i) {const float y=top+(bottom-top)*i/4;draw->AddLine(ImVec2(left,y),ImVec2(right,y),ImGui::GetColorU32(ui::border));}
            if(independentRightAxis) {
                draw->AddLine(ImVec2(left,top),ImVec2(left,bottom),ImGui::GetColorU32(colors[0]));
                draw->AddLine(ImVec2(right,top),ImVec2(right,bottom),ImGui::GetColorU32(colors[1]));
                for(int tick=0;tick<=2;++tick) {
                    const float fraction=tick*.5f,y=bottom-(bottom-top)*fraction;
                    for(int axis=0;axis<2;++axis) {
                        const double scaled=low[axis]+(high[axis]-low[axis])*fraction;
                        const double value=logarithmic?std::pow(10.,scaled):scaled;
                        char text[48];std::snprintf(text,sizeof(text),"%.2g",value);
                        const auto extent=ImGui::CalcTextSize(text);
                        const float x=axis==0?left-extent.x-4:right+4;
                        draw->AddText(ImVec2(x,y-extent.y*.5f),ImGui::GetColorU32(colors[axis]),text);
                    }
                }
            }
            // Draw a min/max envelope per pixel column. Full history stays in
            // memory and brief spikes remain visible without millions of vertices.
            const int buckets=static_cast<int>(std::clamp(static_cast<double>(right-left),1.,16384.));
            for(std::size_t c=0;c<series.size();++c)if(const auto it=columns.find(series[c].first);it!=columns.end()) {
                const int axis=independentRightAxis&&c==1?1:0;
                std::vector<double> minimum(buckets,std::numeric_limits<double>::infinity()),maximum(buckets,-std::numeric_limits<double>::infinity());
                const auto count=std::min(iterations.size(),it->second.size());
                double total=0;
                for(std::size_t i=0;i<count;++i) {
                    const double raw=it->second[i];
                    if(cumulative&&std::isfinite(raw)&&raw>0)total+=raw;
                    if(!std::isfinite(iterations[i])||iterations[i]<first)continue;
                    const double y=scale(cumulative?total:raw);if(!std::isfinite(y))continue;
                    const double fraction=std::clamp((static_cast<double>(iterations[i])-first)/(last-first),0.,1.);
                    const int x=static_cast<int>(fraction*(buckets-1));
                    minimum[x]=std::min(minimum[x],y);maximum[x]=std::max(maximum[x],y);
                }
                std::optional<ImVec2> previous;
                for(int x=0;x<buckets;++x)if(std::isfinite(minimum[x])) {
                    const auto point=[&](double y) {
                        const double fraction=std::clamp((y-low[axis])/(high[axis]-low[axis]),0.,1.);
                        return ImVec2(left+x,static_cast<float>(bottom-fraction*(bottom-top)));
                    };
                    const auto a=point(minimum[x]),b=point(maximum[x]);
                    draw->AddLine(a,b,ImGui::GetColorU32(colors[c%5]),1.3f);
                    if(previous)draw->AddLine(*previous,a,ImGui::GetColorU32(colors[c%5]),1.3f);else draw->AddCircleFilled(a,2,ImGui::GetColorU32(colors[c%5]));
                    previous=b;
                }
            }
            char range[160];
            if(independentRightAxis)std::snprintf(range,sizeof(range),"Iteration %.0f — %.0f%s",first,last,logarithmic?"  ·  Log Y":"");
            else std::snprintf(range,sizeof(range),integerCounts?"Iteration %.0f — %.0f    %sY %.0f … %.0f":"Iteration %.0f — %.0f    %sY %.4g … %.4g",first,last,logarithmic?"log10 ":"",low[0],high[0]);
            draw->AddText(ImVec2(left,bottom+3),ImGui::GetColorU32(ui::muted),range);
            if(hovered) {
                const double fraction=std::clamp(static_cast<double>((ImGui::GetIO().MousePos.x-left)/(right-left)),0.,1.);
                if(const auto index=nearestIteration(first+fraction*(last-first),first,last)) {
                    const float x=left+static_cast<float>((iterations[*index]-first)/(last-first))*(right-left);
                    draw->AddLine(ImVec2(x,top),ImVec2(x,bottom),ImGui::GetColorU32(ui::muted));
                    ImGui::BeginTooltip();ImGui::TextUnformatted(label);
                    ImGui::Text("Iteration %.0f",iterations[*index]);
                    if(cumulative)ImGui::TextDisabled("Accumulated from the start of the run");
                    ImGui::Separator();
                    for(std::size_t c=0;c<series.size();++c) {
                        const double value=sample(series[c].first,*index,cumulative);
                        if(std::isfinite(value)) {
                            ImGui::TextColored(colors[c%5],integerCounts?"%s: %.0f":"%s: %.9g",series[c].second,value);
                            const int axis=independentRightAxis&&c==1?1:0;
                            const double y=scale(value);
                            if(std::isfinite(y))draw->AddCircleFilled(ImVec2(x,bottom-static_cast<float>(std::clamp(
                                (y-low[axis])/(high[axis]-low[axis]),0.,1.))*(bottom-top)),3.5f,ImGui::GetColorU32(colors[c%5]));
                        } else ImGui::TextDisabled("%s: not recorded",series[c].second);
                    }
                    ImGui::EndTooltip();
                }
            }
            ImGui::PopID();
        }
    };
}
