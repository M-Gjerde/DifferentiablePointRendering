#pragma once
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace viewer {
    // A bounded plain-text terminal transcript. tqdm uses CR and ANSI erase/up
    // sequences rather than appending a new line for each progress refresh.
    class TrainingLog {
        std::filesystem::path path_;
        std::uintmax_t offset_=0;
        std::vector<std::string> lines_{1};
        std::size_t row_=0,column_=0;
        enum class Escape { None, Start, Csi, Osc, OscEnd } escape_=Escape::None;
        std::string control_;
        std::size_t limit_;
        void csi(char command) {
            int value=0;
            for(char c:control_) {if(!std::isdigit(static_cast<unsigned char>(c)))break; value=std::min(100000,value*10+c-'0');}
            const auto count=static_cast<std::size_t>(std::max(value,1));
            if(command=='A')row_=row_>count?row_-count:0;
            if(command=='B') {row_=std::min(row_+count,lines_.size());if(row_==lines_.size())lines_.emplace_back();}
            if(command=='C')column_=std::min(column_+count,limit_);
            if(command=='D')column_=column_>count?column_-count:0;
            if(command=='G')column_=count-1;
            if(command=='K') {
                auto& line=lines_[row_];
                if(value==2)line.assign(std::min(column_,limit_),' ');
                else if(value==1)std::fill_n(line.begin(),std::min(column_+1,line.size()),' ');
                else if(column_<line.size())line.resize(column_);
            }
        }
        void trim() {
            std::size_t bytes=0;
            for(const auto& line:lines_)bytes+=line.size()+1;
            std::size_t remove=0;
            while(bytes>limit_&&remove+1<lines_.size()) {bytes-=lines_[remove].size()+1;++remove;}
            if(remove) {lines_.erase(lines_.begin(),lines_.begin()+remove);row_=row_>=remove?row_-remove:0;}
            auto& line=lines_[row_];
            if(line.size()>limit_) {
                std::size_t cut=line.size()-limit_;
                while(cut<line.size()&&(static_cast<unsigned char>(line[cut])&0xc0)==0x80)++cut;
                line.erase(0,cut);column_=column_>=cut?column_-cut:0;
            }
        }
    public:
        explicit TrainingLog(std::size_t limit=128*1024):limit_(std::max<std::size_t>(limit,1)){}
        void reset() {offset_=0;lines_={""};row_=column_=0;escape_=Escape::None;control_.clear();}
        void append(const std::string& chunk) {
            for(unsigned char c:chunk) {
                if(escape_==Escape::Osc) {if(c==7)escape_=Escape::None;else if(c==27)escape_=Escape::OscEnd;continue;}
                if(escape_==Escape::OscEnd) {escape_=c=='\\'?Escape::None:Escape::Osc;continue;}
                if(escape_==Escape::Start) {escape_=c=='['?Escape::Csi:c==']'?Escape::Osc:Escape::None;control_.clear();continue;}
                if(escape_==Escape::Csi) {
                    if(c>=0x40&&c<=0x7e) {csi(static_cast<char>(c));escape_=Escape::None;}
                    else if(control_.size()<64)control_+=static_cast<char>(c);
                    continue;
                }
                if(c==27) {escape_=Escape::Start;continue;}
                if(c=='\r') {column_=0;continue;}
                if(c=='\n') {++row_;column_=0;if(row_==lines_.size())lines_.emplace_back();continue;}
                if(c=='\b') {if(column_)--column_;continue;}
                if(c=='\t') {column_=std::min(((column_/8)+1)*8,limit_);continue;}
                if(c<32||c==127)continue;
                auto& line=lines_[row_];
                if(column_>=line.size())line.resize(column_+1,' ');
                line[column_++]=static_cast<char>(c);
                if(line.size()>limit_*2)trim();
            }
            trim();
        }
        [[nodiscard]] std::string text() const {
            std::string result;
            for(std::size_t i=0;i<lines_.size();++i) {if(i)result+='\n';result+=lines_[i];}
            return result;
        }
        bool update(const std::filesystem::path& path) {
            std::error_code error;const auto size=std::filesystem::file_size(path,error);
            if(error)return false;
            if(path!=path_||size<offset_) {reset();path_=path;}
            if(size==offset_)return false;
            std::ifstream file(path,std::ios::binary);if(!file)return false;
            file.seekg(static_cast<std::streamoff>(offset_));
            std::string chunk(static_cast<std::size_t>(std::min<std::uintmax_t>(size-offset_,256*1024)), '\0');
            file.read(chunk.data(),chunk.size());chunk.resize(static_cast<std::size_t>(file.gcount()));
            offset_+=chunk.size();append(chunk);return true;
        }
    };
}
