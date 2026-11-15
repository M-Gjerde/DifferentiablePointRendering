#pragma once
#include <chrono>
#include <algorithm>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>

namespace viewer {
    struct FolderTimestamp {
        std::chrono::system_clock::time_point value=std::chrono::system_clock::time_point::min();
        bool creation=false;
    };
    // Linux birth time is distinct from ctime (metadata changes). Some file
    // systems do not provide it; only those entries use modification time.
    inline FolderTimestamp folderTimestamp(const std::filesystem::path& path) {
#if defined(__linux__) && defined(STATX_BTIME)
        struct statx info{};
        if(::statx(AT_FDCWD,path.c_str(),0,STATX_BTIME,&info)==0&&(info.stx_mask&STATX_BTIME))
            return {std::chrono::system_clock::time_point(
                std::chrono::seconds(info.stx_btime.tv_sec)+std::chrono::nanoseconds(info.stx_btime.tv_nsec)),true};
#endif
        std::error_code error;
        const auto modified=std::filesystem::last_write_time(path,error);
        if(!error)return {std::filesystem::file_time_type::clock::to_sys(modified),false};
        return {};
    }
    inline std::string folderTimestampLabel(const FolderTimestamp& timestamp) {
        if(timestamp.value==std::chrono::system_clock::time_point::min())return "Unavailable";
        const auto time=std::chrono::system_clock::to_time_t(timestamp.value);
        std::tm local{};
        if(!::localtime_r(&time,&local))return "Unavailable";
        char text[40];std::strftime(text,sizeof(text),"%Y-%m-%d %H:%M:%S",&local);
        return std::string(text)+(timestamp.creation?"":" *");
    }
    struct ReconstructionFolder {
        std::filesystem::path path;
        FolderTimestamp created;
    };
    inline std::vector<ReconstructionFolder> reconstructionFolders(const std::filesystem::path& root) {
        std::vector<ReconstructionFolder> folders;
        std::error_code error;
        for(std::filesystem::directory_iterator it(root,std::filesystem::directory_options::skip_permission_denied,error),end;
            it!=end;it.increment(error)) {
            if(error)break;
            if(it->is_directory(error)&&!it->path().filename().string().starts_with('.'))
                folders.push_back({it->path(),folderTimestamp(it->path())});
            error.clear();
        }
        std::sort(folders.begin(),folders.end(),[](const auto& a,const auto& b) {
            return a.created.value!=b.created.value?a.created.value>b.created.value:a.path.string()<b.path.string();
        });
        return folders;
    }
}
