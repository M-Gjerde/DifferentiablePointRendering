#pragma once
#include "DatasetLibrary.h"
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

namespace viewer {
    // Run the system GTK folder picker without blocking the viewer's event loop.
    class DatasetBrowser {
        pid_t pid_=-1;
        int output_=-1;
        std::string selection_;

        void closeOutput() {
            if(output_>=0) {::close(output_);output_=-1;}
        }
        void stop() {
            if(pid_>0) {
                ::kill(pid_,SIGKILL);
                while(::waitpid(pid_,nullptr,0)<0&&errno==EINTR) {}
                pid_=-1;
            }
            closeOutput();
        }
    public:
        DatasetBrowser()=default;
        DatasetBrowser(const DatasetBrowser&)=delete;
        DatasetBrowser& operator=(const DatasetBrowser&)=delete;
        ~DatasetBrowser() {stop();}
        [[nodiscard]] bool active() const {return pid_>0;}

        void open(std::filesystem::path folder) {
            if(active())return;
            std::error_code error;
            while(!folder.empty()&&!std::filesystem::is_directory(folder,error)&&folder!=folder.parent_path())
                folder=folder.parent_path();
            if(folder.empty())folder=std::filesystem::current_path();
            int pipe[2];
            if(::pipe2(pipe,O_CLOEXEC)<0)throw std::runtime_error("Could not open dataset picker: "+std::string(std::strerror(errno)));
            if(::fcntl(pipe[0],F_SETFL,O_NONBLOCK)<0) {
                const auto reason=std::string(std::strerror(errno));
                ::close(pipe[0]);::close(pipe[1]);
                throw std::runtime_error("Could not open dataset picker: "+reason);
            }
            std::array<std::string,5> arguments{"zenity","--file-selection","--directory",
                "--title=Choose dataset folder","--filename="+folder.string()+"/"};
            std::array<char*,6> argv{};
            for(std::size_t i=0;i<arguments.size();++i)argv[i]=arguments[i].data();
            posix_spawn_file_actions_t actions;
            int result=::posix_spawn_file_actions_init(&actions);
            if(!result) {
                result=::posix_spawn_file_actions_addclose(&actions,pipe[0]);
                if(!result)result=::posix_spawn_file_actions_adddup2(&actions,pipe[1],STDOUT_FILENO);
                if(!result)result=::posix_spawn_file_actions_addclose(&actions,pipe[1]);
                if(!result)result=::posix_spawnp(&pid_,argv[0],&actions,nullptr,argv.data(),::environ);
                ::posix_spawn_file_actions_destroy(&actions);
            }
            ::close(pipe[1]);
            if(result) {
                ::close(pipe[0]);pid_=-1;
                throw std::runtime_error(result==ENOENT?"Dataset folder selection requires zenity on PATH":
                    "Could not open dataset picker: "+std::string(std::strerror(result)));
            }
            output_=pipe[0];selection_.clear();
        }
        std::optional<std::filesystem::path> poll() {
            if(!active())return {};
            int status=0;
            const auto result=::waitpid(pid_,&status,WNOHANG);
            if(result<0&&errno==EINTR)return {};
            if(result<0) {
                pid_=-1;closeOutput();
                throw std::runtime_error("Could not wait for the dataset picker");
            }
            if(result>0)pid_=-1;
            char buffer[4096];
            for(;;) {
                const auto count=::read(output_,buffer,sizeof(buffer));
                if(count>0) {
                    selection_.append(buffer,static_cast<std::size_t>(count));
                    if(selection_.size()>4096) {stop();throw std::runtime_error("Dataset folder path is too long");}
                } else if(count<0&&errno==EINTR)continue;
                else if(count<0&&errno!=EAGAIN&&errno!=EWOULDBLOCK) {
                    stop();throw std::runtime_error("Could not read the dataset folder selection");
                } else break;
            }
            if(result==0)return {};
            closeOutput();
            if(WIFEXITED(status)&&WEXITSTATUS(status)==1)return {}; // Cancel.
            if(!WIFEXITED(status)||WEXITSTATUS(status)!=0)throw std::runtime_error("The native dataset picker failed");
            if(!selection_.empty()&&selection_.back()=='\n')selection_.pop_back();
            if(selection_.empty())return {};
            if(selection_.size()>=2048)throw std::runtime_error("Dataset folder path is too long");
            const auto folder=std::filesystem::path(selection_).lexically_normal();
            if(!isDatasetDirectory(folder))throw std::runtime_error("Dataset needs images/, scene.xml and points.ply");
            return folder;
        }
    };
}
