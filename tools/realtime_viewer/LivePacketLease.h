#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace viewer {
    class LivePacketLease {
        struct Directory {
            std::filesystem::path path;
            ~Directory() {std::error_code ignored;std::filesystem::remove(path,ignored);}
        };
        std::shared_ptr<Directory> directory_;
        std::filesystem::path path_;
        std::filesystem::path source_;
        friend class LivePacketLeaseCache;
        LivePacketLease(std::shared_ptr<Directory> directory,std::filesystem::path path,std::filesystem::path source)
            :directory_(std::move(directory)),path_(std::move(path)),source_(std::move(source)) {}
    public:
        ~LivePacketLease() {std::error_code ignored;std::filesystem::remove(path_,ignored);}
        LivePacketLease(const LivePacketLease&)=delete;
        LivePacketLease& operator=(const LivePacketLease&)=delete;
        [[nodiscard]] const std::filesystem::path& path() const {return path_;}
        [[nodiscard]] const std::filesystem::path& source() const {return source_;}
    };

    // Only the owned optimizer session's immutable packets may be leased. A
    // hard link survives publisher cleanup without copying a large point cloud.
    // Latest packets are cached; consumers explicitly retain displayed/exported
    // packets, and their files live until the final consumer releases them.
    class LivePacketLeaseCache {
        std::size_t capacity_;
        std::deque<std::shared_ptr<const LivePacketLease>> recent_;
        std::vector<std::weak_ptr<const LivePacketLease>> leases_;
        std::vector<std::pair<std::filesystem::path,std::weak_ptr<LivePacketLease::Directory>>> directories_;
        inline static std::atomic<unsigned long long> sequence_{0};
        [[nodiscard]] std::shared_ptr<LivePacketLease::Directory> directory(const std::filesystem::path& session) {
            std::erase_if(directories_,[](const auto& entry){return entry.second.expired();});
            for(const auto& [owner,weak]:directories_)if(owner==session)if(auto existing=weak.lock())return existing;
            auto result=std::make_shared<LivePacketLease::Directory>();
            for(unsigned attempt=0;attempt<100;++attempt) {
                const auto tick=std::chrono::steady_clock::now().time_since_epoch().count();
                const auto candidate=session/(".viewer_reader_"+std::to_string(tick)+"_"+std::to_string(sequence_++));
                std::error_code error;
                if(std::filesystem::create_directory(candidate,error)) {
                    result->path=candidate;
                    std::filesystem::permissions(candidate,std::filesystem::perms::owner_all,std::filesystem::perm_options::replace,error);
                    if(error)throw std::filesystem::filesystem_error("Cannot protect live packet lease directory",candidate,error);
                    directories_.emplace_back(session,result);return result;
                }
                if(error&&error!=std::errc::file_exists)throw std::filesystem::filesystem_error("Cannot create live packet lease directory",candidate,error);
            }
            throw std::runtime_error("Cannot reserve a live packet lease directory");
        }
        static bool packetName(const std::filesystem::path& path) {
            const auto name=path.filename().string();
            constexpr const char* prefix="live-";constexpr const char* suffix=".ply";
            if(!name.starts_with(prefix)||!name.ends_with(suffix)||name.size()<=9)return false;
            return std::all_of(name.begin()+5,name.end()-4,[](char c){return c>='0'&&c<='9';});
        }
    public:
        explicit LivePacketLeaseCache(std::size_t capacity=4):capacity_(std::max<std::size_t>(1,capacity)) {}
        [[nodiscard]] std::shared_ptr<const LivePacketLease> acquire(
            const std::filesystem::path& source,const std::filesystem::path& ownedSession) {
            const auto session=std::filesystem::canonical(ownedSession);
            const auto owner=std::filesystem::canonical(source.parent_path());
            if(owner!=session||!packetName(source))throw std::runtime_error("Live packet is outside the viewer-owned session");
            const auto absoluteSource=owner/source.filename();
            std::erase_if(leases_,[](const auto& item){return item.expired();});
            for(const auto& weak:leases_)if(auto lease=weak.lock();lease&&lease->source()==absoluteSource)return lease;
            const auto status=std::filesystem::symlink_status(absoluteSource);
            if(status.type()!=std::filesystem::file_type::regular)throw std::runtime_error("Live packet is missing or is not a regular owned file");
            auto leaseDirectory=directory(session);
            const auto destination=leaseDirectory->path/source.filename();
            std::filesystem::create_hard_link(absoluteSource,destination);
            auto lease=std::shared_ptr<const LivePacketLease>(new LivePacketLease(leaseDirectory,destination,absoluteSource));
            recent_.push_back(lease);leases_.push_back(lease);
            while(recent_.size()>capacity_)recent_.pop_front();
            return lease;
        }
        [[nodiscard]] std::shared_ptr<const LivePacketLease> retain(const std::filesystem::path& path) const {
            for(const auto& weak:leases_)if(auto lease=weak.lock();lease&&lease->path()==path)return lease;
            return {};
        }
        [[nodiscard]] std::size_t cachedCount() const {return recent_.size();}
        void clear() {recent_.clear();std::erase_if(leases_,[](const auto& lease){return lease.expired();});}
    };
}
