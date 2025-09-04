// ============================
// File: Pale.Assets.Manager.ixx
// ============================
module;

#include <memory>
#include <functional>
#include <shared_mutex>
#include <thread>
#include <future>
#include <filesystem>

export module Pale.Assets:Manager;


import Pale.Assets.Core;
import :Registry;


export namespace Pale {


    class AssetManager {
    public:
        explicit AssetManager(std::size_t cacheCapacity = 512)
        : m_cache(cacheCapacity), m_running(true), m_watcherThread(&AssetManager::watchLoop, this) {}


        ~AssetManager() {
            m_running = false;
            if (m_watcherThread.joinable()) m_watcherThread.join();
        }


        AssetRegistry& registry() { return m_registry; }


        template <typename T>
        void registerLoader(AssetType type, std::shared_ptr<IAssetLoader<T>> loader) {
            std::unique_lock lk(m_loaderMtx);
            m_loaders[type] = [loader](const AssetHandle& id, const AssetMeta& m) -> std::shared_ptr<IAsset> {
                return std::static_pointer_cast<IAsset>(loader->load(id, m));
            };
        }

        template <typename T>
        AssetPtr<T> get(const AssetHandle& id) {
            if (auto ptr = std::static_pointer_cast<T>(m_cache.get(id)); ptr)
                return ptr;
            auto meta = m_registry.meta(id);
            if (!meta)
                return {};
            LoaderFn loader;
            { std::shared_lock lk(m_loaderMtx);
                if (auto it = m_loaders.find(meta->type); it != m_loaders.end()) loader = it->second; }
            if (!loader)
                return {};
            auto asset = loader(id, *meta);
            if (!asset)
                return {};
            m_cache.put(id, asset);
            return std::static_pointer_cast<T>(asset);
        }

        void prefetch(const AssetHandle& id) {
            if (m_futures.contains(id)) return;
            auto meta = m_registry.meta(id);
            if (!meta) return;
            auto loader = getLoader(meta->type);
            if (!loader) return;
            m_futures[id] = std::async(std::launch::async, [this, id, meta, loader]{
            auto a = loader(id, *meta);
            if (a) m_cache.put(id, std::move(a));
            });
        }


        void enableHotReload(bool on) { m_hotReload.store(on); }

    private:
        using LoaderFn = std::function<std::shared_ptr<IAsset>(const AssetHandle&, const AssetMeta&)>;


        LoaderFn getLoader(AssetType t) {
            std::shared_lock lk(m_loaderMtx);
            if (auto it = m_loaders.find(t); it != m_loaders.end()) return it->second;
            return {};
        }


        void watchLoop() {
            using namespace std::chrono_literals;
            while (m_running.load()) {
                if (!m_hotReload.load()) { std::this_thread::sleep_for(200ms); continue; }
                auto snap = m_registry.snapshot();
                for (auto& [id, meta] : snap) {
                    if (!std::filesystem::exists(meta.path)) continue;
                    auto nowWrite = std::filesystem::last_write_time(meta.path);
                    if (meta.lastWrite != std::filesystem::file_time_type{} && nowWrite != meta.lastWrite) {
                        m_cache.erase(id); // invalidate; next get() reloads
                    }
                }
                std::this_thread::sleep_for(400ms);
            }
        }


        AssetRegistry m_registry{};
        AssetCache m_cache{};
        std::unordered_map<AssetType, LoaderFn> m_loaders{};
        std::shared_mutex m_loaderMtx{};
        std::unordered_map<AssetHandle, std::future<void>> m_futures{};


        std::atomic<bool> m_running{false};
        std::atomic<bool> m_hotReload{false};
        std::thread m_watcherThread{};
    };


} // namespace Pale
