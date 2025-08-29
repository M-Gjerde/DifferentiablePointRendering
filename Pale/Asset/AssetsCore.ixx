//
// Created by magnus on 8/28/25.
//

module;
#include <string>
#include <string_view>
#include <filesystem>
#include <unordered_map>
#include <optional>
#include <memory>
#include <variant>
#include <vector>
#include <list>
#include <mutex>
#include <shared_mutex>
#include <functional>
#include <future>
#include <thread>
#include <chrono>


export module Pale.Assets:Core;


import Pale.UUID; // Provides Pale::UUID


export namespace Pale {
    using Clock = std::chrono::steady_clock;
    using AssetHandle = UUID;


    enum class AssetType { Unknown, Mesh, Material, Shader };


    struct AssetMeta {
        AssetType type{AssetType::Unknown};
        std::filesystem::path path{};
        std::filesystem::file_time_type lastWrite{};
        bool valid() const { return !path.empty(); }
    };


    // Base asset interface for shared_ptr type erasure
    struct IAsset {
        virtual ~IAsset() = default;
    };


    // Smart pointer alias for assets
    template<typename T>
    using AssetPtr = std::shared_ptr<T>;

    // Tiny LRU cache
    // -----------------
    class AssetCache {
    public:
        explicit AssetCache(std::size_t capacity = 256) : m_capacity(capacity) {
        }

        void setCapacity(std::size_t c) {
            std::scoped_lock lk(m_mtx);
            m_capacity = c;
            evict();
        }


        void put(const AssetHandle &id, std::shared_ptr<IAsset> asset) {
            std::scoped_lock lk(m_mtx);
            if (auto it = m_map.find(id); it != m_map.end()) {
                touch(it->second);
                it->second->value = std::move(asset);
                return;
            }
            m_list.emplace_front(Node{id, std::move(asset)});
            m_map.emplace(id, m_list.begin());
            evict();
        }


        std::shared_ptr<IAsset> get(const AssetHandle &id) {
            std::scoped_lock lk(m_mtx);
            auto it = m_map.find(id);
            if (it == m_map.end()) return {};
            touch(it->second);
            return it->second->value;
        }


        void erase(const AssetHandle &id) {
            std::scoped_lock lk(m_mtx);
            if (auto it = m_map.find(id); it != m_map.end()) {
                m_list.erase(it->second);
                m_map.erase(it);
            }
        }

    private:
        struct Node {
            AssetHandle id;
            std::shared_ptr<IAsset> value;
        };


        void touch(std::list<Node>::iterator it) {
            m_list.splice(m_list.begin(), m_list, it);
        }


        void evict() {
            while (m_map.size() > m_capacity) {
                auto &node = m_list.back();
                m_map.erase(node.id);
                m_list.pop_back();
            }
        }


        std::size_t m_capacity{};
        std::list<Node> m_list{};
        std::unordered_map<AssetHandle, std::list<Node>::iterator> m_map{};
        std::mutex m_mtx{};
    };

    // Loader concept via interface
    template <typename T>
    struct IAssetLoader {
        virtual ~IAssetLoader() = default;
        virtual AssetPtr<T> load(const AssetHandle& id, const AssetMeta& meta) = 0;
    };
}

