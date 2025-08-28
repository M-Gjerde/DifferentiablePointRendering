//
// Created by magnus on 8/28/25.
//

module;
#include <entt/entt.hpp>


export module Pale.Scene:Entity;

import Pale.Log;
import Pale.UUID;

export namespace Pale {
    class Scene;

    class Entity {
    public:
        Entity() = default;

        Entity(entt::entity handle, Scene *scene);

        Entity(const Entity &other) = default;

        template<typename T, typename... Args>
        T &addComponent(Args &&... args);

        template<typename T, typename... Args>
        T &addOrReplaceComponent(Args &&... args);

        template<typename T, typename... Args>
        T &getOrAddComponent(Args &&... args);

        template<typename T>
        T &getComponent();

        template<typename T>
        bool hasComponent();

        template<typename T>
        void removeComponent();

        const std::string &getName();

        UUID getUUID();

        bool operator==(const Entity &other) const {
            return m_entityHandle == other.m_entityHandle && m_scene == other.m_scene;
        }

        bool operator!=(const Entity &other) const {
            return !(*this == other);
        }

        operator bool() const { return m_entityHandle != entt::null; }
        operator entt::entity() const { return m_entityHandle; }
        operator uint32_t() const { return static_cast<uint32_t>(m_entityHandle); }

        Scene *getScene() { return m_scene; }

    private:
        friend class Scene;
        entt::entity m_entityHandle{entt::null};
        Scene *m_scene = nullptr;
    };
}
