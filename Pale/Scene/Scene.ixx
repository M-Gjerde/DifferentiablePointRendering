//
// Created by magnus on 8/27/25.
//

module;
#include <entt/entt.hpp>

export module Pale.Scene;

import Pale.UUID;
import Pale.Log;
import Pale.Scene.Components;

export import :Entity;

export namespace Pale {
    class Scene {
    public:
        Scene();
        ~Scene();

        Entity createEntity(const std::string &name);

        Entity createEntityWithUUID(UUID uuid, const std::string &name);

        void destroyEntity(Entity entity);

        template<typename... Components>
        auto getAllEntitiesWith()
        {
            return m_registry.view<Components...>();
        }

        template<class T>
        void onComponentUpdated(Entity entity, T &component) {
        };

        template<typename T>
        void onComponentAdded(Entity entity, T &component) {
        };

        template<class T>
        void onComponentRemoved(Entity entity, T &component) {
        };

        entt::registry &getRegistry() { return m_registry; }

    private:
        friend class Entity;

        entt::registry m_registry;
    };


    template<typename T>
    bool Entity::hasComponent() {
        return m_scene->getRegistry().any_of<T>(m_entityHandle);
    }

    template<typename T, typename... Args>
    T &Entity::addComponent(Args &&... args) {
        Log::PA_ASSERT(!hasComponent<T>(), "Entity already has component!");
        T &component = m_scene->getRegistry().emplace<T>(m_entityHandle, std::forward<Args>(args)...);
        m_scene->onComponentAdded<T>(*this, component);
        return component;
    }


    template<typename T, typename... Args>
    T& Entity::addOrReplaceComponent(Args&&... args)
    {
        T& component = m_scene->getRegistry().emplace_or_replace<T>(m_entityHandle, std::forward<Args>(args)...);
        m_scene->onComponentAdded<T>(*this, component);
        return component;
    }



    template<typename T, typename... Args>
    T& Entity::getOrAddComponent(Args&&... args)
    {
        if (hasComponent<T>())
            return getComponent<T>();

        return addComponent<T>(std::forward<Args>(args)...);
    }
    template<typename T>
    T& Entity::getComponent()
    {
        Log::PA_ASSERT(hasComponent<T>(), "Entity does not have component!");
        return m_scene->getRegistry().get<T>(m_entityHandle);
    }


    template<typename T>
    void Entity::removeComponent()
    {
        Log::PA_ASSERT(hasComponent<T>(), "Entity does not have component!");
        auto component =  m_scene->getRegistry().get<T>(m_entityHandle);
        m_scene->onComponentRemoved<T>(*this, component);
        m_scene->getRegistry().remove<T>(m_entityHandle);
    }


    const std::string& Entity::getName() { return getComponent<TagComponent>().tag; }
    UUID Entity::getUUID()                  { return getComponent<IDComponent>().ID; }

}
