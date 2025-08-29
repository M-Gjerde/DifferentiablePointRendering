//
// Created by magnus on 8/27/25.
//
module;

#include <entt/entt.hpp>
module Pale.Scene;

import Pale.Log;
import Pale.Scene.Components;

namespace Pale {

    Scene::Scene() = default;

    Scene::~Scene() {
        // Destroy all entities one by one rather than calling m_Registry.clear()
        // This ensures component on_destroy signals are fired in the correct order.
        // Note that the scene entity does not have an IDComponent, so it will not
        // be destroyed here.
        auto view = getAllEntitiesWith<IDComponent>();
        for (entt::entity entity : view) {                  // entt::entity is explicit; 'auto e' also OK
            destroyEntity({ entity, this });
        }

        // Destroy anything that's left (should be just the scene entity)
        m_registry.clear();
    }


    Entity Scene::createEntityWithUUID(UUID uuid, const std::string &name) {
        Entity entity = {m_registry.create(), this};
        entity.addComponent<IDComponent>(uuid);
        entity.addComponent<TransformComponent>();
        auto &tag = entity.addComponent<TagComponent>();
        tag.tag = name.empty() ? "Entity" : name;
        Log::PA_INFO("Created Entity with UUID: {} and Tag: {}",entity.getUUID().operator std::string(), entity.getName());

        return entity;
    }

    void Scene::destroyEntity(Entity entity) {

        Log::PA_INFO("Deleting Entity with UUID: {} and Tag: {}", entity.getUUID().operator std::string(), entity.getName());

        // For components that require deinitialization remove them here: Check if they exists first.

        m_registry.destroy(entity.m_entityHandle);

    }


    Entity Scene::createEntity(const std::string &name) {
        return createEntityWithUUID(UUID(), name);
    }


}