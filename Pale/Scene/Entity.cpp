//
// Created by magnus on 8/28/25.
//
module;

#include <entt/entt.hpp>


module Pale.Scene;

import Pale.Log;

namespace Pale {
    Entity::Entity(entt::entity handle, Scene *scene)
        : m_entityHandle(handle), m_scene(scene) {
    }



}
