//
// Created by magnus on 8/28/25.
//
module;

#include <random>
#include <stdexcept>

module Pale.UUID;


namespace Pale {

    static std::random_device s_RandomDevice;
    static std::mt19937_64 s_Engine(s_RandomDevice());
    static std::uniform_int_distribution<uint64_t> s_UniformDistribution;

    UUID::UUID()
            : m_UUID(s_UniformDistribution(s_Engine))
    {
    }

    UUID::UUID(uint64_t uuid)
            : m_UUID(uuid)
    {
    }

    UUID::UUID(const std::string &str) {
        try {
            m_UUID = std::stoull(str);
        } catch (const std::exception& e) {
            throw std::invalid_argument("Invalid UUID string: " + str);
        }
    }
}

