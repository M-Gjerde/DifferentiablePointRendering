//
// Created by magnus on 8/28/25.
//
module;

#include <cstdint>
#include <string>

export module Pale.UUID;


export namespace Pale {

    class UUID
    {
    public:
        UUID();
        explicit UUID(uint64_t uuid);
        UUID(const UUID&) = default;
        // New: construct from string
        explicit UUID(const std::string& str);

        auto operator<=>(const UUID&) const = default;

        explicit operator uint64_t() const { return m_UUID; }
        explicit operator std::string() const { return std::to_string(m_UUID); }
    private:
        uint64_t m_UUID;
    };

}



export namespace std {
    template<>
    struct hash<Pale::UUID> {
        std::size_t operator()(const Pale::UUID& uuid) const noexcept {
            return static_cast<uint64_t>(uuid);
        }
    };
}