// xorshift_sycl.hpp
#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>

namespace Pale::rng {

// ---------- SplitMix64 for robust seeding ----------
struct SplitMix64 {
    uint64_t currentState;

    explicit SplitMix64(uint64_t initialState) : currentState(initialState) {}

    inline uint64_t nextUint64() {
        uint64_t z = (currentState += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
};

// ---------- Xorshift32 (Marsaglia) ----------
struct Xorshift32 {
    uint32_t currentState;

    explicit Xorshift32(uint32_t seed) : currentState(seed ? seed : 0xA341316Cu) {}

    inline uint32_t nextUint32() {
        uint32_t x = currentState;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        currentState = x;
        return x;
    }

    inline float nextFloat01() {                  // [0,1)
        // Use top 24 bits to avoid denorms and keep uniform mantissa fill
        return static_cast<float>(nextUint32() >> 8) * 0x1.0p-24f;
    }
};

// ---------- Xorshift64* (better quality than plain 64) ----------
struct Xorshift64Star {
    uint64_t currentState;

    explicit Xorshift64Star(uint64_t seed) : currentState(seed ? seed : 0x9E3779B97F4A7C15ull) {}

    inline uint64_t nextUint64() {
        uint64_t x = currentState;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        currentState = x;
        return x * 0x2545F4914F6CDD1Dull;
    }

    inline double nextDouble01() {                // [0,1)
        // Use top 53 bits for full double mantissa precision
        return (nextUint64() >> 11) * 0x1.0p-53;
    }
};

// ---------- Xorshift128 (32-bit state, very fast) ----------
struct Xorshift128 {
    uint32_t stateX, stateY, stateZ, stateW;

    Xorshift128(uint32_t sx, uint32_t sy, uint32_t sz, uint32_t sw)
        : stateX(sx ? sx : 0x1u),
          stateY(sy ? sy : 0x9E3779B9u),
          stateZ(sz ? sz : 0x7F4A7C15u),
          stateW(sw ? sw : 0x94D049BBu) {}

    explicit Xorshift128(uint64_t seed) {
        // Fill with SplitMix64 to avoid correlated seeds
        SplitMix64 sm(seed);
        stateX = static_cast<uint32_t>(sm.nextUint64());
        stateY = static_cast<uint32_t>(sm.nextUint64());
        stateZ = static_cast<uint32_t>(sm.nextUint64());
        stateW = static_cast<uint32_t>(sm.nextUint64());
        if (!(stateX | stateY | stateZ | stateW)) stateW = 0xA511E9B3u; // avoid all-zero
    }

    uint32_t nextUint() {
        uint32_t t = stateX ^ (stateX << 11);
        stateX = stateY; stateY = stateZ; stateZ = stateW;
        stateW = (stateW ^ (stateW >> 19)) ^ (t ^ (t >> 8));
        return stateW;
    }

    float nextFloat() {                  // [0,1)
        return static_cast<float>(nextUint() >> 8) * 0x1.0p-24f;
    }
};

// ---------- Helper: per-item deterministic seeding ----------
inline uint64_t makePerItemSeed1D(uint64_t baseSeed, sycl::id<1> globalId) {
    // Mix base seed with global id to minimize collisions
    uint64_t mixed = baseSeed ^ (0x9E3779B97F4A7C15ull + static_cast<uint64_t>(globalId[0]));
    // One SplitMix64 step to decorrelate contiguous ids
    SplitMix64 sm(mixed);
    return sm.nextUint64();
}

} // namespace pale::rng
