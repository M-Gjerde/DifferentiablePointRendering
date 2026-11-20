#pragma once

namespace Pale {
// Shared by the native optimizer and Python verification through pale.BETA_*.
inline constexpr float kBetaMin = -2.0f;
inline constexpr float kBetaMax = 5.0f;

// Material defaults for surfels from legacy point clouds.
inline constexpr float kDefaultSurfelRoughness = 0.5f;
inline constexpr float kDefaultSurfelMetalness = 0.0f;
inline constexpr float kMinimumGgxRoughness = 0.01f;
}
