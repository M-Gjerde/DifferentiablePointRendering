#pragma once

namespace Pale {

// Shared by the device optimizer and Python's beta validation through the
// pale.BETA_MIN / pale.BETA_MAX module attributes.
inline constexpr float kBetaMin = -2.0f;
inline constexpr float kBetaMax = 5.0f;

} // namespace Pale
