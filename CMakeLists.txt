cmake_minimum_required(VERSION 4.1)
project(DifferentiablePointRendering LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Important: do NOT enable CMake's experimental import std ---
# (These trigger clang-scan-deps and collide with syclcc-launcher.)
# set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD ... )
# set(CMAKE_CXX_MODULE_STD 1)
# set_target_properties(${PROJECT_NAME} PROPERTIES CXX_MODULE_STD ON)
# set_target_properties(${PROJECT_NAME} PROPERTIES CMAKE_CXX_COMPILER_IMPORT_STD ON)

find_package(AdaptiveCpp REQUIRED)

add_executable(${PROJECT_NAME}
        main.cpp
        Src/DeviceSelector.cpp
        # If you keep a .ixx file, treat it like a header for now (see note below)

)

target_sources(${PROJECT_NAME}  PUBLIC
        FILE_SET CXX_MODULES FILES   Src/DeviceSelector.ixx
)

# Debug/Release backend choice (tweak to your needs)
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(ACPP_TARGETS "omp")        # or set to your GPU backend: cuda / hip
else()
    set(ACPP_TARGETS "omp")
    target_compile_options(${PROJECT_NAME} PRIVATE -O0 -g)
    message(STATUS "[VkRenderINFO]: Compiling using AdaptiveCpp without optimization")
endif()

message(STATUS "[VkRenderINFO]: Using AdaptiveCpp with targets: ${ACPP_TARGETS} for ${PROJECT_NAME}")

# Add SYCL to target via AdaptiveCpp's helper (sets the launcher etc.)
add_sycl_to_target(TARGET ${PROJECT_NAME} TARGETS ${ACPP_TARGETS})

# Optional: if you have non-SYCL translation units that must not be passed
# through the launcher, you can move them into a separate static lib and link it.
