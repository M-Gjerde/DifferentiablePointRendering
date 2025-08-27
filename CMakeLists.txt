cmake_minimum_required(VERSION 4.1)
project(DifferentiablePointRendering LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(AdaptiveCpp REQUIRED)

## Adding Vendor Libraries
add_subdirectory(vendor/pugixml)
include_directories(vendor/pugixml/src)

# Wrap vulkan.hpp as a module
add_library(pugixml_mod)
target_sources(pugixml_mod PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS vendor/pugixml
        FILES vendor/pugixml/src/pugixml.cppm
)

add_executable(${PROJECT_NAME}
        main.cpp
        Src/DeviceSelector.cpp
        Src/Scene/Scene.cpp
)


## Module files
target_sources(${PROJECT_NAME}  PUBLIC
        FILE_SET CXX_MODULES FILES
        Src/DeviceSelector.ixx
        Src/Scene/Scene.ixx
        Src/Scene/SceneSerializer.ixx
)

target_link_libraries(${PROJECT_NAME} PRIVATE pugixml_mod)

# Debug/Release backend choice
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(ACPP_TARGETS "omp")        # or set to your GPU backend: cuda / hip
else()
    set(ACPP_TARGETS "omp")
    target_compile_options(${PROJECT_NAME} PRIVATE -O0 -g)
    message(STATUS "[VkRenderINFO]: Compiling using AdaptiveCpp without optimization")
endif()

message(STATUS "[VkRenderINFO]: Using AdaptiveCpp with targets: ${ACPP_TARGETS} for ${PROJECT_NAME}")

# Add SYCL to target via AdaptiveCpp's helper
add_sycl_to_target(TARGET ${PROJECT_NAME} TARGETS ${ACPP_TARGETS})
