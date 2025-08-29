cmake_minimum_required(VERSION 3.5)
project(DifferentiablePointRendering LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
find_package(AdaptiveCpp REQUIRED)

## Adding Vendor Libraries
add_subdirectory(vendor/pugixml)
include_directories(vendor/pugixml/src)

# --- Pugi module wrapper (module interface + link reqs)
add_library(pugixml_iface INTERFACE)
target_include_directories(pugixml_iface INTERFACE ${CMAKE_SOURCE_DIR}/vendor/pugixml/src)
target_link_libraries(pugixml_iface INTERFACE pugixml)

# Entt
set(ENTT_DIR vendor/entt)
add_subdirectory(${ENTT_DIR})

## SPD LOG

include(FetchContent)
set(SPDLOG_USE_STD_FORMAT ON)
FetchContent_Declare(
        spdlog
        SOURCE_DIR "${CMAKE_SOURCE_DIR}/vendor/spdlog"
        BINARY_DIR "${CMAKE_BINARY_DIR}/vendor/spdlog-build"
)

## GLM
FetchContent_Declare(
        glm
        SOURCE_DIR "${CMAKE_SOURCE_DIR}/vendor/glm"
        BINARY_DIR "${CMAKE_BINARY_DIR}/vendor/glm-build"
)
## assimp
FetchContent_Declare(
        assimp
        SOURCE_DIR "${CMAKE_SOURCE_DIR}/vendor/assimp"
        BINARY_DIR "${CMAKE_BINARY_DIR}/vendor/assimp-build"
)

## yaml_cpp
FetchContent_Declare(
        yaml_cpp
        SOURCE_DIR "${CMAKE_SOURCE_DIR}/vendor/yaml-cpp"
        BINARY_DIR "${CMAKE_BINARY_DIR}/vendor/yaml-cpp-build"
)


## SOURCES

add_executable(${PROJECT_NAME}
        Pale/main.cpp

        Pale/Core/DeviceSelector.cpp
        Pale/Core/UUID.cpp
        Pale/Scene/Entity.cpp
        Pale/Scene/Scene.cpp
        Pale/Scene/SceneSerializer.cpp
        Pale/Renderer/PathTracer.cpp
        Pale/Renderer/Camera.cpp
        Pale/Utils/stb_image_impl.cpp
)


## Module files
target_sources(${PROJECT_NAME}  PUBLIC
        FILE_SET CXX_MODULES FILES
        Pale/Core/DeviceSelector.ixx
        Pale/Core/UUID.ixx
        Pale/Core/Log.ixx
        Pale/Scene/Scene.ixx
        Pale/Scene/Components.ixx
        Pale/Scene/Entity.ixx
        Pale/Scene/SceneSerializer.ixx
        Pale/Asset/AssetRegistry.ixx
        Pale/Asset/Assets.ixx
        Pale/Asset/AssetsCore.ixx
        Pale/Asset/AssetManager.ixx
        Pale/Asset/AssimpMeshLoader.ixx
        Pale/Asset/IAssetIndex.ixx
        Pale/Asset/IAssetAccess.ixx
        Pale/Asset/MaterialYamlLoader.ixx
        Pale/Asset/MeshAsset.ixx
        Pale/Asset/MaterialAsset.ixx
        Pale/Asset/MaterialBakerCommon.ixx
        Pale/Asset/MaterialBakerMitsuba.ixx
        Pale/Renderer/Camera.ixx
        Pale/Renderer/Framebuffer.ixx
        Pale/Renderer/PathTracer.ixx
        Pale/Renderer/Sensors.ixx
        Pale/Renderer/SceneGPU.ixx
        Pale/Renderer/PathTracerConfig.ixx
        Pale/Utils/ImageIO.ixx

)

# Link your exe to the module wrapper (must come after add_executable)
FetchContent_MakeAvailable(spdlog glm assimp yaml_cpp)
target_link_libraries(${PROJECT_NAME}
        PRIVATE
        pugixml
        EnTT::EnTT
        spdlog
        glm
        assimp
        yaml-cpp
)

# Link core directory
include_directories(${PROJECT_NAME} PRIVATE Pale)

# Link vendor directories
target_include_directories(${PROJECT_NAME} PRIVATE
        "vendor/yaml-cpp/include"
)
# STB_Image
include_directories(${PROJECT_NAME} PRIVATE vendor/stb_image)


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
