cmake_minimum_required(VERSION 4.0.0)
project(DifferentiablePointRendering
        VERSION 0.0.1
        DESCRIPTION "A PBDR Application"
        LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

find_package(AdaptiveCpp REQUIRED)

add_executable(${PROJECT_NAME})

# Adding Vendor Libraries
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

# SOURCES
set(KERNEL_SOURCES
        Pale/Renderer/Kernels/SyclBridge.cpp
)


target_sources(${PROJECT_NAME} PRIVATE
        Pale/main.cpp
        Pale/Core/DeviceSelector.cpp
        Pale/Core/UUID.cpp
        Pale/Scene/Entity.cpp
        Pale/Scene/Scene.cpp
        Pale/Scene/SceneSerializer.cpp
        Pale/Renderer/PathTracer.cpp
        Pale/Renderer/Camera.cpp
        Pale/Renderer/SceneBuild.cpp
        Pale/Utils/stb_image_impl.cpp
        ${KERNEL_SOURCES}
)

## Module files
target_sources(${PROJECT_NAME}
        PUBLIC FILE_SET CXX_MODULES FILES
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
        Pale/Renderer/SceneBuild.ixx
        Pale/Renderer/SceneUpload.ixx
        Pale/Renderer/BVH.ixx
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


#set(ACPP_TARGETS "omp.library-only" CACHE STRING "")
#set(ACPP_USE_ACCELERATED_CPU OFF CACHE BOOL "")
target_compile_options(${PROJECT_NAME} PRIVATE -g -O0)


# Add SYCL to target via AdaptiveCpp's helper
target_compile_options(${PROJECT_NAME} PRIVATE -fopenmp=libomp)  # or -fno-openmp

add_sycl_to_target(TARGET ${PROJECT_NAME})

# Cleanup old SYCL kernels in case a recompilation doesn't trigger a replacement
set(ACPP_APP_BASE_DIR "$ENV{HOME}/.acpp/apps")
# Script that cleans up old dirs
set(CLEAN_SCRIPT "${CMAKE_BINARY_DIR}/cleanup.cmake")
file(WRITE "${CLEAN_SCRIPT}" "
file(GLOB PROJECT_DIRS \"${ACPP_APP_BASE_DIR}/${PROJECT_NAME}*\")
foreach(DIR IN LISTS PROJECT_DIRS)
if(IS_DIRECTORY \"\${DIR}\")
    message(STATUS \"Found \${DIR}\")
    file(REMOVE_RECURSE \"\${DIR}\")
    message(STATUS \"Removed \${DIR}\")
endif()
endforeach()
")
# Define a target that runs the script
add_custom_target(clean_acpp ALL
        COMMAND ${CMAKE_COMMAND} -P "${CLEAN_SCRIPT}"
        COMMENT "Cleaning AdaptiveCpp app dirs"
)
# Ensure it runs before your executable
add_dependencies(${PROJECT_NAME} clean_acpp)
