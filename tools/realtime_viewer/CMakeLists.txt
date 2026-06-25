cmake_minimum_required(VERSION 3.28)
project(PaleRealtimeViewer
        VERSION 0.1.0
        DESCRIPTION "ImGui realtime viewer for Pale renderer"
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF)

get_filename_component(PALE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(PALE_ROOT "${PALE_ROOT}" CACHE PATH "DifferentiablePointRendering repository root")

find_package(AdaptiveCpp REQUIRED)
find_package(OpenEXR REQUIRED)
find_package(OpenGL REQUIRED)

include(FetchContent)

set(SPDLOG_USE_STD_FORMAT ON)

add_subdirectory("${PALE_ROOT}/vendor/pugixml" "${CMAKE_BINARY_DIR}/vendor/pugixml-build")
add_subdirectory("${PALE_ROOT}/vendor/entt" "${CMAKE_BINARY_DIR}/vendor/entt-build")
add_subdirectory("${PALE_ROOT}/vendor/tinyply" "${CMAKE_BINARY_DIR}/vendor/tinyply-build")

set(ASSIMP_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ZLIB ON CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(spdlog
        SOURCE_DIR "${PALE_ROOT}/vendor/spdlog"
        BINARY_DIR "${CMAKE_BINARY_DIR}/vendor/spdlog-build")
FetchContent_Declare(glm
        SOURCE_DIR "${PALE_ROOT}/vendor/glm"
        BINARY_DIR "${CMAKE_BINARY_DIR}/vendor/glm-build")
FetchContent_Declare(assimp
        SOURCE_DIR "${PALE_ROOT}/vendor/assimp"
        BINARY_DIR "${CMAKE_BINARY_DIR}/vendor/assimp-build")
FetchContent_Declare(yaml_cpp
        SOURCE_DIR "${PALE_ROOT}/vendor/yaml-cpp"
        BINARY_DIR "${CMAKE_BINARY_DIR}/vendor/yaml-cpp-build")

FetchContent_MakeAvailable(spdlog glm assimp yaml_cpp)

option(REALTIME_VIEWER_FETCH_UI_DEPS "Download Dear ImGui and GLFW when source dirs are not provided" ON)
set(IMGUI_SOURCE_DIR "" CACHE PATH "Dear ImGui source directory")
set(IMGUIZMO_SOURCE_DIR "" CACHE PATH "ImGuizmo source directory")
set(GLFW_SOURCE_DIR "" CACHE PATH "GLFW source directory")

if (NOT IMGUI_SOURCE_DIR)
    if (NOT REALTIME_VIEWER_FETCH_UI_DEPS)
        message(FATAL_ERROR "Set IMGUI_SOURCE_DIR or enable REALTIME_VIEWER_FETCH_UI_DEPS")
    endif ()
    FetchContent_Declare(imgui
            GIT_REPOSITORY https://github.com/ocornut/imgui.git
            GIT_TAG v1.90.9)
    FetchContent_Populate(imgui)
    set(IMGUI_SOURCE_DIR "${imgui_SOURCE_DIR}")
endif ()

set(PALE_REALTIME_GLFW_TARGET "")

if (GLFW_SOURCE_DIR)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory("${GLFW_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/vendor/glfw-build")
    set(PALE_REALTIME_GLFW_TARGET glfw)
else ()
    find_package(glfw3 CONFIG QUIET)
    if (TARGET glfw)
        set(PALE_REALTIME_GLFW_TARGET glfw)
    elseif (TARGET glfw3)
        set(PALE_REALTIME_GLFW_TARGET glfw3)
    else ()
        find_package(PkgConfig QUIET)
        if (PkgConfig_FOUND)
            pkg_check_modules(GLFW3 QUIET IMPORTED_TARGET glfw3)
            if (TARGET PkgConfig::GLFW3)
                set(PALE_REALTIME_GLFW_TARGET PkgConfig::GLFW3)
            endif ()
        endif ()
    endif ()

    if (NOT PALE_REALTIME_GLFW_TARGET)
        if (NOT REALTIME_VIEWER_FETCH_UI_DEPS)
            message(FATAL_ERROR "Set GLFW_SOURCE_DIR, install glfw3, or enable REALTIME_VIEWER_FETCH_UI_DEPS")
        endif ()
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(glfw
                GIT_REPOSITORY https://github.com/glfw/glfw.git
                GIT_TAG 3.4)
        FetchContent_MakeAvailable(glfw)
        set(PALE_REALTIME_GLFW_TARGET glfw)
    endif ()
endif ()

add_library(imgui STATIC
        "${IMGUI_SOURCE_DIR}/imgui.cpp"
        "${IMGUI_SOURCE_DIR}/imgui_draw.cpp"
        "${IMGUI_SOURCE_DIR}/imgui_tables.cpp"
        "${IMGUI_SOURCE_DIR}/imgui_widgets.cpp"
        "${IMGUI_SOURCE_DIR}/backends/imgui_impl_glfw.cpp"
        "${IMGUI_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp")
target_include_directories(imgui
        PUBLIC
        "${IMGUI_SOURCE_DIR}"
        "${IMGUI_SOURCE_DIR}/backends")
target_link_libraries(imgui PUBLIC ${PALE_REALTIME_GLFW_TARGET} OpenGL::GL)

if (NOT IMGUIZMO_SOURCE_DIR)
    if (NOT REALTIME_VIEWER_FETCH_UI_DEPS)
        message(FATAL_ERROR "Set IMGUIZMO_SOURCE_DIR or enable REALTIME_VIEWER_FETCH_UI_DEPS")
    endif ()
    FetchContent_Declare(imguizmo
            GIT_REPOSITORY https://github.com/CedricGuillemet/ImGuizmo.git
            GIT_TAG master)
    FetchContent_Populate(imguizmo)
    set(IMGUIZMO_SOURCE_DIR "${imguizmo_SOURCE_DIR}")
endif ()

set(IMGUIZMO_CPP "${IMGUIZMO_SOURCE_DIR}/ImGuizmo.cpp")
set(IMGUIZMO_INCLUDE_DIR "${IMGUIZMO_SOURCE_DIR}")
if (NOT EXISTS "${IMGUIZMO_CPP}")
    set(IMGUIZMO_CPP "${IMGUIZMO_SOURCE_DIR}/src/ImGuizmo.cpp")
    set(IMGUIZMO_INCLUDE_DIR "${IMGUIZMO_SOURCE_DIR}/src")
endif ()
if (NOT EXISTS "${IMGUIZMO_CPP}")
    message(FATAL_ERROR "Could not find ImGuizmo.cpp in ${IMGUIZMO_SOURCE_DIR} or ${IMGUIZMO_SOURCE_DIR}/src")
endif ()

add_library(imguizmo STATIC
        "${IMGUIZMO_CPP}")
target_include_directories(imguizmo
        PUBLIC
        "${IMGUIZMO_INCLUDE_DIR}")
target_link_libraries(imguizmo PUBLIC imgui)

set(KERNEL_SOURCES
        "${PALE_ROOT}/Pale/Renderer/Kernels/SyclBridge.cpp"
        "${PALE_ROOT}/Pale/Renderer/Kernels/AdjointKernels.cpp"
        "${PALE_ROOT}/Pale/Renderer/Kernels/PrimalKernels.cpp"
        "${PALE_ROOT}/Pale/Renderer/Kernels/PostprocessKernel.cpp")

set(ENGINE_SOURCES
        "${PALE_ROOT}/Pale/Core/DeviceSelector.cpp"
        "${PALE_ROOT}/Pale/Core/ScopedTimer.cpp"
        "${PALE_ROOT}/Pale/Core/UUID.cpp"
        "${PALE_ROOT}/Pale/Scene/Entity.cpp"
        "${PALE_ROOT}/Pale/Scene/Scene.cpp"
        "${PALE_ROOT}/Pale/Scene/SceneSerializer.cpp"
        "${PALE_ROOT}/Pale/Renderer/PathTracer.cpp"
        "${PALE_ROOT}/Pale/Renderer/Camera.cpp"
        "${PALE_ROOT}/Pale/Renderer/SceneBuild.cpp"
        "${PALE_ROOT}/Pale/Utils/stb_image_impl.cpp"
        ${KERNEL_SOURCES})

add_library(PaleLibRealtime ${ENGINE_SOURCES})
target_sources(PaleLibRealtime
        PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS "${PALE_ROOT}"
        FILES
        "${PALE_ROOT}/Pale/Core/DeviceSelector.ixx"
        "${PALE_ROOT}/Pale/Core/UUID.ixx"
        "${PALE_ROOT}/Pale/Core/Log.ixx"
        "${PALE_ROOT}/Pale/Scene/Scene.ixx"
        "${PALE_ROOT}/Pale/Scene/Components.ixx"
        "${PALE_ROOT}/Pale/Scene/Entity.ixx"
        "${PALE_ROOT}/Pale/Scene/SceneSerializer.ixx"
        "${PALE_ROOT}/Pale/Asset/AssetRegistry.ixx"
        "${PALE_ROOT}/Pale/Asset/Assets.ixx"
        "${PALE_ROOT}/Pale/Asset/AssetsCore.ixx"
        "${PALE_ROOT}/Pale/Asset/AssetManager.ixx"
        "${PALE_ROOT}/Pale/Asset/AssimpMeshLoader.ixx"
        "${PALE_ROOT}/Pale/Asset/PLYPointLoader.ixx"
        "${PALE_ROOT}/Pale/Asset/IAssetIndex.ixx"
        "${PALE_ROOT}/Pale/Asset/IAssetAccess.ixx"
        "${PALE_ROOT}/Pale/Asset/MaterialYamlLoader.ixx"
        "${PALE_ROOT}/Pale/Asset/MeshAsset.ixx"
        "${PALE_ROOT}/Pale/Asset/PointAsset.ixx"
        "${PALE_ROOT}/Pale/Asset/MaterialAsset.ixx"
        "${PALE_ROOT}/Pale/Asset/MaterialBakerCommon.ixx"
        "${PALE_ROOT}/Pale/Asset/MaterialBakerMitsuba.ixx"
        "${PALE_ROOT}/Pale/Renderer/Camera.ixx"
        "${PALE_ROOT}/Pale/Renderer/Framebuffer.ixx"
        "${PALE_ROOT}/Pale/Renderer/PathTracer.ixx"
        "${PALE_ROOT}/Pale/Renderer/Sensors.ixx"
        "${PALE_ROOT}/Pale/Renderer/SceneBuild.ixx"
        "${PALE_ROOT}/Pale/Renderer/SceneUpload.ixx"
        "${PALE_ROOT}/Pale/Renderer/BVH.ixx"
        "${PALE_ROOT}/Pale/Utils/ImageIO.ixx"
        "${PALE_ROOT}/Pale/Utils/StringFormatting.ixx")

target_compile_features(PaleLibRealtime PUBLIC cxx_std_20)
target_compile_options(PaleLibRealtime
        PUBLIC
        -fopenmp=libomp
        PRIVATE
        $<$<CONFIG:Debug>:-g -O0 -fno-inline -fno-omit-frame-pointer>
        $<$<CONFIG:Release>:-O3>)
target_link_libraries(PaleLibRealtime
        PRIVATE
        pugixml
        EnTT::EnTT
        glm
        assimp
        yaml-cpp
        tinyply
        PUBLIC
        spdlog::spdlog_header_only
        OpenEXR::OpenEXR)
target_include_directories(PaleLibRealtime
        PUBLIC
        "${PALE_ROOT}/Pale"
        "${PALE_ROOT}/vendor/entt/src"
        "${PALE_ROOT}/vendor/glm"
        PRIVATE
        "${PALE_ROOT}/vendor/yaml-cpp/include"
        "${PALE_ROOT}/vendor/stb_image"
        "${PALE_ROOT}/vendor/tinyply/source")
add_sycl_to_target(TARGET PaleLibRealtime)

add_executable(PaleRealtimeViewer main.cpp)
target_compile_options(PaleRealtimeViewer
        PRIVATE
        -fopenmp=libomp
        $<$<CONFIG:Debug>:-g -O0 -fno-inline -fno-omit-frame-pointer>
        $<$<CONFIG:Release>:-O3>)
target_compile_definitions(PaleRealtimeViewer
        PRIVATE
        PALE_DEFAULT_ASSET_DIR=\"${PALE_ROOT}/Assets\")
target_link_libraries(PaleRealtimeViewer
        PRIVATE
        PaleLibRealtime
        imgui
        imguizmo
        OpenGL::GL)
add_sycl_to_target(TARGET PaleRealtimeViewer)
