cmake_minimum_required(VERSION 3.28)
project(PaleRealtimeViewer
        VERSION 1.0.0
        DESCRIPTION "ImGui realtime viewer for Pale renderer"
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF)

get_filename_component(PALE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(PALE_ROOT "${PALE_ROOT}" CACHE PATH "DifferentiablePointRendering repository root")

# Read literal training debug defaults without importing Python or its runtime
# dependencies. Fail visibly if these fields cease to be literals.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${PALE_ROOT}/python/config.py")
file(READ "${PALE_ROOT}/python/config.py" PALE_TRAINING_CONFIG)
foreach(field densification_grad_abs_min inactive_transport_min_score)
    string(REGEX MATCH "\n    ${field}: float = ([0-9.eE+-]+)[ \r\n#]" match "${PALE_TRAINING_CONFIG}")
    if(NOT match)
        message(FATAL_ERROR "Expected a numeric literal for ${field} in python/config.py")
    endif()
    set(${field} "${CMAKE_MATCH_1}")
endforeach()
string(REGEX MATCH "\n    target_color_space: str = \"([a-z]+)\"" match "${PALE_TRAINING_CONFIG}")
set(target_color_space "${CMAKE_MATCH_1}")
configure_file(TrainingDebugDefaults.h.in generated/TrainingDebugDefaults.h @ONLY)

find_package(AdaptiveCpp REQUIRED)
find_package(OpenEXR REQUIRED)
find_package(OpenGL REQUIRED)

include(FetchContent)
# Show download progress and avoid update checks on subsequent runs to reduce stalls
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
# Allow using FetchContent_Populate(imgui/imguizmo) with declared details under CMake 4.0+
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

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
            GIT_TAG docking
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE
            GIT_SUBMODULES "")
    # Populate only if not already fetched to avoid repeated network calls
    FetchContent_GetProperties(imgui)
    if (NOT imgui_POPULATED)
        FetchContent_Populate(imgui)
    endif ()
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
                GIT_TAG 3.4
                GIT_SHALLOW TRUE
                GIT_PROGRESS TRUE)
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
            GIT_TAG master
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE
            GIT_SUBMODULES "")
    # Populate only if not already fetched to avoid repeated network calls
    FetchContent_GetProperties(imguizmo)
    if (NOT imguizmo_POPULATED)
        FetchContent_Populate(imguizmo)
    endif ()
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
# The cached world-space overlap diagnostic parallelizes host footprint queries.
target_link_options(PaleRealtimeViewer PRIVATE -fopenmp=libomp)
target_compile_options(PaleRealtimeViewer
        PRIVATE
        -fopenmp=libomp
        $<$<CONFIG:Debug>:-g -O0 -fno-inline -fno-omit-frame-pointer>
        $<$<CONFIG:Release>:-O3>)
target_compile_definitions(PaleRealtimeViewer
        PRIVATE
        PALE_DEFAULT_ASSET_DIR=\"${PALE_ROOT}/Assets\")
target_include_directories(PaleRealtimeViewer
        PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
        "${PALE_ROOT}/vendor/stb_image")
target_link_libraries(PaleRealtimeViewer
        PRIVATE
        PaleLibRealtime
        yaml-cpp
        imgui
        imguizmo
        OpenGL::GL)
add_sycl_to_target(TARGET PaleRealtimeViewer)

# Explicitly built, headless forward-render benchmark (no UI or training updates).
#add_executable(PaleSplatEventsBenchmark EXCLUDE_FROM_ALL
#        "${PALE_ROOT}/python/test/benchmark_splat_events.cpp")
#target_link_libraries(PaleSplatEventsBenchmark PRIVATE PaleLibRealtime)
#add_sycl_to_target(TARGET PaleSplatEventsBenchmark)

# Explicit CPU-only launch/protocol/native-loader integration check; no GPU work.
add_executable(PaleViewerTrainingIntegration EXCLUDE_FROM_ALL
        "${PALE_ROOT}/test/test_viewer_training_integration.cpp")
target_compile_options(PaleViewerTrainingIntegration PRIVATE -UNDEBUG)
target_include_directories(PaleViewerTrainingIntegration PRIVATE "${PALE_ROOT}")
target_link_libraries(PaleViewerTrainingIntegration PRIVATE PaleLibRealtime yaml-cpp imgui)
add_sycl_to_target(TARGET PaleViewerTrainingIntegration)

# Viewport geometry regression; Dear ImGui only, with no display or SYCL work.
add_executable(PaleViewerViewportLayout EXCLUDE_FROM_ALL
        "${PALE_ROOT}/test/test_viewport_layout.cpp")
target_compile_options(PaleViewerViewportLayout PRIVATE -UNDEBUG)
target_include_directories(PaleViewerViewportLayout PRIVATE "${PALE_ROOT}")
target_link_libraries(PaleViewerViewportLayout PRIVATE imgui)

add_executable(PaleViewerLivePacketLease EXCLUDE_FROM_ALL
        "${PALE_ROOT}/test/test_viewer_live_packet_lease.cpp")
target_compile_options(PaleViewerLivePacketLease PRIVATE -UNDEBUG)
target_include_directories(PaleViewerLivePacketLease PRIVATE "${PALE_ROOT}")

add_executable(PaleViewerMetricsAxes EXCLUDE_FROM_ALL
        "${PALE_ROOT}/test/test_training_metrics_axes.cpp")
target_compile_options(PaleViewerMetricsAxes PRIVATE -UNDEBUG)
target_include_directories(PaleViewerMetricsAxes PRIVATE "${PALE_ROOT}")
target_link_libraries(PaleViewerMetricsAxes PRIVATE imgui)

foreach(check IN ITEMS responsive_work preview_recovery)
    add_executable(PaleViewer_${check} EXCLUDE_FROM_ALL "${PALE_ROOT}/test/test_viewer_${check}.cpp")
    target_compile_options(PaleViewer_${check} PRIVATE -UNDEBUG)
    target_include_directories(PaleViewer_${check} PRIVATE "${PALE_ROOT}")
endforeach()
add_executable(PaleViewerDeviceAllocation EXCLUDE_FROM_ALL "${PALE_ROOT}/test/test_device_allocation.cpp")
target_compile_options(PaleViewerDeviceAllocation PRIVATE -UNDEBUG)
target_include_directories(PaleViewerDeviceAllocation PRIVATE "${PALE_ROOT}")
add_executable(PaleViewerMeshExport EXCLUDE_FROM_ALL "${PALE_ROOT}/test/test_viewer_mesh_export_controls.cpp")
target_compile_options(PaleViewerMeshExport PRIVATE -UNDEBUG)
target_include_directories(PaleViewerMeshExport PRIVATE "${PALE_ROOT}")
target_link_libraries(PaleViewerMeshExport PRIVATE imgui yaml-cpp)
add_executable(PaleViewerLogScroll EXCLUDE_FROM_ALL "${PALE_ROOT}/test/test_viewer_log_scroll.cpp")
target_compile_options(PaleViewerLogScroll PRIVATE -UNDEBUG)
target_include_directories(PaleViewerLogScroll PRIVATE "${PALE_ROOT}")
target_link_libraries(PaleViewerLogScroll PRIVATE imgui yaml-cpp)

# Optional UI verification without a display server or reconstruction device.
find_library(PALE_VIEWER_EGL EGL)
if(PALE_VIEWER_EGL)
    add_executable(PaleViewerTexture EXCLUDE_FROM_ALL "${PALE_ROOT}/test/test_viewport_texture.cpp")
    target_compile_options(PaleViewerTexture PRIVATE -UNDEBUG)
    target_include_directories(PaleViewerTexture PRIVATE "${PALE_ROOT}")
    target_link_libraries(PaleViewerTexture PRIVATE OpenGL::GL "${PALE_VIEWER_EGL}")
    add_executable(PaleViewerWorkspacePreview EXCLUDE_FROM_ALL "${PALE_ROOT}/test/viewer_workspace_preview.cpp")
    target_compile_options(PaleViewerWorkspacePreview PRIVATE -UNDEBUG)
    target_include_directories(PaleViewerWorkspacePreview PRIVATE "${PALE_ROOT}" "${PALE_ROOT}/vendor/stb_image")
    target_link_libraries(PaleViewerWorkspacePreview PRIVATE yaml-cpp imgui OpenGL::GL "${PALE_VIEWER_EGL}")
endif()
