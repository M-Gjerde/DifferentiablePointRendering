cmake_minimum_required(VERSION 3.28)
project(DifferentiablePointRendering
        VERSION 0.0.1
        DESCRIPTION "A PBDR Application"
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # PIC everywhere
set(BUILD_SHARED_LIBS OFF)               # prefer static vendors we control

find_package(AdaptiveCpp REQUIRED)

find_package(OpenEXR REQUIRED)

# -------- Vendors --------
add_subdirectory(vendor/pugixml)
set(ENTT_DIR vendor/entt)
add_subdirectory(${ENTT_DIR})
set(TINY_PLY_DIR vendor/tinyply)
add_subdirectory(${TINY_PLY_DIR})
include(FetchContent)
set(SPDLOG_USE_STD_FORMAT ON)

FetchContent_Declare(spdlog SOURCE_DIR "${CMAKE_SOURCE_DIR}/vendor/spdlog" BINARY_DIR "${CMAKE_BINARY_DIR}/vendor/spdlog-build")
FetchContent_Declare(glm SOURCE_DIR "${CMAKE_SOURCE_DIR}/vendor/glm" BINARY_DIR "${CMAKE_BINARY_DIR}/vendor/glm-build")
# assimp
set(ASSIMP_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ZLIB ON CACHE BOOL "" FORCE)   # internal zlib
set(ASSIMP_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ZLIB ON CACHE BOOL "" FORCE)
# Avoid tools/tests to keep install/build minimal
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE) # critical: stop assimp installing anything

FetchContent_Declare(assimp
        SOURCE_DIR "${CMAKE_SOURCE_DIR}/vendor/assimp"
        BINARY_DIR "${CMAKE_BINARY_DIR}/vendor/assimp-build")

FetchContent_Declare(yaml_cpp SOURCE_DIR "${CMAKE_SOURCE_DIR}/vendor/yaml-cpp" BINARY_DIR "${CMAKE_BINARY_DIR}/vendor/yaml-cpp-build")
FetchContent_MakeAvailable(spdlog glm assimp yaml_cpp)

# -------- Engine library (all sources + modules + kernels) --------
set(KERNEL_SOURCES
        Pale/Renderer/Kernels/SyclBridge.cpp
        Pale/Renderer/Kernels/AdjointKernels.cpp
        Pale/Renderer/Kernels/PrimalKernels.cpp
        Pale/Renderer/Kernels/PostprocessKernel.cpp
)

set(ENGINE_SOURCES
        Pale/Core/DeviceSelector.cpp
        Pale/Core/ScopedTimer.cpp
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

add_library(PaleLib ${ENGINE_SOURCES})

# C++20 module interfaces belong to the library
target_sources(PaleLib
        PUBLIC
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
        Pale/Asset/PLYPointLoader.ixx
        Pale/Asset/IAssetIndex.ixx
        Pale/Asset/IAssetAccess.ixx
        Pale/Asset/MaterialYamlLoader.ixx
        Pale/Asset/MeshAsset.ixx
        Pale/Asset/PointAsset.ixx
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
        Pale/Utils/StringFormatting.ixx
)

# ---- PaleLib ----
target_compile_features(PaleLib PUBLIC cxx_std_20)
target_compile_options(PaleLib
        PUBLIC
        -fopenmp=libomp
        PRIVATE
        $<$<CONFIG:Debug>:-g -O0 -fno-inline -fno-omit-frame-pointer>
)

target_link_libraries(PaleLib
        PRIVATE
        pugixml
        EnTT::EnTT
        glm
        assimp
        yaml-cpp
        tinyply
        PUBLIC
        spdlog::spdlog_header_only
        OpenEXR::OpenEXR
)

target_include_directories(PaleLib
        PUBLIC
        ${CMAKE_SOURCE_DIR}/Pale
        vendor/entt/src
        vendor/glm
        PRIVATE
        vendor/yaml-cpp/include
        vendor/stb_image
        ${TINY_PLY_DIR}/source
)

# Enable SYCL for the engine library (device code lives here)
add_sycl_to_target(TARGET PaleLib)

# -------- Executable (main only) --------
add_executable(${PROJECT_NAME} Pale/main.cpp)
target_compile_options(${PROJECT_NAME}
        PRIVATE
        -fopenmp=libomp
        $<$<CONFIG:Debug>:-g -O0 -fno-inline -fno-omit-frame-pointer>
)
target_link_libraries(${PROJECT_NAME} PRIVATE PaleLib)
# If main.cpp itself submits SYCL (unlikely), also decorate the exe:
add_sycl_to_target(TARGET ${PROJECT_NAME})

# -------- Cleanup of old AdaptiveCpp caches --------
set(ACPP_APP_BASE_DIR "$ENV{HOME}/.acpp/apps")
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


# ---------- Toggle ----------
option(BUILD_PYBIND "Build the Python module (pybind11)" OFF)

# ---------- ... your existing CMakeLists up to cleanup.cmake ----------

# ===== Optional pybind11 module =====
if (BUILD_PYBIND)
    message(STATUS "BUILD_PYBIND=ON: building Python module 'pale'")

    # pybind11: prefer vendored, else FetchContent
    if (EXISTS "${CMAKE_SOURCE_DIR}/vendor/pybind11/CMakeLists.txt")
        add_subdirectory("${CMAKE_SOURCE_DIR}/vendor/pybind11")
    else ()
        include(FetchContent)
        FetchContent_Declare(pybind11
                GIT_REPOSITORY https://github.com/pybind/pybind11.git
                GIT_TAG v2.13.5
        )
        FetchContent_MakeAvailable(pybind11)
    endif ()

    # Use current Python (helps conda/venv)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

    # Module target
    add_library(pale MODULE Pale/pale_bindings.cpp)
    target_link_libraries(pale PRIVATE PaleLib pybind11::module)
    target_compile_features(pale PRIVATE cxx_std_20)
    target_compile_options(pale
            PRIVATE
            -fopenmp=libomp
            $<$<CONFIG:Debug>:-g -O0 -fno-inline -fno-omit-frame-pointer>
    )

    # If the bindings submit SYCL, decorate the module as well
    add_sycl_to_target(TARGET pale)

    # Produce 'pale.so' without 'lib' prefix or 'd' postfix
    set_target_properties(pale PROPERTIES
            PREFIX ""                # pale.so
            OUTPUT_NAME "pale"
            DEBUG_POSTFIX ""         # avoid paled.so
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN YES
    )

    # Install into the active interpreter's site-packages
    set(PYTHON_SITE "${Python3_SITEARCH}")
    if (NOT PYTHON_SITE)
        set(PYTHON_SITE "${Python3_SITELIB}")
    endif ()

    include(GNUInstallDirs)
    install(TARGETS pale
            LIBRARY DESTINATION "${PYTHON_SITE}"
            RUNTIME DESTINATION "${PYTHON_SITE}"
            ARCHIVE DESTINATION "${PYTHON_SITE}"
    )

    add_custom_target(install_pale_debug
            COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}" --config Debug
            DEPENDS pale)
else ()
    message(STATUS "BUILD_PYBIND=OFF: skipping Python module")
endif ()
