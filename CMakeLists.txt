cmake_minimum_required(VERSION 3.18)
project(DPE_MVS_REFACTORED LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(CUDAToolkit REQUIRED)

option(DPE_DEBUG_SYNC "Synchronize after each CUDA kernel" OFF)
option(DPE_DEBUG_DUMP "Write intermediate debug results" OFF)

add_executable(DPE
    src/main.cpp
    src/common/io.cpp
    src/scene/scene.cpp
    src/scene/reconstruction_state.cpp
    src/preprocessing/edge_detection.cpp
    src/runtime/cuda_context.cu
    src/runtime/gpu_workspace.cu
    src/runtime/gpu_scene.cu
    src/dpe/dpe.cpp
    src/dpe/dpe.cu
    src/pipeline/dpe_pipeline.cpp
    src/fusion/fusion.cpp
)

target_include_directories(DPE PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(DPE PRIVATE
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    CUDA::cudart
)

target_compile_options(DPE PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=128>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

if(DPE_DEBUG_SYNC)
    target_compile_definitions(DPE PRIVATE DPE_DEBUG_SYNC=1)
endif()
if(DPE_DEBUG_DUMP)
    target_compile_definitions(DPE PRIVATE DPE_DEBUG_DUMP=1)
endif()

set(DPE_CUDA_ARCH "86" CACHE STRING "CUDA architecture, e.g. 75, 86, 89")
set_target_properties(DPE PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "${DPE_CUDA_ARCH}"
)
