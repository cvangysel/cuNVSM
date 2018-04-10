cmake_minimum_required (VERSION 3.8)
project (cuNVSM)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/third_party)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Boost 1.65.1 REQUIRED)
find_package(CUDA 8.0 REQUIRED)
find_package(CUDNN 5.1 REQUIRED)
find_package(ExtraCUDA REQUIRED) # nvToolsExt
find_package(Glog 0.3.4 REQUIRED)
find_package(HDF5 1.6.10 REQUIRED COMPONENTS CXX)
find_package(Indri 5.11 REQUIRED)
find_package(PerfTools 2.5 REQUIRED) # Google PerfTools
find_package(Protobuf 3.5.1 REQUIRED)

function(add_external_src TARGET_NAME MODULE_NAME)
    if(NOT TARGET ${TARGET_NAME})
        configure_file(third_party/${MODULE_NAME}-CMakeLists.txt
                       ${MODULE_NAME}-download/CMakeLists.txt)
        execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}-download )
        execute_process(COMMAND ${CMAKE_COMMAND} --build .
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}-download )

        # Prevent GoogleTest from overriding our compiler/linker options
        # when building with Visual Studio
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

        add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}-src
                         ${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}-build
                         EXCLUDE_FROM_ALL)
    endif(NOT TARGET ${TARGET_NAME})
endfunction()

add_external_src(gtest googletest)
add_external_src(gflags gflags)
add_external_src(device_matrix device_matrix)

set(GFLAGS_INCLUDE_DIRS
    ${CMAKE_CURRENT_BINARY_DIR}/gflags-build/include)
set(DEVICE_MATRIX_INCLUDE_DIRS
    ${CMAKE_CURRENT_BINARY_DIR}/device_matrix-src/include)

set(LIBRARIES
    -lstdc++
    cnmem
    gflags-static
    ${Boost_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_cudadevrt_LIBRARY}
    ${CUDNN_LIBRARIES}
    ${EXTRACUDA_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${PERFTOOLS_LIBRARIES}
    ${PROTOBUF_LIBRARY}
    ${INDRI_LIBRARIES}
)

set(EXEC_LIBRARIES device_matrix ${LIBRARIES})

set(TEST_LIBRARIES
    gtest_main gmock device_matrix_debug ${LIBRARIES})

set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3 -funroll-loops")

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -arch=sm_52 -use_fast_math -O3 -default-stream per-thread)

SET(INCLUDE_DIRS
    "${gtest_SOURCE_DIR}/include"
    "${gmock_SOURCE_DIR}/include"
    "${cnmem_SOURCE_DIR}/include"
    ${Boost_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
    ${CUDNN_INCLUDE_DIRS}
    ${DEVICE_MATRIX_INCLUDE_DIRS}
    ${GFLAGS_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${INDRI_INCLUDE_DIRS}
    ${PERFTOOLS_INCLUDE_DIRS}
)

foreach(INCLUDE_DIR ${INCLUDE_DIRS})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -isystem ${INCLUDE_DIR})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${INCLUDE_DIR}")
endforeach()

enable_testing()

subdirs(cpp py)
