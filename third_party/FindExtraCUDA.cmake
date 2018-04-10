include(FindPackageHandleStandardArgs)

find_library(CUDA_NVTX_LIBRARY nvToolsExt
             PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

find_package_handle_standard_args(
    ExtraCUDA
    DEFAULT_MSG
    CUDA_NVTX_LIBRARY)

if(ExtraCUDA_FOUND)
  set(EXTRACUDA_LIBRARIES ${CUDA_NVTX_LIBRARY})
  message(STATUS "Found Extra CUDA libraries    (library: ${EXTRACUDA_LIBRARIES})")
endif()