include(FindPackageHandleStandardArgs)

find_path(INDRI_INCLUDE_DIR indri/DiskIndex.hpp)
find_library(INDRI_LIBRARY NAMES indri)
find_library(Z_LIBRARY NAMES z)
find_library(M_LIBRARY NAMES m)

find_package_handle_standard_args(
    Indri
    DEFAULT_MSG
    INDRI_INCLUDE_DIR
    INDRI_LIBRARY
    Z_LIBRARY
    M_LIBRARY)

if(INDRI_FOUND)
  set(INDRI_INCLUDE_DIRS ${INDRI_INCLUDE_DIR})
  set(INDRI_LIBRARIES ${INDRI_LIBRARY} ${Z_LIBRARY} ${M_LIBRARY})
  message(STATUS "Found Indri    (include: ${INDRI_INCLUDE_DIR}, library: ${INDRI_LIBRARIES})")
endif()