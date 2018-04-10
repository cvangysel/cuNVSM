protobuf_generate_cpp(PROTOBUF_CPP_SOURCE_FILES PROTOBUF_HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../proto/nvsm.proto)
add_library(nvsm_proto ${PROTOBUF_CPP_SOURCE_FILES} ${PROTOBUF_HEADER_FILES})
target_link_libraries(nvsm_proto ${PROTOBUF_LIBRARY})

SET(CUNVSM_SRC_FILES
    model.cu
    objective.cu
    labels.cu intermediate_results.cu gradient_check.cu
    params.cu storage.cu
    updates.cu updates_adagrad.cu updates_adam.cu
    data.cu data_indri.cpp data_async.cpp data_multi.cpp data_repeating.cpp
    hdf5.cu cuda_utils.cu cudnn_utils.cu)

include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR})
include_directories(BEFORE ${cuNVSM_SOURCE_DIR}/include)

SET(RELEASE_OPTS -DFLOATING_POINT_TYPE=float32 -DNDEBUG=1)
SET(DEBUG_OPTS -DFLOATING_POINT_TYPE=float64 -UNDEBUG)

cuda_add_library(nvsm ${CUNVSM_SRC_FILES} OPTIONS ${RELEASE_OPTS})
target_compile_options(nvsm PUBLIC ${RELEASE_OPTS})
set_target_properties(nvsm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_dependencies(nvsm nvsm_proto)
target_link_libraries(nvsm nvsm_proto)

cuda_add_library(nvsm_debug ${CUNVSM_SRC_FILES} OPTIONS ${DEBUG_OPTS})
target_compile_options(nvsm_debug PUBLIC ${DEBUG_OPTS})
set_target_properties(nvsm_debug PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_dependencies(nvsm_debug nvsm_proto)
target_link_libraries(nvsm_debug nvsm_proto)

cuda_add_executable(cuNVSMTrainModel main.cu OPTIONS ${RELEASE_OPTS})
target_link_libraries(cuNVSMTrainModel nvsm nvsm_proto ${EXEC_LIBRARIES})
set_target_properties(cuNVSMTrainModel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(model_tests model_tests.cu OPTIONS ${DEBUG_OPTS})
target_link_libraries(model_tests nvsm_debug nvsm_proto ${TEST_LIBRARIES})
add_test(NAME model_tests COMMAND model_tests)
set_target_properties(model_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(gradient_checking_tests gradient_checking_tests.cu OPTIONS ${DEBUG_OPTS})
target_link_libraries(gradient_checking_tests nvsm_debug nvsm_proto ${TEST_LIBRARIES})
add_test(NAME gradient_checking_tests COMMAND gradient_checking_tests)
set_target_properties(gradient_checking_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(updates_tests updates_tests.cu OPTIONS ${DEBUG_OPTS})
target_link_libraries(updates_tests nvsm_debug nvsm_proto ${TEST_LIBRARIES})
add_test(NAME updates_tests COMMAND updates_tests)
set_target_properties(updates_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(cuda_utils_tests cuda_utils_tests.cu OPTIONS ${DEBUG_OPTS})
target_link_libraries(cuda_utils_tests nvsm_debug nvsm_proto ${TEST_LIBRARIES})
add_test(NAME cuda_utils_tests COMMAND cuda_utils_tests)
set_target_properties(cuda_utils_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(intermediate_results_tests intermediate_results_tests.cu OPTIONS ${DEBUG_OPTS})
target_link_libraries(intermediate_results_tests nvsm_debug nvsm_proto ${TEST_LIBRARIES})
add_test(NAME intermediate_results_tests COMMAND intermediate_results_tests)
set_target_properties(intermediate_results_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(cudnn_utils_tests cudnn_utils_tests.cu OPTIONS ${DEBUG_OPTS})
target_link_libraries(cudnn_utils_tests nvsm_debug nvsm_proto ${TEST_LIBRARIES})
add_test(NAME cudnn_utils_tests COMMAND cudnn_utils_tests)
set_target_properties(cudnn_utils_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(data_tests data_tests.cpp OPTIONS ${DEBUG_OPTS})
target_link_libraries(data_tests nvsm_debug nvsm_proto ${TEST_LIBRARIES})
add_test(NAME data_tests COMMAND data_tests)
set_target_properties(data_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(utils_tests utils_tests.cpp OPTIONS ${DEBUG_OPTS})
target_link_libraries(utils_tests nvsm_debug nvsm_proto ${TEST_LIBRARIES})
add_test(NAME utils_tests COMMAND utils_tests)
set_target_properties(utils_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
