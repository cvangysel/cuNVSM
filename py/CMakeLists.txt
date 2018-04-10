PROTOBUF_GENERATE_PYTHON(PROTOBUF_PY_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../proto/nvsm.proto)

add_custom_target(pynvsm
                  ALL cp -R ${CMAKE_CURRENT_SOURCE_DIR}/nvsm ${CMAKE_CURRENT_BINARY_DIR}
                  DEPENDS ${PROTOBUF_PY_SOURCE_FILES})

function(add_py_executable EXECUTABLE_NAME SCRIPT_NAME)
    add_custom_target(${EXECUTABLE_NAME}
                      ALL cd ${CMAKE_CURRENT_SOURCE_DIR} && cp ${SCRIPT_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE_NAME} && chmod +x ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE_NAME}
                      DEPENDS ${PROTOBUF_PY_SOURCE_FILES})
endfunction()

add_py_executable(cuNVSMQuery query.py)
add_py_executable(cuNVSMVisualize visualize.py)
add_py_executable(cuNVSMDumpVocabulary dump_vocabulary_list.py)