llama_add_compile_flags()

function(advanced_quantizer_test source)
    set(TEST_SOURCES ${source} ${ARGN})
    get_filename_component(TEST_TARGET ${source} NAME_WE)

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE llama llama-common)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()

    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS quantizer)
endfunction()

function(advanced_quantizer_build source)
    set(TEST_SOURCES ${source} ${ARGN})
    get_filename_component(TEST_TARGET ${source} NAME_WE)

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE llama llama-common)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

advanced_quantizer_test(test-quantize-fns.cpp)
advanced_quantizer_test(test-quantize-perf.cpp)
if (TARGET ggml-cuda)
    advanced_quantizer_test(test-quantize-cuda-classic.cpp)
    target_link_libraries(test-quantize-cuda-classic PRIVATE ggml-cuda)
endif()

# Build-only helper: requires a caller-supplied GGUF model path for useful runs.
advanced_quantizer_build(test-quantize-stats.cpp)
