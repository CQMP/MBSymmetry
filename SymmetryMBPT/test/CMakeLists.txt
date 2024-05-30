include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

include(GoogleTest)

set(test_src
        tests_main.cpp
        test_dyson.cpp
        test_cugw.cpp
        )

add_executable(runtests
        ${test_src}
        )
target_link_libraries(runtests PUBLIC gtest_main common-lib cuda-lib solver-lib scmbpt-lib cpu-lib)

if (TEST_XML_OUTPUT)
    add_test(NAME all_tests
            COMMAND runtests --gtest_output="xml:report.xml")
else ()
    gtest_discover_tests(runtests)
endif ()
