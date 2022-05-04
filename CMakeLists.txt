cmake_minimum_required(VERSION 3.0.0)
set (PROJECT OpenCL_test)
project(${PROJECT})

set(CMAKE_CXX_STANDARD 17)

set(SRCs main.cpp SparseMatrix.h SparseMatrix.cpp)
add_definitions(
        -DCL_HPP_TARGET_OPENCL_VERSION=210
        -DCL_HPP_ENABLE_EXCEPTIONS
)

add_executable(${PROJECT} ${SRCs})

# set(TCP_DIR ../TCP)
target_link_libraries(${PROJECT} PRIVATE OpenCL)
# target_include_directories(${PROJECT} PUBLIC ${TCP_DIR})

file(COPY kernels DESTINATION .)
file(COPY data DESTINATION .)
