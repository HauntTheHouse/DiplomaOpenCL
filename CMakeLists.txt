cmake_minimum_required(VERSION 3.0.0)
set (PROJECT OpenCL_test)
project(${PROJECT})

set(CMAKE_CXX_STANDARD 17)

set(SRCs main.cpp SparseMatrix.h SparseMatrix.cpp)
add_definitions(
        -DCL_HPP_TARGET_OPENCL_VERSION=210
        -DCL_HPP_ENABLE_EXCEPTIONS
)


#add_subdirectory("libs/OpenCL-CLHPP")
#add_subdirectory("libs/OpenCL-Headers")
#add_subdirectory("libs/OpenCL-ICD-Loader")

add_subdirectory(lib)

add_executable(${PROJECT} ${SRCs})
target_link_libraries(${PROJECT} PRIVATE OpenCL)
target_include_directories(${PROJECT} PUBLIC "lib/include")

#find_library(OpenCLICDLoader HINTS lib/share/cmake)

file(COPY kernels DESTINATION .)
file(COPY data DESTINATION .)
