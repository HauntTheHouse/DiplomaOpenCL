cmake_minimum_required(VERSION 3.0.0)
set (PROJECT DiplomaOpenCL)
project(${PROJECT})

set(CMAKE_CXX_STANDARD 17)

file(GLOB SRCS
     "src/*.cpp"
     "src/*.h"
)

find_package(OpenCL REQUIRED)

add_executable(${PROJECT} ${SRCS})

add_definitions(-DVIENNACL_WITH_OPENCL)

set(VIENNACL_SRC_PATH "libs/viennacl/libviennacl/src")
add_library(viennacl SHARED ${VIENNACL_SRC_PATH}/backend.cpp
                            ${VIENNACL_SRC_PATH}/blas1.cpp ${VIENNACL_SRC_PATH}/blas1_host.cpp ${VIENNACL_SRC_PATH}/blas1_opencl.cpp
                            ${VIENNACL_SRC_PATH}/blas2.cpp ${VIENNACL_SRC_PATH}/blas2_host.cpp ${VIENNACL_SRC_PATH}/blas2_opencl.cpp
                            ${VIENNACL_SRC_PATH}/blas3.cpp ${VIENNACL_SRC_PATH}/blas3_host.cpp ${VIENNACL_SRC_PATH}/blas3_opencl.cpp)
target_link_libraries(viennacl PUBLIC ${OpenCL_LIBRARIES})

target_include_directories(viennacl PUBLIC
    "libs/viennacl/libviennacl/include"
    "libs/OpenCL-Headers"
    "libs/viennacl"
)

target_link_libraries(${PROJECT} PRIVATE
    viennacl
)

target_include_directories(${PROJECT} PUBLIC
    "libs/OpenCL-CLHPP/include"
    "libs/OpenCL-Headers"
    "libs/viennacl"
    "libs/viennacl/libviennacl/include"
)

file(COPY kernels DESTINATION .)
file(COPY data DESTINATION .)
