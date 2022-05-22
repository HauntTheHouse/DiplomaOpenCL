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
target_link_libraries(${PROJECT} PRIVATE ${OpenCL_LIBRARIES})
#target_include_directories(${PROJECT} PUBLIC ${OpenCL_INCLUDE_DIRS})
target_include_directories(${PROJECT} PUBLIC "libs/OpenCL-CLHPP/include" "libs/OpenCL-Headers")

file(COPY kernels DESTINATION .)
file(COPY data DESTINATION .)
