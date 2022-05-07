project(OpenCL)

add_library(OpenCL bin/OpenCL.dll)
set_target_properties(OpenCL PROPERTIES LINKER_LANGUAGE C)