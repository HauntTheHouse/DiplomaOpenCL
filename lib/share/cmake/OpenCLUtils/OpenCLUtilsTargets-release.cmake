#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "OpenCL::OpenCLUtils" for configuration "Release"
set_property(TARGET OpenCL::OpenCLUtils APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(OpenCL::OpenCLUtils PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/OpenCLUtils.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS OpenCL::OpenCLUtils )
list(APPEND _IMPORT_CHECK_FILES_FOR_OpenCL::OpenCLUtils "${_IMPORT_PREFIX}/lib/OpenCLUtils.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
