#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "OpenCL::OpenCLUtilsCpp" for configuration "Release"
set_property(TARGET OpenCL::OpenCLUtilsCpp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(OpenCL::OpenCLUtilsCpp PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/OpenCLUtilsCpp.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS OpenCL::OpenCLUtilsCpp )
list(APPEND _IMPORT_CHECK_FILES_FOR_OpenCL::OpenCLUtilsCpp "${_IMPORT_PREFIX}/lib/OpenCLUtilsCpp.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
