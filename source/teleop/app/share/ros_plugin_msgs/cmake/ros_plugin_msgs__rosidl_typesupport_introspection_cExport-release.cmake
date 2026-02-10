#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ros_plugin_msgs::ros_plugin_msgs__rosidl_typesupport_introspection_c" for configuration "Release"
set_property(TARGET ros_plugin_msgs::ros_plugin_msgs__rosidl_typesupport_introspection_c APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ros_plugin_msgs::ros_plugin_msgs__rosidl_typesupport_introspection_c PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libros_plugin_msgs__rosidl_typesupport_introspection_c.so"
  IMPORTED_SONAME_RELEASE "libros_plugin_msgs__rosidl_typesupport_introspection_c.so"
  )

list(APPEND _cmake_import_check_targets ros_plugin_msgs::ros_plugin_msgs__rosidl_typesupport_introspection_c )
list(APPEND _cmake_import_check_files_for_ros_plugin_msgs::ros_plugin_msgs__rosidl_typesupport_introspection_c "${_IMPORT_PREFIX}/lib/libros_plugin_msgs__rosidl_typesupport_introspection_c.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
