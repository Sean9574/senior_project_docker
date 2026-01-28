cmake_minimum_required(VERSION 3.5)
project(ros2_numpy)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

ament_python_install_package(${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(rclpy REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)

  set(_pytest_tests
    test/test_pointclouds.py
    test/test_images.py
    test/test_occupancygrids.py
    test/test_geometry.py
    test/test_quat.py
    # Add other test files here
  )
  foreach(_test_path ${_pytest_tests})
    get_filename_component(_test_name ${_test_path} NAME_WE)
    ament_add_pytest_test(${_test_name} ${_test_path}
      APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}
      TIMEOUT 60
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
  endforeach()
endif()

##############
ament_export_dependencies(ament_cmake)
ament_export_dependencies(ament_cmake_python)
ament_package()
