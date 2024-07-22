find_package(ament_cmake_gtest REQUIRED)

ament_add_gtest(test_kalman_filter test_kalman_filter.cpp)
target_link_libraries(test_kalman_filter ${PROJECT_NAME}::math)