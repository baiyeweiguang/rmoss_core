find_package(ament_cmake_gtest REQUIRED)

find_package(Eigen3 REQUIRED)

ament_add_gtest(test_kalman_filter test_kalman_filter.cpp)
target_link_libraries(test_kalman_filter ${PROJECT_NAME}::math)
target_include_directories(test_kalman_filter PUBLIC ${Eigen3_INCLUDE_DIRS})