cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(StaticNet)

add_subdirectory(StaticNet)

add_executable(TestStaticNet
    "test.cpp"
)

target_link_libraries(TestStaticNet
    PRIVATE
        StaticNet
)

add_subdirectory(examples/mnist)
