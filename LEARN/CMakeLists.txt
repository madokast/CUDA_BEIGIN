cmake_minimum_required(VERSION 3.17)
project(LEARN CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(LEARN main.cu)

set_target_properties(
    LEARN
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)