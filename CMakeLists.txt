# CMakeLists.txt
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(vae LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/VAE)

add_library(vae MODULE VAE/vae.cu VAE/vae_pybind.cpp)
set_target_properties(vae PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(vae PROPERTIES PREFIX "")

target_link_libraries(vae PRIVATE pybind11::module)
