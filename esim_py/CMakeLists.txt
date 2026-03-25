cmake_minimum_required(VERSION 3.10)
project(esim_py)

find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
# filesystem only (Boost 1.90+ Homebrew: no separate boost_system CMake package)
find_package(Boost REQUIRED COMPONENTS filesystem)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 11)

add_library(libesim STATIC src/esim.cpp)
target_include_directories(libesim PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(libesim PUBLIC Eigen3::Eigen Boost::filesystem ${OpenCV_LIBS})

pybind11_add_module(esim_py src/bindings.cpp)

target_link_libraries(esim_py PRIVATE libesim pybind11::embed)
