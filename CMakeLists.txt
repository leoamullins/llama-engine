cmake_minimum_required(VERSION 3.20)
project(llama_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # compile_commands.json for clangd

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(main src/main.cpp src/llama.cpp src/mapped_file.cpp)
target_include_directories(main PRIVATE include)
target_compile_options(main PRIVATE -Wall -Wextra)
