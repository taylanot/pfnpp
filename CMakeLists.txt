cmake_minimum_required(VERSION 3.18)
project(PFNPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

file(GLOB SOURCES "*.cpp")

foreach(SOURCE ${SOURCES})
    # Get filename without extension (e.g. "main" from "main.cpp")
    get_filename_component(EXEC_NAME ${SOURCE} NAME_WE)
    add_executable(${EXEC_NAME} ${SOURCE})
    target_link_libraries(${EXEC_NAME} "${TORCH_LIBRARIES}")
    target_include_directories(${EXEC_NAME} PUBLIC ${TORCH_INCLUDE_DIRS})
    # Enable CUDA for GPU if using CUDA-enabled LibTorch
    set_property(TARGET ${EXEC_NAME} PROPERTY CXX_STANDARD 23)


endforeach()


