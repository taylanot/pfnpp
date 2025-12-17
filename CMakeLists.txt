cmake_minimum_required(VERSION 3.18)
project(LibTorchExample LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Torch package
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

# Create executable
add_executable(main main.cpp)

# Link Torch
target_link_libraries(main "${TORCH_LIBRARIES}")

# Include Torch headers
target_include_directories(main PUBLIC ${TORCH_INCLUDE_DIRS})

# Enable CUDA for GPU if using CUDA-enabled LibTorch
set_property(TARGET main PROPERTY CXX_STANDARD 23)
