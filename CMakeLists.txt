cmake_minimum_required(VERSION 3.4.3)
project(DLTuner CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(LLVM_BUILD_TOOLS ON)

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
add_definitions(-fno-rtti)

# Now build our tools
add_executable(dlt dlt.cpp Transform.cpp Layout.cpp Evaluate.cpp)

# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs support irreader bitwriter bitreader DataStructure ipo)

# Link against LLVM libraries
target_link_libraries(dlt ${llvm_libs})
