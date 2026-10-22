cmake_minimum_required(VERSION 4.3)

project(Parse_json_airlines LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(nlohmann_json CONFIG REQUIRED)
find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)

add_executable(json_learning main.cpp)

target_link_libraries(json_learning PRIVATE
        nlohmann_json::nlohmann_json
        Arrow::arrow_shared
        Parquet::parquet_shared
)