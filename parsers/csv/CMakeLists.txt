cmake_minimum_required(VERSION 4.3)

project(csv_in_excel_to_parquet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Xlnt CONFIG REQUIRED)
find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)

add_executable(csv_in_excel_to_parquet main.cpp)

target_link_libraries(csv_in_excel_to_parquet PRIVATE
        xlnt::xlnt
        Arrow::arrow_shared
        Parquet::parquet_shared
)