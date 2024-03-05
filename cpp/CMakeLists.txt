cmake_minimum_required(VERSION 3.1...3.27)

project(
  DynamicSketch
  VERSION 1.0
  LANGUAGES C CXX)

include_directories(include)

add_executable(DynamicSketch
    src/xxhash.c
    src/xxh_x86dispatch.c
    src/BOBHash32.cpp
    src/countmin.cc
    src/DynamicSketch.cpp
    src/ElasticSketch.cpp
    src/HeavyPart.cpp
    src/LightPart.cpp
    src/main.cpp
    src/Murmurhash.cpp
    src/NaiveSketch.cpp
    src/prng.cc
    src/rand48.cc
    src/Dictionary.cpp
    src/LinkedCellSketch.cpp
)
