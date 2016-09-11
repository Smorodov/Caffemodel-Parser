# ----------------------------------------------------------------------------
# минимальная версия CMake, способная переварить этот файл
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 2.8)
set (PROJ_NAME Caffemodel_parser)
project(${PROJ_NAME})
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
# ----------------------------------------------------------------------------  
# ищем все .cpp и .h файлы и добавляем их в наш проект 
# ----------------------------------------------------------------------------
  set(folder_source main.cpp caffe.pb.cc)
  set(folder_header  )
  SOURCE_GROUP("Source Files" FILES ${folder_source})
  SOURCE_GROUP("Header Files" FILES ${folder_header})

  include_directories( protobuf-master/src) 

# add_definitions(-DPROTOBUF_USE_DLLS)
# ----------------------------------------------------------------------------  
# создаем проект
# ----------------------------------------------------------------------------
  ADD_EXECUTABLE(${PROJ_NAME} ${folder_source} ${folder_header})
# ----------------------------------------------------------------------------
# добавляем Lib-ы opencv 
# ---------------------------------------------------------------------------- 
  TARGET_LINK_LIBRARIES(${PROJ_NAME} debug ${CMAKE_BINARY_DIR}/Debug/libprotobufd.lib optimized ${CMAKE_BINARY_DIR}/Release/libprotobuf.lib)

  add_subdirectory(./protobuf-master/cmake)



