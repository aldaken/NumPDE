# define executable
set(sources
  lfen.cc lfen_main.cc lfen.h
)

add_executable(lecturecodes.lfen ${sources})

target_link_libraries(lecturecodes.lfen
  PUBLIC Eigen3::Eigen
)
