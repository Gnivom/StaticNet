add_executable(MnistExample
  mnist.cpp
  mnist_reader.cpp mnist_reader.h
)
target_link_libraries(MnistExample
  PRIVATE
  StaticNet
)
file(COPY data/ DESTINATION data/)
