#pragma once

#include <vector>
#include <array>
#include <filesystem>

namespace mnist_reader
{
  constexpr size_t IMAGE_WIDTH = 28;
  constexpr size_t IMAGE_HEIGHT = 28;
  constexpr size_t IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;

  struct MnistDataPoint {
    std::array<double, IMAGE_SIZE> image;
    std::array<double, 10> oneHotLabel;
  };
  using MnistDataSet = std::vector<MnistDataPoint>;

  MnistDataSet readDataSet(std::filesystem::path imagesFile, std::filesystem::path labelsFile);
}
