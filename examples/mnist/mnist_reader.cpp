#include "mnist_reader.h"

#include <fstream>
#include <assert.h>
#include <cstddef>
#include <cstdint>

namespace
{
  int32_t toInt(std::array<std::byte, 4> bts)
  {
    // MNIST data is stored in high endian
    auto I = [](std::byte b) { return std::to_integer<int32_t>(b); };
    return (I(bts[0]) << 24) | (I(bts[1]) << 16) | (I(bts[2]) << 8) | (I(bts[3]) << 0);
  }

  int32_t readInt(std::basic_ifstream<std::byte>& stream) {
    std::array<std::byte, 4> buffer;
    stream.read(buffer.data(), buffer.size());
    return toInt(buffer);
  }

  unsigned char readChar(std::basic_ifstream<std::byte>& stream){
    std::byte byte;
    stream.read(&byte, 1);
    return static_cast<unsigned char>(byte);
  }

  std::vector<std::array<double, 10>> readLabels(std::filesystem::path file)
  {
    std::basic_ifstream<std::byte> stream(file, std::ios::binary);
    if (!stream.is_open()) {
      assert(false && "Error reading mnist. Can't open label file.");
      return {};
    }

    const int32_t magicNumber = readInt(stream);
    if (magicNumber != 2049) {
      assert(false && "Error reading mnist. Wrong label file?");
      return {};
    }

    const int32_t size = readInt(stream);
    std::vector<std::array<double, 10>> labels(size);
    for (auto& oneHotLabel : labels) {
      oneHotLabel.fill(0.);
      oneHotLabel[readChar(stream)] = 1.;
    }

    return labels;
  }

  std::vector<std::array<double, mnist_reader::IMAGE_SIZE>> readImages(std::filesystem::path file)
  {
    std::basic_ifstream<std::byte> stream(file, std::ios::binary);
    if (!stream.is_open()) {
      assert(false && "Error reading mnist. Can't open image file.");
      return {};
    }

    const int32_t magicNumber = readInt(stream);
    if (magicNumber != 2051) {
      assert(false && "Error reading mnist. Wrong image file?");
      return {};
    }
    const int32_t size = readInt(stream);
    const int32_t rows = readInt(stream);
    const int32_t cols = readInt(stream);
    if (rows != mnist_reader::IMAGE_HEIGHT || cols != mnist_reader::IMAGE_WIDTH) {
      assert(false && "Error reading mnist. Wrong image dimensions");
      return {};
    }

    std::vector<std::array<double, mnist_reader::IMAGE_SIZE>> images(size);
    for (auto& image : images)
      for (double& pixel : image)
        pixel = static_cast<double>(readChar(stream)) / 255;

    return images;
  }
}

namespace mnist_reader
{

  MnistDataSet readDataSet(std::filesystem::path imagesFile, std::filesystem::path labelsFile)
  {
    auto images = readImages(imagesFile);
    auto labels = readLabels(labelsFile);

    if (images.size() != labels.size()) {
      assert(false && "Error reading mnist. Images and labels don't match");
      return {};
    }

    MnistDataSet dataSet;
    for (size_t i = 0; i < images.size(); ++i) {
      dataSet.push_back({images[i], labels[i]});
    }
    return dataSet;
  }

}
