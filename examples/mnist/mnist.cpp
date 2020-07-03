#include "neural_net.h"
#include "backward_prop.h"
#include "activation.h"
#include "loss.h"
#include "optimize.h"

#include "mnist_reader.h"

#include <cassert>
#include <iostream>
#include <fstream>
#include <utility>

constexpr size_t InputSize = 28 * 28;

using namespace mnist_reader;
using namespace staticnet;

auto digest(const MnistDataSet& dataSet)
{
  std::vector<InputVector<IMAGE_SIZE>> images;
  std::vector<std::array<double, 10>> labels;
  for (const MnistDataPoint& point : dataSet) {
    images.emplace_back(point.image);
    labels.push_back(point.oneHotLabel);
  }
  return std::pair(images, labels);
}

template<class TNetwork>
double evaluate(const MnistDataSet& test, int N, const TNetwork& network)
{
  int hits = 0;
  int misses = 0;
  for (int i = 0; i < N; ++i)
  {
    auto [input, correctOutput] = test[i];
    ForwardProp forward(network, input);
    int guess = int(std::max_element(forward._Y.begin(), forward._Y.end()) - forward._Y.begin());
    int correct = int(std::max_element(correctOutput.begin(), correctOutput.end()) - correctOutput.begin());
    if (guess == correct)
      hits += 1;
    else
      misses += 1;
  }
  return double(hits)/(hits+misses);
}

template<template<class TNetwork, class TLoss> class Optimizer>
double mnist_train(const MnistDataSet& train, const MnistDataSet& test, int epochs)
{
  std::cout << "########## RESET ##########" << std::endl;

  NeuralNetwork network = InputVector<InputSize> {} | DENSE<10>{} | activation::SoftMax {};
  Optimizer optimizer(network);

  for (int e = 0; e < epochs; ++e) {
    std::cout << evaluate(train, 1000, network) << " " << evaluate(test, 1000, network) << std::endl;
    optimizer.optimize(digest(train), 0.01);
  }

  return evaluate(test, 1000, network);
}

int main()
{
  const MnistDataSet train = [](){
    MnistDataSet train = readDataSet("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    KnuthShuffle(train);
    return train;
  }();
  const MnistDataSet test = readDataSet("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

  double sgd = mnist_train<SgdOptimizer>(train, test, 10);
  std::cout << "SgdOptimizer: " << sgd << std::endl;
  assert(sgd > 0.89);

  double minibatch = mnist_train<BatchOptimizer>(train, test, 10);
  std::cout << "BatchOptimizer: " << minibatch << std::endl;
  assert(minibatch > 0.89);

  double momentum = mnist_train<MomentumOptimizer>(train, test, 10);
  std::cout << "MomentumOptimizer: " << momentum << std::endl;
  assert(momentum > 0.89);
}
