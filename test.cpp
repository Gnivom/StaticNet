#include "neural_net.h"
#include "optimize.h"
#include "activation.h"
#include "loss.h"

#include <cassert>
#include <iostream>

using namespace staticnet;

bool produceConstant() {
  NeuralNetwork network = InputVector<10> {} | DENSE<1> {} | TanH {};

  const double desiredValue = 0.1337;
  MeanSquareError<1> lossFunction {{desiredValue}};

  double loss = 0.0;
  for (int i = 0; i < 1000; ++i) {
    ForwardProp forward(network, Randomize);
    BackwardProp backward(network, forward, lossFunction);
    backward.UpdateNeuralNet(0.1, 0.0);
  }
  return loss < 0.00001;
}

bool produceLinear() {
  NeuralNetwork network = InputVector<10> {} | DENSE<10> {} | Linear {};

  double loss = 0.0;

  for (int i = 0; i < 10000; ++i) {
    InputVector<10> input = Randomize;
    MeanSquareError lossFunction {input._data};

    ForwardProp forward(network, input);
    BackwardProp backward(network, forward, lossFunction);
    backward.UpdateNeuralNet(0.1, 0.0);
  }

  return loss < 0.00001;
}

bool categorize() {
  NeuralNetwork network = InputVector<1> {} | DENSE<2> {} | SoftMax {};

  double loss = 0.0;

  for (int i = 0; i < 10000; ++i) {
    InputVector<1> input = Randomize;
    std::array<double, 2> goal = {input._data[0] < 0 ? 0.0 : 1.0, input._data[0] < 0 ? 1.0 : 0.0};
    CrossEntropy lossFunction {goal};
    ForwardProp forward(network, input);
    BackwardProp backward(network, forward, lossFunction);
    backward.UpdateNeuralNet(1.0, 0.000001);
  }

  return loss < 1.0;
}

int main()
{
#define TEST(F) const bool F##_success = F(); assert(F##_success); if (!F##_success) return 1;
  TEST(produceConstant);
  TEST(produceLinear);
  TEST(categorize);
#undef TEST
  return 0;
}
