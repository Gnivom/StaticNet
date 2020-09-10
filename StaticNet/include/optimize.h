#pragma once

#include "neural_net.h"
#include "forward_prop.h"
#include "backward_prop.h"
#include "loss.h"
#include "activation.h"

#include <cassert>
#include <vector>
#include <optional>
#include <future>
#include <algorithm>

namespace staticnet {

  namespace detail
  {
    template<class TNetwork>
    struct DefaultLossFunction { using type = loss::MeanSquareError<TNetwork::NOut>; };
    template<class... Ts>
    struct DefaultLossFunction<NeuralNetwork<activation::SoftMax, Ts...>> { using type = loss::CrossEntropy<NeuralNetwork<activation::SoftMax, Ts...>::NOut>; };
  }

  template<class TNetwork, class TLossFunction = typename detail::DefaultLossFunction<TNetwork>::type>
  class Optimizer
  {
  public:
    explicit Optimizer(TNetwork& network) : _network(network) {}
  protected:
    TNetwork& _network;
  };

  template<class TNetwork, class TLossFunction = typename detail::DefaultLossFunction<TNetwork>::type>
  class SgdOptimizer : Optimizer<TNetwork, TLossFunction>
  {
    using Parent = Optimizer<TNetwork, TLossFunction>;
    using TInputs = std::vector<typename TNetwork::InputType>;
    using TOutputs = std::vector<typename TNetwork::TOut>;
  public:
    explicit SgdOptimizer(TNetwork& network) : Parent(network) {}
    void optimize(std::pair<TInputs, TOutputs> insOuts, double vLearnRate, double vDecay = 0.0)
    {
      auto [inputs, correctOutput] = insOuts;
      for (int i = 0; i < inputs.size(); ++i) {
        TLossFunction lossFunction {correctOutput[i]};
        ForwardProp forward(this->_network, inputs[i]);
        BackwardProp backward(this->_network, forward, lossFunction);
        backward.UpdateNeuralNet(vLearnRate, vDecay);
      }
    }
  };

  template<class TNetwork, class TLossFunction = typename detail::DefaultLossFunction<TNetwork>::type>
  class BatchOptimizer : Optimizer<TNetwork, TLossFunction>
  {
    using Parent = Optimizer<TNetwork, TLossFunction>;
    using TInputs = std::vector<typename TNetwork::InputType>;
    using TOutputs = std::vector<typename TNetwork::TOut>;
  public:
    explicit BatchOptimizer(TNetwork& network, int batchSize = 256) : Parent(network), _batchSize(batchSize) {}
    void optimize(std::pair<TInputs, TOutputs> insOuts, double vLearnRate, double vDecay = 0.0)
    {
      auto [inputs, correctOutput] = insOuts;

      using PortableGradient = typename BackwardProp<TNetwork, TLossFunction>::PortableGradient;
      const int NUM_THREADS = 24;

      for (int batch = 0; batch*_batchSize < inputs.size(); ++batch) {
        const int begin = batch*_batchSize;
        const int end = std::min((batch+1)*_batchSize, int(inputs.size()));
        std::vector<std::future<PortableGradient>> futures;
        for (int ti = 0; ti < NUM_THREADS; ++ti) {
          futures.push_back(std::async([&, ti]{
            PortableGradient gradientSum;
            for (int i = begin + ti; i < end; i += NUM_THREADS) {
              TLossFunction lossFunction {correctOutput[i]};
              ForwardProp forward(this->_network, inputs[i]);
              BackwardProp backward(this->_network, forward, lossFunction);
              gradientSum += backward.GetGradient();
            }
            return gradientSum;
          }));
        }
        PortableGradient gradientSum;
        for (auto& f : futures)
          gradientSum += f.get();
        // TODO: don't have to regenerate a BackwardProp
        TLossFunction lossFunction {correctOutput[0]};
        ForwardProp forward(this->_network, inputs[0]);
        BackwardProp backward(this->_network, forward, lossFunction);
        backward.ApplyGradient(std::move(gradientSum) * vLearnRate, vDecay * _batchSize);
      }
    }

  private:
    const int _batchSize;
  };

  template<class TNetwork, class TLossFunction = typename detail::DefaultLossFunction<TNetwork>::type>
  class MomentumOptimizer : Optimizer<TNetwork, TLossFunction>
  {
    using Parent = Optimizer<TNetwork, TLossFunction>;
    using TInputs = std::vector<typename TNetwork::InputType>;
    using TOutputs = std::vector<typename TNetwork::TOut>;
  public:
    explicit MomentumOptimizer(TNetwork& network, double vFriction = 0.1) : Parent(network), _vFriction(vFriction) {}
    void optimize(std::pair<TInputs, TOutputs> insOuts, double vLearnRate, double vDecay = 0.0)
    {
      auto [inputs, correctOutput] = insOuts;
      typename BackwardProp<TNetwork, TLossFunction>::PortableGradient momentum;
      for (int i = 0; i < inputs.size(); ++i) {
        TLossFunction lossFunction {correctOutput[i]};
        ForwardProp forward(this->_network, inputs[i]);
        BackwardProp backward(this->_network, forward, lossFunction);

        auto acceleration = backward.GetGradient();
        momentum = std::move(momentum) * (1.0-_vFriction) + std::move(acceleration) * _vFriction;

        auto change = momentum;
        backward.ApplyGradient(std::move(change *= vLearnRate), vDecay);
      }
    }
  private:
    const double _vFriction;
  };

}
