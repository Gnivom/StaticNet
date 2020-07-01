#pragma once

#include "neural_net.h"

// Because auto doesn't work for member variables
#define AutoVar( lhs, rhs ) decltype( rhs ) lhs = rhs;
#define AutoVarRef( lhs, rhs ) decltype( rhs )& lhs = rhs;
#define AutoVarCRef( lhs, rhs ) const decltype( rhs )& lhs = rhs;

namespace staticnet
{
  template<typename TNeuralNet, size_t N = TNeuralNet::_N>
  struct ForwardProp
  {
    constexpr static size_t _N = N;
    ForwardProp<TNeuralNet, N - 1> _Prev;

    const TNeuralNet& _NeuralNet;

    ForwardProp(const TNeuralNet& NeuralNet, typename TNeuralNet::InputType input)
      : _Prev(NeuralNet, std::move(input)), _NeuralNet(NeuralNet)
    {}
    ForwardProp(const ForwardProp&) = delete;
    ForwardProp(ForwardProp&&) = delete;

    AutoVarCRef(_Layer, GetLayer<N>::Get(_NeuralNet));
    AutoVarCRef(_X, _Prev._Y);
    AutoVar(_Z, _Layer._W* _X + _Layer._B);
    AutoVar(_Y, _Layer._A.f(_Z));
  };
  template<typename TNeuralNet>
  struct ForwardProp<TNeuralNet, 0>
  {
    const TNeuralNet& _NeuralNet;
    typename TNeuralNet::InputType _Input;
    ForwardProp(const TNeuralNet& NeuralNet, typename TNeuralNet::InputType input)
      : _NeuralNet(NeuralNet), _Input(std::move(input))
    {}
    ForwardProp(const ForwardProp&) = delete;
    ForwardProp(ForwardProp&&) = delete;
    AutoVarCRef(_Layer, GetLayer<0>::Get(_NeuralNet));
    AutoVar(_Y, _Layer.GetInput(_Input));
  };
}

#undef AutoVar
#undef AutoVarRef
#undef AutoVarCRef

