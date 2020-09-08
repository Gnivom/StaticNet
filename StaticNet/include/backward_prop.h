#pragma once

#include "neural_net.h"
#include "forward_prop.h"

// Because auto doesn't work for member variables
#define AutoVar( lhs, rhs ) decltype( rhs ) lhs = rhs;
#define AutoVarRef( lhs, rhs ) decltype( rhs )& lhs = rhs;
#define AutoVarCRef( lhs, rhs ) const decltype( rhs )& lhs = rhs;

namespace staticnet
{
  template<class TNeuralNet, class FLossFunction, size_t MAX_N = TNeuralNet::_N, size_t N = 1>
  struct BackwardProp
  {
    const ForwardProp<TNeuralNet, N>& _ForwardProp;
    BackwardProp<TNeuralNet, FLossFunction, MAX_N, N + 1> _Prev;
    TNeuralNet& _NeuralNet;

    BackwardProp(TNeuralNet& NeuralNet, ForwardProp<TNeuralNet, MAX_N>& ForwardProp, FLossFunction LossFunction)
      : _ForwardProp(GetLayer<N>::Get(ForwardProp)), _Prev(NeuralNet, ForwardProp, std::move(LossFunction)), _NeuralNet(NeuralNet)
    {}
    BackwardProp(const BackwardProp&) = delete;
    BackwardProp(BackwardProp&&) = delete;

    struct PortableGradient {
      decltype(GetLayer<N>::Get(TNeuralNet())._B) _dE_dB;
      decltype(GetLayer<N>::Get(TNeuralNet())._W) _dE_dW;
      typename decltype(_Prev)::PortableGradient _PrevGradient;
      PortableGradient(decltype(_dE_dB) eB, decltype(_dE_dW) eW, decltype(_PrevGradient) prev) : _dE_dB(std::move(eB)), _dE_dW(std::move(eW)), _PrevGradient(std::move(prev)) {}
      PortableGradient() { _dE_dB.fill(0.0); _dE_dW.fill(0.0); }
      PortableGradient& operator+=(PortableGradient&& o) { _dE_dB += std::move(o._dE_dB); _dE_dW += std::move(o._dE_dW); _PrevGradient += std::move(o._Prev); return *this; }
      PortableGradient& operator*=(double v) { _dE_dB *= v; _dE_dW *= v; _PrevGradient *= v; return *this; }
      PortableGradient operator*(double v) && { (*this) *= v; return std::move(*this); }
      PortableGradient operator+(PortableGradient&& o) && { (*this) += std::move(o); return std::move(*this); }
    };
    PortableGradient GetGradient() const {
      return {GetTranspose(_dE_dB), _dE_dW, _Prev.GetGradient()};
    }
    void ApplyGradient(const PortableGradient& gradient, double vDecay) const {
      _Layer._B *= 1.0 - vDecay;
      _Layer._W *= 1.0 - vDecay;
      _Layer._B -= gradient._dE_dB;
      _Layer._W -= gradient._dE_dW;
      _Prev.ApplyGradient(gradient._PrevGradient, vDecay);
    }
    void UpdateNeuralNet(double vLearnRate, double vDecay) const {
      ApplyGradient(GetGradient() * vLearnRate, vDecay);
    }

    AutoVarRef(_Layer, GetLayer<N>::Access(_NeuralNet));

    // Prep
    AutoVarRef(_dE_dY, _Prev._dE_dX); // For each i, ROW
    AutoVar(_dY_dZ, _Layer._A.df(_ForwardProp._Z)); // For each i
    AutoVar(_dE_dZ, _dE_dY* _dY_dZ); // For each i

    // For W
    AutoVar(_dE_dW, MatrixMultiplier<typename decltype(_Layer._W)::MatrixType>(_Layer._W)(GetTranspose(_dE_dZ), GetTranspose(_ForwardProp._X)));

    // For B
    AutoVarCRef(_dE_dB, _dE_dZ);

    // For next step
    AutoVarCRef(_dZ_dX, _Layer._W); // dZ[i] / dX[j] = _dZ_dX[i][j]
    AutoVar(_dE_dX, _dE_dZ* _dZ_dX); // For each j, ROW
  };

  template<class TNeuralNet, class FLossFunction, size_t MAX_N>
  struct BackwardProp<TNeuralNet, FLossFunction, MAX_N, MAX_N>
  {
    constexpr static size_t N = MAX_N;
    const ForwardProp<TNeuralNet, N>& _ForwardProp;
    TNeuralNet& _NeuralNet;
    FLossFunction _LossFunction;

    BackwardProp(TNeuralNet& NeuralNet, ForwardProp<TNeuralNet, MAX_N>& ForwardProp, FLossFunction LossFunction)
      : _ForwardProp(GetLayer<MAX_N>::Get(ForwardProp)), _NeuralNet(NeuralNet), _LossFunction(std::move(LossFunction))
    {}
    BackwardProp(const BackwardProp&) = delete;
    BackwardProp(BackwardProp&&) = delete;

    struct PortableGradient {
      decltype(TNeuralNet()._B) _dE_dB;
      decltype(TNeuralNet()._W) _dE_dW;
      PortableGradient() { _dE_dB.fill(0.0); _dE_dW.fill(0.0); }
      PortableGradient(decltype(_dE_dB) eB, decltype(_dE_dW) eW) : _dE_dB(std::move(eB)), _dE_dW(std::move(eW)) {}
      PortableGradient& operator+=(PortableGradient&& o) { _dE_dB += std::move(o._dE_dB); _dE_dW += std::move(o._dE_dW); return *this; }
      PortableGradient& operator*=(double v) { _dE_dB *= v; _dE_dW *= v; return *this; }
      PortableGradient operator*(double v) && { (*this) *= v; return std::move(*this); }
      PortableGradient operator+(PortableGradient&& o) && { (*this) += std::move(o); return std::move(*this); }
    };
    PortableGradient GetGradient() const {
      return {GetTranspose(_dE_dB), _dE_dW};
    }
    void ApplyGradient(const PortableGradient& gradient, double vDecay) const {
      _Layer._B *= 1.0 - vDecay;
      _Layer._W *= 1.0 - vDecay;
      _Layer._B -= gradient._dE_dB;
      _Layer._W -= gradient._dE_dW;
    }
    void UpdateNeuralNet(double vLearnRate, double vDecay) const {
      ApplyGradient(GetGradient() * vLearnRate, vDecay);
    }

    AutoVarRef(_Layer, GetLayer<N>::Access(_NeuralNet));

    // Prep
    AutoVarCRef(_dE_dY, GetTranspose(_LossFunction.df(_ForwardProp._Y)));
    AutoVar(_dY_dZ, _Layer._A.df(_ForwardProp._Z)); // For each i
    AutoVar(_dE_dZ, _dE_dY * _dY_dZ); // For each i

    // For W
    AutoVar(_dE_dW, MatrixMultiplier<typename decltype(_Layer._W)::MatrixType>(_Layer._W)(GetTranspose(_dE_dZ), GetTranspose(_ForwardProp._X)));

    // For B
    AutoVarCRef(_dE_dB, _dE_dZ);

    // For next step
    AutoVarCRef(_dZ_dX, _Layer._W); // dZ[i] / dX[j] = _dZ_dX[i][j]
    AutoVar(_dE_dX, _dE_dZ* _dZ_dX); // For each j, ROW
  };
}

#undef AutoVar
#undef AutoVarRef
#undef AutoVarCRef
