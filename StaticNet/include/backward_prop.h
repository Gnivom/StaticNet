#pragma once

#include "neural_net.h"
#include "forward_prop.h"

// Because auto doesn't work for member variables
#define AutoVar( lhs, rhs ) decltype( rhs ) lhs = rhs;
#define AutoVarRef( lhs, rhs ) decltype( rhs )& lhs = rhs;
#define AutoVarCRef( lhs, rhs ) const decltype( rhs )& lhs = rhs;

namespace staticnet
{
  template<typename TNeuralNet, typename FLossFunction, size_t MAX_N = TNeuralNet::_N, size_t N = 1>
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
      decltype(TNeuralNet()._B) _E_B;
      decltype(TNeuralNet()._W) _E_W;
      typename decltype(_Prev)::PortableGradient _Prev;
      PortableGradient(decltype(_E_B) eB, decltype(_E_W) eW, decltype(_Prev) prev) : _E_B(std::move(eB)), _E_W(std::move(eW)), _Prev(std::move(prev)) {}
      PortableGradient() { _E_B.fill(0.0); _E_W.fill(0.0); }
      PortableGradient& operator+=(PortableGradient&& o) { _E_B += std::move(o._E_B); _E_W += std::move(o._E_W); _Prev += std::move(o._Prev); return *this; }
      PortableGradient& operator*=(double v) { _E_B *= v; _E_W *= v; _Prev *= v; return *this; }
      PortableGradient operator*(double v) && { (*this) *= v; return std::move(*this); }
      PortableGradient operator+(PortableGradient&& o) && { (*this) += std::move(o); return std::move(*this); }
    };
    PortableGradient GetGradient() const {
      return {GetTranspose(_E_B), _E_W, _Prev.GetGradient()};
    }
    void ApplyGradient(const PortableGradient& gradient, double vDecay) const {
      _Layer._B *= 1.0 - vDecay;
      _Layer._W *= 1.0 - vDecay;
      _Layer._B -= gradient._E_B;
      _Layer._W -= gradient._E_W;
      _Prev.ApplyGradient(gradient._Prev, vDecay);
    }
    void UpdateNeuralNet(double vLearnRate, double vDecay) const {
      ApplyGradient(GetGradient() * vLearnRate, vDecay);
    }

    AutoVarRef(_Layer, GetLayer<N>::Access(_NeuralNet));

    // Prep
    AutoVarRef(_E_Y, _Prev._E_X); // For each i, ROW
    AutoVar(_Y_Z, _Layer._A.df(ToArray(_ForwardProp._Z))); // For each i
    AutoVar(_E_Z, _E_Y* _Y_Z); // For each i

    // For W
    AutoVar(_E_W, MatrixMultiplier<typename decltype(_Layer._W)::MatrixType>(_Layer._W)(GetTranspose(_E_Z), GetTranspose(_ForwardProp._X)));

    // For B
    AutoVarCRef(_E_B, _E_Z);

    // For next step
    AutoVarCRef(_Z_X, _Layer._W); // dZ[i] / dX[j] = _Z_X[i][j]
    AutoVar(_E_X, _E_Z* _Z_X); // For each j, ROW
  };

  template<typename TNeuralNet, typename FLossFunction, size_t MAX_N>
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
      decltype(TNeuralNet()._B) _E_B;
      decltype(TNeuralNet()._W) _E_W;
      PortableGradient() { _E_B.fill(0.0); _E_W.fill(0.0); }
      PortableGradient(decltype(_E_B) eB, decltype(_E_W) eW) : _E_B(std::move(eB)), _E_W(std::move(eW)) {}
      PortableGradient& operator+=(PortableGradient&& o) { _E_B += std::move(o._E_B); _E_W += std::move(o._E_W); return *this; }
      PortableGradient& operator*=(double v) { _E_B *= v; _E_W *= v; return *this; }
      PortableGradient operator*(double v) && { (*this) *= v; return std::move(*this); }
      PortableGradient operator+(PortableGradient&& o) && { (*this) += std::move(o); return std::move(*this); }
    };
    PortableGradient GetGradient() const {
      return {GetTranspose(_E_B), _E_W};
    }
    void ApplyGradient(const PortableGradient& gradient, double vDecay) const {
      _Layer._B *= 1.0 - vDecay;
      _Layer._W *= 1.0 - vDecay;
      _Layer._B -= gradient._E_B;
      _Layer._W -= gradient._E_W;
    }
    void UpdateNeuralNet(double vLearnRate, double vDecay) const {
      ApplyGradient(GetGradient() * vLearnRate, vDecay);
    }

    AutoVarRef(_Layer, GetLayer<N>::Access(_NeuralNet));

    // Prep
    AutoVarCRef(_E_Y, GetTranspose(_LossFunction.df(_ForwardProp._Y)));
    AutoVar(_Y_Z, _Layer._A.df(ToArray(_ForwardProp._Z))); // For each i
    AutoVar(_E_Z, _E_Y* _Y_Z); // For each i

    // For W
    AutoVar(_E_W, MatrixMultiplier<typename decltype(_Layer._W)::MatrixType>(_Layer._W)(GetTranspose(_E_Z), GetTranspose(_ForwardProp._X)));

    // For B
    AutoVarCRef(_E_B, _E_Z);

    // For next step
    AutoVarCRef(_Z_X, _Layer._W); // dZ[i] / dX[j] = _Z_X[i][j]
    AutoVar(_E_X, _E_Z* _Z_X); // For each j, ROW
  };
}

#undef AutoVar
#undef AutoVarRef
#undef AutoVarCRef
