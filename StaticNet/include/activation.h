#pragma once

#include "matrix.h"
#include "util.h"

#include <cmath>
#include <ratio>

namespace staticnet
{

  namespace activation
  {

    struct Linear
    {
      template<typename T>
      const T& f(const T& Mat) const { return Mat; }
      template<typename T>
      double df(const T& Mat) const { return 1.0; }
    };

    namespace detail {
      // CRTP
      template<typename Implementation>
      struct ActivationBase
      {
        template<typename TArray>
        TArray f(const TArray& Mat) const { return Transform(Mat, static_cast<const Implementation*>(this)->_f); }
        template<size_t N>
        DiagonalMatrix<N> df(const Matrix<N, 1>& Mat) const
        {
          DiagonalMatrix<N> Ret;
          for (int row = 0; row < N; ++row)
            Ret._Diagonal[row] = static_cast<const Implementation*>(this)->_df(Mat[row][0]);
          return Ret;
        }
        template<size_t N>
        DiagonalMatrix<N> df(const std::array<double, N>& Mat) const
        {
          DiagonalMatrix<N> Ret;
          for (int row = 0; row < N; ++row)
            Ret._Diagonal[row] = static_cast<const Implementation*>(this)->_df(Mat[row]);
          return Ret;
        }
      };
    }

    struct TanH: public detail::ActivationBase<TanH> {
      std::function<double(double)> _f = [](double x) { return std::tanh(x); };
      std::function<double(double)> _df = [](double x) { return 1.0 / std::pow(std::cosh(x), 2); };
      std::function<double(double)> _inverse = [](double x) { return 0.5*std::log((1.0+x)/(1.0-x)); };
    };
    struct Abs: public detail::ActivationBase<Abs> {
      std::function<double(double)> _f = [](double x) { return x >= 0.0 ? x : -x; };
      std::function<double(double)> _df = [](double x) { return x >= 0.0 ? 1.0 : -1.0; };
    };
    struct Relu: public detail::ActivationBase<Relu> {
      std::function<double(double)> _f = [](double x) { return std::max(x, 0.0); };
      std::function<double(double)> _df = [](double x) { return x >= 0.0 ? 1.0 : 0.0; };
    };
    struct Squash: public detail::ActivationBase<Squash> {
      std::function<double(double)> _f = [](double x) { return x*x/(1.0+x*x); };
      std::function<double(double)> _df = [](double x) { return 2.0*x/square(x*x+1.0); };
    };
    struct HeartBeat: public detail::ActivationBase<HeartBeat> {
      std::function<double(double)> _f = [](double x) { return 3.0*x/square(x*x+1.0); };
      std::function<double(double)> _df = [](double x) { return 3.0-9.0*x*x/cube(x*x+1.0); };
    };
    template<size_t MILLI_THRESHOLD_UNSIGNED>
    struct SymLeakRelu: public detail::ActivationBase<SymLeakRelu<MILLI_THRESHOLD_UNSIGNED>>
    {
      constexpr static double THRESHOLD = double(MILLI_THRESHOLD_UNSIGNED) * 0.001; // TODO C++20, just take THRESHOLD as template parameter
      constexpr static double vAlpha = 0.05;
      std::function<double(double)> _f = [](double x) { return x > THRESHOLD ? vAlpha * THRESHOLD + (x-THRESHOLD): x < -THRESHOLD ? vAlpha * -THRESHOLD + (x+THRESHOLD): vAlpha * x; };
      std::function<double(double)> _df = [](double x) { return x > THRESHOLD ? 1.0 : x < -THRESHOLD ? 1.0 : vAlpha; };
    };

    struct SoftMax
    {
      template<size_t N>
      std::array<double, N> f(const std::array<double, N>& Mat, int nOverrideSize = -1) const
      {
        std::array<double, N> Ret;
        double vSum = 0.0;
        for (int i = 0; i < N; ++i)
        {
          if (i == nOverrideSize)
            break;
          vSum += (Ret[i] = std::exp(Mat[i]));
        }
        Ret *= 1.0 / vSum;
        for (int i = nOverrideSize; i < N; ++i)
          Ret[i] = Mat[i];
        return Ret;
      }
      template<typename TPropagationData>
      TPropagationData f(const TPropagationData& Data) const { return TPropagationData(f(Data._data, Data._SizePerDepth*Data._Depth)); }
      struct Gradient {};
      template<size_t N>
      Gradient df(const std::array<double, N>& x) const
      {
        // Hacky type-safe optimization to exploit simple softmax+cross-entropy gradient. See CrossEntropy::Gradient.
        return Gradient {};
      }
    };

  }
}
