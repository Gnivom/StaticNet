#pragma once

#include "matrix.h"
#include "loss.h"

#include <type_traits>
#include <functional>
#include <iostream>

namespace staticnet
{
  ///////////////// Helpers to construct a network structure

  template<size_t N>
  struct DENSE {};
  template<size_t HEIGHT, size_t WIDTH, size_t CONV_RADIUS, size_t OUTPUT_DEPTH=1>
  struct CONVOLUTION2D{};

  struct RandomTag {}; inline RandomTag Randomize;
  template<size_t N>
  struct InputVector {
    using IS_MODEL_STRUCTURE = void;
    constexpr static size_t SIZE = N;
    std::array<double, N> _data;
    InputVector(): _data({}) {}
    InputVector(RandomTag) { for (double& x : _data) x = GetSignedUnitRand(); }
    InputVector(std::array<double, N> data): _data(std::move(data)) {}
    std::array<double, N> Get() const { return _data; }
  };
  template<class... Ts>
  struct ModelStructure {
    using IS_MODEL_STRUCTURE = void;
    ModelStructure() {}
    ModelStructure(Ts...) {}
    template<class T>
    ModelStructure<T, Ts...> prepend(T) const { return {}; }
  };
  template<class TList, class Prepend, class = typename TList::IS_MODEL_STRUCTURE>
  auto operator|(TList list, Prepend p) {
    return ModelStructure(list).prepend(p);
  }

  template<class T, class INPUT_TYPE, size_t INPUT_SIZE>
  struct MatrixType {};
  template<size_t N, class INPUT_TYPE, size_t INPUT_SIZE>
  struct MatrixType<DENSE<N>, INPUT_TYPE, INPUT_SIZE>
  {
    static auto Create()
    {
      Matrix<N, INPUT_SIZE> Ret;
      Ret.Randomize();
      return Ret;
    }
  };
  template<size_t HEIGHT, size_t WIDTH, size_t CONV_RADIUS, size_t OUTPUT_DEPTH, class INPUT_TYPE, size_t INPUT_SIZE>
  struct MatrixType<CONVOLUTION2D<HEIGHT, WIDTH, CONV_RADIUS, OUTPUT_DEPTH>, INPUT_TYPE, INPUT_SIZE> {
    static_assert(INPUT_SIZE==WIDTH*HEIGHT, "Wrong dimensions on CONVOLUTION2D");
    static auto Create()
    {
      constexpr size_t OUTPUT_SIZE_PER_DEPTH = INPUT_SIZE;
      constexpr size_t NEIGHBORS_PER_INPUT = (CONV_RADIUS*2+1)*(CONV_RADIUS*2+1);
      SparseMatrix<OUTPUT_SIZE_PER_DEPTH*OUTPUT_DEPTH, INPUT_SIZE, NEIGHBORS_PER_INPUT*OUTPUT_DEPTH> Ret;
      for (size_t row = 0; row < HEIGHT; ++row) {
        for (size_t col = 0; col < WIDTH; ++col) {
          const size_t m = row*WIDTH + col;
          for (int drow = -int(CONV_RADIUS); drow <= int(CONV_RADIUS); ++drow) {
            if (int(row) + drow < 0) continue;
            if (int(row) + drow >= HEIGHT) continue;
            for (int dcol = -int(CONV_RADIUS); dcol <= int(CONV_RADIUS); ++dcol) {
              if (int(col) + dcol < 0) continue;
              if (int(col) + dcol >= WIDTH) continue;
              const int relativePosition = drow*WIDTH + dcol;
              const size_t sharedIndex = (drow+CONV_RADIUS)*(2*CONV_RADIUS+1)+dcol+CONV_RADIUS;
              if (sharedIndex >= NEIGHBORS_PER_INPUT) {
                BringError(0.0);
              }
              for (size_t out_depth = 0; out_depth < OUTPUT_DEPTH; ++out_depth) {
                const size_t n = out_depth*OUTPUT_SIZE_PER_DEPTH + m + relativePosition;
                Ret._entries.emplace_back(n, m, sharedIndex);
              }
            }
          }
        }
      }
      Ret.Randomize();
      return Ret;
    }
  };

  ///////////////// The actual neural network

  template<class... Ts>
  struct NeuralNetwork {};
  template<class TInput>
  struct NeuralNetwork<TInput> // 'Layer 0'
  {
    NeuralNetwork(ModelStructure<TInput>){}
    NeuralNetwork(double vFillAll = 0.0) {}
    using InputType = TInput;
    using MyType = NeuralNetwork<TInput>;
    constexpr static size_t _N = 0;
    static auto GetInput(TInput input) { return std::move(input).Get(); }
    MyType& operator+=(const MyType& Other) { return *this; }

    struct HelperStruct { auto operator()(TInput&& t) const { return GetInput(std::move(t)); } };
    using TOut = std::invoke_result_t<HelperStruct, TInput&&>;
    TOut TOutDummy() const { return TOut(); }
    constexpr static size_t NOut = TInput::SIZE;
  };
  template<class Activation, class WType, class... Ts>
  struct NeuralNetwork<Activation, WType, Ts...>
  {
    static_assert(sizeof...(Ts) % 2 == 1, "Neural network types should start with N pairs of (Activation, WType) and end with a InputType");
    using InputType = typename NeuralNetwork<Ts...>::InputType;
    using MyType = NeuralNetwork<WType, Activation, Ts...>;

    NeuralNetwork(ModelStructure<Activation, WType, Ts...>) : NeuralNetwork(){}
    NeuralNetwork() { _W.Randomize(); _B.Randomize(); }
    NeuralNetwork(double vFillAll): _Prev(vFillAll) { _B.fill(vFillAll); _W.fill(vFillAll); }

    MyType& operator+=(const MyType& Other)
    {
      _Prev += Other._Prev;
      _W += Other._W;
      _B += Other._B;
      return *this;
    }

    NeuralNetwork<Ts...> _Prev;
    constexpr static size_t In = decltype(_Prev)::NOut;
    constexpr static size_t _N = decltype(_Prev)::_N+1;
    using MyMatrixType = MatrixType<WType, decltype(_Prev.TOutDummy()), In>;
    decltype(MyMatrixType::Create()) _W = MyMatrixType::Create();
    constexpr static size_t NOut = decltype(_W)::_N;
    Matrix<NOut, 1> _B;
    Activation _A;
    using TOut = decltype(_A.f(_W* _Prev.TOutDummy() + _B));
    TOut TOutDummy() const { return TOut(); }
  };
  template<class... Ts>
  NeuralNetwork(ModelStructure<Ts...>) -> NeuralNetwork<Ts...>; // Deduction guide

  ///////////////// Helper for finding the N'th layer of a NeuralNetwork (or of a ForwardProp)
  template<size_t N>
  struct GetLayer
  {
    template<class TLayer>
    static auto Get(const TLayer& Network) ->
      typename std::enable_if<N == TLayer::_N, const decltype(Network)&>::type
    {
      return Network;
    }
    template<class TLayer>
    static auto Get(const TLayer& Network) ->
      typename std::enable_if<N < TLayer::_N, const decltype(Get(Network._Prev))&>::type
    {return Get(Network._Prev);}

    template<class TLayer>
    static auto Access(TLayer& Network) ->
      typename std::enable_if<N == TLayer::_N, decltype(Network)&>::type
    {
      return Network;
    }
    template<class TLayer>
    static auto Access(TLayer& Network) ->
      typename std::enable_if<N < TLayer::_N, decltype(Access(Network._Prev))&>::type
    {return Access(Network._Prev);}
  };

}
