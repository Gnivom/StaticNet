#pragma once

#include "matrix.h"
#include "propagation_data.h"
#include "loss.h"

#include <type_traits>
#include <functional>
#include <iostream>

namespace staticnet
{
  ///////////////// Helpers to construct a network structure

  template<size_t N>
  struct DENSE {};
  template<typename Constructor>
  struct SPARSE {};

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

  template<typename T, typename INPUT_TYPE, size_t INPUT_SIZE>
  struct MatrixType {};
  template<size_t N, typename INPUT_TYPE, size_t INPUT_SIZE>
  struct MatrixType<DENSE<N>, INPUT_TYPE, INPUT_SIZE>
  {
    static auto Create()
    {
      Matrix<N, INPUT_SIZE> Ret;
      Ret.Randomize();
      return Ret;
    }
  };
  template<typename Constructor, typename INPUT_TYPE, size_t INPUT_SIZE>
  struct MatrixType<SPARSE<Constructor>, INPUT_TYPE, INPUT_SIZE> {
    static auto Create() { return Constructor::template Create<INPUT_TYPE>(); }
  };
  template<size_t OUT_DEPTH, size_t RADIUS>
  struct CreateConv {
    template<typename TPropagationData> // Input type
    static auto Create()
    {
      return TPropagationData::CreateConvLayer(SIZET<OUT_DEPTH>(), RADIUS);
    }
  };
  struct CreateInternalizer {
    template<typename TPropagationData>
    static auto Create()
    {
      return CreateInternalizerLayer<TPropagationData>();
    }
  };

  ///////////////// The actual neural network

  template<typename... Ts>
  struct NeuralNetwork {};
  template<typename TInput>
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
  template<typename Activation, typename WType, typename... Ts>
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
    using MatrixType = MatrixType<WType, decltype(_Prev.TOutDummy()), In>;
    decltype(MatrixType::Create()) _W = MatrixType::Create();
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
    template<typename TLayer>
    static auto Get(const TLayer& Network) ->
      typename std::enable_if<N == TLayer::_N, const decltype(Network)&>::type
    {
      return Network;
    }
    template<typename TLayer>
    static auto Get(const TLayer& Network) ->
      typename std::enable_if<N < TLayer::_N, const decltype(Get(Network._Prev))&>::type
    {return Get(Network._Prev);}

    template<typename TLayer>
    static auto Access(TLayer& Network) ->
      typename std::enable_if<N == TLayer::_N, decltype(Network)&>::type
    {
      return Network;
    }
    template<typename TLayer>
    static auto Access(TLayer& Network) ->
      typename std::enable_if<N < TLayer::_N, decltype(Access(Network._Prev))&>::type
    {return Access(Network._Prev);}
  };

}
