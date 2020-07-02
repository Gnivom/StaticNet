#pragma once
#include "matrix.h"
#include "activation.h"

namespace staticnet
{
  namespace loss
  {

    template<size_t SIZE>
    struct MeanSquareError {
      MeanSquareError(std::array<double, SIZE> answer): _answer(std::move(answer)) {}
      std::array<double, SIZE> _answer;

      double f(std::array<double, SIZE> guess) const {
        double loss = 0.0;
        for (size_t i = 0; i < guess.size(); ++i) {
          loss += (guess[i] - _answer[i]) * (guess[i] - _answer[i]);
        }
        return loss;
      }
      std::array<double, SIZE> df(std::array<double, SIZE> guess) const {
        for (size_t i = 0; i < guess.size(); ++i) {
          guess[i] -= _answer[i];
        }
        return guess;
      }
    };

    template<size_t sSIZE>
    struct CrossEntropy {
      constexpr static size_t SIZE = sSIZE;
      CrossEntropy(std::array<double, sSIZE> answer): _answer(std::move(answer)) {}
      std::array<double, SIZE> _answer;

      double f(std::array<double, SIZE> guess) const {
        double loss = 0.0;
        for (size_t i = 0; i < guess.size(); ++i) {
          loss -= _answer[i] * std::log(guess[i]);
        }
        return loss;
      }

      // Hacky type-safe optimization to exploit simple softmax+cross-entropy gradient
      struct Gradient {
        using TCrossEntropy = CrossEntropy;
        std::array<double, SIZE> _answer;
        std::array<double, SIZE> _guess;
      };
      struct TransposedGradient {
        using TCrossEntropy = CrossEntropy;
        std::array<double, SIZE> _answer;
        std::array<double, SIZE> _guess;
        Matrix<1, SIZE> operator*(activation::SoftMax::Gradient) const {
          Matrix<1, SIZE> result;
          for (size_t i = 0; i < SIZE; ++i) {
            result[0][i] = _guess[i] - _answer[i];
          }
          return result;
        }
      };
      Gradient df(std::array<double, SIZE> guess) const {
        return Gradient {_answer, std::move(guess)};
      }
    };

    template<class TGradient, class TTransposedGradient = typename TGradient::TCrossEntropy::TransposedGradient>
    TTransposedGradient GetTranspose(TGradient&& gradient) {
      return {std::move(gradient._answer), std::move(gradient._guess)};
    }

  }

}
