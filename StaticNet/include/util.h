#pragma once

#include <vector>
#include <array>
#include <thread>
#include <set>
#include <future>
#include <chrono>

namespace staticnet
{

  template<class T>
  constexpr T square(T x)
  {
    return x*x;
  }
  template<class T>
  constexpr T cube(T x)
  {
    return x*x*x;
  }

  unsigned int SafeRand();

  template<class T>
  void KnuthShuffle(T& V)
  {
    const size_t N = V.size();
    if (N == 0) return;
    for (size_t i = 0; i < N-1; ++i)
    {
      const size_t j = (SafeRand() % (N-i-1)) + i;
      std::swap(V[i], V[j]);
    }
  }

}
