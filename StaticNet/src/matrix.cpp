#include "matrix.h"
#include "util.h"
#include <iostream>

namespace staticnet
{

  void BringError(double dif)
  {
    std::cout << "ERROR " << dif << std::endl;
    throw std::exception();
  }

  double GetSignedUnitRand() // Between -1 and 1
  {
    return (double(SafeRand() % (1LL << 16)) / (1LL << 16)) * (SafeRand() % 2 ? 1 : -1);
  }

  std::vector<double> GetRandVector(size_t N) // Between -1 and 1
  {
    std::vector<double> Ret;
    for (int i = 0; i < N; ++i)
      Ret.push_back(GetSignedUnitRand());
    return Ret;
  }

  double Get(const Matrix<1, 1>& M)
  {
    return M[0][0];
  }

  Matrix<1, 1> Mat11(double x) { return Matrix<1, 1>(x); }

}