#pragma once

#include <array>
#include <vector>
#include <functional>
#include <type_traits>
#include <iostream>
#include <cmath>
#include <memory>
#include <cassert>

namespace staticnet
{

  void BringError(double dif);

  template<bool B, class LHS, class RHS>
  struct ConditionalMatrixMultiplication
  {};
  template<class LHS, class RHS>
  struct ConditionalMatrixMultiplication< true, LHS, RHS >
  {
    typedef decltype(typename LHS::MatrixType()* typename RHS::MatrixType()) type;
  };
  template<class LHS, class RHS>
  auto operator*(const LHS& lhs, const RHS& rhs) ->
    typename ConditionalMatrixMultiplication<
    !std::is_same<LHS, typename LHS::MatrixType>::value ||
    !std::is_same<RHS, typename RHS::MatrixType>::value,
    LHS, RHS
    >::type
  {
    return static_cast<typename LHS::MatrixType>(lhs) * static_cast<typename RHS::MatrixType>(rhs);
  }

  template<size_t N>
  using SIZET = std::integral_constant<size_t, N>;

  namespace NMatrix
  {
    template<bool B>
    struct DataType {};
    template<>
    struct DataType<true>// true = vector
    {
      template<size_t N, size_t M>
      auto operator()(SIZET<N> n, SIZET<M> m) { return std::vector<std::array<double, M>>(N); }
    };
    template<>
    struct DataType<false>// false = array
    {
      template<size_t N, size_t M>
      auto operator()(SIZET<N> n, SIZET<M> m) { return std::array<std::array<double, M>, N>{}; }
    };
    constexpr bool IsTooLarge(size_t N)
    {
      return N > 1000;
    }
  }

  template<size_t N, size_t M>
  class Matrix
  {
  public:
    using MatrixType = Matrix<N, M>;
    using GradientType = MatrixType;
    constexpr static size_t _N = N;
    constexpr static size_t _M = M;
    using DataType = typename NMatrix::DataType< NMatrix::IsTooLarge(N * M) >;

    explicit Matrix(double v = 0.0): _data(DataType()(SIZET<N>(), SIZET<M>())) { fill(v); }
    //  Matrix(const std::array<double, N>& data) { static_assert(M == 1); for (int n = 0; n<N; ++n) _data[n][0] = data[n]; }
    Matrix(const Matrix& Other): _data(Other._data) {}
    Matrix(Matrix&& Other) noexcept: _data(std::move(Other._data)) {}
    ~Matrix() {}

    Matrix& operator=(const Matrix& Other) { _data = Other._data; return *this; }
    Matrix& operator=(Matrix&& Other) { _data = std::move(Other._data); return *this; }

    Matrix<N, M>& operator-=(const Matrix<N, M>& Other);
    Matrix<N, M>& operator+=(const Matrix<N, M>& Other);
    Matrix<N, M> operator+(const Matrix<N, M>& Other) const;
    Matrix<N, M> operator-(const Matrix<N, M>& Other) const;
    template<size_t P>
    Matrix<N, P> operator*(const Matrix<M, P>& Other) const;
    std::array<double, N> operator*(const std::array<double, M>& X) const;
    Matrix<N, M>& operator*=(double x);
    Matrix<N, M> operator*(double x) const;

    inline auto& operator[] (size_t i) { return _data[i]; }
    inline const auto& operator[] (size_t i) const { return _data[i]; }

    Matrix<M, N> GetTranspose() const;
    Matrix<N, M> MultiplyEachByTranspose(const Matrix<M, N>& Other) const;

    void SubtractEachRow(const Matrix<1, M>& Dif);
    inline void fill(double x) { for (auto& Row : _data) Row.fill(x); }
    void Randomize();

    decltype(DataType()(SIZET<N>(), SIZET<M>())) _data;
  };

  template<size_t N, size_t M>
  Matrix<N, M>& Matrix<N, M>::operator-=(const Matrix<N, M>& Other)
  {
    for (int n = 0; n < N; ++n) for (int m = 0; m < M; ++m)
    {
      _data[n][m] -= Other._data[n][m];
      if (Other._data[n][m] > 100 || Other._data[n][m] < -100)
        BringError(Other._data[n][m]);
    }
    return *this;
  }
  template<size_t N, size_t M>
  Matrix<N, M>& Matrix<N, M>::operator+=(const Matrix<N, M>& Other)
  {
    for (int n = 0; n < N; ++n) for (int m = 0; m < M; ++m)
    {
      _data[n][m] += Other._data[n][m];
      if (Other._data[n][m] > 100 || Other._data[n][m] < -100)
        BringError(Other._data[n][m]);
    }
    return *this;
  }

  template<size_t N, size_t M>
  Matrix<N, M> Matrix<N, M>::operator+(const Matrix<N, M>& Other) const
  {
    Matrix<N, M> Ret;
    for (int n = 0; n < N; ++n) for (int m = 0; m < M; ++m)
      Ret[n][m] += _data[n][m] + Other._data[n][m];
    return Ret;
  }
  template<size_t N, size_t M>
  Matrix<N, M> Matrix<N, M>::operator-(const Matrix<N, M>& Other) const
  {
    Matrix<N, M> Ret;
    for (int n = 0; n < N; ++n) for (int m = 0; m < M; ++m)
      Ret[n][m] += _data[n][m] - Other._data[n][m];
    return Ret;
  }

  template<size_t N, size_t M>
  template<size_t P>
  Matrix<N, P> Matrix<N, M>::operator*(const Matrix<M, P>& Other) const
  {
    Matrix<N, P> Ret;
    if constexpr (P == 1) // Optimization
    {
      const auto OtherRow = std::move(Other.GetTranspose()._data[0]);
      for (int n = 0; n < N; ++n)
      {
        double vRetElement = 0.0;
        const auto& MyRow = _data[n];
        for (int m = 0; m < M; ++m)
        {
          vRetElement += MyRow[m] * OtherRow[m];
        }
        Ret[n][0] = vRetElement;
      }
    } else
    {
      for (int n = 0; n < N; ++n)
      {
        auto& RetRow = Ret[n];
        const auto& MyRow = _data[n];
        for (int m = 0; m < M; ++m)
        {
          const auto& MyElement = MyRow[m];
          const auto& OtherRow = Other[m];
          for (int p = 0; p < P; ++p)
            RetRow[p] += MyElement * OtherRow[p];
        }
      }
    }
    return Ret;
  }

  template<size_t N, size_t M>
  std::array<double, N> Matrix<N, M>::operator*(const std::array<double, M>& X) const
  {
    std::array<double, N> Ret = {};
    for (int n = 0; n < N; ++n)
    {
      const auto& MyRow = _data[n];
      double& vRet = Ret[n];
      for (int m = 0; m < M; ++m)
      {
        vRet += MyRow[m] * X[m];
      }
    }
    return Ret;
  }

  template<size_t N, size_t M>
  Matrix<N, M>& Matrix<N, M>::operator*=(double x)
  {
    for (int n = 0; n < N; ++n) for (int m = 0; m < M; ++m)
      (*this)[n][m] *= x;
    return *this;
  }
  template<size_t N, size_t M>
  Matrix<N, M> Matrix<N, M>::operator*(double x) const
  {
    Matrix<N, M> ret = *this;
    ret *= x;
    return ret;
  }

  template<size_t N, size_t M>
  inline Matrix<M, N> Matrix<N, M>::GetTranspose() const
  {
    Matrix<M, N> Ret;
    for (int n = 0; n < N; ++n) for (int m = 0; m < M; ++m)
      Ret[m][n] = (*this)[n][m];
    return Ret;
  }
  template<size_t N>
  Matrix<1, N> GetTranspose(const std::array<double, N>& X)
  {
    Matrix<1, N> Ret;
    Ret[0] = X;
    return Ret;
  }
  template<size_t N, size_t M>
  Matrix<M, N> GetTranspose(const Matrix<N, M>& X)
  {
    return X.GetTranspose();
  }

  template<size_t N, size_t M>
  Matrix<N, M> Matrix<N, M>::MultiplyEachByTranspose(const Matrix<M, N>& Other) const
  {
    Matrix<N, M> Ret;
    for (int n = 0; n < N; ++n) for (int m = 0; m < M; ++m)
      Ret[n][m] = (*this)[n][m] * Other[m][n];
    return Ret;
  }

  double GetSignedUnitRand();
  std::vector<double> GetRandVector(size_t N);
  template<size_t N, size_t M>
  void Matrix<N, M>::Randomize()
  {
    for (auto& Row : _data) for (double& x : Row)
      x = GetSignedUnitRand();
  }

  template<size_t N, size_t M>
  void Matrix<N, M>::SubtractEachRow(const Matrix<1, M>& Dif)
  {
    for (int n = 0; n < N; ++n) for (int m = 0; m < M; ++m)
      _data[n][m] -= Dif[0][m];
  }

  double Get(const Matrix<1, 1>& M);

  template<size_t N>
  std::array<double, N> Transform(const std::array<double, N>& Mat, std::function<double(double)> f)
  {
    std::array<double, N> Ret;
    for (int n = 0; n < N; ++n)
      Ret[n] = f(Mat[n]);
    return Ret;
  }
  template<size_t N, size_t M>
  Matrix<N, M> Transform(const Matrix<N, M>& Mat, std::function<double(double)> f)
  {
    Matrix<N, M> Ret;
    for (int n = 0; n < N; ++n) for (int m = 0; m < M; ++m)
      Ret[n][m] = f(Mat[n][m]);
    return Ret;
  }
  template<size_t N, size_t M>
  Matrix<M, N> TransformTranspose(const Matrix<N, M>& Mat, std::function<double(double)> f)
  {
    Matrix<M, N> Ret;
    for (int n = 0; n < N; ++n) for (int m = 0; m < M; ++m)
      Ret[m][n] = f(Mat[n][m]);
    return Ret;
  }

  template<size_t N>
  class DiagonalMatrix
  {
  public:
    DiagonalMatrix() {}
    DiagonalMatrix(double v) { _Diagonal.fill(v); }
    Matrix<N, N> CreateMatrix() const
    {
      Matrix<N, N> Ret(0.0);
      for (int n = 0; n < N; ++n)
        Ret[n][n] = _Diagonal[n];
      return Ret;
    }
    std::array<double, N> _Diagonal;
  };

  template<size_t N>
  DiagonalMatrix<N> DiagonalMatrixFromCol(const Matrix<N, 1>& Col)
  {
    DiagonalMatrix<N> Ret;
    for (int n = 0; n < N; ++n)
      Ret._Diagonal[n] = Col[n][0];
    return Ret;
  }
  template<size_t N>
  DiagonalMatrix<N> DiagonalMatrixFromRow(const Matrix<1, N>& Row)
  {
    DiagonalMatrix<N> Ret;
    for (int n = 0; n < N; ++n)
      Ret._Diagonal[n] = Row[0][n];
    return Ret;
  }

  template<size_t N, size_t M>
  Matrix<N, M> ColToNRows(const Matrix<M, 1>& Col, SIZET<N> Dummy)
  {
    Matrix<N, M> Ret;
    for (int n = 0; n < N; ++n) for (int m = 0; m < M; ++m)
      Ret[n][m] = Col[m][0];
    return Ret;
  }

  Matrix<1, 1> Mat11(double x);

  template<size_t N, size_t M>
  Matrix<N, M>& operator*=(Matrix<N, M>& Mat, const DiagonalMatrix<M>& D)
  {
    for (int n = 0; n < N; ++n)
    {
      for (int m = 0; m < M; ++m)
      {
        Mat[n][m] *= D._Diagonal[m];
      }
    }
    return Mat;
  }
  template<size_t N, size_t M>
  Matrix<N, M> operator*(const Matrix<N, M>& Mat, const DiagonalMatrix<M>& D)
  {
    auto Ret = Mat;
    Ret *= D;
    return Ret;
  }

  template<size_t N>
  std::array<double, N> operator+(const std::array<double, N>& lhs, const Matrix<N, 1>& rhs)
  {
    std::array<double, N> Ret = lhs;
    for (int n = 0; n < N; ++n)
      Ret[n] += rhs[n][0];
    return Ret;
  }

  template<size_t N, size_t M, size_t NUM_SHARED>
  class SparseMatrix
  {
  private:
    std::array<double, NUM_SHARED> _SharedValues;
  public:
    using MatrixType = SparseMatrix<N, M, NUM_SHARED>;
    constexpr static size_t _N = N;
    constexpr static size_t _M = M;
    struct SEntry
    {
      SEntry(size_t n, size_t m, size_t sharedIndex): _n(n), _m(m), _sharedIndex(sharedIndex) { assert(sharedIndex < NUM_SHARED); }
      const size_t _sharedIndex; // The index of the shared value in _SharedValues
      size_t _n; size_t _m; // Row and column indices for this entry
      inline double& value(SparseMatrix& parent) { return parent._SharedValues[_sharedIndex]; }
      inline const double& value(const SparseMatrix& parent) const { return parent._SharedValues[_sharedIndex]; }
    };
    std::vector< SEntry > _entries;
    SparseMatrix() { _SharedValues.fill(0.0); }
    explicit SparseMatrix(std::vector<double> SharedValues): _SharedValues(std::move(SharedValues)) {}
    explicit SparseMatrix(const decltype(_entries)& entries): _entries(entries) {}

    void fill(double v) { for (double& val : _SharedValues) val = v; }
    void Randomize() { for (double& val : _SharedValues) val = GetSignedUnitRand(); }

    std::array<double, N> operator*(const std::array<double, M>& X) const;
    template<size_t P>
    Matrix<N, P> operator*(const Matrix<M, P>& Other) const;
    SparseMatrix& operator-=(const Matrix<N, M>& Other);
    SparseMatrix& operator-=(const SparseMatrix<N, M, NUM_SHARED>& Other);
    SparseMatrix& operator+=(const SparseMatrix<N, M, NUM_SHARED>& Other);
    SparseMatrix& operator*=(double v);

    struct GradientType {
      GradientType() { _gradients.fill(0.0); }
      std::array<double, NUM_SHARED> _gradients;
      GradientType& operator*=(double v) { for (double& g : _gradients) g *= v; return *this; }
      GradientType& operator+=(const GradientType& o) { for (size_t i = 0; i < NUM_SHARED; ++i) _gradients[i] += o._gradients[i]; return *this; }
      void fill(double v) { _gradients.fill(v); }
    };
    SparseMatrix& operator-=(const typename SparseMatrix<N, M, NUM_SHARED>::GradientType& Other);
  };

  template<size_t N, size_t M, size_t NUM_SHARED>
  std::array<double, N> SparseMatrix<N, M, NUM_SHARED>::operator*(const std::array<double, M>& X) const
  {
    std::array<double, N> Ret = {};
    for (const auto& entry : _entries)
      Ret[entry._n] += X[entry._m] * entry.value(*this);
    return Ret;
  }
  template<size_t N, size_t M, size_t NUM_SHARED>
  template<size_t P>
  Matrix<N, P> SparseMatrix<N, M, NUM_SHARED>::operator*(const Matrix<M, P>& Other) const
  {
    Matrix<N, P> Ret(0.0);
    for (const auto& entry : _entries)
    {
      auto& RetRow = Ret[entry._n];
      auto& OtherRow = Other[entry._m];
      for (int p = 0; p < P; ++p)
        RetRow[p] += OtherRow[p] * entry.value();
    }
    return Ret;
  }
  template<size_t N, size_t M, size_t P, size_t NUM_SHARED>
  Matrix<N, P> operator*(const Matrix<N, M>& lhs, const SparseMatrix<M, P, NUM_SHARED>& rhs)
  {
    Matrix<N, P> Ret(0.0);
    for (int n = 0; n < N; ++n)
    {
      auto& RetRow = Ret[n];
      const auto& lhsRow = lhs[n];
      for (const auto& entry : rhs._entries)
      {
        const auto m = entry._n;
        const auto p = entry._m;
        RetRow[p] += lhsRow[m] * entry.value(rhs);
      }
    }
    return Ret;
  }
  template<size_t N, size_t M, size_t NUM_SHARED>
  SparseMatrix<N, M, NUM_SHARED>& SparseMatrix<N, M, NUM_SHARED>::operator*=(double v)
  {
    for (double& shared : _SharedValues)
      shared *= v;
    return *this;
  }
  template<size_t N, size_t M, size_t NUM_SHARED>
  SparseMatrix<N, M, NUM_SHARED>& SparseMatrix<N, M, NUM_SHARED>::operator-=(const typename SparseMatrix<N, M, NUM_SHARED>::GradientType& Other)
  {
    for (size_t i = 0; i < NUM_SHARED; ++i)
      _SharedValues[i] -= Other._gradients[i];
    return *this;
  }

  template<size_t N>
  std::array<double, N> operator-(const std::array<double, N> lhs, const std::array<double, N> rhs)
  {
    std::array<double, N> Ret;
    for (int i = 0; i < N; ++i)
      Ret[i] = lhs[i] - rhs[i];
    return Ret;
  }
  template<size_t N>
  std::array<double, N>& operator*=(std::array<double, N>& lhs, double rhs)
  {
    for (double& x : lhs)
      x *= rhs;
    return lhs;
  }
  template<size_t N>
  std::array<double, N>& operator*(const std::array<double, N>& lhs, double rhs)
  {
    auto Ret = lhs;
    Ret *= rhs;
    return Ret;
  }

}
