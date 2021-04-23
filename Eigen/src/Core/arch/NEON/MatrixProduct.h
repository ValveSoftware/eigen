// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2021 Everton Constantino (everton.constantino@hotmail.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIX_PRODUCT_NEON_H
#define EIGEN_MATRIX_PRODUCT_NEON_H

#ifdef __DEBUG__
#include <iostream>
#endif

namespace Eigen {

namespace internal {

template<int Architecture, int CPU, typename LhsScalar, typename RhsScalar>
constexpr int SHAPES_COUNT = 4;

constexpr int SHAPES_DIMENSION = 6;
constexpr int SHAPES_LHS_DIMENSION = 0;
constexpr int SHAPES_DEP_DIMENSION = 1;
constexpr int SHAPES_RHS_DIMENSION = 2;
constexpr int SHAPES_RHS_POINTER = 3;
constexpr int SHAPES_LHS_POINTER = 4;
constexpr int SHAPES_DEP_POINTER = 5;
constexpr int SHAPES_POINTER_END = -1;

template<int Architecture, int CPU, typename Scalar, bool isLhs>
constexpr int PACK_SHAPES_COUNT = 2;
constexpr int PACK_SHAPES_DIMENSION = 3;
constexpr int PACK_SHAPES_POINTER = 2;
constexpr int PACK_SHAPES_END = -1;

// lhs_progress x depth_progress x rhs_progress (depth_progress > 1 matrix ops) x pointer to next rhs_progress on the shapes map
template<int Architecture, int CPU, typename LhsScalar, typename RhsScalar>
constexpr int SHAPES[SHAPES_COUNT<Architecture, CPU, LhsScalar,RhsScalar>][SHAPES_DIMENSION] = 
  { {1,1,1,SHAPES_POINTER_END, SHAPES_POINTER_END, SHAPES_POINTER_END},
    {4,1,1,                 0,                  0, SHAPES_POINTER_END},
    {1,1,4,                 1, SHAPES_POINTER_END, SHAPES_POINTER_END},
    {4,1,4,                 1,                  2, SHAPES_POINTER_END}};

// d1progress x d2progress
template<int Architecture, int CPU, typename Scalar, bool isLhs>
constexpr int PACK_SHAPES[PACK_SHAPES_COUNT<Architecture, CPU, Scalar, isLhs>][PACK_SHAPES_DIMENSION] = {{1,1,PACK_SHAPES_END},{4,1,0}};

template<int Architecture, int CPU, typename Index, typename Scalar, bool isLhs, typename DataMapper, bool Conjugate, bool PanelMode, int StorageOrder, int M, int N>
struct PackingOperator
{
  EIGEN_STRONG_INLINE Scalar* operator()(Index d1Idx, Index d2Idx, Scalar *block, const DataMapper& data)
  {
#ifdef __DEBUG__
    std::cout << M << "x" << N << " ( " << d1Idx << ", " << d2Idx <<") -> ( " << d1Idx + M << ", " << d2Idx + N << ") ";
#endif
    Scalar *c = block;
    for(auto i = 0; i < M; i++)
      for(auto j = 0; j < N; j++)
      {
        if(isLhs)
          *c = data(d1Idx + i, d2Idx + j);
        else
          *c = data(d2Idx + j, d1Idx + i);
#ifdef __DEBUG__
        std::cout << *c << " ";
#endif
        c++;
      }
#ifdef __DEBUG__
    std::cout << std::endl;
#endif
    return c;
  }
};

template<int Architecture, int CPU, typename Index, typename Scalar, bool isLhs, typename DataMapper, bool Conjugate, bool PanelMode, int StorageOrder, int D1PROGRESS, int IDX>
struct PackingInnerStruct
{
  EIGEN_STRONG_INLINE Scalar* operator()(Index d1Idx, Index d2Idx, Scalar *block, const DataMapper& data, Index d1Size, Index d2Size, Index stride, Index offset)
  {
    constexpr auto d2Progress = PACK_SHAPES<Architecture, CPU, Scalar, isLhs>[IDX][1];
    PackingOperator<Architecture, CPU, Index, Scalar, isLhs, DataMapper, Conjugate, PanelMode, StorageOrder, D1PROGRESS, d2Progress> po;

    for(;d2Idx + d2Progress <= d2Size; d2Idx+=d2Progress)
    {
      block = po(d1Idx, d2Idx, block, data);
    }

    if(PACK_SHAPES<Architecture, CPU, Scalar, isLhs>[IDX-1][0] == D1PROGRESS)
    {
      PackingInnerStruct<Architecture, CPU, Index, Scalar, isLhs, DataMapper, Conjugate, PanelMode, StorageOrder, D1PROGRESS, IDX-1> pis;
      block = pis(d1Idx, d2Idx, block, data, d1Size, d2Size, stride, offset);
    }
    return block;
  }
};

template<int Architecture, int CPU, typename Index, typename Scalar, bool isLhs, typename DataMapper, bool Conjugate, bool PanelMode, int StorageOrder, int D1PROGRESS>
struct PackingInnerStruct<Architecture, CPU, Index, Scalar, isLhs, DataMapper, Conjugate, PanelMode, StorageOrder, D1PROGRESS, 0>
{
  EIGEN_STRONG_INLINE Scalar* operator()(Index d1Idx, Index d2Idx, Scalar *block, const DataMapper& data, Index d1Size, Index d2Size, Index stride, Index offset)
  {
    constexpr auto d2Progress = PACK_SHAPES<Architecture, CPU, Scalar, isLhs>[0][1];
    for(;d2Idx + d2Progress <= d2Size; d2Idx+=d2Progress)
    {
      PackingOperator<Architecture, CPU, Index, Scalar, isLhs, DataMapper, Conjugate, PanelMode, StorageOrder, D1PROGRESS, d2Progress> po;
      block = po(d1Idx, d2Idx, block, data);
    }
    return block;
  }
};

template<int Architecture, int CPU, typename Index, typename Scalar, bool isLhs, typename DataMapper, bool Conjugate, bool PanelMode, int StorageOrder, int PACK_SHAPE_IDX>
struct PackingStruct
{
  PackingStruct<Architecture, CPU, Index, Scalar, isLhs, DataMapper, Conjugate, PanelMode, StorageOrder, PACK_SHAPES<Architecture, CPU, Scalar, isLhs>[PACK_SHAPE_IDX][PACK_SHAPES_POINTER]> ps;

  EIGEN_STRONG_INLINE Scalar* operator()(Index d1Idx, Scalar *block, const DataMapper& data, Index d1Size, Index d2Size, Index stride, Index offset)
  {
    constexpr auto d1Progress = PACK_SHAPES<Architecture, CPU, Scalar, isLhs>[PACK_SHAPE_IDX][0];

    for(; d1Idx + d1Progress <= d1Size; d1Idx += d1Progress)
    {
      PackingInnerStruct<Architecture, CPU, Index, Scalar, isLhs, DataMapper, Conjugate, PanelMode, StorageOrder, d1Progress, PACK_SHAPE_IDX> pis;
      block = pis(d1Idx, 0, block, data, d1Size, d2Size, stride, offset);
    }
    return ps(d1Idx, block, data, d1Size, d2Size, stride, offset);
  }
};

template<int Architecture, int CPU, typename Index, typename Scalar, bool isLhs, typename DataMapper, bool Conjugate, bool PanelMode, int StorageOrder>
struct PackingStruct<Architecture, CPU, Index, Scalar, isLhs, DataMapper, Conjugate, PanelMode, StorageOrder, -1>
{
  EIGEN_STRONG_INLINE Scalar* operator()(Index, Scalar *block, const DataMapper&, Index, Index, Index, Index) { return block; }
};

template<int Architecture, int CPU, typename Index, typename Scalar, typename DataMapper, bool Conjugate, bool PanelMode, int StorageOrder>
struct lhs_pack
{
  EIGEN_STRONG_INLINE void operator()(Scalar *blockA, const DataMapper &lhs, Index depth, Index rows, Index stride, Index offset)
  {
    PackingStruct<Architecture, CPU, Index, Scalar, true, DataMapper, Conjugate, PanelMode, StorageOrder, PACK_SHAPES_COUNT<Architecture, CPU, Scalar, true>-1> ps;
    ps(0, blockA, lhs, rows, depth, stride, offset);
  }
};

template<int Architecture, int CPU, typename Index, typename Scalar, typename DataMapper, bool Conjugate, bool PanelMode, int StorageOrder>
struct rhs_pack
{
  EIGEN_STRONG_INLINE void operator()(Scalar *blockB, const DataMapper &rhs, Index depth, Index cols, Index stride, Index offset)
  {
    PackingStruct<Architecture, CPU, Index, Scalar, false, DataMapper, Conjugate, PanelMode, StorageOrder, PACK_SHAPES_COUNT<Architecture, CPU, Scalar, false>-1> ps;
    ps(0, blockB, rhs, cols, depth, stride, offset);
  }
};

template<int Architecture, int CPU, typename Index, typename Scalar, typename DataMapper, bool isLhs, int IDX>
struct PackMapCalculator
{
  PackMapCalculator<Architecture, CPU, Index, Scalar, DataMapper, isLhs, PACK_SHAPES<Architecture, CPU, Scalar, isLhs>[IDX][PACK_SHAPES_POINTER]> pmc;
  EIGEN_STRONG_INLINE Index getPosition(Index pos, Index d2Size)
  {
    constexpr auto d1Progress = PACK_SHAPES<Architecture, CPU, Scalar, isLhs>[IDX][0];
    Index v = (pos / d1Progress) * d1Progress;
    return v*d2Size + pmc.getPosition(pos - v, d2Size);
  }
};

template<int Architecture, int CPU, typename Index, typename Scalar, typename DataMapper, bool isLhs>
struct PackMapCalculator<Architecture, CPU, Index, Scalar, DataMapper, isLhs, -1>
{
  EIGEN_STRONG_INLINE Index getPosition(Index, Index) { return Index(0); }
};

template<int Architecture, int CPU, typename Index, typename Scalar, typename DataMapper, bool isLhs>
struct PackMap
{
  const Scalar *pBase;
  const Scalar *pCur;
  Index stride;
  Index offset;
  Index d2Size;
  PackMapCalculator<Architecture, CPU, Index, Scalar, DataMapper, isLhs, PACK_SHAPES_COUNT<Architecture, CPU, Scalar, isLhs>-1> pmc;

  PackMap(const Scalar *base, Index d2Size, Index stride, Index offset) : pBase(base), pCur(base), d2Size(d2Size), stride(stride), offset(offset) {}

  EIGEN_STRONG_INLINE void resetCur() { pCur = pBase; }
  EIGEN_STRONG_INLINE void moveTo(Index p1) { pCur = pBase + pmc.getPosition(p1, d2Size); }
  EIGEN_STRONG_INLINE void advance(int progress) { pCur += progress; }
};

template<int Architecture, int CPU, typename Scalar, typename ResScalar, typename DataMapper, int M, int N>
struct Accumulator
{
  Scalar dt[M][N];

  EIGEN_STRONG_INLINE void zero()
  {
    for(auto i = 0; i < M; i++)
    {
      for(auto j = 0; j < N; j++)
      {
        dt[i][j] = Scalar(0);
      }
    }
  }

  template<typename ResPacket>
  EIGEN_STRONG_INLINE void scale(ResScalar alpha, const ResPacket& pAlpha)
  {
    for(auto i = 0; i < M; i++)
    {
      for(auto j = 0; j < N; j++)
      {
        dt[i][j] *= alpha;
      }
    }
  }

  EIGEN_STRONG_INLINE void store(const DataMapper& dest, Index row, Index col)
  {
    for(auto i = 0; i < M; i++)
    {
      for(auto j = 0; j < N; j++)
      {
        dest(row + i, col + j) += dt[i][j];
      }
    }
  }
};

template<int Architecture, int CPU, typename Index, typename LhsScalar, typename LhsPackMap, typename RhsScalar, typename RhsPackMap, typename AccScalar, typename ResScalar, typename Accumulator, int M, int K, int N>
struct MicroKernel
{
  EIGEN_STRONG_INLINE void operator()(LhsPackMap& lhsPackMap, 
                                      RhsPackMap& rhsPackMap, 
                                      Index rowIdx, Index colIdx, Index depthIdx,
                                      Accumulator& acc)
  {
#ifdef __DEBUG__
    std::cout << "Kernel " << M << " x " << K << " x " << N << " @ " << rowIdx << ", " << depthIdx << ", " << colIdx << std::endl;
    std::cout << "LHS ";
    for(auto i = 0; i < M; i++)
    {
      for(auto j = 0; j < K; j++)
      {
        std::cout << lhsPackMap.pCur[i*K + j] << " ";
      }
    }
    std::cout << std::endl << "RHS ";
    for(auto i = 0; i < K; i++)
    {
      for(auto j = 0; j < N; j++)
      {
        std::cout << rhsPackMap.pCur[i*N + j] << " ";
      }
    }
    std::cout << std::endl;
#endif
    const RhsScalar *pRhs = rhsPackMap.pCur;
    for(auto i = 0; i < N; i++)
    {
      const LhsScalar *pLhs = lhsPackMap.pCur;
      for(auto j = 0; j < M; j++)
      {
        acc.dt[j][i] += pRhs[i]*pLhs[j];
      }
    }
    lhsPackMap.advance(M*K);
    rhsPackMap.advance(K*N);
  };
};

template<int Architecture, int CPU, typename Index, typename LhsScalar, typename LhsPackMap, typename RhsScalar, typename RhsPackMap, typename AccScalar, typename ResScalar, typename ResPacket, typename DataMapper, int RHS_SHAPE_IDX, int LHS_SHAPE_IDX, int IDX>
struct DepthLoopStruct
{
  static constexpr auto PREVIOUS = SHAPES<Architecture, CPU, LhsScalar, RhsScalar>[IDX][SHAPES_DEP_POINTER];

  DepthLoopStruct<Architecture, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, ResPacket, DataMapper, RHS_SHAPE_IDX, LHS_SHAPE_IDX, PREVIOUS> depthLS;

  EIGEN_STRONG_INLINE void operator()(Index rowIdx, Index colIdx, Index depthIdx, const DataMapper& res,
                          Index rows, Index depth, Index cols, ResScalar alpha, const ResPacket& pAlpha, LhsPackMap& lhsPackMap, RhsPackMap& rhsPackMap)
  {
    constexpr auto rhsProgress      = SHAPES<Architecture, CPU, LhsScalar, RhsScalar>[RHS_SHAPE_IDX][SHAPES_RHS_DIMENSION];
    constexpr auto lhsProgress      = SHAPES<Architecture, CPU, LhsScalar, RhsScalar>[LHS_SHAPE_IDX][SHAPES_LHS_DIMENSION];
    constexpr auto depthProgress    = SHAPES<Architecture, CPU, LhsScalar, RhsScalar>[IDX][SHAPES_DEP_DIMENSION];

    typedef Accumulator<Architecture, CPU, AccScalar, ResScalar, DataMapper, lhsProgress, rhsProgress> AccumulatorType;

    MicroKernel<Architecture, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, AccumulatorType, lhsProgress, depthProgress, rhsProgress> mkt;
    AccumulatorType acc;
    acc.zero();
    for(; depthIdx + depthProgress <= depth; depthIdx+=depthProgress)
    {
        mkt(lhsPackMap, rhsPackMap, rowIdx, colIdx, depthIdx, acc);
    }
    acc.scale(alpha, pAlpha);
    acc.store(res, rowIdx, colIdx);

    depthLS(rowIdx, colIdx, depthIdx, res, rows, depth, cols, alpha, pAlpha, lhsPackMap, rhsPackMap);
  }
};

template<int Architecture, int CPU, typename Index, typename LhsScalar, typename LhsPackMap, typename RhsScalar, typename RhsPackMap, typename AccScalar, typename ResScalar, typename ResPacket, typename DataMapper, int RHS_SHAPE_IDX, int LHS_SHAPE_IDX>
struct DepthLoopStruct<Architecture, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, ResPacket, DataMapper, RHS_SHAPE_IDX, LHS_SHAPE_IDX, -1>
{
  EIGEN_STRONG_INLINE void operator()(Index, Index, Index, const DataMapper&,
                          Index, Index, Index, ResScalar, const ResPacket&, LhsPackMap&, RhsPackMap&) {}
};

template<int Architecture, int CPU, typename Index, typename LhsScalar, typename LhsPackMap, typename RhsScalar, typename RhsPackMap, typename AccScalar, typename ResScalar, typename ResPacket, typename DataMapper, int RHS_SHAPE_IDX, int IDX>
struct LhsLoopStruct
{
  static constexpr auto PREVIOUS = SHAPES<Architecture, CPU, LhsScalar, RhsScalar>[IDX][SHAPES_LHS_POINTER];
  LhsLoopStruct<Architecture, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, ResPacket, DataMapper, RHS_SHAPE_IDX, PREVIOUS> lhsLS;

  EIGEN_STRONG_INLINE void operator()(Index rowIdx, int colIdx, const DataMapper& res,
                          Index rows, Index depth, Index cols, ResScalar alpha, const ResPacket& pAlpha, LhsPackMap& lhsPackMap, RhsPackMap& rhsPackMap)
  {
    constexpr auto lhsProgress = SHAPES<Architecture, CPU, LhsScalar, RhsScalar>[IDX][SHAPES_LHS_DIMENSION];

    DepthLoopStruct<Architecture, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, ResPacket, DataMapper, RHS_SHAPE_IDX, IDX, IDX> depthLS;
    for(;rowIdx + lhsProgress <= rows; rowIdx+=lhsProgress)
    {
      lhsPackMap.moveTo(rowIdx);
      rhsPackMap.moveTo(colIdx);
      depthLS(rowIdx, colIdx, 0, res, rows, depth, cols, alpha, pAlpha, lhsPackMap, rhsPackMap);
    }
    lhsLS(rowIdx, colIdx, res, rows, depth, cols, alpha, pAlpha, lhsPackMap, rhsPackMap);
  }
};

template<int Architecture, int CPU, typename Index, typename LhsScalar, typename LhsPackMap, typename RhsScalar, typename RhsPackMap, typename AccScalar, typename ResScalar, typename ResPacket, typename DataMapper, int RHS_SHAPE_IDX>
struct LhsLoopStruct<Architecture, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, ResPacket, DataMapper, RHS_SHAPE_IDX, -1>
{
  EIGEN_STRONG_INLINE void operator()(Index, Index, const DataMapper&,
                          Index, Index, Index, ResScalar, const ResPacket&, LhsPackMap&, RhsPackMap&) {}
};

template<int Architecture, int CPU, typename Index, typename LhsScalar, typename LhsPackMap, typename RhsScalar, typename RhsPackMap, typename AccScalar, typename ResScalar, typename ResPacket, typename DataMapper, int IDX>
struct RhsLoopStruct
{
  static constexpr auto PREVIOUS = SHAPES<Architecture, CPU, LhsScalar, RhsScalar>[IDX][SHAPES_RHS_POINTER];
  RhsLoopStruct<Architecture, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, ResPacket, DataMapper, PREVIOUS> rhsLS;

  EIGEN_STRONG_INLINE void operator()(Index colIdx, const DataMapper& res,
                          Index rows, Index depth, Index cols, ResScalar alpha, const ResPacket& pAlpha, LhsPackMap& lhsPackMap, RhsPackMap& rhsPackMap)
  {
    constexpr auto rhsProgress = SHAPES<Architecture, CPU, LhsScalar, RhsScalar>[IDX][SHAPES_RHS_DIMENSION];

    for(;colIdx + rhsProgress <= cols; colIdx+=rhsProgress)
    {
      LhsLoopStruct<Architecture, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, ResPacket, DataMapper, IDX, IDX> lhsLS;
      lhsLS(0, colIdx, res, rows, depth, cols, alpha, pAlpha, lhsPackMap, rhsPackMap);
    }
    rhsLS(colIdx, res, rows, depth, cols, alpha, pAlpha, lhsPackMap, rhsPackMap);
  }
};

template<int Architecture, int CPU, typename Index, typename LhsScalar, typename LhsPackMap, typename RhsScalar, typename RhsPackMap, typename AccScalar, typename ResScalar, typename ResPacket, typename DataMapper>
struct RhsLoopStruct<Architecture, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, ResPacket, DataMapper, -1>
{
  EIGEN_STRONG_INLINE void operator()(Index colIdx, const DataMapper&,
                          Index, Index, Index, ResScalar, const ResPacket&, LhsPackMap&, RhsPackMap&) {}
};

template<int Architecture, int CPU, typename ResScalar, typename AccScalar, typename LhsScalar, typename RhsScalar, typename Index, typename DataMapper>
EIGEN_STRONG_INLINE void gemm(const DataMapper& res, const LhsScalar* blockA, const RhsScalar* blockB,
          Index rows, Index depth, Index cols, ResScalar alpha, Index strideA, Index strideB, Index offsetA, Index offsetB)
{
  using ResPacket = typename unpacket_traits<ResScalar>::type;
  typedef PackMap<Architecture, CPU, Index, LhsScalar, DataMapper, true> LhsPackMap;
  typedef PackMap<Architecture, CPU, Index, RhsScalar, DataMapper, false> RhsPackMap;

#ifdef __DEBUG__
  std::cout << "blockA" << std::endl;
  for(auto i = 0; i < rows*depth; i++)
  {
    if(i % 4 == 0 && i > 0)
      std::cout << std::endl;
    std::cout << blockA[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "blockB" << std::endl;
  for(auto i = 0; i < depth*cols; i++)
  {
    if(i % 4 == 0 && i > 0)
      std::cout << std::endl;
    std::cout << blockB[i] << " ";
  }
  std::cout << std::endl;
#endif
  asm __volatile__("#BEGING_GEBP\n\t");
  RhsLoopStruct<Architecture, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, ResPacket, DataMapper, SHAPES_COUNT<0, 0, LhsScalar, RhsScalar>-1> rhsLS;
  LhsPackMap lhsPackMap(blockA, depth, strideA, offsetA);
  RhsPackMap rhsPackMap(blockB, depth, strideB, offsetB);

  ResPacket pAlpha = pset1<ResPacket>(alpha);

  rhsLS(0, res, rows, depth, cols, alpha, pAlpha, lhsPackMap, rhsPackMap);
  asm __volatile__("#END_GEBP\n\t");
}

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<float, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>
{
  void operator()(float* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
void gemm_pack_rhs<float, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>
  ::operator()(float* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset)
{
  rhs_pack<0, 0, Index, float, DataMapper, Conjugate, PanelMode, ColMajor> pack;
  pack(blockB, rhs, depth, cols, stride, offset);
}

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<float, Index, DataMapper, nr, RowMajor, Conjugate, PanelMode>
{
  void operator()(float* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
void gemm_pack_rhs<float, Index, DataMapper, nr, RowMajor, Conjugate, PanelMode>
  ::operator()(float* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset)
{
  rhs_pack<0, 0, Index, float, DataMapper, Conjugate, PanelMode, RowMajor> pack;
  pack(blockB, rhs, depth, cols, stride, offset);
}

template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<float, Index, DataMapper, Pack1, Pack2, Packet, RowMajor, Conjugate, PanelMode>
{
  void operator()(float* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
void gemm_pack_lhs<float, Index, DataMapper, Pack1, Pack2, Packet, RowMajor, Conjugate, PanelMode>
  ::operator()(float* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset)
{
  lhs_pack<0, 0, Index, float, DataMapper, Conjugate, PanelMode, RowMajor> pack;
  pack(blockA, lhs, depth, rows, stride, offset);
}

template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<float, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode>
{
  void operator()(float* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
void gemm_pack_lhs<float, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode>
  ::operator()(float* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset)
{
  lhs_pack<0, 0, Index, float, DataMapper, Conjugate, PanelMode, ColMajor> pack;
  pack(blockA, lhs, depth, rows, stride, offset);
}

template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<float, float, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
{
  void operator()(const DataMapper& res, const float* blockA, const float* blockB,
                  Index rows, Index depth, Index cols, float alpha,
                  Index strideA=-1, Index strideB=-1, Index offsetA=0, Index offsetB=0);
};

template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
void gebp_kernel<float, float, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
  ::operator()(const DataMapper& res, const float* blockA, const float* blockB,
               Index rows, Index depth, Index cols, float alpha,
               Index strideA, Index strideB, Index offsetA, Index offsetB)
  {
    gemm<0, 0, float, float, float, float, Index, DataMapper>(res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB);
  }
} // end namespace internal

} // end namespace Eigen
#endif // EIGEN_MATRIX_PRODUCT_NEON_H