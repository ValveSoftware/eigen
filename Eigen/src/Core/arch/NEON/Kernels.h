// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2021 Everton Constantino (everton.constantino@hotmail.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_KERNELS_NEON_H
#define EIGEN_KERNELS_NEON_H

namespace Eigen {

namespace internal {

template<int CPU, typename LhsScalar, typename RhsScalar>
constexpr int SHAPES_COUNT<0, CPU, LhsScalar, RhsScalar> = 7;

// lhs_progress x depth_progress x rhs_progress (depth_progress > 1 matrix ops) x pointer to next rhs_progress on the shapes map
template<int CPU, typename LhsScalar, typename RhsScalar>
constexpr int SHAPES<0, CPU, LhsScalar, RhsScalar>[SHAPES_COUNT<0, CPU, LhsScalar,RhsScalar>][SHAPES_DIMENSION] =
  { {1,1,1,SHAPES_POINTER_END, SHAPES_POINTER_END, SHAPES_POINTER_END},
    {4,1,1,                 0,                  0, SHAPES_POINTER_END},
    {1,1,4,                 1, SHAPES_POINTER_END, SHAPES_POINTER_END},
    {4,1,4,                 1,                  2, SHAPES_POINTER_END},
    {4,4,4,                 1,                  2,                  3},
    {8,1,4,                 1,                  4, SHAPES_POINTER_END},
    {8,4,4,                 1,                  4,                  5}};

template<int CPU, typename Scalar, typename ResScalar, typename DataMapper>
struct Accumulator<0, CPU, Scalar, ResScalar, DataMapper, 4, 1>
{
  using LinearMapper = typename DataMapper::LinearMapper;
  using AccPacket = typename packet_traits<Scalar>::type;
  using ResPacket = typename packet_traits<ResScalar>::type;

  AccPacket _acc;

  EIGEN_STRONG_INLINE void zero()
  {
    _acc = pset1<AccPacket>(0);
  }

  template<typename ResPacket_>
  EIGEN_STRONG_INLINE void scale(ResScalar alpha, const ResPacket_& pAlpha)
  {
    _acc *= pAlpha;
  }

  EIGEN_STRONG_INLINE void store(const DataMapper& dest, Index row, Index col)
  {
    LinearMapper r0 = dest.getLinearMapper(row, col + 0);

    r0.storePacket(0, r0.template loadPacket<ResPacket>(0) + _acc);
  }
};

template<int CPU, typename Scalar, typename ResScalar, typename DataMapper>
struct Accumulator<0, CPU, Scalar, ResScalar, DataMapper, 1, 4>
{
  using LinearMapper = typename DataMapper::LinearMapper;
  using AccPacket = typename packet_traits<Scalar>::type;
  using ResPacket = typename packet_traits<ResScalar>::type;

  AccPacket _acc;

  EIGEN_STRONG_INLINE void zero()
  {
    _acc = pset1<AccPacket>(0);
  }

  template<typename ResPacket_>
  EIGEN_STRONG_INLINE void scale(ResScalar alpha, const ResPacket_& pAlpha)
  {
    _acc *= pAlpha;
  }

  EIGEN_STRONG_INLINE void store(const DataMapper& dest, Index row, Index col)
  {
    dest(row, col + 0) = _acc[0];
    dest(row, col + 1) = _acc[1];
    dest(row, col + 2) = _acc[2];
    dest(row, col + 3) = _acc[3];
  }
};

template<int CPU, typename Scalar, typename ResScalar, typename DataMapper>
struct Accumulator<0, CPU, Scalar, ResScalar, DataMapper, 4, 4>
{
  using LinearMapper = typename DataMapper::LinearMapper;
  using AccPacket = typename packet_traits<Scalar>::type;
  using ResPacket = typename packet_traits<ResScalar>::type;

  PacketBlock<AccPacket, 4> _acc;

  EIGEN_STRONG_INLINE void zero()
  {
    _acc.packet[0] = pset1<AccPacket>(0);
    _acc.packet[1] = pset1<AccPacket>(0);
    _acc.packet[2] = pset1<AccPacket>(0);
    _acc.packet[3] = pset1<AccPacket>(0);
  }

  template<typename ResPacket_>
  EIGEN_STRONG_INLINE void scale(ResScalar alpha, const ResPacket_& pAlpha)
  {
    _acc.packet[0] *= pAlpha;
    _acc.packet[1] *= pAlpha;
    _acc.packet[2] *= pAlpha;
    _acc.packet[3] *= pAlpha;
  }

  EIGEN_STRONG_INLINE void store(const DataMapper& dest, Index row, Index col)
  {
    LinearMapper r0 = dest.getLinearMapper(row, col + 0);
    LinearMapper r1 = dest.getLinearMapper(row, col + 1);
    LinearMapper r2 = dest.getLinearMapper(row, col + 2);
    LinearMapper r3 = dest.getLinearMapper(row, col + 3);

    r0.storePacket(0, r0.template loadPacket<ResPacket>(0) + _acc.packet[0]);
    r1.storePacket(0, r1.template loadPacket<ResPacket>(0) + _acc.packet[1]);
    r2.storePacket(0, r2.template loadPacket<ResPacket>(0) + _acc.packet[2]);
    r3.storePacket(0, r3.template loadPacket<ResPacket>(0) + _acc.packet[3]);
  }
};

template<int CPU, typename Scalar, typename ResScalar, typename DataMapper>
struct Accumulator<0, CPU, Scalar, ResScalar, DataMapper, 8, 4>
{
  using LinearMapper = typename DataMapper::LinearMapper;
  using AccPacket = typename packet_traits<Scalar>::type;
  using ResPacket = typename packet_traits<ResScalar>::type;

  PacketBlock<AccPacket, 4> _acc1;
  PacketBlock<AccPacket, 4> _acc2;

  EIGEN_STRONG_INLINE void zero()
  {
    _acc1.packet[0] = pset1<AccPacket>(0);
    _acc1.packet[1] = pset1<AccPacket>(0);
    _acc1.packet[2] = pset1<AccPacket>(0);
    _acc1.packet[3] = pset1<AccPacket>(0);

    _acc2.packet[0] = pset1<AccPacket>(0);
    _acc2.packet[1] = pset1<AccPacket>(0);
    _acc2.packet[2] = pset1<AccPacket>(0);
    _acc2.packet[3] = pset1<AccPacket>(0);
  }

  template<typename ResPacket_>
  EIGEN_STRONG_INLINE void scale(ResScalar alpha, const ResPacket_& pAlpha)
  {
    _acc1.packet[0] *= pAlpha;
    _acc1.packet[1] *= pAlpha;
    _acc1.packet[2] *= pAlpha;
    _acc1.packet[3] *= pAlpha;

    _acc2.packet[0] *= pAlpha;
    _acc2.packet[1] *= pAlpha;
    _acc2.packet[2] *= pAlpha;
    _acc2.packet[3] *= pAlpha;
  }

  EIGEN_STRONG_INLINE void store(const DataMapper& dest, Index row, Index col)
  {
    LinearMapper r00 = dest.getLinearMapper(row + 0, col + 0);
    LinearMapper r01 = dest.getLinearMapper(row + 0, col + 1);
    LinearMapper r02 = dest.getLinearMapper(row + 0, col + 2);
    LinearMapper r03 = dest.getLinearMapper(row + 0, col + 3);

    LinearMapper r10 = dest.getLinearMapper(row + 4, col + 0);
    LinearMapper r11 = dest.getLinearMapper(row + 4, col + 1);
    LinearMapper r12 = dest.getLinearMapper(row + 4, col + 2);
    LinearMapper r13 = dest.getLinearMapper(row + 4, col + 3);


    r00.storePacket(0, r00.template loadPacket<ResPacket>(0) + _acc1.packet[0]);
    r01.storePacket(0, r01.template loadPacket<ResPacket>(0) + _acc1.packet[1]);
    r02.storePacket(0, r02.template loadPacket<ResPacket>(0) + _acc1.packet[2]);
    r03.storePacket(0, r03.template loadPacket<ResPacket>(0) + _acc1.packet[3]);

    r10.storePacket(0, r10.template loadPacket<ResPacket>(0) + _acc2.packet[0]);
    r11.storePacket(0, r11.template loadPacket<ResPacket>(0) + _acc2.packet[1]);
    r12.storePacket(0, r12.template loadPacket<ResPacket>(0) + _acc2.packet[2]);
    r13.storePacket(0, r13.template loadPacket<ResPacket>(0) + _acc2.packet[3]);
  }
};

#define MICRO_8x1x4() \
    pLhs = pload<LhsPacket>(lhsPackMap.pCur); \
    lhsPackMap.advance(4*1); \
    pLhs2 = pload<LhsPacket>(lhsPackMap.pCur); \
    pRhs = pload<RhsPacket>(rhsPackMap.pCur); \
    pRhs0 = pset1<RhsPacket>(pRhs[0]); \
    pRhs1 = pset1<RhsPacket>(pRhs[1]); \
    pRhs2 = pset1<RhsPacket>(pRhs[2]); \
    pRhs3 = pset1<RhsPacket>(pRhs[3]); \
    acc._acc1.packet[0] += pLhs*pRhs0; \
    acc._acc1.packet[1] += pLhs*pRhs1; \
    acc._acc1.packet[2] += pLhs*pRhs2; \
    acc._acc1.packet[3] += pLhs*pRhs3; \
    acc._acc2.packet[0] += pLhs2*pRhs0; \
    acc._acc2.packet[1] += pLhs2*pRhs1; \
    acc._acc2.packet[2] += pLhs2*pRhs2; \
    acc._acc2.packet[3] += pLhs2*pRhs3; \
    lhsPackMap.advance(4*1); \
    rhsPackMap.advance(1*4);

template<int CPU, typename Index, typename LhsScalar, typename LhsPackMap, typename RhsScalar, typename RhsPackMap, typename AccScalar, typename ResScalar, typename Accumulator>
struct MicroKernel<0, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, Accumulator, 8, 4, 4>
{
  EIGEN_STRONG_INLINE void operator()(LhsPackMap& lhsPackMap, 
                                      RhsPackMap& rhsPackMap, 
                                      Index rowIdx, Index colIdx, Index depthIdx,
                                      Accumulator& acc)
  {
    using LhsPacket = typename packet_traits<LhsScalar>::type;
    using RhsPacket = typename packet_traits<RhsScalar>::type;

    asm __volatile__("#BEGIN_NEON_MICROKERNEL_8x4x4\n\t");

    LhsPacket pLhs, pLhs2;
    RhsPacket pRhs, pRhs0, pRhs1, pRhs2, pRhs3;

    MICRO_8x1x4();
    MICRO_8x1x4();
    MICRO_8x1x4();
    MICRO_8x1x4();

    asm __volatile__("#END_NEON_MICROKERNEL_8x4x4\n\t");
  };
};

template<int CPU, typename Index, typename LhsScalar, typename LhsPackMap, typename RhsScalar, typename RhsPackMap, typename AccScalar, typename ResScalar, typename Accumulator>
struct MicroKernel<0, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, Accumulator, 8, 1, 4>
{
  EIGEN_STRONG_INLINE void operator()(LhsPackMap& lhsPackMap, 
                                      RhsPackMap& rhsPackMap, 
                                      Index rowIdx, Index colIdx, Index depthIdx,
                                      Accumulator& acc)
  {
    using LhsPacket = typename packet_traits<LhsScalar>::type;
    using RhsPacket = typename packet_traits<RhsScalar>::type;

    asm __volatile__("#BEGIN_NEON_MICROKERNEL_8x1x4\n\t");

    LhsPacket pLhs, pLhs2;
    RhsPacket pRhs, pRhs0, pRhs1, pRhs2, pRhs3;

    MICRO_8x1x4();

    asm __volatile__("#END_NEON_MICROKERNEL_8x1x4\n\t");
  };
};

#define MICRO_4x1x4() \
    pLhs = pload<LhsPacket>(lhsPackMap.pCur); \
    pRhs = pload<RhsPacket>(rhsPackMap.pCur); \
    pRhs0 = pset1<RhsPacket>(pRhs[0]); \
    pRhs1 = pset1<RhsPacket>(pRhs[1]); \
    pRhs2 = pset1<RhsPacket>(pRhs[2]); \
    pRhs3 = pset1<RhsPacket>(pRhs[3]); \
    acc._acc.packet[0] += pLhs*pRhs0; \
    acc._acc.packet[1] += pLhs*pRhs1; \
    acc._acc.packet[2] += pLhs*pRhs2; \
    acc._acc.packet[3] += pLhs*pRhs3; \
    lhsPackMap.advance(4*1); \
    rhsPackMap.advance(1*4);

template<int CPU, typename Index, typename LhsScalar, typename LhsPackMap, typename RhsScalar, typename RhsPackMap, typename AccScalar, typename ResScalar, typename Accumulator>
struct MicroKernel<0, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, Accumulator, 4, 4, 4>
{
  EIGEN_STRONG_INLINE void operator()(LhsPackMap& lhsPackMap, 
                                      RhsPackMap& rhsPackMap, 
                                      Index rowIdx, Index colIdx, Index depthIdx,
                                      Accumulator& acc)
  {
    using LhsPacket = typename packet_traits<LhsScalar>::type;
    using RhsPacket = typename packet_traits<RhsScalar>::type;

    asm __volatile__("#BEGIN_NEON_MICROKERNEL_4x4x4\n\t");
    LhsPacket pLhs;
    RhsPacket pRhs, pRhs0, pRhs1, pRhs2, pRhs3;

    MICRO_4x1x4();
    MICRO_4x1x4();
    MICRO_4x1x4();
    MICRO_4x1x4();
    asm __volatile__("#END_NEON_MICROKERNEL_4x4x4\n\t");
  };
};

template<int CPU, typename Index, typename LhsScalar, typename LhsPackMap, typename RhsScalar, typename RhsPackMap, typename AccScalar, typename ResScalar, typename Accumulator>
struct MicroKernel<0, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, Accumulator, 4, 1, 4>
{
  EIGEN_STRONG_INLINE void operator()(LhsPackMap& lhsPackMap, 
                                      RhsPackMap& rhsPackMap, 
                                      Index rowIdx, Index colIdx, Index depthIdx,
                                      Accumulator& acc)
  {
    using LhsPacket = typename packet_traits<LhsScalar>::type;
    using RhsPacket = typename packet_traits<RhsScalar>::type;

    asm __volatile__("#BEGIN_NEON_MICROKERNEL_4x1x4\n\t");

    LhsPacket pLhs;
    RhsPacket pRhs, pRhs0, pRhs1, pRhs2, pRhs3;

    MICRO_4x1x4();

    asm __volatile__("#END_NEON_MICROKERNEL_4x1x4\n\t");
  };
};

template<int CPU, typename Index, typename LhsScalar, typename LhsPackMap, typename RhsScalar, typename RhsPackMap, typename AccScalar, typename ResScalar, typename Accumulator>
struct MicroKernel<0, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, Accumulator, 4, 1, 1>
{
  EIGEN_STRONG_INLINE void operator()(LhsPackMap& lhsPackMap, 
                                      RhsPackMap& rhsPackMap, 
                                      Index rowIdx, Index colIdx, Index depthIdx,
                                      Accumulator& acc)
  {
    using LhsPacket = typename packet_traits<LhsScalar>::type;

    asm __volatile__("#BEGIN_NEON_MICROKERNEL_4x1x1\n\t");

    LhsPacket pLhs = pload<LhsPacket>(lhsPackMap.pCur);
    RhsScalar rhs = *rhsPackMap.pCur;

    acc._acc += pLhs*rhs;

    lhsPackMap.advance(4*1);
    rhsPackMap.advance(1);
    asm __volatile__("#END_NEON_MICROKERNEL_4x1x1\n\t");
  };
};

template<int CPU, typename Index, typename LhsScalar, typename LhsPackMap, typename RhsScalar, typename RhsPackMap, typename AccScalar, typename ResScalar, typename Accumulator>
struct MicroKernel<0, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, Accumulator, 1, 1, 4>
{
  EIGEN_STRONG_INLINE void operator()(LhsPackMap& lhsPackMap, 
                                      RhsPackMap& rhsPackMap, 
                                      Index rowIdx, Index colIdx, Index depthIdx,
                                      Accumulator& acc)
  {
    using RhsPacket = typename packet_traits<RhsScalar>::type;

    asm __volatile__("#BEGIN_NEON_MICROKERNEL_1x1x4\n\t");

    RhsPacket pRhs = pload<RhsPacket>(rhsPackMap.pCur);
    LhsScalar lhs = *lhsPackMap.pCur;

    acc._acc += pRhs*lhs;

    lhsPackMap.advance(1);
    rhsPackMap.advance(4*1);
    asm __volatile__("#END_NEON_MICROKERNEL_1x1x4\n\t");
  };
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_KERNELS_NEON_H