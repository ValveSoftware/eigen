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

#ifdef __ENABLE_VECTOR_KERNELS__

#define MICRO_12x1x4() \
    pRhs = pload<RhsPacket>(rhsPackMap.pCur); \
    rhsPackMap.advance(1*4); \
    pRhs0 = pset1<RhsPacket>(pRhs[0]); \
    pRhs1 = pset1<RhsPacket>(pRhs[1]); \
    pRhs2 = pset1<RhsPacket>(pRhs[2]); \
    pRhs3 = pset1<RhsPacket>(pRhs[3]); \
    pLhs = pload<LhsPacket>(lhsPackMap.pCur); \
    lhsPackMap.advance(4*1); \
    acc._acc1.packet[0] += pLhs*pRhs0; \
    acc._acc1.packet[1] += pLhs*pRhs1; \
    acc._acc1.packet[2] += pLhs*pRhs2; \
    acc._acc1.packet[3] += pLhs*pRhs3; \
    pLhs2 = pload<LhsPacket>(lhsPackMap.pCur); \
    lhsPackMap.advance(4*1); \
    acc._acc2.packet[0] += pLhs2*pRhs0; \
    acc._acc2.packet[1] += pLhs2*pRhs1; \
    acc._acc2.packet[2] += pLhs2*pRhs2; \
    acc._acc2.packet[3] += pLhs2*pRhs3; \
    pLhs3 = pload<LhsPacket>(lhsPackMap.pCur); \
    acc._acc3.packet[0] += pLhs3*pRhs0; \
    acc._acc3.packet[1] += pLhs3*pRhs1; \
    acc._acc3.packet[2] += pLhs3*pRhs2; \
    acc._acc3.packet[3] += pLhs3*pRhs3; \
    lhsPackMap.advance(4*1);

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

#define MICRO_12x1x1() \
    pLhs = pload<LhsPacket>(lhsPackMap.pCur); \
    pRhs = pset1<RhsPacket>(*rhsPackMap.pCur); \
    acc._acc.packet[0] = pmadd(pRhs, pLhs, acc._acc.packet[0]); \
    lhsPackMap.advance(4*1); \
    pLhs = pload<LhsPacket>(lhsPackMap.pCur); \
    acc._acc.packet[1] = pmadd(pRhs, pLhs, acc._acc.packet[1]); \
    lhsPackMap.advance(4*1); \
    pLhs = pload<LhsPacket>(lhsPackMap.pCur); \
    acc._acc.packet[2] = pmadd(pRhs, pLhs, acc._acc.packet[2]); \
    lhsPackMap.advance(4*1); \
    rhsPackMap.advance(1);

#define MICRO_8x1x1() \
    pLhs = pload<LhsPacket>(lhsPackMap.pCur); \
    pRhs = pset1<RhsPacket>(*rhsPackMap.pCur); \
    acc._acc.packet[0] = pmadd(pRhs, pLhs, acc._acc.packet[0]); \
    lhsPackMap.advance(4*1); \
    pLhs = pload<LhsPacket>(lhsPackMap.pCur); \
    acc._acc.packet[1] = pmadd(pRhs, pLhs, acc._acc.packet[1]); \
    lhsPackMap.advance(4*1); \
    rhsPackMap.advance(1);

#define MICRO_4x1x1() \
    pLhs = pload<LhsPacket>(lhsPackMap.pCur); \
    pRhs = pset1<RhsPacket>(*rhsPackMap.pCur); \
    acc._acc += pRhs*pLhs; \
    lhsPackMap.advance(4*1); \
    rhsPackMap.advance(1);

template<int CPU, typename Scalar, typename ResScalar, typename DataMapper>
struct Accumulator<0, CPU, Scalar, ResScalar, DataMapper, 12, 1>
{
  using LinearMapper = typename DataMapper::LinearMapper;
  using AccPacket = typename packet_traits<Scalar>::type;
  using ResPacket = typename packet_traits<ResScalar>::type;

  PacketBlock<AccPacket,3> _acc;

  EIGEN_STRONG_INLINE void zero()
  {
    _acc.packet[0] = pset1<AccPacket>(0);
    _acc.packet[1] = pset1<AccPacket>(0);
    _acc.packet[2] = pset1<AccPacket>(0);
  }

  EIGEN_STRONG_INLINE void prefetch(const DataMapper&, Index, Index) {}

  template<typename ResPacket_>
  EIGEN_STRONG_INLINE void scale(ResScalar alpha, const ResPacket_& pAlpha)
  {
    _acc.packet[0] *= pAlpha;
    _acc.packet[1] *= pAlpha;
    _acc.packet[2] *= pAlpha;
  }

  EIGEN_STRONG_INLINE void store(const DataMapper& dest, Index row, Index col)
  {
    PacketBlock<ResPacket, 1> block;
    block.packet[0] = dest.template loadPacket<ResPacket>(row + 0, col) + _acc.packet[0];
    dest.template storePacketBlock<AccPacket, 1>(row + 0, col, block);
    block.packet[0] = dest.template loadPacket<ResPacket>(row + 4, col) + _acc.packet[1];
    dest.template storePacketBlock<AccPacket, 1>(row + 4, col, block);
    block.packet[0] = dest.template loadPacket<ResPacket>(row + 8, col) + _acc.packet[2];
    dest.template storePacketBlock<AccPacket, 1>(row + 8, col, block);
  }
};

template<int CPU, typename Scalar, typename ResScalar, typename DataMapper>
struct Accumulator<0, CPU, Scalar, ResScalar, DataMapper, 8, 1>
{
  using LinearMapper = typename DataMapper::LinearMapper;
  using AccPacket = typename packet_traits<Scalar>::type;
  using ResPacket = typename packet_traits<ResScalar>::type;

  PacketBlock<AccPacket,2> _acc;

  EIGEN_STRONG_INLINE void zero()
  {
    _acc.packet[0] = pset1<AccPacket>(0);
    _acc.packet[1] = pset1<AccPacket>(0);
  }

  EIGEN_STRONG_INLINE void prefetch(const DataMapper&, Index, Index) {}

  template<typename ResPacket_>
  EIGEN_STRONG_INLINE void scale(ResScalar alpha, const ResPacket_& pAlpha)
  {
    _acc.packet[0] *= pAlpha;
    _acc.packet[1] *= pAlpha;
  }

  EIGEN_STRONG_INLINE void store(const DataMapper& dest, Index row, Index col)
  {
    PacketBlock<ResPacket, 1> block;
    block.packet[0] = dest.template loadPacket<ResPacket>(row, col) + _acc.packet[0];
    dest.template storePacketBlock<AccPacket, 1>(row, col, block);
    block.packet[0] = dest.template loadPacket<ResPacket>(row + 4, col) + _acc.packet[1];
    dest.template storePacketBlock<AccPacket, 1>(row + 4, col, block);
  }
};

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
  
  EIGEN_STRONG_INLINE void prefetch(const DataMapper&, Index, Index) {}

  template<typename ResPacket_>
  EIGEN_STRONG_INLINE void scale(ResScalar alpha, const ResPacket_& pAlpha)
  {
    _acc *= pAlpha;
  }

  EIGEN_STRONG_INLINE void store(const DataMapper& dest, Index row, Index col)
  {
    PacketBlock<ResPacket, 1> block;
    block.packet[0] = dest.template loadPacket<ResPacket>(row, col) + _acc;
    dest.template storePacketBlock<AccPacket, 1>(row, col, block);
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

  EIGEN_STRONG_INLINE void prefetch(const DataMapper&, Index, Index) {}

  template<typename ResPacket_>
  EIGEN_STRONG_INLINE void scale(ResScalar alpha, const ResPacket_& pAlpha)
  {
    _acc *= pAlpha;
  }

  EIGEN_STRONG_INLINE void store(const DataMapper& dest, Index row, Index col)
  {
    ResPacket r = dest.template gatherPacket<ResPacket>(row, col) + _acc;
    dest.template scatterPacket<ResPacket>(row, col, r);
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

  EIGEN_STRONG_INLINE void prefetch(const DataMapper& dest, Index row, Index col)
  {
    dest.getLinearMapper(row, col + 0).prefetch(0);
    dest.getLinearMapper(row, col + 1).prefetch(0);
    dest.getLinearMapper(row, col + 2).prefetch(0);
    dest.getLinearMapper(row, col + 3).prefetch(0);
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
    constexpr auto PacketSize = unpacket_traits<ResPacket>::size;

    LinearMapper r0 = dest.getLinearMapper(row, col + 0);
    LinearMapper r1 = dest.getLinearMapper(row, col + 1);
    LinearMapper r2 = dest.getLinearMapper(row, col + 2);
    LinearMapper r3 = dest.getLinearMapper(row, col + 3);

    r0.storePacket(0*PacketSize, r0.template loadPacket<ResPacket>(0*PacketSize) + _acc.packet[0]);
    r1.storePacket(0*PacketSize, r1.template loadPacket<ResPacket>(0*PacketSize) + _acc.packet[1]);
    r2.storePacket(0*PacketSize, r2.template loadPacket<ResPacket>(0*PacketSize) + _acc.packet[2]);
    r3.storePacket(0*PacketSize, r3.template loadPacket<ResPacket>(0*PacketSize) + _acc.packet[3]);
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

  EIGEN_STRONG_INLINE void prefetch(const DataMapper& dest, Index row, Index col)
  {
    dest.getLinearMapper(row + 0, col + 0).prefetch(0);
    dest.getLinearMapper(row + 0, col + 1).prefetch(0);
    dest.getLinearMapper(row + 0, col + 2).prefetch(0);
    dest.getLinearMapper(row + 0, col + 3).prefetch(0);
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
    constexpr auto PacketSize = unpacket_traits<ResPacket>::size;

    LinearMapper r0 = dest.getLinearMapper(row, col + 0);
    LinearMapper r1 = dest.getLinearMapper(row, col + 1);
    LinearMapper r2 = dest.getLinearMapper(row, col + 2);
    LinearMapper r3 = dest.getLinearMapper(row, col + 3);

    r0.storePacket(0*PacketSize, r0.template loadPacket<ResPacket>(0*PacketSize) + _acc1.packet[0]);
    r1.storePacket(0*PacketSize, r1.template loadPacket<ResPacket>(0*PacketSize) + _acc1.packet[1]);
    r2.storePacket(0*PacketSize, r2.template loadPacket<ResPacket>(0*PacketSize) + _acc1.packet[2]);
    r3.storePacket(0*PacketSize, r3.template loadPacket<ResPacket>(0*PacketSize) + _acc1.packet[3]);

    r0.storePacket(1*PacketSize, r0.template loadPacket<ResPacket>(1*PacketSize) + _acc2.packet[0]);
    r1.storePacket(1*PacketSize, r1.template loadPacket<ResPacket>(1*PacketSize) + _acc2.packet[1]);
    r2.storePacket(1*PacketSize, r2.template loadPacket<ResPacket>(1*PacketSize) + _acc2.packet[2]);
    r3.storePacket(1*PacketSize, r3.template loadPacket<ResPacket>(1*PacketSize) + _acc2.packet[3]);
  }
};

template<int CPU, typename Scalar, typename ResScalar, typename DataMapper>
struct Accumulator<0, CPU, Scalar, ResScalar, DataMapper, 12, 4>
{
  using LinearMapper = typename DataMapper::LinearMapper;
  using AccPacket = typename packet_traits<Scalar>::type;
  using ResPacket = typename packet_traits<ResScalar>::type;

  PacketBlock<AccPacket, 4> _acc1;
  PacketBlock<AccPacket, 4> _acc2;
  PacketBlock<AccPacket, 4> _acc3;

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

    _acc3.packet[0] = pset1<AccPacket>(0);
    _acc3.packet[1] = pset1<AccPacket>(0);
    _acc3.packet[2] = pset1<AccPacket>(0);
    _acc3.packet[3] = pset1<AccPacket>(0);
  }

  EIGEN_STRONG_INLINE void prefetch(const DataMapper& dest, Index row, Index col)
  {
    dest.getLinearMapper(row + 0, col + 0).prefetch(0);
    dest.getLinearMapper(row + 0, col + 1).prefetch(0);
    dest.getLinearMapper(row + 0, col + 2).prefetch(0);
    dest.getLinearMapper(row + 0, col + 3).prefetch(0);
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

    _acc3.packet[0] *= pAlpha;
    _acc3.packet[1] *= pAlpha;
    _acc3.packet[2] *= pAlpha;
    _acc3.packet[3] *= pAlpha;
  }

  EIGEN_STRONG_INLINE void store(const DataMapper& dest, Index row, Index col)
  {
    constexpr auto PacketSize = unpacket_traits<ResPacket>::size;

    LinearMapper r0 = dest.getLinearMapper(row, col + 0);
    LinearMapper r1 = dest.getLinearMapper(row, col + 1);
    LinearMapper r2 = dest.getLinearMapper(row, col + 2);
    LinearMapper r3 = dest.getLinearMapper(row, col + 3);

    r0.storePacket(0*PacketSize, r0.template loadPacket<ResPacket>(0*PacketSize) + _acc1.packet[0]);
    r1.storePacket(0*PacketSize, r1.template loadPacket<ResPacket>(0*PacketSize) + _acc1.packet[1]);
    r2.storePacket(0*PacketSize, r2.template loadPacket<ResPacket>(0*PacketSize) + _acc1.packet[2]);
    r3.storePacket(0*PacketSize, r3.template loadPacket<ResPacket>(0*PacketSize) + _acc1.packet[3]);

    r0.storePacket(1*PacketSize, r0.template loadPacket<ResPacket>(1*PacketSize) + _acc2.packet[0]);
    r1.storePacket(1*PacketSize, r1.template loadPacket<ResPacket>(1*PacketSize) + _acc2.packet[1]);
    r2.storePacket(1*PacketSize, r2.template loadPacket<ResPacket>(1*PacketSize) + _acc2.packet[2]);
    r3.storePacket(1*PacketSize, r3.template loadPacket<ResPacket>(1*PacketSize) + _acc2.packet[3]);

    r0.storePacket(2*PacketSize, r0.template loadPacket<ResPacket>(2*PacketSize) + _acc3.packet[0]);
    r1.storePacket(2*PacketSize, r1.template loadPacket<ResPacket>(2*PacketSize) + _acc3.packet[1]);
    r2.storePacket(2*PacketSize, r2.template loadPacket<ResPacket>(2*PacketSize) + _acc3.packet[2]);
    r3.storePacket(2*PacketSize, r3.template loadPacket<ResPacket>(2*PacketSize) + _acc3.packet[3]);
  }
};

template<int CPU, typename Index, typename LhsScalar, typename LhsPackMap, typename RhsScalar, typename RhsPackMap, typename AccScalar, typename ResScalar, typename Accumulator>
struct MicroKernel<0, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, Accumulator, 4, __UNROLL__ , 4>
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
#if __UNROLL__ > 4
    MICRO_4x1x4();
    MICRO_4x1x4();
    MICRO_4x1x4();
    MICRO_4x1x4();
#endif

    asm __volatile__("#END_NEON_MICROKERNEL_4x4x4\n\t");
  };
};

template<int CPU, typename Index, typename LhsScalar, typename LhsPackMap, typename RhsScalar, typename RhsPackMap, typename AccScalar, typename ResScalar, typename Accumulator>
struct MicroKernel<0, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, Accumulator, 8, __UNROLL__, 4>
{
  EIGEN_STRONG_INLINE void operator()(LhsPackMap& lhsPackMap, 
                                      RhsPackMap& rhsPackMap, 
                                      Index rowIdx, Index colIdx, Index depthIdx,
                                      Accumulator& acc)
  {
    using LhsPacket = typename packet_traits<LhsScalar>::type;
    using RhsPacket = typename packet_traits<RhsScalar>::type;

    asm __volatile__("#BEGIN_NEON_MICROKERNEL_8x8x4\n\t");

    LhsPacket pLhs, pLhs2;
    RhsPacket pRhs, pRhs0, pRhs1, pRhs2, pRhs3;

#if __UNROLL__ == 8
#ifdef __ENABLE_PREFETCH__
    prefetch(rhsPackMap.pCur + (48+0));
#endif
    MICRO_8x1x4();
    MICRO_8x1x4();
    MICRO_8x1x4();
    MICRO_8x1x4();
#ifdef __ENABLE_PREFETCH__
    prefetch(rhsPackMap.pCur + (48+16));
#endif
    MICRO_8x1x4();
    MICRO_8x1x4();
    MICRO_8x1x4();
    MICRO_8x1x4();
#else
    MICRO_8x1x4();
    MICRO_8x1x4();
    MICRO_8x1x4();
    MICRO_8x1x4();
#endif
    asm __volatile__("#END_NEON_MICROKERNEL_8x8x4\n\t");
  };
};

template<int CPU, typename Index, typename LhsScalar, typename LhsPackMap, typename RhsScalar, typename RhsPackMap, typename AccScalar, typename ResScalar, typename Accumulator>
struct MicroKernel<0, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, Accumulator, 12, __UNROLL__, 4>
{
  EIGEN_STRONG_INLINE void operator()(LhsPackMap& lhsPackMap, 
                                      RhsPackMap& rhsPackMap, 
                                      Index rowIdx, Index colIdx, Index depthIdx,
                                      Accumulator& acc)
  {
    using LhsPacket = typename packet_traits<LhsScalar>::type;
    using RhsPacket = typename packet_traits<RhsScalar>::type;

    asm __volatile__("#BEGIN_NEON_MICROKERNEL_12x8x4\n\t");

    LhsPacket pLhs, pLhs2, pLhs3;
    RhsPacket pRhs, pRhs0, pRhs1, pRhs2, pRhs3;

#if __UNROLL__ == 8
#ifdef __ENABLE_PREFETCH__
    prefetch(rhsPackMap.pCur);
#endif
    MICRO_12x1x4();
    MICRO_12x1x4();
    MICRO_12x1x4();
    MICRO_12x1x4();
    MICRO_12x1x4();
    MICRO_12x1x4();
    MICRO_12x1x4();
    MICRO_12x1x4();
#else
    MICRO_12x1x4();
    MICRO_12x1x4();
    MICRO_12x1x4();
    MICRO_12x1x4();
#endif
    asm __volatile__("#END_NEON_MICROKERNEL_12x8x4\n\t");
  };
};

template<int CPU, typename Index, typename LhsScalar, typename LhsPackMap, typename RhsScalar, typename RhsPackMap, typename AccScalar, typename ResScalar, typename Accumulator>
struct MicroKernel<0, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, Accumulator, 12, 1, 4>
{
  EIGEN_STRONG_INLINE void operator()(LhsPackMap& lhsPackMap, 
                                      RhsPackMap& rhsPackMap, 
                                      Index rowIdx, Index colIdx, Index depthIdx,
                                      Accumulator& acc)
  {
    using LhsPacket = typename packet_traits<LhsScalar>::type;
    using RhsPacket = typename packet_traits<RhsScalar>::type;

    asm __volatile__("#BEGIN_NEON_MICROKERNEL_12x1x4\n\t");

    LhsPacket pLhs, pLhs2, pLhs3;
    RhsPacket pRhs, pRhs0, pRhs1, pRhs2, pRhs3;

    MICRO_12x1x4();

    asm __volatile__("#END_NEON_MICROKERNEL_12x1x4\n\t");
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
struct MicroKernel<0, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, Accumulator, 12, __UNROLL__, 1>
{
  EIGEN_STRONG_INLINE void operator()(LhsPackMap& lhsPackMap,
                                      RhsPackMap& rhsPackMap,
                                      Index rowIdx, Index colIdx, Index depthIdx,
                                      Accumulator& acc)
  {
    using LhsPacket = typename packet_traits<LhsScalar>::type;
    using RhsPacket = typename packet_traits<RhsScalar>::type;

    LhsPacket pLhs;
    RhsPacket pRhs;

    asm __volatile__("#BEGIN_NEON_MICROKERNEL_4x1x1\n\t");

    MICRO_12x1x1();
    MICRO_12x1x1();
    MICRO_12x1x1();
    MICRO_12x1x1();
    MICRO_12x1x1();
    MICRO_12x1x1();
    MICRO_12x1x1();
    MICRO_12x1x1();

    asm __volatile__("#END_NEON_MICROKERNEL_4x1x1\n\t");
  };
};

template<int CPU, typename Index, typename LhsScalar, typename LhsPackMap, typename RhsScalar, typename RhsPackMap, typename AccScalar, typename ResScalar, typename Accumulator>
struct MicroKernel<0, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, Accumulator, 12, 1, 1>
{
  EIGEN_STRONG_INLINE void operator()(LhsPackMap& lhsPackMap,
                                      RhsPackMap& rhsPackMap,
                                      Index rowIdx, Index colIdx, Index depthIdx,
                                      Accumulator& acc)
  {
    using LhsPacket = typename packet_traits<LhsScalar>::type;
    using RhsPacket = typename packet_traits<RhsScalar>::type;

    LhsPacket pLhs;
    RhsPacket pRhs;

    asm __volatile__("#BEGIN_NEON_MICROKERNEL_4x1x1\n\t");

    MICRO_12x1x1();

    asm __volatile__("#END_NEON_MICROKERNEL_4x1x1\n\t");
  };
};

template<int CPU, typename Index, typename LhsScalar, typename LhsPackMap, typename RhsScalar, typename RhsPackMap, typename AccScalar, typename ResScalar, typename Accumulator>
struct MicroKernel<0, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, Accumulator, 8, __UNROLL__, 1>
{
  EIGEN_STRONG_INLINE void operator()(LhsPackMap& lhsPackMap,
                                      RhsPackMap& rhsPackMap,
                                      Index rowIdx, Index colIdx, Index depthIdx,
                                      Accumulator& acc)
  {
    using LhsPacket = typename packet_traits<LhsScalar>::type;
    using RhsPacket = typename packet_traits<RhsScalar>::type;

    asm __volatile__("#BEGIN_NEON_MICROKERNEL_4x1x1\n\t");

    LhsPacket pLhs;
    RhsPacket pRhs;

    MICRO_8x1x1();
    MICRO_8x1x1();
    MICRO_8x1x1();
    MICRO_8x1x1();
    MICRO_8x1x1();
    MICRO_8x1x1();
    MICRO_8x1x1();
    MICRO_8x1x1();

    asm __volatile__("#END_NEON_MICROKERNEL_4x1x1\n\t");
  };
};

template<int CPU, typename Index, typename LhsScalar, typename LhsPackMap, typename RhsScalar, typename RhsPackMap, typename AccScalar, typename ResScalar, typename Accumulator>
struct MicroKernel<0, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, Accumulator, 8, 1, 1>
{
  EIGEN_STRONG_INLINE void operator()(LhsPackMap& lhsPackMap,
                                      RhsPackMap& rhsPackMap,
                                      Index rowIdx, Index colIdx, Index depthIdx,
                                      Accumulator& acc)
  {
    using LhsPacket = typename packet_traits<LhsScalar>::type;
    using RhsPacket = typename packet_traits<RhsScalar>::type;

    asm __volatile__("#BEGIN_NEON_MICROKERNEL_4x1x1\n\t");

    LhsPacket pLhs;
    RhsPacket pRhs;

    MICRO_8x1x1();

    asm __volatile__("#END_NEON_MICROKERNEL_4x1x1\n\t");
  };
};

template<int CPU, typename Index, typename LhsScalar, typename LhsPackMap, typename RhsScalar, typename RhsPackMap, typename AccScalar, typename ResScalar, typename Accumulator>
struct MicroKernel<0, CPU, Index, LhsScalar, LhsPackMap, RhsScalar, RhsPackMap, AccScalar, ResScalar, Accumulator, 4, __UNROLL__, 1>
{
  EIGEN_STRONG_INLINE void operator()(LhsPackMap& lhsPackMap,
                                      RhsPackMap& rhsPackMap,
                                      Index rowIdx, Index colIdx, Index depthIdx,
                                      Accumulator& acc)
  {
    using LhsPacket = typename packet_traits<LhsScalar>::type;
    using RhsPacket = typename packet_traits<RhsScalar>::type;

    asm __volatile__("#BEGIN_NEON_MICROKERNEL_4x1x1\n\t");

    LhsPacket pLhs;
    RhsPacket pRhs;

    MICRO_4x1x1();
    MICRO_4x1x1();
    MICRO_4x1x1();
    MICRO_4x1x1();
    MICRO_4x1x1();
    MICRO_4x1x1();
    MICRO_4x1x1();
    MICRO_4x1x1();

    asm __volatile__("#END_NEON_MICROKERNEL_4x1x1\n\t");
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
    using RhsPacket = typename packet_traits<RhsScalar>::type;

    asm __volatile__("#BEGIN_NEON_MICROKERNEL_4x1x1\n\t");

    LhsPacket pLhs;
    RhsPacket pRhs;

    MICRO_4x1x1();

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
    using LhsPacket = typename packet_traits<LhsScalar>::type;

    asm __volatile__("#BEGIN_NEON_MICROKERNEL_1x1x4\n\t");

    LhsPacket pLhs = pset1<LhsPacket>(*lhsPackMap.pCur);
    RhsPacket pRhs = pload<RhsPacket>(rhsPackMap.pCur);

    acc._acc += pLhs*pRhs;

    lhsPackMap.advance(1);
    rhsPackMap.advance(4*1);
    asm __volatile__("#END_NEON_MICROKERNEL_1x1x4\n\t");
  };
};

#endif // __ENABLE_VECTOR_KERNELS__

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_KERNELS_NEON_H