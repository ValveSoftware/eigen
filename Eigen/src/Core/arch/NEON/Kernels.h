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

    asm __volatile__("#BEGIN_NEON_MICROKERNEL_3x1\n\t");
    LhsPacket pLhs = pload<LhsPacket>(lhsPackMap.pCur);
    RhsPacket pRhs = pload<RhsPacket>(rhsPackMap.pCur);
    RhsPacket pRhs0 = pset1<RhsPacket>(pRhs[0]);
    RhsPacket pRhs1 = pset1<RhsPacket>(pRhs[1]);
    RhsPacket pRhs2 = pset1<RhsPacket>(pRhs[2]);
    RhsPacket pRhs3 = pset1<RhsPacket>(pRhs[3]);

    acc._acc.packet[0] += pLhs*pRhs0;
    acc._acc.packet[1] += pLhs*pRhs1;
    acc._acc.packet[2] += pLhs*pRhs2;
    acc._acc.packet[3] += pLhs*pRhs3;

    lhsPackMap.advance(4*1);
    rhsPackMap.advance(1*4);
    asm __volatile__("#END_NEON_MICROKERNEL_3x1\n\t");
  };
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_KERNELS_NEON_H