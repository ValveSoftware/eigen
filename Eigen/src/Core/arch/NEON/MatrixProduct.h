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


#ifdef __OLD__
template<typename Scalar, typename Packet, typename Index, bool IsLhs = true>
class PackMap
{
  const int packetSize = packet_traits<Scalar>::size;
  const Scalar *packed_block;
  const Scalar *residue_block;
  Index packed_stride;
  Index residue_size;
  Index rows, cols;
  Index offset, stride;
  Scalar *cur;
public:
  PackMap(const Scalar *packed_block, const Scalar *residue_block, Index rows, Index cols, Index offset, Index stride) : packed_block(packed_block), residue_block(residue_block), rows(rows), cols(cols), offset(offset), stride(stride)
  {
    if(IsLhs)
    {
      packed_stride = (rows / packetSize) * packetSize;
      residue_size = rows % packetSize;
    }
    else {
      packed_stride = (cols / packetSize) * packetSize;
      residue_size = cols % packetSize;
    }
  };

  PackMap(const Scalar *packed_block, Index rows, Index cols, Index offset, Index stride) : packed_block(packed_block), rows(rows), cols(cols)
  {
    if(IsLhs)
    {
      packed_stride = (rows / packetSize) * packetSize;
      residue_block = packed_block + packed_stride*cols;
      residue_size = rows % packetSize;
    }
    else {
      packed_stride = (cols / packetSize) * packetSize;
      residue_block = packed_block + packed_stride*rows;
      residue_size = cols % packetSize;
    }

  };

  EIGEN_STRONG_INLINE Index get_packed_size()
  {
    return packed_stride;
  };

  EIGEN_STRONG_INLINE Index get_residue_size()
  {
    return residue_size;
  };

  EIGEN_STRONG_INLINE const Scalar* get_packed_at(Index at)
  {
    return IsLhs ? packed_block + at : packed_block + at*packetSize*rows;
  };

  EIGEN_STRONG_INLINE const Scalar* get_residue_at(Index at)
  {
    return residue_block + stride*at;
  };
};

template<typename ResScalar, typename AccScalar, typename LhsScalar, typename RhsScalar, typename Index, typename DataMapper>
EIGEN_STRONG_INLINE void gemm(const DataMapper& res, const LhsScalar* blockA, const RhsScalar* blockB,
          Index rows, Index depth, Index cols, ResScalar alpha, Index strideA, Index strideB, Index offsetA, Index offsetB)
{
  using AccPacket = typename packet_traits<AccScalar>::type;
  using LhsPacket = typename packet_traits<LhsScalar>::type;
  using RhsPacket = typename packet_traits<RhsScalar>::type;
  using ResPacket = typename packet_traits<ResScalar>::type;
  using LinearMapper = typename DataMapper::LinearMapper;

  if( strideA == -1 ) strideA = depth;
  if( strideB == -1 ) strideB = depth;

  ResPacket pAlpha = pset1<ResPacket>(alpha);

#ifdef __DEBUG__
  std::cout << "blockA" << std::endl;
  for(auto i = 0; i < rows*depth; i++)
  {
    if(i % strideA == 0 && i > 0)
      std::cout << std::endl;
    std::cout << blockA[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "blockB" << std::endl;
  for(auto i = 0; i < depth*cols; i++)
  {
    if(i % strideB == 0 && i > 0)
      std::cout << std::endl;
    std::cout << blockB[i] << " ";
  }
  std::cout << std::endl;
#endif

  int accLhsProgress = 4;
  int accRhsProgress = 4;

  PackMap<LhsScalar, LhsPacket, Index> lhsMap(blockA, rows, depth, offsetA, strideA);
  PackMap<RhsScalar, RhsPacket, Index, false> rhsMap(blockB, depth, cols, offsetB, strideB);
  auto col = 0;
  for(; col + accRhsProgress <= rhsMap.get_packed_size(); col+=accRhsProgress)
  {
    auto row = 0;
    for(; row + 3*accLhsProgress <= lhsMap.get_packed_size(); row+=3*accLhsProgress)
    {
      const LhsScalar *lhs_ptr1 = lhsMap.get_packed_at(row + 0*accLhsProgress);
      const LhsScalar *lhs_ptr2 = lhsMap.get_packed_at(row + 1*accLhsProgress);
      const LhsScalar *lhs_ptr3 = lhsMap.get_packed_at(row + 2*accLhsProgress);
      const RhsScalar *rhs_ptr = rhsMap.get_packed_at(col/accRhsProgress);

      PacketBlock<AccPacket, 4> acc1;
      acc1.packet[0] = pset1<AccPacket>(0);
      acc1.packet[1] = pset1<AccPacket>(0);
      acc1.packet[2] = pset1<AccPacket>(0);
      acc1.packet[3] = pset1<AccPacket>(0);

      PacketBlock<AccPacket, 4> acc2;
      acc2.packet[0] = pset1<AccPacket>(0);
      acc2.packet[1] = pset1<AccPacket>(0);
      acc2.packet[2] = pset1<AccPacket>(0);
      acc2.packet[3] = pset1<AccPacket>(0);

      PacketBlock<AccPacket, 4> acc3;
      acc3.packet[0] = pset1<AccPacket>(0);
      acc3.packet[1] = pset1<AccPacket>(0);
      acc3.packet[2] = pset1<AccPacket>(0);
      acc3.packet[3] = pset1<AccPacket>(0);

      LinearMapper r00 = res.getLinearMapper(row + 0*accLhsProgress, col + 0);
      LinearMapper r01 = res.getLinearMapper(row + 0*accLhsProgress, col + 1);
      LinearMapper r02 = res.getLinearMapper(row + 0*accLhsProgress, col + 2);
      LinearMapper r03 = res.getLinearMapper(row + 0*accLhsProgress, col + 3);

      LinearMapper r10 = res.getLinearMapper(row + 1*accLhsProgress, col + 0);
      LinearMapper r11 = res.getLinearMapper(row + 1*accLhsProgress, col + 1);
      LinearMapper r12 = res.getLinearMapper(row + 1*accLhsProgress, col + 2);
      LinearMapper r13 = res.getLinearMapper(row + 1*accLhsProgress, col + 3);

      LinearMapper r20 = res.getLinearMapper(row + 2*accLhsProgress, col + 0);
      LinearMapper r21 = res.getLinearMapper(row + 2*accLhsProgress, col + 1);
      LinearMapper r22 = res.getLinearMapper(row + 2*accLhsProgress, col + 2);
      LinearMapper r23 = res.getLinearMapper(row + 2*accLhsProgress, col + 3);

      auto k = 0;
      for(; k < depth; k++)
      {
        RhsPacket prhs = pload<RhsPacket>(rhs_ptr);
        PacketBlock<RhsPacket, 4> pbrhs;
        pbrhs.packet[0] = pset1<RhsPacket>(prhs[0]);
        pbrhs.packet[1] = pset1<RhsPacket>(prhs[1]);
        pbrhs.packet[2] = pset1<RhsPacket>(prhs[2]);
        pbrhs.packet[3] = pset1<RhsPacket>(prhs[3]);

        LhsPacket plhs1 = pload<LhsPacket>(lhs_ptr1);
        LhsPacket plhs2 = pload<LhsPacket>(lhs_ptr2);
        LhsPacket plhs3 = pload<LhsPacket>(lhs_ptr3);

        acc1.packet[0] += plhs1*pbrhs.packet[0];
        acc1.packet[1] += plhs1*pbrhs.packet[1];
        acc1.packet[2] += plhs1*pbrhs.packet[2];
        acc1.packet[3] += plhs1*pbrhs.packet[3];

        acc2.packet[0] += plhs2*pbrhs.packet[0];
        acc2.packet[1] += plhs2*pbrhs.packet[1];
        acc2.packet[2] += plhs2*pbrhs.packet[2];
        acc2.packet[3] += plhs2*pbrhs.packet[3];

        acc3.packet[0] += plhs3*pbrhs.packet[0];
        acc3.packet[1] += plhs3*pbrhs.packet[1];
        acc3.packet[2] += plhs3*pbrhs.packet[2];
        acc3.packet[3] += plhs3*pbrhs.packet[3];

        lhs_ptr1 += (rows/accLhsProgress)*accLhsProgress;
        lhs_ptr2 += (rows/accLhsProgress)*accLhsProgress;
        lhs_ptr3 += (rows/accLhsProgress)*accLhsProgress;
        rhs_ptr += accRhsProgress;
      }

      r00.storePacket(0,r00.template loadPacket<ResPacket>(0) + acc1.packet[0]);
      r01.storePacket(0,r01.template loadPacket<ResPacket>(0) + acc1.packet[1]);
      r02.storePacket(0,r02.template loadPacket<ResPacket>(0) + acc1.packet[2]);
      r03.storePacket(0,r03.template loadPacket<ResPacket>(0) + acc1.packet[3]);

      r10.storePacket(0,r10.template loadPacket<ResPacket>(0) + acc2.packet[0]);
      r11.storePacket(0,r11.template loadPacket<ResPacket>(0) + acc2.packet[1]);
      r12.storePacket(0,r12.template loadPacket<ResPacket>(0) + acc2.packet[2]);
      r13.storePacket(0,r13.template loadPacket<ResPacket>(0) + acc2.packet[3]);

      r20.storePacket(0,r20.template loadPacket<ResPacket>(0) + acc3.packet[0]);
      r21.storePacket(0,r21.template loadPacket<ResPacket>(0) + acc3.packet[1]);
      r22.storePacket(0,r22.template loadPacket<ResPacket>(0) + acc3.packet[2]);
      r23.storePacket(0,r23.template loadPacket<ResPacket>(0) + acc3.packet[3]);
    }
    for(; row + 2*accLhsProgress <= lhsMap.get_packed_size(); row+=2*accLhsProgress)
    {
      const LhsScalar *lhs_ptr1 = lhsMap.get_packed_at(row + 0*accLhsProgress);
      const LhsScalar *lhs_ptr2 = lhsMap.get_packed_at(row + 1*accLhsProgress);
      const RhsScalar *rhs_ptr = rhsMap.get_packed_at(col/accRhsProgress);

      PacketBlock<AccPacket, 4> acc1;
      acc1.packet[0] = pset1<AccPacket>(0);
      acc1.packet[1] = pset1<AccPacket>(0);
      acc1.packet[2] = pset1<AccPacket>(0);
      acc1.packet[3] = pset1<AccPacket>(0);

      PacketBlock<AccPacket, 4> acc2;
      acc2.packet[0] = pset1<AccPacket>(0);
      acc2.packet[1] = pset1<AccPacket>(0);
      acc2.packet[2] = pset1<AccPacket>(0);
      acc2.packet[3] = pset1<AccPacket>(0);

      LinearMapper r00 = res.getLinearMapper(row + 0*accLhsProgress, col + 0);
      LinearMapper r01 = res.getLinearMapper(row + 0*accLhsProgress, col + 1);
      LinearMapper r02 = res.getLinearMapper(row + 0*accLhsProgress, col + 2);
      LinearMapper r03 = res.getLinearMapper(row + 0*accLhsProgress, col + 3);

      LinearMapper r10 = res.getLinearMapper(row + 1*accLhsProgress, col + 0);
      LinearMapper r11 = res.getLinearMapper(row + 1*accLhsProgress, col + 1);
      LinearMapper r12 = res.getLinearMapper(row + 1*accLhsProgress, col + 2);
      LinearMapper r13 = res.getLinearMapper(row + 1*accLhsProgress, col + 3);

      auto k = 0;
      for(; k < depth; k++)
      {
        RhsPacket prhs = pload<RhsPacket>(rhs_ptr);
        PacketBlock<RhsPacket, 4> pbrhs;
        pbrhs.packet[0] = pset1<RhsPacket>(prhs[0]);
        pbrhs.packet[1] = pset1<RhsPacket>(prhs[1]);
        pbrhs.packet[2] = pset1<RhsPacket>(prhs[2]);
        pbrhs.packet[3] = pset1<RhsPacket>(prhs[3]);

        LhsPacket plhs1 = pload<LhsPacket>(lhs_ptr1);
        LhsPacket plhs2 = pload<LhsPacket>(lhs_ptr2);

        acc1.packet[0] += plhs1*pbrhs.packet[0];
        acc1.packet[1] += plhs1*pbrhs.packet[1];
        acc1.packet[2] += plhs1*pbrhs.packet[2];
        acc1.packet[3] += plhs1*pbrhs.packet[3];

        acc2.packet[0] += plhs2*pbrhs.packet[0];
        acc2.packet[1] += plhs2*pbrhs.packet[1];
        acc2.packet[2] += plhs2*pbrhs.packet[2];
        acc2.packet[3] += plhs2*pbrhs.packet[3];

        lhs_ptr1 += (rows/accLhsProgress)*accLhsProgress;
        lhs_ptr2 += (rows/accLhsProgress)*accLhsProgress;
        rhs_ptr += accRhsProgress;
      }

      r00.storePacket(0,r00.template loadPacket<ResPacket>(0) + acc1.packet[0]);
      r01.storePacket(0,r01.template loadPacket<ResPacket>(0) + acc1.packet[1]);
      r02.storePacket(0,r02.template loadPacket<ResPacket>(0) + acc1.packet[2]);
      r03.storePacket(0,r03.template loadPacket<ResPacket>(0) + acc1.packet[3]);

      r10.storePacket(0,r10.template loadPacket<ResPacket>(0) + acc2.packet[0]);
      r11.storePacket(0,r11.template loadPacket<ResPacket>(0) + acc2.packet[1]);
      r12.storePacket(0,r12.template loadPacket<ResPacket>(0) + acc2.packet[2]);
      r13.storePacket(0,r13.template loadPacket<ResPacket>(0) + acc2.packet[3]);
    }
    for(; row + accLhsProgress <= lhsMap.get_packed_size(); row+=accLhsProgress)
    {
      const LhsScalar *lhs_ptr = lhsMap.get_packed_at(row);
      const RhsScalar *rhs_ptr = rhsMap.get_packed_at(col/accRhsProgress);
      PacketBlock<AccPacket, 4> acc;
      acc.packet[0] = pset1<AccPacket>(0);
      acc.packet[1] = pset1<AccPacket>(0);
      acc.packet[2] = pset1<AccPacket>(0);
      acc.packet[3] = pset1<AccPacket>(0);

      LinearMapper r0 = res.getLinearMapper(row, col + 0);
      LinearMapper r1 = res.getLinearMapper(row, col + 1);
      LinearMapper r2 = res.getLinearMapper(row, col + 2);
      LinearMapper r3 = res.getLinearMapper(row, col + 3);

      auto k = 0;
      for(; k < depth; k++)
      {
        RhsPacket prhs = pload<RhsPacket>(rhs_ptr);
        PacketBlock<RhsPacket, 4> pbrhs;
        pbrhs.packet[0] = pset1<RhsPacket>(prhs[0]);
        pbrhs.packet[1] = pset1<RhsPacket>(prhs[1]);
        pbrhs.packet[2] = pset1<RhsPacket>(prhs[2]);
        pbrhs.packet[3] = pset1<RhsPacket>(prhs[3]);

        LhsPacket plhs = pload<LhsPacket>(lhs_ptr);

#ifdef __NDEBUG__
        std::cout << "(" << row << "," << k << "," << col << ")" << std::endl;
        std::cout << "lhs " << plhs[0] << " " << plhs[1] << " " << plhs[2] << " " << plhs[3] << std::endl;
        std::cout << "rhs " << prhs[0] << " " << prhs[1] << " " << prhs[2] << " " << prhs[3] << std::endl;
#endif
        acc.packet[0] += plhs*pbrhs.packet[0];
        acc.packet[1] += plhs*pbrhs.packet[1];
        acc.packet[2] += plhs*pbrhs.packet[2];
        acc.packet[3] += plhs*pbrhs.packet[3];

        lhs_ptr += (rows/accLhsProgress)*accLhsProgress;
        rhs_ptr += accRhsProgress;
      }

      r0.storePacket(0,r0.template loadPacket<ResPacket>(0) + acc.packet[0]);
      r1.storePacket(0,r1.template loadPacket<ResPacket>(0) + acc.packet[1]);
      r2.storePacket(0,r2.template loadPacket<ResPacket>(0) + acc.packet[2]);
      r3.storePacket(0,r3.template loadPacket<ResPacket>(0) + acc.packet[3]);
    }
    auto row_residue = 0;
    for(;row < rows; row++)
    {
      const LhsScalar *lhs_ptr = lhsMap.get_residue_at(row_residue);
      const RhsScalar *rhs_ptr = rhsMap.get_packed_at(col/accRhsProgress);
      PacketBlock<AccPacket, 1> acc;
      acc.packet[0] = pset1<AccPacket>(0);

      auto k = 0;
      for(; k < depth; k++)
      {
        RhsPacket prhs = pload<RhsPacket>(rhs_ptr);
        LhsPacket plhs = pset1<LhsPacket>(*lhs_ptr);

#ifdef __NDEBUG__
        std::cout << "(" << row << "," << k << "," << col << ")" << std::endl;
        std::cout << "lhs " << plhs[0] << " " << plhs[1] << " " << plhs[2] << " " << plhs[3] << std::endl;
        std::cout << "rhs " << prhs[0] << " " << prhs[1] << " " << prhs[2] << " " << prhs[3] << std::endl;
#endif
        acc.packet[0] += (*lhs_ptr)*prhs;

        lhs_ptr++;
        rhs_ptr += accRhsProgress;
      }

      res(row, col + 0) += acc.packet[0][0];
      res(row, col + 1) += acc.packet[0][1];
      res(row, col + 2) += acc.packet[0][2];
      res(row, col + 3) += acc.packet[0][3];
      row_residue++;
    }
  }
  auto col_residue = 0;
  for(; col < cols; col++)
  {
    auto row = 0;
    for(; row + accLhsProgress <= lhsMap.get_packed_size(); row+=accLhsProgress)
    {
      const LhsScalar *lhs_ptr = lhsMap.get_packed_at(row);
      const RhsScalar *rhs_ptr = rhsMap.get_residue_at(col_residue);
      PacketBlock<AccPacket, 1> acc;
      acc.packet[0] = pset1<AccPacket>(0);

      LinearMapper r0 = res.getLinearMapper(row, col + 0);

      auto k = 0;
      for(; k < depth; k++)
      {
        RhsPacket prhs = pset1<RhsPacket>(*rhs_ptr);

        LhsPacket plhs = pload<LhsPacket>(lhs_ptr);

#ifdef __NDEBUG__
        std::cout << "(" << row << "," << k << "," << col << ")" << std::endl;
        std::cout << "lhs " << plhs[0] << " " << plhs[1] << " " << plhs[2] << " " << plhs[3] << std::endl;
        std::cout << "rhs " << prhs[0] << " " << prhs[1] << " " << prhs[2] << " " << prhs[3] << std::endl;
#endif
        acc.packet[0] += plhs*prhs;

        lhs_ptr += (rows/accLhsProgress)*accLhsProgress;
        rhs_ptr++;
      }

      r0.storePacket(0,r0.template loadPacket<ResPacket>(0) + acc.packet[0]);
    }
    auto row_residue = 0;
    for(;row < rows; row++)
    {
      const LhsScalar *lhs_ptr = lhsMap.get_residue_at(row_residue);
      const RhsScalar *rhs_ptr = rhsMap.get_residue_at(col_residue);
      AccScalar acc = 0;
      auto k = 0;
      for(; k < depth; k++)
      {
#ifdef __NDEBUG__
        std::cout << "(" << row << "," << k << "," << col << ")" << std::endl;
        std::cout << "lhs " << plhs[0] << " " << plhs[1] << " " << plhs[2] << " " << plhs[3] << std::endl;
        std::cout << "rhs " << prhs[0] << " " << prhs[1] << " " << prhs[2] << " " << prhs[3] << std::endl;
#endif
        acc += (*lhs_ptr)*(*rhs_ptr);

        lhs_ptr++;
        rhs_ptr++;
      }

      //r0.storePacket(0,r0.template loadPacket<ResPacket>(0) + acc.packet[0]);
      res(row, col) += acc;
      row_residue++;
    }
    col_residue++;
  }
}

template<typename ResScalar, typename AccScalar, typename LhsScalar, typename RhsScalar, typename Index, typename DataMapper>
EIGEN_STRONG_INLINE void gemm_old(const DataMapper& res, const LhsScalar* blockA, const RhsScalar* blockB,
          Index rows, Index depth, Index cols, ResScalar alpha, Index strideA, Index strideB, Index offsetA, Index offsetB)
{
  using AccPacket = typename packet_traits<AccScalar>::type;
  using LhsPacket = typename packet_traits<LhsScalar>::type;
  using RhsPacket = typename packet_traits<RhsScalar>::type;
  using ResPacket = typename packet_traits<ResScalar>::type;

  ResPacket pAlpha = pset1<ResPacket>(alpha);

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

  if( strideA == -1 ) strideA = depth;
  if( strideB == -1 ) strideB = depth;

  int accLhsProgress = 4;
  int accRhsProgress = 4;

  PackMap<LhsScalar, LhsPacket, Index> lhsMap(blockA, rows, depth, offsetA, strideA);
  PackMap<RhsScalar, RhsPacket, Index, false> rhsMap(blockB, depth, cols, offsetB, strideB);
  auto col = 0;
  for(; col < rhsMap.get_packed_size(); col+=accRhsProgress)
  {
    for(auto k = 0; k < depth; k++)
    {
      const LhsScalar *lhs_ptr = lhsMap.get_packed_at(k);
      const RhsScalar *rhs_ptr = rhsMap.get_packed_at(col/accRhsProgress) + k*accRhsProgress;
      PacketBlock<AccPacket, 4> acc;
      RhsPacket prhs = pload<RhsPacket>(rhs_ptr);
      PacketBlock<RhsPacket, 4> pbrhs;
      pbrhs.packet[0] = pset1<RhsPacket>(prhs[0]);
      pbrhs.packet[1] = pset1<RhsPacket>(prhs[1]);
      pbrhs.packet[2] = pset1<RhsPacket>(prhs[2]);
      pbrhs.packet[3] = pset1<RhsPacket>(prhs[3]);
      auto row = 0;
      using LinearMapper = typename DataMapper::LinearMapper;
      for(; row < lhsMap.get_packed_size(); row+=accLhsProgress)
      {
        LinearMapper r0 = res.getLinearMapper(row, col + 0);
        LinearMapper r1 = res.getLinearMapper(row, col + 1);
        LinearMapper r2 = res.getLinearMapper(row, col + 2);
        LinearMapper r3 = res.getLinearMapper(row, col + 3);

        LhsPacket plhs = pload<LhsPacket>(lhs_ptr);
#ifdef __NDEBUG__
        std::cout << "(" << row << "," << k << "," << col << ")" << std::endl;
        std::cout << "lhs " << plhs[0] << " " << plhs[1] << " " << plhs[2] << " " << plhs[3] << std::endl;
        std::cout << "rhs " << prhs[0] << " " << prhs[1] << " " << prhs[2] << " " << prhs[3] << std::endl;
#endif
        acc.packet[0] = plhs*pbrhs.packet[0];
        acc.packet[1] = plhs*pbrhs.packet[1];
        acc.packet[2] = plhs*pbrhs.packet[2];
        acc.packet[3] = plhs*pbrhs.packet[3];

        r0.storePacket(0,r0.template loadPacket<ResPacket>(0) + acc.packet[0]);
        r1.storePacket(0,r1.template loadPacket<ResPacket>(0) + acc.packet[1]);
        r2.storePacket(0,r2.template loadPacket<ResPacket>(0) + acc.packet[2]);
        r3.storePacket(0,r3.template loadPacket<ResPacket>(0) + acc.packet[3]);
        lhs_ptr += accLhsProgress;
      }
      auto residue = 0;
      for(;row < rows; row++)
      {
        LhsScalar lhs = *(lhsMap.get_residue_at(residue) + k);
#ifdef __NDEBUG__
        std::cout << "(" << row << "," << k << "," << col << ")" << std::endl;
        std::cout << "lhs " << lhs << " (" << prhs[0] << " " << prhs[1] << " " << prhs[2] << " " << prhs[3] << ")" << std::endl;
#endif
        res(row, col + 0) += lhs*prhs[0];
        res(row, col + 1) += lhs*prhs[1];
        res(row, col + 2) += lhs*prhs[2];
        res(row, col + 3) += lhs*prhs[3];
        residue++;
      }
    }
  }
  auto colResidue = 0;
  for(;col < cols; col++)
  {
    for(auto k = 0; k < depth; k++)
    {
      const LhsScalar *lhs_ptr = lhsMap.get_packed_at(k);
      const RhsScalar *rhs_ptr = rhsMap.get_residue_at(colResidue) + k;
      AccPacket acc;

      RhsPacket prhs = pset1<RhsPacket>(*rhs_ptr);

      auto row = 0;
      using LinearMapper = typename DataMapper::LinearMapper;
      for(; row < lhsMap.get_packed_size(); row+=accLhsProgress)
      {
        LinearMapper r0 = res.getLinearMapper(row, col + 0);

        LhsPacket plhs = pload<LhsPacket>(lhs_ptr);
#ifdef __DEBUG__
        std::cout << "(" << row << "," << k << "," << col << ")" << std::endl;
        std::cout << "lhs " << plhs[0] << " " << plhs[1] << " " << plhs[2] << " " << plhs[3] << std::endl;
        std::cout << "rhs " << prhs[0] << " " << prhs[1] << " " << prhs[2] << " " << prhs[3] << std::endl;
#endif
        acc = plhs*prhs;

        r0.storePacket(0,r0.template loadPacket<ResPacket>(0) + acc);
        lhs_ptr += accLhsProgress;
      }
      auto residue = 0;
      for(;row < rows; row++)
      {
        LhsScalar lhs = *(lhsMap.get_residue_at(residue) + k);
#ifdef __DEBUG__
        std::cout << "(" << row << "," << k << "," << col << ")" << std::endl;
        std::cout << "lhs " << lhs << " (" << prhs[0] << " " << prhs[1] << " " << prhs[2] << " " << prhs[3] << ")" << std::endl;
#endif
        res(row, col + 0) += lhs*prhs[0];
        residue++;
      }
    }
    colResidue++;
  }
}
#endif

template<int Architecture, int CPU, typename LhsScalar, typename RhsScalar>
constexpr int SHAPES_COUNT = 4;

constexpr int SHAPES_DIMENSION = 4;
constexpr int SHAPES_LHS_DIMENSION = 0;
constexpr int SHAPES_DEP_DIMENSION = 1;
constexpr int SHAPES_RHS_DIMENSION = 2;
constexpr int SHAPES_POINTER = 3;
constexpr int SHAPES_POINTER_END = -1;

template<int Architecture, int CPU, typename Scalar, bool isLhs>
constexpr int PACK_SHAPES_COUNT = 2;
constexpr int PACK_SHAPES_DIMENSION = 3;
constexpr int PACK_SHAPES_POINTER = 2;
constexpr int PACK_SHAPES_END = -1;


// lhs_progress x depth_progress x rhs_progress (depth_progress > 1 matrix ops) x pointer to next rhs_progress on the shapes map
template<int Architecture, int CPU, typename LhsScalar, typename RhsScalar>
constexpr int SHAPES[SHAPES_COUNT<Architecture, CPU, LhsScalar,RhsScalar>][SHAPES_DIMENSION] = {{1,1,1,SHAPES_POINTER_END},{4,1,1,0},{1,1,4,1},{4,1,4,1}};

// d1progress x d2progress
template<int Architecture, int CPU, typename Scalar, bool isLhs>
constexpr int PACK_SHAPES[PACK_SHAPES_COUNT<Architecture, CPU, Scalar, isLhs>][PACK_SHAPES_DIMENSION] = {{1,1,PACK_SHAPES_END},{4,1,0}};

template<int Architecture, int CPU, typename Scalar>
constexpr int PACK_SHAPES<Architecture, CPU, Scalar, false>[PACK_SHAPES_COUNT<Architecture, CPU, Scalar, false>][PACK_SHAPES_DIMENSION] = {{1,1,PACK_SHAPES_END},{4,1,0}};

template<int Architecture, int CPU, typename Index, typename Scalar, bool isLhs, typename DataMapper, bool Conjugate, bool PanelMode, int StorageOrder, int M, int N>
struct PackingOperator
{
  EIGEN_STRONG_INLINE Scalar* operator()(Index d1Idx, Index d2Idx, Scalar *block, const DataMapper& data)
  {
    std::cout << M << "x" << N << " ( " << d1Idx << ", " << d2Idx <<") -> ( " << d1Idx + M << ", " << d2Idx + N << ") ";
    Scalar *c = block;
    for(auto i = 0; i < M; i++)
      for(auto j = 0; j < N; j++)
      {
        if(isLhs)
          *c = data(d1Idx + i, d2Idx + j);
        else
          *c = data(d2Idx + j, d1Idx + i);
        std::cout << *c << " ";
        c++;
      }
    std::cout << std::endl;
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
  Index d2Size;
  PackMapCalculator<Architecture, CPU, Index, Scalar, DataMapper, isLhs, PACK_SHAPES_COUNT<Architecture, CPU, Scalar, isLhs>-1> pmc;

  PackMap(const Scalar *base, Index d2Size) : pBase(base), pCur(base), d2Size(d2Size) {}

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

  EIGEN_STRONG_INLINE void scale(ResScalar alpha)
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
        dest(row + i, col + j) = dt[i][j];
      }
    }
  }
};

template<int Architecture, int CPU, typename Index, typename LhsScalar, typename RhsScalar, typename AccScalar, typename ResScalar, typename DataMapper, int SHAPE_IDX, int M, int K, int N>
struct MicroKernel
{
  EIGEN_STRONG_INLINE void operator()(PackMap<Architecture, CPU, Index, LhsScalar, DataMapper, true>& lhsPackMap, 
                                      PackMap<Architecture, CPU, Index, RhsScalar, DataMapper, false>& rhsPackMap, 
                                      Index rowIdx, Index colIdx, Index depthIdx,
                                      Accumulator<Architecture, CPU, AccScalar, ResScalar, DataMapper, M, N>& acc)
  {
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

template<int Architecture, int CPU, typename Index, typename LhsScalar, typename RhsScalar, typename AccScalar, typename ResScalar, typename DataMapper, int RHS_SHAPE_IDX, int LHS_SHAPE_IDX, int IDX>
struct DepthLoopStruct
{
  DepthLoopStruct<Architecture, CPU, Index, LhsScalar, RhsScalar, AccScalar, ResScalar, DataMapper, RHS_SHAPE_IDX, LHS_SHAPE_IDX, IDX-1> depthLS;
  EIGEN_STRONG_INLINE void operator()(Index rowIdx, Index colIdx, Index depthIdx, const DataMapper& res, const LhsScalar* blockA, const RhsScalar*blockB, 
                          Index rows, Index depth, Index cols, ResScalar alpha, Index strideA, Index strideB, Index offsetA, Index offsetB, PackMap<Architecture, CPU, Index, LhsScalar, DataMapper, true>& lhsPackMap, PackMap<Architecture, CPU, Index, RhsScalar, DataMapper, false>& rhsPackMap)
  {
    constexpr auto rhsProgress      = SHAPES<Architecture, CPU, LhsScalar, RhsScalar>[RHS_SHAPE_IDX][SHAPES_RHS_DIMENSION];
    constexpr auto lhsProgress      = SHAPES<Architecture, CPU, LhsScalar, RhsScalar>[LHS_SHAPE_IDX][SHAPES_LHS_DIMENSION];
    constexpr auto depthProgress    = SHAPES<Architecture, CPU, LhsScalar, RhsScalar>[IDX][SHAPES_DEP_DIMENSION];

    if(rhsProgress == SHAPES<Architecture, CPU, LhsScalar, RhsScalar>[IDX][SHAPES_RHS_DIMENSION] && lhsProgress == SHAPES<Architecture, CPU, LhsScalar, RhsScalar>[IDX][SHAPES_LHS_DIMENSION])
    {
      MicroKernel<Architecture, CPU, Index, LhsScalar, RhsScalar, AccScalar, ResScalar, DataMapper, IDX, lhsProgress, depthProgress, rhsProgress> mkt;
      Accumulator<Architecture, CPU, AccScalar, ResScalar, DataMapper, lhsProgress, rhsProgress> acc;
      acc.zero();
      for(; depthIdx + depthProgress <= depth; depthIdx+=depthProgress)
      {
          mkt(lhsPackMap, rhsPackMap, rowIdx, colIdx, depthIdx, acc);
      }
      acc.scale(alpha);
      acc.store(res, rowIdx, colIdx);
    }
    depthLS(rowIdx, colIdx, depthIdx, res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB, lhsPackMap, rhsPackMap);
  }
};

template<int Architecture, int CPU, typename Index, typename LhsScalar, typename RhsScalar, typename AccScalar, typename ResScalar, typename DataMapper, int RHS_SHAPE_IDX, int LHS_SHAPE_IDX>
struct DepthLoopStruct<Architecture, CPU, Index, LhsScalar, RhsScalar, AccScalar, ResScalar, DataMapper, RHS_SHAPE_IDX, LHS_SHAPE_IDX, -1>
{
  EIGEN_STRONG_INLINE void operator()(Index, Index, Index, const DataMapper&, const LhsScalar*, const RhsScalar*,
                          Index, Index, Index, ResScalar, Index, Index, Index, Index, PackMap<Architecture, CPU, Index, LhsScalar, DataMapper, true>&, PackMap<Architecture, CPU, Index, RhsScalar, DataMapper, false>&) {}
};

template<int Architecture, int CPU, typename Index, typename LhsScalar, typename RhsScalar, typename AccScalar, typename ResScalar, typename DataMapper, int RHS_SHAPE_IDX, int IDX>
struct LhsLoopStruct
{
  LhsLoopStruct<Architecture, CPU, Index, LhsScalar, RhsScalar, AccScalar, ResScalar, DataMapper, RHS_SHAPE_IDX, IDX-1> lhsLS;
  EIGEN_STRONG_INLINE void operator()(Index rowIdx, int colIdx, const DataMapper& res, const LhsScalar* blockA, const RhsScalar*blockB, 
                          Index rows, Index depth, Index cols, ResScalar alpha, Index strideA, Index strideB, Index offsetA, Index offsetB, PackMap<Architecture, CPU, Index, LhsScalar, DataMapper, true>& lhsPackMap, PackMap<Architecture, CPU, Index, RhsScalar, DataMapper, false>& rhsPackMap)
  {
    constexpr auto lhsProgress = SHAPES<Architecture, CPU, LhsScalar, RhsScalar>[IDX][SHAPES_LHS_DIMENSION];

    DepthLoopStruct<Architecture, CPU, Index, LhsScalar, RhsScalar, AccScalar, ResScalar, DataMapper, RHS_SHAPE_IDX, IDX, IDX> depthLS;
    for(;rowIdx + lhsProgress <= rows; rowIdx+=lhsProgress)
    {
      lhsPackMap.moveTo(rowIdx);
      rhsPackMap.moveTo(colIdx);
      depthLS(rowIdx, colIdx, 0, res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB, lhsPackMap, rhsPackMap);
    }
    lhsLS(rowIdx, colIdx, res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB, lhsPackMap, rhsPackMap);
  }
};

template<int Architecture, int CPU, typename Index, typename LhsScalar, typename RhsScalar, typename AccScalar, typename ResScalar, typename DataMapper, int RHS_SHAPE_IDX>
struct LhsLoopStruct<Architecture, CPU, Index, LhsScalar, RhsScalar, AccScalar, ResScalar, DataMapper, RHS_SHAPE_IDX, -1>
{
  EIGEN_STRONG_INLINE void operator()(Index, Index, const DataMapper&, const LhsScalar*, const RhsScalar*,
                          Index, Index, Index, ResScalar, Index, Index, Index, Index, PackMap<Architecture, CPU, Index, LhsScalar, DataMapper, true>&, PackMap<Architecture, CPU, Index, RhsScalar, DataMapper, false>&) {}
};

template<int Architecture, int CPU, typename Index, typename LhsScalar, typename RhsScalar, typename AccScalar, typename ResScalar, typename DataMapper, int IDX>
struct RhsLoopStruct
{
  static constexpr auto PREVIOUS = SHAPES<Architecture, CPU, LhsScalar, RhsScalar>[IDX][SHAPES_POINTER];
  RhsLoopStruct<Architecture, CPU, Index, LhsScalar, RhsScalar, AccScalar, ResScalar, DataMapper, PREVIOUS> rhsLS;

  EIGEN_STRONG_INLINE void operator()(Index colIdx, const DataMapper& res, const LhsScalar* blockA, const RhsScalar*blockB, 
                          Index rows, Index depth, Index cols, ResScalar alpha, Index strideA, Index strideB, Index offsetA, Index offsetB, PackMap<Architecture, CPU, Index, LhsScalar, DataMapper, true>& lhsPackMap, PackMap<Architecture, CPU, Index, RhsScalar, DataMapper, false>& rhsPackMap)
  {
    constexpr auto rhsProgress = SHAPES<Architecture, CPU, LhsScalar, RhsScalar>[IDX][SHAPES_RHS_DIMENSION];

    for(;colIdx + rhsProgress <= cols; colIdx+=rhsProgress)
    {
      LhsLoopStruct<Architecture, CPU, Index, LhsScalar, RhsScalar, AccScalar, ResScalar, DataMapper, IDX, IDX> lhsLS;
      lhsLS(0, colIdx, res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB, lhsPackMap, rhsPackMap);
    }
    rhsLS(colIdx, res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB, lhsPackMap, rhsPackMap);
  }
};

template<int Architecture, int CPU, typename Index, typename LhsScalar, typename RhsScalar, typename AccScalar, typename ResScalar, typename DataMapper>
struct RhsLoopStruct<Architecture, CPU, Index, LhsScalar, RhsScalar, AccScalar, ResScalar, DataMapper, -1>
{
  EIGEN_STRONG_INLINE void operator()(Index colIdx, const DataMapper&, const LhsScalar*, const RhsScalar*, 
                          Index, Index, Index, ResScalar, Index, Index, Index, Index, PackMap<Architecture, CPU, Index, LhsScalar, DataMapper, true>&, PackMap<Architecture, CPU, Index, RhsScalar, DataMapper, false>&) {}
};

template<int Architecture, int CPU, typename ResScalar, typename AccScalar, typename LhsScalar, typename RhsScalar, typename Index, typename DataMapper>
EIGEN_STRONG_INLINE void gemm(const DataMapper& res, const LhsScalar* blockA, const RhsScalar* blockB,
          Index rows, Index depth, Index cols, ResScalar alpha, Index strideA, Index strideB, Index offsetA, Index offsetB)
{
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

  RhsLoopStruct<Architecture, CPU, Index, LhsScalar, RhsScalar, AccScalar, ResScalar, DataMapper, SHAPES_COUNT<0, 0, LhsScalar, RhsScalar>-1> rhsLS;
  PackMap<Architecture, CPU, Index, LhsScalar, DataMapper, true> lhsPackMap(blockA, depth);
  PackMap<Architecture, CPU, Index, RhsScalar, DataMapper, false> rhsPackMap(blockB, depth);
  rhsLS(0, res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB, lhsPackMap, rhsPackMap);
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