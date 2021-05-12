// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2021 Everton Constantino (everton.constantino@hotmail.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKING_OPS_NEON_H
#define EIGEN_PACKING_OPS_NEON_H

namespace Eigen {

namespace internal {

#ifdef __ENABLE_CUSTOM_PACKING__

template<int CPU, typename Scalar, bool isLhs>
constexpr int PACK_SHAPES_COUNT<0, CPU, Scalar, isLhs> = 3;

template<int CPU, typename Scalar>
constexpr int PACK_SHAPES_COUNT<0, CPU, Scalar, true> = 4;

template<int CPU, typename Scalar, bool isLhs>
constexpr int PACK_SHAPES<0, CPU, Scalar, isLhs>[PACK_SHAPES_COUNT<0, CPU, Scalar, isLhs>][PACK_SHAPES_DIMENSION] = {{1,1,PACK_SHAPES_END},{4,1,0},{4,4,0}};

template<int CPU, typename Scalar>
constexpr int PACK_SHAPES<0, CPU, Scalar, true>[PACK_SHAPES_COUNT<0, CPU, Scalar, true>][PACK_SHAPES_DIMENSION] = {{1,1,PACK_SHAPES_END},{4,1,0},{4,4,0},{8,1,2}};

template<int CPU, typename Index, typename Scalar, bool isLhs, typename DataMapper, bool Conjugate, bool PanelMode, int StorageOrder>
struct PackingOperator<0, CPU, Index, Scalar, isLhs, DataMapper, Conjugate, PanelMode, StorageOrder, 4, 4>
{
  EIGEN_STRONG_INLINE Scalar* operator()(Index d1Idx, Index d2Idx, Scalar *block, const DataMapper& data)
  {
    using Packet = typename packet_traits<Scalar>::type;
    constexpr int vectorSize = packet_traits<Scalar>::size;

    Scalar *c = block;

    if(!isLhs)
    {
      int tD = d1Idx;
      d1Idx = d2Idx;
      d2Idx = tD;
    }

    if(isLhs && StorageOrder == ColMajor || !isLhs && StorageOrder == RowMajor)
    {
      Packet p0 = data.template loadPacket<Packet>(d1Idx, d2Idx + 0);
      Packet p1 = data.template loadPacket<Packet>(d1Idx, d2Idx + 1);
      Packet p2 = data.template loadPacket<Packet>(d1Idx, d2Idx + 2);
      Packet p3 = data.template loadPacket<Packet>(d1Idx, d2Idx + 3);

      pstore<Scalar>(c + 0*vectorSize, p0);
      pstore<Scalar>(c + 1*vectorSize, p1);
      pstore<Scalar>(c + 2*vectorSize, p2);
      pstore<Scalar>(c + 3*vectorSize, p3);
      c+=4*vectorSize;
    } else {
      PacketBlock<Packet, 4> pblock;

      pblock.packet[0] = data.template loadPacket<Packet>(d1Idx, d2Idx + 0);
      pblock.packet[1] = data.template loadPacket<Packet>(d1Idx, d2Idx + 1);
      pblock.packet[2] = data.template loadPacket<Packet>(d1Idx, d2Idx + 2);
      pblock.packet[3] = data.template loadPacket<Packet>(d1Idx, d2Idx + 3);

      ptranspose(pblock);

      pstore<Scalar>(c + 0*vectorSize, pblock.packet[0]);
      pstore<Scalar>(c + 1*vectorSize, pblock.packet[1]);
      pstore<Scalar>(c + 2*vectorSize, pblock.packet[2]);
      pstore<Scalar>(c + 3*vectorSize, pblock.packet[3]);
      c+=4*vectorSize;
    }
    return c;
  }
};

template<int CPU, typename Index, typename Scalar, bool isLhs, typename DataMapper, bool Conjugate, bool PanelMode, int StorageOrder>
struct PackingOperator<0, CPU, Index, Scalar, isLhs, DataMapper, Conjugate, PanelMode, StorageOrder, 8, 1>
{
  EIGEN_STRONG_INLINE Scalar* operator()(Index d1Idx, Index d2Idx, Scalar *block, const DataMapper& data)
  {
    using Packet = typename packet_traits<Scalar>::type;
    Scalar *c = block;
    if(isLhs && StorageOrder == ColMajor)
    {
        Packet p = data.template loadPacket<Packet>(d1Idx + 0, d2Idx);
        pstore<Scalar>(c, p);
        c+=4;
        p = data.template loadPacket<Packet>(d1Idx + 4, d2Idx);
        pstore<Scalar>(c, p);
        c+=4;
    } else if(!isLhs && StorageOrder == RowMajor) {
        Packet p = data.template loadPacket<Packet>(d2Idx, d1Idx + 0);
        pstore<Scalar>(c, p);
        c+=4;
        p = data.template loadPacket<Packet>(d2Idx, d1Idx + 4);
        pstore<Scalar>(c, p);
        c+=4;
    } else {
      if(isLhs)
      {
        *c = data(d1Idx + 0, d2Idx + 0);
        c++;
        *c = data(d1Idx + 1, d2Idx + 0);
        c++;
        *c = data(d1Idx + 2, d2Idx + 0);
        c++;
        *c = data(d1Idx + 3, d2Idx + 0);
        c++;
        *c = data(d1Idx + 0, d2Idx + 4);
        c++;
        *c = data(d1Idx + 1, d2Idx + 4);
        c++;
        *c = data(d1Idx + 2, d2Idx + 4);
        c++;
        *c = data(d1Idx + 3, d2Idx + 4);
        c++;
      } else {
        *c = data(d2Idx, d1Idx + 0);
        c++;
        *c = data(d2Idx, d1Idx + 1);
        c++;
        *c = data(d2Idx, d1Idx + 2);
        c++;
        *c = data(d2Idx, d1Idx + 3);
        c++;
        *c = data(d2Idx + 4, d1Idx + 0);
        c++;
        *c = data(d2Idx + 4, d1Idx + 1);
        c++;
        *c = data(d2Idx + 4, d1Idx + 2);
        c++;
        *c = data(d2Idx + 4, d1Idx + 3);
        c++;
      }
    }
    return c;
  }
};

template<int CPU, typename Index, typename Scalar, bool isLhs, typename DataMapper, bool Conjugate, bool PanelMode, int StorageOrder>
struct PackingOperator<0, CPU, Index, Scalar, isLhs, DataMapper, Conjugate, PanelMode, StorageOrder, 4, 1>
{
  EIGEN_STRONG_INLINE Scalar* operator()(Index d1Idx, Index d2Idx, Scalar *block, const DataMapper& data)
  {
    using Packet = typename packet_traits<Scalar>::type;
    Scalar *c = block;
    if(isLhs && StorageOrder == ColMajor)
    {
        Packet p = data.template loadPacket<Packet>(d1Idx, d2Idx);
        pstore<Scalar>(c, p);
        c+=4;
    } else if(!isLhs && StorageOrder == RowMajor) {
        Packet p = data.template loadPacket<Packet>(d2Idx, d1Idx);
        pstore<Scalar>(c, p);
        c+=4;
    } else {
      if(isLhs)
      {
        *c = data(d1Idx + 0, d2Idx);
        c++;
        *c = data(d1Idx + 1, d2Idx);
        c++;
        *c = data(d1Idx + 2, d2Idx);
        c++;
        *c = data(d1Idx + 3, d2Idx);
        c++;
      } else {
        *c = data(d2Idx, d1Idx + 0);
        c++;
        *c = data(d2Idx, d1Idx + 1);
        c++;
        *c = data(d2Idx, d1Idx + 2);
        c++;
        *c = data(d2Idx, d1Idx + 3);
        c++;
      }
    }
    return c;
  }
};

#endif // __ENABLE_CUSTOM_PACKING__

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_PACKING_OPS_NEON_H