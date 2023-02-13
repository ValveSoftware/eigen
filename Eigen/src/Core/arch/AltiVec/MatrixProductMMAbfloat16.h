#ifndef EIGEN_MATRIX_PRODUCT_MMA_BFLOAT16_ALTIVEC_H
#define EIGEN_MATRIX_PRODUCT_MMA_BFLOAT16_ALTIVEC_H

#if EIGEN_COMP_LLVM
#define BFLOAT16_UNROLL _Pragma("unroll 8")
#else
#define BFLOAT16_UNROLL _Pragma("GCC unroll(8)")
#endif

namespace Eigen {

namespace internal {

template<bool zero>
EIGEN_ALWAYS_INLINE Packet8bf loadBfloat16(const bfloat16* indexA)
{
  Packet8bf lhs1 = ploadu<Packet8bf>(indexA);
  if(zero){
    Packet8bf lhs2 = pset1<Packet8bf>(Eigen::bfloat16(0));
    return vec_mergeh(lhs1.m_val, lhs2.m_val);
  } else {
    return lhs1;
  }
}

template<bool zero>
EIGEN_ALWAYS_INLINE Packet8bf loadRhsBfloat16(const bfloat16* blockB, Index strideB, Index i)
{
  return loadBfloat16<zero>(blockB + strideB*i);
}

template<Index num_acc, Index num_packets, bool zero, bool rhsExtraCols, bool lhsExtraRows>
EIGEN_ALWAYS_INLINE void KLoop
(
  const bfloat16* indexA,
  const bfloat16* indexB,
  __vector_quad (&quad_acc)[num_acc],
  Index strideB,
  Index k,
  Index offsetB,
  Index extra_cols,
  Index extra_rows
)
{
  Packet8bf lhs = loadBfloat16<zero>(indexA + k*(lhsExtraRows ? extra_rows : num_packets)); //a packet of bfloat16 has 8 elements
  Packet8bf rhs[num_acc];

  for(Index i = 0; i < (num_acc - (rhsExtraCols ? 1 : 0)); i++){
    rhs[i] = loadRhsBfloat16<zero>(indexB + k*4, strideB, i);
  }
  if(rhsExtraCols) {
    rhs[num_acc - 1] = loadRhsBfloat16<zero>(indexB + k*extra_cols - offsetB, strideB, num_acc - 1);
  }

  BFLOAT16_UNROLL
  for (Index i = 0; i < num_acc; i++) {
    __builtin_mma_xvbf16ger2pp(&(quad_acc[i]), reinterpret_cast<Packet16uc>(rhs[i].m_val), reinterpret_cast<Packet16uc>(lhs.m_val));
  }
}

template <bool rhsExtraCols, bool lhsExtraRows>
EIGEN_ALWAYS_INLINE void storeResults(Packet4f (&acc)[4], Index rows, const Packet4f pAlpha, float* result, Index extra_cols, Index extra_rows)
{
  Index x = 0;
  do{
    Packet4f result_block = ploadu<Packet4f>(result);
    result_block = pmadd(acc[x], pAlpha, result_block);
    if (lhsExtraRows) {
      pstoreu_partial(result, result_block, extra_rows);
    } else {
      pstoreu(result, result_block);
    }
    result += rows;
  } while (++x < (rhsExtraCols ? extra_cols : 4));
}

#define MAX_BFLOAT16_ACC   8

template<const Index num_acc, const Index num_packets, bool rhsExtraCols, bool lhsExtraRows>
void colLoopBody(Index& col, Index depth, Index cols, Index rows, const Packet4f pAlpha, const bfloat16* indexA, const bfloat16* indexB, Index strideB, Index offsetB, float* result, Index extra_rows)
{
  const Index step = (num_acc * 4); //each accumulator has 4 elements
  const Index extra_cols = (rhsExtraCols) ? (cols & 3) : 0;

  do{
    for(Index offset_row = 0; offset_row < num_packets; offset_row += 4, indexA += 8, result += 4) {
      Index k;
      Packet4f acc[num_acc][4];
      __vector_quad quad_acc[num_acc];

      BFLOAT16_UNROLL
      for(k = 0; k < num_acc; k++)
        __builtin_mma_xxsetaccz(&(quad_acc[k]));

      for(k = 0; k + 2 <= depth; k += 2){
        KLoop<num_acc, num_packets, false, rhsExtraCols, lhsExtraRows>(indexA, indexB, quad_acc, strideB, k, offsetB, extra_cols, extra_rows);
      }
      if(depth&1){
        KLoop<num_acc, num_packets, true, rhsExtraCols, lhsExtraRows>(indexA - offset_row, indexB, quad_acc, strideB, k, offsetB, extra_cols, extra_rows);
      }

      BFLOAT16_UNROLL
      for(k = 0; k < num_acc; k++)
        __builtin_mma_disassemble_acc((void*)acc[k], &(quad_acc[k]));

      for(k = 0; k < (num_acc - 1); k++){
        storeResults<false, lhsExtraRows>(acc[k], rows, pAlpha, result + k*4*rows, extra_cols, extra_rows);
      }
      storeResults<rhsExtraCols, lhsExtraRows>(acc[k], rows, pAlpha, result + k*4*rows, extra_cols, extra_rows);
    }

    indexA -= num_packets*2;
    indexB += strideB*num_acc;
    result += (rows*step - num_packets);
  } while(!rhsExtraCols && (num_acc == MAX_BFLOAT16_ACC) && (step <= cols - (col += step)));
}

template<const Index num_acc, const Index num_packets, bool rhsExtraCols, bool lhsExtraRows>
EIGEN_ALWAYS_INLINE void colLoopBodyExtraN(Index col, Index depth, Index cols, Index rows, const Packet4f pAlpha, const bfloat16* indexA, const bfloat16* blockB, Index strideB, Index offsetB, float* result, Index extra_rows)
{
  if (MAX_BFLOAT16_ACC > num_acc) {
    colLoopBody<num_acc + (rhsExtraCols ? 1 : 0), num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB, result, extra_rows);
  }
}

template<const Index num_packets, bool rhsExtraCols, bool lhsExtraRows>
void colLoopBodyExtra(Index col, Index depth, Index cols, Index rows, const Packet4f pAlpha, const bfloat16* indexA, const bfloat16* blockB, Index strideB, Index offsetB, float* result, Index extra_rows)
{
  switch ((cols - col) >> 2) {
  case 7:
    colLoopBodyExtraN<7, num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB, result, extra_rows);
    break;
  case 6:
    colLoopBodyExtraN<6, num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB, result, extra_rows);
    break;
  case 5:
    colLoopBodyExtraN<5, num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB, result, extra_rows);
    break;
  case 4:
    colLoopBodyExtraN<4, num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB, result, extra_rows);
    break;
  case 3:
    colLoopBodyExtraN<3, num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB, result, extra_rows);
    break;
  case 2:
    colLoopBodyExtraN<2, num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB, result, extra_rows);
    break;
  case 1:
    colLoopBodyExtraN<1, num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB, result, extra_rows);
    break;
  default:
    if (rhsExtraCols) {
      colLoopBody<1, num_packets, true, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB, result, extra_rows);
    }
    break;
  }
}

template<const Index num_packets, bool lhsExtraRows = false>
EIGEN_ALWAYS_INLINE void colLoops(Index depth, Index cols, Index rows, const Packet4f pAlpha, const bfloat16* indexA, const bfloat16* blockB, Index strideB, Index offsetB, float* result, Index extra_rows = 0)
{
  Index col = 0;
  if (cols >= (MAX_BFLOAT16_ACC * 4)) {
    colLoopBody<MAX_BFLOAT16_ACC, num_packets, false, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, 0, result, extra_rows);
    blockB += (strideB >> 2)*col;
    result += rows*col;
  }
  if (cols & 3) {
    colLoopBodyExtra<num_packets, true, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB, result, extra_rows);
  } else {
    colLoopBodyExtra<num_packets, false, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, 0, result, extra_rows);
  }
}

EIGEN_ALWAYS_INLINE Packet8bf convertF16toF32(const float *res)
{
  Packet16uc fp16_0 = __builtin_vsx_xvcvspbf16(reinterpret_cast<Packet16uc>(ploadu<Packet4f>(res + 0)));
  Packet16uc fp16_1 = __builtin_vsx_xvcvspbf16(reinterpret_cast<Packet16uc>(ploadu<Packet4f>(res + 4)));
  return vec_pack(reinterpret_cast<Packet4ui>(fp16_0), reinterpret_cast<Packet4ui>(fp16_1));
}

template<typename Index, typename Packet, typename RhsPacket, typename DataMapper, const Index accRows, const Index accCols>
void gemmMMAbfloat16(const DataMapper& res, const bfloat16* blockA, const bfloat16* blockB, Index rows, Index depth, Index cols, bfloat16 alpha, Index strideA, Index strideB, Index offsetA, Index offsetB)
{
  if(rows == 0 || cols == 0 || depth == 0) return;
  float falpha = Eigen::bfloat16_impl::bfloat16_to_float(alpha);
  if (falpha == float(0)) return;
  const Packet4f pAlpha = pset1<Packet4f>(falpha);
  ei_declare_aligned_stack_constructed_variable(float, result, cols*rows, 0);

  typedef typename DataMapper::LinearMapper LinearMapper;
  Packet8us z = pset1<Packet8us>(0);
  for(Index j = 0; j < cols; j++){
    const LinearMapper res2 = res.getLinearMapper(0, j);
    float *result2 = result + j*rows;
    Index i = 0;
    for(; i + 32 <= rows; i+=32){
      Packet8us r32_0 = res2.template loadPacket<Packet8bf>(i +  0).m_val;
      Packet8us r32_1 = res2.template loadPacket<Packet8bf>(i +  8).m_val;
      Packet8us r32_2 = res2.template loadPacket<Packet8bf>(i + 16).m_val;
      Packet8us r32_3 = res2.template loadPacket<Packet8bf>(i + 24).m_val;
      pstore(result2 + i +  0, reinterpret_cast<Packet4f>(vec_mergeh(z, r32_0)));
      pstore(result2 + i +  4, reinterpret_cast<Packet4f>(vec_mergel(z, r32_0)));
      pstore(result2 + i +  8, reinterpret_cast<Packet4f>(vec_mergeh(z, r32_1)));
      pstore(result2 + i + 12, reinterpret_cast<Packet4f>(vec_mergel(z, r32_1)));
      pstore(result2 + i + 16, reinterpret_cast<Packet4f>(vec_mergeh(z, r32_2)));
      pstore(result2 + i + 20, reinterpret_cast<Packet4f>(vec_mergel(z, r32_2)));
      pstore(result2 + i + 24, reinterpret_cast<Packet4f>(vec_mergeh(z, r32_3)));
      pstore(result2 + i + 28, reinterpret_cast<Packet4f>(vec_mergel(z, r32_3)));
    }
    for(; i + 16 <= rows; i+=16){
      Packet8us r32_0 = res2.template loadPacket<Packet8bf>(i +  0).m_val;
      Packet8us r32_1 = res2.template loadPacket<Packet8bf>(i +  8).m_val;
      pstore(result2 + i +  0, reinterpret_cast<Packet4f>(vec_mergeh(z, r32_0)));
      pstore(result2 + i +  4, reinterpret_cast<Packet4f>(vec_mergel(z, r32_0)));
      pstore(result2 + i +  8, reinterpret_cast<Packet4f>(vec_mergeh(z, r32_1)));
      pstore(result2 + i + 12, reinterpret_cast<Packet4f>(vec_mergel(z, r32_1)));
    }
    for(; i + 8 <= rows; i+=8){
      Packet8us r32_0 = res2.template loadPacket<Packet8bf>(i +  0).m_val;
      pstore(result2 + i +  0, reinterpret_cast<Packet4f>(vec_mergeh(z, r32_0)));
      pstore(result2 + i +  4, reinterpret_cast<Packet4f>(vec_mergel(z, r32_0)));
    }
    for(; i + 4 <= rows; i+=4){
      Packet8us r32_0 = res2.template loadPacketPartial<Packet8bf>(i +  0, 4).m_val;
      pstore(result2 + i +  0, reinterpret_cast<Packet4f>(vec_mergeh(z, r32_0)));
    }
    for(; i < rows; i++){
      result2[i] = res2(i);
    }
  }

  Index row = 0;
  Index col;

  if( strideA == -1 ) strideA = depth;
  if( strideB == -1 ) strideB = depth;
  //Packing is done in blocks.
  //There's 4 possible sizes of blocks
  //Blocks of 8 columns with 16 elements (8x16)
  //Blocks of 8 columns with 8 elements (8x8). This happens when there's 16 > rows >= 8
  //Blocks of 8 columns with 4 elements (8x4). This happens when there's 8 > rows >= 4
  //Blocks of 8 columns with < 4 elements. This happens when there's less than 4 remaining rows

  //Loop for LHS standard block (8x16)
  const Index standard_block_size = 16;
  const Index standard_blocks_quantity = rows/standard_block_size; //Number of standard blocks
  Index bigSuffix = (2*8) * (strideA-offsetA);
  const bfloat16* indexA = blockA;
  const bfloat16* indexB = blockB + 4*offsetB;
  Index block_index;
  strideB *= 4;
  offsetB *= 3;
  for(block_index = 0; block_index < standard_blocks_quantity; block_index++){
    indexA += 2*8*offsetA;
    colLoops<16>(depth, cols, rows, pAlpha, indexA, indexB, strideB, offsetB, result + row);
    row += 16;
    indexA += bigSuffix;
  }
  //LHS (8x8) block
  if(rows & 8){
    indexA += 1*8*offsetA;
    colLoops<8>(depth, cols, rows, pAlpha, indexA, indexB, strideB, offsetB, result + row);
    row += 8;
    indexA += (bigSuffix >> 1);
  }
  //LHS (8x4) block
  if(rows & 4){
    indexA += 1*4*offsetA;
    colLoops<4>(depth, cols, rows, pAlpha, indexA, indexB, strideB, offsetB, result + row);
    row += 4;
    indexA += (bigSuffix >> 2);
  }
  //extra rows
  Index extra_rows = rows & 3;
  if(extra_rows){
    //This index is the beginning of remaining block.
    colLoops<4, true>(depth, cols, rows, pAlpha, indexA, indexB, strideB, offsetB, result + row, extra_rows);
  }

  //Convert back to bfloat16
  for(col = 0; col + 4 <= cols; col += 4){
    const DataMapper res2 = res.getSubMapper(0, col);
    for(row = 0; row + 8 <= rows; row += 8){
      //get and save block
      PacketBlock<Packet8bf,4> block;
      for(Index j = 0; j < 4; j++){
        block.packet[j].m_val = convertF16toF32(result + (col + j)*rows + row);
      }

      res2.template storePacketBlock<Packet8bf,4>(row, 0, block);
    }
    //extra rows
    while(row < rows){
      for(Index col_off = 0; col_off < 4; col_off++){
        res2(row, col_off) = Eigen::bfloat16(result[(col+col_off)*rows+row]);
      }
      row++;
    }

  }
  //extra cols
  while(col < cols){
    const LinearMapper res2 = res.getLinearMapper(0, col);
    float *result2 = result + col*rows;
    Index r = 0;
    for(; r + 8 <= rows; r += 8){
      Packet8bf fp16 = convertF16toF32(result2 + r);
      res2.template storePacket<Packet8bf>(r, fp16);
    }
    for(; r< rows; r++){
      res2(r) = Eigen::bfloat16(result2[r]);
    }
    col++;
  }
}


}
}
#endif //EIGEN_MATRIX_PRODUCT_MMA_BFLOAT16_ALTIVEC_H
