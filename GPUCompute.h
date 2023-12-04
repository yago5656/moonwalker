/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *da
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

// CUDA Kernel main function
// Compute SecpK1 keys and calculate RIPEMD160(SHA256(key)) then check prefix
// For the kernel, we use a 16 bits prefix lookup table which correspond to ~3 Base58 characters
// A second level lookup table contains 32 bits prefix (if used)
// (The CPU computes the full address and check the full prefix)
// 
// We use affine coordinates for elliptic curve point (ie Z=1)


__device__ __noinline__ void CheckPoint(uint32_t *_h, int32_t incr, int32_t endo, int32_t mode,prefix_t *prefix, 
                                        uint32_t *lookup32, uint32_t maxFound, uint32_t *out,int type) {

  uint32_t   off;
  prefixl_t  l32;
  prefix_t   pr0;
  prefix_t   hit;
  uint32_t   pos;
  uint32_t   st;
  uint32_t   ed;
  uint32_t   mi;
  uint32_t   lmi;
  uint32_t   tid = (blockIdx.x*blockDim.x) + threadIdx.x;
  char       add[48];
  
  if (prefix == NULL) {

    // No lookup compute address and return
    char *pattern = (char *)lookup32;
    _GetAddress(type, _h, add);
    if (_Match(add, pattern)) {
      // found
      goto addItem;
    }
 
  } else {
    
    // Lookup table
    pr0 = *(prefix_t *)(_h);
    hit = prefix[pr0];

    if (hit) {

      if (lookup32) {
        off = lookup32[pr0];
        l32 = _h[0];
        st = off;
        ed = off + hit - 1;
        while (st <= ed) {
          mi = (st + ed) / 2;
          lmi = lookup32[mi];
          if (l32 < lmi) {
            ed = mi - 1;
          } else if (l32 == lmi) {
            // found
            goto addItem;
          } else {
            st = mi + 1;
          }
        }
        return;
      }

    addItem:

      pos = atomicAdd(out, 1);
      if (pos < maxFound) {
        out[pos*ITEM_SIZE32 + 1] = tid;
        out[pos*ITEM_SIZE32 + 2] = (uint32_t)(incr << 16) | (uint32_t)(mode << 15) | (uint32_t)(endo);
        out[pos*ITEM_SIZE32 + 3] = _h[0];
        out[pos*ITEM_SIZE32 + 4] = _h[1];
        out[pos*ITEM_SIZE32 + 5] = _h[2];
        out[pos*ITEM_SIZE32 + 6] = _h[3];
        out[pos*ITEM_SIZE32 + 7] = _h[4];
      }

    }

  }

}

// -----------------------------------------------------------------------------------------

#define CHECK_POINT(_h,incr,endo,mode)  CheckPoint(_h,incr,endo,mode,prefix,lookup32,maxFound,out,P2PKH)
#define CHECK_POINT_P2SH(_h,incr,endo,mode)  CheckPoint(_h,incr,endo,mode,prefix,lookup32,maxFound,out,P2SH)

__device__ __noinline__ void CheckHashComp(prefix_t *prefix, uint64_t *px, uint8_t isOdd, int32_t incr, 
                                           uint32_t *lookup32, uint32_t maxFound, uint32_t *out) {

  uint32_t   h[5];
  //uint64_t   pe1x[4];
  //uint64_t   pe2x[4];

  _GetHash160Comp(px, isOdd, (uint8_t *)h);
  CHECK_POINT(h, incr, 0, true);
  /*
  _ModMult(pe1x, px, _beta);
  _GetHash160Comp(pe1x, isOdd, (uint8_t *)h);
  CHECK_POINT(h, incr, 1, true);
  _ModMult(pe2x, px, _beta2);
  _GetHash160Comp(pe2x, isOdd, (uint8_t *)h);
  CHECK_POINT(h, incr, 2, true);

  _GetHash160Comp(px, !isOdd, (uint8_t *)h);
  CHECK_POINT(h, -incr, 0, true);
  _GetHash160Comp(pe1x, !isOdd, (uint8_t *)h);
  CHECK_POINT(h, -incr, 1, true);
  _GetHash160Comp(pe2x, !isOdd, (uint8_t *)h);
  CHECK_POINT(h, -incr, 2, true);
  */

}

__device__ __noinline__ void CheckHashP2SHComp(prefix_t *prefix, uint64_t *px, uint8_t isOdd, int32_t incr,
  uint32_t *lookup32, uint32_t maxFound, uint32_t *out) {

  uint32_t   h[5];
  uint64_t   pe1x[4];
  uint64_t   pe2x[4];

  _GetHash160P2SHComp(px, isOdd, (uint8_t *)h);
  CHECK_POINT_P2SH(h, incr, 0, true);
  _ModMult(pe1x, px, _beta);
  _GetHash160P2SHComp(pe1x, isOdd, (uint8_t *)h);
  CHECK_POINT_P2SH(h, incr, 1, true);
  _ModMult(pe2x, px, _beta2);
  _GetHash160P2SHComp(pe2x, isOdd, (uint8_t *)h);
  CHECK_POINT_P2SH(h, incr, 2, true);

  _GetHash160P2SHComp(px, !isOdd, (uint8_t *)h);
  CHECK_POINT_P2SH(h, -incr, 0, true);
  _GetHash160P2SHComp(pe1x, !isOdd, (uint8_t *)h);
  CHECK_POINT_P2SH(h, -incr, 1, true);
  _GetHash160P2SHComp(pe2x, !isOdd, (uint8_t *)h);
  CHECK_POINT_P2SH(h, -incr, 2, true);

}

// -----------------------------------------------------------------------------------------

__device__ __noinline__ void CheckHashUncomp(prefix_t *prefix, uint64_t *px, uint64_t *py, int32_t incr, 
                                             uint32_t *lookup32, uint32_t maxFound, uint32_t *out) {

  uint32_t   h[5];
  //uint64_t   pe1x[4];
  //uint64_t   pe2x[4];
  //uint64_t   pyn[4];

  _GetHash160(px, py, (uint8_t *)h);
  CHECK_POINT(h, incr, 0, false);
  /*
  _ModMult(pe1x, px, _beta);
  _GetHash160(pe1x, py, (uint8_t *)h);
  CHECK_POINT(h, incr, 1, false);
  _ModMult(pe2x, px, _beta2);
  _GetHash160(pe2x, py, (uint8_t *)h);
  CHECK_POINT(h, incr, 2, false);

  ModNeg256(pyn,py);

  _GetHash160(px, pyn, (uint8_t *)h);
  CHECK_POINT(h, -incr, 0, false);
  _GetHash160(pe1x, pyn, (uint8_t *)h);
  CHECK_POINT(h, -incr, 1, false);
  _GetHash160(pe2x, pyn, (uint8_t *)h);
  CHECK_POINT(h, -incr, 2, false);
  */
}

__device__ __noinline__ void CheckHashP2SHUncomp(prefix_t *prefix, uint64_t *px, uint64_t *py, int32_t incr,
  uint32_t *lookup32, uint32_t maxFound, uint32_t *out) {

  uint32_t   h[5];
  uint64_t   pe1x[4];
  uint64_t   pe2x[4];
  uint64_t   pyn[4];

  _GetHash160P2SHUncomp(px, py, (uint8_t *)h);
  CHECK_POINT_P2SH(h, incr, 0, false);
  _ModMult(pe1x, px, _beta);
  _GetHash160P2SHUncomp(pe1x, py, (uint8_t *)h);
  CHECK_POINT_P2SH(h, incr, 1, false);
  _ModMult(pe2x, px, _beta2);
  _GetHash160P2SHUncomp(pe2x, py, (uint8_t *)h);
  CHECK_POINT_P2SH(h, incr, 2, false);

  ModNeg256(pyn, py);

  _GetHash160P2SHUncomp(px, pyn, (uint8_t *)h);
  CHECK_POINT_P2SH(h, -incr, 0, false);
  _GetHash160P2SHUncomp(pe1x, pyn, (uint8_t *)h);
  CHECK_POINT_P2SH(h, -incr, 1, false);
  _GetHash160P2SHUncomp(pe2x, pyn, (uint8_t *)h);
  CHECK_POINT_P2SH(h, -incr, 2, false);

}

// -----------------------------------------------------------------------------------------

__device__ __noinline__ void CheckHash(uint32_t mode, prefix_t *prefix, uint64_t *px, uint64_t *py, int32_t incr, 
                                       uint32_t *lookup32, uint32_t maxFound, uint32_t *out) {

  switch (mode) {
  case SEARCH_COMPRESSED:
    CheckHashComp(prefix, px, (uint8_t)(py[0] & 1), incr, lookup32, maxFound, out);
    break;
  case SEARCH_UNCOMPRESSED:
    CheckHashUncomp(prefix, px, py, incr, lookup32, maxFound, out);
    break;
  case SEARCH_BOTH:
    CheckHashComp(prefix, px, (uint8_t)(py[0] & 1), incr, lookup32, maxFound, out);
    CheckHashUncomp(prefix, px, py, incr, lookup32, maxFound, out);
    break;
  }

}

__device__ __noinline__ void CheckP2SHHash(uint32_t mode, prefix_t *prefix, uint64_t *px, uint64_t *py, int32_t incr,
  uint32_t *lookup32, uint32_t maxFound, uint32_t *out) {

  switch (mode) {
  case SEARCH_COMPRESSED:
    CheckHashP2SHComp(prefix, px, (uint8_t)(py[0] & 1), incr, lookup32, maxFound, out);
    break;
  case SEARCH_UNCOMPRESSED:
    CheckHashP2SHUncomp(prefix, px, py, incr, lookup32, maxFound, out);
    break;
  case SEARCH_BOTH:
    CheckHashP2SHComp(prefix, px, (uint8_t)(py[0] & 1), incr, lookup32, maxFound, out);
    CheckHashP2SHUncomp(prefix, px, py, incr, lookup32, maxFound, out);
    break;
  }

}

#define CHECK_PREFIX(incr) CheckHash(mode, sPrefix, px, py, j*GRP_SIZE + (incr), lookup32, maxFound, out)

// -----------------------------------------------------------------------------------------
//Point Mult
// //Secp256k1 Point Addition implementation
#define SubP(r) { \
  USUBO1(r[0], 0xFFFFFFFEFFFFFC2FULL); \
  USUBC1(r[1], 0xFFFFFFFFFFFFFFFFULL); \
  USUBC1(r[2], 0xFFFFFFFFFFFFFFFFULL); \
  USUBC1(r[3], 0xFFFFFFFFFFFFFFFFULL); \
  USUB1(r[4], 0ULL);}

__device__ void _ModAdd256(uint64_t* r, uint64_t* a, uint64_t* b)
{
    uint64_t rr[5];
    uint64_t rx[1];
    rx[0] = { 0 };

    UADDO(rr[0], a[0], b[0]);
    UADDC(rr[1], a[1], b[1]);
    UADDC(rr[2], a[2], b[2]);
    UADDC(rr[3], a[3], b[3]);
    UADD(rr[4], rx[0], rx[0]);

    Load256(r, rr);

    SubP(rr);

    if (_IsPositive(rr)) {
        Load256(r, rr);
    }
}


__device__ void _ModSub256(uint64_t* r, uint64_t* a, uint64_t* b)
{
    uint64_t t;
    uint64_t T[4];

    USUBO(r[0], a[0], b[0]);
    USUBC(r[1], a[1], b[1]);
    USUBC(r[2], a[2], b[2]);
    USUBC(r[3], a[3], b[3]);
    USUB(t, 0ULL, 0ULL);

    T[0] = 0xFFFFFFFEFFFFFC2FULL & t;
    T[1] = 0xFFFFFFFFFFFFFFFFULL & t;
    T[2] = 0xFFFFFFFFFFFFFFFFULL & t;
    T[3] = 0xFFFFFFFFFFFFFFFFULL & t;

    UADDO1(r[0], T[0]);
    UADDC1(r[1], T[1]);
    UADDC1(r[2], T[2]);
    UADD1(r[3], T[3]);

}

// ---------------------------------------------------------------------------------------

__device__ void _ModSub256(uint64_t* r, uint64_t* b)
{

    uint64_t t;
    uint64_t T[4];
    USUBO(r[0], r[0], b[0]);
    USUBC(r[1], r[1], b[1]);
    USUBC(r[2], r[2], b[2]);
    USUBC(r[3], r[3], b[3]);
    USUB(t, 0ULL, 0ULL);
    T[0] = 0xFFFFFFFEFFFFFC2FULL & t;
    T[1] = 0xFFFFFFFFFFFFFFFFULL & t;
    T[2] = 0xFFFFFFFFFFFFFFFFULL & t;
    T[3] = 0xFFFFFFFFFFFFFFFFULL & t;
    UADDO1(r[0], T[0]);
    UADDC1(r[1], T[1]);
    UADDC1(r[2], T[2]);
    UADD1(r[3], T[3]);

}
__device__ void _PointAddSecp256k1(uint64_t* p1x, uint64_t* p1y, uint64_t* p1z, uint64_t* p2x, uint64_t* p2y)
{
    uint64_t u[4];
    uint64_t v[4];

    uint64_t us2[4];
    uint64_t vs2[4];
    uint64_t vs3[4];

    uint64_t a[4];

    uint64_t us2w[4];
    uint64_t vs2v2[4];
    uint64_t vs3u2[4];
    uint64_t _2vs2v2[4];

    _ModMult(u, p2y, p1z);
    _ModMult(v, p2x, p1z);

    _ModSub256(u, u, p1y);
    _ModSub256(v, v, p1x);

    _ModSqr(us2, u);
    _ModSqr(vs2, v);

    _ModMult(vs3, vs2, v);
    _ModMult(us2w, us2, p1z);
    _ModMult(vs2v2, vs2, p1x);

    _ModAdd256(_2vs2v2, vs2v2, vs2v2);

    _ModSub256(a, us2w, vs3);
    _ModSub256(a, _2vs2v2);

    _ModMult(p1x, v, a);
    _ModMult(vs3u2, vs3, p1y);

    _ModSub256(p1y, vs2v2, a);
    _ModMult(p1y, p1y, u);

    _ModSub256(p1y, vs3u2);
    _ModMult(p1z, vs3, p1z);
}
//Cuda Secp256k1 Point Multiplication
//Takes 32-byte privKey + gTable and outputs 64-byte public key [qx,qy]
#define NUM_GTABLE_CHUNK 16    //number of GTable chunks that are pre-computed and stored in memory
#define NUM_GTABLE_VALUE 65536 //number of GTable values per chunk (all possible states) (2 ^ (bits_per_chunk))
__constant__ int CHUNK_FIRST_ELEMENT[NUM_GTABLE_CHUNK] = {
  65536 * 0,  65536 * 1,  65536 * 2,  65536 * 3,
  65536 * 4,  65536 * 5,  65536 * 6,  65536 * 7, 
  65536 * 8,  65536 * 9,  65536 * 10,  65536 * 11,
  65536 * 12,  65536 * 13, 65536 * 14, 65536 * 15
};
#define SIZE_LONG 8            // Each Long is 8 bytes
#define SIZE_HASH160 20        // Each Hash160 is 20 bytes
#define SIZE_PRIV_KEY 32 	   // Length of the private key that is generated from input seed (in bytes)
#define NUM_GTABLE_CHUNK 16    // Number of GTable chunks that are pre-computed and stored in global memory
#define NUM_GTABLE_VALUE 65536 // Number of GTable values per chunk (all possible states) (2 ^ NUM_GTABLE_CHUNK)
#define SIZE_GTABLE_POINT 32   // Each Point in GTable consists of two 32-byte coordinates (X and Y)
#define IDX_CUDA_THREAD ((blockIdx.x * blockDim.x) + threadIdx.x)
#define COUNT_GTABLE_POINTS (NUM_GTABLE_CHUNK * NUM_GTABLE_VALUE)
#define COUNT_CUDA_THREADS (BLOCKS_PER_GRID * THREADS_PER_BLOCK)

__device__ void _PointMultiSecp256k1(uint64_t* qxm, uint64_t* qym, uint16_t* privKey, uint8_t* gTableX, uint8_t* gTableY) {

    int chunk = 0;
    uint64_t qz[5] = { 1, 0, 0, 0, 0 };
	
    uint64_t gx1[4];
    uint64_t gy1[4];
	
    uint64_t *qxmm[4];
    uint64_t *qymm[4];
	
	gx1[0] = { qxm[0] };
    gy1[0] = { qym[0] };
	
	gx1[1] = { qxm[1] };
    gy1[1] = { qym[1] };
	
	gx1[2] = { qxm[2] };
    gy1[2] = { qym[2] };
	
	gx1[3] = { qxm[3] };
    gy1[3] = { qym[3] };
	
	
			
    //Find the first non-zero point [qx,qy]
    for (; chunk < NUM_GTABLE_CHUNK; chunk++) {
        if (privKey[chunk] > 0) {
			
			int index2 = (CHUNK_FIRST_ELEMENT[chunk] + (privKey[chunk] - 1)) * SIZE_GTABLE_POINT;
			
            memcpy(qxm, gTableX + index2, SIZE_GTABLE_POINT);
            memcpy(qym, gTableY + index2, SIZE_GTABLE_POINT);
            //cudaMemcpy(qxm, gTableX + index2, SIZE_GTABLE_POINT, cudaMemcpyDeviceToHost);
            //cudaMemcpy(qym, gTableY + index2, SIZE_GTABLE_POINT, cudaMemcpyDeviceToHost);
            chunk++;
            break;
        }
    }
			
    //Add the remaining chunks together
    for (; chunk < NUM_GTABLE_CHUNK; chunk++) {
        if (privKey[chunk] > 0) {
            uint64_t gx[4];
            uint64_t gy[4];

            int index3 = (CHUNK_FIRST_ELEMENT[chunk] + (privKey[chunk] - 1)) * SIZE_GTABLE_POINT;

            memcpy(gx, gTableX + index3, SIZE_GTABLE_POINT);
            memcpy(gy, gTableY + index3, SIZE_GTABLE_POINT);

            _PointAddSecp256k1(qxm, qym, qz, gx, gy);
        }
    }

    //Performing modular inverse on qz to obtain the public key [qx,qy]
    _ModInv(qz);
    _ModMult(qxm, qz);
    _ModMult(qym, qz);
}

#define GET_HASH_LAST_8_BYTES(l,h) { l = \
	  static_cast<uint64_t>(h[19]) | \
      static_cast<uint64_t>(h[18]) << 8 | \
      static_cast<uint64_t>(h[17]) << 16 | \
      static_cast<uint64_t>(h[16]) << 24 | \
      static_cast<uint64_t>(h[15]) << 32 | \
      static_cast<uint64_t>(h[14]) << 40 | \
      static_cast<uint64_t>(h[13]) << 48 | \
      static_cast<uint64_t>(h[12]) << 56; }

__device__ unsigned long long totThr = 0;

__device__ void printConvergentThreadCount(int line); // Pass __LINE__


__device__ void ComputeKeys(uint8_t* gTableXCPU, uint8_t* gTableYCPU) {

}

// -----------------------------------------------------------------------------------------

#define CHECK_PREFIX_P2SH(incr) CheckP2SHHash(mode, sPrefix, px, py, j*GRP_SIZE + (incr), lookup32, maxFound, out)

__device__ void ComputeKeysP2SH(uint32_t mode, uint64_t *startx, uint64_t *starty,
  prefix_t *sPrefix, uint32_t *lookup32, uint32_t maxFound, uint32_t *out) {

  }

// -----------------------------------------------------------------------------------------
// Optimized kernel for compressed P2PKH address only

__device__ void ComputeKeysComp(uint64_t *startx, uint64_t *starty, prefix_t *sPrefix, uint32_t *lookup32, uint32_t maxFound, uint32_t *out) {

}

