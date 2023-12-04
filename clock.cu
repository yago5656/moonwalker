#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <stdint.h>

#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>

   
#include "Vanity.h"
#include "Base58.h"
#include "Bech32.h"
#include "hash/sha256.h"
#include "hash/sha512.h"
#include "IntGroup.h"
#include "Wildcard.h"
#include "Timer.h"
#include "hash/ripemd160.h"
#include <algorithm>
#include <vector>
#include "SECP256K1.cpp"

#include "GPUGroup.h"
#include "GPUMath.h"
#include "GPUHash.h"
#include "GPUBase58.h"
#include "GPUWildcard.h"
#include "GPUCompute.h"
#include "GPUEngine.h"
  
#define BLOCKS 256
#define THREADS_PER_BLOCK 256
     


__device__ unsigned long long int totThr2 = 0;

__global__ void keyFinderKernel(uint8_t* gTableXCPU, uint8_t* gTableYCPU)
{

//we use atomicadd to verify how many threads are alive, this number is used to define the starting and end ranges for each thread   
atomicAdd(&totThr2, 1);

   //how many threads ?
    __int128_t index = blockIdx.x * blockDim.x + threadIdx.x;
    
	//initial definitions, change your search ranges here. here we are searching for #20
    __int128_t start = 0xD0000;
    __int128_t end =   0xDFFFF;
    __int128_t range =  end - start;
    __int128_t rangeend;
    __int128_t rangestart;

     //calculate the range for each thread	
     rangeend =  (((range / (totThr2 * 1)) * (index + 1)) + start);
     rangestart = (((range / (totThr2 * 1)) * index) + start);
	 
 
//some hashes to search, comment the actual line and uncomment the hash you need to search, don't forget to change start and end ranges above. in this case whe are searching for #20

//     uint8_t aa[20] = { 0x95, 0xa1, 0x56, 0xcd, 0x21, 0xb4, 0xa6, 0x9d, 0xe9, 0x69, 0xeb, 0x67, 0x16, 0x86, 0x4f, 0x4c, 0x8b, 0x82, 0xa8, 0x2a }; //address HASH160 40 bit 
//   uint8_t aa[20] = { 0x68, 0x13, 0x3e, 0x19, 0xb2, 0xdf, 0xb9, 0x03, 0x4e, 0xdf, 0x98, 0x30, 0xa2, 0x00, 0xcf, 0xdf, 0x38, 0xc9, 0x0c, 0xbd }; //address HASH160 61 bit
 //  uint8_t aa[20] = { 0x9a, 0x01, 0x22, 0x60, 0xd0, 0x1c, 0x51, 0x13, 0xdf, 0x66, 0xc8, 0xa8, 0x43, 0x8c, 0x9f, 0x7a, 0x1e, 0x3d, 0x5d, 0xac }; //address HASH160 46 bit
 //  uint8_t aa[20] = { 0x36, 0xaf, 0x65, 0x9e, 0xdb, 0xe9, 0x44, 0x53, 0xf6, 0x34, 0x4e, 0x92, 0x0d, 0x14, 0x3f, 0x17, 0x78, 0x65, 0x3a, 0xe7 }; //address HASH160 52 bit    
//   uint8_t aa[20] = { 0xf0, 0x22, 0x5b, 0xfc, 0x68, 0xa6, 0xe1, 0x7e, 0x87, 0xcd, 0x8b, 0x5e, 0x60, 0xae, 0x3b, 0xe1, 0x8f, 0x12, 0x07, 0x53 }; //address HASH160 45 bit    
//   uint8_t aa[20] = { 0xd1, 0x56, 0x2e, 0xb3, 0x73, 0x57, 0xf9, 0xe6, 0xfc, 0x41, 0xcb, 0x23, 0x59, 0xf4, 0xd3, 0xed, 0xa4, 0x03, 0x23, 0x29 }; //address HASH160 41 bit
//   uint8_t aa[20] = { 0xf6, 0xd6, 0x7d, 0x79, 0x83, 0xbf, 0x70, 0x45, 0x0f, 0x29, 0x5c, 0x9c, 0xb8, 0x28, 0xda, 0xab, 0x26, 0x5f, 0x1b, 0xfa }; //address HASH160 35 bit
//   uint8_t aa[20] = { 0xd3, 0x9c, 0x47, 0x04, 0x66, 0x4e, 0x1d, 0xeb, 0x76, 0xc9, 0x33, 0x1e, 0x63, 0x75, 0x64, 0xc2, 0x57, 0xd6, 0x8a, 0x08 }; //address HASH160 30 bit

    uint8_t aa[20] = { 0xb9, 0x07, 0xc3, 0xa2, 0xa3, 0xb2, 0x77, 0x89, 0xdf, 0xb5, 0x09, 0xb7, 0x30, 0xdd, 0x47, 0x70, 0x3c, 0x27, 0x28, 0x68 }; //address HASH160 20 bit
	 
//   uint8_t aa[20] = { 0x20, 0xd4, 0x5a, 0x6a, 0x76, 0x25, 0x35, 0x70, 0x0c, 0xe9, 0xe0, 0xb2, 0x16, 0xe3, 0x19, 0x94, 0x33, 0x5d, 0xb8, 0xa5 }; //address HASH160 66 bit
//   uint8_t aa[20] = { 0x73, 0x94, 0x37, 0xbb, 0x3d, 0xd6, 0xd1, 0x98, 0x3e, 0x66, 0x62, 0x9c, 0x5f, 0x08, 0xc7, 0x0e, 0x52, 0x76, 0x93, 0x71 }; //address HASH160 67 bit
//   uint8_t aa[20] = { 0xe0, 0xb8, 0xa2, 0xba, 0xee, 0x1b, 0x77, 0xfc, 0x70, 0x34, 0x55, 0xf3, 0x9d, 0x51, 0x47, 0x74, 0x51, 0xfc, 0x8c, 0xfc }; //address HASH160 68 bit 
//   uint8_t aa[20] = { 0x95, 0xa1, 0x56, 0xcd, 0x21, 0xb4, 0xa6, 0x9d, 0xe9, 0x69, 0xeb, 0x67, 0x16, 0x86, 0x4f, 0x4c, 0x8b, 0x82, 0xa8, 0x2a }; //address HASH160 40 bit 
//	 uint8_t aa[20] = { 0x52, 0xe7, 0x63, 0xa7, 0xdd, 0xc1, 0xaa, 0x4f, 0xa8, 0x11, 0x57, 0x8c, 0x49, 0x1c, 0x1b, 0xc7, 0xfd, 0x57, 0x01, 0x37 }; //address HASH160 65 bit 
	
	
	//we will take and compare only the last 8 bytes of hash160
	
    uint64_t hash160Last8Bytesa;
    uint64_t hash160Last8Bytesb;
    uint64_t hash160Last8Bytesb2;
    uint64_t hash160Last8Bytesb3;
 	
    GET_HASH_LAST_8_BYTES(hash160Last8Bytesa, aa);
	
	uint64_t x,y,x1,y1,x2,y2;
	
//the loop, we will test 3 variations of the key (x, x+1, x-1)
while (true) {
    __int128_t ii;
	
    for (ii = rangestart; ii < rangeend; ii++) {

  uint64_t  qx[4]= { 0x000000000000000, 0x000000000000000,0x000000000000000,0x000000000000000 };
  uint64_t  qy[4]= { 0x000000000000000, 0x000000000000000,0x000000000000000,0x000000000000000 };

  uint64_t   qx2[4]= { 0x000000000000000, 0x000000000000000,0x000000000000000,0x000000000000000 };
  uint64_t   qy2[4]= { 0x000000000000000, 0x000000000000000,0x000000000000000,0x000000000000000 };
 
  uint64_t  qx3[4]= { 0x000000000000000, 0x000000000000000,0x000000000000000,0x000000000000000 };
  uint64_t   qy3[4]= { 0x000000000000000, 0x000000000000000,0x000000000000000,0x000000000000000 };

//empty the actual hash160 bytes
    uint64_t hash160Last8Bytesb = 0;
    uint64_t hash160Last8Bytesb2 = 0;
    uint64_t hash160Last8Bytesb3 = 0;
	
//we take the 128 bit integer and split it in two 64 bit numbers to work with uint64 array
y = static_cast<uint64_t>(ii >> 64);
x = static_cast<uint64_t>(ii);

y1 = y;
x1 = x + 1;

y2 = y;
x2 = x - 1;

          //the priv keys
          uint64_t curi1[4] = { x, y, 0x000000000000000, 0x000000000000000 };
          uint64_t curi2[4] = { x1, y1, 0x000000000000000, 0x000000000000000 };
          uint64_t curi3[4] = { x2, y2, 0x000000000000000, 0x000000000000000 };

            //we take the array and turn into uint16 to work with point multiplication
            uint16_t* pv = (uint16_t*)(&curi1);
            uint16_t* pv1 = (uint16_t*)(&curi2);
            uint16_t* pv2 = (uint16_t*)(&curi3);
			
			//point multiplication, we take the integer and multiply by G 
            _PointMultiSecp256k1(qx, qy, pv, gTableXCPU, gTableYCPU);
            _PointMultiSecp256k1(qx2, qy2, pv1, gTableXCPU, gTableYCPU);
            _PointMultiSecp256k1(qx3, qy3, pv2, gTableXCPU, gTableYCPU);

			uint8_t hash160[SIZE_HASH160];
            uint8_t hash1602[SIZE_HASH160];
            uint8_t hash1603[SIZE_HASH160];
			
//is Y odd or even ?
int qy0x = 0;
if (qy[0] % 2) { qy0x = 1; };

int qy1x = 0;
if (qy2[0] % 2) { qy1x = 1; };

int qy2x = 0;
if (qy3[0] % 2) { qy2x = 1; };

            //we take the result and calculate hash160
            _GetHash160Comp(qx, (uint8_t)(qy0x), hash160);
            _GetHash160Comp(qx2, (uint8_t)(qy1x), hash1602);
            _GetHash160Comp(qx3, (uint8_t)(qy2x), hash1603);

            //last 8 bytes
            GET_HASH_LAST_8_BYTES(hash160Last8Bytesb, hash160);
            GET_HASH_LAST_8_BYTES(hash160Last8Bytesb2, hash1602);
            GET_HASH_LAST_8_BYTES(hash160Last8Bytesb3, hash1603);

            //and finally we compare with our hash160, if found the program stops
			
            if (hash160Last8Bytesb == hash160Last8Bytesa) {
                uint64_t xx;
                char foo[20];
                printf("FOUND PRIVKEY 0x%" PRIx64 " 0x%" PRIx64 " 0x % " PRIx64 " \n", (uint64_t)curi1[2], (uint64_t)curi1[1], (uint64_t)curi1[0]);
                asm("trap;");
            }
			
            if (hash160Last8Bytesb2 == hash160Last8Bytesa) {
                uint64_t xx;
                char foo[20];
                printf("FOUND PRIVKEY 0x%" PRIx64 " 0x%" PRIx64 " 0x % " PRIx64 " \n", (uint64_t)curi2[2], (uint64_t)curi2[1], (uint64_t)curi2[0]);
                asm("trap;");
            }

            if (hash160Last8Bytesb3 == hash160Last8Bytesa) {
                uint64_t xx;
                char foo[20];
                printf("FOUND PRIVKEY 0x%" PRIx64 " 0x%" PRIx64 " 0x % " PRIx64 " \n", (uint64_t)curi3[2], (uint64_t)curi3[1], (uint64_t)curi3[0]);
                asm("trap;");
            }


    }
}
            }

#define NUM_GTABLE_CHUNK 16    // Number of GTable chunks that are pre-computed and stored in global memory
#define NUM_GTABLE_VALUE 65536 // Number of GTable values per chunk (all possible states) (2 ^ NUM_GTABLE_CHUNK)
#define SIZE_GTABLE_POINT 32   // Each Point in GTable consists of two 32-byte coordinates (X and Y)
#define COUNT_GTABLE_POINTS (NUM_GTABLE_CHUNK * NUM_GTABLE_VALUE)

void loadGTable(uint8_t* gTableX, uint8_t* gTableY) {
    std::cout << "loadGTable started" << std::endl;
   
    Secp256K1 *secp = new Secp256K1();
    secp->Init2(); 
   
    for (int i = 0; i < NUM_GTABLE_CHUNK; i++)
    {
        for (int j = 0; j < NUM_GTABLE_VALUE - 1; j++)
        {
            int element = (i * NUM_GTABLE_VALUE) + j;
            Point p = secp->GTable2[element];
            for (int b = 0; b < 32; b++) {
                gTableX[(element * SIZE_GTABLE_POINT) + b] = p.x.GetByte64(b);
                gTableY[(element * SIZE_GTABLE_POINT) + b] = p.y.GetByte64(b);
            }
        }
    }

    std::cout << "loadGTable finished!" << std::endl;
}
int main()
{
    printf("MoonWalker YABF v0.2 beta\n");
  curandState *d_state;
  cudaMalloc(&d_state, sizeof(curandState));

    uint8_t* gTableXCPU = new uint8_t[COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT];
    uint8_t* gTableYCPU = new uint8_t[COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT];
    uint8_t* gTableXGPU;
    uint8_t* gTableYGPU;
    loadGTable(gTableXCPU, gTableYCPU);


   printf("Allocating gTableX \n");
   
    cudaMalloc((void**)&gTableXGPU, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT);
    cudaMemset(gTableXGPU, 0, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT);
    cudaMemcpy(gTableXGPU, gTableXCPU, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT, cudaMemcpyHostToDevice);
    printf("Allocating gTableY \n");
    cudaMalloc((void**)&gTableYGPU, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT);
    cudaMemset(gTableYGPU, 0, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT);
    cudaMemcpy(gTableYGPU, gTableYCPU, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT, cudaMemcpyHostToDevice);
	
     printf("Go ! \n");
	 
    keyFinderKernel << <BLOCKS, THREADS_PER_BLOCK >> > (gTableXGPU, gTableYGPU);
	
cudaError_t errSync  = cudaGetLastError();
cudaError_t errAsync = cudaDeviceSynchronize();
if (errSync != cudaSuccess) 
  printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
if (errAsync != cudaSuccess)
  printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

}



