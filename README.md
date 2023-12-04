# Moonwalker - Yet Another Brute Forcer

## Description

This is my attempt to solve bitcoin puzzle

## Key Concepts

I would like to write a compreensive script to beginners work with cuda

## Supported SM Architectures

[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Only Linux !

## Supported CPU Architecture

x86_64, ppc64le, armv7l

## Prerequisites

Download and install the [CUDA Toolkit 12.2](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.

## Build and Run

### Linux
The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:
```
$ cd <sample_dir>
$ make
```

### Run
```
$ ./clock
```
### Result (In this case looking for #20 puzzle)
```
MoonWalker YABF v0.2 beta
loadGTable started
loadGTable finished!
Allocating gTableX
Allocating gTableY
Go !
FOUND PRIVKEY 0x0 0x0 0x d2c55
Async kernel error: unspecified launch failure
```
That error message it's just my dirty way to break the execution

### TODO
This script actually is slow !
we hope community help me to improve speed and fix some probable points in order to get more speed here !

