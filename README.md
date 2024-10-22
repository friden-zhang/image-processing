# Image Processing Library

### Color Convert
A high performance color conversion library, which will support most of the common color spaces. Every color conversion will provide four algorithms:
1. Native Cpu Algorithm
2. Parallel Algorithm of CPU(We will use TBB for that)
3. SIMD Algorithm
4. GPU Algorithm(We will use CUDA for that)

### Memory Management
Because it support CPU/GPU algorithms, We will manage the memory allocation of malloc/cudaMalloc


### Pipeline
We will use stdexec to build the pipeline, stdexec is a c++20 library that provides a set of tools for building parallel and distributed applications. 
It provides a set of abstractions and patterns for building concurrent and parallel applications.


### Performance

#### RGB to Grayscale benchmark
```
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Benchmark                                                                                                                               Time             CPU   Iterations
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
BenchmarkRGB2Gray<image_processing::color_convert::AlgoType::kNativeCpu, image_processing::color_convert::MemLayout::Packed>    106341944 ns    106284844 ns            5
BenchmarkRGB2Gray<image_processing::color_convert::AlgoType::kParallelCpu, image_processing::color_convert::MemLayout::Packed>   26486531 ns     26170631 ns           27
BenchmarkRGB2Gray<image_processing::color_convert::AlgoType::kSimdCpu, image_processing::color_convert::MemLayout::Packed>       82201786 ns     82166220 ns            9
BenchmarkRGB2Gray<image_processing::color_convert::AlgoType::kNativeCpu, image_processing::color_convert::MemLayout::Planar>    119646622 ns    119605222 ns            6
BenchmarkRGB2Gray<image_processing::color_convert::AlgoType::kParallelCpu, image_processing::color_convert::MemLayout::Planar>   37736761 ns     29439502 ns           22
BenchmarkRGB2Gray<image_processing::color_convert::AlgoType::kSimdCpu, image_processing::color_convert::MemLayout::Planar>       38048058 ns     37969202 ns           18
```