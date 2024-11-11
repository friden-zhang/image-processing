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

##### ARM on Raspberry Pi 5
```
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Benchmark                                                                                                                                Time             CPU   Iterations
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
BenchmarkRGB2Gray<image_processing::color_convert::AlgoType::kNativeCpu, image_processing::color_convert::MemLayout::Packed>      17848556 ns     17843701 ns           39
BenchmarkRGB2Gray<image_processing::color_convert::AlgoType::kParallelCpu, image_processing::color_convert::MemLayout::Packed>     9508353 ns      9497135 ns           67
BenchmarkRGB2Gray<image_processing::color_convert::AlgoType::kSimdCpu, image_processing::color_convert::MemLayout::Packed>        10534898 ns     10532252 ns           67
BenchmarkRGB2Gray<image_processing::color_convert::AlgoType::kNativeCpu, image_processing::color_convert::MemLayout::Planar>      17647402 ns     17645217 ns           40
BenchmarkRGB2Gray<image_processing::color_convert::AlgoType::kParallelCpu, image_processing::color_convert::MemLayout::Planar>     9833754 ns      9830831 ns           72
BenchmarkRGB2Gray<image_processing::color_convert::AlgoType::kSimdCpu, image_processing::color_convert::MemLayout::Planar>         3725357 ns      3725055 ns          186
BenchmarkRGBA2Gray<image_processing::color_convert::AlgoType::kNativeCpu, image_processing::color_convert::MemLayout::Packed>     14154560 ns     14153194 ns           45
BenchmarkRGBA2Gray<image_processing::color_convert::AlgoType::kParallelCpu, image_processing::color_convert::MemLayout::Packed>    7895368 ns      7893207 ns           87
BenchmarkRGBA2Gray<image_processing::color_convert::AlgoType::kSimdCpu, image_processing::color_convert::MemLayout::Packed>        3635769 ns      3634868 ns          196
BenchmarkRGBA2Gray<image_processing::color_convert::AlgoType::kNativeCpu, image_processing::color_convert::MemLayout::Planar>     13238522 ns     13236712 ns           53
BenchmarkRGBA2Gray<image_processing::color_convert::AlgoType::kParallelCpu, image_processing::color_convert::MemLayout::Planar>    7632800 ns      7623984 ns           93
BenchmarkRGBA2Gray<image_processing::color_convert::AlgoType::kSimdCpu, image_processing::color_convert::MemLayout::Planar>        3591226 ns      3590762 ns          196
```