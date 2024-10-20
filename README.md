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