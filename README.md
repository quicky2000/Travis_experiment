# my_cuda

Continuous integration with [Travis-Ci](https://app.travis-ci.com/github/quicky2000/my_cuda) : ![Build Status](https://app.travis-ci.com/github/quicky2000/my_cuda.svg?branch=master)

This object allow to compile very simple projects with and without CUDA for test purpose

License
-------
Please see [LICENSE](LICENSE) for info on the license.

Build
-----

Build process is the same used in [Travis file](.travis.yml)
Reference build can be found [here](https://travis-ci.com/quicky2000/my_cuda)

CUDA
-----
CUDA code is designed to run on a **Nvidia GPU Tesla T4**

### Build with CUDA code enabled

With CMake and CUDA 10, g++ > 9 is not supported so use the following command to force use of gcc 8

```
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=/usr/bin/gcc-8 -DCMAKE_CXX_COMPILER=/usr/bin/g++-8 $QUICKY_REPOSITORY/my_cuda/
```

With CMake and CUDA 11.1 nvcc is not in default path so specify location of nvcc with following command to force use of gcc 10

```
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.1/bin/nvcc $QUICKY_REPOSITORY/my_cuda/
```
