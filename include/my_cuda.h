/* -*- C++ -*- */
/*    This file provide some convenience code to make code compilable
      without CUDA in particular cases
      Copyright (C) 2016  Julien Thevenon ( julien_thevenon at yahoo.fr )

      This program is free software: you can redistribute it and/or modify
      it under the terms of the GNU General Public License as published by
      the Free Software Foundation, either version 3 of the License, or
      (at your option) any later version.

      This program is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
      GNU General Public License for more details.

      You should have received a copy of the GNU General Public License
      along with this program.  If not, see <http://www.gnu.org/licenses/>
*/

#ifndef _MY_CUDA_H_
#define _MY_CUDA_H_

#include <cinttypes>
#include <type_traits>
#include <cstdlib>
#include <iostream>
#include <cstring>

#ifndef __NVCC__
#include <array>
#include <functional>
#include <cassert>

class dim3
{
  public:
  inline dim3(uint32_t p_x = 1, uint32_t p_y = 1, uint32_t p_z = 1):
    x(p_x),
    y(p_y),
    z(p_z)
  {
  }

   uint32_t x;
   uint32_t y;
   uint32_t z;
};

#define CUDA_KERNEL(name,...) void name(const dim3 & threadIdx, const dim3 & blockIdx, const dim3 & blockDim, const dim3 & gridDim,__VA_ARGS__)
#define CUDA_METHOD_HD_I inline
#define CUDA_METHOD_D_I inline
#define gpuErrChk(ans) ans

#define cudaFuncCachePreferL1 1
inline void cudaDeviceSetCacheConfig(unsigned int)
{
}

#define cudaFree free
#define cudaMalloc(ptr,size) (*ptr) = (std::remove_pointer<decltype(ptr)>::type)malloc(size);
#define cudaMemcpy(dest, src , size, direction) memcpy(dest, src, size);
#define cudaGetLastError() {}
#define cudaDeviceSynchronize() {}

#define launch_kernels(name,grid,block,...) { dim3 l_blockIdx(0,0,0);                                          \
  for(l_blockIdx.z = 0 ; l_blockIdx.z < grid.z ; ++l_blockIdx.z)                                               \
    {                                                                                                          \
      for(l_blockIdx.y = 0 ; l_blockIdx.y < grid.y ; ++l_blockIdx.y)                                           \
	{                                                                                                      \
	  for(l_blockIdx.x = 0 ; l_blockIdx.x < grid.x ; ++l_blockIdx.x)                                       \
	    {                                                                                                  \
	      dim3 l_threadIdx(0,0,0);                                                                         \
	      for(l_threadIdx.z = 0 ; l_threadIdx.z < block.z ; ++l_threadIdx.z)                               \
		{                                                                                              \
		  for(l_threadIdx.y = 0 ; l_threadIdx.y < block.y ; ++l_threadIdx.y)                           \
		    {                                                                                          \
		      for(l_threadIdx.x = 0 ; l_threadIdx.x < block.x ; ++l_threadIdx.x)                       \
			{                                                                                      \
			  name(l_threadIdx, l_blockIdx, block, grid,__VA_ARGS__);                              \
			}                                                                                      \
		    }                                                                                          \
		}                                                                                              \
	    }                                                                                                  \
	}                                                                                                      \
    }                                                                                                          \
}

inline
int32_t __ffs(int32_t p_value)
{
    return ::ffs(p_value);
}

/**
 * Variable to use when a variable declared in a CUDA thread will take
 * different values depending on the thread in which it is
 * @tparam T
 */
template <typename T>
class pseudo_CUDA_thread_variable
{
  public:
    pseudo_CUDA_thread_variable(T p_value)
    {
        dim3 threadIdx{0, 1, 1};
        for (threadIdx.x = 0; threadIdx.x < 32; ++threadIdx.x)
        {
            m_value[threadIdx.x] = p_value;
        }
    }

    pseudo_CUDA_thread_variable(std::function<T(const dim3 &)> p_init_func)
    {
        dim3 threadIdx{0, 1, 1};
        for (threadIdx.x = 0; threadIdx.x < 32; ++threadIdx.x)
        {
            m_value[threadIdx.x] = p_init_func(threadIdx);
        }
    }

    auto begin()
    {
        return m_value.begin();
    }

    auto end()
    {
        return m_value.end();
    }

    T operator[](dim3 p_dim) const
    {
        assert(p_dim.x < 32);
        return m_value[p_dim.x];
    }

    T & operator[](dim3 p_dim)
    {
        assert(p_dim.x < 32);
        return m_value[p_dim.x];
    }

    pseudo_CUDA_thread_variable & operator=(std::function<T(const dim3 &)> p_init_func)
    {
        dim3 threadIdx{0, 1, 1};
        for (threadIdx.x = 0; threadIdx.x < 32; ++threadIdx.x)
        {
            m_value[threadIdx.x] = p_init_func(threadIdx);
        }
        return *this;
    }

    pseudo_CUDA_thread_variable & operator&=(const pseudo_CUDA_thread_variable & p_value)
    {
        dim3 threadIdx{0, 1, 1};
        for (threadIdx.x = 0; threadIdx.x < 32; ++threadIdx.x)
        {
            m_value[threadIdx.x] &= p_value[threadIdx];
        }
        return *this;
    }

    pseudo_CUDA_thread_variable operator&(const pseudo_CUDA_thread_variable & p_value)
    {
        return pseudo_CUDA_thread_variable([&](dim3 threadIdx){return m_value[threadIdx.x] & p_value[threadIdx.x];});
    }

    pseudo_CUDA_thread_variable & operator>>(int p_shift)
    {
        dim3 threadIdx{0, 1, 1};
        for (threadIdx.x = 0; threadIdx.x < 32; ++threadIdx.x)
        {
            m_value[threadIdx.x] = m_value[threadIdx] >> p_shift;
        }
        return *this;
    }

  private:
    std::array<T,32> m_value;
};

inline
uint32_t
__ballot_sync(uint32_t p_mask, std::function<uint32_t(dim3)> p_condition)
{
    uint32_t l_bit = 1;
    uint32_t l_result = 0;
    for(dim3 threadIdx{0, 1, 1}; p_mask && (threadIdx.x < 32); ++threadIdx.x)
    {
        if(l_bit & p_mask)
        {
            p_mask &= ~l_bit;
            l_result |= (p_condition(threadIdx.x) != 0) << threadIdx.x;
        }
        l_bit = l_bit << 1u;
    }
    return l_result;
}

template <typename T>
uint32_t
__ballot_sync(uint32_t p_mask, const pseudo_CUDA_thread_variable<T> & p_condition)
{
    uint32_t l_bit = 1;
    uint32_t l_result = 0;
    for(unsigned int l_threadIdx_x = 0; p_mask && (l_threadIdx_x < 32); ++l_threadIdx_x)
    {
        if(l_bit & p_mask)
        {
            p_mask &= ~l_bit;
            l_result |= (p_condition[l_threadIdx_x] != 0) << l_threadIdx_x;
        }
        l_bit = l_bit << 1u;
    }
    return l_result;
}

template <typename T>
uint32_t
__all_sync(uint32_t p_mask, const pseudo_CUDA_thread_variable<T> & p_condition)
{
    bool l_all = true;
    uint32_t l_bit = 1;
    for (unsigned int l_threadIdx_x = 0; l_all && l_threadIdx_x < 32;++l_threadIdx_x)
    {
        if(l_bit & p_mask)
        {
            p_mask &= ~l_bit;
            l_all = static_cast<uint32_t>(p_condition[l_threadIdx_x]);
        }
        l_bit = l_bit << 1u;
    }
    return l_all;
}

template <typename T>
uint32_t
__any_sync(uint32_t p_mask, const pseudo_CUDA_thread_variable<T> & p_condition)
{
    uint32_t l_bit = 1;
    for(unsigned int l_threadIdx_x = 0; l_threadIdx_x < 32; ++l_threadIdx_x)
    {
        if((l_bit & p_mask) && p_condition[l_threadIdx_x])
        {
            return true;
        }
        l_bit = l_bit << 1u;
    }
    return false;
}

#define __global__
#define __host__
#define __device__
#define __constant__
#else // __NVCC__
#define CUDA_KERNEL(name,...) __global__ void name(__VA_ARGS__)
#define CUDA_METHOD_HD_I __device__ __host__ inline
#define CUDA_METHOD_D_I __device__ inline

#define launch_kernels(name,grid,block,...) { name<<<grid,block>>>(__VA_ARGS__);}

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t p_code, const char * p_file, int p_line, bool p_abort = true)
{
   if (p_code != cudaSuccess) 
   {
      std::cerr << "GPUassert: " << cudaGetErrorString(p_code) << " @ " << p_file << ":" << p_line << std::endl ;
      if (p_abort)
	{
	  exit(p_code);
	}
   }
}
#endif // __NVCC__

#endif // _MY_CUDA_H_
// EOF
