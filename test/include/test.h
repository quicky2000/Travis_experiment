/* -*- C++ -*- */
/*    Copyright (C) 2023  Julien Thevenon ( julien_thevenon at yahoo.fr )

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

#ifndef MY_CUDA_TEST_H
#define MY_CUDA_TEST_H
#include "my_cuda.h"
#include "example_object.h"
#include "CUDA_memory_managed_array.h"
#include "CUDA_memory_managed_pointer.h"
#include "CUDA_info.h"
#include "CUDA_utils.h"
#ifdef ENABLE_CUDA_CODE
#include <nvfunctional>
#endif // ENABLE_CUDA_CODE
#include <memory>

#include "CUDA_print.h"

__device__
void apply_lambda(my_cuda::CUDA_memory_managed_array<uint32_t> & p_array
#ifdef ENABLE_CUDA_CODE
                 ,nvstd::function<uint32_t(uint32_t)> p_lamda
#else // ENABLE_CUDA_CODE
                 ,std::function<uint32_t(uint32_t)> p_lamda
                 ,const dim3 & threadIdx
#endif // ENABLE_CUDA_CODE
                 )
{
    p_array[threadIdx.x] = p_lamda(p_array[threadIdx.x]);
}

__global__
void kernel(example_object & p_object
           ,my_cuda::CUDA_memory_managed_ptr<example_object> & p_object_ptr
           ,my_cuda::CUDA_memory_managed_array<uint32_t> & p_array
           )
{
#ifndef ENABLE_CUDA_CODE
    dim3 threadIdx{0,0,0};
    for(threadIdx.y = 0; threadIdx.y < 4; ++threadIdx.y)
    {
        for(threadIdx.x = 0; threadIdx.x < 16; ++threadIdx.x)
        {
#endif // ENABLE_CUDA_CODE
    printf("Thread %i %i: %i %i -> %i\n", threadIdx.x, threadIdx.y, (width_t::base_type)p_object.get_width(), (height_t::base_type)p_object.get_height(), (area_t::base_type)p_object_ptr->compute_area());
#ifndef ENABLE_CUDA_CODE
        }
    }
#endif // ENABLE_CUDA_CODE

#ifndef ENABLE_CUDA_CODE
    for(threadIdx.y = 0; threadIdx.y < 4; ++threadIdx.y)
    {
        for(threadIdx.x = 0; threadIdx.x < 16; ++threadIdx.x)
        {
#endif // ENABLE_CUDA_CODE
    p_array[threadIdx.x] = threadIdx.x;
    auto l_lambda =[&](uint32_t p_value) -> u_int32_t
    {
        return p_value * ((threadIdx.x % 2 ) ? 10 : 1);
    };
#ifdef ENABLE_CUDA_CODE
    apply_lambda(p_array, l_lambda);
    my_cuda::print_mask(2, 0xFFFF, "Value = %" PRIu32 "\n", p_array[threadIdx.x]);
#else // ENABLE_CUDA_CODE
    apply_lambda(p_array, l_lambda, threadIdx);
    my_cuda::print_mask(2, 0xFFFF, threadIdx, "Value = %" PRIu32 "\n", p_array[threadIdx.x]);
#endif // ENABLE_CUDA_CODE
#ifndef ENABLE_CUDA_CODE
        }
    }
#endif // ENABLE_CUDA_CODE
}

__global__
void kernel_min(my_cuda::CUDA_memory_managed_array<uint32_t> & p_array_dest
               ,const my_cuda::CUDA_memory_managed_array<uint32_t> & p_array_src
               )
{
#ifdef ENABLE_CUDA_CODE
    my_cuda::print_all(2, "Value = %" PRIu32 "\n", p_array_src[threadIdx.x]);
#else // ENABLE_CUDA_CODE
    for(dim3 threadIdx{0, 0, 0}; threadIdx.x < 32; ++threadIdx.x)
    {
        my_cuda::print_all(2, threadIdx, "Value = %" PRIu32 "\n", p_array_src[threadIdx.x]);
    }
#endif // ENABLE_CUDA_CODE

#ifndef ENABLE_CUDA_CODE
    dim3 threadIdx{0,0,0};
    for(threadIdx.x = 0; threadIdx.x < 32; ++threadIdx.x)
    {
#endif // ENABLE_CUDA_CODE

#ifdef ENABLE_CUDA_CODE
    p_array_dest[threadIdx.x] = my_cuda::reduce_min_sync(p_array_src[threadIdx.x]);
    my_cuda::print_single(2, "Min = %" PRIu32 "\n", p_array_dest[threadIdx.x]);
#else // ENABLE_CUDA_CODE
    pseudo_CUDA_thread_variable<uint32_t> l_var{[&](dim3 threadIdx){return p_array_src[threadIdx.x];}};
    p_array_dest[threadIdx.x] = my_cuda::reduce_min_sync(l_var);
    my_cuda::print_single(2, threadIdx, "Min = %" PRIu32 "\n", p_array_dest[threadIdx.x]);
#endif // ENABLE_CUDA_CODE

#ifndef ENABLE_CUDA_CODE
     }
#endif // ENABLE_CUDA_CODE
}

__global__
void kernel_max(my_cuda::CUDA_memory_managed_array<uint32_t> & p_array_dest
               ,const my_cuda::CUDA_memory_managed_array<uint32_t> & p_array_src
               )
{
#ifndef ENABLE_CUDA_CODE
    dim3 threadIdx{0,0,0};
    for(threadIdx.x = 0; threadIdx.x < 32; ++threadIdx.x)
    {
#endif // ENABLE_CUDA_CODE

#ifdef ENABLE_CUDA_CODE
    p_array_dest[threadIdx.x] = my_cuda::reduce_max_sync(p_array_src[threadIdx.x]);
    my_cuda::print_single(2, "Max = %" PRIu32 "\n", p_array_dest[threadIdx.x]);
#else // ENABLE_CUDA_CODE
    pseudo_CUDA_thread_variable<uint32_t> l_var{[&](dim3 threadIdx){return p_array_src[threadIdx.x];}};
    p_array_dest[threadIdx.x] = my_cuda::reduce_max_sync(l_var);
    my_cuda::print_single(2, threadIdx, "Max = %" PRIu32 "\n", p_array_dest[threadIdx.x]);
#endif // ENABLE_CUDA_CODE

#ifndef ENABLE_CUDA_CODE
    }
#endif // ENABLE_CUDA_CODE
}

__global__
void kernel_add(my_cuda::CUDA_memory_managed_array<uint32_t> & p_array_dest
               ,const my_cuda::CUDA_memory_managed_array<uint32_t> & p_array_src
               )
{
#ifndef ENABLE_CUDA_CODE
    dim3 threadIdx{0,0,0};
    for(threadIdx.x = 0; threadIdx.x < 32; ++threadIdx.x)
    {
#endif // ENABLE_CUDA_CODE

#ifdef ENABLE_CUDA_CODE
    p_array_dest[threadIdx.x] = my_cuda::reduce_add_sync(p_array_src[threadIdx.x]);
    my_cuda::print_single(2, "Sum = %" PRIu32 "\n", p_array_dest[threadIdx.x]);
#else // ENABLE_CUDA_CODE
    pseudo_CUDA_thread_variable<uint32_t> l_var{[&](dim3 threadIdx){return p_array_src[threadIdx.x];}};
    p_array_dest[threadIdx.x] = my_cuda::reduce_add_sync(l_var);
    my_cuda::print_single(2, threadIdx, "Sum = %" PRIu32 "\n", p_array_dest[threadIdx.x]);
#endif // ENABLE_CUDA_CODE

#ifndef ENABLE_CUDA_CODE
    }
#endif // ENABLE_CUDA_CODE
}

void launcher()
{
    std::unique_ptr<example_object> l_object{new example_object((height_t)4, (width_t)10)};
    std::unique_ptr<my_cuda::CUDA_memory_managed_ptr<example_object>> l_object_ptr{new my_cuda::CUDA_memory_managed_ptr<example_object>(l_object.get())};
    std::unique_ptr<my_cuda::CUDA_memory_managed_array<uint32_t>> l_cuda_array{new my_cuda::CUDA_memory_managed_array<uint32_t>(32, 0)};

    // Reset CUDA error status
    cudaGetLastError();
    std::cout << "Launch kernels" << std::endl;
    dim3 dimBlock(16, 4);
    dim3 dimGrid( 1, 1);
#ifdef ENABLE_CUDA_CODE
    kernel<<<dimGrid, dimBlock>>>(*l_object, *l_object_ptr, *l_cuda_array);
#else // ENABLE_CUDA_CODE
    kernel(*l_object, *l_object_ptr, *l_cuda_array);
#endif // ENABLE_CUDA_CODE
    cudaDeviceSynchronize();
    gpuErrChk(cudaGetLastError());
    std::cout << "Object height : " << l_object->get_height() << std::endl;

    std::unique_ptr<my_cuda::CUDA_memory_managed_array<uint32_t>> l_cuda_array_result{new my_cuda::CUDA_memory_managed_array<uint32_t>(32, 0)};

    // Reset CUDA error status
    cudaGetLastError();
    std::cout << "Launch kernel_min" << std::endl;
    dim3 dim_block_warp(32, 1);
#ifdef ENABLE_CUDA_CODE
    kernel_min<<<dimGrid, dim_block_warp>>>(*l_cuda_array_result, *l_cuda_array);
#else // ENABLE_CUDA_CODE
    kernel_min(*l_cuda_array_result, *l_cuda_array);
#endif // ENABLE_CUDA_CODE
    cudaDeviceSynchronize();
    gpuErrChk(cudaGetLastError());

    // Reset CUDA error status
    cudaGetLastError();
    std::cout << "Launch kernel_max" << std::endl;
#ifdef ENABLE_CUDA_CODE
    kernel_max<<<dimGrid, dim_block_warp>>>(*l_cuda_array_result, *l_cuda_array);
#else // ENABLE_CUDA_CODE
    kernel_max(*l_cuda_array_result, *l_cuda_array);
#endif // ENABLE_CUDA_CODE
    cudaDeviceSynchronize();
    gpuErrChk(cudaGetLastError());

    // Reset CUDA error status
    cudaGetLastError();
    std::cout << "Launch kernel_add" << std::endl;
#ifdef ENABLE_CUDA_CODE
    kernel_add<<<dimGrid, dim_block_warp>>>(*l_cuda_array_result, *l_cuda_array);
#else // ENABLE_CUDA_CODE
    kernel_add(*l_cuda_array_result, *l_cuda_array);
#endif // ENABLE_CUDA_CODE
    cudaDeviceSynchronize();
    gpuErrChk(cudaGetLastError());
}

#endif //MY_CUDA_TEST_H
// EOF
