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
#include "my_cuda.h"
#include "example_object.h"
#include "CUDA_memory_managed_array.h"
#include "CUDA_memory_managed_pointer.h"
#include "CUDA_info.h"

__global__
void kernel(example_object & p_object
           ,my_cuda::CUDA_memory_managed_ptr<example_object> & p_object_ptr
           ,my_cuda::CUDA_memory_managed_array<uint32_t> & p_array
           )
{
    printf("Thread %i %i: %i %i -> %i\n", threadIdx.x, threadIdx.y, (width_t::base_type)p_object.get_width(), (height_t::base_type)p_object.get_height(), (area_t::base_type)p_object_ptr->compute_area());
    p_array[threadIdx.x] = threadIdx.x;
}

void launch_kernel()
{
    int l_nb_cuda_device = my_cuda::CUDA_info();

    if(!l_nb_cuda_device)
    {
        return;
    }

    std::unique_ptr<example_object> l_object{new example_object((height_t)4, (width_t)10)};
    std::unique_ptr<my_cuda::CUDA_memory_managed_ptr<example_object>> l_object_ptr{new my_cuda::CUDA_memory_managed_ptr<example_object>(l_object.get())};
    std::unique_ptr<my_cuda::CUDA_memory_managed_array<uint32_t>> l_cuda_array{new my_cuda::CUDA_memory_managed_array<uint32_t>(32, 0)};

    // Reset CUDA error status
    cudaGetLastError();
    std::cout << "Launch kernels" << std::endl;
    dim3 dimBlock(16, 4);
    dim3 dimGrid( 1, 1);
    kernel<<<dimGrid, dimBlock>>>(*l_object, *l_object_ptr, *l_cuda_array);
    cudaDeviceSynchronize();
    gpuErrChk(cudaGetLastError());
    std::cout << "Object height : " << l_object->get_height() << std::endl;
    for(unsigned int l_index = 0; l_index < 32; ++l_index)
    {
        std::cout << "Array[" << l_index << "] = " << (*l_cuda_array)[l_index] << std::endl;
    }
}



