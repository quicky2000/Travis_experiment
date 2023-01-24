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

__global__
void kernel(example_object & p_object)
{
    printf("Thread %i : Integer value %i", threadIdx.x, p_object.get_integer());
    p_object.set_integer(p_object.get_integer() + 1);
}

void launch_kernel()
{

    std::unique_ptr<example_object> l_object{new example_object(10)};

    // Reset CUDA error status
    cudaGetLastError();
    std::cout << "Launch kernels" << std::endl;
    dim3 dimBlock(16, 4);
    dim3 dimGrid( 1, 1);
    kernel<<<dimGrid, dimBlock>>>(*l_object);
    cudaDeviceSynchronize();
    gpuErrChk(cudaGetLastError());
    std::cout << "Object integer : " << l_object->get_integer() << std::endl;
}


