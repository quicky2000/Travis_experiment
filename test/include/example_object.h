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


#ifndef MY_CUDA_EXAMPLE_OBJECT_H
#define MY_CUDA_EXAMPLE_OBJECT_H

#include "my_cuda.h"
#include "CUDA_memory_managed_item.h"

  class example_object: public my_cuda::CUDA_memory_managed_item
{
  public:

    inline
    __host__
    example_object(uint32_t p_value);

    inline
    __host__ __device__
    uint32_t
    get_integer() const;

    inline
    __host__ __device__
    void
    set_integer(uint32_t p_value);

  private:

    uint32_t m_integer;

};

__host__
example_object::example_object(uint32_t p_value)
:m_integer(p_value)
{
}

uint32_t
example_object::get_integer() const
{
    return m_integer;
}

void
example_object::set_integer(uint32_t p_value)
{
    m_integer = p_value;
}

#endif //MY_CUDA_EXAMPLE_OBJECT_H
// EOF