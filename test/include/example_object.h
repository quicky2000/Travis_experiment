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
#include "CUDA_strong_primitive.h"

using height_t = my_cuda::CUDA_strong_primitive<uint32_t, struct height>;
using width_t = my_cuda::CUDA_strong_primitive<uint32_t, struct width>;
using area_t = my_cuda::CUDA_strong_primitive<uint32_t, struct area>;

class example_object: public my_cuda::CUDA_memory_managed_item
{
  public:

    inline
    __host__
    example_object(height_t p_height
                  ,width_t p_width
                  );

    inline
    __host__ __device__
    void
    set(height_t p_value);

    inline
    __host__ __device__
    void
    set(width_t p_value);

    inline
    __host__ __device__
    height_t
    get_height() const;

    inline
    __host__ __device__
    width_t
    get_width() const;

    inline
    __host__ __device__
    area_t
    compute_area() const;

  private:

    height_t m_height;
    width_t m_with;

};

//-----------------------------------------------------------------------------
__host__
example_object::example_object(height_t p_height
                              ,width_t p_width
                              )
:m_height(p_height)
,m_with(p_width)
{
}

//-----------------------------------------------------------------------------
__host__ __device__
void
example_object::set(height_t p_value)
{
    m_height = p_value;
}

//-----------------------------------------------------------------------------
__host__ __device__
void
example_object::set(width_t p_value)
{
    m_with = p_value;
}

//-----------------------------------------------------------------------------
__host__ __device__
height_t
example_object::get_height() const
{
    return m_height;
}

//-----------------------------------------------------------------------------
__host__ __device__
width_t
example_object::get_width() const
{
    return m_with;
}

//-----------------------------------------------------------------------------
__host__ __device__
area_t
example_object::compute_area() const
{
    return (area_t )((width_t::base_type)m_with) * ((height_t::base_type)m_height);
}

#endif //MY_CUDA_EXAMPLE_OBJECT_H
// EOF