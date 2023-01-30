/*
      This file is part of my_cuda
      Copyright (C) 2022  Julien Thevenon ( julien_thevenon at yahoo.fr )

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
#ifndef MY_CUDA_CUDA_UTILS_H
#define MY_CUDA_CUDA_UTILS_H

#include "my_cuda.h"
#ifndef ENABLE_CUDA_CODE
#include <numeric>
#endif // ENABLE_CUDA_CODE

namespace my_cuda
{
    /**
     * Compute sum of warp values and return the value
     * @param p_word
     * @return sum of p_words
     */
    inline static
#ifdef ENABLE_CUDA_CODE
    __device__
    uint32_t
    reduce_add_sync(uint32_t p_word)
    {
        unsigned l_mask = 0xFFFF;
        unsigned int l_width = 16;
        do
        {
            p_word += __shfl_down_sync(l_mask, p_word, l_width);
            l_width = l_width >> 1;
            l_mask = l_mask >> l_width;
        }
        while(l_width);
        return __shfl_sync(0xFFFFFFFFu, p_word, 0);
    }
#else // ENABLE_CUDA_CODE
    uint32_t
    reduce_add_sync(pseudo_CUDA_thread_variable<uint32_t> & p_word)
    {
        uint32_t l_total = std::accumulate(p_word.begin(), p_word.end(), 0);
        std::transform(p_word.begin(), p_word.end(), p_word.begin(), [=](uint32_t){return l_total;});
        return l_total;
    }
#endif // ENABLE_CUDA_CODE

    /**
     * Compute min of warp values and return the value
     * @param p_word
     * @return min of p_words
     */
    inline static
#ifdef ENABLE_CUDA_CODE
    __device__
    uint32_t
    reduce_min_sync(uint32_t p_word)
    {
        unsigned l_mask = 0xFFFF;
        unsigned int l_width = 16;
        do
        {
            uint32_t l_received_word = __shfl_down_sync(l_mask, p_word, l_width);
            p_word = l_received_word < p_word ? l_received_word : p_word;
            l_width = l_width >> 1;
            l_mask = l_mask >> l_width;
        }
        while(l_width);
        return __shfl_sync(0xFFFFFFFFu, p_word, 0);
    }
#else // ENABLE_CUDA_CODE
    uint32_t
    reduce_min_sync(pseudo_CUDA_thread_variable<uint32_t> & p_word)
    {
        uint32_t l_min = *std::min_element(p_word.begin(), p_word.end());
        std::transform(p_word.begin(), p_word.end(), p_word.begin(), [=](uint32_t){return l_min;});
        return l_min;
    }
#endif // ENABLE_CUDA_CODE

    /**
     * Compute max of warp values and return the value
     * @param p_word
     * @return max of p_words
     */
    inline static
#ifdef ENABLE_CUDA_CODE
    __device__
    uint32_t
    reduce_max_sync(uint32_t p_word)
    {
        unsigned l_mask = 0xFFFF;
        unsigned int l_width = 16;
        do
        {
            uint32_t l_received_word = __shfl_down_sync(l_mask, p_word, l_width);
            p_word = l_received_word > p_word ? l_received_word : p_word;
            l_width = l_width >> 1;
            l_mask = l_mask >> l_width;
        }
        while(l_width);
        return __shfl_sync(0xFFFFFFFFu, p_word, 0);
    }
#else // ENABLE_CUDA_CODE
    uint32_t
    reduce_max_sync(pseudo_CUDA_thread_variable<uint32_t> & p_word)
    {
        uint32_t l_max = *std::max_element(p_word.begin(), p_word.end());
        std::transform(p_word.begin(), p_word.end(), p_word.begin(), [=](uint32_t){return l_max;});
        return l_max;
    }
#endif // ENABLE_CUDA_CODE
}
#endif //MY_CUDA_CUDA_UTILS_H
// EOF