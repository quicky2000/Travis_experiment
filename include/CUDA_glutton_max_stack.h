/*
      This file is part of edge_matching_puzzle
      Copyright (C) 2021  Julien Thevenon ( julien_thevenon at yahoo.fr )

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

#ifndef EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_MAX_STACK_H
#define EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_MAX_STACK_H

#include "CUDA_memory_managed_item.h"
#include "CUDA_memory_managed_array.h"
#include "CUDA_piece_position_info2.h"
#include <cinttypes>

namespace edge_matching_puzzle
{
    /**
     * Stack of glutton max algorithm. Each level of stack corresponds to a
     * puzzle level. Level 0 is the beginning of the puzzle when no piece has
     * been set.
     * For each level we store the available transitions for each free position
     * As order of locations where pieces are set is not defined relation
     * between position and index is not constant through level and must be
     * computed
     */
    class CUDA_glutton_max_stack: public CUDA_memory_managed_item
    {

      public:

        inline explicit
        CUDA_glutton_max_stack(uint32_t p_size
                              ,uint32_t p_nb_pieces
                              );

        inline
        ~CUDA_glutton_max_stack();

        inline
        __device__ __host__
        const CUDA_piece_position_info2 &
        get_position_info(uint32_t p_info_index) const;

        inline
        __device__ __host__
        const CUDA_piece_position_info2 &
        get_best_candidate_info(uint32_t p_info_index) const;

        inline
        __device__ __host__
        CUDA_piece_position_info2 &
        get_best_candidate_info(uint32_t p_info_index);

        inline
        void set_position_info(uint32_t p_info_index
                              ,const CUDA_piece_position_info2 & p_info
                              );

        inline
        __device__ __host__
        void push();

        inline
        __device__ __host__
        void pop();

        inline
        __device__ __host__
        uint32_t get_size() const;

        inline
        __device__ __host__
        uint32_t get_level() const;

        /**
         * Number of positions available at current level
         * @return number of positions
         */
        inline
        __device__ __host__
        uint32_t get_nb_position() const;

        /**
         * Store correspondant between index of position info and position index
         * @param p_index Index of position info
         * @param p_position_index Position index
         */
        inline
        void set_position_index(uint32_t p_index
                               ,uint32_t p_position_index
                               );

        /**
         * Indicate at which index information related to position index is stored
         * @param p_position_index
         * @return index in info array here information is stored
         */
        inline
        __host__ __device__
        uint32_t get_index_of_position(uint32_t p_position_index) const;

        /**
         * Indicate which position corresponds to info stored at index
         * @param p_index index in info array
         * @return Position index whose info is stored
         */
        inline
        __host__ __device__
        uint32_t get_position_of_index(uint32_t p_index) const;

        /**
         * Indicate if a position is associated to this index.
         * Dependant on current stack level
         * @param p_index index in array
         * @return true if a position is associated
         */
        inline
        __device__ __host__
        bool is_position_index_used(unsigned int p_index) const;

        /**
         * Indicate if position is indexed. Once occupied a position is no more
         * indexed so indexed == free
         * @param p_position_index position index
         * @return true if position is not set
         */
        inline
        __device__ __host__
        bool is_position_free(unsigned int p_position_index) const;

        /**
         * Indicate if piece designed by piece index is used or not
         * @param p_piece_index
         * @return true if piece not used, false if used
         */
        inline
        __host__ __device__
        bool is_piece_available(unsigned int p_piece_index)const;

        /**
         * Indicate that piece designed by piece index is not used
         * @param p_piece_index
         */
        inline
        __host__ __device__
        void set_piece_available(unsigned int p_piece_index);

        /**
         * Indicate that piece designed by piece index is used
         * @param p_piece_index
         */
        inline
        __host__ __device__
        void set_piece_unavailable(unsigned int p_piece_index);

        /**
         * Alias to improve code readibility
         */
        typedef uint16_t t_piece_info;
        typedef t_piece_info t_piece_infos[8];

        /**
         * Return piece information related to a thread
         * @param p_thread_id
         * @return piece information
         */
        inline
        __device__
        t_piece_infos & get_thread_piece_info();

        /**
         * Reset values of piece info depending on piece availabilitys
         * @param p_thread_id thread ID
         */
        inline
        __device__
        void clear_piece_info();

      private:

        /**
         * Return information related to this position index at this level
         * @param p_level
         * @param p_info_index
         * @return Position information: available transitions
         */
        inline
        __device__ __host__
        const CUDA_piece_position_info2 &
        get_position_info(uint32_t p_level
                         ,uint32_t p_info_index
                         ) const;

        /**
         * Return information related to this position index at this level
         * @param p_level
         * @param p_info_index
         * @return Position information: available transitions
         */
        inline
        __device__ __host__
        CUDA_piece_position_info2 &
        get_position_info(uint32_t p_level
                         ,uint32_t p_info_index
                         );

        /**
         * Return information related to best candidate at this position index
         * at this level
         * @param p_level
         * @param p_info_index
         * @return Position information: available transitions
         */
        inline
        __device__ __host__
        const CUDA_piece_position_info2 & get_best_candidate_info(uint32_t p_level
                                                                 ,uint32_t p_info_index
                                                                 ) const;

        /**
         * Return information related to best candidate at this position index
         * at this level
         * @param p_level
         * @param p_info_index
         * @return Position information: available transitions
         */
        inline
        __device__ __host__
        CUDA_piece_position_info2 & get_best_candidate_info(uint32_t p_level
                                                           ,uint32_t p_info_index
                                                           );


        inline
        __device__ __host__
        uint32_t compute_situation_index(uint32_t p_level) const;

        /**
         * Help method to compute word index in a bitfield composed of 32 bits
         * words
         * @param p_index
         * @return word index
         */
        inline static
        __device__ __host__
        uint32_t compute_word_index(uint32_t p_index);

        /**
         * Help method to compute bit index in a word for a bitfield composed
         * of 32 bits words
         * @param p_index
         * @return bit index
         */
        inline static
        __device__ __host__
        uint32_t compute_bit_index(uint32_t p_index);

        uint32_t m_size;

        uint32_t m_level;

        /**
         * Store correspondence between position info and position
         */
        CUDA_memory_managed_array<uint32_t> m_index_to_position;

        /**
         * Store correspondence between position and position info
         */
        CUDA_memory_managed_array<uint32_t> m_position_to_index;

        /**
         * Store available pieces
         */
         uint32_t m_available_pieces[8];

         /**
          * Store piece infos
          */
          t_piece_infos m_thread_piece_infos[32];

        CUDA_piece_position_info2 * m_position_infos;
        CUDA_piece_position_info2 * m_best_candidate_infos;
    };

    //-------------------------------------------------------------------------
    CUDA_glutton_max_stack::CUDA_glutton_max_stack(uint32_t p_size
                                                  ,uint32_t p_nb_pieces
                                                  )
    :m_size(p_size)
    ,m_level{0}
    ,m_index_to_position(p_size, std::numeric_limits<uint32_t>::max())
    ,m_position_to_index(p_nb_pieces, std::numeric_limits<uint32_t>::max())
    ,m_available_pieces{0, 0, 0, 0, 0, 0, 0, 0}
    ,m_thread_piece_infos{{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         ,{0, 0, 0, 0, 0, 0, 0, 0}
                         }
    ,m_position_infos{new CUDA_piece_position_info2[(p_size * (p_size + 1)) / 2]}
    ,m_best_candidate_infos{new CUDA_piece_position_info2[(p_size * (p_size + 1)) / 2]}
    {
    }

    //-------------------------------------------------------------------------
    CUDA_glutton_max_stack::~CUDA_glutton_max_stack()
    {
        delete[] m_best_candidate_infos;
        delete[] m_position_infos;
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    const CUDA_piece_position_info2 &
    CUDA_glutton_max_stack::get_position_info(uint32_t p_info_index) const
    {
        return get_position_info(m_level, p_info_index);
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    const CUDA_piece_position_info2 &
    CUDA_glutton_max_stack::get_best_candidate_info(uint32_t p_info_index) const
    {
        return get_best_candidate_info(m_level, p_info_index);
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    CUDA_piece_position_info2 &
    CUDA_glutton_max_stack::get_best_candidate_info(uint32_t p_info_index)
    {
        return get_best_candidate_info(m_level, p_info_index);
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    uint32_t
    CUDA_glutton_max_stack::compute_situation_index(uint32_t p_level) const
    {
        assert(p_level < m_size);
        return (m_size * (m_size + 1) - (m_size - p_level) * (m_size - p_level + 1)) / 2;
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    uint32_t
    CUDA_glutton_max_stack::get_size() const
    {
        return m_size;
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    const CUDA_piece_position_info2 &
    CUDA_glutton_max_stack::get_position_info(uint32_t p_level
                                             ,uint32_t p_info_index
                                             ) const
    {
        assert(p_info_index < m_size - p_level);
        return m_position_infos[compute_situation_index(p_level + p_info_index)];
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    CUDA_piece_position_info2 &
    CUDA_glutton_max_stack::get_position_info(uint32_t p_level
                                             ,uint32_t p_info_index
                                             )
    {
        assert(p_info_index < m_size - p_level);
        return m_position_infos[compute_situation_index(p_level + p_info_index)];
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    const CUDA_piece_position_info2 &
    CUDA_glutton_max_stack::get_best_candidate_info(uint32_t p_level
                                                   ,uint32_t p_info_index
                                                   ) const
    {
        assert(p_info_index < m_size - p_level);
        return m_best_candidate_infos[compute_situation_index(p_level + p_info_index)];
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    CUDA_piece_position_info2 &
    CUDA_glutton_max_stack::get_best_candidate_info(uint32_t p_level
                                                   ,uint32_t p_info_index
                                                   )
    {
        assert(p_info_index < m_size - p_level);
        return m_best_candidate_infos[compute_situation_index(p_level + p_info_index)];
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    void
    CUDA_glutton_max_stack::push()
    {
        assert(m_level < m_size);
        ++m_level;
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    void
    CUDA_glutton_max_stack::pop()
    {
        assert(m_level);
        --m_level;
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    uint32_t
    CUDA_glutton_max_stack::get_level() const
    {
        return m_level;
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    uint32_t
    CUDA_glutton_max_stack::get_nb_position() const
    {
        return m_size - m_level;
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_max_stack::set_position_index(uint32_t p_index,
                                               uint32_t p_position_index
                                              )
    {
        // Should check m_position_to_index array size but consider that the
        // check is done by caller
        assert(p_index < m_size);
        m_position_to_index[p_position_index] = p_index;
        m_index_to_position[p_index] = p_position_index;
    }

    //-------------------------------------------------------------------------
    __host__ __device__
    uint32_t
    CUDA_glutton_max_stack::get_index_of_position(uint32_t p_position_index) const
    {
        // Should check array size but consider that is method is mainly used
        // with a position index comming from get_position_of_index so correct
        // by construction and when considering neighbourood check on piece
        // color avoid out of boundaries positions
        //assert(p_position_index < m_nb_pieces);
        return m_position_to_index[p_position_index];
    }

    //-------------------------------------------------------------------------
    __host__ __device__
    uint32_t
    CUDA_glutton_max_stack::get_position_of_index(uint32_t p_index) const
    {
        assert(p_index < m_size);
        return m_index_to_position[p_index];
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    bool
    CUDA_glutton_max_stack::is_position_index_used(unsigned int p_index) const
    {
        assert(p_index < m_size);
        return (p_index < m_size - m_level) && m_index_to_position[p_index] != 0xFFFFFFFFu; //std::numeric_limits<uint32_t>::max();
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    bool
    CUDA_glutton_max_stack::is_position_free(unsigned int p_position_index) const
    {
        // Should check array size
        //assert(p_position_index < m_nb_pieces);
        return m_position_to_index[p_position_index] < m_size - m_level;
    }

    //-------------------------------------------------------------------------
    void
    CUDA_glutton_max_stack::set_position_info(uint32_t p_info_index
                                             ,const CUDA_piece_position_info2 & p_info
                                             )
    {
        get_position_info(m_level, p_info_index) = p_info;
    }

    //-------------------------------------------------------------------------
    __host__ __device__
    bool
    CUDA_glutton_max_stack::is_piece_available(unsigned int p_piece_index) const
    {
        assert(p_piece_index < 256);
        return m_available_pieces[compute_word_index(p_piece_index)] & (1u << compute_bit_index(p_piece_index));
    }

    //-------------------------------------------------------------------------
    __host__ __device__
    void
    CUDA_glutton_max_stack::set_piece_available(unsigned int p_piece_index)
    {
        assert(p_piece_index < 256);
        assert(!is_piece_available(p_piece_index));
        m_available_pieces[compute_word_index(p_piece_index)] |= (1u << compute_bit_index(p_piece_index));
    }

    //-------------------------------------------------------------------------
    __host__ __device__
    void
    CUDA_glutton_max_stack::set_piece_unavailable(unsigned int p_piece_index)
    {
        assert(p_piece_index < 256);
        assert(is_piece_available(p_piece_index));
        m_available_pieces[compute_word_index(p_piece_index)] &= ~(1u << compute_bit_index(p_piece_index));
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    uint32_t
    CUDA_glutton_max_stack::compute_word_index(uint32_t p_index)
    {
        unsigned int l_word_index = p_index / 32;
        return l_word_index;
    }

    //-------------------------------------------------------------------------
    __device__ __host__
    uint32_t
    CUDA_glutton_max_stack::compute_bit_index(uint32_t p_index)
    {
        unsigned int l_bit_index = p_index % 32;
        return l_bit_index;
    }

    //-------------------------------------------------------------------------
    __device__
    CUDA_glutton_max_stack::t_piece_infos &
    CUDA_glutton_max_stack::get_thread_piece_info()
    {
        assert(threadIdx.x < 32);
        return m_thread_piece_infos[threadIdx.x];
    }

    //-------------------------------------------------------------------------
    __device__
    void
    CUDA_glutton_max_stack::clear_piece_info()
    {
        assert(threadIdx.x < 32);
        for(unsigned int l_index = 0; l_index < 8; ++l_index)
        {
            m_thread_piece_infos[threadIdx.x][l_index] = is_piece_available(8 * threadIdx.x + l_index) ? 0 : 0xFFFF;
        }
    }

}
#endif //EDGE_MATCHING_PUZZLE_CUDA_GLUTTON_MAX_STACK_H
// EOF
