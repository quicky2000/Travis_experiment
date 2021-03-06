/*    This file is part of edge matching puzzle
      The aim of this software is to find some solutions
      of edge matching puzzle
      Copyright (C) 2014  Julien Thevenon ( julien_thevenon at yahoo.fr )

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
#ifndef EMP_PIECE_BORDER_H
#define EMP_PIECE_BORDER_H

#include "emp_piece.h"
#include "quicky_exception.h"
#include <cinttypes>

namespace edge_matching_puzzle
{
    class emp_piece_border: public emp_piece
    {
      public:
        inline
        emp_piece_border(const emp_types::t_piece_id & p_id
                        ,const std::array<emp_types::t_color_id, ((unsigned int)(emp_types::t_orientation::WEST)) + 1> & p_colours
                        );

        inline explicit
        emp_piece_border(const emp_piece & p_piece);

        inline
        const std::pair<emp_types::t_color_id, emp_types::t_color_id> & get_border_colors()const;

        inline
        const std::pair<emp_types::t_orientation,emp_types::t_orientation> & get_colors_orientations()const;

        inline
        const emp_types::t_color_id & get_center_color()const;

        inline
        emp_types::t_orientation get_center_orientation()const;

        inline
        emp_types::t_orientation get_border_orientation()const;

        /**
         * Compute piece orientation to match given argument border orientation
         * @param p_border_orientation border orientation
         * @return piece orientation
         */
        inline
        emp_types::t_orientation compute_orientation(emp_types::t_orientation p_border_orientation)const;

      private:
        inline
        void init();

        emp_types::t_orientation m_border_orientation;
        emp_types::t_orientation m_center_orientation;
        std::pair<emp_types::t_color_id, emp_types::t_color_id> m_border_colors;
        std::pair<emp_types::t_orientation,emp_types::t_orientation> m_colors_orientations;
        emp_types::t_color_id m_center_color;
    };

    //----------------------------------------------------------------------------
    emp_piece_border::emp_piece_border(const emp_types::t_piece_id & p_id
                                      ,const std::array<emp_types::t_color_id,((unsigned int)(emp_types::t_orientation::WEST))+1> & p_colours
                                      )
    : emp_piece(p_id,p_colours)
    ,m_border_orientation{}
    ,m_center_orientation{}
    ,m_border_colors{}
    ,m_colors_orientations{}
    ,m_center_color{}
    {
        init();
    }

    //----------------------------------------------------------------------------
    emp_piece_border::emp_piece_border(const emp_piece & p_piece)
    : emp_piece(p_piece.get_id(),p_piece.get_colors())
    ,m_border_orientation{}
    ,m_center_orientation{}
    ,m_border_colors{}
    ,m_colors_orientations{}
    ,m_center_color{}
    {
        init();
    }

    //----------------------------------------------------------------------------
    const emp_types::t_color_id & emp_piece_border::get_center_color()const
	{
        return m_center_color;
	}
    
	//----------------------------------------------------------------------------
    void emp_piece_border::init()
    {
        if(get_kind() != emp_types::t_kind::BORDER)
        {
            throw quicky_exception::quicky_logic_exception("Try to build a border piece with a "+emp_types::kind2string(get_kind()),__LINE__,__FILE__);
        }
        for(auto l_orientation:emp_types::get_orientations())
        {
            if(!get_color(l_orientation))
            {
                m_border_orientation = l_orientation;
                m_center_orientation = emp_types::get_opposite(l_orientation);
                break;
            }
        }
        m_colors_orientations.first = emp_types::get_previous_orientation(m_border_orientation);
        m_colors_orientations.second = emp_types::get_next_orientation(m_border_orientation);
        m_border_colors.first = get_color(m_colors_orientations.first);
        m_border_colors.second = get_color(m_colors_orientations.second);
        m_center_color = get_color(m_center_orientation);
    }

    //----------------------------------------------------------------------------
    emp_types::t_orientation emp_piece_border::get_center_orientation()const
    {
        return m_center_orientation;
    }

    //----------------------------------------------------------------------------
    emp_types::t_orientation emp_piece_border::get_border_orientation()const
    {
        return m_border_orientation;
    }

    //----------------------------------------------------------------------------
    const std::pair<emp_types::t_color_id, emp_types::t_color_id> & emp_piece_border::get_border_colors()const
	{
        return m_border_colors;
	}

	//----------------------------------------------------------------------------
	const std::pair<emp_types::t_orientation,emp_types::t_orientation> & emp_piece_border::get_colors_orientations()const
	{
        return m_colors_orientations;
	}

    //----------------------------------------------------------------------------
    emp_types::t_orientation
    emp_piece_border::compute_orientation(emp_types::t_orientation p_border_orientation)const
    {
        return static_cast<emp_types::t_orientation>((static_cast<unsigned int>(m_border_orientation) + 4 - static_cast<unsigned int>(p_border_orientation)) % 4);
    }

}
#endif // EMP_PIECE_BORDER_H
//EOF
