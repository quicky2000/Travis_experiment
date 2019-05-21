/*    This file is part of edge_matching_puzzle
      Copyright (C) 2019  Julien Thevenon ( julien_thevenon at yahoo.fr )

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

#ifndef _EMP_SYSTEM_EQUATION_H_
#define _EMP_SYSTEM_EQUATION_H_

#include "feature_if.h"
#include "emp_piece_db.h"
#include "emp_FSM_info.h"
#include "emp_gui.h"
#include "emp_FSM_situation.h"
#include "emp_variable_generator.h"
#include "emp_types.h"
#include "emp_se_step_info.h"
#include <string>
#include <vector>
#include <memory>

namespace edge_matching_puzzle
{
    class emp_se_step_info;
    class emp_strategy_generator;

    class feature_system_equation: public feature_if
    {
      public:
        feature_system_equation(const emp_piece_db & p_db
                               ,std::unique_ptr<emp_strategy_generator> & p_strategy_generator
                               ,const emp_FSM_info & p_info
                               ,const std::string & p_initial_situation
                               ,const std::string & p_hint_string
                               ,emp_gui & p_gui
                               );

        // Method inherited from feature if
        void run() override;

        // End of method inherited from feature if

        ~feature_system_equation() override = default;

      private:

        emp_FSM_situation extract_situation(const std::vector<emp_se_step_info> & p_stack
                                           ,unsigned int p_step
                                           );

#ifdef DEBUG
        void print_bitfield(const emp_types::bitfield & p_bitfield);
#endif // DEBUG

        /**
         * Indicate that piece corresponding to this variable should no more
         * be checked to detect if it cannot be used
         * @param p_variable variable indicating piece and its location
         * @return index in piece check list
         */
        unsigned int mark_checked(const simplex_variable & p_variable);

        /**
         * Strategy generator that will determine order of positions
         */
        const std::unique_ptr<emp_strategy_generator> m_strategy_generator;

        /**
         * Contains initial situation
         */
        emp_FSM_situation m_initial_situation;

        /**
         * Contains hint situation
         */
        emp_FSM_situation m_hint_situation;

        /**
         * Generate variables of equation system representing the puzzle
         */
        emp_variable_generator m_variable_generator;


        std::vector<emp_types::bitfield> m_pieces_and_masks;

        /**
         * Masks used to check if a position can still be used
         */
        std::vector<emp_types::bitfield> m_positions_check_mask;

        /**
         * Masks used to check if a piece can still be used
         * Boolean indicate if make is usable or not:
         * true : mask can be used, piece still not used
         * false: mask cannot be used, piece is used
         */
        std::vector<std::pair<bool,emp_types::bitfield> > m_pieces_check_mask;

        /**
         * Graphical interface for situation display
         */
        emp_gui & m_gui;

        /**
         * Puzzle information
         */
        const emp_FSM_info & m_info;
    };
}
#endif // _EMP_SYSTEM_EQUATION_H_
// EOF