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

#include <emp_types.h>

namespace edge_matching_puzzle
{
  const std::string emp_types::m_kind_strings[((uint32_t)emp_types::t_kind::UNDEFINED) + 1] = {"Center","Border","Corner","Undefined"};
  const std::string emp_types::m_orientation_strings[((uint32_t)emp_types::t_orientation::WEST) + 1] = {"North","East","South","West"};
  const char emp_types::m_short_orientation_strings[((uint32_t)emp_types::t_orientation::WEST) + 1] = {'N','E','S','W'};
}
//EOF
