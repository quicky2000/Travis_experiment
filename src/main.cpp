/*    This file is part of edge_matching_puzzle
      The aim of this software is to find some solutions
      to edge matching  puzzles
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
#include "signal_handler.h"
#include "parameter_manager.h"
#include "emp_piece.h"
#include "emp_pieces_parser.h"
#include "emp_gui.h"
#include "emp_piece_db.h"
#include "emp_FSM.h"

#include "feature_display_all.h"
#include "feature_display_max.h"
#include "feature_display_solutions.h"
#include "feature_dump_solutions.h"
#include "feature_compute_stats.h"
#include "feature_dump_summary.h"
#include "feature_display_dump.h"


#include "emp_spiral_strategy_generator.h"
#include "emp_strategy.h"

#include "quicky_exception.h"
#include <unistd.h>


#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <set>
#include <map>

using namespace edge_matching_puzzle;
using namespace parameter_manager;


//------------------------------------------------------------------------------
int main(int argc,char ** argv)
{
  try
    {
      // Defining application command line parameters
      parameter_manager::parameter_manager l_param_manager("edge_matching_puzzle.exe","--",3);
      parameter_if l_definition_file("definition",false);
      l_param_manager.add(l_definition_file);
      parameter_if l_ressources_path("ressources",false);
      l_param_manager.add(l_ressources_path);
      parameter_if l_feature_name_parameter("feature_name",false);
      l_param_manager.add(l_feature_name_parameter);
      parameter_if l_dump_file("dump_file",true);
      l_param_manager.add(l_dump_file);

      // Treating parameters
      l_param_manager.treat_parameters(argc,argv);

      std::string l_dump_file_name = l_dump_file.value_set() ? l_dump_file.get_value<std::string>() : "results.bin";

      // Get puzzle description
      std::vector<emp_piece> l_pieces;
      emp_pieces_parser l_piece_parser(l_definition_file.get_value<std::string>().c_str());
      unsigned int l_width = 0;
      unsigned int l_height = 0;
      l_piece_parser.parse(l_width,l_height,l_pieces);

      std::cout << l_pieces.size() << " pieces loaded" << std::endl ;
      if(l_pieces.size() != l_width * l_height)
        {
          std::stringstream l_stream_width;
          l_stream_width << l_width;
          std::stringstream l_stream_height;
          l_stream_height << l_height;
          std::stringstream l_stream_nb_pieces;
          l_stream_nb_pieces << l_pieces.size();
          throw quicky_exception::quicky_logic_exception("Inconsistency between puzzle dimensions ("+l_stream_width.str()+"*"+l_stream_height.str()+") and piece number ("+l_stream_nb_pieces.str()+")",__LINE__,__FILE__);
        }


      emp_gui l_gui(l_width,l_height,l_ressources_path.get_value<std::string>().c_str(),l_pieces);

      emp_piece_db l_piece_db(l_pieces,l_width,l_height);
      emp_FSM_info l_info(l_width,l_height,l_piece_db.get_piece_id_size(),l_piece_db.get_dumped_piece_id_size());

      emp_FSM_situation::init(l_info);


      feature_if * l_feature = NULL;
      std::string l_feature_name = l_feature_name_parameter.get_value<std::string>();
      if("display_all" == l_feature_name)
        {
          l_feature = new feature_display_all(l_piece_db,l_info,l_gui);
        }
      else if("display_max" == l_feature_name)
        {
          l_feature = new feature_display_max(l_piece_db,l_info,l_gui);
        }
      else if("display_solutions" == l_feature_name)
        {
          l_feature = new feature_display_solutions(l_piece_db,l_info,l_gui);
        }
      else if("dump_solutions" == l_feature_name)
        {
          l_feature = new feature_dump_solutions(l_piece_db,l_info,l_gui,l_dump_file_name);
        }
      else if("dump_summary" == l_feature_name)
        {
          l_feature = new feature_dump_summary(l_dump_file_name,l_info);
        }
      else if("display_dump" == l_feature_name)
        {
          l_feature = new feature_display_dump(l_dump_file_name,l_info,l_gui);
        }
      else if("compute_stats" == l_feature_name)
        {
          l_feature = new feature_compute_stats(l_piece_db,l_info,l_gui);
        }
      else if("new_strategy" == l_feature_name)
	{
          // No need to delte this objetct, it will be done in emp_strategy destructor
	  emp_spiral_strategy_generator * l_generator = new emp_spiral_strategy_generator(l_info.get_width(),l_info.get_height());
	  l_generator->generate();
	  l_feature = new emp_strategy(*l_generator,l_piece_db,l_gui,l_info,l_dump_file_name);
	}
      else
        {
          throw quicky_exception::quicky_logic_exception("Unsupported feature \""+l_feature_name+"\"",__LINE__,__FILE__);
        }
      l_feature->run();
      delete l_feature;
    }
  catch(quicky_exception::quicky_runtime_exception & e)
    {
      std::cout << "ERROR : " << e.what() << std::endl ;
      return(-1);
    }
  catch(quicky_exception::quicky_logic_exception & e)
    {
      std::cout << "ERROR : " << e.what() << std::endl ;
      return(-1);
    }
  return 0;
  
}
//EOF
