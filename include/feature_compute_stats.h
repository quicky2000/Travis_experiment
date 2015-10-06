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

#ifndef FEATURE_COMPUTE_STATS
#define FEATURE_COMPUTE_STATS

#include "emp_gui.h"
#include "algo_based_feature.h"
#include "algorithm_deep_raw.h"
#include <string>
#include <unistd.h>
#include <fstream>
  
namespace edge_matching_puzzle
{
  class feature_compute_stats: public algo_based_feature<FSM_framework::algorithm_deep_raw>
  {
  public:
    inline feature_compute_stats(const emp_piece_db & p_db,
                                 const emp_FSM_info & p_info,
                                 emp_gui & p_gui);
    inline ~feature_compute_stats(void);
    // Methods to implement inherited from algo_based_feature
    inline const std::string& get_class_name(void) const;
    inline void print_status(void);
    // End of Methods to implement inherited from algo_based_feature
  private:
    // Methods to implement inherited from algo_based_feature
    inline void display_specific_situation(const emp_FSM_situation & p_situation);
    // End of method to implement inherited from algo_based_feature
    inline void print_stats(std::ostream & p_stream);
    std::ofstream m_report_file;
    uint64_t m_nb;
    uint64_t m_nb_final;
    unsigned int m_length;
    uint64_t * m_invalid_stats;
    uint64_t * m_intermediate_stats;
    static const std::string m_class_name;
  };

  //----------------------------------------------------------------------------
  feature_compute_stats::feature_compute_stats(const emp_piece_db & p_db,
                                               const emp_FSM_info & p_info,
                                               emp_gui & p_gui):
    algo_based_feature<FSM_framework::algorithm_deep_raw>(p_db,p_info,p_gui),
    m_report_file(NULL),
    m_nb(0),
    m_nb_final(0),
    m_length(p_info.get_width()*p_info.get_height()),
    m_invalid_stats(new uint64_t[m_length]),
    m_intermediate_stats(new uint64_t[m_length])
      {
        memset(m_invalid_stats,0,m_length*sizeof(uint64_t));
        memset(m_intermediate_stats,0,m_length*sizeof(uint64_t));
        std::stringstream l_stream_width;
        l_stream_width << p_info.get_width();
        std::stringstream l_stream_height;
        l_stream_height << p_info.get_height();
        std::string l_file_name = "stats_"+l_stream_width.str()+"_"+l_stream_height.str()+".txt";
        m_report_file.open(l_file_name.c_str());
        if(!m_report_file.is_open()) throw quicky_exception::quicky_runtime_exception("Unable to create file \""+l_file_name+"\"",__LINE__,__FILE__);
      }

    //----------------------------------------------------------------------------
    void feature_compute_stats::print_stats(std::ostream & p_stream)
    {
      p_stream << "Nunber of invalid situations per level : " << std::endl ;
      uint64_t l_nb_invalid = 0;
      for(unsigned int l_index = 0 ; l_index < m_length ; ++l_index)
        {
          p_stream << l_index + 1 << "\t:" << m_invalid_stats[l_index] << std::endl ;
          l_nb_invalid += m_invalid_stats[l_index];
        }
      p_stream << std::endl << "Nunber of intermediate situations per level : " << std::endl ;
      uint64_t l_nb_intermediate = 0;
      for(unsigned int l_index = 0 ; l_index < m_length ; ++l_index)
        {
          p_stream << l_index + 1 << "\t:" << m_intermediate_stats[l_index] << std::endl ;
          l_nb_intermediate += m_intermediate_stats[l_index];
        }
      p_stream << std::endl << "Total final situations : " <<  m_nb_final << std::endl ;
      p_stream << std::endl << "Total invalid situations : " <<  l_nb_invalid << std::endl ;
      p_stream << std::endl << "Total intermediate situations : " <<  l_nb_intermediate << std::endl ;
      p_stream << std::endl << "Total situations : " << m_nb << std::endl ;
    }
    //----------------------------------------------------------------------------
    void feature_compute_stats::print_status(void)
    {
      print_stats(std::cout);
    }

    //----------------------------------------------------------------------------
    feature_compute_stats::~feature_compute_stats(void)
      {
        print_stats(m_report_file);
        m_report_file.close();
        delete[] m_invalid_stats;
        delete[] m_intermediate_stats;
      }

    //----------------------------------------------------------------------------
    void feature_compute_stats::display_specific_situation(const emp_FSM_situation & p_situation)
    {
      ++m_nb;
      if(p_situation.is_valid() && !p_situation.is_final())
        {
          ++(m_intermediate_stats[p_situation.get_level()-1]);
        }
      else if(!p_situation.is_valid())
        {
          ++(m_invalid_stats[p_situation.get_level()-1]);
        }
      else
        {
          ++m_nb_final;
        }
    }

    //----------------------------------------------------------------------------
    const std::string & feature_compute_stats::get_class_name(void) const
      {
        return m_class_name;
      }
}

#endif // FEATURE_COMPUTE_STATS
//EOF
