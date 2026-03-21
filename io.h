/*
  * Author: Ozgur Taylan Turan
  * Date: 21 March 2026
  * Description: A rough funciton for reading csv files.
  *
*/

#include <torch/torch.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

namespace io 
{
  torch::Tensor read_csv( const std::filesystem::path& file_path,
                          bool has_header = true )
  {
    std::ifstream file(file_path);

    if (!file.is_open())
      throw std::runtime_error("Could not open file");

    std::string line;
    std::vector<float> data;
    size_t rows = 0;
    size_t cols = 0;

    if (has_header)
      std::getline(file, line);

    while (std::getline(file, line))
    {
      std::stringstream line_stream(line);
      std::string cell;

      size_t current_cols = 0;

      while (std::getline(line_stream, cell, ','))
      {
        data.push_back(std::stof(cell));
        current_cols++;
      }

      if (cols == 0)
        cols = current_cols;

      rows++;
    }

    torch::Tensor tensor = torch::from_blob( data.data(),
                                             {(long)rows, (long)cols},
                                             torch::kFloat32 ).clone();

    return tensor;
  }
}
