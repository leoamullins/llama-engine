#include "llama.hpp"
#include <exception>
#include <iostream>

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    std::cerr << "usage: " << argv[0] << " <checkpoint>\n";
    return 1;
  }

  try
  {
    llama::Model model(argv[1]);
  }
  catch (const std::exception &e)
  {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
