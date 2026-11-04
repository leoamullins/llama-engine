#include "llama.hpp"
#include <exception>
#include <iostream>

int main(int argc, char **argv)
{
  if (argc != 4)
  {
    std::cerr << "usage: " << argv[0] << " <checkpoint> <vocab> <prompt>\n";
    return 1;
  }

  try
  {
    llama::Model model(argv[1]);
    llama::Tokenizer tokenizer(argv[2]);
    llama::Sampler sampler(/*temperature=*/0.8f, /*top_p=*/0.9f, /*seed=*/42, tokenizer.vocab_size());

    const std::string prompt = argv[3];
    std::cout << prompt << llama::generate(model, tokenizer, sampler, prompt, 100) << "\n";
  }
  catch (const std::exception &e)
  {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
