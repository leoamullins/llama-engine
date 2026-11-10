#include "llama.hpp"
#include <exception>
#include <iostream>

struct Args
{
  std::string checkpoint, vocab, prompt;
  int steps = 100;
  float temp = 0.8f;
  bool stop_at_eot = true;
};

static Args parse_args(int argc, char **argv)
{
  Args a;
  std::vector<std::string> positional;

  for (int i = 1; i < argc; ++i)
  {
    std::string arg = argv[i];

    auto next_value = [&]() -> std::string
    {
      if (i + 1 >= argc)
      {
        throw std::invalid_argument(arg + " needs a value");
      }
      return argv[++i];
    };

    if (arg == "--steps")
    {
      a.steps = std::stoi(next_value());
    }
    else if (arg == "--temp")
    {
      a.temp = std::stof(next_value());
    }
    else if (arg == "--no-eot")
    {
      a.stop_at_eot = false;
    }
    else if (arg.rfind("--", 0) == 0)
    {
      throw std::invalid_argument("unkown flag: " + arg);
    }
    else
    {
      positional.push_back(arg);
    }
  }
  if (positional.size() != 3)
  {
    throw std::invalid_argument("usage: main <checkpoint> <vocab> <prompt> [--steps N] [--temp T] [--no-eot]");
  }

  a.checkpoint = positional[0];
  a.vocab = positional[1];
  a.prompt = positional[2];

  return a;
}

int main(int argc, char **argv)
{
  try
  {
    Args args = parse_args(argc, argv);

    llama::Model model(args.checkpoint);
    llama::Tokenizer tokenizer(args.vocab);
    llama::Sampler sampler(args.temp, /*top_p=*/0.9f, /*seed=*/42, tokenizer.vocab_size());

    const std::string prompt = args.prompt;
    std::cout << prompt << llama::generate(model, tokenizer, sampler, prompt, args.steps) << "\n";
  }
  catch (const std::exception &e)
  {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
