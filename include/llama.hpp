#pragma once
#include <cstdint>
#include <mapped_file.hpp>
#include <random>
#include <stdio.h>
#include <string>
#include <vector>

namespace llama
{

  struct Config
  {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len;
    float rope_theta, norm_eps;
  };

  struct LayerWeights
  {
    const float *attn_norm, *wq, *wk, *wv, *wo;
    const float *ffn_norm, *w1, *w2, *w3; // SwiGLU
  };

  struct Weights
  {
    const float *tok_embeddings;
    std::vector<LayerWeights> layers;
    const float *final_norm, *output;
  };

  class KVCache
  {
  public:
    KVCache(const Config &cfg);
    float *k(int layer, int pos);
    float *v(int layer, int pos);

  private:
    int max_seq_len_;
    int kv_dim_;
    std::vector<float> k_, v_;
  };

  struct RunState
  {
    std::vector<float> x, xb, xb2, q, hb, hb2, att, logits;
    RunState(const Config &cfg);
  };

  class Model
  {
  public:
    explicit Model(const std::string &checkpoint_path);
    const float *forward(int token, int pos);
    const Config &config() const;

  private:
    MappedFile file_;
    Config cfg_;
    Weights w_;
    KVCache cache_;
    RunState s_;
  };

  class Sampler
  {
  public:
    Sampler(float temperature, float top_p, uint64_t seed, int real_vocab_size);
    int sample(const float *logits, int vocab_size);

  private:
    float temperature_, top_p_;
    std::mt19937_64 rng_;
    int real_vocab_size_;
  };

  class Tokenizer
  {
  public:
    explicit Tokenizer(const std::string &vocab_path);
    const std::string &decode(int id) const;
    std::string decode(const std::vector<int> &ids) const;
    std::vector<int> encode(const std::string &text) const;
    int vocab_size() const;

  private:
    std::vector<std::string> pieces_;
    std::unordered_map<std::string, int> ranks_;
    std::vector<int> encode_chunk(const std::string &chunk) const;
  };

  std::string generate(Model &, Tokenizer &, Sampler &, const std::string &prompt,
                       int max_new_tokens);

} // namespace llama
