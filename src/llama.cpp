// In progress
#include "llama.hpp"
#include "mapped_file.hpp"
#include <vector>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <cmath>

namespace llama
{
  namespace
  {
    Config read_config(const MappedFile &file)
    {
      if (file.size() < sizeof(Config))
      {
        throw std::runtime_error("File too small to contain Config");
      }
      Config cfg;
      std::memcpy(&cfg, file.data(), sizeof(Config));

      if (cfg.dim <= 0 || cfg.hidden_dim <= 0 || cfg.n_layers <= 0 || cfg.n_heads <= 0 ||
          cfg.n_kv_heads <= 0 || cfg.vocab_size <= 0 || cfg.max_seq_len_ <= 0)
      {
        throw std::runtime_error("Config fields must be positive");
      }
      if (cfg.dim % cfg.n_heads != 0 || cfg.n_heads % cfg.n_kv_heads != 0)
      {
        throw std::runtime_error("n_heads must divide dim and n_kv_heads must divide n_heads");
      }

      const std::size_t dim = cfg.dim, hidden = cfg.hidden_dim, vocab = cfg.vocab_size;
      const std::size_t kv_dim = cfg.n_kv_heads * dim / cfg.n_heads;

      const std::size_t per_layer = dim + 2 * dim * dim + 2 * kv_dim * dim + dim + 3 * hidden * dim;

      const std::size_t n_floats = vocab * dim + cfg.n_layers * per_layer + dim + vocab * dim;

      if (file.size() != sizeof(Config) + n_floats * sizeof(float))
      {
        throw std::runtime_error("File size does not match Config");
      }
      if ((cfg.dim / cfg.n_heads) % 2 != 0)
      {
        throw std::runtime_error("dim / n_heads must be even");
      }
      return cfg;
    }
  }

  KVCache::KVCache(const Config &cfg) : max_seq_len_(cfg.max_seq_len_),
                                        kv_dim_(cfg.n_kv_heads * cfg.dim / cfg.n_heads),
                                        k_(static_cast<std::size_t>(cfg.n_layers) * max_seq_len_ * kv_dim_),
                                        v_(k_.size()) {}

  float *KVCache::k(int layer, int pos)
  {
    return k_.data() + (static_cast<std::size_t>(layer * max_seq_len_ + pos) * kv_dim_);
  }

  float *KVCache::v(int layer, int pos)
  {
    return v_.data() + (static_cast<std::size_t>(layer * max_seq_len_ + pos) * kv_dim_);
  }

  RunState::RunState(const Config &cfg) : x(cfg.dim), xb(cfg.dim), xb2(cfg.dim), q(cfg.dim), hb(cfg.hidden_dim), hb2(cfg.hidden_dim), att(cfg.n_heads * cfg.max_seq_len_), logits(cfg.vocab_size) {}

  Model::Model(const std::string &checkpoint_path)
      : file_(checkpoint_path), cfg_(read_config(file_)), w_(), cache_(cfg_), s_(cfg_)
  {
    const float *p = reinterpret_cast<const float *>(file_.data() + sizeof(Config));
    auto take = [&p](std::size_t n)
    {
      const float *t = p;
      p += n;
      return t;
    };

    const std::size_t dim = cfg_.dim, hidden = cfg_.hidden_dim, vocab = cfg_.vocab_size;
    const std::size_t kv_dim = cfg_.n_kv_heads * dim / cfg_.n_heads;

    w_.tok_embeddings = take(vocab * dim);
    w_.layers.resize(cfg_.n_layers);
    for (LayerWeights &l : w_.layers)
    {
      l.attn_norm = take(dim);
      l.wq = take(dim * dim);
      l.wo = take(dim * dim);
      l.wk = take(kv_dim * dim);
      l.wv = take(kv_dim * dim);
      l.ffn_norm = take(dim);
      l.w1 = take(hidden * dim);
      l.w3 = take(hidden * dim);
      l.w2 = take(dim * hidden);
    }
    w_.final_norm = take(dim);
    w_.output = take(vocab * dim);
  }

  const Config &Model::config() const
  {
    return cfg_;
  }

  namespace
  {

    void rms_norm(float *out, const float *x, const float *weight, std::size_t n, float eps)
    {
      float sum = 0.0f;
      for (std::size_t i = 0; i < n; ++i)
      {
        sum += x[i] * x[i];
      }

      const float scale = 1.0f / std::sqrt(sum / n + eps);
      for (std::size_t i = 0; i < n; ++i)
      {
        out[i] = x[i] * scale * weight[i];
      }
    }

    void matvec(float *out, const float *m, const float *x, std::size_t rows, std::size_t cols)
    {
      for (std::size_t r = 0; r < rows; ++r)
      {
        float sum = 0.0f;
        for (std::size_t c = 0; c < cols; ++c)
        {
          sum += m[r * cols + c] * x[c];
        }
        out[r] = sum;
      }
    }

    void softmax(float *out, const float *x, std::size_t n)
    {
      float max_val = x[0];
      for (std::size_t i = 1; i < n; ++i)
      {
        if (x[i] > max_val)
        {
          max_val = x[i];
        }
      }
      float sum = 0.0f;
      for (std::size_t i = 0; i < n; ++i)
      {
        out[i] = std::exp(x[i] - max_val);
        sum += out[i];
      }
      for (std::size_t i = 0; i < n; ++i)
      {
        out[i] /= sum;
      }
    }

    void silu(float *out, const float *x, std::size_t n)
    {
      for (std::size_t i = 0; i < n; ++i)
      {
        out[i] = x[i] / (1.0f + std::exp(-x[i]));
      }
    }

    void apply_rope(float *x, std::size_t head_dim, int pos, float theta)
    {
      for (std::size_t i = 0; i < head_dim / 2; ++i)
      {
        const float exponent = static_cast<float>(2 * i) / head_dim;
        const float angle = pos / std::pow(theta, exponent);
        const float c = std::cos(angle), s = std::sin(angle);
        const float a = x[2 * i], b = x[2 * i + 1];
        x[2 * i] = a * c - b * s;
        x[2 * i + 1] = a * s + b * c;
      }
    }
  }

  const float *Model::forward(int token, int pos)
  {
    if (token < 0 || token >= cfg_.vocab_size)
    {
      throw std::out_of_range("token out of range");
    }
    if (pos < 0 || pos >= cfg_.max_seq_len_)
    {
      throw std::out_of_range("pos out of range");
    }

    const std::size_t dim = cfg_.dim, vocab = cfg_.vocab_size;

    // each row of tok_embeddings is one tokens vector
    std::memcpy(s_.x.data(), w_.tok_embeddings + static_cast<std::size_t>(token) * dim, dim * sizeof(float));

    for (const LayerWeights &l : w_.layers)
    {
      // attention then feed forward
    }
    rms_norm(s_.x.data(), s_.x.data(), w_.final_norm, dim, cfg_.norm_eps);
    matvec(s_.logits.data(), w_.output, s_.xb.data(), vocab, dim);
    return s_.logits.data();
  }
}
