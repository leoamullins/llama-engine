// In progress
#include "llama.hpp"
#include "mapped_file.hpp"
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <climits>
#include <limits>
#include <algorithm>
#include <numeric>

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
          cfg.n_kv_heads <= 0 || cfg.vocab_size <= 0 || cfg.max_seq_len <= 0)
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

  KVCache::KVCache(const Config &cfg) : max_seq_len_(cfg.max_seq_len),
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

  RunState::RunState(const Config &cfg) : x(cfg.dim), xb(cfg.dim), xb2(cfg.dim), q(cfg.dim), hb(cfg.hidden_dim), hb2(cfg.hidden_dim), att(cfg.n_heads * cfg.max_seq_len), logits(cfg.vocab_size) {}

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
      const std::size_t half = head_dim / 2;
      for (std::size_t i = 0; i < head_dim / 2; ++i)
      {
        const float exponent = static_cast<float>(2 * i) / head_dim;
        const float angle = pos / std::pow(theta, exponent);
        const float c = std::cos(angle), s = std::sin(angle);
        const float a = x[i], b = x[i + half];
        x[i] = a * c - b * s;
        x[i + half] = a * s + b * c;
      }
    }
  }

  const float *Model::forward(int token, int pos)
  {
    if (token < 0 || token >= cfg_.vocab_size)
    {
      throw std::out_of_range("token out of range");
    }
    if (pos < 0 || pos >= cfg_.max_seq_len)
    {
      throw std::out_of_range("pos out of range");
    }

    const std::size_t dim = cfg_.dim, vocab = cfg_.vocab_size, hidden = cfg_.hidden_dim;

    // each row of tok_embeddings is one tokens vector
    std::memcpy(s_.x.data(), w_.tok_embeddings + static_cast<std::size_t>(token) * dim, dim * sizeof(float));

    for (int layer = 0; layer < cfg_.n_layers; ++layer)
    {
      const LayerWeights &l = w_.layers[layer];

      rms_norm(s_.xb.data(), s_.x.data(), l.attn_norm, dim, cfg_.norm_eps);

      // calculate q, k, v
      // using the KV Cache to store k and v for each layer and position
      matvec(s_.q.data(), l.wq, s_.xb.data(), dim, dim);

      float *k = cache_.k(layer, pos);
      float *v = cache_.v(layer, pos);
      matvec(k, l.wk, s_.xb.data(), cfg_.n_kv_heads * dim / cfg_.n_heads, dim);
      matvec(v, l.wv, s_.xb.data(), cfg_.n_kv_heads * dim / cfg_.n_heads, dim);

      std::size_t head_dim = dim / cfg_.n_heads;

      for (int h = 0; h < cfg_.n_heads; ++h)
      {
        apply_rope(s_.q.data() + h * head_dim, head_dim, pos, cfg_.rope_theta);
      }
      for (int h = 0; h < cfg_.n_kv_heads; ++h)
      {
        apply_rope(k + h * head_dim, head_dim, pos, cfg_.rope_theta);
      }

      // attention scores
      const int kv_mul = cfg_.n_heads / cfg_.n_kv_heads;
      const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

      for (int h = 0; h < cfg_.n_heads; ++h)
      {
        const float *q_h = s_.q.data() + h * head_dim;
        float *att = s_.att.data() + h * cfg_.max_seq_len;
        const int kv_h = h / kv_mul;

        for (int t = 0; t <= pos; ++t)
        {
          const float *k_t = cache_.k(layer, t) + kv_h * head_dim;
          float score = 0.0f;
          for (std::size_t i = 0; i < head_dim; ++i)
          {
            score += q_h[i] * k_t[i];
          }
          att[t] = score * scale;
        }

        softmax(att, att, pos + 1);

        float *out_h = s_.xb.data() + h * head_dim;
        std::fill(out_h, out_h + head_dim, 0.0f);

        for (int i = 0; i <= pos; ++i)
        {
          const float *v_t = cache_.v(layer, i) + kv_h * head_dim;
          const float a = att[i];
          for (std::size_t j = 0; j < head_dim; ++j)
          {
            out_h[j] += a * v_t[j];
          }
        }
      }
      // residual + attention output proj
      matvec(s_.xb2.data(), l.wo, s_.xb.data(), dim, dim);
      for (std::size_t i = 0; i < dim; ++i)
      {
        s_.x[i] += s_.xb2[i];
      }

      // rms norm before ffn (x is storing prenorm vals)
      rms_norm(s_.xb.data(), s_.x.data(), l.ffn_norm, dim, cfg_.norm_eps);

      // project up with w1 and w3
      matvec(s_.hb.data(), l.w1, s_.xb.data(), hidden, dim);  // hb = w1 * x (gate)
      matvec(s_.hb2.data(), l.w3, s_.xb.data(), hidden, dim); // hb2 = w3 * x (up)

      silu(s_.hb.data(), s_.hb.data(), hidden);
      for (std::size_t i = 0; i < hidden; ++i)
      {
        s_.hb[i] *= s_.hb2[i];
      }

      // project back down
      matvec(s_.xb2.data(), l.w2, s_.hb.data(), dim, hidden); // xb2 = w2 * hb

      for (std::size_t i = 0; i < dim; ++i)
      {
        s_.x[i] += s_.xb2[i];
      }
    }

    rms_norm(s_.xb.data(), s_.x.data(), w_.final_norm, dim, cfg_.norm_eps);
    matvec(s_.logits.data(), w_.output, s_.xb.data(), vocab, dim);
    return s_.logits.data();
  }

  Sampler::Sampler(float temperature, float top_p, uint64_t seed, int real_vocab_size) : temperature_(temperature), top_p_(top_p), rng_(seed), real_vocab_size_(static_cast<std::size_t>(real_vocab_size)) {}

  int Sampler::sample(const float *logits, int vocab_size)
  {
    if (vocab_size <= 0)
    {
      throw std::invalid_argument("vocab_size must be positive");
    }
    if (temperature_ < 0.0f)
    {
      throw std::invalid_argument("temperature must be non-negative");
    }
    if (top_p_ < 0.0f || top_p_ > 1.0f)
    {
      throw std::invalid_argument("top_p must be in [0, 1]");
    }
    if (temperature_ == 0.0f)
    {
      // argmax, skipping padded tokens
      int n_real = std::min(real_vocab_size_, vocab_size);
      int max_idx = 0;
      for (int i = 1; i < n_real; ++i)
      {
        if (logits[i] > logits[max_idx])
        {
          max_idx = i;
        }
      }
      return max_idx;
    }

    std::vector<float> probs(static_cast<std::size_t>(vocab_size));
    // copy logits into probs buffer, since logits is const and we may need to modify it for softmax
    std::memcpy(probs.data(), logits, static_cast<std::size_t>(vocab_size) * sizeof(float));

    // masking padded tokens
    for (int i = real_vocab_size_; i < static_cast<int>(probs.size()); ++i)
    {
      probs[static_cast<std::size_t>(i)] = -std::numeric_limits<float>::infinity();
    }

    for (size_t i = 0; i < static_cast<std::size_t>(vocab_size); ++i)
    {
      probs[i] /= temperature_;
    }
    softmax(probs.data(), probs.data(), static_cast<std::size_t>(vocab_size));

    // top-p: sort token ids by probability, keep the smallest prefix whose mass reaches top_p_
    std::vector<int> order(static_cast<std::size_t>(vocab_size));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b)
              { return probs[static_cast<std::size_t>(a)] > probs[static_cast<std::size_t>(b)]; });

    float kept_mass = 0.0f;
    std::size_t n_kept = 0;
    while (n_kept < order.size())
    {
      kept_mass += probs[static_cast<std::size_t>(order[n_kept])];
      ++n_kept;
      if (kept_mass >= top_p_)
      {
        break;
      }
    }

    // generate a random idx within the kept set, scaled so it doesn't need renormalizing
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_) * kept_mass;
    float cumulative = 0.0f;
    for (std::size_t i = 0; i < n_kept; ++i)
    {
      cumulative += probs[static_cast<std::size_t>(order[i])];
      if (r < cumulative)
      {
        return order[i];
      }
    }
    // fallback to the last kept token if not found
    return order[n_kept - 1];
  }

  Tokenizer::Tokenizer(const std::string &vocab_path)
  {
    std::ifstream vocab_file(vocab_path);
    if (!vocab_file.is_open())
    {
      throw std::runtime_error("Failed to open vocab file: " + vocab_path);
    }

    // first 4 bytes are the number of pieces (int32_t)
    int32_t count = 0;
    vocab_file.read(reinterpret_cast<char *>(&count), sizeof(count));

    if (!vocab_file || count <= 0)
    {
      throw std::runtime_error("Invalid vocab file format or empty vocab: " + vocab_path);
    }

    // reserve space for pieces to avoid reallocations, then read each piece and for the the ranks map
    pieces_.reserve(static_cast<std::size_t>(count));
    ranks_.reserve(static_cast<std::size_t>(count));
    for (int32_t i = 0; i < count; ++i)
    {
      int32_t length = 0;
      vocab_file.read(reinterpret_cast<char *>(&length), sizeof(length));

      std::string piece(length, '\0');
      vocab_file.read(piece.data(), length);

      ranks_[piece] = i;

      pieces_.push_back(std::move(piece));
    }
  }

  const std::string &Tokenizer::decode(int id) const
  {
    if (id < 0 || static_cast<std::size_t>(id) >= pieces_.size())
    {
      throw std::out_of_range("Token ID out of range");
    }
    return pieces_[static_cast<std::size_t>(id)];
  }

  std::string Tokenizer::decode(const std::vector<int> &ids) const
  {
    std::string result;
    for (int id : ids)
    {
      result += decode(id);
    }
    return result;
  }

  int Tokenizer::vocab_size() const
  {
    return static_cast<int>(pieces_.size());
  }

  std::vector<int> Tokenizer::encode_chunk(const std::string &chunk) const
  {
    std::vector<std::string> parts;

    parts.reserve(chunk.size());
    for (char c : chunk)
    {
      std::string s(1, c);
      parts.push_back(std::move(s));
    }

    // since the encoder was BPE (byte pair encoded) we need to merge the pieces according to the ranks_ map
    while (true)
    {
      int best_rank = INT_MAX;
      int best_i = -1;

      for (std::size_t i = 0; i + 1 < parts.size(); ++i)
      {
        auto it = ranks_.find(parts[i] + parts[i + 1]);
        if (it != ranks_.end() && it->second < best_rank)
        {
          best_rank = it->second;
          best_i = static_cast<int>(i);
        }
      }
      if (best_i == -1)
      {
        break;
      }
      parts[best_i] += parts[best_i + 1];
      parts.erase(parts.begin() + best_i + 1);
    }

    std::vector<int> ids;
    ids.reserve(parts.size());
    for (const std::string &part : parts)
    {
      ids.push_back(ranks_.at(part));
    }
    return ids;
  }

  std::vector<int> Tokenizer::encode(const std::string &text) const
  {
    std::vector<int> ids;
    std::string current;
    for (char c : text)
    {
      // a space starts a new word, and stays attached to the front of it
      if (c == ' ' && !current.empty())
      {
        std::vector<int> chunk_ids = encode_chunk(current);
        ids.insert(ids.end(), chunk_ids.begin(), chunk_ids.end());
        current.clear();
      }
      current += c;
    }
    if (!current.empty())
    {
      std::vector<int> chunk_ids = encode_chunk(current);
      ids.insert(ids.end(), chunk_ids.begin(), chunk_ids.end());
    }
    return ids;
  }
}
