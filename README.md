# llama-engine

A from-scratch C++ inference engine for a Llama 3 style model I pretrained myself
([leoamullins/llama3](https://github.com/leoamullins/llama3)). No dependencies beyond
the C++17 standard library: the checkpoint is memory-mapped and every operation
(attention, RoPE, SwiGLU, RMSNorm, sampling, BPE tokenization) is written by hand.

## Architecture

The engine runs the same model as the training code:

- Token embeddings tied to the output head
- Pre-norm transformer blocks with RMSNorm
- Grouped-query attention with a KV cache
- Rotary position embeddings (rotate-half layout)
- SwiGLU feed-forward network
- GPT-2 tokenizer (50,257 tokens, vocab padded to 50,304 in the model)
- Temperature and top-p (nucleus) sampling

The default model is 12 layers, 768 dim, 12 query heads, 4 KV heads, 2688 FFN
hidden dim, 1024 context.

## Usage

Build:

```bash
cmake -B build
cmake --build build
```

Export a training checkpoint to the engine's binary format, and the GPT-2 vocab
(both need `torch`/`tiktoken`):

```bash
python scripts/export_weights.py path/to/best.pt model.bin
python scripts/export_vocab.py
```

Generate:

```bash
./build/main model.bin gpt2_vocab.bin "Photosynthesis is the process by which plants"
```

Sampling is currently fixed at temperature 0.8, top-p 0.9, seed 42, 100 new tokens.

## Example outputs

Unedited outputs from the model, which was trained on FineWeb-Edu. It's a small
base model, so expect fluent text with shaky facts and some repetition.

**Prompt:** `Photosynthesis is the process by which plants`

> Photosynthesis is the process by which plants absorb water from the water they absorb.
> In plants, the primary photosynthesis process occurs in the plants’ leaves. The photosynthesis process is controlled by a system of energy exchanges between the plant and the surrounding air. This system is called the chlorophyll cycle.
> By capturing sunlight, plants convert this energy into chemical energy, which is then stored in the plants’ leaves. Photosynthesis is an important process in plants. Photosynthesis is a process by which plants convert sunlight into chemical

**Prompt:** `In 1492, Christopher Columbus`

> In 1492, Christopher Columbus arrived in America. This was a great opportunity for the English to spread their new culture and culture. Christopher Columbus sailed to America in 1492, and with his father Christopher Columbus in 1610, he made his first voyage to the New World.
> This voyage was also a great help to the Spanish to learn how to build a ship and sail a ship to Spain. He made the journey to England in 1493, and he had the opportunity to sail to Spain, but also to the Americas.

**Prompt:** `The water cycle has three main stages:`

> The water cycle has three main stages:
> \- Chlorosis: The water cycle is the first stage in which a filter removes chlorine from the water.
> \- Chlorosis: The chlorine is released when chlorine combines with other elements in the water.
> \- Chlorosis: The chlorine is released when chlorine combines with other elements in the water.
> The chlorine cycle is the third stage in which a filter removes chlorine from the water.
> What are the different types of chlorine filters?
> Chlorine filters are made of a

**Prompt:** `Functional Analysis is the best`

> Functional Analysis is the best choice for both beginners and experienced researchers.
> \- The core principles of clinical presentation and analysis are the same.
> \- The clinical presentation is more complex.
> \- The clinical presentation is more complex.
> \- The clinical presentation is more complicated.
> \- The clinical presentation is more complex.
> \- ...

## Correctness

The forward pass was checked against the PyTorch training model on a small
randomly initialised config: logits agree to within 1e-6 across multiple
positions, covering RoPE, the KV cache, GQA head mapping, and the SwiGLU weight
mapping in `export_weights.py`.

## Performance

About 12 tokens/s on an Apple M3 (single-threaded, fp32, naive `matvec`).

## Next steps

- Match GPT-2's regex pre-tokenization (currently splits on spaces only)
- Command-line flags for sampling settings, streaming output
- Multithreaded and SIMD `matvec`, then fp16/int8 weights
- GPU kernels (HIP)
