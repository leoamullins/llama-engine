import struct

import tiktoken


def main(out_path="gpt2_vocab.bin"):
    enc = tiktoken.get_encoding("gpt2")
    n = enc.n_vocab

    with open(out_path, "wb") as f:
        f.write(struct.pack("<i", n))
        for i in range(n):
            piece = enc.decode_single_token_bytes(i)
            f.write(struct.pack("<i", len(piece)))
            f.write(piece)
    print(f"wrote {n} tokens to {out_path}")


if __name__ == "__main__":
    main()
