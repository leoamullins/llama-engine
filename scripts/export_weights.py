import struct
import sys

import torch


def write(f, t):
    f.write(
        t.detach().cpu().to(torch.float32).contiguous().numpy().astype("<f4").tobytes()
    )


def main(ckpt_path, out_path="model.bin"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = {
        k.removeprefix("_orig_mod.").removeprefix("module."): v
        for k, v in ckpt["model"].items()
    }
    cfg = ckpt["model_cfg"]

    dim, n_heads = cfg["n_embd"], cfg["n_head"]
    hidden = sd["transformer.h.0.mlp.swiglu.w1.weight"].shape[0]

    with open(out_path, "wb") as f:
        # Config: 7 int32 then 2 float32, matching struct Config in llama.hpp
        f.write(
            struct.pack(
                "<7i2f",
                dim,
                hidden,
                cfg["n_layer"],
                n_heads,
                cfg["n_groups"],
                cfg["vocab_size"],
                cfg["block_size"],
                cfg["theta"],
                cfg["eps"],
            )
        )

        # wte is tied to lm_head
        write(f, sd["lm_head.weight"])

        for i in range(cfg["n_layer"]):
            p = f"transformer.h.{i}."
            write(f, sd[p + "rms_1.rmsnorm.weight"])
            # loader order is wq, wo, wk, wv
            write(f, sd[p + "attn.q_proj.weight"])
            write(f, sd[p + "attn.out_proj.weight"])
            write(f, sd[p + "attn.k_proj.weight"])
            write(f, sd[p + "attn.v_proj.weight"])
            write(f, sd[p + "rms_2.rmsnorm.weight"])
            # training names: w1 gate, w2 up, w3 down
            # engine names:   w1 gate, w3 up, w2 down
            write(f, sd[p + "mlp.swiglu.w1.weight"])  # -> engine w1
            write(f, sd[p + "mlp.swiglu.w2.weight"])  # -> engine w3
            write(f, sd[p + "mlp.swiglu.w3.weight"])  # -> engine w2

        write(f, sd["transformer.rms_f.rmsnorm.weight"])
        write(f, sd["lm_head.weight"])

    print(
        f"wrote {out_path} (dim={dim}, hidden={hidden}, layers={cfg['n_layer']}, "
        f"heads={n_heads}, kv_heads={cfg['n_groups']}, vocab={cfg['vocab_size']})"
    )


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: export_weights.py <best.pt> [out.bin]")
    main(*sys.argv[1:])
