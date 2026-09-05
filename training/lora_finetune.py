"""LoRA domain adaptation of Moondream2 on BigEarthNet land-cover captions.

The build report's recipe (`get_peft_model` + `Trainer`) cannot work here:
`HfMoondream` is an inference-only wrapper with no trainable `forward`, so
there is nothing for `Trainer` to call. What Moondream does expose is the
uncached full-sequence path — `text._produce_hidden` plus `lm_head` — which is
exactly a training forward pass, so the loop below is written directly against
it. `peft.inject_adapter_in_model` adds the LoRA layers to the text tower in
place, since `model.text` is a bare `nn.ModuleDict` rather than a
`PreTrainedModel`.

What it teaches: the base model describes a Sentinel-2 patch as a generic
aerial photo. The adapter teaches it to name CORINE land-cover classes —
"arable land, broad-leaved forest and inland waters" — which is the remote-
sensing domain adaptation the requirements ask for.

Sequence layout per sample, matching what Moondream's own query path builds:

    [bos] [729 image patch embeddings] [query prefix] [question] [query suffix] [answer] [eos]
    <------------ loss masked out ------------------------------------------> <-- loss -->

The vision encoder runs under no_grad — only the text LoRA is trained.

Run: venv/Scripts/python.exe -m training.lora_finetune
Fits an RTX 4060 8GB at batch 1. Kaggle T4x2 is the fallback if it OOMs.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "sample_data" / "bigearthnet" / "train"
OUT = ROOT / "backend" / "models" / "final_adapter"
IMAGE_PREFIX_LEN = 730  # bos + 729 patch embeddings; text._produce_hidden assumes this


def load_parts():
    """-> (hf_model, MoondreamModel, moondream module, text module)."""
    from backend.tools.vlm import load_model

    model = load_model()
    md = model.model
    moondream_mod = sys.modules[type(md).__module__]
    text_mod = sys.modules[type(md).__module__.rsplit(".", 1)[0] + ".text"]
    return model, md, moondream_mod, text_mod


def build_sample(md, moondream_mod, image, question, answer):
    """-> (inputs_embeds [1,T,D], labels [1,T]) with loss only on the answer."""
    templates = md.config.tokenizer.templates["query"]
    prompt_ids = (
        templates["prefix"]
        + md.tokenizer.encode(" " + question.strip()).ids
        + templates["suffix"]
    )
    answer_ids = md.tokenizer.encode(" " + answer.strip()).ids + [
        md.config.tokenizer.eos_id
    ]

    with torch.no_grad():
        img_emb = md._run_vision_encoder(image)  # (729, dim)
        bos = moondream_mod.text_encoder(
            torch.tensor([[md.config.tokenizer.bos_id]], device=md.device), md.text
        )
    token_ids = torch.tensor([prompt_ids + answer_ids], device=md.device)
    token_emb = moondream_mod.text_encoder(token_ids, md.text)

    inputs_embeds = torch.cat([bos, img_emb[None], token_emb], dim=1)
    labels = torch.full(
        (1, inputs_embeds.size(1)), -100, dtype=torch.long, device=md.device
    )
    labels[0, -len(answer_ids):] = torch.tensor(answer_ids, device=md.device)
    return inputs_embeds, labels


def produce_hidden(inputs_embeds, text, config, text_mod):
    """Uncached full-sequence forward through the text tower.

    Same computation as Moondream's own `text._produce_hidden`, but that one
    builds its attention mask with `torch.zeros(...)` on the CPU, which crashes
    `scaled_dot_product_attention` the moment the model is on a GPU. The mask is
    bidirectional across the image prefix and causal after it, exactly as
    Moondream defines it.
    """
    q_len = inputs_embeds.size(1)
    prefix = min(IMAGE_PREFIX_LEN, q_len)
    attn_mask = torch.zeros(q_len, q_len, dtype=torch.bool, device=inputs_embeds.device)
    attn_mask[:prefix, :prefix] = True
    for i in range(prefix, q_len):
        attn_mask[i, : i + 1] = True

    layers = sys.modules[text_mod.__name__.rsplit(".", 1)[0] + ".layers"]
    hidden = inputs_embeds
    for block in text.blocks:
        normed = layers.layer_norm(hidden, block.ln)
        hidden = hidden + text_mod._attn(
            x=normed, w=block.attn, freqs_cis=text.freqs_cis, attn_mask=attn_mask,
            n_heads=config.n_heads, n_kv_heads=config.n_kv_heads,
        ) + layers.mlp(normed, block.mlp)
    return hidden


def forward_loss(md, text_mod, inputs_embeds, labels):
    hidden = produce_hidden(inputs_embeds, md.text, md.config.text, text_mod)
    # `lm_head` keeps only the last position (it is the generation path);
    # `_lm_head` returns logits for the whole sequence, which is what training needs.
    logits = text_mod._lm_head(hidden, md.text)
    # Next-token prediction: position t predicts token t+1.
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)).float(),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def attach_lora(md, rank=8, alpha=16, dropout=0.05):
    """LoRA on every attention projection in the text tower."""
    from peft import LoraConfig, inject_adapter_in_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        # Confirmed against model.named_modules(), not guessed: Moondream's text
        # blocks are blocks.N.attn.{qkv,proj}, not the q_proj/v_proj of a Llama.
        target_modules=[
            name
            for name, module in md.text.named_modules()
            if isinstance(module, torch.nn.Linear) and name.endswith(("attn.qkv", "attn.proj"))
        ],
    )
    if not config.target_modules:
        raise RuntimeError("No attention projections found — check module names first.")
    for param in md.parameters():
        param.requires_grad = False
    inject_adapter_in_model(config, md.text)
    trainable = [p for p in md.text.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.float()  # LoRA params in fp32 so small updates survive
    return config, trainable


HOLDOUT_PATCHES = 20  # never trained on, so the before/after comparison is honest


def _split():
    rows = [json.loads(line)
            for line in (DATA / "train.jsonl").read_text(encoding="utf-8").splitlines() if line]
    held = set(sorted({r["image"] for r in rows})[-HOLDOUT_PATCHES:])
    return [r for r in rows if r["image"] not in held], [r for r in rows if r["image"] in held]


def load_dataset(limit=None):
    rows, _ = _split()
    random.Random(0).shuffle(rows)
    return rows[:limit] if limit else rows


def holdout():
    return _split()[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="cap the sample count")
    parser.add_argument("--smoke", action="store_true",
                        help="4 steps on one sample, assert the loss falls, save nothing")
    args = parser.parse_args()

    if args.smoke:
        return smoke()

    if not (DATA / "train.jsonl").is_file():
        sys.exit("No training set. Run: python -m training.prepare_bigearthnet train 300")

    rows = load_dataset(args.limit)
    model, md, moondream_mod, text_mod = load_parts()
    config, trainable = attach_lora(md, rank=args.rank)
    n_params = sum(p.numel() for p in trainable)
    print(f"{len(rows)} samples | LoRA r={args.rank} on {len(config.target_modules)} "
          f"modules | {n_params:,} trainable params")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    md.text.train()
    started, running, step = time.time(), 0.0, 0

    for epoch in range(args.epochs):
        for i, row in enumerate(rows, 1):
            with Image.open(DATA / row["image"]) as raw:
                image = raw.convert("RGB")
            embeds, labels = build_sample(md, moondream_mod, image, row["question"], row["answer"])
            loss = forward_loss(md, text_mod, embeds, labels) / args.accum
            loss.backward()
            running += loss.item() * args.accum

            if i % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0:
                    print(f"epoch {epoch + 1} step {step} "
                          f"({i}/{len(rows)}) loss {running / (args.accum * 5):.4f} "
                          f"| {time.time() - started:.0f}s "
                          f"| {torch.cuda.max_memory_allocated() / 1e9:.1f}GB",
                          flush=True)
                    running = 0.0

    save_adapter(md, config)
    print(f"adapter saved to {OUT}  ({time.time() - started:.0f}s total)")


def save_adapter(md, config):
    from peft import get_peft_model_state_dict

    OUT.mkdir(parents=True, exist_ok=True)
    state = {k: v.to(torch.float16).cpu() for k, v in get_peft_model_state_dict(md.text).items()}
    torch.save(state, OUT / "adapter_model.bin")
    (OUT / "adapter_config.json").write_text(json.dumps({
        "r": config.r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "target_modules": list(config.target_modules),
        "base_model": "vikhyatk/moondream2",
        "trained_on": "BigEarthNet v2 Sentinel-2 land-cover captions",
    }, indent=2), encoding="utf-8")


def apply_adapter(md, adapter_dir=OUT, alpha=None):
    """Re-attach a saved adapter to a freshly loaded model. Used by vlm.py.

    `alpha` overrides the trained lora_alpha at load time. LoRA's contribution
    scales as alpha/r, so a lower alpha dials the adaptation down without
    retraining — the knob that trades CORINE vocabulary against the base model's
    general captioning, which two epochs of templated answers erodes.
    """
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

    meta = json.loads((Path(adapter_dir) / "adapter_config.json").read_text(encoding="utf-8"))
    meta["applied_alpha"] = alpha if alpha is not None else meta["lora_alpha"]
    config = LoraConfig(
        r=meta["r"], lora_alpha=meta["applied_alpha"], lora_dropout=0.0,
        bias="none", target_modules=meta["target_modules"],
    )
    inject_adapter_in_model(config, md.text)
    state = torch.load(Path(adapter_dir) / "adapter_model.bin", map_location="cpu")
    dtype = next(md.text.parameters()).dtype
    set_peft_model_state_dict(md.text, {k: v.to(dtype) for k, v in state.items()})
    md.text.eval()
    return meta


def smoke():
    """Runnable check for the training path: the loss must actually go down.

    Catches the two ways this silently breaks — a wrong `lm_head` (the
    generation one returns only the last position) and a detached graph from
    freezing the wrong parameters.
    """
    from PIL import Image as PILImage

    images = sorted((DATA / "images").glob("*.png"))
    if not images:
        sys.exit("No rendered patches. Run: python -m training.prepare_bigearthnet train 300")

    model, md, moondream_mod, text_mod = load_parts()
    config, trainable = attach_lora(md)
    assert trainable, "LoRA injected no trainable parameters"

    with PILImage.open(images[0]) as raw:
        image = raw.convert("RGB")
    question = "What land cover types are present in this image?"
    answer = "The land cover present is: arable land, broad-leaved forest and inland waters."

    optimizer = torch.optim.AdamW(trainable, lr=2e-4)
    losses = []
    for _ in range(4):
        embeds, labels = build_sample(md, moondream_mod, image, question, answer)
        assert embeds.size(1) > IMAGE_PREFIX_LEN, embeds.shape
        assert int((labels != -100).sum()) > 5, "answer tokens are not supervised"
        loss = forward_loss(md, text_mod, embeds, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"loss did not fall on a memorisation task: {losses}"
    print(f"lora_finetune smoke: loss {losses[0]:.3f} -> {losses[-1]:.3f} over 4 steps, "
          f"{torch.cuda.max_memory_allocated() / 1e9:.1f}GB peak")


if __name__ == "__main__":
    main()
