import json
import argparse
import torch
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import opencc
from pypinyin import pinyin, Style


# ---------------------------------------------------------------------------
# NLLB setup
# ---------------------------------------------------------------------------

def setup_nllb_model(model_name, src_lang="eng_Latn"):
    """Load NLLB model for seq2seq translation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "device": device}


def translate_batch_nllb(texts, nllb_dict, tgt_lang="zho_Hans", max_length=256):
    """Translate a batch of texts with NLLB."""
    model = nllb_dict["model"]
    tokenizer = nllb_dict["tokenizer"]
    device = nllb_dict["device"]

    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_new_tokens=max_length,
        )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Script conversion
# ---------------------------------------------------------------------------

def to_traditional(text, converter):
    return converter.convert(text)


def to_pinyin_str(text):
    result = pinyin(text, style=Style.TONE, heteronym=False)
    return " ".join(tok[0] for tok in result)


# ---------------------------------------------------------------------------
# Data selection
# ---------------------------------------------------------------------------

def select_representative_examples(dataset, n=100):
    """Pick n evenly-spaced examples for representative coverage."""
    total = len(dataset)
    if total <= n:
        return dataset
    step = total / n
    indices = [int(i * step) for i in range(n)]
    return dataset.select(indices)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Translate AdvBench prompts EN→ZH with NLLB (simplified, traditional, pinyin)"
    )
    parser.add_argument(
        "--nllb_model",
        default="facebook/nllb-200-distilled-600M",
        help="NLLB model name",
    )
    parser.add_argument(
        "--num_examples", type=int, default=100, help="Number of examples to translate"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="NLLB batch size")
    parser.add_argument(
        "--output_dir",
        default="./advbench_translated/",
        help="Output directory for results",
    )
    parser.add_argument("--src_lang", default="eng_Latn", help="NLLB source language code")
    parser.add_argument("--tgt_lang", default="zho_Hans", help="NLLB target language code")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("Loading AdvBench dataset...")
    ds = load_dataset("walledai/AdvBench", split="train")
    print(f"Total examples: {len(ds)}")

    subset = select_representative_examples(ds, args.num_examples)
    print(f"Selected {len(subset)} representative examples")

    # ------------------------------------------------------------------
    # 2. Load NLLB
    # ------------------------------------------------------------------
    print(f"Loading NLLB model: {args.nllb_model} ...")
    nllb = setup_nllb_model(args.nllb_model, src_lang=args.src_lang)
    print(f"Using device: {nllb['device']}")

    # ------------------------------------------------------------------
    # 3. Setup script converters
    # ------------------------------------------------------------------
    s2t_converter = opencc.OpenCC("s2t")

    # ------------------------------------------------------------------
    # 4. Translate + convert
    # ------------------------------------------------------------------
    prompts = subset["prompt"]
    results = []

    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Translating (NLLB)"):
        batch = prompts[i : i + args.batch_size]
        simplified_batch = translate_batch_nllb(batch, nllb, tgt_lang=args.tgt_lang)

        for j, (eng, simplified) in enumerate(zip(batch, simplified_batch)):
            traditional = to_traditional(simplified, s2t_converter)
            pinyin_text = to_pinyin_str(simplified)
            results.append(
                {
                    "id": i + j,
                    "english": eng,
                    "chinese_simplified": simplified,
                    "chinese_traditional": traditional,
                    "chinese_pinyin": pinyin_text,
                }
            )

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    output_file = os.path.join(args.output_dir, f"advbench_nllb_{args.num_examples}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Saved {len(results)} translated examples to {output_file}")

    for r in results[:3]:
        print(f"\n[{r['id']}] EN:   {r['english'][:80]}")
        print(f"     ZH-S: {r['chinese_simplified']}")
        print(f"     ZH-T: {r['chinese_traditional']}")
        print(f"     PY:   {r['chinese_pinyin'][:80]}")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
