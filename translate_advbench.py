import json
import argparse
import os
import time
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
import opencc


# ---------------------------------------------------------------------------
# GPT-based translation
# ---------------------------------------------------------------------------

SIMPLIFIED_SYSTEM_PROMPT = (
    "You are a translator. Translate the following English text into "
    "Simplified Chinese. Output ONLY the translation, nothing else."
)

PINYIN_SYSTEM_PROMPT = (
    "You are a translator. Translate the following English text into Chinese "
    "Pinyin (with tone marks). Output ONLY the pinyin, nothing else. "
    "Do not include Chinese characters, explanations, or punctuation other "
    "than what appears in standard pinyin."
)


def translate_with_gpt(
    text: str,
    client: OpenAI,
    system_prompt: str,
    model: str = "gpt-4.1-mini",
    max_retries: int = 3,
) -> str:
    """Translate text using an OpenAI model with a given system prompt."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  GPT translation attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                return ""
    return ""


# ---------------------------------------------------------------------------
# Script conversion
# ---------------------------------------------------------------------------

def to_traditional(text, converter):
    return converter.convert(text)


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
        description="Translate AdvBench prompts EN→ZH with GPT (simplified, traditional, pinyin)"
    )
    parser.add_argument(
        "--num_examples", type=int, default=100, help="Number of examples to translate"
    )
    parser.add_argument(
        "--output_dir",
        default="./advbench_translated/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--openai_api_key",
        default=None,
        help="OpenAI API key (falls back to OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--openai_base_url",
        default=None,
        help="Custom base URL for the OpenAI-compatible API",
    )
    parser.add_argument(
        "--translation_model",
        default="gpt-4.1",
        help="OpenAI model to use for translation (default: gpt-4.1)",
    )

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
    # 2. Setup OpenAI client
    # ------------------------------------------------------------------
    openai_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: OpenAI API key is required. Use --openai_api_key or set OPENAI_API_KEY.")
        return

    client_kwargs = {"api_key": openai_key}
    if args.openai_base_url:
        client_kwargs["base_url"] = args.openai_base_url
    client = OpenAI(**client_kwargs)
    print(f"Using GPT model '{args.translation_model}' for all translations")

    # ------------------------------------------------------------------
    # 3. Setup script converters
    # ------------------------------------------------------------------
    s2t_converter = opencc.OpenCC("s2t")

    # ------------------------------------------------------------------
    # 4. Translate + convert
    # ------------------------------------------------------------------
    prompts = subset["prompt"]
    results = []

    for i, eng in enumerate(tqdm(prompts, desc="Translating (GPT)")):
        simplified = translate_with_gpt(
            eng, client, SIMPLIFIED_SYSTEM_PROMPT, model=args.translation_model
        )
        traditional = to_traditional(simplified, s2t_converter)
        pinyin_text = translate_with_gpt(
            eng, client, PINYIN_SYSTEM_PROMPT, model=args.translation_model
        )

        results.append(
            {
                "id": i,
                "english": eng,
                "chinese_simplified": simplified,
                "chinese_traditional": traditional,
                "chinese_pinyin": pinyin_text,
            }
        )

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    output_file = os.path.join(args.output_dir, f"advbench_gpt_{args.num_examples}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Saved {len(results)} translated examples to {output_file}")

    for r in results[:3]:
        print(f"\n[{r['id']}] EN:   {r['english'][:80]}")
        print(f"     ZH-S: {r['chinese_simplified']}")
        print(f"     ZH-T: {r['chinese_traditional']}")
        print(f"     PY:   {r['chinese_pinyin'][:80]}")


if __name__ == "__main__":
    main()
