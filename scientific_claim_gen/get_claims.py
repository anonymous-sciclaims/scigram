import argparse
import os
import re

from json import JSONDecodeError
from pathlib import Path
from random import random

import srsly
import torch

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)

from scientific_claim_gen.utils import llama_request


def parse_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--hf_token",
        help="Your huggingface token. Required to run llama3",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--input_path",
        help="Path to the file with the claim prompts",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="output dir path",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--model_name",
        help="model name. Default: meta-llama/Meta-Llama-3-8B-Instruct",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        type=str,
    )
    parser.add_argument(
        "--batch_size",
        help="Batch size. Default: 8",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--seed",
        help="Random seed. Default: 42",
        default=42,
        type=int,
    )
    args = parser.parse_args()
    print(args)
    return args


def main(
    hf_token: str,
    input_path: str,
    output_dir: str,
    model_name: str,
    batch_size: int,
    seed: int,
):
    random.seed(seed)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    dataset = [x for x in srsly.read_jsonl(os.path.join(input_path))]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=hf_token, quantization_config=bnb_config, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=hf_token, quantization_config=bnb_config
    )
    if not Path(os.path.join(output_dir, "claims")).exists():
        os.mkdir(os.path.join(output_dir, "claims"))
    prompts = [dataset[x : x + batch_size] for x in range(0, len(dataset), batch_size)]
    for i, prompt_batch in tqdm(enumerate(prompts), total=len(prompts)):
        try:
            claims_batch, response = llama_request(
                prompts=prompt_batch, model=model, tokenizer=tokenizer
            )
            res_claims = []
            for claim_batch, prompt in zip(claims_batch, prompts):
                for claim in claim_batch:
                    if re.match("([1-9]|[1-9][0-9]|100). ", claim) is not None:
                        clean_claim = claim.split(". ")[1]
                    else:
                        clean_claim = claim
                    res_claims.append(
                        {"claim": clean_claim, "prompt": prompt, "metadata": response}
                    )
            srsly.write_jsonl(
                os.path.join(output_dir, "claims", "claims_part" + str(i) + ".jsonl"),
                res_claims,
            )
        except JSONDecodeError:
            print("Error with prompt {" + prompt + "}")


if __name__ == "__main__":
    main(**vars(parse_flags()))
