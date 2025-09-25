import argparse
import random

import srsly
import huggingface_hub

from tqdm import tqdm
from transformers import AutoTokenizer

from scientific_claim_gen.utils import get_combinations


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
        help="Path to the file with sentences and selected terminology",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--output_path",
        help="output path",
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
        "--n_atomic_facts",
        help="Number of atomic facts per combination. Default: 50",
        default=50,
        type=int,
    )
    parser.add_argument(
        "--max_tokens",
        help="Max tokens per claim. Default: 12",
        default=12,
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
    output_path: str,
    model_name: str,
    n_atomic_facts: int,
    max_tokens: int,
    seed: int,
):
    random.seed(seed)
    huggingface_hub.login(token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = [x for x in srsly.read_jsonl(input_path)]
    n_source_sentences = len(dataset)
    print(len(dataset))
    if n_source_sentences is not None:
        dataset = random.choices(dataset, k=n_source_sentences)

    term_lists = []
    for doc in tqdm(dataset):
        combinations = get_combinations(doc["lemmas"])
        for term_list in combinations:
            term_list = set(term_list)
            if term_list not in term_lists:
                term_lists.append(term_list)

    prompts = []
    for term_list in tqdm(term_lists):
        system_prompt = f"""
        Your task is to generate a list of {n_atomic_facts} atomic facts that contain different combinations of the given terms.

        Please follow this criteria:
        - Limit yourself to the scientific domains of life sciences, earth sciences, and physical sciences at the middle school level.
        - Pay attention to commonsense.
        - Make the facts brief and concise yet easy to understand and meaningful.
        - Use different grammatical constructions and limit length to {max_tokens} tokens or less.

        Arrange your output as a jsonl file where each line is {{"claim": <fact>}}.
        """  # noqa: E501
        user_prompt = f"""
        TERMS: {term_list}.

        OUTPUT: {{"output": "<INSERT OUTPUT HERE>"}}
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if prompt not in prompts:
            prompts.append(prompt)
    srsly.write_jsonl(output_path, prompts)
    print(len(prompts))


if __name__ == "__main__":
    main(**vars(parse_flags()))
