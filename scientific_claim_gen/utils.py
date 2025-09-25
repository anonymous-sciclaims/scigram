import itertools
from typing import List

from transformers import AutoModel, AutoTokenizer


def get_combinations(lemmas):
    res = []
    for L in range(len(lemmas) + 1):
        for subset in itertools.combinations(lemmas, L):
            sublist = [x for x in subset]
            if len(sublist) > 0:
                res.append(sublist)
    return res


def llama_request(prompts: List[str], model: AutoModel, tokenizer: AutoTokenizer):
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    generate_ids = model.generate(
        inputs.input_ids.to("cuda"),
        attention_mask=inputs.attention_mask.to("cuda"),
        max_new_tokens=1500,
    )
    response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    claims_batch = [
        x.replace(prompt, "").split("\n") for x, prompt in zip(response, prompts)
    ]
    return claims_batch, response