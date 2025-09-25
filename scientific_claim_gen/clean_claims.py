import argparse
import os
from collections import defaultdict

import srsly

from tqdm import tqdm


def parse_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--input_path",
        help="Input path",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--output_path",
        help="Output path",
        required=True,
        type=str,
    )
    args = parser.parse_args()
    print(args)
    return args


def main(
    input_path: str,
    output_path: str,
):
    dataset = [x for x in srsly.read_jsonl(input_path)]

    clean_dataset = []
    aux_dict = defaultdict(list)
    for i, doc in tqdm(enumerate(dataset), total=len(dataset)):
        doc = dataset[i]
        if type(doc["claim"]) is dict:
            doc = doc["claim"]
        clean_claim = doc["claim"].split(":")[-1]
        aux_dict[clean_claim].append(i)

    first_indexes = []
    for k, v in tqdm(aux_dict.items()):
        first_indexes.append(v[0])

    for i in tqdm(first_indexes):
        doc = dataset[i]
        if type(doc["claim"]) is dict:
            doc = doc["claim"]
        clean_claim = doc["claim"].split(":")[-1]
        doc["claim"] = clean_claim
        clean_dataset.append(doc)

    srsly.write_jsonl(output_path, clean_dataset)


if __name__ == "__main__":
    main(**vars(parse_flags()))
