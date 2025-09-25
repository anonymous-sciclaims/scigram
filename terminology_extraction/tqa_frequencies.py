import os
import json
import argparse

from collections import defaultdict

import srsly
from nltk import ngrams as get_ngrams
from tqdm import tqdm

from terminology_extraction.utils import NGRAM_TYPES, get_clean_term


def parse_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-t",
        "--tqa_path",
        required=True,
        help="tqa path",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="output path",
        type=str,
    )
    args = parser.parse_args()
    print(args)
    return args


def main(tqa_path: str, output_path: str):

    out_dict = {
        "unigrams": defaultdict(int),
        "bigrams": defaultdict(int),
        "trigrams": defaultdict(int),
    }

    json_files = [
        "train/tqa_v1_train.json",
        "val/tqa_v1_val.json",
        "test/tqa_v2_test.json",
    ]

    for json_file in json_files:
        json_path = os.path.join(tqa_path, json_file)
        dataset = srsly.read_json(json_path)
        for lesson in tqdm(dataset):
            topics = lesson["topics"]
            for topic_id in topics.keys():
                topic = topics[topic_id]
                text = topic["topicName"] + ". " + topic["content"]["text"]
                for n, ngram_label in NGRAM_TYPES.items():
                    for ngram in get_ngrams(text.split(), n):
                        term = get_clean_term(ngram=ngram)
                        out_dict[ngram_label][term] = out_dict[ngram_label][term] + 1

    with open(output_path, "w", errors="replace") as outfile:
        json.dump(out_dict, outfile, indent=4)


if __name__ == "__main__":
    main(**vars(parse_flags()))
