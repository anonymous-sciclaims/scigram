import json
import argparse

from collections import defaultdict

import srsly

from terminology_extraction.utils import NGRAM_TYPES


def parse_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-m",
        "--metadata_path",
        help="path to tqa_metadata.json",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="output path",
        required=True,
        type=str,
    )
    args = parser.parse_args()
    print(args)
    return args


def main(
    metadata_path: str,
    output_path: str,
):
    vocab = {ngram_label: defaultdict(list) for ngram_label in NGRAM_TYPES.values()}

    dataset = srsly.read_json(metadata_path)

    for topic in dataset:
        terms = topic["terms"]
        split = topic["split"]
        for lemma in terms.keys():
            n = len(lemma.split())
            ngram_label = NGRAM_TYPES[n]
            if split not in vocab[ngram_label][lemma]:
                vocab[ngram_label][lemma].append(split)

    with open(output_path, "w", errors="replace") as outfile:
        json.dump(vocab, outfile, indent=4)


if __name__ == "__main__":
    main(**vars(parse_flags()))
