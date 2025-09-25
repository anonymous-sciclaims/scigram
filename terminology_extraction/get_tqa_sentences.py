import json
import argparse

import srsly

from tqdm import tqdm


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
    dataset = srsly.read_json(metadata_path)

    indexed_sentences = []
    with open(output_path, "w", encoding="utf-8") as fout:
        for topic in tqdm(dataset):
            terms = topic["terms"]
            split = topic["split"]
            for lemma, lemma_info in terms.items():
                for sentence in lemma_info["sentences"]:
                    if sentence not in indexed_sentences:
                        json_line = {"sentence": sentence, "split": split}
                        fout.write(json.dumps(json_line) + "\n")
                        indexed_sentences.append(sentence)


if __name__ == "__main__":
    main(**vars(parse_flags()))
