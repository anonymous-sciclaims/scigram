import json
import argparse

import srsly

from nltk.stem import WordNetLemmatizer
from flair.models import SequenceTagger
from transformers import AutoModel, AutoTokenizer

from terminology_extraction.utils import load_tqa_with_splits, \
    generate_terminology_wise_topics


def parse_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t", "--tqa_path", help="tqa folder", required=True, type=str)
    parser.add_argument(
        "-o",
        "--output_path",
        help="output path",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-b",
        "--bnc_freqs_path",
        help="bnc frequencies path",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-f",
        "--tqa_freqs_path",
        help="tqa frequencies path",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-s",
        "--stopwords_path",
        help="stopwords path",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-w",
        "--weirdness_index_threshold",
        default=2.0,
        help="weirdness index threshold. Default: 2.0",
        type=float,
    )
    parser.add_argument(
        "-c",
        "--chunker_name",
        default="flair/chunk-english-fast",
        help="chunker. Default: flair/chunk-english-fast",
        type=str,
    )
    parser.add_argument(
        "-g",
        "--tagger_name",
        default="flair/pos-english-fast",
        help="pos tagger. Default: flair/pos-english-fast",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        default="roberta-base",
        help="transformer model. Default: roberta-base",
        type=str,
    )
    args = parser.parse_args()
    print(args)
    return args


def main(
    tqa_path: str,
    output_path: str,
    bnc_freqs_path: str,
    tqa_freqs_path: str,
    stopwords_path: str,
    weirdness_index_threshold: float,
    chunker_name: str,
    tagger_name: str,
    model_name_or_path: str,
):
    json_files = [
        "train/tqa_v1_train.json",
        "val/tqa_v1_val.json",
        "test/tqa_v2_test.json",
    ]

    dataset = load_tqa_with_splits(root_path=tqa_path, json_files=json_files)

    bnc_freqs = srsly.read_json(bnc_freqs_path)
    tqa_freqs = srsly.read_json(tqa_freqs_path)

    stopwords = srsly.read_json(stopwords_path)

    chunker = SequenceTagger.load(chunker_name)
    tagger = SequenceTagger.load(tagger_name)
    lemmatizer = WordNetLemmatizer()

    model = AutoModel.from_pretrained(model_name_or_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    out_topics = generate_terminology_wise_topics(
        dataset=dataset,
        chunker=chunker,
        tagger=tagger,
        model=model,
        tokenizer=tokenizer,
        lemmatizer=lemmatizer,
        bnc_freqs=bnc_freqs,
        tqa_freqs=tqa_freqs,
        weirdness_index_threshold=weirdness_index_threshold,
        stopwords=stopwords,
    )

    with open(output_path, "w", errors="replace") as outfile:
        json.dump(out_topics, outfile, indent=4)


if __name__ == "__main__":
    main(**vars(parse_flags()))
