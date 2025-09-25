import json
import argparse

from typing import Dict, List, Tuple

import srsly

from nltk import ngrams as get_ngrams
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer

from terminology_extraction.utils import NGRAM_TYPES, clean_ngram


def parse_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-v",
        "--vocab_path",
        required=True,
        type=str,
        help="vocab file path",
    )
    parser.add_argument(
        "-s",
        "--sentence_path",
        required=True,
        type=str,
        help="sentence file path",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        type=str,
        help="output path",
    )
    args = parser.parse_args()
    print(args)
    return args


def get_valid_terms(
    lemmatizer: WordNetLemmatizer, words: List[str], vocab: Dict
) -> Tuple[List, List]:
    # Find lemmas and terms from the vocabulary among a list of words.
    lemmas = []
    terms = []
    for n, ngram_label in NGRAM_TYPES.items():
        ngrams = get_ngrams(words, n)
        for ngram in ngrams:
            lemma, term = clean_ngram(ngram=ngram, lemmatizer=lemmatizer)
            if lemma in vocab[ngram_label].keys():
                if lemma not in lemmas:
                    lemmas.append(lemma)
                if term not in terms:
                    terms.append(term)
    return lemmas, terms


def main(vocab_path: str, sentence_path: str, output_path: str):
    vocab = srsly.read_json(vocab_path)

    with open(sentence_path, "r", encoding="utf-8", errors="surrogatepass") as f:
        lines = f.readlines()

    lemmatizer = WordNetLemmatizer()

    with open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(lines):
            json_line = json.loads(line)
            sentence = json_line["sentence"]
            split = json_line["split"]
            words = sentence.split()
            lemmas, terms = get_valid_terms(
                lemmatizer=lemmatizer, words=words, vocab=vocab
            )
            if lemmas and terms:
                out_json_line = {
                    "sentence": sentence,
                    "lemmas": sorted(lemmas),
                    "terms": sorted(terms),
                    "split": split,
                }
                fout.write(json.dumps(out_json_line) + "\n")


if __name__ == "__main__":
    main(**vars(parse_flags()))
