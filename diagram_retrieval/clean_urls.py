import argparse
from collections import defaultdict

import srsly
import textdistance as td

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
    parser.add_argument(
        "--n_sentences_per_image",
        help="Number of sentences per image. Default: 5",
        default=5,
        type=int,
    )
    args = parser.parse_args()
    print(args)
    return args


def main(
        input_path,
        output_path,
        n_sentences_per_image,
):
    dataset = srsly.read_jsonl(input_path)

    images_dict = defaultdict(list)
    images_with_rank = defaultdict(list)
    sentences_dict = defaultdict(list)
    for doc in dataset:
        if "images" in doc:
            for i, image in enumerate(doc["images"]):
                if not image["image"].endswith(".gif"):
                    if image["image"] not in sentences_dict[doc["claim"]]:
                        sentences_dict[doc["claim"]].append(image["image"])
                    if doc["claim"] not in images_dict[image["image"]]:
                        images_dict[image["image"]].append(doc["claim"])
                        images_with_rank[image["image"]].append(
                            {"claim": doc["claim"], "rank": i}
                        )

    images_to_delete = []
    for image, sentences in images_dict.items():
        if len(sentences) < n_sentences_per_image:
            images_to_delete.append(image)

    for image in tqdm(images_to_delete):
        del images_with_rank[image]
        del images_dict[image]

    filter_dataset = defaultdict(list)
    for rank in range(5):
        print("RANK", rank)
        for image, sentences in tqdm(images_with_rank.items()):
            ranked_sentences = [x["claim"] for x in sentences if x["rank"] == rank]
            for s1 in ranked_sentences:
                if (
                    len(filter_dataset[image]) < n_sentences_per_image
                    and s1 in sentences_dict
                ):
                    valid = True
                    other_imgs = [
                        x for x in sentences_dict[s1][rank + 1 :] if x in images_dict
                    ]
                    if len(images_dict[image]) > n_sentences_per_image:
                        for other_img in other_imgs:
                            if len(images_dict[other_img]) == n_sentences_per_image:
                                valid = False
                                break
                    if valid:
                        for s2 in filter_dataset[image]:
                            score = td.levenshtein.normalized_similarity(s1, s2)
                            if score > 0.75:
                                valid = False
                                break
                    if valid:
                        filter_dataset[image].append(s1)
                        for other_img in other_imgs:
                            images_dict[other_img].remove(s1)
                        images_dict[image].remove(s1)
                        del sentences_dict[s1]

    final_dataset = [
        {"image_url": image, "texts": sentences}
        for image, sentences in filter_dataset.items()
        if len(sentences) == n_sentences_per_image
    ]

    srsly.write_jsonl(output_path, final_dataset)


if __name__ == "__main__":
    main(**vars(parse_flags()))
