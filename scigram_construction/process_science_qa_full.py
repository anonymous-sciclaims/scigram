import argparse
import os
from pathlib import Path

import srsly
from datasets import load_dataset
from tqdm import tqdm


def parse_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--scienceqa_path",
        help="Path to save ScienceQA. Default: data/scienceqa",
        default="data/scienceqa",
        type=str,
    )
    args = parser.parse_args()
    print(args)
    return args


def main(
    scienceqa_path: str,
):
    CHOICE_LETTERS = ["a", "b", "c", "d", "e"]
    if not Path(scienceqa_path).exists():
        os.mkdir(scienceqa_path)

    ds = load_dataset("derek-thomas/ScienceQA")
    question_elems = []
    for split in ds:
        split_path = os.path.join(scienceqa_path, split)
        if not Path(split_path).exists():
            os.mkdir(split_path)
        for q_id, ex in tqdm(enumerate(ds[split])):
            question = ex["question"]
            answers = {CHOICE_LETTERS[i]: choice for i, choice in enumerate(ex["choices"])}
            lesson = ex["lecture"]
            out_doc = {
                    "id": q_id,
                    "question": question,
                    "subject": ex["subject"],
                    "grade": ex["grade"],
                    "answers": answers,
                    "correct_answer": CHOICE_LETTERS[ex["answer"]],
                    "lesson": lesson,
                }
            if ex["image"]:
                image_path = os.path.join(split_path, split + "_" + str(q_id) + ".png")
                if not Path(image_path).exists():
                    ex["image"].save(image_path, "PNG")
                out_doc["diagram"] = f"{split}_{str(q_id)}.png"
            if ex["hint"]:
                out_doc["context"] = ex["hint"]
            question_elems.append(out_doc)
        srsly.write_jsonl(os.path.join(scienceqa_path, f"{split}_full.jsonl"), question_elems)


if __name__ == "__main__":
    main(**vars(parse_flags()))