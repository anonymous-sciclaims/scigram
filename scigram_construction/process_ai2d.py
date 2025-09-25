import argparse
import hashlib

import srsly
from datasets import load_dataset
from tqdm import tqdm

CHOICE_LETTERS = ["a", "b", "c", "d"]


def parse_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--ai2d_path",
        help="Path to save AI2D. Default: data/ai2d-all",
        default="data/ai2d-all",
        type=str,
    )
    args = parser.parse_args()
    print(args)
    return args


def main(
    ai2d_path: str,
):
    ds = load_dataset("lmms-lab/LLaVA-OneVision-Data", "ai2d(internvl)", split="train")
    question_elems = []
    images = []
    cont = 0
    for ex in tqdm(ds):
        image = ex["image"]
        image_id = ex["id"]
        image_path = f"{ai2d_path}/images_fix/{image_id}"
        if image_id not in images:
            doc = srsly.read_json(f"{ai2d_path}/questions/{image_id}.json")
            for question, q_dict in doc["questions"].items():
                options = {CHOICE_LETTERS[i]: choice for i, choice in enumerate(q_dict["answerTexts"])}
                correct_answer = CHOICE_LETTERS[q_dict["correctAnswer"]]
                image.save(image_path)
                question_elems.append(
                    {
                        "id": cont,
                        "question": question,
                        "answers": options,
                        "diagram": image_id,
                        "correct_answer": correct_answer,
                    }
                )
                images.append(image_id)
                cont+=1
    srsly.write_jsonl(f"{ai2d_path}/train_fix.jsonl", question_elems)

    ds = load_dataset("lmms-lab/ai2d", split="test")
    question_elems = []
    images = {}
    cont = 0
    for q_id, ex in tqdm(enumerate(ds), total=len(ds)):
        image = ex["image"]
        im_hash = hashlib.md5(image.tobytes()).hexdigest()
        question = ex["question"]
        options = ex["options"]
        correct_answer = CHOICE_LETTERS[int(ex["answer"])]
        if im_hash not in images:
            image_id = f"{cont}_test.png"
            image_path = f"{ai2d_path}/images_fix/{image_id}"
            image.save(image_path)
            images[im_hash] = image_id
            cont += 1
        else:
            image_id = images[im_hash]
        question_elems.append(
            {
                "id": q_id,
                "question": question,
                "answers": options,
                "diagram": image_id,
                "correct_answer": correct_answer,
            }
        )
    srsly.write_jsonl(f"{ai2d_path}/test_fix.jsonl", question_elems)

    ds = load_dataset("lmms-lab/ai2d-no-mask", split="test")
    question_elems = []
    images = {}
    cont = 0
    for q_id, ex in tqdm(enumerate(ds), total=len(ds)):
        image = ex["image"]
        im_hash = hashlib.md5(image.tobytes()).hexdigest()
        question = ex["question"]
        options = ex["options"]
        correct_answer = CHOICE_LETTERS[int(ex["answer"])]
        if im_hash not in images:
            image_id = f"{cont}_test_unmasked.png"
            image_path = f"{ai2d_path}/images_fix/{image_id}"
            image.save(image_path)
            images[im_hash] = image_id
            cont += 1
        else:
            image_id = images[im_hash]
        question_elems.append(
            {
                "id": q_id,
                "question": question,
                "answers": options,
                "diagram": image_id,
                "correct_answer": correct_answer,
            }
        )
    srsly.write_jsonl(f"{ai2d_path}/test_fix_unmasked.jsonl", question_elems)


if __name__ == "__main__":
    main(**vars(parse_flags()))
