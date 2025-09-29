import argparse

import srsly
from tqdm import tqdm
import random
from vllm import LLM, SamplingParams
import json_repair
from PIL import Image
import os
import numpy as np
from collections import Counter

from scigram_construction.utils import answer_randomizer


def parse_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--image_dir_path",
        help="Path to the data dir where all the images are.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--input_path",
        help="Path to the scigram base json with the images and the queries",
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
        "--model_name",
        help="Name of the model used for generation. Default: Qwen/Qwen2-VL-7B-Instruct",
        default="Qwen/Qwen2-VL-7B-Instruct",
        type=str,
    )
    parser.add_argument(
        "--batch_size",
        help="Batch Size. Default: 1000",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--temperature",
        help="LLM's temperature. Default: 0",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--max_tokens",
        help="LLM's max tokens. Default: 1024",
        default=1024,
        type=int,
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        help="Percentage of GPU used with VLLM. Default: 0.8",
        default=0.8,
        type=float,
    )
    parser.add_argument(
        "--seed",
        help="Random Seed. Default: 42",
        default=42,
        type=int,
    )
    args = parser.parse_args()
    print(args)
    return args


def main(
    image_dir_path,
    input_path,
    output_path,
    model_name,
    batch_size,
    temperature,
    max_tokens,
    gpu_memory_utilization,
    seed,
):
    random.seed(seed)
    images = []
    for doc in tqdm([x for x in srsly.read_json(input_path)]):
        images.append(doc["image"])
    images_counter = Counter(images)
    images_sample = list(set(images))

    model = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    out_dataset = []
    cont = 1
    for i in tqdm(range(0, len(images_sample), batch_size)):
        prompts = []
        image_paths = []
        batch = images_sample[i : i + batch_size]
        for image_path in batch:
            image = Image.open(os.path.join(image_dir_path, image_path)).convert("RGB")
            h, w, c = np.shape(image)
            if h > 28 or w > 28:
                prompt = (f"Formulate {images_counter[image_path] * 2} multiple "
                          f"choice questions with 4 possible answers grounded "
                          f"in the diagram. The resulting questions must be "
                          f"middle school level questions in the subjects of "
                          f"life sciences, earth sciences or physical sciences. "
                          f"The questions should be answered using the elements "
                          f"from the image. For your output, follow this "
                          f"structure:")
                prompt += ('[{"question": <question>, "answers": {"a": '
                           '<answer a>, "b": <answer b>, "c": <answer c>, '
                           '"d": < answer d>}, "correct_answer": '
                           '<correct_letter>}, ...].')
                prompt_dict = {
                    "prompt": f"<|im_start|>user\n<|vision_start|><|image_pad|>"
                              f"<|vision_end|>{prompt}<|im_end|>\n<|im_start|>"
                              f"assistant\n",
                    "multi_modal_data": {"image": image},
                }
                prompts.append(prompt_dict)
                image_paths.append(image_path)
        outputs = model.generate(prompts, sampling_params)
        for output, image_path in zip(outputs, image_paths):
            response = output.outputs[0].text
            try:
                mcqa_doc = json_repair.loads(response)
                for q in mcqa_doc:
                    if "question" in q and "correct_answer" in q and "answers" in q:
                        question = q["question"]
                        correct_answer = q["correct_answer"]
                        answers = q["answers"]
                        if correct_answer in answers:
                            answers, correct_answer = answer_randomizer(answers, correct_answer)
                            answers_text = "Answer choices:\n"
                            for k, v in answers.items():
                                answers_text = answers_text + f"{k}) {v}\n"
                            q_prompt = (f'<image>\nTake a look at the diagram '
                                        f'and answer the following question by '
                                        f'choosing one of the possible answers.'
                                        f'\nQuestion: "{question}" \n'
                                        f'{answers_text}')
                            q_prompt = (
                                q_prompt
                                + '\nArrange your output as json such as '
                                  '{"answer": "<your choice>"}. Your choice '
                                  'must be the associated letter to your '
                                  'answer.'
                            )
                            conv1 = {"from": "human", "value": q_prompt}
                            conv2 = {
                                "from": "gpt",
                                "value": '{"answer": "' + correct_answer + '"}',
                            }
                            conversations = [conv1, conv2]
                            out_dataset.append(
                                {
                                    "id": cont,
                                    "image": image_path,
                                    "conversations": conversations,
                                }
                            )
                            cont += 1
            except (RecursionError, TypeError):
                print("Error with model output")

    random.shuffle(out_dataset)
    srsly.write_json(output_path, out_dataset)


if __name__ == "__main__":
    main(**vars(parse_flags()))
