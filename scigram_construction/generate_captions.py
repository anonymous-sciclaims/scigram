import argparse

import srsly
from tqdm import tqdm
import random
from vllm import LLM, SamplingParams
from PIL import Image
import os
import numpy as np

PREFIXES = [
    "Can you provide a detailed caption explaining the key elements and purpose of this diagram?",
    "Could you describe this diagram in detail, including its components and how they relate to each other?",
    "What does this diagram illustrate, and can you provide an in-depth explanation of its parts?",
    "Please give a thorough caption for this diagram, covering all the important aspects and their significance.",
    "Can you break down the meaning of this diagram and provide a comprehensive caption?",
    "I need a detailed caption for this diagram—please include explanations of each section and their functions.",
    "Could you elaborate on this diagram with a caption that explains the concepts and relationships shown?",
    "What does this diagram represent? Please provide a caption that clearly defines each element.",
    "Can you create a caption that not only labels but also explains the purpose of this diagram?",
    "Please provide an insightful caption for this diagram, detailing its structure, purpose, and key takeaways."
    "Can you write a detailed caption that explains the meaning and significance of this diagram?"
    "Please provide an in-depth description of this diagram, covering all the key details."
    "Could you generate a comprehensive caption that explains the function of each part of this diagram?"
    "What does this diagram convey? Please summarize its purpose in a detailed caption."
    "Can you clarify the relationships between the elements in this diagram with an explanatory caption?"
    "I need a precise yet detailed caption that captures the essence of this diagram—can you help?"
    "Please describe this diagram thoroughly, ensuring the caption highlights all important aspects."
    "Could you write a caption that not only labels the diagram but also explains its overall meaning?"
    "What insights does this diagram offer? Please provide a caption that explains them clearly."
    "Can you craft a well-explained caption that makes this diagram easy to understand for others?"
]


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
        help="LLM's temperature. Default: 0.7",
        default=0.7,
        type=float,
    )
    parser.add_argument(
        "--top_p",
        help="LLM's top p. Default: 0.7",
        default=0.7,
        type=float,
    )
    parser.add_argument(
        "--max_tokens",
        help="LLM's max tokens. Default: 512",
        default=512,
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
    top_p,
    max_tokens,
    gpu_memory_utilization,
    seed,
):
    random.seed(seed)
    images = []
    for doc in tqdm(srsly.read_json(input_path)):
        images.append(doc["image"])
    images_sample = list(set(images))
    model = LLM(model=model_name, trust_remote_code=True,
                gpu_memory_utilization=gpu_memory_utilization)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    out_dataset = []
    cont = 1
    for i in tqdm(range(0, len(images_sample), batch_size)):
        prompts = []
        image_paths = []
        batch = images_sample[i:i + batch_size]
        for image_path in batch:
            image = Image.open(
                os.path.join(image_dir_path, image_path)).convert("RGB")
            h, w, c = np.shape(image)
            if h > 28 or w > 28:
                prompt = ('Provide a paragraph with a brief description of the '
                          'diagram. Pay special attention to the main '
                          'components of the diagram and the relations between '
                          'them. If visible, also reflect space and temporal '
                          'information, linking it to the components and '
                          'relations in the diagram.')
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
            for out in output.outputs:
                response = out.text
                caption = response
                prefix = random.choice(PREFIXES)
                conv1 = {'from': 'human', 'value': f"<image>\n {prefix}"}
                conv2 = {'from': 'gpt', 'value': caption}
                conversations = [conv1, conv2]
                out_dataset.append({"id": cont, "image": image_path,
                                    "conversations": conversations})
                cont += 1

    random.shuffle(out_dataset)
    srsly.write_json(output_path, out_dataset)


if __name__ == "__main__":
    main(**vars(parse_flags()))

