import argparse
import random

import srsly

from scigram_construction.utils import process_ai2d, process_scienceqa, \
    process_tqa, process_extras


def parse_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--ai2d_path",
        help="Path to the training file of AI2D.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--scienceqa_path",
        help="Path to the training file of ScienceQA.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--tqa_path",
        help="Path to the training file of TQA.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--output_path",
        help="Output path",
        required=True,
        type=str,
    )
    args = parser.parse_args()
    print(args)
    return args


def main(
        ai2d_path,
        scienceqa_path,
        tqa_path,
        output_path,
):
    random.seed(42)

    ai2d_dataset = process_ai2d(input_path=ai2d_path)
    print(len(ai2d_dataset))
    scienceqa_dataset = process_scienceqa(input_path=scienceqa_path)
    print(len(scienceqa_dataset))
    tqa_dataset = process_tqa(input_path=tqa_path)
    print(len(tqa_dataset))
    extras_dataset = process_extras()
    print(len(extras_dataset))

    out_dataset = ai2d_dataset + scienceqa_dataset + tqa_dataset + extras_dataset
    random.shuffle(out_dataset)
    print(len(out_dataset))
    srsly.write_json(output_path, out_dataset)


if __name__ == "__main__":
    main(**vars(parse_flags()))
