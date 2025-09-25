import argparse
import os

from pathlib import Path
from functools import partial

import srsly

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from diagram_retrieval.utils import add_urls


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
        "--restart_timer",
        help="Time in seconds to restart the query after failing. Default: 10.",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--wait_time",
        help="Time in seconds between requests. Default: 1.5",
        default=1.5,
        type=float,
    )
    parser.add_argument(
        "--num_workers",
        help="Number of simultaneous workers. Default: 4",
        default=4,
        type=int,
    )
    args = parser.parse_args()
    print(args)
    return args


def main(
        input_path,
        output_path,
        restart_timer,
        wait_time,
        n_workers,
):
    dataset = [x for x in srsly.read_jsonl(input_path)]

    cache_dir = "data/.cache/"
    if not Path(cache_dir).exists():
        os.mkdir(cache_dir)

    n_processed_queries = len(os.listdir(cache_dir))
    process_map(
        partial(add_urls, cache_dir, wait_time, restart_timer, n_processed_queries),
        dataset,
        max_workers=n_workers,
    )
    out_dataset = []
    for doc in tqdm(dataset):
        out_doc = {k: v for k, v in doc.items()}
        cache_file = os.path.join(cache_dir, str(doc["id"]) + ".jsonl")
        out_doc["images"] = [x for x in srsly.read_jsonl(cache_file)]
        out_dataset.append(out_doc)
    srsly.write_jsonl(output_path, out_dataset)


if __name__ == "__main__":
    main(**vars(parse_flags()))
