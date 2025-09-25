import os
import time
from pathlib import Path

import srsly
from duckduckgo_search import DDGS


def add_urls(cache_dir, wait_time, restart_timer, n_processed_queries, doc):
    cache_file = os.path.join(cache_dir, str(doc["id"]) + ".jsonl")
    if not Path(cache_file).exists():
        try:
            ddgs = DDGS()
            query = doc["claim"][:-1] + " diagram"
            results = ddgs.images(
                keywords=query,
                region="wt-wt",
                safesearch="off",
                size=None,
                color=None,
                type_image=None,
                layout=None,
                license_image=None,
                max_results=5,
            )
            srsly.write_jsonl(cache_file, results)
            time.sleep(wait_time)
        except (ValueError, Exception) as e:
            new_n_processed_queries = len(os.listdir(cache_dir))
            print(
                f"Error {e}. Restarting in {restart_timer} minutes. "
                f"{new_n_processed_queries - n_processed_queries} queries processed "
                f"during this loop. {new_n_processed_queries} queries processed "
                f"in total."
            )
            time.sleep(restart_timer * 60)
            ddgs.__exit__()
            add_urls(
                doc=doc,
                cache_dir=cache_dir,
                restart_timer=restart_timer + 10,
                wait_time=wait_time,
                n_processed_queries=new_n_processed_queries,
            )
        ddgs.__exit__()