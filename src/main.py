import os
import json
import pandas as pd
from dateutil import parser
from datetime import datetime
from location_predictions import location as loc, data_processing
from time_normalizer import time_prediction as time
from data_preprocessing import (
    add_articles_to_df,
    filter_articles,
    filter_negative_articles,
)
from reddit import download_posts

MAIN_PATH = os.path.join(
    os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))), os.pardir
)


def get_config_interval():
    """
    Reads config file to extract appropriate
    date interval to extract articles from.

    Returns
    -------
    tuple
        (start_date, end_date)
    """
    with open(os.path.join(MAIN_PATH, "config.json")) as f:
        config = json.load(f)

    if config["interval"]["default"] == "yes":
        log = []
        with open(os.path.join(MAIN_PATH, "history.log")) as f:
            for line in f:
                log.append(parser.parse(line.strip().split("\t")[2]))
        if len(log) == 0:
            start_date = datetime(2019, 1, 1)
        else:
            log.sort()
            start_date = log[-1]

        end_date = datetime.now()

    else:
        if config["interval"]["start"]["date"]:
            start_date = parser.parse(config["interval"]["start"]["date"])
        else:
            start_date = datetime(2019, 1, 1)

        if config["interval"]["end"]["now"] == "yes":
            end_date = datetime.now()
        elif config["interval"]["end"]["date"]:
            end_date = parser.parse(config["interval"]["end"]["date"])
        else:
            end_date = datetime.now()

    with open(os.path.join(MAIN_PATH, "history.log"), "a+") as f:
        f.write("\t".join([str(datetime.now()), str(start_date), str(end_date)]) + "\n")

    return start_date, end_date


def main():
    filtered_reddit_articles_df = pd.read_csv('./data/output/article_sample.tsv', sep='\t')
    # start_date, end_date = get_config_interval()

    # reddit_df = download_posts(start_date, end_date)
    # reddit_articles_df = add_articles_to_df(reddit_df)
    # filtered_reddit_articles_df = filter_negative_articles(
    #     filter_articles(reddit_articles_df)
    # )

    filtered_reddit_articles_df["article_publish_date"] = filtered_reddit_articles_df[
        "article_publish_date"
    ].map(lambda x: str(parser.parse(str(x)) if x and x == x else x))

    clean_data = data_processing.prepare_date(filtered_reddit_articles_df)
    loc_results = loc.predict(clean_data)
    time_results = time.get_final_result(clean_data, filtered_reddit_articles_df)
    data_processing.get_final_result(
        filtered_reddit_articles_df, loc_results, time_results
    )


if __name__ == "__main__":
    main()
