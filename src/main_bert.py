import os
import io
import json
import torch
import pickle
import geocoder
import pandas as pd
from bert_utils import *
from dateutil import parser
from datetime import datetime
from joblib import Parallel, delayed
from location_predictions import location as loc, data_processing
from time_normalizer.time_main import time_date_normalization
from data_preprocessing import (
    add_articles_to_df,
    filter_articles,
    filter_negative_articles,
)
from reddit import download_posts

MAIN_PATH = os.path.join(
    os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))), os.pardir
)
MODEL_PATH = os.path.join(MAIN_PATH, "models")
DATA_PATH = os.path.join(MAIN_PATH, "data")
BATCH_SIZE = 16


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def get_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def get_lat_lng_radius(location_name):
    lat, lng, radius = None, None, None
    geocoded = geocoder.arcgis(location_name).json
    if geocoded:
        geoloc = geocoded["bbox"]
        lat, lng = geocoded["lat"], geocoded["lng"]
        ne, sw = geoloc["northeast"], geoloc["southwest"]
        radius = loc.get_radius(ne, sw)

    return (lat, lng, radius)


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
    # filtered_reddit_articles_df = pd.read_csv(
    #     "./data/output/article_sample.tsv", sep="\t"
    # )
    start_date, end_date = get_config_interval()

    reddit_df = download_posts(start_date, end_date)
    reddit_articles_df = add_articles_to_df(reddit_df)
    filtered_reddit_articles_df = filter_negative_articles(
        filter_articles(reddit_articles_df)
    )

    filtered_reddit_articles_df = filtered_reddit_articles_df.dropna(
        subset=["article_publish_date"]
    )
    filtered_reddit_articles_df["article_publish_date"] = filtered_reddit_articles_df[
        "article_publish_date"
    ].astype(str)

    with open(
        os.path.join(MODEL_PATH, "landslide_detection-QA-2-epoch_2-40.model"), "rb"
    ) as f:
        model = CPU_Unpickler(f).load()

    text = filtered_reddit_articles_df["article_text"].to_numpy().tolist()
    publish_dates = (
        filtered_reddit_articles_df["article_publish_date"].to_numpy().tolist()
    )

    preds_cats = []
    preds_trigs = []
    preds_spans = []

    with torch.no_grad():
        for i in range(len(SPAN_ID2L)):
            preds_spans.append([])
        for batch in tqdm(
            get_batch(text, BATCH_SIZE), total=round(len(text) / BATCH_SIZE)
        ):
            cats, trigs, spans = predict(model, batch)
            preds_cats.extend(cats)
            preds_trigs.extend(trigs)
            for i, span in enumerate(spans):
                preds_spans[i].extend(span)

    predicted_locations = preds_spans[SPAN_L2ID["LOC"]]
    predicted_times = preds_spans[SPAN_L2ID["TIME"]]
    predicted_dates = preds_spans[SPAN_L2ID["DATE"]]
    predicted_cas = preds_spans[SPAN_L2ID["CAS"]]

    predicted_geos = Parallel(n_jobs=-1, verbose=1)(
        delayed(get_lat_lng_radius)(location_name)
        for location_name in predicted_locations
    )
    lats = [geo[0] for geo in predicted_geos]
    lngs = [geo[1] for geo in predicted_geos]
    radius = [geo[2] for geo in predicted_geos]

    predicted_intervals = [
        time_date_normalization(predicted_date, publish_date)
        for predicted_date, publish_date in zip(predicted_dates, publish_dates)
    ]

    for i, interval in enumerate(predicted_intervals):
        if not interval and predicted_times[i]:
            predicted_intervals[i] = time_date_normalization(
                predicted_times[i], publish_dates[i]
            )

    interval_start = [interval[0] for interval in predicted_intervals]
    interval_end = [interval[1] for interval in predicted_intervals]

    filtered_reddit_articles_df["landslide_category"] = preds_cats
    filtered_reddit_articles_df["landslide_trigger"] = preds_trigs
    filtered_reddit_articles_df["location"] = predicted_locations
    filtered_reddit_articles_df["latitude"] = lats
    filtered_reddit_articles_df["longtitude"] = lngs
    filtered_reddit_articles_df["radius_km"] = radius
    filtered_reddit_articles_df["interval_start"] = interval_start
    filtered_reddit_articles_df["interval_end"] = interval_end
    filtered_reddit_articles_df["cas"] = predicted_cas

    filtered_reddit_articles_df.to_csv(os.path.join(DATA_PATH, "output", "result_bert.csv"))

    # loc_results = loc.predict(clean_data)
    # time_results = time.get_final_result(clean_data, filtered_reddit_articles_df)
    # data_processing.get_final_result(
    # filtered_reddit_articles_df, loc_results, time_results
    # )


if __name__ == "__main__":
    main()
