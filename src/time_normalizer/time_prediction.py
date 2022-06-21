from time_normalizer.time_main import time_date_normalization
from collections import defaultdict
import os
import pickle
import pandas as pd
from dateutil.relativedelta import relativedelta
from dateparser import parse

MAIN_PATH = os.path.join(
    os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))),
    os.pardir,
    os.pardir,
)
MODEL_PATH = os.path.join(MAIN_PATH, "models")


def merge_locs_dates(data, model_name):

    """
    Given the file with the data and the name of the model, removes NAs, runs the model
    over the data and returns the predictions and the model's predictions confidence
    """

    with open(os.path.join(MODEL_PATH, model_name), "rb") as f:
        model = pickle.load(f)
    data = data.fillna("missing_value")  # replace NAN with missing_value string
    data["locations"] = data[["GPE", "LOC"]].agg(
        " ".join, axis=1
    )  # join GPE and LOC by " "
    data["dates"] = data[["DATE", "TIME"]].agg(
        " ".join, axis=1
    )  # join DATE and TIME by " "
    data = data.drop(columns=["GPE", "LOC", "DATE", "TIME"])
    return model.predict(data), model.predict_proba(data)


def data_filtering_post_processing(data, predictions, predict_proba):

    """
    Given the file with the data, creates a normalization column that reflects if the most confident prediction
    for a separate document contains a date or a time phrase, and if not - exhaustively looks for it in the remaining
    document sentences
    """

    data = data[["id", "text", "DATE", "TIME"]].copy()
    data["label"] = predictions.tolist()
    data["pos_label_confidence"] = predict_proba[:, 1].tolist()
    normalization = len(data) * [False]
    idx_max = data.groupby("id")["pos_label_confidence"].idxmax().to_dict()
    for i, idx in idx_max.items():
        current_proba = data.query("id == @i")["pos_label_confidence"]
        if type(data["DATE"][idx]) != float or type(data["TIME"][idx]) != float:
            normalization[idx] = True
        else:
            while type(data["DATE"][idx]) == float and type(data["TIME"][idx]) == float:
                try:
                    current_proba = current_proba.drop(idx)  # drop the current idxmax
                    idx = current_proba.idxmax()
                    if (
                        type(data["DATE"][idx]) != float
                        or type(data["TIME"][idx]) != float
                    ):
                        normalization[idx] = True
                        break
                except:
                    break
    data["normalization"] = normalization
    return data


def publication_gold_date_merge(
    publication_dates_file, nasa_catalog_gold, current_dataset
):

    """
    Given the processed dataset with the predictions and normalization boolean column, adds additional columns
    with gold_date from Nasa Global Landslide Catalog Point and the file with the extracted publication dates
    """

    publication_date = publication_dates_file[["id", "article_publish_date"]].copy()
    # gold_date = pd.read_csv(nasa_catalog_gold).reset_index()[['index', 'article_publish_date']].rename(columns={'index': 'id',
    #                                                                                                    'event_date': 'gold_date'})
    current_dataset = publication_date.merge(current_dataset, on="id").rename(
        columns={"article_publish_date": "publication_dates"}
    )
    # current_dataset = current_dataset.merge(gold_date, on='id').reset_index()
    return current_dataset


def phrase_normalization(data):

    """
    Merges date and time columns into one and runs the wrapper time_date_normalization function to predict
    temporal interval for positively labeled sentences, returns a dataframe with the predicted intervals as
    a separate column
    """

    data["dates"] = data.fillna("")[["DATE", "TIME"]].agg("|".join, axis=1)
    data = data.drop(columns=["DATE", "TIME", "text"])

    date_time = defaultdict()
    for row in data.iterrows():
        date_time[row[1][0]] = ""
        publication_date = row[1][1]
        if type(publication_date) == float or "." in str(publication_date):
            continue
        phrases = row[1][5]
        normalization = row[1][4]
        if normalization == True and len(phrases) > 1:
            phrases = phrases.split("|")
            for phrase in phrases:
                if phrase:
                    try:
                        date_start, date_end = time_date_normalization(
                            phrase, publication_date
                        )
                        if date_start:
                            date_time[row[1][0]] += (
                                str(
                                    (
                                        date_start.strftime("%Y/%m/%d, %H:%M")
                                        + "-"
                                        + date_end.strftime("%Y/%m/%d, %H:%M")
                                    )
                                )
                                + "\n"
                            )
                        else:
                            date_time[row[1][0]] += "None" + "\n"
                    except:
                        date_time[row[1][0]] += "None" + "\n"
    date_intervals = pd.DataFrame.from_dict(date_time, orient="index").reset_index()
    data["index"] = list(data["id"])
    data = data.merge(date_intervals, on="index").rename(
        columns={0: "normalized_interval"}
    )
    return data


def id_interval_extraction(data):

    """
    Given a dataframe with the normalized date/time intervals for each sentence, returns a dictionary with unique
    article ids and the related date/time intervals
    """

    idx = defaultdict()
    for row in data.iterrows():
        index = row[1][0]
        if index not in idx.keys():
            idx[index] = None
        normilized_date = row[1][7]
        if normilized_date:
            idx[index] = row[1][7]
    return idx


# def discrete_date_plus_confidence(data):
#
#     """
#     Given a df with calculated intervals, calculates the centroids of these intervals and their confidence interval,
#     returns a df with document index, intervals, discrete values and confidence intervals
#     """
#
#     discrete_date = defaultdict(str)
#     confidence = defaultdict(str)
#     for row in data.iterrows():
#         intervals = [row[1][7]]
#         if intervals:
#             intervals = intervals[0].split('\n')
#             for interval in intervals:
#                 if len(interval) > 0:
#                     interval = interval.strip('\n')
#                     if interval == 'None':
#                         discrete_date[row[1][0]] += 'None' + '\n'
#                         confidence[row[1][0]] += 'None' + '\n'
#                     else:
#                         date_start, date_end = interval.split('-')
#                         date_start, date_end = parse(date_start), parse(date_end)
#                         delta_hours = (date_end - date_start).total_seconds() / 60 / 60 / 2
#                         date_start += relativedelta(hours=delta_hours)
#                         discrete_date[row[1][0]] += str(date_start.strftime("%Y/%m/%d, %H:%M")) + '\n'
#                         confidence[row[1][0]] = confidence[row[1][0]] + f'± {round(delta_hours, 2)} hours \n'
#
#     discrete_date = pd.DataFrame.from_dict(discrete_date, orient='index').reset_index().rename(
#         columns={0: 'discrete_date'})
#     confidence = pd.DataFrame.from_dict(confidence, orient='index').reset_index().rename(columns={0: 'confidence'})
#     intervals = id_interval_extraction(data)
#     intervals = pd.DataFrame.from_dict(intervals, orient='index').reset_index().rename(columns={0: 'interval'})
#     merged_columns = intervals.merge(discrete_date, on='index')
#     merged_columns = merged_columns.merge(confidence, on='index')
#     return merged_columns


def descrete_date_plus_confidence(data):
    discrete_date = defaultdict(str)
    potential_discrete_dates = defaultdict(str)
    confidence = defaultdict(str)
    potential_confidence = defaultdict(str)
    interval_to_normalize = defaultdict(str)
    potential_intervals = defaultdict(str)
    for row in data.iterrows():
        intervals = [row[1][7]]
        if intervals:
            intervals = intervals[0].split("\n")
            for interval in intervals:
                if len(interval) > 0:
                    interval = interval.strip("\n")
                    if interval == "None":
                        if discrete_date[row[1][0]]:
                            potential_discrete_dates[row[1][0]] += "None" + "\n"
                            potential_confidence[row[1][0]] += "None" + "\n"
                            potential_intervals[row[1][0]] += interval + "\n"
                    else:
                        date_start, date_end = interval.split("-")
                        date_start, date_end = parse(date_start), parse(date_end)
                        delta_hours = (
                            (date_end - date_start).total_seconds() / 60 / 60 / 2
                        )
                        date_start += relativedelta(hours=delta_hours)
                        if discrete_date[row[1][0]]:
                            potential_discrete_dates[row[1][0]] += (
                                str(date_start.strftime("%Y/%m/%d, %H:%M")) + "\n"
                            )
                            potential_confidence[
                                row[1][0]
                            ] += f"± {round(delta_hours, 2)} hours \n"
                            potential_intervals[row[1][0]] += interval + "\n"
                        else:
                            interval_to_normalize[row[1][0]] += interval + "\n"
                            discrete_date[row[1][0]] += (
                                str(date_start.strftime("%Y/%m/%d, %H:%M")) + "\n"
                            )
                            confidence[row[1][0]] += (
                                confidence[row[1][0]]
                                + f"± {round(delta_hours, 2)} hours \n"
                            )

    discrete_date = (
        pd.DataFrame.from_dict(discrete_date, orient="index")
        .reset_index()
        .rename(columns={0: "discrete_date"})
    )
    potential_discrete_dates = (
        pd.DataFrame.from_dict(potential_discrete_dates, orient="index")
        .reset_index()
        .rename(columns={0: "potential_discrete_dates"})
    )
    confidence = (
        pd.DataFrame.from_dict(confidence, orient="index")
        .reset_index()
        .rename(columns={0: "confidence"})
    )
    interval_to_normalize = (
        pd.DataFrame.from_dict(interval_to_normalize, orient="index")
        .reset_index()
        .rename(columns={0: "interval_to_normalize"})
    )
    potential_intervals = (
        pd.DataFrame.from_dict(potential_intervals, orient="index")
        .reset_index()
        .rename(columns={0: "potential_intervals"})
    )
    potential_confidence = (
        pd.DataFrame.from_dict(potential_confidence, orient="index")
        .reset_index()
        .rename(columns={0: "potential_confidence"})
    )
    intervals = id_interval_extraction(data)
    intervals = (
        pd.DataFrame.from_dict(intervals, orient="index")
        .reset_index()
        .rename(columns={0: "interval"})
    )
    merged_columns = (
        intervals.merge(interval_to_normalize, on="index", how="outer")
        .merge(potential_intervals, on="index", how="outer")
        .merge(discrete_date, on="index", how="outer")
        .merge(confidence, on="index", how="outer")
        .merge(potential_discrete_dates, on="index", how="outer")
        .merge(potential_confidence, on="index", how="outer")
    )
    merged_columns = merged_columns.fillna(value="None")
    return merged_columns[
        [
            "index",
            "interval_to_normalize",
            "discrete_date",
            "confidence",
            "potential_intervals",
            "potential_discrete_dates",
            "potential_confidence",
        ]
    ]


def get_final_result(clean_data, original_data):
    original_data["id"] = list(range(len(original_data)))
    prediction, prob = merge_locs_dates(clean_data, "date_time.model")
    filtered_data = data_filtering_post_processing(clean_data, prediction, prob)
    merged_df = publication_gold_date_merge(original_data, original_data, filtered_data)
    df_time_date = phrase_normalization(merged_df)
    date_time_output = descrete_date_plus_confidence(df_time_date)
    return date_time_output
