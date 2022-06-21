import os
import pickle

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import geocoder
import geopy.distance


MAIN_PATH = os.path.join(
    os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))),
    os.pardir,
    os.pardir,
)
MODEL_PATH = os.path.join(MAIN_PATH, "models")


def merge_locs_dates(data):
    """Fill NAs and merge location/date columns"""
    data = data.fillna("")  # replace NAN with empty string
    data["locations"] = data[["GPE", "LOC"]].agg(
        "|".join, axis=1
    )  # join GPE and LOC by |
    data["dates"] = data[["DATE", "TIME"]].agg(
        "|".join, axis=1
    )  # join DATE and TIME by |
    data = data.drop(
        columns=["GPE", "LOC", "DATE", "TIME"]
    )  # keep only the joined column
    return data


def preprocess(nasa, ner):
    """
    prepare data for pos sentence prediction model

    Parameters
    ----------
    nasa : pandas DataFrame
        nasa dataset containing location_description
    ner : pandas DataFrame
        dataset containing id, text, GPE, LOC, DATE, TIME columns from previous step

    Returns
    ----------
        a pandas DataFrame with id, text, location_description, locations, dates, pos_sentence columns
    """
    nasa = nasa.reset_index()
    nasa = nasa.rename(columns={"index": "id"})  # id: index in NASA dataset
    df = pd.merge(ner, nasa, how="left", on="id")
    df = merge_locs_dates(
        df[["id", "text", "location_description", "GPE", "LOC", "DATE", "TIME"]]
    )

    data = df.to_dict()  # transform dataframe into dictionary

    data["pos_sentence"] = {}
    n = defaultdict(int)
    data["number_of_pos_sent"] = {}
    data["contain_pos_sent"] = {}

    # exact match
    # iterate over each sentence in the data
    for i in range(len(data["text"])):
        if data["location_description"][i] in data["text"][i]:
            data["pos_sentence"][i] = "Yes"
            n[data["id"][i]] += 1
        else:
            data["pos_sentence"][i] = "No"

    # partial match
    for i in range(len(data["text"])):
        if n[data["id"][i]] == 0:
            locs = list(filter(None, data["locations"][i].split("|")))
            if any(loc in data["location_description"][i] for loc in locs):
                data["pos_sentence"][i] = "Yes"
                n[data["id"][i]] += 1
            else:
                data["pos_sentence"][i] = "No"

    # count how many pos_sentence each document has
    # count how many documents contain gold place name, how many doesn't
    for i in range(len(data["text"])):
        if n[data["id"][i]] == 0:
            data["number_of_pos_sent"][i] = 0
            data["contain_pos_sent"][i] = False
        else:
            data["number_of_pos_sent"][i] = n[data["id"][i]]
            data["contain_pos_sent"][i] = True

    return pd.DataFrame(data)[pd.DataFrame(data)["contain_pos_sent"]]


def train(data):
    """Train and return the model for pos sentence prediction"""
    df_train, df_test = train_test_split(data, test_size=0.20, random_state=123)
    X_train, y_train = df_train["text"], df_train["pos_sentence"]
    X_test, y_test = df_test["text"], df_test["pos_sentence"]
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2)),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    model.fit(X_train, y_train)
    test_scores = classification_report(y_test, model.predict(X_test))
    return model, test_scores


def get_distance(p1, p2):
    """Get the geographical distance between two points"""
    if p1 and p2:
        return round(geopy.distance.geodesic(p1, p2).km, 3)
    else:
        return None


def get_radius(p1, p2):
    """Get the radius of a region"""
    if p1 and p2:
        return round(geopy.distance.geodesic(p1, p2).km, 3) / 2
    else:
        return None


def get_outlier_idx(centroid, points):
    """
    Parameters:
        centroid: a tuple of centroid;
        points: a list of tuples
    Return:
        the index of the point that should be removed
    """
    dists = [get_distance(centroid, point) for point in points]
    return dists.index(max(dists))


def get_smallest_region_idx(locs):
    """
    Parameters
    ----------
    locs : list of dictionary
        a list of dictionary containing latitude, longitude,
        northeast point, southwest point for all the location
        entities in the positive sentence

    Returns
    ----------
        an integer indicating the index of the location entity
        that has the smallest region
    """
    dists = [get_distance(loc["northeast"], loc["southwest"]) for loc in locs]
    return dists.index(min(dists))


def predict(df):  # df=example, model=best_model
    """Get the most likely locations, latitude, longitude based on pred model

    Parameters
    ----------
    df:
        a data frame containing document ID (id) and tokenized sentences (text) for each document,
        extracted location entities (locations), and extracted date entities (dates)
    model:
        the prediction model (logistic model trained on NASA dataset)

    Returns
    -------
        a data frame with locations, the most likely location, latitude, longitude, diameter
    """
    with open(os.path.join(MODEL_PATH, "loc_model"), "rb") as f:
        model = pickle.load(f)
    df = merge_locs_dates(df)

    # get predict_proba
    pd.options.mode.chained_assignment = None  # silent warning message
    df["predict_proba"] = model.predict_proba(df["text"])[:, 1]

    result = {
        "locations": defaultdict(str),
        "location": defaultdict(str),
        "latitude": defaultdict(float),
        "longitude": defaultdict(float),
        "radius_km": defaultdict(float),
    }

    # get a dict of idxmax for each document
    idx_max = df.groupby("id")["predict_proba"].idxmax().to_dict()

    data = df.to_dict()
    for i, idx in idx_max.items():  # i: index of the document; idx: index of the df
        # ensure the `locations` column of the `idxmax` row is not empty
        current_proba = df.query("id == @i")["predict_proba"]
        while data["locations"][idx] == "|":
            try:
                current_proba = current_proba.drop(idx)  # drop the current idxmax
                idx = current_proba.idxmax()  # get the idxmax of the rest
            except ValueError:
                # print(f"All locations in document {i} are empty!")
                idx = -1  # set idx=-1 if all locations are empty
                break

        # store the locations, latitude, longitude in result dict
        if idx != -1:
            result["locations"][i] = data["locations"][idx]

            locs = list(filter(None, data["locations"][idx].split("|")))
            geolocs = []
            for loc in locs:

                geocoded = geocoder.arcgis(loc).json
                if geocoded:
                    geoloc = geocoded["bbox"]
                    geoloc["lat"], geoloc["lng"] = geocoded["lat"], geocoded["lng"]
                    geolocs.append(geoloc)

            if len(geolocs) > 2:
                # remove the farthest outlier
                lats, lngs = [loc["lat"] for loc in geolocs], [
                    loc["lng"] for loc in geolocs
                ]
                mean_lat, mean_lng = np.mean(lats), np.mean(lngs)  # get the centroid
                x = get_outlier_idx(
                    (mean_lat, mean_lng), [(lat, lng) for lat, lng in zip(lats, lngs)]
                )
                del geolocs[x]
                del locs[x]
                # get the index of location with the smallest region
                j = get_smallest_region_idx(geolocs)
                location = locs[j]
                lat, lng = geolocs[j]["lat"], geolocs[j]["lng"]
                ne, sw = geolocs[j]["northeast"], geolocs[j]["southwest"]
            elif len(geolocs) == 2:
                j = get_smallest_region_idx(geolocs)
                location = locs[j]
                lat, lng = geolocs[j]["lat"], geolocs[j]["lng"]
                ne, sw = geolocs[j]["northeast"], geolocs[j]["southwest"]
            elif len(geolocs) == 1:
                location = locs[0]
                lat, lng = geolocs[0]["lat"], geolocs[0]["lng"]
                ne, sw = geolocs[0]["northeast"], geolocs[0]["southwest"]
            else:
                # print(f"Locations in document {i} cannot be geocoded!")
                location, lat, lng, ne, sw = None, None, None, None, None
        else:
            result["locations"][i] = None
            location, lat, lng, ne, sw = None, None, None, None, None

        result["location"][i] = location
        result["latitude"][i], result["longitude"][i] = lat, lng
        result["radius_km"][i] = get_radius(ne, sw)

    return pd.DataFrame(result)
