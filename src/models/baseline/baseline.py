import os
import pickle
import numpy as np
from joblib import Parallel, delayed
from flair.models import SequenceTagger
from nltk.tokenize.regexp import RegexpTokenizer
from extraction.casualties import casualties
from extraction.time.landslide_event_time import LandslideEventTime
from extraction.location.landslide_event_location import LandslideEventLocation


TOKENIZER = RegexpTokenizer("\w+|\$[\d\.]+|\S+")

MAIN_PATH = os.path.join(
    os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))),
    os.pardir,
    os.pardir,
    os.pardir
)
MODEL_PATH = os.path.join(MAIN_PATH, "models")


def extract_casualties(text):
    """
    Rule based method to extract casualties if a certain text contains a
    token number about casualties.

    Parameters
    ----------
    text : str


    Returns
    -------
    str
        number of casualties if any
    """
    tokens = TOKENIZER.tokenize(text)
    for i, token in enumerate(tokens):
        if casualties.is_num(token):
            if i + 4 > len(tokens):
                return ""
            if (
                tokens[i + 1].lower() == "dead"
                or tokens[i + 1].lower() == "died"
                or tokens[i + 1].lower() == "killed"
                or tokens[i + 1].lower() == "buried"
                or tokens[i + 2].lower() == "dead"
                or tokens[i + 2].lower() == "died"
                or tokens[i + 2].lower() == "killed"
                or tokens[i + 2].lower() == "buried"
                or tokens[i + 3].lower() == "dead"
                or tokens[i + 3].lower() == "died"
                or tokens[i + 3].lower() == "killed"
                or tokens[i + 3].lower() == "buried"
            ):
                return casualties.format_num(token)
    return ""


def is_time_sentence_invalid(row):
    if type(row["DATE"]) != float or type(row["TIME"]) != float:
        return True
    else:
        return False


def is_location_sentence_invalid(row):
    if type(row["GPE"]) != float or type(row["LOC"]) != float:
        return True
    else:
        return False


def predict_categories(texts):
    """
    Predicts landslide categories with a logistic regression model.

    Parameters
    ----------
    texts : list(str)
        list of strings to predict

    Returns
    -------
    list(str)
        list of categories
    """
    with open(os.path.join(MODEL_PATH, "category.model"), "rb") as f:
        model = pickle.load(f)

    categories = model.predict(texts)

    return categories


def predict_triggers(texts):
    """
    Predicts landslide triggers with a logistic regression model.

    Parameters
    ----------
    texts : list(str)
        list of strings to predict

    Returns
    -------
    list(str)
        list of triggers
    """
    with open(os.path.join(MODEL_PATH, "trigger.model"), "rb") as f:
        model = pickle.load(f)

    triggers = model.predict(texts)

    return triggers


def predict_casualties(texts):
    """
    Predicts casualties with a rule based method.

    Parameters
    ----------
    texts : list(str)
        list of strings to predict

    Returns
    -------
    list(str)
        list of casualties
    """
    casualties = [extract_casualties(text) for text in texts]

    return casualties


def predict_datetimes(sentence_df, publication_dates):
    predicted_event_times = [None] * sentence_df.groupby("id").size()

    with open(os.path.join(MODEL_PATH, "date_time.model"), "rb") as f:
        model = pickle.load(f)

    time_probs = model.predict_proba(sentence_df["text"])[:, 1]

    sentence_df["time_sentence_is_positive_confidence"] = time_probs
    sentence_df = sentence_df[sentence_df.apply(is_time_sentence_invalid)].copy()
    sentence_df = sentence_df[
        sentence_df.groupby("id")["time_sentence_is_positive_confidence"].idxmax()
    ].copy()

    for idx in range(sentence_df.shape[0]):
        phrases = sentence_df["dates"].iloc[idx].split("|")
        publication_date = publication_dates(sentence_df["id"].iloc[idx])
        predicted_event_times[sentence_df["id"].iloc[idx]] = LandslideEventTime(
            phrases, publication_date
        )

    return predicted_event_times


def predict_locations(sentence_df):
    predicted_event_locations = [None] * sentence_df.groupby("id").size()
    with open(os.path.join(MODEL_PATH, "location.model"), "rb") as f:
        model = pickle.load(f)

    location_probs = model.predict_proba(sentence_df["text"])[:, 1]

    sentence_df["time_sentence_is_positive_confidence"] = location_probs
    sentence_df = sentence_df[sentence_df.apply(is_location_sentence_invalid)].copy()
    sentence_df = sentence_df[
        sentence_df.groupby("id")["time_sentence_is_positive_confidence"].idxmax()
    ].copy()

    locations_candidates = sentence_df["locations"].to_numpy()
    extracted_event_locations = Parallel(n_jobs=-1, verbose=1)(
        delayed(LandslideEventLocation)(locations.split("|"))
        for locations in locations_candidates
    )

    for id, event_location in zip(sentence_df["id"].to_numpy(), extracted_event_locations):
        predicted_event_locations[id] = event_location

    return predicted_event_locations


def predict(article_df):
    model_tagger = SequenceTagger.load("ner-ontonotes-fast")
    sentence_df = data_processing.prepare_date(article_df, model_tagger)

    sentence_df = data_processing.merge_locs_dates(sentence_df)

    articles = article_df["article_text"].to_numpy().tolist()
    publication_dates = article_df["article_publish_date"].to_numpy().tolist()

    event_locations = predict_locations(sentence_df)
    event_times = predict_datetimes(sentence_df, publication_dates)
    casualties = predict_casualties(articles)
    categories = predict_categories(articles)
    triggers = predict_triggers(articles)

    return (
        event_locations,
        event_times,
        casualties,
        categories,
        triggers
    )
