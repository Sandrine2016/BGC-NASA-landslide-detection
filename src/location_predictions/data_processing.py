import os
from urllib.request import Request, urlopen
import pandas as pd
from flair.data import Sentence
import spacy as spacy
from flair.models import SequenceTagger
from src.location_predictions import check_duplicates


MAIN_PATH = os.path.join(
    os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))),
    os.pardir,
    os.pardir,
)
DATA_PATH = os.path.join(MAIN_PATH, "data")


def get_non_date(df):
    non_dates = []
    for idx, row in df.iterrows():
        if type(row["article_publish_date"]) == float:
            non_dates.append(row["id"])
    return non_dates


def update_date(df, non_date_lst, tagger):
    date_dict = {}
    for idx, i in enumerate(non_date_lst):
        try:
            req = Request(
                list(df[df["id"] == i]["source_link"])[0],
                headers={"User-Agent": "Mozilla/5.0"},
            )
            webpage = urlopen(req, timeout=10).read()
            if "datepublish" in str(webpage).lower():
                text = str(webpage)[str(webpage).lower().index("datepublish") :][:300]
                sentence = Sentence(text)
                tagger.predict(sentence)
                for entity in sentence.get_spans("ner"):
                    if entity.tag == "DATE":
                        date_dict[i] = entity.text
                        break
        except Exception:
            continue

    for idx, row in df.iterrows():
        if type(row["article_publish_date"]) == float:
            if row["id"] in date_dict:
                df.loc[idx, "article_publish_date"] = date_dict[row["id"]]
    return df


def get_test_data(df, tagger, nlp):
    GPE = []
    LOC = []
    DATE = []
    TIME = []
    INDEX = []
    SENTENCE = []
    for idx, row in df.iterrows():
        if type(row["article_text"]) != float:
            for sent in nlp(row["article_text"]).sents:
                sub_sent = sent.text.strip()
                if sub_sent:
                    sentence = Sentence(sub_sent)
                    tagger.predict(sentence)
                    sub_gpe = []
                    sub_date = []
                    sub_time = []
                    sub_loc = []
                    for entity in sentence.get_spans("ner"):
                        if entity.tag == "GPE":
                            sub_gpe.append(entity.text)
                        if entity.tag == "DATE":
                            sub_date.append(entity.text)
                        if entity.tag == "TIME":
                            sub_time.append(entity.text)
                        if entity.tag == "LOC":
                            sub_loc.append(entity.text)
                    if sub_loc or sub_date or sub_gpe or sub_time:
                        GPE.append("|".join(sub_gpe))
                        LOC.append("|".join(sub_loc))
                        DATE.append("|".join(sub_date))
                        TIME.append("|".join(sub_time))
                        INDEX.append(idx)
                        SENTENCE.append(sub_sent)
    return pd.DataFrame(
        {
            "id": INDEX,
            "text": SENTENCE,
            "GPE": GPE,
            "LOC": LOC,
            "DATE": DATE,
            "TIME": TIME,
        }
    )


def prepare_date(df):
    model_nlp = spacy.load("en_core_web_sm")
    model_tagger = SequenceTagger.load("ner-ontonotes-fast")
    # non_dates = get_non_date(df)
    # df = update_date(df, non_dates, model_tagger)

    clean_data = get_test_data(df, model_tagger, model_nlp)
    return clean_data


def get_final_result(original_data, loc, time, nasa):
    original_data = original_data.reset_index()
    loc_results = loc.reset_index()
    original_data = original_data.merge(loc_results, on='index', how='left')
    original_data = original_data.merge(time, on='index', how='left')
    original_data = original_data.loc[original_data['category'] == 'pos']
    original_data = original_data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'index', 'category', 'id'])
    original_data = check_duplicates.remove_duplicates(original_data, nasa)
    original_data.to_csv('../result.csv')
