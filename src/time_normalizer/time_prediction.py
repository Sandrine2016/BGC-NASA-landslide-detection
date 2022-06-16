from time_main import time_date_normalization
from collections import defaultdict
import pickle
import pandas as pd



def merge_locs_dates(data, model_name):

    """
    Given the file with the data and the name of the model, removes NAs, runs the model
    over the data and returns the predictions and the model's predictions confidence
    """

    with open(model_name, 'rb') as f:
        model = pickle.load(f)
    data = pd.read_csv(data)
    data = data.fillna('missing_value')  # replace NAN with missing_value string
    data['locations'] = data[['GPE', 'LOC']].agg(' '.join, axis=1)  # join GPE and LOC by " "
    data['dates'] = data[['DATE', 'TIME']].agg(' '.join, axis=1)  # join DATE and TIME by " "
    data = data.drop(columns=['GPE', 'LOC', 'DATE', 'TIME'])
    return model.predict(data), model.predict_proba(data)


def data_filtering_post_processing(data, predictions, predict_proba):

    """
    Given the file with the data, creates a normalization column that reflects if the most confident prediction
    for a separate document contains a date or a time phrase, and if not - exhaustively looks for it in the remaining
    document sentences
    """

    data = pd.read_csv(data)[['id', 'text', 'DATE', 'TIME']]
    data['label'] = predictions.tolist()
    data['pos_label_confidence'] = predict_proba[:, 1].tolist()
    normalization = len(data) * [False]
    idx_max = data.groupby('id')['pos_label_confidence'].idxmax().to_dict()
    print(len(idx_max))
    for i, idx in idx_max.items():
        current_proba = data.query('id == @i')['pos_label_confidence']
        if type(data['DATE'][idx]) != float or type(data['TIME'][idx]) != float:
            normalization[idx] = True
        else:
            while type(data['DATE'][idx]) == float and type(data['TIME'][idx]) == float:
                try:
                    current_proba = current_proba.drop(idx)  # drop the current idxmax
                    idx = current_proba.idxmax()
                    if type(data['DATE'][idx]) != float or type(data['TIME'][idx]) != float:
                        normalization[idx] = True
                        break
                except:
                    break
    data['normalization'] = normalization
    return data


def publication_gold_date_merge(publication_dates_file, nasa_catalog_gold, current_dataset):

    """
    Given the processed dataset with the predictions and normalization boolean column, adds additional columns
    with gold_date from Nasa Global Landslide Catalog Point and the file with the extracted publication dates
    """

    publication_date = pd.read_csv(publication_dates_file)[['id', 'dates']]
    gold_date = pd.read_csv(nasa_catalog_gold).reset_index()[['index', 'event_date']].rename(columns={'index': 'id',
                                                                                                       'event_date': 'gold_date'})
    current_dataset = publication_date.merge(current_dataset, on='id').rename(columns={'dates': 'publication_dates'})
    current_dataset = current_dataset.merge(gold_date, on='id').reset_index()
    return current_dataset


def phrase_normalization(data):

    """
    Merges date and time columns into one and runs the wrapper time_date_normalization function to predict
    temporal interval for positively labeled sentences, returns a dataframe with the predicted intervals as
    a separate column
    """

    data['dates'] = data.fillna('')[['DATE', 'TIME']].agg('|'.join, axis=1)
    data = data.drop(columns=['DATE', 'TIME', 'text'])

    date_time = defaultdict()
    for row in data.iterrows():

        date_time[row[1][0]] = ""
        publication_date = row[1][2]
        if type(publication_date) == float or '.' in publication_date:
            continue
        phrases = row[1][7]
        normalization = row[1][5]
        if normalization == True and len(phrases) > 1:
            phrases = phrases.split('|')
            for phrase in phrases:
                if phrase:
                    try:
                        date_start, date_end = time_date_normalization(phrase, publication_date)
                        if date_start:
                            date_time[row[0]] += str((date_start.strftime("%Y/%m/%d, %H:%M") + '-' + date_end.strftime(
                                "%Y/%m/%d, %H:%M"))) + '\n'
                        else:
                            date_time[row[0]] += "None" + '\n'
                    except:
                        date_time[row[0]] += "None" + '\n'

    date_intervals = pd.DataFrame.from_dict(date_time, orient='index').reset_index()
    data = data.merge(date_intervals, on='index').rename(columns={0: 'normalized_interval'})
    return data

def id_interval_extraction(data):

    """
    Given a dataframe with the normalized date/time intervals for each sentence, returns a dictionary with unique
    article ids and the related date/time intervals
    """
    idx = defaultdict()
    for row in data.iterrows():
        index = row[1][1]
        if index not in idx.keys():
            idx[index] = None
        normilized_date = row[1][8]
        if normilized_date:
            idx[index] = row[1][8]
    return idx