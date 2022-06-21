import pandas as pd
from collections import defaultdict
import geocoder
import geopy.distance
import math
from datetime import datetime


def get_distance2(pred_lat, pred_lng, gold_lat, gold_lng):
    if pd.isnull(pred_lat):
        return None
    else:
        return round(geopy.distance.geodesic((pred_lat, pred_lng), (gold_lat, gold_lng)).km, 3)


def get_gold_radius(accuracy):
    if not pd.isnull(accuracy):
        if accuracy == 'exact' or accuracy == 'Known exactly':
            output = 0.1
        if accuracy.lower() == 'unknown':
            output = 100
        if accuracy.endswith('km'):
            output = float(accuracy[:-2])
    else:
        output = None
    return output


def get_correct(r_pred, r_gold, d):
    if d <= r_pred + r_gold:
        correct = True
    else:
        correct = False
    return correct


def get_precision_recall_f1(r_pred, r_gold, d):
    """Get location evaluation metrics based on intersected area"""
    if r_pred >= r_gold:
        r1 = r_pred
        r2 = r_gold
    else:
        r1 = r_gold
        r2 = r_pred

    if d >= r1 + r2:
        a_intersection = 0
    elif d <= r1 - r2:
        a_intersection = math.pi * (r2**2)
    else:
        d1 = (r1**2 - r2**2 + d**2)/(2*d)
        d2 = d - d1
        a1 = (r1**2) * math.acos(d1/r1) - d1 * math.sqrt(r1**2 - d1**2)
        a2 = (r2**2) * math.acos(d2/r2) - d2 * math.sqrt(r2**2 - d2**2)
        a_intersection = a1 + a2

    a_pred = math.pi * (r_pred**2)
    a_gold = math.pi * (r_gold**2)

    precision = a_intersection/a_pred
    recall = a_intersection/a_gold
    
    if precision != 0 and recall != 0:
        f1 = 2*precision*recall/(precision+recall)
    else:
        f1 = 0
    return precision, recall, f1


def drop_predicted_duplicates(df):
    """Drop rows where both date, loc are empty or duplicated"""
    idxs = df.query('discrete_date != discrete_date & location != location').index.to_list()
    if idxs:
        df = df.drop([idxs], axis=0)
    df = df.drop_duplicates(subset=['location', 'discrete_date'])
    df = df.reset_index().drop('index', axis=1)
    return df


def get_potential_duplicates(pred, gold):
    """
    Get potential_duplicates (nasa dataset indices where the dates are duplicated)
    
    Parameters:
    -----------
        pred: predicted data frame containing id, locations, location, latitude, 
              longitude, radius_km, interval, date, confidence columns
        gold: nasa dataset
    Returns:
        df: a data frame containing potential_duplicates column
    """
    df = pred.to_dict()
    df['potential_duplicates'] = defaultdict(str)
    
    gold['event_date'] = pd.to_datetime(gold['event_date'], format='%Y-%m-%d %H:%M', errors='coerce')  # transform event_date into datetime format
    gold = gold.dropna(subset=['event_date'])  # drop rows that are not in datetime format
    gold = gold.reset_index()  # keep the original nasa dataset index in 'index' column
    
    for i in range(len(pred)):
        data = pred.iloc[[i]].merge(gold[['index','event_date']], how='cross')  # index: original nasa dataset index
        if "interval_to_normalize" in pred and not pd.isnull(pred['interval_to_normalize'][i]) and "-" in pred['interval_to_normalize'][i]:   # If interval has valid date interval
            start, end = pred['interval_to_normalize'][i].split('-')  # pred['interval'][i].split('\n')[0].split('-')   start, end = '1997/01/01, 00:00', '1997/01/20, 00:00'
            start, end = datetime.strptime(start.strip(), '%Y/%m/%d, %H:%M'), datetime.strptime(end.strip(), '%Y/%m/%d, %H:%M')
            ids = data.query('@start <= event_date <= @end').index.to_list()  # ids: data index      
        elif "interval_start" in pred and "interval_end" in pred:
            start = pred["interval_start"].iloc[i]
            end = pred["interval_end"].iloc[i]
            ids = data.query('@start <= event_date <= @end').index.to_list()  # ids: data index      
        elif not pd.isnull(pred['article_publish_date'][i]): 
            date = datetime.strptime(pred['article_publish_date'][i][:10], '%Y-%m-%d')
            ids = data.query('event_date == @date').index.to_list()  
        else:
            ids = None
        
        if ids:
            idxs = data.iloc[ids]['index'].to_list()  # idxs: original nasa dataset index
            df['potential_duplicates'][i] = ','.join([str(i) for i in idxs])   
        else:
            df['potential_duplicates'][i] = ""
    return pd.DataFrame(df)  # "": no duplicated date or no date


def drop_nasa_duplicates(pred, gold):
    """
    Remove rows that are already in NASA dataset based on location and time
    
    Parameters:
    -----------
        pred: predicted data frame containing id, locations, location, latitude, 
              longitude, radius_km, interval, date, potential_duplicates columns
        gold: nasa dataset
    Returns:
        df: a data frame without duplicated rows in nasa dataset
    """
    df = pred.iloc[:,:-1].copy() 
    gold = gold.rename(columns={"latitude": "gold_latitude", "longitude": "gold_longitude"})
    gold = gold[['location_description', 'location_accuracy', 'gold_latitude', 'gold_longitude']]
    
    for i in range(len(pred)):
        if pred['potential_duplicates'][i] != "":
            idxs = pred['potential_duplicates'][i].split(',')  # find index of the potential duplicates (indexes seperated by ",")
            data = pred.iloc[[i]].merge(gold.iloc[idxs], how='cross')  # full join the current pred row with potential duplicate rows in nasa dataset
            data = data.assign(distance_km=data.apply(lambda x: get_distance2(x.latitude, x.longitude, x.gold_latitude, x.gold_longitude), axis=1))  
            data = data.assign(gold_radius_km=data.apply(lambda x: get_gold_radius(x.location_accuracy), axis=1))
            data = data.assign(correct=data.apply(lambda x: get_correct(x.radius_km, x.gold_radius_km, x.distance_km), axis=1))
            data[['precision', 'recall', 'f1_score']] = data.apply(lambda x: get_precision_recall_f1(x.radius_km, x.gold_radius_km, x.distance_km), axis=1, result_type='expand')
            if not data.query('precision > 0.5').empty:  # correct == True
                df = df.drop([i], axis=0)
    return df


def remove_duplicates(pred, gold):
    """Remove all the duplicates of prediction table and nasa dataset"""
    pred = drop_predicted_duplicates(pred)
    pred = get_potential_duplicates(pred, gold)
    pred = drop_nasa_duplicates(pred, gold)
    return pred  # remove original index: pred.iloc[:,1:]

