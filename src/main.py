import pandas as pd
from src.location_predictions import location as loc , data_processing
from src.time_normalizer import time_prediction as time

if __name__ == '__main__':
    original_data = pd.read_csv('../data/output/article_sample.tsv', sep='\t')
    clean_data = data_processing.prepare_date(original_data)
    loc_results = loc.predict(clean_data)
    time_results = time.get_final_result(clean_data, original_data)
    data_processing.get_final_result(original_data, loc_results, time_results)



