import pandas as pd
from src.location_predictions import location as loc , data_processing

if __name__ == '__main__':
    test = pd.read_csv('../data/output/article_sample.tsv' , sep='\t')
    clean_data = data_processing.prepare_date(test)
    print(clean_data)
    results = loc.predict(clean_data)
    results.to_csv('test.csv')
    print(results)

