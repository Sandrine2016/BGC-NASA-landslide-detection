import data_processing
import pandas as pd
import location as loc


if __name__ == '__main__':
    test = pd.read_csv('../data/output/article_sample.tsv', sep='\t')
    clean_data = data_processing.prepare_date(test)
    print(clean_data)
    results = loc.predict(clean_data)
    results.to_csv('test.csv')
    print(results)

