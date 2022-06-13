import spacy as spacy
from flair.models import SequenceTagger
import data_processing
import pandas as pd

if __name__ == '__main__':
    test = pd.read_csv('../data/output/article_sample.tsv', sep='\t')
    model_nlp = spacy.load('en_core_web_sm')
    model_tagger = SequenceTagger.load('ner-ontonotes-fast')
    non_dates = data_processing.get_non_date(test)
    updated_data = data_processing.update_date(test, non_dates, model_tagger)
    test_data = data_processing.get_test_data(updated_data, model_tagger, model_nlp)
    print(test_data)
    print(test_data.iloc[0])

