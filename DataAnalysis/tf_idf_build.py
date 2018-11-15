from gensim import corpora, models
import pandas as pd
import pickle as pic
from ast import literal_eval


class TfIdfGenerator:

    def __init__(self, generate_data_path, original_tokens_path=''):
        self.__original_tokens_path = original_tokens_path
        self.__generate_data_path = generate_data_path
        self.__news_tokens = None

    def generate(self, news_tokens=None, auto_save=True):
        """
        This method builds TF-IDF vectors for news.

        :param news_tokens: news id and its tokens, if None, method will load tokens from __original_tokens_path
        :param auto_save: if true,method will automatically save news id and its vector to a csv file.
        :return: a pd.DataFrame of news id and its vector
        """
        if news_tokens is None:
            assert (self.__original_tokens_path != ''), 'No data source.'
            self.__news_tokens = pd.read_csv(self.__original_tokens_path, encoding='utf-8')
        else:
            self.__news_tokens = news_tokens
        pure_tokens = self.__news_tokens['tokens'].values.tolist()
        pure_tokens = [literal_eval(t) for t in pure_tokens]  # transform list-like strings to list
        word_dict = corpora.Dictionary(pure_tokens)  # Used in LSA or LDA algo
        news_bow = [word_dict.doc2bow(t) for t in pure_tokens]
        algo = models.TfidfModel(news_bow)
        corpus_tfidf = algo[news_bow]
        news_vec = []
        for t in corpus_tfidf:
            news_vec.append([v for (_, v) in t])
        newsid_tfvec = pd.DataFrame({'news_id': self.__news_tokens['news_id'].values.tolist(), 'tf_vec': news_vec})
        if auto_save:
            f = open(self.__generate_data_path + 'corpus_tfidf.pdata', 'wb')
            pic.dump(corpus_tfidf, f)
            f.close()
            word_dict.save(self.__generate_data_path + 'word_dict.dict')
            newsid_tfvec.to_csv(self.__generate_data_path + 'newsid_tfvec.csv', index=False, encoding='utf-8')
        return newsid_tfvec
