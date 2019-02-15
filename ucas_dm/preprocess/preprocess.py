# coding=utf-8
import pandas as pd
import jieba.posseg as pseg
import codecs
import os
from gensim import corpora, models
from ast import literal_eval


class PreProcessor:

    def __init__(self, source_data_path):
        self.__source_data_path = source_data_path

    def extract_news(self):
        """
        This method extract news from data and save them to a csv file.

        :return: A pandas.DataFrame with two attributes: news_id and content
        """
        data = pd.read_csv(filepath_or_buffer=self.__source_data_path, sep="\\t", names=[
            'user_id', 'news_id', 'view_time', 'title', 'content', 'publish_time'], encoding="utf-8")
        data = data[['news_id', 'title', 'content']]
        data = data.fillna('')
        data = data.drop_duplicates().reset_index(drop=True)
        data['content'] = data['title'] + data['content']
        id_content = data[['news_id', 'content']]
        return id_content

    @classmethod
    def generate_tokens(cls, id_content):
        """
        This method generate tokens for news.

        :param id_content: A pandas.DataFrame of news id(integer) and its content(string)  \
        \|column1: news_id\|column2: content\|

        :return: A pd.DataFrame of news id and its tokens
        """
        dir_path = os.path.split(__file__)[0]
        stop_words_path = dir_path + "/stop_words/stop.txt"
        id_content.columns = ['news_id', 'content']
        stop_words = codecs.open(stop_words_path, encoding='utf8').readlines()
        stop_words = [w.strip() for w in stop_words]
        stop_flags = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']

        def _tokenization(text):
            result = []
            words = pseg.cut(text)
            for word, flag in words:
                if flag not in stop_flags and word not in stop_words:
                    result.append(word)
            return result

        res = []
        for i in range(id_content.shape[0]):
            content = id_content.loc[i, 'content']
            result = _tokenization(content)
            res.append(result)

        assert (id_content.shape[0] == len(
            res)), "The number of id_content's rows doesn't match the length of tokenization result."
        id_tokens = pd.DataFrame({'news_id': id_content['news_id'], 'tokens': res})
        return id_tokens

    @classmethod
    def build_tf_idf(cls, id_tokens):
        """
        This method builds TF-IDF vectors for news.

        :param id_tokens: A pandas.DataFrame contains news id and its tokens.  \|column1: news_id\|column2: tokens\|
        :return: A dict - {"id_tfvec": A pandas.DataFrame contains news id and its tf-idf vector \
        \|column1: news_id\|column2: tf_vec\| ,"gensim_pack":{"word2dict": important parameter if package "gensim" is \
        used for further process, "corpus": important parameter if package "gensim" is used for further process}}
        """
        id_tokens.columns = ['news_id', 'tokens']
        pure_tokens = id_tokens['tokens'].values.tolist()
        if isinstance(pure_tokens[0], str):
            pure_tokens = [literal_eval(t) for t in pure_tokens]  # transform list-like strings to list
        word_dict = corpora.Dictionary(pure_tokens)  # Used in LSA or LDA algorithm
        news_bow = [word_dict.doc2bow(t) for t in pure_tokens]
        algo = models.TfidfModel(news_bow)
        corpus_tfidf = algo[news_bow]
        news_vec = []
        for t in corpus_tfidf:
            news_vec.append([v for (_, v) in t])
        id_tfvec = pd.DataFrame({'news_id': id_tokens['news_id'], 'tf_vec': news_vec})
        return {"id_tfvec": id_tfvec, "gensim_pack": {"id2word": word_dict, "corpus": corpus_tfidf}}

    def extract_logs(self):
        """
        This method extract user's browsing history from source data and save it to a csv file.

        :return: A pandas.DataFrame with 3 attributes: user_id, news_id, view_time
        """
        data = pd.read_csv(filepath_or_buffer=self.__source_data_path, sep="\\t", names=[
            'user_id', 'news_id', 'view_time', 'title', 'content', 'publish_time'], encoding="utf-8")
        user_log = data[['user_id', 'news_id', 'view_time']]
        user_log['view_time'] = pd.to_datetime(user_log['view_time'], unit='s')
        user_log = user_log.drop_duplicates().reset_index(drop=True)
        return user_log
