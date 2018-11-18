from Recommend.algo_set.baseline import BaseLineAlgo
from Recommend.algo_set.collaborate_based_algo import CollaborateBasedAlgo
from Recommend.algo_set.topic_based_algo import TopicBasedAlgo
from Recommend.utils import Evaluator
from ast import literal_eval
from gensim import corpora
import pandas as pd
import pickle as pic


def string2vec(item_vector):
    """
    Because the vectors would be in string type when item vectors were loaded from csv file,
    this func transform vector-like string data in item_vector into real vector.
    """
    vector_like = item_vector['tf_vec'].values.tolist()
    real_vecs = [literal_eval(t) for t in vector_like]
    return pd.DataFrame({'item_id': item_vector['news_id'], 'vec': real_vecs})


data_path = "../DataAnalysis/generate_datas/"
with open(data_path + 'corpus_tfidf.pdata', 'rb') as f:
    corpus_tfidf = pic.load(f)
word_dict = corpora.Dictionary.load(data_path + 'word_dict.dict')
news_tfvec = pd.read_csv(data_path + 'newsid_tfvec.csv')
news_tfvec = string2vec(news_tfvec)
user_log = pd.read_csv(data_path + 'userid_newsid.csv')
lsi = TopicBasedAlgo(item_vector=news_tfvec, topic_n=150, corpus=corpus_tfidf, id2word=word_dict, chunksize=400,
                     topic_type='lsi')
cb = CollaborateBasedAlgo('cosine', True, 5)
eva = Evaluator(user_log)
print(eva.evaluate(algo=BaseLineAlgo(), k=10, n_splits=3, shuffle=True, random_state=112, debug=True, verbose=True))
print(eva.evaluate(algo=lsi, k=10, n_splits=3, shuffle=True, random_state=112, debug=True, verbose=True, auto_log=True))
print(eva.evaluate(algo=cb, k=4, n_splits=3, shuffle=True, random_state=112, debug=True, verbose=True))
