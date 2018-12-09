from Recommend.algo_set.baseline import BaseLineAlgo
from Recommend.algo_set.collaborate_based_algo import CollaborateBasedAlgo
from Recommend.algo_set.topic_based_algo import TopicBasedAlgo
from Recommend.utils import Evaluator
from gensim import corpora
import pandas as pd
import pickle as pic

data_path = "../DataAnalysis/generate_datas/"
with open(data_path + 'corpus_tfidf.pdata', 'rb') as f:
    corpus_tfidf = pic.load(f)
word_dict = corpora.Dictionary.load(data_path + 'word_dict.dict')
news_tfvec = pd.read_csv(data_path + 'newsid_tfvec.csv')
user_log = pd.read_csv(data_path + 'userid_newsid.csv')
lsi = TopicBasedAlgo(news_tfvec['news_id'], topic_n=150, corpus=corpus_tfidf, id2word=word_dict, chunksize=400,
                     topic_type='lsi')
cb = CollaborateBasedAlgo('cosine', False, 10)
eva = Evaluator(user_log)
print(
    eva.evaluate(algo=BaseLineAlgo(), k=5, n_jobs=3, split_date='2014-3-21', debug=False, verbose=True, auto_log=True))
print(eva.evaluate(algo=lsi, k=25, n_jobs=4, split_date='2014-3-21', debug=False, verbose=True, auto_log=True))
print(eva.evaluate(algo=cb, k=10, split_date='2014-3-21', debug=False, verbose=True))
