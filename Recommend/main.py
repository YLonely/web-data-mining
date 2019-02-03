from Recommend.prediction_algorithms.baseline import BaseLineAlgo
from Recommend.prediction_algorithms.collaborate_based_algo import CollaborateBasedAlgo
from Recommend.prediction_algorithms.topic_based_algo import TopicBasedAlgo
from Recommend.prediction_algorithms.nmf import NMF
from Recommend.utils import Evaluator
from gensim import corpora
import pandas as pd
import pickle as pic

data_path = "../DataAnalysis/generate_datas/"
id_content = pd.read_csv(data_path + "newsid_content.csv")
user_log = pd.read_csv(data_path + 'userid_newsid.csv')
k_list = [15]
n_jobs = 6
eva = Evaluator(user_log)
# initial_p = TopicBasedAlgo.preprocess(id_content)
#
# lda = TopicBasedAlgo(initial_params=initial_p, topic_n=100, topic_type='lsi', chunksize=1000)
# lda.save('./tmp/lda_model')
ll = TopicBasedAlgo.load('./tmp/lda_model')
eva.evaluate(algo=ll, k=k_list, n_jobs=n_jobs, split_date='2014-3-21', auto_log=True, debug=True)
#
# for factor in [10, 15, 20]:
#     nmf = NMF(n_factors=factor, random_state=112)
#     eva.evaluate(algo=nmf, k=k_list, n_jobs=n_jobs, split_date='2014-3-21', auto_log=True)
#
# for k_neighbor in [5, 10, 15, 20]:
#     cb = CollaborateBasedAlgo(user_based=True, k=k_neighbor)
#     eva.evaluate(algo=cb, k=k_list, n_jobs=n_jobs, split_date='2014-3-21', auto_log=True)
#
# for k_neighbor in [15]:
#     cb = CollaborateBasedAlgo(user_based=False, k=k_neighbor, sim_func='pearson')
#     eva.evaluate(algo=cb, k=k_list, n_jobs=6, split_date='2014-3-21', auto_log=True, debug=False)
