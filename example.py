from ucas_dm.prediction_algorithms import BaseLineAlgo
from ucas_dm.prediction_algorithms import CollaborateBasedAlgo
from ucas_dm.prediction_algorithms import TopicBasedAlgo
from ucas_dm.prediction_algorithms import NMF
from ucas_dm.prediction_algorithms import InitialParams
from ucas_dm.utils import Evaluator
import pandas as pd

data_path = "./source_data/"
id_content = pd.read_csv(data_path + "newsid_content.csv")
user_log = pd.read_csv(data_path + 'userid_newsid.csv')
eva = Evaluator(user_log)
initial_params = TopicBasedAlgo.preprocess(id_content)
initial_params.save('./tmp/initial_p')
# initial_params = InitialParams.load('/tmp/initial_p')
lsi = TopicBasedAlgo(initial_params=initial_params, topic_n=100, topic_type='lsi', chunksize=1000)
lsi.save('./tmp/lsi_model')
# lsi = TopicBasedAlgo.load('./tmp/lsi_model')
k_list = [5, 10, 15]
n_jobs = 6
eva.evaluate(algo=lsi, k=k_list, n_jobs=n_jobs, split_date='2014-3-21', auto_log=True, debug=True)

for factor in [10, 15, 20]:
    nmf = NMF(n_factors=factor, random_state=112)
    eva.evaluate(algo=nmf, k=k_list, n_jobs=n_jobs, split_date='2014-3-21', auto_log=True)

for k_neighbor in [5, 10, 15, 20]:
    cb = CollaborateBasedAlgo(user_based=True, k=k_neighbor)
    eva.evaluate(algo=cb, k=k_list, n_jobs=n_jobs, split_date='2014-3-21', auto_log=True)

for k_neighbor in [5, 10, 15, 20]:
    cb = CollaborateBasedAlgo(user_based=False, k=k_neighbor, sim_func='pearson')
    eva.evaluate(algo=cb, k=k_list, n_jobs=6, split_date='2014-3-21', auto_log=True, debug=False)
