from .surprise_base_algo import SurpriseBaseAlgo
from surprise.prediction_algorithms import KNNBasic
import pandas as pd
import math
from surprise import Dataset, Reader


class CollaborateBasedAlgo(SurpriseBaseAlgo):
    """
    Collaborative filtering algorithm.
    """

    def __init__(self, sim_func='cosine', user_based=True, k=1):
        """
        :param sim_func: similarity function: 'cosine','msd','pearson','pearson_baseline'
        :param user_based: True--> user-user filtering strategy;False--> item-item filtering strategy
        :param k: The (max) number of neighbors to take into account for aggregation
        """
        super().__init__()
        self._user_based = user_based
        self._sim_func = sim_func
        self._k = k

    def train(self, train_set):
        # News recommendation is a typical case that use users' implicit feedback to give recommendations, train set
        # only contains binary or unary data (1 for seen, 0 for unseen). According to some papers, normalizing user
        # vectors to unit vectors will increase the accuracy of recommending with binary data.
        if self._surprise_model is None:
            self._surprise_model = self._init_surprise_model()
        train_set = pd.DataFrame(train_set)
        train_set.columns = ['user_id', 'item_id']
        self._user_log = train_set.copy()
        train_set = train_set.drop_duplicates()
        groups = train_set.groupby(['user_id'])
        id_to_group_size = {}
        for user_id, group in groups:
            id_to_group_size[user_id] = group.shape[0]
        train_set['rate'] = 1
        train_set['rate'] = train_set.apply(lambda row: 1 / math.sqrt(id_to_group_size[row['user_id']]), axis=1)
        reader = Reader(rating_scale=(0, 1))
        train_s = Dataset.load_from_df(train_set, reader)
        ''' train surprise-framework based model '''
        self._surprise_model.fit(train_s.build_full_trainset())
        return self

    def _init_surprise_model(self):
        sim_options = {'name': self._sim_func, 'user_based': self._user_based}
        return KNNBasic(k=self._k, sim_options=sim_options)

    def to_dict(self):
        """
        See :meth:`BaseAlgo.to_dict <base_algo.BaseAlgo.to_dict>` for more details.
        """
        return {'type': 'Collaborative filtering', 'user_based': self._user_based, 'sim_fun': self._sim_func,
                'k': self._k}
