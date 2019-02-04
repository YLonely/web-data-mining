from .base_algo import BaseAlgo
import numpy as np
import pandas as pd


class BaseLineAlgo(BaseAlgo):
    """
    A simple recommend algorithm that recommend items in random.
    Use it as a base-line.
    """

    def __init__(self):
        super().__init__()
        self._user_log = None

    def train(self, train_set):
        self._user_log = pd.DataFrame(train_set)
        self._user_log.columns = ['user_id', 'item_id']

    def top_k_recommend(self, u_id, k):
        specific_user_log = self._user_log[self._user_log['user_id'] == u_id]
        viewed_num = specific_user_log.shape[0]
        assert (viewed_num != 0), "User id doesn't exist"
        predict_rate_log = self._user_log.copy()
        predict_rate_log = predict_rate_log[['item_id']].drop_duplicates()
        predict_rate_log = predict_rate_log[~predict_rate_log['item_id'].isin(specific_user_log['item_id'])]
        predict_rate_log['prate'] = np.random.rand(predict_rate_log.shape[0])
        predict_rate_log = predict_rate_log.sort_values(by=['prate'], ascending=False)
        predict_rate_log = predict_rate_log[:k]
        top_k_rate = predict_rate_log['prate'].values.tolist()
        top_k_item = predict_rate_log['item_id'].values.tolist()
        return top_k_rate, top_k_item

    def to_dict(self):
        """
        See :meth:`BaseAlgo.to_dict <base_algo.BaseAlgo.to_dict>` for more details.
        """
        return {'type': 'BaseLineAlgo'}
