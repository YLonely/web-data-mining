from surprise import Dataset, Reader, dump
from .base_algo import BaseAlgo
import pandas as pd


class SurpriseBaseAlgo(BaseAlgo):
    """
        Do not use this class directly.
        This is the base class for all other sub-class which use the algorithms from
        Python recommend package--'Surprise'.
        Inherit from this base class will obtain some basic features.
    """

    def __init__(self):
        super().__init__()
        self._user_log = None
        self._surprise_model = None

    def train(self, train_set):
        if self._surprise_model is None:
            self._surprise_model = self._init_surprise_model()  # Initialize prediction model
        self._user_log = pd.DataFrame(train_set)
        self._user_log.columns = ['user_id', 'item_id']
        ''' Cause there is no rate in this situation, so just simply set rate to 1'''
        rate_log = self._user_log.copy()
        rate_log = rate_log.drop_duplicates()
        rate_log['rate'] = 1
        reader = Reader(rating_scale=(0, 1))
        train_s = Dataset.load_from_df(rate_log, reader)
        ''' train surprise-framework based model '''
        self._surprise_model.fit(train_s.build_full_trainset())
        return self

    def _init_surprise_model(self):
        """
        Sub-class should implement this method which return a prediction algorithm from package 'Surprise'.

        :return: A surprise-based recommend model
        """
        raise NotImplementedError()

    def top_k_recommend(self, u_id, k):
        specific_user_log = self._user_log[self._user_log['user_id'] == u_id]
        viewed_num = specific_user_log.shape[0]
        assert (viewed_num != 0), "User id doesn't exist"
        predict_rate_log = self._user_log.copy()
        predict_rate_log = predict_rate_log[['item_id']].drop_duplicates()
        predict_rate_log = predict_rate_log[~predict_rate_log['item_id'].isin(specific_user_log['item_id'])]
        predict_rate_log['prate'] = predict_rate_log.apply(lambda row: self.predict(u_id, row['item_id']), axis=1)
        predict_rate_log = predict_rate_log.sort_values(by=['prate'], ascending=False)
        predict_rate_log = predict_rate_log[:k]
        top_k_rate = predict_rate_log['prate'].values.tolist()
        top_k_item = predict_rate_log['item_id'].values.tolist()
        return top_k_rate, top_k_item

    def predict(self, u_id, i_id):
        """
        Predict the rate of user 'u_id' give to the item 'i_id'

        :param u_id: user id
        :param i_id: item id
        :return: rate value
        """
        _, _, _, est, _ = self._surprise_model.predict(u_id, i_id)
        return est

    def to_dict(self):
        raise NotImplementedError()

    @classmethod
    def load(cls, fname):
        res = super(SurpriseBaseAlgo, cls).load(fname)
        assert (hasattr(res, '_surprise_model')), 'Not a standard SurpriseBaseAlgo class.'
        setattr(res, '_surprise_model', dump.load(fname + '.surprise'))
        return res

    def save(self, fname, *args):
        if len(args) == 0:
            ignore = ['_surprise_model']
        else:
            ignore = args[0].append('_surprise_model')
        dump.dump(fname + '.surprise', algo=self._surprise_model)
        super().save(fname, ignore)
