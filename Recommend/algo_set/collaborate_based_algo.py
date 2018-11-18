from surprise import prediction_algorithms
from Recommend.algo_set.surprise_base_algo import SurpriseBaseAlgo


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
        self._user_based = user_based
        self._sim_func = sim_func
        self._k = k
        super().__init__()

    def _init_surprise_model(self):
        """
        See :meth:`SurpriseBaseAlgo._init_surprise_model <surprise_base_algo.SurpriseBaseAlgo._init_surprise_model>` for more details.

        :return: KNNBasic() from surprise
        """
        sim_options = {'name': self._sim_func, 'user_based': self._user_based}
        return prediction_algorithms.KNNBasic(k=self._k, sim_options=sim_options)

    def to_dict(self):
        """
        See :meth:`BaseAlgo.to_dict <base_algo.BaseAlgo.to_dict>` for more details.
        """
        return {'type': 'Collaborative filtering', 'user_based': self._user_based, 'sim_fun': self._sim_func,
                'k': self._k}
