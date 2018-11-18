from Recommend.algo_set.surprise_base_algo import SurpriseBaseAlgo
from surprise import prediction_algorithms


class BaseLineAlgo(SurpriseBaseAlgo):
    """
    A simple recommend algorithm that recommend items in random.
    Use it as a benchmark or a base-line.
    """

    def __init__(self):
        super().__init__()

    def _init_surprise_model(self):
        """
        See :meth:`SurpriseBaseAlgo._init_surprise_model <surprise_base_algo.SurpriseBaseAlgo._init_surprise_model>` for more details.


        :return: NormalPredictor() from surprise.
        """
        return prediction_algorithms.random_pred.NormalPredictor()

    def to_dict(self):
        """
        See :meth:`BaseAlgo.to_dict <base_algo.BaseAlgo.to_dict>` for more details.
        """
        return {'type': 'BaseLineAlgo'}
