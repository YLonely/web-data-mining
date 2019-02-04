from .surprise_base_algo import SurpriseBaseAlgo
from surprise.prediction_algorithms import matrix_factorization


class NMF(SurpriseBaseAlgo):
    """
    A collaborative filtering algorithm based on Non-negative Matrix Factorization.
    """

    def __init__(self, n_factors=15, n_epochs=80, random_state=0, reg_pu=0.05, reg_qi=0.05):
        """
        :param n_factors: The number of factors. Default is 20.
        :param n_epochs: The number of iteration of the SGD procedure. Default is 20.
        :param random_state: random_state (int, RandomState instance from numpy, or None) – Determines the RNG that \
         will be used for initialization. If int, random_state will be used as a seed for a new RNG. This is useful to \
         get the same initialization over multiple calls to fit(). If RandomState instance, this same instance is used \
         as RNG. If None, the current RNG from numpy is used. Default is 0.
        :param reg_pu:  The regularization term for users λu. Default is 0.05.
        :param reg_qi: The regularization term for items λi. Default is 0.05.
        """
        super().__init__()
        self._n_factors = n_factors
        self._n_epochs = n_epochs
        self._random_state = random_state
        self._reg_pu = reg_pu
        self._reg_qi = reg_qi
        self._surprise_model = self._init_surprise_model()

    def _init_surprise_model(self):
        return matrix_factorization.NMF(n_factors=self._n_factors, random_state=self._random_state,
                                        n_epochs=self._n_epochs, reg_pu=self._reg_pu, reg_qi=self._reg_qi)

    def to_dict(self):
        """
        See :meth:`BaseAlgo.to_dict <base_algo.BaseAlgo.to_dict>` for more details.
        """
        return {'type': 'NMF', 'factors': self._n_factors, 'epochs': self._n_epochs,
                'random_state': self._random_state, 'reg_pu': self._reg_pu, 'reg_qi': self._reg_qi}