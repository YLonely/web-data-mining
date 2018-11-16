from Recommend.algo_set.surprise_base_algo import SurpriseBaseAlgo
from surprise.prediction_algorithms import matrix_factorization


class SVDpp(SurpriseBaseAlgo):
    """
    The SVD++ algorithm, an extension of SVD taking into account implicit ratings.
    """

    def __init__(self, n_factors=20, n_epochs=20, init_mean=0, init_std_dev=0.1, lr_all=0.007, reg_all=0.2):
        """
        :param n_factors: The number of factors. Default is 20.
        :param n_epochs: The number of iteration of the SGD procedure. Default is 20.
        :param init_mean: The mean of the normal distribution for factor vectors initialization. Default is 0.
        :param init_std_dev: The standard deviation of the normal distribution for factor vectors initialization. Default is 0.1.
        :param lr_all: The learning rate for all parameters. Default is 0.007.
        :param reg_all: The regularization term for all parameters. Default is 0.02.
        """
        self._n_factors = n_factors
        self._n_epochs = n_epochs
        self._init_mean = init_mean
        self._init_std_dev = init_std_dev
        self._lr_all = lr_all
        self._reg_all = reg_all
        super().__init__()

    def _init_surprise_model(self):
        """
        See :meth:`SurpriseBaseAlgo._init_surprise_model <surprise_base_algo.SurpriseBaseAlgo._init_surprise_model>` for more details.

        :return: SVDpp() from surprise
        """
        return matrix_factorization.SVDpp(self._n_factors, self._n_epochs, self._init_mean, self._init_std_dev,
                                          self._lr_all, self._reg_all)
