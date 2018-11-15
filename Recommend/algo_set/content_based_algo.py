from Recommend.algo_set.base_algo import BaseAlgo
import numpy as np
import pandas as pd
import faiss


class ContentBasedAlgo(BaseAlgo):
    """
    Do not use this class directly.
    The base class of content-based algorithms
    """

    def __init__(self, item_vector, dimension):
        """
        :param item_vector: Should be a pd.DataFrame contains item_id and its' vectors
        :param dimension: Vector's dimensions. Cause item vectors are usually TF-IDF vectors which may have different dimensions among them, sub-class should at least handle this situation and make all the vectors be in same dimensions by implementing method _item_vector_process()
        """
        super().__init__()
        self._dim = dimension
        self._item_vector = pd.DataFrame(item_vector)
        self._item_vector.columns = ['item_id', 'vec']
        self._item_vector = self._item_vector_process(self._item_vector)
        self._retrieval_model = self._generate_retrieval_model()
        self._user_vector = None
        self._user_log = None

    def train(self, train_set):
        """
        Calculate user model for every user.

        See :meth:`BaseAlog.train <base_algo.BaseAlgo.train>` for more details.
        """
        self._user_log = pd.DataFrame(train_set)
        self._user_log.columns = ['user_id', 'item_id']
        '''Calculate user model'''
        user_ids = self._user_log['user_id'].drop_duplicates().values.tolist()
        user_vecs = [self._build_user_model(id) for id in user_ids]
        self._user_vector = pd.DataFrame({'user_id': user_ids, 'vec': user_vecs})
        return self

    def top_k_recommend(self, u_id, k):
        """
        See :meth:`BaseAlog.top_k_recommend <base_algo.BaseAlgo.top_k_recommend>` for more details.
        """
        if self._retrieval_model is None:
            raise RuntimeError('Run method train() first.')
        specific_user_log = self._user_log[self._user_log['user_id'] == u_id]
        viewed_num = specific_user_log.shape[0]
        assert (viewed_num != 0), "User id doesn't exist."
        specific_user_vec = self._user_vector[self._user_vector['user_id'] == u_id]['vec'].values[0]
        specific_user_vec = np.array(specific_user_vec)
        normal_specific_user_vec = _vector_normalize(np.array([specific_user_vec]).astype('float32'))
        ''' k+viewed_num make sure that we have at least k unseen items '''
        distance, index = self._retrieval_model.search(normal_specific_user_vec, k + viewed_num)
        item_res = self._item_vector.loc[index[0]]
        res = pd.DataFrame({'dis': distance[0], 'item_id': item_res['item_id']})
        res = res[~res['item_id'].isin(specific_user_log['item_id'])]
        res = res[:k]
        return res['dis'].values.tolist(), res['item_id'].values.tolist()

    @classmethod
    def load(cls, fname):
        """
        See :meth:`BaseAlog.load <base_algo.BaseAlgo.load>` for more details.
        """
        res = super(ContentBasedAlgo, cls).load(fname)
        assert (hasattr(res, '_retrieval_model')), 'Not a standard ContentBasedAlgo class.'
        res._retrieval_model = res._generate_retrieval_model()
        return res

    def save(self, fname, ignore=None):
        """
        See :meth:`BaseAlog.save <base_algo.BaseAlgo.save>` for more details.
        """
        if ignore is None:
            ignore = []
        ignore = ignore.append('_retrieval_model')
        super().save(fname, ignore)

    def _generate_retrieval_model(self):
        """
        Use some retrieval model(faiss) to speed up the vector indexing

        :return: Ready-to-work retrieval model from faiss
        """
        real_vecs = self._item_vector['vec'].values.tolist()
        item_vector_array = np.array(real_vecs)
        item_vector_array = _vector_normalize(item_vector_array.astype('float32'))
        retrieval_model = faiss.IndexFlatIP(self._dim)
        retrieval_model.add(item_vector_array)
        return retrieval_model

    def _build_user_model(self, user_id):
        """
        This method builds vector model for a user.

        :param user_id: user id
        :return: user's vector model in numpy.array type.
        """
        specific_user_log = self._user_log[self._user_log['user_id'] == user_id]
        specific_user_log.drop_duplicates(inplace=True)
        log_vecs = pd.merge(specific_user_log, self._item_vector, how='left', on=['item_id'])
        assert (sum(log_vecs['vec'].notnull()) == log_vecs.shape[0]), 'Item vector sheet has null values'
        return _calc_cos_dis_median(log_vecs['vec'].values)

    def _item_vector_process(self, item_vector):
        """
        Cause item vectors are usually TF-IDF vectors which may have different dimensions among them,
        sub-class should at least handle this situation and make all the vectors be in same
        dimensions by implementing method _item_vector_process()

        :param item_vector: A pd.DataFrame contains items id and its' vector
        :return: A pd.DataFrame of items' id and its' processed vectors
        """
        pass


def _vector_normalize(vectors_array):
    vector_len_list = np.sqrt((vectors_array ** 2).sum(axis=1, keepdims=True))
    res = vectors_array / vector_len_list
    return res


def _calc_cos_dis_median(vectors_array):  # np.array
    """
    Like the geometric median calculation,but use a different distance function.
    This func find a median in the vector space that have the smallest sum of distance
    among other vectors in vectors_array. it uses cosine value to indicate distance
    """
    # normalization
    vectors_array = np.array(vectors_array.tolist())
    normal_vectors_array = _vector_normalize(vectors_array)
    sqrt_sum = np.sqrt(((normal_vectors_array.sum(axis=0)) ** 2).sum())
    part_sum = normal_vectors_array.sum(axis=0)
    x1 = part_sum / sqrt_sum
    x2 = -x1
    distance_sum1 = (normal_vectors_array * x1).sum()
    distance_sum2 = (normal_vectors_array * x2).sum()
    if distance_sum1 > distance_sum2:
        res = x1
    else:
        res = x2
    return res
