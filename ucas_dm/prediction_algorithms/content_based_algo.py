from .base_algo import BaseAlgo
import numpy as np
import pandas as pd
import faiss
import multiprocessing as mp
from multiprocessing import cpu_count


class ContentBasedAlgo(BaseAlgo):
    """
    Content-based prediction algorithm
    """

    def __init__(self, item_vector, dimension):
        """
        :param item_vector: Should be a pd.DataFrame contains item_id(integer) and its vector([float]) | id | vector |
        :param dimension: Vector's dimensions.
        """
        super().__init__()
        self._dimension = dimension
        self._item_vector = pd.DataFrame(item_vector)
        self._item_vector.columns = ['item_id', 'vec']
        self._retrieval_model = self._generate_retrieval_model()
        self._user_vector = {}
        self._user_log = None

    def train(self, train_set):
        """
        Main job is calculating user model for every user. Use multi-process to speed up the training.

        See :meth:`BaseAlog.train <base_algo.BaseAlgo.train>` for more details.
        """

        class TrainJob(mp.Process):
            def __init__(self, func, result_list, *args):
                super().__init__()
                self.func = func
                self.args = args
                self.res = result_list

            def run(self):
                self.res.append(self.func(*self.args))

        self._user_log = pd.DataFrame(train_set)
        self._user_log.columns = ['user_id', 'item_id']
        self._user_log.drop_duplicates(inplace=True)
        '''Calculate user model'''
        manager = mp.Manager()
        res_list = manager.list()
        user_ids = self._user_log['user_id'].drop_duplicates().values.tolist()
        part = 2
        cpus = cpu_count()
        job_list = []
        jobs = int(cpus / part)  # Use 1/2 of the cpus
        if jobs <= 0:
            jobs = 1
        part_ids_num = int((len(user_ids) + jobs - 1) / jobs)
        for i in range(jobs):
            part_ids = user_ids[i * part_ids_num:i * part_ids_num + part_ids_num]
            j = TrainJob(self._build_user_model, res_list, part_ids)
            job_list.append(j)
            j.start()
        for job in job_list:
            job.join()
        for ids_dict in res_list:
            for key in ids_dict.keys():
                self._user_vector[key] = ids_dict[key]
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
        specific_user_vec = self._user_vector[u_id]
        normal_specific_user_vec = ContentBasedAlgo._vector_normalize(np.array([specific_user_vec]).astype('float32'))
        ''' k+viewed_num make sure that we have at least k unseen items '''
        distance, index = self._retrieval_model.search(normal_specific_user_vec, k + viewed_num)
        item_res = self._item_vector.loc[index[0]]
        res = pd.DataFrame({'dis': distance[0], 'item_id': item_res['item_id']})
        res = res[~res['item_id'].isin(specific_user_log['item_id'])]
        res = res[:k]
        ''' return top-k smallest cosine distance and the ids of the items which hold that distance. '''
        return res['dis'].values.tolist(), res['item_id'].values.tolist()

    @classmethod
    def load(cls, fname):
        """
        See :meth:`BaseAlog.load <base_algo.BaseAlgo.load>` for more details.
        """
        res = super(ContentBasedAlgo, cls).load(fname)
        assert (hasattr(res, '_retrieval_model')), 'Not a standard ContentBasedAlgo class.'
        setattr(res, '_retrieval_model', faiss.read_index(fname + ".retrieval"))
        return res

    def save(self, fname, ignore=None):
        """
        See :meth:`BaseAlog.save <base_algo.BaseAlgo.save>` for more details.
        """
        if ignore is None:
            ignore = []
        ignore.append('_retrieval_model')
        faiss.write_index(self._retrieval_model, fname + ".retrieval")
        super().save(fname, ignore)

    def _generate_retrieval_model(self):
        """
        Use the retrieval model(faiss) to speed up the vector indexing

        :return: Ready-to-work retrieval model from faiss
        """
        real_vecs = self._item_vector['vec'].values.tolist()
        item_vector_array = np.array(real_vecs)
        item_vector_array = ContentBasedAlgo._vector_normalize(item_vector_array.astype('float32'))
        retrieval_model = faiss.IndexFlatIP(self._dimension)
        retrieval_model.add(item_vector_array)
        return retrieval_model

    def _build_user_model(self, user_ids):
        """
        This method will calculate user model for all users in user_ids.

        :param user_ids: users' id list
        :return: A dict contains user's id and vector.
        """
        res_dict = {}
        for user_id in user_ids:
            specific_user_log = self._user_log[self._user_log['user_id'] == user_id]
            log_vecs = pd.merge(specific_user_log, self._item_vector, how='left', on=['item_id'])
            assert (sum(log_vecs['vec'].notnull()) == log_vecs.shape[0]), 'Item vector sheet has null values'
            res_dict[user_id] = ContentBasedAlgo._calc_dim_average(np.array(log_vecs['vec'].values.tolist()))
        return res_dict

    def to_dict(self):
        pass

    @staticmethod
    def _calc_dim_average(vectors_array):
        """
        This func calculate the average value on every dimension of vectors_array, but it only count none-zero values.

        :param vectors_array: np.array contains a list of vectors.
        :return: A vector has the average value in every dimension.
        """
        array = np.array(vectors_array)
        threshold = 0.001
        res = array.sum(axis=0, dtype='float32')
        valid_count = (array > threshold).sum(axis=0)
        valid_count[valid_count == 0] = 1
        res /= valid_count
        return res

    @staticmethod
    def _vector_normalize(vectors_array):
        vector_len_list = np.sqrt((vectors_array ** 2).sum(axis=1, keepdims=True))
        # handle all-zero vectors
        vector_len_list[vector_len_list == 0] = 1
        res = vectors_array / vector_len_list
        return res
