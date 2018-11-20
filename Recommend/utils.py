from sklearn.model_selection import KFold
from Recommend.algo_set.base_algo import BaseAlgo
import pandas as pd
import json
import multiprocessing as mp
import time


class Evaluator:
    """
    This class provide some methods to evaluate the performance of a recommend algorithm.
    Two measures are supported for now: Recall and Precision
    """

    def __init__(self, data_set):
        """
        :param data_set: User view log.
        """
        self.__data_set = pd.DataFrame(data_set)
        self.__data_set.columns = ['user_id', 'item_id']
        self.__data_set.drop_duplicates(inplace=True)

    def evaluate(self, algo=BaseAlgo(), k=1, n_splits=2, shuffle=False, random_state=None, n_jobs=1, debug=False,
                 verbose=False, auto_log=False):
        """
        :param algo: recommend algorithm
        :param k: recommend top-k items
        :param n_splits: means n-flod evaluation
        :param shuffle: Whether to shuffle the data before splitting into batches.
        :param random_state: random seed
        :param n_jobs: The maximum number of evaluating in parallel. Use multi-thread to speed up the evaluating.
        :param debug: if true, the evaluator will use 5000 instances in data set to run the test.
        :param verbose: whether to print recall and precision value in every test round.
        :param auto_log: if true, Evaluator will automatically save performance data to './performance.log'
        :return: average recall and precision
        """
        assert (n_jobs > 0), 'n_jobs must be greater than 0.'
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        recall_log = []
        precision_log = []
        if debug:
            data_set = self.__data_set[:5000]
        else:
            data_set = self.__data_set
        i = 1
        start_time = time.time()
        for train, test in kf.split(data_set):
            train_set = data_set.iloc[train, :]
            test_set = data_set.iloc[test, :]
            # print("training.", flush=True)
            algo.train(train_set)
            # print("training over.", flush=True)
            recall, precision = _job_dispatch(algo, k, train_set, test_set, n_jobs)
            if verbose:
                print("Round %d finish. Recall: %.3f%% Precision: %.3f%%" % (i, 100 * recall, 100 * precision))
                i = i + 1
            recall_log.append(recall)
            precision_log.append(precision)
        end_time = time.time()
        if verbose:
            print("Totally cost: %.1f(s)" % (end_time - start_time))
        rec = sum(recall_log) / len(recall_log)
        pre = sum(precision_log) / len(precision_log)
        if auto_log:
            _log_to_file(algo.to_dict(), recall=rec, precision=pre, shuffle=shuffle, random_state=random_state,
                         time_cost=end_time - start_time)
        return rec, pre


def _log_to_file(algo_dict, **kwargs):
    """
    This func will save algorithm's dict data and it's performance data to ./performance.log in json format.

    :param algo_dict: algorithm's dict format.
    :param kwargs: Performance data of the algorithm.
    """
    print("Saving to ./performance.log......")
    for k in kwargs.keys():
        algo_dict[k] = kwargs[k]
    with open('./performance.log', 'a', encoding='utf-8') as f:
        json.dump(algo_dict, f, ensure_ascii=False, indent=4)


def _job_dispatch(algo, k, train_set, test_set, n_jobs):
    class TestJob(mp.Process):
        def __init__(self, func, result_list, *args):
            super().__init__()
            self.func = func
            self.args = args
            self.res = result_list

        def run(self):
            self.res.append(self.func(*self.args))

    manager = mp.Manager()
    res_list = manager.list()
    user_ids = train_set[['user_id']].drop_duplicates()
    user_num = user_ids.shape[0]
    part_num = int((user_num + n_jobs - 1) / n_jobs)
    job_list = []
    recall = []
    precision = []
    for i in range(n_jobs):
        part_users = user_ids[i * part_num:i * part_num + part_num]
        part_train_set = train_set[train_set['user_id'].isin(part_users['user_id'])]
        part_test_set = test_set[test_set['user_id'].isin(part_users['user_id'])]
        j = TestJob(_one_round_test, res_list, algo, k, part_train_set, part_test_set)
        job_list.append(j)
        j.start()
    for job in job_list:
        job.join()
    # All processes finished here.
    for i in range(len(res_list)):
        recall.append(res_list[i][0])
        precision.append(res_list[i][1])
    return sum(recall) / len(recall), sum(precision) / len(precision)


def _one_round_test(algo, k, train_set, test_set):
    user_id_series = train_set['user_id'].drop_duplicates()
    recall_log = []
    precision_log = []
    for user_id in user_id_series:
        user_viewed_in_test = test_set[test_set['user_id'] == user_id].drop_duplicates()
        if user_viewed_in_test.shape[0] == 0:
            continue
        _, recommend = algo.top_k_recommend(user_id, k)
        recommend = pd.DataFrame({'item_id': recommend})
        assert (recommend.shape[0] > 0), 'Recommend error'
        cover_number = (recommend[recommend['item_id'].isin(user_viewed_in_test['item_id'])]).shape[0]
        recall_log.append(cover_number / user_viewed_in_test.shape[0])
        precision_log.append(cover_number / recommend.shape[0])
    recall = sum(recall_log) / len(recall_log)
    precision = sum(precision_log) / len(precision_log)
    return recall, precision
