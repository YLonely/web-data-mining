from .prediction_algorithms.base_algo import BaseAlgo
import pandas as pd
import json
import multiprocessing as mp
import time
import os


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
        self.__data_set.columns = ['user_id', 'item_id', 'view_time']
        self.__data_set['view_time'] = pd.to_datetime(self.__data_set['view_time'])
        self.__data_set.drop_duplicates(inplace=True)

    def evaluate(self, algo=BaseAlgo(), k=[], n_jobs=1, split_date='2000-1-1', debug=False, verbose=False,
                 auto_log=False):
        """
        :param algo: recommend algorithm
        :param k: list of integers represent the number of recommended items.
        :param n_jobs: The maximum number of evaluating in parallel. Use multi-thread to speed up the evaluating.
        :param split_date: on which date we split the log data into train and test.
        :param debug: if true, the evaluator will use 5000 instances in data set to run the test.
        :param verbose: whether to print the total time that evaluation cost.
        :param auto_log: if true, Evaluator will automatically save performance data to './performance.log'
        :return: average recall and precision
        """
        assert (n_jobs > 0), 'n_jobs must be greater than 0.'
        if debug:
            data_set = self.__data_set[:5000]
        else:
            data_set = self.__data_set
        train_set = data_set[data_set['view_time'] < split_date][['user_id', 'item_id']]
        test_set = data_set[data_set['view_time'] >= split_date][['user_id', 'item_id']]
        res = []
        start_time = time.time()
        algo.train(train_set)
        end_time1 = time.time()
        for _k in k:
            s_time = time.time()
            recall, precision = Evaluator._job_dispatch(algo, _k, train_set, test_set, n_jobs)
            e_time = time.time()
            res.append((recall, precision))
            if verbose:
                print("Totally cost: %.1f(s)" % (e_time - s_time + end_time1 - start_time))
            if auto_log:
                Evaluator._log_to_file(algo.to_dict(), recall=recall, precision=precision, k_recommend=_k,
                                       train_time=end_time1 - start_time, recommend_time=e_time - s_time, n_jobs=n_jobs,
                                       debug=debug)
        return res

    @staticmethod
    def _log_to_file(algo_dict, **kwargs):
        """
        This func will save algorithm's dict data and it's performance data to ./performance.log in json format.

        :param algo_dict: algorithm's dict format.
        :param kwargs: Performance data of the algorithm.
        """
        file_path = "./performance.log"
        print("Saving to " + file_path)
        for k in kwargs.keys():
            algo_dict[k] = kwargs[k]
        if os.path.exists(file_path):
            with open(file_path, 'r+', encoding='utf-8') as f:
                logs = json.load(f)
                f.seek(0)
                keys = logs.keys()
                last_index = int(list(keys)[-1].replace("record", ''))
                new_key = "record" + str(last_index + 1)
                logs[new_key] = algo_dict
                json.dump(logs, f, ensure_ascii=False, indent=4)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                logs = {"record1": algo_dict}
                json.dump(logs, f, ensure_ascii=False, indent=4)

    @staticmethod
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
            j = TestJob(Evaluator._one_round_test, res_list, algo, k, part_train_set, part_test_set)
            job_list.append(j)
            j.start()
        for job in job_list:
            job.join()
        # All processes finished here.
        for i in range(len(res_list)):
            recall.append(res_list[i][0])
            precision.append(res_list[i][1])
        return sum(recall) / len(recall), sum(precision) / len(precision)

    @staticmethod
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
