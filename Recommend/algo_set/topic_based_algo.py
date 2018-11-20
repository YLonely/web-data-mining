from Recommend.algo_set.content_based_algo import ContentBasedAlgo
from gensim import models
import pandas as pd
import numpy as np


class TopicBasedAlgo(ContentBasedAlgo):
    def __init__(self, item_ids, topic_n=100, corpus=None, id2word=None, chunksize=100, topic_type='lda',
                 power_iters=2, extra_samples=100, passes=1):
        """
        :param item_ids: A list or array contains items' ids.
        :param topic_n: The number of requested latent topics to be extracted from the training corpus.
        :param corpus: Stream of document vectors or sparse matrix of shape (num_terms, num_documents).used by LSI \
        or LDA model.
        :param id2word: ID to word mapping ,used by LSI or LDA model.
        :param chunksize: Number of documents to be used in each training chunk.
        :param topic_type: 'lsi' or 'lda'
        :param power_iters: (**LSI parameter**)Number of power iteration steps to be used. Increasing the number \
        of power iterations improves accuracy, but lowers performance.
        :param extra_samples: (**LSI parameter**)Extra samples to be used besides the rank k. Can improve accuracy.
        :param passes: (**LDA parameter**)Number of passes through the corpus during training.
        """
        self._item_ids = item_ids
        self._topic_n = topic_n
        self._topic = topic_type
        self._corpus = corpus
        self._id2word = id2word
        self._topic_model = None
        self._chunksize = chunksize
        self._power_iters = power_iters
        self._extra_samples = extra_samples
        self._passes = passes
        item_vectors = self._generate_item_vector()
        super().__init__(item_vectors, topic_n)

    @classmethod
    def load(cls, fname):
        res = super(TopicBasedAlgo, cls).load(fname)
        assert (hasattr(res, '_topic')), 'Not a standard TopicBasedAlgo class.'
        topic = getattr(res, '_topic')
        if topic == 'lsi':
            model_fname = '.'.join([fname, 'lsi'])
            setattr(res, '_topic_model', models.LsiModel.load(model_fname))
        elif topic == 'lda':
            model_fname = '.'.join([fname, 'lda'])
            setattr(res, '_topic_model', models.LdaModel.load(model_fname))
        return res

    def save(self, fname, *args):
        ignore = ['_topic_model']
        if self._topic_model is not None:
            self._topic_model.save('.'.join([fname, self._topic]))
        super().save(fname, ignore)

    def _generate_item_vector(self):
        """
        Use LDA or LSI algorithm to process TF-IDF vector and generate new item vectors.

        :return: DataFrame contains item id and it's new vector
        """
        if self._topic == 'lsi':
            self._topic_model = models.LsiModel(corpus=self._corpus, num_topics=self._topic_n,
                                                id2word=self._id2word, chunksize=self._chunksize,
                                                power_iters=self._power_iters, extra_samples=self._extra_samples)
        elif self._topic == 'lda':
            self._topic_model = models.LdaModel(corpus=self._corpus, num_topics=self._topic_n,
                                                id2word=self._id2word, chunksize=self._chunksize,
                                                update_every=1, passes=self._passes, dtype=np.float64)
        else:
            raise ValueError(self._topic)
        vecs = self._topic_model[self._corpus]
        pure_vecs = []
        for vec in vecs:
            if len(vec) != self._topic_n:
                pure_vecs.append(_rebuild_vector(vec, self._topic_n))
            else:
                pure_vecs.append([v for (index, v) in vec])
        return pd.DataFrame({'item_id': self._item_ids, 'vec': pure_vecs})

    def to_dict(self):
        """
        See :meth:`BaseAlgo.to_dict <base_algo.BaseAlgo.to_dict>` for more details.
        """
        res = {'type': self._topic, 'topic_num': self._topic_n, 'chunksize': self._chunksize}
        if self._topic == 'lsi':
            res['power_iters'] = self._power_iters
            res['extra_samples'] = self._extra_samples
        else:
            res['passes'] = self._passes
        return res


def _rebuild_vector(partial_vector, dim):
    res = [0] * dim
    for (index, value) in partial_vector:
        res[index] = value
    return res
