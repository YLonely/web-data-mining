from Recommend.algo_set.content_based_algo import ContentBasedAlgo
from gensim import models
import pandas as pd
import numpy as np


class TopicBasedAlgo(ContentBasedAlgo):
    def __init__(self, item_vector=None, topic_n=100, corpus=None, id2word=None, chunksize=100, topic_type='lda',
                 power_iters=2, extra_samples=100, passes=1):
        """
        :param item_vector: See :meth:`ContentBasedAlgo.__init__ <content_based_algo.ContentBasedAlgo.__init__>` \
        for more details.
        :param topic_n: dimension. :meth:`ContentBasedAlgo.__init__ <content_based_algo.ContentBasedAlgo.__init__>` \
        for more details.
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
        self.__topic = topic_type
        self.__corpus = corpus
        self.__id2word = id2word
        self.__topic_model = None
        self.__chunksize = chunksize
        self.__power_iters = power_iters
        self.__extra_samples = extra_samples
        self.__passes = passes
        super().__init__(item_vector, topic_n)

    @classmethod
    def load(cls, fname):
        res = super(TopicBasedAlgo, cls).load(fname)
        assert (hasattr(res, '__topic')), 'Not a standard TopicBasedAlgo class.'
        if not getattr(res, '_initial'):
            topic = getattr(res, '__topic')
            if topic == 'lsi':
                model_fname = '.'.join([fname, 'lsi'])
                setattr(res, '__topic_model', models.LsiModel.load(model_fname))
            elif topic == 'lda':
                model_fname = '.'.join([fname, 'lda'])
                setattr(res, '__topic_model', models.LdaModel.load(model_fname))
        return res

    def save(self, fname, *args):
        ignore = ['__topic_model']
        super().save(fname, ignore)
        if self.__topic_model is not None:
            self.__topic_model.save('.'.join([fname, self.__topic]))

    def _item_vector_process(self, item_vector):
        """
        Use LDA or LSI algorithm to process the item vectors.

        See :meth:`ContentBasedAlgo._item_vector_process <content_based_algo.ContentBasedAlgo._item_vector_process>` \
        for more details.
        """
        if self.__topic == 'lsi':
            self.__topic_model = models.LsiModel(corpus=self.__corpus, num_topics=self._dim, id2word=self.__id2word,
                                                 chunksize=self.__chunksize, power_iters=self.__power_iters,
                                                 extra_samples=self.__extra_samples)
        elif self.__topic == 'lda':
            self.__topic_model = models.LdaModel(corpus=self.__corpus, num_topics=self._dim, id2word=self.__id2word,
                                                 chunksize=self.__chunksize, update_every=1, passes=self.__passes,
                                                 dtype=np.float64)
        else:
            raise ValueError(self.__topic)
        vecs = self.__topic_model[self.__corpus]
        pure_vecs = []
        for vec in vecs:
            if len(vec) != self._dim:
                pure_vecs.append(_rebuild_vector(vec, self._dim))
            else:
                pure_vecs.append([v for (index, v) in vec])
        return pd.DataFrame({'item_id': self._item_vector['item_id'], 'vec': pure_vecs})

    def to_dict(self):
        """
        See :meth:`BaseAlgo.to_dict <base_algo.BaseAlgo.to_dict>` for more details.
        """
        s = super().to_dict()
        res = {'type': self.__topic, 'topic_num': s['dimension'], 'chunksize': self.__chunksize}
        if self.__topic == 'lsi':
            res['power_iters'] = self.__power_iters
            res['extra_samples'] = self.__extra_samples
        else:
            res['passes'] = self.__passes
        return res


def _rebuild_vector(partial_vector, dim):
    res = [0] * dim
    for (index, value) in partial_vector:
        res[index] = value
    return res
