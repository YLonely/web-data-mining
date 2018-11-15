from Recommend.algo_set.content_based_algo import ContentBasedAlgo
from gensim import models
import pandas as pd


class TopicBasedAlgo(ContentBasedAlgo):
    def __init__(self, item_vector, topic_n, corpus, id2word, topic_type, alpha='auto', eta='auto'):
        """
        :param item_vector: See :meth:`ContentBasedAlgo.__init__ <content_based_algo.ContentBasedAlgo.__init__>` for more details.
        :param topic_n: dimension. :meth:`ContentBasedAlgo.__init__ <content_based_algo.ContentBasedAlgo.__init__>` for more details.
        :param corpus: Stream of document vectors or sparse matrix of shape (num_terms, num_documents).used by LSI or LDA model.
        :param id2word: ID to word mapping ,used by LSI or LDA model.
        :param topic_type: 'lsi' or 'lda'
        :param alpha: parameter of LDA algorithm
        :param eta: parameter of LDA algorithm
        """
        self.__topic = topic_type
        self.__corpus = corpus
        self.__id2word = id2word
        self.__topic_model = None
        self.__alpha = alpha
        self.__eta = eta
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

        See :meth:`ContentBasedAlgo._item_vector_process <content_based_algo.ContentBasedAlgo._item_vector_process>` for more details.
        """
        if self.__topic == 'lsi':
            self.__topic_model = models.LsiModel(corpus=self.__corpus, num_topics=self._dim,
                                                 id2word=self.__id2word)
        elif self.__topic == 'lda':
            self.__topic_model = models.LdaModel(corpus=self.__corpus, num_topics=self._dim,
                                                 id2word=self.__id2word, alpha=self.__alpha, eta=self.__eta)
        else:
            raise ValueError(self.__topic)
        vecs = self.__topic_model[self.__corpus]
        pure_vecs = []
        for vec in vecs:
            if len(vec) != self._dim:
                pure_vecs.append([1e-8] * self._dim)
            else:
                pure_vecs.append([v for (index, v) in vec])
        return pd.DataFrame({'item_id': self._item_vector['item_id'], 'vec': pure_vecs})
