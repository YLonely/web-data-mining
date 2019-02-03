import pickle as pic


class BaseAlgo:
    """
    Do not use this class directly.
    The interface of all recommend algorithms.
    """

    def __init__(self):
        pass

    def train(self, train_set):
        """
        Do some train-set-dependent work here: for example calculate sims between users or items

        :param train_set: A pandas.DataFrame contains two attributes: user_id and item_id,which \
        represents the user view record during a period of time.
        :return: return a model that is ready to give recommend
        """
        raise NotImplementedError()

    def top_k_recommend(self, u_id, k):
        """
        Calculate the top-K recommend items

        :param u_id: users' identity (user's id)
        :param k: the number of the items that the recommender should return
        :return: (v,id) v is a list contains predict rate or distance, id is a list contains top-k highest rated or \
        nearest items
        """
        raise NotImplementedError()

    def to_dict(self):
        """
        Convert algorithm model to a dict which contains algorithm's type and it's main hyper-parameters.

        :return: A dict contains type and hyper-parameters.
        """
        raise NotImplementedError()

    @classmethod
    def load(cls, fname):
        """
        Load an object previously saved from a file

        :param fname: file path
        :return: object loaded from file
        """
        with open(fname, 'rb') as f:
            obj = pic.load(f)
        return obj

    def save(self, fname, ignore=None):
        """
        Save an object to a file.

        :param fname: file path
        :param ignore: a set of attributes that should't be saved by super class, but subclass may have to handle \
        these special attributes.
        """
        if ignore is not None:
            for attr in ignore:
                if hasattr(self, attr):
                    setattr(self, attr, None)
        with open(fname, 'wb') as f:
            pic.dump(self, f)
