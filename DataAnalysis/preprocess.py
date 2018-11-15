# coding=utf-8
import pandas as pd
import jieba.posseg as pseg
import codecs
import pickle as pic


class PreProcessor:

    def __init__(self, original_data_path, generate_data_path):
        self.__generate_data_path = generate_data_path
        self.__original_data_path = original_data_path

    def extract_news(self, auto_save=True):
        """
        This func extract news from data and save them to csv file.

        :param auto_save: If true,func will automatically save news id with its content to a csv file.

        :return: A pd.DataFrame with two attributes: news_id and content
        """
        data = pd.read_csv(filepath_or_buffer=self.__original_data_path, sep="\\t", names=[
            'user_id', 'news_id', 'view_time', 'title', 'content', 'publish_time'], encoding="utf-8")
        data = data[['news_id', 'title', 'content']]
        data = data.fillna('')
        data = data.drop_duplicates().reset_index(drop=True)
        data['content'] = data['title'] + data['content']
        newsid_content = data[['news_id', 'content']]
        if auto_save:
            newsid_content.to_csv(self.__generate_data_path + "newsid_content.csv", index=False, encoding='utf-8')
        return newsid_content

    def generate_tokens(self, newsid_content, stop_words_path, auto_save=True):
        """
        This func generate tokens for news,and records the ids of short news.

        :param newsid_content: a pd.DataFrame of news id and its content
        :param stop_words_path: file path of chinese stop words
        :param auto_save: same above 'extract_news(..auto_save)'

        :return: a pair contains two stuff: a pd.DataFrame of news id and its tokens,and the news id of the short-content news
        """
        stop_words = codecs.open(stop_words_path, encoding='utf8').readlines()
        stop_words = [w.strip() for w in stop_words]
        stop_flags = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']

        def _tokenization(text):
            result = []
            words = pseg.cut(text)
            for word, flag in words:
                if flag not in stop_flags and word not in stop_words:
                    result.append(word)
            return result

        short_ids = []
        useful_ids = []
        res = []
        threshold = 2
        for i in range(newsid_content.shape[0]):
            news_id = newsid_content.loc[i, 'news_id']
            content = newsid_content.loc[i, 'content']
            result = _tokenization(content)
            if len(result) < threshold:
                short_ids.append(news_id)
            else:
                useful_ids.append(news_id)
                res.append(result)

        assert (len(useful_ids) == len(res)), "news ids length error"
        newsid_tokens = pd.DataFrame({'news_id': useful_ids, 'tokens': res})
        if auto_save:
            newsid_tokens.to_csv(self.__generate_data_path + 'newsid_tokens.csv', index=False, encoding='utf-8')
            f = open(self.__generate_data_path + 'short_ids.pdata', 'wb')
            pic.dump(short_ids, f)
            f.close()
        return [newsid_tokens, short_ids]

    def extract_view_log(self, news_to_avoid=[], auto_save=True):
        """
        This func extract user view log from data and save to csv file.
        Note that some news' content may be very short after segmentation. just erase them from user view log.

        :param news_to_avoid: content-short news id
        :param auto_save: same above

        :return: user view log
        """
        data = pd.read_csv(filepath_or_buffer=self.__original_data_path, sep="\\t", names=[
            'user_id', 'news_id', 'view_time', 'title', 'content', 'publish_time'], encoding="utf-8")
        userid_newsid = data[['user_id', 'news_id']]
        userid_newsid = userid_newsid.drop_duplicates().reset_index(drop=True)
        userid_newsid = userid_newsid[~userid_newsid['news_id'].isin(news_to_avoid)]
        if auto_save:
            userid_newsid.to_csv(self.__generate_data_path + "userid_newsid.csv", index=False, encoding='utf-8')
        return userid_newsid
