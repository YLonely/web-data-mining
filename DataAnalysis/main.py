# coding=utf-8
from Recommend.preprocess.preprocess import PreProcessor
from DataAnalysis.tf_idf_build import TfIdfGenerator

original_data_path = './datas/'
generate_data_path = './generate_datas/'
stop_words_path = './datas/stop.txt'
pp = PreProcessor(original_data_path+'data.txt', generate_data_path)
newsid_content = pp.extract_news()
[newsid_tokens, short_ids] = pp.generate_tokens(newsid_content, stop_words_path)
_ = pp.extract_view_log(short_ids)
gen = TfIdfGenerator(generate_data_path, generate_data_path + 'newsid_tokens.csv')
_ = gen.generate()


