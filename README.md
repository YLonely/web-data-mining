# UCAS-DM
![](https://img.shields.io/badge/version-1.0.0-green.svg)
![](https://img.shields.io/badge/docs-passing-green.svg)
![](https://img.shields.io/badge/python-3.x-blue.svg)
![](https://img.shields.io/badge/License-MIT-blue.svg)

介绍
--------
UCAS-DM(UCAS-DataMining)是一个较为简易的推荐算法库，专门为国科大-网络数据挖掘的**新闻推荐**课程大作业所设计，其主要包含了数据预处理，推荐算法以及算法性能评测三个部分，该算法库旨在为使用者提供更加方便的数据处理与分析的接口，让使用者能够将精力更加专注于数据的分析以及算法的调整上。

完整的API文档请点这里[Docs](http://YLonely.github.io/web-data-mining)

环境依赖
--------
### 系统环境
由于本算法库使用到了[Faiss](https://github.com/facebookresearch/faiss)库用于推荐算法的加速，而Faiss目前仅支持Linux与OSX，因此请在*nix的环境下使用本算法库。
### Python版本
本Python包在Python3.6.5的环境下运行通过，因此不支持Python2.x的Python版本，推荐在Python3.5~3.6的环境下使用本算法库。

### 推荐的环境搭建方式:white_check_mark:
1. 安装相应版本的Anaconda以获得Python环境以及所需的包(例如numpy，pandas等)。点击[这里](https://repo.continuum.io/archive/)获得历史版本的安装包
2. 根据[指导](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md)通过conda安装Faiss包(CPU版本即可)。

安装
------
直接通过pip进行安装即可
```
pip install ucas-dm==1.0.0
```
安装本库时会自动检查并安装(若缺少)`numpy`，`pandas`，`gensim`，`jieba`，`scikit-surprise`等库，但并不会自动检查安装`Faiss`。

简单教程
----
### 数据预处理
```python
import pandas as pd
from ucas_dm.preprocess import Preprocessor

path_to_source_data = ".."
pp = Preprocessor(path_to_source_data)
news_id_and_its_content =  pp.extract_news()
news_id_and_its_content.to_csv(path_or_buf = "./news_id_and_its_content.csv", index = False)

user_logs = pp.extract_logs()
user_logs.to_csv(path_or_buf = "./user_logs.csv", index = False)
```
`Preprocessor`类对外提供了`extract_news`，`generate_tokens`，`build_tf_idf`，以及`extract_logs`方法，并不推荐直接使用`generate_tokens`以及`build_tf_idf`方法，这两个方法会被推荐算法库中的算法调用。`extract_news`以及`extract_logs`方法分别从原始数据中抽取出新闻内容以及用户的浏览历史，返回的均是`pandas.DataFrame`类型的数据，以上的两个方法均默认原始数据具有以下的形式:

| 用户id | 新闻id | 浏览时间 | 标题  | 新闻内容 | 发布时间 |
| :----: | :----: | :------: | :---: | :------: | :------: |
|   ……   |   ……   |    ……    |  ……   |    ……    |    ……    |

数据预处理的部分并不具有很好的可重用性，因此如果原始数据有更改，则需要修改数据或者预处理的代码。本项目的*source_data*文件夹中已包含了从原始数据中抽取出的新闻内容与用户浏览记录，可以直接使用。

### 使用推荐算法
```python
import pandas as pd
from ucas_dm.prediction_algorithms import CollaborateBasedAlgo, TopicBasedAlgo

k_items = 35
user_logs = pd.read_csv("./user_logs.csv")
k_neighbors = 10
cb = CollaborateBasedAlgo(user_based = True, k = k_neighbors)
cb.train(user_logs)
recommend1 = cb.top_k_recommend(u_id = 12345, k = k_items)[0]
cb.save("./cb.model")

news_id_and_its_content = pd.read_csv("./news_id_and_its_content.csv")
initial_params = TopicBasedAlgo.preprocess(news_id_and_its_content)
lsi = TopicBasedAlgo(initial_params = initial_params, topic_n = 100, topic_type = 'lsi', chunksize = 100)
lsi.train(user_logs)
recommend2 = lsi.top_k_recommend(u_id = 12345, k = k_items)[0]
```
所有的推荐算法都被集中到了`prediction_algorithms`包中，目前可以直接使用的推荐算法有BaseLineAlgo(随机推荐算法，可以作为基准),CollaborateBasedAlgo(协同过滤推荐算法),NMF(基于非负矩阵分解的协同过滤),TopicBasedAlgo(使用了话题模型的基于内容推荐算法)，这些算法均直接或间接地实现了BaseAlgo接口，这些算法在初始化之后，使用之前均需要调用`train`方法并传入用户浏览历史进行训练，之后调用`top_k_recommend`方法获得针对某用户的前`k`个推荐的物品(`top_k_recommend`返回的是推荐的前`k`个物品的id列表以及这`k`个物品推荐度的列表，推荐度的解释根据不同的推荐算法而不同，例如在协同过滤推荐算法中，推荐度是相似度与相似用户或物品评分的积，在基于内容推荐算法中，推荐度是被推荐物品与用户兴趣模型的相似度)，在使用上TopicBasedAlgo稍有不同，它需要调用`preprocess`方法对数据进行进一步处理，得到初始化时所必须的参数`initial_params`，该参数可以保存，也可以从文件中读取，实际类型来源于`prediction_algorithms.InitialParams`。所有的算法模型都可以使用`save`和`load`进行存取。

### 算法评价
```python
from ucas_dm.utils import Evaluator
from ucas_dm.prediction_algorithms import CollaborateBasedAlgo
import pandas as pd

k_list = [5, 10, 15]
k_neighbors = 30
user_logs = pd.read_csv("./user_logs.csv")
cb = CollaborateBasedAlgo(user_based = True, k = k_neighbors)
eva.evaluate(algo = cb, k = k_list, n_jobs = 2, split_date = '2014-3-21', auto_log = True)
```
使用`utils`包中的`Evaluator`可以方便地对不同的推荐模型在不同的数据集上的推荐性能进行评测，`Evaluator`使用`evaluate`方法进行评测，该方法的`k`参数表示在推荐时为每位用户推荐多少物品，该参数允许传入一个列表，这样可以方便地对多个参数进行评测。目前`Evaluator`仅允许按照浏览时间对数据集进行划分，即划分为训练集与测试集。若自动记录`auto_log`打开，`evaluate`方法会自动记录推荐算法的性能参数，以json的格式存储到`./performance.log`文件中。

反馈
-----
欢迎反馈，如果有bug什么的我尽量修改:blush: