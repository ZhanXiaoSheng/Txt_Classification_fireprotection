# Txt_Classification_fireprotection
这是jupyter版本 后面的CNN版的batch是pycharm版，有一些细节没有上传

消防警情报警信息文本分类，主要步骤如下：
1.提取文本信息
2.去停用词，分词 test_pre.py(text_som的前半部分即)
3.生成词向量（减少训练参数和训练时间）word2vec.py
4.分割数据集（8：2分为训练集和验证集），label为3类
5.建模，分别用LSTM、CNN、Bi-LSTM、LSTM+Attention比较
6.作图
