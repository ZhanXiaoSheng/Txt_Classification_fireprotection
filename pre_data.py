import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
'''
    读取分词好的数据
    data2.txt
'''

corpus = []
Embedding_dim = 200
max_length = 50
for line in open('C:\\Users\\zhangsheng\\Desktop\\DSjjcl\\文本分类（报警信息）\\data2.txt','r').readlines():
    corpus.append(line.strip())

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)

word_index = tokenizer.word_index
#  将句子转换成数字，最大长度为50
data_oh = pad_sequences(sequences=sequences, maxlen=max_length, padding='post')

#  导入案件数据，提取标签y
filepath = 'C:\\Users\\zhangsheng\\Desktop\\DSjjcl\\version2\\new_all_df.csv'
all_df = pd.read_csv(filepath, encoding='gb18030')
labEncode = LabelEncoder()
all_df['案件类型'] = labEncode.fit_transform(all_df['案件类型'])
new_df = all_df['案件类型']
all_labels = new_df.values

#  转化为one-hot形式
labels = to_categorical(all_labels)

#  分割数据集
msk = np.random.rand(len(new_df)) < 0.8
train = new_df[msk]
test = new_df[~msk]

index = train.index
x_train = data_oh[index][0:40000]
x_val = data_oh[index][40000:]

y_train = labels[index][0:40000]
y_val = labels[index][40000:]

test_index = test.index
x_test = data_oh[test_index]
y_test = labels[test_index]

#print(len(word_index))
# print(corpus[:5])
# print(new_df[:5])
