import gensim
from gensim.models import word2vec, Word2Vec
import numpy as np
from keras.layers import Embedding
import pre_data

word_index = pre_data.word_index
Embedding_dim = pre_data.Embedding_dim
max_length = pre_data.max_length
word2vec_model = Word2Vec.load('word2vec.model')
embedding_matrix = np.zeros((len(word_index)+1, Embedding_dim))
print(embedding_matrix)

for word, i in word_index.items():
    if word in word2vec_model:
        embedding_matrix[i] = np.asarray(word2vec_model[word], dtype='float32')

print('*********')
print(embedding_matrix.shape)
print(embedding_matrix)
embedding_layer = Embedding(len(word_index)+1,
                            Embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_length,
                            trainable=False)
