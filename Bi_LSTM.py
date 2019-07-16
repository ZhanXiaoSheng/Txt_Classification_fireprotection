from keras.layers import Dense, Input, Flatten, Dropout, LSTM, Embedding, Bidirectional
from keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D
from keras.optimizers import RMSprop
import numpy as np
from sklearn import metrics
import pre_data


#  传递参数
word_index = pre_data.word_index
Embedding_dim = pre_data.Embedding_dim
max_length = pre_data.max_length
x_train = pre_data.x_train
x_val = pre_data.x_val
x_test = pre_data.x_test
y_train = pre_data.y_train
y_val = pre_data.y_val
y_test = pre_data.y_test

Bi_model = Sequential()
Bi_model.add(Embedding(input_dim=len(word_index)+1, output_dim=Embedding_dim, input_length=max_length))
Bi_model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
# Bi_model.add(SeqSelfAttention(attention_activation='sigmoid'))
Bi_model.add(Dropout(0.2))
Bi_model.add(Dense(3, activation='softmax'))
Bi_model.summary()
Bi_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=["accuracy"])
Bi_history = Bi_model.fit(x_train, y_train, validation_data=[x_val, y_val], epochs=50, batch_size=128)

# prediction
test_prediction = Bi_model.predict(x_test)
# precision ,recall
print(metrics.classification_report(np.argmax(test_prediction, axis=1), np.argmax(y_test, axis=1), digits=4))

# save model, weights
Bi_model.save('BiLSTM.h5')
Bi_model.save_weights('BiLSTM_weights.h5')