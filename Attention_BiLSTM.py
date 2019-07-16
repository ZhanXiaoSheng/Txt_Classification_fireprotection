from keras.layers import Dense, Input, Flatten, Dropout, LSTM, Embedding, Bidirectional
from keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D
from keras.optimizers import RMSprop
import pre_data
from keras.models import Model
from keras import regularizers
from keras_self_attention import SeqSelfAttention
from sklearn import metrics
import numpy as np
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


inputs = Input(name='inputs', shape=[max_length])
layer = Embedding(len(word_index)+1, input_length=max_length, output_dim=Embedding_dim, name='Embedding'
               )(inputs)
layer = Bidirectional(LSTM(units=16, return_sequences=True), name='Bi-LSTM')(layer)
layer = Dropout(0.5)(layer)
layer = SeqSelfAttention(units=8, name='attention')(layer)
layer = Flatten()(layer)
layer = Dense(3, kernel_regularizer=regularizers.l2(0.1), activation='softmax')(layer)
AttBi_model = Model(inputs=inputs, outputs=layer)
AttBi_model.summary()

AttBi_model.compile(
    optimizer=RMSprop(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

AttBi_history = AttBi_model.fit(x_train, y_train, validation_data=[x_val, y_val], epochs=50, batch_size=128)

# prediction
test_prediction = AttBi_model.predict(x_test)

# Recall,precision
print(metrics.classification_report(np.argmax(test_prediction, axis=1), np.argmax(y_test, axis=1), digits=4))
# save model and weights
AttBi_model.save_weights('AttBi_weights.h5')
AttBi_model.save('AttBi.h5')
