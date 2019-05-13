import os
import pickle

import pandas as pd
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Read in data
data = pd.read_csv("data_no_short_words.csv")
data.pub_date = pd.to_datetime(data.pub_date)
data.capitalised_index = [list(map(int, bodge.replace("[", "").replace("]", "").split())) for bodge in
          data.capitalised_index.str.replace("'", "")]

plt.plot(data.pub_date, data.capitalised_index.str.len(), 'x')
plt.title("Rabidness Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Capitalised Words")
plt.show()


# Set up callbacks and helper functions
embedding_dim = 100
num_words = 50000
tensorboard = TensorBoard(log_dir='logs/')
checkpoints = ModelCheckpoint('model_checkpoint1.h5',
                              monitor='val_loss', mode='min', save_best_only=True)
earlystopping = EarlyStopping(monitor='val_loss', patience=10)

binarizer = MultiLabelBinarizer()
if os.path.isfile("tokenizer.pickle") is True:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
else:
    tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
                          lower=True)  # TODO; pretrained embed
    tokenizer.fit_on_texts(data.headline.values)
    word_index = tokenizer.word_index
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Generate dataset
x = tokenizer.texts_to_sequences(data.headline.values)
x = pad_sequences(x)

y = binarizer.fit_transform(data.capitalised_index)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# Train model
model = Sequential()
model.add(Embedding(num_words, embedding_dim, input_length=x.shape[1]))  # None?
model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=150, batch_size=8192, verbose=1, validation_data=(x_test, y_test),
          callbacks=[tensorboard, earlystopping, checkpoints])
scores = model.evaluate(x_test, y_test, verbose=1)
