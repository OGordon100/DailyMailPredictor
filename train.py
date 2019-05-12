import pandas as pd
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

# Read in data
data = pd.read_csv("data.csv")

# Set up callbacks and helper functions
embedding_dim = 100
num_words = 50000
tensorboard = TensorBoard(log_dir='logs/')
checkpoints = ModelCheckpoint('model_checkpoint.h5',
                              monitor='val_loss', mode='min', save_best_only=True)
earlystopping = EarlyStopping(monitor='val_loss', patience=10)

binarizer = MultiLabelBinarizer()
tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(data.headline.values)
word_index = tokenizer.word_index
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Generate dataset
x = tokenizer.texts_to_sequences(data.capitalised_index.values)
x = pad_sequences(x)

y_init = [list(map(int, bodge.replace("[", "").replace("]", "").split())) for bodge in
          data.capitalised_index.str.replace("'", "")]
y = binarizer.fit_transform(y_init)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

model = Sequential()
model.add(Embedding(num_words, embedding_dim, input_length=x.shape[1]))  # None?
model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=150, batch_size=2048, verbose=1, validation_data=(x_test, y_test),
          callbacks=[tensorboard, earlystopping, checkpoints])
scores = model.evaluate(x_test, y_test, verbose=1)
