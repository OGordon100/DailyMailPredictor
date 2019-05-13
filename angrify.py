import pickle

import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from titlecase import titlecase


def load_stuff(model_name='model_checkpoint_data_no_short_words_256.h5', tokenizer_name='tokenizer.pickle'):
    with open(tokenizer_name, 'rb') as handle:
        tokenizer = pickle.load(handle)
    model = load_model(model_name)

    return tokenizer, model


def angrify(input_string, predict_thresh=0.2):
    predict_raw = model.predict(
        pad_sequences(tokenizer.texts_to_sequences([input_string]), maxlen=model.input_shape[1]))
    predict_caps_index = np.nonzero(np.round(predict_raw - predict_thresh + 0.5))[-1]

    raw = titlecase(input_string.lower()).split()
    for i in predict_caps_index:
        raw[i] = raw[i].upper()

    print(' '.join(raw))


if __name__ == "__main__":
    tokenizer, model = load_stuff()
    angrify("scientists fume as equipment fails again! is labour to blame?")

