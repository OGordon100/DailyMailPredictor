import pickle

import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from titlecase import titlecase


class Angrifier:
    def load_stuff(self, model_name='model_checkpoint_data_no_short_words_256.h5', tokenizer_name='tokenizer.pickle'):
        with open(tokenizer_name, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.model = load_model(model_name)


    def angrify(self, input_string, predict_thresh=0.2):
        predict_raw = self.model.predict(
            pad_sequences(self.tokenizer.texts_to_sequences([input_string]), maxlen=self.model.input_shape[1]))
        predict_caps_index = np.nonzero(np.round(predict_raw - predict_thresh + 0.5))[-1]

        raw = titlecase(input_string.lower()).split()
        for i in predict_caps_index:
            raw[i] = raw[i].upper()

        print(' '.join(raw))


if __name__ == "__main__":
    angrifier = Angrifier()
    angrifier.load_stuff()
    angrifier.angrify("scientists fume as equipment fails again! is labour to blame?")

