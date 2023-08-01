from nltk.stem import WordNetLemmatizer
import string
import nltk
import random
import numpy as np
from Model import Model


class Chatbot:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('wordnet')
        self.words = []
        self.lemmatizer = WordNetLemmatizer()
        self.model = Model()

    def prepare_data(self, file_content):
        self.__lemmatize_data()

    def create_model(self):
        pass

    def set(self, parameter, value):
        self.parameter = value

    def __lemmatize_data(self):
        # Lemmatize lower vocab words
        self.words = [self.lemmatizer.lemmatize(word.lower()) for word in self.words if word not in string.punctuation]

    # One-Hot Encoding
    def one_hot_encoding(self):
        training = []
        out_empty = [0] * len(self.classes)
        # create word ensemble
        for idx, doc in enumerate(self.doc_X):
            bow = []
            text = self.lemmatizer.lemmatize(doc.lower())
            for word in self.words:
                bow.append(1) if word in text else bow.append(0)

            output_row = list(out_empty)
            output_row[self.classes.index(self.doc_y[idx])] = 1

            training.append([bow, output_row])
        # shuffle data and add it to training set
        random.shuffle(training)
        training = np.array(training, dtype=object)
        return training
