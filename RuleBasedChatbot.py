import nltk
from Chatbot import Chatbot
import numpy as np
import random


class RuleBasedChatbot(Chatbot):
    def __init__(self):
        Chatbot.__init__(self)
        self.classes = []
        self.doc_X = []
        self.doc_y = []
        self.training = None
        self.intents = None

    def prepare_data(self, file_content):
        self.__parse_file(file_content)
        Chatbot.prepare_data(self, file_content)
        # sort vocab and words
        self.words = sorted(set(self.words))
        self.classes = sorted(set(self.classes))
        self.training = Chatbot.one_hot_encoding(self)

    def clean_text(self, text):
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return tokens

    def bag_of_words(self, text, vocab):
        tokens = self.clean_text(text)
        bow = [0] * len(vocab)
        for w in tokens:
            for idx, word in enumerate(vocab):
                if word == w:
                    bow[idx] = 1
        return np.array(bow)

    def pred_class(self, text):
        bow = self.bag_of_words(text, self.words)
        result = self.model.model.predict(np.array([bow]))[0]
        thresh = 0.2
        y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
        y_pred.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in y_pred:
            return_list.append(self.classes[r[0]])
        self.intents = return_list

    def get_response(self, message, intents_json):
        self.pred_class(message)
        tag = self.intents[0]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                return random.choice(i["responses"])

    def __parse_file(self, file_content):
        # Tokenize patterns and fill the lists
        for intent in file_content["intents"]:
            for pattern in intent["patterns"]:
                tokens = nltk.word_tokenize(pattern)
                self.words.extend(tokens)
                self.doc_X.append(pattern)
                self.doc_y.append(intent["tag"])

            # add tag to class if it doesn't already exist
            if intent["tag"] not in self.classes:
                self.classes.append(intent["tag"])
