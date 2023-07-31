import json
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

nltk.download('punkt')
nltk.download('wordnet')


def read_file(filename):
    with open(filename) as intents:
        return json.load(intents)


data = read_file('train.json')
lemmatizer = WordNetLemmatizer()

words = []
classes = []
doc_X = []
doc_y = []

# Tokenize patterns and fill the lists
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_X.append(pattern)
        doc_y.append(intent["tag"])

    # add tag to class if it doesn't already exist
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

# Lemmatize lower vocab words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]

# sort vocab and words
words = sorted(set(words))
classes = sorted(set(classes))

print(words)
print(classes)
print(doc_X)
print(doc_y)


# One-Hot Encoding
def one_hot_encoding():
    training = []
    out_empty = [0] * len(classes)
    # create word ensemble
    for idx, doc in enumerate(doc_X):
        bow = []
        text = lemmatizer.lemmatize(doc.lower())
        for word in words:
            bow.append(1) if word in text else bow.append(0)

        output_row = list(out_empty)
        output_row[classes.index(doc_y[idx])] = 1

        training.append([bow, output_row])
    # shuffle data and add it to training set
    random.shuffle(training)
    training = np.array(training, dtype=object)
    return training


training = one_hot_encoding()
train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

input_shape = (len(train_X[0]),)
output_shape = len(train_y[0])
epochs = 200
lr = 0.01


def create_model():
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(output_shape, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    adam = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
    return model


model = create_model()
print(model.summary())
model.fit(x=train_X, y=train_y, epochs=200, verbose=1)


def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)


def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0]
    thresh = 0.2
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


print('Chatbot launched !')
while True:
    message = input("")
    intents = pred_class(message, words, classes)
    result = get_response(intents, data)
    print(result)
