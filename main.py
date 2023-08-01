import numpy as np
from RuleBasedChatbot import RuleBasedChatbot
import json
import argparse

parser = argparse.ArgumentParser(
                    prog='Chatbot',
                    description='Minimal rule-based chatbot')

parser.add_argument('--train', action='store_true', help="Train a model from scratch")
parser.add_argument('--overwrite', action='store_true', help="Overwrite existing ./model/model.keras model saving")
args = parser.parse_args()
TRAIN = args.train
OVERWRITE = args.overwrite


def read_file(filename):
    with open(filename) as intents:
        return json.load(intents)


chatbot = RuleBasedChatbot()

data = read_file('dataset.json')
chatbot.prepare_data(data)

print('Words :\n', chatbot.words)
print('Classes :\n', chatbot.classes)
print('Doc X :\n', chatbot.doc_X)
print('Doc y :\n', chatbot.doc_y)

train_X = np.array(list(chatbot.training[:, 0]))
train_y = np.array(list(chatbot.training[:, 1]))

chatbot.model.input_shape = (len(train_X[0]),)
chatbot.model.output_shape = len(train_y[0])
chatbot.model.epochs = 200
chatbot.model.lr = 0.01

if TRAIN:
    chatbot.model.create_model()
    chatbot.model.fit_model(train_X, train_y)
    chatbot.model.save_model(OVERWRITE)
else:
    chatbot.model.load_model()


print('Chat loaded. Go !')
while True:
    message = input("")
    result = chatbot.get_response(message, data)
    print(result)
