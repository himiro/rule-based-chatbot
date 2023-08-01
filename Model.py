import tensorflow as tf
from tensorflow.keras import Sequential, saving
from tensorflow.keras.layers import Dense, Dropout


class Model:
    def __init__(self):
        self.model = None
        self.input_shape = 2
        self.output_shape = None
        self.lr = None
        self.epochs = None

    def create_model(self):
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=self.input_shape, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(self.output_shape, activation="softmax"))
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.lr,
            decay_steps=10000,
            decay_rate=0.9)
        adam = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

    def fit_model(self, X, y):
        print(self.model.summary())
        self.model.fit(x=X, y=y, epochs=self.epochs, verbose=1)

    def save_model(self, overwrite=False):
        self.model.save('./model/model.keras', overwrite=overwrite)

    def load_model(self):
        self.model = saving.load_model('./model/model.keras')