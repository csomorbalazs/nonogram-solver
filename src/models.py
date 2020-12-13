import math
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv1D, MaxPooling2D, Dropout, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from datagen import load_data
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input folder, consisting training and validation data" , default=".")
    parser.add_argument("-o", "--output", type=str, help="Compiled model is saved to here", default="compiled")
    args = parser.parse_args()
    for arg in vars(args):
        if getattr(args, arg) is None:
            parser.print_help()
            sys.exit(1)

    x_train, y_train, _, _, _, _ = load_data(args.input)

    model = build_model(x_train, y_train)
    model.save(args.output)


def build_model(x_train, y_train):
    kernel_size = math.ceil(math.sqrt(x_train.shape[1]) / 2)

    model = Sequential()
    model.add(Input(x_train.shape[1]))
    model.add(Reshape((x_train.shape[1], 1)))
    model.add(Conv1D(128, kernel_size=kernel_size, strides=kernel_size, input_shape=(x_train.shape[1], 1),
                     activation='relu'))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1200, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))

    optimizer = Adam(learning_rate=0.002)
    model.compile(loss='mae', optimizer=optimizer)
    model.summary()

    return model


if __name__ == '__main__':
    main()