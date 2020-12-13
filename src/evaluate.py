import sys
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from tensorflow.keras.models import load_model
from datagen import load_data
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input folder, consisting test data" , default=".")
    parser.add_argument("-m", "--model", type=str, help="Model path", default='trained/')

    args = parser.parse_args()
    for arg in vars(args):
        if getattr(args, arg) is None:
            parser.print_help()
            sys.exit(1)
    _,_,_,_, x_test, y_test = load_data(args.input)
    model = load_model(args.model)
    evaluate(model, x_test, y_test)

def plot(missed_cells,  y_predicted, correct_guesses):
    plt.hist(missed_cells,
             bins=y_predicted.shape[1],
             weights=np.ones(len(missed_cells)) / len(missed_cells),
             edgecolor='white', color='orange')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title(f'The model solved {100 * correct_guesses / len(missed_cells):.2f}% of nonograms')
    plt.xlabel('Number of incorrect fields')
    plt.show()



def evaluate(model, x_test, y_test):
    y_predicted = model.predict(x_test)
    y_predicted = np.rint(y_predicted).astype(int)

    missed_cells = []

    for predicted_cells, real_cells in zip(y_predicted, y_test):
        mae = mean_absolute_error(predicted_cells, real_cells)
        missed_cells.append(int(mae * len(real_cells)))

    correct_guesses = len([c for c in missed_cells if c == 0])

    plot(missed_cells, y_predicted, correct_guesses)


if __name__ == '__main__':
    main()