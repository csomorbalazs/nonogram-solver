import itertools
import os
import sys
import random
from nonogram import Nonogram
import numpy as np
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--samples", type=int, help="Number of generated samples")
    parser.add_argument("-r", "--rows", type=int, help="Number of rows in a nonogram")
    parser.add_argument("-c", "--columns", type=int, help="Number of columns in a nonogram")
    parser.add_argument("-t", "--train", type=float, help="Train split [0.0, 1.0]")
    parser.add_argument("-v", "--valid", type=float, help="Valid split [0.0, 1.0]")
    parser.add_argument("-o", "--output", type=str, help="Output folder")

    args = parser.parse_args()
    for arg in vars(args):
        if getattr(args, arg) is None:
            parser.print_help()
            sys.exit(1)

    Path(args.output).mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_valid, y_valid, x_test, y_test = generate_data(args.samples, args.rows, args.columns, args.train, args.valid)

    for fname, data in zip(["x_train", "y_train", "x_valid", "y_valid", "x_test", "y_test"],
                           [x_train, y_train, x_valid, y_valid, x_test, y_test]):
        np.savetxt(os.path.join(args.output, fname + ".csv"), data, delimiter=",")


def load_data(folder):
    data = ()
    for fname in ["x_train", "y_train", "x_valid", "y_valid", "x_test", "y_test"]:
        data = (*data, np.loadtxt(os.path.join(folder, fname + ".csv"), delimiter=","))

    return data


def random_nonogram(row_count, col_count):
    cell_count = row_count*col_count

    # nonograms with half of the cells colored are the hardest
    colored_cell_count = np.random.binomial(cell_count, 0.5)

    fields = list(itertools.product(range(row_count), range(col_count)))

    colored_fields = random.sample(fields, k=colored_cell_count)

    nonogram = Nonogram(
        row_count,
        col_count,
        colored_cells=colored_fields)

    nonogram.calculate_descriptors()
    nonogram.reset_cells()

    if nonogram.solve():
        return nonogram
    else:
        return random_nonogram(row_count, col_count)

def x_from_nonogram(nonogram: Nonogram):
    flat_row_descriptor = np.array(nonogram.row_descriptors).flatten()
    flat_col_descriptor = np.array(nonogram.col_descriptors).flatten()

    return np.concatenate((flat_row_descriptor, flat_col_descriptor))

def y_from_nonogram(nonogram: Nonogram):
  return np.array(nonogram.cells).flatten()

def generate_data(num_nonograms=10000, rows=4, columns=4, train_split=0.7, valid_split=0.2):
    nonograms = [random_nonogram(rows, columns) for _ in range(num_nonograms)]

    N = len(nonograms)

    train_nonograms = nonograms[:int(N * train_split)]
    valid_nonograms = nonograms[int(N * train_split):int(N * (train_split + valid_split))]
    test_nonograms = nonograms[int(N * (train_split + valid_split)):]
    x_train = np.array([x_from_nonogram(n) for n in train_nonograms])
    y_train = np.array([y_from_nonogram(n) for n in train_nonograms])
    x_valid = np.array([x_from_nonogram(n) for n in valid_nonograms])
    y_valid = np.array([y_from_nonogram(n) for n in valid_nonograms])
    x_test = np.array([x_from_nonogram(n) for n in test_nonograms])
    y_test = np.array([y_from_nonogram(n) for n in test_nonograms])

    return x_train, y_train, x_valid, y_valid, x_test, y_test



if __name__ == '__main__':
    main()