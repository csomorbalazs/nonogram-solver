from typing import List, Tuple
import random
import numpy as np
import itertools
import math
import re


def pad(list, target_len):
    return [0]*(target_len - len(list)) + list[:target_len]


class Nonogram():
    def __init__(self,
                 row_count: int,
                 col_count: int,
                 row_descriptors: List[List[int]] = None,
                 col_descriptors: List[List[int]] = None,
                 colored_cells: List[Tuple[int]] = None):
        self.row_count = row_count
        self.col_count = col_count

        self.row_descriptor_len = math.ceil(col_count / 2)
        self.col_descriptor_len = math.ceil(row_count / 2)

        if not row_descriptors:
            row_descriptors = [[] for _ in range(row_count)]
        if not col_descriptors:
            col_descriptors = [[] for _ in range(col_count)]

        self.row_descriptors = [pad(row, self.row_descriptor_len)
                                for row in row_descriptors]
        self.col_descriptors = [pad(col, self.col_descriptor_len)
                                for col in col_descriptors]

        self.cells = [[0 for _ in range(col_count)]
                      for _ in range(row_count)]

        if colored_cells:
            self.colored_cells = colored_cells

    @property
    def colored_cells(self) -> List[Tuple[int]]:
        solutions = []
        for i in range(len(self.row_descriptors)):
            for j in range(len(self.col_descriptors)):
                if self.cells[i][j] == 1:
                    solutions.append((i, j))
        return solutions

    @colored_cells.setter
    def colored_cells(self, solutions):
        for i in range(self.row_count):
            for j in range(self.col_count):
                self.cells[i][j] = 1 if (i, j) in solutions else 0

    def print(self, show_zeros=False) -> None:
        row_descriptor_len = self.row_descriptor_len
        col_descriptor_len = self.col_descriptor_len

        output = ""

        # column numbers
        for i in range(col_descriptor_len-1, -1, -1):
            output += "  " * row_descriptor_len
            for col in self.col_descriptors:
                output += str(col[-(i+1)]) + " " if len(col) > i else "  "

            output += "\n"

        # row numbers
        for i, row in enumerate(self.row_descriptors):
            for r in range(row_descriptor_len-1, -1, -1):
                output += str(row[-(r+1)]) + " " if len(row) > r else "  "

            # body
            for j, col in enumerate(self.col_descriptors):
                output += "■ " if self.cells[i][j] == 1 else "□ "

            output += "\n"

        # don't display 0 values
        if not show_zeros:
            output = re.sub(r"(?<=[^0-9])0", " ", output)
        print(output)

    def is_valid(self) -> bool:
        for i in range(len(self.row_descriptors)):
            row_pieces = []
            row_open_piece = False

            for j in range(len(self.col_descriptors)):
                if self.cells[i][j] == 1:
                    if row_open_piece:
                        row_pieces[-1] += 1
                    else:
                        row_pieces.append(1)
                        row_open_piece = True
                else:
                    row_open_piece = False

            if pad(row_pieces, self.row_descriptor_len) != self.row_descriptors[i]:
                return False

        for i in range(len(self.col_descriptors)):
            col_pieces = []
            col_open_piece = False

            for j in range(len(self.row_descriptors)):
                if self.cells[j][i] == 1:
                    if col_open_piece:
                        col_pieces[-1] += 1
                    else:
                        col_pieces.append(1)
                        col_open_piece = True
                else:
                    col_open_piece = False

            if pad(col_pieces, self.col_descriptor_len) != self.col_descriptors[i]:
                return False

        return True

    def calculate_descriptors(self):
        row_descriptors = []
        col_descriptors = []

        for i in range(self.row_count):
            row_descriptor = []
            row_open_piece = False

            for j in range(self.col_count):
                if self.cells[i][j] == 1:
                    if row_open_piece:
                        row_descriptor[-1] += 1
                    else:
                        row_descriptor.append(1)
                        row_open_piece = True
                else:
                    row_open_piece = False

            row_descriptors.append(row_descriptor)

        for i in range(self.col_count):
            col_descriptor = []
            col_open_piece = False

            for j in range(self.row_count):
                if self.cells[j][i] == 1:
                    if col_open_piece:
                        col_descriptor[-1] += 1
                    else:
                        col_descriptor.append(1)
                        col_open_piece = True
                else:
                    col_open_piece = False

            col_descriptors.append(col_descriptor)

        self.row_descriptors = [
            pad(r, self.row_descriptor_len) for r in row_descriptors]
        self.col_descriptors = [
            pad(c, self.col_descriptor_len) for c in col_descriptors]

        return self.row_descriptors, self.col_descriptors


def random_nonogram(row_count, col_count) -> Nonogram:
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

    return nonogram


if __name__ == "__main__":

    nonogram = random_nonogram(row_count=8, col_count=8)

    nonogram.print()

    print("Row descriptors:", nonogram.row_descriptors)
    print("Column descriptors: ", nonogram.col_descriptors)
    print("Cells: ", nonogram.cells)
