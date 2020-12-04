from typing import List, Tuple
import random
import numpy as np
import itertools
import math
import re


def pad(list, target_len):
    return [0]*(target_len - len(list)) + list[:target_len]


class Nonogram():
    display_char = {
        -1: "?",
        0: "□",
        1: "■"
    }

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

        self.reset_cells()

        if colored_cells:
            self.colored_cells = colored_cells

    def reset_cells(self):
        self.cells = np.ones((self.row_count, self.col_count), dtype=int) * -1

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
                output += Nonogram.display_char[self.cells[i][j]] + " "

            output += "\n"

        # don't display 0 values
        if not show_zeros:
            output = re.sub(r"(?<=[^0-9])0", " ", output)
        print(output)

    def is_solved(self) -> bool:
        if np.isin(-1, self.cells):
            return False

        real_row_descriptors = self.__calculate_row_descriptors()

        if real_row_descriptors != self.row_descriptors:
            return False

        real_col_descriptors = self.__calculate_col_descriptors()

        if real_col_descriptors != self.col_descriptors:
            return False

        return True

    def calculate_descriptors(self):
        self.row_descriptors = self.__calculate_row_descriptors()
        self.col_descriptors = self.__calculate_col_descriptors()

        return self.row_descriptors, self.col_descriptors

    def __calculate_row_descriptors(self):
        row_descriptors = []

        for row in self.cells:
            row_descriptors.append(self.__calculate_descriptor(row))

        return row_descriptors

    def __calculate_col_descriptors(self):
        col_descriptors = []

        for col in self.cells.T:
            col_descriptors.append(self.__calculate_descriptor(col))

        return col_descriptors

    def __calculate_descriptor(self, list: List[int]):
        descriptor = []
        open_piece = False
        pad_len = math.ceil(len(list) / 2)

        for elem in list:
            if elem == 1:
                if open_piece:
                    descriptor[-1] += 1
                else:
                    descriptor.append(1)
                    open_piece = True
            else:
                open_piece = False

        return pad(descriptor, pad_len)

    def solve_line(self, index: int):
        if index < self.row_count:
            row = self.cells[index]
            row_descriptor = self.row_descriptors[index]

            solved_row = self.__solve_line(row, row_descriptor)
            reward = np.sum(row == -1) - np.sum(solved_row == -1)

            self.cells[index] = solved_row
            return reward
        else:
            index -= self.row_count
            col = self.cells[:, index]
            col_descriptor = self.col_descriptors[index]

            solved_col = self.__solve_line(col, col_descriptor)
            reward = np.sum(col == -1) - np.sum(solved_col == -1)

            self.cells[:, index] = solved_col
            return reward

    def __solve_line(self, line, line_descriptor):
        # More efficient algorithm could be implemented
        possible_lines = []
        filled_cell_count = sum(line_descriptor)

        line_indexes = range(len(line))

        for filled_indexes in itertools.combinations(line_indexes, filled_cell_count):
            possible_solution = [0 for _ in range(len(line))]

            for i in filled_indexes:
                possible_solution[i] = 1

            if pad(self.__calculate_descriptor(possible_solution),
                   len(line_descriptor)) == line_descriptor:

                matches_line = True
                for r, s in zip(line, possible_solution):
                    if r != -1 and r != s:
                        matches_line = False
                        break

                if matches_line:
                    possible_lines.append(np.array(possible_solution))

        result = [True] * len(possible_lines[0])
        last_line = possible_lines[0]

        for curr_line in possible_lines:
            for i, (curr, last) in enumerate(zip(curr_line, last_line)):
                if curr != last:
                    result[i] = False
            last_line = curr_line

        solved_line = [e if result[i] else -1 for i,
                       e in enumerate(possible_lines[0])]

        return solved_line


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

    nonogram = random_nonogram(5, 5)
    nonogram.reset_cells()

    for i in range(1000):
        nonogram.solve_line(i % 10)
        nonogram.print()
        if nonogram.is_solved():
            print("Solved!")
            break

    print("Row descriptors:", nonogram.row_descriptors)
    print("Column descriptors: ", nonogram.col_descriptors)
    print("Cells: ", nonogram.cells)
