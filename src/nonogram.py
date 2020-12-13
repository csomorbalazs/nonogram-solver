import itertools
import math
import re
from typing import List, Tuple
import numpy as np

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

        self.row_descriptors = [self.pad(row, self.row_descriptor_len)
                                for row in row_descriptors]
        self.col_descriptors = [self.pad(col, self.col_descriptor_len)
                                for col in col_descriptors]

        self.reset_cells()

        if colored_cells:
            self.colored_cells = colored_cells

    def pad(self, list, target_len):
        return [0] * (target_len - len(list)) + list[:target_len]

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

    def __str__(self):
        return self.print(stdout=False)

    def print(self, show_zeros=False, stdout=True) -> str:
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
        if stdout:
            print(output)

        return output

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

        return self.pad(descriptor, pad_len)

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

            if self.pad(self.__calculate_descriptor(possible_solution),
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

        return np.array(solved_line)

    def solve(self) -> bool:
        i = 0
        passive_since = 0
        while not self.is_solved():
            if self.solve_line(i % (self.row_count + self.col_count)) == 0:
                passive_since += 1
            else:
                passive_since = 0

            if passive_since >= self.row_count + self.col_count:
                return False
            i += 1

        return True