#!/usr/bin/env python3

import pandas as pd
import numpy as np
from math import gcd
from typing import List, Optional, Tuple

class SudokuBoard:
    def __init__(self):
        """
        grid: 9x9 board
            - None means empty cell
            - otherwise an integer digit [0..9]
        """
        self.grid = [[None]*9 for _ in range(9)]
        
        # We might exclude exactly one digit from 0..9 in the final solution.
        self.excluded_digit = None

        # Track best found solution data (global, for use during backtracking).
        self.best_gcd = 0
        self.best_grid = None

    def load_board(self, filename: str) -> None:
        """
        Load board from .csv or .xlsx, storing digits and None for '-'.
        If negative numbers appear, convert them to positive.
        """
        if filename.endswith('.csv'):
            df = pd.read_csv(filename, header=None)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(filename, header=None)
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx")

        print("Initial DataFrame:")
        print(df, "\n")

        # Replace '-' with None
        df = df.replace('-', None)
        
        board_array = df.to_numpy()

        # Convert negative to positive if needed
        float_grid = pd.DataFrame(board_array).astype(float)
        board_array = np.where(
            pd.notna(float_grid) & (float_grid < 0),
            np.abs(float_grid),
            board_array
        )

        # Convert valid numbers to int (or None if empty)
        for i in range(9):
            for j in range(9):
                val = board_array[i][j]
                if pd.notna(val):
                    self.grid[i][j] = int(float(val))  # ensure integer
                else:
                    self.grid[i][j] = None
        
        print("Parsed initial grid:")
        self.print_board()
        print()

    def print_board(self) -> None:
        """Pretty-print the Sudoku grid."""
        for i in range(9):
            if i > 0 and i % 3 == 0:
                print("-"*21)
            row_str = []
            for j in range(9):
                if j > 0 and j % 3 == 0:
                    row_str.append("|")
                val = self.grid[i][j]
                row_str.append(str(val) if val is not None else ".")
            print(" ".join(row_str))

    # ------------------------------------------------------------------------
    # Constraint checks
    # ------------------------------------------------------------------------
    def used_in_row(self, row: int) -> set:
        return {d for d in self.grid[row] if d is not None}

    def used_in_col(self, col: int) -> set:
        return {self.grid[r][col] for r in range(9) if self.grid[r][col] is not None}

    def used_in_box(self, row: int, col: int) -> set:
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        digits = set()
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if self.grid[r][c] is not None:
                    digits.add(self.grid[r][c])
        return digits

    def valid_candidates(self, row: int, col: int) -> List[int]:
        """
        Return a list of valid digits for grid[row][col] considering:
        - Standard Sudoku row, col, 3×3 constraints
        - Excluded digit (if any)
        """
        if self.grid[row][col] is not None:
            # Cell is already filled
            return []

        used = self.used_in_row(row) \
               | self.used_in_col(col) \
               | self.used_in_box(row, col)
        
        candidates = []
        for d in range(10):
            # If this digit is excluded for the entire puzzle, skip
            if self.excluded_digit == d:
                continue
            # If digit is already used in row/col/box, skip
            if d in used:
                continue
            candidates.append(d)
        return candidates

    # ------------------------------------------------------------------------
    # Helpers for building row integers and updating GCD
    # ------------------------------------------------------------------------
    def row_to_int(self, row: int) -> Optional[int]:
        """
        Convert row’s 9 digits into a single integer. Return None if any cell is empty.
        Leading zeros are automatically handled by int(...) in Python.
        """
        if any(self.grid[row][j] is None for j in range(9)):
            return None  # row not fully filled
        row_str = "".join(str(self.grid[row][j]) for j in range(9))
        return int(row_str)

    def compute_gcd_of_completed_rows(self, up_to_row: int) -> int:
        """
        Compute GCD of all fully completed rows up to 'up_to_row' (inclusive).
        If a row is not filled entirely, skip it in this partial GCD.
        """
        row_vals = []
        for i in range(up_to_row+1):
            val = self.row_to_int(i)
            if val is not None: 
                row_vals.append(val)
        # If no fully completed rows yet, partial GCD is effectively 0
        if not row_vals:
            return 0
        g = row_vals[0]
        for rv in row_vals[1:]:
            g = gcd(g, rv)
        return g

    # ------------------------------------------------------------------------
    # Backtracking strategy:
    #  - We fill the grid row by row, left to right
    #  - Once a row is complete, we compute partial GCD
    #  - If partial GCD <= self.best_gcd, prune immediately
    #  - Continue until the entire grid is filled
    # ------------------------------------------------------------------------
    def backtrack_fill(self, row: int, col: int) -> None:
        """
        Recursive function to fill the grid starting from (row, col).
        We track and update self.best_gcd, self.best_grid when we find solutions
        with higher GCD.
        """
        # If we've moved past row 8 → puzzle is filled
        if row == 9:
            # The board is fully filled; compute full GCD
            final_gcd = self._compute_gcd_all_rows()
            if final_gcd > self.best_gcd:
                self.best_gcd = final_gcd
                self.best_grid = [r[:] for r in self.grid]  # deep copy
            return

        # If we've moved past col 8, go to next row
        if col == 9:
            # We just finished a row; compute partial gcd to prune quickly
            partial_gcd = self.compute_gcd_of_completed_rows(row)
            if partial_gcd <= self.best_gcd:
                # Prune if partial gcd can't exceed best_gcd
                return
            else:
                # Move to next row
                self.backtrack_fill(row+1, 0)
            return

        # If cell is already fixed, skip to next cell
        if self.grid[row][col] is not None:
            self.backtrack_fill(row, col+1)
            return

        # Try valid candidates
        for candidate in self.valid_candidates(row, col):
            self.grid[row][col] = candidate
            self.backtrack_fill(row, col+1)
            self.grid[row][col] = None  # backtrack

    def _compute_gcd_all_rows(self) -> int:
        """Compute the GCD across all 9 row-integers, assuming the board is fully filled."""
        row_vals = []
        for i in range(9):
            row_str = "".join(str(self.grid[i][j]) for j in range(9))
            row_vals.append(int(row_str))
        g = row_vals[0]
        for rv in row_vals[1:]:
            g = gcd(g, rv)
        return g

    def solve_for_excluded_digit(self, digit_to_exclude: int) -> None:
        """
        Attempt solving under the assumption that 'digit_to_exclude' is the globally unused digit.
        If that digit appears in the initial puzzle, skip immediately.
        """
        # If puzzle already has digit_to_exclude in the givens, this scenario is invalid
        if any(digit_to_exclude in row_vals for row_vals in self.grid):
            return
        
        self.excluded_digit = digit_to_exclude
        
        # Reset best_gcd for this run, or we can keep a global best across all digits.
        # We'll keep a global best across all digits, so we just do not reset self.best_gcd here.
        
        # Start backtracking from top-left cell
        self.backtrack_fill(0, 0)

    def solve_max_gcd(self) -> None:
        """
        Overall driver:
          - For each digit in [0..9], try excluding it (unless it's present in puzzle givens).
          - Keep track of the best GCD solution across all these attempts.
        """
        self.best_gcd = 0
        self.best_grid = None

        for d in range(10):
            self.solve_for_excluded_digit(d)

        if self.best_grid is not None:
            self.grid = self.best_grid
            print(f"Max GCD found: {self.best_gcd}")
            print("Corresponding solution:")
            self.print_board()
            print("\nMiddle row number:", self._middle_row_as_number())
        else:
            print("No valid solution found under any excluded digit scenario.")

    def _middle_row_as_number(self) -> str:
        """
        Return the 9-digit number formed by the middle row (row=4).
        It's required by the puzzle as the final answer.
        """
        row_str = "".join(str(self.grid[4][j]) for j in range(9))
        return row_str

# ------------------------------------------------------------------------
# Usage Example
# ------------------------------------------------------------------------
if __name__ == "__main__":
    board = SudokuBoard()
    board.load_board("board.csv")  # <- Put your puzzle CSV file here
    board.solve_max_gcd()
