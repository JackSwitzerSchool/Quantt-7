import pandas as pd
import numpy as np
from math import gcd
from typing import List, Optional, Tuple

class SudokuBoard:
    def __init__(self):
        # We'll store the puzzle as a 2D list of optional ints
        # None indicates an empty cell.
        self.grid = [[None]*9 for _ in range(9)]
        
        # If you want to track which digit (0–9) is not used at all in the puzzle,
        # you can store that here once you’ve identified it. For now, just keep placeholder.
        self.excluded_digit = None  

    def load_board(self, filename: str) -> None:
        """Load board from .csv or .xlsx, storing digits and None for '-'."""
        if filename.endswith('.csv'):
            df = pd.read_csv(filename, header=None)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(filename, header=None)
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx")

        print("Initial DataFrame:")
        print(df)

        # Replace '-' with None
        df = df.replace('-', None)
        
        # Convert to numpy array for uniform handling
        board_array = df.to_numpy()

        # If negative numbers appear, convert to positive
        float_grid = pd.DataFrame(board_array).astype(float)
        board_array = np.where(
            pd.notna(float_grid) & (float_grid < 0),
            np.abs(float_grid),
            board_array
        )

        # Convert valid numbers to int (or None if empty)
        for i in range(len(board_array)):
            for j in range(len(board_array[i])):
                if pd.notna(board_array[i][j]):
                    self.grid[i][j] = int(float(board_array[i][j]))
                else:
                    self.grid[i][j] = None

        print("\nParsed grid:")
        self.print_board()

    def print_board(self) -> None:
        """Pretty-print the Sudoku grid."""
        for i in range(9):
            if i > 0 and i % 3 == 0:
                print("-" * 21)
            row_str = []
            for j in range(9):
                if j > 0 and j % 3 == 0:
                    row_str.append("|")
                val = self.grid[i][j]
                row_str.append(str(val) if val is not None else ".")
            print(" ".join(row_str))

    def used_digits_in_row(self, row: int) -> set:
        return {d for d in self.grid[row] if d is not None}

    def used_digits_in_col(self, col: int) -> set:
        return {self.grid[r][col] for r in range(9) if self.grid[r][col] is not None}

    def used_digits_in_box(self, row: int, col: int) -> set:
        # Identify the 3x3 box’s top-left coordinates
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        digits = set()
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if self.grid[r][c] is not None:
                    digits.add(self.grid[r][c])
        return digits

    def valid_candidates(self, row: int, col: int) -> List[int]:
        """Return a list of valid digits for grid[row][col], considering standard Sudoku constraints
           plus the puzzle’s detail that only 9 of the 10 digits [0..9] are used overall."""
        if self.grid[row][col] is not None:
            return []

        used = self.used_digits_in_row(row) \
               | self.used_digits_in_col(col) \
               | self.used_digits_in_box(row, col)

        # If you know which digit is excluded from the entire puzzle, remove it from the set of possibilities.
        # For now, assume we might exclude each possibility at some stage, so we check all digits [0..9].
        candidates = []
        for digit in range(10):
            if digit not in used:
                # If you have logic for an excluded digit, filter out that as well
                if self.excluded_digit is not None and digit == self.excluded_digit:
                    continue
                candidates.append(digit)

        return candidates

    def is_solved(self) -> bool:
        """Check if every cell is filled and constraints are satisfied."""
        for i in range(9):
            for j in range(9):
                if self.grid[i][j] is None:
                    return False
        return True

    def compute_gcd_of_rows(self) -> int:
        """Form each row’s 9-digit integer and compute the GCD across all 9 rows."""
        # If leading digit is 0, the integer has a leading zero — that’s allowed, but be mindful
        # to treat it as an int. The row "0 3 5 ..." becomes 035... → 35..., etc.
        # However, we can parse as a string then cast to int.
        row_numbers = []
        for i in range(9):
            row_str = "".join(str(d) for d in self.grid[i])
            row_val = int(row_str)
            row_numbers.append(row_val)
        # Compute overall GCD
        current_gcd = row_numbers[0]
        for val in row_numbers[1:]:
            current_gcd = gcd(current_gcd, val)
        return current_gcd

    def find_empty_cell(self) -> Optional[Tuple[int, int]]:
        """Find the next empty cell (row,col). Return None if board is full."""
        for i in range(9):
            for j in range(9):
                if self.grid[i][j] is None:
                    return (i, j)
        return None

    def backtrack_solver(self, best_solution: dict) -> bool:
        """
        Standard backtracking approach:
          - Pick an empty cell
          - Try each candidate
          - Recurse
          - If puzzle gets fully solved, check GCD and update best_solution if better
          - Backtrack if needed.
        
        best_solution: a dictionary for storing {'gcd': int, 'grid': deep_copy_of_grid}.
        We pass it around to track the highest GCD found so far.
        """
        empty_cell = self.find_empty_cell()
        if empty_cell is None:
            # No empty cells => valid solution
            current_gcd = self.compute_gcd_of_rows()
            if current_gcd > best_solution['gcd']:
                best_solution['gcd'] = current_gcd
                # Make a copy of the current grid so we can restore it if needed
                best_solution['grid'] = [row[:] for row in self.grid]
            return False  # Return False to keep searching for possibly better GCD solutions

        row, col = empty_cell
        candidates = self.valid_candidates(row, col)
        for candidate in candidates:
            self.grid[row][col] = candidate
            # Recurse
            self.backtrack_solver(best_solution)
            self.grid[row][col] = None
        
        return False  # We want to explore all solutions to find the max GCD

    def solve_max_gcd(self) -> None:
        """
        Wrapper that uses backtracking to find the solution that yields
        the maximum GCD among row-integers.
        """
        # Prepare a structure to keep track of the best GCD found so far.
        best_solution = {
            'gcd': 0,
            'grid': None
        }

        # Because one digit is excluded, you can loop over each digit in [0..9],
        # set self.excluded_digit, and see if it yields a valid puzzle with a big GCD.
        # Or you can attempt to discover that digit automatically. For now, we’ll do a naive approach:
        for digit_to_exclude in range(10):
            self.excluded_digit = digit_to_exclude
            
            # Check feasibility quickly (if the excluded digit already appears in the puzzle, skip)
            if any(digit_to_exclude in row for row in self.grid):
                continue
            
            # Now run backtracking
            self.backtrack_solver(best_solution)

        # After exploring all possible excluded digits, set the final answer back
        if best_solution['grid'] is not None:
            self.grid = best_solution['grid']
            print(f"Max GCD found: {best_solution['gcd']}")
            print("Final board configuration:")
            self.print_board()
        else:
            print("No valid solution was found.")

# Example usage
if __name__ == "__main__":
    board = SudokuBoard()
    board.load_board("board.csv")
    board.solve_max_gcd()
