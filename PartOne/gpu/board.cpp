#include "board.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <numeric>
#include <cmath>
#include <algorithm>

SudokuBoard::SudokuBoard() {
    // Initialize empty grid
    for (auto& row : grid) {
        row.fill(std::nullopt);
    }
    excluded_digit = std::nullopt;
}

bool SudokuBoard::load_board(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::string line;
    int row = 0;
    
    while (std::getline(file, line) && row < BOARD_SIZE) {
        std::stringstream ss(line);
        std::string cell;
        int col = 0;
        
        while (std::getline(ss, cell, ',') && col < BOARD_SIZE) {
            grid[row][col] = parse_cell(cell);
            col++;
        }
        
        row++;
    }

    std::cout << "Loaded board:" << std::endl;
    print_board();
    return true;
}

void SudokuBoard::print_board() const {
    for (int i = 0; i < BOARD_SIZE; i++) {
        if (i > 0 && i % 3 == 0) {
            std::cout << std::string(21, '-') << std::endl;
        }
        
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (j > 0 && j % 3 == 0) {
                std::cout << "| ";
            }
            
            if (grid[i][j].has_value()) {
                std::cout << static_cast<int>(grid[i][j].value()) << " ";
            } else {
                std::cout << ". ";
            }
        }
        std::cout << std::endl;
    }
}

std::vector<SudokuBoard::cell_t> SudokuBoard::get_flat_board() const {
    std::vector<cell_t> flat_board(BOARD_SIZE * BOARD_SIZE, -1);
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (grid[i][j].has_value()) {
                flat_board[i * BOARD_SIZE + j] = grid[i][j].value();
            }
        }
    }
    return flat_board;
}

void SudokuBoard::set_from_flat_board(const std::vector<cell_t>& flat_board) {
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            cell_t val = flat_board[i * BOARD_SIZE + j];
            grid[i][j] = (val >= 0) ? std::optional<cell_t>(val) : std::nullopt;
        }
    }
}

bool SudokuBoard::is_solved() const {
    // Check if all cells are filled
    for (const auto& row : grid) {
        for (const auto& cell : row) {
            if (!cell.has_value()) return false;
        }
    }
    return true;
}

SudokuBoard::gcd_t SudokuBoard::compute_gcd_of_rows() const {
    std::vector<gcd_t> row_numbers;
    
    for (const auto& row : grid) {
        std::string row_str;
        for (const auto& cell : row) {
            if (!cell.has_value()) return 0;
            row_str += std::to_string(cell.value());
        }
        row_numbers.push_back(std::stoll(row_str));
    }
    
    // Compute GCD of all rows
    gcd_t result = row_numbers[0];
    for (size_t i = 1; i < row_numbers.size(); i++) {
        result = std::gcd(result, row_numbers[i]);
    }
    return result;
}

bool SudokuBoard::is_valid_digit(const std::string& str) {
    return !str.empty() && str.find_first_not_of("0123456789") == std::string::npos;
}

std::optional<SudokuBoard::cell_t> SudokuBoard::parse_cell(const std::string& str) {
    // Trim whitespace
    std::string trimmed = str;
    trimmed.erase(0, trimmed.find_first_not_of(" \t\r\n"));
    trimmed.erase(trimmed.find_last_not_of(" \t\r\n") + 1);
    
    // Check for empty cell markers
    if (trimmed.empty() || trimmed == "-") {
        return std::nullopt;
    }
    
    // Parse valid digits
    if (is_valid_digit(trimmed)) {
        return static_cast<cell_t>(std::stoi(trimmed));
    }
    
    return std::nullopt;
} 