#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <optional>
#include <array>

#define DATA_PATH "../data/input/board.csv"
#define OUT_DIR "../data/output"

class SudokuBoard {
public:
    static constexpr int BOARD_SIZE = 9;
    using cell_t = int8_t;  // For the grid values
    using gcd_t = int64_t;  // For GCD calculations

    SudokuBoard();
    
    // Load board from CSV file
    bool load_board(const std::string& filename);
    
    // Display methods
    void print_board() const;
    
    // Getters/Setters for CUDA integration
    std::vector<cell_t> get_flat_board() const;
    void set_from_flat_board(const std::vector<cell_t>& flat_board);
    
    // Helper methods for solving
    bool is_solved() const;
    gcd_t compute_gcd_of_rows() const;

private:
    std::array<std::array<std::optional<cell_t>, BOARD_SIZE>, BOARD_SIZE> grid;
    std::optional<cell_t> excluded_digit;

    // Helper methods
    static bool is_valid_digit(const std::string& str);
    static std::optional<cell_t> parse_cell(const std::string& str);
}; 